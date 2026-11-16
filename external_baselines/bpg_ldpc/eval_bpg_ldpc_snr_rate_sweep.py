import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from decompjscc_utils import compute_ms_ssim_batch, compute_ssim_batch, ms_ssim_metadata
from external_baselines.bpg_ldpc.codec import BPGCodec, search_bpg_qp
from external_baselines.bpg_ldpc.digital_link import (
    bits_to_bytes,
    bytes_to_bits,
    run_5g_ldpc_awgn,
)
from external_baselines.bpg_ldpc.mcs import mcs_from_dict
from external_baselines.shared import sha256_file
from swinjscc_hr_dataset import HRCroppedEvalImageDataset, split_path_list


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Python-controlled BPG+5G LDPC")
    parser.add_argument(
        "--bpgenc",
        default=str(REPO_ROOT / "external_baselines/bpg_ldpc/bin/bpgenc"),
    )
    parser.add_argument(
        "--bpgdec",
        default=str(REPO_ROOT / "external_baselines/bpg_ldpc/bin/bpgdec"),
    )
    parser.add_argument(
        "--mcs-calibration",
        default=str(
            REPO_ROOT
            / "paper_data/updates/bpg_i5_unified_20261104/config/mcs.json"
        ),
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/swinjscc_hr"))
    parser.add_argument("--val-dirs", default="")
    parser.add_argument("--eval-split", default="kodak", choices=("kodak", "clic2021", "combined"))
    parser.add_argument("--rates", default="32,64,96,128,192")
    parser.add_argument("--snr-db-values", default="1,4,7,10,13")
    parser.add_argument("--eval-crop-multiple", default=128, type=int)
    parser.add_argument("--noise-realizations", default=3, type=int)
    parser.add_argument("--seed", default=20260811, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder-iterations", default=5, type=int)
    parser.add_argument("--encoder-mode", default=8, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--minimum-qp", default=0, type=int)
    parser.add_argument("--maximum-qp", default=51, type=int)
    parser.add_argument("--report-ms-ssim", action="store_true")
    parser.add_argument("--save-image-id", default="")
    parser.add_argument("--reconstruction-dir", default="")
    parser.add_argument("--require-idle-gpu", action="store_true")
    parser.add_argument("--cache-qp", action="store_true")
    parser.add_argument(
        "--output",
        default="",
    )
    return parser


def _default_eval_dirs(data_dir: str, split: str) -> str:
    root = Path(data_dir)
    paths = {
        "kodak": (root / "Kodak",),
        "clic2021": (root / "CLIC2021/test",),
        "combined": (root / "Kodak", root / "CLIC2021/test"),
    }[split]
    return ",".join(str(path) for path in paths)


def _per_image_psnr(reference: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = ((reference * 255.0 - reconstructed * 255.0) ** 2).mean().clamp_min(1e-10)
    return float(10.0 * torch.log10((255.0**2) / mse))


def _noise_seed(base_seed: int, image_id: str, rate: int, snr_db: float, realization: int) -> int:
    key = f"{base_seed}|{image_id}|{rate}|{snr_db}|{realization}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _aggregate_realizations(image_rows):
    grouped = {}
    for row in image_rows:
        key = (row["rate"], row["snr_db"], row["realization"])
        grouped.setdefault(key, []).append(row)
    results = []
    for (rate, snr_db, realization), rows in sorted(grouped.items()):
        results.append(
            {
                "rate": rate,
                "snr_db": snr_db,
                "realization": realization,
                "psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
                "ssim": float(np.mean([row["ssim"] for row in rows])),
                **({"ms_ssim": float(np.mean([row["ms_ssim"] for row in rows]))}
                   if "ms_ssim" in rows[0] else {}),
                "cbr": float(np.mean([row["actual_cbr"] for row in rows])),
                "target_cbr": rate / 1536.0,
                "decode_failure_rate": float(np.mean([row["decode_failure"] for row in rows])),
                "crc_failure_rate": float(np.mean([row["crc_failure"] for row in rows])),
                "codec_failure_rate": float(np.mean([row["codec_failure"] for row in rows])),
                "budget_failure_rate": float(np.mean([row["budget_failure"] for row in rows])),
                "block_error_rate": float(np.mean([row["block_error_rate"] for row in rows])),
                "num_images": len(rows),
            }
        )
    return results


def _aggregate_conditions(realization_rows):
    grouped = {}
    for row in realization_rows:
        grouped.setdefault((row["rate"], row["snr_db"]), []).append(row)
    return [
        {
            "rate": rate,
            "snr_db": snr_db,
            "psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
            **({"ms_ssim": float(np.mean([row["ms_ssim"] for row in rows]))}
               if "ms_ssim" in rows[0] else {}),
            "cbr": float(np.mean([row["cbr"] for row in rows])),
            "target_cbr": rate / 1536.0,
            "decode_failure_rate": float(np.mean([row["decode_failure_rate"] for row in rows])),
            "crc_failure_rate": float(np.mean([row["crc_failure_rate"] for row in rows])),
            "codec_failure_rate": float(np.mean([row["codec_failure_rate"] for row in rows])),
            "budget_failure_rate": float(np.mean([row["budget_failure_rate"] for row in rows])),
            "block_error_rate": float(np.mean([row["block_error_rate"] for row in rows])),
            "noise_realizations": len(rows),
            "num_images": rows[0]["num_images"],
        }
        for (rate, snr_db), rows in sorted(grouped.items())
    ]


def run(args, *, dataset=None):
    rates = [int(item.strip()) for item in args.rates.split(",") if item.strip()]
    snr_values = [float(item.strip()) for item in args.snr_db_values.split(",") if item.strip()]
    if not rates or not snr_values or args.noise_realizations <= 0:
        raise ValueError("rates, SNR values, and noise realizations must be non-empty/positive")
    if args.save_image_id and not args.reconstruction_dir:
        raise ValueError("save-image-id requires reconstruction-dir")
    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT
        / f"exp_log/decompjscc/baselines/bpg_ldpc/awgn/evaluation/{args.eval_split}_grid.json"
    )
    if output.exists():
        raise FileExistsError(f"Evaluation already exists; use a new output path: {output}")
    if args.save_image_id:
        render_root = Path(args.reconstruction_dir)
        render_paths = [render_root / "original.png"] + [
            render_root / "bpg_ldpc" / f"rate{rate}_snr{snr:g}.png"
            for rate in rates for snr in snr_values
        ]
        for path in render_paths:
            if path.exists():
                raise FileExistsError(f"Reconstruction already exists; use a new directory: {path}")
    calibration_path = Path(args.mcs_calibration)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("test_images_used_for_selection") is not False:
        raise ValueError("MCS calibration must explicitly exclude test images")
    selections = calibration["selection"]
    codec = BPGCodec(args.bpgenc, args.bpgdec, encoder_mode=args.encoder_mode)
    device = torch.device(args.device)
    if args.require_idle_gpu:
        from experiments.evaluation import require_idle_gpu
        require_idle_gpu(device)
    directories = args.val_dirs or _default_eval_dirs(args.data_dir, args.eval_split)
    if dataset is None:
        dataset = HRCroppedEvalImageDataset(
            split_path_list(directories), crop_multiple=args.eval_crop_multiple)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    image_rows = []
    qp_search_rows = []
    for image_index, (reference_batch, image_ids) in enumerate(loader):
        if args.max_images and image_index >= args.max_images:
            break
        if args.require_idle_gpu:
            require_idle_gpu(device)
        image_id = str(image_ids[0])
        reference = reference_batch.to(device)
        _, _, height, width = reference.shape
        with tempfile.TemporaryDirectory(prefix="decompjscc_bpg_image_") as directory:
            source_png = Path(directory) / "source.png"
            to_pil_image(reference_batch[0]).save(source_png, format="PNG")
            encoded = {}

            def encode_qp(qp):
                if not args.cache_qp:
                    return codec.encode(source_png, qp)
                if qp not in encoded:
                    encoded[qp] = codec.encode(source_png, qp)
                return encoded[qp]

            for rate in rates:
                max_channel_uses = math.floor((rate / 1536.0) * reference[0].numel())
                for snr_db in snr_values:
                    selection_key = str(float(snr_db))
                    if selection_key not in selections:
                        raise KeyError(f"No calibrated MCS for SNR {snr_db}")
                    mcs = mcs_from_dict(selections[selection_key])
                    qp_result = search_bpg_qp(
                        encode_qp,
                        max_channel_uses=max_channel_uses,
                        mcs=mcs,
                        minimum_qp=args.minimum_qp,
                        maximum_qp=args.maximum_qp,
                    )
                    qp_search_rows.append(
                        {
                            "image_id": image_id,
                            "rate": rate,
                            "snr_db": snr_db,
                            "channel_use_budget": max_channel_uses,
                            "selected_qp": qp_result.qp,
                            "selected_channel_uses": qp_result.channel_uses,
                            "trials": list(qp_result.trials),
                        }
                    )
                    for realization in range(args.noise_realizations):
                        budget_failure = qp_result.bitstream is None
                        crc_failure = False
                        codec_failure = False
                        link_stats = None
                        reconstructed = torch.zeros_like(reference)
                        seed = _noise_seed(args.seed, image_id, rate, snr_db, realization)
                        if not budget_failure:
                            payload_bits = bytes_to_bits(qp_result.bitstream, device=device)
                            link = run_5g_ldpc_awgn(
                                payload_bits,
                                information_block_bits=mcs.information_block_bits,
                                coded_block_bits=mcs.coded_block_bits,
                                bits_per_symbol=mcs.bits_per_symbol,
                                snr_db=snr_db,
                                seed=seed,
                                decoder_iterations=args.decoder_iterations,
                                crc_bits=24,
                                device=device,
                            )
                            link_stats = link.stats
                            crc_failure = not link.stats.crc_pass
                            if link.stats.decode_success:
                                try:
                                    decoded_bytes = bits_to_bytes(
                                        link.decoded_bits,
                                        valid_bits=len(qp_result.bitstream) * 8,
                                    )
                                    decoded_image = codec.decode(decoded_bytes)
                                    decoded_tensor = pil_to_tensor(decoded_image).float().div(255.0)
                                    if tuple(decoded_tensor.shape[-2:]) != (height, width):
                                        raise ValueError("decoded BPG image size mismatch")
                                    reconstructed = decoded_tensor.unsqueeze(0).to(device)
                                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                                    codec_failure = True
                        decode_failure = budget_failure or crc_failure or codec_failure
                        psnr_db = _per_image_psnr(reference, reconstructed)
                        ssim = float(compute_ssim_batch(reference, reconstructed, device).mean())
                        extra_metrics = (
                            {"ms_ssim": float(compute_ms_ssim_batch(reference, reconstructed)[0])}
                            if args.report_ms_ssim else {}
                        )
                        if args.save_image_id == image_id and realization == 0:
                            from experiments.evaluation import save_image
                            render_root = Path(args.reconstruction_dir)
                            save_image(render_root / "original.png", reference)
                            save_image(render_root / "bpg_ldpc" / f"rate{rate}_snr{snr_db:g}.png", reconstructed)
                        channel_uses = 0 if link_stats is None else link_stats.channel_uses
                        payload_bit_count = 0 if qp_result.bitstream is None else len(qp_result.bitstream) * 8
                        image_rows.append(
                            {
                                "image_id": image_id,
                                "rate": rate,
                                "snr_db": snr_db,
                                "realization": realization,
                                "evaluation_seed": seed,
                                "target_cbr": rate / 1536.0,
                                "actual_cbr": channel_uses / reference[0].numel(),
                                "channel_use_budget": max_channel_uses,
                                "channel_uses": channel_uses,
                                "payload_bits": payload_bit_count,
                                "qp": qp_result.qp,
                                "mcs": mcs.name,
                                "bits_per_symbol": mcs.bits_per_symbol,
                                "code_rate": mcs.code_rate,
                                "num_code_blocks": 0 if link_stats is None else link_stats.num_code_blocks,
                                "block_error_rate": (
                                    1.0
                                    if link_stats is None
                                    else link_stats.crc_failed_blocks / link_stats.num_code_blocks
                                ),
                                "budget_failure": budget_failure,
                                "crc_failure": crc_failure,
                                "codec_failure": codec_failure,
                                "decode_failure": decode_failure,
                                "psnr_db": psnr_db,
                                "ssim": ssim,
                                **extra_metrics,
                            }
                        )
        print(
            json.dumps(
                {
                    "processed_images": image_index + 1,
                    "total_images": min(len(dataset), args.max_images or len(dataset)),
                    "last_image_id": image_id,
                }
            ),
            flush=True,
        )

    realization_rows = _aggregate_realizations(image_rows)
    results = _aggregate_conditions(realization_rows)
    payload = {
        "method": "BPG+5G_LDPC",
        "protocol": "decompjscc_unified_awgn_v1",
        "channel": "complex AWGN, unit average symbol power, Es/N0",
        "codec": codec.metadata(),
        "sionna_version": importlib.metadata.version("sionna"),
        "mcs_calibration": str(calibration_path.resolve()),
        "mcs_calibration_sha256": sha256_file(calibration_path),
        "mcs_selected_without_test_images": True,
        "crc": "CRC24A per LDPC information block",
        "failed_reconstruction": "all-zero RGB image included in PSNR/SSIM",
        "decompjscc_support_mask_in_payload_cbr": False,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "noise_realizations": args.noise_realizations,
        "results": results,
        "realization_results": realization_rows,
        "per_image_results": image_rows,
        "qp_search_results": qp_search_rows,
        **({"ms_ssim": ms_ssim_metadata()} if args.report_ms_ssim else {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    processed_images = len({row["image_id"] for row in image_rows})
    print(
        json.dumps(
            {"output": str(output), "conditions": len(results), "images": processed_images},
            indent=2,
        )
    )
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
