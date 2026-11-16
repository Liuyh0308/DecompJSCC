"""Shared model loading, metrics and evaluation for all paper neural methods."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import external_baselines.unified_runtime as baseline
from experiments import source_evaluation as source_eval
from experiments.resources import resource_path as resolve_resource_path
from decompjscc_utils import (REPO_ROOT, compute_ms_ssim_batch, compute_per_image_psnr,
    compute_ssim_batch, ms_ssim_metadata, sha256_file)
from models.swinjscc_source_wrapper import load_source_checkpoint
from swinjscc_hr_dataset import HRCroppedEvalImageDataset, split_path_list

ROOT = REPO_ROOT
BUNDLE = ROOT / "paper_data/updates/bpg_i5_unified_20261104"
DEFAULT_SELECTION = ROOT / "paper_data/updates/fig5_20261105/selection.json"
FIXED_SOURCE_METHODS = ("decompjscc_wo_rcsb", "swinjscc_sa")
METHODS = ("decompjscc", "swinjscc_universal", "sa_ra_jscc", "snr_eq_jscc", "adjscc", *FIXED_SOURCE_METHODS)
FIXED_METHODS = (*FIXED_SOURCE_METHODS, "adjscc", "snr_eq_jscc")
FIXED_RATES = (32, 48, 64, 96, 128, 192)
SNR_VALUES = "1,2.5,4,5.5,7,8.5,10,11.5,13"
METRICS = ("psnr_db", "ssim", "ms_ssim", "cbr")
_GPU_PROCESS_IDS = {}
SOURCE_MODELS = {
    "decompjscc_wo_rcsb": {"recipe": "no_modnet_snr_stageall_cross_attnres_decoder_adapter", "model": "SwinJSCC_w/o_SAandRA"},
    "swinjscc_sa": {"recipe": "snr_adaptive", "model": "SwinJSCC_w/_SA"},
}


def parse_rates(value: str) -> list[int]:
    rates = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("rates must contain positive integers")
    return rates

def parse_snr_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("snr-db-values must not be empty")
    return values

def seed_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

def _default_eval_dirs(data_dir: str, split: str) -> str:
    root = Path(data_dir)
    paths = {
        "kodak": (root / "Kodak",),
        "clic2021": (root / "CLIC2021/test",),
        "combined": (root / "Kodak", root / "CLIC2021/test"),
    }[split]
    return ",".join(str(path) for path in paths)

def build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=args.data_dir,
        hr_val_dirs=args.val_dirs,
        hr_eval_split=args.eval_split,
        hr_eval_crop_multiple=args.eval_crop_multiple,
        batch_size=args.batch_size,
    )

def build_eval_loader(args: SimpleNamespace) -> DataLoader:
    directories = args.hr_val_dirs or _default_eval_dirs(args.data_dir, args.hr_eval_split)
    dataset = HRCroppedEvalImageDataset(
        split_path_list(directories),
        crop_multiple=args.hr_eval_crop_multiple,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

def require_idle_gpu(device: torch.device) -> None:
    if device.type != "cuda":
        return
    def process_ids():
        result = subprocess.run(
            ["nvidia-smi", f"--id={device.index or 0}", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}

    key = (os.getpid(), device.index or 0)
    active = process_ids()
    if key not in _GPU_PROCESS_IDS:
        if active:
            raise RuntimeError(f"GPU already in use by processes {sorted(active)}")
        # NVML reports host PIDs, which differ from os.getpid() in this container.
        probe = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        active = process_ids()
        if len(active) != 1:
            raise RuntimeError(f"Cannot safely identify our GPU context: {sorted(active)}")
        _GPU_PROCESS_IDS[key] = next(iter(active))
        del probe
    others = sorted(active - {_GPU_PROCESS_IDS[key]})
    if others:
        raise RuntimeError(f"GPU is in use by other processes {others}; stopping before the next batch")

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def forward_model(method: str, model, images, snr_db: float, rate: int, *, return_aux=False):
    if method in baseline.METHODS:
        return baseline.call_model(method, model, images, snr_db, rate, return_aux=return_aux)
    kwargs = {"snr_db": snr_db, "rate": rate}
    if "return_aux" in inspect.signature(model.forward).parameters:
        kwargs["return_aux"] = return_aux
    return model(images, **kwargs)

def save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(path)

def aggregate_images(rows: list[dict]) -> tuple[dict, list[dict]]:
    if not rows:
        raise ValueError("No evaluated images")
    realizations = []
    for index in sorted({row["realization"] for row in rows}):
        selected = [row for row in rows if row["realization"] == index]
        realizations.append({
            "realization": index, "num_images": len(selected),
            **{key: float(np.mean([row[key] for row in selected])) for key in METRICS},
        })
    image_sets = [{row["image_id"] for row in rows if row["realization"] == value["realization"]}
                  for value in realizations]
    if any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError("Realizations must evaluate identical images")
    result = {
        **{key: float(np.mean([row[key] for row in realizations])) for key in METRICS},
        "num_images": realizations[0]["num_images"], "noise_realizations": len(realizations),
    }
    return result, realizations

def resource_path(value):
    return resolve_resource_path(value, ROOT)

def adopted_evidence(method, rate, *, bundle=None):
    bundle = Path(bundle or BUNDLE)
    index_path = bundle / "provenance/local_source_index.json"
    manifest = json.loads((bundle / "manifest.json").read_text())
    registered = [row for row in manifest["files"]
                  if row["path"] == "provenance/local_source_index.json"]
    if len(registered) != 1 or sha256_file(index_path) != registered[0]["sha256"]:
        raise ValueError("Adopted source index differs from the delivered bundle manifest")
    index = json.loads(index_path.read_text())
    matches = [row for row in index["conditions"]
               if row["method"] == method and "checkpoint_sha256" in row
               and (method not in FIXED_METHODS or row["rate"] == rate)]
    if not matches or len({row["checkpoint_sha256"] for row in matches}) != 1:
        raise ValueError(f"No unique adopted checkpoint identity for {method}, k={rate}")
    source = matches[0]["raw_sources"][0]
    relative = Path(source["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Raw evidence must remain within the data bundle")
    path = bundle / relative
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"Adopted raw evidence SHA mismatch: {path}")
    saved = json.loads(path.read_text())
    identity = saved["identity"]
    expected = matches[0]["checkpoint_sha256"]
    if (saved.get("status") != "complete" or identity["method"] != method
            or identity["rate"] != matches[0]["rate"] or identity["checkpoint_sha256"] != expected
            or saved["checkpoint_sha256"] != expected):
        raise ValueError("Adopted raw model identity differs from its source index")
    return {"saved": saved, "checkpoint_sha256": expected,
            "raw_path": str(path), "raw_sha256": source["sha256"],
            "index_path": str(index_path), "index_sha256": sha256_file(index_path)}

def source_runtime(method, rate, checkpoint, config_path):
    config = source_eval.load_json_config(str(config_path))
    args = source_eval.build_parser().parse_args([
        "--checkpoint-path", str(checkpoint), "--config-path", str(config_path),
        "--rates", str(rate), "--snr-db-values", SNR_VALUES,
        "--batch-size", "1", "--strict-checkpoint",
    ])
    runtime = source_eval.build_source_eval_runtime_args(args, config)
    spec = SOURCE_MODELS[method]
    if (runtime.recipe_stage != spec["recipe"] or runtime.source_model != spec["model"]
            or parse_rates(runtime.C) != [rate] or runtime.channel_type != "awgn"
            or runtime.model_size != "base"):
        raise ValueError("Checkpoint config is not the requested fixed-CBR architecture")
    if method == "decompjscc_wo_rcsb" and (
            runtime.attnres_query_mode != "snr_depth_logit_gate"
            or runtime.attnres_condition_gate_mode != "snr"
            or not runtime.attnres_condition_gate_use_reference
            or runtime.attnres_cross_stage_force_skip
            or runtime.attnres_cross_stage_skip_threshold != 0):
        raise ValueError("The fixed-CBR control must retain the adopted SNR/cross-stage mechanism")
    return runtime, config

def require_new_output(path):
    if Path(path).exists():
        raise FileExistsError(f"Use a new output directory; existing results are read-only: {path}")


def build_model(method, rate, *, checkpoint=None, config=None):
    """Load the adopted weight or an explicitly requested new experiment."""
    evidence = None if checkpoint else adopted_evidence(method, rate)
    saved = evidence["saved"] if evidence else {}
    path = resource_path(checkpoint or saved["checkpoint"])
    checkpoint_sha = sha256_file(path)
    if evidence and checkpoint_sha != evidence["checkpoint_sha256"]:
        raise ValueError(f"The adopted checkpoint changed: {path}")
    record = {"checkpoint": str(path), "checkpoint_sha256": checkpoint_sha}
    if method in FIXED_SOURCE_METHODS:
        config_path = resource_path(config or saved["config_path"])
        config_sha = sha256_file(config_path)
        if evidence and config_sha != saved["config_sha256"]:
            raise ValueError("The adopted fixed-CBR config changed")
        runtime, model_config = source_runtime(method, rate, path, config_path)
        model = source_eval.build_source_model(runtime)
        if (hasattr(model, "rcsb_rate_adapter") or model.net.encoder.head_list.out_features != rate
                or Path(inspect.getfile(model.net.channel.__class__)).resolve()
                != (ROOT / "SwinJSCC_base/channel.py").resolve()):
            raise ValueError("Expected the fixed-width interface and shared source channel")
        if method == "decompjscc_wo_rcsb" and (
                not model.enable_cross_stage_attnres or len(model.encoder_stage_attnres) != 4
                or not hasattr(model, "decoder_snr_adapter")):
            raise ValueError("Missing stage-all/cross-stage attention or decoder SNR adapter")
        loaded = load_source_checkpoint(model, str(path), strict=True)
        record.update(config_path=str(config_path), config_sha256=config_sha,
                      source_checkpoint_load=loaded, model_config=model_config,
                      checkpoint_load={"strict": True, "missing_keys": [], "unexpected_keys": []},
                      training_completion=saved.get("training_completion"), fixed_rate=rate)
    elif method in ("decompjscc", "swinjscc_universal"):
        from experiments import complexity
        config_path = resource_path(config) if config else Path(complexity.ROLE_DEFAULTS[method]["config"])
        args = SimpleNamespace(role=method, image_size=256, model_size="base", batch_size=1, snr_db=7.0)
        model, loaded, model_config = complexity.build_role_model(
            args, [32, 64, 96, 128, 192], path, config_path)
        record.update(checkpoint_load=loaded, model_config=model_config)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("method") != method:
            raise ValueError("Checkpoint method mismatch")
        model_config = dict(payload["model_config"])
        if method == "adjscc":
            if model_config["transmit_channels"] * 16 != rate:
                raise ValueError("ADJSCC checkpoint CBR mismatch")
            model = baseline.ADJSCC(baseline.ADJSCCConfig(**model_config),
                                   channel=baseline.SourceAWGNChannelAdapter())
        elif method == "snr_eq_jscc":
            if model_config["transmit_channels"] != rate:
                raise ValueError("SNR-EQ checkpoint CBR mismatch")
            model_config["pass_channel"] = True
            model = baseline.SNREQJSCC(baseline.SNREQJSCCConfig(**model_config))
        elif method == "sa_ra_jscc":
            model_config["pass_channel"] = True
            model = baseline.SARAJSCC(baseline.SARAJSCCConfig(**model_config))
        else:
            raise ValueError(f"Unknown method: {method}")
        loaded = model.load_state_dict(payload["model_state_dict"], strict=True)
        record.update(model_config=model_config, checkpoint_load={
            "strict": True, "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys})
    record["checkpoint_origin"] = "adopted_paper" if evidence else "explicit_custom_checkpoint"
    if evidence:
        record["adopted_model_evidence"] = {key: value for key, value in evidence.items() if key != "saved"}
    return model.eval(), record


build_existing_model = build_model


def measure_images(args, model, loader, rate, snr):
    """Keep the original realization-first RNG stream and full-image metrics."""
    device = resolve_device(args.device)
    image_rows = []
    with torch.no_grad():
        for realization in range(args.noise_realizations):
            seed_randomness(args.seed + realization)
            for index, (images, ids) in enumerate(loader):
                if args.max_images and index >= args.max_images:
                    break
                if index % 10 == 0:
                    require_idle_gpu(device)
                images = images.to(device)
                output = forward_model(args.method, model, images, snr, rate)
                reconstructed = output["reconstructed"].clamp(0, 1)
                scores = {
                    "psnr_db": float(compute_per_image_psnr(images, reconstructed)[0]),
                    "ssim": float(compute_ssim_batch(images, reconstructed, device)[0]),
                    "ms_ssim": float(compute_ms_ssim_batch(images, reconstructed)[0]),
                    "cbr": float(output["cbr"]),
                }
                if not all(np.isfinite(value) for value in scores.values()):
                    raise ValueError("Nonfinite evaluation metric")
                if not np.isclose(scores["cbr"], rate / 1536, rtol=0, atol=1e-12):
                    raise ValueError("Model output does not use the requested CBR")
                image_rows.append({"image_id": str(ids[0]), "realization": realization,
                                   "evaluation_seed": args.seed + realization, **scores})
                if str(ids[0]) == args.save_image_id and realization == 0:
                    root = Path(args.output_dir) / "reconstructions"
                    save_image(root / "original.png", images)
                    save_image(root / args.method / f"rate{rate}_snr{snr:g}.png", reconstructed)
    return image_rows


def run_grid(args):
    plan = grid_plan(args)
    root = Path(args.output_dir)
    require_new_output(root)
    dataset = HRCroppedEvalImageDataset(
        split_path_list(_default_eval_dirs(args.data_dir, args.eval_split)), crop_multiple=128)
    paths = [Path(path) for path in dataset.image_files[:args.max_images or None]]
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError("Evaluation images require nonempty, unique IDs")
    image_ids = [p.name for p in paths]
    image_hashes = {p.name: sha256_file(p) for p in paths}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = resolve_device(args.device)
    require_idle_gpu(device)
    model, summaries = None, []
    for rate in plan["rates"]:
        if model is None or args.method in FIXED_METHODS:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, record = build_model(args.method, rate, checkpoint=args.checkpoint, config=args.config)
            model = model.to(device)
        for snr in plan["snr_db_values"]:
            started = time.perf_counter()
            rows = measure_images(args, model, loader, rate, snr)
            result, realizations = aggregate_images(rows)
            result.update(rate=rate, snr_db=snr)
            identity = {
                "method": args.method, "eval_split": args.eval_split, "rate": rate, "snr_db": snr,
                "seed": args.seed, "noise_realizations": args.noise_realizations,
                "checkpoint_sha256": record["checkpoint_sha256"], "image_ids": image_ids,
                "image_sha256": image_hashes, "ms_ssim": ms_ssim_metadata(),
                "eval_crop_multiple": 128, "batch_size": 1, "save_image_id": args.save_image_id,
                "evaluator_sha256": sha256_file(Path(__file__)),
                "metric_utils_sha256": sha256_file(ROOT / "decompjscc_utils.py"),
                "channel_sha256": sha256_file(ROOT / "SwinJSCC_base/channel.py"),
            }
            if "config_sha256" in record:
                identity["config_sha256"] = record["config_sha256"]
            payload = {"status": "complete", "identity": identity, **record, "result": result,
                       "realization_results": realizations, "per_image_results": rows,
                       "elapsed_seconds": time.perf_counter() - started,
                       "cross_architecture_common_noise": False,
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
            write_json(root / args.method / args.eval_split / f"rate{rate}_snr{snr:g}.json", payload)
            summaries.append(result)
            print(json.dumps({"method": args.method, "split": args.eval_split, **result}), flush=True)
    result = {"method": args.method, "eval_split": args.eval_split,
              "ms_ssim": ms_ssim_metadata(), "results": summaries}
    write_json(root / args.method / args.eval_split / "summary.json", result)
    return result

def grid_plan(args):
    rates = parse_rates(args.rates)
    snrs = parse_snr_values(args.snr_db_values)
    if not rates or len(set(rates)) != len(rates) or any(rate <= 0 for rate in rates):
        raise ValueError("Choose distinct positive latent widths")
    if (not snrs or len(set(snrs)) != len(snrs) or any(not math.isfinite(snr) for snr in snrs)
            or args.noise_realizations < 1 or args.max_images < 0):
        raise ValueError("Invalid SNR/realization/image selection")
    if args.method not in FIXED_METHODS and any(rate > 320 for rate in rates):
        raise ValueError("The paper joint models have 320 latent channels; requested width is unsupported")
    if args.method in ("adjscc", "snr_eq_jscc") and any(rate not in FIXED_RATES for rate in rates):
        raise ValueError(f"The fixed-CBR evaluator supports only {FIXED_RATES}")
    if bool(args.config) and not args.checkpoint:
        raise ValueError("--config requires an explicit --checkpoint")
    if args.checkpoint:
        if args.method in FIXED_METHODS and len(rates) != 1:
            raise ValueError("A custom fixed-CBR checkpoint requires exactly one --rates value")
        if args.method in (*FIXED_SOURCE_METHODS, "decompjscc", "swinjscc_universal") and not args.config:
            raise ValueError("Source-model checkpoints require an explicit --config")
    elif args.method in FIXED_METHODS and any(rate not in (32, 96, 192) for rate in rates):
        raise ValueError("Adopted fixed-CBR controls exist only at k=32,96,192")
    return {"status": "dry_run", "method": args.method, "dataset": args.eval_split,
            "rates": rates, "snr_db_values": snrs, "noise_realizations": args.noise_realizations,
            "seed": args.seed, "output_dir": str(args.output_dir),
            "checkpoint_origin": "explicit_custom_checkpoint" if args.checkpoint else "adopted_paper"}

def read_selection(path, panels=""):
    data = json.loads(Path(path).read_text())
    if (data.get("seed") != 20260811 or data.get("realization") != 0
            or data.get("dataset") != "kodak" or data.get("eval_crop_multiple") != 128):
        raise ValueError("Reconstruction keeps Kodak, seed20260811, realization0 and crop_multiple128")
    rows = data.get("rows", [])
    names = [row["panel"] for row in rows]
    if not rows or len(set(names)) != len(names) or any(not name.isalnum() for name in names):
        raise ValueError("Selection must have unique alphanumeric panel names")
    selected = panels.split(",") if panels else names
    if len(set(selected)) != len(selected) or any(name not in names for name in selected):
        raise ValueError("Choose existing distinct panel names, separated by commas")
    for row in rows:
        if (type(row["rate"]) is not int or row["rate"] <= 0
                or not math.isfinite(float(row["snr_db"]))):
            raise ValueError("Panel conditions require a positive integer width and finite SNR")
        relative = Path(row["source_image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Selected images must use project-relative paths")
        image = ROOT / relative
        if image.name != row["image_id"] or sha256_file(image) != row["original_sha256"]:
            raise ValueError("Selection image identity changed")
        crop = row["crop_xywh"]
        if len(crop) != 4 or any(type(value) is not int for value in crop):
            raise ValueError("Crop requires four integer coordinates")
        x, y, width, height = crop
        with Image.open(image) as source:
            max_width, max_height = source.width // 128 * 128, source.height // 128 * 128
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > max_width or y + height > max_height:
            raise ValueError("Detail crop is outside the evaluated image")
    return dict(data, rows=[row for row in rows if row["panel"] in selected])

def validate_reconstruction(args, selection):
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError("Choose distinct neural methods")
    if args.part == "bpg":
        if any(row["snr_db"] not in (4, 10) for row in selection["rows"]):
            raise ValueError("The adopted qualitative BPG replay supports the frozen 4/10dB protocol")
        protocol = selection.get("bpg_protocol", {})
        if any(protocol.get(key) != value for key, value in {
            "encoder_mode": 8, "decoder_iterations": 5, "information_block_length": 8352,
            "crc": "CRC24A",
            "calibration_sha256": "6fafac37ccdc945a61fceb04c26a227debbb1de94e9ae26598930b946fc722d7",
        }.items()):
            raise ValueError("Selection changes the adopted BPG+LDPC protocol")
        if args.device != "cuda:0":
            raise ValueError("The BPG replay uses cuda:0 after physical-GPU selection")


BPG_CALIBRATION = ROOT / "paper_data/updates/bpg_i5_unified_20261104/config/mcs.json"
BPG_CALIBRATION_SHA = "6fafac37ccdc945a61fceb04c26a227debbb1de94e9ae26598930b946fc722d7"
BPG_BIN = ROOT / "external_baselines/bpg_ldpc/bin"


def bpg_parser():
    from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep as backend
    parser = backend.build_parser()
    parser.set_defaults(encoder_mode=8, decoder_iterations=5, report_ms_ssim=True,
                        cache_qp=True, require_idle_gpu=True,
                        mcs_calibration=str(BPG_CALIBRATION),
                        bpgenc=str(BPG_BIN / "bpgenc"), bpgdec=str(BPG_BIN / "bpgdec"))
    return parser


def run_bpg(args, *, dataset=None):
    from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep as backend
    from external_baselines.bpg_ldpc.calibration import validate_calibration
    if Path(args.mcs_calibration).resolve() == BPG_CALIBRATION.resolve():
        if sha256_file(BPG_CALIBRATION) != BPG_CALIBRATION_SHA:
            raise ValueError("The adopted AMC calibration changed")
        validate_calibration(json.loads(BPG_CALIBRATION.read_text()))
        if args.encoder_mode != 8 or args.decoder_iterations != 5:
            raise ValueError("The adopted AMC requires BPG mode8 and five decoder iterations")
    # Preserve the four-core codec affinity used for the adopted BPG outputs.
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, sorted(previous)[:4])
    try:
        return backend.run(args, dataset=dataset)
    finally:
        os.sched_setaffinity(0, previous)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grid = commands.add_parser("grid", help="PSNR and RGB five-scale MS-SSIM for any neural method")
    grid.add_argument("--method", choices=METHODS, required=True)
    grid.add_argument("--eval-split", choices=("kodak", "clic2021"), required=True)
    grid.add_argument("--rates", default="32,96,192")
    grid.add_argument("--snr-db-values", default=SNR_VALUES)
    grid.add_argument("--data-dir", default=str(ROOT / "datasets/swinjscc_hr"))
    grid.add_argument("--noise-realizations", type=int, default=3)
    grid.add_argument("--seed", type=int, default=20260811)
    grid.add_argument("--max-images", type=int, default=0)
    grid.add_argument("--save-image-id", default="")
    grid.add_argument("--checkpoint")
    grid.add_argument("--config")
    reconstruction = commands.add_parser("reconstruction", help="Replay selected Kodak panels")
    reconstruction.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    reconstruction.add_argument("--panels", default="", help="Comma-separated selection rows, e.g. a,b")
    reconstruction.add_argument("--part", choices=("neural", "bpg"), default="neural")
    reconstruction.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    for command in (grid, reconstruction):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--run", action="store_true", help="Execute; omitted means a read-only plan")
    bpg = commands.add_parser("bpg", parents=[bpg_parser()], add_help=False,
                              help="BPG+5G LDPC using the frozen AMC or an explicit calibration")
    bpg.add_argument("--run", action="store_true")
    source = commands.add_parser("source", parents=[source_eval.build_parser()], add_help=False,
                                 help="Source-model sweeps, including channel transfer and ablations")
    source.add_argument("--run", action="store_true")
    from external_baselines.bpg_ldpc import calibration
    commands.add_parser("calibrate", parents=[calibration.build_parser()], add_help=False,
                        help="Independent random-bit AMC calibration (no test image metrics)")
    merge = commands.add_parser("merge-calibration", help="Merge the nine completed calibration conditions")
    merge.add_argument("--output-root", type=Path, required=True)
    return parser


def execute(args):
    if args.command == "grid":
        plan = grid_plan(args)
        return run_grid(args) if args.run else plan
    if args.command == "reconstruction":
        selection = read_selection(args.selection, args.panels)
        validate_reconstruction(args, selection)
        if args.run:
            from experiments.reconstruction import run
            return run(args, selection)
        return {"status": "dry_run", "part": args.part, "methods": args.methods,
                "panels": [row["panel"] for row in selection["rows"]],
                "output_dir": str(args.output_dir), "realization": 0}
    if args.command == "bpg":
        if not args.output:
            raise ValueError("Specify a new --output JSON path")
        require_new_output(args.output)
        return run_bpg(args) if args.run else {"status": "dry_run", "arguments": vars(args)}
    if args.command == "source":
        if not args.output_json:
            raise ValueError("Specify a new --output-json path")
        require_new_output(args.output_json)
        return source_eval.run(args) if args.run else {"status": "dry_run", "arguments": vars(args)}
    from external_baselines.bpg_ldpc import calibration
    if args.command == "merge-calibration":
        return calibration.merge_calibration(args.output_root)
    if args.run or args.smoke:
        return calibration.run(args)
    return {"status": "dry_run", "snr_db": args.snr_db, "reliability": calibration.protocol_settings(),
            "candidate_plans": [calibration.candidate_plan(mcs)
                                for mcs in calibration.candidate_order(calibration.candidate_grid())]}
