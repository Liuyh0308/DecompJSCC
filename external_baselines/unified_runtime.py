"""Unified DecompJSCC-data runtime for external neural baselines."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from decompjscc_utils import compute_psnr_from_pixel_mse, compute_ssim_batch, iter_limited
from distributed_utils import add_distributed_args, create_dist_context, setup_seed
from external_baselines.adjscc.reproduction import ADJSCC, ADJSCCConfig
from external_baselines.sa_ra_jscc.model import SARAJSCC, SARAJSCCConfig
from external_baselines.shared import (
    OFFICIAL_FIXED_CHECKPOINT,
    SourceAWGNChannelAdapter,
    load_compatible_fixed_checkpoint,
    sha256_file,
)
from external_baselines.snr_eq_jscc.model import SNREQJSCC, SNREQJSCCConfig
from swinjscc_hr_dataset import HRCroppedEvalImageDataset, HRRandomCropImageDataset, split_path_list


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("adjscc", "snr_eq_jscc", "sa_ra_jscc")
SNR_VALUES = (1.0, 4.0, 7.0, 10.0, 13.0)
RATE_VALUES = (32, 64, 96, 128, 192)


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("rate list must contain positive integers")
    return values


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("SNR list must not be empty")
    return values


def _default_train_dirs(data_dir: str) -> str:
    root = Path(data_dir)
    return ",".join(
        str(path)
        for path in (
            root / "CLIC2020",
            root / "CLIC2021/train",
            root / "CLIC2021/valid",
            root / "CLIC2022/val",
            root / "DIV2K/DIV2K_train_HR",
            root / "DIV2K/DIV2K_valid_HR",
        )
    )


def _default_eval_dirs(data_dir: str, split: str) -> str:
    root = Path(data_dir)
    paths = {
        "kodak": (root / "Kodak",),
        "clic2021": (root / "CLIC2021/test",),
        "combined": (root / "Kodak", root / "CLIC2021/test"),
    }[split]
    return ",".join(str(path) for path in paths)


def variant_name(method: str, rate: int) -> str:
    if method == "sa_ra_jscc":
        return "joint"
    denominators = {32: 48, 48: 32, 64: 24, 96: 16, 128: 12, 192: 8}
    if rate in denominators:
        return f"cbr_1_{denominators[rate]}"
    return f"rate_{rate}"


def output_dirs(output_root: str | Path, method: str, rate: int) -> dict[str, Path]:
    root = Path(output_root)
    variant = variant_name(method, rate)
    relative = Path("decompjscc/baselines") / method / "awgn" / variant
    return {
        "checkpoint": root / "checkpoints" / relative,
        "experiment": root / "exp_log" / relative,
        "run": root / "runs" / relative,
    }


def build_model(
    method: str,
    *,
    rate: int,
    image_size: int,
    model_size: str,
    pass_channel: bool,
    snr_eq_hidden_dim: int = 16,
    snr_eq_penalty_weight: float = 1e5,
    sa_ra_hidden_dim: int = 320,
    sa_ra_mask_temperature: float = 0.1,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if method == "adjscc":
        if rate % 16:
            raise ValueError("ADJSCC rate must map to an integer transmit channel count")
        config = ADJSCCConfig(transmit_channels=rate // 16)
        channel = SourceAWGNChannelAdapter() if pass_channel else None
        return ADJSCC(config, channel=channel), asdict(config)
    if method == "snr_eq_jscc":
        config = SNREQJSCCConfig(
            transmit_channels=rate,
            image_size=image_size,
            model_size=model_size,
            hidden_dim=snr_eq_hidden_dim,
            penalty_weight=snr_eq_penalty_weight,
            pass_channel=pass_channel,
        )
        return SNREQJSCC(config), asdict(config)
    if method == "sa_ra_jscc":
        config = SARAJSCCConfig(
            image_size=image_size,
            model_size=model_size,
            mlp_hidden_dim=sa_ra_hidden_dim,
            mask_temperature=sa_ra_mask_temperature,
            pass_channel=pass_channel,
        )
        return SARAJSCC(config), asdict(config)
    raise ValueError(f"Unsupported method: {method}")


def call_model(
    method: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    snr_db: torch.Tensor | float,
    rate: int,
    *,
    return_aux: bool,
) -> dict[str, Any]:
    if method == "adjscc":
        return model(images, snr_db, return_aux=return_aux)
    return model(images, snr_db, rate=rate, return_aux=return_aux)


def build_train_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train {method} under the unified DecompJSCC HR/AWGN protocol"
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/swinjscc_hr"))
    parser.add_argument("--hr-train-dirs", default="")
    parser.add_argument("--hr-val-dirs", default="")
    parser.add_argument("--hr-eval-split", default="kodak", choices=("kodak", "clic2021", "combined"))
    parser.add_argument("--hr-crop-size", default=256, type=int)
    parser.add_argument("--hr-eval-crop-multiple", default=128, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--rate", default=192, type=int)
    parser.add_argument("--rates", default="32,64,96,128,192")
    parser.add_argument("--snr-db-values", default="1,4,7,10,13")
    parser.add_argument("--global-batch-size", default=16, type=int)
    parser.add_argument(
        "--micro-batch-size",
        default=16,
        type=int,
        help="Per-rank loader batch; use accumulation only when smaller than effective batch.",
    )
    parser.add_argument("--val-batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--validate-every", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fixed-init-checkpoint", default=str(OFFICIAL_FIXED_CHECKPOINT))
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--max-train-batches", default=0, type=int)
    parser.add_argument("--max-val-batches", default=0, type=int)
    parser.add_argument("--snr-eq-hidden-dim", default=16, type=int)
    parser.add_argument("--snr-eq-penalty-weight", default=1e5, type=float)
    parser.add_argument("--sa-ra-hidden-dim", default=320, type=int)
    parser.add_argument("--sa-ra-mask-temperature", default=0.1, type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-idle-gpu", action="store_true",
                        help="Single-GPU runs only: stop if another compute process appears.")
    add_distributed_args(parser)
    return parser


def _check_training_gpu(args, device: torch.device) -> None:
    if not args.require_idle_gpu:
        return
    if args.dist_mode != "single" or device.type != "cuda":
        raise ValueError("--require-idle-gpu requires single-GPU CUDA training")
    # The evaluator imports this runtime, so reuse its guard through a delayed import.
    from experiments.evaluation import require_idle_gpu
    require_idle_gpu(device)


def _file_manifest(files: list[str]) -> dict[str, Any]:
    normalized = "\n".join(str(Path(path).resolve()) for path in sorted(files))
    return {
        "num_images": len(files),
        "path_list_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def _sample_training_condition(
    method: str,
    snr_values: tuple[float, ...],
    rates: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | float, int]:
    rate = int(random.choice(rates))
    if method == "snr_eq_jscc":
        indices = torch.randint(0, len(snr_values), (batch_size,), device=device)
        table = torch.tensor(snr_values, device=device)
        return table[indices], rate
    return float(random.choice(snr_values)), rate


def _reset_train_resolution(model: torch.nn.Module, crop_size: int) -> None:
    base = model.module if hasattr(model, "module") else model
    net = getattr(base, "net", None)
    if net is None:
        return
    scale = 2**net.downsample
    net.encoder.update_resolution(crop_size, crop_size)
    net.decoder.update_resolution(crop_size // scale, crop_size // scale)
    net.H = net.W = crop_size


def _train_epoch(
    method,
    epoch,
    model,
    loader,
    optimizer,
    snr_values,
    rates,
    args,
    dist_ctx,
    accumulation_steps,
):
    model.train()
    dist_ctx.set_epoch(loader, epoch)
    totals = torch.zeros(3, device=dist_ctx.device)
    count = 0
    usable_micro_batches = len(loader) // accumulation_steps * accumulation_steps
    if args.max_train_batches:
        usable_micro_batches = min(
            usable_micro_batches,
            args.max_train_batches * accumulation_steps,
        )
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= usable_micro_batches:
            break
        if batch_index % 10 == 0:
            _check_training_gpu(args, dist_ctx.device)
        images = images.to(dist_ctx.device)
        snr_db, rate = _sample_training_condition(
            method,
            snr_values,
            rates,
            images.shape[0],
            dist_ctx.device,
        )
        should_step = (batch_index + 1) % accumulation_steps == 0
        sync_context = nullcontext()
        if not should_step and hasattr(model, "no_sync"):
            sync_context = model.no_sync()
        with sync_context:
            output = call_model(method, model, images, snr_db, rate, return_aux=True)
            (output["loss"] / accumulation_steps).backward()
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals += torch.tensor(
            [
                float(output["loss"].detach().item()),
                compute_psnr_from_pixel_mse(output["mse"].detach()),
                float(output["cbr"]),
            ],
            device=dist_ctx.device,
        )
        count += 1
    metrics = dist_ctx.reduce_mean(totals / max(count, 1))
    return {"loss": float(metrics[0]), "psnr": float(metrics[1]), "cbr": float(metrics[2])}


def _validate(method, model, loader, snr_values, rates, args, dist_ctx):
    base = model.module if hasattr(model, "module") else model
    totals = {"psnr": 0.0, "ssim": 0.0, "cbr": 0.0}
    points = 0
    if dist_ctx.is_main:
        base.eval()
        with torch.no_grad():
            for rate in rates:
                for snr_db in snr_values:
                    point = {"psnr": 0.0, "ssim": 0.0, "cbr": 0.0}
                    batches = 0
                    for batch_index, (images, _) in enumerate(loader):
                        if batch_index % 10 == 0:
                            _check_training_gpu(args, dist_ctx.device)
                        images = images.to(dist_ctx.device)
                        output = call_model(
                            method,
                            base,
                            images,
                            float(snr_db),
                            int(rate),
                            return_aux=False,
                        )
                        reconstructed = output["reconstructed"].clamp(0.0, 1.0)
                        point["psnr"] += compute_psnr_from_pixel_mse(output["mse"])
                        point["ssim"] += float(
                            compute_ssim_batch(images.clamp(0.0, 1.0), reconstructed, dist_ctx.device).mean()
                        )
                        point["cbr"] += float(output["cbr"])
                        batches += 1
                        if args.max_val_batches and batch_index + 1 >= args.max_val_batches:
                            break
                    for name in totals:
                        totals[name] += point[name] / max(batches, 1)
                    points += 1
        totals = {name: value / max(points, 1) for name, value in totals.items()}
    if dist_ctx.mode.value == "ddp":
        dist_ctx.barrier()
    return totals


def run_training(method: str, argv: list[str] | None = None) -> None:
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    args = build_train_parser(method).parse_args(argv)
    snr_values = parse_floats(args.snr_db_values)
    configured_rates = parse_ints(args.rates) if method == "sa_ra_jscc" else (int(args.rate),)
    dirs = output_dirs(args.output_root, method, args.rate)
    dry_metadata = {
        "method": method,
        "protocol": "decompjscc_unified_awgn_v1",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "global_batch_size": args.global_batch_size,
        "seed": args.seed,
        "snr_db_values": snr_values,
        "rates": configured_rates,
        "output_dirs": {name: str(path) for name, path in dirs.items()},
    }
    if args.dry_run:
        print(json.dumps(dry_metadata, indent=2))
        return
    if args.epochs <= 0 or args.validate_every <= 0:
        raise ValueError("epochs and validate-every must be positive")

    dist_ctx = create_dist_context(args)
    _check_training_gpu(args, dist_ctx.device)
    if args.global_batch_size % dist_ctx.world_size:
        raise ValueError("global-batch-size must be divisible by the DDP world size")
    local_effective_batch = args.global_batch_size // dist_ctx.world_size
    if args.micro_batch_size <= 0 or local_effective_batch % args.micro_batch_size:
        raise ValueError("micro-batch-size must be positive and divide the per-rank effective batch")
    local_batch_size = args.micro_batch_size
    accumulation_steps = local_effective_batch // local_batch_size
    setup_seed(args.seed, dist_ctx)
    process_seed = args.seed + int(dist_ctx.local_rank)
    random.seed(process_seed)
    np.random.seed(process_seed)

    train_dirs = args.hr_train_dirs or _default_train_dirs(args.data_dir)
    val_dirs = args.hr_val_dirs or _default_eval_dirs(args.data_dir, args.hr_eval_split)
    train_dataset = HRRandomCropImageDataset(split_path_list(train_dirs), crop_size=args.hr_crop_size)
    val_dataset = HRCroppedEvalImageDataset(
        split_path_list(val_dirs),
        crop_multiple=args.hr_eval_crop_multiple,
    )
    train_loader = dist_ctx.get_dataloader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        is_train=True,
    )
    val_loader = dist_ctx.get_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        is_train=False,
    )

    model, model_config = build_model(
        method,
        rate=args.rate,
        image_size=args.image_size,
        model_size=args.model_size,
        pass_channel=True,
        snr_eq_hidden_dim=args.snr_eq_hidden_dim,
        snr_eq_penalty_weight=args.snr_eq_penalty_weight,
        sa_ra_hidden_dim=args.sa_ra_hidden_dim,
        sa_ra_mask_temperature=args.sa_ra_mask_temperature,
    )
    model = model.to(dist_ctx.device)
    initialization = None
    start_epoch = 0
    best_psnr = float("-inf")
    best_epoch = 0
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        start_epoch = int(resume_payload["epoch"])
        best_psnr = float(resume_payload.get("best_validation_psnr", float("-inf")))
        best_epoch = int(resume_payload.get("best_validation_epoch", 0))
    elif method in {"snr_eq_jscc", "sa_ra_jscc"}:
        initialization = load_compatible_fixed_checkpoint(model, args.fixed_init_checkpoint)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model = dist_ctx.wrap_model(model)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    if resume_payload is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    metadata = {
        **dry_metadata,
        "model_config": model_config,
        "local_batch_size": local_batch_size,
        "local_effective_batch_size": local_effective_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "world_size": dist_ctx.world_size,
        "optimizer": "Adam",
        "validation_every_epochs": args.validate_every,
        "checkpoint_selection": "maximum Kodak validation PSNR over evaluated epochs",
        "max_train_optimizer_steps_per_epoch": args.max_train_batches or None,
        "max_validation_batches": args.max_val_batches or None,
        "torch_version": torch.__version__,
        "fixed_initialization": initialization,
        "resume_checkpoint": args.resume_checkpoint or None,
        "data": {
            "train_dirs": train_dirs,
            "validation_dirs": val_dirs,
            "crop_size": args.hr_crop_size,
            "eval_crop_multiple": args.hr_eval_crop_multiple,
            "train": _file_manifest(train_dataset.image_files),
            "validation": _file_manifest(val_dataset.image_files),
        },
        "channel": "SwinJSCC_base.channel.Channel/AWGN",
        "support_mask_in_payload_cbr": False,
        "require_idle_gpu": args.require_idle_gpu,
    }
    writer = None
    if dist_ctx.is_main:
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        _write_json(dirs["experiment"] / "training_metadata.json", metadata)
        writer = SummaryWriter(dirs["run"] / time.strftime("%Y%m%d_%H%M%S"))

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        train_metrics = _train_epoch(
            method,
            epoch,
            model,
            train_loader,
            optimizer,
            snr_values,
            configured_rates,
            args,
            dist_ctx,
            accumulation_steps,
        )
        should_validate = (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs
        val_metrics = (
            _validate(method, model, val_loader, snr_values, configured_rates, args, dist_ctx)
            if should_validate
            else {"psnr": float("nan"), "ssim": float("nan"), "cbr": float("nan")}
        )
        _reset_train_resolution(model, args.hr_crop_size)
        epoch_wall_seconds = time.time() - epoch_started
        if dist_ctx.is_main:
            if should_validate and val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                best_epoch = epoch + 1
            state = {
                "method": method,
                "protocol": "decompjscc_unified_awgn_v1",
                "epoch": epoch + 1,
                "model_state_dict": dist_ctx.get_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "training_metadata": metadata,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "best_validation_psnr": best_psnr,
                "best_validation_epoch": best_epoch,
            }
            torch.save(state, dirs["checkpoint"] / "model_last.pth")
            if should_validate and val_metrics["psnr"] == best_psnr:
                torch.save(state, dirs["checkpoint"] / "model_best.pth")
            _write_json(
                dirs["experiment"] / "latest_metrics.json",
                {
                    "method": method,
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "best_validation_psnr": best_psnr,
                    "best_validation_epoch": best_epoch,
                    "epoch_wall_seconds": epoch_wall_seconds,
                },
            )
            _append_json_line(
                dirs["experiment"] / "training_history.jsonl",
                {
                    "method": method,
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "best_validation_psnr": best_psnr,
                    "best_validation_epoch": best_epoch,
                    "epoch_wall_seconds": epoch_wall_seconds,
                },
            )
            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch + 1)
                if should_validate:
                    for name, value in val_metrics.items():
                        writer.add_scalar(f"validation/{name}", value, epoch + 1)
            print(
                f"{method} epoch {epoch + 1}/{args.epochs}: "
                f"loss={train_metrics['loss']:.4f}, train_psnr={train_metrics['psnr']:.3f}, "
                f"val_psnr={val_metrics['psnr']:.3f}, "
                f"wall={epoch_wall_seconds:.1f}s",
                flush=True,
            )
        dist_ctx.barrier()

    if writer is not None:
        writer.close()
    dist_ctx.cleanup()


def build_eval_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate {method} under the unified DecompJSCC HR/AWGN protocol"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/swinjscc_hr"))
    parser.add_argument("--val-dirs", default="")
    parser.add_argument("--eval-split", default="kodak", choices=("kodak", "clic2021", "combined"))
    parser.add_argument("--rate", default=192, type=int)
    parser.add_argument("--rates", default="32,64,96,128,192")
    parser.add_argument("--snr-db-values", default="1,4,7,10,13")
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--eval-crop-multiple", default=128, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--noise-realizations", default=3, type=int)
    parser.add_argument("--seed", default=20260811, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-val-batches", default=0, type=int)
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _seed_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _per_image_psnr(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = ((reference * 255.0 - reconstructed * 255.0) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10((255.0**2) / mse)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["rate"]), float(row["snr_db"])), []).append(row)
    return [
        {
            "rate": rate,
            "snr_db": snr,
            "psnr_db": float(np.mean([row["psnr_db"] for row in values])),
            "ssim": float(np.mean([row["ssim"] for row in values])),
            "cbr": float(np.mean([row["cbr"] for row in values])),
            "noise_realizations": len(values),
            "num_images": int(values[0]["num_images"]),
        }
        for (rate, snr), values in sorted(grouped.items())
    ]


def run_evaluation(method: str, argv: list[str] | None = None) -> Path:
    args = build_eval_parser(method).parse_args(argv)
    rates = parse_ints(args.rates) if method == "sa_ra_jscc" else (args.rate,)
    snr_values = parse_floats(args.snr_db_values)
    dirs = output_dirs(args.output_root, method, args.rate)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else dirs["checkpoint"] / "model_best.pth"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("method") != method:
        raise ValueError(f"Checkpoint method {payload.get('method')} does not match {method}")
    config = dict(payload["model_config"])
    if method == "adjscc":
        config_obj = ADJSCCConfig(**config)
        model = ADJSCC(config_obj, channel=SourceAWGNChannelAdapter())
    elif method == "snr_eq_jscc":
        config["pass_channel"] = True
        model = SNREQJSCC(SNREQJSCCConfig(**config))
    else:
        config["pass_channel"] = True
        model = SARAJSCC(SARAJSCCConfig(**config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = _resolve_device(args.device)
    model = model.to(device).eval()
    directories = args.val_dirs or _default_eval_dirs(args.data_dir, args.eval_split)
    dataset = HRCroppedEvalImageDataset(
        split_path_list(directories),
        crop_multiple=args.eval_crop_multiple,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    condition_rows = []
    image_rows = []
    with torch.no_grad():
        for rate in rates:
            for snr_db in snr_values:
                for realization in range(args.noise_realizations):
                    evaluation_seed = args.seed + realization
                    _seed_randomness(evaluation_seed)
                    totals = {"psnr_db": 0.0, "ssim": 0.0, "cbr": 0.0}
                    current_images = []
                    for batch_index, (images, image_ids) in enumerate(loader):
                        images = images.to(device)
                        output = call_model(
                            method,
                            model,
                            images,
                            float(snr_db),
                            int(rate),
                            return_aux=False,
                        )
                        reconstructed = output["reconstructed"].clamp(0.0, 1.0)
                        psnr = _per_image_psnr(images.clamp(0.0, 1.0), reconstructed)
                        ssim = compute_ssim_batch(images.clamp(0.0, 1.0), reconstructed, device)
                        for index, image_id in enumerate(image_ids):
                            row = {
                                "image_id": str(image_id),
                                "rate": int(rate),
                                "snr_db": float(snr_db),
                                "realization": realization,
                                "evaluation_seed": evaluation_seed,
                                "psnr_db": float(psnr[index]),
                                "ssim": float(ssim[index]),
                                "cbr": float(output["cbr"]),
                            }
                            current_images.append(row)
                            image_rows.append(row)
                            for name in totals:
                                totals[name] += row[name]
                        if args.max_val_batches and batch_index + 1 >= args.max_val_batches:
                            break
                    if not current_images:
                        raise RuntimeError("Evaluation loader produced no images")
                    condition_rows.append(
                        {
                            "rate": int(rate),
                            "snr_db": float(snr_db),
                            "realization": realization,
                            "evaluation_seed": evaluation_seed,
                            **{name: value / len(current_images) for name, value in totals.items()},
                            "num_images": len(current_images),
                        }
                    )
    result = {
        "method": method,
        "protocol": "decompjscc_unified_awgn_v1",
        "channel": "SwinJSCC_base.channel.Channel/AWGN",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_split": args.eval_split,
        "seed": args.seed,
        "noise_realizations": args.noise_realizations,
        "cross_architecture_common_noise": False,
        "support_mask_in_payload_cbr": False,
        "results": _aggregate(condition_rows),
        "realization_results": condition_rows,
        "per_image_results": image_rows,
    }
    output_path = (
        Path(args.output)
        if args.output
        else dirs["experiment"] / "evaluation" / f"{args.eval_split}_grid.json"
    )
    _write_json(output_path, result)
    print(json.dumps({"method": method, "output": str(output_path), "conditions": len(result["results"])}, indent=2))
    return output_path
