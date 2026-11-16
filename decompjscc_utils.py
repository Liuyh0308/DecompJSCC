"""Shared paths, tensor helpers, metrics, and checkpoint I/O for DecompJSCC."""

from __future__ import annotations

import hashlib
import inspect
import math
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:  # pragma: no cover - optional evaluation dependency
    ssim_fn = None


REPO_ROOT = Path(__file__).resolve().parent
MAIN_CHECKPOINT = REPO_ROOT / "checkpoints/decompjscc/awgn/model_best.pth"
MAIN_EXPERIMENT_DIR = REPO_ROOT / "exp_log/decompjscc/main"
MAIN_RUN_DIR = REPO_ROOT / "runs/decompjscc/main"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_implementation_metadata(model) -> dict:
    """Describe the exact shared source-channel implementation used by a model."""

    channel = model.net.channel
    channel_class = type(channel)
    source_file = inspect.getsourcefile(channel_class)
    if source_file is None:
        raise RuntimeError(f"Cannot resolve source for channel class {channel_class!r}")
    source_path = Path(source_file).resolve()
    try:
        relative_source = source_path.relative_to(REPO_ROOT)
    except ValueError:
        relative_source = source_path
    channel_type = str(channel.chan_type)
    metadata = {
        "class_name": channel_class.__name__,
        "module": channel_class.__module__,
        "source_path": str(relative_source),
        "source_sha256": sha256_file(source_path),
        "channel_type": channel_type,
        "complex_packing": "flattened tensor first half as real and second half as imaginary",
        "power_normalization": "unit average complex power before channel",
    }
    if channel_type == "rayleigh":
        metadata["rayleigh_protocol"] = {
            "fading": "iid nonnegative Rayleigh magnitude per complex symbol with E[h^2]=1",
            "random_phase": False,
            "receiver_csi": False,
            "equalization": "none",
            "noise_sigma": "sqrt(1/(2*10^(snr_db/10))) per real component",
        }
    return metadata


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def iter_limited(loader: Iterable, max_batches: Optional[int]) -> Iterator:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def expand_scalar_like(
    value,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float = 0.0,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    if isinstance(value, torch.Tensor):
        output = value.to(device=device, dtype=dtype).reshape(-1)
        if output.numel() == 1:
            output = output.expand(batch_size)
        if output.numel() != batch_size:
            raise ValueError(f"condition batch mismatch: expected {batch_size}, got {output.numel()}")
        return output
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def expand_rate_like(
    value,
    batch_size: int,
    device: torch.device,
    default: int,
    max_value: Optional[int] = None,
) -> torch.Tensor:
    if value is None:
        output = torch.full((batch_size,), int(default), device=device, dtype=torch.long)
    elif isinstance(value, torch.Tensor):
        output = value.to(device=device, dtype=torch.long).reshape(-1)
        if output.numel() == 1:
            output = output.expand(batch_size)
        if output.numel() != batch_size:
            raise ValueError(f"rate batch mismatch: expected {batch_size}, got {output.numel()}")
    else:
        output = torch.full((batch_size,), int(value), device=device, dtype=torch.long)
    output = output.clamp_min(1)
    if max_value is not None:
        output = output.clamp_max(int(max_value))
    return output


def build_hard_channel_mask(scores: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Select exactly top-k latent channels for each sample."""

    batch_size, num_channels = scores.shape
    rate = rate.to(device=scores.device, dtype=torch.long).reshape(batch_size).clamp_(1, num_channels)
    indices = scores.argsort(dim=1, descending=True)
    rank = torch.arange(num_channels, device=scores.device).unsqueeze(0).expand(batch_size, -1)
    sorted_mask = (rank < rate.unsqueeze(1)).to(dtype=scores.dtype)
    mask = torch.zeros_like(scores)
    mask.scatter_(1, indices, sorted_mask)
    return mask


def compute_psnr_from_pixel_mse(mse: torch.Tensor) -> float:
    value = float(mse.detach().mean().clamp_min(1e-10).item())
    return float(10.0 * math.log10(255.0 * 255.0 / value))


def compute_per_image_psnr(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(reference * 255.0, reconstructed * 255.0, reduction="none")
    mse = mse.flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10((255.0**2) / mse)


def compute_ssim_batch(reference: torch.Tensor, reconstructed: torch.Tensor, device: torch.device) -> torch.Tensor:
    if ssim_fn is None:
        return torch.full((reference.shape[0],), 0.5, device=device)
    reference_np = reference.detach().cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_np = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 1)
    height, width = reference_np.shape[1:3]
    win_size = min(7, height, width)
    win_size = win_size - 1 if win_size % 2 == 0 else win_size
    win_size = max(3, win_size)
    values = [
        ssim_fn(reference_np[index], reconstructed_np[index], channel_axis=2, data_range=1.0, win_size=win_size)
        for index in range(reference_np.shape[0])
    ]
    return torch.tensor(values, device=device)


def ms_ssim_metadata() -> dict:
    import importlib.metadata

    return {
        "implementation": "pytorch_msssim.ms_ssim",
        "version": importlib.metadata.version("pytorch-msssim"),
        "scales": 5,
        "weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        "window_size": 11,
        "window_sigma": 1.5,
        "data_range": 1.0,
        "color_space": "RGB",
        "aggregation": "per-image score, then arithmetic mean",
        "higher_is_better": True,
        "training_loss_changed": False,
    }


@torch.no_grad()
def compute_ms_ssim_batch(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Standard five-scale scores, separate from the source model's training loss."""
    from pytorch_msssim import ms_ssim

    if reference.shape != reconstructed.shape or reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("MS-SSIM requires matching NCHW RGB tensors")
    if min(reference.shape[-2:]) <= 160:
        raise ValueError("Five-scale MS-SSIM with window 11 requires both dimensions > 160")
    return ms_ssim(reference, reconstructed, data_range=1.0, size_average=False)


def get_training_output_dirs(output_root: str | Path, channel_type: str, stage: str) -> dict[str, Path]:
    """Return the standard checkpoint, experiment-log, and TensorBoard directories."""

    root = Path(output_root)
    suffix = Path("decompjscc/training") / channel_type / stage
    return {
        "checkpoint": root / "checkpoints" / suffix,
        "experiment": root / "exp_log" / suffix,
        "run": root / "runs" / suffix,
    }


def load_decompjscc_checkpoint(model, checkpoint_path: str | Path, *, strict: bool = True) -> dict:
    """Load a DecompJSCC checkpoint without rewriting the source file."""

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.12 compatibility
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if strict:
        model.load_state_dict(state_dict, strict=True)
        skipped: list[str] = []
        loaded = len(state_dict)
    else:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        skipped = [key for key in state_dict if key not in compatible]
        model.load_state_dict(compatible, strict=False)
        loaded = len(compatible)
    return {
        "checkpoint_path": str(path),
        "loaded": loaded,
        "skipped": skipped,
        "epoch": checkpoint.get("epoch"),
    }


def capture_training_rng_state() -> dict:
    """Capture the process RNG state needed for exact single-process continuation."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
    }


def restore_training_rng_state(state: dict) -> None:
    required = {"python", "numpy", "torch", "cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"rng_state is missing required keys: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def load_training_state(
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    """Restore optimizer, best metric, and RNG from a resumable checkpoint."""

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.12 compatibility
        checkpoint = torch.load(path, map_location="cpu")
    required = {"epoch", "best_psnr", "optimizer_state_dict", "rng_state"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"checkpoint {path} is not resumable; missing required keys: {', '.join(missing)}"
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)
    restore_training_rng_state(checkpoint["rng_state"])
    return {
        "start_epoch": int(checkpoint["epoch"]),
        "best_psnr": float(checkpoint["best_psnr"]),
    }


def paper_metric_name(name: str) -> str:
    """Convert private backend diagnostic names to the paper terminology."""

    public = name.replace("attnres", "attention_residual")
    public = public.replace("decoder_snr_", "decoder_snr_adapter_")
    public = public.replace("rcsb_active_mean", "rcsb_active_latent_mean")
    return public.replace("rcsb_active_rms", "rcsb_active_latent_rms")


def paper_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {paper_metric_name(key): value for key, value in metrics.items()}
