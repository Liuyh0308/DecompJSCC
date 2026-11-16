"""Shared parsing, tensor, and metric helpers for the SwinJSCC backend."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Union

import torch

try:
    from skimage.metrics import structural_similarity as ssim_fn

    SSIM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ssim_fn = None
    SSIM_AVAILABLE = False


ScalarLike = Union[float, torch.Tensor]
RateLike = Union[int, torch.Tensor]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_snr_list(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("snr step must be > 0")
    if end < start:
        raise ValueError("snr end must be >= start")
    values = []
    current = float(start)
    limit = float(end) + float(step) * 1e-6
    while current <= limit:
        values.append(round(current, 6))
        current += float(step)
    return values


def expand_scalar_like(
    value: Optional[ScalarLike],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float = 0.0,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    if isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=dtype).reshape(-1)
        if out.numel() == 1:
            out = out.expand(batch_size)
        if out.numel() != batch_size:
            raise ValueError(f"condition batch mismatch: expected {batch_size}, got {out.numel()}")
        return out
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def expand_rate_like(
    value: Optional[RateLike],
    batch_size: int,
    device: torch.device,
    default: int,
    max_value: Optional[int] = None,
) -> torch.Tensor:
    if value is None:
        out = torch.full((batch_size,), int(default), device=device, dtype=torch.long)
    elif isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=torch.long).reshape(-1)
        if out.numel() == 1:
            out = out.expand(batch_size)
        if out.numel() != batch_size:
            raise ValueError(f"rate batch mismatch: expected {batch_size}, got {out.numel()}")
    else:
        out = torch.full((batch_size,), int(value), device=device, dtype=torch.long)
    out = out.clamp_min(1)
    if max_value is not None:
        out = out.clamp_max(int(max_value))
    return out


def build_hard_channel_mask(scores: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Select each sample's top-k channels according to its rate budget."""
    batch_size, num_channels = scores.shape
    rate = rate.to(device=scores.device, dtype=torch.long).reshape(batch_size).clamp_(1, num_channels)
    _, indices = scores.sort(dim=1, descending=True)
    rank = torch.arange(num_channels, device=scores.device).unsqueeze(0).expand(batch_size, -1)
    sorted_mask = (rank < rate.unsqueeze(1)).to(dtype=scores.dtype)
    mask = torch.zeros_like(scores)
    mask.scatter_(1, indices, sorted_mask)
    return mask


def iter_limited(loader: Iterable, max_batches: Optional[int]) -> Iterator:
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


def compute_ssim_batch(img1: torch.Tensor, img2: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not SSIM_AVAILABLE:
        return torch.ones(img1.shape[0], device=device) * 0.5

    values = []
    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    height, width = img1_np.shape[1], img1_np.shape[2]
    win_size = min(7, min(height, width))
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)

    for idx in range(img1_np.shape[0]):
        try:
            val = ssim_fn(
                img1_np[idx],
                img2_np[idx],
                channel_axis=2,
                data_range=1.0,
                win_size=win_size,
            )
        except TypeError:  # pragma: no cover - older skimage
            val = ssim_fn(
                img1_np[idx],
                img2_np[idx],
                multichannel=True,
                data_range=1.0,
                win_size=win_size,
            )
        values.append(val)
    return torch.tensor(values, device=device)
