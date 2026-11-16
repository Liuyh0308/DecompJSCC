"""Shared runtime helpers for unified-protocol external baselines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from models.swinjscc_source_wrapper import _ensure_swinjscc_base_on_path


OFFICIAL_FIXED_CHECKPOINT = Path("checkpoints/pretraining/swinjscc_fixed/model_best.pth")


class SourceAWGNChannelAdapter(torch.nn.Module):
    """Expose the bundled SwinJSCC AWGN channel to non-Swin baselines."""

    def __init__(self):
        super().__init__()
        _ensure_swinjscc_base_on_path()
        from channel import Channel  # type: ignore

        args = SimpleNamespace(channel_type="awgn", multiple_snr="1,4,7,10,13")
        config = SimpleNamespace(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            CUDA=torch.cuda.is_available(),
            logger=None,
        )
        self.channel = Channel(args, config)

    def feature_pass_channel(
        self,
        feature: torch.Tensor,
        snr_db: float,
        average_power: torch.Tensor | float | bool = False,
    ) -> torch.Tensor:
        return self.channel.forward(feature, snr_db, average_power)

    def forward(
        self,
        features: torch.Tensor,
        snr_db: torch.Tensor | float,
    ) -> torch.Tensor:
        return pass_source_channel(self, features, snr_db)


def pass_source_channel(
    net: torch.nn.Module,
    features: torch.Tensor,
    snr_db: torch.Tensor | float,
    *,
    average_power: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Call the bundled SwinJSCC channel, including per-image conditions."""
    values = torch.as_tensor(snr_db, device=features.device, dtype=features.dtype).reshape(-1)
    if values.numel() == 1:
        if average_power is not None:
            powers = torch.as_tensor(
                average_power,
                device=features.device,
                dtype=features.dtype,
            ).reshape(-1)
            if powers.numel() == features.shape[0] and powers.numel() > 1:
                values = values.expand(features.shape[0])
            elif powers.numel() != 1:
                raise ValueError("average_power must be scalar or contain one value per image")
        if values.numel() == 1:
            avg = False if average_power is None else average_power
            return net.feature_pass_channel(features, float(values.item()), avg)
    if values.numel() != features.shape[0]:
        raise ValueError("snr_db must be scalar or contain one value per image")

    powers = None
    if average_power is not None:
        powers = torch.as_tensor(
            average_power,
            device=features.device,
            dtype=features.dtype,
        ).reshape(-1)
        if powers.numel() != features.shape[0]:
            raise ValueError("average_power must contain one value per image")
    outputs = []
    for index, snr in enumerate(values):
        avg = False if powers is None else powers[index]
        outputs.append(net.feature_pass_channel(features[index : index + 1], float(snr.item()), avg))
    return torch.cat(outputs, dim=0)


def load_compatible_fixed_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path = OFFICIAL_FIXED_CHECKPOINT,
) -> dict[str, Any]:
    """Load every shape-compatible tensor from the same official fixed model."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    backbone_loaded = sum(
        name.startswith(("net.encoder.layers.", "net.decoder.layers."))
        for name in compatible
    )
    if backbone_loaded == 0:
        raise RuntimeError(f"No Swin backbone weights loaded from {path}")
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "source_epoch": int(payload.get("epoch", -1)),
        "loaded_tensors": len(compatible),
        "loaded_backbone_tensors": int(backbone_loaded),
        "missing_tensors": len(incompatible.missing_keys),
        "unexpected_tensors": len(incompatible.unexpected_keys),
        "shape_skipped_tensors": sum(
            name in target_state and target_state[name].shape != tensor.shape
            for name, tensor in source_state.items()
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_vector(
    value: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if values.numel() == 1:
        return values.expand(batch_size)
    if values.numel() != batch_size:
        raise ValueError("condition must be scalar or contain one value per image")
    return values
