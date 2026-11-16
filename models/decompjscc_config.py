"""Fixed public configuration for the DecompJSCC paper method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecompJSCCConfig:
    channel_type: str = "awgn"
    rate_channels: tuple[int, ...] = (32, 64, 96, 128, 192)
    snr_db_values: tuple[float, ...] = (1.0, 4.0, 7.0, 10.0, 13.0)
    image_size: int = 256
    model_size: str = "base"
    batch_size: int = 1
    distortion_metric: str = "MSE"
    trainset: str = "DIV2K"
    testset: str = "kodak"
    embed_dims: tuple[int, ...] | None = None
    encoder_depths: tuple[int, ...] | None = None
    decoder_depths: tuple[int, ...] | None = None
    num_heads: tuple[int, ...] | None = None

    METHOD_NAME = "DecompJSCC"
    METHOD_VERSION = "1.0"
    SNR_REFERENCE_DB = 7.0
    SNR_NORMALIZATION_DB = 20.0

    def __post_init__(self) -> None:
        if self.channel_type not in {"awgn", "rayleigh"}:
            raise ValueError(f"Unsupported channel_type={self.channel_type}")
        if not self.rate_channels or any(int(rate) <= 0 for rate in self.rate_channels):
            raise ValueError("rate_channels must contain positive channel budgets")
        if not self.snr_db_values:
            raise ValueError("snr_db_values must not be empty")
        if int(self.image_size) <= 0:
            raise ValueError("image_size must be positive")

    def to_public_metadata(self) -> dict[str, Any]:
        return {
            "method": self.METHOD_NAME,
            "method_version": self.METHOD_VERSION,
            "components": {
                "attention_residual": "SNRConditionedAttentionResidual",
                "snr_conditioning": "SNRDepthLogitConditioner",
                "cross_stage_fusion": "CrossStageAttentionResidual",
                "decoder_snr_adaptation": "DecoderSNRAdapter",
                "rate_control": "RateControlledSparseBottleneck",
                "gain_normalization": "ActiveRMSGainNormalization",
                "power_normalization": "ChannelPowerNormalization",
            },
            "channel_type": self.channel_type,
            "rate_channels": [int(rate) for rate in self.rate_channels],
            "snr_db_values": [float(snr) for snr in self.snr_db_values],
            "snr_reference_db": self.SNR_REFERENCE_DB,
            "snr_normalization_db": self.SNR_NORMALIZATION_DB,
            "image_size": int(self.image_size),
            "model_size": self.model_size,
            "distortion_metric": self.distortion_metric,
        }

    def encoder_depths_list(self) -> list[int]:
        if self.encoder_depths is not None:
            return list(self.encoder_depths)
        return {"small": [2, 2, 2, 2], "base": [2, 2, 6, 2], "large": [2, 2, 18, 2]}[self.model_size]

    def decoder_depths_list(self) -> list[int]:
        if self.decoder_depths is not None:
            return list(self.decoder_depths)
        return {"small": [2, 2, 2, 2], "base": [2, 6, 2, 2], "large": [2, 18, 2, 2]}[self.model_size]
