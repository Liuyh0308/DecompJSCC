"""Complete PyTorch SNR-EQ-JSCC model for the unified AWGN protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from external_baselines.shared import condition_vector, pass_source_channel
from external_baselines.snr_eq_jscc.reproduction import (
    SNREQConfig,
    SNRConditionedWindowAttention,
    snr_eq_loss,
)
from models.swinjscc_source_wrapper import (
    SOURCE_MODEL_WO_SA_RA,
    SourceSwinJSCCWrapper,
    build_source_swinjscc_args,
)


@dataclass(frozen=True)
class SNREQJSCCConfig:
    transmit_channels: int = 192
    image_size: int = 256
    model_size: str = "base"
    hidden_dim: int = 16
    penalty_weight: float = 1e5
    pass_channel: bool = True

    @property
    def cbr(self) -> float:
        return self.transmit_channels / 1536.0


class SNREQJSCC(SourceSwinJSCCWrapper):
    """SwinJSCC without Channel ModNet, using SNR embedding and CAQ."""

    def __init__(self, config: SNREQJSCCConfig | None = None):
        self.reproduction_config = config or SNREQJSCCConfig()
        args = build_source_swinjscc_args(
            recipe_stage="fixed",
            model=SOURCE_MODEL_WO_SA_RA,
            C=str(self.reproduction_config.transmit_channels),
            multiple_snr="1,4,7,10,13",
            model_size=self.reproduction_config.model_size,
            img_size=self.reproduction_config.image_size,
        )
        super().__init__(args, pass_channel=self.reproduction_config.pass_channel)
        self.net.H = self.net.W = self.reproduction_config.image_size
        attention_config = SNREQConfig(
            hidden_dim=self.reproduction_config.hidden_dim,
            penalty_weight=self.reproduction_config.penalty_weight,
        )
        self.snr_eq_attentions = []
        for owner in (self.net.encoder, self.net.decoder):
            for layer in owner.layers:
                for block in layer.blocks:
                    old = block.attn
                    replacement = SNRConditionedWindowAttention(
                        dim=old.dim,
                        window_size=tuple(old.window_size),
                        num_heads=old.num_heads,
                        config=attention_config,
                        qkv_bias=old.qkv.bias is not None,
                        qk_scale=old.scale,
                        attention_dropout=old.attn_drop.p,
                        projection_dropout=old.proj_drop.p,
                    )
                    replacement.relative_position_bias_table.data.copy_(
                        old.relative_position_bias_table.data
                    )
                    replacement.qkv.load_state_dict(old.qkv.state_dict())
                    replacement.proj.load_state_dict(old.proj.state_dict())
                    block.attn = replacement
                    self.snr_eq_attentions.append(replacement)

    def _set_condition(self, snr_db: torch.Tensor | float, batch_size: int) -> None:
        for attention in self.snr_eq_attentions:
            attention.set_condition(snr_db, batch_size=batch_size)

    def _penalty_parameters(
        self,
        snr_db: torch.Tensor | float,
        batch_size: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = []
        biases = []
        for attention in self.snr_eq_attentions:
            if attention.last_aux is None:
                raise RuntimeError("SNR-EQ attention did not record its condition parameters")
            scale = attention.last_aux["query_scale"].reshape(batch_size, -1).mean(dim=1)
            bias = attention.last_aux["query_bias"].reshape(batch_size, -1).mean(dim=1)
            scales.append(scale)
            biases.append(bias)
        condition = condition_vector(
            snr_db,
            batch_size=batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        return condition, torch.stack(scales, dim=1).mean(dim=1), torch.stack(biases, dim=1).mean(dim=1)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        rate: int | None = None,
        *,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        if rate is not None and int(rate) != self.reproduction_config.transmit_channels:
            raise ValueError("SNR-EQ is fixed-rate; rate must match transmit_channels")
        batch_size, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            scale = 2**self.net.downsample
            self.net.decoder.update_resolution(height // scale, width // scale)
            self.net.H, self.net.W = height, width
        self._set_condition(snr_db, batch_size)
        feature = self.net.encoder(
            image,
            snr_db,
            self.reproduction_config.transmit_channels,
            self.source_model_name,
        )
        noisy_feature = (
            pass_source_channel(self.net, feature, snr_db)
            if self.pass_channel
            else feature
        )
        reconstructed = self.net.decoder(noisy_feature, snr_db, self.source_model_name)
        reconstructed_clamped = reconstructed.clamp(0.0, 1.0)
        mse = self.net.squared_difference(image * 255.0, reconstructed_clamped * 255.0).mean()
        base_loss = self.net.distortion_loss.forward(image, reconstructed_clamped).mean()
        condition, query_scale, query_bias = self._penalty_parameters(
            snr_db,
            batch_size,
            image,
        )
        loss, loss_parts = snr_eq_loss(
            base_loss,
            condition,
            query_scale,
            query_bias,
            penalty_weight=self.reproduction_config.penalty_weight,
        )
        output: dict[str, Any] = {
            "reconstructed": reconstructed,
            "cbr": self.reproduction_config.cbr,
            "snr_db": snr_db,
            "mse": mse,
            "loss": loss,
        }
        if return_aux:
            output["snr_eq_aux"] = {
                **loss_parts,
                "query_scale": query_scale,
                "query_bias": query_bias,
            }
        return output
