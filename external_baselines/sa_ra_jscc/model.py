"""Complete PyTorch SA-RA-JSCC model for the unified AWGN protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from external_baselines.sa_ra_jscc.reproduction import SARAConfig, SARALatentConditioner
from external_baselines.shared import pass_source_channel
from models.swinjscc_source_wrapper import (
    SOURCE_MODEL_RA,
    SourceSwinJSCCWrapper,
    build_source_swinjscc_args,
)


@dataclass(frozen=True)
class SARAJSCCConfig:
    image_size: int = 256
    model_size: str = "base"
    feature_dim: int = 320
    mlp_hidden_dim: int = 320
    mask_temperature: float = 0.1
    pass_channel: bool = True


class SARAJSCC(SourceSwinJSCCWrapper):
    """Swin backbone followed by the paper's SNR and semantic-rate modules."""

    def __init__(self, config: SARAJSCCConfig | None = None):
        self.reproduction_config = config or SARAJSCCConfig()
        args = build_source_swinjscc_args(
            recipe_stage="rate_adaptive",
            model=SOURCE_MODEL_RA,
            C="32,64,96,128,192",
            multiple_snr="1,4,7,10,13",
            model_size=self.reproduction_config.model_size,
            img_size=self.reproduction_config.image_size,
        )
        super().__init__(args, pass_channel=self.reproduction_config.pass_channel)
        self.net.H = self.net.W = self.reproduction_config.image_size
        if self.net.encoder.embed_dims[-1] != self.reproduction_config.feature_dim:
            raise ValueError("feature_dim must match the final Swin encoder dimension")
        self.latent_conditioner = SARALatentConditioner(
            SARAConfig(
                feature_dim=self.reproduction_config.feature_dim,
                mlp_hidden_dim=self.reproduction_config.mlp_hidden_dim,
                mask_temperature=self.reproduction_config.mask_temperature,
            )
        )
        for name, parameter in self.net.encoder.named_parameters():
            if name.startswith(("bm_list.", "sm_list.")):
                parameter.requires_grad = False

    def _encode_backbone(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.net.encoder.patch_embed(image)
        for layer in self.net.encoder.layers:
            feature = layer(feature)
        return self.net.encoder.norm(feature)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        rate: int,
        *,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        rate = int(rate)
        if rate <= 0 or rate > self.reproduction_config.feature_dim:
            raise ValueError("rate must be within the final latent channel dimension")
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            scale = 2**self.net.downsample
            self.net.decoder.update_resolution(height // scale, width // scale)
            self.net.H, self.net.W = height, width
        feature = self._encode_backbone(image)
        semantic_rate = rate / 1536.0
        transmitted, conditioner_aux = self.latent_conditioner(
            feature,
            snr_db,
            semantic_rate,
            active_channels=rate,
        )
        hard_mask = conditioner_aux["channel_mask"][:, None, :].expand_as(transmitted)
        active_count = hard_mask.sum().clamp_min(1.0)
        average_power = transmitted.square().sum() / active_count
        noisy_feature = (
            pass_source_channel(
                self.net,
                transmitted,
                snr_db,
                average_power=average_power,
            )
            if self.pass_channel
            else transmitted
        )
        noisy_feature = noisy_feature * hard_mask
        reconstructed = self.net.decoder(noisy_feature, snr_db, self.source_model_name)
        reconstructed_clamped = reconstructed.clamp(0.0, 1.0)
        mse = self.net.squared_difference(image * 255.0, reconstructed_clamped * 255.0).mean()
        loss = self.net.distortion_loss.forward(image, reconstructed_clamped).mean()
        output: dict[str, Any] = {
            "reconstructed": reconstructed,
            "cbr": semantic_rate,
            "snr_db": snr_db,
            "rate": rate,
            "mse": mse,
            "loss": loss,
        }
        if return_aux:
            output["sa_ra_aux"] = conditioner_aux
        return output
