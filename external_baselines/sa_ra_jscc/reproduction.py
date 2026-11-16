"""Paper-specified SNR/rate residual adaptation and hard top-C masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SARAConfig:
    feature_dim: int = 320
    mlp_hidden_dim: int = 320
    mask_temperature: float = 0.1


class ScalarConditionPreprocessor(nn.Module):
    """Three FC layers with ReLU, ReLU, and Sigmoid activations."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor | float, batch_size: int) -> torch.Tensor:
        values = _condition_column(condition, batch_size, self.layers[0].weight)
        return self.layers(values)


class ConditionResidualAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.preprocessor = ScalarConditionPreprocessor(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, channels]")
        condition_vector = self.preprocessor(condition, features.shape[0])
        modulated = features * condition_vector[:, None, :]
        residual = torch.sigmoid(self.mlp(modulated))
        return features + residual, {
            "condition_vector": condition_vector,
            "residual": residual,
        }


class SARALatentConditioner(nn.Module):
    """Apply the paper's SNR module followed by its semantic-rate module."""

    def __init__(self, config: SARAConfig | None = None):
        super().__init__()
        self.config = config or SARAConfig()
        self.snr_adapter = ConditionResidualAdapter(
            self.config.feature_dim,
            self.config.mlp_hidden_dim,
        )
        self.rate_adapter = ConditionResidualAdapter(
            self.config.feature_dim,
            self.config.mlp_hidden_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
        snr_db: torch.Tensor | float,
        semantic_rate: torch.Tensor | float,
        *,
        channel_scores: torch.Tensor | None = None,
        active_channels: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError("the feature channel dimension does not match SARAConfig")
        snr_adapted, snr_aux = self.snr_adapter(features, snr_db)
        rate_adapted, rate_aux = self.rate_adapter(snr_adapted, semantic_rate)
        output = rate_adapted
        aux = {
            "snr_condition_vector": snr_aux["condition_vector"],
            "snr_residual": snr_aux["residual"],
            "rate_condition_vector": rate_aux["condition_vector"],
            "rate_residual": rate_aux["residual"],
        }
        if channel_scores is None and active_channels is None:
            return output, aux
        if active_channels is None:
            raise ValueError("active_channels is required when channel_scores are provided")
        if channel_scores is None:
            channel_scores = rate_aux["residual"].mean(dim=1)
        hard_mask = hard_topk_channel_mask(channel_scores, active_channels)
        mask = straight_through_topk_mask(
            channel_scores,
            hard_mask,
            temperature=self.config.mask_temperature,
        ) if self.training else hard_mask
        output = output * mask[:, None, :]
        aux["channel_scores"] = channel_scores
        aux["channel_mask"] = hard_mask
        aux["channel_mask_surrogate"] = mask
        return output, aux


def hard_topk_channel_mask(channel_scores: torch.Tensor, active_channels: int) -> torch.Tensor:
    if channel_scores.ndim != 2:
        raise ValueError("channel_scores must have shape [batch, channels]")
    channels = channel_scores.shape[1]
    if active_channels <= 0 or active_channels > channels:
        raise ValueError("active_channels must be within [1, channels]")
    indices = torch.topk(channel_scores, active_channels, dim=1, largest=True, sorted=True).indices
    mask = torch.zeros_like(channel_scores)
    return mask.scatter(1, indices, 1.0)


def straight_through_topk_mask(
    channel_scores: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    active_channels = int(hard_mask[0].sum().item())
    threshold = torch.topk(channel_scores, active_channels, dim=1).values[:, -1:]
    soft_mask = torch.sigmoid((channel_scores - threshold) / float(temperature))
    return hard_mask + soft_mask - soft_mask.detach()


def mse_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstruction, target)


def _condition_column(
    condition: torch.Tensor | float,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(condition, device=reference.device, dtype=reference.dtype).reshape(-1)
    if values.numel() == 1:
        return values.reshape(1, 1).expand(batch_size, 1)
    if values.numel() != batch_size:
        raise ValueError("condition must be scalar or contain one value per sample")
    return values.reshape(batch_size, 1)
