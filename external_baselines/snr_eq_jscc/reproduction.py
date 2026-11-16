"""Paper-specified SNR embedding, adaptive query, and penalty terms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SNREQConfig:
    hidden_dim: int = 16
    penalty_weight: float = 1e5


class ScalarNetwork(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, *, nonnegative: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.nonnegative = nonnegative

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        output = self.layers(condition)
        return F.relu(output) if self.nonnegative else output


class SNRConditionedWindowAttention(nn.Module):
    """Swin window attention with the SNR-EQ embedding and query equations."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        *,
        config: SNREQConfig | None = None,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.config = config or SNREQConfig()

        relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_positions, num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.snr_embedding = ScalarNetwork(dim, self.config.hidden_dim)
        self.query_scale = ScalarNetwork(1, self.config.hidden_dim, nonnegative=True)
        self.query_bias = ScalarNetwork(1, self.config.hidden_dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        # Keep the source SwinJSCC parameter names so a fixed checkpoint can
        # initialize every compatible attention weight without key rewriting.
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self._condition: torch.Tensor | float | None = None
        self._condition_batch_size: int | None = None
        self.last_aux: dict[str, torch.Tensor] | None = None
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def set_condition(
        self,
        snr_db: torch.Tensor | float,
        *,
        batch_size: int,
    ) -> None:
        self._condition = snr_db
        self._condition_batch_size = int(batch_size)

    def forward(
        self,
        features: torch.Tensor,
        snr_db: torch.Tensor | float | None = None,
        *,
        batch_size: int | None = None,
        mask: torch.Tensor | None = None,
        return_aux: bool = False,
        add_token: bool = False,
        token_num: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.dim:
            raise ValueError("features must have shape [windows*batch, tokens, dim]")
        window_batch, tokens, channels = features.shape
        expected_tokens = self.window_size[0] * self.window_size[1]
        if tokens != expected_tokens:
            raise ValueError("token count must match the configured window size")
        if add_token or token_num:
            raise ValueError("SNR-EQ reproduction only supports source Swin windows without extra tokens")
        condition = self._condition if snr_db is None else snr_db
        condition_batch_size = self._condition_batch_size if batch_size is None else batch_size
        if condition is None:
            raise ValueError("set_condition() or snr_db is required before attention forward")
        snr = _condition_per_window(
            condition,
            window_batch,
            condition_batch_size,
            features.device,
            features.dtype,
        )

        embedded = features + self.snr_embedding(snr)[:, None, :]
        qkv = self.qkv(embedded).reshape(
            window_batch,
            tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query_scale = self.query_scale(snr)
        query_bias = self.query_bias(snr)
        query = query_scale[:, None, None, :] * query + query_bias[:, None, None, :]
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            if mask.shape[1:] != (tokens, tokens) or window_batch % num_windows:
                raise ValueError("mask shape is incompatible with the window batch")
            attention = attention.reshape(
                window_batch // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(window_batch, self.num_heads, tokens, tokens)
        attention = self.attn_drop(torch.softmax(attention, dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(window_batch, tokens, channels)
        output = self.proj_drop(self.proj(output))
        compact_aux = {
            "query_scale": query_scale.reshape(-1),
            "query_bias": query_bias.reshape(-1),
            "snr_per_window": snr.reshape(-1),
        }
        self.last_aux = compact_aux
        if not return_aux:
            return output
        aux = {**compact_aux, "attention": attention}
        return output, aux


def correlation_penalty(condition: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    condition = condition.reshape(-1).to(torch.float32)
    parameters = parameters.reshape(-1).to(device=condition.device, dtype=torch.float32)
    if condition.numel() != parameters.numel():
        raise ValueError("condition and parameters must have the same number of elements")
    if condition.numel() < 2:
        return torch.zeros((), device=condition.device)
    condition = condition - condition.mean()
    parameters = parameters - parameters.mean()
    denominator = torch.sqrt(condition.square().sum() * parameters.square().sum()).clamp_min(1e-12)
    correlation = torch.sum(condition * parameters) / denominator
    return F.relu(correlation)


def snr_eq_loss(
    mse: torch.Tensor,
    snr_db: torch.Tensor,
    query_scale: torch.Tensor,
    query_bias: torch.Tensor,
    *,
    penalty_weight: float = 1e5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale_penalty = correlation_penalty(snr_db, query_scale)
    bias_penalty = correlation_penalty(snr_db, query_bias)
    penalty = scale_penalty + bias_penalty
    total = mse + float(penalty_weight) * penalty
    return total, {
        "mse": mse,
        "scale_penalty": scale_penalty,
        "bias_penalty": bias_penalty,
        "penalty": penalty,
    }


def _condition_per_window(
    condition: torch.Tensor | float,
    window_batch: int,
    batch_size: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.as_tensor(condition, device=device, dtype=dtype).reshape(-1)
    if values.numel() == 1:
        return values.reshape(1, 1).expand(window_batch, 1)
    if values.numel() == window_batch:
        return values.reshape(window_batch, 1)
    if batch_size is None or values.numel() != batch_size or window_batch % batch_size:
        raise ValueError("provide scalar SNR, one SNR per window, or batch_size with one SNR per image")
    return values.repeat_interleave(window_batch // batch_size).reshape(window_batch, 1)
