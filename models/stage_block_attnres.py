# -*- coding: utf-8 -*-
"""Internal layer-block attention residual implementation for DecompJSCC."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from swinjscc_utils import expand_rate_like, expand_scalar_like


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)


ATTNRES_QUERY_MODES = {
    "static",
    "condition",
    "content",
    "content_condition_gate",
    "sourcewise_logit_gate",
    "layerwise_snr_gate",
    "snr_depth_logit_gate",
}
ATTNRES_CONDITION_GATE_MODES = {"snr_rate", "rate", "snr"}


class AffineFreeRMSNorm(nn.Module):
    """无仿射参数的 RMS 归一化（仅按最后一维缩放）。

    用于 Block AttnRes 中对 depth source 做 key 侧归一化，使 softmax 源选择
    主要由方向/相关性决定，而非各源幅值差异（见 PRD：affine-free RMSNorm-style key normalization）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale


class StageBlockAttnRes(nn.Module):
    """编码器最后一层 SwinJSCC stage 上的 Block Attention Residuals（按 AttnRes layer的粒度）。

    将每个 SwinTransformerBlock 拆成两个 AttnRes 层（WindowAttention 分支与 MLP 分支），
    在每个 AttnRes 层前对「已完成 block 表征 + 当前 partial_block」做 depth-source attention 混合，
    再在 Swin 子模块内做 norm/attn/mlp；下游预算路径不由此模块改写。
    """

    def __init__(
        self,
        d_semantic: int,
        num_swin_blocks: int,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "static",
        condition_dim: Optional[int] = None,
        snr_norm_denominator: float = 20.0,
        condition_reference_snr_db: float = 7.0,
        condition_reference_rate_ratio: float = 0.125,
        condition_gate_scale: float = 0.5,
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr_rate",
        attnres_condition_gate_use_reference: bool = True,
        enable_cross_stage_attnres: bool = False,
        attnres_cross_stage_skip_threshold: float = 0.0,
        attnres_cross_stage_force_skip: bool = False,
    ):
        super().__init__()
        if attnres_query_mode not in ATTNRES_QUERY_MODES:
            raise ValueError(f"Unsupported attnres_query_mode: {attnres_query_mode}")
        if attnres_condition_gate_mode not in ATTNRES_CONDITION_GATE_MODES:
            raise ValueError(f"Unsupported attnres_condition_gate_mode: {attnres_condition_gate_mode}")
        if attnres_cross_stage_skip_threshold < 0:
            raise ValueError("attnres_cross_stage_skip_threshold must be non-negative")
        self.d_semantic = int(d_semantic)
        self.num_swin_blocks = int(num_swin_blocks)
        self.num_attnres_layers = self.num_swin_blocks * 2
        self.attnres_block_size = int(attnres_block_size)
        if self.attnres_block_size <= 0:
            raise ValueError(f"attnres_block_size must be positive, got {attnres_block_size}")

        self.scale = float(self.d_semantic) ** -0.5
        self.key_norm = AffineFreeRMSNorm(self.d_semantic)
        self.attnres_query_mode = attnres_query_mode
        self.use_condition_query = self.attnres_query_mode == "condition"
        self.use_content_query = self.attnres_query_mode in {
            "content",
            "content_condition_gate",
            "sourcewise_logit_gate",
            "layerwise_snr_gate",
        }
        self.use_condition_gate = self.attnres_query_mode == "content_condition_gate"
        self.use_sourcewise_logit_gate = self.attnres_query_mode == "sourcewise_logit_gate"
        self.use_layerwise_snr_gate = self.attnres_query_mode == "layerwise_snr_gate"
        self.use_snr_depth_logit_gate = self.attnres_query_mode == "snr_depth_logit_gate"
        self.condition_gate_mode = attnres_condition_gate_mode
        self.snr_norm_denominator = float(snr_norm_denominator)
        self.condition_reference_snr_db = float(condition_reference_snr_db)
        self.condition_reference_rate = max(1.0, float(self.d_semantic) * float(condition_reference_rate_ratio))
        self.condition_gate_scale = float(condition_gate_scale)
        self.attnres_content_query_scale = float(attnres_content_query_scale)
        self.attnres_use_layer_embedding = bool(attnres_use_layer_embedding)
        self.attnres_condition_gate_use_reference = bool(attnres_condition_gate_use_reference)
        self.enable_cross_stage_attnres = bool(enable_cross_stage_attnres)
        self.attnres_cross_stage_skip_threshold = float(attnres_cross_stage_skip_threshold)
        self.attnres_cross_stage_force_skip = bool(attnres_cross_stage_force_skip)
        # 每层一个可学习伪 query；零初始化 → 初始 logits 为 0 → 各 depth source 均匀权重
        self.pseudo_queries = nn.Parameter(torch.zeros(self.num_attnres_layers, self.d_semantic))
        self.output_query = nn.Parameter(torch.zeros(self.d_semantic))
        num_query_points = self.num_attnres_layers + 1
        if self.use_snr_depth_logit_gate:
            self.snr_depth_logit_scale = nn.Parameter(torch.zeros(num_query_points))
        if self.enable_cross_stage_attnres:
            self.cross_stage_gates = nn.Parameter(torch.zeros(num_query_points))
        if self.use_condition_query:
            condition_dim = int(condition_dim or self.d_semantic)
            snr_dim = condition_dim // 2
            rate_dim = condition_dim - snr_dim
            self.snr_condition_proj = nn.Sequential(nn.Linear(1, snr_dim), nn.GELU())
            self.rate_condition_proj = nn.Sequential(nn.Linear(1, rate_dim), nn.GELU())
            self.layer_query_delta = nn.Sequential(
                nn.Linear(condition_dim, self.d_semantic),
                nn.GELU(),
                nn.Linear(self.d_semantic, self.d_semantic, bias=False),
            )
            self.output_query_delta = nn.Sequential(
                nn.Linear(condition_dim, self.d_semantic),
                nn.GELU(),
                nn.Linear(self.d_semantic, self.d_semantic, bias=False),
            )
            self._reset_condition_query_parameters()
        if self.use_content_query:
            self.layer_embeddings = (
                nn.Embedding(self.num_attnres_layers + 1, self.d_semantic)
                if self.attnres_use_layer_embedding
                else None
            )
            self.content_query_delta = nn.Sequential(
                nn.Linear(self.d_semantic * 2, self.d_semantic),
                nn.GELU(),
                nn.Linear(self.d_semantic, self.d_semantic, bias=False),
            )
            self.content_output_query_delta = nn.Sequential(
                nn.Linear(self.d_semantic * 2, self.d_semantic),
                nn.GELU(),
                nn.Linear(self.d_semantic, self.d_semantic, bias=False),
            )
            if self.use_condition_gate:
                # 当 condition_gate_mode 为 "snr_rate" 时，condition_gate_scalar_dim 为 2，否则为 1。
                condition_gate_scalar_dim = 2 if self.condition_gate_mode == "snr_rate" else 1
                # 条件门控网络，输入维度为 d_semantic + condition_gate_scalar_dim，输出维度为 1，末层无 bias。
                self.condition_gate = nn.Sequential(
                    nn.Linear(self.d_semantic + condition_gate_scalar_dim, self.d_semantic),
                    nn.GELU(),
                    nn.Linear(self.d_semantic, 1, bias=False),
                )
            if self.use_layerwise_snr_gate:
                num_query_points = self.num_attnres_layers + 1
                self.layerwise_snr_gate_scale = nn.Parameter(torch.zeros(num_query_points))
                self.layerwise_snr_gate_bias = nn.Parameter(torch.zeros(num_query_points))
            if self.use_sourcewise_logit_gate:
                num_query_points = self.num_attnres_layers + 1
                self.sourcewise_content_score = nn.Sequential(
                    nn.Linear(self.d_semantic * 2, self.d_semantic),
                    nn.GELU(),
                    nn.Linear(self.d_semantic, 1, bias=False),
                )
                self.sourcewise_logit_scale = nn.Parameter(torch.zeros(num_query_points))
                self.sourcewise_snr_gate_scale = nn.Parameter(torch.zeros(num_query_points))
                self.sourcewise_snr_gate_bias = nn.Parameter(torch.zeros(num_query_points))
            self._reset_content_query_parameters()

    def reset_identity_parameters(self) -> None:
        """将伪 query 与输出 query 置零，恢复「均匀 depth 混合」的初始行为（用于全局 init 之后对齐 PRD）。"""
        nn.init.zeros_(self.pseudo_queries)
        nn.init.zeros_(self.output_query)
        if self.use_condition_query:
            self._reset_condition_query_parameters()
        if self.use_content_query:
            self._reset_content_query_parameters()
        if self.use_snr_depth_logit_gate:
            nn.init.zeros_(self.snr_depth_logit_scale)
        if self.enable_cross_stage_attnres:
            nn.init.zeros_(self.cross_stage_gates)

    def _reset_condition_query_parameters(self) -> None:
        """仅重置 condition delta 的末层，保持 zero-init uniform 行为且保留前层可学习投影。"""
        nn.init.zeros_(self.layer_query_delta[-1].weight)
        nn.init.zeros_(self.output_query_delta[-1].weight)
        if self.layer_query_delta[-1].bias is not None:
            nn.init.zeros_(self.layer_query_delta[-1].bias)
        if self.output_query_delta[-1].bias is not None:
            nn.init.zeros_(self.output_query_delta[-1].bias)

    def _reset_content_query_parameters(self) -> None:
        """重置 content/query 分支末层，使初始行为仍为 zero-query uniform depth mixing。"""
        nn.init.zeros_(self.content_query_delta[-1].weight)
        nn.init.zeros_(self.content_output_query_delta[-1].weight)
        if self.content_query_delta[-1].bias is not None:
            nn.init.zeros_(self.content_query_delta[-1].bias)
        if self.content_output_query_delta[-1].bias is not None:
            nn.init.zeros_(self.content_output_query_delta[-1].bias)
        if self.use_condition_gate:
            nn.init.zeros_(self.condition_gate[-1].weight)
            if self.condition_gate[-1].bias is not None:
                nn.init.zeros_(self.condition_gate[-1].bias)
        if self.use_layerwise_snr_gate:
            nn.init.zeros_(self.layerwise_snr_gate_scale)
            nn.init.zeros_(self.layerwise_snr_gate_bias)
        if self.use_sourcewise_logit_gate:
            nn.init.zeros_(self.sourcewise_logit_scale)
            nn.init.zeros_(self.sourcewise_snr_gate_scale)
            nn.init.zeros_(self.sourcewise_snr_gate_bias)

    def _encode_condition_inputs(self, snr_input: torch.Tensor, rate_input: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.snr_condition_proj(snr_input),
                self.rate_condition_proj(rate_input),
            ],
            dim=-1,
        )

    def _compute_condition_deltas(
        self,
        snr_db,
        rate,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """计算 batch/sample 级 condition query delta；未启用时返回 `(None, None)`。"""
        if not self.use_condition_query:
            return None, None
        snr = expand_scalar_like(snr_db, batch_size, device, dtype, default=0.0).unsqueeze(-1)
        rate_tensor = expand_rate_like(rate, batch_size, device, default=self.d_semantic, max_value=self.d_semantic)
        rate_value = rate_tensor.to(dtype=dtype).unsqueeze(-1)
        snr_input = snr / self.snr_norm_denominator
        rate_input = rate_value / float(self.d_semantic)
        cond_emb = self._encode_condition_inputs(snr_input, rate_input)
        ref_snr_input = torch.full_like(snr_input, self.condition_reference_snr_db / self.snr_norm_denominator)
        ref_rate_input = torch.full_like(rate_input, self.condition_reference_rate / float(self.d_semantic))
        ref_emb = self._encode_condition_inputs(ref_snr_input, ref_rate_input)
        # Centering prevents this branch from becoming a static query offset; base pseudo_queries own static depth preference.
        layer_delta = self.layer_query_delta(cond_emb) - self.layer_query_delta(ref_emb)
        output_delta = self.output_query_delta(cond_emb) - self.output_query_delta(ref_emb)
        return layer_delta, output_delta

    def _layer_embedding(
        self,
        layer_index: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.layer_embeddings is None:
            return torch.zeros(batch_size, self.d_semantic, device=device, dtype=dtype)
        layer_emb = self.layer_embeddings.weight[layer_index].to(device=device, dtype=dtype)
        return layer_emb.unsqueeze(0).expand(batch_size, -1)

    def _content_summary(self, sources: list[torch.Tensor]) -> torch.Tensor:
        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values.detach())
        return keys.mean(dim=(0, 2))

    def _content_delta(
        self,
        sources: list[torch.Tensor],
        layer_index: int,
        is_output: bool = False,
    ) -> torch.Tensor:
        summary = self._content_summary(sources)
        layer_emb = self._layer_embedding(layer_index, summary.shape[0], summary.device, summary.dtype)
        delta_input = torch.cat([summary, layer_emb], dim=-1)
        branch = self.content_output_query_delta if is_output else self.content_query_delta
        return branch(delta_input) * self.attnres_content_query_scale

    def _condition_gate(
        self,
        snr_db,
        rate,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_index: int,
    ) -> torch.Tensor:
        # 将 snr_db 和 rate 转换为 tensor，并扩展到 batch 维度。
        snr = expand_scalar_like(snr_db, batch_size, device, dtype, default=0.0).unsqueeze(-1)
        rate_tensor = expand_rate_like(rate, batch_size, device, default=self.d_semantic, max_value=self.d_semantic)
        rate_value = rate_tensor.to(dtype=dtype).unsqueeze(-1)
        # 将 snr_db 和 rate 转换为归一化值。
        snr_input = snr / self.snr_norm_denominator
        rate_input = rate_value / float(self.d_semantic)
        # 创建参考条件输入。
        ref_snr_input = torch.full_like(snr_input, self.condition_reference_snr_db / self.snr_norm_denominator)
        ref_rate_input = torch.full_like(rate_input, self.condition_reference_rate / float(self.d_semantic))
        # 创建层嵌入。
        layer_emb = self._layer_embedding(layer_index, batch_size, device, dtype)
        # 根据 condition_gate_mode 选择条件门控的输入部分。
        if self.condition_gate_mode == "snr_rate":
            # 当 condition_gate_mode 为 "snr_rate" 时，条件门控的输入部分为 snr_input、rate_input 和 layer_emb。
            cond_parts = [snr_input, rate_input, layer_emb]
            ref_parts = [ref_snr_input, ref_rate_input, layer_emb]
        elif self.condition_gate_mode == "rate":
            # 当 condition_gate_mode 为 "rate" 时，条件门控的输入部分为 rate_input 和 layer_emb。
            cond_parts = [rate_input, layer_emb]
            ref_parts = [ref_rate_input, layer_emb]
        elif self.condition_gate_mode == "snr":
            # 当 condition_gate_mode 为 "snr" 时，条件门控的输入部分为 snr_input 和 layer_emb。
            cond_parts = [snr_input, layer_emb]
            ref_parts = [ref_snr_input, layer_emb]
        else:
            raise RuntimeError(f"Unsupported condition_gate_mode={self.condition_gate_mode}")
        # 将条件输入拼接起来，得到条件输入。
        cond_input = torch.cat(cond_parts, dim=-1)
        # 将参考条件输入拼接起来，得到参考条件输入。
        ref_input = torch.cat(ref_parts, dim=-1)
        # 默认做 reference centering；no-ref 消融只使用当前 cond_input，避免把 ref_snr 注入门控路径。
        centered_gate = self.condition_gate(cond_input)
        if self.attnres_condition_gate_use_reference:
            centered_gate = centered_gate - self.condition_gate(ref_input)
        # 将条件门控缩放后，加上 1，得到最终的条件门控。
        return 1.0 + self.condition_gate_scale * torch.tanh(centered_gate)

    def _layerwise_snr_gate(
        self,
        snr_db,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_index: int,
    ) -> torch.Tensor:
        """每个 AttnRes query 点一个轻量 SNR gate，避免 2 标量直接生成高维 query。"""
        snr_input = (
            expand_scalar_like(snr_db, batch_size, device, dtype, default=13.0).unsqueeze(-1)
            / self.snr_norm_denominator
        )
        scale = self.layerwise_snr_gate_scale[layer_index].to(device=device, dtype=dtype)
        bias = self.layerwise_snr_gate_bias[layer_index].to(device=device, dtype=dtype)
        centered_gate = scale * snr_input + bias
        if self.attnres_condition_gate_use_reference:
            ref_snr = torch.full_like(snr_input, self.condition_reference_snr_db / self.snr_norm_denominator)
            centered_gate = centered_gate - (scale * ref_snr + bias)
        return 1.0 + self.condition_gate_scale * torch.tanh(centered_gate)

    def _sourcewise_snr_gate(
        self,
        snr_db,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_index: int,
    ) -> torch.Tensor:
        snr_input = (
            expand_scalar_like(snr_db, batch_size, device, dtype, default=13.0).unsqueeze(-1)
            / self.snr_norm_denominator
        )
        scale = self.sourcewise_snr_gate_scale[layer_index].to(device=device, dtype=dtype)
        bias = self.sourcewise_snr_gate_bias[layer_index].to(device=device, dtype=dtype)
        centered_gate = scale * snr_input + bias
        if self.attnres_condition_gate_use_reference:
            ref_snr = torch.full_like(snr_input, self.condition_reference_snr_db / self.snr_norm_denominator)
            centered_gate = centered_gate - (scale * ref_snr + bias)
        return 1.0 + self.condition_gate_scale * torch.tanh(centered_gate)

    def _sourcewise_logit_bias(
        self,
        sources: list[torch.Tensor],
        snr_db,
        layer_index: int,
    ) -> torch.Tensor:
        """内容/SNR 驱动的 source-wise logit bias，直接调 depth-source 选择概率。"""
        if not self.use_sourcewise_logit_gate:
            raise RuntimeError("source-wise logit bias is only available in sourcewise_logit_gate mode")
        if not sources:
            raise ValueError("Source-wise logit gate requires at least one source")

        values = torch.stack(sources, dim=0)
        num_sources, batch_size, token_count, _ = values.shape
        device, dtype = values.device, values.dtype
        source_summary = self.key_norm(values.detach()).mean(dim=2)
        layer_emb = self._layer_embedding(layer_index, batch_size, device, dtype)
        layer_emb = layer_emb.unsqueeze(0).expand(num_sources, -1, -1)
        content_score = self.sourcewise_content_score(torch.cat([source_summary, layer_emb], dim=-1)).squeeze(-1)
        detached_values = values.detach()
        if num_sources == 1:
            source_delta = detached_values
        else:
            source_delta = torch.cat([detached_values[:1], detached_values[1:] - detached_values[:-1]], dim=0)
        innovation_score = torch.log1p(source_delta.pow(2).mean(dim=(2, 3)))
        raw_score = content_score + innovation_score
        raw_score = raw_score - raw_score.mean(dim=0, keepdim=True)
        snr_gate = self._sourcewise_snr_gate(snr_db, batch_size, device, dtype, layer_index).squeeze(-1)
        source_scale = self.sourcewise_logit_scale[layer_index].to(device=device, dtype=dtype)
        bias = source_scale * snr_gate.unsqueeze(0) * raw_score
        return bias.unsqueeze(-1).expand(num_sources, batch_size, token_count)

    def _snr_depth_logit_bias(
        self,
        sources: list[torch.Tensor],
        snr_db,
        layer_index: int,
    ) -> torch.Tensor:
        """用低维 SNR 条件调 depth source logits，避免 SNR 直接生成高维 query。

        正系数时，高于参考 SNR 会偏向更深 source，低于参考 SNR 会偏向更浅/稳定 source。
        系数 zero-init，因此初始完全不改变 AttnRes 的 uniform 行为。
        """
        if not self.use_snr_depth_logit_gate:
            raise RuntimeError("SNR depth logit bias is only available in snr_depth_logit_gate mode")
        if not sources:
            raise ValueError("SNR depth logit gate requires at least one source")

        values = torch.stack(sources, dim=0)
        num_sources, batch_size, token_count, _ = values.shape
        device, dtype = values.device, values.dtype
        if num_sources == 1:
            return torch.zeros(num_sources, batch_size, token_count, device=device, dtype=dtype)

        snr_input = (
            expand_scalar_like(snr_db, batch_size, device, dtype, default=self.condition_reference_snr_db).unsqueeze(-1)
            / self.snr_norm_denominator
        )
        if self.attnres_condition_gate_use_reference:
            ref_snr = torch.full_like(snr_input, self.condition_reference_snr_db / self.snr_norm_denominator)
            snr_delta = snr_input - ref_snr
        else:
            snr_delta = snr_input
        source_rank = torch.linspace(-1.0, 1.0, num_sources, device=device, dtype=dtype).view(num_sources, 1, 1)
        scale = self.snr_depth_logit_scale[layer_index].to(device=device, dtype=dtype)
        bias = scale * source_rank * snr_delta.view(1, batch_size, 1)
        return bias.expand(num_sources, batch_size, token_count)

    def _mix_sources(
        self,
        sources: list[torch.Tensor],
        query: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 depth sources 做 scaled dot-product → softmax(source 维) → 加权求和得到混合表征与权重。"""
        if not sources:
            raise ValueError("Block AttnRes requires at least one source")
        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values)
        query = query.to(device=values.device, dtype=values.dtype)
        if query.ndim == 1:
            logits = torch.einsum("d,sbnd->sbn", query, keys) * self.scale
        elif query.ndim == 2:
            if query.shape[0] != values.shape[1]:
                raise ValueError(f"condition query batch mismatch: expected {values.shape[1]}, got {query.shape[0]}")
            logits = torch.einsum("bd,sbnd->sbn", query, keys) * self.scale
        else:
            raise ValueError(f"query must have rank 1 or 2, got shape {tuple(query.shape)}")
        if logit_bias is not None:
            if logit_bias.shape != logits.shape:
                raise ValueError(f"logit_bias shape {tuple(logit_bias.shape)} does not match logits {tuple(logits.shape)}")
            logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
        weights = torch.softmax(logits, dim=0)
        mixed = torch.einsum("sbn,sbnd->bnd", weights, values)
        return mixed, weights

    @staticmethod
    def _window_attention_branch(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """复现 Swin block 的 WindowAttention 分支（norm1 + shift/roll + window attn + 反变换），不含残差加回。"""
        height, width = block.input_resolution
        batch_size, length, channels = x.shape
        if length != height * width:
            raise ValueError("input feature has wrong size for Swin attention branch")

        x = block.norm1(x)
        x = x.view(batch_size, height, width, channels)
        if block.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, block.window_size)
        x_windows = x_windows.view(-1, block.window_size * block.window_size, channels)
        attn_windows = block.attn(x_windows, mask=block.attn_mask)
        attn_windows = attn_windows.view(-1, block.window_size, block.window_size, channels)
        shifted_x = window_reverse(attn_windows, block.window_size, height, width)

        if block.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x.view(batch_size, height * width, channels)

    def forward(
        self,
        x: torch.Tensor,
        swin_blocks: nn.ModuleList,
        snr_db=None,
        rate=None,
        external_sources: Optional[list[torch.Tensor]] = None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """在最后一个 stage 的输入 `x` 上逐步跑完各 Swin block，返回 stage 输出与 attnres 诊断字典。

        `attnres_block_size` 控制每累计多少个 AttnRes layer后把当前 partial 并入 completed（默认 attnres block size=2, 即一整块 SwinJSCC block）。
        """
        if len(swin_blocks) != self.num_swin_blocks:
            raise ValueError(f"expected {self.num_swin_blocks} Swin blocks, got {len(swin_blocks)}")

        layer_delta, output_delta = self._compute_condition_deltas(
            snr_db,
            rate,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        completed_blocks: list[torch.Tensor] = [x]
        partial_block: Optional[torch.Tensor] = None
        weights_per_attnres_layer: list[torch.Tensor] = []
        source_counts: list[int] = []
        content_delta_norms: list[torch.Tensor] = []
        content_delta_vectors: list[torch.Tensor] = []
        output_content_delta_norm: Optional[torch.Tensor] = None
        output_content_delta_vector: Optional[torch.Tensor] = None
        condition_gates: list[torch.Tensor] = []
        sourcewise_logit_bias_terms: list[torch.Tensor] = []
        sourcewise_snr_gates: list[torch.Tensor] = []
        snr_depth_logit_bias_terms: list[torch.Tensor] = []
        cross_stage_weights_per_layer: list[torch.Tensor] = []
        cross_stage_gate_values: list[torch.Tensor] = []
        cross_stage_sources = list(external_sources or [])
        attnres_layer_index = 0

        def available_sources() -> list[torch.Tensor]:
            """当前 step 可用的 depth sources：已完成各段 + 可选的进行中 partial。"""
            if partial_block is None:
                return list(completed_blocks)
            return [*completed_blocks, partial_block]

        def maybe_complete_block() -> None:
            """每 `attnres_block_size` 个 AttnRes 层结束时将 partial 封存进 completed。"""
            nonlocal partial_block
            if attnres_layer_index % self.attnres_block_size == 0 and partial_block is not None:
                completed_blocks.append(partial_block)
                partial_block = None

        def build_query(base_query: torch.Tensor, sources: list[torch.Tensor], layer_index: int, is_output: bool = False):
            nonlocal output_content_delta_norm, output_content_delta_vector
            query = base_query
            if layer_delta is not None and output_delta is not None:
                delta = output_delta if is_output else layer_delta
                return query.unsqueeze(0) + delta
            # 如果使用 content query，则计算 content delta 并应用条件门控。
            if self.use_content_query:
                # 计算 content delta。
                delta = self._content_delta(sources, layer_index, is_output=is_output)
                # 如果使用条件门控，则计算条件门控并应用。
                if self.use_condition_gate:
                    # 计算条件门控。
                    gate = self._condition_gate(
                        snr_db,
                        rate,
                        batch_size=x.shape[0],
                        device=x.device,
                        dtype=x.dtype,
                        layer_index=layer_index,
                    )
                    if return_aux:
                        condition_gates.append(gate.detach())
                    delta = delta * gate
                if self.use_layerwise_snr_gate:
                    gate = self._layerwise_snr_gate(
                        snr_db,
                        batch_size=x.shape[0],
                        device=x.device,
                        dtype=x.dtype,
                        layer_index=layer_index,
                    )
                    if return_aux:
                        condition_gates.append(gate.detach())
                    delta = delta * gate
                if return_aux:
                    if is_output:
                        output_content_delta_norm = delta.norm(dim=-1).mean()
                        output_content_delta_vector = delta.mean(dim=0)
                    else:
                        content_delta_norms.append(delta.norm(dim=-1).mean())
                        content_delta_vectors.append(delta.mean(dim=0))
                return query.unsqueeze(0) + delta
            return query

        def build_logit_bias(sources: list[torch.Tensor], layer_index: int) -> Optional[torch.Tensor]:
            biases: list[torch.Tensor] = []
            if self.use_sourcewise_logit_gate:
                sourcewise_bias = self._sourcewise_logit_bias(sources, snr_db=snr_db, layer_index=layer_index)
                if return_aux:
                    sourcewise_logit_bias_terms.append(sourcewise_bias.detach().abs().mean())
                    sourcewise_snr_gates.append(
                        self._sourcewise_snr_gate(
                            snr_db,
                            batch_size=x.shape[0],
                            device=x.device,
                            dtype=x.dtype,
                            layer_index=layer_index,
                        ).detach()
                    )
                biases.append(sourcewise_bias)
            if self.use_snr_depth_logit_gate:
                snr_depth_bias = self._snr_depth_logit_bias(sources, snr_db=snr_db, layer_index=layer_index)
                if return_aux:
                    snr_depth_logit_bias_terms.append(snr_depth_bias.detach().abs().mean())
                biases.append(snr_depth_bias)
            if not biases:
                return None
            bias = biases[0]
            for extra_bias in biases[1:]:
                bias = bias + extra_bias
            return bias

        def add_cross_stage_context(h: torch.Tensor, query: torch.Tensor, layer_index: int) -> torch.Tensor:
            if not self.enable_cross_stage_attnres or not cross_stage_sources:
                return h
            gate = torch.tanh(self.cross_stage_gates[layer_index]).to(device=h.device, dtype=h.dtype)
            if self.attnres_cross_stage_force_skip:
                return h
            if self.attnres_cross_stage_skip_threshold > 0.0:
                gate_abs = float(gate.detach().abs().max().item())
                if gate_abs < self.attnres_cross_stage_skip_threshold:
                    return h
            cross_bias = None
            if self.use_snr_depth_logit_gate:
                cross_bias = self._snr_depth_logit_bias(cross_stage_sources, snr_db=snr_db, layer_index=layer_index)
            cross_h, cross_weights = self._mix_sources(cross_stage_sources, query, logit_bias=cross_bias)
            if return_aux:
                cross_stage_weights_per_layer.append(cross_weights.detach())
                cross_stage_gate_values.append(gate.detach().reshape(1))
            return h + gate * cross_h

        for block in swin_blocks:
            sources = available_sources()
            query = build_query(self.pseudo_queries[attnres_layer_index], sources, attnres_layer_index)
            h, weights = self._mix_sources(sources, query, logit_bias=build_logit_bias(sources, attnres_layer_index))
            h = add_cross_stage_context(h, query, attnres_layer_index)
            attn_out = self._window_attention_branch(block, h)
            partial_block = attn_out if partial_block is None else partial_block + attn_out
            if return_aux:
                weights_per_attnres_layer.append(weights.detach())
                source_counts.append(len(sources))
            attnres_layer_index += 1
            maybe_complete_block()

            sources = available_sources()
            query = build_query(self.pseudo_queries[attnres_layer_index], sources, attnres_layer_index)
            h, weights = self._mix_sources(sources, query, logit_bias=build_logit_bias(sources, attnres_layer_index))
            h = add_cross_stage_context(h, query, attnres_layer_index)
            mlp_out = block.mlp(block.norm2(h))
            partial_block = mlp_out if partial_block is None else partial_block + mlp_out
            if return_aux:
                weights_per_attnres_layer.append(weights.detach())
                source_counts.append(len(sources))
            attnres_layer_index += 1
            maybe_complete_block()

        if partial_block is not None:
            completed_blocks.append(partial_block)

        # 全部 Swin block 结束后，用独立 output_query 在「各 completed block 输出」上再混合一次得到 z
        output_query = build_query(
            self.output_query,
            completed_blocks,
            self.num_attnres_layers,
            is_output=True,
        )
        output, output_weights = self._mix_sources(
            completed_blocks,
            output_query,
            logit_bias=build_logit_bias(completed_blocks, self.num_attnres_layers),
        )
        output = add_cross_stage_context(output, output_query, self.num_attnres_layers)
        if not return_aux:
            return output, {}
        entropy_terms = [
            -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0).mean()
            for weights in weights_per_attnres_layer
        ]
        entropy_terms.append(
            -(output_weights.clamp_min(1e-8) * output_weights.clamp_min(1e-8).log()).sum(dim=0).mean()
        )
        all_flat_weights = torch.cat(
            [weights.reshape(-1) for weights in [*weights_per_attnres_layer, output_weights]]
        )
        # 训练脚本仅聚合标量键；完整权重张量供调试/可视化
        aux = {
            "attnres_weights": weights_per_attnres_layer,
            "attnres_output_weights": output_weights.detach(),
            "attnres_source_counts": torch.tensor(source_counts, device=x.device, dtype=torch.long),
            "attnres_output_source_count": torch.tensor(len(completed_blocks), device=x.device, dtype=torch.long),
            "attnres_entropy": torch.stack(entropy_terms).mean().detach(),
            "attnres_weight_std": all_flat_weights.std().detach(),
            "attnres_source_count_mean": torch.tensor(
                float(sum(source_counts)) / max(len(source_counts), 1),
                device=x.device,
                dtype=x.dtype,
            ),
        }
        if layer_delta is not None and output_delta is not None:
            aux.update(
                {
                    "attnres_cond_delta_norm": layer_delta.norm(dim=-1).mean().detach(),
                    "attnres_cond_output_delta_norm": output_delta.norm(dim=-1).mean().detach(),
                    "attnres_cond_delta_vector": layer_delta.mean(dim=0).detach(),
                    "attnres_cond_output_delta_vector": output_delta.mean(dim=0).detach(),
                }
            )
        if content_delta_norms:
            aux.update(
                {
                    "attnres_content_delta_norm": torch.stack(content_delta_norms).mean().detach(),
                    "attnres_content_output_delta_norm": output_content_delta_norm.detach()
                    if output_content_delta_norm is not None
                    else torch.zeros((), device=x.device, dtype=x.dtype),
                    "attnres_content_delta_vector": torch.stack(content_delta_vectors).mean(dim=0).detach(),
                    "attnres_content_output_delta_vector": output_content_delta_vector.detach()
                    if output_content_delta_vector is not None
                    else torch.zeros(self.d_semantic, device=x.device, dtype=x.dtype),
                }
            )
        if condition_gates:
            all_gates = torch.cat([gate.reshape(-1) for gate in condition_gates])
            aux.update(
                {
                    "attnres_condition_gate_mean": all_gates.mean().detach(),
                    "attnres_condition_gate_std": all_gates.std(unbiased=False).detach(),
                }
            )
        if sourcewise_logit_bias_terms:
            all_sourcewise_gates = torch.cat([gate.reshape(-1) for gate in sourcewise_snr_gates])
            aux.update(
                {
                    "attnres_sourcewise_logit_bias_abs_mean": torch.stack(sourcewise_logit_bias_terms).mean().detach(),
                    "attnres_sourcewise_snr_gate_mean": all_sourcewise_gates.mean().detach(),
                    "attnres_sourcewise_snr_gate_std": all_sourcewise_gates.std(unbiased=False).detach(),
                    "attnres_sourcewise_logit_scale_mean": self.sourcewise_logit_scale.detach().abs().mean(),
                    "attnres_sourcewise_snr_gate_scale_mean": self.sourcewise_snr_gate_scale.detach().abs().mean(),
                }
            )
        if snr_depth_logit_bias_terms:
            aux.update(
                {
                    "attnres_snr_depth_logit_bias_abs_mean": torch.stack(snr_depth_logit_bias_terms).mean().detach(),
                    "attnres_snr_depth_logit_scale_mean": self.snr_depth_logit_scale.detach().abs().mean(),
                }
            )
        if self.use_layerwise_snr_gate:
            aux.update(
                {
                    "attnres_layerwise_snr_gate_scale_mean": self.layerwise_snr_gate_scale.detach().abs().mean(),
                }
            )
        if cross_stage_gate_values:
            cross_entropy_terms = [
                -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0).mean()
                for weights in cross_stage_weights_per_layer
            ]
            all_cross_weights = torch.cat([weights.reshape(-1) for weights in cross_stage_weights_per_layer])
            all_cross_gates = torch.cat(cross_stage_gate_values)
            aux.update(
                {
                    "attnres_cross_stage_weights": cross_stage_weights_per_layer[:-1],
                    "attnres_cross_stage_output_weights": cross_stage_weights_per_layer[-1],
                    "attnres_cross_stage_source_count": torch.tensor(
                        len(cross_stage_sources), device=x.device, dtype=torch.long
                    ),
                    "attnres_cross_stage_entropy": torch.stack(cross_entropy_terms).mean().detach(),
                    "attnres_cross_stage_weight_std": all_cross_weights.std().detach(),
                    "attnres_cross_stage_gate_abs_mean": all_cross_gates.abs().mean().detach(),
                    "attnres_cross_stage_gate_mean": all_cross_gates.mean().detach(),
                }
            )
        return output, aux


FinalStageBlockAttnRes = StageBlockAttnRes
