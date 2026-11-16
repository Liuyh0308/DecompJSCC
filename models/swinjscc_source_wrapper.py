# -*- coding: utf-8 -*-
"""Source-faithful wrapper around the bundled `SwinJSCC_base` implementation."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from random import choice
from types import SimpleNamespace
from typing import Iterable, Optional

import torch
import torch.nn as nn

from swinjscc_utils import build_hard_channel_mask, expand_rate_like, expand_scalar_like
from models.decompjscc_config import DecompJSCCConfig
from models.stage_block_attnres import StageBlockAttnRes


SOURCE_MODEL_WO_SA_RA = "SwinJSCC_w/o_SAandRA"
SOURCE_MODEL_SA = "SwinJSCC_w/_SA"
SOURCE_MODEL_RA = "SwinJSCC_w/_RA"
SOURCE_MODEL_SA_RA = "SwinJSCC_w/_SAandRA"

SOURCE_HR_RATE_LIST = "32,64,96,128,192"
SOURCE_HR_SNR_LIST = "1,4,7,10,13"


def _ensure_swinjscc_base_on_path() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "SwinJSCC_base"
    base_dir_str = str(base_dir)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)


def _ensure_timm_layers_compat() -> None:
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    timm_mod = sys.modules.setdefault("timm", types.ModuleType("timm"))
    models_mod = sys.modules.setdefault("timm.models", types.ModuleType("timm.models"))
    layers_mod = types.ModuleType("timm.models.layers")

    class DropPath(nn.Identity):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        return (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    layers_mod.DropPath = DropPath
    layers_mod.to_2tuple = to_2tuple
    layers_mod.trunc_normal_ = trunc_normal_
    timm_mod.models = models_mod
    models_mod.layers = layers_mod
    sys.modules["timm.models.layers"] = layers_mod


def _source_swinjscc_class():
    _ensure_swinjscc_base_on_path()
    _ensure_timm_layers_compat()
    from network import SwinJSCC  # type: ignore

    return SwinJSCC


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _freeze_inactive_source_modules(net: nn.Module, source_model_name: str) -> None:
    if source_model_name != SOURCE_MODEL_WO_SA_RA:
        return

    # The official w/o_SAandRA class still registers ModNet modules, but its
    # forward path never uses them. Freezing keeps DDP strict without changing
    # the source forward semantics.
    for owner in (net.encoder, net.decoder):
        for attr_name in ("bm_list", "sm_list"):
            module = getattr(owner, attr_name, None)
            if isinstance(module, nn.Module):
                _set_module_requires_grad(module, False)


def _to_int_list(values: Optional[Iterable[int]]) -> Optional[list[int]]:
    if values is None:
        return None
    return [int(item) for item in values]


def source_model_name_for_recipe_stage(recipe_stage: str) -> str:
    mapping = {
        "fixed": SOURCE_MODEL_WO_SA_RA,
        "fixed_attnres": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_attnres": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_attnres_decoder_adapter": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_stageall_attnres": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_stageall_attnres_decoder_adapter": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_stageall_cross_attnres_decoder_adapter": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_stage1_final_attnres": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_snr_stage1_final_attnres_decoder_adapter": SOURCE_MODEL_WO_SA_RA,
        "no_modnet_universal": SOURCE_MODEL_RA,
        "no_modnet_sall_cross_universal": SOURCE_MODEL_RA,
        "snr_adaptive": SOURCE_MODEL_SA,
        "snr_adaptive_attnres": SOURCE_MODEL_SA,
        "rate_adaptive": SOURCE_MODEL_RA,
        "rcsb_rate_adaptive": SOURCE_MODEL_RA,
        "attnres_rcsb_rate_adaptive": SOURCE_MODEL_RA,
        "universal": SOURCE_MODEL_SA_RA,
        "rcsb_universal_snr_warmup": SOURCE_MODEL_SA_RA,
        "rcsb_universal_joint": SOURCE_MODEL_SA_RA,
    }
    try:
        return mapping[recipe_stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported source SwinJSCC recipe_stage={recipe_stage}") from exc


def source_rate_list_for_recipe_stage(recipe_stage: str) -> str:
    if recipe_stage in {
        "fixed",
        "fixed_attnres",
        "no_modnet_snr",
        "no_modnet_snr_attnres",
        "no_modnet_snr_attnres_decoder_adapter",
        "no_modnet_snr_stageall_attnres",
        "no_modnet_snr_stageall_attnres_decoder_adapter",
        "no_modnet_snr_stageall_cross_attnres_decoder_adapter",
        "no_modnet_snr_stage1_final_attnres",
        "no_modnet_snr_stage1_final_attnres_decoder_adapter",
        "snr_adaptive",
        "snr_adaptive_attnres",
    }:
        return "192"
    if recipe_stage in {
        "no_modnet_universal",
        "no_modnet_sall_cross_universal",
        "rate_adaptive",
        "rcsb_rate_adaptive",
        "attnres_rcsb_rate_adaptive",
        "universal",
        "rcsb_universal_snr_warmup",
        "rcsb_universal_joint",
    }:
        return SOURCE_HR_RATE_LIST
    raise ValueError(f"Unsupported source SwinJSCC recipe_stage={recipe_stage}")


def source_snr_list_for_recipe_stage(recipe_stage: str) -> str:
    if recipe_stage in {"fixed", "fixed_attnres", "rate_adaptive", "rcsb_rate_adaptive", "attnres_rcsb_rate_adaptive"}:
        return "13"
    if recipe_stage in {
        "no_modnet_snr",
        "no_modnet_snr_attnres",
        "no_modnet_snr_attnres_decoder_adapter",
        "no_modnet_snr_stageall_attnres",
        "no_modnet_snr_stageall_attnres_decoder_adapter",
        "no_modnet_snr_stageall_cross_attnres_decoder_adapter",
        "no_modnet_snr_stage1_final_attnres",
        "no_modnet_snr_stage1_final_attnres_decoder_adapter",
        "no_modnet_universal",
        "no_modnet_sall_cross_universal",
        "snr_adaptive",
        "snr_adaptive_attnres",
        "universal",
        "rcsb_universal_snr_warmup",
        "rcsb_universal_joint",
    }:
        return SOURCE_HR_SNR_LIST
    raise ValueError(f"Unsupported source SwinJSCC recipe_stage={recipe_stage}")


@dataclass
class SourceSwinJSCCArgs:
    trainset: str = "DIV2K"
    testset: str = "kodak"
    distortion_metric: str = "MSE"
    model: str = SOURCE_MODEL_SA_RA
    channel_type: str = "awgn"
    C: str = SOURCE_HR_RATE_LIST
    multiple_snr: str = SOURCE_HR_SNR_LIST
    model_size: str = "base"
    img_size: int = 256
    batch_size: int = 16
    embed_dims: Optional[list[int]] = None
    depths: Optional[list[int]] = None
    num_heads: Optional[list[int]] = None
    decoder_depths: Optional[list[int]] = None


def build_source_swinjscc_args(
    recipe_stage: str = "universal",
    trainset: str = "DIV2K",
    testset: str = "kodak",
    distortion_metric: str = "MSE",
    model: Optional[str] = None,
    channel_type: str = "awgn",
    C: Optional[str] = None,
    multiple_snr: Optional[str] = None,
    model_size: str = "base",
    img_size: int = 256,
    batch_size: int = 16,
    embed_dims: Optional[Iterable[int]] = None,
    depths: Optional[Iterable[int]] = None,
    num_heads: Optional[Iterable[int]] = None,
    decoder_depths: Optional[Iterable[int]] = None,
) -> SourceSwinJSCCArgs:
    source_model = model or source_model_name_for_recipe_stage(recipe_stage)
    return SourceSwinJSCCArgs(
        trainset=trainset,
        testset=testset,
        distortion_metric=distortion_metric,
        model=source_model,
        channel_type=channel_type,
        C=C or source_rate_list_for_recipe_stage(recipe_stage),
        multiple_snr=multiple_snr or source_snr_list_for_recipe_stage(recipe_stage),
        model_size=model_size,
        img_size=int(img_size),
        batch_size=int(batch_size),
        embed_dims=_to_int_list(embed_dims),
        depths=_to_int_list(depths),
        num_heads=_to_int_list(num_heads),
        decoder_depths=_to_int_list(decoder_depths),
    )


def _source_depths_for_model_size(model_size: str) -> tuple[list[int], list[int]]:
    if model_size == "small":
        return [2, 2, 2, 2], [2, 2, 2, 2]
    if model_size == "base":
        return [2, 2, 6, 2], [2, 6, 2, 2]
    if model_size == "large":
        return [2, 2, 18, 2], [2, 18, 2, 2]
    raise ValueError(f"Unsupported source SwinJSCC model_size={model_size}")


def _channel_number_for_source_encoder(args: SourceSwinJSCCArgs) -> Optional[int]:
    if args.trainset == "CIFAR10":
        return int(str(args.C).split(",")[0])
    if args.model in {SOURCE_MODEL_WO_SA_RA, SOURCE_MODEL_SA}:
        return int(str(args.C).split(",")[0])
    return None


def build_source_swinjscc_config(
    args: SourceSwinJSCCArgs,
    pass_channel: bool = True,
    logger=None,
) -> SimpleNamespace:
    if args.trainset != "DIV2K":
        raise ValueError("Source-faithful wrapper currently supports the HR DIV2K protocol path")

    embed_dims = list(args.embed_dims or [128, 192, 256, 320])
    encoder_depths, decoder_depths_default = _source_depths_for_model_size(args.model_size)
    encoder_depths = list(args.depths or encoder_depths)
    decoder_depths = list(args.decoder_depths or args.depths or decoder_depths_default)
    num_heads = list(args.num_heads or [4, 6, 8, 10])
    decoder_embed_dims = list(reversed(embed_dims))
    decoder_num_heads = list(reversed(num_heads))
    channel_number = _channel_number_for_source_encoder(args)

    encoder_kwargs = dict(
        model=args.model,
        img_size=(int(args.img_size), int(args.img_size)),
        patch_size=2,
        in_chans=3,
        embed_dims=embed_dims,
        depths=encoder_depths,
        num_heads=num_heads,
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    decoder_kwargs = dict(
        model=args.model,
        img_size=(int(args.img_size), int(args.img_size)),
        embed_dims=decoder_embed_dims,
        depths=decoder_depths,
        num_heads=decoder_num_heads,
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    return SimpleNamespace(
        seed=42,
        pass_channel=bool(pass_channel),
        CUDA=torch.cuda.is_available(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        norm=False,
        logger=logger,
        image_dims=(3, int(args.img_size), int(args.img_size)),
        batch_size=int(args.batch_size),
        downsample=4,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
    )


class SourceSwinJSCCWrapper(nn.Module):
    """Thin wrapper that preserves `SwinJSCC_base.network.SwinJSCC` forward semantics."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
    ):
        super().__init__()
        self.source_args = args
        self.source_config = build_source_swinjscc_config(args, pass_channel=pass_channel, logger=logger)
        self.source_model_name = args.model
        self.pass_channel = bool(pass_channel)
        swinjscc_cls = _source_swinjscc_class()
        self.net = swinjscc_cls(args, self.source_config)
        _freeze_inactive_source_modules(self.net, self.source_model_name)

    @staticmethod
    def flatten_feature_to_source_complex(feature: torch.Tensor) -> torch.Tensor:
        """Match `SwinJSCC_base.channel.Channel.forward`: first half real, second half imaginary."""
        flat = feature.reshape(-1)
        length = flat.shape[0]
        return flat[: length // 2] + flat[length // 2:] * 1j

    def forward(self, image: torch.Tensor, snr_db: Optional[float] = None, rate: Optional[int] = None) -> dict:
        recon_image, cbr, snr, mse, loss = self.net(
            image,
            given_SNR=snr_db,
            given_rate=rate,
        )
        return {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(snr),
            "mse": mse,
            "loss": loss,
        }


class SourceFixedAttnResWrapper(SourceSwinJSCCWrapper):
    """Source-faithful fixed model with final-stage Block AttnRes before the fixed channel head."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "static",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr_rate",
        attnres_condition_gate_use_reference: bool = True,
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_WO_SA_RA:
            raise ValueError("SourceFixedAttnResWrapper requires the official fixed tensor contract")
        final_layer = self.net.encoder.layers[-1]
        self.final_stage_attnres = StageBlockAttnRes(
            d_semantic=self.net.encoder.embed_dims[-1],
            num_swin_blocks=len(final_layer.blocks),
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )

    def _encode_fixed_attnres(
        self,
        image: torch.Tensor,
        snr_db=None,
        rate=None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        final_layer_index = len(self.net.encoder.layers) - 1
        attnres_aux: dict[str, object] = {}
        for layer_index, layer in enumerate(self.net.encoder.layers):
            if layer_index != final_layer_index:
                x = layer(x)
                continue
            if layer.downsample is not None:
                x = layer.downsample(x)
            x, attnres_aux = self.final_stage_attnres(
                x,
                layer.blocks,
                snr_db=snr_db,
                rate=rate,
                return_aux=return_aux,
            )
        x = self.net.encoder.norm(x)
        return self.net.encoder.head_list(x), attnres_aux

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_fixed_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
        return output


class SourceDecoderSNRAdapter(nn.Module):
    """Lightweight receiver-side SNR FiLM adapter with identity-safe initialization."""

    def __init__(self, d_feature: int, hidden_dim: Optional[int] = None, residual_scale: float = 0.1):
        super().__init__()
        self.d_feature = int(d_feature)
        self.residual_scale = float(residual_scale)
        hidden = int(hidden_dim or max(32, self.d_feature // 2))
        self.condition_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_feature * 2),
        )
        nn.init.zeros_(self.condition_mlp[-1].weight)
        nn.init.zeros_(self.condition_mlp[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        snr_db,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, _, d_feature = feature.shape
        if d_feature != self.d_feature:
            raise ValueError(f"Expected decoder input dim {self.d_feature}, got {d_feature}")
        snr_input = expand_scalar_like(
            snr_db,
            batch_size=batch_size,
            device=feature.device,
            dtype=feature.dtype,
            default=13.0,
        ).unsqueeze(-1) / 20.0
        gamma_beta = self.condition_mlp(snr_input).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gain = 1.0 + self.residual_scale * torch.tanh(gamma)
        shift = self.residual_scale * torch.tanh(beta)
        adapted = feature * gain + shift
        if not return_aux:
            return adapted, {}
        return adapted, {
            "gain_mean": gain.mean(),
            "gain_std": gain.std(),
            "shift_rms": torch.sqrt(shift.pow(2).mean()),
        }


class SourceNoModNetSNRAttnResDecoderAdapterWrapper(SourceFixedAttnResWrapper):
    """No-SA-ModNet SNR-adaptive AttnRes encoder with receiver-side SNR adapter."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
        enable_cross_stage_attnres: bool = False,
    ):
        super().__init__(
            args,
            pass_channel=pass_channel,
            logger=logger,
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )
        self.decoder_snr_adapter = SourceDecoderSNRAdapter(self.net.encoder.head_list.out_features)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_fixed_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        decoder_feature, decoder_snr_aux = self.decoder_snr_adapter(
            noisy_feature,
            chan_param,
            return_aux=return_aux,
        )
        recon_image = self.net.decoder(decoder_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
            output["decoder_snr_aux"] = decoder_snr_aux
        return output


class SourceAllStageAttnResWrapper(SourceSwinJSCCWrapper):
    """No-SA-ModNet fixed-rate model with Block AttnRes applied inside every encoder stage."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
        enable_cross_stage_attnres: bool = False,
        attnres_cross_stage_skip_threshold: float = 0.0,
        attnres_cross_stage_force_skip: bool = False,
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_WO_SA_RA:
            raise ValueError("SourceAllStageAttnResWrapper requires the official fixed tensor contract")
        self.enable_cross_stage_attnres = bool(enable_cross_stage_attnres)
        self.attnres_cross_stage_skip_threshold = float(attnres_cross_stage_skip_threshold)
        self.attnres_cross_stage_force_skip = bool(attnres_cross_stage_force_skip)
        self.encoder_stage_attnres = nn.ModuleList(
            [
                StageBlockAttnRes(
                    d_semantic=self.net.encoder.embed_dims[stage_index],
                    num_swin_blocks=len(layer.blocks),
                    attnres_block_size=attnres_block_size,
                    attnres_query_mode=attnres_query_mode,
                    attnres_content_query_scale=attnres_content_query_scale,
                    attnres_use_layer_embedding=attnres_use_layer_embedding,
                    attnres_condition_gate_mode=attnres_condition_gate_mode,
                    attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
                    enable_cross_stage_attnres=enable_cross_stage_attnres and stage_index > 1,
                    attnres_cross_stage_skip_threshold=attnres_cross_stage_skip_threshold,
                    attnres_cross_stage_force_skip=attnres_cross_stage_force_skip,
                )
                for stage_index, layer in enumerate(self.net.encoder.layers)
            ]
        )

    @staticmethod
    def _prefix_stage_aux(stage_index: int, aux: dict[str, object]) -> dict[str, object]:
        return {f"stage{stage_index}_{key}": value for key, value in aux.items()}

    def _project_stage_output_to_current(
        self,
        source: torch.Tensor,
        source_stage_index: int,
        target_stage_index: int,
    ) -> torch.Tensor:
        projected = source
        for stage_index in range(source_stage_index + 1, target_stage_index + 1):
            downsample = self.net.encoder.layers[stage_index].downsample
            if downsample is not None:
                projected = downsample(projected)
        return projected

    def _cross_stage_sources_for_stage(
        self,
        stage_outputs: list[torch.Tensor],
        target_stage_index: int,
    ) -> list[torch.Tensor]:
        if not self.enable_cross_stage_attnres or target_stage_index <= 1:
            return []
        external_sources: list[torch.Tensor] = []
        # The immediate previous stage is already represented by the current stage input after PatchMerging.
        for source_stage_index, source in enumerate(stage_outputs[: target_stage_index - 1]):
            external_sources.append(
                self._project_stage_output_to_current(source, source_stage_index, target_stage_index)
            )
        return external_sources

    def _encode_all_stage_attnres(
        self,
        image: torch.Tensor,
        snr_db=None,
        rate=None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        attnres_aux: dict[str, object] = {}
        stage_outputs: list[torch.Tensor] = []
        for stage_index, (layer, stage_attnres) in enumerate(
            zip(self.net.encoder.layers, self.encoder_stage_attnres)
        ):
            if layer.downsample is not None:
                x = layer.downsample(x)
            external_sources = self._cross_stage_sources_for_stage(stage_outputs, stage_index)
            x, stage_aux = stage_attnres(
                x,
                layer.blocks,
                snr_db=snr_db,
                rate=rate,
                external_sources=external_sources,
                return_aux=return_aux,
            )
            if return_aux:
                attnres_aux.update(self._prefix_stage_aux(stage_index, stage_aux))
            stage_outputs.append(x)
        x = self.net.encoder.norm(x)
        return self.net.encoder.head_list(x), attnres_aux

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_all_stage_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
        return output


class SourceAllStageAttnResDecoderAdapterWrapper(SourceAllStageAttnResWrapper):
    """All-stage encoder AttnRes plus an identity-safe receiver-side SNR adapter."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
        enable_cross_stage_attnres: bool = False,
        attnres_cross_stage_skip_threshold: float = 0.0,
        attnres_cross_stage_force_skip: bool = False,
    ):
        super().__init__(
            args,
            pass_channel=pass_channel,
            logger=logger,
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
            enable_cross_stage_attnres=enable_cross_stage_attnres,
            attnres_cross_stage_skip_threshold=attnres_cross_stage_skip_threshold,
            attnres_cross_stage_force_skip=attnres_cross_stage_force_skip,
        )
        self.decoder_snr_adapter = SourceDecoderSNRAdapter(self.net.encoder.head_list.out_features)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_all_stage_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        decoder_feature, decoder_snr_aux = self.decoder_snr_adapter(
            noisy_feature,
            chan_param,
            return_aux=return_aux,
        )
        recon_image = self.net.decoder(decoder_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
            output["decoder_snr_aux"] = decoder_snr_aux
        return output


class SourceSelectedStageAttnResWrapper(SourceSwinJSCCWrapper):
    """No-SA-ModNet fixed-rate model with Block AttnRes on selected encoder stages."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        stage_indices: Iterable[int] = (1, -1),
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_WO_SA_RA:
            raise ValueError("SourceSelectedStageAttnResWrapper requires the official fixed tensor contract")
        stage_count = len(self.net.encoder.layers)
        resolved_indices = []
        for stage_index in stage_indices:
            resolved_index = int(stage_index)
            if resolved_index < 0:
                resolved_index = stage_count + resolved_index
            if resolved_index < 0 or resolved_index >= stage_count:
                raise ValueError(f"Invalid AttnRes stage index {stage_index} for {stage_count} encoder stages")
            if resolved_index not in resolved_indices:
                resolved_indices.append(resolved_index)
        self.attnres_stage_indices = tuple(sorted(resolved_indices))
        self.encoder_stage_attnres = nn.ModuleDict(
            {
                str(stage_index): StageBlockAttnRes(
                    d_semantic=self.net.encoder.embed_dims[stage_index],
                    num_swin_blocks=len(self.net.encoder.layers[stage_index].blocks),
                    attnres_block_size=attnres_block_size,
                    attnres_query_mode=attnres_query_mode,
                    attnres_content_query_scale=attnres_content_query_scale,
                    attnres_use_layer_embedding=attnres_use_layer_embedding,
                    attnres_condition_gate_mode=attnres_condition_gate_mode,
                    attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
                )
                for stage_index in self.attnres_stage_indices
            }
        )

    @staticmethod
    def _prefix_stage_aux(stage_index: int, aux: dict[str, object]) -> dict[str, object]:
        return {f"stage{stage_index}_{key}": value for key, value in aux.items()}

    def _encode_selected_stage_attnres(
        self,
        image: torch.Tensor,
        snr_db=None,
        rate=None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        attnres_aux: dict[str, object] = {}
        selected = set(self.attnres_stage_indices)
        for stage_index, layer in enumerate(self.net.encoder.layers):
            if stage_index not in selected:
                x = layer(x)
                continue
            if layer.downsample is not None:
                x = layer.downsample(x)
            stage_attnres = self.encoder_stage_attnres[str(stage_index)]
            x, stage_aux = stage_attnres(
                x,
                layer.blocks,
                snr_db=snr_db,
                rate=rate,
                return_aux=return_aux,
            )
            if return_aux:
                attnres_aux.update(self._prefix_stage_aux(stage_index, stage_aux))
        x = self.net.encoder.norm(x)
        return self.net.encoder.head_list(x), attnres_aux

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_selected_stage_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
        return output


class SourceStage1FinalAttnResWrapper(SourceSelectedStageAttnResWrapper):
    """Encoder AttnRes on stage1 plus final stage; no decoder SNR adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, stage_indices=(1, -1), **kwargs)


class SourceStage1FinalAttnResDecoderAdapterWrapper(SourceStage1FinalAttnResWrapper):
    """Stage1+final encoder AttnRes plus an identity-safe receiver-side SNR adapter."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
    ):
        super().__init__(
            args,
            pass_channel=pass_channel,
            logger=logger,
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )
        self.decoder_snr_adapter = SourceDecoderSNRAdapter(self.net.encoder.head_list.out_features)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_selected_stage_attnres(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        decoder_feature, decoder_snr_aux = self.decoder_snr_adapter(
            noisy_feature,
            chan_param,
            return_aux=return_aux,
        )
        recon_image = self.net.decoder(decoder_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
            output["decoder_snr_aux"] = decoder_snr_aux
        return output


class SourceSNRAdaptiveAttnResWrapper(SourceSwinJSCCWrapper):
    """Source-faithful SNR-adaptive model with final-stage Block AttnRes before Channel ModNet."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = True,
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_SA:
            raise ValueError("SourceSNRAdaptiveAttnResWrapper requires the official SA tensor contract")
        final_layer = self.net.encoder.layers[-1]
        self.final_stage_attnres = StageBlockAttnRes(
            d_semantic=self.net.encoder.embed_dims[-1],
            num_swin_blocks=len(final_layer.blocks),
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )

    def _encode_attnres_semantic(
        self,
        image: torch.Tensor,
        snr_db=None,
        rate=None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        final_layer_index = len(self.net.encoder.layers) - 1
        attnres_aux: dict[str, object] = {}
        for layer_index, layer in enumerate(self.net.encoder.layers):
            if layer_index != final_layer_index:
                x = layer(x)
                continue
            if layer.downsample is not None:
                x = layer.downsample(x)
            x, attnres_aux = self.final_stage_attnres(
                x,
                layer.blocks,
                snr_db=snr_db,
                rate=rate,
                return_aux=return_aux,
            )
        return self.net.encoder.norm(x), attnres_aux

    def _apply_encoder_channel_modnet(self, z: torch.Tensor, snr_db) -> torch.Tensor:
        batch_size, token_count, _ = z.shape
        snr_batch = expand_scalar_like(
            snr_db,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
            default=13.0,
        ).unsqueeze(-1)
        temp = None
        for index in range(self.net.encoder.layer_num):
            if index == 0:
                temp = self.net.encoder.sm_list[index](z.detach())
            else:
                temp = self.net.encoder.sm_list[index](temp)
            snr_gate = self.net.encoder.bm_list[index](snr_batch).unsqueeze(1).expand(-1, token_count, -1)
            temp = temp * snr_gate
        mod_val = self.net.encoder.sigmoid(self.net.encoder.sm_list[-1](temp))
        return z * mod_val

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        semantic, attnres_aux = self._encode_attnres_semantic(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        feature = self.net.encoder.head_list(self._apply_encoder_channel_modnet(semantic, chan_param))
        cbr = feature.numel() / 2 / image.numel()
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output["attnres_aux"] = attnres_aux
        return output


@dataclass
class SourceRCSBRateOutput:
    z_tx: torch.Tensor
    gate: torch.Tensor
    value_gain: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    active_mean: torch.Tensor
    active_rms: torch.Tensor


class SourceRCSBRateAdapter(nn.Module):
    """Rate-conditioned soft modulation followed by exact hard channel budget."""

    def __init__(self, d_semantic: int, hidden_dim: Optional[int] = None, gain_mode: str = "sigmoid"):
        super().__init__()
        self.d_semantic = int(d_semantic)
        if gain_mode not in {
            "sigmoid",
            "active_norm",
            "active_rms",
            "decoupled_active_rms",
            "decoupled_st_active_rms",
            "decoupled_gumbel_st_active_rms",
        }:
            raise ValueError(f"Unsupported RCSB rate gain_mode={gain_mode}")
        self.gain_mode = gain_mode
        self.use_decoupled_score_value = self.gain_mode in {
            "decoupled_active_rms",
            "decoupled_st_active_rms",
            "decoupled_gumbel_st_active_rms",
        }
        self.use_gumbel_st_score_mask = self.gain_mode == "decoupled_gumbel_st_active_rms"
        self.use_st_score_mask = self.gain_mode in {
            "decoupled_st_active_rms",
            "decoupled_gumbel_st_active_rms",
        }
        self.st_mask_temperature = 1.0
        hidden = int(hidden_dim or round(self.d_semantic * 1.5))
        self.feature_proj = nn.Sequential(
            nn.Linear(self.d_semantic, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.rate_modulator = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.gate_head = nn.Linear(hidden, self.d_semantic)
        self.score_head = nn.Linear(hidden, self.d_semantic) if self.use_decoupled_score_value else None
        if self.score_head is not None and not self.use_st_score_mask:
            _set_module_requires_grad(self.score_head, False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _channel_masks(
        self,
        scores: torch.Tensor,
        rate_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_gumbel_st_score_mask and self.training:
            eps = torch.finfo(scores.dtype).eps
            uniform = torch.rand_like(scores).clamp_(min=eps, max=1.0 - eps)
            mask_scores = scores - torch.log(-torch.log(uniform))
        else:
            mask_scores = scores

        # hard_mask 是真正的预算执行结果：每个样本严格选择 top-k 个通道，保证 CBR/rate 语义不变。
        hard_mask = build_hard_channel_mask(mask_scores, rate_budget)
        if not self.use_st_score_mask or (self.use_gumbel_st_score_mask and not self.training):
            return hard_mask, hard_mask

        # ST 模式只改变反向传播路径：forward 数值仍等于 hard_mask，backward 通过 soft_mask 给 score_head 梯度。
        batch_size, num_channels = scores.shape
        k_index = rate_budget.to(device=scores.device, dtype=torch.long).reshape(batch_size).clamp_(1, num_channels) - 1
        sorted_scores = mask_scores.sort(dim=1, descending=True).values
        threshold = sorted_scores.gather(1, k_index.unsqueeze(1)).detach()
        soft_mask = torch.sigmoid((mask_scores - threshold) / self.st_mask_temperature)
        st_mask = hard_mask + soft_mask - soft_mask.detach()
        return hard_mask, st_mask

    def forward(self, z: torch.Tensor, rate, snr_db=None) -> SourceRCSBRateOutput:
        batch_size, _, d_semantic = z.shape
        if d_semantic != self.d_semantic:
            raise ValueError(f"Expected semantic dim {self.d_semantic}, got {d_semantic}")
        rate_budget = expand_rate_like(
            rate,
            batch_size=batch_size,
            device=z.device,
            default=self.d_semantic,
            max_value=self.d_semantic,
        )
        rate_cond = rate_budget.to(dtype=z.dtype).unsqueeze(-1)
        _ = snr_db
        # hidden 使用 z.detach()，保持 RCSB 控制器学习通道选择/增益，不直接把梯度回灌到 z 表征形成。
        hidden = self.feature_proj(z.detach())
        rate_hidden = hidden * self.rate_modulator(rate_cond).unsqueeze(1)
        # 根据 gain_mode 选择 score/value 的耦合方式。
        if self.use_decoupled_score_value:
            # decoupled 模式下 value gate 和 score head 分开：gate 控制传输幅度，scores 控制 top-k 选择。
            gate = torch.sigmoid(self.gate_head(rate_hidden))
            scores = self.score_head(rate_hidden).mean(dim=1)
        else:
            # 默认/active_norm/active_rms 仍使用同一路 gate 同时产生 value gain 和 score。
            gate = torch.sigmoid(self.gate_head(rate_hidden))
            scores = gate.mean(dim=1)

        # mask_c [B, D]：每样本 top-k 通道 hard mask；transmit_mask_c 在 ST 模式下 forward 仍等于 hard mask。
        mask_c, transmit_mask_c = self._channel_masks(scores, rate_budget)
        # hard_mask 用于统计 active 通道集合，保证 active_mean/active_rms 的归一化只看真实被选中的通道。
        hard_mask = mask_c.unsqueeze(1).expand_as(z)
        # transmit_mask 用于实际传输；普通模式为 hard mask，ST 模式带 soft-backward 梯度。
        transmit_mask = transmit_mask_c.unsqueeze(1).expand_as(z)
        # active_count 是每个样本选中的 token-channel 元素数，等于 rate_budget * token_count。
        active_count = hard_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        # active_mean 是选中通道上的 gate 均值，用于 active_norm。
        active_mean = (gate * hard_mask).sum(dim=(1, 2), keepdim=True) / active_count
        # active_rms 是选中通道上的 gate RMS，用于 active_rms / decoupled active-RMS 等模式。
        active_rms = torch.sqrt((gate.pow(2) * hard_mask).sum(dim=(1, 2), keepdim=True) / active_count)

        if self.gain_mode == "active_norm":
            # active_norm 约束选中通道的平均 value gain 接近 1。
            value_gain = gate / active_mean.clamp_min(1e-6)
        elif self.gain_mode in {
            "active_rms",
            "decoupled_active_rms",
            "decoupled_st_active_rms",
            "decoupled_gumbel_st_active_rms",
        }:
            # active-RMS 系列约束选中通道的 RMS value gain 接近 1，稳定信道功率尺度。
            value_gain = gate / active_rms.clamp_min(1e-6)
        else:
            # sigmoid 基线直接使用 gate 作为 value gain。
            value_gain = gate
        z_rate = z * value_gain
        return SourceRCSBRateOutput(
            z_tx=z_rate * transmit_mask,
            gate=gate,
            value_gain=value_gain,
            scores=scores,
            mask=transmit_mask,
            active_mean=active_mean,
            active_rms=active_rms,
        )


class SourceRCSBRateAdaptiveWrapper(SourceSwinJSCCWrapper):
    """Source-faithful RCSB rate-adaptive model with official RA tensor contract."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        gain_mode: str = "sigmoid",
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_RA:
            raise ValueError("SourceRCSBRateAdaptiveWrapper requires the official RA tensor contract")
        _set_module_requires_grad(self.net.encoder.bm_list, False)
        _set_module_requires_grad(self.net.encoder.sm_list, False)
        self.rcsb_rate_adapter = SourceRCSBRateAdapter(
            d_semantic=self.net.encoder.embed_dims[-1],
            gain_mode=gain_mode,
        )

    def _encode_semantic(self, image: torch.Tensor) -> torch.Tensor:
        x = self.net.encoder.patch_embed(image)
        for layer in self.net.encoder.layers:
            x = layer(x)
        return self.net.encoder.norm(x)

    def _encode_semantic_with_aux(self, image: torch.Tensor, snr_db=None, rate=None) -> tuple[torch.Tensor, dict[str, object]]:
        return self._encode_semantic(image), {}

    def forward(self, image: torch.Tensor, snr_db: Optional[float] = None, rate: Optional[int] = None) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_semantic_with_aux(image, snr_db=chan_param, rate=channel_number)
        rcsb_out = self.rcsb_rate_adapter(feature, channel_number, snr_db=chan_param)
        cbr = channel_number / (2 * 3 * 2 ** (self.net.downsample * 2))
        active_count = rcsb_out.mask.sum().clamp_min(1.0)
        avg_pwr = torch.sum(rcsb_out.z_tx ** 2) / active_count
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(rcsb_out.z_tx, chan_param, avg_pwr)
        else:
            noisy_feature = rcsb_out.z_tx
        noisy_feature = noisy_feature * rcsb_out.mask

        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
            "rcsb_aux": {
                "gate_mean": rcsb_out.gate.mean(),
                "gate_std": rcsb_out.gate.std(),
                "value_gain_mean": rcsb_out.value_gain.mean(),
                "value_gain_std": rcsb_out.value_gain.std(),
                "selected_gate_mean": (rcsb_out.gate * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_mean": (rcsb_out.value_gain * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_rms": torch.sqrt(
                    (rcsb_out.value_gain.pow(2) * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0)
                ),
                "active_mean": rcsb_out.active_mean.mean(),
                "active_rms": rcsb_out.active_rms.mean(),
                "score_mean": rcsb_out.scores.mean(),
                "mask_mean": rcsb_out.mask.mean(),
            },
        }
        if attnres_aux:
            output["attnres_aux"] = attnres_aux
        return output


class SourceRCSBUniversalWrapper(SourceSwinJSCCWrapper):
    """Universal RCSB model: official SNR ModNet contract plus RCSB rate control."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        gain_mode: str = "active_rms",
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger)
        if self.source_model_name != SOURCE_MODEL_SA_RA:
            raise ValueError("SourceRCSBUniversalWrapper requires the official SA+RA tensor contract")
        # RCSB replaces the official encoder Rate ModNet; keep it frozen to avoid unused trainable params.
        _set_module_requires_grad(self.net.encoder.bm_list, False)
        _set_module_requires_grad(self.net.encoder.sm_list, False)
        self.rcsb_rate_adapter = SourceRCSBRateAdapter(
            d_semantic=self.net.encoder.embed_dims[-1],
            gain_mode=gain_mode,
        )

    def _encode_semantic(self, image: torch.Tensor) -> torch.Tensor:
        x = self.net.encoder.patch_embed(image)
        for layer in self.net.encoder.layers:
            x = layer(x)
        return self.net.encoder.norm(x)

    def _apply_encoder_snr_modnet(self, z: torch.Tensor, snr_db) -> torch.Tensor:
        batch_size, token_count, _ = z.shape
        snr_batch = expand_scalar_like(
            snr_db,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
            default=13.0,
        ).unsqueeze(-1)
        temp = None
        for index in range(self.net.encoder.layer_num):
            if index == 0:
                temp = self.net.encoder.sm_list1[index](z.detach())
            else:
                temp = self.net.encoder.sm_list1[index](temp)
            snr_gate = self.net.encoder.bm_list1[index](snr_batch).unsqueeze(1).expand(-1, token_count, -1)
            temp = temp * snr_gate
        mod_val = self.net.encoder.sigmoid1(self.net.encoder.sm_list1[-1](temp))
        return z * mod_val

    def forward(self, image: torch.Tensor, snr_db: Optional[float] = None, rate: Optional[int] = None) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature = self._apply_encoder_snr_modnet(self._encode_semantic(image), chan_param)
        rcsb_out = self.rcsb_rate_adapter(feature, channel_number)
        cbr = channel_number / (2 * 3 * 2 ** (self.net.downsample * 2))
        active_count = rcsb_out.mask.sum().clamp_min(1.0)
        avg_pwr = torch.sum(rcsb_out.z_tx ** 2) / active_count
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(rcsb_out.z_tx, chan_param, avg_pwr)
        else:
            noisy_feature = rcsb_out.z_tx
        noisy_feature = noisy_feature * rcsb_out.mask

        recon_image = self.net.decoder(noisy_feature, chan_param, self.source_model_name)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        return {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
            "rcsb_aux": {
                "gate_mean": rcsb_out.gate.mean(),
                "gate_std": rcsb_out.gate.std(),
                "value_gain_mean": rcsb_out.value_gain.mean(),
                "value_gain_std": rcsb_out.value_gain.std(),
                "selected_gate_mean": (rcsb_out.gate * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_mean": (rcsb_out.value_gain * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_rms": torch.sqrt(
                    (rcsb_out.value_gain.pow(2) * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0)
                ),
                "active_mean": rcsb_out.active_mean.mean(),
                "active_rms": rcsb_out.active_rms.mean(),
                "score_mean": rcsb_out.scores.mean(),
                "mask_mean": rcsb_out.mask.mean(),
            },
        }


class SourceAttnResRCSBRateAdaptiveWrapper(SourceRCSBRateAdaptiveWrapper):
    """Final-stage Block AttnRes latent formation followed by source-faithful RCSB rate control."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        gain_mode: str = "decoupled_st_active_rms",
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "rate",
        attnres_condition_gate_use_reference: bool = True,
    ):
        super().__init__(args, pass_channel=pass_channel, logger=logger, gain_mode=gain_mode)
        final_layer = self.net.encoder.layers[-1]
        self.final_stage_attnres = StageBlockAttnRes(
            d_semantic=self.net.encoder.embed_dims[-1],
            num_swin_blocks=len(final_layer.blocks),
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )

    def _encode_semantic_with_aux(self, image: torch.Tensor, snr_db=None, rate=None) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        final_layer_index = len(self.net.encoder.layers) - 1
        attnres_aux: dict[str, object] = {}
        for layer_index, layer in enumerate(self.net.encoder.layers):
            if layer_index != final_layer_index:
                x = layer(x)
                continue
            if layer.downsample is not None:
                x = layer.downsample(x)
            x, attnres_aux = self.final_stage_attnres(x, layer.blocks, snr_db=snr_db, rate=rate)
        return self.net.encoder.norm(x), attnres_aux


class SourceNoModNetUniversalWrapper(SourceAttnResRCSBRateAdaptiveWrapper):
    """No-ModNet universal path: SNR-gated encoder AttnRes, rate-only RCSB, decoder SNR adapter."""

    def __init__(
        self,
        args: SourceSwinJSCCArgs,
        pass_channel: bool = True,
        logger=None,
        gain_mode: str = "active_rms",
        attnres_block_size: int = 2,
        attnres_query_mode: str = "content_condition_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = False,
    ):
        super().__init__(
            args,
            pass_channel=pass_channel,
            logger=logger,
            gain_mode=gain_mode,
            attnres_block_size=attnres_block_size,
            attnres_query_mode=attnres_query_mode,
            attnres_content_query_scale=attnres_content_query_scale,
            attnres_use_layer_embedding=attnres_use_layer_embedding,
            attnres_condition_gate_mode=attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
        )
        self.decoder_snr_adapter = SourceDecoderSNRAdapter(self.net.decoder.embed_dims[0])

    def _tokens_to_decoder_image(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels = tokens.shape
        return tokens.reshape(batch_size, self.net.decoder.H, self.net.decoder.W, channels).permute(0, 3, 1, 2)

    def _decode_no_modnet_tokens(self, tokens: torch.Tensor, snr_db=None, rate=None) -> tuple[torch.Tensor, dict[str, object]]:
        _ = snr_db, rate
        x = tokens
        for layer in self.net.decoder.layers:
            x = layer(x)
        return self._tokens_to_decoder_image(x), {}

    def forward(self, image: torch.Tensor, snr_db: Optional[float] = None, rate: Optional[int] = None) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_semantic_with_aux(image, snr_db=chan_param, rate=channel_number)
        rcsb_out = self.rcsb_rate_adapter(feature, channel_number, snr_db=chan_param)
        cbr = channel_number / (2 * 3 * 2 ** (self.net.downsample * 2))
        active_count = rcsb_out.mask.sum().clamp_min(1.0)
        avg_pwr = torch.sum(rcsb_out.z_tx ** 2) / active_count
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(rcsb_out.z_tx, chan_param, avg_pwr)
        else:
            noisy_feature = rcsb_out.z_tx
        noisy_feature = noisy_feature * rcsb_out.mask
        decoder_feature, decoder_snr_aux = self.decoder_snr_adapter(noisy_feature, chan_param)

        recon_image, decoder_attnres_aux = self._decode_no_modnet_tokens(
            decoder_feature,
            snr_db=chan_param,
            rate=channel_number,
        )
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
            "attnres_aux": attnres_aux,
            "rcsb_aux": {
                "gate_mean": rcsb_out.gate.mean(),
                "gate_std": rcsb_out.gate.std(),
                "value_gain_mean": rcsb_out.value_gain.mean(),
                "value_gain_std": rcsb_out.value_gain.std(),
                "selected_gate_mean": (rcsb_out.gate * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_mean": (rcsb_out.value_gain * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0),
                "selected_gain_rms": torch.sqrt(
                    (rcsb_out.value_gain.pow(2) * rcsb_out.mask).sum() / rcsb_out.mask.sum().clamp_min(1.0)
                ),
                "active_mean": rcsb_out.active_mean.mean(),
                "active_rms": rcsb_out.active_rms.mean(),
                "score_mean": rcsb_out.scores.mean(),
                "mask_mean": rcsb_out.mask.mean(),
            },
            "decoder_snr_aux": decoder_snr_aux,
        }
        if decoder_attnres_aux:
            output["decoder_attnres_aux"] = decoder_attnres_aux
        return output


class DecompJSCC(SourceRCSBRateAdaptiveWrapper):
    """最终 DecompJSCC：全阶段/跨阶段 SNR 路由、RCSB rate 控制和 decoder SNR 校正。"""

    method_name = DecompJSCCConfig.METHOD_NAME
    method_version = DecompJSCCConfig.METHOD_VERSION

    def __init__(
        self,
        args: SourceSwinJSCCArgs | DecompJSCCConfig,
        pass_channel: bool = True,
        logger=None,
        gain_mode: str = "active_rms",
        attnres_block_size: int = 2,
        attnres_query_mode: str = "snr_depth_logit_gate",
        attnres_content_query_scale: float = 1.0,
        attnres_use_layer_embedding: bool = True,
        attnres_condition_gate_mode: str = "snr",
        attnres_condition_gate_use_reference: bool = True,
        attnres_cross_stage_skip_threshold: float = 0.0,
        attnres_cross_stage_force_skip: bool = False,
    ):
        # isinstance(x, T)：判断 x 是否是类型/类 T 的实例（或其子类实例）。
        # 这里允许 __init__ 同时接收两种入参：
        # 1) DecompJSCCConfig：先保存配置，再转换成父类需要的 SourceSwinJSCCArgs
        # 2) 已是 SourceSwinJSCCArgs：不转换，decompjscc_config 记为 None
        self.decompjscc_config = args if isinstance(args, DecompJSCCConfig) else None
        if isinstance(args, DecompJSCCConfig):
            # 把 DecompJSCCConfig 里的字段映射成 build_source_swinjscc_args 需要的参数格式
            # （例如 rate_channels / snr_db_values 转成逗号分隔字符串 C / multiple_snr）
            args = build_source_swinjscc_args(
                recipe_stage="no_modnet_sall_cross_universal",
                trainset=args.trainset,
                testset=args.testset,
                distortion_metric=args.distortion_metric,
                channel_type=args.channel_type,
                C=",".join(str(int(rate)) for rate in args.rate_channels),
                multiple_snr=",".join(f"{float(snr):g}" for snr in args.snr_db_values),
                model_size=args.model_size,
                img_size=args.image_size,
                batch_size=args.batch_size,
                embed_dims=args.embed_dims,
                depths=args.encoder_depths,
                decoder_depths=args.decoder_depths,
                num_heads=args.num_heads,
            )
        super().__init__(args, pass_channel=pass_channel, logger=logger, gain_mode=gain_mode)
        self.enable_cross_stage_attnres = True
        self.encoder_stage_attnres = nn.ModuleList(
            [
                StageBlockAttnRes(
                    d_semantic=self.net.encoder.embed_dims[stage_index],
                    num_swin_blocks=len(layer.blocks),
                    attnres_block_size=attnres_block_size,
                    attnres_query_mode=attnres_query_mode,
                    attnres_content_query_scale=attnres_content_query_scale,
                    attnres_use_layer_embedding=attnres_use_layer_embedding,
                    attnres_condition_gate_mode=attnres_condition_gate_mode,
                    attnres_condition_gate_use_reference=attnres_condition_gate_use_reference,
                    enable_cross_stage_attnres=stage_index > 1,
                    attnres_cross_stage_skip_threshold=attnres_cross_stage_skip_threshold,
                    attnres_cross_stage_force_skip=attnres_cross_stage_force_skip,
                )
                for stage_index, layer in enumerate(self.net.encoder.layers)
            ]
        )
        self.decoder_snr_adapter = SourceDecoderSNRAdapter(self.net.decoder.embed_dims[0])

    @property
    def attention_residuals(self):
        return self.encoder_stage_attnres

    @property
    def rcsb(self):
        return self.rcsb_rate_adapter

    @property
    def receiver_snr_adapter(self):
        return self.decoder_snr_adapter

    def public_metadata(self) -> dict:
        # 如果 DecompJSCCConfig 已存在，直接返回其 public_metadata
        # 否则从 source_args 重建一个 DecompJSCCConfig
        config = self.decompjscc_config
        # 如果 DecompJSCCConfig 不存在，从 source_args 重建一个
        if config is None:
            config = DecompJSCCConfig(
                channel_type=self.source_args.channel_type,
                rate_channels=tuple(int(value) for value in str(self.source_args.C).split(",")),
                snr_db_values=tuple(float(value) for value in str(self.source_args.multiple_snr).split(",")),
                image_size=self.source_args.img_size,
                model_size=self.source_args.model_size,
                batch_size=self.source_args.batch_size,
                distortion_metric=self.source_args.distortion_metric,
                trainset=self.source_args.trainset,
                testset=self.source_args.testset,
            )
        return config.to_public_metadata()

    @staticmethod
    def _prefix_stage_aux(stage_index: int, aux: dict[str, object]) -> dict[str, object]:
        return {f"stage{stage_index}_{key}": value for key, value in aux.items()}

    def _project_stage_output_to_current(
        self,
        source: torch.Tensor,
        source_stage_index: int,
        target_stage_index: int,
    ) -> torch.Tensor:
        projected = source
        for stage_index in range(source_stage_index + 1, target_stage_index + 1):
            downsample = self.net.encoder.layers[stage_index].downsample
            if downsample is not None:
                projected = downsample(projected)
        return projected

    def _cross_stage_sources_for_stage(
        self,
        stage_outputs: list[torch.Tensor],
        target_stage_index: int,
    ) -> list[torch.Tensor]:
        if target_stage_index <= 1:
            return []
        external_sources: list[torch.Tensor] = []
        # The immediate previous stage is already represented by the current stage input after PatchMerging.
        for source_stage_index, source in enumerate(stage_outputs[: target_stage_index - 1]):
            external_sources.append(
                self._project_stage_output_to_current(source, source_stage_index, target_stage_index)
            )
        return external_sources

    def _encode_semantic_with_aux(
        self,
        image: torch.Tensor,
        snr_db=None,
        rate=None,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        x = self.net.encoder.patch_embed(image)
        attnres_aux: dict[str, object] = {}
        stage_outputs: list[torch.Tensor] = []
        for stage_index, (layer, stage_attnres) in enumerate(
            zip(self.net.encoder.layers, self.encoder_stage_attnres)
        ):
            if layer.downsample is not None:
                x = layer.downsample(x)
            external_sources = self._cross_stage_sources_for_stage(stage_outputs, stage_index)
            x, stage_aux = stage_attnres(
                x,
                layer.blocks,
                snr_db=snr_db,
                rate=rate,
                external_sources=external_sources,
                return_aux=return_aux,
            )
            if return_aux:
                attnres_aux.update(self._prefix_stage_aux(stage_index, stage_aux))
            stage_outputs.append(x)
        return self.net.encoder.norm(x), attnres_aux

    def _tokens_to_decoder_image(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels = tokens.shape
        return tokens.reshape(batch_size, self.net.decoder.H, self.net.decoder.W, channels).permute(0, 3, 1, 2)

    def _decode_no_modnet_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for layer in self.net.decoder.layers:
            x = layer(x)
        return self._tokens_to_decoder_image(x)

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None,
        rate: Optional[int] = None,
        return_aux: bool = True,
    ) -> dict:
        _, _, height, width = image.shape
        if height != self.net.H or width != self.net.W:
            self.net.encoder.update_resolution(height, width)
            self.net.decoder.update_resolution(height // (2 ** self.net.downsample), width // (2 ** self.net.downsample))
            self.net.H = height
            self.net.W = width

        chan_param = float(snr_db) if snr_db is not None else float(choice(self.net.multiple_snr))
        channel_number = int(rate) if rate is not None else int(choice(self.net.channel_number))
        feature, attnres_aux = self._encode_semantic_with_aux(
            image,
            snr_db=chan_param,
            rate=channel_number,
            return_aux=return_aux,
        )
        rcsb_out = self.rcsb_rate_adapter(feature, channel_number, snr_db=chan_param)
        cbr = channel_number / (2 * 3 * 2 ** (self.net.downsample * 2))
        active_count = rcsb_out.mask.sum().clamp_min(1.0)
        avg_pwr = torch.sum(rcsb_out.z_tx ** 2) / active_count
        if self.pass_channel:
            noisy_feature = self.net.feature_pass_channel(rcsb_out.z_tx, chan_param, avg_pwr)
        else:
            noisy_feature = rcsb_out.z_tx
        noisy_feature = noisy_feature * rcsb_out.mask
        decoder_feature, decoder_snr_aux = self.decoder_snr_adapter(
            noisy_feature,
            chan_param,
            return_aux=return_aux,
        )

        recon_image = self._decode_no_modnet_tokens(decoder_feature)
        mse = self.net.squared_difference(image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0)
        loss = self.net.distortion_loss.forward(image, recon_image.clamp(0.0, 1.0))
        output = {
            "reconstructed": recon_image,
            "cbr": float(cbr),
            "snr_db": float(chan_param),
            "mse": mse.mean(),
            "loss": loss.mean(),
        }
        if return_aux:
            output.update(
                {
                    "attnres_aux": attnres_aux,
                    "rcsb_aux": {
                        "gate_mean": rcsb_out.gate.mean(),
                        "gate_std": rcsb_out.gate.std(),
                        "value_gain_mean": rcsb_out.value_gain.mean(),
                        "value_gain_std": rcsb_out.value_gain.std(),
                        "selected_gate_mean": (rcsb_out.gate * rcsb_out.mask).sum()
                        / rcsb_out.mask.sum().clamp_min(1.0),
                        "selected_gain_mean": (rcsb_out.value_gain * rcsb_out.mask).sum()
                        / rcsb_out.mask.sum().clamp_min(1.0),
                        "selected_gain_rms": torch.sqrt(
                            (rcsb_out.value_gain.pow(2) * rcsb_out.mask).sum()
                            / rcsb_out.mask.sum().clamp_min(1.0)
                        ),
                        "active_mean": rcsb_out.active_mean.mean(),
                        "active_rms": rcsb_out.active_rms.mean(),
                        "score_mean": rcsb_out.scores.mean(),
                        "mask_mean": rcsb_out.mask.mean(),
                    },
                    "decoder_snr_aux": decoder_snr_aux,
                }
            )
        return output


SourceAllStageCrossAttnResRCSBUniversalWrapper = DecompJSCC


def load_source_checkpoint(
    model: SourceSwinJSCCWrapper,
    checkpoint_path: str,
    strict: bool = False,
) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.12 compatibility
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return {"loaded": len(state_dict), "skipped": []}

    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        current_value = current.get(key)
        if current_value is None or tuple(current_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "skipped": skipped}
