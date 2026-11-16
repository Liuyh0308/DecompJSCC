# -*- coding: utf-8 -*-
"""Source training loop shared by SwinJSCC and the paper's DecompJSCC stages."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from decompjscc_utils import capture_training_rng_state, load_training_state
from distributed_utils import DistributedContext, add_distributed_args, create_dist_context, setup_seed
from models.swinjscc_source_wrapper import (
    SOURCE_HR_RATE_LIST,
    SOURCE_HR_SNR_LIST,
    SOURCE_MODEL_RA,
    SOURCE_MODEL_SA,
    SOURCE_MODEL_SA_RA,
    SOURCE_MODEL_WO_SA_RA,
    DecompJSCC,
    SourceAllStageAttnResDecoderAdapterWrapper,
    SourceAllStageAttnResWrapper,
    SourceAttnResRCSBRateAdaptiveWrapper,
    SourceFixedAttnResWrapper,
    SourceNoModNetUniversalWrapper,
    SourceNoModNetSNRAttnResDecoderAdapterWrapper,
    SourceRCSBRateAdaptiveWrapper,
    SourceRCSBUniversalWrapper,
    SourceSNRAdaptiveAttnResWrapper,
    SourceStage1FinalAttnResDecoderAdapterWrapper,
    SourceStage1FinalAttnResWrapper,
    SourceSwinJSCCWrapper,
    build_source_swinjscc_args,
    load_source_checkpoint,
)
from swinjscc_utils import build_snr_list, compute_ssim_batch, iter_limited, parse_float_list, parse_int_list
from swinjscc_hr_dataset import HRCroppedEvalImageDataset, HRRandomCropImageDataset, split_path_list


DEFAULT_HR_DATA_ROOT = "datasets/swinjscc_hr"


SOURCE_MODELS = [SOURCE_MODEL_WO_SA_RA, SOURCE_MODEL_SA, SOURCE_MODEL_RA, SOURCE_MODEL_SA_RA]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-faithful SwinJSCC HR reconstruction training")
    parser.add_argument(
        "--recipe-stage",
        default="universal",
        choices=[
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
            "no_modnet_universal",
            "no_modnet_sall_cross_universal",
            "snr_adaptive",
            "snr_adaptive_attnres",
            "rate_adaptive",
            "rcsb_rate_adaptive",
            "attnres_rcsb_rate_adaptive",
            "universal",
            "rcsb_universal_snr_warmup",
            "rcsb_universal_joint",
        ],
    )
    parser.add_argument("--source-model", default="", choices=["", *SOURCE_MODELS])
    parser.add_argument("--data-dir", default=DEFAULT_HR_DATA_ROOT, type=str)
    parser.add_argument("--hr-train-dirs", default="", type=str)
    parser.add_argument("--hr-val-dirs", default="", type=str)
    parser.add_argument("--hr-eval-split", default="kodak", choices=["kodak", "clic2021", "combined"])
    parser.add_argument("--img-size", default=256, type=int)
    parser.add_argument("--hr-crop-size", default=256, type=int)
    parser.add_argument("--hr-eval-crop-multiple", default=128, type=int)
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--C", default="", type=str)
    parser.add_argument("--multiple-snr", default="", type=str)
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--distortion-metric", default="MSE", choices=["MSE", "MS-SSIM"])
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--val-batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--checkpoint-path", default="checkpoints/decompjscc/training/source", type=str)
    parser.add_argument("--exp-log-dir", default="exp_log/decompjscc/training/source", type=str)
    parser.add_argument("--log-dir", default="runs/decompjscc/training/source", type=str)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without training")
    parser.add_argument("--resume-checkpoint", default="", type=str)
    parser.add_argument("--resume-strict", action="store_true")
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help="Resume epoch, Adam, best metric, and RNG state from --resume-checkpoint.",
    )
    parser.add_argument(
        "--freeze-mode",
        default="none",
        choices=["none", "snr_modnet_only", "snr_attnres_adapter_only", "rcsb_rate_adapter_only"],
    )
    parser.add_argument(
        "--rcsb-rate-gain-mode",
        default="sigmoid",
        choices=[
            "sigmoid",
            "active_norm",
            "active_rms",
            "decoupled_active_rms",
            "decoupled_st_active_rms",
            "decoupled_gumbel_st_active_rms",
        ],
    )
    parser.add_argument(
        "--attnres-query-mode",
        default="static",
        choices=[
            "static",
            "condition",
            "content",
            "content_condition_gate",
            "sourcewise_logit_gate",
            "layerwise_snr_gate",
            "snr_depth_logit_gate",
        ],
    )
    parser.add_argument("--attnres-content-query-scale", default=1.0, type=float)
    parser.add_argument(
        "--attnres-condition-gate-mode",
        default="snr_rate",
        choices=["snr_rate", "rate", "snr"],
    )
    parser.add_argument("--no-attnres-condition-gate-reference", action="store_true")
    parser.add_argument("--no-attnres-layer-embedding", action="store_true")
    parser.add_argument(
        "--attnres-cross-stage-skip-threshold",
        default=0.0,
        type=float,
        help="Inference/pruning knob: skip cross-stage AttnRes residual when |tanh(gate)| is below this value.",
    )
    parser.add_argument(
        "--attnres-cross-stage-force-skip",
        action="store_true",
        help="Disable cross-stage AttnRes residual while keeping the cross-stage model/checkpoint structure.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-train-batches", default=None, type=int)
    parser.add_argument("--max-val-batches", default=None, type=int)
    add_distributed_args(parser)
    return parser


def _default_source_train_dirs(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "CLIC2020"),
        os.path.join(data_dir, "CLIC2021", "train"),
        os.path.join(data_dir, "CLIC2021", "valid"),
        os.path.join(data_dir, "CLIC2022", "val"),
        os.path.join(data_dir, "DIV2K", "DIV2K_train_HR"),
        os.path.join(data_dir, "DIV2K", "DIV2K_valid_HR"),
    ]
    return ",".join(candidates)


def _default_source_val_dirs(data_dir: str, split: str) -> str:
    split_dirs = {
        "kodak": [os.path.join(data_dir, "Kodak")],
        "clic2021": [os.path.join(data_dir, "CLIC2021", "test")],
        "combined": [os.path.join(data_dir, "Kodak"), os.path.join(data_dir, "CLIC2021", "test")],
    }
    return ",".join(split_dirs[split])


def build_source_args_from_runtime(args: argparse.Namespace):
    return build_source_swinjscc_args(
        recipe_stage=args.recipe_stage,
        trainset="DIV2K",
        testset="CLIC21" if args.hr_eval_split == "clic2021" else args.hr_eval_split,
        distortion_metric=args.distortion_metric,
        model=args.source_model or None,
        channel_type=args.channel_type,
        C=args.C or None,
        multiple_snr=args.multiple_snr or None,
        model_size=args.model_size,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )


def _choose(values: Iterable[int | float]) -> int | float:
    items = list(values)
    if not items:
        raise ValueError("Cannot sample from an empty condition list")
    return random.choice(items)


def sample_source_condition(source_args, dist_ctx: DistributedContext | None = None) -> tuple[float, int]:
    if dist_ctx is not None and dist_ctx.mode.value == "ddp":
        condition = torch.zeros(2, dtype=torch.float32, device=dist_ctx.device)
        if dist_ctx.is_main:
            condition[0] = float(_choose(parse_float_list(source_args.multiple_snr)))
            condition[1] = float(_choose(parse_int_list(source_args.C)))
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(condition, src=0)
        return float(condition[0].item()), int(round(float(condition[1].item())))

    snr = float(_choose(parse_float_list(source_args.multiple_snr)))
    rate = int(_choose(parse_int_list(source_args.C)))
    return snr, rate


def build_source_datasets(args: argparse.Namespace):
    if not args.hr_train_dirs:
        args.hr_train_dirs = _default_source_train_dirs(args.data_dir)
    if not args.hr_val_dirs:
        args.hr_val_dirs = _default_source_val_dirs(args.data_dir, args.hr_eval_split)
    train_dataset = HRRandomCropImageDataset(split_path_list(args.hr_train_dirs), crop_size=args.hr_crop_size)
    val_dataset = HRCroppedEvalImageDataset(
        split_path_list(args.hr_val_dirs),
        crop_multiple=args.hr_eval_crop_multiple,
    )
    return train_dataset, val_dataset


def _psnr_from_mse(mse: torch.Tensor) -> float:
    mse_value = float(mse.detach().mean().clamp_min(1e-10).item())
    return float(10.0 * np.log10(255.0 * 255.0 / mse_value))


def collect_source_aux_scalars(outputs: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for aux_key, prefix in (
        ("attnres_aux", "attnres"),
        ("decoder_attnres_aux", "decoder_attnres"),
        ("rcsb_aux", "rcsb"),
        ("decoder_snr_aux", "decoder_snr"),
    ):
        aux = outputs.get(aux_key)
        if not isinstance(aux, dict):
            continue
        for key, value in aux.items():
            metric_name = key
            if prefix == "decoder_attnres" and metric_name.startswith("attnres_"):
                metric_name = metric_name[len("attnres_") :]
            metric_key = metric_name if metric_name.startswith(f"{prefix}_") else f"{prefix}_{metric_name}"
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                metrics[metric_key] = float(value.detach().item())
            elif isinstance(value, (int, float)):
                metrics[metric_key] = float(value)
    return metrics


def train_epoch(epoch, model, train_loader, optimizer, source_args, runtime_args, dist_ctx: DistributedContext):
    model.train()
    dist_ctx.set_epoch(train_loader, epoch)
    accumulation_steps = int(runtime_args.gradient_accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    total_loss = 0.0
    total_psnr = 0.0
    total_cbr = 0.0
    total_aux: dict[str, float] = {}
    num_batches = 0
    optimizer_steps = 0
    condition = None
    optimizer.zero_grad()

    for micro_batch_index, (images, _) in enumerate(train_loader):
        if runtime_args.max_train_batches is not None and optimizer_steps >= runtime_args.max_train_batches:
            break
        if micro_batch_index % accumulation_steps == 0:
            condition = sample_source_condition(source_args, dist_ctx)
        images = images.to(dist_ctx.device)
        snr_db, rate = condition
        outputs = model(images, snr_db=snr_db, rate=rate)
        loss = outputs["loss"]
        (loss / accumulation_steps).backward()
        if (micro_batch_index + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1

        total_loss += float(loss.item())
        total_psnr += _psnr_from_mse(outputs["mse"])
        total_cbr += float(outputs["cbr"])
        for key, value in collect_source_aux_scalars(outputs).items():
            total_aux[key] = total_aux.get(key, 0.0) + value
        num_batches += 1

    metrics = torch.tensor(
        [
            total_loss / max(num_batches, 1),
            total_psnr / max(num_batches, 1),
            total_cbr / max(num_batches, 1),
        ],
        device=dist_ctx.device,
    )
    metrics = dist_ctx.reduce_mean(metrics)
    result = {
        "loss": float(metrics[0].item()),
        "psnr": float(metrics[1].item()),
        "cbr": float(metrics[2].item()),
    }
    for key in sorted(total_aux):
        aux_metric = torch.tensor(total_aux[key] / max(num_batches, 1), device=dist_ctx.device)
        aux_metric = dist_ctx.reduce_mean(aux_metric)
        result[key] = float(aux_metric.item())
    return result


def validate(model, val_loader, source_args, runtime_args, dist_ctx: DistributedContext):
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    zero = {"psnr": 0.0, "ssim": 0.0, "cbr": 0.0}

    if dist_ctx.is_main or dist_ctx.mode.value != "ddp":
        snr_values = parse_float_list(source_args.multiple_snr)
        rate_values = parse_int_list(source_args.C)
        total_psnr = 0.0
        total_ssim = 0.0
        total_cbr = 0.0
        total_aux: dict[str, float] = {}
        total_aux_count = 0
        num_points = 0
        with torch.no_grad():
            for snr_db in snr_values:
                for rate in rate_values:
                    point_psnr = 0.0
                    point_ssim = 0.0
                    point_cbr = 0.0
                    num_batches = 0
                    for images, _ in iter_limited(val_loader, runtime_args.max_val_batches):
                        images = images.to(dist_ctx.device)
                        outputs = base_model(images, snr_db=float(snr_db), rate=int(rate))
                        reconstructed = outputs["reconstructed"].clamp(0.0, 1.0)
                        point_psnr += _psnr_from_mse(outputs["mse"])
                        point_ssim += float(compute_ssim_batch(images.clamp(0, 1), reconstructed, dist_ctx.device).mean().item())
                        point_cbr += float(outputs["cbr"])
                        for key, value in collect_source_aux_scalars(outputs).items():
                            total_aux[key] = total_aux.get(key, 0.0) + value
                        total_aux_count += 1
                        num_batches += 1
                    total_psnr += point_psnr / max(num_batches, 1)
                    total_ssim += point_ssim / max(num_batches, 1)
                    total_cbr += point_cbr / max(num_batches, 1)
                    num_points += 1
        result = {
            "psnr": total_psnr / max(num_points, 1),
            "ssim": total_ssim / max(num_points, 1),
            "cbr": total_cbr / max(num_points, 1),
        }
        for key in sorted(total_aux):
            result[key] = total_aux[key] / max(total_aux_count, 1)
    else:
        result = zero

    if dist_ctx.mode.value == "ddp":
        dist_ctx.barrier()
    if not dist_ctx.is_main:
        return zero
    return result


def reset_source_train_resolution(model, runtime_args):
    base_model = model.module if hasattr(model, "module") else model
    train_size = int(runtime_args.hr_crop_size)
    downsample = int(getattr(base_model.net, "downsample", 4))
    base_model.net.encoder.update_resolution(train_size, train_size)
    base_model.net.decoder.update_resolution(train_size // (2 ** downsample), train_size // (2 ** downsample))
    base_model.net.H = train_size
    base_model.net.W = train_size


def configure_source_trainable_parameters(model, freeze_mode: str) -> dict:
    if freeze_mode == "none":
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
        return {"trainable": trainable, "frozen": frozen}

    trainable_prefixes_by_mode = {
        "snr_modnet_only": (
            "net.encoder.bm_list1.",
            "net.encoder.sm_list1.",
            "net.decoder.bm_list.",
            "net.decoder.sm_list.",
        ),
        "snr_attnres_adapter_only": (
            "final_stage_attnres.",
            "encoder_stage_attnres.",
            "decoder_snr_adapter.",
        ),
        "rcsb_rate_adapter_only": (
            "rcsb_rate_adapter.",
        ),
    }
    if freeze_mode not in trainable_prefixes_by_mode:
        raise ValueError(f"Unsupported freeze_mode={freeze_mode}")

    for param in model.parameters():
        param.requires_grad = False
    trainable_prefixes = trainable_prefixes_by_mode[freeze_mode]
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
    rcsb_rate_adapter = getattr(model, "rcsb_rate_adapter", None)
    if (
        freeze_mode == "rcsb_rate_adapter_only"
        and rcsb_rate_adapter is not None
        and getattr(rcsb_rate_adapter, "use_decoupled_score_value", False)
        and not getattr(rcsb_rate_adapter, "use_st_score_mask", False)
    ):
        score_head = getattr(rcsb_rate_adapter, "score_head", None)
        if score_head is not None:
            for param in score_head.parameters():
                param.requires_grad = False
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    if trainable == 0:
        raise ValueError(f"freeze_mode={freeze_mode} selected no trainable parameters")
    return {"trainable": trainable, "frozen": frozen}


def build_train_source_model(args, source_args):
    if args.recipe_stage == "fixed_attnres":
        return SourceFixedAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_attnres":
        return SourceFixedAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_attnres_decoder_adapter":
        return SourceNoModNetSNRAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_attnres":
        return SourceAllStageAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_attnres_decoder_adapter":
        return SourceAllStageAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_cross_attnres_decoder_adapter":
        return SourceAllStageAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
            enable_cross_stage_attnres=True,
            attnres_cross_stage_skip_threshold=args.attnres_cross_stage_skip_threshold,
            attnres_cross_stage_force_skip=args.attnres_cross_stage_force_skip,
        )
    if args.recipe_stage == "no_modnet_snr_stage1_final_attnres":
        return SourceStage1FinalAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stage1_final_attnres_decoder_adapter":
        return SourceStage1FinalAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_universal":
        return SourceNoModNetUniversalWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "no_modnet_sall_cross_universal":
        return DecompJSCC(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
            attnres_cross_stage_skip_threshold=args.attnres_cross_stage_skip_threshold,
            attnres_cross_stage_force_skip=args.attnres_cross_stage_force_skip,
        )
    if args.recipe_stage == "snr_adaptive_attnres":
        return SourceSNRAdaptiveAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage == "rcsb_rate_adaptive":
        return SourceRCSBRateAdaptiveWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
        )
    if args.recipe_stage == "attnres_rcsb_rate_adaptive":
        return SourceAttnResRCSBRateAdaptiveWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=not args.no_attnres_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=not args.no_attnres_condition_gate_reference,
        )
    if args.recipe_stage in {"rcsb_universal_snr_warmup", "rcsb_universal_joint"}:
        return SourceRCSBUniversalWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
        )
    return SourceSwinJSCCWrapper(source_args, pass_channel=True)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.dry_run:
        print(json.dumps({"runtime": vars(args), "source_args": vars(build_source_args_from_runtime(args))}, indent=2))
        return
    dist_ctx = create_dist_context(args)
    if args.resume_training_state and dist_ctx.mode.value != "single":
        raise ValueError("--resume-training-state currently requires --dist-mode single")
    if args.resume_training_state and not args.resume_checkpoint:
        raise ValueError("--resume-training-state requires --resume-checkpoint")
    setup_seed(args.seed, dist_ctx)
    random.seed(args.seed + int(getattr(dist_ctx, "local_rank", 0)))
    np.random.seed(args.seed + int(getattr(dist_ctx, "local_rank", 0)))

    source_args = build_source_args_from_runtime(args)
    train_dataset, val_dataset = build_source_datasets(args)
    train_loader = dist_ctx.get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        is_train=True,
    )
    val_loader = dist_ctx.get_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        is_train=False,
    )

    model = build_train_source_model(args, source_args).to(dist_ctx.device)
    if args.resume_checkpoint:
        load_source_checkpoint(model, args.resume_checkpoint, strict=args.resume_strict)
    trainable_summary = configure_source_trainable_parameters(model, args.freeze_mode)
    runtime_config = vars(args).copy()
    runtime_config["attnres_condition_gate_use_reference"] = not args.no_attnres_condition_gate_reference
    # Pre-resolve Swin attention-mask buffer shapes to prevent DDP collective mismatch.
    _dummy = torch.randn(1, 3, int(args.img_size), int(args.img_size), device=dist_ctx.device)
    _snr, _rate = sample_source_condition(source_args)
    _ = model(_dummy, snr_db=float(_snr), rate=int(_rate))
    model = dist_ctx.wrap_model(model)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args.lr)

    start_epoch = 0
    best_psnr = float("-inf")
    if args.resume_training_state:
        resume = load_training_state(args.resume_checkpoint, optimizer, dist_ctx.device)
        start_epoch = int(resume["start_epoch"])
        best_psnr = float(resume["best_psnr"])
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint epoch {start_epoch} already reached target --epochs {args.epochs}"
            )

    writer = None
    if dist_ctx.is_main:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        os.makedirs(args.exp_log_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.exp_log_dir, "config.json"), "w", encoding="utf-8") as fh:
            payload = runtime_config.copy()
            payload["source_args"] = vars(source_args)
            payload["trainable_summary"] = trainable_summary
            payload["start_epoch"] = start_epoch
            json.dump(payload, fh, indent=2)
        writer = SummaryWriter(os.path.join(args.log_dir, time.strftime("%Y%m%d_%H%M%S")))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, source_args, args, dist_ctx)
        val_metrics = validate(model, val_loader, source_args, args, dist_ctx)
        reset_source_train_resolution(model, args)

        if dist_ctx.is_main:
            is_best = val_metrics["psnr"] > best_psnr
            if is_best:
                best_psnr = val_metrics["psnr"]
            state = {
                "epoch": epoch + 1,
                "model_state_dict": dist_ctx.get_state_dict(model),
                "source_args": vars(source_args),
                "config": runtime_config,
                "best_psnr": best_psnr,
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": capture_training_rng_state(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            torch.save(state, os.path.join(args.checkpoint_path, "last.pth"))
            if is_best:
                torch.save(state, os.path.join(args.checkpoint_path, "best.pth"))
                torch.save(state, os.path.join(args.checkpoint_path, "best_psnr.pth"))
            with open(os.path.join(args.exp_log_dir, "training_history.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train": train_metrics,
                            "validation": val_metrics,
                            "best_psnr": best_psnr,
                        }
                    )
                    + "\n"
                )
            if writer is not None:
                writer.add_scalar("loss/train", train_metrics["loss"], epoch)
                writer.add_scalar("psnr/train", train_metrics["psnr"], epoch)
                writer.add_scalar("psnr/val", val_metrics["psnr"], epoch)
                writer.add_scalar("ssim/val", val_metrics["ssim"], epoch)
                writer.add_scalar("cbr/train", train_metrics["cbr"], epoch)
                writer.add_scalar("cbr/val", val_metrics["cbr"], epoch)
                for key, value in train_metrics.items():
                    if key not in {"loss", "psnr", "cbr"}:
                        writer.add_scalar(f"aux_train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    if key not in {"psnr", "ssim", "cbr"}:
                        writer.add_scalar(f"aux_val/{key}", value, epoch)
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"train_psnr={train_metrics['psnr']:.2f}, "
                f"val_psnr={val_metrics['psnr']:.2f}, "
                f"val_ssim={val_metrics['ssim']:.4f}, "
                f"val_cbr={val_metrics['cbr']:.4f}"
            )
        dist_ctx.barrier()

    if writer is not None:
        writer.close()
    dist_ctx.cleanup()


if __name__ == "__main__":
    main()
