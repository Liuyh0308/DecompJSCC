# -*- coding: utf-8 -*-
"""Evaluate source-faithful SwinJSCC checkpoints on an SNR sweep."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from decompjscc_utils import channel_implementation_metadata
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
    source_model_name_for_recipe_stage,
    source_rate_list_for_recipe_stage,
)
from swinjscc_utils import build_snr_list, compute_ssim_batch, parse_float_list, parse_int_list
from swinjscc_hr_dataset import HRCroppedEvalImageDataset, split_path_list
from experiments.source_training import (
    DEFAULT_HR_DATA_ROOT,
    _default_source_val_dirs,
)
from experiments.resources import load_manifest, resource_path


SOURCE_MODELS = [SOURCE_MODEL_WO_SA_RA, SOURCE_MODEL_SA, SOURCE_MODEL_RA, SOURCE_MODEL_SA_RA]


def infer_exp_log_dir_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    marker = f"{os.sep}checkpoints{os.sep}"
    if marker in checkpoint_dir:
        return checkpoint_dir.replace(marker, f"{os.sep}exp_log{os.sep}", 1)
    return checkpoint_dir


def resolve_config_path(checkpoint_path: str, config_path: str) -> str:
    if config_path:
        return config_path
    checkpoint = resource_path(checkpoint_path).resolve()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for record in load_manifest(repo_root)["weights"]:
        if resource_path(record["checkpoint"]).resolve() == checkpoint and record.get("config"):
            return str(resource_path(record["config"]))
    candidate_dirs = [
        infer_exp_log_dir_from_checkpoint(checkpoint_path),
        os.path.dirname(os.path.abspath(checkpoint_path)),
    ]
    for base_dir in candidate_dirs:
        candidate = os.path.join(base_dir, "config.json")
        if os.path.isfile(candidate):
            return candidate
    return ""


def resolve_output_json_path(checkpoint_path: str, output_json: str) -> str:
    if output_json:
        return output_json
    return os.path.join(infer_exp_log_dir_from_checkpoint(checkpoint_path), "source_snr_sweep.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-faithful SwinJSCC SNR sweep")
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--config-path", default="", type=str)
    parser.add_argument("--output-json", default="", type=str)
    parser.add_argument(
        "--recipe-stage",
        default=None,
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
    parser.add_argument("--data-dir", default=None, type=str)
    parser.add_argument("--hr-val-dirs", default="", type=str)
    parser.add_argument("--hr-eval-split", default=None, choices=["kodak", "clic2021", "combined"])
    parser.add_argument("--img-size", default=None, type=int)
    parser.add_argument("--hr-eval-crop-multiple", default=None, type=int)
    parser.add_argument("--model-size", default=None, choices=["small", "base", "large"])
    parser.add_argument("--C", default="", type=str)
    parser.add_argument("--rate", default=None, type=int)
    parser.add_argument("--rates", default="", type=str)
    parser.add_argument("--multiple-snr", default="", type=str)
    parser.add_argument("--channel-type", default=None, choices=["awgn", "rayleigh"])
    parser.add_argument("--distortion-metric", default=None, choices=["MSE", "MS-SSIM"])
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument(
        "--rcsb-rate-gain-mode",
        default=None,
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
        default=None,
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
    parser.add_argument("--attnres-content-query-scale", default=None, type=float)
    parser.add_argument(
        "--attnres-condition-gate-mode",
        default=None,
        choices=["snr_rate", "rate", "snr"],
    )
    parser.add_argument("--no-attnres-condition-gate-reference", action="store_true")
    parser.add_argument("--no-attnres-layer-embedding", action="store_true")
    parser.add_argument("--attnres-cross-stage-skip-threshold", default=None, type=float)
    parser.add_argument("--attnres-cross-stage-force-skip", action="store_true")
    parser.add_argument("--snr-start", default=1.0, type=float)
    parser.add_argument("--snr-end", default=13.0, type=float)
    parser.add_argument("--snr-step", default=3.0, type=float)
    parser.add_argument("--snr-db-values", default="", type=str)
    parser.add_argument("--noise-realizations", default=1, type=int)
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-val-batches", default=0, type=int)
    parser.add_argument(
        "--include-inference-aux",
        action="store_true",
        help="Request diagnostic aux outputs during eval. Disabled by default for lower inference latency.",
    )
    return parser


def load_json_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick(cli_value, cfg: dict, key: str, default):
    if cli_value not in (None, ""):
        return cli_value
    if cfg.get(key) not in (None, ""):
        return cfg[key]
    source_args = cfg.get("source_args", {})
    if source_args.get(key) not in (None, ""):
        return source_args[key]
    return default


def _pick_attnres_condition_gate_use_reference(cli_args: argparse.Namespace, cfg: dict) -> bool:
    if cli_args.no_attnres_condition_gate_reference:
        return False
    if "attnres_condition_gate_use_reference" in cfg:
        return bool(cfg["attnres_condition_gate_use_reference"])
    if "no_attnres_condition_gate_reference" in cfg:
        return not bool(cfg["no_attnres_condition_gate_reference"])
    return True


def build_source_eval_runtime_args(cli_args: argparse.Namespace, cfg: dict) -> SimpleNamespace:
    recipe_stage = _pick(cli_args.recipe_stage, cfg, "recipe_stage", "universal")
    c_value = _pick(cli_args.C, cfg, "C", source_rate_list_for_recipe_stage(recipe_stage))
    snr_values = (
        parse_float_list(cli_args.snr_db_values)
        if cli_args.snr_db_values
        else build_snr_list(float(cli_args.snr_start), float(cli_args.snr_end), float(cli_args.snr_step))
    )
    if not snr_values:
        raise ValueError("snr-db-values must contain at least one value")
    default_rate = int(cli_args.rate) if cli_args.rate is not None else int(parse_int_list(str(c_value))[0])
    rates = parse_int_list(cli_args.rates) if cli_args.rates else [default_rate]
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("rates must contain positive integers")
    if cli_args.noise_realizations <= 0:
        raise ValueError("noise-realizations must be positive")
    data_dir = _pick(cli_args.data_dir, cfg, "data_dir", DEFAULT_HR_DATA_ROOT)
    if not cli_args.data_dir:
        data_dir = str(resource_path(data_dir))
    if cli_args.hr_val_dirs:
        val_dirs = cli_args.hr_val_dirs
    elif cli_args.data_dir or cli_args.hr_eval_split:
        # Explicit data/split selection must not inherit an old Kodak directory.
        val_dirs = ""
    else:
        val_dirs = ",".join(str(resource_path(path)) for path in split_path_list(
            _pick("", cfg, "hr_val_dirs", "")))
    return SimpleNamespace(
        recipe_stage=recipe_stage,
        source_model=_pick(cli_args.source_model, cfg, "model", source_model_name_for_recipe_stage(recipe_stage)),
        data_dir=data_dir,
        hr_val_dirs=val_dirs,
        hr_eval_split=_pick(cli_args.hr_eval_split, cfg, "hr_eval_split", "kodak"),
        img_size=int(_pick(cli_args.img_size, cfg, "img_size", 256)),
        hr_eval_crop_multiple=int(_pick(cli_args.hr_eval_crop_multiple, cfg, "hr_eval_crop_multiple", 128)),
        model_size=_pick(cli_args.model_size, cfg, "model_size", "base"),
        C=str(c_value),
        rate=rates[0],
        rates=rates,
        snr_values=snr_values,
        multiple_snr=",".join(str(int(v)) if float(v).is_integer() else str(v) for v in snr_values),
        channel_type=_pick(cli_args.channel_type, cfg, "channel_type", "awgn"),
        distortion_metric=_pick(cli_args.distortion_metric, cfg, "distortion_metric", "MSE"),
        batch_size=int(_pick(cli_args.batch_size, cfg, "batch_size", 1)),
        rcsb_rate_gain_mode=_pick(cli_args.rcsb_rate_gain_mode, cfg, "rcsb_rate_gain_mode", "sigmoid"),
        attnres_query_mode=_pick(cli_args.attnres_query_mode, cfg, "attnres_query_mode", "static"),
        attnres_content_query_scale=float(
            _pick(cli_args.attnres_content_query_scale, cfg, "attnres_content_query_scale", 1.0)
        ),
        attnres_condition_gate_mode=_pick(cli_args.attnres_condition_gate_mode, cfg, "attnres_condition_gate_mode", "snr_rate"),
        attnres_condition_gate_use_reference=_pick_attnres_condition_gate_use_reference(cli_args, cfg),
        attnres_use_layer_embedding=not bool(cli_args.no_attnres_layer_embedding)
        if cli_args.no_attnres_layer_embedding
        else bool(cfg.get("attnres_use_layer_embedding", True)),
        attnres_cross_stage_skip_threshold=float(
            _pick(cli_args.attnres_cross_stage_skip_threshold, cfg, "attnres_cross_stage_skip_threshold", 0.0)
        ),
        attnres_cross_stage_force_skip=bool(
            cli_args.attnres_cross_stage_force_skip or cfg.get("attnres_cross_stage_force_skip", False)
        ),
        include_inference_aux=bool(cli_args.include_inference_aux),
        noise_realizations=int(cli_args.noise_realizations),
        strict_checkpoint=bool(cli_args.strict_checkpoint),
        seed=int(cli_args.seed),
        max_val_batches=int(cli_args.max_val_batches),
    )


def _psnr_from_mse(mse: torch.Tensor) -> float:
    mse_value = float(mse.detach().mean().clamp_min(1e-10).item())
    return float(10.0 * np.log10(255.0 * 255.0 / mse_value))


def seed_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_image_psnr_db(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = ((reference * 255.0 - reconstructed * 255.0) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10((255.0**2) / mse)


def build_eval_loader(args: SimpleNamespace) -> DataLoader:
    if not args.hr_val_dirs:
        args.hr_val_dirs = _default_source_val_dirs(args.data_dir, args.hr_eval_split)
    dataset = HRCroppedEvalImageDataset(
        split_path_list(args.hr_val_dirs),
        crop_multiple=args.hr_eval_crop_multiple,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


def build_source_model(args: SimpleNamespace) -> SourceSwinJSCCWrapper:
    source_args = build_source_swinjscc_args(
        recipe_stage=args.recipe_stage,
        trainset="DIV2K",
        testset="CLIC21" if args.hr_eval_split == "clic2021" else args.hr_eval_split,
        distortion_metric=args.distortion_metric,
        model=args.source_model,
        channel_type=args.channel_type,
        C=args.C,
        multiple_snr=args.multiple_snr or SOURCE_HR_SNR_LIST,
        model_size=args.model_size,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    if args.recipe_stage == "fixed_attnres":
        return SourceFixedAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_attnres":
        return SourceFixedAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_attnres_decoder_adapter":
        return SourceNoModNetSNRAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_attnres":
        return SourceAllStageAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_attnres_decoder_adapter":
        return SourceAllStageAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stageall_cross_attnres_decoder_adapter":
        return SourceAllStageAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
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
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_snr_stage1_final_attnres_decoder_adapter":
        return SourceStage1FinalAttnResDecoderAdapterWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_universal":
        return SourceNoModNetUniversalWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage == "no_modnet_sall_cross_universal":
        return DecompJSCC(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
            attnres_cross_stage_skip_threshold=args.attnres_cross_stage_skip_threshold,
            attnres_cross_stage_force_skip=args.attnres_cross_stage_force_skip,
        )
    if args.recipe_stage == "snr_adaptive_attnres":
        return SourceSNRAdaptiveAttnResWrapper(
            source_args,
            pass_channel=True,
            attnres_query_mode=args.attnres_query_mode,
            attnres_content_query_scale=args.attnres_content_query_scale,
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
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
            attnres_use_layer_embedding=args.attnres_use_layer_embedding,
            attnres_condition_gate_mode=args.attnres_condition_gate_mode,
            attnres_condition_gate_use_reference=args.attnres_condition_gate_use_reference,
        )
    if args.recipe_stage in {"rcsb_universal_snr_warmup", "rcsb_universal_joint"}:
        return SourceRCSBUniversalWrapper(
            source_args,
            pass_channel=True,
            gain_mode=args.rcsb_rate_gain_mode,
        )
    return SourceSwinJSCCWrapper(source_args, pass_channel=True)


def _forward_source_model(model, images: torch.Tensor, snr_db: float, rate: int, return_aux: bool):
    if "return_aux" in inspect.signature(model.forward).parameters:
        return model(images, snr_db=snr_db, rate=rate, return_aux=return_aux)
    return model(images, snr_db=snr_db, rate=rate)


def evaluate_at_snr(
    model,
    loader,
    snr_db: float,
    rate: int,
    device: torch.device,
    max_val_batches: int,
    return_aux: bool = False,
) -> dict:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_cbr = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = _forward_source_model(
                model,
                images,
                snr_db=float(snr_db),
                rate=int(rate),
                return_aux=return_aux,
            )
            reconstructed = outputs["reconstructed"].clamp(0.0, 1.0)
            total_psnr += _psnr_from_mse(outputs["mse"])
            total_ssim += float(compute_ssim_batch(images.clamp(0, 1), reconstructed, device).mean().item())
            total_cbr += float(outputs["cbr"])
            num_batches += 1
            if max_val_batches > 0 and batch_idx + 1 >= max_val_batches:
                break
    return {
        "snr_db": float(snr_db),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "cbr": total_cbr / max(num_batches, 1),
        "rate": int(rate),
    }


def evaluate_condition(
    model,
    loader,
    *,
    snr_db: float,
    rate: int,
    device: torch.device,
    max_val_batches: int,
    return_aux: bool,
    realization: int,
    evaluation_seed: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_cbr = 0.0
    image_rows = []
    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(loader):
            images = images.to(device)
            outputs = _forward_source_model(
                model,
                images,
                snr_db=float(snr_db),
                rate=int(rate),
                return_aux=return_aux,
            )
            reconstructed = outputs["reconstructed"].clamp(0.0, 1.0)
            reference = images.clamp(0.0, 1.0)
            psnr_values = per_image_psnr_db(reference, reconstructed)
            ssim_values = compute_ssim_batch(reference, reconstructed, device)
            cbr = float(outputs["cbr"])
            for image_idx, image_id in enumerate(image_ids):
                row = {
                    "image_id": str(image_id),
                    "rate": int(rate),
                    "snr_db": float(snr_db),
                    "realization": int(realization),
                    "evaluation_seed": int(evaluation_seed),
                    "psnr_db": float(psnr_values[image_idx].item()),
                    "ssim": float(ssim_values[image_idx].item()),
                    "cbr": cbr,
                }
                image_rows.append(row)
                total_psnr += row["psnr_db"]
                total_ssim += row["ssim"]
                total_cbr += row["cbr"]
            if max_val_batches > 0 and batch_idx + 1 >= max_val_batches:
                break
    num_images = len(image_rows)
    if num_images == 0:
        raise RuntimeError("Evaluation loader produced no images")
    psnr = total_psnr / num_images
    return (
        {
            "rate": int(rate),
            "snr_db": float(snr_db),
            "realization": int(realization),
            "evaluation_seed": int(evaluation_seed),
            "psnr": psnr,
            "psnr_db": psnr,
            "ssim": total_ssim / num_images,
            "cbr": total_cbr / num_images,
            "num_images": num_images,
        },
        image_rows,
    )


def aggregate_condition_results(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["rate"]), float(row["snr_db"])), []).append(row)
    results = []
    for (rate, snr_db), values in sorted(grouped.items()):
        psnr = float(np.mean([row["psnr_db"] for row in values]))
        results.append(
            {
                "rate": rate,
                "snr_db": snr_db,
                "psnr": psnr,
                "psnr_db": psnr,
                "ssim": float(np.mean([row["ssim"] for row in values])),
                "cbr": float(np.mean([row["cbr"] for row in values])),
                "noise_realizations": len(values),
                "num_images": int(values[0]["num_images"]),
            }
        )
    return results


def run(cli_args) -> None:
    config_path = resolve_config_path(cli_args.checkpoint_path, cli_args.config_path)
    cfg = load_json_config(config_path)
    args = build_source_eval_runtime_args(cli_args, cfg)
    seed_randomness(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(cli_args.gpu_id)
    device = torch.device(f"cuda:{cli_args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = build_source_model(args).to(device)
    ckpt_info = load_source_checkpoint(model, cli_args.checkpoint_path, strict=args.strict_checkpoint)
    loader = build_eval_loader(args)
    realization_results = []
    per_image_results = []
    for rate in args.rates:
        for snr_db in args.snr_values:
            for realization in range(args.noise_realizations):
                evaluation_seed = args.seed + realization
                seed_randomness(evaluation_seed)
                result, image_rows = evaluate_condition(
                    model,
                    loader,
                    snr_db=snr_db,
                    rate=rate,
                    device=device,
                    max_val_batches=args.max_val_batches,
                    return_aux=args.include_inference_aux,
                    realization=realization,
                    evaluation_seed=evaluation_seed,
                )
                realization_results.append(result)
                per_image_results.extend(image_rows)
    results = aggregate_condition_results(realization_results)
    avg_psnr = sum(item["psnr"] for item in results) / max(len(results), 1)
    avg_ssim = sum(item["ssim"] for item in results) / max(len(results), 1)
    payload = {
        "checkpoint_path": cli_args.checkpoint_path,
        "config_path": config_path,
        "checkpoint_load": ckpt_info,
        "channel_type": args.channel_type,
        "channel_implementation": channel_implementation_metadata(model),
        "source_model": args.source_model,
        "C": args.C,
        "rate": args.rates[0] if len(args.rates) == 1 else None,
        "rates": args.rates,
        "multiple_snr": args.multiple_snr,
        "rcsb_rate_gain_mode": args.rcsb_rate_gain_mode,
        "attnres_query_mode": args.attnres_query_mode,
        "attnres_content_query_scale": args.attnres_content_query_scale,
        "attnres_condition_gate_mode": args.attnres_condition_gate_mode,
        "attnres_condition_gate_use_reference": args.attnres_condition_gate_use_reference,
        "attnres_use_layer_embedding": args.attnres_use_layer_embedding,
        "include_inference_aux": args.include_inference_aux,
        "noise_realizations": args.noise_realizations,
        "common_random_numbers": True,
        "strict_checkpoint": args.strict_checkpoint,
        "seed": args.seed,
        "hr_eval_split": args.hr_eval_split,
        "results": results,
        "realization_results": realization_results,
        "per_image_results": per_image_results,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "best_psnr": max(results, key=lambda item: item["psnr"]),
    }
    output_path = resolve_output_json_path(cli_args.checkpoint_path, cli_args.output_json)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(
        json.dumps(
            {
                "checkpoint_path": cli_args.checkpoint_path,
                "output_json": output_path,
                "checkpoint_load": ckpt_info,
                "conditions": len(results),
                "realization_results": len(realization_results),
                "per_image_results": len(per_image_results),
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
            },
            indent=2,
        )
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
