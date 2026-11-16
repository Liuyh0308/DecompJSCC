"""Depth routing, cross-stage interventions and rate-controller analysis."""

from __future__ import annotations

import contextlib
import math
import random
import re
from collections import defaultdict
import numpy as np
import torch
from decompjscc_utils import compute_psnr_from_pixel_mse, compute_ssim_batch
from models.stage_block_attnres import StageBlockAttnRes
import argparse
import csv
import json
from pathlib import Path
from decompjscc_utils import MAIN_CHECKPOINT, MAIN_EXPERIMENT_DIR, REPO_ROOT, load_decompjscc_checkpoint, paper_metrics
from experiments.evaluation import build_eval_loader, build_loader_args, parse_rates, parse_snr_values, resolve_device
from models.decompjscc import DecompJSCC, DecompJSCCConfig
import hashlib
import itertools
import statistics
import sys
from decompjscc_utils import load_decompjscc_checkpoint, parse_int_list
from experiments.evaluation import resolve_device
from swinjscc_hr_dataset import HRCroppedEvalImageDataset
from torch.utils.data import DataLoader
from decompjscc_utils import REPO_ROOT
from experiments.evaluation import build_existing_model, forward_model, require_idle_gpu, write_json
from experiments.evaluation import seed_randomness


def seed_analysis_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def summarize_weight_tensor(weights: torch.Tensor, layer_name: str) -> dict:
    weights = weights.detach().float().cpu()
    if weights.ndim != 3:
        raise ValueError(f"Expected [source,batch,token] weights, got {tuple(weights.shape)}")
    source_count = int(weights.shape[0])
    source_weights = weights.mean(dim=(1, 2))
    source_weights = source_weights / source_weights.sum().clamp_min(1e-12)
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0).mean()
    positions = torch.linspace(0.0, 1.0, source_count) if source_count > 1 else torch.zeros(1)
    normalized_entropy = entropy / math.log(float(source_count)) if source_count > 1 else torch.tensor(0.0)
    peak_source = int(source_weights.argmax())
    return {
        "layer_name": layer_name,
        "source_count": source_count,
        "source_weights": [float(value) for value in source_weights],
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "depth_center": float((source_weights * positions).sum()),
        "peak_source": peak_source,
        "peak_weight": float(source_weights[peak_source]),
    }

def summarize_attention_aux(aux: dict[str, object]) -> list[dict]:
    summaries: list[dict] = []
    stage_prefixes = sorted(
        {
            match.group(1)
            for key in aux
            for match in [re.match(r"^(stage\d+)_attnres_weights$", key)]
            if match
        },
        key=lambda value: int(value[len("stage") :]),
    )
    for stage_prefix in stage_prefixes:
        weights_per_layer = aux.get(f"{stage_prefix}_attnres_weights", [])
        for layer_index, weights in enumerate(weights_per_layer):
            summaries.append(summarize_weight_tensor(weights, f"{stage_prefix}_layer_{layer_index}"))
        output_weights = aux.get(f"{stage_prefix}_attnres_output_weights")
        if isinstance(output_weights, torch.Tensor):
            summaries.append(summarize_weight_tensor(output_weights, f"{stage_prefix}_output"))
    return summaries

@contextlib.contextmanager
def snr_conditioning_intervention(model: torch.nn.Module, enabled: bool):
    if not enabled:
        yield
        return
    originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, StageBlockAttnRes) and module.use_snr_depth_logit_gate:
                originals.append((module.snr_depth_logit_scale, module.snr_depth_logit_scale.detach().clone()))
                module.snr_depth_logit_scale.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)

def _merge_layer_items(items: list[dict]) -> dict:
    source_count = int(items[0]["source_count"])
    source_weights = np.mean([item["source_weights"] for item in items], axis=0)
    peak_counts = np.bincount([int(item["peak_source"]) for item in items], minlength=source_count)
    return {
        "layer_name": items[0]["layer_name"],
        "source_count": source_count,
        "source_weights": [float(value) for value in source_weights],
        "entropy": float(np.mean([item["entropy"] for item in items])),
        "normalized_entropy": float(np.mean([item["normalized_entropy"] for item in items])),
        "depth_center": float(np.mean([item["depth_center"] for item in items])),
        "peak_source": int(peak_counts.argmax()),
        "peak_weight": float(np.mean([item["peak_weight"] for item in items])),
    }

def _layer_sort_key(name: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"stage(\d+)_(layer_(\d+)|output)", name)
    if match is None:
        return (999, 999, 999)
    return (int(match.group(1)), 1 if match.group(2) == "output" else 0, int(match.group(3) or 0))

def _collect_aux_scalars(outputs: dict) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for aux_name, prefix in (("attnres_aux", "attnres"), ("rcsb_aux", "rcsb"), ("decoder_snr_aux", "decoder_snr")):
        aux = outputs.get(aux_name)
        if not isinstance(aux, dict):
            continue
        for name, value in aux.items():
            key = name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                scalars[key] = float(value.detach())
            elif isinstance(value, (int, float)):
                scalars[key] = float(value)
    return scalars

def evaluate_condition(
    model,
    loader,
    *,
    snr_db: float,
    rate: int,
    device: torch.device,
    max_val_batches: int,
    intervention: str,
    evaluation_seed: int,
) -> dict:
    if intervention not in {"none", "snr_conditioning_off"}:
        raise ValueError(f"Unsupported intervention={intervention}")
    seed_analysis_randomness(evaluation_seed)
    totals = {"psnr": 0.0, "ssim": 0.0, "cbr": 0.0}
    layer_collection: dict[str, list[dict]] = defaultdict(list)
    scalar_collection: dict[str, list[float]] = defaultdict(list)
    batches = 0
    model.eval()
    with torch.no_grad(), snr_conditioning_intervention(model, intervention == "snr_conditioning_off"):
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images, snr_db=snr_db, rate=rate, return_aux=True)
            reconstructed = outputs["reconstructed"].clamp(0.0, 1.0)
            totals["psnr"] += compute_psnr_from_pixel_mse(outputs["mse"])
            totals["ssim"] += float(compute_ssim_batch(images.clamp(0, 1), reconstructed, device).mean())
            totals["cbr"] += float(outputs["cbr"])
            for summary in summarize_attention_aux(outputs.get("attnres_aux", {})):
                layer_collection[summary["layer_name"]].append(summary)
            for key, value in _collect_aux_scalars(outputs).items():
                scalar_collection[key].append(value)
            batches += 1
            if max_val_batches > 0 and batch_index + 1 >= max_val_batches:
                break
    return {
        "snr_db": float(snr_db),
        "rate": int(rate),
        "intervention": intervention,
        **{key: value / max(batches, 1) for key, value in totals.items()},
        "num_batches": batches,
        "evaluation_seed": evaluation_seed,
        "routing_layers": [
            _merge_layer_items(layer_collection[name]) for name in sorted(layer_collection, key=_layer_sort_key)
        ],
        "aux_scalars": {
            key: float(sum(values) / len(values)) for key, values in sorted(scalar_collection.items())
        },
    }


INTERVENTIONS = {"none", "snr_conditioning_off"}

def parse_interventions(value: str) -> list[str]:
    interventions = [item.strip() for item in value.split(",") if item.strip()]
    if not interventions:
        return ["none"]
    unsupported = [item for item in interventions if item not in INTERVENTIONS]
    if unsupported:
        raise ValueError(f"Unsupported interventions: {unsupported}")
    if "none" not in interventions:
        interventions.insert(0, "none")
    return interventions

def routing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze DecompJSCC multi-depth latent routing")
    parser.add_argument("--checkpoint", default=str(MAIN_CHECKPOINT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets/swinjscc_hr"))
    parser.add_argument("--val-dirs", default="")
    parser.add_argument("--eval-split", default="kodak", choices=("kodak", "clic2021", "combined"))
    parser.add_argument("--rates", default="96")
    parser.add_argument("--snr-db-values", default="1,4,7,10,13")
    parser.add_argument("--interventions", default="none,snr_conditioning_off")
    parser.add_argument("--channel-type", default="awgn", choices=("awgn", "rayleigh"))
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--eval-crop-multiple", default=128, type=int)
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--distortion-metric", default="MSE", choices=("MSE", "MS-SSIM"))
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-val-batches", default=0, type=int)
    return parser

def _public_result(result: dict, intervention: str) -> dict:
    cleaned = dict(result)
    cleaned.pop("ablation_mode", None)
    scalars = cleaned.get("aux_scalars")
    if isinstance(scalars, dict):
        cleaned["aux_scalars"] = paper_metrics(scalars)
    cleaned["intervention"] = intervention
    return cleaned

def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return MAIN_EXPERIMENT_DIR / args.channel_type / "routing_interventions" / args.eval_split

def _write_layer_csv(path: Path, conditions: list[dict]) -> None:
    fields = [
        "rate",
        "snr_db",
        "intervention",
        "layer_block",
        "source_index",
        "source_count",
        "mean_weight",
        "normalized_entropy",
        "depth_center",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            for intervention, result in condition["intervention_results"].items():
                for layer in result.get("routing_layers", []):
                    for source_index, weight in enumerate(layer["source_weights"]):
                        writer.writerow(
                            {
                                "rate": condition["rate"],
                                "snr_db": condition["snr_db"],
                                "intervention": intervention,
                                "layer_block": layer["layer_name"],
                                "source_index": source_index,
                                "source_count": layer["source_count"],
                                "mean_weight": weight,
                                "normalized_entropy": layer["normalized_entropy"],
                                "depth_center": layer["depth_center"],
                            }
                        )

def routing_main() -> None:
    args = routing_parser().parse_args()
    rates = parse_rates(args.rates)
    snr_values = parse_snr_values(args.snr_db_values)
    interventions = parse_interventions(args.interventions)
    config = DecompJSCCConfig(
        channel_type=args.channel_type,
        rate_channels=tuple(rates),
        snr_db_values=tuple(snr_values),
        image_size=args.image_size,
        model_size=args.model_size,
        batch_size=args.batch_size,
        distortion_metric=args.distortion_metric,
        testset=args.eval_split,
    )
    device = resolve_device(args.device)
    model = DecompJSCC(config).to(device).eval()
    checkpoint_load = load_decompjscc_checkpoint(model, args.checkpoint, strict=True)
    loader = build_eval_loader(build_loader_args(args))

    conditions = []
    condition_index = 0
    for rate in rates:
        for snr_db in snr_values:
            evaluation_seed = args.seed + condition_index
            results = {}
            for intervention in interventions:
                result = evaluate_condition(
                    model,
                    loader,
                    snr_db=snr_db,
                    rate=rate,
                    device=device,
                    max_val_batches=args.max_val_batches,
                    intervention=intervention,
                    evaluation_seed=evaluation_seed,
                )
                results[intervention] = _public_result(result, intervention)
            conditions.append(
                {
                    "rate": rate,
                    "snr_db": snr_db,
                    "evaluation_seed": evaluation_seed,
                    "intervention_results": results,
                }
            )
            condition_index += 1

    payload = {
        **config.to_public_metadata(),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_load": checkpoint_load,
        "eval_split": args.eval_split,
        "interventions": interventions,
        "seed": args.seed,
        "common_random_numbers": True,
        "conditions": conditions,
    }
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "decompjscc_routing.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_layer_csv(output_dir / "decompjscc_layer_block_routing.csv", conditions)
    print(json.dumps({"method": "DecompJSCC", "output_dir": str(output_dir), "conditions": len(conditions)}, indent=2))


ROLE_SPECS = {
    "decompjscc": {
        "gain_mode": "active_rms",
        "checkpoint": "checkpoints/decompjscc/awgn/model_best.pth",
    },
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def support_metadata_bits(total_channels: int, rate: int) -> float:
    if not 0 <= rate <= total_channels:
        raise ValueError("rate must be within the channel dimension")
    return float(
        (
            math.lgamma(total_channels + 1)
            - math.lgamma(rate + 1)
            - math.lgamma(total_channels - rate + 1)
        )
        / math.log(2.0)
    )

def support_comparison(smaller: np.ndarray, larger: np.ndarray) -> dict:
    smaller = np.asarray(smaller, dtype=bool)
    larger = np.asarray(larger, dtype=bool)
    if smaller.shape != larger.shape:
        raise ValueError("support masks must have the same shape")
    intersection = int(np.logical_and(smaller, larger).sum())
    union = int(np.logical_or(smaller, larger).sum())
    missing = int(np.logical_and(smaller, np.logical_not(larger)).sum())
    return {
        "intersection_channels": intersection,
        "union_channels": union,
        "missing_from_larger": missing,
        "is_nested": missing == 0,
        "smaller_retention": float(intersection / max(1, int(smaller.sum()))),
        "jaccard": float(intersection / max(1, union)),
    }

def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size != x.size or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson_correlation(_average_ranks(x), _average_ranks(y))

def mask_diversity(masks: list[np.ndarray]) -> dict:
    if not masks:
        raise ValueError("mask list must not be empty")
    matrix = np.stack([np.asarray(mask, dtype=bool) for mask in masks])
    probabilities = matrix.mean(axis=0)
    nontrivial = np.logical_and(probabilities > 0.0, probabilities < 1.0)
    entropy = np.zeros_like(probabilities, dtype=np.float64)
    entropy[nontrivial] = -(
        probabilities[nontrivial] * np.log2(probabilities[nontrivial])
        + (1.0 - probabilities[nontrivial]) * np.log2(1.0 - probabilities[nontrivial])
    )
    jaccards = []
    for first, second in itertools.combinations(matrix, 2):
        comparison = support_comparison(first, second)
        jaccards.append(comparison["jaccard"])
    return {
        "images": int(matrix.shape[0]),
        "unique_masks": len({mask.tobytes() for mask in matrix}),
        "varying_channels": int(nontrivial.sum()),
        "mean_channel_entropy_bits": float(entropy.mean()),
        "pairwise_jaccard_mean": float(np.mean(jaccards)) if jaccards else 1.0,
        "pairwise_jaccard_median": float(np.median(jaccards)) if jaccards else 1.0,
        "pairwise_jaccard_min": float(np.min(jaccards)) if jaccards else 1.0,
        "pairwise_jaccard_max": float(np.max(jaccards)) if jaccards else 1.0,
    }

def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else None

def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _set_resolution(model: DecompJSCC, image: torch.Tensor) -> None:
    _, _, height, width = image.shape
    if height == model.net.H and width == model.net.W:
        return
    model.net.encoder.update_resolution(height, width)
    model.net.decoder.update_resolution(
        height // (2 ** model.net.downsample),
        width // (2 ** model.net.downsample),
    )
    model.net.H = height
    model.net.W = width

def _build_model(
    checkpoint: Path,
    gain_mode: str,
    rates: list[int],
    snr_db: float,
    device: torch.device,
) -> tuple[DecompJSCC, dict]:
    config = DecompJSCCConfig(
        rate_channels=tuple(rates),
        snr_db_values=(float(snr_db),),
        image_size=256,
        model_size="base",
        batch_size=1,
    )
    model = DecompJSCC(config, gain_mode=gain_mode)
    checkpoint_load = load_decompjscc_checkpoint(model, checkpoint, strict=True)
    return model.to(device).eval(), checkpoint_load

def _audit_role(
    role: str,
    model: DecompJSCC,
    dataset: HRCroppedEvalImageDataset,
    rates: list[int],
    snr_db: float,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict[tuple[str, int], np.ndarray]]:
    rate_rows: list[dict] = []
    nested_rows: list[dict] = []
    supports: dict[tuple[str, int], np.ndarray] = {}
    reference_rate = rates[0]

    with torch.inference_mode():
        for image_index in range(len(dataset)):
            image, image_id = dataset[image_index]
            image = image.unsqueeze(0).to(device)
            _set_resolution(model, image)

            features = {}
            for rate in rates:
                feature, _ = model._encode_semantic_with_aux(
                    image,
                    snr_db=snr_db,
                    rate=rate,
                    return_aux=False,
                )
                features[rate] = feature
            reference_feature = features[reference_rate]

            for rate_index, rate in enumerate(rates):
                feature = features[rate]
                latent_max_abs = float((feature - reference_feature).abs().max().item())
                output = model.rcsb_rate_adapter(feature, rate, snr_db=snr_db)
                token_mask_consistent = bool(
                    torch.equal(output.mask, output.mask[:, :1, :].expand_as(output.mask))
                )
                channel_mask = output.mask[0, 0].detach().to(dtype=torch.bool).cpu().numpy()
                supports[(str(image_id), rate)] = channel_mask
                selected = int(channel_mask.sum())
                active_count = output.mask.sum().clamp_min(1.0)
                selected_gain_rms = torch.sqrt(
                    (output.value_gain.pow(2) * output.mask).sum() / active_count
                )
                pre_channel_active_power = (output.z_tx.pow(2).sum() / active_count).clamp_min(1e-12)
                normalized_tx = output.z_tx / torch.sqrt(pre_channel_active_power * 2.0)
                normalized_active_pair_power = (
                    normalized_tx.pow(2).sum() / active_count * 2.0
                )

                torch.manual_seed(seed + image_index * 100 + rate_index)
                noisy = model.net.feature_pass_channel(
                    output.z_tx,
                    snr_db,
                    pre_channel_active_power,
                ) * output.mask
                noise = (noisy - output.z_tx) * output.mask
                received_snr_db = 10.0 * torch.log10(
                    output.z_tx.pow(2).sum().clamp_min(1e-12)
                    / noise.pow(2).sum().clamp_min(1e-12)
                )

                scores = output.scores[0].detach().cpu().numpy()
                value_gain = output.value_gain[0].mean(dim=0).detach().cpu().numpy()
                selected_scores = scores[channel_mask]
                selected_value_gain = value_gain[channel_mask]
                rate_rows.append(
                    {
                        "role": role,
                        "image_id": str(image_id),
                        "height": int(image.shape[-2]),
                        "width": int(image.shape[-1]),
                        "rate": rate,
                        "reference_rate": reference_rate,
                        "latent_max_abs_vs_reference": latent_max_abs,
                        "latent_rate_invariant_1e-7": latent_max_abs <= 1e-7,
                        "selected_channels": selected,
                        "latent_channels": int(output.scores.shape[-1]),
                        "cardinality_exact": selected == rate,
                        "token_mask_consistent": token_mask_consistent,
                        "selected_indices": " ".join(str(index) for index in np.flatnonzero(channel_mask)),
                        "score_value_pearson": pearson_correlation(scores, value_gain),
                        "score_value_spearman": spearman_correlation(scores, value_gain),
                        "selected_score_value_pearson": pearson_correlation(
                            selected_scores, selected_value_gain
                        ),
                        "selected_score_value_spearman": spearman_correlation(
                            selected_scores, selected_value_gain
                        ),
                        "selected_gain_rms": float(selected_gain_rms.item()),
                        "pre_channel_active_power": float(pre_channel_active_power.item()),
                        "normalized_active_pair_power": float(normalized_active_pair_power.item()),
                        "actual_received_snr_db": float(received_snr_db.item()),
                        "payload_cbr": float(rate / (2 * 3 * 2 ** (model.net.downsample * 2))),
                        "support_metadata_lower_bound_bits": support_metadata_bits(
                            output.scores.shape[-1], rate
                        ),
                    }
                )

            for smaller_rate, larger_rate in itertools.combinations(rates, 2):
                comparison = support_comparison(
                    supports[(str(image_id), smaller_rate)],
                    supports[(str(image_id), larger_rate)],
                )
                nested_rows.append(
                    {
                        "role": role,
                        "image_id": str(image_id),
                        "smaller_rate": smaller_rate,
                        "larger_rate": larger_rate,
                        "adjacent_rates": rates.index(larger_rate) == rates.index(smaller_rate) + 1,
                        **comparison,
                    }
                )
    return rate_rows, nested_rows, supports

def _summarize(
    rate_rows: list[dict],
    nested_rows: list[dict],
    diversity_rows: list[dict],
) -> dict:
    role_rate: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rate_rows:
        role_rate[(row["role"], row["rate"])].append(row)
    metric_summary = []
    for (role, rate), rows in sorted(role_rate.items()):
        metric_summary.append(
            {
                "role": role,
                "rate": rate,
                "images": len(rows),
                "latent_max_abs": max(row["latent_max_abs_vs_reference"] for row in rows),
                "latent_invariant_count": sum(row["latent_rate_invariant_1e-7"] for row in rows),
                "cardinality_exact_count": sum(row["cardinality_exact"] for row in rows),
                "token_mask_consistent_count": sum(row["token_mask_consistent"] for row in rows),
                "score_value_pearson_mean": _mean(
                    [row["score_value_pearson"] for row in rows]
                ),
                "score_value_spearman_mean": _mean(
                    [row["score_value_spearman"] for row in rows]
                ),
                "selected_gain_rms_mean": _mean([row["selected_gain_rms"] for row in rows]),
                "selected_gain_rms_max_abs_from_one": max(
                    abs(row["selected_gain_rms"] - 1.0) for row in rows
                ),
                "pre_channel_active_power_mean": _mean(
                    [row["pre_channel_active_power"] for row in rows]
                ),
                "normalized_active_pair_power_mean": _mean(
                    [row["normalized_active_pair_power"] for row in rows]
                ),
                "actual_received_snr_db_mean": _mean(
                    [row["actual_received_snr_db"] for row in rows]
                ),
                "actual_received_snr_db_median": _median(
                    [row["actual_received_snr_db"] for row in rows]
                ),
                "support_metadata_lower_bound_bits": rows[0][
                    "support_metadata_lower_bound_bits"
                ],
            }
        )

    nested_summary = []
    for role in ROLE_SPECS:
        role_rows = [row for row in nested_rows if row["role"] == role]
        for scope, rows in (
            ("all_pairs", role_rows),
            ("adjacent_pairs", [row for row in role_rows if row["adjacent_rates"]]),
        ):
            nested_summary.append(
                {
                    "role": role,
                    "scope": scope,
                    "pairs": len(rows),
                    "nested_pairs": sum(row["is_nested"] for row in rows),
                    "violation_pairs": sum(not row["is_nested"] for row in rows),
                    "missing_channels_total": sum(row["missing_from_larger"] for row in rows),
                    "smaller_retention_mean": _mean(
                        [row["smaller_retention"] for row in rows]
                    ),
                }
            )

    return {
        "role_rate_metrics": metric_summary,
        "nestedness": nested_summary,
        "mask_diversity": diversity_rows,
    }

def _write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    latent_channels = payload["protocol"]["latent_channels"]
    lines = [
        "# DecompJSCC RCSB 机制审计",
        "",
        "口径：Kodak 全图中心裁剪到 128 的倍数，主方法 strict-load 自身 checkpoint；SNR、rate 与随机种子见本次 protocol。",
        "",
        "## 逐 rate 性质",
        "",
        "| 角色 | rate | latent max abs | exact top-k | score/value Pearson | gain RMS 偏差 | 实测接收 SNR (dB) | support bits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["role_rate_metrics"]:
        lines.append(
            "| {role} | {rate} | {latent_max_abs:.3g} | {cardinality_exact_count}/{images} | "
            "{pearson} | {gain_error:.3g} | {snr:.3f} | {bits:.2f} |".format(
                role=row["role"],
                rate=row["rate"],
                latent_max_abs=row["latent_max_abs"],
                cardinality_exact_count=row["cardinality_exact_count"],
                images=row["images"],
                pearson=(
                    f"{row['score_value_pearson_mean']:.4f}"
                    if row["score_value_pearson_mean"] is not None
                    else "-"
                ),
                gain_error=row["selected_gain_rms_max_abs_from_one"],
                snr=row["actual_received_snr_db_mean"],
                bits=row["support_metadata_lower_bound_bits"],
            )
        )
    lines.extend(
        [
            "",
            "## Nestedness",
            "",
            "| 角色 | 范围 | nested / total | violation | mean retention |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["nestedness"]:
        lines.append(
            "| {role} | {scope} | {nested_pairs}/{pairs} | {violation_pairs} | {retention:.4f} |".format(
                **row,
                retention=row["smaller_retention_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Mask diversity",
            "",
            "| 角色 | rate | unique / images | varying channels | mean pairwise Jaccard |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["mask_diversity"]:
        lines.append(
            "| {role} | {rate} | {unique_masks}/{images} | {varying_channels} | {pairwise_jaccard_mean:.4f} |".format(
                **row
            )
        )
    total_role_rate = sum(row["images"] for row in summary["role_rate_metrics"])
    invariant = sum(row["latent_invariant_count"] for row in summary["role_rate_metrics"])
    cardinality = sum(row["cardinality_exact_count"] for row in summary["role_rate_metrics"])
    main_adjacent = next(
        row
        for row in summary["nestedness"]
        if row["role"] == "decompjscc" and row["scope"] == "adjacent_pairs"
    )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- latent formation 的 rate 不变检查为 {invariant}/{total_role_rate}，精确 top-k cardinality 为 {cardinality}/{total_role_rate}。",
            f"- 主方法相邻 rate 的 nestedness 违例为 {main_adjacent['violation_pairs']}/{main_adjacent['pairs']}；因此不能声称 guaranteed progressive nested support。",
            "",
            f"解释边界：最终 latent 维度 D={latent_channels}，`support_metadata_lower_bound_bits=log2 C(D,k)` 是若要显式传输任意内容相关 support 所需的组合索引下界；它不是当前模型实际码流中已经计费的比特。Nestedness 若存在违例，只能说明各 rate 独立 top-k，不得声称 progressive embedded bitstream。",
            "",
            "逐图片 support、功率、相关性、nestedness 和 mask diversity 分别见同目录 CSV；输入文件与 checkpoint SHA256 见 `run_manifest.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def rcsb_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen DecompJSCC RCSB properties")
    parser.add_argument(
        "--main-checkpoint",
        default=str(Path(__file__).resolve().parents[1] / ROLE_SPECS["decompjscc"]["checkpoint"]),
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parents[1] / "datasets/swinjscc_hr/Kodak"),
    )
    parser.add_argument("--rates", default="32,64,96,128,192")
    parser.add_argument("--snr-db", default=7.0, type=float)
    parser.add_argument("--seed", default=20260826, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "exp_log/decompjscc/analysis/awgn/rcsb_audit"
        ),
    )
    return parser

def rcsb_main() -> None:
    args = rcsb_parser().parse_args()
    rates = parse_int_list(args.rates)
    if rates != sorted(set(rates)) or rates[-1] != 192:
        raise ValueError("rates must be unique, ascending, and end at latent dimension 192")
    device = resolve_device(args.device)
    dataset = HRCroppedEvalImageDataset([args.dataset_dir], crop_multiple=128)
    if args.max_images > 0:
        dataset.image_files = dataset.image_files[: args.max_images]

    checkpoints = {
        "decompjscc": Path(args.main_checkpoint),
    }
    checkpoint_records = {}
    all_rate_rows = []
    all_nested_rows = []
    role_supports = {}
    for role, spec in ROLE_SPECS.items():
        model, checkpoint_load = _build_model(
            checkpoints[role],
            spec["gain_mode"],
            rates,
            args.snr_db,
            device,
        )
        rate_rows, nested_rows, supports = _audit_role(
            role,
            model,
            dataset,
            rates,
            args.snr_db,
            args.seed,
            device,
        )
        all_rate_rows.extend(rate_rows)
        all_nested_rows.extend(nested_rows)
        role_supports[role] = supports
        checkpoint_records[role] = {
            "path": str(checkpoints[role]),
            "sha256": sha256_file(checkpoints[role]),
            "load": checkpoint_load,
            "gain_definition": "coupled active-RMS",
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diversity_rows = []
    for role in ROLE_SPECS:
        for rate in rates:
            masks = [
                role_supports[role][(Path(path).name, rate)] for path in dataset.image_files
            ]
            diversity_rows.append({"role": role, "rate": rate, **mask_diversity(masks)})

    latent_channels = {row["latent_channels"] for row in all_rate_rows}
    if len(latent_channels) != 1:
        raise ValueError(f"roles disagree on latent channel dimension: {sorted(latent_channels)}")
    latent_channels_value = latent_channels.pop()

    payload = {
        "material_passport": {
            "origin": "experiment-agent",
            "mode": "run",
            "verification_status": "MEASURED",
            "version": "decompjscc_rcsb_audit_main_v2",
        },
        "protocol": {
            "dataset": "Kodak",
            "images": len(dataset),
            "crop_multiple": 128,
            "snr_db": args.snr_db,
            "rates": rates,
            "latent_channels": latent_channels_value,
            "seed": args.seed,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": "float32",
            "latent_invariance_tolerance": 1e-7,
        },
        "definitions": {
            "actual_received_snr_db": "10*log10(sum(z_tx^2)/sum((channel(z_tx)-z_tx)^2)) on active entries",
            "normalized_active_pair_power": "2*mean(channel_tx^2) over active real-valued entries",
            "support_metadata_lower_bound_bits": "log2 binomial(D, k), where D is the measured latent channel dimension; not charged by current payload CBR",
            "nestedness": "support(k_small) is a subset of support(k_large) for the same image",
        },
        "checkpoints": checkpoint_records,
        "summary": _summarize(
            all_rate_rows,
            all_nested_rows,
            diversity_rows,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_image_rate.csv", all_rate_rows)
    _write_csv(output_dir / "nestedness.csv", all_nested_rows)
    _write_csv(output_dir / "mask_diversity.csv", diversity_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "command": [sys.executable, *sys.argv],
        "protocol": payload["protocol"],
        "checkpoints": checkpoint_records,
        "images": [
            {
                "path": str(path),
                "sha256": sha256_file(Path(path)),
            }
            for path in dataset.image_files
        ],
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in (
                "per_image_rate.csv",
                "nestedness.csv",
                "mask_diversity.csv",
                "summary.json",
                "summary.md",
            )
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "images": len(dataset)}, indent=2))


def crossstage_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=96)
    parser.add_argument("--snr-db-values", default="1,13")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "datasets/swinjscc_hr/Kodak")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Use a new output path: {args.output_json}")
    snr_values = parse_snr_values(args.snr_db_values)
    device = torch.device(args.device)
    require_idle_gpu(device)
    model, record = build_existing_model("decompjscc", args.rate)
    model.to(device)
    dataset = HRCroppedEvalImageDataset([str(args.data_dir)], crop_multiple=128)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for snr in snr_values:
            seed_randomness(args.seed)
            for images, ids in loader:
                require_idle_gpu(device)
                aux = forward_model("decompjscc", model, images.to(device), snr, args.rate, return_aux=True)["attnres_aux"]
                for stage in (2, 3):
                    layers = list(aux[f"stage{stage}_attnres_cross_stage_weights"])
                    layers.append(aux[f"stage{stage}_attnres_cross_stage_output_weights"])
                    gates = model.encoder_stage_attnres[stage].cross_stage_gates.tanh().cpu().tolist()
                    if len(layers) != len(gates):
                        raise ValueError("Cross-stage mixing point/gate mismatch")
                    for point, weights in enumerate(layers):
                        means = weights.mean(dim=(1, 2)).cpu().tolist()
                        if not np.isclose(sum(means), 1, atol=1e-6):
                            raise ValueError("Cross-stage weights do not sum to one")
                        for source, value in enumerate(means):
                            rows.append({
                                "image_id": str(ids[0]), "snr_db": snr, "target_stage": stage + 1,
                                "mixing_point": point + 1, "source_stage": source + 1,
                                "source_weight": value, "injection_gate": gates[point],
                            })
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["snr_db"], row["target_stage"], row["mixing_point"], row["source_stage"])].append(row)
    summary = []
    for (snr, stage, point, source), values in sorted(grouped.items()):
        summary.append({
            "snr_db": snr, "target_stage": stage, "mixing_point": point, "source_stage": source,
            "num_images": len(values), "mean_source_weight": float(np.mean([v["source_weight"] for v in values])),
            "injection_gate": values[0]["injection_gate"],
        })
    path = args.output_json
    write_json(path, {
        **record, "dataset": str(args.data_dir), "num_images": len(dataset), "rate": args.rate,
        "snr_db_values": snr_values, "seed": args.seed, "summary": summary, "per_image_results": rows,
        "interpretation": "Descriptive cross-stage routing. The injection gates are fixed learned parameters, not SNR-dependent gates. Stage 3 has only one external source and thus weight 1.",
    })
    print(path)
