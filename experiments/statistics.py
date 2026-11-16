"""Aggregate existing measurements and analyze interpolation and channel transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import math


@dataclass(frozen=True)
class EvaluationInput:
    method: str
    split: str
    path: Path

def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected-rates must not be empty")
    return values

def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected-snr-values must not be empty")
    return values

def parse_input_spec(value: str) -> EvaluationInput:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("input must use METHOD:SPLIT:PATH")
    return EvaluationInput(method=parts[0], split=parts[1], path=Path(parts[2]))

def parse_expected_images(values: Iterable[str]) -> dict[str, int]:
    parsed = {}
    for value in values:
        split, separator, count = value.partition("=")
        if not separator or not split or int(count) <= 0:
            raise ValueError("expected-images must use SPLIT=POSITIVE_COUNT")
        parsed[split] = int(count)
    return parsed

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _psnr(row: dict) -> float:
    if "psnr_db" in row:
        return float(row["psnr_db"])
    return float(row["psnr"])

def _normalize_row(row: dict, source: EvaluationInput, row_type: str) -> dict:
    normalized = {
        "method": source.method,
        "split": source.split,
        "rate": int(row["rate"]),
        "snr_db": float(row["snr_db"]),
        "psnr_db": _psnr(row),
        "ssim": float(row["ssim"]),
        "cbr": float(row["cbr"]),
    }
    if row_type in {"realization", "image"}:
        normalized["realization"] = int(row["realization"])
        normalized["evaluation_seed"] = int(row["evaluation_seed"])
    if row_type == "image":
        normalized["image_id"] = str(row["image_id"])
    return normalized

def _assert_exact_keys(actual: set, expected: set, label: str) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} mismatch: missing={missing[:8]} extra={extra[:8]}")

def validate_payload(
    payload: dict,
    source: EvaluationInput,
    expected_rates: tuple[int, ...],
    expected_snrs: tuple[float, ...],
    expected_realizations: int,
    expected_images: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not payload.get("common_random_numbers", False):
        raise ValueError(f"{source.path}: common_random_numbers is not true")
    declared_split = payload.get("eval_split", payload.get("hr_eval_split"))
    if declared_split != source.split:
        raise ValueError(f"{source.path}: declared split {declared_split!r} != {source.split!r}")
    if int(payload.get("noise_realizations", expected_realizations)) != expected_realizations:
        raise ValueError(f"{source.path}: unexpected noise realization count")
    checkpoint_load = payload.get("checkpoint_load", {})
    skipped = checkpoint_load.get("skipped_keys", checkpoint_load.get("skipped", []))
    if skipped:
        raise ValueError(f"{source.path}: checkpoint load skipped keys: {skipped}")

    cells = [_normalize_row(row, source, "cell") for row in payload["results"]]
    realization_rows = [
        _normalize_row(row, source, "realization") for row in payload["realization_results"]
    ]
    image_rows = [_normalize_row(row, source, "image") for row in payload["per_image_results"]]

    expected_cells = {(rate, snr) for rate in expected_rates for snr in expected_snrs}
    _assert_exact_keys(
        {(row["rate"], row["snr_db"]) for row in cells},
        expected_cells,
        f"{source.method}/{source.split} cells",
    )
    expected_realization_keys = {
        (rate, snr, realization)
        for rate in expected_rates
        for snr in expected_snrs
        for realization in range(expected_realizations)
    }
    _assert_exact_keys(
        {(row["rate"], row["snr_db"], row["realization"]) for row in realization_rows},
        expected_realization_keys,
        f"{source.method}/{source.split} realization rows",
    )

    image_ids = sorted({row["image_id"] for row in image_rows})
    if len(image_ids) != expected_images:
        raise ValueError(
            f"{source.method}/{source.split}: expected {expected_images} images, got {len(image_ids)}"
        )
    expected_image_keys = {
        (image_id, rate, snr, realization)
        for image_id in image_ids
        for rate in expected_rates
        for snr in expected_snrs
        for realization in range(expected_realizations)
    }
    _assert_exact_keys(
        {
            (row["image_id"], row["rate"], row["snr_db"], row["realization"])
            for row in image_rows
        },
        expected_image_keys,
        f"{source.method}/{source.split} per-image rows",
    )
    return cells, realization_rows, image_rows

def _mean_rows(rows: list[dict], metric: str) -> float:
    return float(np.mean([row[metric] for row in rows]))

def summarize_cells(cells: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cells:
        grouped[(row["method"], row["split"])].append(row)
    summaries = []
    for (method, split), rows in sorted(grouped.items()):
        by_rate = {}
        for rate in sorted({row["rate"] for row in rows}):
            rate_rows = [row for row in rows if row["rate"] == rate]
            by_rate[str(rate)] = {
                "psnr_db": _mean_rows(rate_rows, "psnr_db"),
                "ssim": _mean_rows(rate_rows, "ssim"),
                "cbr": _mean_rows(rate_rows, "cbr"),
            }
        by_snr = {}
        for snr in sorted({row["snr_db"] for row in rows}):
            snr_rows = [row for row in rows if row["snr_db"] == snr]
            by_snr[str(snr)] = {
                "psnr_db": _mean_rows(snr_rows, "psnr_db"),
                "ssim": _mean_rows(snr_rows, "ssim"),
                "cbr": _mean_rows(snr_rows, "cbr"),
            }
        worst = min(rows, key=lambda row: row["psnr_db"])
        summaries.append(
            {
                "method": method,
                "split": split,
                "num_cells": len(rows),
                "macro_psnr_db": _mean_rows(rows, "psnr_db"),
                "macro_ssim": _mean_rows(rows, "ssim"),
                "macro_cbr": _mean_rows(rows, "cbr"),
                "by_rate": by_rate,
                "by_snr": by_snr,
                "worst_cell": {
                    "rate": worst["rate"],
                    "snr_db": worst["snr_db"],
                    "psnr_db": worst["psnr_db"],
                    "ssim": worst["ssim"],
                    "cbr": worst["cbr"],
                },
            }
        )
    return summaries

def _bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> dict:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "num_images": int(values.size),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }

def paired_comparison(
    image_rows: list[dict],
    primary_method: str,
    baseline_method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    by_split_method: dict[tuple[str, str], dict[tuple, dict]] = defaultdict(dict)
    for row in image_rows:
        key = (row["image_id"], row["rate"], row["snr_db"], row["realization"])
        by_split_method[(row["split"], row["method"])][key] = row

    output = {
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "difference": f"{primary_method} - {baseline_method}",
        "splits": [],
    }
    splits = sorted({row["split"] for row in image_rows})
    for split_index, split in enumerate(splits):
        primary = by_split_method.get((split, primary_method))
        baseline = by_split_method.get((split, baseline_method))
        if not primary or not baseline:
            continue
        _assert_exact_keys(set(primary), set(baseline), f"paired rows for {split}")

        per_image_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
        per_cell_metric: dict[tuple[int, float, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key in sorted(primary):
            image_id, rate, snr, _ = key
            for metric in ("psnr_db", "ssim"):
                difference = primary[key][metric] - baseline[key][metric]
                per_image_metric[(image_id, metric)].append(difference)
                per_cell_metric[(rate, snr, metric)][image_id].append(difference)

        macro = {}
        for metric_index, metric in enumerate(("psnr_db", "ssim")):
            values = np.asarray(
                [
                    np.mean(per_image_metric[(image_id, metric)])
                    for image_id in sorted({key[0] for key in per_image_metric if key[1] == metric})
                ],
                dtype=np.float64,
            )
            macro[metric] = _bootstrap_mean_ci(
                values,
                bootstrap_samples,
                bootstrap_seed + split_index * 1000 + metric_index,
            )

        cells = []
        for cell_index, (rate, snr, metric) in enumerate(sorted(per_cell_metric)):
            by_image = per_cell_metric[(rate, snr, metric)]
            values = np.asarray(
                [np.mean(by_image[image_id]) for image_id in sorted(by_image)], dtype=np.float64
            )
            cells.append(
                {
                    "rate": rate,
                    "snr_db": snr,
                    "metric": metric,
                    **_bootstrap_mean_ci(
                        values,
                        bootstrap_samples,
                        bootstrap_seed + 10000 + split_index * 1000 + cell_index,
                    ),
                }
            )
        output["splits"].append(
            {
                "split": split,
                "macro": macro,
                "cells": cells,
                "extend_primary_comparison_to_10_realizations": abs(macro["psnr_db"]["mean"])
                < 0.10,
            }
        )
    return output

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _checkpoint_path(payload: dict) -> Path | None:
    value = payload.get("checkpoint", payload.get("checkpoint_path", ""))
    return Path(value) if value else None

def summarize_evaluations(
    inputs: list[EvaluationInput],
    output_dir: Path,
    expected_rates: tuple[int, ...],
    expected_snrs: tuple[float, ...],
    expected_realizations: int,
    expected_images: dict[str, int],
    primary_method: str,
    baseline_method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_channel_type: str | None = None,
) -> dict:
    if len({(item.method, item.split) for item in inputs}) != len(inputs):
        raise ValueError("Each method/split pair must appear exactly once")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cells = []
    all_realizations = []
    all_images = []
    manifest_inputs = []
    checkpoint_hash_cache = {}
    seed_contract_by_split: dict[str, dict[int, int]] = {}
    image_ids_by_split: dict[str, set[str]] = {}
    shared_channel_implementation = None

    for source in inputs:
        if source.split not in expected_images:
            raise ValueError(f"No expected image count configured for split {source.split}")
        payload = json.loads(source.path.read_text(encoding="utf-8"))
        if expected_channel_type is not None:
            if payload.get("channel_type") != expected_channel_type:
                raise ValueError(
                    f"{source.path}: channel_type={payload.get('channel_type')!r}, "
                    f"expected {expected_channel_type!r}"
                )
            channel_implementation = payload.get("channel_implementation")
            if not channel_implementation or not channel_implementation.get("source_sha256"):
                raise ValueError(f"{source.path}: missing channel implementation fingerprint")
            if shared_channel_implementation is None:
                shared_channel_implementation = channel_implementation
            elif shared_channel_implementation != channel_implementation:
                raise ValueError(
                    f"{source.path}: channel implementation differs across methods"
                )
        cells, realizations, images = validate_payload(
            payload,
            source,
            expected_rates,
            expected_snrs,
            expected_realizations,
            expected_images[source.split],
        )
        realization_seed_map = {row["realization"]: row["evaluation_seed"] for row in realizations}
        existing_seeds = seed_contract_by_split.setdefault(source.split, realization_seed_map)
        if existing_seeds != realization_seed_map:
            raise ValueError(f"CRN seed mismatch for split {source.split}: {source.method}")
        current_image_ids = {row["image_id"] for row in images}
        existing_images = image_ids_by_split.setdefault(source.split, current_image_ids)
        if existing_images != current_image_ids:
            raise ValueError(f"Image ID mismatch for split {source.split}: {source.method}")

        checkpoint = _checkpoint_path(payload)
        checkpoint_sha256 = None
        if checkpoint is not None and checkpoint.exists():
            resolved = checkpoint.resolve()
            checkpoint_sha256 = checkpoint_hash_cache.setdefault(resolved, sha256_file(resolved))
        manifest_inputs.append(
            {
                "method": source.method,
                "split": source.split,
                "path": str(source.path),
                "input_sha256": sha256_file(source.path),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_sha256": checkpoint_sha256,
                "image_ids": sorted(current_image_ids),
                "evaluation_seeds": realization_seed_map,
                "channel_implementation": payload.get("channel_implementation"),
            }
        )
        all_cells.extend(cells)
        all_realizations.extend(realizations)
        all_images.extend(images)

    summaries = summarize_cells(all_cells)
    paired = paired_comparison(
        all_images,
        primary_method,
        baseline_method,
        bootstrap_samples,
        bootstrap_seed,
    )
    summary_payload = {
        "material_passport": {
            "origin": "experiment-agent",
            "mode": "validate",
            "verification_status": "ANALYZED",
            "version": "decompjscc_formal_eval_summary_v1",
        },
        "expected_rates": list(expected_rates),
        "expected_snr_db_values": list(expected_snrs),
        "expected_realizations": expected_realizations,
        "channel_type": expected_channel_type,
        "channel_implementation": shared_channel_implementation,
        "summaries": summaries,
        "paired_comparison": paired,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    _write_csv(
        output_dir / "cell_summary.csv",
        all_cells,
        ["method", "split", "rate", "snr_db", "psnr_db", "ssim", "cbr"],
    )
    _write_csv(
        output_dir / "realization_results.csv",
        all_realizations,
        [
            "method",
            "split",
            "rate",
            "snr_db",
            "realization",
            "evaluation_seed",
            "psnr_db",
            "ssim",
            "cbr",
        ],
    )
    _write_csv(
        output_dir / "per_image_results.csv",
        all_images,
        [
            "method",
            "split",
            "image_id",
            "rate",
            "snr_db",
            "realization",
            "evaluation_seed",
            "psnr_db",
            "ssim",
            "cbr",
        ],
    )
    paired_cells = []
    for split in paired["splits"]:
        for row in split["cells"]:
            paired_cells.append({"split": split["split"], **row})
    _write_csv(
        output_dir / "paired_cells.csv",
        paired_cells,
        [
            "split",
            "rate",
            "snr_db",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "num_images",
            "bootstrap_samples",
            "bootstrap_seed",
        ],
    )

    per_rate_dir = output_dir / "per_rate"
    per_rate_dir.mkdir(exist_ok=True)
    for (method, split, rate), rows in sorted(
        _group_rows(all_cells, lambda row: (row["method"], row["split"], row["rate"])).items()
    ):
        path = per_rate_dir / f"{method}_{split}_rate{rate}.json"
        path.write_text(json.dumps({"method": method, "split": split, "rate": rate, "results": rows}, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "inputs": manifest_inputs,
        "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary_payload

def _group_rows(rows: list[dict], key_fn) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped

def summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize formal DecompJSCC evaluations")
    parser.add_argument("--input", action="append", required=True, type=parse_input_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-rates", required=True, type=parse_csv_ints)
    parser.add_argument("--expected-snr-values", required=True, type=parse_csv_floats)
    parser.add_argument("--expected-realizations", default=3, type=int)
    parser.add_argument("--expected-images", action="append", required=True)
    parser.add_argument("--primary-method", default="decompjscc")
    parser.add_argument("--baseline-method", default="swinjscc_universal")
    parser.add_argument("--bootstrap-samples", default=10000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260811, type=int)
    parser.add_argument("--expected-channel-type", choices=("awgn", "rayleigh"), default=None)
    return parser

def summary_main() -> None:
    args = summary_parser().parse_args()
    if args.expected_realizations <= 0:
        raise ValueError("expected-realizations must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    summary = summarize_evaluations(
        inputs=args.input,
        output_dir=args.output_dir,
        expected_rates=args.expected_rates,
        expected_snrs=args.expected_snr_values,
        expected_realizations=args.expected_realizations,
        expected_images=parse_expected_images(args.expected_images),
        primary_method=args.primary_method,
        baseline_method=args.baseline_method,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_channel_type=args.expected_channel_type,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summaries": len(summary["summaries"]),
                "paired_splits": len(summary["paired_comparison"]["splits"]),
            },
            indent=2,
        )
    )



def parse_interpolation_input(value: str) -> tuple[str, str, Path, Path, Path]:
    parts = value.split(':', 4)
    if len(parts) != 5 or not parts[0] or (not parts[1]):
        raise ValueError('input must be METHOD:SPLIT:SEEN:SNR_SLICE:RATE_SLICE')
    return (parts[0], parts[1], Path(parts[2]), Path(parts[3]), Path(parts[4]))

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_cells(path: Path) -> dict[tuple[int, float], dict[str, float]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    cells = {}
    for row in payload['results']:
        key = (int(row['rate']), float(row['snr_db']))
        if key in cells:
            raise ValueError(f'duplicate cell {key} in {path}')
        cells[key] = {'psnr_db': float(row.get('psnr_db', row.get('psnr'))), 'ssim': float(row['ssim'])}
    if not cells:
        raise ValueError(f'no result cells in {path}')
    return cells

def bracket(values: list[float], point: float) -> tuple[float, float]:
    lower = [value for value in values if value < point]
    upper = [value for value in values if value > point]
    if not lower or not upper:
        raise ValueError(f'point {point} is not bracketed by seen values {values}')
    return (max(lower), min(upper))

def audit_axis(method: str, split: str, axis: str, seen: dict[tuple[int, float], dict[str, float]], unseen: dict[tuple[int, float], dict[str, float]]) -> list[dict]:
    rows = []
    for (rate, snr_db), metrics in sorted(unseen.items()):
        if axis == 'snr':
            seen_values = sorted((cell_snr for cell_rate, cell_snr in seen if cell_rate == rate))
            lower, upper = bracket(seen_values, snr_db)
            lower_key, upper_key = ((rate, lower), (rate, upper))
            point = snr_db
        else:
            seen_values = sorted((float(cell_rate) for cell_rate, cell_snr in seen if cell_snr == snr_db))
            lower, upper = bracket(seen_values, float(rate))
            lower_key, upper_key = ((int(lower), snr_db), (int(upper), snr_db))
            point = float(rate)
        fraction = (point - lower) / (upper - lower)
        for metric in ('psnr_db', 'ssim'):
            lower_value = seen[lower_key][metric]
            unseen_value = metrics[metric]
            upper_value = seen[upper_key][metric]
            linear_value = lower_value + fraction * (upper_value - lower_value)
            rows.append({'method': method, 'split': split, 'axis': axis, 'rate': rate, 'snr_db': snr_db, 'metric': metric, 'lower_condition': lower, 'upper_condition': upper, 'lower_value': lower_value, 'unseen_value': unseen_value, 'upper_value': upper_value, 'within_adjacent_endpoints': lower_value <= unseen_value <= upper_value, 'linear_interpolation_value': linear_value, 'linear_residual': unseen_value - linear_value})
    return rows

def analyze_interpolation(inputs: list[str], output_dir: Path) -> dict:
    rows = []
    input_manifest = []
    for value in inputs:
        method, split, seen_path, snr_path, rate_path = parse_interpolation_input(value)
        seen = load_cells(seen_path)
        rows.extend(audit_axis(method, split, 'snr', seen, load_cells(snr_path)))
        rows.extend(audit_axis(method, split, 'rate', seen, load_cells(rate_path)))
        input_manifest.append({'method': method, 'split': split, 'seen_grid': str(seen_path), 'seen_grid_sha256': file_sha256(seen_path), 'snr_slice': str(snr_path), 'snr_slice_sha256': file_sha256(snr_path), 'rate_slice': str(rate_path), 'rate_slice_sha256': file_sha256(rate_path)})
    violations = [row for row in rows if not row['within_adjacent_endpoints']]
    payload = {'material_passport': {'origin': 'experiment-agent', 'mode': 'validate', 'verification_status': 'ANALYZED', 'version': 'decompjscc_interpolation_audit_v1'}, 'criterion': 'unseen aggregate metric lies within its two adjacent seen-condition endpoint values', 'inputs': input_manifest, 'checks': len(rows), 'violations': len(violations), 'all_checks_pass': not violations, 'rows': rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'monotonicity.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / 'monotonicity.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return payload

def interpolation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Audit DecompJSCC unseen interpolation points')
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser

def interpolation_main() -> None:
    args = interpolation_parser().parse_args()
    result = analyze_interpolation(args.input, args.output_dir)
    print(json.dumps({key: result[key] for key in ('checks', 'violations', 'all_checks_pass')}, indent=2))

KEY_FIELDS = ('method', 'split', 'image_id', 'rate', 'snr_db', 'realization', 'evaluation_seed')
METRICS = ('psnr_db', 'ssim')

def channel_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _row_key(row: dict[str, str]) -> tuple:
    return (row['method'], row['split'], row['image_id'], int(row['rate']), float(row['snr_db']), int(row['realization']), int(row['evaluation_seed']))

def load_rows(path: Path, methods: set[str]) -> dict[tuple, dict[str, float]]:
    rows = {}
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(KEY_FIELDS + METRICS + ('cbr',)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path} is missing fields: {sorted(missing)}')
        for raw in reader:
            if raw['method'] not in methods:
                continue
            key = _row_key(raw)
            if key in rows:
                raise ValueError(f'duplicate row {key} in {path}')
            rows[key] = {'psnr_db': float(raw['psnr_db']), 'ssim': float(raw['ssim']), 'cbr': float(raw['cbr'])}
    if not rows:
        raise ValueError(f'no requested methods found in {path}')
    return rows

def bootstrap_mean(values: list[float], samples: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError('cannot bootstrap an empty sample')
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return {'mean': float(array.mean()), 'ci95_low': float(np.quantile(means, 0.025)), 'ci95_high': float(np.quantile(means, 0.975)), 'num_images': int(array.size), 'bootstrap_samples': samples, 'bootstrap_seed': seed}

def _summarize_by_image(rows: list[dict], group_fields: tuple[str, ...], metric: str, bootstrap_samples: int, seed_start: int) -> list[dict]:
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = tuple((row[field] for field in group_fields))
        grouped[group][row['image_id']].append(float(row[f'delta_{metric}']))
    summaries = []
    for offset, (group, image_values) in enumerate(sorted(grouped.items())):
        values = [float(np.mean(entries)) for entries in image_values.values()]
        summary = bootstrap_mean(values, bootstrap_samples, seed_start + offset)
        summaries.append({**dict(zip(group_fields, group)), 'metric': metric, **summary})
    return summaries

def _difference_in_differences(paired_rows: list[dict], primary_method: str, baseline_method: str, bootstrap_samples: int, seed_start: int) -> list[dict]:
    by_method_key = {(row['method'], row['split'], row['image_id'], row['rate'], row['snr_db'], row['realization'], row['evaluation_seed']): row for row in paired_rows}
    image_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in paired_rows:
        if row['method'] != primary_method:
            continue
        common = (row['split'], row['image_id'], row['rate'], row['snr_db'], row['realization'], row['evaluation_seed'])
        baseline = by_method_key.get((baseline_method, *common))
        if baseline is None:
            raise ValueError(f'missing baseline pair for {common}')
        for metric in METRICS:
            value = row[f'delta_{metric}'] - baseline[f'delta_{metric}']
            image_values[row['split'], metric][row['image_id']].append(value)
    summaries = []
    for offset, ((split, metric), per_image) in enumerate(sorted(image_values.items())):
        values = [float(np.mean(entries)) for entries in per_image.values()]
        summaries.append({'split': split, 'metric': metric, **bootstrap_mean(values, bootstrap_samples, seed_start + offset)})
    return summaries

def analyze_channel_transfer(awgn_csv: Path, rayleigh_csv: Path, output_dir: Path, methods: tuple[str, ...], primary_method: str, baseline_method: str, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    method_set = set(methods)
    if primary_method not in method_set or baseline_method not in method_set:
        raise ValueError('primary and baseline methods must be included in methods')
    awgn = load_rows(awgn_csv, method_set)
    rayleigh = load_rows(rayleigh_csv, method_set)
    if awgn.keys() != rayleigh.keys():
        missing_rayleigh = sorted(awgn.keys() - rayleigh.keys())[:3]
        missing_awgn = sorted(rayleigh.keys() - awgn.keys())[:3]
        raise ValueError(f'AWGN and Rayleigh row keys differ: missing Rayleigh={missing_rayleigh}, missing AWGN={missing_awgn}')
    paired_rows = []
    for key in sorted(awgn):
        method, split, image_id, rate, snr_db, realization, evaluation_seed = key
        awgn_row = awgn[key]
        rayleigh_row = rayleigh[key]
        if not np.isclose(awgn_row['cbr'], rayleigh_row['cbr'], rtol=0.0, atol=1e-12):
            raise ValueError(f'CBR differs between channels for {key}')
        paired_rows.append({'method': method, 'split': split, 'image_id': image_id, 'rate': rate, 'snr_db': snr_db, 'realization': realization, 'evaluation_seed': evaluation_seed, 'cbr': awgn_row['cbr'], 'awgn_psnr_db': awgn_row['psnr_db'], 'rayleigh_psnr_db': rayleigh_row['psnr_db'], 'delta_psnr_db': rayleigh_row['psnr_db'] - awgn_row['psnr_db'], 'awgn_ssim': awgn_row['ssim'], 'rayleigh_ssim': rayleigh_row['ssim'], 'delta_ssim': rayleigh_row['ssim'] - awgn_row['ssim']})
    macro = []
    cells = []
    for metric_offset, metric in enumerate(METRICS):
        macro.extend(_summarize_by_image(paired_rows, ('method', 'split'), metric, bootstrap_samples, bootstrap_seed + metric_offset * 1000))
        cells.extend(_summarize_by_image(paired_rows, ('method', 'split', 'rate', 'snr_db'), metric, bootstrap_samples, bootstrap_seed + 10000 + metric_offset * 1000))
    sensitivity = _difference_in_differences(paired_rows, primary_method, baseline_method, bootstrap_samples, bootstrap_seed + 20000)
    payload = {'material_passport': {'origin': 'experiment-agent', 'mode': 'validate', 'verification_status': 'ANALYZED', 'version': 'decompjscc_channel_transfer_v1'}, 'difference': 'Rayleigh - AWGN; negative values indicate degradation', 'method_sensitivity_difference': f'({primary_method} Rayleigh-AWGN) - ({baseline_method} Rayleigh-AWGN)；正值表示主方法在信道迁移中损失更少', 'inputs': {'awgn_csv': str(awgn_csv), 'awgn_csv_sha256': channel_file_sha256(awgn_csv), 'rayleigh_csv': str(rayleigh_csv), 'rayleigh_csv_sha256': channel_file_sha256(rayleigh_csv)}, 'methods': list(methods), 'paired_rows': len(paired_rows), 'macro': macro, 'cells': cells, 'method_sensitivity': sensitivity}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'channel_transfer.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'channel_transfer.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    _write_summary(output_dir / 'summary.md', payload)
    return payload

def _write_summary(path: Path, payload: dict) -> None:
    lines = ['# DecompJSCC AWGN 到 Rayleigh 零样本迁移', '', '指标差值定义为 `Rayleigh - AWGN`；负值表示迁移到 Rayleigh 后性能下降。', '置信区间以图像为 bootstrap 单位，每张图先对全部网格条件和信道实现取平均。', '', '## 宏平均退化', '', '| 方法 | 数据集 | 指标 | 均值 | 95% CI | 图像数 |', '|---|---|---:|---:|---:|---:|']
    for row in payload['macro']:
        lines.append(f"| {row['method']} | {row['split']} | {row['metric']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['num_images']} |")
    lines.extend(['', '## 方法间信道敏感性', '', f"差值定义：`{payload['method_sensitivity_difference']}`。", '', '| 数据集 | 指标 | 均值 | 95% CI | 图像数 |', '|---|---:|---:|---:|---:|'])
    for row in payload['method_sensitivity']:
        lines.append(f"| {row['split']} | {row['metric']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['num_images']} |")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple((item.strip() for item in value.split(',') if item.strip()))
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError('methods must contain at least two unique names')
    return methods

def channel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Analyze paired DecompJSCC AWGN-to-Rayleigh transfer')
    parser.add_argument('--awgn-csv', required=True, type=Path)
    parser.add_argument('--rayleigh-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--methods', default='decompjscc,swinjscc_universal', type=parse_methods)
    parser.add_argument('--primary-method', default='decompjscc')
    parser.add_argument('--baseline-method', default='swinjscc_universal')
    parser.add_argument('--bootstrap-samples', default=10000, type=int)
    parser.add_argument('--bootstrap-seed', default=20260811, type=int)
    return parser

def channel_main() -> None:
    args = channel_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError('bootstrap-samples must be positive')
    result = analyze_channel_transfer(args.awgn_csv, args.rayleigh_csv, args.output_dir, args.methods, args.primary_method, args.baseline_method, args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps({'output_dir': str(args.output_dir), 'paired_rows': result['paired_rows'], 'macro_summaries': len(result['macro']), 'cell_summaries': len(result['cells'])}, indent=2))

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'exp_log/decompjscc/analysis/rayleigh/adaptation_comparison'
RAYLEIGH_KEY_FIELDS = ('method', 'split', 'image_id', 'rate', 'snr_db', 'realization', 'evaluation_seed')
RAYLEIGH_METRICS = ('psnr_db', 'ssim')
SETTINGS = ('zero_shot', 'matched_training')

def rayleigh_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def rayleigh_parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple((int(item.strip()) for item in value.split(',') if item.strip()))
    if not values:
        raise ValueError('expected at least one integer')
    return values

def rayleigh_parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple((float(item.strip()) for item in value.split(',') if item.strip()))
    if not values:
        raise ValueError('expected at least one float')
    return values

def rayleigh_parse_expected_images(values: list[str]) -> dict[str, int]:
    output = {}
    for value in values:
        split, separator, count = value.partition('=')
        if not separator or not split or int(count) <= 0:
            raise ValueError('expected-images must use SPLIT=POSITIVE_COUNT')
        output[split] = int(count)
    return output

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = set(RAYLEIGH_KEY_FIELDS + RAYLEIGH_METRICS + ('cbr',))
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path} is missing fields: {sorted(missing)}')
        return list(reader)

def _rayleigh_row_key(row: dict) -> tuple:
    return (str(row['method']), str(row['split']), str(row['image_id']), int(row['rate']), float(row['snr_db']), int(row['realization']), int(row['evaluation_seed']))

def _normalize_rows(path: Path, methods: set[str]) -> dict[tuple, dict[str, float]]:
    output = {}
    for raw in _read_csv(path):
        if raw['method'] not in methods:
            continue
        key = _rayleigh_row_key(raw)
        if key in output:
            raise ValueError(f'duplicate row key {key} in {path}')
        values = {metric: float(raw[metric]) for metric in RAYLEIGH_METRICS}
        values['cbr'] = float(raw['cbr'])
        if not all((math.isfinite(value) for value in values.values())):
            raise ValueError(f'non-finite metric for {key} in {path}')
        output[key] = values
    if not output:
        raise ValueError(f'no requested method rows found in {path}')
    return output

def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path

def _load_setting(directory: Path, methods: set[str]) -> tuple[dict, dict, dict]:
    csv_path = directory / 'per_image_results.csv'
    summary_path = directory / 'summary.json'
    manifest_path = directory / 'run_manifest.json'
    for path in (csv_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if summary.get('channel_type') != 'rayleigh':
        raise ValueError(f'{summary_path}: expected Rayleigh channel')
    channel = summary.get('channel_implementation')
    if not channel or not channel.get('source_sha256'):
        raise ValueError(f'{summary_path}: missing channel implementation fingerprint')
    raw_inputs = manifest.get('inputs', [])
    for index, item in enumerate(raw_inputs):
        if item.get('channel_implementation') != channel:
            raise ValueError(f'{manifest_path}: raw input channel metadata differs')
        recorded_path = _resolve_recorded_path(str(item['path']))
        if recorded_path.is_file():
            actual = rayleigh_sha256_file(recorded_path)
            if actual != item.get('input_sha256'):
                raise ValueError(f'raw input hash mismatch: {recorded_path}')
            raw_inputs[index] = {**item, 'verified_on_disk': True}
        else:
            raw_inputs[index] = {**item, 'verified_on_disk': False}
    provenance = {'directory': str(directory), 'per_image_results': str(csv_path), 'per_image_results_sha256': rayleigh_sha256_file(csv_path), 'summary': str(summary_path), 'summary_sha256': rayleigh_sha256_file(summary_path), 'run_manifest': str(manifest_path), 'run_manifest_sha256': rayleigh_sha256_file(manifest_path), 'raw_inputs': raw_inputs}
    return (_normalize_rows(csv_path, methods), channel, provenance)

def _validate_rows(rows: dict[tuple, dict[str, float]], *, setting: str, methods: tuple[str, ...], expected_rates: tuple[int, ...], expected_snrs: tuple[float, ...], expected_realizations: int, expected_images: dict[str, int]) -> None:
    actual_methods = {key[0] for key in rows}
    if actual_methods != set(methods):
        raise ValueError(f'{setting}: method set differs: {sorted(actual_methods)}')
    actual_splits = {key[1] for key in rows}
    if actual_splits != set(expected_images):
        raise ValueError(f'{setting}: split set differs: {sorted(actual_splits)}')
    for method in methods:
        for split, image_count in expected_images.items():
            selected = [key for key in rows if key[0] == method and key[1] == split]
            image_ids = {key[2] for key in selected}
            if len(image_ids) != image_count:
                raise ValueError(f'{setting}/{method}/{split}: expected {image_count} images, got {len(image_ids)}')
            expected = {(image_id, rate, snr, realization) for image_id in image_ids for rate in expected_rates for snr in expected_snrs for realization in range(expected_realizations)}
            actual = {(key[2], key[3], key[4], key[5]) for key in selected}
            if actual != expected:
                raise ValueError(f'{setting}/{method}/{split}: incomplete condition grid')
            seed_by_realization: dict[int, set[int]] = defaultdict(set)
            for key in selected:
                seed_by_realization[key[5]].add(key[6])
            if any((len(values) != 1 for values in seed_by_realization.values())):
                raise ValueError(f'{setting}/{method}/{split}: inconsistent seeds')

def _bootstrap_mean(values: list[float], samples: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError('bootstrap values must be a non-empty vector')
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return {'mean': float(array.mean()), 'ci95_low': float(np.quantile(means, 0.025)), 'ci95_high': float(np.quantile(means, 0.975)), 'num_images': int(array.size), 'bootstrap_samples': int(samples), 'bootstrap_seed': int(seed)}

def _mean_grouped_rows(settings: dict[str, dict[tuple, dict[str, float]]]) -> tuple[list[dict], list[dict], list[dict]]:
    cells: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    snrs: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    macros: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    for setting, rows in settings.items():
        for key, values in rows.items():
            method, split, _, rate, snr, _, _ = key
            cells[setting, method, split, rate, snr].append(values)
            snrs[setting, method, split, snr].append(values)
            macros[setting, method, split].append(values)

    def summarize(grouped: dict[tuple, list[dict]], fields: tuple[str, ...]) -> list[dict]:
        output = []
        for group, values in sorted(grouped.items()):
            output.append({**dict(zip(fields, group)), **{metric: float(np.mean([item[metric] for item in values])) for metric in RAYLEIGH_METRICS + ('cbr',)}})
        return output
    return (summarize(cells, ('setting', 'method', 'split', 'rate', 'snr_db')), summarize(snrs, ('setting', 'method', 'split', 'snr_db')), summarize(macros, ('setting', 'method', 'split')))

def _difference_observations(settings: dict[str, dict[tuple, dict[str, float]]], primary_method: str, baseline_method: str) -> dict[tuple[str, str], list[dict]]:
    observations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for setting, rows in settings.items():
        for key, primary in rows.items():
            if key[0] != primary_method:
                continue
            _, split, image_id, rate, snr, realization, seed = key
            baseline_key = (baseline_method, *key[1:])
            baseline = rows[baseline_key]
            observations['method_difference', setting].append({'split': split, 'image_id': image_id, 'rate': rate, 'snr_db': snr, 'realization': realization, 'evaluation_seed': seed, **{metric: primary[metric] - baseline[metric] for metric in RAYLEIGH_METRICS}})
    zero_rows = settings['zero_shot']
    matched_rows = settings['matched_training']
    for method in (primary_method, baseline_method):
        for key, matched in matched_rows.items():
            if key[0] != method:
                continue
            _, split, image_id, rate, snr, realization, seed = key
            zero = zero_rows[key]
            observations['adaptation_gain', method].append({'split': split, 'image_id': image_id, 'rate': rate, 'snr_db': snr, 'realization': realization, 'evaluation_seed': seed, **{metric: matched[metric] - zero[metric] for metric in RAYLEIGH_METRICS}})
    zero_difference = {(row['split'], row['image_id'], row['rate'], row['snr_db'], row['realization'], row['evaluation_seed']): row for row in observations['method_difference', 'zero_shot']}
    subject = f'{primary_method}_vs_{baseline_method}'
    for matched in observations['method_difference', 'matched_training']:
        key = (matched['split'], matched['image_id'], matched['rate'], matched['snr_db'], matched['realization'], matched['evaluation_seed'])
        zero = zero_difference[key]
        observations['difference_in_differences', subject].append({**{field: matched[field] for field in key_fields_without_method()}, **{metric: matched[metric] - zero[metric] for metric in RAYLEIGH_METRICS}})
    return observations

def key_fields_without_method() -> tuple[str, ...]:
    return RAYLEIGH_KEY_FIELDS[1:]

def _summarize_effects(observations: dict[tuple[str, str], list[dict]], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    output = []
    next_seed = bootstrap_seed
    scopes = (('macro', ('split',)), ('snr', ('split', 'snr_db')), ('cell', ('split', 'rate', 'snr_db')))
    for (comparison, subject), rows in sorted(observations.items()):
        for scope, group_fields in scopes:
            grouped: dict[tuple, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for row in rows:
                group = tuple((row[field] for field in group_fields))
                for metric in RAYLEIGH_METRICS:
                    grouped[group][row['image_id']][metric].append(row[metric])
            for group, per_image in sorted(grouped.items()):
                for metric in RAYLEIGH_METRICS:
                    values = [float(np.mean(per_image[image_id][metric])) for image_id in sorted(per_image)]
                    output.append({'comparison': comparison, 'subject': subject, 'scope': scope, 'split': group[0], 'rate': group[group_fields.index('rate')] if 'rate' in group_fields else '', 'snr_db': group[group_fields.index('snr_db')] if 'snr_db' in group_fields else '', 'metric': metric, **_bootstrap_mean(values, bootstrap_samples, next_seed)})
                    next_seed += 1
    return output

def _write_rayleigh_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_rayleigh_summary(path: Path, payload: dict) -> None:
    lines = ['# DecompJSCC Rayleigh zero-shot 与 matched-training 对比', '', '所有区间均为 10,000 次 image-level paired bootstrap 95% CI。每张图先对目标条件和三个 fading realization 取平均；训练 seed 只有 1 个。', '', '## 方法间差值', '', '差值定义为 `DecompJSCC - official SwinJSCC universal`。', '', '| 设置 | 数据集 | 指标 | 均值 | 95% CI |', '|---|---|---:|---:|---:|']
    macros = [row for row in payload['effects'] if row['scope'] == 'macro']
    for row in macros:
        if row['comparison'] != 'method_difference':
            continue
        lines.append(f"| {row['subject']} | {row['split']} | {row['metric']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines.extend(['', '## 匹配训练增益', '', '差值定义为 `matched training - zero-shot`。', '', '| 方法 | 数据集 | 指标 | 均值 | 95% CI |', '|---|---|---:|---:|---:|'])
    for row in macros:
        if row['comparison'] != 'adaptation_gain':
            continue
        lines.append(f"| {row['subject']} | {row['split']} | {row['metric']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines.extend(['', '## Difference-in-differences', '', '定义为 `(DecompJSCC matched-zero-shot) - (official matched-zero-shot)`；负值表示 official 从匹配训练中获得的提升更大。', '', '| 数据集 | 指标 | 均值 | 95% CI |', '|---|---:|---:|---:|'])
    for row in macros:
        if row['comparison'] != 'difference_in_differences':
            continue
        lines.append(f"| {row['split']} | {row['metric']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines.extend(['', '## 证据边界', '', '- 本分析支持 source-faithful、无随机相位/CSI/equalization 的 Rayleigh 协议，不外推到其他衰落定义。', '- image bootstrap 描述测试图像不确定性，不包含多训练 seed 变异。', '- matched training 后的绝对性能和方法间优势分开解释；不把更高最终性能等同于更大的适配增益。'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def analyze_rayleigh_adaptation(*, zero_dir: Path, matched_dir: Path, output_dir: Path, methods: tuple[str, ...], primary_method: str, baseline_method: str, expected_rates: tuple[int, ...], expected_snrs: tuple[float, ...], expected_realizations: int, expected_images: dict[str, int], bootstrap_samples: int, bootstrap_seed: int) -> dict:
    if set(methods) != {primary_method, baseline_method}:
        raise ValueError('methods must contain exactly primary and baseline')
    method_set = set(methods)
    settings = {}
    channels = {}
    inputs = {}
    for setting, directory in (('zero_shot', zero_dir), ('matched_training', matched_dir)):
        rows, channel, provenance = _load_setting(directory, method_set)
        _validate_rows(rows, setting=setting, methods=methods, expected_rates=expected_rates, expected_snrs=expected_snrs, expected_realizations=expected_realizations, expected_images=expected_images)
        settings[setting] = rows
        channels[setting] = channel
        inputs[setting] = provenance
    if channels['zero_shot'] != channels['matched_training']:
        raise ValueError('channel implementation differs between settings')
    if settings['zero_shot'].keys() != settings['matched_training'].keys():
        missing_matched = sorted(settings['zero_shot'].keys() - settings['matched_training'].keys())[:3]
        missing_zero = sorted(settings['matched_training'].keys() - settings['zero_shot'].keys())[:3]
        raise ValueError(f'zero-shot and matched row keys differ: missing matched={missing_matched}, missing zero-shot={missing_zero}')
    for key in settings['zero_shot']:
        if not np.isclose(settings['zero_shot'][key]['cbr'], settings['matched_training'][key]['cbr'], rtol=0.0, atol=1e-12):
            raise ValueError(f'CBR differs between settings for {key}')
    cells, snrs, macros = _mean_grouped_rows(settings)
    observations = _difference_observations(settings, primary_method, baseline_method)
    effects = _summarize_effects(observations, bootstrap_samples, bootstrap_seed)
    payload = {'material_passport': {'origin': 'experiment-agent', 'mode': 'validate', 'verification_status': 'ANALYZED', 'version': 'decompjscc_rayleigh_adaptation_v1'}, 'definitions': {'method_difference': f'{primary_method} - {baseline_method}', 'adaptation_gain': 'matched_training - zero_shot', 'difference_in_differences': f'({primary_method} matched-zero) - ({baseline_method} matched-zero)', 'bootstrap_unit': 'image after averaging requested conditions and realizations', 'training_seeds': 1}, 'validation': {'paired_rows_per_setting': len(settings['zero_shot']), 'expected_rates': list(expected_rates), 'expected_snr_db_values': list(expected_snrs), 'expected_realizations': expected_realizations, 'expected_images': expected_images, 'channel_implementation': channels['zero_shot']}, 'inputs': inputs, 'macro_performance': macros, 'effects': effects}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rayleigh_csv(output_dir / 'cell_performance.csv', cells, ['setting', 'method', 'split', 'rate', 'snr_db', 'psnr_db', 'ssim', 'cbr'])
    _write_rayleigh_csv(output_dir / 'snr_performance.csv', snrs, ['setting', 'method', 'split', 'snr_db', 'psnr_db', 'ssim', 'cbr'])
    _write_rayleigh_csv(output_dir / 'effects.csv', effects, ['comparison', 'subject', 'scope', 'split', 'rate', 'snr_db', 'metric', 'mean', 'ci95_low', 'ci95_high', 'num_images', 'bootstrap_samples', 'bootstrap_seed'])
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    _write_rayleigh_summary(output_dir / 'summary.md', payload)
    output_paths = [output_dir / 'cell_performance.csv', output_dir / 'snr_performance.csv', output_dir / 'effects.csv', summary_path, output_dir / 'summary.md']
    manifest = {'protocol': 'decompjscc_rayleigh_adaptation_v1', 'python': platform.python_version(), 'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed, 'channel_implementation': channels['zero_shot'], 'inputs': inputs, 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': rayleigh_sha256_file(path)} for path in output_paths]}
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return payload

def rayleigh_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Analyze DecompJSCC zero-shot and matched Rayleigh results')
    parser.add_argument('--zero-shot-dir', type=Path, required=True)
    parser.add_argument('--matched-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--methods', default='decompjscc,swinjscc_universal')
    parser.add_argument('--primary-method', default='decompjscc')
    parser.add_argument('--baseline-method', default='swinjscc_universal')
    parser.add_argument('--expected-rates', type=rayleigh_parse_csv_ints, default=(32, 64, 96, 128, 192))
    parser.add_argument('--expected-snr-values', type=rayleigh_parse_csv_floats, default=(1.0, 4.0, 7.0, 10.0, 13.0))
    parser.add_argument('--expected-realizations', type=int, default=3)
    parser.add_argument('--expected-images', action='append', default=['kodak=24', 'clic2021=60'])
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260811)
    return parser

def rayleigh_main() -> None:
    args = rayleigh_parser().parse_args()
    methods = tuple((item.strip() for item in args.methods.split(',') if item.strip()))
    result = analyze_rayleigh_adaptation(zero_dir=args.zero_shot_dir, matched_dir=args.matched_dir, output_dir=args.output_dir, methods=methods, primary_method=args.primary_method, baseline_method=args.baseline_method, expected_rates=args.expected_rates, expected_snrs=args.expected_snr_values, expected_realizations=args.expected_realizations, expected_images=rayleigh_parse_expected_images(args.expected_images), bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({'output_dir': str(args.output_dir), 'paired_rows_per_setting': result['validation']['paired_rows_per_setting'], 'effects': len(result['effects'])}, indent=2))
