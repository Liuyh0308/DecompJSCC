"""Summarize DecompJSCC versus BPG+LDPC without assuming common noise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RATES = (32, 64, 96, 128, 192)
SNR_VALUES = (1.0, 4.0, 7.0, 10.0, 13.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DecompJSCC versus BPG+LDPC")
    parser.add_argument(
        "--decomp-dir",
        required=True,
        help="Directory containing the evaluated DecompJSCC split JSON files",
    )
    parser.add_argument(
        "--bpg-dir",
        required=True,
        help="Directory containing the evaluated BPG+LDPC split grid JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "exp_log/decompjscc/analysis/awgn/external_baselines/bpg_ldpc_comparison"
        ),
    )
    parser.add_argument("--bootstrap-samples", default=10000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260827, type=int)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_realizations(rows: list[dict]) -> dict[tuple[str, int, float], dict]:
    grouped: dict[tuple[str, int, float], list[dict]] = {}
    for row in rows:
        key = (str(row["image_id"]), int(row["rate"]), float(row["snr_db"]))
        grouped.setdefault(key, []).append(row)
    averaged = {}
    for key, values in grouped.items():
        if len(values) != 3 or {int(row["realization"]) for row in values} != {0, 1, 2}:
            raise ValueError(f"Expected three realizations for {key}")
        averaged[key] = {
            "psnr_db": float(np.mean([float(row["psnr_db"]) for row in values])),
            "ssim": float(np.mean([float(row["ssim"]) for row in values])),
            "cbr": float(np.mean([float(row["cbr"] if "cbr" in row else row["actual_cbr"]) for row in values])),
        }
    return averaged


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> dict:
    if values.ndim != 1 or not values.size:
        raise ValueError("bootstrap input must be a non-empty vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "num_images": int(values.size),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(
    split: str,
    decomp_path: Path,
    bpg_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    decomp_payload = json.loads(decomp_path.read_text(encoding="utf-8"))
    bpg_payload = json.loads(bpg_path.read_text(encoding="utf-8"))
    if decomp_payload["eval_split"] != split or bpg_payload["eval_split"] != split:
        raise ValueError(f"Split declaration mismatch for {split}")
    if bpg_payload.get("mcs_selected_without_test_images") is not True:
        raise ValueError("BPG MCS selection must exclude test images")
    decomp = _average_realizations(decomp_payload["per_image_results"])
    bpg = _average_realizations(bpg_payload["per_image_results"])
    if set(decomp) != set(bpg):
        raise ValueError(f"Per-image condition keys differ for {split}")
    expected_images = 24 if split == "kodak" else 60
    expected_keys = expected_images * len(RATES) * len(SNR_VALUES)
    if len(decomp) != expected_keys:
        raise ValueError(f"Expected {expected_keys} averaged rows for {split}, got {len(decomp)}")

    bpg_cells = {
        (int(row["rate"]), float(row["snr_db"])): row for row in bpg_payload["results"]
    }
    cell_rows = []
    per_image_macro: dict[str, dict[str, list[float]]] = {}
    for cell_index, (rate, snr_db) in enumerate(
        (rate, snr) for rate in RATES for snr in SNR_VALUES
    ):
        keys = sorted(key for key in decomp if key[1:] == (rate, snr_db))
        psnr_difference = np.asarray(
            [decomp[key]["psnr_db"] - bpg[key]["psnr_db"] for key in keys]
        )
        ssim_difference = np.asarray(
            [decomp[key]["ssim"] - bpg[key]["ssim"] for key in keys]
        )
        psnr_ci = _bootstrap(
            psnr_difference,
            bootstrap_samples,
            bootstrap_seed + cell_index * 2,
        )
        ssim_ci = _bootstrap(
            ssim_difference,
            bootstrap_samples,
            bootstrap_seed + cell_index * 2 + 1,
        )
        bpg_cell = bpg_cells[(rate, snr_db)]
        cell_rows.append(
            {
                "split": split,
                "rate": rate,
                "snr_db": snr_db,
                "decomp_psnr_db": float(np.mean([decomp[key]["psnr_db"] for key in keys])),
                "bpg_psnr_db": float(np.mean([bpg[key]["psnr_db"] for key in keys])),
                "psnr_difference_db": psnr_ci["mean"],
                "psnr_ci95_low": psnr_ci["ci95_low"],
                "psnr_ci95_high": psnr_ci["ci95_high"],
                "decomp_ssim": float(np.mean([decomp[key]["ssim"] for key in keys])),
                "bpg_ssim": float(np.mean([bpg[key]["ssim"] for key in keys])),
                "ssim_difference": ssim_ci["mean"],
                "decomp_cbr": float(np.mean([decomp[key]["cbr"] for key in keys])),
                "bpg_actual_cbr": float(np.mean([bpg[key]["cbr"] for key in keys])),
                "bpg_decode_failure_rate": float(bpg_cell["decode_failure_rate"]),
            }
        )
        for key, psnr_value, ssim_value in zip(keys, psnr_difference, ssim_difference):
            metrics = per_image_macro.setdefault(key[0], {"psnr_db": [], "ssim": []})
            metrics["psnr_db"].append(float(psnr_value))
            metrics["ssim"].append(float(ssim_value))

    macro_rows = []
    for image_id, metrics in sorted(per_image_macro.items()):
        macro_rows.append(
            {
                "split": split,
                "image_id": image_id,
                "psnr_difference_db": float(np.mean(metrics["psnr_db"])),
                "ssim_difference": float(np.mean(metrics["ssim"])),
            }
        )
    macro_psnr = _bootstrap(
        np.asarray([row["psnr_difference_db"] for row in macro_rows]),
        bootstrap_samples,
        bootstrap_seed + 1000,
    )
    macro_ssim = _bootstrap(
        np.asarray([row["ssim_difference"] for row in macro_rows]),
        bootstrap_samples,
        bootstrap_seed + 1001,
    )
    summary = {
        "split": split,
        "difference": "DecompJSCC - BPG+5G_LDPC",
        "paired_unit": "image after averaging three independent-noise realizations per cell",
        "common_random_noise_across_architectures": False,
        "macro_psnr_difference_db": macro_psnr,
        "macro_ssim_difference": macro_ssim,
        "decomp_macro_psnr_db": float(np.mean([row["decomp_psnr_db"] for row in cell_rows])),
        "bpg_macro_psnr_db": float(np.mean([row["bpg_psnr_db"] for row in cell_rows])),
        "bpg_failure_cells": sum(row["bpg_decode_failure_rate"] > 0 for row in cell_rows),
    }
    return summary, cell_rows, macro_rows


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    decomp_dir = Path(args.decomp_dir)
    bpg_dir = Path(args.bpg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = []
    all_cells = []
    all_images = []
    inputs = []
    for split_index, split in enumerate(("kodak", "clic2021")):
        decomp_path = decomp_dir / f"{split}.json"
        bpg_path = bpg_dir / f"{split}_grid.json"
        summary, cells, images = summarize_split(
            split,
            decomp_path,
            bpg_path,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + split_index * 10000,
        )
        split_summaries.append(summary)
        all_cells.extend(cells)
        all_images.extend(images)
        inputs.extend(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (decomp_path, bpg_path)
        )
    payload = {
        "protocol": "decompjscc_bpg_ldpc_paired_image_summary_v1",
        "command": sys.argv,
        "inputs": inputs,
        "splits": split_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "cell_comparison.csv", all_cells)
    _write_csv(output_dir / "per_image_macro_difference.csv", all_images)
    markdown = [
        "# DecompJSCC 与 BPG+5G LDPC 正式比较",
        "",
        "差值统一定义为 `DecompJSCC - BPG+5G_LDPC`。跨架构不假设共同噪声；每个 cell 先在图像内平均三次 realization，再按图像配对 bootstrap。",
        "",
        "| 数据集 | DecompJSCC macro PSNR | BPG+LDPC macro PSNR | paired 差值（95% CI） | BPG 有失败的 cell |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in split_summaries:
        ci = row["macro_psnr_difference_db"]
        markdown.append(
            f"| {row['split']} | {row['decomp_macro_psnr_db']:.4f} | "
            f"{row['bpg_macro_psnr_db']:.4f} | {ci['mean']:+.4f} "
            f"[{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}] | "
            f"{row['bpg_failure_cells']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "splits": len(split_summaries)}, indent=2))


if __name__ == "__main__":
    main()
