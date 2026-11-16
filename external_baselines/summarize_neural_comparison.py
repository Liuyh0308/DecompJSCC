"""Summarize the paper's joint 5x5 grid for three neural methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RATES = (32, 64, 96, 128, 192)
SNR_VALUES = (1.0, 4.0, 7.0, 10.0, 13.0)
DISPLAY_NAMES = {
    "DecompJSCC": "DecompJSCC",
    "Official_SwinJSCC_Universal": "SwinJSCC universal",
    "SA_RA_JSCC": "SA-RA-JSCC",
}
SCOPE_LABELS = {
    "primary_joint": "主文 joint SNR-rate 证据",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the joint 5x5 grid on Kodak and CLIC2021"
    )
    parser.add_argument("--input-dir", type=Path,
                        default=REPO_ROOT / "paper_data/evidence/claims/joint_grid")
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "outputs/neural_comparison"
        ),
    )
    parser.add_argument("--bootstrap-samples", default=10000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260830, type=int)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(split: str, input_dir: str | Path | None = None) -> dict[str, tuple[Path, ...]]:
    base = Path(input_dir) if input_dir is not None else REPO_ROOT / "paper_data/evidence/claims/joint_grid"
    return {
        "DecompJSCC": (base / f"decompjscc_{split}.json",),
        "Official_SwinJSCC_Universal": (base / f"swinjscc_universal_{split}.json",),
        "SA_RA_JSCC": (base / f"sa_ra_jscc_{split}.json",),
    }


def _average_realizations(rows: list[dict]) -> dict[tuple[str, int, float], dict]:
    grouped: dict[tuple[str, int, float], list[dict]] = {}
    for row in rows:
        key = (str(row["image_id"]), int(row["rate"]), float(row["snr_db"]))
        grouped.setdefault(key, []).append(row)
    averaged = {}
    for key, values in grouped.items():
        realizations = {int(row["realization"]) for row in values}
        if len(values) != 3 or realizations != {0, 1, 2}:
            raise ValueError(f"Expected three realizations for {key}")
        averaged[key] = {
            "psnr_db": float(np.mean([float(row["psnr_db"]) for row in values])),
            "ssim": float(np.mean([float(row["ssim"]) for row in values])),
            "cbr": float(np.mean([float(row["cbr"]) for row in values])),
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


def _group_specs() -> tuple[dict, ...]:
    return (
        {
            "name": "joint_5x5",
            "title": "Joint SNR-rate 主比较（5x5）",
            "scope": "primary_joint",
            # This group was index 1 in the original multi-group bootstrap.
            "bootstrap_seed_offset": 10000,
            "reference": "DecompJSCC",
            "rates": RATES,
            "methods": (
                "DecompJSCC",
                "Official_SwinJSCC_Universal",
                "SA_RA_JSCC",
            ),
        },
    )


def summarize_group(
    split: str,
    group: dict,
    methods: dict[str, dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rates = tuple(int(rate) for rate in group["rates"])
    expected_cells = {(rate, snr) for rate in rates for snr in SNR_VALUES}
    selected: dict[str, dict] = {}
    for method in group["methods"]:
        rows = {
            key: value
            for key, value in methods[method].items()
            if key[1] in rates and key[2] in SNR_VALUES
        }
        cells = {(key[1], key[2]) for key in rows}
        if cells != expected_cells:
            raise ValueError(f"Unexpected condition grid for {split}/{group['name']}/{method}")
        selected[method] = rows

    reference_method = str(group["reference"])
    reference = selected[reference_method]
    expected_images = 24 if split == "kodak" else 60
    expected_keys = expected_images * len(expected_cells)
    if len(reference) != expected_keys:
        raise ValueError(
            f"Expected {expected_keys} rows for {split}/{group['name']}, got {len(reference)}"
        )
    for method, rows in selected.items():
        if set(rows) != set(reference):
            raise ValueError(f"Per-image keys differ for {split}/{group['name']}/{method}")

    cell_rows = []
    summary_rows = []
    image_rows = []
    for method_index, method in enumerate(group["methods"]):
        rows = selected[method]
        per_image: dict[str, dict[str, list[float]]] = {}
        for cell_index, (rate, snr_db) in enumerate(
            (rate, snr) for rate in rates for snr in SNR_VALUES
        ):
            keys = sorted(key for key in reference if key[1:] == (rate, snr_db))
            method_psnr = np.asarray([rows[key]["psnr_db"] for key in keys])
            reference_psnr = np.asarray([reference[key]["psnr_db"] for key in keys])
            method_ssim = np.asarray([rows[key]["ssim"] for key in keys])
            reference_ssim = np.asarray([reference[key]["ssim"] for key in keys])
            psnr_difference = reference_psnr - method_psnr
            ssim_difference = reference_ssim - method_ssim
            psnr_ci = _bootstrap(
                psnr_difference,
                bootstrap_samples,
                bootstrap_seed + method_index * 1000 + cell_index * 2,
            )
            ssim_ci = _bootstrap(
                ssim_difference,
                bootstrap_samples,
                bootstrap_seed + method_index * 1000 + cell_index * 2 + 1,
            )
            cell_rows.append(
                {
                    "split": split,
                    "group": group["name"],
                    "scope": group["scope"],
                    "reference_method": reference_method,
                    "method": method,
                    "rate": rate,
                    "snr_db": snr_db,
                    "psnr_db": float(method_psnr.mean()),
                    "ssim": float(method_ssim.mean()),
                    "cbr": float(np.mean([rows[key]["cbr"] for key in keys])),
                    "reference_minus_method_psnr_db": psnr_ci["mean"],
                    "psnr_ci95_low": psnr_ci["ci95_low"],
                    "psnr_ci95_high": psnr_ci["ci95_high"],
                    "reference_minus_method_ssim": ssim_ci["mean"],
                }
            )
            for key, psnr_value, ssim_value in zip(
                keys, psnr_difference, ssim_difference
            ):
                metrics = per_image.setdefault(key[0], {"psnr_db": [], "ssim": []})
                metrics["psnr_db"].append(float(psnr_value))
                metrics["ssim"].append(float(ssim_value))

        method_cells = [
            row
            for row in cell_rows
            if row["split"] == split
            and row["group"] == group["name"]
            and row["method"] == method
        ]
        macro_image_rows = []
        for image_id, metrics in sorted(per_image.items()):
            row = {
                "split": split,
                "group": group["name"],
                "scope": group["scope"],
                "reference_method": reference_method,
                "method": method,
                "image_id": image_id,
                "reference_minus_method_psnr_db": float(np.mean(metrics["psnr_db"])),
                "reference_minus_method_ssim": float(np.mean(metrics["ssim"])),
            }
            image_rows.append(row)
            macro_image_rows.append(row)
        psnr_ci = _bootstrap(
            np.asarray(
                [row["reference_minus_method_psnr_db"] for row in macro_image_rows]
            ),
            bootstrap_samples,
            bootstrap_seed + method_index * 1000 + 900,
        )
        ssim_ci = _bootstrap(
            np.asarray([row["reference_minus_method_ssim"] for row in macro_image_rows]),
            bootstrap_samples,
            bootstrap_seed + method_index * 1000 + 901,
        )
        summary_rows.append(
            {
                "split": split,
                "group": group["name"],
                "scope": group["scope"],
                "reference_method": reference_method,
                "method": method,
                "macro_psnr_db": float(np.mean([row["psnr_db"] for row in method_cells])),
                "macro_ssim": float(np.mean([row["ssim"] for row in method_cells])),
                "reference_minus_method_psnr_db": psnr_ci,
                "reference_minus_method_ssim": ssim_ci,
            }
        )
    return summary_rows, cell_rows, image_rows


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    all_summaries = []
    all_cells = []
    all_images = []
    inputs = []
    for split_index, split in enumerate(("kodak", "clic2021")):
        paths = _input_paths(split, args.input_dir)
        methods = {}
        for method, method_paths in paths.items():
            merged = {}
            for path in method_paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                declared_split = payload.get("eval_split", payload.get("hr_eval_split"))
                if declared_split != split:
                    raise ValueError(f"Split mismatch in {path}")
                averaged = _average_realizations(payload["per_image_results"])
                overlap = set(merged).intersection(averaged)
                if overlap:
                    raise ValueError(f"Duplicate condition keys across inputs for {method}")
                merged.update(averaged)
                inputs.append(
                    {
                        "method": method,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    }
                )
            methods[method] = merged
        for group in _group_specs():
            summaries, cells, images = summarize_group(
                split,
                group,
                methods,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=(
                    args.bootstrap_seed + split_index * 100000 + group["bootstrap_seed_offset"]
                ),
            )
            all_summaries.extend(summaries)
            all_cells.extend(cells)
            all_images.extend(images)

    payload = {
        "protocol": "decompjscc_dimension_matched_neural_summary_v2",
        "command": sys.argv,
        "comparison_difference": "reference_method minus method",
        "paired_unit": "image after averaging three independent-noise realizations per cell",
        "common_random_noise_across_architectures": False,
        "groups": [
            {
                "name": group["name"],
                "title": group["title"],
                "scope": group["scope"],
                "bootstrap_seed_offset": group["bootstrap_seed_offset"],
                "reference": group["reference"],
                "rates": list(group["rates"]),
                "methods": list(group["methods"]),
            }
            for group in _group_specs()
        ],
        "inputs": inputs,
        "summaries": all_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "cell_comparison.csv", all_cells)
    _write_csv(output_dir / "per_image_macro_difference.csv", all_images)
    markdown = [
        "# DecompJSCC 与外部神经基线正式比较",
        "",
        "差值定义为 `DecompJSCC - method`。每个 cell 先在图像内平均 3 次独立噪声 realization，再按图像配对 bootstrap；不假设跨架构共同噪声。每个数据集包含五档码率和五档SNR，共25个条件。",
    ]
    for group in _group_specs():
        markdown.extend(
            [
                "",
                f"## {group['title']}",
                "",
                f"参考方法：{DISPLAY_NAMES[group['reference']]}；证据范围：{SCOPE_LABELS[group['scope']]}。",
                "",
                "| 数据集 | 方法 | macro PSNR | Reference - 方法（95% CI） |",
                "|---|---|---:|---:|",
            ]
        )
        for row in all_summaries:
            if row["group"] != group["name"]:
                continue
            ci = row["reference_minus_method_psnr_db"]
            markdown.append(
                f"| {row['split']} | {DISPLAY_NAMES[row['method']]} | "
                f"{row['macro_psnr_db']:.4f} | {ci['mean']:+.4f} "
                f"[{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}] |"
            )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_rows": len(all_summaries),
                "cell_rows": len(all_cells),
                "image_rows": len(all_images),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
