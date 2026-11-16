"""Tests for evaluation summaries and interpolation/channel analyses."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.statistics import (
    EvaluationInput,
    analyze_channel_transfer,
    analyze_interpolation,
    analyze_rayleigh_adaptation,
    summarize_evaluations,
    rayleigh_parser,
)


class TestInterpolationAudit(unittest.TestCase):
    def test_channel_comparison_requires_explicit_result_inputs(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rayleigh_parser().parse_args([])
        args = rayleigh_parser().parse_args([
            "--zero-shot-dir", "outputs/zero", "--matched-dir", "outputs/matched",
        ])
        self.assertEqual(args.zero_shot_dir, Path("outputs/zero"))
        self.assertEqual(args.matched_dir, Path("outputs/matched"))

    @staticmethod
    def _write(path: Path, rows: list[dict]) -> None:
        path.write_text(json.dumps({"results": rows}), encoding="utf-8")

    def test_reports_bracketed_points_and_violations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen = root / "seen.json"
            snr = root / "snr.json"
            rate = root / "rate.json"
            self._write(
                seen,
                [
                    {"rate": 32, "snr_db": 1, "psnr_db": 10, "ssim": 0.1},
                    {"rate": 32, "snr_db": 4, "psnr_db": 13, "ssim": 0.4},
                    {"rate": 64, "snr_db": 1, "psnr_db": 12, "ssim": 0.2},
                    {"rate": 64, "snr_db": 4, "psnr_db": 15, "ssim": 0.5},
                ],
            )
            self._write(snr, [{"rate": 32, "snr_db": 2.5, "psnr_db": 11.5, "ssim": 0.25}])
            self._write(rate, [{"rate": 48, "snr_db": 1, "psnr_db": 14, "ssim": 0.15}])

            result = analyze_interpolation([f"method:split:{seen}:{snr}:{rate}"], root / "output")

            self.assertEqual(result["checks"], 4)
            self.assertEqual(result["violations"], 1)
            self.assertFalse(result["all_checks_pass"])
            self.assertTrue((root / "output/monotonicity.json").is_file())
            self.assertTrue((root / "output/monotonicity.csv").is_file())


class ChannelTransferAnalysisTest(unittest.TestCase):
    def test_paired_degradation_and_method_sensitivity(self):
        fields = [
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
        ]
        awgn_values = {
            ("decompjscc", "image1.png"): (30.0, 0.90),
            ("decompjscc", "image2.png"): (32.0, 0.92),
            ("swinjscc_universal", "image1.png"): (29.0, 0.88),
            ("swinjscc_universal", "image2.png"): (31.0, 0.90),
        }
        rayleigh_values = {
            ("decompjscc", "image1.png"): (25.0, 0.80),
            ("decompjscc", "image2.png"): (28.0, 0.84),
            ("swinjscc_universal", "image1.png"): (23.0, 0.76),
            ("swinjscc_universal", "image2.png"): (26.0, 0.80),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            awgn_path = root / "awgn.csv"
            rayleigh_path = root / "rayleigh.csv"
            for path, values in (
                (awgn_path, awgn_values),
                (rayleigh_path, rayleigh_values),
            ):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for (method, image_id), (psnr, ssim) in values.items():
                        writer.writerow(
                            {
                                "method": method,
                                "split": "kodak",
                                "image_id": image_id,
                                "rate": 96,
                                "snr_db": 7,
                                "realization": 0,
                                "evaluation_seed": 20260811,
                                "psnr_db": psnr,
                                "ssim": ssim,
                                "cbr": 0.0625,
                            }
                        )

            result = analyze_channel_transfer(
                awgn_path,
                rayleigh_path,
                root / "output",
                ("decompjscc", "swinjscc_universal"),
                "decompjscc",
                "swinjscc_universal",
                bootstrap_samples=100,
                bootstrap_seed=7,
            )

            macro = {
                (row["method"], row["metric"]): row["mean"]
                for row in result["macro"]
            }
            sensitivity = {
                row["metric"]: row["mean"]
                for row in result["method_sensitivity"]
            }
            self.assertEqual(result["paired_rows"], 4)
            self.assertAlmostEqual(macro[("decompjscc", "psnr_db")], -4.5)
            self.assertAlmostEqual(
                macro[("swinjscc_universal", "psnr_db")], -5.5
            )
            self.assertAlmostEqual(sensitivity["psnr_db"], 1.0)
            self.assertTrue((root / "output" / "channel_transfer.csv").is_file())
            self.assertTrue((root / "output" / "summary.md").is_file())


class RayleighAdaptationAnalysisTest(unittest.TestCase):
    def _write_setting(
        self,
        root: Path,
        setting: str,
        values: dict[tuple[str, str], tuple[float, float]],
    ) -> Path:
        directory = root / setting
        directory.mkdir()
        fields = [
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
        ]
        with (directory / "per_image_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for (method, image_id), (psnr, ssim) in values.items():
                writer.writerow(
                    {
                        "method": method,
                        "split": "kodak",
                        "image_id": image_id,
                        "rate": 96,
                        "snr_db": 7,
                        "realization": 0,
                        "evaluation_seed": 20260811,
                        "psnr_db": psnr,
                        "ssim": ssim,
                        "cbr": 0.0625,
                    }
                )
        channel = {
            "source_path": "SwinJSCC_base/channel.py",
            "source_sha256": "channel-sha",
            "channel_type": "rayleigh",
        }
        (directory / "summary.json").write_text(
            json.dumps({"channel_type": "rayleigh", "channel_implementation": channel}),
            encoding="utf-8",
        )
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "inputs": [
                        {
                            "method": method,
                            "split": "kodak",
                            "path": f"{setting}-{method}.json",
                            "input_sha256": f"{setting}-{method}-sha",
                            "checkpoint": f"{setting}-{method}.pth",
                            "checkpoint_sha256": f"{setting}-{method}-checkpoint-sha",
                            "image_ids": ["image1.png", "image2.png"],
                            "evaluation_seeds": {"0": 20260811},
                            "channel_implementation": channel,
                        }
                        for method in ("decompjscc", "swinjscc_universal")
                    ]
                }
            ),
            encoding="utf-8",
        )
        return directory

    def test_method_adaptation_and_difference_in_differences(self):
        zero_values = {
            ("decompjscc", "image1.png"): (25.0, 0.80),
            ("decompjscc", "image2.png"): (27.0, 0.82),
            ("swinjscc_universal", "image1.png"): (24.0, 0.77),
            ("swinjscc_universal", "image2.png"): (26.0, 0.79),
        }
        matched_values = {
            ("decompjscc", "image1.png"): (28.0, 0.86),
            ("decompjscc", "image2.png"): (30.0, 0.88),
            ("swinjscc_universal", "image1.png"): (26.0, 0.82),
            ("swinjscc_universal", "image2.png"): (28.0, 0.84),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero_dir = self._write_setting(root, "zero_shot", zero_values)
            matched_dir = self._write_setting(
                root, "matched_training", matched_values
            )
            result = analyze_rayleigh_adaptation(
                zero_dir=zero_dir,
                matched_dir=matched_dir,
                output_dir=root / "output",
                methods=("decompjscc", "swinjscc_universal"),
                primary_method="decompjscc",
                baseline_method="swinjscc_universal",
                expected_rates=(96,),
                expected_snrs=(7.0,),
                expected_realizations=1,
                expected_images={"kodak": 2},
                bootstrap_samples=100,
                bootstrap_seed=7,
            )

            macro = {
                (row["comparison"], row["subject"], row["metric"]): row["mean"]
                for row in result["effects"]
                if row["scope"] == "macro"
            }
            self.assertAlmostEqual(
                macro[("method_difference", "zero_shot", "psnr_db")], 1.0
            )
            self.assertAlmostEqual(
                macro[("method_difference", "matched_training", "psnr_db")],
                2.0,
            )
            self.assertAlmostEqual(
                macro[("adaptation_gain", "decompjscc", "psnr_db")], 3.0
            )
            self.assertAlmostEqual(
                macro[("adaptation_gain", "swinjscc_universal", "psnr_db")],
                2.0,
            )
            self.assertAlmostEqual(
                macro[
                    (
                        "difference_in_differences",
                        "decompjscc_vs_swinjscc_universal",
                        "psnr_db",
                    )
                ],
                1.0,
            )
            self.assertEqual(result["validation"]["paired_rows_per_setting"], 4)
            for name in (
                "summary.json",
                "summary.md",
                "cell_performance.csv",
                "snr_performance.csv",
                "effects.csv",
                "run_manifest.json",
            ):
                self.assertTrue((root / "output" / name).is_file())

    def test_rejects_unpaired_settings(self):
        values = {
            ("decompjscc", "image1.png"): (25.0, 0.80),
            ("decompjscc", "image2.png"): (27.0, 0.82),
            ("swinjscc_universal", "image1.png"): (24.0, 0.77),
            ("swinjscc_universal", "image2.png"): (26.0, 0.79),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero_dir = self._write_setting(root, "zero_shot", values)
            matched_dir = self._write_setting(root, "matched_training", values)
            with (matched_dir / "per_image_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["image_id"] = "image3.png"
            rows[3]["image_id"] = "image3.png"
            with (matched_dir / "per_image_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "row keys differ"):
                analyze_rayleigh_adaptation(
                    zero_dir=zero_dir,
                    matched_dir=matched_dir,
                    output_dir=root / "output",
                    methods=("decompjscc", "swinjscc_universal"),
                    primary_method="decompjscc",
                    baseline_method="swinjscc_universal",
                    expected_rates=(96,),
                    expected_snrs=(7.0,),
                    expected_realizations=1,
                    expected_images={"kodak": 2},
                    bootstrap_samples=10,
                    bootstrap_seed=7,
                )


def build_payload(offset: float, *, wrong_seed: bool = False, channel_sha: str | None = None) -> dict:
    rates = (32, 64)
    snrs = (1.0, 4.0)
    image_ids = ("a.png", "b.png")
    realization_results = []
    per_image_results = []
    for rate in rates:
        for snr in snrs:
            for realization in range(2):
                seed = 10 + realization + (1 if wrong_seed else 0)
                image_psnr = []
                image_ssim = []
                for image_index, image_id in enumerate(image_ids):
                    psnr = 20.0 + offset + rate / 100.0 + snr / 10.0 + image_index
                    ssim = 0.7 + offset / 100.0 + image_index / 100.0
                    image_psnr.append(psnr)
                    image_ssim.append(ssim)
                    per_image_results.append(
                        {
                            "image_id": image_id,
                            "rate": rate,
                            "snr_db": snr,
                            "realization": realization,
                            "evaluation_seed": seed,
                            "psnr_db": psnr,
                            "ssim": ssim,
                            "cbr": rate / 1536.0,
                        }
                    )
                realization_results.append(
                    {
                        "rate": rate,
                        "snr_db": snr,
                        "realization": realization,
                        "evaluation_seed": seed,
                        "psnr_db": sum(image_psnr) / len(image_psnr),
                        "ssim": sum(image_ssim) / len(image_ssim),
                        "cbr": rate / 1536.0,
                    }
                )
    results = []
    for rate in rates:
        for snr in snrs:
            matching = [
                row for row in realization_results if row["rate"] == rate and row["snr_db"] == snr
            ]
            results.append(
                {
                    "rate": rate,
                    "snr_db": snr,
                    "psnr_db": sum(row["psnr_db"] for row in matching) / len(matching),
                    "ssim": sum(row["ssim"] for row in matching) / len(matching),
                    "cbr": rate / 1536.0,
                }
            )
    payload = {
        "eval_split": "kodak",
        "noise_realizations": 2,
        "common_random_numbers": True,
        "checkpoint_load": {"skipped": []},
        "results": results,
        "realization_results": realization_results,
        "per_image_results": per_image_results,
    }
    if channel_sha is not None:
        payload.update(
            {
                "channel_type": "rayleigh",
                "channel_implementation": {
                    "class_name": "Channel",
                    "source_sha256": channel_sha,
                },
            }
        )
    return payload


class EvaluationSummaryTest(unittest.TestCase):
    def test_summarizes_and_bootstraps_paired_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_path = root / "main.json"
            baseline_path = root / "baseline.json"
            main_path.write_text(json.dumps(build_payload(0.5)), encoding="utf-8")
            baseline_path.write_text(json.dumps(build_payload(0.0)), encoding="utf-8")
            output_dir = root / "summary"
            summary = summarize_evaluations(
                inputs=[
                    EvaluationInput("decompjscc", "kodak", main_path),
                    EvaluationInput("swinjscc_universal", "kodak", baseline_path),
                ],
                output_dir=output_dir,
                expected_rates=(32, 64),
                expected_snrs=(1.0, 4.0),
                expected_realizations=2,
                expected_images={"kodak": 2},
                primary_method="decompjscc",
                baseline_method="swinjscc_universal",
                bootstrap_samples=100,
                bootstrap_seed=7,
            )
            comparison = summary["paired_comparison"]["splits"][0]
            self.assertAlmostEqual(comparison["macro"]["psnr_db"]["mean"], 0.5)
            self.assertFalse(comparison["extend_primary_comparison_to_10_realizations"])
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "per_image_results.csv").is_file())
            self.assertEqual(len(list((output_dir / "per_rate").glob("*.json"))), 4)

    def test_rejects_cross_method_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_path = root / "main.json"
            baseline_path = root / "baseline.json"
            main_path.write_text(json.dumps(build_payload(0.5)), encoding="utf-8")
            baseline_path.write_text(json.dumps(build_payload(0.0, wrong_seed=True)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CRN seed mismatch"):
                summarize_evaluations(
                    inputs=[
                        EvaluationInput("decompjscc", "kodak", main_path),
                        EvaluationInput("swinjscc_universal", "kodak", baseline_path),
                    ],
                    output_dir=root / "summary",
                    expected_rates=(32, 64),
                    expected_snrs=(1.0, 4.0),
                    expected_realizations=2,
                    expected_images={"kodak": 2},
                    primary_method="decompjscc",
                    baseline_method="swinjscc_universal",
                    bootstrap_samples=100,
                    bootstrap_seed=7,
                )

    def test_rejects_cross_method_channel_implementation_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_path = root / "main.json"
            baseline_path = root / "baseline.json"
            main_path.write_text(json.dumps(build_payload(0.5, channel_sha="a")), encoding="utf-8")
            baseline_path.write_text(
                json.dumps(build_payload(0.0, channel_sha="b")), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "channel implementation differs"):
                summarize_evaluations(
                    inputs=[
                        EvaluationInput("decompjscc", "kodak", main_path),
                        EvaluationInput("swinjscc_universal", "kodak", baseline_path),
                    ],
                    output_dir=root / "summary",
                    expected_rates=(32, 64),
                    expected_snrs=(1.0, 4.0),
                    expected_realizations=2,
                    expected_images={"kodak": 2},
                    primary_method="decompjscc",
                    baseline_method="swinjscc_universal",
                    bootstrap_samples=100,
                    bootstrap_seed=7,
                    expected_channel_type="rayleigh",
                )


if __name__ == "__main__":
    unittest.main()
