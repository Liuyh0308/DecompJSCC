"""Focused contracts for the shared evaluation entry point; no model inference."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiments import evaluation as evaluation
from experiments import reconstruction
from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep as bpg


class EvaluationTests(unittest.TestCase):
    def test_source_config_is_resolved_from_public_checkpoint_index(self):
        from experiments import source_evaluation as source
        for checkpoint, config in [
            ("checkpoints/decompjscc/awgn/model_best.pth", "configs/models/decompjscc.json"),
            ("checkpoints/swinjscc_sa/awgn/cbr_1_16/model_best.pth", "configs/models/swinjscc_sa_cbr_1_16.json"),
        ]:
            self.assertEqual(source.resolve_config_path(checkpoint, ""), str(evaluation.ROOT / config))
        self.assertEqual(source.resolve_config_path("unused.pth", "custom.json"), "custom.json")

    def test_source_config_data_paths_follow_checkout(self):
        from experiments import source_evaluation as source
        args = source.build_parser().parse_args(["--checkpoint-path", "unused.pth"])
        runtime = source.build_source_eval_runtime_args(args, {
            "data_dir": "/workspace/datasets/swinjscc_hr",
            "hr_val_dirs": "/workspace/datasets/swinjscc_hr/Kodak",
        })
        self.assertEqual(runtime.data_dir, str(evaluation.ROOT / "datasets/swinjscc_hr"))
        self.assertEqual(runtime.hr_val_dirs, str(evaluation.ROOT / "datasets/swinjscc_hr/Kodak"))

    def test_source_explicit_data_selection_overrides_saved_validation_directory(self):
        from experiments import source_evaluation as source
        args = source.build_parser().parse_args([
            "--checkpoint-path", "unused.pth", "--data-dir", "/custom/images",
            "--hr-eval-split", "clic2021",
        ])
        runtime = source.build_source_eval_runtime_args(args, {
            "hr_val_dirs": "/workspace/datasets/swinjscc_hr/Kodak",
        })
        self.assertEqual(runtime.data_dir, "/custom/images")
        self.assertEqual(runtime.hr_val_dirs, "")
        self.assertEqual(runtime.hr_eval_split, "clic2021")

    def args(self, *extra):
        return evaluation.build_parser().parse_args([
            "grid", "--method", "decompjscc", "--eval-split", "kodak",
            "--output-dir", "/tmp/decompjscc-unittest-unused", *extra])

    def test_grid_dry_run_has_no_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            args = self.args("--output-dir", str(output), "--snr-db-values", "1,7")
            result = evaluation.execute(args)
            self.assertEqual(result["rates"], [32, 96, 192])
            self.assertEqual(result["snr_db_values"], [1, 7])
            self.assertEqual(result["noise_realizations"], 3)
            self.assertFalse(output.exists())

    def test_fixed_rate_and_custom_checkpoint_validation(self):
        for rate in (32, 96, 192):
            self.assertEqual(evaluation.grid_plan(self.args(
                "--method", "swinjscc_sa", "--rates", str(rate)))["rates"], [rate])
        for extra in (("--method", "swinjscc_sa", "--rates", "64"),
                      ("--checkpoint", "new.pth"),
                      ("--rates", "32,32"), ("--snr-db-values", "nan")):
            with self.assertRaises(ValueError):
                evaluation.grid_plan(self.args(*extra))

    def test_adopted_models_use_the_delivered_source_index(self):
        for method in evaluation.METHODS:
            for rate in ((32, 96, 192) if method in evaluation.FIXED_METHODS else (32,)):
                evidence = evaluation.adopted_evidence(method, rate)
                self.assertEqual(evidence["saved"]["identity"]["method"], method)
                self.assertEqual(len(evidence["checkpoint_sha256"]), 64)

    def test_aggregate_equal_image_weight_and_realizations(self):
        rows = [dict(image_id=image, realization=realization, psnr_db=value,
                     ssim=value / 100, ms_ssim=value / 100, cbr=32 / 1536)
                for realization, values in enumerate(((10, 20), (20, 30), (30, 40)))
                for image, value in zip(("a.png", "b.png"), values)]
        mean, realizations = evaluation.aggregate_images(rows)
        self.assertEqual(mean["psnr_db"], 25)
        self.assertEqual(mean["num_images"], 2)
        self.assertEqual(len(realizations), 3)
        with self.assertRaises(ValueError):
            evaluation.aggregate_images(rows[:-1])

    def test_measurement_order_and_seeds(self):
        args = self.args("--device", "cpu")
        image = torch.ones(1, 3, 4, 4)
        loader = [(image, ["a.png"]), (image, ["b.png"])]
        calls = []
        def forward(method, model, images, snr, rate):
            calls.append((snr, rate))
            return {"reconstructed": images, "cbr": rate / 1536}
        with patch.object(evaluation, "forward_model", forward), \
                patch.object(evaluation, "seed_randomness") as seed, \
                patch.object(evaluation, "compute_per_image_psnr", return_value=[30]), \
                patch.object(evaluation, "compute_ssim_batch", return_value=[.9]), \
                patch.object(evaluation, "compute_ms_ssim_batch", return_value=[.95]):
            rows = evaluation.measure_images(args, object(), loader, 32, 7)
        self.assertEqual([call.args[0] for call in seed.call_args_list], [20260811, 20260812, 20260813])
        self.assertEqual([(r["image_id"], r["realization"]) for r in rows],
                         [(name, repeat) for repeat in range(3) for name in ("a.png", "b.png")])
        self.assertEqual(calls, [(7, 32)] * 6)

    def test_neural_reconstruction_replays_preceding_images(self):
        loader = [(torch.ones(1, 3, 4, 4), [name]) for name in ("first.png", "target.png")]
        with patch.object(evaluation, "forward_model", side_effect=lambda *a: {
                "reconstructed": a[2], "cbr": 32 / 1536}) as forward, \
                patch.object(evaluation, "compute_per_image_psnr", return_value=[30]), \
                patch.object(evaluation, "compute_ms_ssim_batch", return_value=[.95]), \
                patch.object(evaluation, "seed_randomness") as seed:
            _, scores, count = reconstruction.replay_to_image(
                "decompjscc", object(), loader,
                {"image_id": "target.png", "rate": 32, "snr_db": 4}, torch.device("cpu"))
        self.assertEqual(forward.call_count, 2)
        seed.assert_called_once_with(20260811)
        self.assertEqual(count, 2)
        self.assertEqual(scores["psnr_db"], 30)

    def test_metric_functions_match_history_despite_checkpoint_path_changes(self):
        fixture = json.loads((evaluation.ROOT / 'tests/fixtures/metric_functions.json').read_text())
        self.assertEqual(fixture['historical_file_sha256'], reconstruction.HISTORICAL_METRIC_SHA)
        tree = ast.parse((evaluation.ROOT / 'decompjscc_utils.py').read_text())
        nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        nodes.update({'import:' + node.names[0].name: node for node in tree.body
                      if isinstance(node, ast.Import)})
        nodes['ssim_fn_import'] = next(node for node in tree.body if isinstance(node, ast.Try))
        for name, expected in fixture['nodes'].items():
            with self.subTest(node=name):
                node = nodes[name]
                for child in ast.walk(node):
                    if hasattr(child, 'type_params') and not child.type_params:
                        child._fields = tuple(field for field in child._fields if field != 'type_params')
                digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest, expected)

    def test_reconstruction_history_is_complete_without_exp_log(self):
        selection = json.loads((evaluation.ROOT / (
            "paper_data/updates/fig5_20261105/selection.json")).read_text())
        paths = set()
        for method in reconstruction.METHODS:
            for row in selection["rows"]:
                with self.subTest(method=method, panel=row["panel"]):
                    path = reconstruction.history_path(method, row)
                    self.assertTrue(path.is_relative_to(reconstruction.HISTORY_ROOT))
                    data = json.loads(path.read_text())
                    identity = data["identity"]
                    _, observation, returned_path = reconstruction.read_history(
                        method, row, identity["image_ids"], identity["image_sha256"])
                    self.assertEqual(returned_path, path)
                    self.assertEqual(observation["image_id"], row["image_id"])
                    paths.add(path)
        self.assertEqual(len(paths), 28)

    def test_bpg_defaults_are_the_adopted_protocol(self):
        args = evaluation.build_parser().parse_args(["bpg", "--output", "/tmp/unused.json"])
        self.assertEqual((args.encoder_mode, args.decoder_iterations), (8, 5))
        self.assertEqual(Path(args.mcs_calibration), evaluation.BPG_CALIBRATION)
        self.assertTrue(args.report_ms_ssim)
        self.assertTrue(args.cache_qp)
        self.assertEqual(evaluation.sha256_file(evaluation.BPG_CALIBRATION), evaluation.BPG_CALIBRATION_SHA)

    def test_bpg_failed_observations_remain_in_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            args = evaluation.bpg_parser().parse_args([
                "--device", "cpu", "--rates", "32", "--snr-db-values", "1",
                "--output", str(Path(temp) / "result.json")])
            codec = SimpleNamespace(metadata=lambda: {"encoder_mode": 8})
            qp = SimpleNamespace(bitstream=None, qp=None, channel_uses=0, trials=[])
            with patch.object(bpg, "BPGCodec", return_value=codec), \
                    patch.object(bpg, "search_bpg_qp", return_value=qp), \
                    patch.object(bpg, "compute_ssim_batch", return_value=torch.tensor([0.])), \
                    patch.object(bpg, "compute_ms_ssim_batch", return_value=torch.tensor([0.])), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = bpg.run(args, dataset=[(torch.ones(3, 128, 128), "a.png")])
            self.assertEqual(len(result["per_image_results"]), 3)
            self.assertTrue(all(row["budget_failure"] and row["decode_failure"]
                                for row in result["per_image_results"]))
            self.assertEqual(result["results"][0]["psnr_db"], 0)
            self.assertEqual(result["results"][0]["ms_ssim"], 0)

    def test_missing_or_wrong_checkpoint_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(evaluation.baseline, "ADJSCC") as factory:
            with self.assertRaises(FileNotFoundError):
                evaluation.build_model("adjscc", 32, checkpoint=Path(temp) / "missing.pth")
            with patch.object(evaluation, "sha256_file", return_value="example"), \
                    patch.object(torch, "load", return_value={"method": "snr_eq_jscc"}):
                with self.assertRaisesRegex(ValueError, "method mismatch"):
                    evaluation.build_model("adjscc", 32, checkpoint=Path(temp) / "wrong.pth")
            factory.assert_not_called()

    def test_grid_matches_frozen_loop_rng_and_metrics(self):
        reference_path = evaluation.ROOT / (
            "tests/fixtures/"
            "eval_decompjscc_paper_comparison_20260916.py")
        tree = ast.parse(reference_path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        class Dataset:
            def __init__(self, paths):
                self.image_files = paths
            def __len__(self):
                return len(self.image_files)
            def __getitem__(self, index):
                return torch.full((3, 8, 8), .25 + .25 * index), Path(self.image_files[index]).name
        def forward(method, model, images, snr, rate):
            noise = torch.rand_like(images) + float(np.random.rand()) + evaluation.random.random()
            return {"reconstructed": (images + noise / 20).clamp(0, 1), "cbr": rate / 1536}
        def score(reference, reconstruction, *unused):
            return reconstruction.mean().reshape(1)
        def builder(*unused, **kwargs):
            return SimpleNamespace(to=lambda device: model), {"checkpoint_sha256": "synthetic"}
        model = SimpleNamespace(to=lambda device: model)
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("a.png", "b.png")]
            for path in paths:
                Image.new("RGB", (8, 8)).save(path)
            dataset = Dataset([str(path) for path in paths])
            old_results, new_results = [], []
            args = self.args("--device", "cpu", "--rates", "32,96", "--snr-db-values", "1,7",
                             "--output-dir", str(Path(temp) / "old"))
            args.reuse_dir = None
            namespace = dict(vars(evaluation), __file__=str(reference_path),
                HRCroppedEvalImageDataset=lambda *a, **k: dataset,
                build_existing_model=builder, forward_model=forward,
                compute_ssim_batch=score, compute_ms_ssim_batch=score,
                write_json=lambda path, data: old_results.append(data) if "per_image_results" in data else None)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(reference_path), "exec"), namespace)
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["run"](args)
            args.output_dir = Path(temp) / "new"
            with patch.object(evaluation, "HRCroppedEvalImageDataset", return_value=dataset), \
                    patch.object(evaluation, "build_model", builder), \
                    patch.object(evaluation, "forward_model", forward), \
                    patch.object(evaluation, "compute_ssim_batch", score), \
                    patch.object(evaluation, "compute_ms_ssim_batch", score), \
                    patch.object(evaluation, "write_json", side_effect=lambda path, data:
                        new_results.append(data) if "per_image_results" in data else None), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.run_grid(args)
            self.assertEqual(len(new_results), 4)
            for old, new in zip(old_results, new_results):
                self.assertEqual(old["per_image_results"], new["per_image_results"])
                self.assertEqual(old["realization_results"], new["realization_results"])
                self.assertEqual(old["result"], new["result"])

    def test_bpg_success_crc_and_codec_failure_share_one_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            args = evaluation.bpg_parser().parse_args([
                "--device", "cpu", "--rates", "32", "--snr-db-values", "1",
                "--output", str(Path(temp) / "result.json")])
            codec = SimpleNamespace(metadata=lambda: {"encoder_mode": 8})
            qp = SimpleNamespace(bitstream=b"\0", qp=20, channel_uses=512, trials=[])
            links = [SimpleNamespace(decoded_bits=torch.zeros(8, dtype=torch.uint8), stats=SimpleNamespace(
                crc_pass=success, decode_success=success, channel_uses=512,
                num_code_blocks=1, crc_failed_blocks=int(not success))) for success in (True, False, True)]
            def score(reference, reconstruction, *unused):
                return reconstruction.mean().reshape(1)
            with patch.object(bpg, "BPGCodec", return_value=codec), \
                    patch.object(bpg, "search_bpg_qp", return_value=qp), \
                    patch.object(bpg, "run_5g_ldpc_awgn", side_effect=links), \
                    patch.object(bpg, "compute_ssim_batch", score), \
                    patch.object(bpg, "compute_ms_ssim_batch", score), \
                    patch.object(codec, "decode", create=True,
                                 side_effect=[Image.new("RGB", (128, 128), (128, 128, 128)), ValueError("bad codec")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = bpg.run(args, dataset=[(torch.ones(3, 128, 128), "a.png")])
            rows = result["per_image_results"]
            self.assertEqual([row["decode_failure"] for row in rows], [False, True, True])
            self.assertEqual([row["crc_failure"] for row in rows], [False, True, False])
            self.assertEqual([row["codec_failure"] for row in rows], [False, False, True])
            self.assertEqual(result["results"][0]["decode_failure_rate"], 2 / 3)
            self.assertEqual(result["results"][0]["psnr_db"], rows[0]["psnr_db"] / 3)
            self.assertEqual(result["results"][0]["ms_ssim"], rows[0]["ms_ssim"] / 3)


if __name__ == "__main__":
    unittest.main()
