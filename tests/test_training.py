"""CPU contracts for parameterized fixed-CBR training and cumulative-best resume."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train
from experiments import fixed_training as training, source_training


class ParameterizedTrainingTests(unittest.TestCase):
    def test_width_and_output_are_parameters_not_model_variants(self):
        for rate, denominator in ((32, 48), (64, 24), (96, 16), (128, 12), (192, 8)):
            args = training.runtime_args(training.METHODS[0], rate, 400, "/tmp/new-project", "repeat")
            self.assertEqual(args.C, str(rate))
            self.assertEqual(args.epochs, 400)
            self.assertIn(f"/cbr_1_{denominator}/repeat", args.checkpoint_path)
            self.assertTrue(args.checkpoint_path.startswith("/tmp/new-project/checkpoints/"))
            self.assertEqual((args.batch_size, args.gradient_accumulation_steps), (8, 2))
        self.assertEqual(training.runtime_args(training.METHODS[0]).C, "32")

    def test_one_entrypoint_dispatches_without_changing_arguments(self):
        self.assertEqual(train.BACKENDS, ("source", "fixed", "adjscc", "snr_eq_jscc", "sa_ra_jscc"))
        for name, module in (("fixed", training), ("source", source_training)):
            with self.subTest(name=name), mock.patch.object(module, "main") as run:
                train.main([name, "--epochs", "400"])
                run.assert_called_once_with(["--epochs", "400"])
        with mock.patch("external_baselines.unified_runtime.run_training") as run:
            train.main(["snr_eq_jscc", "--rate", "96", "--dry-run"])
            run.assert_called_once_with("snr_eq_jscc", ["--rate", "96", "--dry-run"])

    def test_fixed_initialization_uses_one_public_resource_path(self):
        from external_baselines.shared import OFFICIAL_FIXED_CHECKPOINT

        expected = Path("checkpoints/pretraining/swinjscc_fixed/model_best.pth")
        self.assertEqual(OFFICIAL_FIXED_CHECKPOINT, expected)
        self.assertEqual(training.INITIAL_CHECKPOINT, training.ROOT / expected)

    def test_source_dry_run_does_not_create_training_context(self):
        with mock.patch.object(source_training, "create_dist_context") as context:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                train.main(["source", "--recipe-stage", "rate_adaptive", "--dry-run"])
            config = json.loads(output.getvalue())
            self.assertEqual(config["runtime"]["recipe_stage"], "rate_adaptive")
            self.assertEqual(config["source_args"]["model"], "SwinJSCC_w/_RA")
            context.assert_not_called()

    def test_fixed_c96_dry_run_needs_no_special_script(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train.main(["fixed", "--method", "decompjscc_wo_rcsb", "--rate", "96"])
        config = json.loads(output.getvalue())
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["runtime"]["C"], "96")

    def test_invalid_output_and_resume_arguments_fail_before_gpu_checks(self):
        with self.assertRaises(ValueError):
            training.output_dirs(training.METHODS[0], run_name="../old")
        with self.assertRaises(ValueError):
            training.runtime_args(training.METHODS[0], epochs=0)
        with mock.patch.object(training, "check_idle_gpu") as gpu:
            with self.assertRaisesRegex(ValueError, "both"):
                training.train(training.METHODS[0], resume_checkpoint="last.pth")
            gpu.assert_not_called()


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.method = training.METHODS[0]
        self.args = training.runtime_args(self.method, rate=96, epochs=400)
        self.model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        for parameter in self.model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(300 * 166.), "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        self.audit = dict(training.training_protocol(rate=96), architecture={"fixed_channels": 96},
                          training_images=[], validation_images=[], torch_version=torch.__version__)
        self.rng = training.capture_training_rng_state()
        self.rng["cuda"] = [torch.random.get_rng_state().clone()]
        self.last = {
            "epoch": 300, "best_psnr": 30., "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": self.rng, "model_state_dict": self.model.state_dict(),
            "config": vars(self.args), "training_metadata": self.audit,
            "method": self.method, "val_metrics": {"psnr": 29.},
        }
        self.best = copy.deepcopy(self.last)
        self.best.update(epoch=297, val_metrics={"psnr": 30.})
        self.save()

    def save(self):
        for role in ("last", "best"):
            torch.save(getattr(self, role), self.root / f"{role}.pth")

    def load(self):
        return training.load_parent_state(self.method, self.args, self.audit,
                                          self.root / "last.pth", self.root / "best.pth")

    def test_parent_identity_optimizer_and_cumulative_best(self):
        parent = self.load()
        self.assertEqual((parent["audit"]["start_epoch"], parent["audit"]["best_epoch"]), (300, 297))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        training.restore_parent_optimizer(optimizer, parent["last"]["optimizer_state_dict"])
        for state in optimizer.state.values():
            self.assertEqual(int(state["step"]), 49800)
            self.assertEqual(state["step"].device.type, "cpu")

    def test_rejects_rate_best_and_optimizer_mismatches(self):
        original = copy.deepcopy(self.last)
        mutations = (
            lambda: self.last["config"].update(C="32"),
            lambda: self.last.update(best_psnr=31.),
            lambda: self.last["optimizer_state_dict"]["state"][0].update(step=torch.tensor(1.)),
            lambda: self.last["rng_state"].update(cuda=None),
            lambda: self.last.update(epoch=300.5),
            lambda: self.last["config"].update(attnres_cross_stage_force_skip=True),
        )
        for mutate in mutations:
            self.last = copy.deepcopy(original)
            mutate()
            self.save()
            with self.assertRaises(ValueError):
                self.load()

    def test_training_loop_preserves_parent_best_without_improvement(self):
        paths_root = self.root / "new_run"
        args = training.runtime_args(self.method, 96, 302, paths_root)
        source_args = training.source.build_source_args_from_runtime(args)
        ctx = mock.Mock(device=torch.device("cuda:0"))
        ctx.get_dataloader.side_effect = [list(range(332)), list(range(24))]
        audit = copy.deepcopy(self.audit)
        audit.update(epochs=302, rates=[96])
        parent_best_sha = training.sha256_file(self.root / "best.pth")
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(mock.patch.object(training, "preflight", return_value=(
                args, source_args, self.model, [], [], audit)))
            stack.enter_context(mock.patch.object(self.model, "to", return_value=self.model))
            stack.enter_context(mock.patch.object(training, "create_dist_context", return_value=ctx))
            stack.enter_context(mock.patch.object(training, "check_idle_gpu"))
            stack.enter_context(mock.patch("experiments.evaluation.require_idle_gpu"))
            stack.enter_context(mock.patch.object(training, "smoke_batch", return_value={}))
            stack.enter_context(mock.patch.object(training, "verify_serialized_state", return_value={}))
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_name", return_value="mocked"))
            stack.enter_context(mock.patch.object(training, "SummaryWriter"))
            restore = stack.enter_context(mock.patch.object(training, "restore_training_rng_state"))
            stack.enter_context(mock.patch.object(training, "capture_training_rng_state", return_value=self.rng))
            loop = stack.enter_context(mock.patch.object(training.source, "train_epoch", return_value={"loss": 1.}))
            stack.enter_context(mock.patch.object(training.source, "validate", side_effect=[{"psnr": 29.}, {"psnr": 30.}]))
            stack.enter_context(mock.patch.object(training.source, "reset_source_train_resolution"))
            training.train(self.method, rate=96, epochs=302, output_root=paths_root,
                           resume_checkpoint=self.root / "last.pth", resume_best_checkpoint=self.root / "best.pth")
        self.assertEqual([call.args[0] for call in loop.call_args_list], [300, 301])
        restore.assert_called_once()
        ctx.cleanup.assert_called_once()
        paths = training.output_dirs(self.method, 96, paths_root)
        self.assertEqual(training.sha256_file(paths["checkpoint"] / "model_best.pth"), parent_best_sha)
        selected = json.loads((paths["experiment"] / "selection.json").read_text())
        self.assertEqual(selected["best_epoch"], 297)
        self.assertTrue(selected["inherited_parent_best"])
        self.assertEqual(torch.load(paths["checkpoint"] / "model_last.pth", weights_only=False)["epoch"], 302)


if __name__ == "__main__":
    unittest.main()
