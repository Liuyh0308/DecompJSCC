"""Model, checkpoint and numerical contracts shared by the public entry points."""

from __future__ import annotations

import gc
import math
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models.swinjscc_source_wrapper import (
    DecompJSCC,
    SourceAllStageCrossAttnResRCSBUniversalWrapper,
    build_source_swinjscc_args,
)
from models.stage_block_attnres import FinalStageBlockAttnRes, StageBlockAttnRes


class TestRootDecompJSCCNamingContract(unittest.TestCase):
    def test_old_class_names_are_compatibility_aliases(self):
        self.assertIs(FinalStageBlockAttnRes, StageBlockAttnRes)
        self.assertIs(SourceAllStageCrossAttnResRCSBUniversalWrapper, DecompJSCC)

    def test_decompjscc_keeps_historical_registered_module_prefixes(self):
        args = build_source_swinjscc_args(
            recipe_stage="no_modnet_sall_cross_universal",
            trainset="DIV2K",
            img_size=32,
            model_size="small",
            batch_size=1,
            embed_dims=[16, 24, 32, 40],
            depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
        )
        model = DecompJSCC(args, pass_channel=False)

        self.assertEqual(type(model).__name__, "DecompJSCC")
        self.assertEqual(
            {key.split(".", 1)[0] for key in model.state_dict()},
            {"net", "encoder_stage_attnres", "rcsb_rate_adapter", "decoder_snr_adapter"},
        )


class TestDecompJSCCPublicContract(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        from models.decompjscc import DecompJSCCConfig

        return DecompJSCCConfig(
            image_size=32,
            model_size="small",
            batch_size=1,
            embed_dims=(16, 24, 32, 40),
            encoder_depths=(1, 1, 1, 1),
            decoder_depths=(1, 1, 1, 1),
            num_heads=(1, 1, 1, 1),
        )

    @staticmethod
    def _legacy_tiny_model():
        args = build_source_swinjscc_args(
            recipe_stage="no_modnet_sall_cross_universal",
            trainset="DIV2K",
            img_size=32,
            model_size="small",
            batch_size=1,
            embed_dims=[16, 24, 32, 40],
            depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
        )
        return SourceAllStageCrossAttnResRCSBUniversalWrapper(
            args,
            pass_channel=False,
            gain_mode="active_rms",
            attnres_query_mode="snr_depth_logit_gate",
            attnres_condition_gate_mode="snr",
            attnres_condition_gate_use_reference=True,
        )

    def test_public_metadata_uses_paper_names_without_legacy_recipe_fields(self):
        config = self._tiny_config()
        metadata = config.to_public_metadata()

        self.assertEqual(metadata["method"], "DecompJSCC")
        self.assertEqual(metadata["components"]["rate_control"], "RateControlledSparseBottleneck")
        self.assertEqual(metadata["components"]["decoder_snr_adaptation"], "DecoderSNRAdapter")
        self.assertEqual(metadata["components"]["snr_conditioning"], "SNRDepthLogitConditioner")
        self.assertEqual(metadata["components"]["attention_residual"], "SNRConditionedAttentionResidual")
        self.assertEqual(metadata["components"]["gain_normalization"], "ActiveRMSGainNormalization")
        self.assertEqual(metadata["components"]["power_normalization"], "ChannelPowerNormalization")
        self.assertNotIn("recipe_stage", metadata)
        self.assertNotIn("rcsb_rate_gain_mode", metadata)
        self.assertNotIn("attnres_query_mode", metadata)

    def test_model_has_single_decompjscc_identity_and_paper_component_accessors(self):
        from models.decompjscc import DecoderSNRAdapter, DecompJSCC, RCSB, StageBlockAttnRes

        model = DecompJSCC(self._tiny_config(), pass_channel=False)

        self.assertEqual(type(model).__name__, "DecompJSCC")
        self.assertIs(model.attention_residuals, model.encoder_stage_attnres)
        self.assertIs(model.rcsb, model.rcsb_rate_adapter)
        self.assertIs(model.receiver_snr_adapter, model.decoder_snr_adapter)
        self.assertIsInstance(model.rcsb, RCSB)
        self.assertIsInstance(model.receiver_snr_adapter, DecoderSNRAdapter)
        self.assertTrue(all(isinstance(stage, StageBlockAttnRes) for stage in model.attention_residuals))
        self.assertTrue(all(stage.enable_cross_stage_attnres for stage in model.attention_residuals[2:]))
        self.assertEqual(model.rcsb.gain_mode, "active_rms")
        self.assertTrue(all(stage.use_snr_depth_logit_gate for stage in model.attention_residuals))

    def test_canonical_model_preserves_legacy_state_dict_keys(self):
        from models.decompjscc import DecompJSCC

        legacy = self._legacy_tiny_model()
        canonical = DecompJSCC(self._tiny_config(), pass_channel=False)

        self.assertEqual(set(legacy.state_dict()), set(canonical.state_dict()))
        canonical.load_state_dict(legacy.state_dict(), strict=True)

    def test_canonical_model_is_numerically_equivalent_to_legacy_model(self):
        from models.decompjscc import DecompJSCC

        torch.manual_seed(17)
        legacy = self._legacy_tiny_model().eval()
        canonical = DecompJSCC(self._tiny_config(), pass_channel=False).eval()
        canonical.load_state_dict(legacy.state_dict(), strict=True)
        image = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            legacy_output = legacy(image, snr_db=4.0, rate=32, return_aux=False)
            canonical_output = canonical(image, snr_db=4.0, rate=32, return_aux=False)

        self.assertTrue(torch.equal(legacy_output["reconstructed"], canonical_output["reconstructed"]))
        self.assertEqual(legacy_output["cbr"], canonical_output["cbr"])
        self.assertEqual(legacy_output["snr_db"], canonical_output["snr_db"])

    def test_canonical_model_accepts_fractional_snr_evaluation_points(self):
        from models.decompjscc import DecompJSCC, DecompJSCCConfig

        base = self._tiny_config()
        config = DecompJSCCConfig(
            image_size=base.image_size,
            model_size=base.model_size,
            batch_size=base.batch_size,
            embed_dims=base.embed_dims,
            encoder_depths=base.encoder_depths,
            decoder_depths=base.decoder_depths,
            num_heads=base.num_heads,
            snr_db_values=(2.5, 5.5),
        )
        model = DecompJSCC(config, pass_channel=False)

        self.assertEqual(model.net.multiple_snr, [2.5, 5.5])

    def test_checkpoint_loader_accepts_historical_checkpoint_without_rewriting_it(self):
        from models.decompjscc import DecompJSCC, load_decompjscc_checkpoint

        legacy = self._legacy_tiny_model()
        canonical = DecompJSCC(self._tiny_config(), pass_channel=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pth"
            torch.save({"model_state_dict": legacy.state_dict(), "epoch": 3}, checkpoint_path)
            result = load_decompjscc_checkpoint(canonical, checkpoint_path, strict=True)

        self.assertEqual(result["loaded"], len(legacy.state_dict()))
        self.assertEqual(result["skipped"], [])
        self.assertEqual(result["epoch"], 3)

    def test_canonical_checkpoints_strict_load_with_frozen_state_dict_schemas(self):
        from experiments import source_evaluation as source_eval
        from models.swinjscc_source_wrapper import load_source_checkpoint

        repo_root = Path(__file__).resolve().parents[1]
        contract = json.loads(
            (repo_root / "configs/checkpoint_contract.json").read_text(encoding="utf-8")
        )
        missing = [row["path"] for row in contract["checkpoints"].values()
                   if not (repo_root / row["path"]).is_file()]
        if missing:
            self.skipTest("Checkpoint integration requires downloaded weights: " + ", ".join(missing))
        parser = source_eval.build_parser()
        for experiment_id, expected in contract["checkpoints"].items():
            with self.subTest(experiment_id=experiment_id):
                checkpoint_path = repo_root / expected["path"]
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                cli_args = parser.parse_args(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--snr-db-values",
                        "7",
                        "--rates",
                        "96",
                        "--strict-checkpoint",
                    ]
                )
                runtime_args = source_eval.build_source_eval_runtime_args(cli_args, checkpoint.get("config", {}))
                model = source_eval.build_source_model(runtime_args)
                report = load_source_checkpoint(model, str(checkpoint_path), strict=True)

                self.assertEqual(report["loaded"], expected["state_dict_key_count"])
                self.assertEqual(
                    {key: tuple(value.shape) for key, value in model.state_dict().items()},
                    {key: tuple(value.shape) for key, value in checkpoint["model_state_dict"].items()},
                )
                del model, checkpoint
                gc.collect()

    def test_root_repository_has_required_asset_directories(self):
        from experiments.resources import inspect_directories

        repo_root = Path(__file__).resolve().parents[1]
        report = inspect_directories(repo_root)
        self.assertEqual(report["method"], "DecompJSCC")
        self.assertEqual(report["missing_directories"], [])

    def test_root_only_layout_keeps_configs_and_empty_asset_placeholders(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertFalse((repo_root / "decompjscc").exists())
        for directory in ("configs", "checkpoints", "datasets"):
            self.assertTrue((repo_root / directory).is_dir())

    def test_active_backend_utils_do_not_use_plugin_naming(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "single_task_plugin_utils.py").exists())
        self.assertTrue((repo_root / "swinjscc_utils.py").is_file())

        active_files = (
            "models/swinjscc_source_wrapper.py",
            "models/stage_block_attnres.py",
            "experiments/source_training.py",
            "experiments/source_evaluation.py",
            "experiments/mechanisms.py",
        )
        for relative_path in active_files:
            source = (repo_root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("single_task_plugin_utils", source)

    def test_directory_preparer_creates_only_local_asset_directories(self):
        from experiments.resources import prepare_directories

        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_directories(tmpdir)
            root = Path(tmpdir)
            self.assertEqual(result["created_directories"], ["datasets", "checkpoints"])
            self.assertEqual(result["missing_directories"], [])
            self.assertTrue(all((root / name).is_dir() for name in result["required_directories"]))
            self.assertEqual(list(root.rglob("*.pth")), [])

    def test_public_parsers_do_not_expose_historical_model_families(self):
        from experiments import complexity, evaluation, fixed_training, mechanisms

        forbidden = (
            "recipe-stage",
            "no_modnet",
            "sall_cross",
            "rate_first",
            "snr_first",
            "active_rms",
            "query-mode",
            "gain-mode",
        )
        parsers = (fixed_training.build_parser(), evaluation.build_parser(),
                   mechanisms.routing_parser(), complexity.profile_parser())
        for parser in parsers:
            help_text = parser.format_help().lower()
            for token in forbidden:
                self.assertNotIn(token, help_text)

    def test_public_entrypoints_replace_configuration_specific_wrappers(self):
        repo_root = Path(__file__).resolve().parents[1]
        for name in ("train.py", "eval.py", "analyze.py"):
            self.assertTrue((repo_root / name).is_file(), msg=name)
        for pattern in ("train_*.py", "run_*.py", "eval_*.py", "prepare_decompjscc_fig*.py"):
            self.assertEqual(list(repo_root.glob(pattern)), [], msg=pattern)

    def test_source_freeze_scopes_match_paper_training_stages(self):
        from experiments.source_training import configure_source_trainable_parameters
        from models.decompjscc import DecompJSCC

        scopes = {
            "rcsb_rate_adapter_only": ("rcsb_rate_adapter.",),
            "snr_attnres_adapter_only": ("encoder_stage_attnres.", "decoder_snr_adapter."),
        }
        for scope, prefixes in scopes.items():
            with self.subTest(scope=scope):
                model = DecompJSCC(self._tiny_config(), pass_channel=False)
                summary = configure_source_trainable_parameters(model, scope)
                self.assertGreater(summary["trainable"], 0)
                self.assertGreater(summary["frozen"], 0)
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.requires_grad, name.startswith(prefixes), msg=name)

        model = DecompJSCC(self._tiny_config(), pass_channel=False)
        flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        configure_source_trainable_parameters(model, "none")
        self.assertEqual(flags, {name: parameter.requires_grad for name, parameter in model.named_parameters()})

    def test_paper_scripts_default_to_canonical_main_checkpoint(self):
        from experiments import mechanisms as routing, evaluation as evaluate, complexity as profile

        expected_checkpoint = "checkpoints/decompjscc/awgn/model_best.pth"
        eval_args = evaluate.build_parser().parse_args([
            "grid", "--method", "decompjscc", "--eval-split", "clic2021",
            "--output-dir", "/tmp/model-contract-evaluation",
        ])
        routing_args = routing.routing_parser().parse_args([])
        profile_args = profile.profile_parser().parse_args([])

        self.assertIsNone(eval_args.checkpoint)
        self.assertEqual(evaluate.grid_plan(eval_args)["checkpoint_origin"], "adopted_paper")
        self.assertTrue(routing_args.checkpoint.endswith(expected_checkpoint))
        self.assertTrue(profile_args.checkpoint.endswith(expected_checkpoint))
        self.assertEqual(evaluate.parse_rates(eval_args.rates), [32, 96, 192])
        self.assertEqual(evaluate.parse_snr_values(eval_args.snr_db_values),
                         [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0])
        self.assertEqual(routing.parse_interventions(routing_args.interventions), ["none", "snr_conditioning_off"])
        self.assertEqual(profile.parse_rates(profile_args.rates), [32, 96, 192])

    def test_complexity_profiler_exposes_only_frozen_paper_roles(self):
        from experiments import complexity as profile

        parser = profile.profile_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertEqual(
            tuple(actions["role"].choices),
            ("decompjscc", "swinjscc_universal"),
        )
        args = parser.parse_args(["--role", "swinjscc_universal"])
        checkpoint, config_path = profile.resolve_role_paths(args)
        self.assertTrue(str(checkpoint).endswith("checkpoints/swinjscc_joint/awgn/model_best.pth"))
        self.assertTrue(str(config_path).endswith("configs/models/swinjscc_joint.json"))

    def test_complexity_summary_enforces_frozen_protocol(self):
        from experiments import complexity as summary

        payload = {
            **summary.EXPECTED_PROTOCOL,
            "role": "decompjscc",
            "device_name": "NVIDIA GeForce RTX 4090",
            "material_passport": {"verification_status": "MEASURED"},
            "checkpoint_load": {"loaded": 461, "skipped": []},
            "total_parameters": 24_320_382,
            "state_size_mb": 247.9,
            "checkpoint_sha256": "a" * 64,
            "profiles": [
                {
                    "rate": rate,
                    "latency_median_ms": 13.0,
                    "latency_p95_ms": 14.0,
                    "peak_memory_mb": 248.0,
                }
                for rate in summary.ROLE_RATES["decompjscc"]
            ],
        }
        rows = summary.validate_profile_payload(payload, "decompjscc")
        self.assertEqual([row["rate"] for row in rows], [32, 96, 192])

        payload["warmup"] = 19
        with self.assertRaisesRegex(ValueError, "warmup"):
            summary.validate_profile_payload(payload, "decompjscc")

    def test_rcsb_audit_support_metrics_detect_non_nested_masks(self):
        from experiments import mechanisms as audit

        smaller = torch.tensor([1, 1, 0, 0], dtype=torch.bool).numpy()
        larger = torch.tensor([0, 1, 1, 1], dtype=torch.bool).numpy()
        comparison = audit.support_comparison(smaller, larger)
        self.assertFalse(comparison["is_nested"])
        self.assertEqual(comparison["missing_from_larger"], 1)
        self.assertAlmostEqual(comparison["jaccard"], 0.25)

        diversity = audit.mask_diversity([smaller, larger])
        self.assertEqual(diversity["unique_masks"], 2)
        self.assertEqual(diversity["varying_channels"], 3)
        self.assertAlmostEqual(audit.support_metadata_bits(4, 2), math.log2(6.0))

    def test_rayleigh_channel_metadata_identifies_shared_source_protocol(self):
        from dataclasses import replace

        from decompjscc_utils import channel_implementation_metadata
        from models.decompjscc import DecompJSCC

        model = DecompJSCC(replace(self._tiny_config(), channel_type="rayleigh"), pass_channel=False)
        metadata = channel_implementation_metadata(model)
        self.assertEqual(metadata["source_path"], "SwinJSCC_base/channel.py")
        self.assertEqual(metadata["channel_type"], "rayleigh")
        self.assertFalse(metadata["rayleigh_protocol"]["random_phase"])
        self.assertFalse(metadata["rayleigh_protocol"]["receiver_csi"])
        self.assertEqual(metadata["rayleigh_protocol"]["equalization"], "none")

    def test_training_state_resume_restores_optimizer_and_rng(self):
        import random

        import numpy as np

        from decompjscc_utils import capture_training_rng_state, load_training_state

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=3e-5)
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        rng_state = capture_training_rng_state()
        expected_random = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = float(torch.rand(()))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "last.pth"
            torch.save(
                {
                    "epoch": 7,
                    "best_psnr": 31.25,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": rng_state,
                },
                checkpoint_path,
            )
            restored_parameter = torch.nn.Parameter(torch.tensor([0.0]))
            restored_optimizer = torch.optim.Adam([restored_parameter], lr=1e-4)
            result = load_training_state(checkpoint_path, restored_optimizer, torch.device("cpu"))

        self.assertEqual(result, {"start_epoch": 7, "best_psnr": 31.25})
        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], 3e-5)
        self.assertAlmostEqual(random.random(), expected_random)
        self.assertAlmostEqual(float(np.random.random()), expected_numpy)
        self.assertAlmostEqual(float(torch.rand(())), expected_torch)

    def test_training_state_resume_rejects_weight_only_checkpoint(self):
        from decompjscc_utils import load_training_state

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=3e-5)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "weight_only.pth"
            torch.save({"epoch": 7, "model_state_dict": {}}, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "optimizer_state_dict"):
                load_training_state(checkpoint_path, optimizer, torch.device("cpu"))

    def test_source_checkpoint_loader_accepts_full_rng_metadata(self):
        import numpy as np

        from models.swinjscc_source_wrapper import load_source_checkpoint

        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "full_state.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "rng_state": {"numpy": np.random.get_state()},
                },
                checkpoint_path,
            )
            report = load_source_checkpoint(model, str(checkpoint_path), strict=True)

        self.assertEqual(report, {"loaded": 2, "skipped": []})

    def test_gradient_accumulation_reuses_one_condition_per_optimizer_step(self):
        from experiments import source_training as training

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.conditions = []

            def forward(self, images, snr_db, rate):
                self.conditions.append((snr_db, rate))
                loss = (self.weight * images.mean()).square()
                return {"loss": loss, "mse": loss.detach(), "cbr": 0.1}

        class SingleProcessContext:
            device = torch.device("cpu")
            mode = SimpleNamespace(value="single")

            @staticmethod
            def set_epoch(loader, epoch):
                return None

            @staticmethod
            def reduce_mean(value):
                return value

        model = RecordingModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = [(torch.ones(1, 1, 1, 1), None) for _ in range(4)]
        runtime_args = SimpleNamespace(max_train_batches=2, gradient_accumulation_steps=2)
        config = SimpleNamespace(multiple_snr="1,4", C="32,64")
        with mock.patch.object(training, "sample_source_condition", side_effect=[(1.0, 32), (4.0, 64)]) as sampler:
            training.train_epoch(0, model, loader, optimizer, config, runtime_args, SingleProcessContext())

        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(model.conditions, [(1.0, 32), (1.0, 32), (4.0, 64), (4.0, 64)])
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 2)

    def test_evaluator_computes_psnr_per_image(self):
        from decompjscc_utils import compute_per_image_psnr

        reference = torch.zeros(2, 3, 2, 2)
        reconstructed = reference.clone()
        reconstructed[0] = 1.0 / 255.0
        reconstructed[1] = 2.0 / 255.0
        values = compute_per_image_psnr(reference, reconstructed)

        self.assertEqual(tuple(values.shape), (2,))
        self.assertAlmostEqual(float(values[0]), 10.0 * torch.log10(torch.tensor(255.0**2)).item(), places=5)
        self.assertLess(float(values[1]), float(values[0]))

    def test_routing_payload_translates_backend_scalar_names(self):
        from experiments.mechanisms import _public_result

        public = _public_result(
            {
                "ablation_mode": "snr_logit_off",
                "aux_scalars": {
                    "attnres_stage0_attnres_entropy": 0.7,
                    "decoder_snr_gain_mean": 1.0,
                    "rcsb_active_rms": 0.9,
                },
            },
            "snr_conditioning_off",
        )

        self.assertNotIn("ablation_mode", public)
        self.assertEqual(public["intervention"], "snr_conditioning_off")
        self.assertIn("attention_residual_stage0_attention_residual_entropy", public["aux_scalars"])
        self.assertIn("decoder_snr_adapter_gain_mean", public["aux_scalars"])
        self.assertIn("rcsb_active_latent_rms", public["aux_scalars"])


if __name__ == "__main__":
    unittest.main()
