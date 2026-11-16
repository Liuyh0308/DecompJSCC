"""Analysis contracts retained after entry-point consolidation."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import complexity, mechanisms


class AnalysisTest(unittest.TestCase):
    def test_rcsb_audit_only_exposes_adopted_main_method(self):
        self.assertEqual(tuple(mechanisms.ROLE_SPECS), ("decompjscc",))
        actions = {action.dest for action in mechanisms.rcsb_parser()._actions}
        self.assertIn("main_checkpoint", actions)
        self.assertNotIn("controller_checkpoint", actions)
        self.assertEqual(
            Path(mechanisms.rcsb_parser().parse_args([]).main_checkpoint),
            mechanisms.MAIN_CHECKPOINT,
        )

    def test_rcsb_report_does_not_require_controller_results(self):
        summary = mechanisms._summarize([], [{
            "role": "decompjscc", "adjacent_rates": True, "is_nested": True,
            "missing_from_larger": 0, "smaller_retention": 1.0,
        }], [])
        self.assertNotIn("main_controller_support_overlap", summary)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            mechanisms._write_markdown(output, {
                "protocol": {"latent_channels": 192}, "summary": summary,
            })
            self.assertIn("DecompJSCC RCSB", output.read_text())
            self.assertNotIn("controller", output.read_text())

    def test_weights_average_batch_and_tokens(self):
        weights = torch.tensor([[[0.2, 0.4]], [[0.8, 0.6]]])
        result = mechanisms.summarize_weight_tensor(weights, "stage4_output")
        self.assertAlmostEqual(result["source_weights"][0], 0.3)
        self.assertAlmostEqual(result["depth_center"], 0.7)

    def test_profile_uses_selected_rate(self):
        wrapped = complexity.ComparisonForward("decompjscc", torch.nn.Identity(), rate=96)
        images = torch.ones(1, 3, 8, 8)
        with patch.object(complexity, "forward_model", return_value=images) as forward:
            wrapped(images)
        self.assertEqual(forward.call_args.args[4], 96)

    def test_profile_can_override_rate(self):
        wrapped = complexity.ComparisonForward("decompjscc", torch.nn.Identity(), rate=96)
        with patch.object(complexity, "forward_model", return_value=None) as forward:
            wrapped(torch.ones(1, 3, 8, 8), rate=32)
        self.assertEqual(forward.call_args.args[4], 32)

    def test_merged_modules_have_no_shadowed_definitions(self):
        root = Path(__file__).resolve().parents[1] / "experiments"
        for name in ("complexity", "mechanisms", "statistics"):
            tree = ast.parse((root / f"{name}.py").read_text())
            names = [node.name for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            self.assertEqual(len(names), len(set(names)), name)


if __name__ == "__main__":
    unittest.main()
