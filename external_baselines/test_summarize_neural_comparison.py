import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from external_baselines import summarize_neural_comparison as summary


class NeuralComparisonGroupTest(unittest.TestCase):
    def test_only_current_joint_group_is_available(self):
        groups = summary._group_specs()
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["name"], "joint_5x5")
        self.assertEqual(group["reference"], "DecompJSCC")
        self.assertEqual(group["rates"], (32, 64, 96, 128, 192))
        self.assertEqual(group["methods"], (
            "DecompJSCC", "Official_SwinJSCC_Universal", "SA_RA_JSCC"))
        self.assertEqual(group["bootstrap_seed_offset"], 10000)

    def test_six_inputs_are_current_paper_evidence(self):
        paths = [path for split in ("kodak", "clic2021")
                 for files in summary._input_paths(split).values() for path in files]
        self.assertEqual(len(paths), 6)
        expected = summary.REPO_ROOT / "paper_data/evidence/claims/joint_grid"
        self.assertTrue(all(path.parent == expected and path.is_file() for path in paths))

    def test_input_directory_is_overridable_and_output_is_separate(self):
        args = summary.build_parser().parse_args(["--input-dir", "/tmp/new-grid"])
        self.assertTrue(all(path.parent == Path("/tmp/new-grid")
                            for files in summary._input_paths("kodak", args.input_dir).values()
                            for path in files))
        self.assertEqual(Path(args.output_dir), summary.REPO_ROOT / "outputs/neural_comparison")

    def test_command_retains_historical_joint_bootstrap_seed_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("kodak", "clic2021"):
                for files in summary._input_paths(split, root).values():
                    files[0].write_text(json.dumps({"eval_split": split, "per_image_results": []}))
            with patch("sys.argv", ["summary", "--input-dir", str(root),
                                    "--output-dir", str(root / "new-output")]), \
                    patch.object(summary, "_average_realizations", return_value={}), \
                    patch.object(summary, "summarize_group", return_value=([], [], [])) as summarize, \
                    patch.object(summary, "_write_csv"), contextlib.redirect_stdout(io.StringIO()):
                summary.main()
            self.assertEqual([call.kwargs["bootstrap_seed"] for call in summarize.call_args_list],
                             [20260830 + 10000, 20260830 + 100000 + 10000])

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["summary", "--output-dir", directory]), \
                    patch.object(summary, "_input_paths") as inputs:
                with self.assertRaises(FileExistsError):
                    summary.main()
                inputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
