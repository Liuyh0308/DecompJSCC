"""Read-only validation of external weights and adopted result identities."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.resources import download_weights, inspect_paper_resources, resource_path, weight_records


class PaperResourcesTest(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "configs/paper_resources.json"
        manifest.parent.mkdir(parents=True)
        payload = b"test weight bytes, not a model"
        (root / "best.pth").write_bytes(payload)
        manifest.write_text(json.dumps({"weights": [{
            "role": "adopted",
            "checkpoint": "best.pth", "sha256": hashlib.sha256(payload).hexdigest(),
        }]}))
        return manifest

    def download_fixture(self, root):
        manifest = self.fixture(root)
        data = json.loads(manifest.read_text())
        data["model_repository"] = {
            "repo_id": "test/decompjscc", "repo_type": "model", "revision": "1" * 40,
        }
        data["weights"][0].update({
            "checkpoint": "checkpoints/decompjscc/awgn/model_best.pth",
            "hf_file": "weights/decompjscc_awgn_best.pth",
        })
        manifest.write_text(json.dumps(data))
        return manifest, root / "best.pth"

    def test_correct_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = inspect_paper_resources(root, verify_hashes=True)
            self.assertEqual(result["checkpoint_count"], 1)
            self.assertEqual(result["failures"], [])

    def test_missing_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "best.pth").unlink()
            self.assertEqual(inspect_paper_resources(root)["failures"][0]["reason"], "missing")

    def test_replaced_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "best.pth").write_bytes(b"different")
            self.assertEqual(inspect_paper_resources(root, verify_hashes=True)["failures"][0]["reason"], "sha256_mismatch")

    def test_manifest_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest.write_text(json.dumps({"weights": [{"role": "adopted", "checkpoint": "../best.pth"}]}))
            with self.assertRaises(ValueError):
                inspect_paper_resources(root)

    def test_historical_resource_uses_only_explicit_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            data["aliases"] = {"/workspace/checkpoints/old/best.pth": "checkpoints/new/best.pth"}
            manifest.write_text(json.dumps(data))
            self.assertEqual(resource_path("/workspace/checkpoints/old/best.pth", root),
                             root / "checkpoints/new/best.pth")
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                resource_path("/workspace/checkpoints/unregistered.pth", root)

    def test_dataset_prefix_mapping_preserves_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            data["path_prefix_aliases"] = {"/workspace/datasets/swinjscc_hr": "datasets/swinjscc_hr"}
            manifest.write_text(json.dumps(data))
            self.assertEqual(resource_path("/workspace/datasets/swinjscc_hr/CLIC2021/test", root),
                             root / "datasets/swinjscc_hr/CLIC2021/test")
            with self.assertRaisesRegex(ValueError, "within"):
                resource_path("/workspace/datasets/swinjscc_hr/../../secret", root)

    def test_download_selection_includes_claims_but_not_training_by_default(self):
        manifest = {"weights": [{"role": role} for role in
                    ("adopted", "claim_ablation", "five_rate_model_bank", "initialization")],
                    "history_repository": {"weights": [{"role": "optional_resume_parent"}]}}
        self.assertEqual([row["role"] for row in weight_records(manifest)],
                         ["adopted", "claim_ablation", "five_rate_model_bank"])
        self.assertEqual([row["role"] for row in weight_records(manifest, include_pretraining=True)],
                         ["adopted", "claim_ablation", "five_rate_model_bank", "initialization"])

    def test_published_download_selection_excludes_archived_parents(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/paper_resources.json").read_text())
        self.assertEqual(len(weight_records(manifest)), 20)
        self.assertEqual(len(weight_records(manifest, include_pretraining=True)), 26)
        self.assertEqual(len(manifest["weights"]), 26)
        self.assertEqual(sum(row["role"] == "initialization" for row in manifest["weights"]), 6)
        for record in manifest["weights"]:
            with self.subTest(checkpoint=record["checkpoint"]):
                self.assertNotEqual(record["role"], "optional_resume_parent")
                self.assertFalse(record["hf_file"].startswith("resume/"))

    def test_archived_parents_retain_local_aliases_and_explicit_configs(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/paper_resources.json").read_text())
        archived = manifest["history_repository"].get("weights", [])
        self.assertEqual(len(archived), 4)
        active_ids = {row["id"] for row in manifest["weights"]}
        self.assertTrue(active_ids.isdisjoint(row["id"] for row in archived))
        for record in archived:
            with self.subTest(checkpoint=record["checkpoint"]):
                self.assertEqual(record["role"], "optional_resume_parent")
                self.assertTrue(record["former_paper_hf_file"].startswith("resume/"))
                self.assertNotEqual(record["hf_file"], record["former_paper_hf_file"])
                self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(record["bytes"], 0)
                self.assertEqual(record["historical_path"], "checkpoints/" + record["source_hf_file"])
                for historical in (record["historical_path"], "/workspace/" + record["historical_path"]):
                    self.assertEqual(resource_path(historical, root), root / record["checkpoint"])
                config = root / record["config"]
                self.assertTrue(config.is_file())
                self.assertEqual(hashlib.sha256(config.read_bytes()).hexdigest(), record["config_sha256"])

    def test_download_uses_remote_name_and_revision_with_original_local_path(self):
        for revision, expected_revision in ((None, "1" * 40), ("2" * 40, "2" * 40)):
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, cached = self.download_fixture(root)
                with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as fetch:
                    result = download_weights(root, revision=revision)
                fetch.assert_called_once_with(
                    repo_id="test/decompjscc", repo_type="model",
                    filename="weights/decompjscc_awgn_best.pth", revision=expected_revision,
                )
                self.assertEqual(result, {
                    "downloaded": ["checkpoints/decompjscc/awgn/model_best.pth"],
                    "already_present": [],
                })
                destination = root / "checkpoints/decompjscc/awgn/model_best.pth"
                self.assertEqual(destination.read_bytes(), b"test weight bytes, not a model")
                self.assertFalse(destination.is_symlink())
                self.assertFalse((root / "weights").exists())

    def test_download_include_pretraining_excludes_archived_parents(self):
        roles = ("adopted", "claim_ablation", "five_rate_model_bank", "initialization")
        for include_pretraining, expected_roles in ((False, roles[:3]), (True, roles)):
            with self.subTest(include_pretraining=include_pretraining), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, cached = self.download_fixture(root)
                data = json.loads(manifest.read_text())
                base = data["weights"][0]
                data["weights"] = [dict(base, role=role, checkpoint=f"checkpoints/{role}.pth",
                                        hf_file=f"weights/{role}.pth") for role in roles]
                data["history_repository"] = {
                    "repo_id": "test/history", "repo_type": "model", "revision": "3" * 40,
                    "weights": [dict(base, role="optional_resume_parent",
                                     checkpoint="checkpoints/optional_resume_parent.pth",
                                     hf_file="archive/parent.pth",
                                     former_paper_hf_file="resume/parent.pth")],
                }
                manifest.write_text(json.dumps(data))
                with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as fetch:
                    result = download_weights(root, include_pretraining=include_pretraining)
                expected = [f"checkpoints/{role}.pth" for role in expected_roles]
                self.assertEqual(result, {"downloaded": expected, "already_present": []})
                self.assertEqual(fetch.call_count, len(expected))
                self.assertEqual([call.kwargs["filename"] for call in fetch.call_args_list],
                                 [f"weights/{role}.pth" for role in expected_roles])
                self.assertTrue(all(call.kwargs["repo_id"] == "test/decompjscc"
                                    for call in fetch.call_args_list))
                self.assertEqual(sorted(str(path.relative_to(root)) for path in
                                        (root / "checkpoints").glob("*.pth")), sorted(expected))
                self.assertFalse((root / "checkpoints/optional_resume_parent.pth").exists())
                for relative in expected:
                    self.assertEqual((root / relative).read_bytes(), b"test weight bytes, not a model")

    def test_download_rejects_corrupt_cache_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cached = self.download_fixture(root)
            cached.write_bytes(b"corrupt download")
            with patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                with self.assertRaisesRegex(ValueError, "Downloaded checkpoint SHA differs"):
                    download_weights(root)
            self.assertFalse((root / "checkpoints/decompjscc/awgn/model_best.pth").exists())

    def test_download_preserves_existing_checkpoints_without_fetching(self):
        for matching in (True, False):
            with self.subTest(matching=matching), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, cached = self.download_fixture(root)
                destination = root / "checkpoints/decompjscc/awgn/model_best.pth"
                destination.parent.mkdir(parents=True)
                existing = cached.read_bytes() if matching else b"existing different weight"
                destination.write_bytes(existing)
                with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as fetch:
                    if matching:
                        self.assertEqual(download_weights(root), {
                            "downloaded": [],
                            "already_present": ["checkpoints/decompjscc/awgn/model_best.pth"],
                        })
                    else:
                        with self.assertRaisesRegex(FileExistsError, "Existing checkpoint SHA differs"):
                            download_weights(root)
                fetch.assert_not_called()
                self.assertEqual(destination.read_bytes(), existing)


if __name__ == "__main__":
    unittest.main()
