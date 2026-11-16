"""Prepare or validate the root project's DecompJSCC data and output directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REQUIRED_DIRS = ("datasets", "checkpoints")
MANIFEST = "configs/paper_resources.json"


def load_manifest(repo_root: str | Path) -> dict:
    return json.loads((Path(repo_root) / MANIFEST).read_text())


def resource_path(value: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve explicit aliases without falling back to historical server files."""
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    value = str(value)
    manifest = load_manifest(root)
    mapped = manifest.get("aliases", {}).get(value)
    if mapped is None:
        for source, destination in manifest.get("path_prefix_aliases", {}).items():
            if Path(value).is_relative_to(source):
                mapped = str(Path(destination) / Path(value).relative_to(source))
                break
    if mapped is not None:
        path = Path(mapped)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Resource alias must remain within this checkout: {mapped}")
        return root / path
    path = Path(value)
    if path.is_absolute():
        if path.is_relative_to(root):
            return path
        if path.is_relative_to("/workspace"):
            raise ValueError(f"Unregistered historical resource: {value}")
        return path
    if ".." in path.parts:
        raise ValueError(f"Resource path cannot escape this checkout: {value}")
    return root / path


def weight_records(manifest: dict, *, include_pretraining: bool = False) -> list[dict]:
    return [row for row in manifest["weights"] if include_pretraining or row["role"] in
            ("adopted", "claim_ablation", "five_rate_model_bank")]


def inspect_paper_resources(repo_root: str | Path, *, verify_hashes: bool = False) -> dict:
    """Check the adopted weight/evidence index without importing a model."""
    root = Path(repo_root).resolve()
    manifest = load_manifest(root)
    records = weight_records(manifest)
    failures = []
    for record in records:
        for field, hash_field in (("checkpoint", "sha256"),
                                  ("source_result", "source_result_sha256"),
                                  ("config", "config_sha256")):
            if field not in record:
                continue
            relative = Path(record[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Resource index requires a relative path: {relative}")
            path = root / relative
            if not path.is_file():
                failures.append({"path": str(relative), "reason": "missing",
                                 "action": "download weights" if field == "checkpoint" else "restore repository file"})
            elif verify_hashes:
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != record[hash_field]:
                    failures.append({"path": str(relative), "reason": "sha256_mismatch"})
    return {"checkpoint_count": len(records), "sha256_checked": verify_hashes,
            "failures": failures}


def download_weights(repo_root: str | Path, *, include_pretraining: bool = False,
                     revision: str | None = None) -> dict:
    """Explicitly download indexed weights; never use old server checkpoint paths."""
    root = Path(repo_root).resolve()
    manifest = load_manifest(root)
    records = weight_records(manifest, include_pretraining=include_pretraining)
    pending = [row["checkpoint"] for row in records if not row.get("hf_file")]
    if pending:
        raise ValueError("Hugging Face publication is pending for: " + ", ".join(pending))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub to download model weights") from error
    repository = manifest["model_repository"]
    downloaded, present = [], []
    for record in records:
        destination = resource_path(record["checkpoint"], root)
        if destination.exists():
            with destination.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != record["sha256"]:
                raise FileExistsError(f"Existing checkpoint SHA differs: {destination}")
            present.append(record["checkpoint"])
            continue
        cached = Path(hf_hub_download(
            repo_id=repository["repo_id"], repo_type=repository.get("repo_type", "model"),
            filename=record["hf_file"], revision=revision or repository.get("revision"),
        ))
        with cached.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Downloaded checkpoint SHA differs: {record['checkpoint']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with cached.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        downloaded.append(record["checkpoint"])
    return {"downloaded": downloaded, "already_present": present}


def inspect_directories(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    missing = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    return {
        "method": "DecompJSCC",
        "repo_root": str(root),
        "required_directories": list(REQUIRED_DIRS),
        "missing_directories": missing,
    }


def prepare_directories(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    created = []
    for name in REQUIRED_DIRS:
        path = root / name
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            created.append(name)
    return {**inspect_directories(root), "created_directories": created}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DecompJSCC datasets and output directories")
    parser.add_argument("action", choices=("validate", "prepare", "download"))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--check-paper-resources", action="store_true",
                        help="Check adopted checkpoint/config/raw paths without model inference")
    parser.add_argument("--sha256", action="store_true",
                        help="Also hash indexed resources (requires --check-paper-resources)")
    parser.add_argument("--include-pretraining", action="store_true",
                        help="Also download training initializations; historical resume files are excluded")
    parser.add_argument("--revision", help="Explicit Hugging Face revision for downloads")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sha256 and not args.check_paper_resources:
        raise SystemExit("--sha256 requires --check-paper-resources")
    if args.action == "download":
        print(json.dumps(download_weights(args.repo_root, include_pretraining=args.include_pretraining,
                                          revision=args.revision), indent=2))
        return
    result = prepare_directories(args.repo_root) if args.action == "prepare" else inspect_directories(args.repo_root)
    if args.check_paper_resources:
        result["paper_resources"] = inspect_paper_resources(args.repo_root, verify_hashes=args.sha256)
    print(json.dumps(result, indent=2))
    if result["missing_directories"] or result.get("paper_resources", {}).get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
