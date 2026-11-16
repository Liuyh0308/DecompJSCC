"""Qualitative reconstruction using the same models, noise streams and BPG link as grids."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments import evaluation as paper
from swinjscc_hr_dataset import HRCroppedEvalImageDataset

ROOT = paper.ROOT
HISTORY_ROOT = ROOT / "paper_data/evidence/reconstruction"
# Recorded metric source identity; current code hashes are reported separately.
HISTORICAL_METRIC_SHA = "681b13b1822be0a70e8b24b2bc9359be3293de0d3d74f711c4256a8fee4a4134"
METHODS = paper.METHODS
TOLERANCES = {"psnr_db": 1e-4, "ms_ssim": 1e-5, "cbr": 1e-12}
CODE = ("eval.py", "experiments/evaluation.py", "experiments/reconstruction.py",
        "experiments/source_evaluation.py", "experiments/complexity.py",
        "swinjscc_hr_dataset.py", "decompjscc_utils.py", "models/decompjscc.py",
        "models/swinjscc_source_wrapper.py", "SwinJSCC_base/channel.py")
build_model = paper.build_model


def source_path(value):
    return paper.resource_path(value).resolve(strict=True)

def history_path(method, row):
    return HISTORY_ROOT / method / "kodak" / f"rate{row['rate']}_snr{row['snr_db']:g}.json"

def read_history(method, row, image_ids, image_hashes):
    path = history_path(method, row)
    data = json.loads(path.read_text())
    identity = data["identity"]
    expected = {"method": method, "eval_split": "kodak", "rate": row["rate"],
                "snr_db": row["snr_db"], "seed": 20260811, "noise_realizations": 3,
                "eval_crop_multiple": 128, "batch_size": 1,
                "image_ids": image_ids, "image_sha256": image_hashes,
                "ms_ssim": paper.ms_ssim_metadata(),
                "metric_utils_sha256": HISTORICAL_METRIC_SHA,
                "channel_sha256": paper.sha256_file(ROOT / "SwinJSCC_base/channel.py")}
    if data.get("status") != "complete" or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Historical dataset/model/metric protocol mismatch: {path}")
    if data["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Historical checkpoint identity mismatch")
    observations = [r for r in data["per_image_results"]
                    if r["image_id"] == row["image_id"] and r["realization"] == 0]
    if len(observations) != 1 or observations[0]["evaluation_seed"] != 20260811:
        raise ValueError("Expected exactly one historical realization-0 observation")
    return data, observations[0], path

def replay_to_image(method, model, loader, row, device):
    # Replaying preceding images preserves the original condition-level RNG stream.
    paper.seed_randomness(20260811)
    with torch.no_grad():
        for index, (images, ids) in enumerate(loader):
            if index % 10 == 0:
                paper.require_idle_gpu(device)
            images = images.to(device)
            output = paper.forward_model(method, model, images, row["snr_db"], row["rate"])
            if str(ids[0]) != row["image_id"]:
                continue
            reconstruction = output["reconstructed"].clamp(0, 1)
            scores = {
                "psnr_db": float(paper.compute_per_image_psnr(images, reconstruction)[0]),
                "ms_ssim": float(paper.compute_ms_ssim_batch(images, reconstruction)[0]),
                "cbr": float(output["cbr"]),
            }
            if (not all(np.isfinite(v) for v in scores.values())
                    or not 0 <= scores["ms_ssim"] <= 1
                    or not np.isclose(scores["cbr"], row["rate"] / 1536, rtol=0, atol=1e-12)):
                raise ValueError("Invalid reconstruction metrics or transmission budget")
            return reconstruction, scores, index + 1
    raise ValueError(f"Selected image is absent from Kodak: {row['image_id']}")

def metric_comparison(scores, previous):
    differences = {key: scores[key] - previous[key] for key in TOLERANCES}
    return {"historical_metrics": {key: previous[key] for key in TOLERANCES},
            "differences": differences, "absolute_tolerances": TOLERANCES,
            "matches_history": all(abs(v) <= TOLERANCES[key] for key, v in differences.items()),
            "reported_metrics": "newly measured reconstruction; never substituted with historical values"}

def run_neural(args, selection):
    selection_path = Path(args.selection).resolve(strict=True)
    selected_methods = args.methods
    if (not selected_methods or len(set(selected_methods)) != len(selected_methods)
            or any(method not in METHODS for method in selected_methods)):
        raise ValueError("Choose distinct supported neural methods")
    output = Path(args.output_dir).resolve()
    for method in selected_methods:
        if (output / method).exists():
            raise FileExistsError(f"Refusing to overwrite existing method output: {output / method}")
    dataset = HRCroppedEvalImageDataset([str(ROOT / "datasets/swinjscc_hr/Kodak")], crop_multiple=128)
    images = [Path(p) for p in dataset.image_files]
    image_ids = [p.name for p in images]
    if len(image_ids) != 24 or len(set(image_ids)) != 24:
        raise ValueError("Expected the original 24 Kodak images")
    image_hashes = {p.name: paper.sha256_file(p) for p in images}
    for row in selection["rows"]:
        if image_hashes.get(row["image_id"]) != row["original_sha256"]:
            raise ValueError("Selected source differs from evaluated Kodak input")
    histories = {(method, row["panel"]): read_history(method, row, image_ids, image_hashes)
                 for method in selected_methods for row in selection["rows"]}
    historical_hashes = {str(path): paper.sha256_file(path) for _, _, path in histories.values()}
    source_hashes = {name: paper.sha256_file(ROOT / name) for name in CODE}
    selection_sha = paper.sha256_file(selection_path)
    device = paper.resolve_device(args.device)
    paper.require_idle_gpu(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    summaries = []
    for method in selected_methods:
        folder = output / method
        folder.mkdir(parents=True, exist_ok=False)
        model, model_record, last_rate = None, None, None
        observations = []
        for row in selection["rows"]:
            rate = row["rate"]
            if model is None or (rate != last_rate and method in paper.FIXED_METHODS):
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model, model_record = build_model(method, rate)
                model = model.to(device).eval()
                last_rate = rate
            historical, previous, historical_path = histories[method, row["panel"]]
            if model_record["checkpoint_sha256"] != historical["checkpoint_sha256"]:
                raise ValueError("Model is not the checkpoint used by the adopted Fig.5")
            reconstruction, scores, replayed_images = replay_to_image(method, model, loader, row, device)
            image_path = folder / f"row_{row['panel']}.png"
            paper.save_image(image_path, reconstruction)
            comparison = metric_comparison(scores, previous)
            record = {
                "method": method, "panel": row["panel"], "rate": rate, "snr_db": row["snr_db"],
                "image_id": row["image_id"], "realization": 0, "evaluation_seed": 20260811,
                "psnr_db": scores["psnr_db"], "ms_ssim": scores["ms_ssim"],
                "actual_cbr": scores["cbr"], "checkpoint": model_record["checkpoint"],
                "checkpoint_sha256": model_record["checkpoint_sha256"],
                "png": str(image_path), "png_sha256": paper.sha256_file(image_path),
            }
            provenance = {
                "historical_json": str(historical_path),
                "historical_json_sha256": historical_hashes[str(historical_path)],
                "selection": str(selection_path), "selection_sha256": selection_sha,
                "source_image": str(source_path(row["source_image"])),
                "original_sha256": row["original_sha256"], "crop_xywh": row["crop_xywh"],
                "code_sha256": source_hashes, "checkpoint_load": model_record["checkpoint_load"],
                "ms_ssim": paper.ms_ssim_metadata(), "eval_crop_multiple": 128, "batch_size": 1,
                "replay_order": image_ids[:replayed_images],
                "replayed_images": replayed_images, "image_sha256": image_hashes,
                "noise_protocol": "seed 20260811 before the unchanged Kodak DataLoader order; realization 0",
                "cross_architecture_common_noise": False,
                "metric_comparison": comparison,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            }
            paper.write_json(folder / f"row_{row['panel']}.json",
                             {"status": "complete", "record": record, "provenance": provenance})
            observations.append(record)
            print(json.dumps({"method": method, "panel": row["panel"], **scores,
                              "history_comparison": comparison}), flush=True)
            del reconstruction
        if paper.sha256_file(selection_path) != selection_sha or any(
                paper.sha256_file(ROOT / name) != sha for name, sha in source_hashes.items()):
            raise ValueError("Selection or source code changed during reconstruction")
        if any(paper.sha256_file(path) != historical_hashes[str(path)]
               for _, _, path in histories.values()) or any(
                   paper.sha256_file(path) != image_hashes[path.name] for path in images):
            raise ValueError("Historical observations or original images changed during reconstruction")
        summary = {"status": "complete", "method": method, "observations": observations,
                   "selection_sha256": selection_sha, "training_performed": False}
        paper.write_json(folder / "summary.json", summary)
        summaries.append(summary)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"status": "complete", "methods": selected_methods,
            "observations": sum(len(item["observations"]) for item in summaries),
            "output_dir": str(output)}


def run_bpg(args, selection):
    records = []
    for row in selection["rows"]:
        dataset = HRCroppedEvalImageDataset([str(ROOT / "datasets/swinjscc_hr/Kodak")], crop_multiple=128)
        dataset.image_files = [path for path in dataset.image_files if Path(path).name == row["image_id"]]
        if len(dataset) != 1:
            raise ValueError(f"Missing selected Kodak image: {row['image_id']}")
        folder = Path(args.output_dir) / f"row_{row['panel']}"
        command = paper.bpg_parser().parse_args([
            "--eval-split", "kodak", "--rates", str(row["rate"]),
            "--snr-db-values", str(row["snr_db"]), "--noise-realizations", "1",
            "--seed", str(selection["seed"]), "--device", args.device,
            "--save-image-id", row["image_id"], "--reconstruction-dir", str(folder / "reconstructions"),
            "--output", str(folder / "grid.json"),
        ])
        raw = paper.run_bpg(command, dataset=dataset)
        value = raw["per_image_results"][0]
        png = folder / "reconstructions/bpg_ldpc" / f"rate{row['rate']}_snr{row['snr_db']:g}.png"
        record = dict(value, method="bpg_ldpc", panel=row["panel"], png=str(png),
                      png_sha256=paper.sha256_file(png))
        paper.write_json(Path(args.output_dir) / f"row_{row['panel']}.json", {
            "status": "complete", "record": record,
            "provenance": {"selection": str(args.selection),
                           "source_image": row["source_image"], "original_sha256": row["original_sha256"],
                           "raw_output": str(folder / "grid.json"),
                           "mcs_calibration_sha256": raw["mcs_calibration_sha256"],
                           "failure_policy": "all measured outcomes retained without retries"}})
        records.append(record)
    result = {"status": "complete", "method": "bpg_ldpc", "observations": records}
    paper.write_json(Path(args.output_dir) / "summary.json", result)
    return result


def run(args, selection):
    paper.require_new_output(args.output_dir)
    return run_neural(args, selection) if args.part == "neural" else run_bpg(args, selection)
