"""Train or continue fixed-CBR SNR-only controls using the source epoch loop."""

import argparse
import contextlib
import copy
import fcntl
import io
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from decompjscc_utils import (
    capture_training_rng_state, channel_implementation_metadata,
    restore_training_rng_state, sha256_file,
)
from distributed_utils import create_dist_context, setup_seed
from experiments import source_training as source


ROOT = Path(__file__).resolve().parents[1]
INITIAL_CHECKPOINT = ROOT / "checkpoints/pretraining/swinjscc_fixed/model_best.pth"
INITIAL_SHA256 = "dbec37749f08da2aaadd20706ef660232789d381ba44bbea05d8539730956264"
METHODS = ("decompjscc_wo_rcsb", "swinjscc_sa")
EPOCHS = 300
PHYSICAL_BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
GLOBAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS
HEAD_KEYS = {
    "net.encoder.head_list.weight", "net.encoder.head_list.bias",
    "net.decoder.head_list.weight",
}


def training_protocol(rate=None, epochs=EPOCHS):
    microbatches = 2663 // PHYSICAL_BATCH_SIZE
    if microbatches % ACCUMULATION_STEPS:
        raise ValueError("The frozen train pool must contain only complete accumulation groups")
    result = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "microbatches_per_epoch": microbatches,
        "optimizer_steps_per_epoch": microbatches // ACCUMULATION_STEPS,
        "samples_per_epoch": microbatches * PHYSICAL_BATCH_SIZE,
        "dropped_samples_per_epoch": 2663 - microbatches * PHYSICAL_BATCH_SIZE,
        "loss_scale_per_microbatch": 1 / ACCUMULATION_STEPS,
        "condition_sampling": "one SNR per optimizer step, shared across accumulated microbatches",
        "channel_normalization_scope": "physical batch, not the effective accumulated batch",
        "precision": "float32", "epochs": epochs, "seed": 42,
        "learning_rate": 0.0001, "optimizer": "Adam", "distortion_metric": "MSE",
    }
    if rate is not None:
        result.update(fixed_channels=rate, target_cbr=rate / 1536)
    return result


def output_dirs(method, rate=32, output_root=None, run_name=""):
    if method not in METHODS:
        raise ValueError(f"Unsupported SNR-only method: {method}")
    relative = ("controls/snr_only" if method == METHODS[0] else "baselines/swinjscc_sa")
    if rate not in (32, 64, 96, 128, 192):
        raise ValueError("Fixed channels must be one of 32, 64, 96, 128, 192")
    if run_name and (Path(run_name).name != run_name or run_name in (".", "..")):
        raise ValueError("--run-name must be one directory name")
    relative += f"/awgn/cbr_1_{1536 // rate}"
    if run_name:
        relative += f"/{run_name}"
    root = Path(output_root) if output_root is not None else ROOT
    return {name: root / prefix / "decompjscc" / relative for name, prefix in (
        ("checkpoint", "checkpoints"), ("experiment", "exp_log"), ("tensorboard", "runs"))}


def runtime_args(method, rate=32, epochs=EPOCHS, output_root=None, run_name=""):
    if epochs <= 0:
        raise ValueError("Target epoch count must be positive")
    paths = output_dirs(method, rate, output_root, run_name)
    recipe = ("no_modnet_snr_stageall_cross_attnres_decoder_adapter"
              if method == METHODS[0] else "snr_adaptive")
    return source.build_parser().parse_args([
        "--recipe-stage", recipe, "--data-dir", str(ROOT / "datasets/swinjscc_hr"),
        "--C", str(rate), "--multiple-snr", "1,4,7,10,13", "--channel-type", "awgn",
        "--model-size", "base", "--distortion-metric", "MSE", "--img-size", "256",
        "--hr-crop-size", "256", "--hr-eval-split", "kodak", "--hr-eval-crop-multiple", "128",
        "--epochs", str(epochs), "--batch-size", str(PHYSICAL_BATCH_SIZE),
        "--gradient-accumulation-steps", str(ACCUMULATION_STEPS), "--val-batch-size", "1",
        "--num-workers", "0", "--lr", "0.0001", "--seed", "42",
        "--dist-mode", "single", "--gpu-ids", "0", "--freeze-mode", "none",
        "--attnres-query-mode", "snr_depth_logit_gate", "--attnres-condition-gate-mode", "snr",
        "--checkpoint-path", str(paths["checkpoint"]), "--exp-log-dir", str(paths["experiment"]),
        "--log-dir", str(paths["tensorboard"]), "--resume-checkpoint", str(INITIAL_CHECKPOINT),
    ])


def check_empty_outputs(paths):
    for path in paths.values():
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def check_idle_gpu():
    result = subprocess.run([
        "nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError(f"GPU 0 is occupied; no training started: {result.stdout.strip()}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def initialize_from_fixed(model, checkpoint_path=INITIAL_CHECKPOINT, expected_sha=INITIAL_SHA256):
    digest = sha256_file(checkpoint_path)
    if digest != expected_sha:
        raise ValueError("Official fixed initialization checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    current = model.state_dict()
    skipped = {
        key: {"source_shape": list(value.shape),
              "target_shape": list(current[key].shape) if key in current else None}
        for key, value in state.items()
        if key not in current or current[key].shape != value.shape
    }
    expected_skipped = HEAD_KEYS if model.net.encoder.head_list.out_features != 192 else set()
    if set(skipped) != expected_skipped:
        raise ValueError(f"Expected only incompatible fixed-width interface tensors; got {skipped}")
    inherited = {key: value for key, value in state.items() if key not in skipped}
    if len(state) != 493 or len(inherited) != 493 - len(expected_skipped):
        raise ValueError("Unexpected fixed-backbone state schema")
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Initialization checkpoint contains nonfinite tensors")
    result = model.load_state_dict(inherited, strict=False)
    new_keys = sorted(set(current) - set(state))
    if result.unexpected_keys or set(result.missing_keys) != set(new_keys) | expected_skipped:
        raise ValueError("Unexpected shape-filtered initialization result")
    return {
        "path": str(checkpoint_path), "sha256": digest, "source_epoch": payload.get("epoch"),
        "source_state_count": len(state), "loaded_state_count": len(inherited),
        "skipped": skipped, "new_parameter_keys": new_keys,
        "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
        "optimizer_inherited": False,
    }


def architecture_metadata(model, method, rate=32):
    if method not in METHODS:
        raise ValueError("Unsupported SNR-only method")
    if model.net.encoder.head_list.out_features != rate or model.net.decoder.head_list.in_features != rate:
        raise ValueError(f"SNR-only control must use a direct fixed C{rate} transmission interface")
    if hasattr(model, "rcsb_rate_adapter"):
        raise ValueError("SNR-only controls must not contain RCSB")
    result = {
        "class_name": type(model).__name__, "source_model": model.source_model_name,
        "fixed_channels": rate, "cbr": rate / 1536, "rcsb": False,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "channel": channel_implementation_metadata(model),
    }
    if method == METHODS[0]:
        stages = model.encoder_stage_attnres
        if len(stages) != 4 or not model.enable_cross_stage_attnres or not hasattr(model, "decoder_snr_adapter"):
            raise ValueError("Missing approved DecompJSCC SNR-only components")
        for stage in stages:
            if (stage.attnres_query_mode != "snr_depth_logit_gate"
                    or stage.condition_reference_snr_db != 7 or stage.snr_norm_denominator != 20
                    or not stage.attnres_condition_gate_use_reference):
                raise ValueError("SNR-depth conditioning deviates from the frozen model")
        if [stage.enable_cross_stage_attnres for stage in stages] != [False, False, True, True]:
            raise ValueError("Unexpected cross-stage deployment")
        result.update(stage_all=True, cross_stage_edges=[[1, 3], [1, 4], [2, 4]],
                      query_mode="snr_depth_logit_gate", snr_reference=7, snr_scale=20,
                      decoder_snr_adapter=True)
    elif model.source_model_name != "SwinJSCC_w/_SA":
        raise ValueError("Official SA control has the wrong source model")
    return result


def preflight(method, rate=32, epochs=EPOCHS, output_root=None, run_name="", code_paths=None):
    args = runtime_args(method, rate, epochs, output_root, run_name)
    setup_seed(args.seed)
    source_args = source.build_source_args_from_runtime(args)
    train_dataset, val_dataset = source.build_source_datasets(args)
    if (len(train_dataset), len(val_dataset)) != (2663, 24):
        raise ValueError("Expected full 2663-image HR train pool and 24 Kodak validation images")
    if set(train_dataset.image_files) & set(val_dataset.image_files):
        raise ValueError("Training/validation image paths overlap")
    with contextlib.redirect_stdout(io.StringIO()):
        model = source.build_train_source_model(args, source_args)
    initialization = initialize_from_fixed(model)
    architecture = architecture_metadata(model, method, rate)
    rng = capture_training_rng_state()
    with contextlib.redirect_stdout(io.StringIO()):
        clone = source.build_train_source_model(args, source_args)
    clone.load_state_dict(model.state_dict(), strict=True)
    del clone
    restore_training_rng_state(rng)
    code_paths = code_paths or [
        "train.py", "experiments/fixed_training.py", "experiments/source_training.py",
        "models/swinjscc_source_wrapper.py", "models/stage_block_attnres.py",
        "SwinJSCC_base/network.py", "SwinJSCC_base/encoder.py", "SwinJSCC_base/decoder.py",
        "SwinJSCC_base/channel.py", "swinjscc_hr_dataset.py", "swinjscc_utils.py",
        "decompjscc_utils.py", "distributed_utils.py",
    ]
    audit = {
        "method": method, "architecture": architecture, "initialization": initialization,
        "strict_reload": True, "training_image_count": len(train_dataset),
        "validation_image_count": len(val_dataset), "training_images": train_dataset.image_files,
        "validation_images": val_dataset.image_files,
        "code_sha256": {path: sha256_file(ROOT / path) for path in code_paths},
        "selection": "highest mean per-image PSNR over Kodak and SNR 1,4,7,10,13; first best on ties",
        **training_protocol(rate if rate != 32 else None, epochs),
        "training_snr_db": [1, 4, 7, 10, 13], "rates": [rate],
        "learning_rate_schedule": None, "torch_version": torch.__version__,
        "gpu_pilot_status": "not_run",
    }
    return args, source_args, model, train_dataset, val_dataset, audit


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def smoke_batch(model, device, rate=32):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Pilot requires a fresh model with no existing gradients")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Pilot must retain float32 precision")
    rng = capture_training_rng_state()
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    modes = {module: module.training for module in model.modules()}
    optimizer = output = images = None
    measured_cuda = device.type == "cuda"
    try:
        model.train()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.0001)
        if measured_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        losses = []
        for _ in range(ACCUMULATION_STEPS):
            images = torch.rand(PHYSICAL_BATCH_SIZE, 3, 256, 256, device=device)
            output = model(images, snr_db=7.0, rate=rate)
            if not torch.isfinite(output["loss"]) or not math.isclose(output["cbr"], rate / 1536):
                raise ValueError(f"Fixed-C{rate} forward smoke failed")
            (output["loss"] / ACCUMULATION_STEPS).backward()
            losses.append(output["loss"].item())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"Fixed-C{rate} backward smoke contains nonfinite gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Pilot Adam update produced nonfinite parameters")
        if not optimizer.state or any(
                isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                for state in optimizer.state.values() for value in state.values()):
            raise ValueError("Pilot Adam state is missing or nonfinite")
        if measured_cuda:
            torch.cuda.synchronize(device)
        result = {
            "status": "passed", "device": str(device), "measured_cuda": measured_cuda,
            "physical_batch_size": PHYSICAL_BATCH_SIZE, "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "global_batch_size": GLOBAL_BATCH_SIZE, "microbatches": ACCUMULATION_STEPS,
            "optimizer_steps": 1, "loss": sum(losses) / len(losses), "cbr": rate / 1536,
            "optimizer_state_finite": True, "updated_parameters_finite": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if measured_cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if measured_cuda else None,
        }
    finally:
        output = images = optimizer = None
        model.zero_grad(set_to_none=True)
        try:
            model.load_state_dict(initial, strict=True)
            if any(not torch.equal(value.detach().cpu(), initial[name])
                   for name, value in model.state_dict().items()):
                raise ValueError("Pilot failed to restore the initial model state")
        finally:
            for module, mode in modes.items():
                module.training = mode
            restore_training_rng_state(rng)
            if measured_cuda:
                torch.cuda.empty_cache()
    result.update(initial_state_restored=True, rng_restored=True, gradients_cleared=True)
    return result


def verify_serialized_state(model):
    with tempfile.TemporaryDirectory(prefix="decompjscc_snr_only_pilot_") as folder:
        path = Path(folder) / "model.pth"
        torch.save({"model_state_dict": model.state_dict()}, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return {"strict": True, "state_count": len(payload["model_state_dict"]),
                "sha256": sha256_file(path)}


class GPUCheckedLoader:
    """Keep the shared source epoch loop unchanged while checking GPU ownership."""

    def __init__(self, loader, device, check):
        self.loader, self.device, self.check = loader, device, check

    def __iter__(self):
        for batch in self.loader:
            self.check(self.device)
            yield batch


def load_parent_state(method, args, audit, checkpoint, best_checkpoint):
    """Validate a resumable run without replacing the historical identity checks."""
    paths = {"last": Path(checkpoint), "best": Path(best_checkpoint)}
    payloads = {role: torch.load(path, map_location="cpu", weights_only=False)
                for role, path in paths.items()}
    last, best = payloads["last"], payloads["best"]
    required = {"epoch", "best_psnr", "optimizer_state_dict", "rng_state",
                "model_state_dict", "config", "training_metadata", "method", "val_metrics"}
    for role, payload in payloads.items():
        if missing := required.difference(payload):
            raise ValueError(f"Parent {role} is not resumable: missing {sorted(missing)}")
        if payload["method"] != method:
            raise ValueError("Parent method differs from the requested method")
        for key in ("C", "recipe_stage", "source_model", "multiple_snr", "channel_type", "model_size",
                    "distortion_metric", "img_size", "hr_crop_size", "hr_eval_crop_multiple",
                    "batch_size", "gradient_accumulation_steps", "val_batch_size", "num_workers",
                    "lr", "seed", "freeze_mode", "hr_eval_split", "attnres_query_mode",
                    "attnres_content_query_scale", "attnres_condition_gate_mode",
                    "attnres_cross_stage_skip_threshold", "attnres_cross_stage_force_skip",
                    "no_attnres_condition_gate_reference",
                    "no_attnres_layer_embedding", "max_train_batches", "max_val_batches"):
            if payload["config"].get(key) != getattr(args, key):
                raise ValueError(f"Parent training configuration differs: {key}")
        metadata = payload["training_metadata"]
        if metadata.get("architecture") != audit["architecture"]:
            raise ValueError("Parent model/data differs: architecture")
        for key in ("training_images", "validation_images"):
            recorded = metadata.get(key)
            if recorded is None or [str(Path(path).resolve()) for path in recorded] != [
                    str(Path(path).resolve()) for path in audit[key]]:
                raise ValueError(f"Parent model/data differs: {key}")
        if metadata.get("torch_version") != torch.__version__:
            raise ValueError("Resume requires the parent's PyTorch version")
        for key, value in training_protocol().items():
            if key != "epochs" and metadata.get(key) != value:
                raise ValueError(f"Parent training protocol differs: {key}")
    if type(last["epoch"]) is not int or type(best["epoch"]) is not int:
        raise ValueError("Parent epoch counts must be integers")
    start_epoch = last["epoch"]
    if not 0 < best["epoch"] <= start_epoch < args.epochs:
        raise ValueError("--epochs is the final total and must exceed the parent last epoch")
    if (not math.isfinite(last["best_psnr"]) or last["best_psnr"] != best["best_psnr"]
            or best["val_metrics"]["psnr"] != last["best_psnr"]):
        raise ValueError("Parent best checkpoint does not match the cumulative best metric")
    if set(last["rng_state"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Parent must contain complete Python/NumPy/torch/CUDA RNG state")
    cuda_rng = last["rng_state"]["cuda"]
    if cuda_rng is None or len(cuda_rng) != 1 or any(
            not torch.is_tensor(value) or value.dtype != torch.uint8
            for value in (last["rng_state"]["torch"], *cuda_rng)):
        raise ValueError("Parent must contain the original single-GPU RNG state")
    for payload in payloads.values():
        if any(value.is_floating_point() and not torch.isfinite(value).all()
               for value in payload["model_state_dict"].values()):
            raise ValueError("Parent contains nonfinite model tensors")
    state = last["optimizer_state_dict"]
    expected_steps = start_epoch * training_protocol()["optimizer_steps_per_epoch"]
    if not state.get("state") or not state.get("param_groups"):
        raise ValueError("Parent must preserve Adam state")
    for group in state["param_groups"]:
        if group["lr"] != args.lr or group.get("capturable", False) or group.get("fused", False):
            raise ValueError("Parent Adam configuration differs")
    for values in state["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(values):
            raise ValueError("Parent Adam moments are incomplete")
        if float(values["step"]) != expected_steps:
            raise ValueError("Parent Adam step does not match the completed epoch count")
        if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Parent Adam state is nonfinite")
    payloads["audit"] = {
        "start_epoch": start_epoch, "best_epoch": int(best["epoch"]),
        "parent_files": {str(path): sha256_file(path) for path in paths.values()},
        "resume_checkpoint": str(paths["last"]), "parent_best_checkpoint": str(paths["best"]),
        "optimizer_restore_required": True, "rng_restore_required": True,
        "selection_scope": f"first highest validation PSNR across epochs 1-{args.epochs}",
    }
    return payloads


def restore_parent_optimizer(optimizer, state):
    # Adam keeps non-capturable step scalars on CPU even for CUDA parameters.
    optimizer.load_state_dict(copy.deepcopy(state))
    actual = optimizer.state_dict()
    if actual["param_groups"] != state["param_groups"]:
        raise ValueError("Restored Adam groups differ")
    for index, expected in state["state"].items():
        for key, value in expected.items():
            if torch.is_tensor(value) and not torch.equal(actual["state"][index][key].cpu(), value.cpu()):
                raise ValueError(f"Restored Adam {key} differs")
        if actual["state"][index]["step"].device.type != "cpu":
            raise ValueError("Non-capturable Adam steps must remain on CPU")


def train(method, rate=32, epochs=EPOCHS, output_root=None, run_name="", *,
          resume_checkpoint=None, resume_best_checkpoint=None, code_paths=None, check_gpu=None):
    if bool(resume_checkpoint) != bool(resume_best_checkpoint):
        raise ValueError("Resume requires both the last and cumulative best checkpoints")
    paths = output_dirs(method, rate, output_root, run_name)
    check_empty_outputs(paths)
    paths["experiment"].parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as cleanup, (
            paths["experiment"].parent / f"{paths['experiment'].name}.training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_empty_outputs(paths)
        (check_gpu or check_idle_gpu)()
        # Register our host PID before any model/seed operation creates a CUDA context.
        from experiments.evaluation import require_idle_gpu
        require_idle_gpu(torch.device("cuda:0"))
        args, source_args, model, train_dataset, val_dataset, audit = preflight(
            method, rate, epochs, output_root, run_name, code_paths)
        parent = None
        if resume_checkpoint:
            parent = load_parent_state(method, args, audit, resume_checkpoint, resume_best_checkpoint)
            model.load_state_dict(parent["best"]["model_state_dict"], strict=True)
            model.load_state_dict(parent["last"]["model_state_dict"], strict=True)
            source.reset_source_train_resolution(model, args)
            args.resume_checkpoint = str(resume_checkpoint)
            args.resume_training_state = True
            audit["parent_run"] = parent["audit"]
        dist_ctx = create_dist_context(args)
        cleanup.callback(dist_ctx.cleanup)
        if dist_ctx.device.type != "cuda":
            raise RuntimeError("Formal training requires CUDA; CPU fallback is disabled")
        model = model.to(dist_ctx.device)
        audit["gpu_batch_smoke"] = smoke_batch(model, dist_ctx.device, rate)
        audit["gpu_pilot_status"] = "passed"
        audit["serialized_reload"] = verify_serialized_state(model)
        audit["gpu_name"] = torch.cuda.get_device_name(dist_ctx.device)
        train_loader = dist_ctx.get_dataloader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
                                              num_workers=0, drop_last=True, is_train=True)
        if len(train_loader) != audit["microbatches_per_epoch"] or len(train_loader) % ACCUMULATION_STEPS:
            raise ValueError("Unexpected incomplete accumulation group in the formal training loader")
        val_loader = dist_ctx.get_dataloader(val_dataset, batch_size=1, shuffle=False,
                                            num_workers=0, drop_last=False, is_train=False)
        train_loader = GPUCheckedLoader(train_loader, dist_ctx.device, require_idle_gpu)
        val_loader = GPUCheckedLoader(val_loader, dist_ctx.device, require_idle_gpu)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        if parent:
            restore_parent_optimizer(optimizer, parent["last"]["optimizer_state_dict"])
            audit["parent_run"]["optimizer_restored"] = True
        config = vars(args).copy()
        config.update(method=method, source_args=vars(source_args), training_metadata=audit,
                      attnres_condition_gate_use_reference=True)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        write_json(paths["experiment"] / "config.json", config)
        write_json(paths["experiment"] / "initialization_audit.json", audit)
        start_epoch = int(parent["last"]["epoch"]) if parent else 0
        best_psnr = float(parent["last"]["best_psnr"]) if parent else float("-inf")
        best_epoch = int(parent["best"]["epoch"]) if parent else None
        if parent:
            destination = paths["checkpoint"] / "model_best.pth"
            shutil.copy2(resume_best_checkpoint, destination)
            if sha256_file(destination) != sha256_file(resume_best_checkpoint):
                raise ValueError("Inherited cumulative best differs from its source")
        resume_rng = parent["last"]["rng_state"] if parent else None
        parent = None
        with SummaryWriter(str(paths["tensorboard"])) as writer:
            if resume_rng is not None:
                restore_training_rng_state(resume_rng)
            for epoch in range(start_epoch, epochs):
                start = time.monotonic()
                training = source.train_epoch(epoch, model, train_loader, optimizer, source_args, args, dist_ctx)
                validation = source.validate(model, val_loader, source_args, args, dist_ctx)
                source.reset_source_train_resolution(model, args)
                if any(not math.isfinite(value) for metrics in (training, validation) for value in metrics.values()):
                    raise ValueError(f"Nonfinite metric at epoch {epoch + 1}")
                is_best = validation["psnr"] > best_psnr
                if is_best:
                    best_epoch = epoch + 1
                best_psnr = max(best_psnr, validation["psnr"])
                payload = {
                    "epoch": epoch + 1, "method": method, "model_state_dict": model.state_dict(),
                    "source_args": vars(source_args), "config": config, "training_metadata": audit,
                    "best_psnr": best_psnr, "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": capture_training_rng_state(), "train_metrics": training,
                    "val_metrics": validation,
                }
                save_checkpoint(paths["checkpoint"] / "model_last.pth", payload)
                if is_best:
                    save_checkpoint(paths["checkpoint"] / "model_best.pth", payload)
                history = {"epoch": epoch + 1, "train": training, "validation": validation,
                           "best_psnr": best_psnr, "elapsed_seconds": time.monotonic() - start}
                with (paths["experiment"] / "training_history.jsonl").open("a") as handle:
                    handle.write(json.dumps(history) + "\n")
                write_json(paths["experiment"] / "selection.json", {
                    "completed_epoch": epoch + 1, "target_epoch": epochs,
                    "best_epoch": best_epoch, "best_psnr": best_psnr,
                    "inherited_parent_best": bool(start_epoch and best_epoch <= start_epoch),
                    "parent_best_checkpoint": str(resume_best_checkpoint) if start_epoch else None,
                })
                for split, values in (("train", training), ("validation", validation)):
                    for key, value in values.items():
                        writer.add_scalar(f"{split}/{key}", value, epoch + 1)
                writer.flush()
                print(json.dumps(history), flush=True)
        if resume_checkpoint:
            for path, expected in audit["parent_run"]["parent_files"].items():
                if sha256_file(Path(path)) != expected:
                    raise ValueError(f"Parent artifact changed: {path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--rate", type=int, choices=(32, 64, 96, 128, 192), default=32,
                        help="Fixed latent width k; CBR=k/1536, not LDPC code rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Final total epoch count, including resumed epochs")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--run-name", default="", help="Optional new output subdirectory; existing outputs are protected")
    parser.add_argument("--resume-checkpoint", type=Path, help="Parent model_last.pth with Adam and RNG state")
    parser.add_argument("--resume-best-checkpoint", type=Path, help="Parent cumulative model_best.pth; required for resume")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="Read-only CPU initialization/data audit")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if bool(args.resume_checkpoint) != bool(args.resume_best_checkpoint):
        parser.error("Resume requires both --resume-checkpoint and --resume-best-checkpoint")
    options = dict(rate=args.rate, epochs=args.epochs, output_root=args.output_root, run_name=args.run_name)
    if args.run:
        train(args.method, **options, resume_checkpoint=args.resume_checkpoint,
              resume_best_checkpoint=args.resume_best_checkpoint)
    elif args.preflight:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            parser.error("CPU preflight requires CUDA_VISIBLE_DEVICES='' to avoid the live GPU job")
        runtime, _, _, _, _, audit = preflight(args.method, **options)
        if args.resume_checkpoint:
            audit["parent_run"] = load_parent_state(
                args.method, runtime, audit, args.resume_checkpoint, args.resume_best_checkpoint)["audit"]
        print(json.dumps(audit, indent=2))
    else:
        runtime = runtime_args(args.method, **options)
        if args.resume_checkpoint:
            runtime.resume_checkpoint = str(args.resume_checkpoint)
            runtime.resume_training_state = True
        print(json.dumps({"method": args.method, "runtime": vars(runtime),
                          "training_protocol": training_protocol(args.rate, args.epochs),
                          "resume_best_checkpoint": str(args.resume_best_checkpoint) if args.resume_best_checkpoint else None,
                          "gpu_pilot_status": "not_run",
                          "initial_sha256": INITIAL_SHA256, "dry_run": True}, indent=2))


if __name__ == "__main__":
    main()
