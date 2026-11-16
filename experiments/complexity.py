"""Model loading and parameter, FLOP and latency measurements."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
import torch
import experiments.source_evaluation as source_eval
from decompjscc_utils import MAIN_CHECKPOINT, MAIN_EXPERIMENT_DIR, REPO_ROOT, load_decompjscc_checkpoint
from experiments.evaluation import parse_rates, resolve_device
from models.decompjscc import DecompJSCC, DecompJSCCConfig
from models.swinjscc_source_wrapper import load_source_checkpoint
import csv
import gc
from torch.utils.flop_counter import FlopCounterMode
from experiments.evaluation import build_existing_model, forward_model, require_idle_gpu, write_json


ROLE_DEFAULTS = {
    "decompjscc": {
        "checkpoint": MAIN_CHECKPOINT,
        "config": REPO_ROOT / "configs/models/decompjscc.json",
        "gain_mode": "active_rms",
    },
    "swinjscc_universal": {
        "checkpoint": REPO_ROOT / "checkpoints/swinjscc_joint/awgn/model_best.pth",
        "config": REPO_ROOT / "configs/models/swinjscc_joint.json",
    },
}

def profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the paper-defined DecompJSCC model")
    parser.add_argument("--role", default="decompjscc", choices=tuple(ROLE_DEFAULTS))
    parser.add_argument("--checkpoint", default=str(MAIN_CHECKPOINT))
    parser.add_argument("--config-path", default="")
    parser.add_argument("--output-json", default=str(MAIN_EXPERIMENT_DIR / "awgn/complexity/profile.json"))
    parser.add_argument("--output-markdown", default=str(MAIN_EXPERIMENT_DIR / "awgn/complexity/profile.md"))
    parser.add_argument("--rates", default="32,96,192")
    parser.add_argument("--snr-db", default=7.0, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--model-size", default="base", choices=("small", "base", "large"))
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iterations", default=80, type=int)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    return parser

def _state_size_mb(model: torch.nn.Module) -> float:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()) / (1024.0**2)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_role_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    defaults = ROLE_DEFAULTS[args.role]
    default_main = str(MAIN_CHECKPOINT)
    checkpoint = Path(defaults["checkpoint"] if args.role != "decompjscc" and args.checkpoint == default_main else args.checkpoint)
    config_path = Path(args.config_path) if args.config_path else Path(defaults["config"])
    return checkpoint, config_path

def build_role_model(
    args: argparse.Namespace,
    rates: list[int],
    checkpoint: Path,
    config_path: Path,
) -> tuple[torch.nn.Module, dict, dict]:
    if args.role == "decompjscc":
        config = DecompJSCCConfig(
            rate_channels=tuple(rates),
            snr_db_values=(args.snr_db,),
            image_size=args.image_size,
            model_size=args.model_size,
            batch_size=args.batch_size,
        )
        model = DecompJSCC(config, gain_mode=ROLE_DEFAULTS[args.role]["gain_mode"])
        checkpoint_load = load_decompjscc_checkpoint(model, checkpoint, strict=True)
        return model, checkpoint_load, config.to_public_metadata()

    config_payload = source_eval.load_json_config(str(config_path))
    source_cli = source_eval.build_parser().parse_args(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--config-path",
            str(config_path),
            "--rates",
            ",".join(str(rate) for rate in rates),
            "--snr-db-values",
            f"{args.snr_db:g}",
            "--img-size",
            str(args.image_size),
            "--model-size",
            args.model_size,
            "--batch-size",
            str(args.batch_size),
            "--channel-type",
            "awgn",
            "--strict-checkpoint",
        ]
    )
    runtime = source_eval.build_source_eval_runtime_args(source_cli, config_payload)
    model = source_eval.build_source_model(runtime)
    checkpoint_load = load_source_checkpoint(model, str(checkpoint), strict=True)
    metadata = {
        "method": "SwinJSCC",
        "source_model": runtime.source_model,
        "rate_channels": rates,
        "snr_db_values": [float(args.snr_db)],
    }
    return model, checkpoint_load, metadata

def _latency(model, image, snr_db: float, rate: int, warmup: int, iterations: int) -> dict[str, float]:
    use_cuda = image.is_cuda
    timings = []
    accepts_return_aux = "return_aux" in inspect.signature(model.forward).parameters

    def forward_once():
        if accepts_return_aux:
            return model(image, snr_db=snr_db, rate=rate, return_aux=False)
        return model(image, snr_db=snr_db, rate=rate)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            forward_once()
        if use_cuda:
            torch.cuda.synchronize(image.device)
        for _ in range(max(1, iterations)):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward_once()
                end.record()
                torch.cuda.synchronize(image.device)
                timings.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter()
                forward_once()
                timings.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "latency_median_ms": float(statistics.median(timings)),
        "latency_p95_ms": float(ordered[p95_index]),
    }

def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        f"# {payload['role']} 复杂度统计",
        "",
        f"- 参数量：{payload['total_parameters'] / 1e6:.3f} M",
        f"- 模型状态大小：{payload['state_size_mb']:.2f} MB",
        "",
        "| rate | median latency (ms) | p95 latency (ms) | peak memory (MB) |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["profiles"]:
        lines.append(
            "| {rate} | {median} | {p95} | {peak} |".format(
                rate=row["rate"],
                median=f"{row['latency_median_ms']:.2f}" if "latency_median_ms" in row else "-",
                p95=f"{row['latency_p95_ms']:.2f}" if "latency_p95_ms" in row else "-",
                peak=f"{row['peak_memory_mb']:.1f}" if "peak_memory_mb" in row else "-",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def profile_main() -> None:
    args = profile_parser().parse_args()
    rates = parse_rates(args.rates)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint, config_path = resolve_role_paths(args)
    model, checkpoint_load, metadata = build_role_model(args, rates, checkpoint, config_path)
    payload = {
        **metadata,
        "material_passport": {
            "origin": "experiment-agent",
            "mode": "run",
            "verification_status": "MEASURED",
            "version": "decompjscc_complexity_profile_v2",
        },
        "role": args.role,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_file_size_mb": checkpoint.stat().st_size / (1024.0**2),
        "config_path": str(config_path),
        "checkpoint_load": checkpoint_load,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "channel_type": "awgn",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "snr_db": args.snr_db,
        "precision": "float32",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "diagnostic_outputs": False,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_size_mb": _state_size_mb(model),
        "profiles": [],
    }
    if not args.skip_latency:
        model = model.to(device).eval()
        image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        for rate in rates:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row = {"rate": rate, **_latency(model, image, args.snr_db, rate, args.warmup, args.iterations)}
            if device.type == "cuda":
                row["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
            payload["profiles"].append(row)
    else:
        payload["profiles"] = [{"rate": rate} for rate in rates]

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(Path(args.output_markdown), payload)
    print(json.dumps({"role": args.role, "output_json": str(output_json)}, indent=2))


class ComparisonForward(torch.nn.Module):
    def __init__(self, method, model, rate=192):
        super().__init__()
        self.method = method
        self.model = model
        self.rate = rate

    def forward(self, images, snr_db=7.0, rate=None, return_aux=False):
        return forward_model(self.method, self.model, images, snr_db,
                             self.rate if rate is None else rate, return_aux=return_aux)

def comparison_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="decompjscc,swinjscc_universal,sa_ra_jscc,snr_eq_jscc,adjscc")
    parser.add_argument("--rate", type=int, default=192)
    args = parser.parse_args()
    device = torch.device(args.device)
    folder = Path(args.output_dir)
    if (folder / "summary.csv").exists():
        raise FileExistsError("Use a new output folder rather than overwriting measured profiles")
    rows = []
    for method in args.methods.split(","):
        require_idle_gpu(device)
        torch.manual_seed(42)
        model, record = build_existing_model(method, args.rate)
        wrapper = ComparisonForward(method, model, args.rate).to(device).eval()
        image = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            wrapper(image)
        flops = counter.get_total_flops()
        if flops <= 0:
            raise RuntimeError("FLOP counter returned zero")
        supported = {str(op): count for op, count in counter.flop_counts["Global"].items()}
        require_idle_gpu(device)
        timings = _latency(wrapper, image, 7.0, args.rate, 20, 80)
        require_idle_gpu(device)
        row = {
            "method": method, "rate": args.rate, "target_cbr": args.rate / 1536,
            "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "counted_gflops": flops / 1e9, **timings,
        }
        write_json(folder / f"{method}.json", {
            **record, **row, "counted_ops": supported,
            "flops_convention": "torch FlopCounterMode supported operators; multiply-add = 2 FLOPs; not all scalar operators counted",
            "scope": "end-to-end model forward, no auxiliary outputs; excludes image I/O and reported quality metrics",
            "warmup": 20, "iterations": 80, "precision": "float32", "batch_size": 1,
            "image_shape": [1, 3, 256, 256], "snr_db": 7, "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        })
        rows.append(row)
        print(row, flush=True)
        del wrapper, model, image, counter
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 五种现有权重的复杂度比较", "",
        f"共同条件：CBR={args.rate}/1536，SNR=7 dB，256x256 RGB，batch=1，FP32，warmup=20，测量80次。",
        "运算量为 PyTorch FlopCounterMode 支持算子的计数，乘加计2 FLOPs；完整算子清单见逐方法JSON，未将未支持的逐元素操作冒充已计数。", "",
        "| 方法 | 参数量 (M) | 已计数 GFLOPs | 中位时延 (ms) | P95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['parameters_m']:.3f} | {row['counted_gflops']:.3f} | "
                     f"{row['latency_median_ms']:.3f} | {row['latency_p95_ms']:.3f} |")
    lines.extend(["| BPG+LDPC | N/A | N/A | N/A | N/A |", "",
                  "传统链路未计入神经网络参数与 GPU 前向时延排名。时延包含各 wrapper 的收发前向，非纯编码器时间。",
                  "固定 CBR 方法此处只统计一个专用模型，不将其参数量解释为覆盖全部 CBR 的模型库大小。"])
    (folder / "table_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


ROLE_RATES = {
    "decompjscc": (32, 96, 192),
    "swinjscc_universal": (96,),
}

EXPECTED_PROTOCOL = {
    "channel_type": "awgn",
    "image_size": 256,
    "batch_size": 1,
    "snr_db": 7.0,
    "precision": "float32",
    "warmup": 20,
    "iterations": 80,
    "diagnostic_outputs": False,
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_profile_payload(payload: dict, role: str) -> list[dict]:
    if payload.get("role") != role:
        raise ValueError(f"{role}: payload role is {payload.get('role')!r}")
    for key, expected in EXPECTED_PROTOCOL.items():
        if payload.get(key) != expected:
            raise ValueError(f"{role}: {key}={payload.get(key)!r}, expected {expected!r}")
    if "RTX 4090" not in str(payload.get("device_name", "")):
        raise ValueError(f"{role}: expected RTX 4090, got {payload.get('device_name')!r}")
    if payload.get("material_passport", {}).get("verification_status") != "MEASURED":
        raise ValueError(f"{role}: profile is not marked MEASURED")

    checkpoint_load = payload.get("checkpoint_load", {})
    skipped = checkpoint_load.get("skipped", checkpoint_load.get("skipped_keys", []))
    if skipped:
        raise ValueError(f"{role}: checkpoint skipped keys: {skipped}")
    if int(checkpoint_load.get("loaded", 0)) <= 0:
        raise ValueError(f"{role}: checkpoint loaded-key count is missing")

    profiles = payload.get("profiles", [])
    expected_rates = ROLE_RATES[role]
    actual_rates = tuple(int(row["rate"]) for row in profiles)
    if actual_rates != expected_rates:
        raise ValueError(f"{role}: rates={actual_rates}, expected {expected_rates}")

    required_metrics = ("latency_median_ms", "latency_p95_ms", "peak_memory_mb")
    rows = []
    for profile in profiles:
        if any(metric not in profile for metric in required_metrics):
            raise ValueError(f"{role}: incomplete measured profile at rate {profile.get('rate')}")
        rows.append(
            {
                "role": role,
                "rate": int(profile["rate"]),
                "parameters_m": float(payload["total_parameters"]) / 1e6,
                "state_size_mb": float(payload["state_size_mb"]),
                "latency_median_ms": float(profile["latency_median_ms"]),
                "latency_p95_ms": float(profile["latency_p95_ms"]),
                "peak_memory_mb": float(profile["peak_memory_mb"]),
                "checkpoint_sha256": str(payload["checkpoint_sha256"]),
            }
        )
    return rows

def summarize(input_dir: Path) -> dict:
    inputs = []
    rows = []
    environment = None
    for role in ROLE_RATES:
        path = input_dir / f"{role}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(validate_profile_payload(payload, role))

        checkpoint = Path(payload["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        recorded_checkpoint_sha = str(payload["checkpoint_sha256"])
        if sha256_file(checkpoint) != recorded_checkpoint_sha:
            raise ValueError(f"{role}: checkpoint SHA256 changed after profiling")
        inputs.append(
            {
                "role": role,
                "profile_path": str(path),
                "profile_sha256": sha256_file(path),
                "checkpoint_path": payload["checkpoint"],
                "checkpoint_sha256": recorded_checkpoint_sha,
                "checkpoint_loaded_keys": int(payload["checkpoint_load"]["loaded"]),
            }
        )
        current_environment = {
            "device": payload["device"],
            "device_name": payload["device_name"],
            "torch_version": payload["torch_version"],
            "cuda_version": payload["cuda_version"],
        }
        if environment is None:
            environment = current_environment
        elif environment != current_environment:
            raise ValueError(f"{role}: runtime environment differs from preceding profiles")

    return {
        "material_passport": {
            "origin": "experiment-agent",
            "mode": "run",
            "verification_status": "MEASURED",
            "version": "decompjscc_complexity_summary_v1",
        },
        "protocol": {**EXPECTED_PROTOCOL, "seed": 20260826, "return_aux": False},
        "environment": environment,
        "inputs": inputs,
        "rows": rows,
    }

def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "role",
        "rate",
        "parameters_m",
        "state_size_mb",
        "latency_median_ms",
        "latency_p95_ms",
        "peak_memory_mb",
        "checkpoint_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_markdown(path: Path, payload: dict) -> None:
    protocol = payload["protocol"]
    environment = payload["environment"]
    lines = [
        "# DecompJSCC 复杂度汇总",
        "",
        (
            f"口径：{environment['device_name']}，batch={protocol['batch_size']}，"
            f"{protocol['image_size']}x{protocol['image_size']}，FP32，SNR={protocol['snr_db']:g} dB，"
            f"warmup={protocol['warmup']}，iterations={protocol['iterations']}，不返回诊断 aux。"
        ),
        "",
        "| 角色 | rate | 参数量 (M) | 状态大小 (MB) | median (ms) | p95 (ms) | peak memory (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {role} | {rate} | {parameters_m:.3f} | {state_size_mb:.2f} | "
            "{latency_median_ms:.3f} | {latency_p95_ms:.3f} | {peak_memory_mb:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "说明：四个角色分别 strict-load 自身冻结 checkpoint；checkpoint 与单项 profile 的 SHA256 见 `summary.json`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize frozen DecompJSCC complexity profiles")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser

def summary_main() -> None:
    args = summary_parser().parse_args()
    payload = summarize(Path(args.input_dir))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(Path(args.output_csv), payload["rows"])
    write_markdown(Path(args.output_markdown), payload)
    print(json.dumps({"output_json": str(output_json), "rows": len(payload["rows"])}, indent=2))
