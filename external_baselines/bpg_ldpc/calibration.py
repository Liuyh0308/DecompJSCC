"""Independent random-bit AMC qualification for a 1% channel-frame error target."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "exp_log/decompjscc/analysis/awgn/bpg_reliable_amc_m8_i5_20261104"
SNRS = (1., 2.5, 4., 5.5, 7., 8.5, 10., 11.5, 13.)
MAX_CHANNEL_USE_BUDGET = 1376256
TARGET_IMAGE_FAILURE = .01
FAMILY_ALPHA = .05
FAMILY_SIZE = 9 * 16
ALPHA_PER_CANDIDATE = FAMILY_ALPHA / FAMILY_SIZE
CHUNK_BLOCKS = 64
SEED = 20261103
CODE_FILES = ("external_baselines/bpg_ldpc/calibration.py",
              "external_baselines/bpg_ldpc/digital_link.py",
              "external_baselines/bpg_ldpc/mcs.py",
              "experiments/evaluation.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protocol_settings():
    return {
        "target_image_failure_probability": TARGET_IMAGE_FAILURE,
        "confidence_familywise": 1 - FAMILY_ALPHA,
        "max_channel_use_budget": MAX_CHANNEL_USE_BUDGET,
        "qualification": "zero_error_fixed_sample",
        "candidate_family_size": FAMILY_SIZE,
        "alpha_per_candidate": ALPHA_PER_CANDIDATE,
        "chunk_blocks": CHUNK_BLOCKS,
        "scope": "channel_error_only; budget and codec failures remain scored separately",
        "frame_bound": "union_bound_Bmax_times_p; no independent-frame-block assumption required",
        "acceptance": "zero CRC and payload errors over the fixed required sample; early rejection only",
        "geometry_from_test_images": True,
        "test_image_pixels_or_metrics_used_for_selection": False,
        "assumptions": "Independent random valid payload blocks and AWGN in the calibrated link; statistical qualification is not a deterministic guarantee for each BPG bitstream or padded final block",
    }


def max_code_blocks(mcs):
    return MAX_CHANNEL_USE_BUDGET * int(mcs["bits_per_symbol"]) // int(mcs["coded_block_bits"])


def required_trials(blocks, target=TARGET_IMAGE_FAILURE, alpha=ALPHA_PER_CANDIDATE):
    if not isinstance(blocks, int) or blocks <= 0 or not 0 < target < 1 or not 0 < alpha < 1:
        raise ValueError("Expected positive frame blocks and probabilities within (0,1)")
    return math.ceil(math.log(alpha) / math.log1p(-target / blocks))


def zero_error_upper(trials, alpha=ALPHA_PER_CANDIDATE):
    if trials <= 0:
        return 1.
    return -math.expm1(math.log(alpha) / trials)


def candidate_grid():
    from external_baselines.bpg_ldpc.mcs import candidate_mcs
    return [mcs.to_dict() for mcs in candidate_mcs()]


def candidate_order(grid):
    return sorted(grid, key=lambda m: (-m["spectral_efficiency"], m["bits_per_symbol"]))


def candidate_plan(mcs):
    blocks = max_code_blocks(mcs)
    return {**mcs, "max_code_blocks": blocks, "block_failure_limit": TARGET_IMAGE_FAILURE / blocks,
            "required_trials": required_trials(blocks), "alpha": ALPHA_PER_CANDIDATE,
            "trials": 0, "crc_failed_blocks": 0, "payload_bit_errors": 0,
            "bler": None, "block_error_upper_bound": None, "image_error_upper_bound": None,
            "qualified": False, "status": "not_tested_after_selection", "seconds": 0., "chunks": 0}


def chunk_seed(seed, snr, name, chunk_index, purpose):
    key = f"reliable-amc-v1|{int(seed)}|{float(snr)}|{name}|{chunk_index}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def qualify_candidate(plan, snr, seed, run_batch, emit):
    row = dict(plan)
    row["status"] = "running"
    while row["trials"] < row["required_trials"]:
        count = min(CHUNK_BLOCKS, row["required_trials"] - row["trials"])
        index = row["chunks"]
        payload_seed = chunk_seed(seed, snr, row["name"], index, "payload")
        noise_seed = chunk_seed(seed, snr, row["name"], index, "noise")
        batch = run_batch(row, snr, count, payload_seed, noise_seed)
        stats = batch["stats"]
        if stats["num_code_blocks"] != count:
            raise ValueError("Actual link block count differs from the requested sample")
        row["trials"] += count
        row["chunks"] += 1
        row["crc_failed_blocks"] += stats["crc_failed_blocks"]
        row["payload_bit_errors"] += stats["bit_errors"]
        row["seconds"] += batch["seconds"]
        row["bler"] = row["crc_failed_blocks"] / row["trials"]
        failed = row["crc_failed_blocks"] > 0 or row["payload_bit_errors"] > 0
        if failed:
            row["status"] = "rejected_observed_error"
        elif row["trials"] == row["required_trials"]:
            row["qualified"] = True
            row["status"] = "qualified"
            row["block_error_upper_bound"] = zero_error_upper(row["trials"])
            row["image_error_upper_bound"] = row["max_code_blocks"] * row["block_error_upper_bound"]
        emit({"snr_db": snr, "mcs": row["name"], "chunk_index": index,
              "payload_seed": payload_seed, "noise_seed": noise_seed,
              "requested_blocks": count, **batch}, dict(row))
        if failed:
            break
    return row


def validate_calibration(data, *, require_all_snrs=True):
    if (data.get("protocol") != "decompjscc_bpg_reliable_amc_m8_i5_v1"
            or data.get("reliability") != protocol_settings()
            or data.get("decoder_iterations") != 5
            or data.get("calibration_data") != "fixed_random_bits_only"
            or data.get("test_images_used_for_selection") is not False
            or data.get("test_image_pixels_or_metrics_used_for_selection") is not False
            or data.get("geometry_from_test_images") is not True):
        raise ValueError("Unexpected reliability calibration protocol")
    grid = candidate_grid()
    if data.get("candidate_grid") != grid:
        raise ValueError("Candidate grid changed")
    results = data.get("results", [])
    snrs = [r["snr_db"] for r in results]
    if len(snrs) != len(set(snrs)) or not set(snrs).issubset(SNRS):
        raise ValueError("Duplicate or unsupported SNRs")
    if (require_all_snrs and set(snrs) != set(SNRS)) or (not require_all_snrs and len(snrs) != 1):
        raise ValueError("Incomplete calibration SNR inventory")
    if set(data.get("selection", {})) != {str(float(snr)) for snr in snrs}:
        raise ValueError("Selection SNR inventory differs")
    for result in results:
        candidates = result["candidates"]
        ordered = candidate_order(grid)
        if [r["name"] for r in candidates] != [r["name"] for r in ordered]:
            raise ValueError("Candidate priority/order differs")
        selected = None
        for row, mcs in zip(candidates, ordered):
            plan = candidate_plan(mcs)
            if any(row.get(k) != plan[k] for k in (*mcs, "max_code_blocks", "block_failure_limit",
                                                  "required_trials", "alpha")):
                raise ValueError("Candidate dimensions or fixed qualification sample changed")
            count = row["trials"]
            if not isinstance(count, int) or not 0 <= count <= plan["required_trials"]:
                raise ValueError("Invalid tested block count")
            if (type(row["crc_failed_blocks"]) is not int or not 0 <= row["crc_failed_blocks"] <= count
                    or type(row["payload_bit_errors"]) is not int
                    or not 0 <= row["payload_bit_errors"] <= count * 8328):
                raise ValueError("Impossible calibration error counts")
            if row["bler"] != (row["crc_failed_blocks"] / count if count else None):
                raise ValueError("Observed BLER does not match calibration counts")
            failed = row["crc_failed_blocks"] > 0 or row["payload_bit_errors"] > 0
            if selected is not None:
                if count != 0 or row["status"] != "not_tested_after_selection" or row["qualified"]:
                    raise ValueError("Candidates after the first qualified one must remain untested")
            elif row["qualified"]:
                if count != plan["required_trials"] or failed or row["status"] != "qualified":
                    raise ValueError("Premature or errored qualification")
                upper = zero_error_upper(count)
                if (row["block_error_upper_bound"] != upper
                        or row["image_error_upper_bound"] != upper * row["max_code_blocks"]
                        or row["image_error_upper_bound"] > TARGET_IMAGE_FAILURE):
                    raise ValueError("Invalid error probability certificate")
                selected = row
            elif count <= 0 or not failed or row["status"] != "rejected_observed_error":
                raise ValueError("An earlier candidate was neither qualified nor explicitly rejected")
        threshold_met = selected is not None
        if selected is None:
            selected = next(r for r in candidates if r["name"] == "4QAM_R1_3")
        if (result["threshold_met"] is not threshold_met or result["selected_mcs"] != selected
                or data["selection"][str(float(result["snr_db"]))] != selected):
            raise ValueError("Selection/fallback is not determined by qualification")
    return data


def verify_audit(data):
    audit_path = Path(data["audit_path"])
    if sha256(audit_path) != data["audit_sha256"]:
        raise ValueError("Calibration audit SHA mismatch")
    result = data["results"][0]
    snr = result["snr_db"]
    candidates = {r["name"]: r for r in result["candidates"]}
    totals = {name: {"trials": 0, "chunks": 0, "crc_failed_blocks": 0,
                     "payload_bit_errors": 0, "seconds": 0.} for name in candidates}
    with audit_path.open() as stream:
        for line in stream:
            batch = json.loads(line)
            name, index = batch["mcs"], batch["chunk_index"]
            total, candidate = totals[name], candidates[name]
            if (batch["snr_db"] != snr or index != total["chunks"]
                    or batch["payload_seed"] != chunk_seed(data["seed"], snr, name, index, "payload")
                    or batch["noise_seed"] != chunk_seed(data["seed"], snr, name, index, "noise")
                    or total["crc_failed_blocks"] or total["payload_bit_errors"]):
                raise ValueError("Audit seed/order differs or continues after rejection")
            count = min(CHUNK_BLOCKS, candidate["required_trials"] - total["trials"])
            stats = batch["stats"]
            if (count <= 0 or batch["requested_blocks"] != count
                    or stats["num_code_blocks"] != count
                    or stats["payload_bits"] != count * 8328
                    or stats["information_block_bits"] != 8352
                    or stats["coded_block_bits"] != candidate["coded_block_bits"]
                    or stats["bits_per_symbol"] != candidate["bits_per_symbol"]
                    or stats["decoder_iterations"] != 5 or stats["crc_bits"] != 24
                    or stats["base_graph"] not in ("bg1", "bg2") or stats["lifting_size"] <= 0):
                raise ValueError("Audit does not represent the required finite 5G LDPC samples")
            if (not math.isfinite(batch["seconds"]) or batch["seconds"] < 0
                    or not 0 <= stats["crc_failed_blocks"] <= count
                    or not 0 <= stats["bit_errors"] <= count * 8328
                    or stats["decode_success"] != (stats["crc_failed_blocks"] == stats["bit_errors"] == 0)):
                raise ValueError("Invalid calibration timing/error records")
            for field in ("payload_unpacked_bits_sha256", "decoded_unpacked_bits_sha256"):
                if len(batch[field]) != 64 or any(c not in "0123456789abcdef" for c in batch[field]):
                    raise ValueError("Invalid calibration bit-vector SHA")
            total["trials"] += count
            total["chunks"] += 1
            total["crc_failed_blocks"] += stats["crc_failed_blocks"]
            total["payload_bit_errors"] += stats["bit_errors"]
            total["seconds"] += batch["seconds"]
    for name, total in totals.items():
        if any(candidates[name][key] != value for key, value in total.items()):
            raise ValueError("Candidate qualification totals do not match the actual link audit")
    return audit_path


def merge_calibration(output_root):
    folder = Path(output_root) / "calibration"
    output = folder / "mcs.json"
    if output.exists():
        raise FileExistsError(output)
    parts, sources = [], {}
    for snr in SNRS:
        path = folder / f"snr{snr:g}.json"
        part = json.loads(path.read_text())
        validate_calibration(part, require_all_snrs=False)
        if part.get("status") != "complete" or part["results"][0]["snr_db"] != snr or part["seed"] != SEED:
            raise ValueError("Wrong/incomplete single-SNR calibration")
        audit_path = verify_audit(part)
        if audit_path.resolve() != path.with_name(path.stem + "_audit.jsonl").resolve():
            raise ValueError("Unexpected calibration audit location")
        for name, expected in part["code_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Frozen calibration code changed")
        sources[str(path)] = sha256(path)
        sources[str(audit_path)] = part["audit_sha256"]
        parts.append(part)
    first = parts[0]
    combined = {key: first[key] for key in (
        "status", "method", "protocol", "calibration_data", "test_images_used_for_selection",
        "test_image_pixels_or_metrics_used_for_selection", "geometry_from_test_images", "geometry_note",
        "reliability", "seed", "decoder_iterations", "encoder_mode", "crc", "snr_definition",
        "sionna_version", "candidate_grid", "note")}
    combined.update(results=[part["results"][0] for part in parts],
                    selection={key: value for part in parts for key, value in part["selection"].items()},
                    source_sha256=sources, code_sha256=first["code_sha256"], completed_at=now(),
                    all_snr_thresholds_met=all(part["results"][0]["threshold_met"] for part in parts))
    validate_calibration(combined)
    write_json(output, combined)
    return {"path": str(output), "sha256": sha256(output),
            "selection": {snr: row["name"] for snr, row in combined["selection"].items()},
            "all_snr_thresholds_met": combined["all_snr_thresholds_met"]}


def write_json(path, data, *, update=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if update:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    else:
        with path.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")


def make_runner(device_name):
    import torch
    from external_baselines.bpg_ldpc.digital_link import run_5g_ldpc_awgn
    from experiments.evaluation import require_idle_gpu
    device = torch.device(device_name)
    require_idle_gpu(device)

    def run_batch(mcs, snr, count, payload_seed, noise_seed):
        require_idle_gpu(device)
        generator = torch.Generator(device="cpu").manual_seed(payload_seed)
        payload = torch.randint(0, 2, (count * 8328,), generator=generator, dtype=torch.uint8)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = run_5g_ldpc_awgn(
                payload, information_block_bits=8352, coded_block_bits=mcs["coded_block_bits"],
                bits_per_symbol=mcs["bits_per_symbol"], snr_db=snr, seed=noise_seed,
                decoder_iterations=5, crc_bits=24, device=device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        stats = result.stats.to_dict()
        if (stats["information_block_bits"] != 8352 or stats["decoder_iterations"] != 5
                or stats["crc_bits"] != 24 or stats["base_graph"] not in ("bg1", "bg2")
                or stats["lifting_size"] <= 0):
            raise ValueError("Actual finite-5G-LDPC link differs from the frozen protocol")
        return {"stats": stats, "seconds": elapsed,
                "payload_unpacked_bits_sha256": hashlib.sha256(payload.numpy().tobytes()).hexdigest(),
                "decoded_unpacked_bits_sha256": hashlib.sha256(
                    result.decoded_bits.cpu().numpy().tobytes()).hexdigest()}

    return run_batch, {"logical_device": str(device), "cpu_affinity": sorted(os.sched_getaffinity(0))}


def run(args):
    output = Path(args.output) if args.output else DEFAULT_ROOT / "calibration" / f"snr{args.snr_db:g}.json"
    output = output.resolve()
    audit_path = output.with_name(output.stem + "_audit.jsonl")
    progress_path = output.with_name(output.stem + "_progress.json")
    if any(path.exists() for path in (output, audit_path, progress_path)):
        raise FileExistsError("Existing calibration output/audit/progress is never overwritten")
    grid = candidate_grid()
    plans = [candidate_plan(mcs) for mcs in candidate_order(grid)]
    sources = {name: sha256(ROOT / name) for name in CODE_FILES}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    state = {"status": "running", "started_at": started, "snr_db": args.snr_db,
             "output": str(output), "audit": str(audit_path), "smoke": args.smoke}
    write_json(progress_path, state)
    try:
        run_batch, hardware = make_runner(args.device)
        if args.smoke:
            mcs = next(row for row in grid if row["name"] == "4QAM_R1_3")
            payload_seed = chunk_seed(args.seed, args.snr_db, mcs["name"], 0, "smoke_payload")
            noise_seed = chunk_seed(args.seed, args.snr_db, mcs["name"], 0, "smoke_noise")
            batch = run_batch(mcs, args.snr_db, CHUNK_BLOCKS, payload_seed, noise_seed)
            result = {"status": "smoke_complete", "selection_produced": False,
                      "snr_db": args.snr_db, "candidate": mcs, "payload_seed": payload_seed,
                      "noise_seed": noise_seed, "blocks": CHUNK_BLOCKS,
                      "seconds_per_block": batch["seconds"] / CHUNK_BLOCKS,
                      "hardware": hardware, "code_sha256": sources, **batch}
            write_json(output, result)
            state.update(status="smoke_complete", seconds_per_block=result["seconds_per_block"])
        else:
            last_print = time.monotonic()
            with audit_path.open("x") as audit:
                def emit(batch, candidate):
                    nonlocal last_print
                    audit.write(json.dumps(batch, allow_nan=False) + "\n")
                    audit.flush()
                    state.update(updated_at=now(), current_candidate=candidate)
                    write_json(progress_path, state, update=True)
                    if candidate["status"] != "running" or time.monotonic() - last_print >= 60:
                        print(json.dumps({"snr_db": args.snr_db, "candidate": candidate}), flush=True)
                        last_print = time.monotonic()
                for index, plan in enumerate(plans):
                    plans[index] = qualify_candidate(plan, args.snr_db, args.seed, run_batch, emit)
                    if plans[index]["qualified"]:
                        break
            qualified = [row for row in plans if row["qualified"]]
            selected = qualified[0] if qualified else next(row for row in plans if row["name"] == "4QAM_R1_3")
            row = {"snr_db": args.snr_db, "threshold_met": bool(qualified),
                   "selected_mcs": selected, "candidates": plans}
            result = {
                "status": "complete", "method": "BPG+5G_LDPC",
                "protocol": "decompjscc_bpg_reliable_amc_m8_i5_v1",
                "calibration_data": "fixed_random_bits_only", "test_images_used_for_selection": False,
                "test_image_pixels_or_metrics_used_for_selection": False,
                "geometry_from_test_images": True,
                "geometry_note": "Only maximum channel-use budget from frozen image dimensions and CBRs is used",
                "reliability": protocol_settings(), "seed": args.seed, "decoder_iterations": 5,
                "encoder_mode": 8, "crc": "CRC24A per information block",
                "snr_definition": "complex-symbol Es/N0", "sionna_version": importlib.metadata.version("sionna"),
                "candidate_grid": grid, "results": [row], "selection": {str(float(args.snr_db)): selected},
                "hardware": hardware, "code_sha256": sources,
                "audit_path": str(audit_path), "audit_sha256": sha256(audit_path),
                "started_at": started, "completed_at": now(),
                "note": "Post-hoc reliability redesign after observing earlier image failures; no smoothness or network-win acceptance criterion",
            }
            validate_calibration(result, require_all_snrs=False)
            if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
                raise ValueError("Calibration code changed during the run")
            write_json(output, result)
            state.update(status="complete", selected_mcs=selected["name"], threshold_met=bool(qualified))
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["updated_at"] = now()
        write_json(progress_path, state, update=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-db", type=float, choices=SNRS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--smoke", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.run or args.smoke:
        result = run(args)
    else:
        result = {"status": "dry_run", "snr_db": args.snr_db, "reliability": protocol_settings(),
                  "candidate_plans": [candidate_plan(mcs) for mcs in candidate_order(candidate_grid())]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
