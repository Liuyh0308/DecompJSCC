"""Mechanism, complexity and existing-result analysis for the paper."""

import argparse
import importlib
import sys


COMMANDS = {
    "routing": ("mechanisms", "routing_main", "Stage depth weights and SNR intervention"),
    "crossstage": ("mechanisms", "crossstage_main", "Cross-stage source weights"),
    "rcsb": ("mechanisms", "rcsb_main", "Channel support and active-RMS diagnostics"),
    "profile": ("complexity", "comparison_main", "Compare parameter counts, FLOPs and latency"),
    "profile-role": ("complexity", "profile_main", "Profile one model or ablation"),
    "profile-summary": ("complexity", "summary_main", "Summarize existing profiles"),
    "summary": ("statistics", "summary_main", "Aggregate raw evaluation observations"),
    "interpolation": ("statistics", "interpolation_main", "Analyze unseen SNR/rate conditions"),
    "channel-transfer": ("statistics", "channel_main", "Compare AWGN and Rayleigh measurements"),
    "rayleigh": ("statistics", "rayleigh_main", "Compare zero-shot and matched channel training"),
    "resources": ("resources", "main", "Check or prepare local experiment directories"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS,
                        help="; ".join(f"{name}: {item[2]}" for name, item in COMMANDS.items()))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    module, function, _ = COMMANDS[args.command]
    old_argv = sys.argv
    try:
        sys.argv = [f"analyze.py {args.command}", *argv[1:]]
        getattr(importlib.import_module(f"experiments.{module}"), function)()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
