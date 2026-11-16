"""Unified training entry point for the paper models and fixed-CBR controls."""

import argparse
import sys


BACKENDS = ("source", "fixed", "adjscc", "snr_eq_jscc", "sa_ra_jscc")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s {source,fixed,adjscc,snr_eq_jscc,sa_ra_jscc} [options]",
        epilog="Use 'train.py <method> --help' for its parameters. See README.md for paper recipes.",
    )
    parser.add_argument("method", choices=BACKENDS)
    method = parser.parse_args(argv[:1]).method
    options = argv[1:]
    if method == "source":
        from experiments.source_training import main as run
    elif method == "fixed":
        from experiments.fixed_training import main as run
    else:
        from external_baselines.unified_runtime import run_training
        return run_training(method, options)
    return run(options)


if __name__ == "__main__":
    main()
