"""Evaluate paper models with explicit methods, conditions and output paths."""

import json

from experiments.evaluation import build_parser, execute


if __name__ == "__main__":
    print(json.dumps(execute(build_parser().parse_args()), indent=2, default=str))
