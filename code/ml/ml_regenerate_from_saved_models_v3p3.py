#!/usr/bin/env python3
"""Regenerate v3p3 full-record products from validated saved CatBoost models.

This thin entry point deliberately delegates to the sole active ML workflow so
the physical-event identity, post-prediction resolution, and output schemas
cannot drift between fitting and saved-model regeneration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_catboost_conformal_loyo_v3p3_physical_event import main as ml_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--impute-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate ML figures after reconstructing saved-model products.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    forwarded = [
        str(Path(__file__).with_name("ml_catboost_conformal_loyo_v3p3_physical_event.py")),
        "--repo", str(repo),
        "--impute_only",
        "--impute_draws", str(args.impute_draws),
        "--seed", str(args.seed),
    ]
    if args.output_dir:
        forwarded.extend(["--output_dir", args.output_dir])
    if not args.figures:
        forwarded.append("--no-figures")
    sys.argv = forwarded
    ml_main()


if __name__ == "__main__":
    main()
