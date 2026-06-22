"""Command-line interface for the MVP pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crys_jepa_gnome_sc.io import load_candidates, write_ranked_candidates
from crys_jepa_gnome_sc.pipeline import screen_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crys-jepa-gnome-sc",
        description="Screen and rank GNoME-like superconducting crystal candidates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen", help="run the screening pipeline")
    screen.add_argument("input", type=Path, help="CSV or JSON candidate file")
    screen.add_argument("--out", type=Path, required=True, help="output CSV or JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "screen":
            candidates = load_candidates(args.input)
            ranked = screen_candidates(candidates)
            write_ranked_candidates(ranked, args.out)
            print(f"Wrote {len(ranked)} ranked candidates to {args.out}")
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1
