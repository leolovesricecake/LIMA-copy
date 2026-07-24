"""Compare two target-compatible schema-v2 runs."""

from __future__ import annotations

import argparse
import json

from mobius.evaluation.compare import compare_runs


def build_parser() -> argparse.ArgumentParser:
    """Build the run-comparison parser."""

    parser = argparse.ArgumentParser(description="Compare two evaluated runs.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Print or save right-minus-left metric deltas."""

    args = build_parser().parse_args(argv)
    report = compare_runs(args.left, args.right)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

