"""Collect direct paper tables for one cell or aggregate one paper version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.paper_results import (
    collect_paper_cell,
    summarize_paper_version,
)


def _q_values(raw: str) -> list[int]:
    """Parse a nonempty comma-separated percentage grid."""

    values = sorted(
        {
            int(value.strip())
            for value in str(raw).split(",")
            if value.strip()
        }
    )
    if not values or any(value <= 0 or value > 100 for value in values):
        raise argparse.ArgumentTypeError(
            "q values must be comma-separated percentages in 1..100."
        )
    return values


def _parse_runs(values: Sequence[str]) -> Dict[str, str]:
    """Parse repeated ROLE=PATH arguments while rejecting duplicate roles."""

    output: Dict[str, str] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"Run must use ROLE=PATH syntax: {value!r}")
        role, path = str(value).split("=", 1)
        normalized = role.strip().upper()
        if not normalized or not path.strip():
            raise ValueError(f"Run must use nonempty ROLE=PATH: {value!r}")
        if normalized in output:
            raise ValueError(f"Duplicate run role: {normalized}")
        output[normalized] = path.strip()
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the direct paper-result collection CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate one dataset/model/seed paper cell or aggregate all cells "
            "for one explicit paper version."
        )
    )
    parser.add_argument("--paper-version", required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--audit-dir", action="append", default=[])
    parser.add_argument("--output-root", default="results/paper")
    parser.add_argument("--q-values", type=_q_values, default="5,10,20,50")
    parser.add_argument("--analysis-seed", type=int, default=260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Collect one paper cell or aggregate an existing paper version."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.summarize:
        if args.run or args.audit_dir:
            raise ValueError(
                "--summarize cannot be combined with run or audit inputs."
            )
        destination = summarize_paper_version(
            args.paper_version,
            output_root=args.output_root,
        )
        print(f"[paper-results-summarized] output={destination}")
        return
    if not args.run:
        raise ValueError(
            "Paper cell collection requires repeated --run ROLE=PATH inputs."
        )
    destination = collect_paper_cell(
        args.paper_version,
        _parse_runs(args.run),
        args.audit_dir,
        output_root=args.output_root,
        q_values=args.q_values,
        analysis_seed=args.analysis_seed,
        bootstrap=args.bootstrap,
        overwrite=args.overwrite,
    )
    print(f"[paper-cell-complete] output={destination}")


if __name__ == "__main__":
    main()
