"""Run zero-query representation, exact-structure, and interaction audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.representation import run_representation_audit


def _positive_ints(raw: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""

    values = sorted(
        {
            int(value.strip())
            for value in str(raw).split(",")
            if value.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "Expected one or more comma-separated positive integers."
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the representation-audit command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare finite-query Möbius/Fourier recovery, audit exact "
            "full-table structure, and evaluate full-table pair rankings "
            "without model queries."
        )
    )
    parser.add_argument("--mobius-run", required=True)
    parser.add_argument("--proxyspex-run", required=True)
    parser.add_argument("--heldout-audit", required=True)
    parser.add_argument("--output-root", default="results/audits")
    parser.add_argument("--compression-k", type=_positive_ints, default="1,2,4,8,16,32")
    parser.add_argument("--max-exact-features", type=int, default=9)
    parser.add_argument("--geometry-max-columns", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260730)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute one content-addressed representation audit."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    destination = run_representation_audit(
        args.mobius_run,
        args.proxyspex_run,
        args.heldout_audit,
        output_root=args.output_root,
        compression_k=args.compression_k,
        max_exact_features=args.max_exact_features,
        geometry_max_columns=args.geometry_max_columns,
        seed=args.seed,
        bootstrap=args.bootstrap,
        max_samples=args.max_samples,
    )
    print(f"[representation-audit-complete] output={destination}")


if __name__ == "__main__":
    main()
