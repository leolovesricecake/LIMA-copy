"""Build an exact all-pair deletion oracle for C and ProxySPEX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.exact_pairs import run_exact_pair_audit


def build_parser() -> argparse.ArgumentParser:
    """Build the exact pair-oracle command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Query full, singleton-deletion, and pair-deletion masks once per "
            "sample, then compare C and ProxySPEX with exact pair rankings."
        )
    )
    parser.add_argument("--mobius-run", required=True)
    parser.add_argument("--proxyspex-run", required=True)
    parser.add_argument("--output-root", default="results/audits")
    parser.add_argument(
        "--cache-path",
        default="results/.cache/value_oracle.sqlite3",
    )
    parser.add_argument("--device")
    parser.add_argument("--min-features", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260731)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one content-addressed exact pair oracle."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if int(args.min_features) < 2:
        raise ValueError("--min-features must be at least 2.")
    if int(args.max_features) < int(args.min_features):
        raise ValueError("--max-features must be at least --min-features.")
    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive.")
    destination = run_exact_pair_audit(
        args.mobius_run,
        args.proxyspex_run,
        output_root=args.output_root,
        cache_path=args.cache_path,
        device=args.device,
        min_features=args.min_features,
        max_features=args.max_features,
        max_samples=args.max_samples,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    print(f"[exact-pair-audit-complete] output={destination}")


if __name__ == "__main__":
    main()
