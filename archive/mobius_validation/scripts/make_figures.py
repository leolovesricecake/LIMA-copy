from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.utils import atomic_write_text, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Placeholder figure entry point for Mobius verification results.")
    parser.add_argument("--results-dir", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = resolve_project_path(
        args.results_dir,
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "exact_default",
    )
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        figures / "README.md",
        "Figure generation is intentionally separated from aggregation. "
        "Use aggregate/sample_level_metrics.csv and aggregate/recovery_metrics.csv as inputs.\n",
    )


if __name__ == "__main__":
    main()
