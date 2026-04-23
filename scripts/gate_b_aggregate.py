#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lima_llm.eval import (
    aggregate_gate_b_runs,
    collect_gate_b_runs,
    write_gate_b_aggregate_csv,
    write_gate_b_aggregate_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Gate B multi-seed reports")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--min-runs", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_json = Path(args.output_json) if args.output_json else (results_root / "gate_b_aggregate.json")
    output_csv = Path(args.output_csv) if args.output_csv else (results_root / "gate_b_aggregate.csv")

    runs = collect_gate_b_runs(results_root=results_root)
    payload = aggregate_gate_b_runs(runs=runs, min_runs=args.min_runs)
    write_gate_b_aggregate_json(payload=payload, output_path=output_json)
    write_gate_b_aggregate_csv(payload=payload, output_path=output_csv)

    print(
        f"[gate-b] root={results_root} runs={payload.get('run_count', 0)} groups={payload.get('group_count', 0)}"
    )
    print(f"[gate-b] json={output_json}")
    print(f"[gate-b] csv={output_csv}")


if __name__ == "__main__":
    main()
