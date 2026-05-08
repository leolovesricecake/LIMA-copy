#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lima_llm.eval.equivalence import compare_run_dirs


def _parse_ignored_paths(raw_values):
    ignored = []
    for raw in raw_values:
        parts = tuple(part for part in raw.strip().split(".") if part != "")
        if parts:
            ignored.append(parts)
    return tuple(ignored)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Gate C run directories for strict equivalence")
    parser.add_argument("--reference-run-dir", type=str, required=True)
    parser.add_argument("--candidate-run-dir", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--ignore-path",
        action="append",
        default=[
            "metrics_secondary.runtime_seconds",
            "metrics_secondary.forward_counters_delta",
        ],
        help="Dot-path in eval_report.json to ignore. Can be repeated.",
    )
    parser.add_argument("--expected-sample-count", type=int, default=None)
    parser.add_argument(
        "--prereq-report",
        type=str,
        default=None,
        help="Optional report path from a previous gate. Must contain {'passed': true}.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.prereq_report:
        prereq_path = Path(args.prereq_report)
        if not prereq_path.exists():
            raise SystemExit(f"[gate-c-compare] missing prereq report: {prereq_path}")
        prereq = json.loads(prereq_path.read_text(encoding="utf-8"))
        if not bool(prereq.get("passed", False)):
            raise SystemExit(f"[gate-c-compare] prereq gate did not pass: {prereq_path}")

    ignored_paths = _parse_ignored_paths(args.ignore_path)
    report = compare_run_dirs(
        reference_run_dir=Path(args.reference_run_dir),
        candidate_run_dir=Path(args.candidate_run_dir),
        tolerance=float(args.tolerance),
        ignored_eval_paths=ignored_paths,
    )
    if args.expected_sample_count is not None and bool(report.get("passed", False)):
        got = int(report.get("sample_check", {}).get("sample_count", -1))
        want = int(args.expected_sample_count)
        if got != want:
            report = {
                **report,
                "passed": False,
                "reason": "sample_count_mismatch",
                "expected_sample_count": want,
                "actual_sample_count": got,
            }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"[gate-c-compare] wrote {output_path}")

    print(payload)
    if not bool(report.get("passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
