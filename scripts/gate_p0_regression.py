#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(rows: list[dict], metric_name: str) -> float | None:
    for row in rows:
        if str(row.get("metric")) == metric_name:
            try:
                return float(row.get("delta"))
            except Exception:
                return None
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P0 gate: equivalence first, speed second")
    p.add_argument("--baseline-run-dir", type=str, required=True)
    p.add_argument("--current-run-dir", type=str, required=True)
    p.add_argument("--baseline-report", type=str, required=True)
    p.add_argument("--current-report", type=str, required=True)
    p.add_argument("--baseline-results-root", type=str, required=True)
    p.add_argument("--current-results-root", type=str, required=True)
    p.add_argument("--primary-method", type=str, default="ours")
    p.add_argument("--reference-method", type=str, default="gradient")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--equivalence-tol", type=float, default=1e-6)
    return p


def main() -> None:
    args = build_parser().parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    eq_path = out.parent / "p0_equivalence.json"
    report_diff_path = out.parent / "p0_eval_report_diff.json"
    snap_base = out.parent / "p0_snapshot_baseline.json"
    snap_cur = out.parent / "p0_snapshot_current.json"
    snap_diff = out.parent / "p0_snapshot_diff.json"

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_explain_equivalence.py"),
            "--left-run-dir",
            args.baseline_run_dir,
            "--right-run-dir",
            args.current_run_dir,
            "--tolerance",
            str(float(args.equivalence_tol)),
            "--output-json",
            str(eq_path),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_report_diff.py"),
            "--left",
            args.baseline_report,
            "--right",
            args.current_report,
            "--left-name",
            "baseline",
            "--right-name",
            "current",
            "--output",
            str(report_diff_path),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis_snapshot.py"),
            "--results-root",
            args.baseline_results_root,
            "--primary-method",
            args.primary_method,
            "--reference-method",
            args.reference_method,
            "--output-json",
            str(snap_base),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis_snapshot.py"),
            "--results-root",
            args.current_results_root,
            "--primary-method",
            args.primary_method,
            "--reference-method",
            args.reference_method,
            "--output-json",
            str(snap_cur),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis_snapshot_diff.py"),
            "--baseline",
            str(snap_base),
            "--current",
            str(snap_cur),
            "--output-json",
            str(snap_diff),
        ]
    )

    eq = _read_json(eq_path)
    report_diff = _read_json(report_diff_path)
    snap_diff_payload = _read_json(snap_diff)

    eq_pass = bool(eq.get("passed", False))
    total_seconds_delta = _metric(snap_diff_payload.get("rows", []), "timing.total_seconds")
    speed_pass = (total_seconds_delta is not None and total_seconds_delta <= 0.0) if eq_pass else False

    payload = {
        "gate_name": "p0_regression",
        "equivalence_pass": eq_pass,
        "speed_pass": bool(speed_pass),
        "overall_pass": bool(eq_pass and speed_pass),
        "policy": {
            "order": ["equivalence", "speed"],
            "speed_rule": "timing.total_seconds delta <= 0",
        },
        "artifacts": {
            "equivalence": str(eq_path),
            "eval_report_diff": str(report_diff_path),
            "snapshot_baseline": str(snap_base),
            "snapshot_current": str(snap_cur),
            "snapshot_diff": str(snap_diff),
        },
        "summary": {
            "total_seconds_delta": total_seconds_delta,
            "eval_report_metrics_rows": report_diff.get("rows", []),
            "equivalence_first_failure": eq.get("first_failure"),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate-p0] overall_pass={payload['overall_pass']}")
    print(f"[gate-p0] output={out}")


if __name__ == "__main__":
    main()
