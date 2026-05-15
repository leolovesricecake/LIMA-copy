#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _aggregate_snapshot_metric(
    rows: List[dict],
    metric_name: str,
    method: str | None = None,
) -> Dict[str, Any] | None:
    matched = []
    for row in rows:
        if str(row.get("metric")) != metric_name:
            continue
        if method is not None and str(row.get("method")) != str(method):
            continue
        matched.append(row)
    if not matched:
        return None

    baseline = 0.0
    current = 0.0
    for row in matched:
        baseline += _safe_float(row.get("baseline"))
        current += _safe_float(row.get("current"))
    delta = current - baseline
    return {
        "metric": metric_name,
        "method": method,
        "matched_rows": len(matched),
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "improvement_ratio": ((baseline - current) / baseline) if abs(baseline) > 1e-12 else None,
    }


def _select_git_info(run_cfg: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        run_cfg.get("provenance", {}).get("git", {}),
        report.get("provenance", {}).get("git", {}),
    ]
    for git in candidates:
        if not isinstance(git, dict):
            continue
        if git.get("commit") or git.get("commit_short") or git.get("dirty") is not None:
            return git
    return {}


def _provenance_checks(
    baseline_run_cfg: Dict[str, Any],
    current_run_cfg: Dict[str, Any],
    baseline_report: Dict[str, Any],
    current_report: Dict[str, Any],
) -> Dict[str, Any]:
    checks = {
        "device_match": str(baseline_run_cfg.get("device")) == str(current_run_cfg.get("device")),
        "deterministic_match": bool(baseline_run_cfg.get("deterministic")) == bool(current_run_cfg.get("deterministic")),
        "eval_granularity_match": str(baseline_run_cfg.get("eval_granularity", "token")) == str(
            current_run_cfg.get("eval_granularity", "token")
        ),
    }

    git_base = _select_git_info(baseline_run_cfg, baseline_report)
    git_cur = _select_git_info(current_run_cfg, current_report)
    checks["git_commit_available"] = bool(git_base.get("commit")) and bool(git_cur.get("commit"))
    checks["git_dirty_available"] = git_base.get("dirty") is not None and git_cur.get("dirty") is not None
    checks["git_commit_match"] = str(git_base.get("commit")) == str(git_cur.get("commit"))
    checks["git_dirty_match"] = bool(git_base.get("dirty")) == bool(git_cur.get("dirty"))

    return {
        "passed": all(bool(v) for v in checks.values()),
        "checks": checks,
        "baseline": {
            "device": baseline_run_cfg.get("device"),
            "deterministic": baseline_run_cfg.get("deterministic"),
            "eval_granularity": baseline_run_cfg.get("eval_granularity", "token"),
            "git": git_base,
        },
        "current": {
            "device": current_run_cfg.get("device"),
            "deterministic": current_run_cfg.get("deterministic"),
            "eval_granularity": current_run_cfg.get("eval_granularity", "token"),
            "git": git_cur,
        },
    }


def _metric_drift_checks(
    report_diff_rows: List[dict],
    tol: float,
) -> Dict[str, Any]:
    tracked = [
        "metrics_primary.log_odds",
        "metrics_primary.comprehensiveness",
        "metrics_primary.sufficiency",
        "metrics_primary.aopc",
    ]
    values = []
    checks: Dict[str, bool] = {}
    for key in tracked:
        row = next((r for r in report_diff_rows if str(r.get("metric")) == key), None)
        if row is None:
            checks[key] = False
            values.append({"metric": key, "abs_diff": None, "delta": None, "present": False})
            continue
        abs_diff = _safe_float(row.get("abs_diff"))
        checks[key] = abs_diff <= tol
        values.append(
            {
                "metric": key,
                "abs_diff": abs_diff,
                "delta": _safe_float(row.get("delta")),
                "present": True,
            }
        )
    return {
        "passed": all(bool(v) for v in checks.values()),
        "tolerance": float(tol),
        "checks": checks,
        "values": values,
    }


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
    p.add_argument("--equivalence-tol", type=float, default=5e-4)
    p.add_argument("--speed-improve-ratio", type=float, default=0.2)
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
    baseline_run_cfg = _read_json(Path(args.baseline_run_dir) / "run_config.json")
    current_run_cfg = _read_json(Path(args.current_run_dir) / "run_config.json")
    baseline_report = _read_json(Path(args.baseline_report))
    current_report = _read_json(Path(args.current_report))

    explain_equivalence_pass = bool(eq.get("passed", False))
    provenance = _provenance_checks(
        baseline_run_cfg=baseline_run_cfg,
        current_run_cfg=current_run_cfg,
        baseline_report=baseline_report,
        current_report=current_report,
    )
    metric_drift = _metric_drift_checks(
        report_diff_rows=list(report_diff.get("rows", [])),
        tol=float(args.equivalence_tol),
    )
    equivalence_pass = bool(explain_equivalence_pass and provenance["passed"] and metric_drift["passed"])

    speed_info = _aggregate_snapshot_metric(
        rows=list(snap_diff_payload.get("rows", [])),
        metric_name="timing.total_seconds",
        method=str(args.primary_method),
    )
    speed_pass = False
    if equivalence_pass and speed_info is not None:
        improvement = speed_info.get("improvement_ratio")
        speed_pass = (improvement is not None) and (float(improvement) >= float(args.speed_improve_ratio))

    payload = {
        "gate_name": "p0_regression",
        "equivalence_pass": bool(equivalence_pass),
        "speed_pass": bool(speed_pass),
        "overall_pass": bool(equivalence_pass and speed_pass),
        "policy": {
            "order": ["equivalence", "speed"],
            "equivalence_rule": "sample equivalence + provenance match + primary metric abs_diff <= tolerance",
            "speed_rule": f"timing.total_seconds improvement_ratio >= {float(args.speed_improve_ratio):.3f}",
        },
        "artifacts": {
            "equivalence": str(eq_path),
            "eval_report_diff": str(report_diff_path),
            "snapshot_baseline": str(snap_base),
            "snapshot_current": str(snap_cur),
            "snapshot_diff": str(snap_diff),
        },
        "summary": {
            "explain_equivalence_pass": explain_equivalence_pass,
            "provenance": provenance,
            "metric_drift": metric_drift,
            "speed": speed_info,
            "eval_report_metrics_rows": report_diff.get("rows", []),
            "equivalence_first_failure": eq.get("first_failure"),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate-p0] overall_pass={payload['overall_pass']}")
    print(f"[gate-p0] output={out}")


if __name__ == "__main__":
    main()
