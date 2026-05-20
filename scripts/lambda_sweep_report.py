#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_samples(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return out
    for path in sorted(sample_dir.glob("*.json")):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        sample_id = str(payload.get("sample_id", path.stem))
        out[sample_id] = payload
    return out


def _explain_seconds_total(samples: Dict[str, Dict[str, Any]]) -> float:
    total = 0.0
    for payload in samples.values():
        total += _safe_float(payload.get("metadata", {}).get("elapsed_seconds"), 0.0)
    return total


def _gold_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    gold = report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    return {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comp": _safe_float(gold.get("comprehensiveness")),
        "suff": _safe_float(gold.get("sufficiency")),
        "a_c": _safe_float(gold.get("aopc_comprehensiveness")),
        "a_s": _safe_float(gold.get("aopc_sufficiency")),
    }


def _directional_gain(metric: str, baseline: float, current: float) -> float:
    # Positive means better under the chosen policy:
    # log_odds↑, comp↑, suff↓, a-c↑, a-s↓
    if metric in {"suff", "a_s"}:
        return baseline - current
    return current - baseline


def _oom_shrink_events(report: Dict[str, Any]) -> int:
    sec = report.get("metrics_secondary", {})
    return int(_safe_float(sec.get("forward_counters_delta", {}).get("oom_shrink_events"), 0.0))


def _failed_samples(report: Dict[str, Any]) -> int:
    sec = report.get("metrics_secondary", {})
    diag = sec.get("method_diagnostics", {})
    return int(_safe_float(diag.get("failed_samples"), 0.0))


@dataclass
class StabilityDiff:
    compared_samples: int
    selected_changed: int
    ranking_changed: int
    trace_changed: int
    trace_total_score_max_abs_diff: float

    def to_dict(self) -> Dict[str, Any]:
        denom = max(1, self.compared_samples)
        return {
            "compared_samples": int(self.compared_samples),
            "selected_changed": int(self.selected_changed),
            "ranking_changed": int(self.ranking_changed),
            "trace_changed": int(self.trace_changed),
            "selected_changed_ratio": float(self.selected_changed / denom),
            "ranking_changed_ratio": float(self.ranking_changed / denom),
            "trace_changed_ratio": float(self.trace_changed / denom),
            "trace_total_score_max_abs_diff": float(self.trace_total_score_max_abs_diff),
        }


def _sample_stability_diff(
    baseline_samples: Dict[str, Dict[str, Any]],
    current_samples: Dict[str, Dict[str, Any]],
    trace_tol: float,
) -> StabilityDiff:
    shared_ids = sorted(set(baseline_samples.keys()).intersection(current_samples.keys()))
    selected_changed = 0
    ranking_changed = 0
    trace_changed = 0
    trace_max = 0.0

    for sample_id in shared_ids:
        left = baseline_samples[sample_id]
        right = current_samples[sample_id]

        if left.get("selected_chunk_ids") != right.get("selected_chunk_ids"):
            selected_changed += 1
        if left.get("chunk_ranking") != right.get("chunk_ranking"):
            ranking_changed += 1

        left_trace = left.get("trace", [])
        right_trace = right.get("trace", [])
        local_max = 0.0
        for lrow, rrow in zip(left_trace, right_trace):
            d = abs(_safe_float(lrow.get("total_score")) - _safe_float(rrow.get("total_score")))
            if d > local_max:
                local_max = d
        if local_max > trace_tol:
            trace_changed += 1
        if local_max > trace_max:
            trace_max = local_max

    return StabilityDiff(
        compared_samples=len(shared_ids),
        selected_changed=selected_changed,
        ranking_changed=ranking_changed,
        trace_changed=trace_changed,
        trace_total_score_max_abs_diff=trace_max,
    )


def _build_row(
    baseline_run_dir: Path,
    current_run_dir: Path,
    metric_tol: float,
    trace_tol: float,
) -> Dict[str, Any]:
    base_report = _read_json(baseline_run_dir / "eval_report.json")
    cur_report = _read_json(current_run_dir / "eval_report.json")
    base_cfg = _read_json(baseline_run_dir / "run_config.json")
    cur_cfg = _read_json(current_run_dir / "run_config.json")
    base_samples = _load_samples(baseline_run_dir)
    cur_samples = _load_samples(current_run_dir)

    b = _gold_metrics(base_report)
    c = _gold_metrics(cur_report)

    direction = {key: _directional_gain(key, b[key], c[key]) for key in b}
    improved = {key: (val > float(metric_tol)) for key, val in direction.items()}
    regressed = {key: (val < -float(metric_tol)) for key, val in direction.items()}
    improved_count = sum(1 for v in improved.values() if v)
    regressed_count = sum(1 for v in regressed.values() if v)

    stability = _sample_stability_diff(base_samples, cur_samples, trace_tol=trace_tol)
    base_eval_seconds = _safe_float(base_report.get("metrics_secondary", {}).get("runtime_seconds"))
    cur_eval_seconds = _safe_float(cur_report.get("metrics_secondary", {}).get("runtime_seconds"))
    base_explain_seconds = _explain_seconds_total(base_samples)
    cur_explain_seconds = _explain_seconds_total(cur_samples)

    return {
        "baseline_run_dir": str(baseline_run_dir),
        "run_dir": str(current_run_dir),
        "lambda": str(cur_cfg.get("lambdas", "")),
        "seed": cur_cfg.get("seed"),
        "search": cur_cfg.get("search"),
        "chunker": cur_cfg.get("chunker"),
        "directional_gain": direction,
        "improved": improved,
        "regressed": regressed,
        "improved_count": int(improved_count),
        "regressed_count": int(regressed_count),
        "faithfulness_pass": bool(improved_count >= 4 and regressed_count == 0),
        "metrics_baseline": b,
        "metrics_current": c,
        "timing": {
            "eval_runtime_seconds_baseline": base_eval_seconds,
            "eval_runtime_seconds_current": cur_eval_seconds,
            "eval_runtime_seconds_delta": cur_eval_seconds - base_eval_seconds,
            "explain_seconds_baseline": base_explain_seconds,
            "explain_seconds_current": cur_explain_seconds,
            "explain_seconds_delta": cur_explain_seconds - base_explain_seconds,
        },
        "stability": stability.to_dict(),
        "reliability": {
            "baseline_failed_samples": _failed_samples(base_report),
            "current_failed_samples": _failed_samples(cur_report),
            "baseline_oom_shrink_events": _oom_shrink_events(base_report),
            "current_oom_shrink_events": _oom_shrink_events(cur_report),
        },
        "provenance": {
            "baseline_device": base_cfg.get("device"),
            "current_device": cur_cfg.get("device"),
            "baseline_deterministic": base_cfg.get("deterministic"),
            "current_deterministic": cur_cfg.get("deterministic"),
            "baseline_commit": base_report.get("provenance", {}).get("git", {}).get("commit_short"),
            "current_commit": cur_report.get("provenance", {}).get("git", {}).get("commit_short"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize lambda sweep faithfulness and side effects")
    parser.add_argument("--baseline-run-dir", type=str, required=True)
    parser.add_argument("--candidate-run-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--metric-tol", type=float, default=5e-4)
    parser.add_argument("--trace-tol", type=float, default=1e-8)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def _flatten_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "run_dir": row["run_dir"],
                "lambda": row["lambda"],
                "improved_count": row["improved_count"],
                "regressed_count": row["regressed_count"],
                "faithfulness_pass": row["faithfulness_pass"],
                "gain_log_odds": row["directional_gain"]["log_odds"],
                "gain_comp": row["directional_gain"]["comp"],
                "gain_suff": row["directional_gain"]["suff"],
                "gain_a_c": row["directional_gain"]["a_c"],
                "gain_a_s": row["directional_gain"]["a_s"],
                "eval_seconds_delta": row["timing"]["eval_runtime_seconds_delta"],
                "explain_seconds_delta": row["timing"]["explain_seconds_delta"],
                "selected_changed_ratio": row["stability"]["selected_changed_ratio"],
                "ranking_changed_ratio": row["stability"]["ranking_changed_ratio"],
                "trace_total_score_max_abs_diff": row["stability"]["trace_total_score_max_abs_diff"],
                "failed_samples": row["reliability"]["current_failed_samples"],
                "oom_shrink_events": row["reliability"]["current_oom_shrink_events"],
            }
        )
    return out


def main() -> None:
    args = build_parser().parse_args()
    baseline_run_dir = Path(args.baseline_run_dir)
    candidate_run_dirs = [Path(p) for p in args.candidate_run_dirs]

    rows = [
        _build_row(
            baseline_run_dir=baseline_run_dir,
            current_run_dir=run_dir,
            metric_tol=float(args.metric_tol),
            trace_tol=float(args.trace_tol),
        )
        for run_dir in candidate_run_dirs
    ]
    rows = sorted(rows, key=lambda x: (not bool(x["faithfulness_pass"]), -int(x["improved_count"]), str(x["lambda"])))

    payload = {
        "baseline_run_dir": str(baseline_run_dir),
        "metric_tolerance": float(args.metric_tol),
        "trace_tolerance": float(args.trace_tol),
        "candidate_count": len(rows),
        "rows": rows,
    }

    out_json = Path(args.output_json) if args.output_json else (baseline_run_dir.parent / "lambda_sweep_report.json")
    out_csv = Path(args.output_csv) if args.output_csv else (baseline_run_dir.parent / "lambda_sweep_report.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = _flatten_rows(rows)
    fieldnames = [
        "run_dir",
        "lambda",
        "improved_count",
        "regressed_count",
        "faithfulness_pass",
        "gain_log_odds",
        "gain_comp",
        "gain_suff",
        "gain_a_c",
        "gain_a_s",
        "eval_seconds_delta",
        "explain_seconds_delta",
        "selected_changed_ratio",
        "ranking_changed_ratio",
        "trace_total_score_max_abs_diff",
        "failed_samples",
        "oom_shrink_events",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)

    print(f"[lambda-sweep] candidates={len(rows)}")
    print(f"[lambda-sweep] json={out_json}")
    print(f"[lambda-sweep] csv={out_csv}")


if __name__ == "__main__":
    main()
