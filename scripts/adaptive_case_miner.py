#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
        payload = _read_json(path)
        sample_id = str(payload.get("sample_id", path.stem))
        out[sample_id] = payload
    return out


def _trace_max_abs_diff(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_trace = left.get("trace", [])
    right_trace = right.get("trace", [])
    local_max = 0.0
    for lrow, rrow in zip(left_trace, right_trace):
        diff = abs(_safe_float(lrow.get("total_score")) - _safe_float(rrow.get("total_score")))
        if diff > local_max:
            local_max = diff
    return float(local_max)


def _sample_proxy_score(
    *,
    selected_changed: bool,
    ranking_changed: bool,
    trace_max_abs_diff: float,
    chunk_count_delta: int,
    short_singleton_risk: bool,
    very_long_fragmentation_risk: bool,
) -> float:
    score = 0.0
    if selected_changed:
        score += 1.0
    if ranking_changed:
        score += 1.0
    score += min(2.0, float(trace_max_abs_diff) / 0.05)
    score += min(1.5, abs(float(chunk_count_delta)) / 20.0)
    if short_singleton_risk:
        score += 1.0
    if very_long_fragmentation_risk:
        score += 1.5
    return float(score)


def _trigger_kind(*, floor_applied: bool, guard_applied: bool) -> str:
    if floor_applied and guard_applied:
        return "both"
    if floor_applied:
        return "floor_only"
    if guard_applied:
        return "guard_only"
    return "none"


def build_case_report(
    *,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    very_long_max_chunks: int = 48,
    very_long_ratio_threshold: float = 24.0,
    short_floor_min_words: int = 15,
    top_k: int = 50,
) -> Dict[str, Any]:
    baseline_samples = _load_samples(baseline_run_dir)
    candidate_samples = _load_samples(candidate_run_dir)
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))

    rows: List[Dict[str, Any]] = []
    very_long_explosive_cases: List[Dict[str, Any]] = []
    short_singleton_cases: List[Dict[str, Any]] = []
    selected_changed = 0
    ranking_changed = 0

    for sample_id in common_ids:
        left = baseline_samples[sample_id]
        right = candidate_samples[sample_id]
        left_sel = list(left.get("selected_chunk_ids", []))
        right_sel = list(right.get("selected_chunk_ids", []))
        left_rank = list(left.get("chunk_ranking", []))
        right_rank = list(right.get("chunk_ranking", []))
        sel_changed = left_sel != right_sel
        rank_changed = left_rank != right_rank
        if sel_changed:
            selected_changed += 1
        if rank_changed:
            ranking_changed += 1

        trace_max = _trace_max_abs_diff(left, right)
        left_chunk_count = int(len(left.get("chunks", [])))
        right_chunk_count = int(len(right.get("chunks", [])))
        chunk_delta = right_chunk_count - left_chunk_count

        diag = right.get("metadata", {}).get("chunk_diagnostics", {})
        if not isinstance(diag, dict):
            diag = {}
        bucket = str(diag.get("adaptive_bucket", "unknown"))
        adaptive_features = diag.get("adaptive_features", {})
        if not isinstance(adaptive_features, dict):
            adaptive_features = {}
        stage = diag.get("adaptive_stage_chunk_counts", {})
        if not isinstance(stage, dict):
            stage = {}
        raw_count = _safe_float(stage.get("raw"), 0.0)
        final_count = _safe_float(stage.get("final"), float(right_chunk_count))
        word_count = int(_safe_float(adaptive_features.get("word_count"), 0.0))
        fallback_reason = diag.get("fallback_reason")
        floor_applied = bool(diag.get("adaptive_effective_floor_applied", False))
        guard_applied = bool(diag.get("adaptive_fragmentation_guard_applied", False))
        trigger_kind = _trigger_kind(floor_applied=floor_applied, guard_applied=guard_applied)

        short_singleton_risk = bool(bucket == "short" and word_count >= short_floor_min_words and final_count <= 1.0)
        very_long_fragmentation_risk = False
        if bucket == "very_long" and raw_count > 0:
            if raw_count <= 1.0 and final_count > float(very_long_max_chunks):
                very_long_fragmentation_risk = True
            if (final_count / raw_count) > float(very_long_ratio_threshold):
                very_long_fragmentation_risk = True

        score = _sample_proxy_score(
            selected_changed=sel_changed,
            ranking_changed=rank_changed,
            trace_max_abs_diff=trace_max,
            chunk_count_delta=chunk_delta,
            short_singleton_risk=short_singleton_risk,
            very_long_fragmentation_risk=very_long_fragmentation_risk,
        )

        row = {
            "sample_id": sample_id,
            "proxy_regression_score": score,
            "selected_changed": bool(sel_changed),
            "ranking_changed": bool(rank_changed),
            "trace_total_score_max_abs_diff": float(trace_max),
            "baseline_chunk_count": int(left_chunk_count),
            "candidate_chunk_count": int(right_chunk_count),
            "chunk_count_delta": int(chunk_delta),
            "adaptive_bucket": bucket,
            "word_count": int(word_count),
            "raw_chunk_count": float(raw_count),
            "final_chunk_count": float(final_count),
            "fallback_reason": fallback_reason,
            "adaptive_effective_floor_applied": bool(floor_applied),
            "adaptive_fragmentation_guard_applied": bool(guard_applied),
            "trigger_kind": trigger_kind,
            "short_singleton_risk": bool(short_singleton_risk),
            "very_long_fragmentation_risk": bool(very_long_fragmentation_risk),
            "text_preview": str(right.get("text", "")).replace("\n", " ")[:220],
        }
        rows.append(row)
        if very_long_fragmentation_risk:
            very_long_explosive_cases.append(row)
        if short_singleton_risk:
            short_singleton_cases.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            0 if str(x.get("trigger_kind")) in {"both", "guard_only", "floor_only"} else 1,
            -_safe_float(x.get("proxy_regression_score")),
            -int(bool(x.get("selected_changed", False))),
            -int(bool(x.get("ranking_changed", False))),
            str(x.get("sample_id")),
        ),
    )
    top_cases = rows_sorted[: max(1, int(top_k))]
    trigger_ranked: Dict[str, List[Dict[str, Any]]] = {}
    for kind in ["both", "guard_only", "floor_only", "none"]:
        subset = [row for row in rows_sorted if str(row.get("trigger_kind", "none")) == kind]
        trigger_ranked[kind] = subset[: max(1, min(int(top_k), len(subset)))] if subset else []

    denom = float(len(common_ids)) if common_ids else 1.0
    return {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "sample_count_common": int(len(common_ids)),
        "summary": {
            "selected_changed_ratio": float(selected_changed / denom),
            "ranking_changed_ratio": float(ranking_changed / denom),
            "very_long_fragmentation_risk_count": int(len(very_long_explosive_cases)),
            "short_singleton_risk_count": int(len(short_singleton_cases)),
            "trigger_counts": {
                "both": int(sum(1 for row in rows if row.get("trigger_kind") == "both")),
                "guard_only": int(sum(1 for row in rows if row.get("trigger_kind") == "guard_only")),
                "floor_only": int(sum(1 for row in rows if row.get("trigger_kind") == "floor_only")),
                "none": int(sum(1 for row in rows if row.get("trigger_kind") == "none")),
            },
        },
        "top_cases": top_cases,
        "trigger_ranked_cases": trigger_ranked,
        "very_long_fragmentation_cases": very_long_explosive_cases,
        "short_singleton_cases": short_singleton_cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine adaptive regression-risk cases between two runs")
    parser.add_argument("--baseline-run-dir", type=str, required=True)
    parser.add_argument("--candidate-run-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline = Path(args.baseline_run_dir)
    candidate = Path(args.candidate_run_dir)
    report = build_case_report(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        top_k=int(args.top_k),
    )

    out_json = Path(args.output_json) if args.output_json else (candidate / "adaptive_case_miner.json")
    out_csv = Path(args.output_csv) if args.output_csv else (candidate / "adaptive_case_miner.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = report.get("top_cases", [])
    fieldnames = [
        "sample_id",
        "proxy_regression_score",
        "selected_changed",
        "ranking_changed",
        "trace_total_score_max_abs_diff",
        "baseline_chunk_count",
        "candidate_chunk_count",
        "chunk_count_delta",
        "adaptive_bucket",
        "word_count",
        "raw_chunk_count",
        "final_chunk_count",
        "fallback_reason",
        "adaptive_effective_floor_applied",
        "adaptive_fragmentation_guard_applied",
        "trigger_kind",
        "short_singleton_risk",
        "very_long_fragmentation_risk",
        "text_preview",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"[adaptive-case-miner] json={out_json}")
    print(f"[adaptive-case-miner] csv={out_csv}")


if __name__ == "__main__":
    main()
