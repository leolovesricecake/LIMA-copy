#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


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


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def _extract_boundaries(payload: Dict[str, Any]) -> List[int]:
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    boundaries: List[int] = []
    for chunk in chunks[:-1]:
        if not isinstance(chunk, dict):
            continue
        end_char = chunk.get("end_char")
        try:
            boundaries.append(int(end_char))
        except Exception:
            continue
    return sorted(set(boundaries))


def _top20_count(payload: Dict[str, Any]) -> int:
    chunks = payload.get("chunks", [])
    total = len(chunks) if isinstance(chunks, list) else 0
    return int((float(total) * 20.0) // 100.0)


def _aggregate_chunk_diag(samples: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    fallback = 0
    orphan_total = 0.0
    orphan_samples = 0
    leading_close_total = 0.0
    leading_close_samples = 0
    abbreviation_total = 0.0
    abbreviation_samples = 0
    cross_total = 0.0
    cross_samples = 0
    orphan_merge_total = 0.0
    orphan_merge_samples = 0
    leading_close_fix_total = 0.0
    leading_close_fix_samples = 0
    abbreviation_merge_total = 0.0
    abbreviation_merge_samples = 0
    strategy_counts: Dict[str, int] = {}
    chunk_count_values: List[float] = []
    chunk_len_mean_values: List[float] = []
    top20_count_values: List[float] = []

    for payload in samples.values():
        diag = payload.get("metadata", {}).get("chunk_diagnostics")
        if not isinstance(diag, dict):
            continue
        total += 1
        strategy = str(diag.get("chunk_strategy", "unknown"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        if bool(diag.get("fallback_applied", False)):
            fallback += 1
        orphan = _safe_float(diag.get("singleton_orphan_punctuation_chunks"), 0.0)
        leading_close = _safe_float(diag.get("leading_close_punct_chunks"), 0.0)
        abbreviation = _safe_float(diag.get("abbreviation_singleton_chunks"), 0.0)
        cross = _safe_float(diag.get("cross_newline_boundary_chunks"), 0.0)
        orphan_merge = _safe_float(diag.get("orphan_merge_count"), 0.0)
        leading_close_fix = _safe_float(diag.get("leading_close_punct_fix_count"), 0.0)
        abbreviation_merge = _safe_float(diag.get("abbreviation_merge_count"), 0.0)
        orphan_total += orphan
        leading_close_total += leading_close
        abbreviation_total += abbreviation
        cross_total += cross
        orphan_merge_total += orphan_merge
        leading_close_fix_total += leading_close_fix
        abbreviation_merge_total += abbreviation_merge
        if orphan > 0.0:
            orphan_samples += 1
        if leading_close > 0.0:
            leading_close_samples += 1
        if abbreviation > 0.0:
            abbreviation_samples += 1
        if orphan_merge > 0.0:
            orphan_merge_samples += 1
        if leading_close_fix > 0.0:
            leading_close_fix_samples += 1
        if abbreviation_merge > 0.0:
            abbreviation_merge_samples += 1
        if cross > 0.0:
            cross_samples += 1
        chunk_count_values.append(_safe_float(diag.get("chunk_count"), 0.0))
        chunk_len_mean_values.append(_safe_float(diag.get("chunk_len_chars_mean"), 0.0))
        top20_count_values.append(float(_top20_count(payload)))

    denom = float(total) if total > 0 else 1.0
    return {
        "samples_with_chunk_diagnostics": int(total),
        "chunk_strategy_counts": strategy_counts,
        "fallback_rate": float(fallback / denom),
        "orphan_chunks_mean": float(orphan_total / denom),
        "orphan_samples_ratio": float(orphan_samples / denom),
        "leading_close_punct_mean": float(leading_close_total / denom),
        "leading_close_punct_samples_ratio": float(leading_close_samples / denom),
        "abbreviation_singleton_mean": float(abbreviation_total / denom),
        "abbreviation_singleton_samples_ratio": float(abbreviation_samples / denom),
        "orphan_merge_count_mean": float(orphan_merge_total / denom),
        "orphan_merge_samples_ratio": float(orphan_merge_samples / denom),
        "leading_close_punct_fix_count_mean": float(leading_close_fix_total / denom),
        "leading_close_punct_fix_samples_ratio": float(leading_close_fix_samples / denom),
        "abbreviation_merge_count_mean": float(abbreviation_merge_total / denom),
        "abbreviation_merge_samples_ratio": float(abbreviation_merge_samples / denom),
        "cross_newline_chunks_mean": float(cross_total / denom),
        "cross_newline_samples_ratio": float(cross_samples / denom),
        "chunk_count_mean": (
            float(sum(chunk_count_values) / len(chunk_count_values)) if chunk_count_values else 0.0
        ),
        "top20_count_mean": (
            float(sum(top20_count_values) / len(top20_count_values)) if top20_count_values else 0.0
        ),
        "top20_count_p90": _percentile(top20_count_values, 0.90),
        "chunk_len_chars_mean_of_mean": (
            float(sum(chunk_len_mean_values) / len(chunk_len_mean_values)) if chunk_len_mean_values else 0.0
        ),
    }


def _aggregate_adaptive_diag(samples: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    diags: List[Dict[str, Any]] = []
    for payload in samples.values():
        diag = payload.get("metadata", {}).get("chunk_diagnostics")
        if isinstance(diag, dict):
            diags.append(diag)

    if not diags:
        return {
            "samples_with_chunk_diagnostics": 0,
            "adaptive_enabled_samples": 0,
            "adaptive_enabled_ratio": 0.0,
            "adaptive_bucket_counts": {},
            "adaptive_bucket_ratios": {},
            "very_long_samples": 0,
            "very_long_raw_eq_one_ratio": 0.0,
            "very_long_fragmentation_ratio_mean": 0.0,
            "very_long_fragmentation_delta_mean": 0.0,
            "postprocess_invalid_merge_mean": 0.0,
            "postprocess_short_merge_mean": 0.0,
            "postprocess_long_split_mean": 0.0,
            "postprocess_post_long_invalid_merge_mean": 0.0,
            "postprocess_post_long_short_merge_mean": 0.0,
            "postprocess_adjacent_pack_merge_mean": 0.0,
            "postprocess_invalid_merge_trigger_ratio": 0.0,
            "postprocess_short_merge_trigger_ratio": 0.0,
            "postprocess_long_split_trigger_ratio": 0.0,
            "postprocess_post_long_invalid_merge_trigger_ratio": 0.0,
            "postprocess_post_long_short_merge_trigger_ratio": 0.0,
            "postprocess_adjacent_pack_merge_trigger_ratio": 0.0,
            "stage_raw_mean": 0.0,
            "stage_after_invalid_mean": 0.0,
            "stage_after_short_mean": 0.0,
            "stage_after_long_mean": 0.0,
            "stage_after_post_long_merge_mean": 0.0,
            "stage_final_mean": 0.0,
            "stage_raw_to_final_delta_mean": 0.0,
        }

    bucket_counts: Dict[str, int] = {}
    adaptive_enabled = 0
    invalid_values: List[float] = []
    short_values: List[float] = []
    long_values: List[float] = []
    post_long_invalid_values: List[float] = []
    post_long_short_values: List[float] = []
    adjacent_pack_values: List[float] = []
    stage_raw_values: List[float] = []
    stage_after_invalid_values: List[float] = []
    stage_after_short_values: List[float] = []
    stage_after_long_values: List[float] = []
    stage_after_post_long_values: List[float] = []
    stage_final_values: List[float] = []
    very_long_ratios: List[float] = []
    very_long_deltas: List[float] = []
    very_long_samples = 0
    very_long_raw_eq_one = 0

    for diag in diags:
        enabled = bool(diag.get("adaptive_enabled", False))
        if enabled:
            adaptive_enabled += 1
        bucket = str(diag.get("adaptive_bucket", "unknown"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        post = diag.get("adaptive_postprocess", {})
        if not isinstance(post, dict):
            post = {}
        invalid_values.append(_safe_float(post.get("invalid_merge_count"), 0.0))
        short_values.append(_safe_float(post.get("short_merge_count"), 0.0))
        long_values.append(_safe_float(post.get("long_split_count"), 0.0))
        post_long_invalid_values.append(_safe_float(post.get("post_long_invalid_merge_count"), 0.0))
        post_long_short_values.append(_safe_float(post.get("post_long_short_merge_count"), 0.0))
        adjacent_pack_values.append(_safe_float(post.get("adjacent_pack_merge_count"), 0.0))

        stage = diag.get("adaptive_stage_chunk_counts", {})
        if not isinstance(stage, dict):
            stage = {}
        stage_raw = _safe_float(stage.get("raw"), 0.0)
        stage_final = _safe_float(stage.get("final"), _safe_float(diag.get("chunk_count"), 0.0))
        stage_raw_values.append(stage_raw)
        stage_after_invalid_values.append(_safe_float(stage.get("after_invalid"), 0.0))
        stage_after_short_values.append(_safe_float(stage.get("after_short"), 0.0))
        stage_after_long_values.append(_safe_float(stage.get("after_long"), 0.0))
        stage_after_post_long_values.append(_safe_float(stage.get("after_post_long_merge"), stage_final))
        stage_final_values.append(stage_final)

        if bucket == "very_long":
            very_long_samples += 1
            if int(stage_raw) == 1:
                very_long_raw_eq_one += 1
            if stage_raw > 0.0:
                very_long_ratios.append(float(stage_final / stage_raw))
                very_long_deltas.append(float(stage_final - stage_raw))

    denom = float(len(diags))
    very_long_denom = float(very_long_samples) if very_long_samples > 0 else 1.0
    bucket_ratios = {k: float(v / denom) for k, v in sorted(bucket_counts.items())}
    return {
        "samples_with_chunk_diagnostics": int(len(diags)),
        "adaptive_enabled_samples": int(adaptive_enabled),
        "adaptive_enabled_ratio": float(adaptive_enabled / denom),
        "adaptive_bucket_counts": dict(sorted(bucket_counts.items())),
        "adaptive_bucket_ratios": bucket_ratios,
        "very_long_samples": int(very_long_samples),
        "very_long_raw_eq_one_ratio": float(very_long_raw_eq_one / very_long_denom),
        "very_long_fragmentation_ratio_mean": (
            float(sum(very_long_ratios) / len(very_long_ratios)) if very_long_ratios else 0.0
        ),
        "very_long_fragmentation_delta_mean": (
            float(sum(very_long_deltas) / len(very_long_deltas)) if very_long_deltas else 0.0
        ),
        "postprocess_invalid_merge_mean": float(sum(invalid_values) / denom),
        "postprocess_short_merge_mean": float(sum(short_values) / denom),
        "postprocess_long_split_mean": float(sum(long_values) / denom),
        "postprocess_post_long_invalid_merge_mean": float(sum(post_long_invalid_values) / denom),
        "postprocess_post_long_short_merge_mean": float(sum(post_long_short_values) / denom),
        "postprocess_adjacent_pack_merge_mean": float(sum(adjacent_pack_values) / denom),
        "postprocess_invalid_merge_trigger_ratio": float(sum(1 for x in invalid_values if x > 0.0) / denom),
        "postprocess_short_merge_trigger_ratio": float(sum(1 for x in short_values if x > 0.0) / denom),
        "postprocess_long_split_trigger_ratio": float(sum(1 for x in long_values if x > 0.0) / denom),
        "postprocess_post_long_invalid_merge_trigger_ratio": float(
            sum(1 for x in post_long_invalid_values if x > 0.0) / denom
        ),
        "postprocess_post_long_short_merge_trigger_ratio": float(
            sum(1 for x in post_long_short_values if x > 0.0) / denom
        ),
        "postprocess_adjacent_pack_merge_trigger_ratio": float(
            sum(1 for x in adjacent_pack_values if x > 0.0) / denom
        ),
        "stage_raw_mean": float(sum(stage_raw_values) / denom),
        "stage_after_invalid_mean": float(sum(stage_after_invalid_values) / denom),
        "stage_after_short_mean": float(sum(stage_after_short_values) / denom),
        "stage_after_long_mean": float(sum(stage_after_long_values) / denom),
        "stage_after_post_long_merge_mean": float(sum(stage_after_post_long_values) / denom),
        "stage_final_mean": float(sum(stage_final_values) / denom),
        "stage_raw_to_final_delta_mean": float((sum(stage_final_values) - sum(stage_raw_values)) / denom),
    }


def _explain_seconds_total(samples: Dict[str, Dict[str, Any]]) -> float:
    return float(
        sum(_safe_float(payload.get("metadata", {}).get("elapsed_seconds"), 0.0) for payload in samples.values())
    )


def _drift_summary(
    baseline_samples: Dict[str, Dict[str, Any]],
    candidate_samples: Dict[str, Dict[str, Any]],
    trace_tolerance: float,
) -> Dict[str, Any]:
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    selected_changed = 0
    ranking_changed = 0
    trace_changed = 0
    max_trace_abs_diff = 0.0

    for sample_id in common_ids:
        left = baseline_samples[sample_id]
        right = candidate_samples[sample_id]
        if left.get("selected_chunk_ids") != right.get("selected_chunk_ids"):
            selected_changed += 1
        if left.get("chunk_ranking") != right.get("chunk_ranking"):
            ranking_changed += 1

        local_max = 0.0
        left_trace = left.get("trace", [])
        right_trace = right.get("trace", [])
        for lrow, rrow in zip(left_trace, right_trace):
            diff = abs(_safe_float(lrow.get("total_score")) - _safe_float(rrow.get("total_score")))
            if diff > local_max:
                local_max = diff
        if local_max > trace_tolerance:
            trace_changed += 1
        if local_max > max_trace_abs_diff:
            max_trace_abs_diff = local_max

    denom = float(len(common_ids)) if common_ids else 1.0
    return {
        "sample_count_common": int(len(common_ids)),
        "selected_changed": int(selected_changed),
        "ranking_changed": int(ranking_changed),
        "trace_changed": int(trace_changed),
        "selected_changed_ratio": float(selected_changed / denom),
        "ranking_changed_ratio": float(ranking_changed / denom),
        "trace_changed_ratio": float(trace_changed / denom),
        "trace_total_score_max_abs_diff": float(max_trace_abs_diff),
    }


def _top20_count_drift(
    baseline_samples: Dict[str, Dict[str, Any]],
    candidate_samples: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    if not common_ids:
        return {
            "sample_count_common": 0,
            "baseline_top20_count_mean": 0.0,
            "baseline_top20_count_p90": 0.0,
            "candidate_top20_count_mean": 0.0,
            "candidate_top20_count_p90": 0.0,
            "top20_count_delta_mean": 0.0,
            "top20_count_delta_p90": 0.0,
        }

    baseline_top20 = [_top20_count(baseline_samples[sid]) for sid in common_ids]
    candidate_top20 = [_top20_count(candidate_samples[sid]) for sid in common_ids]
    deltas = [float(c - b) for b, c in zip(baseline_top20, candidate_top20)]

    return {
        "sample_count_common": int(len(common_ids)),
        "baseline_top20_count_mean": float(sum(baseline_top20) / len(baseline_top20)),
        "baseline_top20_count_p90": _percentile([float(x) for x in baseline_top20], 0.90),
        "candidate_top20_count_mean": float(sum(candidate_top20) / len(candidate_top20)),
        "candidate_top20_count_p90": _percentile([float(x) for x in candidate_top20], 0.90),
        "top20_count_delta_mean": float(sum(deltas) / len(deltas)),
        "top20_count_delta_p90": _percentile(deltas, 0.90),
    }


def _boundary_drift_summary(
    baseline_samples: Dict[str, Dict[str, Any]],
    candidate_samples: Dict[str, Dict[str, Any]],
    large_shift_threshold: int = 8,
) -> Dict[str, Any]:
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    if not common_ids:
        return {
            "sample_count_common": 0,
            "sample_count_with_boundaries": 0,
            "boundary_jaccard_mean": 0.0,
            "boundary_jaccard_median": 0.0,
            "boundary_shift_chars_mean": 0.0,
            "boundary_shift_chars_p90": 0.0,
            "boundary_shift_chars_max": 0.0,
            "large_shift_threshold": int(large_shift_threshold),
            "large_shift_boundary_count": 0,
            "large_shift_boundary_ratio": 0.0,
        }

    jaccards: List[float] = []
    shifts: List[float] = []
    sample_with_boundaries = 0

    for sample_id in common_ids:
        baseline_bounds = _extract_boundaries(baseline_samples[sample_id])
        candidate_bounds = _extract_boundaries(candidate_samples[sample_id])

        baseline_set = set(baseline_bounds)
        candidate_set = set(candidate_bounds)
        union = baseline_set.union(candidate_set)
        if len(union) == 0:
            jaccards.append(1.0)
        else:
            jaccards.append(float(len(baseline_set.intersection(candidate_set)) / float(len(union))))

        if baseline_bounds and candidate_bounds:
            sample_with_boundaries += 1
            for bound in baseline_bounds:
                nearest = min(abs(bound - other) for other in candidate_bounds)
                shifts.append(float(nearest))

    large_shift_count = sum(1 for value in shifts if value > float(large_shift_threshold))
    shift_total = len(shifts)

    return {
        "sample_count_common": int(len(common_ids)),
        "sample_count_with_boundaries": int(sample_with_boundaries),
        "boundary_jaccard_mean": float(statistics.mean(jaccards)) if jaccards else 0.0,
        "boundary_jaccard_median": float(statistics.median(jaccards)) if jaccards else 0.0,
        "boundary_shift_chars_mean": float(statistics.mean(shifts)) if shifts else 0.0,
        "boundary_shift_chars_p90": _percentile(shifts, 0.90),
        "boundary_shift_chars_max": float(max(shifts)) if shifts else 0.0,
        "large_shift_threshold": int(large_shift_threshold),
        "large_shift_boundary_count": int(large_shift_count),
        "large_shift_boundary_ratio": (
            float(large_shift_count) / float(shift_total) if shift_total > 0 else 0.0
        ),
    }


def _metrics(report: Dict[str, Any]) -> Dict[str, float]:
    gold = report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    return {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comprehensiveness": _safe_float(gold.get("comprehensiveness")),
        "sufficiency": _safe_float(gold.get("sufficiency")),
        "aopc": _safe_float(gold.get("aopc")),
        "aopc_comprehensiveness": _safe_float(gold.get("aopc_comprehensiveness")),
        "aopc_sufficiency": _safe_float(gold.get("aopc_sufficiency")),
        "runtime_seconds": _safe_float(report.get("metrics_secondary", {}).get("runtime_seconds")),
    }


def _metric_delta_rows(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in baseline:
        bv = float(baseline[key])
        cv = float(candidate.get(key, 0.0))
        out[key] = {
            "baseline": bv,
            "candidate": cv,
            "delta": cv - bv,
            "abs_diff": abs(cv - bv),
            "ratio": (cv / bv) if abs(bv) > 1e-12 else 0.0,
        }
    return out


def build_report(baseline_run_dir: Path, candidate_run_dir: Path, trace_tolerance: float) -> Dict[str, Any]:
    baseline_report = _read_json(baseline_run_dir / "eval_report.json")
    candidate_report = _read_json(candidate_run_dir / "eval_report.json")
    baseline_cfg = _read_json(baseline_run_dir / "run_config.json")
    candidate_cfg = _read_json(candidate_run_dir / "run_config.json")
    baseline_samples = _load_samples(baseline_run_dir)
    candidate_samples = _load_samples(candidate_run_dir)

    metric_deltas = _metric_delta_rows(_metrics(baseline_report), _metrics(candidate_report))
    baseline_explain_seconds = _explain_seconds_total(baseline_samples)
    candidate_explain_seconds = _explain_seconds_total(candidate_samples)

    return {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "provenance": {
            "baseline_device": baseline_cfg.get("device"),
            "candidate_device": candidate_cfg.get("device"),
            "baseline_deterministic": baseline_cfg.get("deterministic"),
            "candidate_deterministic": candidate_cfg.get("deterministic"),
            "baseline_commit": baseline_report.get("provenance", {}).get("git", {}).get("commit_short"),
            "candidate_commit": candidate_report.get("provenance", {}).get("git", {}).get("commit_short"),
        },
        "metric_deltas": metric_deltas,
        "explain_timing": {
            "baseline_explain_seconds": baseline_explain_seconds,
            "candidate_explain_seconds": candidate_explain_seconds,
            "delta_seconds": candidate_explain_seconds - baseline_explain_seconds,
            "improve_ratio": (
                (baseline_explain_seconds - candidate_explain_seconds) / baseline_explain_seconds
                if baseline_explain_seconds > 0
                else 0.0
            ),
        },
        "chunk_diagnostics": {
            "baseline": _aggregate_chunk_diag(baseline_samples),
            "candidate": _aggregate_chunk_diag(candidate_samples),
        },
        "adaptive_diagnostics": {
            "baseline": _aggregate_adaptive_diag(baseline_samples),
            "candidate": _aggregate_adaptive_diag(candidate_samples),
        },
        "top20_count_drift": _top20_count_drift(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
        ),
        "explanation_drift": _drift_summary(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
            trace_tolerance=trace_tolerance,
        ),
        "boundary_drift": _boundary_drift_summary(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase A chunking comparison report")
    parser.add_argument("--baseline-run-dir", type=str, required=True)
    parser.add_argument("--candidate-run-dir", type=str, required=True)
    parser.add_argument("--trace-tolerance", type=float, default=1e-8)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_run_dir = Path(args.baseline_run_dir)
    candidate_run_dir = Path(args.candidate_run_dir)
    report = build_report(
        baseline_run_dir=baseline_run_dir,
        candidate_run_dir=candidate_run_dir,
        trace_tolerance=float(args.trace_tolerance),
    )

    out_json = Path(args.output_json) if args.output_json else (candidate_run_dir / "phase_a_chunking_compare.json")
    out_csv = Path(args.output_csv) if args.output_csv else (candidate_run_dir / "phase_a_chunking_compare.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "gold_comp_abs_diff": report["metric_deltas"]["comprehensiveness"]["abs_diff"],
        "gold_suff_abs_diff": report["metric_deltas"]["sufficiency"]["abs_diff"],
        "gold_log_odds_abs_diff": report["metric_deltas"]["log_odds"]["abs_diff"],
        "gold_aopc_abs_diff": report["metric_deltas"]["aopc"]["abs_diff"],
        "eval_runtime_improve_ratio": (
            (report["metric_deltas"]["runtime_seconds"]["baseline"] - report["metric_deltas"]["runtime_seconds"]["candidate"])
            / report["metric_deltas"]["runtime_seconds"]["baseline"]
            if report["metric_deltas"]["runtime_seconds"]["baseline"] > 0
            else 0.0
        ),
        "explain_seconds_improve_ratio": report["explain_timing"]["improve_ratio"],
        "selected_changed_ratio": report["explanation_drift"]["selected_changed_ratio"],
        "ranking_changed_ratio": report["explanation_drift"]["ranking_changed_ratio"],
        "baseline_orphan_chunks_mean": report["chunk_diagnostics"]["baseline"]["orphan_chunks_mean"],
        "candidate_orphan_chunks_mean": report["chunk_diagnostics"]["candidate"]["orphan_chunks_mean"],
        "baseline_leading_close_punct_mean": report["chunk_diagnostics"]["baseline"]["leading_close_punct_mean"],
        "candidate_leading_close_punct_mean": report["chunk_diagnostics"]["candidate"]["leading_close_punct_mean"],
        "baseline_abbreviation_singleton_mean": report["chunk_diagnostics"]["baseline"]["abbreviation_singleton_mean"],
        "candidate_abbreviation_singleton_mean": report["chunk_diagnostics"]["candidate"][
            "abbreviation_singleton_mean"
        ],
        "baseline_orphan_merge_count_mean": report["chunk_diagnostics"]["baseline"]["orphan_merge_count_mean"],
        "candidate_orphan_merge_count_mean": report["chunk_diagnostics"]["candidate"]["orphan_merge_count_mean"],
        "baseline_leading_close_punct_fix_count_mean": report["chunk_diagnostics"]["baseline"][
            "leading_close_punct_fix_count_mean"
        ],
        "candidate_leading_close_punct_fix_count_mean": report["chunk_diagnostics"]["candidate"][
            "leading_close_punct_fix_count_mean"
        ],
        "baseline_abbreviation_merge_count_mean": report["chunk_diagnostics"]["baseline"][
            "abbreviation_merge_count_mean"
        ],
        "candidate_abbreviation_merge_count_mean": report["chunk_diagnostics"]["candidate"][
            "abbreviation_merge_count_mean"
        ],
        "baseline_fallback_rate": report["chunk_diagnostics"]["baseline"]["fallback_rate"],
        "candidate_fallback_rate": report["chunk_diagnostics"]["candidate"]["fallback_rate"],
        "baseline_adaptive_enabled_ratio": report["adaptive_diagnostics"]["baseline"]["adaptive_enabled_ratio"],
        "candidate_adaptive_enabled_ratio": report["adaptive_diagnostics"]["candidate"]["adaptive_enabled_ratio"],
        "baseline_adaptive_stage_raw_mean": report["adaptive_diagnostics"]["baseline"]["stage_raw_mean"],
        "candidate_adaptive_stage_raw_mean": report["adaptive_diagnostics"]["candidate"]["stage_raw_mean"],
        "baseline_adaptive_stage_final_mean": report["adaptive_diagnostics"]["baseline"]["stage_final_mean"],
        "candidate_adaptive_stage_final_mean": report["adaptive_diagnostics"]["candidate"]["stage_final_mean"],
        "baseline_adaptive_very_long_fragmentation_ratio_mean": report["adaptive_diagnostics"]["baseline"][
            "very_long_fragmentation_ratio_mean"
        ],
        "candidate_adaptive_very_long_fragmentation_ratio_mean": report["adaptive_diagnostics"]["candidate"][
            "very_long_fragmentation_ratio_mean"
        ],
        "baseline_adaptive_long_split_trigger_ratio": report["adaptive_diagnostics"]["baseline"][
            "postprocess_long_split_trigger_ratio"
        ],
        "candidate_adaptive_long_split_trigger_ratio": report["adaptive_diagnostics"]["candidate"][
            "postprocess_long_split_trigger_ratio"
        ],
        "baseline_top20_count_mean": report["top20_count_drift"]["baseline_top20_count_mean"],
        "candidate_top20_count_mean": report["top20_count_drift"]["candidate_top20_count_mean"],
        "baseline_top20_count_p90": report["top20_count_drift"]["baseline_top20_count_p90"],
        "candidate_top20_count_p90": report["top20_count_drift"]["candidate_top20_count_p90"],
        "top20_count_delta_mean": report["top20_count_drift"]["top20_count_delta_mean"],
        "top20_count_delta_p90": report["top20_count_drift"]["top20_count_delta_p90"],
        "boundary_jaccard_mean": report["boundary_drift"]["boundary_jaccard_mean"],
        "boundary_jaccard_median": report["boundary_drift"]["boundary_jaccard_median"],
        "boundary_shift_chars_mean": report["boundary_drift"]["boundary_shift_chars_mean"],
        "boundary_shift_chars_p90": report["boundary_drift"]["boundary_shift_chars_p90"],
        "boundary_shift_chars_max": report["boundary_drift"]["boundary_shift_chars_max"],
        "large_shift_boundary_count": report["boundary_drift"]["large_shift_boundary_count"],
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[phase-a-compare] json={out_json}")
    print(f"[phase-a-compare] csv={out_csv}")


if __name__ == "__main__":
    main()
