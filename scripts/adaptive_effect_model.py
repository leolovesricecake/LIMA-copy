#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


_DIRECTION_SIGNS = {
    "log_odds": -1.0,  # implementation: log(p_perturbed)-log(p_full), lower is better
    "comprehensiveness": 1.0,
    "sufficiency": -1.0,
    "aopc_comprehensiveness": 1.0,
    "aopc_sufficiency": -1.0,
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_chunker(run_dir_name: str) -> str:
    if "chunk-adaptive_" in run_dir_name:
        return "adaptive"
    if "chunk-sentence_v2_" in run_dir_name:
        return "sentence_v2"
    if "chunk-sentence_" in run_dir_name:
        return "sentence"
    return "unknown"


def _find_run_dir_by_chunker(model_dir: Path, chunker: str) -> Path | None:
    matched = [
        path
        for path in sorted(model_dir.glob("chunk-*_method-ours"))
        if path.is_dir() and _detect_chunker(path.name) == chunker
    ]
    if not matched:
        return None
    return matched[0]


def _extract_primary_metrics(eval_report: Dict[str, Any]) -> Dict[str, float]:
    primary = eval_report.get("metrics_primary", {})
    return {
        "log_odds": _safe_float(primary.get("log_odds")),
        "comprehensiveness": _safe_float(primary.get("comprehensiveness")),
        "sufficiency": _safe_float(primary.get("sufficiency")),
        "aopc_comprehensiveness": _safe_float(primary.get("aopc_comprehensiveness")),
        "aopc_sufficiency": _safe_float(primary.get("aopc_sufficiency")),
        "runtime_seconds": _safe_float(eval_report.get("metrics_secondary", {}).get("runtime_seconds")),
    }


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


def _sample_chunk_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    chunks = payload.get("chunks", [])
    chunk_count = int(len(chunks))
    top20_count = int(math.floor(0.2 * float(chunk_count)))
    diag = payload.get("metadata", {}).get("chunk_diagnostics", {})
    if not isinstance(diag, dict):
        diag = {}
    stage = diag.get("adaptive_stage_chunk_counts", {})
    if not isinstance(stage, dict):
        stage = {}
    return {
        "chunk_count": chunk_count,
        "top20_count": top20_count,
        "fallback_applied": bool(diag.get("fallback_applied", False)),
        "bucket": str(diag.get("adaptive_bucket", "unknown")),
        "floor_applied": bool(diag.get("adaptive_effective_floor_applied", False)),
        "guard_applied": bool(diag.get("adaptive_fragmentation_guard_applied", False)),
        "raw_count": _safe_float(stage.get("raw"), 0.0),
        "final_count": _safe_float(stage.get("final"), float(chunk_count)),
    }


def _delta_with_gain(candidate: float, baseline: float, metric: str) -> Dict[str, float]:
    delta = float(candidate) - float(baseline)
    gain = float(_DIRECTION_SIGNS[metric] * delta)
    return {
        "candidate": float(candidate),
        "baseline": float(baseline),
        "delta": float(delta),
        "directional_gain": float(gain),
    }


def _trigger_kind(*, floor_applied: bool, guard_applied: bool) -> str:
    if floor_applied and guard_applied:
        return "both"
    if floor_applied:
        return "floor_only"
    if guard_applied:
        return "guard_only"
    return "none"


def _group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        bucket = str(row.get(key, "unknown"))
        grouped.setdefault(bucket, []).append(row)
    return grouped


def _aggregate_row_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "selected_changed_ratio": 0.0,
            "ranking_changed_ratio": 0.0,
            "chunk_count_mean_baseline": 0.0,
            "chunk_count_mean_candidate": 0.0,
            "top20_count_mean_baseline": 0.0,
            "top20_count_mean_candidate": 0.0,
            "fallback_rate_baseline": 0.0,
            "fallback_rate_candidate": 0.0,
            "raw_to_final_delta_mean_candidate": 0.0,
        }
    denom = float(len(rows))
    return {
        "count": int(len(rows)),
        "selected_changed_ratio": float(sum(1 for row in rows if row["selected_changed"]) / denom),
        "ranking_changed_ratio": float(sum(1 for row in rows if row["ranking_changed"]) / denom),
        "chunk_count_mean_baseline": float(sum(row["baseline_chunk_count"] for row in rows) / denom),
        "chunk_count_mean_candidate": float(sum(row["candidate_chunk_count"] for row in rows) / denom),
        "top20_count_mean_baseline": float(sum(row["baseline_top20_count"] for row in rows) / denom),
        "top20_count_mean_candidate": float(sum(row["candidate_top20_count"] for row in rows) / denom),
        "fallback_rate_baseline": float(sum(1 for row in rows if row["baseline_fallback"]) / denom),
        "fallback_rate_candidate": float(sum(1 for row in rows if row["candidate_fallback"]) / denom),
        "raw_to_final_delta_mean_candidate": float(
            sum(row["candidate_final_count"] - row["candidate_raw_count"] for row in rows) / denom
        ),
    }


def _dataset_shape_summary(sample_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not sample_rows:
        return {
            "chunk_count_mean_baseline": 0.0,
            "chunk_count_mean_candidate": 0.0,
            "top20_zero_ratio_baseline": 0.0,
            "top20_zero_ratio_candidate": 0.0,
            "top20_count_mean_baseline": 0.0,
            "top20_count_mean_candidate": 0.0,
            "fallback_rate_baseline": 0.0,
            "fallback_rate_candidate": 0.0,
        }
    denom = float(len(sample_rows))
    return {
        "chunk_count_mean_baseline": float(sum(row["baseline_chunk_count"] for row in sample_rows) / denom),
        "chunk_count_mean_candidate": float(sum(row["candidate_chunk_count"] for row in sample_rows) / denom),
        "top20_zero_ratio_baseline": float(sum(1 for row in sample_rows if row["baseline_top20_count"] == 0) / denom),
        "top20_zero_ratio_candidate": float(sum(1 for row in sample_rows if row["candidate_top20_count"] == 0) / denom),
        "top20_count_mean_baseline": float(sum(row["baseline_top20_count"] for row in sample_rows) / denom),
        "top20_count_mean_candidate": float(sum(row["candidate_top20_count"] for row in sample_rows) / denom),
        "fallback_rate_baseline": float(sum(1 for row in sample_rows if row["baseline_fallback"]) / denom),
        "fallback_rate_candidate": float(sum(1 for row in sample_rows if row["candidate_fallback"]) / denom),
    }


def build_effect_model(
    *,
    baseline_root: Path,
    candidate_root: Path,
    chunker: str = "adaptive",
) -> Dict[str, Any]:
    baseline_dataset_dirs = {
        path.name: path for path in baseline_root.iterdir() if path.is_dir()
    }
    candidate_dataset_dirs = {
        path.name: path for path in candidate_root.iterdir() if path.is_dir()
    }
    datasets: List[Dict[str, Any]] = []
    for dataset in sorted(set(baseline_dataset_dirs).intersection(candidate_dataset_dirs)):
        baseline_models = sorted(path for path in baseline_dataset_dirs[dataset].glob("model-*") if path.is_dir())
        candidate_models = sorted(path for path in candidate_dataset_dirs[dataset].glob("model-*") if path.is_dir())
        if not baseline_models or not candidate_models:
            continue
        baseline_run_dir = _find_run_dir_by_chunker(baseline_models[0], chunker)
        candidate_run_dir = _find_run_dir_by_chunker(candidate_models[0], chunker)
        if baseline_run_dir is None or candidate_run_dir is None:
            continue

        baseline_eval = _read_json(baseline_run_dir / "eval_report.json")
        candidate_eval = _read_json(candidate_run_dir / "eval_report.json")
        baseline_metrics = _extract_primary_metrics(baseline_eval)
        candidate_metrics = _extract_primary_metrics(candidate_eval)

        baseline_samples = _load_samples(baseline_run_dir)
        candidate_samples = _load_samples(candidate_run_dir)
        common_ids = sorted(set(baseline_samples).intersection(candidate_samples))

        sample_rows: List[Dict[str, Any]] = []
        for sample_id in common_ids:
            base_payload = baseline_samples[sample_id]
            cand_payload = candidate_samples[sample_id]
            base_stats = _sample_chunk_stats(base_payload)
            cand_stats = _sample_chunk_stats(cand_payload)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "bucket": cand_stats["bucket"],
                    "trigger_kind": _trigger_kind(
                        floor_applied=bool(cand_stats["floor_applied"]),
                        guard_applied=bool(cand_stats["guard_applied"]),
                    ),
                    "selected_changed": base_payload.get("selected_chunk_ids") != cand_payload.get("selected_chunk_ids"),
                    "ranking_changed": base_payload.get("chunk_ranking") != cand_payload.get("chunk_ranking"),
                    "baseline_chunk_count": int(base_stats["chunk_count"]),
                    "candidate_chunk_count": int(cand_stats["chunk_count"]),
                    "baseline_top20_count": int(base_stats["top20_count"]),
                    "candidate_top20_count": int(cand_stats["top20_count"]),
                    "baseline_fallback": bool(base_stats["fallback_applied"]),
                    "candidate_fallback": bool(cand_stats["fallback_applied"]),
                    "candidate_raw_count": float(cand_stats["raw_count"]),
                    "candidate_final_count": float(cand_stats["final_count"]),
                }
            )

        by_bucket = {
            bucket: _aggregate_row_group(rows)
            for bucket, rows in sorted(_group_rows(sample_rows, "bucket").items())
        }
        by_trigger = {
            trigger: _aggregate_row_group(rows)
            for trigger, rows in sorted(_group_rows(sample_rows, "trigger_kind").items())
        }

        shape = _dataset_shape_summary(sample_rows)
        metric_deltas = {
            metric: _delta_with_gain(candidate_metrics[metric], baseline_metrics[metric], metric)
            for metric in _DIRECTION_SIGNS
        }
        metric_deltas["runtime_seconds"] = {
            "candidate": float(candidate_metrics["runtime_seconds"]),
            "baseline": float(baseline_metrics["runtime_seconds"]),
            "delta": float(candidate_metrics["runtime_seconds"] - baseline_metrics["runtime_seconds"]),
            "directional_gain": float(baseline_metrics["runtime_seconds"] - candidate_metrics["runtime_seconds"]),
        }

        datasets.append(
            {
                "dataset": dataset,
                "baseline_run_dir": str(baseline_run_dir),
                "candidate_run_dir": str(candidate_run_dir),
                "sample_count_common": int(len(common_ids)),
                "metric_deltas": metric_deltas,
                "shape_summary": {
                    **shape,
                    "chunk_count_mean_delta": float(shape["chunk_count_mean_candidate"] - shape["chunk_count_mean_baseline"]),
                    "top20_zero_ratio_delta": float(shape["top20_zero_ratio_candidate"] - shape["top20_zero_ratio_baseline"]),
                    "fallback_rate_delta": float(shape["fallback_rate_candidate"] - shape["fallback_rate_baseline"]),
                },
                "bucket_groups": by_bucket,
                "trigger_groups": by_trigger,
                "provenance": {
                    "baseline_commit": (
                        baseline_eval.get("provenance", {})
                        .get("git", {})
                        .get("commit")
                    ),
                    "candidate_commit": (
                        candidate_eval.get("provenance", {})
                        .get("git", {})
                        .get("commit")
                    ),
                    "baseline_adaptive_profile": _read_json(baseline_run_dir / "run_config.json").get("adaptive_profile"),
                    "candidate_adaptive_profile": _read_json(candidate_run_dir / "run_config.json").get("adaptive_profile"),
                },
            }
        )

    return {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "chunker": chunker,
        "metric_direction_notes": {
            "log_odds": "log(p_perturbed)-log(p_full), lower is better",
            "comprehensiveness": "higher is better",
            "sufficiency": "lower is better",
            "aopc_comprehensiveness": "higher is better",
            "aopc_sufficiency": "lower is better",
        },
        "datasets": datasets,
    }


def _flatten_effect_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in report.get("datasets", []):
        md = ds.get("metric_deltas", {})
        shape = ds.get("shape_summary", {})
        trg = ds.get("trigger_groups", {})
        very_long = ds.get("bucket_groups", {}).get("very_long", {})
        rows.append(
            {
                "dataset": ds.get("dataset"),
                "sample_count_common": ds.get("sample_count_common"),
                "delta_log_odds": _safe_float(md.get("log_odds", {}).get("delta")),
                "gain_log_odds": _safe_float(md.get("log_odds", {}).get("directional_gain")),
                "delta_comp": _safe_float(md.get("comprehensiveness", {}).get("delta")),
                "gain_comp": _safe_float(md.get("comprehensiveness", {}).get("directional_gain")),
                "delta_suff": _safe_float(md.get("sufficiency", {}).get("delta")),
                "gain_suff": _safe_float(md.get("sufficiency", {}).get("directional_gain")),
                "delta_aopc_c": _safe_float(md.get("aopc_comprehensiveness", {}).get("delta")),
                "gain_aopc_c": _safe_float(md.get("aopc_comprehensiveness", {}).get("directional_gain")),
                "delta_aopc_s": _safe_float(md.get("aopc_sufficiency", {}).get("delta")),
                "gain_aopc_s": _safe_float(md.get("aopc_sufficiency", {}).get("directional_gain")),
                "delta_runtime_seconds": _safe_float(md.get("runtime_seconds", {}).get("delta")),
                "gain_runtime_seconds": _safe_float(md.get("runtime_seconds", {}).get("directional_gain")),
                "delta_chunk_count_mean": _safe_float(shape.get("chunk_count_mean_delta")),
                "delta_top20_zero_ratio": _safe_float(shape.get("top20_zero_ratio_delta")),
                "delta_fallback_rate": _safe_float(shape.get("fallback_rate_delta")),
                "trigger_floor_only_count": int(trg.get("floor_only", {}).get("count", 0)),
                "trigger_guard_only_count": int(trg.get("guard_only", {}).get("count", 0)),
                "trigger_both_count": int(trg.get("both", {}).get("count", 0)),
                "trigger_none_count": int(trg.get("none", {}).get("count", 0)),
                "very_long_selected_changed_ratio": _safe_float(very_long.get("selected_changed_ratio")),
                "very_long_ranking_changed_ratio": _safe_float(very_long.get("ranking_changed_ratio")),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive chunking effect model report")
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--candidate-root", type=str, required=True)
    parser.add_argument("--chunker", type=str, default="adaptive")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_root = Path(args.baseline_root)
    candidate_root = Path(args.candidate_root)
    report = build_effect_model(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        chunker=str(args.chunker),
    )

    out_json = Path(args.output_json) if args.output_json else (candidate_root / "adaptive_effect_model.json")
    out_csv = Path(args.output_csv) if args.output_csv else (candidate_root / "adaptive_effect_model.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _flatten_effect_rows(report)
    fieldnames = [
        "dataset",
        "sample_count_common",
        "delta_log_odds",
        "gain_log_odds",
        "delta_comp",
        "gain_comp",
        "delta_suff",
        "gain_suff",
        "delta_aopc_c",
        "gain_aopc_c",
        "delta_aopc_s",
        "gain_aopc_s",
        "delta_runtime_seconds",
        "gain_runtime_seconds",
        "delta_chunk_count_mean",
        "delta_top20_zero_ratio",
        "delta_fallback_rate",
        "trigger_floor_only_count",
        "trigger_guard_only_count",
        "trigger_both_count",
        "trigger_none_count",
        "very_long_selected_changed_ratio",
        "very_long_ranking_changed_ratio",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[adaptive-effect-model] json={out_json}")
    print(f"[adaptive-effect-model] csv={out_csv}")


if __name__ == "__main__":
    main()
