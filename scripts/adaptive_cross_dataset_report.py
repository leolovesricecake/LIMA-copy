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


def _extract_metrics(eval_report: Dict[str, Any]) -> Dict[str, float]:
    gold = (
        eval_report.get("metrics_by_target", {})
        .get("gold", {})
        .get("metrics_primary", {})
    )
    sec = eval_report.get("metrics_secondary", {})
    return {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comprehensiveness": _safe_float(gold.get("comprehensiveness")),
        "sufficiency": _safe_float(gold.get("sufficiency")),
        "aopc_comprehensiveness": _safe_float(gold.get("aopc_comprehensiveness")),
        "aopc_sufficiency": _safe_float(gold.get("aopc_sufficiency")),
        "runtime_seconds": _safe_float(sec.get("runtime_seconds")),
    }


def _collect_sample_stats(samples_dir: Path) -> Dict[str, Any]:
    sample_paths = sorted(samples_dir.glob("*.json"))
    if not sample_paths:
        return {
            "sample_count": 0,
            "top20_count_zero_ratio": 0.0,
            "top20_count_mean": 0.0,
            "fallback_rate": 0.0,
            "adaptive_bucket_counts": {},
            "adaptive_bucket_ratios": {},
            "raw_to_final_delta_mean": 0.0,
            "fragmentation_index_mean": 0.0,
            "very_long_raw_single_ratio": 0.0,
            "very_long_fragmentation_index_mean": 0.0,
        }

    top20_counts: List[int] = []
    fallback_hits = 0
    adaptive_bucket_counts: Dict[str, int] = {}
    raw_values: List[float] = []
    final_values: List[float] = []
    frag_values: List[float] = []
    very_long_rows = 0
    very_long_raw_single_rows = 0
    very_long_frag_values: List[float] = []

    for path in sample_paths:
        payload = _read_json(path)
        chunks = payload.get("chunks", [])
        total_chunks = int(len(chunks))
        top20_counts.append(int(math.floor(0.2 * float(total_chunks))))

        diag = payload.get("metadata", {}).get("chunk_diagnostics", {})
        if not isinstance(diag, dict):
            continue

        if bool(diag.get("fallback_applied", False)):
            fallback_hits += 1

        bucket = diag.get("adaptive_bucket")
        if isinstance(bucket, str):
            adaptive_bucket_counts[bucket] = adaptive_bucket_counts.get(bucket, 0) + 1

        stage = diag.get("adaptive_stage_chunk_counts", {})
        if isinstance(stage, dict):
            raw = _safe_float(stage.get("raw"))
            final = _safe_float(stage.get("final"))
            if raw > 0:
                raw_values.append(raw)
                final_values.append(final)
                frag_values.append(final / raw)
            if bucket == "very_long" and raw > 0:
                very_long_rows += 1
                if raw <= 1.0:
                    very_long_raw_single_rows += 1
                very_long_frag_values.append(final / raw)

    sample_count = len(sample_paths)
    top20_zero = sum(1 for x in top20_counts if x == 0)
    bucket_ratios = {
        key: (float(value) / float(sample_count))
        for key, value in sorted(adaptive_bucket_counts.items())
    }
    raw_to_final_delta_mean = 0.0
    if raw_values and final_values:
        raw_to_final_delta_mean = (
            float(sum(final_values)) - float(sum(raw_values))
        ) / float(len(raw_values))

    return {
        "sample_count": int(sample_count),
        "top20_count_zero_ratio": float(top20_zero / float(sample_count)),
        "top20_count_mean": float(sum(top20_counts) / float(sample_count)),
        "fallback_rate": float(fallback_hits / float(sample_count)),
        "adaptive_bucket_counts": dict(sorted(adaptive_bucket_counts.items())),
        "adaptive_bucket_ratios": bucket_ratios,
        "raw_to_final_delta_mean": float(raw_to_final_delta_mean),
        "fragmentation_index_mean": (
            float(sum(frag_values) / float(len(frag_values))) if frag_values else 0.0
        ),
        "very_long_raw_single_ratio": (
            float(very_long_raw_single_rows / float(very_long_rows)) if very_long_rows > 0 else 0.0
        ),
        "very_long_fragmentation_index_mean": (
            float(sum(very_long_frag_values) / float(len(very_long_frag_values)))
            if very_long_frag_values
            else 0.0
        ),
    }


def _load_run(run_dir: Path) -> Dict[str, Any]:
    eval_path = run_dir / "eval_report.json"
    run_cfg_path = run_dir / "run_config.json"
    samples_dir = run_dir / "samples"
    if not eval_path.exists() or not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing run files under {run_dir}")

    eval_report = _read_json(eval_path)
    run_cfg = _read_json(run_cfg_path)
    sample_stats = _collect_sample_stats(samples_dir)
    return {
        "run_dir": str(run_dir),
        "chunker": _detect_chunker(run_dir.name),
        "metrics": _extract_metrics(eval_report),
        "sample_stats": sample_stats,
        "provenance": {
            "git_commit": (
                eval_report.get("provenance", {})
                .get("git", {})
                .get("commit")
            ),
            "git_dirty": (
                eval_report.get("provenance", {})
                .get("git", {})
                .get("dirty")
            ),
            "deterministic": bool(run_cfg.get("deterministic", False)),
            "split": run_cfg.get("split"),
            "max_samples": run_cfg.get("max_samples"),
        },
    }


def _metric_delta_rows(candidate: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for metric_name in _DIRECTION_SIGNS:
        cand = float(candidate.get(metric_name, 0.0))
        base = float(baseline.get(metric_name, 0.0))
        delta = cand - base
        rows[metric_name] = {
            "candidate": cand,
            "baseline": base,
            "delta": delta,
            "directional_gain": float(_DIRECTION_SIGNS[metric_name] * delta),
        }
    # runtime is reported separately with lower-better gain.
    cand_rt = float(candidate.get("runtime_seconds", 0.0))
    base_rt = float(baseline.get("runtime_seconds", 0.0))
    rows["runtime_seconds"] = {
        "candidate": cand_rt,
        "baseline": base_rt,
        "delta": cand_rt - base_rt,
        "directional_gain": base_rt - cand_rt,
    }
    return rows


def _pairwise(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    metric_deltas = _metric_delta_rows(candidate["metrics"], baseline["metrics"])
    ss_c = candidate["sample_stats"]
    ss_b = baseline["sample_stats"]
    return {
        "candidate_chunker": candidate["chunker"],
        "baseline_chunker": baseline["chunker"],
        "metric_deltas": metric_deltas,
        "sample_stats_delta": {
            "top20_count_zero_ratio_delta": float(ss_c["top20_count_zero_ratio"] - ss_b["top20_count_zero_ratio"]),
            "top20_count_mean_delta": float(ss_c["top20_count_mean"] - ss_b["top20_count_mean"]),
            "fallback_rate_delta": float(ss_c["fallback_rate"] - ss_b["fallback_rate"]),
            "fragmentation_index_mean_delta": float(ss_c["fragmentation_index_mean"] - ss_b["fragmentation_index_mean"]),
            "raw_to_final_delta_mean_delta": float(ss_c["raw_to_final_delta_mean"] - ss_b["raw_to_final_delta_mean"]),
            "very_long_raw_single_ratio_delta": float(
                ss_c["very_long_raw_single_ratio"] - ss_b["very_long_raw_single_ratio"]
            ),
            "very_long_fragmentation_index_mean_delta": float(
                ss_c["very_long_fragmentation_index_mean"] - ss_b["very_long_fragmentation_index_mean"]
            ),
        },
    }


def build_report(results_root: Path) -> Dict[str, Any]:
    dataset_rows: List[Dict[str, Any]] = []
    for dataset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        model_dirs = sorted(path for path in dataset_dir.glob("model-*") if path.is_dir())
        if not model_dirs:
            continue
        # Phase D currently runs one model; keep first deterministic ordering.
        model_dir = model_dirs[0]
        runs: Dict[str, Dict[str, Any]] = {}
        for run_dir in sorted(path for path in model_dir.glob("chunk-*_method-ours") if path.is_dir()):
            run = _load_run(run_dir)
            runs[run["chunker"]] = run

        pairwise: Dict[str, Any] = {}
        if "adaptive" in runs and "sentence" in runs:
            pairwise["adaptive_vs_sentence"] = _pairwise(runs["adaptive"], runs["sentence"])
        if "adaptive" in runs and "sentence_v2" in runs:
            pairwise["adaptive_vs_sentence_v2"] = _pairwise(runs["adaptive"], runs["sentence_v2"])

        dataset_rows.append(
            {
                "dataset": dataset_dir.name,
                "model_dir": str(model_dir),
                "runs": runs,
                "pairwise": pairwise,
            }
        )

    return {
        "results_root": str(results_root),
        "direction_signs": dict(_DIRECTION_SIGNS),
        "datasets": dataset_rows,
    }


def _flatten_pairwise_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds_row in report.get("datasets", []):
        dataset = str(ds_row.get("dataset"))
        pairwise = ds_row.get("pairwise", {})
        for pair_name, detail in sorted(pairwise.items()):
            md = detail.get("metric_deltas", {})
            sd = detail.get("sample_stats_delta", {})
            rows.append(
                {
                    "dataset": dataset,
                    "pair": pair_name,
                    "candidate_chunker": detail.get("candidate_chunker"),
                    "baseline_chunker": detail.get("baseline_chunker"),
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
                    "delta_top20_zero_ratio": _safe_float(sd.get("top20_count_zero_ratio_delta")),
                    "delta_top20_count_mean": _safe_float(sd.get("top20_count_mean_delta")),
                    "delta_fallback_rate": _safe_float(sd.get("fallback_rate_delta")),
                    "delta_fragmentation_index_mean": _safe_float(sd.get("fragmentation_index_mean_delta")),
                    "delta_raw_to_final_delta_mean": _safe_float(sd.get("raw_to_final_delta_mean_delta")),
                    "delta_very_long_raw_single_ratio": _safe_float(sd.get("very_long_raw_single_ratio_delta")),
                    "delta_very_long_fragmentation_index_mean": _safe_float(
                        sd.get("very_long_fragmentation_index_mean_delta")
                    ),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-dataset adaptive diagnostics report")
    parser.add_argument("--results-root", type=str, default="lima_llm_results-phase-d")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_root)
    report = build_report(root)

    out_json = Path(args.output_json) if args.output_json else (root / "adaptive_cross_dataset_report.json")
    out_csv = Path(args.output_csv) if args.output_csv else (root / "adaptive_cross_dataset_report.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _flatten_pairwise_rows(report)
    fieldnames = [
        "dataset",
        "pair",
        "candidate_chunker",
        "baseline_chunker",
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
        "delta_top20_zero_ratio",
        "delta_top20_count_mean",
        "delta_fallback_rate",
        "delta_fragmentation_index_mean",
        "delta_raw_to_final_delta_mean",
        "delta_very_long_raw_single_ratio",
        "delta_very_long_fragmentation_index_mean",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[adaptive-cross] json={out_json}")
    print(f"[adaptive-cross] csv={out_csv}")


if __name__ == "__main__":
    main()
