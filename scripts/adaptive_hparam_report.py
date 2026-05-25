#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


_LONG_TEXT_DATASETS = {"eraser_movie_reviews", "imdb"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_dataset_reports(search_root: Path) -> List[Path]:
    direct = search_root / "search_trials"
    if direct.exists():
        return sorted(direct.glob("*/adaptive_hparam_search.json"))
    return sorted(search_root.glob("*/adaptive_hparam_search.json"))


def _trial_rows(dataset_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in ("train_trials", "stage2_trials", "dev_trials"):
        for row in dataset_payload.get(key, []) or []:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _preferred_rows(dataset_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ranked_dev = dataset_payload.get("ranked_dev", []) or []
    if ranked_dev:
        return [dict(row) for row in ranked_dev if isinstance(row, Mapping)]
    ranked_train = dataset_payload.get("ranked_train", []) or []
    return [dict(row) for row in ranked_train if isinstance(row, Mapping)]


def _robust_score(dataset: str, row: Mapping[str, Any]) -> float:
    score = _safe_float(row.get("score_summary", {}).get("score"))
    quality = _safe_float(row.get("score_summary", {}).get("quality_gain_sum"))
    runtime_gain = _safe_float(row.get("score_summary", {}).get("runtime_gain_seconds"))
    major_reg = int(_safe_float(row.get("score_summary", {}).get("major_regression_count")))

    drift = row.get("drift_vs_dev_baseline", row.get("drift_vs_stage1_baseline", {}))
    selected_drift = _safe_float(drift.get("selected_changed_ratio"))
    ranking_drift = _safe_float(drift.get("ranking_changed_ratio"))

    reg_penalty = float(major_reg * (2.0 if dataset in _LONG_TEXT_DATASETS else 1.0))
    drift_penalty = 0.5 * selected_drift + 0.5 * ranking_drift
    runtime_bonus = 1e-4 * runtime_gain

    return float(score + quality + runtime_bonus - reg_penalty - drift_penalty)


def _summarize_dataset(dataset_payload: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = str(dataset_payload.get("dataset", ""))
    candidates = _preferred_rows(dataset_payload)
    if not candidates:
        return {
            "dataset": dataset,
            "status": "failed",
            "best_trial_id": "",
            "selection_source": "none",
            "robust_score": 0.0,
            "score": 0.0,
            "quality_gain_sum": 0.0,
            "runtime_gain_seconds": 0.0,
            "major_regression_count": 0,
            "adaptive_overrides_json": "{}",
            "run_dir": "",
        }

    ranked = sorted(
        candidates,
        key=lambda row: (
            -_robust_score(dataset, row),
            -_safe_float(row.get("score_summary", {}).get("score")),
            -_safe_float(row.get("score_summary", {}).get("quality_gain_sum")),
            -_safe_float(row.get("score_summary", {}).get("runtime_gain_seconds")),
            str(row.get("trial_id", "")),
        ),
    )
    best = ranked[0]
    source = "ranked_dev" if (dataset_payload.get("ranked_dev") or []) else "ranked_train"

    drift = best.get("drift_vs_dev_baseline", best.get("drift_vs_stage1_baseline", {}))
    return {
        "dataset": dataset,
        "status": best.get("status"),
        "best_trial_id": best.get("trial_id"),
        "selection_source": source,
        "robust_score": _robust_score(dataset, best),
        "score": _safe_float(best.get("score_summary", {}).get("score")),
        "quality_gain_sum": _safe_float(best.get("score_summary", {}).get("quality_gain_sum")),
        "runtime_gain_seconds": _safe_float(best.get("score_summary", {}).get("runtime_gain_seconds")),
        "major_regression_count": int(_safe_float(best.get("score_summary", {}).get("major_regression_count"))),
        "selected_changed_ratio": _safe_float(drift.get("selected_changed_ratio")),
        "ranking_changed_ratio": _safe_float(drift.get("ranking_changed_ratio")),
        "trace_total_score_max_abs_diff": _safe_float(drift.get("trace_total_score_max_abs_diff")),
        "adaptive_overrides_json": json.dumps(best.get("adaptive_overrides") or {}, ensure_ascii=False, sort_keys=True),
        "run_dir": best.get("run_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize adaptive hparam search outputs")
    parser.add_argument("--search-root", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    search_root = Path(args.search_root)
    report_paths = _discover_dataset_reports(search_root)

    datasets: List[Dict[str, Any]] = []
    for path in report_paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        datasets.append(dict(payload))

    summaries = [_summarize_dataset(ds) for ds in datasets]
    global_override_counts: Dict[str, int] = {}
    for row in summaries:
        key = str(row.get("adaptive_overrides_json", "{}"))
        global_override_counts[key] = global_override_counts.get(key, 0) + 1

    payload = {
        "search_root": str(search_root),
        "dataset_count": int(len(summaries)),
        "dataset_summaries": summaries,
        "global_override_frequency": dict(sorted(global_override_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

    out_json = Path(args.output_json) if args.output_json else (search_root / "adaptive_hparam_report.json")
    out_csv = Path(args.output_csv) if args.output_csv else (search_root / "adaptive_hparam_report.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "dataset",
        "status",
        "best_trial_id",
        "selection_source",
        "robust_score",
        "score",
        "quality_gain_sum",
        "runtime_gain_seconds",
        "major_regression_count",
        "selected_changed_ratio",
        "ranking_changed_ratio",
        "trace_total_score_max_abs_diff",
        "adaptive_overrides_json",
        "run_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"[adaptive-hparam-report] json={out_json}")
    print(f"[adaptive-hparam-report] csv={out_csv}")


if __name__ == "__main__":
    main()
