#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalized_eval_granularity(cfg: Dict[str, Any]) -> str:
    value = str(cfg.get("eval_granularity", "word")).strip().lower()
    if value not in {"word", "token"}:
        return "word"
    return value


def _group_key_from_config(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        cfg.get("dataset"),
        cfg.get("split"),
        cfg.get("model_path"),
        cfg.get("chunker"),
        cfg.get("search"),
        cfg.get("k"),
        cfg.get("lambdas"),
        cfg.get("eval_q_values"),
        _normalized_eval_granularity(cfg),
        cfg.get("max_samples"),
        cfg.get("seed"),
    )


def _group_id_from_config(cfg: Dict[str, Any]) -> str:
    model = Path(str(cfg.get("model_path", ""))).name
    return (
        f"{cfg.get('dataset')}/{cfg.get('split')}"
        f"|model={model}"
        f"|chunk={cfg.get('chunker')}"
        f"|search={cfg.get('search')}"
        f"|k={cfg.get('k')}"
        f"|lam={cfg.get('lambdas')}"
        f"|eval={_normalized_eval_granularity(cfg)}"
        f"|seed={cfg.get('seed')}"
    )


def _collect_explain_stats(sample_dir: Path) -> Dict[str, Any]:
    elapsed: List[float] = []
    chunk_counts: List[float] = []
    selected_counts: List[float] = []
    timing_totals = {
        "chunk_build_seconds": 0.0,
        "search_seconds": 0.0,
        "model_prefetch_seconds": 0.0,
    }
    cache_totals = {
        "subset_requested": 0,
        "subset_cache_hits": 0,
        "subset_cache_misses": 0,
        "prob_cache_hits": 0,
        "prob_cache_misses": 0,
        "embed_cache_hits": 0,
        "embed_cache_misses": 0,
        "evaluate_gains_calls": 0,
    }
    cache_samples = 0
    chunk_diag_samples = 0
    chunk_strategy_counter: Counter[str] = Counter()
    fallback_count = 0
    orphan_chunks_total = 0.0
    orphan_samples = 0
    orphan_merge_total = 0.0
    orphan_merge_samples = 0
    cross_newline_total = 0.0
    cross_newline_samples = 0
    forward_final = {
        "predict_calls": 0,
        "embed_calls": 0,
        "gradient_calls": 0,
    }

    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        meta = payload.get("metadata", {})
        elapsed.append(_safe_float(meta.get("elapsed_seconds")))
        chunk_counts.append(float(len(payload.get("chunks", []))))
        selected_counts.append(float(len(payload.get("selected_chunk_ids", []))))
        timing_payload = meta.get("explain_timing_breakdown", {})
        for key in timing_totals:
            timing_totals[key] += _safe_float(timing_payload.get(key))

        cache_payload = meta.get("objective_cache_stats")
        if isinstance(cache_payload, dict):
            cache_samples += 1
            for key in cache_totals:
                cache_totals[key] += int(_safe_float(cache_payload.get(key), 0.0))

        chunk_diag = meta.get("chunk_diagnostics")
        if isinstance(chunk_diag, dict):
            chunk_diag_samples += 1
            chunk_strategy_counter[str(chunk_diag.get("chunk_strategy", "unknown"))] += 1
            if bool(chunk_diag.get("fallback_applied", False)):
                fallback_count += 1
            orphan_val = _safe_float(chunk_diag.get("singleton_orphan_punctuation_chunks"), 0.0)
            orphan_merge_val = _safe_float(chunk_diag.get("orphan_merge_count"), 0.0)
            cross_val = _safe_float(chunk_diag.get("cross_newline_boundary_chunks"), 0.0)
            orphan_chunks_total += orphan_val
            orphan_merge_total += orphan_merge_val
            cross_newline_total += cross_val
            if orphan_val > 0.0:
                orphan_samples += 1
            if orphan_merge_val > 0.0:
                orphan_merge_samples += 1
            if cross_val > 0.0:
                cross_newline_samples += 1

        fc = meta.get("forward_counters", {})
        for key in forward_final:
            forward_final[key] = max(forward_final[key], int(_safe_float(fc.get(key), 0.0)))

    n = len(elapsed)
    if n == 0:
        return {
            "sample_count": 0,
            "explain_seconds_total": 0.0,
            "explain_seconds_mean": 0.0,
            "explain_seconds_p50": 0.0,
            "explain_seconds_p90": 0.0,
            "explain_seconds_max": 0.0,
            "chunk_count_mean": 0.0,
            "selected_count_mean": 0.0,
            "forward_counters_final": forward_final,
            "explain_timing_breakdown_totals": timing_totals,
            "objective_cache_stats": {
                **cache_totals,
                "samples_with_cache_stats": 0,
                "subset_cache_hit_rate": 0.0,
                "prob_cache_hit_rate": 0.0,
                "embed_cache_hit_rate": 0.0,
            },
            "chunk_diagnostics": {
                "samples_with_chunk_diagnostics": 0,
                "chunk_strategy_counts": {},
                "fallback_rate": 0.0,
                "orphan_chunks_mean": 0.0,
                "orphan_samples_ratio": 0.0,
                "orphan_merge_count_mean": 0.0,
                "orphan_merge_samples_ratio": 0.0,
                "cross_newline_chunks_mean": 0.0,
                "cross_newline_samples_ratio": 0.0,
            },
        }

    elapsed_sorted = sorted(elapsed)

    def _percentile(xs: List[float], q: float) -> float:
        if not xs:
            return 0.0
        idx = int(round((len(xs) - 1) * q))
        idx = max(0, min(len(xs) - 1, idx))
        return float(xs[idx])

    subset_requested = int(cache_totals["subset_requested"])
    prob_total = int(cache_totals["prob_cache_hits"] + cache_totals["prob_cache_misses"])
    embed_total = int(cache_totals["embed_cache_hits"] + cache_totals["embed_cache_misses"])

    return {
        "sample_count": n,
        "explain_seconds_total": float(sum(elapsed)),
        "explain_seconds_mean": float(sum(elapsed) / n),
        "explain_seconds_p50": _percentile(elapsed_sorted, 0.50),
        "explain_seconds_p90": _percentile(elapsed_sorted, 0.90),
        "explain_seconds_max": float(max(elapsed_sorted)),
        "chunk_count_mean": float(sum(chunk_counts) / n),
        "selected_count_mean": float(sum(selected_counts) / n),
        "forward_counters_final": forward_final,
        "explain_timing_breakdown_totals": timing_totals,
        "objective_cache_stats": {
            **cache_totals,
            "samples_with_cache_stats": int(cache_samples),
            "subset_cache_hit_rate": (
                float(cache_totals["subset_cache_hits"]) / float(subset_requested)
                if subset_requested > 0
                else 0.0
            ),
            "prob_cache_hit_rate": (
                float(cache_totals["prob_cache_hits"]) / float(prob_total)
                if prob_total > 0
                else 0.0
            ),
            "embed_cache_hit_rate": (
                float(cache_totals["embed_cache_hits"]) / float(embed_total)
                if embed_total > 0
                else 0.0
            ),
        },
        "chunk_diagnostics": {
            "samples_with_chunk_diagnostics": int(chunk_diag_samples),
            "chunk_strategy_counts": dict(chunk_strategy_counter),
            "fallback_rate": (float(fallback_count) / float(chunk_diag_samples)) if chunk_diag_samples > 0 else 0.0,
            "orphan_chunks_mean": (
                float(orphan_chunks_total) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
            "orphan_samples_ratio": (
                float(orphan_samples) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
            "orphan_merge_count_mean": (
                float(orphan_merge_total) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
            "orphan_merge_samples_ratio": (
                float(orphan_merge_samples) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
            "cross_newline_chunks_mean": (
                float(cross_newline_total) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
            "cross_newline_samples_ratio": (
                float(cross_newline_samples) / float(chunk_diag_samples)
                if chunk_diag_samples > 0
                else 0.0
            ),
        },
    }


def _extract_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    sec = report.get("metrics_secondary", {})
    gold = report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    pred = report.get("metrics_by_target", {}).get("predicted", {}).get("metrics_primary", {})

    eval_seconds = _safe_float(sec.get("runtime_seconds"))
    row = {
        "eval_seconds": eval_seconds,
        "eval_forward_counters": {
            "predict_calls": int(_safe_float(sec.get("forward_counters_delta", {}).get("predict_calls", 0.0))),
            "embed_calls": int(_safe_float(sec.get("forward_counters_delta", {}).get("embed_calls", 0.0))),
            "gradient_calls": int(_safe_float(sec.get("forward_counters_delta", {}).get("gradient_calls", 0.0))),
        },
        "plausibility_f1": _safe_float(sec.get("plausibility_f1")),
        "plausibility_iou": _safe_float(sec.get("plausibility_iou")),
        "sparsity": _safe_float(sec.get("sparsity")),
        "accuracy_full": _safe_float(report.get("metrics_primary", {}).get("accuracy_full")),
        "gold": {
            "log_odds": _safe_float(gold.get("log_odds")),
            "comprehensiveness": _safe_float(gold.get("comprehensiveness")),
            "sufficiency": _safe_float(gold.get("sufficiency")),
            "aopc": _safe_float(gold.get("aopc")),
            "aopc_sufficiency": _safe_float(gold.get("aopc_sufficiency")),
            "aopc_comprehensiveness": _safe_float(gold.get("aopc_comprehensiveness")),
        },
        "predicted": {
            "log_odds": _safe_float(pred.get("log_odds")),
            "comprehensiveness": _safe_float(pred.get("comprehensiveness")),
            "sufficiency": _safe_float(pred.get("sufficiency")),
            "aopc": _safe_float(pred.get("aopc")),
            "aopc_sufficiency": _safe_float(pred.get("aopc_sufficiency")),
            "aopc_comprehensiveness": _safe_float(pred.get("aopc_comprehensiveness")),
        },
    }
    return row


def _pairwise(primary: Dict[str, Any], reference: Dict[str, Any], primary_name: str, reference_name: str) -> Dict[str, Any]:
    pg = primary["metrics"]["gold"]
    rg = reference["metrics"]["gold"]
    pair = {
        "primary": primary_name,
        "reference": reference_name,
        "gold_comp_adv": float(pg["comprehensiveness"] - rg["comprehensiveness"]),
        "gold_suff_adv": float(rg["sufficiency"] - pg["sufficiency"]),
        "gold_pass": bool((pg["comprehensiveness"] > rg["comprehensiveness"]) and (pg["sufficiency"] < rg["sufficiency"])),
        "primary_total_seconds": float(primary["timing"]["total_seconds"]),
        "reference_total_seconds": float(reference["timing"]["total_seconds"]),
        "primary_eval_seconds": float(primary["timing"]["eval_seconds"]),
        "reference_eval_seconds": float(reference["timing"]["eval_seconds"]),
        "primary_explain_seconds": float(primary["timing"]["explain_seconds_total"]),
        "reference_explain_seconds": float(reference["timing"]["explain_seconds_total"]),
    }
    return pair


def build_snapshot(results_root: Path, primary_method: str, reference_method: str) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for config_path in sorted(results_root.glob("**/run_config.json")):
        run_dir = config_path.parent
        report_path = run_dir / "eval_report.json"
        cfg = _read_json(config_path)
        report = _read_json(report_path)
        if cfg is None or report is None:
            continue

        method = str(report.get("report_method") or cfg.get("explain_method") or "unknown").strip().lower()
        explain_stats = _collect_explain_stats(run_dir / "samples")
        metrics = _extract_metrics(report)
        timing = {
            "explain_seconds_total": float(explain_stats["explain_seconds_total"]),
            "eval_seconds": float(metrics["eval_seconds"]),
            "total_seconds": float(explain_stats["explain_seconds_total"] + metrics["eval_seconds"]),
            "explain_seconds_mean": float(explain_stats["explain_seconds_mean"]),
            "explain_seconds_p50": float(explain_stats["explain_seconds_p50"]),
            "explain_seconds_p90": float(explain_stats["explain_seconds_p90"]),
            "explain_seconds_max": float(explain_stats["explain_seconds_max"]),
        }
        run = {
            "group_key": _group_key_from_config(cfg),
            "group_id": _group_id_from_config(cfg),
            "run_dir": str(run_dir),
            "method": method,
            "config": cfg,
            "report_path": str(report_path),
            "sample_count": int(_safe_float(report.get("sample_count"), 0.0)),
            "metrics": metrics,
            "timing": timing,
            "explain": {
                "chunk_count_mean": float(explain_stats["chunk_count_mean"]),
                "selected_count_mean": float(explain_stats["selected_count_mean"]),
                "forward_counters_final": explain_stats["forward_counters_final"],
                "timing_breakdown_totals": explain_stats["explain_timing_breakdown_totals"],
                "objective_cache_stats": explain_stats["objective_cache_stats"],
                "chunk_diagnostics": explain_stats["chunk_diagnostics"],
            },
        }
        runs.append(run)

    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for run in runs:
        key = run["group_key"]
        if key not in groups:
            groups[key] = {
                "group_id": run["group_id"],
                "methods": {},
            }
        groups[key]["methods"][run["method"]] = run

    out_groups: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda x: str(x)):
        entry = groups[key]
        methods = entry["methods"]
        pair = None
        if primary_method in methods and reference_method in methods:
            pair = _pairwise(methods[primary_method], methods[reference_method], primary_method, reference_method)

        out_groups.append(
            {
                "group_id": entry["group_id"],
                "methods": methods,
                "pairwise": pair,
            }
        )

    return {
        "results_root": str(results_root),
        "primary_method": primary_method,
        "reference_method": reference_method,
        "group_count": len(out_groups),
        "groups": out_groups,
    }


def _flatten_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in snapshot.get("groups", []):
        gid = group.get("group_id")
        methods = group.get("methods", {})
        for method, run in sorted(methods.items()):
            gold = run["metrics"]["gold"]
            pred = run["metrics"]["predicted"]
            timing = run["timing"]
            rows.append(
                {
                    "group_id": gid,
                    "eval_granularity": _normalized_eval_granularity(run.get("config", {})),
                    "method": method,
                    "sample_count": run.get("sample_count"),
                    "explain_seconds_total": timing["explain_seconds_total"],
                    "eval_seconds": timing["eval_seconds"],
                    "total_seconds": timing["total_seconds"],
                    "gold_comp": gold["comprehensiveness"],
                    "gold_suff": gold["sufficiency"],
                    "gold_aopc": gold["aopc"],
                    "plaus_f1": run["metrics"]["plausibility_f1"],
                    "plaus_iou": run["metrics"]["plausibility_iou"],
                    "pred_comp": pred["comprehensiveness"],
                    "pred_suff": pred["sufficiency"],
                    "accuracy_full": run["metrics"]["accuracy_full"],
                    "eval_predict_calls": run["metrics"]["eval_forward_counters"]["predict_calls"],
                    "eval_embed_calls": run["metrics"]["eval_forward_counters"]["embed_calls"],
                    "eval_gradient_calls": run["metrics"]["eval_forward_counters"]["gradient_calls"],
                    "explain_predict_calls": run["explain"]["forward_counters_final"]["predict_calls"],
                    "explain_embed_calls": run["explain"]["forward_counters_final"]["embed_calls"],
                    "explain_gradient_calls": run["explain"]["forward_counters_final"]["gradient_calls"],
                    "explain_chunk_build_seconds_total": run["explain"]["timing_breakdown_totals"]["chunk_build_seconds"],
                    "explain_search_seconds_total": run["explain"]["timing_breakdown_totals"]["search_seconds"],
                    "explain_model_prefetch_seconds_total": run["explain"]["timing_breakdown_totals"][
                        "model_prefetch_seconds"
                    ],
                    "explain_subset_cache_hit_rate": run["explain"]["objective_cache_stats"]["subset_cache_hit_rate"],
                    "explain_prob_cache_hit_rate": run["explain"]["objective_cache_stats"]["prob_cache_hit_rate"],
                    "explain_embed_cache_hit_rate": run["explain"]["objective_cache_stats"]["embed_cache_hit_rate"],
                    "explain_evaluate_gains_calls": run["explain"]["objective_cache_stats"]["evaluate_gains_calls"],
                    "chunk_diag_fallback_rate": run["explain"]["chunk_diagnostics"]["fallback_rate"],
                    "chunk_diag_orphan_chunks_mean": run["explain"]["chunk_diagnostics"]["orphan_chunks_mean"],
                    "chunk_diag_orphan_samples_ratio": run["explain"]["chunk_diagnostics"]["orphan_samples_ratio"],
                    "chunk_diag_orphan_merge_count_mean": run["explain"]["chunk_diagnostics"][
                        "orphan_merge_count_mean"
                    ],
                    "chunk_diag_orphan_merge_samples_ratio": run["explain"]["chunk_diagnostics"][
                        "orphan_merge_samples_ratio"
                    ],
                    "chunk_diag_cross_newline_chunks_mean": run["explain"]["chunk_diagnostics"][
                        "cross_newline_chunks_mean"
                    ],
                    "chunk_diag_cross_newline_samples_ratio": run["explain"]["chunk_diagnostics"][
                        "cross_newline_samples_ratio"
                    ],
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build unified analysis snapshot from run directories")
    p.add_argument("--results-root", type=str, default="lima_llm_results")
    p.add_argument("--primary-method", type=str, default="ours")
    p.add_argument("--reference-method", type=str, default="gradient")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_root)
    out_json = Path(args.output_json) if args.output_json else (root / "analysis_snapshot.json")
    out_csv = Path(args.output_csv) if args.output_csv else (root / "analysis_snapshot.csv")

    snapshot = build_snapshot(
        results_root=root,
        primary_method=str(args.primary_method).strip().lower(),
        reference_method=str(args.reference_method).strip().lower(),
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _flatten_rows(snapshot)
    fieldnames = [
        "group_id",
        "eval_granularity",
        "method",
        "sample_count",
        "explain_seconds_total",
        "eval_seconds",
        "total_seconds",
        "gold_comp",
        "gold_suff",
        "gold_aopc",
        "plaus_f1",
        "plaus_iou",
        "pred_comp",
        "pred_suff",
        "accuracy_full",
        "eval_predict_calls",
        "eval_embed_calls",
        "eval_gradient_calls",
        "explain_predict_calls",
        "explain_embed_calls",
        "explain_gradient_calls",
        "explain_chunk_build_seconds_total",
        "explain_search_seconds_total",
        "explain_model_prefetch_seconds_total",
        "explain_subset_cache_hit_rate",
        "explain_prob_cache_hit_rate",
        "explain_embed_cache_hit_rate",
        "explain_evaluate_gains_calls",
        "chunk_diag_fallback_rate",
        "chunk_diag_orphan_chunks_mean",
        "chunk_diag_orphan_samples_ratio",
        "chunk_diag_orphan_merge_count_mean",
        "chunk_diag_orphan_merge_samples_ratio",
        "chunk_diag_cross_newline_chunks_mean",
        "chunk_diag_cross_newline_samples_ratio",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[snapshot] groups={snapshot.get('group_count', 0)}")
    print(f"[snapshot] json={out_json}")
    print(f"[snapshot] csv={out_csv}")


if __name__ == "__main__":
    main()
