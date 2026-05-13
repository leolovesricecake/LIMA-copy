from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _normalized_eval_granularity(run_config: Dict[str, Any]) -> str:
    value = str(run_config.get("eval_granularity", "word")).strip().lower()
    if value not in {"word", "token"}:
        return "word"
    return value


def _group_config(run_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": run_config.get("dataset"),
        "split": run_config.get("split"),
        "model_path": run_config.get("model_path"),
        "chunker": run_config.get("chunker"),
        "search": run_config.get("search"),
        "k": run_config.get("k"),
        "lambdas": run_config.get("lambdas"),
        "eval_q_values": run_config.get("eval_q_values"),
        "eval_granularity": _normalized_eval_granularity(run_config),
        "max_samples": run_config.get("max_samples"),
    }


def _group_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        config.get("dataset"),
        config.get("split"),
        config.get("model_path"),
        config.get("chunker"),
        config.get("search"),
        config.get("k"),
        config.get("lambdas"),
        config.get("eval_q_values"),
        config.get("eval_granularity"),
        config.get("max_samples"),
    )


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_seed(run_config: Dict[str, Any]) -> int | None:
    seed = run_config.get("seed")
    try:
        return int(seed)
    except Exception:
        return None


def _extract_gold_primary(report: Dict[str, Any]) -> Dict[str, float]:
    metrics_by_target = report.get("metrics_by_target", {})
    gold_primary = metrics_by_target.get("gold", {}).get("metrics_primary", {})
    return {
        "comprehensiveness": _safe_float(gold_primary.get("comprehensiveness")),
        "sufficiency": _safe_float(gold_primary.get("sufficiency")),
        "log_odds": _safe_float(gold_primary.get("log_odds")),
        "aopc_sufficiency": _safe_float(gold_primary.get("aopc_sufficiency")),
        "aopc_comprehensiveness": _safe_float(gold_primary.get("aopc_comprehensiveness")),
        "aopc": _safe_float(gold_primary.get("aopc")),
        "deletion_auc": _safe_float(gold_primary.get("deletion_auc")),
        "insertion_auc": _safe_float(gold_primary.get("insertion_auc")),
    }


def _extract_pred_primary(report: Dict[str, Any]) -> Dict[str, float]:
    metrics_by_target = report.get("metrics_by_target", {})
    pred_primary = metrics_by_target.get("predicted", {}).get("metrics_primary", {})
    return {
        "comprehensiveness": _safe_float(pred_primary.get("comprehensiveness")),
        "sufficiency": _safe_float(pred_primary.get("sufficiency")),
        "log_odds": _safe_float(pred_primary.get("log_odds")),
        "aopc_sufficiency": _safe_float(pred_primary.get("aopc_sufficiency")),
        "aopc_comprehensiveness": _safe_float(pred_primary.get("aopc_comprehensiveness")),
        "aopc": _safe_float(pred_primary.get("aopc")),
        "deletion_auc": _safe_float(pred_primary.get("deletion_auc")),
        "insertion_auc": _safe_float(pred_primary.get("insertion_auc")),
    }


def collect_gate_b_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for config_path in sorted(results_root.glob("**/run_config.json")):
        run_dir = config_path.parent
        run_config = _read_json(config_path)
        if run_config is None:
            continue

        report_path = run_dir / "eval_report.json"
        report = _read_json(report_path)
        if report is None:
            continue

        method_from_cfg = str(run_config.get("explain_method", "")).strip().lower()
        method_from_report = str(report.get("report_method", "")).strip().lower()
        method = method_from_report or method_from_cfg or "unknown"

        runs.append(
            {
                "seed": _safe_seed(run_config),
                "method": method,
                "report_path": str(report_path),
                "config": _group_config(run_config),
                "report": report,
            }
        )

    return runs


def aggregate_gate_b_runs(
    runs: Sequence[Dict[str, Any]],
    min_runs: int = 1,
    primary_method: str = "ours",
    reference_method: str = "random",
) -> Dict[str, Any]:
    primary = str(primary_method).strip().lower()
    reference = str(reference_method).strip().lower()

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        key = _group_key(run["config"])
        groups.setdefault(key, []).append(run)

    out_groups: List[Dict[str, Any]] = []
    paired_total = 0

    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        config = dict(items[0]["config"])
        method_counter = Counter(str(item.get("method", "unknown")) for item in items)

        by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for run in items:
            seed = run.get("seed")
            if seed is None:
                continue
            by_seed.setdefault(int(seed), {})[str(run.get("method", "unknown"))] = run

        paired: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for seed in sorted(by_seed.keys()):
            run_primary = by_seed[seed].get(primary)
            run_reference = by_seed[seed].get(reference)
            if run_primary is None or run_reference is None:
                continue
            paired.append((seed, run_primary, run_reference))

        if len(paired) < int(min_runs):
            continue

        seeds = [seed for seed, _, _ in paired]
        paired_total += len(paired)

        gold_comp: List[float] = []
        gold_suff: List[float] = []
        gold_log_odds: List[float] = []
        gold_aopc_suff: List[float] = []
        gold_aopc_comp: List[float] = []
        gold_aopc: List[float] = []
        gold_deletion_auc: List[float] = []
        gold_insertion_auc: List[float] = []

        ref_gold_comp: List[float] = []
        ref_gold_suff: List[float] = []

        gold_comp_adv: List[float] = []
        gold_suff_adv: List[float] = []

        pred_comp: List[float] = []
        pred_suff: List[float] = []
        pred_log_odds: List[float] = []
        pred_aopc_suff: List[float] = []
        pred_aopc_comp: List[float] = []
        pred_aopc: List[float] = []
        pred_deletion_auc: List[float] = []
        pred_insertion_auc: List[float] = []

        runtime_primary: List[float] = []
        runtime_reference: List[float] = []
        predict_calls_primary: List[float] = []
        predict_calls_reference: List[float] = []

        pass_runs = 0
        reports: List[Dict[str, Any]] = []

        for seed, run_primary, run_reference in paired:
            primary_report = run_primary["report"]
            reference_report = run_reference["report"]

            primary_gold = _extract_gold_primary(primary_report)
            reference_gold = _extract_gold_primary(reference_report)
            primary_pred = _extract_pred_primary(primary_report)
            secondary_primary = primary_report.get("metrics_secondary", {})
            secondary_reference = reference_report.get("metrics_secondary", {})

            comp_adv = primary_gold["comprehensiveness"] - reference_gold["comprehensiveness"]
            suff_adv = reference_gold["sufficiency"] - primary_gold["sufficiency"]
            if comp_adv > 0.0 and suff_adv > 0.0:
                pass_runs += 1

            gold_comp.append(primary_gold["comprehensiveness"])
            gold_suff.append(primary_gold["sufficiency"])
            gold_log_odds.append(primary_gold["log_odds"])
            gold_aopc_suff.append(primary_gold["aopc_sufficiency"])
            gold_aopc_comp.append(primary_gold["aopc_comprehensiveness"])
            gold_aopc.append(primary_gold["aopc"])
            gold_deletion_auc.append(primary_gold["deletion_auc"])
            gold_insertion_auc.append(primary_gold["insertion_auc"])

            ref_gold_comp.append(reference_gold["comprehensiveness"])
            ref_gold_suff.append(reference_gold["sufficiency"])

            gold_comp_adv.append(comp_adv)
            gold_suff_adv.append(suff_adv)

            pred_comp.append(primary_pred["comprehensiveness"])
            pred_suff.append(primary_pred["sufficiency"])
            pred_log_odds.append(primary_pred["log_odds"])
            pred_aopc_suff.append(primary_pred["aopc_sufficiency"])
            pred_aopc_comp.append(primary_pred["aopc_comprehensiveness"])
            pred_aopc.append(primary_pred["aopc"])
            pred_deletion_auc.append(primary_pred["deletion_auc"])
            pred_insertion_auc.append(primary_pred["insertion_auc"])

            runtime_primary.append(_safe_float(secondary_primary.get("runtime_seconds")))
            runtime_reference.append(_safe_float(secondary_reference.get("runtime_seconds")))
            predict_calls_primary.append(
                _safe_float(secondary_primary.get("forward_counters_delta", {}).get("predict_calls", 0))
            )
            predict_calls_reference.append(
                _safe_float(secondary_reference.get("forward_counters_delta", {}).get("predict_calls", 0))
            )

            reports.append(
                {
                    "seed": int(seed),
                    "primary_report_path": str(run_primary.get("report_path")),
                    "reference_report_path": str(run_reference.get("report_path")),
                }
            )

        group_id = (
            f"{config.get('dataset')}/{config.get('split')}"
            f"|model={Path(str(config.get('model_path', ''))).name}"
            f"|chunk={config.get('chunker')}"
            f"|search={config.get('search')}"
            f"|k={config.get('k')}"
            f"|lam={config.get('lambdas')}"
            f"|eval={config.get('eval_granularity')}"
        )

        out_groups.append(
            {
                "group_id": group_id,
                "config": config,
                "methods": {
                    "primary": primary,
                    "reference": reference,
                    "available_run_counts": dict(method_counter),
                },
                "n_runs": len(paired),
                "seeds": seeds,
                "gate_b_checks": {
                    "comp_beats_reference_all": bool(min(gold_comp_adv) > 0.0),
                    "suff_beats_reference_all": bool(min(gold_suff_adv) > 0.0),
                    "run_pass_rate": float(pass_runs / len(paired)),
                },
                "gold_metrics": {
                    "log_odds": _stats(gold_log_odds),
                    "comprehensiveness": _stats(gold_comp),
                    "sufficiency": _stats(gold_suff),
                    "aopc_sufficiency": _stats(gold_aopc_suff),
                    "aopc_comprehensiveness": _stats(gold_aopc_comp),
                    "aopc": _stats(gold_aopc),
                    "deletion_auc": _stats(gold_deletion_auc),
                    "insertion_auc": _stats(gold_insertion_auc),
                    "comp_adv_vs_reference": _stats(gold_comp_adv),
                    "suff_adv_vs_reference": _stats(gold_suff_adv),
                },
                "reference_gold_metrics": {
                    "comprehensiveness": _stats(ref_gold_comp),
                    "sufficiency": _stats(ref_gold_suff),
                },
                "predicted_metrics": {
                    "log_odds": _stats(pred_log_odds),
                    "comprehensiveness": _stats(pred_comp),
                    "sufficiency": _stats(pred_suff),
                    "aopc_sufficiency": _stats(pred_aopc_suff),
                    "aopc_comprehensiveness": _stats(pred_aopc_comp),
                    "aopc": _stats(pred_aopc),
                    "deletion_auc": _stats(pred_deletion_auc),
                    "insertion_auc": _stats(pred_insertion_auc),
                },
                "runtime": {
                    "primary_seconds": _stats(runtime_primary),
                    "reference_seconds": _stats(runtime_reference),
                    "primary_predict_calls": _stats(predict_calls_primary),
                    "reference_predict_calls": _stats(predict_calls_reference),
                },
                "reports": reports,
            }
        )

    return {
        "group_count": len(out_groups),
        "run_count": len(runs),
        "paired_run_count": paired_total,
        "min_runs": int(min_runs),
        "primary_method": primary,
        "reference_method": reference,
        "groups": out_groups,
    }


def write_gate_b_aggregate_json(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_gate_b_summary_csv(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in payload.get("groups", []):
        rows.append(
            {
                "group_id": group.get("group_id"),
                "n_runs": group.get("n_runs"),
                "seeds": ",".join(str(x) for x in group.get("seeds", [])),
                "primary_method": group.get("methods", {}).get("primary", ""),
                "reference_method": group.get("methods", {}).get("reference", ""),
                "gold_comp_mean": _safe_float(group.get("gold_metrics", {}).get("comprehensiveness", {}).get("mean")),
                "gold_comp_std": _safe_float(group.get("gold_metrics", {}).get("comprehensiveness", {}).get("std")),
                "gold_suff_mean": _safe_float(group.get("gold_metrics", {}).get("sufficiency", {}).get("mean")),
                "gold_suff_std": _safe_float(group.get("gold_metrics", {}).get("sufficiency", {}).get("std")),
                "ref_comp_mean": _safe_float(group.get("reference_gold_metrics", {}).get("comprehensiveness", {}).get("mean")),
                "ref_suff_mean": _safe_float(group.get("reference_gold_metrics", {}).get("sufficiency", {}).get("mean")),
                "comp_adv_mean": _safe_float(group.get("gold_metrics", {}).get("comp_adv_vs_reference", {}).get("mean")),
                "suff_adv_mean": _safe_float(group.get("gold_metrics", {}).get("suff_adv_vs_reference", {}).get("mean")),
                "run_pass_rate": _safe_float(group.get("gate_b_checks", {}).get("run_pass_rate")),
            }
        )

    fieldnames = [
        "group_id",
        "n_runs",
        "seeds",
        "primary_method",
        "reference_method",
        "gold_comp_mean",
        "gold_comp_std",
        "gold_suff_mean",
        "gold_suff_std",
        "ref_comp_mean",
        "ref_suff_mean",
        "comp_adv_mean",
        "suff_adv_mean",
        "run_pass_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
