from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


GATE_B_REPORT_MODES = ("auto", "legacy", "split")


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
        "eval_random_trials": run_config.get("eval_random_trials"),
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
        config.get("eval_random_trials"),
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


def _extract_gold_comp_suff(report: Dict[str, Any]) -> tuple[float, float]:
    metrics_by_target = report.get("metrics_by_target", {})
    gold_primary = metrics_by_target.get("gold", {}).get("metrics_primary", {})
    return (
        _safe_float(gold_primary.get("comprehensiveness")),
        _safe_float(gold_primary.get("sufficiency")),
    )


def _collect_split_run(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any] | None:
    ours_path = run_dir / "eval_report.ours.json"
    random_path = run_dir / "eval_report.random.json"
    ours_report = _read_json(ours_path)
    random_report = _read_json(random_path)
    if ours_report is None or random_report is None:
        return None

    ours_comp, ours_suff = _extract_gold_comp_suff(ours_report)
    rand_comp, rand_suff = _extract_gold_comp_suff(random_report)
    comp_adv = ours_comp - rand_comp
    suff_adv = rand_suff - ours_suff

    return {
        "seed": _safe_seed(run_config),
        "report_path": str(ours_path),
        "report_paths": [str(ours_path), str(random_path)],
        "report_kind": "split",
        "config": _group_config(run_config),
        "report": ours_report,
        "random_reference_report": random_report,
        "gold_comp_adv_vs_random": comp_adv,
        "gold_suff_adv_vs_random": suff_adv,
        "gate_b_pass_run": bool(comp_adv > 0.0 and suff_adv > 0.0),
    }


def _collect_legacy_run(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any] | None:
    report_path = run_dir / "eval_report.json"
    report = _read_json(report_path)
    if report is None:
        return None

    metrics_by_target = report.get("metrics_by_target", {})
    gold = metrics_by_target.get("gold", {})
    gold_primary = gold.get("metrics_primary", {})
    gold_random = gold.get("baselines", {}).get("random", {})

    ours_comp = _safe_float(gold_primary.get("comprehensiveness"))
    ours_suff = _safe_float(gold_primary.get("sufficiency"))
    rand_comp = _safe_float(gold_random.get("comprehensiveness"))
    rand_suff = _safe_float(gold_random.get("sufficiency"))
    comp_adv = ours_comp - rand_comp
    suff_adv = rand_suff - ours_suff

    return {
        "seed": _safe_seed(run_config),
        "report_path": str(report_path),
        "report_paths": [str(report_path)],
        "report_kind": "legacy",
        "config": _group_config(run_config),
        "report": report,
        "random_reference_report": None,
        "gold_comp_adv_vs_random": comp_adv,
        "gold_suff_adv_vs_random": suff_adv,
        "gate_b_pass_run": bool(comp_adv > 0.0 and suff_adv > 0.0),
    }


def collect_gate_b_runs(results_root: Path, report_mode: str = "auto") -> List[Dict[str, Any]]:
    mode = str(report_mode).lower()
    if mode not in GATE_B_REPORT_MODES:
        raise ValueError(f"Unsupported report_mode: {report_mode}. Expected one of {GATE_B_REPORT_MODES}.")

    runs: List[Dict[str, Any]] = []
    for config_path in sorted(results_root.glob("**/run_config.json")):
        run_dir = config_path.parent
        run_config = _read_json(config_path)
        if run_config is None:
            continue

        if mode in ("auto", "split"):
            split_run = _collect_split_run(run_dir=run_dir, run_config=run_config)
            if split_run is not None:
                runs.append(split_run)
                if mode == "auto":
                    continue

        if mode in ("auto", "legacy"):
            legacy_run = _collect_legacy_run(run_dir=run_dir, run_config=run_config)
            if legacy_run is not None:
                runs.append(legacy_run)

    return runs


def aggregate_gate_b_runs(runs: Sequence[Dict[str, Any]], min_runs: int = 1) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        key = _group_key(run["config"])
        groups.setdefault(key, []).append(run)

    out_groups: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        if len(items) < int(min_runs):
            continue
        config = dict(items[0]["config"])
        seeds = sorted([run["seed"] for run in items if run["seed"] is not None])

        gold_comp = []
        gold_suff = []
        gold_log_odds = []
        gold_aopc_suff = []
        gold_aopc_comp = []
        gold_aopc = []
        gold_deletion_auc = []
        gold_insertion_auc = []
        gold_diag = []
        gold_comp_adv = []
        gold_suff_adv = []
        pred_comp = []
        pred_suff = []
        pred_log_odds = []
        pred_aopc_suff = []
        pred_aopc_comp = []
        pred_aopc = []
        pred_deletion_auc = []
        pred_insertion_auc = []
        runtimes = []
        predict_calls = []
        pass_runs = 0
        kind_counter = Counter()

        for run in items:
            report = run["report"]
            kind_counter[str(run.get("report_kind", "unknown"))] += 1

            metrics_by_target = report.get("metrics_by_target", {})
            gold = metrics_by_target.get("gold", {})
            predicted = metrics_by_target.get("predicted", {})
            gold_primary = gold.get("metrics_primary", {})
            pred_primary = predicted.get("metrics_primary", {})
            secondary = report.get("metrics_secondary", {})

            gold_comp.append(_safe_float(gold_primary.get("comprehensiveness")))
            gold_suff.append(_safe_float(gold_primary.get("sufficiency")))
            gold_log_odds.append(_safe_float(gold_primary.get("log_odds")))
            gold_aopc_suff.append(_safe_float(gold_primary.get("aopc_sufficiency")))
            gold_aopc_comp.append(_safe_float(gold_primary.get("aopc_comprehensiveness")))
            gold_aopc.append(_safe_float(gold_primary.get("aopc")))
            gold_deletion_auc.append(_safe_float(gold_primary.get("deletion_auc")))
            gold_insertion_auc.append(_safe_float(gold_primary.get("insertion_auc")))
            gold_diag.append(_safe_float(gold.get("diagnosticity_vs_random"), default=0.0))
            gold_comp_adv.append(_safe_float(run.get("gold_comp_adv_vs_random")))
            gold_suff_adv.append(_safe_float(run.get("gold_suff_adv_vs_random")))
            pred_comp.append(_safe_float(pred_primary.get("comprehensiveness")))
            pred_suff.append(_safe_float(pred_primary.get("sufficiency")))
            pred_log_odds.append(_safe_float(pred_primary.get("log_odds")))
            pred_aopc_suff.append(_safe_float(pred_primary.get("aopc_sufficiency")))
            pred_aopc_comp.append(_safe_float(pred_primary.get("aopc_comprehensiveness")))
            pred_aopc.append(_safe_float(pred_primary.get("aopc")))
            pred_deletion_auc.append(_safe_float(pred_primary.get("deletion_auc")))
            pred_insertion_auc.append(_safe_float(pred_primary.get("insertion_auc")))
            runtimes.append(_safe_float(secondary.get("runtime_seconds")))
            predict_calls.append(
                _safe_float(
                    secondary.get("forward_counters_delta", {}).get("predict_calls", 0),
                    default=0.0,
                )
            )
            if bool(run.get("gate_b_pass_run", False)):
                pass_runs += 1

        group_id = (
            f"{config.get('dataset')}/{config.get('split')}"
            f"|model={Path(str(config.get('model_path', ''))).name}"
            f"|chunk={config.get('chunker')}"
            f"|search={config.get('search')}"
            f"|k={config.get('k')}"
            f"|lam={config.get('lambdas')}"
        )

        out_groups.append(
            {
                "group_id": group_id,
                "config": config,
                "n_runs": len(items),
                "seeds": seeds,
                "report_kind_counts": dict(kind_counter),
                "gate_b_checks": {
                    "comp_beats_random_all": bool(min(gold_comp_adv) > 0.0),
                    "suff_beats_random_all": bool(min(gold_suff_adv) > 0.0),
                    "run_pass_rate": float(pass_runs / len(items)),
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
                    "diagnosticity_vs_random": _stats(gold_diag),
                    "comp_adv_vs_random": _stats(gold_comp_adv),
                    "suff_adv_vs_random": _stats(gold_suff_adv),
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
                    "seconds": _stats(runtimes),
                    "predict_calls": _stats(predict_calls),
                },
                "reports": [run.get("report_paths", [run["report_path"]]) for run in items],
            }
        )

    return {
        "group_count": len(out_groups),
        "run_count": len(runs),
        "min_runs": int(min_runs),
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
                "report_kinds": json.dumps(group.get("report_kind_counts", {}), ensure_ascii=False),
                "seeds": ",".join(str(x) for x in group.get("seeds", [])),
                "gold_comp_mean": _safe_float(group.get("gold_metrics", {}).get("comprehensiveness", {}).get("mean")),
                "gold_comp_std": _safe_float(group.get("gold_metrics", {}).get("comprehensiveness", {}).get("std")),
                "gold_suff_mean": _safe_float(group.get("gold_metrics", {}).get("sufficiency", {}).get("mean")),
                "gold_suff_std": _safe_float(group.get("gold_metrics", {}).get("sufficiency", {}).get("std")),
                "gold_log_odds_mean": _safe_float(group.get("gold_metrics", {}).get("log_odds", {}).get("mean")),
                "gold_aopc_suff_mean": _safe_float(group.get("gold_metrics", {}).get("aopc_sufficiency", {}).get("mean")),
                "gold_aopc_comp_mean": _safe_float(group.get("gold_metrics", {}).get("aopc_comprehensiveness", {}).get("mean")),
                "gold_aopc_mean": _safe_float(group.get("gold_metrics", {}).get("aopc", {}).get("mean")),
                "gold_deletion_auc_mean": _safe_float(group.get("gold_metrics", {}).get("deletion_auc", {}).get("mean")),
                "gold_insertion_auc_mean": _safe_float(group.get("gold_metrics", {}).get("insertion_auc", {}).get("mean")),
                "gold_diag_mean": _safe_float(group.get("gold_metrics", {}).get("diagnosticity_vs_random", {}).get("mean")),
                "comp_adv_mean": _safe_float(group.get("gold_metrics", {}).get("comp_adv_vs_random", {}).get("mean")),
                "suff_adv_mean": _safe_float(group.get("gold_metrics", {}).get("suff_adv_vs_random", {}).get("mean")),
                "run_pass_rate": _safe_float(group.get("gate_b_checks", {}).get("run_pass_rate")),
            }
        )

    fieldnames = [
        "group_id",
        "n_runs",
        "report_kinds",
        "seeds",
        "gold_comp_mean",
        "gold_comp_std",
        "gold_suff_mean",
        "gold_suff_std",
        "gold_log_odds_mean",
        "gold_aopc_suff_mean",
        "gold_aopc_comp_mean",
        "gold_aopc_mean",
        "gold_deletion_auc_mean",
        "gold_insertion_auc_mean",
        "gold_diag_mean",
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
