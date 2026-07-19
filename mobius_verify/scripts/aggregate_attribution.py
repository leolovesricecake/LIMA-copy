from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.utils import atomic_write_json, atomic_write_text, read_json, resolve_project_path
from mobius_verify.src.reconstruction_metrics import auc_logx
from mobius_verify.src.statistics import bootstrap_ci


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Sparse Mobius attribution results.")
    parser.add_argument("--results-dir", type=str, required=True)
    return parser


def _mean(values: Iterable[object]) -> float | None:
    clean = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return float(sum(clean) / len(clean)) if clean else None


def _threshold_query(
    curve: list[Dict[str, Any]],
    *,
    metric: str,
    threshold: float,
    query_key: str,
) -> float | None:
    eligible = [
        row
        for row in curve
        if isinstance(row.get(metric), (int, float))
        and float(row[metric]) >= float(threshold)
        and isinstance(row.get(query_key), (int, float))
    ]
    return min((float(row[query_key]) for row in eligible), default=None)


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if not str(key).startswith("_")})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _result_rows(results_dir: Path) -> list[Dict[str, Any]]:
    rows = []
    for path in sorted((results_dir / "protocols").glob("*/*/*/*/*/budget_*/seed_*/result.json")):
        payload = read_json(path)
        row: Dict[str, Any] = {
            "status": payload.get("status"),
            "protocol": payload.get("protocol"),
            "value_function": payload.get("value_function"),
            "method": payload.get("method"),
            "task": payload.get("task"),
            "sample_id": payload.get("sample_id"),
            "n_features": payload.get("n_features"),
            "budget": payload.get("budget"),
            "seed": payload.get("seed"),
            "failure_type": payload.get("failure_type"),
            "failure_reason": payload.get("failure_reason"),
        }
        if payload.get("status") == "ok":
            for distribution, metrics in payload.get("surrogate_metrics", {}).items():
                row[f"{distribution}_r2"] = metrics.get("r2")
                row[f"{distribution}_nrmse"] = metrics.get("normalized_rmse")
            attribution = payload.get("attribution_metrics", {})
            row["morf_auc"] = attribution.get("morf_comprehensiveness_auc")
            row["sufficiency_auc"] = attribution.get("sufficiency_gap_auc")
            random_control = payload.get("random_ranking_control", {})
            row["random_morf_auc"] = random_control.get("morf_comprehensiveness_auc")
            verification = payload.get("interaction_verification") or {}
            top_verification = list(verification.get("top") or [])
            row["targeted_sign_accuracy"] = verification.get("sign_accuracy")
            row["targeted_top_abs"] = verification.get("top_mean_abs_true")
            row["targeted_random_abs"] = verification.get("random_mean_abs_true")
            row["targeted_magnitude_enrichment"] = (
                float(row["targeted_top_abs"]) - float(row["targeted_random_abs"])
                if isinstance(row["targeted_top_abs"], (int, float))
                and isinstance(row["targeted_random_abs"], (int, float))
                else None
            )
            row["targeted_normalized_error_mean"] = _mean(
                item.get("normalized_error") for item in top_verification
            )
            row["targeted_absolute_error_mean"] = _mean(
                item.get("absolute_error") for item in top_verification
            )
            ledger = payload.get("query_ledger", {})
            for key in (
                "attribution_budget_used",
                "logical_unique_queries",
                "physical_forwards_caused",
                "global_cache_hits",
                "interaction_verification_queries",
                "evaluation_only_queries",
            ):
                row[key] = ledger.get(key)
            row["support_size"] = payload.get("model", {}).get("coefficient_count")
            timing = payload.get("timing", {})
            row["fit_elapsed_seconds"] = timing.get("fit_elapsed_seconds")
            row["evaluation_elapsed_seconds"] = timing.get("evaluation_elapsed_seconds")
            row["competitor_switch_rate"] = payload.get("competitor_switch_rate")
            row["train_masks_digest"] = payload.get("train_masks_digest")
            row["evaluation_masks_digest"] = payload.get("evaluation_masks_digest")
            n_features = max(2, int(payload.get("n_features", 2)))
            row["normalized_query_budget"] = float(
                int(payload.get("budget", 0)) / (n_features * math.log2(n_features))
            )
            row["_positive_ranking"] = payload.get("rankings", {}).get("positive", [])
            model_payload = payload.get("model", {})
            row["_hyperedges"] = list(
                model_payload.get("hyperedges") or model_payload.get("presence_mobius") or []
            )
            row["_verification"] = verification
        rows.append(row)
    return rows


def _query_curve_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("protocol"),
            row.get("value_function"),
            row.get("method"),
            row.get("task"),
            row.get("sample_id"),
            row.get("seed"),
        )
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        curve = sorted(group, key=lambda row: float(row["normalized_query_budget"]))
        uniform_curve = [row for row in curve if isinstance(row.get("uniform_r2"), (int, float))]
        near_full_curve = [row for row in curve if isinstance(row.get("near_full_r2"), (int, float))]
        output.append(
            {
                "protocol": key[0],
                "value_function": key[1],
                "method": key[2],
                "task": key[3],
                "sample_id": key[4],
                "seed": key[5],
                "uniform_r2_auc": auc_logx(uniform_curve, "normalized_query_budget", "uniform_r2"),
                "near_full_r2_auc": auc_logx(near_full_curve, "normalized_query_budget", "near_full_r2"),
                "logical_queries_at_uniform_r2_08": _threshold_query(
                    curve,
                    metric="uniform_r2",
                    threshold=0.8,
                    query_key="attribution_budget_used",
                ),
                "logical_queries_at_uniform_r2_09": _threshold_query(
                    curve,
                    metric="uniform_r2",
                    threshold=0.9,
                    query_key="attribution_budget_used",
                ),
                "logical_queries_at_near_full_r2_08": _threshold_query(
                    curve,
                    metric="near_full_r2",
                    threshold=0.8,
                    query_key="attribution_budget_used",
                ),
                "logical_queries_at_near_full_r2_09": _threshold_query(
                    curve,
                    metric="near_full_r2",
                    threshold=0.9,
                    query_key="attribution_budget_used",
                ),
                "budget_point_count": len(curve),
            }
        )
    return output


def _stability_rows(rows: list[Dict[str, Any]], *, top_k: int = 5) -> list[Dict[str, Any]]:
    groups: Dict[tuple, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ranking = row.get("_positive_ranking")
        if row.get("status") != "ok" or not isinstance(ranking, list):
            continue
        key = (
            row.get("protocol"),
            row.get("value_function"),
            row.get("method"),
            row.get("task"),
            row.get("sample_id"),
            row.get("budget"),
        )
        edges: Dict[int, float] = {}
        for item in row.get("_hyperedges") or []:
            term = item.get("term")
            coefficient = item.get("coefficient", item.get("refit_coefficient"))
            if not isinstance(term, (int, float)) or not isinstance(coefficient, (int, float)):
                continue
            value = float(coefficient)
            selection = item.get("selection_coefficient")
            if abs(value) <= 1e-12 and isinstance(selection, (int, float)):
                value = float(selection)
            if abs(value) > 1e-12:
                edges[int(term)] = value
        top_edges = [
            term
            for term, _ in sorted(edges.items(), key=lambda item: (-abs(item[1]), item[0]))[
                : int(top_k)
            ]
        ]
        groups[key].append(
            {
                "nodes": [int(value) for value in ranking[: int(top_k)]],
                "top_edges": top_edges,
                "edges": edges,
            }
        )
    output = []
    for key, runs in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        node_jaccards = []
        edge_jaccards = []
        for left_idx in range(len(runs)):
            for right_idx in range(left_idx + 1, len(runs)):
                for field, target in (("nodes", node_jaccards), ("top_edges", edge_jaccards)):
                    left = set(runs[left_idx][field])
                    right = set(runs[right_idx][field])
                    union = left | right
                    target.append(float(len(left & right) / len(union)) if union else 1.0)
        union_terms = sorted({term for run in runs for term in run["edges"]})
        frequencies = []
        sign_consistencies = []
        for term in union_terms:
            coefficients = [float(run["edges"][term]) for run in runs if term in run["edges"]]
            frequencies.append(float(len(coefficients) / len(runs)))
            if len(coefficients) >= 2:
                positive = sum(value > 0 for value in coefficients)
                negative = sum(value < 0 for value in coefficients)
                sign_consistencies.append(float(max(positive, negative) / len(coefficients)))
        output.append(
            {
                "protocol": key[0],
                "value_function": key[1],
                "method": key[2],
                "task": key[3],
                "sample_id": key[4],
                "budget": key[5],
                "seed_count": len(runs),
                "top_k": int(top_k),
                "node_mean_pairwise_jaccard": _mean(node_jaccards),
                "hyperedge_mean_pairwise_jaccard": _mean(edge_jaccards),
                "mean_support_selection_frequency": _mean(frequencies),
                "mean_shared_support_sign_consistency": _mean(sign_consistencies),
                "support_union_size": len(union_terms),
            }
        )
    return output


def _interaction_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        verification = row.get("_verification") or {}
        common = {
            "protocol": row.get("protocol"),
            "value_function": row.get("value_function"),
            "method": row.get("method"),
            "task": row.get("task"),
            "sample_id": row.get("sample_id"),
            "budget": row.get("budget"),
            "seed": row.get("seed"),
            "orientation": verification.get("orientation"),
        }
        for control, items in (
            ("top", verification.get("top") or []),
            ("random", verification.get("random_control") or []),
        ):
            for item in items:
                output.append(
                    {
                        **common,
                        "control": control,
                        "term": item.get("term"),
                        "players": json.dumps(item.get("players", [])),
                        "degree": item.get("degree"),
                        "estimated_coefficient": item.get("estimated_coefficient"),
                        "true_coefficient": item.get("true_coefficient"),
                        "absolute_error": item.get("absolute_error"),
                        "normalized_error": item.get("normalized_error"),
                        "sign_match": item.get("sign_match"),
                    }
                )
    return output


def _paired_query_rows(query_rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in query_rows:
        key = (
            row.get("protocol"),
            row.get("value_function"),
            row.get("task"),
            row.get("sample_id"),
            row.get("seed"),
        )
        groups[key][str(row.get("method"))] = row
    output = []
    for key, methods in groups.items():
        deletion = methods.get("deletion_mobius")
        if deletion is None:
            continue
        comparators = ["additive_lasso", "fourier"] if key[0] == "controlled" else ["proxyspex"]
        for comparator in comparators:
            other = methods.get(comparator)
            if other is None:
                continue
            output_row = {
                "protocol": key[0],
                "value_function": key[1],
                "task": key[2],
                "sample_id": key[3],
                "seed": key[4],
                "left_method": "deletion_mobius",
                "right_method": comparator,
            }
            for metric in ("uniform_r2_auc", "near_full_r2_auc"):
                left_value = deletion.get(metric)
                right_value = other.get(metric)
                output_row[f"delta_{metric}"] = (
                    float(left_value) - float(right_value)
                    if isinstance(left_value, (int, float))
                    and isinstance(right_value, (int, float))
                    else None
                )
            output.append(output_row)
    return output


def _sample_level_ci(
    rows: list[Dict[str, Any]],
    *,
    metric: str,
    filters: Mapping[str, object],
) -> Dict[str, object]:
    groups: Dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            groups[(row.get("task"), row.get("sample_id"))].append(float(value))
    per_sample = [float(sum(values) / len(values)) for values in groups.values() if values]
    return bootstrap_ci(per_sample) if per_sample else {"n": 0}


def _paired_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("protocol"),
            row.get("value_function"),
            row.get("task"),
            row.get("sample_id"),
            row.get("budget"),
            row.get("seed"),
        )
        groups[key][str(row.get("method"))] = row
    output = []
    for key, methods in groups.items():
        left = methods.get("deletion_mobius")
        if left is None:
            continue
        comparisons = ["additive_lasso", "fourier"] if key[0] == "controlled" else ["proxyspex"]
        for comparator in comparisons:
            right = methods.get(comparator)
            if right is None:
                continue
            row = {
                "protocol": key[0],
                "value_function": key[1],
                "task": key[2],
                "sample_id": key[3],
                "budget": key[4],
                "seed": key[5],
                "left_method": "deletion_mobius",
                "right_method": comparator,
            }
            for metric in (
                "uniform_r2",
                "near_full_r2",
                "morf_auc",
                "targeted_sign_accuracy",
                "targeted_magnitude_enrichment",
                "targeted_normalized_error_mean",
                "support_size",
            ):
                left_value = left.get(metric)
                right_value = right.get(metric)
                row[f"delta_{metric}"] = (
                    float(left_value) - float(right_value)
                    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float))
                    else None
                )
            output.append(row)
    return output


def aggregate(results_dir: Path) -> Dict[str, Any]:
    rows = _result_rows(results_dir)
    aggregate_dir = results_dir / "aggregate"
    _write_csv(aggregate_dir / "sample_metrics.csv", rows)
    query_rows = _query_curve_rows(rows)
    stability_rows = _stability_rows(rows)
    interaction_rows = _interaction_rows(rows)
    paired_rows = _paired_rows(rows)
    paired_query_rows = _paired_query_rows(query_rows)
    _write_csv(aggregate_dir / "query_metrics.csv", query_rows)
    _write_csv(aggregate_dir / "stability_metrics.csv", stability_rows)
    _write_csv(aggregate_dir / "interaction_metrics.csv", interaction_rows)
    _write_csv(aggregate_dir / "paired_differences.csv", paired_rows)
    _write_csv(aggregate_dir / "paired_query_auc.csv", paired_query_rows)
    groups: Dict[tuple[str, str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(str(row["protocol"]), str(row["value_function"]), str(row["method"]))].append(row)
    summaries = []
    for (protocol, value_type, method), group in sorted(groups.items()):
        summaries.append(
            {
                "protocol": protocol,
                "value_function": value_type,
                "method": method,
                "n": len(group),
                "uniform_r2_mean": _mean(row.get("uniform_r2") for row in group),
                "near_full_r2_mean": _mean(row.get("near_full_r2") for row in group),
                "fixed_cardinality_r2_mean": _mean(row.get("fixed_cardinality_r2") for row in group),
                "morf_auc_mean": _mean(row.get("morf_auc") for row in group),
                "random_morf_auc_mean": _mean(row.get("random_morf_auc") for row in group),
                "targeted_sign_accuracy_mean": _mean(row.get("targeted_sign_accuracy") for row in group),
                "targeted_normalized_error_mean": _mean(
                    row.get("targeted_normalized_error_mean") for row in group
                ),
                "targeted_magnitude_enrichment_mean": _mean(
                    row.get("targeted_magnitude_enrichment") for row in group
                ),
                "attribution_budget_mean": _mean(row.get("attribution_budget_used") for row in group),
                "physical_values_scored_mean": _mean(
                    row.get("physical_forwards_caused") for row in group
                ),
                "fit_elapsed_seconds_mean": _mean(row.get("fit_elapsed_seconds") for row in group),
                "support_size_mean": _mean(row.get("support_size") for row in group),
            }
        )
    failures = [row for row in rows if row.get("status") != "ok"]
    primary_filters = {"value_function": "predicted_class_margin"}
    controlled_primary = _sample_level_ci(
        paired_query_rows,
        metric="delta_near_full_r2_auc",
        filters={
            **primary_filters,
            "protocol": "controlled",
            "right_method": "additive_lasso",
        },
    )
    native_primary = _sample_level_ci(
        paired_query_rows,
        metric="delta_near_full_r2_auc",
        filters={
            **primary_filters,
            "protocol": "native",
            "right_method": "proxyspex",
        },
    )
    targeted_enrichment = _sample_level_ci(
        rows,
        metric="targeted_magnitude_enrichment",
        filters={
            **primary_filters,
            "protocol": "controlled",
            "method": "deletion_mobius",
            "status": "ok",
        },
    )
    hyperedge_stability = _sample_level_ci(
        stability_rows,
        metric="hyperedge_mean_pairwise_jaccard",
        filters={
            **primary_filters,
            "protocol": "controlled",
            "method": "deletion_mobius",
        },
    )
    config_path = results_dir / "run_config.json"
    run_config = read_json(config_path) if config_path.exists() else {}
    minimum_samples = int(run_config.get("decision_min_samples", 10))
    noninferiority_margin = float(run_config.get("proxyspex_noninferiority_margin", -0.02))
    stability_floor = float(run_config.get("hyperedge_stability_floor", 0.1))

    def enough(summary: Mapping[str, object]) -> bool:
        return int(summary.get("n", 0) or 0) >= minimum_samples

    decision_ready = all(
        enough(summary)
        for summary in (
            controlled_primary,
            native_primary,
            targeted_enrichment,
            hyperedge_stability,
        )
    )
    gates = {
        "controlled_beats_additive": bool(
            enough(controlled_primary) and float(controlled_primary.get("low", -math.inf)) > 0
        ),
        "targeted_beats_random": bool(
            enough(targeted_enrichment) and float(targeted_enrichment.get("low", -math.inf)) > 0
        ),
        "hyperedge_support_not_unstable": bool(
            enough(hyperedge_stability)
            and float(hyperedge_stability.get("mean", -math.inf)) >= stability_floor
        ),
        "native_noninferior_to_proxyspex": bool(
            enough(native_primary)
            and float(native_primary.get("low", -math.inf)) >= noninferiority_margin
        ),
    }
    clear_negative = bool(
        decision_ready
        and (
            float(controlled_primary.get("high", math.inf)) <= 0
            or float(native_primary.get("high", math.inf)) < noninferiority_margin
        )
    )
    main_tasks = {
        str(row.get("task"))
        for row in rows
        if row.get("status") == "ok"
        and row.get("value_function") == "predicted_class_margin"
    }
    if not decision_ready:
        status = "Inconclusive"
        reason = "主实验的 sample-level paired evidence 尚未达到预设最小样本数。"
    elif all(gates.values()):
        status = "Supported" if len(main_tasks) >= 2 else "Partially supported"
        reason = "所有 degree-2 开发门控均通过。"
    elif clear_negative:
        status = "Not supported"
        reason = "主 faithfulness 或 native non-inferiority 出现明确负向证据。"
    else:
        status = "Inconclusive"
        reason = "样本数充分，但预注册门控给出混合证据。"
    hypothesis = {
        "status": status,
        "reason": reason,
        "successful_runs": int(len(rows) - len(failures)),
        "failed_runs": int(len(failures)),
        "summaries": summaries,
        "decision": {
            "value_function": "predicted_class_margin",
            "primary_metric": "near_full_r2_auc",
            "minimum_samples": minimum_samples,
            "proxyspex_noninferiority_margin": noninferiority_margin,
            "hyperedge_stability_floor": stability_floor,
            "controlled_vs_additive": controlled_primary,
            "native_vs_proxyspex": native_primary,
            "targeted_magnitude_enrichment": targeted_enrichment,
            "hyperedge_stability": hyperedge_stability,
            "gates": gates,
        },
    }
    atomic_write_json(aggregate_dir / "hypothesis_results.json", hypothesis)
    _write_csv(aggregate_dir / "method_summary.csv", summaries)
    lines = [
        "# Sparse Mobius Attribution 实验报告",
        "",
        f"- 成功运行：{hypothesis['successful_runs']}",
        f"- 失败运行：{hypothesis['failed_runs']}",
        f"- 当前结论：{status}",
        f"- 原因：{reason}",
        "- 主 value function：predicted-class margin",
        "",
        "## Degree-2 门控",
        "",
        f"- Controlled deletion-Möbius vs additive：{controlled_primary}",
        f"- Native deletion-Möbius vs ProxySPEX：{native_primary}",
        f"- Targeted magnitude enrichment：{targeted_enrichment}",
        f"- Hyperedge stability：{hyperedge_stability}",
        f"- Gates：{gates}",
        "",
        "## 方法汇总",
        "",
    ]
    for summary in summaries:
        lines.append(
            "- "
            f"{summary['protocol']} / {summary['value_function']} / {summary['method']}: "
            f"n={summary['n']}, uniform R2={summary['uniform_r2_mean']}, "
            f"near-full R2={summary['near_full_r2_mean']}, "
            f"targeted sign={summary['targeted_sign_accuracy_mean']}"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "Controlled 结果用于比较相同观测下的坐标与估计器；Native 结果用于比较完整方法。",
            "predicted-class margin 是主结果，raw target score 仅作敏感性分析。",
            "Smoke 或单预算结果没有 query-curve AUC，因此会按规则保持 Inconclusive。",
            "",
        ]
    )
    atomic_write_text(aggregate_dir / "hypothesis_report.md", "\n".join(lines))
    return hypothesis


def main() -> None:
    args = build_parser().parse_args()
    results_dir = resolve_project_path(
        args.results_dir,
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "attribution_mvp",
    )
    result = aggregate(results_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
