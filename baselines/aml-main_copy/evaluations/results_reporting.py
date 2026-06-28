from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config.config import ExpArgs, LossCoefficients
from config.constants import INPUT_TXT
from config.types_enums import EvalMetric
from evaluations.evaluations import evaluate_tokens_attributions
from evaluations.metrics.metrics_utils import MetricsFunctions
from utils.utils_functions import get_model_special_tokens


ALL_METRICS_LONG_RESULTS_FILE_NAME = "all_metrics_results_long.csv"
ALL_METRICS_WIDE_RESULTS_FILE_NAME = "all_metrics_results_wide.csv"
ALL_METRICS_SUMMARY_FILE_NAME = "all_metrics_summary.csv"
ALL_METRICS_REPORT_FILE_NAME = "all_metrics_report.json"
EVAL_REPORT_FILE_NAME = "eval_report.json"
TRAJECTORY_POINTS_CSV = "trajectory_points.csv"
TRAJECTORY_POINTS_JSONL = "trajectory_points.jsonl"
TRAJECTORY_SUMMARY_CSV = "trajectory_summary.csv"

PRIMARY_METRIC_NAME_MAP = {
    EvalMetric.SUFFICIENCY.value: "sufficiency",
    EvalMetric.COMPREHENSIVENESS.value: "comprehensiveness",
    EvalMetric.EVAL_LOG_ODDS.value: "log_odds",
    EvalMetric.AOPC.value: "aopc",
    EvalMetric.AOPC_SUFFICIENCY.value: "aopc_sufficiency",
    EvalMetric.AOPC_COMPREHENSIVENESS.value: "aopc_comprehensiveness",
}

SECONDARY_METRIC_NAME_MAP = {
    EvalMetric.AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY.value: "aopc_comprehensiveness_aopc_sufficiency",
    EvalMetric.COMPREHENSIVENESS_SUFFICIENCY.value: "comprehensiveness_sufficiency",
}


def get_supported_eval_metrics():
    return [metric.value for metric in EvalMetric]


def normalize_model_name(model_identifier: str) -> str:
    raw = str(model_identifier or "").strip()
    if not raw:
        return "unknown-model"
    candidate = Path(raw.rstrip("/")).name or raw
    candidate = candidate.removeprefix("model-")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-_.").lower()
    return normalized or "unknown-model"


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _coerce_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    return [value]


def _coerce_float_list(value) -> List[float]:
    result = []
    for item in _coerce_list(value):
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _coerce_int_list(value) -> List[int]:
    result = []
    for item in _coerce_list(value):
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _label_text_for_index(label_id: int) -> Optional[str]:
    labels_map = getattr(ExpArgs.task, "labels_str_int_maps", None) or {}
    label_names = list(labels_map.keys())
    if 0 <= int(label_id) < len(label_names):
        return str(label_names[int(label_id)])
    return None


def _compute_trajectory_payload(model, explained_tokenizer, ref_token_id, data):
    special_tokens = torch.tensor(get_model_special_tokens(ExpArgs.explained_model_backbone, explained_tokenizer))
    metric_functions = MetricsFunctions(model, explained_tokenizer, ref_token_id, special_tokens)
    return metric_functions.deletion_trajectory(data)


def _build_trajectory_points_from_payload(
        trajectory_payload: Dict[str, Any],
        trajectory_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_label_id = int(trajectory_payload["target_label_id"])
    target_label_text = _label_text_for_index(target_label_id)
    records = []
    for point in trajectory_payload.get("points", []):
        records.append(
            {
                "source_family": "aml",
                "run_id": str(trajectory_context["run_id"]),
                "report_stage": str(trajectory_context["report_stage"]),
                "dataset": str(trajectory_context["dataset"]),
                "split": str(trajectory_context["split"]),
                "model_name": str(trajectory_context["model_name"]),
                "method_name": str(trajectory_context.get("method_name", "aml")),
                "sample_id": str(trajectory_context["sample_id"]),
                "target_label_id": target_label_id,
                "target_label_text": target_label_text,
                "step_index": int(point["step_index"]),
                "total_steps": int(point["total_steps"]),
                "delete_count": int(point["delete_count"]),
                "delete_fraction": float(point["delete_fraction"]),
                "remaining_fraction": float(point["remaining_fraction"]),
                "target_probability": float(point["target_probability"]),
                "prob_drop_from_full": float(point["prob_drop_from_full"]),
                "is_full_text_step": bool(point["is_full_text_step"]),
                "deleted_ids": [int(index) for index in point.get("deleted_ids", [])],
            }
        )
    return records


def evaluate_all_metrics(
        model,
        explained_tokenizer,
        ref_token_id,
        data,
        experiment_path: str,
        step: int,
        epoch: int,
        item_index: str,
        metrics: Optional[Iterable[str]] = None,
        save_support_results: bool = False,
        experiment_name: Optional[str] = None,
        report_stage: Optional[str] = None,
        input_text = None,
        result_row_id: Optional[str] = None,
        selection_metric: Optional[str] = None,
        selection_metric_result: Optional[float] = None,
        selection_mode: Optional[str] = None,
        return_trajectory: bool = False,
        trajectory_context: Optional[Dict[str, Any]] = None):
    metrics = list(metrics or get_supported_eval_metrics())
    original_save_support_results = ExpArgs.is_save_support_results
    target_eval_metric = ExpArgs.target_eval_metric or ExpArgs.eval_metric
    all_metrics_results = []
    trajectory_payload = None
    trajectory_points: List[Dict[str, Any]] = []

    needs_trajectory = return_trajectory or (EvalMetric.AOPC.value in metrics)
    if needs_trajectory:
        trajectory_payload = _compute_trajectory_payload(
            model = model,
            explained_tokenizer = explained_tokenizer,
            ref_token_id = ref_token_id,
            data = data)
        if return_trajectory:
            if trajectory_context is None:
                raise ValueError("trajectory_context is required when return_trajectory=True")
            trajectory_points = _build_trajectory_points_from_payload(trajectory_payload, trajectory_context)

    try:
        ExpArgs.is_save_support_results = save_support_results
        for metric in metrics:
            _, evaluation_item = evaluate_tokens_attributions(model = model,
                                                              explained_tokenizer = explained_tokenizer,
                                                              ref_token_id = ref_token_id,
                                                              data = data,
                                                              experiment_path = experiment_path,
                                                              step = step,
                                                              epoch = epoch,
                                                              item_index = item_index,
                                                              eval_metric = metric,
                                                              trajectory_payload = trajectory_payload if metric == EvalMetric.AOPC.value else None)
            evaluation_item = evaluation_item.copy()
            evaluation_item["target_evaluation_metric"] = target_eval_metric
            if experiment_name is not None:
                evaluation_item["experiment_name"] = experiment_name
            if report_stage is not None:
                evaluation_item["report_stage"] = report_stage
            if result_row_id is not None:
                evaluation_item["result_row_id"] = str(result_row_id)
            if selection_metric is not None:
                evaluation_item["selection_metric"] = selection_metric
            if selection_metric_result is not None:
                evaluation_item["selection_metric_result"] = float(selection_metric_result)
            if selection_mode is not None:
                evaluation_item["selection_mode"] = selection_mode
            if input_text is not None:
                evaluation_item[INPUT_TXT] = normalize_input_text(input_text)
            all_metrics_results.append(evaluation_item)
    finally:
        ExpArgs.is_save_support_results = original_save_support_results

    if not all_metrics_results:
        if return_trajectory:
            return pd.DataFrame(), trajectory_points
        return pd.DataFrame()

    output = pd.concat(all_metrics_results, ignore_index = True)
    if return_trajectory:
        return output, trajectory_points
    return output


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding = "utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii = False) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding = "utf-8", newline = "") as file:
            file.write("")
        return

    fieldnames = list(rows[0].keys())
    normalized_rows = []
    for row in rows:
        normalized = {}
        for field in fieldnames:
            value = row.get(field)
            if isinstance(value, (list, dict)):
                normalized[field] = json.dumps(value, ensure_ascii = False)
            else:
                normalized[field] = value
        normalized_rows.append(normalized)

    with open(path, "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _trajectory_summary_rows(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for point in points:
        key = (
            point["source_family"],
            point["run_id"],
            point["report_stage"],
            point["dataset"],
            point["split"],
            point["model_name"],
            point["method_name"],
            point["step_index"],
            point["total_steps"],
            point["delete_count"],
            point["delete_fraction"],
            point["remaining_fraction"],
        )
        group = grouped.setdefault(
            key,
            {
                "source_family": point["source_family"],
                "run_id": point["run_id"],
                "report_stage": point["report_stage"],
                "dataset": point["dataset"],
                "split": point["split"],
                "model_name": point["model_name"],
                "method_name": point["method_name"],
                "step_index": point["step_index"],
                "total_steps": point["total_steps"],
                "delete_count": point["delete_count"],
                "delete_fraction": point["delete_fraction"],
                "remaining_fraction": point["remaining_fraction"],
                "target_probabilities": [],
                "prob_drops": [],
                "sample_ids": set(),
            },
        )
        group["target_probabilities"].append(float(point["target_probability"]))
        group["prob_drops"].append(float(point["prob_drop_from_full"]))
        group["sample_ids"].add(str(point["sample_id"]))

    rows = []
    for group in grouped.values():
        target_probabilities = np.asarray(group.pop("target_probabilities"), dtype = np.float64)
        prob_drops = np.asarray(group.pop("prob_drops"), dtype = np.float64)
        sample_ids = group.pop("sample_ids")
        group["mean_target_probability"] = float(np.mean(target_probabilities)) if len(target_probabilities) else 0.0
        group["std_target_probability"] = float(np.std(target_probabilities)) if len(target_probabilities) else 0.0
        group["mean_prob_drop_from_full"] = float(np.mean(prob_drops)) if len(prob_drops) else 0.0
        group["std_prob_drop_from_full"] = float(np.std(prob_drops)) if len(prob_drops) else 0.0
        group["sample_count"] = int(len(sample_ids))
        rows.append(group)

    rows.sort(
        key = lambda row: (
            str(row["dataset"]),
            str(row["model_name"]),
            str(row["method_name"]),
            int(row["step_index"]),
        )
    )
    return rows


def _metric_result_lookup(all_metrics_results: pd.DataFrame) -> Dict[str, float]:
    lookup = {}
    for metric_name, frame in all_metrics_results.groupby("evaluation_metric"):
        values = [float(value) for value in frame["metric_result"].tolist()]
        lookup[str(metric_name)] = float(np.mean(values)) if values else 0.0
    return lookup


def _build_primary_metrics(metric_lookup: Dict[str, float]) -> Dict[str, float]:
    return {
        report_name: float(metric_lookup.get(metric_name, 0.0))
        for metric_name, report_name in PRIMARY_METRIC_NAME_MAP.items()
    }


def _build_secondary_metrics(metric_lookup: Dict[str, float], all_metrics_results: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        report_name: float(metric_lookup.get(metric_name, 0.0))
        for metric_name, report_name in SECONDARY_METRIC_NAME_MAP.items()
        if metric_name in metric_lookup
    }
    if "selection_metric_result" in all_metrics_results.columns:
        selection_values = [
            float(value)
            for value in all_metrics_results["selection_metric_result"].tolist()
            if value is not None and not pd.isna(value)
        ]
        if selection_values:
            metrics["selection_metric_result_mean"] = _safe_mean(selection_values)
    return metrics


def _build_per_q_payload(all_metrics_results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    per_q = {
        "comprehensiveness": {},
        "sufficiency": {},
    }

    for metric_name, report_key in (
        (EvalMetric.AOPC_COMPREHENSIVENESS.value, "comprehensiveness"),
        (EvalMetric.AOPC_SUFFICIENCY.value, "sufficiency"),
    ):
        buckets: Dict[int, List[float]] = {}
        subset = all_metrics_results[all_metrics_results["evaluation_metric"] == metric_name]
        for _, row in subset.iterrows():
            steps_k = _coerce_int_list(row.get("steps_k"))
            values = _coerce_float_list(row.get("metric_steps_result"))
            for step_k, value in zip(steps_k, values):
                buckets.setdefault(int(step_k), []).append(float(value))
        per_q[report_key] = {
            str(int(step_k)): _safe_mean(values)
            for step_k, values in sorted(buckets.items(), key = lambda item: item[0])
        }

    return per_q


def _build_eval_report(
        *,
        all_metrics_results: pd.DataFrame,
        summary: pd.DataFrame,
        experiment_name: str,
        report_stage: str,
        dataset_name: str,
        split_name: str,
        model_name: str,
        trajectory_points: Sequence[Dict[str, Any]],
        selected_hyperparameters: Optional[dict],
        extra_metadata: Optional[dict],
        log_odds_reference_token: Optional[str]) -> Dict[str, Any]:
    metric_lookup = _metric_result_lookup(all_metrics_results)
    primary_metrics = _build_primary_metrics(metric_lookup)
    secondary_metrics = _build_secondary_metrics(metric_lookup, all_metrics_results)
    per_q = _build_per_q_payload(all_metrics_results)

    sample_ids = []
    for column in ("result_row_id", "item_index"):
        if column in all_metrics_results.columns:
            sample_ids.extend(str(value) for value in all_metrics_results[column].dropna().tolist())
            if sample_ids:
                break
    sample_count = len(set(sample_ids))
    if sample_count == 0 and trajectory_points:
        sample_count = len({str(point["sample_id"]) for point in trajectory_points})

    selection_mode_counts = {}
    if "selection_mode" in all_metrics_results.columns:
        selection_mode_counts = {
            str(key): int(value)
            for key, value in all_metrics_results["selection_mode"].value_counts(dropna = True).to_dict().items()
        }

    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_method": "aml",
        "report_stage": report_stage,
        "run_id": experiment_name,
        "dataset": dataset_name,
        "split": split_name,
        "model_name": model_name,
        "sample_count": int(sample_count),
        "metric_settings": {
            "protocol": "AML",
            "primary_top_k_percent": 20,
            "aopc_top_k_percentages": [1, 5, 10, 20, 50],
            "aopc_average_denominator": "len(top_k_percentages)+1",
            "plain_aopc_average_denominator": "m+1_including_full_text_step",
            "morf_average_denominator": "num_deletion_steps",
            "perturbation_target": "predicted",
            "perturbation_unit": "token",
            "log_odds_reference_token": log_odds_reference_token,
            "eval_tokens": ExpArgs.eval_tokens,
        },
        "metrics_primary": primary_metrics,
        "metrics_secondary": secondary_metrics,
        "metrics_by_target": {
            "predicted": {
                "metrics_primary": primary_metrics,
                "method_diagnostics": {
                    "evaluated_samples": int(sample_count),
                    "failed_samples": 0,
                    "report_stage": report_stage,
                    "selection_mode_counts": selection_mode_counts,
                },
                "per_q": per_q,
            }
        },
        "artifacts": {
            "eval_report_json": EVAL_REPORT_FILE_NAME,
            "trajectory_summary_csv": TRAJECTORY_SUMMARY_CSV,
            "trajectory_points_csv": TRAJECTORY_POINTS_CSV,
            "trajectory_points_jsonl": TRAJECTORY_POINTS_JSONL,
            "all_metrics_results_long": ALL_METRICS_LONG_RESULTS_FILE_NAME,
            "all_metrics_results_wide": ALL_METRICS_WIDE_RESULTS_FILE_NAME,
            "all_metrics_summary": ALL_METRICS_SUMMARY_FILE_NAME,
            "primary_results": "results.csv",
            "target_metric_support_results": "support_results_df.csv",
        },
        "dataset_diagnostics": {
            "selection_mode_counts": selection_mode_counts,
            "evaluated_metrics": get_supported_eval_metrics(),
            "trajectory_sample_count": int(len({str(point["sample_id"]) for point in trajectory_points})) if trajectory_points else 0,
        },
        "summary_by_metric": summary.to_dict(orient = "records"),
        "selected_hyperparameters": to_serializable(selected_hyperparameters),
        "loss_coefficients": dict(
            prediction_loss_weight = LossCoefficients.prediction_loss_weight,
            regularization_loss_weight = LossCoefficients.regularization_loss_weight,
            inverse_loss_weight = LossCoefficients.inverse_loss_weight),
        "experiment_arguments": get_experiment_arguments_snapshot(),
        "task": to_serializable(vars(ExpArgs.task)) if ExpArgs.task is not None else None,
        "extra_metadata": to_serializable(extra_metadata),
    }
    return report


def save_all_metrics_report(
        all_metrics_results: pd.DataFrame,
        experiment_path: str,
        experiment_name: str,
        report_stage: str,
        primary_results: Optional[pd.DataFrame] = None,
        selected_hyperparameters: Optional[dict] = None,
        extra_metadata: Optional[dict] = None,
        trajectory_points: Optional[Sequence[Dict[str, Any]]] = None,
        dataset_name: Optional[str] = None,
        split_name: Optional[str] = None,
        model_name: Optional[str] = None,
        log_odds_reference_token: Optional[str] = None):
    if all_metrics_results.empty:
        return

    results_path = Path(experiment_path)
    results_path.mkdir(parents = True, exist_ok = True)

    all_metrics_results = all_metrics_results.copy()
    trajectory_points = list(trajectory_points or [])

    all_metrics_results.to_csv(results_path / ALL_METRICS_LONG_RESULTS_FILE_NAME, index = False, encoding = "utf-8-sig")

    wide_results = build_wide_results(all_metrics_results)
    wide_results.to_csv(results_path / ALL_METRICS_WIDE_RESULTS_FILE_NAME, index = False, encoding = "utf-8-sig")

    summary = build_metrics_summary(all_metrics_results)
    summary.to_csv(results_path / ALL_METRICS_SUMMARY_FILE_NAME, index = False, encoding = "utf-8-sig")

    _write_csv(results_path / TRAJECTORY_POINTS_CSV, trajectory_points)
    _write_jsonl(results_path / TRAJECTORY_POINTS_JSONL, trajectory_points)
    _write_csv(results_path / TRAJECTORY_SUMMARY_CSV, _trajectory_summary_rows(trajectory_points))

    metadata = dict(
        schema_version = 2,
        generated_at_utc = datetime.now(timezone.utc).isoformat(),
        experiment_name = experiment_name,
        report_stage = report_stage,
        target_evaluation_metric = ExpArgs.target_eval_metric or ExpArgs.eval_metric,
        evaluated_metrics = get_supported_eval_metrics(),
        num_primary_results = int(len(primary_results)) if primary_results is not None else None,
        num_all_metric_rows = int(len(all_metrics_results)),
        selected_hyperparameters = to_serializable(selected_hyperparameters),
        loss_coefficients = dict(
            prediction_loss_weight = LossCoefficients.prediction_loss_weight,
            regularization_loss_weight = LossCoefficients.regularization_loss_weight,
            inverse_loss_weight = LossCoefficients.inverse_loss_weight),
        experiment_arguments = get_experiment_arguments_snapshot(),
        task = to_serializable(vars(ExpArgs.task)) if ExpArgs.task is not None else None,
        artifacts = dict(
            primary_results = "results.csv",
            target_metric_support_results = "support_results_df.csv",
            all_metrics_results_long = ALL_METRICS_LONG_RESULTS_FILE_NAME,
            all_metrics_results_wide = ALL_METRICS_WIDE_RESULTS_FILE_NAME,
            all_metrics_summary = ALL_METRICS_SUMMARY_FILE_NAME,
            trajectory_points_csv = TRAJECTORY_POINTS_CSV,
            trajectory_points_jsonl = TRAJECTORY_POINTS_JSONL,
            trajectory_summary_csv = TRAJECTORY_SUMMARY_CSV,
            eval_report_json = EVAL_REPORT_FILE_NAME),
        summary_by_metric = summary.to_dict(orient = "records"),
        extra_metadata = to_serializable(extra_metadata))

    report_path = results_path / ALL_METRICS_REPORT_FILE_NAME
    with open(report_path, "w", encoding = "utf-8") as file:
        json.dump(metadata, file, indent = 2, ensure_ascii = False)

    eval_report = _build_eval_report(
        all_metrics_results = all_metrics_results,
        summary = summary,
        experiment_name = experiment_name,
        report_stage = report_stage,
        dataset_name = dataset_name or getattr(ExpArgs.task, "name", "unknown"),
        split_name = split_name or getattr(ExpArgs.task, "dataset_test", "unknown"),
        model_name = model_name or normalize_model_name(ExpArgs.explained_model_path or ExpArgs.explained_model_backbone),
        trajectory_points = trajectory_points,
        selected_hyperparameters = selected_hyperparameters,
        extra_metadata = extra_metadata,
        log_odds_reference_token = log_odds_reference_token)
    with open(results_path / EVAL_REPORT_FILE_NAME, "w", encoding = "utf-8") as file:
        json.dump(eval_report, file, indent = 2, ensure_ascii = False)

    config_report_path = Path(ExpArgs.default_root_dir,
                              "CONFIG",
                              "EXPERIMENT_ARGUMENTS",
                              f"{experiment_name}_{report_stage}_ALL_METRICS.json")
    config_report_path.parent.mkdir(parents = True, exist_ok = True)
    with open(config_report_path, "w", encoding = "utf-8") as file:
        json.dump(metadata, file, indent = 2, ensure_ascii = False)


def build_metrics_summary(all_metrics_results: pd.DataFrame) -> pd.DataFrame:
    summary = all_metrics_results.groupby("evaluation_metric", as_index = False).agg(
        count = ("metric_result", "count"),
        mean = ("metric_result", "mean"),
        median = ("metric_result", "median"),
        min = ("metric_result", "min"),
        max = ("metric_result", "max"),
        std = ("metric_result", "std"))
    summary["std"] = summary["std"].fillna(0.0)

    metric_order = {metric_name: idx for idx, metric_name in enumerate(get_supported_eval_metrics())}
    summary["__metric_order__"] = summary["evaluation_metric"].map(metric_order)
    summary = summary.sort_values("__metric_order__").drop(columns = "__metric_order__")
    return summary


def build_wide_results(all_metrics_results: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "result_row_id",
        "item_index",
        "epoch",
        "step",
        "explained_model_predicted_class",
        INPUT_TXT,
        "target_evaluation_metric",
        "selection_metric",
        "selection_metric_result",
        "selection_mode",
        "experiment_name",
        "report_stage",
    ]
    base_columns = [column for column in base_columns if column in all_metrics_results.columns]

    if not base_columns:
        wide_results = all_metrics_results.copy()
        wide_results["result_row_id"] = list(range(len(wide_results)))
        base_columns = ["result_row_id"]

    pivot_source = all_metrics_results[base_columns + ["evaluation_metric", "metric_result"]].copy()
    wide_results = pivot_source.pivot_table(index = base_columns,
                                            columns = "evaluation_metric",
                                            values = "metric_result",
                                            aggfunc = "first").reset_index()
    wide_results.columns.name = None
    return wide_results


def get_experiment_arguments_snapshot():
    excluded = {"task", "label_vocab_tokens"}
    snapshot = {}
    for key, value in vars(ExpArgs).items():
        if key.startswith("__") or key in excluded:
            continue
        snapshot[key] = to_serializable(value)
    return snapshot


def normalize_input_text(input_text):
    if isinstance(input_text, list):
        if len(input_text) == 1:
            return input_text[0]
        return "\n".join(str(item) for item in input_text)
    return input_text


def to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {key: to_serializable(val) for key, val in vars(value).items() if not key.startswith("__")}
    return str(value)
