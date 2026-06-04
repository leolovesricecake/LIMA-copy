import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from config.config import ExpArgs, LossCoefficients
from config.constants import INPUT_TXT
from config.types_enums import EvalMetric
from evaluations.evaluations import evaluate_tokens_attributions


ALL_METRICS_LONG_RESULTS_FILE_NAME = "all_metrics_results_long.csv"
ALL_METRICS_WIDE_RESULTS_FILE_NAME = "all_metrics_results_wide.csv"
ALL_METRICS_SUMMARY_FILE_NAME = "all_metrics_summary.csv"
ALL_METRICS_REPORT_FILE_NAME = "all_metrics_report.json"


def get_supported_eval_metrics():
    return [metric.value for metric in EvalMetric]


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
        selection_mode: Optional[str] = None) -> pd.DataFrame:
    metrics = list(metrics or get_supported_eval_metrics())
    original_save_support_results = ExpArgs.is_save_support_results
    target_eval_metric = ExpArgs.target_eval_metric or ExpArgs.eval_metric
    all_metrics_results = []

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
                                                              eval_metric = metric)
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

    return pd.concat(all_metrics_results, ignore_index = True)


def save_all_metrics_report(
        all_metrics_results: pd.DataFrame,
        experiment_path: str,
        experiment_name: str,
        report_stage: str,
        primary_results: Optional[pd.DataFrame] = None,
        selected_hyperparameters: Optional[dict] = None,
        extra_metadata: Optional[dict] = None):
    if all_metrics_results.empty:
        return

    results_path = Path(experiment_path)
    results_path.mkdir(parents = True, exist_ok = True)

    all_metrics_results = all_metrics_results.copy()
    all_metrics_results.to_csv(results_path / ALL_METRICS_LONG_RESULTS_FILE_NAME, index = False, encoding = "utf-8-sig")

    wide_results = build_wide_results(all_metrics_results)
    wide_results.to_csv(results_path / ALL_METRICS_WIDE_RESULTS_FILE_NAME, index = False, encoding = "utf-8-sig")

    summary = build_metrics_summary(all_metrics_results)
    summary.to_csv(results_path / ALL_METRICS_SUMMARY_FILE_NAME, index = False, encoding = "utf-8-sig")

    metadata = dict(
        schema_version = 1,
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
            all_metrics_summary = ALL_METRICS_SUMMARY_FILE_NAME),
        summary_by_metric = summary.to_dict(orient = "records"),
        extra_metadata = to_serializable(extra_metadata))

    report_path = results_path / ALL_METRICS_REPORT_FILE_NAME
    with open(report_path, "w", encoding = "utf-8") as file:
        json.dump(metadata, file, indent = 2, ensure_ascii = False)

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
