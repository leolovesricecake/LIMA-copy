#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


_FLOAT_FORWARD_COUNTER_KEYS = {
    "batch_tokenize_seconds",
    "batch_pack_seconds",
    "batch_forward_seconds",
}
_MODEL_INVOCATION_COUNTER_KEYS = ("predict_calls", "embed_calls", "gradient_calls")


def _accumulate_numeric_dict(
    totals: Dict[str, float | int],
    values: Dict[str, Any] | None,
    *,
    float_keys = (),
) -> Dict[str, float | int]:
    float_key_set = {str(key) for key in float_keys}
    payload = values or {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key in float_key_set or isinstance(value, float):
            totals[key] = float(totals.get(key, 0.0)) + float(value)
        else:
            totals[key] = int(totals.get(key, 0)) + int(value)
    return totals


def _divide_numeric_dict(values: Dict[str, float | int], denominator: int) -> Dict[str, float]:
    if denominator <= 0:
        return {}
    return {key: float(value) / float(denominator) for key, value in values.items()}


def _numeric_dict(values: Any) -> Dict[str, float | int]:
    if not isinstance(values, dict):
        return {}
    out: Dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key in _FLOAT_FORWARD_COUNTER_KEYS or isinstance(value, float):
            out[str(key)] = float(value)
        else:
            out[str(key)] = int(value)
    return out


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _counter_value(counters: Dict[str, float | int], key: str) -> float:
    value = counters.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _sum_model_invocations(counters: Dict[str, float | int]) -> float:
    return float(sum(_counter_value(counters, key) for key in _MODEL_INVOCATION_COUNTER_KEYS))


def _has_report_explain_counters(report: Dict[str, Any]) -> bool:
    metrics_secondary = report.get("metrics_secondary", {})
    if isinstance(metrics_secondary, dict) and isinstance(metrics_secondary.get("explain_forward_counters_total"), dict):
        return True
    explain_diagnostics = report.get("explain_diagnostics", {})
    return isinstance(explain_diagnostics, dict) and isinstance(explain_diagnostics.get("forward_counters_total"), dict)


def _add_explain_model_call_fields(row: Dict[str, Any], report: Dict[str, Any]) -> None:
    metrics_secondary = report.get("metrics_secondary", {})
    if not isinstance(metrics_secondary, dict):
        metrics_secondary = {}
    explain_diagnostics = report.get("explain_diagnostics", {})
    if not isinstance(explain_diagnostics, dict):
        explain_diagnostics = {}

    sample_count = _positive_int(report.get("sample_count"))
    explain_sample_count = _positive_int(explain_diagnostics.get("sample_count")) or sample_count

    explain_total = _numeric_dict(metrics_secondary.get("explain_forward_counters_total"))
    if not explain_total:
        explain_total = _numeric_dict(explain_diagnostics.get("forward_counters_total"))
    if not explain_total:
        row["explain_model_calls_total"] = 0.0
        row["explain_model_calls_mean_per_sample"] = 0.0
        return

    total_calls = _sum_model_invocations(explain_total)
    mean_calls = total_calls / float(explain_sample_count) if explain_sample_count > 0 else 0.0
    row["explain_model_calls_total"] = total_calls
    row["explain_model_calls_mean_per_sample"] = mean_calls


def _aggregate_explain_stats_from_samples(method_dir: Path) -> Dict[str, Any]:
    sample_dir = method_dir / "samples"
    if not sample_dir.exists():
        return {}

    sample_jsons = sorted(sample_dir.glob("*.json"))
    if not sample_jsons:
        return {}

    explain_forward_counters_total: Dict[str, float | int] = {}
    explain_timing_totals: Dict[str, float | int] = {}
    explain_elapsed_seconds_total = 0.0
    explain_sample_count = 0

    for sample_json in sample_jsons:
        try:
            payload = json.loads(sample_json.read_text(encoding = "utf-8"))
        except Exception:
            continue
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        explain_sample_count += 1
        _accumulate_numeric_dict(
            explain_forward_counters_total,
            metadata.get("forward_counters_delta"),
            float_keys = _FLOAT_FORWARD_COUNTER_KEYS,
        )
        timing = metadata.get("explain_timing_breakdown")
        if isinstance(timing, dict):
            _accumulate_numeric_dict(explain_timing_totals, timing, float_keys = tuple(timing.keys()))
        if isinstance(metadata.get("elapsed_seconds"), (int, float)):
            explain_elapsed_seconds_total += float(metadata["elapsed_seconds"])

    if explain_sample_count == 0:
        return {}

    return {
        "sample_count": int(explain_sample_count),
        "forward_counters_total": explain_forward_counters_total,
        "forward_counters_mean_per_sample": _divide_numeric_dict(
            explain_forward_counters_total,
            explain_sample_count,
        ),
        "timing_totals": explain_timing_totals,
        "timing_mean_per_sample": _divide_numeric_dict(explain_timing_totals, explain_sample_count),
        "elapsed_seconds_total": float(explain_elapsed_seconds_total),
        "elapsed_seconds_mean_per_sample": float(explain_elapsed_seconds_total) / float(explain_sample_count),
    }


def collect_eval_reports(input_dir: Path) -> List[Dict[str, Any]]:
    """
    收集形如 dataset/model/method/eval_report.json 的实验结果。

    每一行包含：
    - dataset / model / method
    - report_method / split / sample_count
    - metrics_primary 中的所有指标
    - explain_model_calls_total / explain_model_calls_mean_per_sample
    """
    rows: List[Dict[str, Any]] = []

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # 只扫描预期的三级目录：dataset/model/method
    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        print(f'Dataset: {dataset_dir}')
        
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            print(f'\t Model: {model_dir}')

            for method_dir in sorted(model_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                
                report_path = method_dir / "eval_report.json"

                # 如果第三级目录下没有 eval_report.json，则跳过
                if not report_path.is_file():
                    print(f'\t\t T^T Method {method_dir} has no report!')
                    continue
                print(f'\t\t Method: {method_dir}')

                try:
                    with report_path.open("r", encoding="utf-8") as f:
                        report = json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to read JSON: {report_path} ({e})")
                    continue

                explain_stats_fallback = {}
                if not _has_report_explain_counters(report):
                    explain_stats_fallback = _aggregate_explain_stats_from_samples(method_dir)
                    if explain_stats_fallback and not isinstance(report.get("explain_diagnostics"), dict):
                        report["explain_diagnostics"] = explain_stats_fallback
                if explain_stats_fallback:
                    metrics_secondary = report.setdefault("metrics_secondary", {})
                    if not isinstance(metrics_secondary.get("explain_forward_counters_total"), dict):
                        metrics_secondary["explain_forward_counters_total"] = explain_stats_fallback.get(
                            "forward_counters_total",
                            {},
                        )
                    if not isinstance(metrics_secondary.get("explain_forward_counters_mean_per_sample"), dict):
                        metrics_secondary["explain_forward_counters_mean_per_sample"] = explain_stats_fallback.get(
                            "forward_counters_mean_per_sample",
                            {},
                        )
                    if not isinstance(metrics_secondary.get("explain_timing_totals"), dict):
                        metrics_secondary["explain_timing_totals"] = explain_stats_fallback.get("timing_totals", {})
                    if not isinstance(metrics_secondary.get("explain_timing_mean_per_sample"), dict):
                        metrics_secondary["explain_timing_mean_per_sample"] = explain_stats_fallback.get(
                            "timing_mean_per_sample",
                            {},
                        )
                    metrics_secondary.setdefault(
                        "explain_elapsed_seconds_total",
                        explain_stats_fallback.get("elapsed_seconds_total"),
                    )
                    metrics_secondary.setdefault(
                        "explain_elapsed_seconds_mean_per_sample",
                        explain_stats_fallback.get("elapsed_seconds_mean_per_sample"),
                    )

                metrics = report.get("metrics_primary")
                if not isinstance(metrics, dict):
                    print(f"[WARN] Missing or invalid metrics_primary: {report_path}")
                    continue

                row = {
                    "dataset": dataset_dir.name,
                    "model": model_dir.name,
                    "method": method_dir.name,
                    "report_method": report.get("report_method"),
                    "split": report.get("split"),
                    "sample_count": report.get("sample_count"),
                }
                row.update(metrics)
                _add_explain_model_call_fields(row, report)

                rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """
    将收集到的结果写入 CSV。

    指标列会自动取所有 JSON 中出现过的 metrics_primary key 的并集。
    """
    base_columns = ["dataset", "model", "method", "report_method", "split", "sample_count"]

    metric_columns = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in base_columns
        }
    )

    columns = base_columns + metric_columns

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect metrics_primary from eval_report.json files into a CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input result directory, e.g. results/baselines/inseq",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="eval_summary.csv",
        help="Output CSV path. Default: eval_summary.csv",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = input_dir / Path(args.output)

    rows = collect_eval_reports(input_dir)
    write_csv(rows, output_csv)

    print(f"[INFO] Collected {len(rows)} eval reports.")
    print(f"[INFO] Saved CSV to: {output_csv}")


if __name__ == "__main__":
    main()
