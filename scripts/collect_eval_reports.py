#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def collect_eval_reports(input_dir: Path) -> List[Dict[str, Any]]:
    """
    收集形如 dataset/model/method/eval_report.json 的实验结果。

    每一行包含：
    - dataset: 第一级目录名
    - model: 第二级目录名
    - method: 第三级目录名
    - metrics_primary 中的所有指标
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

                metrics = report.get("metrics_primary")
                if not isinstance(metrics, dict):
                    print(f"[WARN] Missing or invalid metrics_primary: {report_path}")
                    continue

                row = {
                    "dataset": dataset_dir.name,
                    "model": model_dir.name,
                    "method": method_dir.name,
                }
                row.update(metrics)
                rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """
    将收集到的结果写入 CSV。

    指标列会自动取所有 JSON 中出现过的 metrics_primary key 的并集。
    """
    base_columns = ["dataset", "model", "method"]

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