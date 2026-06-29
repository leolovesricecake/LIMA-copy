from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.collect_eval_reports import collect_eval_reports, write_csv


def test_collect_eval_reports_includes_call_count_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "emotion" / "model-Qwen2_5-7B-Instruct" / "method-ours"
    run_dir.mkdir(parents = True, exist_ok = True)
    report = {
        "report_method": "ours",
        "split": "test",
        "sample_count": 2,
        "metrics_primary": {
            "accuracy_full": 1.0,
            "comprehensiveness": 0.2,
        },
        "metrics_secondary": {
            "forward_counters_delta": {
                "model_forward_calls": 100,
            },
            "eval_forward_counters_delta": {
                "model_forward_calls": 100,
            },
            "explain_forward_counters_total": {
                "model_forward_calls": 30,
                "predict_calls": 12,
            },
            "explain_forward_counters_mean_per_sample": {
                "model_forward_calls": 15.0,
            },
            "runtime_seconds": 12.5,
        },
        "explain_diagnostics": {
            "forward_counters_total": {
                "model_forward_calls": 30,
            },
            "elapsed_seconds_total": 8.0,
        },
    }
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii = False), encoding = "utf-8")

    rows = collect_eval_reports(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "emotion"
    assert row["report_method"] == "ours"
    assert row["metrics_secondary.forward_counters_delta.model_forward_calls"] == 100
    assert row["metrics_secondary.explain_forward_counters_total.model_forward_calls"] == 30
    assert row["metrics_secondary.explain_forward_counters_mean_per_sample.model_forward_calls"] == 15.0
    assert row["metrics_secondary.runtime_seconds"] == 12.5

    output_csv = tmp_path / "summary.csv"
    write_csv(rows, output_csv)
    with output_csv.open("r", encoding = "utf-8") as file:
        reader = csv.DictReader(file)
        written_row = next(reader)
    assert written_row["metrics_secondary.explain_forward_counters_total.predict_calls"] == "12"
