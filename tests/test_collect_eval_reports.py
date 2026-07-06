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
                "predict_calls": 40,
            },
            "eval_forward_counters_delta": {
                "model_forward_calls": 100,
                "predict_calls": 40,
            },
            "explain_forward_counters_total": {
                "model_forward_calls": 30,
                "predict_calls": 12,
                "embed_calls": 6,
            },
            "explain_forward_counters_mean_per_sample": {
                "model_forward_calls": 15.0,
                "predict_calls": 6.0,
                "embed_calls": 3.0,
            },
            "runtime_seconds": 12.5,
        },
        "explain_diagnostics": {
            "sample_count": 2,
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
    assert row["explain_model_calls_total"] == 18.0
    assert row["explain_model_calls_mean_per_sample"] == 9.0
    assert "metrics_secondary.forward_counters_delta.model_forward_calls" not in row
    assert "metrics_secondary.runtime_seconds" not in row

    output_csv = tmp_path / "summary.csv"
    write_csv(rows, output_csv)
    with output_csv.open("r", encoding = "utf-8") as file:
        reader = csv.DictReader(file)
        written_row = next(reader)
    assert written_row["explain_model_calls_total"] == "18.0"
    assert written_row["explain_model_calls_mean_per_sample"] == "9.0"


def test_collect_eval_reports_prefers_report_explain_counters_over_sample_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "sst2" / "model-Qwen2_5-7B-Instruct" / "method-ours"
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents = True, exist_ok = True)
    report = {
        "report_method": "ours",
        "split": "validation",
        "sample_count": 1,
        "metrics_primary": {"accuracy_full": 1.0},
        "metrics_secondary": {
            "explain_forward_counters_total": {
                "predict_calls": 2,
                "embed_calls": 3,
                "gradient_calls": 0,
            },
        },
        "explain_diagnostics": {"sample_count": 1},
    }
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii = False), encoding = "utf-8")
    sample_payload = {
        "metadata": {
            "forward_counters_delta": {
                "predict_calls": 100,
                "embed_calls": 100,
                "gradient_calls": 100,
            },
        },
    }
    (sample_dir / "s0.json").write_text(json.dumps(sample_payload, ensure_ascii = False), encoding = "utf-8")

    rows = collect_eval_reports(tmp_path)
    assert len(rows) == 1
    assert rows[0]["explain_model_calls_total"] == 5.0
    assert rows[0]["explain_model_calls_mean_per_sample"] == 5.0
