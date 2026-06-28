from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index = False)


def test_plot_morf_curves_reads_aml_and_lima_sources(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    from scripts.plot_morf_curves import main

    aml_run = tmp_path / "aml" / "run-a"
    aml_run.mkdir(parents = True, exist_ok = True)
    _write_csv(
        aml_run / "trajectory_summary.csv",
        [
            dict(
                source_family = "aml",
                run_id = "aml-run-1",
                report_stage = "FINE_TUNE",
                dataset = "sst2",
                split = "validation",
                model_name = "Qwen3-8B",
                method_name = "aml",
                step_index = 0,
                total_steps = 2,
                delete_count = 0,
                delete_fraction = 0.0,
                remaining_fraction = 1.0,
                mean_target_probability = 0.9,
                std_target_probability = 0.0,
                mean_prob_drop_from_full = 0.0,
                std_prob_drop_from_full = 0.0,
                sample_count = 2,
            ),
            dict(
                source_family = "aml",
                run_id = "aml-run-1",
                report_stage = "FINE_TUNE",
                dataset = "sst2",
                split = "validation",
                model_name = "Qwen3-8B",
                method_name = "aml",
                step_index = 1,
                total_steps = 2,
                delete_count = 1,
                delete_fraction = 0.5,
                remaining_fraction = 0.5,
                mean_target_probability = 0.7,
                std_target_probability = 0.1,
                mean_prob_drop_from_full = 0.2,
                std_prob_drop_from_full = 0.1,
                sample_count = 2,
            ),
        ],
    )
    (aml_run / "eval_report.json").write_text(
        json.dumps({"report_method": "aml", "dataset": "sst2", "model_name": "Qwen3-8B", "report_stage": "FINE_TUNE"}),
        encoding = "utf-8",
    )

    lima_run = tmp_path / "lima" / "sst2" / "model-Qwen3-8B" / "chunk-sentence_method-lime"
    lima_run.mkdir(parents = True, exist_ok = True)
    _write_csv(
        lima_run / "trajectory_points.csv",
        [
            dict(
                source_family = "lima_llm",
                run_id = "lime-run-1",
                report_stage = "EVAL",
                dataset = "sst2",
                split = "validation",
                model_name = "Qwen3-8B",
                method_name = "lime",
                sample_id = "s0",
                target_label_id = 1,
                target_label_text = "positive",
                step_index = 0,
                total_steps = 2,
                delete_count = 0,
                delete_fraction = 0.0,
                remaining_fraction = 1.0,
                target_probability = 0.88,
                prob_drop_from_full = 0.0,
                is_full_text_step = True,
                deleted_ids = "[]",
            ),
            dict(
                source_family = "lima_llm",
                run_id = "lime-run-1",
                report_stage = "EVAL",
                dataset = "sst2",
                split = "validation",
                model_name = "Qwen3-8B",
                method_name = "lime",
                sample_id = "s0",
                target_label_id = 1,
                target_label_text = "positive",
                step_index = 1,
                total_steps = 2,
                delete_count = 1,
                delete_fraction = 0.5,
                remaining_fraction = 0.5,
                target_probability = 0.66,
                prob_drop_from_full = 0.22,
                is_full_text_step = False,
                deleted_ids = "[3]",
            ),
        ],
    )
    (lima_run / "eval_report.json").write_text(
        json.dumps({"report_method": "lime", "dataset": "sst2", "model_name": "Qwen3-8B", "report_stage": "EVAL"}),
        encoding = "utf-8",
    )

    output_dir = tmp_path / "plots"
    main(
        [
            "--aml-root",
            str(tmp_path / "aml"),
            "--lima-root",
            str(tmp_path / "lima"),
            "--output-dir",
            str(output_dir),
        ]
    )

    plot_path = output_dir / "qwen3-8b-sst2.png"
    manifest_path = output_dir / "curve_plot_manifest.csv"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert manifest_path.exists()

    manifest = pd.read_csv(manifest_path)
    assert set(manifest["method_name"]) == {"aml", "lime"}
