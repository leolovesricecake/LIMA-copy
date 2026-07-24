"""Tests for the schema-v2 result collection script."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.collect_results import (
    DEFAULT_OUTPUT,
    build_parser,
    collect_results,
    write_csv,
)


def _write_json(path: Path, payload: dict) -> None:
    """Write one compact fixture JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(root: Path) -> Path:
    """Create one complete schema-v2 result fixture."""

    run_dir = root / "sst2" / "Qwen3-8B" / "sparse_mobius" / "b512-o2-s42-std"
    _write_json(
        run_dir / "run.json",
        {
            "run_id": "b512-o2-s42-std",
            "scientific_config": {
                "budget": 512,
                "max_degree": 2,
                "seed": 42,
                "value_function": "target_probability",
                "target_mode": "predicted",
                "chunker": "word",
                "eval_granularity": "token",
                "basis": "deletion_mobius",
                "hierarchy": "none",
                "sampler": {"name": "deletion_mixture"},
                "projector": "signed_equal_share",
                "estimator": {"name": "lasso_cv_ridge_refit"},
                "dataset": {"name": "sst2", "split": "validation"},
                "model": {"model_path": "/models/Qwen3-8B"},
            },
        },
    )
    _write_json(
        run_dir / "status.json",
        {
            "state": "complete",
            "completed_count": 100,
            "skipped_count": 0,
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": "2.0",
            "target": "predicted",
            "sample_count": 100,
            "evaluated_count": 100,
            "failed_count": 0,
            "accuracy": 0.9,
            "faithfulness": {
                "aopc": {"count": 100, "mean": 0.4, "std": 0.1},
                "aupc": {"count": 100, "mean": 0.3, "std": 0.08},
            },
            "per_q": {
                "20": {
                    "comprehensiveness": {
                        "count": 100,
                        "mean": 0.35,
                        "std": 0.2,
                    }
                }
            },
            "attribution_cost": {
                "model_forward_calls": 5000,
                "logical_unique_queries": 51000,
                "physical_values_scored": 49000,
                "attribution_budget_used": 48000,
            },
            "evaluation_cost": {
                "elapsed_seconds": 12.0,
                "model_counter_delta": {
                    "model_forward_calls": 400,
                    "batch_calls": 200,
                    "batch_rows": 2400,
                },
            },
        },
    )
    return run_dir


def test_parser_has_requested_output_default() -> None:
    """Check the requested argument names and default CSV filename."""

    args = build_parser().parse_args(["--input_dir", "results/mobius"])
    assert args.input_dir == "results/mobius"
    assert args.o == DEFAULT_OUTPUT == "results_summary.csv"
    explicit = build_parser().parse_args(
        ["--input_dir", "results/mobius", "--o", "paper.csv"]
    )
    assert explicit.o == "paper.csv"


def test_collect_results_flattens_metrics_config_and_costs(tmp_path: Path) -> None:
    """Check identity, faithfulness, query costs, and per-sample normalization."""

    _make_run(tmp_path)
    rows = collect_results(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "sst2"
    assert row["model"] == "Qwen3-8B"
    assert row["method"] == "sparse_mobius"
    assert row["config"] == "b512-o2-s42-std"
    assert row["faithfulness_aopc_mean"] == 0.4
    assert row["faithfulness_aopc_std"] == 0.1
    assert row["faithfulness_aupc_mean"] == 0.3
    assert row["attribution_model_forward_calls"] == 5000.0
    assert row["attribution_model_forward_calls_per_sample"] == 50.0
    assert row["basis"] == "deletion_mobius"
    assert row["sampler"] == "deletion_mixture"
    assert row["order"] == 2
    assert "q20_comprehensiveness_mean" not in row
    assert "evaluation_model_forward_calls" not in row
    assert "accuracy" not in row


def test_write_csv_preserves_research_columns(tmp_path: Path) -> None:
    """Check deterministic CSV output includes required comparison columns."""

    _make_run(tmp_path / "input")
    rows = collect_results(tmp_path / "input")
    output = write_csv(rows, tmp_path / "summary.csv")
    with output.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["dataset"] == "sst2"
    assert written[0]["faithfulness_aopc_mean"] == "0.4"
    assert written[0]["attribution_model_forward_calls"] == "5000.0"
    assert list(written[0]) == [
        "dataset",
        "model",
        "method",
        "config",
        "target",
        "faithfulness_aopc_mean",
        "faithfulness_aopc_std",
        "faithfulness_aupc_mean",
        "faithfulness_aupc_std",
        "attribution_model_forward_calls",
        "attribution_model_forward_calls_per_sample",
        "budget",
        "order",
        "seed",
        "value_function",
        "chunker",
        "eval_granularity",
        "basis",
        "hierarchy",
        "sampler",
        "projector",
        "sample_count",
        "failed_count",
    ]
