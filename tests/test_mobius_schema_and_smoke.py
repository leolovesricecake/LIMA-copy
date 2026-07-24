"""Schema-v2 and mock end-to-end tests for the modular method."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

from mobius.core.config import resolve_config
from mobius.core.results import ResultStore, build_run_id
from mobius.core.schema import AttributionResult, DatasetBundle, TextChunk, TextSample
from mobius.evaluation.evaluator import evaluate_run
from mobius.methods.sparse.explainer import run_sparse_mobius
from mobius.models.mock import MockSentimentScorer


def _base_config() -> dict:
    """Return a compact deterministic method configuration."""

    return {
        "method": "sparse_mobius",
        "output_level": "standard",
        "budget": 16,
        "seed": 5,
        "max_degree": 2,
        "k": 3,
        "value_function": "target_probability",
        "target_mode": "predicted",
        "chunker": "word",
        "eval_granularity": "word",
        "eval_q_values": [20, 50],
        "min_features": 1,
        "max_features": None,
        "batch_size": 8,
        "targeted_top_k": 0,
        "basis": "deletion_mobius",
        "hierarchy": "none",
        "projector": "signed_equal_share",
        "sampler": {"name": "deletion_mixture"},
        "estimator": {
            "name": "lasso_cv_ridge_refit",
            "alphas": [0.0001, 0.001],
            "l1_ratios": [1.0],
            "cv_folds": 2,
        },
        "dataset": {"name": "inline", "split": "validation"},
        "model": {"type": "mock_sentiment", "device": "cpu"},
    }


def _bundle() -> DatasetBundle:
    """Return one nonlinear sentiment sample for fast fitting."""

    return DatasetBundle(
        dataset_name="inline",
        split="validation",
        samples=[
            TextSample(
                sample_id="s0",
                text="not good but finally moving",
                label=1,
                label_text="positive",
            )
        ],
        label_names=["negative", "positive"],
        verbalizers=["negative", "positive"],
    )


def test_run_id_ignores_runtime_device_but_changes_scientific_axis() -> None:
    """Check canonical hashing of scientific rather than machine-local settings."""

    left = _base_config()
    right = _base_config()
    right["model"]["device"] = "cuda:7"
    assert build_run_id(left) == build_run_id(right)
    right["basis"] = "fourier"
    assert build_run_id(left) != build_run_id(right)


def test_run_id_uses_configured_suffix_before_hash_fallback() -> None:
    """Check optional readable suffixes and safe hash fallback behavior."""

    default_id = build_run_id(_base_config())
    assert default_id.startswith("b16-o2-s5-")
    assert len(default_id.rsplit("-", 1)[-1]) == 8

    configured = _base_config()
    configured["run_suffix"] = "paper-main"
    assert build_run_id(configured) == "b16-o2-s5-paper-main"

    blank = _base_config()
    blank["run_suffix"] = "  "
    resolved_blank = resolve_config(blank)
    assert "run_suffix" not in resolved_blank
    assert len(build_run_id(resolved_blank).rsplit("-", 1)[-1]) == 8

    invalid = _base_config()
    invalid["run_suffix"] = "../escape"
    try:
        build_run_id(invalid)
    except ValueError as error:
        assert "run_suffix" in str(error)
    else:  # pragma: no cover - supports direct execution without pytest
        raise AssertionError("unsafe run_suffix should be rejected")


def test_result_store_honors_minimal_and_debug_output_levels(tmp_path: Path) -> None:
    """Check method summaries and diagnostics are gated by output level."""

    result = AttributionResult(
        sample_id="x",
        gold_label=0,
        predicted_label=0,
        target_label=0,
        text="bad",
        chunks=[TextChunk(0, 0, 3, "bad")],
        node_scores=[1.0],
        ranking=[0],
        selected_ids=[0],
        attribution_cost={"attribution_budget_used": 1},
        method_summary={"fit": "kept"},
        diagnostics={"masks": [0, 1]},
    )
    minimal = ResultStore(
        tmp_path / "minimal",
        _base_config(),
        output_level="minimal",
        overwrite=True,
    )
    minimal.write_sample(result)
    minimal_payload = json.loads(
        (minimal.samples_dir / "x.json").read_text(encoding="utf-8")
    )
    assert "method_summary" not in minimal_payload
    assert not (minimal.run_dir / "diagnostics").exists()

    debug = ResultStore(
        tmp_path / "debug",
        _base_config(),
        output_level="debug",
        overwrite=True,
    )
    debug.write_sample(result)
    debug_payload = json.loads(
        (debug.samples_dir / "x.json").read_text(encoding="utf-8")
    )
    assert debug_payload["method_summary"]["fit"] == "kept"
    assert (debug.diagnostics_dir / "x.json").exists()


def test_result_store_resumes_only_valid_schema_v2_samples(tmp_path: Path) -> None:
    """Check that a second store instance discovers a completed sample."""

    store = ResultStore(tmp_path / "run", _base_config(), overwrite=True)
    result = AttributionResult(
        sample_id="done",
        gold_label=0,
        predicted_label=0,
        target_label=0,
        text="bad",
        chunks=[TextChunk(0, 0, 3, "bad")],
        node_scores=[1.0],
        ranking=[0],
        selected_ids=[0],
        attribution_cost={},
    )
    store.write_sample(result)
    resumed = ResultStore(tmp_path / "run", _base_config(), overwrite=False)
    assert resumed.sample_complete("done")
    assert resumed.completed_ids == ["done"]


def test_evaluator_writes_one_explicit_target_block(tmp_path: Path) -> None:
    """Ensure metrics.json contains exactly one explicit evaluation target."""

    config = _base_config()
    scorer = MockSentimentScorer()
    run_sparse_mobius(
        config,
        _bundle(),
        scorer,
        run_dir=tmp_path / "run",
        cache_path=tmp_path / "cache.sqlite3",
        overwrite=True,
        evaluate=False,
    )
    report = evaluate_run(
        tmp_path / "run",
        _bundle(),
        scorer,
        target="gold",
        eval_granularity="word",
        q_values=[20, 50],
    )
    assert report["target"] == "gold"
    assert "metrics_primary" not in report
    assert "metrics_by_target" not in report
    assert "targets" not in report
    assert inspect.signature(evaluate_run).parameters["target"].default == "predicted"
    assert set(report["faithfulness"]) >= {
        "aopc",
        "aupc",
        "sufficiency",
        "comprehensiveness",
    }
    assert report["protocol"]["aupc_extra_model_calls"] == 0


def test_every_single_axis_ablation_runs_with_mock_scorer(tmp_path: Path) -> None:
    """Exercise both bases, hierarchies, all samplers, and all projectors end to end."""

    cells = [
        ("main", {}),
        ("basis-fourier", {"basis": "fourier"}),
        ("hierarchy-strong", {"hierarchy": "strong"}),
        ("sampler-bernoulli", {"sampler": {"name": "bernoulli"}}),
        ("sampler-uniform", {"sampler": {"name": "uniform_size"}}),
        ("projector-absolute", {"projector": "absolute_equal_share"}),
        ("projector-singleton", {"projector": "singleton_only"}),
    ]
    for name, overrides in cells:
        config = _base_config()
        config.update(overrides)
        scorer = MockSentimentScorer()
        run_dir = tmp_path / name
        status = run_sparse_mobius(
            config,
            _bundle(),
            scorer,
            run_dir=run_dir,
            cache_path=tmp_path / "global-cache.sqlite3",
            overwrite=True,
            evaluate=True,
        )
        assert status["state"] == "complete"
        assert status["completed_count"] == 1
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["target"] == "predicted"
        assert metrics["evaluated_count"] == 1
        assert "aopc" in metrics["faithfulness"]
        assert "aupc" in metrics["faithfulness"]
