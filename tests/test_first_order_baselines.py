"""Shared-value word Occlusion and LIME integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from mobius.core.artifacts import (
    load_observation_artifact,
    load_surrogate_artifact,
)
from mobius.core.schema import DatasetBundle, TextSample
from mobius.methods.first_order import run_first_order
from mobius.models.mock import MockSentimentScorer


def _bundle() -> DatasetBundle:
    """Build one short word-level sentiment sample."""

    return DatasetBundle(
        dataset_name="sst2",
        split="validation",
        samples=[
            TextSample(
                sample_id="s0",
                text="not good but moving",
                label=1,
                label_text="positive",
            )
        ],
        label_names=["negative", "positive"],
        verbalizers=["negative", "positive"],
    )


def _config(method: str) -> dict:
    """Build one aligned first-order configuration."""

    return {
        "method": method,
        "run_suffix": f"test-{method}",
        "budget": 8,
        "seed": 7,
        "max_degree": 1,
        "k": 2,
        "value_function": "predicted_probability",
        "target_mode": "predicted",
        "chunker": "word",
        "eval_granularity": "word",
        "eval_q_values": [5, 10, 20, 50],
        "min_features": 1,
        "max_features": None,
        "batch_size": 8,
        "first_order": {
            "lime_kernel_width": 25.0,
            "lime_ridge_alpha": 1.0,
        },
        "dataset": {"name": "sst2", "split": "validation"},
        "model": {"type": "mock_sentiment", "device": "cpu"},
    }


def test_first_order_methods_emit_complete_schema_v2_runs(tmp_path: Path) -> None:
    """Run both baselines through attribution, evaluation, and artifact writing."""

    for method in ("word_occlusion", "word_lime"):
        run_dir = tmp_path / method
        status = run_first_order(
            _config(method),
            _bundle(),
            MockSentimentScorer(),
            run_dir=run_dir,
            cache_path=tmp_path / "global.sqlite3",
            overwrite=True,
            evaluate=True,
        )
        assert status["state"] == "complete"
        assert status["completed_count"] == 1
        observation = load_observation_artifact(
            run_dir / "observations" / "s0.npz"
        )
        surrogate = load_surrogate_artifact(
            run_dir / "surrogates" / "s0.json"
        )
        sample = json.loads(
            (run_dir / "samples" / "s0.json").read_text(encoding="utf-8")
        )
        assert observation["n_features"] == 4
        assert surrogate["predictor"]["basis"] == "deletion_mobius"
        assert all(
            len(term["players"]) == 1
            for term in surrogate["predictor"]["terms"]
        )
        assert sample["method_summary"]["method"] == method
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "curves-predicted.jsonl").is_file()
        if method == "word_occlusion":
            assert observation["keep_masks"].shape[0] == 5
        else:
            assert observation["keep_masks"].shape[0] == 8
