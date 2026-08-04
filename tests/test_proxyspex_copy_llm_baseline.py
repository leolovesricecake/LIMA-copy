from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mobius.core.artifacts import (
    normalize_surrogate_artifact,
    predict_surrogate,
)
from mobius.text.chunks import (
    build_chunks,
    build_eval_units,
    project_ranking,
    validate_coverage,
)


def _load_module(name: str, relative_path: str):
    """Load the retained runner directly from its repository path."""

    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module(
    "proxyspex_copy_runner_under_test",
    "baselines/shapiq-copy/run_proxyspex_llm_baseline.py",
)


class _CharTokenizer:
    def __call__(self, text, **kwargs):
        """Return one tokenizer offset per input character."""

        ids = list(range(len(text)))
        payload = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


class _Backbone:
    """Provide deterministic probabilities and observable batch-call counters."""

    def __init__(self):
        """Initialize the character tokenizer and counters."""

        self.tokenizer = _CharTokenizer()
        self.single_calls = 0
        self.batch_calls = 0

    def predict_label_probs(self, _text, _verbalizers):
        """Score one full input."""

        self.single_calls += 1
        return np.asarray([0.2, 0.8], dtype=np.float32)

    def predict_label_probs_batch(self, texts, _verbalizers):
        """Score one coalition batch."""

        self.batch_calls += 1
        return np.asarray([[0.2, 0.8] for _ in texts], dtype=np.float32)

    def snapshot_counters(self):
        """Expose enough counters to detect artifact-only model calls."""

        return {
            "predict_calls": self.single_calls + self.batch_calls,
            "model_forward_calls": self.single_calls + self.batch_calls,
            "batch_calls": self.batch_calls,
            "batch_rows": 0,
        }


class _DummyProxySPEX:
    """Provide a deterministic spectral surrogate for runner contract tests."""

    seen_n = []

    def __init__(self, **kwargs):
        """Record the requested player count."""

        self.n = int(kwargs["n"])
        self.__class__.seen_n.append(self.n)

    def approximate(self, budget, game):
        """Evaluate one native coalition and expose fitted ProxySPEX artifacts."""

        if self.n > 0:
            self.coalitions_matrix_ = np.ones((1, self.n), dtype=bool)
            game(self.coalitions_matrix_)
            self.unrefined_fourier_ = {
                (): 0.25,
                (0,): 0.5,
            }
            self.refined_fourier_ = {
                (): 0.25,
                (0,): 0.5,
            }
        return SimpleNamespace(
            dict_values={(idx,): float(self.n - idx) for idx in range(self.n)},
            baseline_value=0.0,
        )

    def predict_refined_fourier(self, coalitions_matrix):
        """Evaluate the deterministic parity surrogate used by the test double."""

        matrix = np.asarray(coalitions_matrix, dtype=bool)
        predictions = np.zeros(matrix.shape[0], dtype=np.float64)
        for interaction, coefficient in self.refined_fourier_.items():
            if not interaction:
                predictions += float(coefficient)
                continue
            parity = np.sum(matrix[:, list(interaction)], axis=1) % 2
            predictions += float(coefficient) * np.where(parity == 0, 1.0, -1.0)
        return predictions


def _args(**overrides):
    """Build the minimal runner argument namespace used by unit tests."""

    defaults = dict(
        chunker="word",
        adaptive_profile="balanced",
        adaptive_overrides=None,
        adaptive_overrides_json=None,
        eval_granularity="token",
        max_length=512,
        target_mode="gold",
        index="FBII",
        max_order=2,
        sampling_weight_mode="uniform_coalition",
        budget=16,
        proxy_model="tree",
        hpo=False,
        proxy_n_jobs=1,
        quiet_proxy=True,
        pairing_trick=False,
        top_order=False,
        seed=7,
        interaction_metadata_limit=16,
        k=2,
        value_function="predicted_probability",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_proxyspex_chunking_token_uses_tokenizer_offsets() -> None:
    """Check token chunks follow tokenizer offsets exactly."""

    result = build_chunks(
        text="abc",
        chunker="token",
        tokenizer=_CharTokenizer(),
    )
    assert [chunk.text for chunk in result.chunks] == ["a", "b", "c"]
    assert [chunk.chunk_id for chunk in result.chunks] == [0, 1, 2]
    assert result.fallback_used is False
    assert result.diagnostics["chunk_strategy_requested"] == "token"


def test_proxyspex_chunking_word_preserves_coverage_and_ids() -> None:
    """Check word chunks retain whitespace and full text coverage."""

    text = "good movie"
    result = build_chunks(text=text, chunker="word", tokenizer=None)
    ok, msg = validate_coverage(text, result.chunks)
    assert ok, msg
    assert [chunk.chunk_id for chunk in result.chunks] == [0, 1]
    assert [chunk.text for chunk in result.chunks] == ["good ", "movie"]
    assert result.diagnostics["chunk_strategy"] == "word"


def test_proxyspex_chunking_adaptive_emits_diagnostics() -> None:
    """Check adaptive chunks expose profile and strategy diagnostics."""

    text = " ".join(f"tok{i}" for i in range(24)) + "."
    result = build_chunks(text=text, chunker="adaptive", tokenizer=None)
    ok, msg = validate_coverage(text, result.chunks)
    assert ok, msg
    assert result.diagnostics["chunk_strategy_requested"] == "adaptive"
    assert result.diagnostics["adaptive_profile"] == "balanced"


def test_copy_runner_defaults_to_word_explanation_and_evaluation() -> None:
    """Check the paper default target and word-level granularities."""

    args = RUNNER.build_parser().parse_args(["--dataset", "sst2", "--model-path", "tiny-local"])
    assert args.chunker == "word"
    assert args.eval_granularity == "word"
    assert args.value_function == "predicted_probability"
    assert args.target_mode == "predicted"


def test_short_sample_hpo_uses_feasible_cross_validation() -> None:
    """Protect the n_samples=4 regression that previously requested five folds."""

    assert RUNNER._expected_proxy_fit_sample_count(2, 512) == 4
    assert RUNNER._proxy_hpo_cv_splits(4) == 2
    assert RUNNER._proxy_hpo_cv_splits(3) is None


def test_predicted_probability_forces_predicted_target() -> None:
    """Check predicted probability ignores an incompatible gold request."""

    sample = SimpleNamespace(
        sample_id="sample-predicted",
        text="good movie",
        label=0,
        label_text="negative",
    )
    bundle = SimpleNamespace(
        verbalizers=["negative", "positive"],
        dataset_name="sst2",
        split="validation",
    )
    result = RUNNER._explain_sample(
        sample=sample,
        bundle=bundle,
        backbone=_Backbone(),
        args=_args(value_function="predicted_probability", target_mode="gold"),
        ProxySPEX=_DummyProxySPEX,
    )
    assert result.target_label == 1
    assert result.method_summary["target_mode"] == "predicted"
    assert result.method_summary["target_mode_requested"] == "gold"


def test_target_probability_can_reproduce_gold_target_behavior() -> None:
    """Check target probability preserves explicit gold-target behavior."""

    sample = SimpleNamespace(
        sample_id="sample-gold",
        text="good movie",
        label=0,
        label_text="negative",
    )
    bundle = SimpleNamespace(
        verbalizers=["negative", "positive"],
        dataset_name="sst2",
        split="validation",
    )
    result = RUNNER._explain_sample(
        sample=sample,
        bundle=bundle,
        backbone=_Backbone(),
        args=_args(value_function="target_probability", target_mode="gold"),
        ProxySPEX=_DummyProxySPEX,
    )
    assert result.target_label == 0
    assert result.method_summary["target_mode"] == "gold"
    assert np.isclose(result.method_summary["attribution_value"], 0.2)


def test_proxy_game_supports_probability_and_margin_values() -> None:
    """Check coalition values support both probability and margin semantics."""

    units = [SimpleNamespace(chunk_id=0, start_char=0, end_char=1, text="x")]
    probability_game = RUNNER.ProxySPEXCoalitionGame(
        units=units,
        player_to_chunk_id=[0],
        backbone=_Backbone(),
        verbalizers=["negative", "positive"],
        target_label=1,
        value_function="predicted_probability",
    )
    margin_game = RUNNER.ProxySPEXCoalitionGame(
        units=units,
        player_to_chunk_id=[0],
        backbone=_Backbone(),
        verbalizers=["negative", "positive"],
        target_label=1,
        value_function="predicted_class_margin",
    )
    coalition = np.asarray([[True]], dtype=bool)
    assert np.isclose(probability_game(coalition)[0], 0.8)
    assert np.isclose(margin_game(coalition)[0], np.log(4.0))


def test_explain_sample_uses_word_chunks_as_players_even_when_eval_is_token() -> None:
    """Check explanation and evaluation granularities remain independent."""

    _DummyProxySPEX.seen_n = []
    sample = SimpleNamespace(
        sample_id="sample-1",
        text="good movie",
        label=1,
        label_text="positive",
    )
    bundle = SimpleNamespace(
        verbalizers=["negative", "positive"],
        dataset_name="sst2",
        split="validation",
    )

    backbone = _Backbone()
    result = RUNNER._explain_sample(
        sample=sample,
        bundle=bundle,
        backbone=backbone,
        args=_args(chunker="word", eval_granularity="token"),
        ProxySPEX=_DummyProxySPEX,
    )

    assert _DummyProxySPEX.seen_n == [2]
    assert [chunk.text for chunk in result.chunks] == ["good ", "movie"]
    assert result.method_summary["proxyspex_chunker"] == "word"
    assert result.method_summary["eval_granularity"] == "token"
    assert result.method_summary["chunk_diagnostics"]["chunk_strategy"] == "word"
    assert result.method_summary["player_to_chunk_id"] == [0, 1]
    assert result.observation_artifact["keep_masks"].shape == (1, 2)
    assert result.observation_artifact["label_scores"].shape == (1, 2)
    assert result.surrogate_artifact["predictor"]["type"] == "refined_fourier"
    assert result.surrogate_artifact["refined_fourier"]["support_size"] == 1
    assert backbone.single_calls == 1
    assert backbone.batch_calls == 1


def test_adaptive_chunks_can_project_to_token_eval_units() -> None:
    """Check adaptive chunk rankings project to complete token rankings."""

    _DummyProxySPEX.seen_n = []
    text = " ".join(f"tok{i}" for i in range(24)) + "."
    sample = SimpleNamespace(sample_id="sample-2", text=text, label=1, label_text="positive")
    bundle = SimpleNamespace(verbalizers=["negative", "positive"], dataset_name="sst2", split="validation")
    backbone = _Backbone()

    result = RUNNER._explain_sample(
        sample=sample,
        bundle=bundle,
        backbone=backbone,
        args=_args(chunker="adaptive", eval_granularity="token", k=3),
        ProxySPEX=_DummyProxySPEX,
    )

    ok, msg = validate_coverage(text, result.chunks)
    assert ok, msg
    eval_result = build_eval_units(text, "token", backbone.tokenizer)
    ranking_units = project_ranking(
        eval_units=eval_result.chunks,
        chunks=result.chunks,
        chunk_ranking=result.ranking,
    )
    assert sorted(ranking_units) == list(range(len(eval_result.chunks)))
    assert result.method_summary["proxyspex_chunker"] == "adaptive"


def test_lightgbm_python_dump_converter_preserves_proxy_tree_function() -> None:
    """Ensure the no-extension converter preserves tree outputs and Fourier extraction."""

    if sys.version_info < (3, 12):
        pytest.skip("The retained shapiq-copy package requires Python >= 3.12.")

    local_source = str(RUNNER._LOCAL_SHAPIQ_SRC)
    if local_source not in sys.path:
        sys.path.insert(0, local_source)
    from shapiq.approximator.proxy.proxyspex import ProxySPEX
    from shapiq.tree.conversion._lightgbm_dump import convert_lightgbm_dump_model

    class _FakeBooster:
        """Expose a deterministic LightGBM-compatible structured model dump."""

        def dump_model(self):
            """Return one branching tree and one constant tree."""

            return {
                "num_tree_per_iteration": 1,
                "tree_info": [
                    {
                        "tree_structure": {
                            "split_index": 0,
                            "split_feature": 0,
                            "threshold": 0.5,
                            "decision_type": "<=",
                            "default_left": True,
                            "internal_count": 8,
                            "left_child": {
                                "leaf_index": 0,
                                "leaf_value": 1.0,
                                "leaf_count": 4,
                            },
                            "right_child": {
                                "split_index": 1,
                                "split_feature": 1,
                                "threshold": 0.5,
                                "decision_type": "<=",
                                "default_left": False,
                                "internal_count": 4,
                                "left_child": {
                                    "leaf_index": 1,
                                    "leaf_value": 2.0,
                                    "leaf_count": 2,
                                },
                                "right_child": {
                                    "leaf_index": 2,
                                    "leaf_value": 3.0,
                                    "leaf_count": 2,
                                },
                            },
                        }
                    },
                    {
                        "tree_structure": {
                            "leaf_value": -0.5,
                            "leaf_count": 8,
                        }
                    },
                ],
            }

    trees = convert_lightgbm_dump_model(_FakeBooster())
    masks = np.asarray(
        [[False, False], [False, True], [True, False], [True, True]],
        dtype=bool,
    )
    tree_predictions = np.asarray(
        [sum(tree.predict_one(row) for tree in trees) for row in masks],
        dtype=np.float64,
    )
    assert np.allclose(tree_predictions, [0.5, 0.5, 1.5, 2.5])
    assert {tree.conversion_backend for tree in trees} == {
        "lightgbm_python_dump"
    }

    approximator = ProxySPEX(
        n=2,
        max_order=2,
        index="FBII",
        proxy_model="tree",
        hpo=False,
        random_state=9,
    )
    approximator.refined_fourier_ = approximator._sklearn_to_fourier(trees)
    assert np.allclose(
        approximator.predict_refined_fourier(masks),
        tree_predictions,
    )


def test_refined_fourier_artifact_matches_native_predictor() -> None:
    """Check serialized refined Fourier predictions equal native ProxySPEX."""

    if sys.version_info < (3, 12):
        pytest.skip("The retained shapiq-copy package requires Python >= 3.12.")

    local_source = str(RUNNER._LOCAL_SHAPIQ_SRC)
    if local_source not in sys.path:
        sys.path.insert(0, local_source)
    from shapiq.approximator.proxy.proxyspex import ProxySPEX

    seen = {}

    def game(matrix):
        """Evaluate a deterministic nonlinear function and retain native inputs."""

        values = np.asarray(matrix, dtype=np.float64)
        seen["matrix"] = values.astype(bool)
        seen["values"] = (
            0.2
            + 0.5 * values[:, 0]
            - 0.3 * values[:, 1]
            + 0.8 * values[:, 0] * values[:, 2]
        )
        return seen["values"]

    approximator = ProxySPEX(
        n=3,
        max_order=2,
        index="FBII",
        proxy_model="tree",
        hpo=False,
        random_state=9,
    )
    approximator.approximate(budget=8, game=game)
    refined = RUNNER._spectral_terms_payload(
        approximator.refined_fourier_
    )
    surrogate = normalize_surrogate_artifact(
        {
            "sample_id": "spectral",
            "method": "proxyspex",
            "n_features": 3,
            "player_to_chunk_id": [0, 1, 2],
            "predictor": {
                "type": "refined_fourier",
                "basis": "fourier",
                "intercept": refined["intercept"],
                "terms": refined["terms"],
            },
        }
    )
    masks = np.asarray(
        [
            [False, False, False],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    assert np.array_equal(
        approximator.coalitions_matrix_,
        seen["matrix"],
    )
    assert np.allclose(approximator.coalition_values_, seen["values"])
    assert np.allclose(
        predict_surrogate(surrogate, masks),
        approximator.predict_refined_fourier(masks),
    )
