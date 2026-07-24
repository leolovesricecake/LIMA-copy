from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mobius.text.chunks import (
    build_chunks,
    build_eval_units,
    project_ranking,
    validate_coverage,
)


def _load_module(name: str, relative_path: str):
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
        ids = list(range(len(text)))
        payload = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


class _Backbone:
    def __init__(self):
        self.tokenizer = _CharTokenizer()

    def predict_label_probs(self, _text, _verbalizers):
        return np.asarray([0.2, 0.8], dtype=np.float32)

    def predict_label_probs_batch(self, texts, _verbalizers):
        return np.asarray([[0.2, 0.8] for _ in texts], dtype=np.float32)

    def snapshot_counters(self):
        return {}


class _DummyProxySPEX:
    seen_n = []

    def __init__(self, **kwargs):
        self.n = int(kwargs["n"])
        self.__class__.seen_n.append(self.n)

    def approximate(self, budget, game):
        if self.n > 0:
            game(np.ones((1, self.n), dtype=bool))
        return SimpleNamespace(
            dict_values={(idx,): float(self.n - idx) for idx in range(self.n)},
            baseline_value=0.0,
        )


def _args(**overrides):
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
    text = "good movie"
    result = build_chunks(text=text, chunker="word", tokenizer=None)
    ok, msg = validate_coverage(text, result.chunks)
    assert ok, msg
    assert [chunk.chunk_id for chunk in result.chunks] == [0, 1]
    assert [chunk.text for chunk in result.chunks] == ["good ", "movie"]
    assert result.diagnostics["chunk_strategy"] == "word"


def test_proxyspex_chunking_adaptive_emits_diagnostics() -> None:
    text = " ".join(f"tok{i}" for i in range(24)) + "."
    result = build_chunks(text=text, chunker="adaptive", tokenizer=None)
    ok, msg = validate_coverage(text, result.chunks)
    assert ok, msg
    assert result.diagnostics["chunk_strategy_requested"] == "adaptive"
    assert result.diagnostics["adaptive_profile"] == "balanced"


def test_copy_runner_defaults_word_explanation_and_token_eval() -> None:
    """Check the fair default target and decoupled chunk/eval granularities."""

    args = RUNNER.build_parser().parse_args(["--dataset", "sst2", "--model-path", "tiny-local"])
    assert args.chunker == "word"
    assert args.eval_granularity == "token"
    assert args.value_function == "predicted_probability"
    assert args.target_mode == "predicted"


def test_short_sample_hpo_uses_feasible_cross_validation() -> None:
    """Protect the n_samples=4 regression that previously requested five folds."""

    assert RUNNER._expected_proxy_fit_sample_count(2, 512) == 4
    assert RUNNER._proxy_hpo_cv_splits(4) == 2
    assert RUNNER._proxy_hpo_cv_splits(3) is None


def test_predicted_probability_forces_predicted_target() -> None:
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

    result = RUNNER._explain_sample(
        sample=sample,
        bundle=bundle,
        backbone=_Backbone(),
        args=_args(chunker="word", eval_granularity="token"),
        ProxySPEX=_DummyProxySPEX,
    )

    assert _DummyProxySPEX.seen_n == [2]
    assert [chunk.text for chunk in result.chunks] == ["good ", "movie"]
    assert result.method_summary["proxyspex_chunker"] == "word"
    assert result.method_summary["eval_granularity"] == "token"
    assert result.method_summary["chunk_diagnostics"]["chunk_strategy"] == "word"
    assert result.method_summary["player_to_chunk_id"] == [0, 1]


def test_adaptive_chunks_can_project_to_token_eval_units() -> None:
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
