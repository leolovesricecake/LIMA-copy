from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lima_llm.chunking.utils import validate_chunk_coverage
from lima_llm.eval.units import build_eval_units, project_chunk_ranking_to_unit_ranking


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHUNKING = _load_module(
    "proxyspex_copy_chunking_under_test",
    "baselines/shapiq-copy/proxyspex_chunking.py",
)
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
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_proxyspex_chunking_token_uses_tokenizer_offsets() -> None:
    result = CHUNKING.build_proxyspex_chunks(
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
    result = CHUNKING.build_proxyspex_chunks(text=text, chunker="word", tokenizer=None)
    ok, msg = validate_chunk_coverage(text, result.chunks)
    assert ok, msg
    assert [chunk.chunk_id for chunk in result.chunks] == [0, 1]
    assert [chunk.text for chunk in result.chunks] == ["good ", "movie"]
    assert result.diagnostics["chunk_strategy"] == "word"


def test_proxyspex_chunking_adaptive_emits_diagnostics() -> None:
    text = " ".join(f"tok{i}" for i in range(24)) + "."
    result = CHUNKING.build_proxyspex_chunks(text=text, chunker="adaptive", tokenizer=None)
    ok, msg = validate_chunk_coverage(text, result.chunks)
    assert ok, msg
    assert result.diagnostics["chunk_strategy_requested"] == "adaptive"
    assert result.diagnostics["adaptive_profile"] == "balanced"


def test_copy_runner_defaults_word_explanation_and_token_eval() -> None:
    args = RUNNER.build_parser().parse_args(["--dataset", "sst2", "--model-path", "tiny-local"])
    assert args.chunker == "word"
    assert args.eval_granularity == "token"


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
    assert result.metadata["proxyspex_chunker"] == "word"
    assert result.metadata["eval_granularity"] == "token"
    assert result.metadata["chunk_diagnostics"]["chunk_strategy"] == "word"
    assert result.metadata["player_to_chunk_id"] == [0, 1]


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

    ok, msg = validate_chunk_coverage(text, result.chunks)
    assert ok, msg
    eval_units, _fallback, _strategy = build_eval_units(text=text, eval_granularity="token", tokenizer=backbone.tokenizer)
    ranking_units = project_chunk_ranking_to_unit_ranking(
        eval_units=eval_units,
        chunks=result.chunks,
        chunk_ranking=result.chunk_ranking,
    )
    assert sorted(ranking_units) == list(range(len(eval_units)))
    assert result.metadata["proxyspex_chunker"] == "adaptive"
