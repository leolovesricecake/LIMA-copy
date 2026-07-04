from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lima_llm.eval.metrics import EMPTY_PERTURBATION_TEXT
from lima_llm.pipeline.io import save_explanation
from lima_llm.types import ExplanationResult, TextChunk


def _load_runner_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "shapiq-main"
        / "run_proxyspex_llm_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("proxyspex_runner_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


class _CharTokenizer:
    def __call__(self, text, **kwargs):
        ids = list(range(len(text)))
        payload = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


def test_compose_coalition_text_uses_selected_players_and_empty_placeholder() -> None:
    units = [
        TextChunk(chunk_id=0, start_char=0, end_char=1, text="a"),
        TextChunk(chunk_id=1, start_char=1, end_char=3, text=" b"),
        TextChunk(chunk_id=2, start_char=3, end_char=5, text=" c"),
    ]
    assert (
        RUNNER._compose_coalition_text(
            units=units,
            player_to_chunk_id=[0, 2],
            coalition_row=[True, False],
        )
        == "a"
    )
    assert (
        RUNNER._compose_coalition_text(
            units=units,
            player_to_chunk_id=[0, 2],
            coalition_row=[True, True],
        )
        == "a c"
    )
    assert (
        RUNNER._compose_coalition_text(
            units=units,
            player_to_chunk_id=[0, 2],
            coalition_row=[False, False],
        )
        == EMPTY_PERTURBATION_TEXT
    )


def test_prompt_visible_span_matches_left_truncation_budget() -> None:
    payload = RUNNER._prompt_visible_text_span_after_left_truncation(
        tokenizer=_CharTokenizer(),
        text="abcd",
        label_text="y",
        max_length=12,
    )
    assert payload["target_token_count"] == 2
    assert payload["max_prompt_token_budget"] == 10
    assert payload["visible_start_char"] == 1
    assert payload["visible_end_char"] == 4
    assert payload["dropped_prompt_token_count"] == 7


def test_active_unit_ids_from_visible_span_drops_left_units() -> None:
    units = [
        TextChunk(chunk_id=0, start_char=0, end_char=1, text="a"),
        TextChunk(chunk_id=1, start_char=1, end_char=2, text="b"),
        TextChunk(chunk_id=2, start_char=2, end_char=3, text="c"),
        TextChunk(chunk_id=3, start_char=3, end_char=4, text="d"),
    ]
    assert RUNNER._active_unit_ids_from_visible_span(units, visible_start=1, visible_end=4) == [1, 2, 3]


def test_project_interactions_to_chunk_scores_uses_signed_equal_share() -> None:
    scores = RUNNER._project_interactions_to_chunk_scores(
        interaction_items=[
            ((0,), 0.6),
            ((0, 1), 0.4),
            ((1, 2), -0.3),
            ((), 1.0),
        ],
        player_to_chunk_id=[2, 4, 5],
        total_chunk_count=6,
    )
    assert scores == pytest.approx([0.0, 0.0, 0.8, 0.0, 0.05, -0.15])


def test_stable_uniform_coalition_sampling_weights_do_not_overflow_for_large_n() -> None:
    weights = RUNNER._sampling_weights(2048, "uniform_coalition")
    assert weights.shape == (2049,)
    assert weights.dtype.kind == "f"
    assert weights[0] > 0
    assert weights[-1] > 0
    assert weights[1024] == pytest.approx(1.0)
    assert weights[7] == pytest.approx(weights[-8])
    assert all(float(x) > 0.0 for x in weights)
    assert all(float(x) < float("inf") for x in weights)


def test_uniform_size_sampling_weights_are_available_for_ablation() -> None:
    weights = RUNNER._sampling_weights(4, "uniform_size")
    assert weights.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_build_proxy_model_tree_is_quiet_fallback_free() -> None:
    args = argparse.Namespace(
        proxy_model="tree",
        seed=7,
        proxy_n_jobs=1,
        quiet_proxy=True,
        hpo=False,
    )
    proxy_model, effective = RUNNER._build_proxy_model(args)
    assert effective == "tree"
    assert proxy_model.__class__.__name__ == "DecisionTreeRegressor"


def test_rank_desc_scores_uses_raw_scores_and_chunk_id_tiebreak() -> None:
    ranking, selected = RUNNER._rank_desc_scores([0.0, 0.5, 0.5, -0.2], k=3)
    assert ranking == [1, 2, 0, 3]
    assert selected == [1, 2, 0]


def test_write_configs_emits_provenance(tmp_path: Path) -> None:
    args = argparse.Namespace(
        dataset="sst2",
        split="validation",
        eraser_root=None,
        sst2_source=None,
        dataset_cache_dir=None,
        max_samples=2,
        model_path="tiny-local",
        device="cpu",
        dtype="float32",
        max_length=32,
        trust_remote_code=False,
        k=2,
        seed=7,
        target_mode="gold",
        eval_q_values="1,5,10,20,50",
        eval_granularity="token",
        budget=16,
        max_order=2,
        index="FBII",
        proxy_model="tree",
        proxy_n_jobs=1,
        quiet_proxy=True,
        sampling_weight_mode="uniform_coalition",
        hpo=False,
        pairing_trick=False,
        top_order=False,
        interaction_metadata_limit=16,
        base_save_dir="results",
        save_dir="baselines/proxyspex",
        resume_check="strict",
        deterministic=False,
    )
    RUNNER._write_configs(
        output_root=tmp_path,
        args=args,
        raw_argv=["--dataset", "sst2"],
        deterministic_info={"enabled": False, "applied": False},
    )
    run_cfg = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    eval_cfg = json.loads((tmp_path / "eval_config.json").read_text(encoding="utf-8"))
    assert run_cfg["method"] == "proxyspex"
    assert eval_cfg["method"] == "proxyspex"
    assert "provenance" in run_cfg
    assert "git" in run_cfg["provenance"]
    assert "command" in eval_cfg["provenance"]


def test_sample_json_schema_round_trips_with_proxyspex_metadata(tmp_path: Path) -> None:
    result = ExplanationResult(
        explain_method="proxyspex",
        sample_id="sample-1",
        dataset="sst2",
        split="validation",
        label=1,
        label_text="positive",
        text="good movie",
        chunks=[TextChunk(chunk_id=0, start_char=0, end_char=4, text="good")],
        chunk_ranking=[0],
        chunk_scores=[0.5],
        selected_chunk_ids=[0],
        selected_text="good",
        scores={"total": 0.5, "target_probability": 0.7, "label_probabilities": [0.3, 0.7]},
        trace=[],
        metadata={
            "proxyspex_method": "proxyspex",
            "projection_strategy": "signed_equal_share",
            "interaction_summary": {"total_interaction_count": 1},
        },
    )
    save_explanation(result, tmp_path)
    payload = json.loads((tmp_path / "samples" / "sample-1.json").read_text(encoding="utf-8"))
    assert payload["explain_method"] == "proxyspex"
    assert payload["metadata"]["projection_strategy"] == "signed_equal_share"
    assert payload["chunk_ranking"] == [0]
    assert payload["selected_chunk_ids"] == [0]


@pytest.mark.skipif(
    sys.version_info < (3, 12)
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("tokenizers") is None
    or importlib.util.find_spec("shapiq") is None,
    reason="Python 3.12+, torch, transformers, tokenizers, and shapiq are required for smoke coverage",
)
def test_proxyspex_runner_smoke_with_tiny_local_hf_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    tokenizers_mod = pytest.importorskip("tokenizers")

    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    model_root = tmp_path / "tiny_model"
    model_root.mkdir(parents=True, exist_ok=True)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "Text": 2,
        ":": 3,
        "Label": 4,
        "negative": 5,
        "positive": 6,
        "good": 7,
        "bad": 8,
        "movie": 9,
        "<": 10,
        "EMPTY": 11,
        ">": 12,
    }
    tokenizer_obj = tokenizers_mod.Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<pad>",
    )
    tokenizer.save_pretrained(model_root)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=2,
            n_head=4,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    model.save_pretrained(model_root)

    sst2_root = tmp_path / "sst2_local"
    sst2_root.mkdir(parents=True, exist_ok=True)
    (sst2_root / "validation.csv").write_text(
        "sentence,label\n"
        "good movie,1\n"
        "bad movie,0\n",
        encoding="utf-8",
    )

    results_root = tmp_path / "results"
    RUNNER.main(
        [
            "--dataset",
            "sst2",
            "--split",
            "validation",
            "--sst2-source",
            str(sst2_root),
            "--model-path",
            str(model_root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-length",
            "32",
            "--max-samples",
            "2",
            "--budget",
            "16",
            "--max-order",
            "2",
            "--proxy-model",
            "tree",
            "--no-hpo",
            "--base-save-dir",
            str(results_root),
            "--save-dir",
            "proxyspex-smoke",
        ]
    )

    report_paths = sorted(results_root.glob("**/eval_report.json"))
    sample_jsons = sorted(results_root.glob("**/samples/*.json"))
    assert len(report_paths) == 1
    assert len(sample_jsons) == 2

    payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
    primary = payload["metrics_primary"]
    assert "log_odds" in primary
    assert "comprehensiveness" in primary
    assert "sufficiency" in primary
    assert "aopc" in primary
    assert "aopc_sufficiency" in primary
    assert "aopc_comprehensiveness" in primary
