from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from lima_llm.pipeline.io import save_explanation
from lima_llm.types import ExplanationResult, TextChunk


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "baselines" / "captum" / "run_captum_llm_baselines.py"
    spec = importlib.util.spec_from_file_location("captum_runner_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def test_compose_prompt_input_ids_inserts_suffix_before_generated_tokens() -> None:
    full_ids = RUNNER._compose_prompt_input_ids(
        text_token_ids=[11, 12],
        continuation_token_ids=[91, 92],
        prefix_token_ids=[1, 2],
        suffix_token_ids=[3, 4],
    )
    assert full_ids == [1, 2, 11, 12, 3, 4, 91, 92]


def test_truncate_text_token_ids_uses_left_truncation_and_reports_dropped_prefix() -> None:
    truncation = RUNNER._truncate_text_token_ids(
        text_token_ids=[10, 11, 12, 13],
        prefix_token_ids=[1, 2],
        suffix_token_ids=[3],
        target_token_ids=[99, 100],
        max_length=7,
    )
    assert truncation["available_text_token_budget"] == 2
    assert truncation["dropped_left_token_count"] == 2
    assert truncation["kept_text_token_ids"] == [12, 13]


def test_rank_desc_scores_uses_raw_score_order_and_id_tiebreak() -> None:
    ranking, selected = RUNNER._rank_desc_scores([0.2, -1.0, 0.2, 0.1], k=2)
    assert ranking == [0, 2, 3, 1]
    assert selected == [0, 2]


def test_method_attr_kwargs_include_forward_mode_for_perturbation_methods() -> None:
    args = argparse.Namespace(
        forward_in_tokens=0,
        num_trials=3,
        n_samples=5,
        n_steps=7,
    )
    fa_kwargs = RUNNER._method_attr_kwargs(args, "feature_ablation")
    assert fa_kwargs["forward_in_tokens"] is False
    assert fa_kwargs["use_cached_outputs"] is False
    assert fa_kwargs["num_trials"] == 3

    lime_kwargs = RUNNER._method_attr_kwargs(args, "lime")
    assert lime_kwargs["forward_in_tokens"] is False
    assert lime_kwargs["use_cached_outputs"] is False
    assert lime_kwargs["n_samples"] == 5

    lig_kwargs = RUNNER._method_attr_kwargs(args, "layer_integrated_gradients")
    assert lig_kwargs == {"n_steps": 7}


def test_write_configs_emits_run_and_eval_payloads_with_provenance(tmp_path: Path) -> None:
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
        methods="feature_ablation",
        method="feature_ablation",
        k=2,
        seed=7,
        target_mode="gold",
        eval_q_values="1,5,10,20,50",
        eval_granularity="token",
        attr_target="log_prob",
        n_steps=8,
        n_samples=8,
        num_trials=1,
        forward_in_tokens=1,
        base_save_dir="results",
        save_dir="baselines/captum",
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
    assert run_cfg["method"] == "feature_ablation"
    assert eval_cfg["method"] == "feature_ablation"
    assert "provenance" in run_cfg
    assert "provenance" in eval_cfg
    assert "git" in run_cfg["provenance"]
    assert "command" in eval_cfg["provenance"]


def test_sample_json_schema_round_trips_with_explanation_result(tmp_path: Path) -> None:
    result = ExplanationResult(
        explain_method="feature_ablation",
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
        metadata={"captum_method": "feature_ablation", "raw_seq_attr": [0.5]},
    )
    save_explanation(result, tmp_path)
    payload = json.loads((tmp_path / "samples" / "sample-1.json").read_text(encoding="utf-8"))
    assert payload["explain_method"] == "feature_ablation"
    assert payload["chunk_ranking"] == [0]
    assert payload["chunk_scores"] == [0.5]
    assert payload["selected_chunk_ids"] == [0]
    assert payload["metadata"]["captum_method"] == "feature_ablation"


def test_factory_forwards_trust_remote_code(monkeypatch) -> None:
    from lima_llm.backbone import factory as factory_mod

    captured = {}

    class _DummyHFBackbone:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(factory_mod, "HFBackbone", _DummyHFBackbone)
    backbone = factory_mod.build_backbone(
        model_path="tiny-local",
        device="cpu",
        use_mock_backbone=False,
        max_length=64,
        embedding_layer_ratio=0.5,
        dtype="float32",
        trust_remote_code=True,
    )
    assert isinstance(backbone, _DummyHFBackbone)
    assert captured["trust_remote_code"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("captum") is None
    or importlib.util.find_spec("tokenizers") is None,
    reason="torch/transformers/captum/tokenizers are required for smoke coverage",
)
def test_captum_runner_smoke_with_tiny_local_hf_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
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
            "--methods",
            "feature_ablation,layer_integrated_gradients",
            "--max-samples",
            "2",
            "--base-save-dir",
            str(results_root),
            "--save-dir",
            "captum-smoke",
        ]
    )

    report_paths = sorted(results_root.glob("**/eval_report.json"))
    sample_jsons = sorted(results_root.glob("**/samples/*.json"))
    assert len(report_paths) == 2
    assert len(sample_jsons) == 4
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        primary = payload["metrics_primary"]
        assert "log_odds" in primary
        assert "comprehensiveness" in primary
        assert "sufficiency" in primary
        assert "aopc_sufficiency" in primary
        assert "aopc_comprehensiveness" in primary
