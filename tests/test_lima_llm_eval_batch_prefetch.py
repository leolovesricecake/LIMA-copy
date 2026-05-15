from __future__ import annotations

import json
from pathlib import Path

import pytest

from lima_llm.backbone import build_backbone
from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.data import load_dataset_bundle
from lima_llm.eval.evaluate import evaluate_saved_explanations
from lima_llm.pipeline.run import main
from lima_llm.utils import parse_q_values


def _build_tiny_eraser(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc-1.txt").write_text("This movie is great. I loved the acting.", encoding="utf-8")
    (docs / "doc-2.txt").write_text("This movie is terrible. Waste of time.", encoding="utf-8")

    rows = [
        {
            "annotation_id": "1",
            "classification": "POS",
            "evidences": [[{"docid": "doc-1", "start_char": 0, "end_char": 19}]],
        },
        {
            "annotation_id": "2",
            "classification": "NEG",
            "evidences": [[{"docid": "doc-2", "start_char": 0, "end_char": 23}]],
        },
    ]
    with open(root / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _canonical_report(report: dict) -> dict:
    out = json.loads(json.dumps(report))
    out["metric_settings"].pop("eval_batch_prefetch_enabled", None)
    out["metric_settings"].pop("prefetch_fallback_policy", None)
    out["metric_settings"].pop("prefetch_length_sort_enabled", None)
    out["metrics_secondary"].pop("runtime_seconds", None)
    out["metrics_secondary"].pop("forward_counters_delta", None)
    out["metrics_secondary"].pop("timing_breakdown", None)
    out["metrics_secondary"].pop("prefetch_stats", None)
    out["metrics_secondary"].pop("backbone_batch_stats", None)
    out.pop("timing_breakdown", None)
    out.pop("prefetch_stats", None)
    out.pop("backbone_batch_stats", None)
    return out


def test_eval_batch_prefetch_keeps_metrics_identical(monkeypatch, tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    results_root = tmp_path / "results"
    main(
        [
            "--dataset",
            "eraser_movie_reviews",
            "--split",
            "validation",
            "--eraser-root",
            str(eraser_root),
            "--mock-backbone",
            "--chunker",
            "sentence",
            "--search",
            "greedy",
            "--k",
            "2",
            "--output-dir",
            str(results_root),
        ]
    )

    run_dir = next(results_root.glob("**/chunk-sentence_search-greedy_k-2_lam-1-1-1-1_seed-42_method-ours"))
    bundle = load_dataset_bundle(
        dataset_name="eraser_movie_reviews",
        split="validation",
        eraser_root=str(eraser_root),
        max_samples=None,
    )
    q_values = parse_q_values("1,5,10,20,50")
    monkeypatch.setenv("LIMA_PREFETCH_FALLBACK_POLICY", "fail")

    backbone_batch = build_backbone(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        device="cpu",
        use_mock_backbone=True,
        max_length=2048,
        embedding_layer_ratio=0.7,
        dtype="bfloat16",
    )
    monkeypatch.setenv("LIMA_EVAL_BATCH_PREFETCH", "1")
    report_batch = evaluate_saved_explanations(
        output_root=run_dir,
        bundle=bundle,
        backbone=backbone_batch,
        verbalizers=bundle.verbalizers,
        q_values=q_values,
        explain_method="ours",
        eval_granularity="token",
    )

    backbone_single = build_backbone(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        device="cpu",
        use_mock_backbone=True,
        max_length=2048,
        embedding_layer_ratio=0.7,
        dtype="bfloat16",
    )
    monkeypatch.setenv("LIMA_EVAL_BATCH_PREFETCH", "0")
    report_single = evaluate_saved_explanations(
        output_root=run_dir,
        bundle=bundle,
        backbone=backbone_single,
        verbalizers=bundle.verbalizers,
        q_values=q_values,
        explain_method="ours",
        eval_granularity="token",
    )

    assert _canonical_report(report_batch) == _canonical_report(report_single)
    assert report_batch["metric_settings"]["perturbation_unit"] == "token"
    assert report_batch["dataset_diagnostics"]["tokenizer_fallback_samples"] == report_batch["sample_count"]
    assert report_batch["prefetch_stats"]["batch_fallback_count"] == 0
    assert report_single["prefetch_stats"]["batch_fallback_count"] == 0


class _BatchFailBackbone(MockBackbone):
    def predict_label_probs_batch(self, texts, verbalizers):
        raise RuntimeError("simulated batch failure")


def test_eval_prefetch_warn_policy_records_fallback(monkeypatch, tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    results_root = tmp_path / "results"
    main(
        [
            "--dataset",
            "eraser_movie_reviews",
            "--split",
            "validation",
            "--eraser-root",
            str(eraser_root),
            "--mock-backbone",
            "--chunker",
            "sentence",
            "--search",
            "greedy",
            "--k",
            "2",
            "--output-dir",
            str(results_root),
        ]
    )

    run_dir = next(results_root.glob("**/chunk-sentence_search-greedy_k-2_lam-1-1-1-1_seed-42_method-ours"))
    bundle = load_dataset_bundle(
        dataset_name="eraser_movie_reviews",
        split="validation",
        eraser_root=str(eraser_root),
        max_samples=None,
    )
    q_values = parse_q_values("1,5,10,20,50")

    monkeypatch.setenv("LIMA_EVAL_BATCH_PREFETCH", "1")
    monkeypatch.setenv("LIMA_PREFETCH_FALLBACK_POLICY", "warn")
    report = evaluate_saved_explanations(
        output_root=run_dir,
        bundle=bundle,
        backbone=_BatchFailBackbone(),
        verbalizers=bundle.verbalizers,
        q_values=q_values,
        explain_method="ours",
        eval_granularity="token",
    )

    assert report["prefetch_stats"]["batch_fallback_count"] > 0
    assert report["prefetch_stats"]["batch_attempt_count"] > 0
    assert report["prefetch_stats"]["batch_success_count"] == 0


def test_eval_prefetch_fail_policy_raises(monkeypatch, tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    results_root = tmp_path / "results"
    main(
        [
            "--dataset",
            "eraser_movie_reviews",
            "--split",
            "validation",
            "--eraser-root",
            str(eraser_root),
            "--mock-backbone",
            "--chunker",
            "sentence",
            "--search",
            "greedy",
            "--k",
            "2",
            "--output-dir",
            str(results_root),
        ]
    )

    run_dir = next(results_root.glob("**/chunk-sentence_search-greedy_k-2_lam-1-1-1-1_seed-42_method-ours"))
    bundle = load_dataset_bundle(
        dataset_name="eraser_movie_reviews",
        split="validation",
        eraser_root=str(eraser_root),
        max_samples=None,
    )
    q_values = parse_q_values("1,5,10,20,50")

    monkeypatch.setenv("LIMA_EVAL_BATCH_PREFETCH", "1")
    monkeypatch.setenv("LIMA_PREFETCH_FALLBACK_POLICY", "fail")
    with pytest.raises(RuntimeError, match="fallback policy is 'fail'"):
        evaluate_saved_explanations(
            output_root=run_dir,
            bundle=bundle,
            backbone=_BatchFailBackbone(),
            verbalizers=bundle.verbalizers,
            q_values=q_values,
            explain_method="ours",
            eval_granularity="token",
        )
