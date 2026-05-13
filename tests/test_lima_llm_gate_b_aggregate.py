import json
from pathlib import Path

from lima_llm.eval.gate_b import aggregate_gate_b_runs, collect_gate_b_runs


def _write_method_run(
    run_dir: Path,
    *,
    seed: int,
    method: str,
    comp: float,
    suff: float,
    pred_comp: float,
    pred_suff: float,
    predict_calls: float,
    runtime: float,
    eval_granularity: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "eraser_movie_reviews",
        "split": "validation",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "chunker": "sentence",
        "search": "greedy",
        "k": 8,
        "lambdas": "1,1,1,1",
        "eval_q_values": "1,5,10,20,50",
        "max_samples": None,
        "seed": seed,
        "explain_method": method,
    }
    if eval_granularity is not None:
        run_config["eval_granularity"] = eval_granularity
    report = {
        "report_method": method,
        "metrics_secondary": {
            "runtime_seconds": runtime,
            "forward_counters_delta": {"predict_calls": predict_calls},
        },
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "log_odds": 0.1,
                    "aopc_sufficiency": 0.2,
                    "aopc_comprehensiveness": 0.3,
                    "aopc": 0.4,
                    "deletion_auc": 0.5,
                    "insertion_auc": 0.6,
                }
            },
            "predicted": {
                "metrics_primary": {
                    "comprehensiveness": pred_comp,
                    "sufficiency": pred_suff,
                    "log_odds": 0.1,
                    "aopc_sufficiency": 0.2,
                    "aopc_comprehensiveness": 0.3,
                    "aopc": 0.4,
                    "deletion_auc": 0.5,
                    "insertion_auc": 0.6,
                }
            },
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False), encoding="utf-8")
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_gate_b_aggregate_pairs_primary_and_reference_methods(tmp_path: Path) -> None:
    root = tmp_path / "results"

    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=0.03,
        pred_comp=0.09,
        pred_suff=0.04,
        predict_calls=120,
        runtime=12.0,
    )
    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-random",
        seed=42,
        method="random",
        comp=0.05,
        suff=0.21,
        pred_comp=0.05,
        pred_suff=0.20,
        predict_calls=90,
        runtime=8.0,
    )

    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-43_method-ours",
        seed=43,
        method="ours",
        comp=0.11,
        suff=0.04,
        pred_comp=0.10,
        pred_suff=0.05,
        predict_calls=125,
        runtime=12.5,
    )
    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-43_method-random",
        seed=43,
        method="random",
        comp=0.06,
        suff=0.20,
        pred_comp=0.06,
        pred_suff=0.19,
        predict_calls=92,
        runtime=8.2,
    )

    # Extra method is allowed but should not block primary/reference pairing.
    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-gradient",
        seed=42,
        method="gradient",
        comp=0.07,
        suff=0.16,
        pred_comp=0.07,
        pred_suff=0.16,
        predict_calls=110,
        runtime=11.0,
    )

    runs = collect_gate_b_runs(root)
    payload = aggregate_gate_b_runs(runs=runs, min_runs=2, primary_method="ours", reference_method="random")

    assert payload["run_count"] == 5
    assert payload["paired_run_count"] == 2
    assert payload["group_count"] == 1

    group = payload["groups"][0]
    assert group["n_runs"] == 2
    assert group["seeds"] == [42, 43]
    assert group["methods"]["primary"] == "ours"
    assert group["methods"]["reference"] == "random"
    assert group["methods"]["available_run_counts"]["gradient"] == 1
    assert group["gate_b_checks"]["comp_beats_reference_all"] is True
    assert group["gate_b_checks"]["suff_beats_reference_all"] is True
    assert group["gate_b_checks"]["run_pass_rate"] == 1.0


def test_gate_b_aggregate_respects_min_runs(tmp_path: Path) -> None:
    root = tmp_path / "results"

    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=0.03,
        pred_comp=0.10,
        pred_suff=0.03,
        predict_calls=120,
        runtime=12.0,
    )
    _write_method_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-random",
        seed=42,
        method="random",
        comp=0.05,
        suff=0.20,
        pred_comp=0.05,
        pred_suff=0.20,
        predict_calls=90,
        runtime=8.0,
    )

    runs = collect_gate_b_runs(root)
    payload = aggregate_gate_b_runs(runs=runs, min_runs=2, primary_method="ours", reference_method="random")
    assert payload["group_count"] == 0


def test_gate_b_aggregate_isolates_eval_granularity(tmp_path: Path) -> None:
    root = tmp_path / "results"

    _write_method_run(
        root / "a_seed42_ours_token",
        seed=42,
        method="ours",
        comp=0.10,
        suff=0.03,
        pred_comp=0.09,
        pred_suff=0.04,
        predict_calls=120,
        runtime=12.0,
        eval_granularity="token",
    )
    _write_method_run(
        root / "a_seed42_random_token",
        seed=42,
        method="random",
        comp=0.05,
        suff=0.21,
        pred_comp=0.05,
        pred_suff=0.20,
        predict_calls=90,
        runtime=8.0,
        eval_granularity="token",
    )
    _write_method_run(
        root / "b_seed42_ours_word",
        seed=42,
        method="ours",
        comp=0.10,
        suff=0.03,
        pred_comp=0.09,
        pred_suff=0.04,
        predict_calls=120,
        runtime=12.0,
        eval_granularity="word",
    )
    _write_method_run(
        root / "b_seed42_random_word",
        seed=42,
        method="random",
        comp=0.05,
        suff=0.21,
        pred_comp=0.05,
        pred_suff=0.20,
        predict_calls=90,
        runtime=8.0,
        eval_granularity="word",
    )

    runs = collect_gate_b_runs(root)
    payload = aggregate_gate_b_runs(runs=runs, min_runs=1, primary_method="ours", reference_method="random")
    assert payload["group_count"] == 2
    group_ids = [group["group_id"] for group in payload["groups"]]
    assert any("|eval=token" in gid for gid in group_ids)
    assert any("|eval=word" in gid for gid in group_ids)


def test_gate_b_aggregate_legacy_missing_granularity_defaults_to_word(tmp_path: Path) -> None:
    root = tmp_path / "results"

    _write_method_run(
        root / "legacy_seed42_ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=0.03,
        pred_comp=0.10,
        pred_suff=0.03,
        predict_calls=120,
        runtime=12.0,
        eval_granularity=None,
    )
    _write_method_run(
        root / "explicit_seed42_random_word",
        seed=42,
        method="random",
        comp=0.05,
        suff=0.20,
        pred_comp=0.05,
        pred_suff=0.20,
        predict_calls=90,
        runtime=8.0,
        eval_granularity="word",
    )

    runs = collect_gate_b_runs(root)
    payload = aggregate_gate_b_runs(runs=runs, min_runs=1, primary_method="ours", reference_method="random")
    assert payload["group_count"] == 1
    assert "|eval=word" in payload["groups"][0]["group_id"]
