import json
from pathlib import Path

from lima_llm.eval.gate_b import aggregate_gate_b_runs, collect_gate_b_runs


def _write_run(run_dir: Path, seed: int, comp: float, suff: float, rand_comp: float, rand_suff: float) -> None:
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
        "eval_random_trials": 5,
        "eval_gradient_baseline": True,
        "max_samples": None,
        "seed": seed,
    }
    report = {
        "metrics_secondary": {"runtime_seconds": 10.0, "forward_counters_delta": {"predict_calls": 100}},
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {"comprehensiveness": comp, "sufficiency": suff},
                "diagnosticity_vs_random": 0.7,
                "baselines": {"random": {"comprehensiveness": rand_comp, "sufficiency": rand_suff}},
            },
            "predicted": {"metrics_primary": {"comprehensiveness": comp, "sufficiency": suff}},
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False), encoding="utf-8")
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_gate_b_aggregate_groups_multi_seed_runs(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42",
        seed=42,
        comp=0.10,
        suff=0.02,
        rand_comp=0.04,
        rand_suff=0.20,
    )
    _write_run(
        root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct" / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-43",
        seed=43,
        comp=0.12,
        suff=0.03,
        rand_comp=0.05,
        rand_suff=0.19,
    )

    runs = collect_gate_b_runs(root)
    payload = aggregate_gate_b_runs(runs, min_runs=2)

    assert payload["run_count"] == 2
    assert payload["group_count"] == 1
    group = payload["groups"][0]
    assert group["n_runs"] == 2
    assert group["seeds"] == [42, 43]
    assert group["gate_b_checks"]["comp_beats_random_all"] is True
    assert group["gate_b_checks"]["suff_beats_random_all"] is True
