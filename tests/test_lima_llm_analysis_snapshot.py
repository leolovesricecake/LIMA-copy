import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(
    run_dir: Path,
    *,
    seed: int,
    method: str,
    comp: float,
    suff: float,
    eval_seconds: float,
    explain_elapsed: float,
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
        "sample_count": 1,
        "metrics_primary": {"accuracy_full": 1.0},
        "metrics_secondary": {
            "runtime_seconds": eval_seconds,
            "plausibility_f1": 0.2,
            "plausibility_iou": 0.1,
            "sparsity": 0.3,
            "forward_counters_delta": {
                "predict_calls": 10,
                "embed_calls": 2,
                "gradient_calls": 1,
            },
        },
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": 0.1,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "aopc": 0.3,
                    "aopc_sufficiency": 0.2,
                    "aopc_comprehensiveness": 0.1,
                }
            },
            "predicted": {
                "metrics_primary": {
                    "log_odds": 0.1,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "aopc": 0.3,
                    "aopc_sufficiency": 0.2,
                    "aopc_comprehensiveness": 0.1,
                }
            },
        },
    }

    sample = {
        "sample_id": "s1",
        "chunks": [{"chunk_id": 0, "start_char": 0, "end_char": 2, "text": "aa"}],
        "selected_chunk_ids": [0],
        "metadata": {
            "elapsed_seconds": explain_elapsed,
            "explain_timing_breakdown": {
                "chunk_build_seconds": 0.1,
                "search_seconds": max(0.0, explain_elapsed - 0.2),
                "model_prefetch_seconds": 0.05,
            },
            "objective_cache_stats": {
                "subset_requested": 10,
                "subset_cache_hits": 4,
                "subset_cache_misses": 6,
                "prob_cache_hits": 8,
                "prob_cache_misses": 2,
                "embed_cache_hits": 7,
                "embed_cache_misses": 3,
                "evaluate_gains_calls": 5,
            },
            "forward_counters": {
                "predict_calls": 3,
                "embed_calls": 4,
                "gradient_calls": 5,
            },
        },
    }

    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False), encoding="utf-8")
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "s1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_analysis_snapshot_and_pairwise(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_snapshot.py"
    mod = _load_module(script, "analysis_snapshot")

    root = tmp_path / "results"
    base = root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct"
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=-0.10,
        eval_seconds=10.0,
        explain_elapsed=2.0,
    )
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-gradient",
        seed=42,
        method="gradient",
        comp=0.12,
        suff=0.20,
        eval_seconds=11.0,
        explain_elapsed=1.0,
    )

    snapshot = mod.build_snapshot(root, primary_method="ours", reference_method="gradient")
    assert snapshot["group_count"] == 1
    group = snapshot["groups"][0]
    assert "ours" in group["methods"]
    assert "gradient" in group["methods"]

    pair = group["pairwise"]
    assert pair is not None
    assert pair["gold_comp_adv"] == -0.01999999999999999
    assert pair["gold_suff_adv"] == 0.30000000000000004
    assert pair["gold_pass"] is False

    ours = group["methods"]["ours"]
    assert ours["explain"]["timing_breakdown_totals"]["chunk_build_seconds"] == 0.1
    assert ours["explain"]["timing_breakdown_totals"]["model_prefetch_seconds"] == 0.05
    assert ours["explain"]["objective_cache_stats"]["evaluate_gains_calls"] == 5
    assert abs(ours["explain"]["objective_cache_stats"]["subset_cache_hit_rate"] - 0.4) <= 1e-12


def test_analysis_snapshot_diff(tmp_path: Path) -> None:
    snap_script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_snapshot.py"
    diff_script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_snapshot_diff.py"
    snap_mod = _load_module(snap_script, "analysis_snapshot")
    diff_mod = _load_module(diff_script, "analysis_snapshot_diff")

    root = tmp_path / "results"
    base = root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct"

    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=-0.10,
        eval_seconds=10.0,
        explain_elapsed=2.0,
    )

    baseline = snap_mod.build_snapshot(root, primary_method="ours", reference_method="gradient")

    # mutate run for current snapshot
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.11,
        suff=-0.09,
        eval_seconds=8.0,
        explain_elapsed=1.0,
    )
    current = snap_mod.build_snapshot(root, primary_method="ours", reference_method="gradient")

    diff = diff_mod.build_diff(baseline, current)
    assert diff["row_count"] > 0
    matched = [
        r for r in diff["rows"]
        if r["method"] == "ours" and r["metric"] == "timing.total_seconds"
    ]
    assert len(matched) == 1
    assert matched[0]["delta"] < 0.0


def test_analysis_snapshot_groups_by_eval_granularity(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_snapshot.py"
    mod = _load_module(script, "analysis_snapshot")

    root = tmp_path / "results"
    base = root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct"

    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours-token",
        seed=42,
        method="ours",
        comp=0.10,
        suff=-0.10,
        eval_seconds=10.0,
        explain_elapsed=2.0,
        eval_granularity="token",
    )
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-gradient-token",
        seed=42,
        method="gradient",
        comp=0.12,
        suff=0.20,
        eval_seconds=11.0,
        explain_elapsed=1.0,
        eval_granularity="token",
    )
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours-word",
        seed=42,
        method="ours",
        comp=0.10,
        suff=-0.10,
        eval_seconds=10.0,
        explain_elapsed=2.0,
        eval_granularity="word",
    )
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-gradient-word",
        seed=42,
        method="gradient",
        comp=0.12,
        suff=0.20,
        eval_seconds=11.0,
        explain_elapsed=1.0,
        eval_granularity="word",
    )

    snapshot = mod.build_snapshot(root, primary_method="ours", reference_method="gradient")
    assert snapshot["group_count"] == 2
    group_ids = [group["group_id"] for group in snapshot["groups"]]
    assert any("|eval=token|" in gid for gid in group_ids)
    assert any("|eval=word|" in gid for gid in group_ids)


def test_analysis_snapshot_legacy_missing_granularity_defaults_to_word(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_snapshot.py"
    mod = _load_module(script, "analysis_snapshot")

    root = tmp_path / "results"
    base = root / "eraser_movie_reviews" / "model-Qwen2_5-7B-Instruct"

    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-ours",
        seed=42,
        method="ours",
        comp=0.10,
        suff=-0.10,
        eval_seconds=10.0,
        explain_elapsed=2.0,
        eval_granularity=None,
    )
    _write_run(
        base / "chunk-sentence_search-greedy_k-8_lam-1-1-1-1_seed-42_method-gradient",
        seed=42,
        method="gradient",
        comp=0.12,
        suff=0.20,
        eval_seconds=11.0,
        explain_elapsed=1.0,
        eval_granularity="word",
    )

    snapshot = mod.build_snapshot(root, primary_method="ours", reference_method="gradient")
    assert snapshot["group_count"] == 1
    assert "|eval=word|" in snapshot["groups"][0]["group_id"]
