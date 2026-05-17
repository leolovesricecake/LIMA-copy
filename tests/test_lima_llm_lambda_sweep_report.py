from __future__ import annotations

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
    lambdas: str,
    log_odds: float,
    comp: float,
    suff: float,
    a_c: float,
    a_s: float,
    eval_seconds: float,
    explain_seconds: float,
    selected_ids: list[int],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "eraser_movie_reviews",
        "split": "validation",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "chunker": "sentence",
        "search": "greedy",
        "k": 8,
        "lambdas": lambdas,
        "seed": 42,
        "deterministic": True,
        "device": "cuda:0",
    }
    report = {
        "metrics_secondary": {
            "runtime_seconds": eval_seconds,
            "forward_counters_delta": {"oom_shrink_events": 0},
            "method_diagnostics": {"failed_samples": 0},
        },
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": log_odds,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "aopc_comprehensiveness": a_c,
                    "aopc_sufficiency": a_s,
                }
            }
        },
        "provenance": {"git": {"commit_short": "abc123"}},
    }
    sample = {
        "sample_id": "s1",
        "selected_chunk_ids": selected_ids,
        "chunk_ranking": [*selected_ids, 9, 10],
        "trace": [
            {"total_score": 0.1},
            {"total_score": 0.2},
        ],
        "metadata": {"elapsed_seconds": explain_seconds},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False), encoding="utf-8")
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "s1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_lambda_sweep_report_direction_and_stability(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "lambda_sweep_report.py"
    mod = _load_module(script, "lambda_sweep_report")

    baseline = tmp_path / "baseline"
    cand_good = tmp_path / "cand_good"
    cand_bad = tmp_path / "cand_bad"

    _write_run(
        baseline,
        lambdas="1,1,1,1",
        log_odds=0.10,
        comp=0.10,
        suff=-0.10,
        a_c=0.10,
        a_s=0.20,
        eval_seconds=100.0,
        explain_seconds=10.0,
        selected_ids=[0, 1],
    )
    _write_run(
        cand_good,
        lambdas="1.1,1,1,0.9",
        log_odds=0.12,
        comp=0.12,
        suff=-0.11,
        a_c=0.11,
        a_s=0.20,
        eval_seconds=95.0,
        explain_seconds=9.0,
        selected_ids=[0, 1],
    )
    _write_run(
        cand_bad,
        lambdas="0.8,1,1.2,1",
        log_odds=0.09,
        comp=0.11,
        suff=-0.09,
        a_c=0.09,
        a_s=0.205,
        eval_seconds=105.0,
        explain_seconds=12.0,
        selected_ids=[1, 2],
    )

    row_good = mod._build_row(baseline, cand_good, metric_tol=5e-4, trace_tol=1e-8)
    row_bad = mod._build_row(baseline, cand_bad, metric_tol=5e-4, trace_tol=1e-8)

    assert row_good["faithfulness_pass"] is True
    assert row_good["improved_count"] >= 4
    assert row_good["stability"]["selected_changed_ratio"] == 0.0

    assert row_bad["faithfulness_pass"] is False
    assert row_bad["regressed_count"] >= 1
    assert row_bad["stability"]["selected_changed_ratio"] == 1.0
