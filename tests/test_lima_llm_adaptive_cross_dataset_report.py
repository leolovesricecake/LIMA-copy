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
    runtime_seconds: float,
    log_odds: float,
    comp: float,
    suff: float,
    aopc_c: float,
    aopc_s: float,
    chunk_count: int,
    fallback_applied: bool,
    bucket: str | None,
    raw_count: int | None,
    final_count: int | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "metrics_secondary": {"runtime_seconds": runtime_seconds},
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": log_odds,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "aopc_comprehensiveness": aopc_c,
                    "aopc_sufficiency": aopc_s,
                }
            }
        },
        "provenance": {"git": {"commit": "abc123", "dirty": False}},
    }
    run_cfg = {"deterministic": True, "split": "validation", "max_samples": 1}

    text = " ".join(f"w{i}" for i in range(max(1, chunk_count * 3)))
    chunks = []
    cur = 0
    for idx in range(chunk_count):
        nxt = cur + 3
        chunks.append({"chunk_id": idx, "start_char": cur, "end_char": nxt, "text": text[cur:nxt]})
        cur = nxt
    if chunks:
        chunks[-1]["end_char"] = len(text)
        chunks[-1]["text"] = text[chunks[-1]["start_char"] :]

    diag = {
        "fallback_applied": fallback_applied,
        "chunk_count": chunk_count,
    }
    if bucket is not None:
        diag["adaptive_bucket"] = bucket
    if raw_count is not None or final_count is not None:
        diag["adaptive_stage_chunk_counts"] = {
            "raw": raw_count or 0,
            "final": final_count or chunk_count,
        }

    sample = {
        "sample_id": "s1",
        "text": text,
        "chunks": chunks,
        "metadata": {"chunk_diagnostics": diag},
    }

    (run_dir / "eval_report.json").write_text(json.dumps(eval_report, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "s1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_adaptive_cross_dataset_report_builds_pairwise_rows(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_cross_dataset_report.py"
    mod = _load_module(script, "adaptive_cross_dataset_report")

    model_dir = tmp_path / "results" / "demo_ds" / "model-Qwen2_5-7B-Instruct"
    _write_run(
        model_dir / "chunk-sentence_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours",
        runtime_seconds=10.0,
        log_odds=-0.1,
        comp=0.1,
        suff=0.2,
        aopc_c=0.3,
        aopc_s=0.4,
        chunk_count=1,
        fallback_applied=True,
        bucket=None,
        raw_count=None,
        final_count=None,
    )
    _write_run(
        model_dir / "chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours",
        runtime_seconds=8.0,
        log_odds=-0.2,
        comp=0.2,
        suff=0.1,
        aopc_c=0.4,
        aopc_s=0.3,
        chunk_count=55,
        fallback_applied=False,
        bucket="very_long",
        raw_count=1,
        final_count=55,
    )

    report = mod.build_report(tmp_path / "results")
    assert len(report["datasets"]) == 1
    ds = report["datasets"][0]
    assert ds["dataset"] == "demo_ds"
    assert "adaptive_vs_sentence" in ds["pairwise"]
    pair = ds["pairwise"]["adaptive_vs_sentence"]
    assert pair["metric_deltas"]["runtime_seconds"]["directional_gain"] == 2.0
    assert pair["sample_stats_delta"]["top20_count_zero_ratio_delta"] < 0.0
    assert pair["sample_stats_delta"]["very_long_raw_single_ratio_delta"] >= 0.0

    rows = mod._flatten_pairwise_rows(report)
    assert len(rows) == 1
    assert rows[0]["pair"] == "adaptive_vs_sentence"


def test_adaptive_cross_dataset_report_cross_root_mode(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_cross_dataset_report.py"
    mod = _load_module(script, "adaptive_cross_dataset_report_cross")

    baseline_model = tmp_path / "baseline" / "demo_ds" / "model-Qwen2_5-7B-Instruct"
    candidate_model = tmp_path / "candidate" / "demo_ds" / "model-Qwen2_5-7B-Instruct"
    _write_run(
        baseline_model / "chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours",
        runtime_seconds=10.0,
        log_odds=-0.10,
        comp=0.10,
        suff=0.20,
        aopc_c=0.30,
        aopc_s=0.40,
        chunk_count=5,
        fallback_applied=False,
        bucket="short",
        raw_count=2,
        final_count=5,
    )
    _write_run(
        candidate_model / "chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours",
        runtime_seconds=8.0,
        log_odds=-0.20,
        comp=0.20,
        suff=0.10,
        aopc_c=0.40,
        aopc_s=0.30,
        chunk_count=6,
        fallback_applied=False,
        bucket="short",
        raw_count=2,
        final_count=6,
    )

    report = mod.build_cross_root_report(
        baseline_root=tmp_path / "baseline",
        candidate_root=tmp_path / "candidate",
    )
    assert report["report_mode"] == "cross_root"
    assert len(report["datasets"]) == 1
    pair = report["datasets"][0]["pairwise"]["adaptive_candidate_vs_adaptive_baseline"]
    assert pair["metric_deltas"]["runtime_seconds"]["directional_gain"] == 2.0
    rows = mod._flatten_pairwise_rows(report)
    assert len(rows) == 1
    assert rows[0]["pair"] == "adaptive_candidate_vs_adaptive_baseline"
