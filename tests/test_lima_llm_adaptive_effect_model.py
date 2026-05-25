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


def _write_sample(
    sample_dir: Path,
    *,
    sample_id: str,
    text: str,
    chunk_count: int,
    selected_ids: list[int],
    ranking: list[int],
    bucket: str,
    floor_applied: bool,
    guard_applied: bool,
    raw_count: int,
    final_count: int,
) -> None:
    chunks = []
    cursor = 0
    width = max(1, len(text) // max(1, chunk_count))
    for idx in range(chunk_count):
        start = cursor
        end = len(text) if idx == chunk_count - 1 else min(len(text), cursor + width)
        if end <= start:
            end = min(len(text), start + 1)
        chunks.append({"chunk_id": idx, "start_char": start, "end_char": end, "text": text[start:end]})
        cursor = end

    payload = {
        "sample_id": sample_id,
        "text": text,
        "chunks": chunks,
        "selected_chunk_ids": selected_ids,
        "chunk_ranking": ranking,
        "metadata": {
            "chunk_diagnostics": {
                "chunk_count": chunk_count,
                "fallback_applied": False,
                "adaptive_bucket": bucket,
                "adaptive_effective_floor_applied": floor_applied,
                "adaptive_fragmentation_guard_applied": guard_applied,
                "adaptive_stage_chunk_counts": {"raw": raw_count, "final": final_count},
            }
        },
    }
    (sample_dir / f"{sample_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_run(
    run_dir: Path,
    *,
    log_odds: float,
    comp: float,
    suff: float,
    aopc_c: float,
    aopc_s: float,
    runtime_seconds: float,
    adaptive_profile: str,
    samples: list[dict],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "metrics_primary": {
            "log_odds": log_odds,
            "comprehensiveness": comp,
            "sufficiency": suff,
            "aopc_comprehensiveness": aopc_c,
            "aopc_sufficiency": aopc_s,
        },
        "metrics_secondary": {"runtime_seconds": runtime_seconds},
        "provenance": {"git": {"commit": "abc123"}},
    }
    run_cfg = {"adaptive_profile": adaptive_profile}
    (run_dir / "eval_report.json").write_text(json.dumps(eval_report, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        _write_sample(sample_dir, **sample)


def test_adaptive_effect_model_builds_cross_root_summary(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_effect_model.py"
    mod = _load_module(script, "adaptive_effect_model")

    base_model = tmp_path / "baseline" / "demo_ds" / "model-Qwen2_5-7B-Instruct"
    cand_model = tmp_path / "candidate" / "demo_ds" / "model-Qwen2_5-7B-Instruct"
    run_name = "chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours"
    base_samples = [
        {
            "sample_id": "s1",
            "text": "a b c d e f g h i j k l.",
            "chunk_count": 1,
            "selected_ids": [0],
            "ranking": [0],
            "bucket": "short",
            "floor_applied": False,
            "guard_applied": False,
            "raw_count": 1,
            "final_count": 1,
        },
        {
            "sample_id": "s2",
            "text": "long " * 300,
            "chunk_count": 60,
            "selected_ids": [0, 1],
            "ranking": [0, 1, 2],
            "bucket": "very_long",
            "floor_applied": False,
            "guard_applied": False,
            "raw_count": 1,
            "final_count": 60,
        },
    ]
    cand_samples = [
        {
            "sample_id": "s1",
            "text": "a b c d e f g h i j k l.",
            "chunk_count": 5,
            "selected_ids": [1],
            "ranking": [1, 0, 2, 3, 4],
            "bucket": "short",
            "floor_applied": True,
            "guard_applied": False,
            "raw_count": 1,
            "final_count": 5,
        },
        {
            "sample_id": "s2",
            "text": "long " * 300,
            "chunk_count": 40,
            "selected_ids": [1, 2],
            "ranking": [1, 2, 0],
            "bucket": "very_long",
            "floor_applied": False,
            "guard_applied": True,
            "raw_count": 1,
            "final_count": 40,
        },
    ]
    _write_run(
        base_model / run_name,
        log_odds=-0.2,
        comp=0.10,
        suff=0.20,
        aopc_c=0.30,
        aopc_s=0.40,
        runtime_seconds=100.0,
        adaptive_profile="balanced",
        samples=base_samples,
    )
    _write_run(
        cand_model / run_name,
        log_odds=-0.3,
        comp=0.12,
        suff=0.18,
        aopc_c=0.33,
        aopc_s=0.35,
        runtime_seconds=90.0,
        adaptive_profile="aggressive",
        samples=cand_samples,
    )

    report = mod.build_effect_model(
        baseline_root=tmp_path / "baseline",
        candidate_root=tmp_path / "candidate",
    )
    assert len(report["datasets"]) == 1
    ds = report["datasets"][0]
    assert ds["dataset"] == "demo_ds"
    assert ds["metric_deltas"]["log_odds"]["directional_gain"] > 0.0
    assert ds["shape_summary"]["chunk_count_mean_delta"] < 0.0 or ds["shape_summary"]["chunk_count_mean_delta"] > 0.0
    assert ds["trigger_groups"]["floor_only"]["count"] == 1
    assert ds["trigger_groups"]["guard_only"]["count"] == 1
    rows = mod._flatten_effect_rows(report)
    assert len(rows) == 1
    assert rows[0]["trigger_floor_only_count"] == 1
    assert rows[0]["trigger_guard_only_count"] == 1
