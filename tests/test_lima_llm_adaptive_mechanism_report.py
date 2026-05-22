from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_text_chunks(segment_texts: List[str]) -> tuple[str, List[Dict]]:
    text = ""
    chunks: List[Dict] = []
    cursor = 0
    for idx, seg in enumerate(segment_texts):
        start = cursor
        text += seg
        cursor += len(seg)
        chunks.append({"chunk_id": idx, "start_char": start, "end_char": cursor, "text": seg})
    return text, chunks


def _write_run(
    run_dir: Path,
    *,
    metrics: Dict[str, float],
    runtime_seconds: float,
    selected_ids: List[int],
    segment_texts: List[str],
    chunk_diagnostics: Dict,
    commit_short: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_report = {
        "metrics_secondary": {"runtime_seconds": runtime_seconds},
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": metrics["log_odds"],
                    "comprehensiveness": metrics["comprehensiveness"],
                    "sufficiency": metrics["sufficiency"],
                    "aopc": metrics["aopc"],
                    "aopc_comprehensiveness": metrics["aopc_comprehensiveness"],
                    "aopc_sufficiency": metrics["aopc_sufficiency"],
                }
            }
        },
        "provenance": {
            "git": {"commit_short": commit_short, "is_dirty": False},
            "environment": {"deterministic": {"enabled": True}},
        },
    }
    run_cfg = {"device": "cuda:0", "deterministic": True}

    text, chunks = _build_text_chunks(segment_texts)
    sample = {
        "sample_id": "sample-1",
        "text": text,
        "chunks": chunks,
        "selected_chunk_ids": list(selected_ids),
        "chunk_ranking": list(selected_ids) + [c["chunk_id"] for c in chunks if c["chunk_id"] not in selected_ids],
        "metadata": {
            "chunk_diagnostics": chunk_diagnostics,
        },
    }

    (run_dir / "eval_report.json").write_text(json.dumps(eval_report, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "sample-1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_adaptive_mechanism_report_builds_expected_sections(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_mechanism_report.py"
    mod = _load_module(script, "adaptive_mechanism_report")

    sentence_dir = tmp_path / "sentence"
    sentence_v2_dir = tmp_path / "sentence_v2"
    adaptive_dir = tmp_path / "adaptive"

    _write_run(
        sentence_dir,
        metrics={
            "log_odds": -0.20,
            "comprehensiveness": 0.10,
            "sufficiency": -0.10,
            "aopc": 0.20,
            "aopc_comprehensiveness": 0.06,
            "aopc_sufficiency": -0.03,
        },
        runtime_seconds=100.0,
        selected_ids=[0, 1],
        segment_texts=["s%02d. " % i for i in range(10)],
        chunk_diagnostics={"chunk_strategy": "sentence"},
        commit_short="sent123",
    )

    _write_run(
        sentence_v2_dir,
        metrics={
            "log_odds": -0.19,
            "comprehensiveness": 0.11,
            "sufficiency": -0.11,
            "aopc": 0.21,
            "aopc_comprehensiveness": 0.061,
            "aopc_sufficiency": -0.031,
        },
        runtime_seconds=90.0,
        selected_ids=[0, 1],
        segment_texts=["v%02d. " % i for i in range(9)],
        chunk_diagnostics={"chunk_strategy": "sentence_v2"},
        commit_short="sv2123",
    )

    _write_run(
        adaptive_dir,
        metrics={
            "log_odds": -0.25,
            "comprehensiveness": 0.12,
            "sufficiency": -0.08,
            "aopc": 0.22,
            "aopc_comprehensiveness": 0.07,
            "aopc_sufficiency": -0.02,
        },
        runtime_seconds=80.0,
        selected_ids=[0, 2],
        segment_texts=["a%02d. " % i for i in range(15)],
        chunk_diagnostics={
            "chunk_strategy": "adaptive",
            "adaptive_enabled": True,
            "adaptive_bucket": "very_long",
            "adaptive_postprocess": {
                "invalid_merge_count": 1,
                "short_merge_count": 1,
                "long_split_count": 1,
                "post_long_invalid_merge_count": 1,
                "post_long_short_merge_count": 1,
                "adjacent_pack_merge_count": 2,
            },
            "adaptive_stage_chunk_counts": {
                "raw": 1,
                "after_invalid": 1,
                "after_short": 1,
                "after_long": 6,
                "after_post_long_merge": 5,
                "final": 4,
            },
        },
        commit_short="adp123",
    )

    report = mod.build_report(sentence_dir=sentence_dir, sentence_v2_dir=sentence_v2_dir, adaptive_dir=adaptive_dir)
    assert "runs" in report
    assert "comparisons" in report
    assert "adaptive_vs_sentence" in report["comparisons"]

    adaptive_summary = report["runs"]["adaptive"]
    assert adaptive_summary["shape"]["chunk_count_mean"] == 15.0
    assert adaptive_summary["adaptive"]["very_long"]["raw_eq_one_ratio"] == 1.0
    assert adaptive_summary["adaptive"]["very_long"]["fragmentation_ratio_mean"] == 4.0

    cmp_sentence = report["comparisons"]["adaptive_vs_sentence"]
    assert cmp_sentence["metric_delta"]["log_odds"]["delta"] == pytest.approx(-0.05, abs=1e-12)
    assert cmp_sentence["top20_count_drift"]["top20_count_delta_mean"] == 1.0
    assert cmp_sentence["runtime"]["improve_ratio"] == pytest.approx(0.2, abs=1e-12)
