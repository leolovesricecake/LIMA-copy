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
    comp: float,
    suff: float,
    runtime_seconds: float,
    selected_ids: list[int],
    orphan_chunks: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics_secondary": {"runtime_seconds": runtime_seconds},
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": 0.1,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "aopc": 0.3,
                    "aopc_comprehensiveness": 0.1,
                    "aopc_sufficiency": 0.2,
                }
            }
        },
        "provenance": {"git": {"commit_short": "abc123"}},
    }
    cfg = {"device": "cuda:0", "deterministic": True}
    sample = {
        "sample_id": "s1",
        "selected_chunk_ids": selected_ids,
        "chunk_ranking": [*selected_ids, 99],
        "trace": [{"total_score": 0.1}, {"total_score": 0.2}],
        "metadata": {
            "elapsed_seconds": 1.0,
            "chunk_diagnostics": {
                "chunk_strategy": "sentence_v2",
                "fallback_applied": False,
                "singleton_orphan_punctuation_chunks": orphan_chunks,
                "cross_newline_boundary_chunks": 0,
                "chunk_count": 2,
                "chunk_len_chars_mean": 20.0,
            },
        },
    }
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "s1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_phase_a_chunking_compare_builds_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase_a_chunking_compare.py"
    mod = _load_module(script, "phase_a_chunking_compare")

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        comp=0.1,
        suff=-0.1,
        runtime_seconds=10.0,
        selected_ids=[0, 1],
        orphan_chunks=1,
    )
    _write_run(
        candidate,
        comp=0.1,
        suff=-0.1,
        runtime_seconds=9.0,
        selected_ids=[0, 1],
        orphan_chunks=0,
    )

    report = mod.build_report(baseline, candidate, trace_tolerance=1e-8)
    assert report["metric_deltas"]["comprehensiveness"]["abs_diff"] == 0.0
    assert report["explanation_drift"]["selected_changed_ratio"] == 0.0
    assert report["chunk_diagnostics"]["baseline"]["orphan_chunks_mean"] == 1.0
    assert report["chunk_diagnostics"]["candidate"]["orphan_chunks_mean"] == 0.0
