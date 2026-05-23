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
    run_dir: Path,
    *,
    sample_id: str,
    chunk_count: int,
    selected_ids: list[int],
    ranking: list[int],
    trace_scores: list[float],
    bucket: str | None = None,
    word_count: int = 0,
    raw_count: int | None = None,
    final_count: int | None = None,
    fallback_reason: str | None = None,
    floor_applied: bool = False,
    guard_applied: bool = False,
) -> None:
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text = " ".join(f"w{i}" for i in range(max(2, word_count)))
    chunks = []
    cursor = 0
    width = max(1, len(text) // max(1, chunk_count))
    for idx in range(chunk_count):
        start = cursor
        end = len(text) if idx == chunk_count - 1 else min(len(text), cursor + width)
        if end <= start:
            end = min(len(text), start + 1)
        chunks.append(
            {
                "chunk_id": idx,
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )
        cursor = end

    diag = {
        "chunk_count": chunk_count,
        "fallback_applied": bool(fallback_reason),
        "fallback_reason": fallback_reason,
    }
    if bucket is not None:
        diag["adaptive_bucket"] = bucket
        diag["adaptive_features"] = {"word_count": word_count}
    if raw_count is not None or final_count is not None:
        diag["adaptive_stage_chunk_counts"] = {
            "raw": raw_count if raw_count is not None else chunk_count,
            "final": final_count if final_count is not None else chunk_count,
        }
    diag["adaptive_effective_floor_applied"] = bool(floor_applied)
    diag["adaptive_fragmentation_guard_applied"] = bool(guard_applied)

    payload = {
        "sample_id": sample_id,
        "text": text,
        "chunks": chunks,
        "selected_chunk_ids": selected_ids,
        "chunk_ranking": ranking,
        "trace": [{"total_score": x} for x in trace_scores],
        "metadata": {"chunk_diagnostics": diag},
    }
    (sample_dir / f"{sample_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_adaptive_case_miner_detects_fragmentation_and_short_singleton(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_case_miner.py"
    mod = _load_module(script, "adaptive_case_miner")

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"

    _write_sample(
        baseline,
        sample_id="s1",
        chunk_count=6,
        selected_ids=[0, 1],
        ranking=[0, 1, 2, 3, 4, 5],
        trace_scores=[0.1, 0.2],
    )
    _write_sample(
        candidate,
        sample_id="s1",
        chunk_count=60,
        selected_ids=[0, 2],
        ranking=list(range(60)),
        trace_scores=[0.8, 1.0],
        bucket="very_long",
        word_count=1400,
        raw_count=1,
        final_count=60,
        guard_applied=True,
    )

    _write_sample(
        baseline,
        sample_id="s2",
        chunk_count=3,
        selected_ids=[0],
        ranking=[0, 1, 2],
        trace_scores=[0.2],
    )
    _write_sample(
        candidate,
        sample_id="s2",
        chunk_count=1,
        selected_ids=[0],
        ranking=[0],
        trace_scores=[0.2],
        bucket="short",
        word_count=20,
        raw_count=1,
        final_count=1,
        fallback_reason="single_chunk_fallback",
    )

    report = mod.build_case_report(
        baseline_run_dir=baseline,
        candidate_run_dir=candidate,
        top_k=10,
    )

    assert report["sample_count_common"] == 2
    assert report["summary"]["very_long_fragmentation_risk_count"] == 1
    assert report["summary"]["short_singleton_risk_count"] == 1
    assert report["summary"]["trigger_counts"]["guard_only"] == 1
    assert report["summary"]["trigger_counts"]["none"] == 1
    assert len(report["top_cases"]) == 2
    top_ids = {row["sample_id"] for row in report["top_cases"]}
    assert {"s1", "s2"} == top_ids
    assert "trigger_ranked_cases" in report
    assert report["top_cases"][0]["trigger_kind"] in {"guard_only", "none"}
