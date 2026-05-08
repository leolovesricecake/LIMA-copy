import json
from pathlib import Path

from lima_llm.eval.equivalence import compare_run_dirs


def _write_sample(run_dir: Path, sample_id: str, selected_ids, score_delta: float = 0.0) -> None:
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "selected_chunk_ids": list(selected_ids),
        "scores": {
            "total": 1.0 + score_delta,
            "confidence": 0.4 + score_delta,
            "effectiveness": 0.1,
            "consistency": 0.3,
            "collaboration": 0.2,
        },
        "trace": [
            {
                "step": 0,
                "selected_chunk_id": int(selected_ids[0]),
                "marginal_gain": 0.4 + score_delta,
                "total_score": 0.4 + score_delta,
                "components": {
                    "confidence": 0.2,
                    "effectiveness": 0.1,
                    "consistency": 0.05,
                    "collaboration": 0.05,
                },
            }
        ],
    }
    (sample_dir / f"{sample_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_eval_report(run_dir: Path, runtime_seconds: float, predict_calls: int) -> None:
    report = {
        "sample_count": 1,
        "metrics_primary": {"accuracy_full": 1.0, "comprehensiveness": 0.1},
        "metrics_secondary": {
            "runtime_seconds": runtime_seconds,
            "forward_counters_delta": {"predict_calls": predict_calls, "embed_calls": 0, "gradient_calls": 0},
            "diagnosticity_vs_random": 0.5,
        },
        "metrics_by_target": {
            "gold": {"metrics_primary": {"comprehensiveness": 0.1, "sufficiency": 0.0}},
            "predicted": {"metrics_primary": {"comprehensiveness": 0.1, "sufficiency": 0.0}},
        },
    }
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_compare_run_dirs_passes_when_only_runtime_and_counters_differ(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_sample(left, "s1", selected_ids=[1], score_delta=0.0)
    _write_sample(right, "s1", selected_ids=[1], score_delta=0.0)
    _write_eval_report(left, runtime_seconds=10.0, predict_calls=123)
    _write_eval_report(right, runtime_seconds=22.0, predict_calls=456)

    result = compare_run_dirs(left, right, tolerance=1e-6)
    assert result["passed"] is True
    assert result["sample_check"]["passed"] is True
    assert result["report_check"]["passed"] is True


def test_compare_run_dirs_fails_on_selected_chunk_ids_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_sample(left, "s1", selected_ids=[1], score_delta=0.0)
    _write_sample(right, "s1", selected_ids=[0], score_delta=0.0)
    _write_eval_report(left, runtime_seconds=10.0, predict_calls=123)
    _write_eval_report(right, runtime_seconds=10.0, predict_calls=123)

    result = compare_run_dirs(left, right, tolerance=1e-6)
    assert result["passed"] is False
    assert result["sample_check"]["passed"] is False
    assert result["sample_check"]["first_failure"]["reason"] == "selected_chunk_ids_mismatch"

