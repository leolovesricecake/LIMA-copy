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


def _write_dataset_report(path: Path, dataset: str, score_a: float, score_b: float) -> None:
    payload = {
        "dataset": dataset,
        "ranked_dev": [
            {
                "trial_id": "dev_a",
                "status": "ok",
                "adaptive_overrides": {"min_effective_chunks": 5},
                "run_dir": f"/tmp/{dataset}/a",
                "score_summary": {
                    "score": score_a,
                    "quality_gain_sum": 0.1,
                    "runtime_gain_seconds": 1.0,
                    "major_regression_count": 0,
                },
                "drift_vs_dev_baseline": {
                    "selected_changed_ratio": 0.1,
                    "ranking_changed_ratio": 0.1,
                    "trace_total_score_max_abs_diff": 0.0,
                },
            },
            {
                "trial_id": "dev_b",
                "status": "ok",
                "adaptive_overrides": {"guard_mode": "soft_band"},
                "run_dir": f"/tmp/{dataset}/b",
                "score_summary": {
                    "score": score_b,
                    "quality_gain_sum": 0.2,
                    "runtime_gain_seconds": 0.0,
                    "major_regression_count": 0,
                },
                "drift_vs_dev_baseline": {
                    "selected_changed_ratio": 0.05,
                    "ranking_changed_ratio": 0.05,
                    "trace_total_score_max_abs_diff": 0.0,
                },
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_adaptive_hparam_report_aggregates_dataset_results(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_hparam_report.py"
    mod = _load_module(script, "adaptive_hparam_report")

    root = tmp_path / "search"
    _write_dataset_report(root / "search_trials" / "emotion" / "adaptive_hparam_search.json", "emotion", 1.0, 0.8)
    _write_dataset_report(root / "search_trials" / "imdb" / "adaptive_hparam_search.json", "imdb", 1.2, 0.9)

    paths = mod._discover_dataset_reports(root)
    assert len(paths) == 2

    rows = [mod._summarize_dataset(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    assert {row["dataset"] for row in rows} == {"emotion", "imdb"}
    assert all(row["status"] == "ok" for row in rows)
    assert all("adaptive_overrides_json" in row for row in rows)
