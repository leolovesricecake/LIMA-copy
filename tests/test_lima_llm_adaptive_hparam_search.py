from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage1_grid_default_shape() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "search_adaptive_hparams.py"
    mod = _load_module(script, "search_adaptive_hparams")

    grid = mod._default_stage1_grid()
    assert len(grid) == 12
    assert grid[0] == {}
    assert any(str(row.get("guard_mode", "")) == "soft_band" for row in grid)


def test_stage2_random_builds_requested_count() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "search_adaptive_hparams.py"
    mod = _load_module(script, "search_adaptive_hparams_stage2")

    space = mod._space_defaults()
    top_rows = [
        {"adaptive_overrides": {}},
        {"adaptive_overrides": {"min_effective_chunks": 6}},
        {"adaptive_overrides": {"guard_mode": "soft_band"}},
    ]
    rows = mod._build_stage2_random(top_rows=top_rows, n_random=8, rng=mod.random.Random(42), space=space)
    assert len(rows) == 8
    assert all(isinstance(row, dict) for row in rows)


def test_score_trial_penalizes_long_text_regressions() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "search_adaptive_hparams.py"
    mod = _load_module(script, "search_adaptive_hparams_score")

    baseline = {
        "log_odds": -0.10,
        "comprehensiveness": 0.10,
        "sufficiency": 0.20,
        "aopc_comprehensiveness": 0.30,
        "aopc_sufficiency": 0.40,
        "runtime_seconds": 100.0,
    }
    worse = {
        "log_odds": -0.09,
        "comprehensiveness": 0.09,
        "sufficiency": 0.21,
        "aopc_comprehensiveness": 0.29,
        "aopc_sufficiency": 0.41,
        "runtime_seconds": 90.0,
    }
    short_ds = mod._score_trial(
        dataset="emotion",
        baseline_metrics=baseline,
        candidate_metrics=worse,
        metric_tol=5e-4,
        runtime_weight=1e-4,
    )
    long_ds = mod._score_trial(
        dataset="imdb",
        baseline_metrics=baseline,
        candidate_metrics=worse,
        metric_tol=5e-4,
        runtime_weight=1e-4,
    )
    assert int(short_ds["major_regression_count"]) >= 1
    assert int(long_ds["major_regression_count"]) >= 1
    assert float(long_ds["score"]) < float(short_ds["score"])
