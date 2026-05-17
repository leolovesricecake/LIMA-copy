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


def test_trace_component_profile_aggregates_steps(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "trace_component_profile.py"
    mod = _load_module(script, "trace_component_profile")

    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_a = {
        "sample_id": "a",
        "selected_chunk_ids": [0, 1],
        "trace": [
            {
                "step": 0,
                "marginal_gain": 1.0,
                "total_score": 1.0,
                "components": {
                    "confidence": 0.2,
                    "effectiveness": 0.1,
                    "consistency": 0.3,
                    "collaboration": 0.4,
                },
            },
            {
                "step": 1,
                "marginal_gain": 0.8,
                "total_score": 1.8,
                "components": {
                    "confidence": 0.25,
                    "effectiveness": 0.2,
                    "consistency": 0.35,
                    "collaboration": 0.45,
                },
            },
        ],
    }
    sample_b = {
        "sample_id": "b",
        "selected_chunk_ids": [2],
        "trace": [
            {
                "step": 0,
                "marginal_gain": 0.9,
                "total_score": 0.9,
                "components": {
                    "confidence": 0.3,
                    "effectiveness": 0.15,
                    "consistency": 0.25,
                    "collaboration": 0.35,
                },
            }
        ],
    }
    (sample_dir / "a.json").write_text(json.dumps(sample_a, ensure_ascii=False), encoding="utf-8")
    (sample_dir / "b.json").write_text(json.dumps(sample_b, ensure_ascii=False), encoding="utf-8")

    profile = mod.build_profile(run_dir)
    assert profile["sample_count"] == 2
    assert abs(profile["mean_selected_chunks"] - 1.5) <= 1e-12
    assert len(profile["step_profile"]) == 2

    step0 = profile["step_profile"][0]
    assert step0["step"] == 0
    assert step0["sample_rows"] == 2
    assert abs(step0["mean_total_score"] - 0.95) <= 1e-12
    assert abs(profile["final_component_means"]["confidence"] - 0.275) <= 1e-12
