from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_sample(path: Path, *, sample_id: str, conf: float, eff: float, cons: float, col: float) -> None:
    payload = {
        "sample_id": sample_id,
        "scores": {
            "confidence": conf,
            "effectiveness": eff,
            "consistency": cons,
            "collaboration": col,
        },
        "metadata": {
            "component_profile": {
                "component_enabled": {
                    "confidence": True,
                    "effectiveness": True,
                    "consistency": True,
                    "collaboration": True,
                },
                "singleton_components": {
                    "0": {
                        "confidence": conf,
                        "effectiveness": eff,
                        "consistency": cons,
                        "collaboration": col,
                        "total": conf + eff + cons + col,
                    },
                    "1": {
                        "confidence": conf + 0.1,
                        "effectiveness": eff + 0.1,
                        "consistency": cons + 0.1,
                        "collaboration": col + 0.1,
                        "total": conf + eff + cons + col + 0.4,
                    },
                },
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_component_correlation_full_builds_rows(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "component_correlation_full.py"
    mod = _load_module(script, "component_correlation_full")

    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    _write_sample(sample_dir / "s1.json", sample_id="s1", conf=0.1, eff=0.2, cons=0.3, col=0.4)
    _write_sample(sample_dir / "s2.json", sample_id="s2", conf=0.2, eff=0.1, cons=0.4, col=0.3)

    report = mod.build_report(run_dir)

    assert report["sample_count"] == 2
    assert report["chunk_row_count"] == 4
    assert report["sample_row_count"] == 2
    assert len(report["rows"]) == 12  # 6 pairs * 2 views

    chunk_pairs = report["views"]["chunk_singleton"]["pairs"]
    sample_pairs = report["views"]["sample_selected_set"]["pairs"]
    assert len(chunk_pairs) == 6
    assert len(sample_pairs) == 6
    for row in chunk_pairs + sample_pairs:
        assert "pearson" in row
        assert "spearman" in row
        assert int(row["n"]) >= 2
