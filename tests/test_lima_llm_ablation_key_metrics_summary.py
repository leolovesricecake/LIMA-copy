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


def _write_run(run_dir: Path, *, lambdas: str, log_odds: float, runtime_seconds: float, conf_calls: int, conf_skip: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics_secondary": {"runtime_seconds": runtime_seconds},
        "metrics_by_target": {
            "gold": {
                "metrics_primary": {
                    "log_odds": log_odds,
                    "comprehensiveness": 0.1,
                    "sufficiency": -0.1,
                    "aopc_comprehensiveness": 0.2,
                    "aopc_sufficiency": -0.2,
                }
            }
        },
        "provenance": {"git": {"commit_short": "abc123"}},
    }
    cfg = {
        "lambdas": lambdas,
        "device": "cuda:0",
        "deterministic": True,
    }
    sample = {
        "sample_id": "s1",
        "selected_chunk_ids": [0, 1],
        "chunk_ranking": [0, 1, 2],
        "scores": {
            "confidence": 0.8 if conf_calls > 0 else 0.0,
            "effectiveness": 0.1,
            "consistency": 0.2,
            "collaboration": 0.05,
            "total": (0.8 if conf_calls > 0 else 0.0) + 0.1 + 0.2 + 0.05,
        },
        "metadata": {
            "elapsed_seconds": 2.0,
            "component_profile": {
                "component_enabled": {
                    "confidence": conf_calls > 0,
                    "effectiveness": True,
                    "consistency": True,
                    "collaboration": True,
                },
                "singleton_components": {
                    "0": {
                        "confidence": 0.8 if conf_calls > 0 else 0.0,
                        "effectiveness": 0.0,
                        "consistency": 0.4,
                        "collaboration": 0.2,
                        "total": (0.8 if conf_calls > 0 else 0.0) + 0.0 + 0.4 + 0.2,
                    }
                },
            },
            "objective_compute_stats": {
                "evaluate_gains_calls": 1,
                "subset_cache_hit_rate": 0.3,
                "prob_cache_hit_rate": 0.5,
                "embed_cache_hit_rate": 0.6,
                "confidence_compute_calls": conf_calls,
                "effectiveness_compute_calls": 3,
                "consistency_compute_calls": 3,
                "collaboration_compute_calls": 3,
                "confidence_skipped_due_to_zero_lambda": conf_skip,
                "effectiveness_skipped_due_to_zero_lambda": 0,
                "consistency_skipped_due_to_zero_lambda": 0,
                "collaboration_skipped_due_to_zero_lambda": 0,
                "component_enabled": {
                    "confidence": conf_calls > 0,
                    "effectiveness": True,
                    "consistency": True,
                    "collaboration": True,
                },
                "timing": {
                    "text_build_seconds": 0.1,
                    "prefetch_seconds": 0.2,
                    "component_compute_seconds": 0.3,
                    "confidence_compute_seconds": 0.01,
                    "effectiveness_compute_seconds": 0.02,
                    "consistency_compute_seconds": 0.03,
                    "collaboration_compute_seconds": 0.04,
                },
            },
        },
    }

    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "s1.json").write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def test_ablation_summary_reports_zero_lambda_skip_check(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "ablation_key_metrics_summary.py"
    mod = _load_module(script, "ablation_key_metrics_summary")

    full_run = tmp_path / "full"
    drop_conf_run = tmp_path / "drop_conf"

    _write_run(full_run, lambdas="1,1,1,1", log_odds=0.4, runtime_seconds=10.0, conf_calls=3, conf_skip=0)
    _write_run(drop_conf_run, lambdas="0,1,1,1", log_odds=0.3, runtime_seconds=9.0, conf_calls=0, conf_skip=3)

    report = mod.build_report(full_run, [drop_conf_run])
    assert len(report["rows"]) == 2
    assert report["metric_directions"]["log_odds"] == "lower_is_better"

    ablation_row = report["rows"][1]
    assert ablation_row["disabled_components"] == ["confidence"]
    checks = ablation_row["zero_lambda_skip_checks"]
    assert checks["confidence"]["disabled"] is True
    assert checks["confidence"]["pass"] is True
    assert checks["all_disabled_components_pass"] is True
    assert ablation_row["quality_first_pass"] is False
    assert ablation_row["speed_first_pass"] is False
    assert "singleton effectiveness=0 is expected by definition when subset size<=1" in ablation_row["analysis_notes"]
    scale = ablation_row["score_scale_summary"]
    assert float(scale["means"]["total_mean"]) > 0.0
    assert float(scale["share_by_mean"]["confidence_share_by_mean"]) == 0.0

    gain = ablation_row["directional_gain_vs_full"]
    assert gain["log_odds"] > 0.0
    assert ablation_row["raw_delta_vs_full"]["log_odds"] < 0.0
