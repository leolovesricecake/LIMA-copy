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


def test_ablation_plan_helper_generates_grid_and_commands(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "ablation_plan_helper.py"
    mod = _load_module(script, "ablation_plan_helper")

    cfg = {
        "dataset": "eraser_movie_reviews",
        "split": "validation",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_length": 2048,
        "embedding_layer_ratio": 0.7,
        "k": 8,
        "chunker": "sentence",
        "fixed_token_size": 64,
        "search": "greedy",
        "seed": 42,
        "deterministic": True,
        "max_samples": 20,
        "output_dir": str(tmp_path / "runs"),
        "resume_check": "strict",
        "run_eval": True,
        "eval_q_values": "1,5,10,20,50",
        "eval_granularity": "token",
        "explain_method": "ours",
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    payload = mod.build_plan(cfg_path, python_bin="python", do_check=False)
    assert len(payload["entries"]) == 5

    tags = [row["tag"] for row in payload["entries"]]
    assert tags == ["full", "drop_conf", "drop_eff", "drop_cons", "drop_col"]

    commands = [row["command"] for row in payload["entries"]]
    assert any("--lambdas 1,1,1,1" in cmd for cmd in commands)
    assert any("--lambdas 0,1,1,1" in cmd for cmd in commands)

    for row in payload["entries"]:
        assert "chunk-sentence_search-greedy_k-8" in row["run_dir"]


def test_ablation_plan_helper_phase_b2_grid(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "ablation_plan_helper.py"
    mod = _load_module(script, "ablation_plan_helper_phase_b2")

    cfg = {
        "dataset": "eraser_movie_reviews",
        "split": "validation",
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cuda:0",
        "k": 8,
        "chunker": "sentence",
        "search": "greedy",
        "seed": 42,
        "output_dir": str(tmp_path / "runs"),
        "explain_method": "ours",
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    payload = mod.build_plan(cfg_path, python_bin="python", do_check=False, plan_set="phase_b2")
    assert payload["plan_set"] == "phase_b2"
    tags = [row["tag"] for row in payload["entries"]]
    assert tags == ["full", "cand_a_drop_col_half", "cand_b_drop_col_zero", "cand_c_drop_cons_half"]
    commands = [row["command"] for row in payload["entries"]]
    assert any("--lambdas 1,1,1,0.5" in cmd for cmd in commands)
    assert any("--lambdas 1,1,1,0" in cmd for cmd in commands)
