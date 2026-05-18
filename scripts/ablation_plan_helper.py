#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ABLATION_GRID: List[Tuple[str, str]] = [
    ("full", "1,1,1,1"),
    ("drop_conf", "0,1,1,1"),
    ("drop_eff", "1,0,1,1"),
    ("drop_cons", "1,1,0,1"),
    ("drop_col", "1,1,1,0"),
]

PHASE_B2_GRID: List[Tuple[str, str]] = [
    ("full", "1,1,1,1"),
    ("cand_a_drop_col_half", "1,1,1,0.5"),
    ("cand_b_drop_col_zero", "1,1,1,0"),
    ("cand_c_drop_cons_half", "1,1,0.5,1"),
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [" ", "\t", "\n", "\"", "'"]):
        return json.dumps(text)
    return text


def _flag_items(cfg: Dict[str, Any], key: str, cli_name: str) -> List[str]:
    value = cfg.get(key)
    if isinstance(value, bool):
        return [cli_name] if value else []
    if value is None:
        return []
    return [cli_name, str(value)]


def _build_run_dir(cfg: Dict[str, Any], lambdas: str) -> Path:
    output_dir = str(cfg.get("output_dir", "lima_llm_results"))
    dataset = str(cfg.get("dataset", ""))
    model_path = str(cfg.get("model_path", ""))
    chunker = str(cfg.get("chunker", "sentence"))
    search = str(cfg.get("search", "greedy"))
    k = int(cfg.get("k", 8))
    seed = int(cfg.get("seed", 42))
    explain_method = str(cfg.get("explain_method", "ours"))

    model_leaf = model_path.split("/")[-1].replace(".", "_")
    run_name = (
        f"chunk-{chunker}_search-{search}_k-{k}"
        f"_lam-{lambdas.replace(',', '-')}_seed-{seed}"
        f"_method-{explain_method}"
    )
    return Path(output_dir) / dataset / f"model-{model_leaf}" / run_name


def _build_base_cmd(cfg: Dict[str, Any], python_bin: str) -> List[str]:
    cmd: List[str] = [python_bin, "-m", "lima_llm.pipeline.run"]
    cmd.extend(_flag_items(cfg, "dataset", "--dataset"))
    cmd.extend(_flag_items(cfg, "split", "--split"))
    cmd.extend(_flag_items(cfg, "eraser_root", "--eraser-root"))
    cmd.extend(_flag_items(cfg, "sst2_source", "--sst2-source"))
    cmd.extend(_flag_items(cfg, "dataset_cache_dir", "--dataset-cache-dir"))
    cmd.extend(_flag_items(cfg, "model_path", "--model-path"))
    cmd.extend(_flag_items(cfg, "device", "--device"))
    cmd.extend(_flag_items(cfg, "dtype", "--dtype"))
    cmd.extend(_flag_items(cfg, "max_length", "--max-length"))
    cmd.extend(_flag_items(cfg, "embedding_layer_ratio", "--embedding-layer-ratio"))
    cmd.extend(_flag_items(cfg, "k", "--k"))
    cmd.extend(_flag_items(cfg, "chunker", "--chunker"))
    cmd.extend(_flag_items(cfg, "fixed_token_size", "--fixed-token-size"))
    cmd.extend(_flag_items(cfg, "search", "--search"))
    cmd.extend(_flag_items(cfg, "seed", "--seed"))
    cmd.extend(_flag_items(cfg, "max_samples", "--max-samples"))
    cmd.extend(_flag_items(cfg, "output_dir", "--output-dir"))
    cmd.extend(_flag_items(cfg, "resume_check", "--resume-check"))
    cmd.extend(_flag_items(cfg, "eval_q_values", "--eval-q-values"))
    cmd.extend(_flag_items(cfg, "eval_granularity", "--eval-granularity"))
    cmd.extend(_flag_items(cfg, "explain_method", "--explain-method"))

    for key, cli_name in [
        ("mock_backbone", "--mock-backbone"),
        ("deterministic", "--deterministic"),
        ("verbose_chunks", "--verbose-chunks"),
        ("run_eval", "--run-eval"),
    ]:
        if bool(cfg.get(key, False)):
            cmd.append(cli_name)

    return cmd


def _command_to_shell(cmd: List[str]) -> str:
    return " ".join(_quote(part) for part in cmd)


def _check_runs(run_dirs: List[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing = 0
    for run_dir in run_dirs:
        report = run_dir / "eval_report.json"
        sample_dir = run_dir / "samples"
        ok = report.exists() and sample_dir.exists() and any(sample_dir.glob("*.json"))
        if not ok:
            missing += 1
        rows.append(
            {
                "run_dir": str(run_dir),
                "exists": bool(run_dir.exists()),
                "has_eval_report": bool(report.exists()),
                "has_samples": bool(sample_dir.exists()),
                "sample_json_count": len(list(sample_dir.glob("*.json"))) if sample_dir.exists() else 0,
            }
        )
    return {"missing_count": int(missing), "rows": rows}


def _grid_for_plan_set(plan_set: str) -> List[Tuple[str, str]]:
    key = str(plan_set).strip().lower()
    if key == "phase_b2":
        return list(PHASE_B2_GRID)
    return list(ABLATION_GRID)


def build_plan(template_run_config: Path, python_bin: str, do_check: bool, plan_set: str = "loo") -> Dict[str, Any]:
    cfg = _read_json(template_run_config)
    base_cmd = _build_base_cmd(cfg, python_bin=python_bin)
    grid = _grid_for_plan_set(plan_set)

    entries: List[Dict[str, Any]] = []
    run_dirs: List[Path] = []
    for tag, lambdas in grid:
        cmd = list(base_cmd)
        cmd.extend(["--lambdas", lambdas])
        run_dir = _build_run_dir(cfg, lambdas=lambdas)
        run_dirs.append(run_dir)
        entries.append(
            {
                "tag": tag,
                "lambdas": lambdas,
                "run_dir": str(run_dir),
                "command": _command_to_shell(cmd),
            }
        )

    payload: Dict[str, Any] = {
        "template_run_config": str(template_run_config),
        "plan_set": str(plan_set),
        "grid": [{"tag": tag, "lambdas": lambdas} for tag, lambdas in grid],
        "entries": entries,
    }

    if do_check:
        payload["check"] = _check_runs(run_dirs)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standard commands and expected run dirs for full + leave-one-out ablation runs"
    )
    parser.add_argument("--template-run-config", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--plan-set", type=str, default="loo", choices=["loo", "phase_b2"])
    parser.add_argument("--check", action="store_true", help="check whether generated run dirs already contain results")
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    template_run_config = Path(args.template_run_config)
    payload = build_plan(
        template_run_config=template_run_config,
        python_bin=str(args.python_bin),
        do_check=bool(args.check),
        plan_set=str(args.plan_set),
    )

    out_json = Path(args.output_json) if args.output_json else (template_run_config.parent / "ablation_plan_helper.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ablation-plan] entries={len(payload.get('entries', []))}")
    print(f"[ablation-plan] json={out_json}")
    for row in payload.get("entries", []):
        print(f"[{row['tag']}] {row['command']}")
    if "check" in payload:
        print(f"[ablation-plan] missing_count={payload['check'].get('missing_count', 0)}")


if __name__ == "__main__":
    main()
