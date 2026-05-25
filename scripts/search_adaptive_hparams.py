#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


_DIRECTION_SIGNS = {
    "log_odds": -1.0,  # implementation: log(p_perturbed)-log(p_full), lower is better
    "comprehensiveness": 1.0,
    "sufficiency": -1.0,
    "aopc_comprehensiveness": 1.0,
    "aopc_sufficiency": -1.0,
}

_DEFAULT_SPLIT_BY_DATASET = {
    "eraser_movie_reviews": "validation",
    "imdb": "test",
    "rotten_tomatoes": "validation",
    "emotion": "validation",
    "sst2": "validation",
}

_LONG_TEXT_DATASETS = {"eraser_movie_reviews", "imdb"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_datasets(raw: str) -> List[str]:
    rows = [part.strip() for part in str(raw).split(",")]
    return [row for row in rows if row]


def _parse_space_json(raw: str | None) -> Dict[str, Any] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("{"):
        obj = json.loads(text)
    else:
        obj = _read_json(Path(text))
    if not isinstance(obj, dict):
        raise ValueError("--space-json must be a JSON object")
    return dict(obj)


def _default_stage1_grid() -> List[Dict[str, Any]]:
    # 12-trial low-budget grid, including baseline {}.
    return [
        {},
        {"min_effective_chunks": 4},
        {"min_effective_chunks": 6},
        {"short_floor_min_words": 12},
        {"short_floor_min_words": 18},
        {"short_floor_signal_mode": "structural"},
        {"min_effective_chunks": 6, "short_floor_min_words": 12, "short_floor_signal_mode": "always"},
        {
            "guard_mode": "soft_band",
            "fragmentation_target_words": 32,
            "fragmentation_target_min_chunks": 24,
            "fragmentation_target_max_chunks": 96,
        },
        {
            "guard_mode": "soft_band",
            "fragmentation_target_words": 28,
            "fragmentation_target_min_chunks": 24,
            "fragmentation_target_max_chunks": 96,
        },
        {"guard_mode": "hard_cap", "ratio_threshold": 18, "fragmentation_target_words": 32},
        {"guard_mode": "hard_cap", "ratio_threshold": 28, "fragmentation_target_words": 32},
        {
            "guard_mode": "soft_band",
            "fragmentation_target_words": 40,
            "fragmentation_target_min_chunks": 16,
            "fragmentation_target_max_chunks": 80,
        },
    ]


def _space_defaults() -> Dict[str, Sequence[Any]]:
    return {
        "min_effective_chunks": [4, 5, 6, 7],
        "short_floor_min_words": [12, 15, 18, 21],
        "short_floor_signal_mode": ["always", "structural"],
        "guard_mode": ["hard_cap", "soft_band"],
        "ratio_threshold": [16, 20, 24, 28, 32],
        "fragmentation_target_words": [24, 28, 32, 36, 40],
        "fragmentation_target_min_chunks": [16, 24, 32],
        "fragmentation_target_max_chunks": [64, 80, 96],
        "fragmentation_target_low_ratio": [0.70, 0.80, 0.90],
        "fragmentation_target_high_ratio": [1.10, 1.20, 1.30],
    }


def _mutate_overrides(
    base: Mapping[str, Any],
    *,
    rng: random.Random,
    space: Mapping[str, Sequence[Any]],
    mutate_k: int = 3,
) -> Dict[str, Any]:
    keys = sorted(space.keys())
    if not keys:
        return dict(base)
    out = dict(base)
    rng.shuffle(keys)
    mutate_n = max(1, min(int(mutate_k), len(keys)))
    for key in keys[:mutate_n]:
        choices = list(space.get(key, []))
        if not choices:
            continue
        out[key] = rng.choice(choices)

    # Keep related bounds coherent.
    if int(out.get("fragmentation_target_max_chunks", 96)) < int(out.get("fragmentation_target_min_chunks", 24)):
        out["fragmentation_target_max_chunks"] = int(out.get("fragmentation_target_min_chunks", 24))
    if float(out.get("fragmentation_target_high_ratio", 1.20)) < float(out.get("fragmentation_target_low_ratio", 0.80)):
        out["fragmentation_target_high_ratio"] = float(out.get("fragmentation_target_low_ratio", 0.80))
    return out


def _build_stage2_random(
    *,
    top_rows: Sequence[Mapping[str, Any]],
    n_random: int,
    rng: random.Random,
    space: Mapping[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    if n_random <= 0:
        return []
    if not top_rows:
        return [{} for _ in range(n_random)]

    seeds = [dict(row.get("adaptive_overrides") or {}) for row in top_rows]
    out: List[Dict[str, Any]] = []
    seen = {json.dumps(seed, sort_keys=True, ensure_ascii=False) for seed in seeds}
    budget = max(32, n_random * 8)
    attempts = 0
    while len(out) < n_random and attempts < budget:
        attempts += 1
        base = rng.choice(seeds)
        candidate = _mutate_overrides(base, rng=rng, space=space, mutate_k=rng.randint(2, 4))
        key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    while len(out) < n_random:
        out.append({})
    return out


def _extract_metrics(eval_report: Dict[str, Any]) -> Dict[str, float]:
    gold = (
        eval_report.get("metrics_by_target", {})
        .get("gold", {})
        .get("metrics_primary", {})
    )
    if not gold:
        gold = eval_report.get("metrics_primary", {})
    sec = eval_report.get("metrics_secondary", {})
    return {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comprehensiveness": _safe_float(gold.get("comprehensiveness")),
        "sufficiency": _safe_float(gold.get("sufficiency")),
        "aopc_comprehensiveness": _safe_float(gold.get("aopc_comprehensiveness")),
        "aopc_sufficiency": _safe_float(gold.get("aopc_sufficiency")),
        "runtime_seconds": _safe_float(sec.get("runtime_seconds")),
    }


def _load_samples(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return out
    for path in sorted(sample_dir.glob("*.json")):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        sample_id = str(payload.get("sample_id", path.stem))
        out[sample_id] = payload
    return out


def _sample_stats(samples: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    if not samples:
        return {
            "sample_count": 0.0,
            "chunk_count_mean": 0.0,
            "top20_zero_ratio": 0.0,
            "fallback_rate": 0.0,
        }
    chunk_counts: List[int] = []
    top20_zero = 0
    fallback_hits = 0
    for payload in samples.values():
        chunks = payload.get("chunks", [])
        n_chunks = int(len(chunks))
        chunk_counts.append(n_chunks)
        if int(n_chunks * 0.2) == 0:
            top20_zero += 1
        diag = payload.get("metadata", {}).get("chunk_diagnostics", {})
        if isinstance(diag, Mapping) and bool(diag.get("fallback_applied", False)):
            fallback_hits += 1
    denom = float(len(chunk_counts))
    return {
        "sample_count": float(len(chunk_counts)),
        "chunk_count_mean": float(sum(chunk_counts) / denom),
        "top20_zero_ratio": float(top20_zero / denom),
        "fallback_rate": float(fallback_hits / denom),
    }


def _compute_drift(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    common = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    if not common:
        return {
            "sample_count_common": 0.0,
            "selected_changed_ratio": 0.0,
            "ranking_changed_ratio": 0.0,
            "trace_total_score_max_abs_diff": 0.0,
        }
    selected_changed = 0
    ranking_changed = 0
    trace_max = 0.0
    for sid in common:
        left = baseline_samples[sid]
        right = candidate_samples[sid]
        if left.get("selected_chunk_ids") != right.get("selected_chunk_ids"):
            selected_changed += 1
        if left.get("chunk_ranking") != right.get("chunk_ranking"):
            ranking_changed += 1
        left_trace = left.get("trace", [])
        right_trace = right.get("trace", [])
        local = 0.0
        for lrow, rrow in zip(left_trace, right_trace):
            diff = abs(_safe_float(lrow.get("total_score")) - _safe_float(rrow.get("total_score")))
            if diff > local:
                local = diff
        if local > trace_max:
            trace_max = local
    denom = float(len(common))
    return {
        "sample_count_common": float(len(common)),
        "selected_changed_ratio": float(selected_changed / denom),
        "ranking_changed_ratio": float(ranking_changed / denom),
        "trace_total_score_max_abs_diff": float(trace_max),
    }


def _score_trial(
    *,
    dataset: str,
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    metric_tol: float,
    runtime_weight: float,
) -> Dict[str, Any]:
    gains: Dict[str, float] = {}
    major_regressions = 0
    for metric, sign in _DIRECTION_SIGNS.items():
        delta = float(candidate_metrics.get(metric, 0.0)) - float(baseline_metrics.get(metric, 0.0))
        gain = float(sign * delta)
        gains[metric] = gain
        if gain < -float(metric_tol):
            major_regressions += 1

    quality_gain_sum = float(sum(gains.values()))
    runtime_gain = float(baseline_metrics.get("runtime_seconds", 0.0) - candidate_metrics.get("runtime_seconds", 0.0))
    regression_penalty = float(major_regressions * (2.0 if dataset in _LONG_TEXT_DATASETS else 1.0))
    total_score = float(quality_gain_sum - regression_penalty + runtime_weight * runtime_gain)

    return {
        "metric_directional_gains": gains,
        "quality_gain_sum": quality_gain_sum,
        "runtime_gain_seconds": runtime_gain,
        "major_regression_count": int(major_regressions),
        "regression_penalty": regression_penalty,
        "score": total_score,
    }


def _model_tag(model_path: str) -> str:
    return str(model_path).split("/")[-1].replace(".", "_")


def _expected_run_dir(
    *,
    output_dir: Path,
    dataset: str,
    model_path: str,
    search: str,
    k: int,
    lambdas: str,
    seed: int,
    explain_method: str,
) -> Path:
    run_name = (
        f"chunk-adaptive_search-{search}_k-{k}"
        f"_lam-{str(lambdas).replace(',', '-')}_seed-{seed}_method-{explain_method}"
    )
    return output_dir / dataset / f"model-{_model_tag(model_path)}" / run_name


def _run_trial(
    *,
    dataset: str,
    split: str,
    sample_ids_file: Path | None,
    trial_output_root: Path,
    trial_id: str,
    adaptive_profile: str,
    adaptive_overrides: Mapping[str, Any],
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    embedding_layer_ratio: float,
    mock_backbone: bool,
    search: str,
    k: int,
    lambdas: str,
    seed: int,
    deterministic: bool,
    explain_method: str,
    eraser_root: str | None,
    sst2_source: str | None,
    dataset_cache_dir: str | None,
    extra_args: Sequence[str],
    resume_check: str,
    reuse_existing: bool,
) -> Dict[str, Any]:
    run_output_dir = trial_output_root / trial_id
    run_dir = _expected_run_dir(
        output_dir=run_output_dir,
        dataset=dataset,
        model_path=model_path,
        search=search,
        k=k,
        lambdas=lambdas,
        seed=seed,
        explain_method=explain_method,
    )
    eval_path = run_dir / "eval_report.json"

    cmd = [
        sys.executable,
        "-m",
        "lima_llm",
        "--dataset",
        dataset,
        "--split",
        split,
        "--model-path",
        model_path,
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-length",
        str(max_length),
        "--embedding-layer-ratio",
        str(embedding_layer_ratio),
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        adaptive_profile,
        "--adaptive-overrides-json",
        json.dumps(dict(adaptive_overrides), ensure_ascii=False, separators=(",", ":")),
        "--search",
        search,
        "--k",
        str(k),
        "--lambdas",
        lambdas,
        "--seed",
        str(seed),
        "--output-dir",
        str(run_output_dir),
        "--resume-check",
        resume_check,
        "--run-eval",
        "--explain-method",
        explain_method,
    ]
    if deterministic:
        cmd.append("--deterministic")
    if mock_backbone:
        cmd.append("--mock-backbone")
    if sample_ids_file is not None:
        cmd.extend(["--sample-ids-file", str(sample_ids_file)])
    if eraser_root:
        cmd.extend(["--eraser-root", str(eraser_root)])
    if sst2_source:
        cmd.extend(["--sst2-source", str(sst2_source)])
    if dataset_cache_dir:
        cmd.extend(["--dataset-cache-dir", str(dataset_cache_dir)])
    if extra_args:
        cmd.extend(list(extra_args))

    launched = False
    return_code = 0
    elapsed = 0.0
    stdout_tail = ""
    stderr_tail = ""
    if not (reuse_existing and eval_path.exists()):
        launched = True
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = float(time.time() - t0)
        return_code = int(proc.returncode)
        stdout_tail = str(proc.stdout)[-2000:]
        stderr_tail = str(proc.stderr)[-2000:]

    if not eval_path.exists():
        return {
            "trial_id": trial_id,
            "status": "failed",
            "run_dir": str(run_dir),
            "run_output_dir": str(run_output_dir),
            "adaptive_overrides": dict(adaptive_overrides),
            "launched": bool(launched),
            "return_code": int(return_code),
            "launch_elapsed_seconds": float(elapsed),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "error": f"eval_report.json missing: {eval_path}",
        }

    eval_report = _read_json(eval_path)
    run_cfg = _read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    samples = _load_samples(run_dir)
    return {
        "trial_id": trial_id,
        "status": "ok",
        "run_dir": str(run_dir),
        "run_output_dir": str(run_output_dir),
        "adaptive_overrides": dict(adaptive_overrides),
        "launched": bool(launched),
        "return_code": int(return_code),
        "launch_elapsed_seconds": float(elapsed),
        "metrics": _extract_metrics(eval_report),
        "sample_stats": _sample_stats(samples),
        "provenance": {
            "git_commit": eval_report.get("provenance", {}).get("git", {}).get("commit"),
            "git_dirty": eval_report.get("provenance", {}).get("git", {}).get("dirty"),
            "device": run_cfg.get("device"),
            "deterministic": run_cfg.get("deterministic"),
            "adaptive_profile": run_cfg.get("adaptive_profile"),
            "adaptive_overrides_json": run_cfg.get("adaptive_overrides_json"),
        },
    }


def _rank_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda r: (
            str(r.get("status")) != "ok",
            -_safe_float(r.get("score_summary", {}).get("score")),
            -_safe_float(r.get("score_summary", {}).get("quality_gain_sum")),
            -_safe_float(r.get("score_summary", {}).get("runtime_gain_seconds")),
            str(r.get("trial_id", "")),
        ),
    )


def _resolve_ids_paths(
    *,
    dataset: str,
    split_root: Path | None,
    train_ids_file: str | None,
    dev_ids_file: str | None,
) -> Tuple[Path | None, Path | None]:
    train_path = Path(train_ids_file) if train_ids_file else None
    dev_path = Path(dev_ids_file) if dev_ids_file else None
    if split_root is not None:
        ds_root = split_root / dataset
        if train_path is None:
            candidate = ds_root / "train_ids.json"
            if candidate.exists():
                train_path = candidate
        if dev_path is None:
            candidate = ds_root / "dev_ids.json"
            if candidate.exists():
                dev_path = candidate
    return train_path, dev_path


def _load_ids_count(path: Path | None) -> int:
    if path is None or (not path.exists()):
        return 0
    obj = _read_json(path)
    if isinstance(obj, Mapping):
        rows = obj.get("sample_ids", [])
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    if not isinstance(rows, list):
        return 0
    return int(len(rows))


def _run_search_for_dataset(
    *,
    dataset: str,
    split: str,
    args,
    split_root: Path | None,
    space: Mapping[str, Sequence[Any]],
    stage1_grid: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    dataset_root = Path(args.output_root) / "search_trials" / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    trial_output_root = dataset_root / "runs"

    train_ids_path, dev_ids_path = _resolve_ids_paths(
        dataset=dataset,
        split_root=split_root,
        train_ids_file=args.train_ids_file,
        dev_ids_file=args.dev_ids_file,
    )
    train_id_count = _load_ids_count(train_ids_path)
    dev_id_count = _load_ids_count(dev_ids_path)

    rng = random.Random(int(args.seed))
    train_rows: List[Dict[str, Any]] = []

    for idx, overrides in enumerate(stage1_grid):
        trial_id = f"stage1_{idx:02d}"
        row = _run_trial(
            dataset=dataset,
            split=split,
            sample_ids_file=train_ids_path,
            trial_output_root=trial_output_root,
            trial_id=trial_id,
            adaptive_profile=args.adaptive_profile,
            adaptive_overrides=overrides,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_length=int(args.max_length),
            embedding_layer_ratio=float(args.embedding_layer_ratio),
            mock_backbone=bool(args.mock_backbone),
            search=args.search,
            k=int(args.k),
            lambdas=args.lambdas,
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
            explain_method=args.explain_method,
            eraser_root=args.eraser_root,
            sst2_source=args.sst2_source,
            dataset_cache_dir=args.dataset_cache_dir,
            extra_args=args.extra_args,
            resume_check=args.resume_check,
            reuse_existing=bool(args.reuse_existing),
        )
        row["stage"] = "stage1"
        train_rows.append(row)

    stage1_ok = [row for row in train_rows if row.get("status") == "ok"]
    if not stage1_ok:
        return {
            "dataset": dataset,
            "split": split,
            "error": "all stage1 trials failed",
            "train_trials": train_rows,
        }

    baseline_stage1 = stage1_ok[0]
    baseline_metrics_stage1 = dict(baseline_stage1.get("metrics", {}))
    baseline_samples_stage1 = _load_samples(Path(baseline_stage1["run_dir"]))
    for row in stage1_ok:
        score = _score_trial(
            dataset=dataset,
            baseline_metrics=baseline_metrics_stage1,
            candidate_metrics=row.get("metrics", {}),
            metric_tol=float(args.metric_tol),
            runtime_weight=float(args.runtime_weight),
        )
        drift = _compute_drift(baseline_samples_stage1, _load_samples(Path(row["run_dir"])))
        row["score_summary"] = score
        row["drift_vs_stage1_baseline"] = drift

    ranked_stage1 = _rank_rows(stage1_ok)

    top_for_stage2 = ranked_stage1[: max(1, int(args.stage2_anchor_topk))]
    stage2_candidates = _build_stage2_random(
        top_rows=top_for_stage2,
        n_random=int(args.stage2_random_count),
        rng=rng,
        space=space,
    )

    stage2_rows: List[Dict[str, Any]] = []
    for idx, overrides in enumerate(stage2_candidates):
        trial_id = f"stage2_{idx:02d}"
        row = _run_trial(
            dataset=dataset,
            split=split,
            sample_ids_file=train_ids_path,
            trial_output_root=trial_output_root,
            trial_id=trial_id,
            adaptive_profile=args.adaptive_profile,
            adaptive_overrides=overrides,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_length=int(args.max_length),
            embedding_layer_ratio=float(args.embedding_layer_ratio),
            mock_backbone=bool(args.mock_backbone),
            search=args.search,
            k=int(args.k),
            lambdas=args.lambdas,
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
            explain_method=args.explain_method,
            eraser_root=args.eraser_root,
            sst2_source=args.sst2_source,
            dataset_cache_dir=args.dataset_cache_dir,
            extra_args=args.extra_args,
            resume_check=args.resume_check,
            reuse_existing=bool(args.reuse_existing),
        )
        row["stage"] = "stage2"
        stage2_rows.append(row)

    stage2_ok = [row for row in stage2_rows if row.get("status") == "ok"]
    for row in stage2_ok:
        score = _score_trial(
            dataset=dataset,
            baseline_metrics=baseline_metrics_stage1,
            candidate_metrics=row.get("metrics", {}),
            metric_tol=float(args.metric_tol),
            runtime_weight=float(args.runtime_weight),
        )
        drift = _compute_drift(baseline_samples_stage1, _load_samples(Path(row["run_dir"])))
        row["score_summary"] = score
        row["drift_vs_stage1_baseline"] = drift

    ranked_train_all = _rank_rows([*stage1_ok, *stage2_ok])

    dev_rows: List[Dict[str, Any]] = []
    if dev_ids_path is not None and dev_ids_path.exists() and int(args.final_dev_topk) > 0:
        dev_candidates = ranked_train_all[: max(1, int(args.final_dev_topk))]
        baseline_dev = None
        for row in dev_candidates:
            dev_trial_id = f"dev_{str(row.get('trial_id'))}"
            dev_row = _run_trial(
                dataset=dataset,
                split=split,
                sample_ids_file=dev_ids_path,
                trial_output_root=trial_output_root,
                trial_id=dev_trial_id,
                adaptive_profile=args.adaptive_profile,
                adaptive_overrides=row.get("adaptive_overrides") or {},
                model_path=args.model_path,
                device=args.device,
                dtype=args.dtype,
                max_length=int(args.max_length),
                embedding_layer_ratio=float(args.embedding_layer_ratio),
                mock_backbone=bool(args.mock_backbone),
                search=args.search,
                k=int(args.k),
                lambdas=args.lambdas,
                seed=int(args.seed),
                deterministic=bool(args.deterministic),
                explain_method=args.explain_method,
                eraser_root=args.eraser_root,
                sst2_source=args.sst2_source,
                dataset_cache_dir=args.dataset_cache_dir,
                extra_args=args.extra_args,
                resume_check=args.resume_check,
                reuse_existing=bool(args.reuse_existing),
            )
            dev_row["stage"] = "dev"
            dev_row["source_train_trial_id"] = row.get("trial_id")
            dev_rows.append(dev_row)
            if baseline_dev is None and dev_row.get("status") == "ok":
                baseline_dev = dev_row

        if baseline_dev is not None:
            baseline_dev_metrics = dict(baseline_dev.get("metrics", {}))
            baseline_dev_samples = _load_samples(Path(baseline_dev["run_dir"]))
            for row in dev_rows:
                if row.get("status") != "ok":
                    continue
                score = _score_trial(
                    dataset=dataset,
                    baseline_metrics=baseline_dev_metrics,
                    candidate_metrics=row.get("metrics", {}),
                    metric_tol=float(args.metric_tol),
                    runtime_weight=float(args.runtime_weight),
                )
                drift = _compute_drift(baseline_dev_samples, _load_samples(Path(row["run_dir"])))
                row["score_summary"] = score
                row["drift_vs_dev_baseline"] = drift

    ranked_dev = _rank_rows([row for row in dev_rows if row.get("status") == "ok"])
    best_row = ranked_dev[0] if ranked_dev else (ranked_train_all[0] if ranked_train_all else None)

    dataset_payload = {
        "dataset": dataset,
        "split": split,
        "adaptive_profile": args.adaptive_profile,
        "search_space": dict(space),
        "stage1_grid_size": int(len(stage1_grid)),
        "stage2_random_count": int(args.stage2_random_count),
        "train_ids_file": str(train_ids_path) if train_ids_path else None,
        "dev_ids_file": str(dev_ids_path) if dev_ids_path else None,
        "train_ids_count": int(train_id_count),
        "dev_ids_count": int(dev_id_count),
        "train_trials": train_rows,
        "stage2_trials": stage2_rows,
        "dev_trials": dev_rows,
        "ranked_train": ranked_train_all,
        "ranked_dev": ranked_dev,
        "best_trial": best_row,
        "created_at_unix": int(time.time()),
    }

    out_json = dataset_root / "adaptive_hparam_search.json"
    _write_json(out_json, dataset_payload)

    flat_rows = []
    for section_name, rows in (
        ("stage1", train_rows),
        ("stage2", stage2_rows),
        ("dev", dev_rows),
    ):
        for row in rows:
            flat = {
                "dataset": dataset,
                "section": section_name,
                "trial_id": row.get("trial_id"),
                "status": row.get("status"),
                "run_dir": row.get("run_dir"),
                "score": _safe_float(row.get("score_summary", {}).get("score")),
                "quality_gain_sum": _safe_float(row.get("score_summary", {}).get("quality_gain_sum")),
                "runtime_gain_seconds": _safe_float(row.get("score_summary", {}).get("runtime_gain_seconds")),
                "major_regression_count": int(_safe_float(row.get("score_summary", {}).get("major_regression_count"))),
                "metric_gain_log_odds": _safe_float(
                    row.get("score_summary", {}).get("metric_directional_gains", {}).get("log_odds")
                ),
                "metric_gain_comp": _safe_float(
                    row.get("score_summary", {}).get("metric_directional_gains", {}).get("comprehensiveness")
                ),
                "metric_gain_suff": _safe_float(
                    row.get("score_summary", {}).get("metric_directional_gains", {}).get("sufficiency")
                ),
                "metric_gain_aopc_c": _safe_float(
                    row.get("score_summary", {}).get("metric_directional_gains", {}).get("aopc_comprehensiveness")
                ),
                "metric_gain_aopc_s": _safe_float(
                    row.get("score_summary", {}).get("metric_directional_gains", {}).get("aopc_sufficiency")
                ),
                "metric_log_odds": _safe_float(row.get("metrics", {}).get("log_odds")),
                "metric_comp": _safe_float(row.get("metrics", {}).get("comprehensiveness")),
                "metric_suff": _safe_float(row.get("metrics", {}).get("sufficiency")),
                "metric_aopc_c": _safe_float(row.get("metrics", {}).get("aopc_comprehensiveness")),
                "metric_aopc_s": _safe_float(row.get("metrics", {}).get("aopc_sufficiency")),
                "runtime_seconds": _safe_float(row.get("metrics", {}).get("runtime_seconds")),
                "top20_zero_ratio": _safe_float(row.get("sample_stats", {}).get("top20_zero_ratio")),
                "chunk_count_mean": _safe_float(row.get("sample_stats", {}).get("chunk_count_mean")),
                "fallback_rate": _safe_float(row.get("sample_stats", {}).get("fallback_rate")),
                "selected_changed_ratio": _safe_float(
                    row.get("drift_vs_stage1_baseline", row.get("drift_vs_dev_baseline", {})).get("selected_changed_ratio")
                ),
                "ranking_changed_ratio": _safe_float(
                    row.get("drift_vs_stage1_baseline", row.get("drift_vs_dev_baseline", {})).get("ranking_changed_ratio")
                ),
                "trace_total_score_max_abs_diff": _safe_float(
                    row.get("drift_vs_stage1_baseline", row.get("drift_vs_dev_baseline", {})).get("trace_total_score_max_abs_diff")
                ),
                "adaptive_overrides_json": json.dumps(row.get("adaptive_overrides") or {}, ensure_ascii=False, sort_keys=True),
            }
            flat_rows.append(flat)

    out_csv = dataset_root / "adaptive_hparam_search.csv"
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)

    return dataset_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-budget adaptive hparam search runner")
    parser.add_argument(
        "--datasets",
        type=str,
        default="eraser_movie_reviews,imdb,rotten_tomatoes,emotion,sst2",
        help="Comma-separated datasets",
    )
    parser.add_argument("--split-map-json", type=str, default=None, help="JSON object or path, e.g. {'imdb':'test'}")
    parser.add_argument("--output-root", type=str, default="lima_llm_results-adaptive-tune")

    parser.add_argument("--split-root", type=str, default="adaptive_tune_splits")
    parser.add_argument("--train-ids-file", type=str, default=None)
    parser.add_argument("--dev-ids-file", type=str, default=None)
    parser.add_argument("--final-dev-topk", type=int, default=5)

    parser.add_argument("--adaptive-profile", type=str, default="balanced")
    parser.add_argument("--space-json", type=str, default=None, help="Optional override search space JSON")
    parser.add_argument("--stage1-grid-json", type=str, default=None, help="Optional stage1 grid list JSON")
    parser.add_argument("--stage2-random-count", type=int, default=8)
    parser.add_argument("--stage2-anchor-topk", type=int, default=3)

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--embedding-layer-ratio", type=float, default=0.7)
    parser.add_argument("--mock-backbone", action="store_true")

    parser.add_argument("--search", type=str, default="greedy")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=str, default="1,1,0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")

    parser.add_argument("--explain-method", type=str, default="ours")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)
    parser.add_argument("--extra-args", type=str, nargs="*", default=[])

    parser.add_argument("--metric-tol", type=float, default=5e-4)
    parser.add_argument("--runtime-weight", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    datasets = _parse_datasets(args.datasets)
    split_map = dict(_DEFAULT_SPLIT_BY_DATASET)
    split_map_override = _parse_space_json(args.split_map_json)
    if split_map_override:
        for k, v in split_map_override.items():
            split_map[str(k)] = str(v)

    search_space = _space_defaults()
    user_space = _parse_space_json(args.space_json)
    if user_space:
        for key, value in user_space.items():
            if isinstance(value, list):
                search_space[str(key)] = list(value)

    stage1_grid = _default_stage1_grid()
    if args.stage1_grid_json:
        raw = _parse_space_json(args.stage1_grid_json)
        if not isinstance(raw, dict) or not isinstance(raw.get("trials"), list):
            raise ValueError("--stage1-grid-json must be JSON object with key 'trials' (list)")
        stage1_grid = [dict(x) for x in raw.get("trials", []) if isinstance(x, Mapping)]

    split_root = Path(args.split_root) if args.split_root else None
    results: List[Dict[str, Any]] = []
    for dataset in datasets:
        split = split_map.get(dataset, "validation")
        print(f"[adaptive-search] dataset={dataset} split={split} stage1={len(stage1_grid)} stage2={args.stage2_random_count}")
        dataset_result = _run_search_for_dataset(
            dataset=dataset,
            split=split,
            args=args,
            split_root=split_root,
            space=search_space,
            stage1_grid=stage1_grid,
        )
        results.append(dataset_result)
        best = dataset_result.get("best_trial")
        if isinstance(best, Mapping):
            print(
                "[adaptive-search] best",
                f"dataset={dataset}",
                f"trial={best.get('trial_id')}",
                f"score={_safe_float(best.get('score_summary', {}).get('score')):.6f}",
            )

    payload = {
        "datasets": datasets,
        "results": results,
        "adaptive_profile": args.adaptive_profile,
        "search_space": search_space,
        "stage1_grid": stage1_grid,
        "stage2_random_count": int(args.stage2_random_count),
        "stage2_anchor_topk": int(args.stage2_anchor_topk),
        "metric_tol": float(args.metric_tol),
        "runtime_weight": float(args.runtime_weight),
        "created_at_unix": int(time.time()),
    }

    out_root = Path(args.output_root)
    out_json = out_root / "adaptive_hparam_search_all.json"
    _write_json(out_json, payload)

    summary_rows: List[Dict[str, Any]] = []
    for ds in results:
        best = ds.get("best_trial")
        if not isinstance(best, Mapping):
            summary_rows.append(
                {
                    "dataset": ds.get("dataset"),
                    "status": "failed",
                    "best_trial_id": "",
                    "score": 0.0,
                    "quality_gain_sum": 0.0,
                    "runtime_gain_seconds": 0.0,
                    "major_regression_count": 0,
                    "adaptive_overrides_json": "{}",
                    "run_dir": "",
                }
            )
            continue
        summary_rows.append(
            {
                "dataset": ds.get("dataset"),
                "status": best.get("status"),
                "best_trial_id": best.get("trial_id"),
                "score": _safe_float(best.get("score_summary", {}).get("score")),
                "quality_gain_sum": _safe_float(best.get("score_summary", {}).get("quality_gain_sum")),
                "runtime_gain_seconds": _safe_float(best.get("score_summary", {}).get("runtime_gain_seconds")),
                "major_regression_count": int(
                    _safe_float(best.get("score_summary", {}).get("major_regression_count"))
                ),
                "adaptive_overrides_json": json.dumps(best.get("adaptive_overrides") or {}, ensure_ascii=False, sort_keys=True),
                "run_dir": best.get("run_dir"),
            }
        )

    out_csv = out_root / "adaptive_hparam_search_all.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    print(f"[adaptive-search] json={out_json}")
    print(f"[adaptive-search] csv={out_csv}")


if __name__ == "__main__":
    main()
