from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from tqdm import tqdm

from ..backbone import build_backbone
from ..chunking import build_chunker
from ..data import load_dataset_bundle
from ..objective.submodular import ObjectiveWeights
from ..utils import (
    build_provenance,
    configure_determinism,
    ensure_dir,
    format_label_distribution,
    parse_lambdas,
    parse_q_values,
    set_seed,
)
from .explainer import ExplainerConfig, TextLIMAExplainer
from .io import rebuild_summary_csv, save_explanation
from .resume import is_sample_completed, sample_output_paths

_ADAPTIVE_HPARAM_KEYS = {
    "short_max_words",
    "medium_max_words",
    "long_max_words",
    "min_effective_chunks",
    "short_floor_min_words",
    "short_floor_signal_mode",
    "guard_mode",
    "ratio_threshold",
    "fragmentation_target_words",
    "target_words",
    "fragmentation_target_min_chunks",
    "target_min",
    "fragmentation_target_max_chunks",
    "target_max",
    "fragmentation_target_low_ratio",
    "band_low",
    "fragmentation_target_high_ratio",
    "band_high",
    "long_split_max_words_by_bucket",
}
_LAMBDA_HPARAM_KEYS = {"lambda1", "lambda2", "lambda3", "lambda4"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIMA LLM v1 pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "eraser_movie_reviews", "imdb", "rotten_tomatoes", "emotion"],
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--embedding-layer-ratio", type=float, default=0.7)
    parser.add_argument("--mock-backbone", action="store_true")

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=str, default="1,1,1,1")
    parser.add_argument(
        "--chunker",
        type=str,
        default="sentence",
        choices=["sentence", "sentence_v2", "fixed_token", "adaptive"],
    )
    parser.add_argument("--fixed-token-size", type=int, default=64)
    parser.add_argument(
        "--adaptive-profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
    )
    parser.add_argument(
        "--adaptive-overrides-json",
        type=str,
        default=None,
        help="Path to json file or inline JSON object for adaptive parameter overrides.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="Optional file (json/txt/csv) listing sample_id to keep, preserving dataset order.",
    )

    parser.add_argument(
        "--hparam-search-split",
        type=str,
        default="",
        help="Optional split for adaptive hparam search; empty means no search.",
    )
    parser.add_argument("--hparam-train-size", type=int, default=60)
    parser.add_argument("--hparam-dev-size", type=int, default=40)
    parser.add_argument(
        "--hparam-search-method",
        type=str,
        default="grid+random",
        choices=["grid", "random", "grid+random"],
    )
    parser.add_argument("--hparam-space-file", type=str, default=None)
    parser.add_argument("--hparam-random-trials", type=int, default=8)
    parser.add_argument("--hparam-max-trials", type=int, default=16)
    parser.add_argument("--hparam-enable-lambda-search", action="store_true")

    parser.add_argument("--search", type=str, default="greedy", choices=["greedy", "bidirectional"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--verbose-chunks", action="store_true")

    parser.add_argument("--output-dir", type=str, default="lima_llm_results")
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])

    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-q-values", type=str, default="1,5,10,20,50")
    parser.add_argument("--eval-granularity", type=str, default="token", choices=["token", "word"])
    parser.add_argument("--explain-method", type=str, default="ours", choices=["ours", "random", "gradient"])
    return parser


def _canonical_split(split: str) -> str:
    x = str(split).strip().lower()
    if x in {"val", "dev"}:
        return "validation"
    return x


def _canonical_hf_split(dataset: str, split: str) -> str:
    canon = _canonical_split(split)
    if str(dataset).strip().lower() == "imdb" and canon == "validation":
        return "test"
    return canon


def _load_adaptive_overrides(raw: str | None) -> Dict[str, object] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("--adaptive-overrides-json inline payload must be a JSON object")
        return dict(obj)
    path = Path(text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"--adaptive-overrides-json file must contain a JSON object: {path}")
    return dict(payload)


def _load_sample_ids(path: str) -> Set[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    ids: List[str] = []
    if suffix == ".json":
        obj = json.loads(raw)
        if isinstance(obj, list):
            ids = [str(x) for x in obj]
        elif isinstance(obj, dict):
            source = obj.get("sample_ids", [])
            if not isinstance(source, list):
                raise ValueError(f"sample_ids must be list in {p}")
            ids = [str(x) for x in source]
        else:
            raise ValueError(f"Unsupported JSON shape for sample ids: {p}")
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if "," in line:
                ids.extend(str(part).strip() for part in line.split(",") if str(part).strip() != "")
            else:
                ids.append(str(line))
    return {x for x in ids if x != ""}


def _filter_samples_by_ids(samples: Sequence, sample_ids: Set[str]):
    selected = [s for s in samples if str(getattr(s, "sample_id", "")) in sample_ids]
    missing = int(max(0, len(sample_ids) - len({str(getattr(s, "sample_id", "")) for s in selected})))
    return selected, missing


def _preview_dataset(bundle, backbone, dry_run: int) -> None:
    print(f"[dry-run] dataset={bundle.dataset_name} split={bundle.split} size={len(bundle.samples)}")
    print(f"[dry-run] label_distribution={format_label_distribution(s.label for s in bundle.samples)}")

    n = min(dry_run, len(bundle.samples))
    for idx in range(n):
        sample = bundle.samples[idx]
        token_len = backbone.tokenize_len(sample.text)
        rationale_chars = sum(end - start for start, end in sample.rationale_char_spans)
        coverage = rationale_chars / max(1, len(sample.text))
        snippet = sample.text.replace("\n", " ")[:120]
        print(
            f"[dry-run] #{idx} id={sample.sample_id} label={sample.label} "
            f"chars={len(sample.text)} tokens={token_len} rationale_cov={coverage:.4f} text={snippet!r}"
        )


def _print_chunk_preview(explainer: TextLIMAExplainer, sample) -> None:
    chunks = explainer.chunker(sample.text)
    print(f"[chunk-preview] sample={sample.sample_id} chunk_count={len(chunks)}")
    for chunk in chunks:
        preview = chunk.text.replace("\n", " ")[:40]
        print(f"  chunk#{chunk.chunk_id} span=[{chunk.start_char},{chunk.end_char}) text={preview!r}")


def _scan_resume(samples, output_root: Path, resume_mode: str):
    pending = []
    completed = 0
    for sample in samples:
        paths = sample_output_paths(output_root, sample.sample_id)
        if is_sample_completed(paths, mode=resume_mode):
            completed += 1
        else:
            pending.append(sample)
    print(
        f"[resume] mode={resume_mode} selected={len(samples)} completed={completed} pending={len(pending)}"
    )
    return pending, completed


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_if_possible(path: Path) -> Dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_hparam_space() -> Dict[str, List[Any]]:
    return {
        "short_max_words": [96, 120],
        "medium_max_words": [384, 448],
        "long_max_words": [960, 1152],
        "min_effective_chunks": [4, 5],
        "short_floor_min_words": [12, 15],
        "short_floor_signal_mode": ["always", "structural"],
        "guard_mode": ["hard_cap", "soft_band"],
        "ratio_threshold": [20, 24, 28],
        "fragmentation_target_words": [28, 32],
    }


def _load_hparam_space(path: str | None) -> Dict[str, List[Any]]:
    if path is None or str(path).strip() == "":
        return _default_hparam_space()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--hparam-space-file must be a JSON object.")
    params = payload.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("--hparam-space-file must contain `parameters` object.")
    out: Dict[str, List[Any]] = {}
    for key, values in params.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Parameter {key!r} must be a non-empty list.")
        out[str(key)] = list(values)
    return out


def _expand_grid(space: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(space.keys())
    if not keys:
        return [{}]
    values = [list(space[k]) for k in keys]
    out: List[Dict[str, Any]] = []
    for row in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, row)})
    return out


def _dedup_candidates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in rows:
        key = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _build_candidates(
    *,
    method: str,
    adaptive_space: Mapping[str, Sequence[Any]],
    lambda_space: Mapping[str, Sequence[Any]],
    enable_lambda_search: bool,
    random_trials: int,
    max_trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    adaptive_grid = _expand_grid(adaptive_space)
    if enable_lambda_search and lambda_space:
        lambda_grid = _expand_grid(lambda_space)
    else:
        lambda_grid = [{}]

    grid_rows: List[Dict[str, Any]] = []
    for adaptive_row in adaptive_grid:
        for lambda_row in lambda_grid:
            grid_rows.append(
                {
                    "adaptive_params": dict(adaptive_row),
                    "lambda_params": dict(lambda_row),
                }
            )

    rng = random.Random(int(seed))
    random_rows: List[Dict[str, Any]] = []
    if int(random_trials) > 0:
        if len(grid_rows) <= int(random_trials):
            random_rows = list(grid_rows)
        else:
            random_rows = rng.sample(grid_rows, k=int(random_trials))

    baseline_row = {"adaptive_params": {}, "lambda_params": {}}
    if method == "grid":
        rows = [baseline_row] + list(grid_rows)
    elif method == "random":
        rows = [baseline_row] + list(random_rows)
    elif method == "grid+random":
        rows = [baseline_row] + list(grid_rows) + list(random_rows)
    else:
        raise ValueError(f"Unsupported hparam search method: {method}")

    deduped = _dedup_candidates(rows)
    budget = max(1, int(max_trials))
    if len(deduped) <= budget:
        return deduped

    baseline = deduped[0]
    if budget == 1:
        return [baseline]

    rest = deduped[1:]
    sample_n = min(len(rest), budget - 1)
    keep_rows = rng.sample(rest, k=sample_n) if sample_n > 0 else []
    keep_rows = sorted(
        keep_rows,
        key=lambda row: json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return [baseline, *keep_rows]


def _split_hparam_space(
    *,
    raw_space: Mapping[str, Sequence[Any]],
    enable_lambda_search: bool,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[float]]]:
    adaptive_space: Dict[str, List[Any]] = {}
    lambda_space: Dict[str, List[float]] = {}

    for raw_key, raw_values in raw_space.items():
        key = str(raw_key).strip()
        if key in _ADAPTIVE_HPARAM_KEYS:
            adaptive_space[key] = list(raw_values)
            continue
        if key in _LAMBDA_HPARAM_KEYS:
            if not enable_lambda_search:
                raise ValueError(
                    f"Lambda key '{key}' found in hparam space but --hparam-enable-lambda-search is disabled."
                )
            values = []
            for v in raw_values:
                values.append(float(v))
            lambda_space[key] = values
            continue
        raise ValueError(
            f"Unknown hparam key '{key}'. Allowed adaptive keys: {sorted(_ADAPTIVE_HPARAM_KEYS)}; "
            f"lambda keys: {sorted(_LAMBDA_HPARAM_KEYS)}."
        )

    return adaptive_space, lambda_space


def _resolve_trial_lambdas(
    *,
    base_lambdas: Tuple[float, float, float, float],
    lambda_params: Mapping[str, Any],
) -> str:
    values = [float(base_lambdas[0]), float(base_lambdas[1]), float(base_lambdas[2]), float(base_lambdas[3])]
    key_to_idx = {"lambda1": 0, "lambda2": 1, "lambda3": 2, "lambda4": 3}
    for key, value in lambda_params.items():
        idx = key_to_idx.get(str(key))
        if idx is None:
            continue
        values[idx] = float(value)
    return ",".join(f"{x:g}" for x in values)


def _sample_id(sample) -> str:
    return str(getattr(sample, "sample_id", ""))


def _split_train_dev(*, samples: Sequence, train_size: int, dev_size: int, seed: int) -> Tuple[List, List]:
    rng = random.Random(int(seed))
    by_label: Dict[int, List[Any]] = {}
    for sample in samples:
        label = int(getattr(sample, "label", -1))
        by_label.setdefault(label, []).append(sample)
    for rows in by_label.values():
        rng.shuffle(rows)

    total = int(len(samples))
    target = min(total, int(train_size) + int(dev_size))
    if target <= 0:
        return [], []
    train_target = min(int(train_size), target)
    dev_target = max(0, target - train_target)

    labels = sorted(by_label.keys())
    counts = {k: len(by_label[k]) for k in labels}
    quotas = {k: int(round(float(target) * float(counts[k]) / float(max(1, total)))) for k in labels}
    while sum(quotas.values()) > target:
        k = max(labels, key=lambda x: (quotas[x], counts[x]))
        if quotas[k] > 0:
            quotas[k] -= 1
        else:
            break
    while sum(quotas.values()) < target:
        k = max(labels, key=lambda x: (counts[x] - quotas[x], counts[x]))
        if quotas[k] < counts[k]:
            quotas[k] += 1
        else:
            break

    picked: List[Any] = []
    for k in labels:
        picked.extend(by_label[k][: min(quotas[k], len(by_label[k]))])
    rng.shuffle(picked)
    return picked[:train_target], picked[train_target : train_target + dev_target]


def _output_root_from_values(
    *,
    output_dir: Path,
    dataset: str,
    model_path: str,
    chunker: str,
    search: str,
    k: int,
    lambdas: str,
    seed: int,
    explain_method: str,
) -> Path:
    return (
        output_dir
        / dataset
        / f"model-{model_path.split('/')[-1].replace('.', '_')}"
        / (
            f"chunk-{chunker}_search-{search}_k-{k}"
            f"_lam-{str(lambdas).replace(',', '-')}_seed-{seed}"
            f"_method-{explain_method}"
        )
    )


def _run_subprocess_trial(
    *,
    args,
    split: str,
    output_dir: Path,
    sample_ids_file: Path,
    adaptive_overrides: Mapping[str, Any],
    lambdas: str,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "lima_llm",
        "--dataset",
        str(args.dataset),
        "--split",
        str(split),
        "--model-path",
        str(args.model_path),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--max-length",
        str(args.max_length),
        "--embedding-layer-ratio",
        str(args.embedding_layer_ratio),
        "--k",
        str(args.k),
        "--lambdas",
        str(lambdas),
        "--chunker",
        str(args.chunker),
        "--fixed-token-size",
        str(args.fixed_token_size),
        "--adaptive-profile",
        str(args.adaptive_profile),
        "--adaptive-overrides-json",
        json.dumps(dict(adaptive_overrides), ensure_ascii=False, separators=(",", ":")),
        "--search",
        str(args.search),
        "--seed",
        str(args.seed),
        "--sample-ids-file",
        str(sample_ids_file),
        "--output-dir",
        str(output_dir),
        "--resume-check",
        str(args.resume_check),
        "--explain-method",
        str(args.explain_method),
        "--eval-q-values",
        str(args.eval_q_values),
        "--eval-granularity",
        str(args.eval_granularity),
        "--run-eval",
    ]
    if bool(args.deterministic):
        cmd.append("--deterministic")
    if bool(args.verbose_chunks):
        cmd.append("--verbose-chunks")
    if bool(args.mock_backbone):
        cmd.append("--mock-backbone")
    if args.eraser_root:
        cmd.extend(["--eraser-root", str(args.eraser_root)])
    if args.sst2_source:
        cmd.extend(["--sst2-source", str(args.sst2_source)])
    if args.dataset_cache_dir:
        cmd.extend(["--dataset-cache-dir", str(args.dataset_cache_dir)])

    t0 = time.time()
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, text=True, env=child_env)
    elapsed = float(time.time() - t0)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed code={proc.returncode}, split={split}")

    run_root = _output_root_from_values(
        output_dir=Path(output_dir),
        dataset=str(args.dataset),
        model_path=str(args.model_path),
        chunker=str(args.chunker),
        search=str(args.search),
        k=int(args.k),
        lambdas=str(lambdas),
        seed=int(args.seed),
        explain_method=str(args.explain_method),
    )
    report_path = run_root / "eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing eval_report.json at {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    primary = report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    if not primary:
        primary = report.get("metrics_primary", {})

    return {
        "elapsed_seconds": elapsed,
        "run_root": str(run_root),
        "report_path": str(report_path),
        "metrics": {
            "log_odds": float(primary.get("log_odds", 0.0)),
            "comprehensiveness": float(primary.get("comprehensiveness", 0.0)),
            "sufficiency": float(primary.get("sufficiency", 0.0)),
            "aopc_comprehensiveness": float(primary.get("aopc_comprehensiveness", 0.0)),
            "aopc_sufficiency": float(primary.get("aopc_sufficiency", 0.0)),
        },
    }


def _quality_gain(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> float:
    delta_lo = float(candidate["log_odds"]) - float(baseline["log_odds"])
    delta_comp = float(candidate["comprehensiveness"]) - float(baseline["comprehensiveness"])
    delta_suff = float(candidate["sufficiency"]) - float(baseline["sufficiency"])
    delta_aopc_c = float(candidate["aopc_comprehensiveness"]) - float(baseline["aopc_comprehensiveness"])
    delta_aopc_s = float(candidate["aopc_sufficiency"]) - float(baseline["aopc_sufficiency"])
    return float((-delta_lo) + delta_comp + (-delta_suff) + delta_aopc_c + (-delta_aopc_s))


def _augment_final_reports_hparam(
    *,
    output_root: Path,
    hparam_search_split: str,
    best_adaptive_overrides: Mapping[str, Any],
    best_lambdas: str,
    lambda_search_enabled: bool,
    summary_path: Path,
) -> None:
    run_cfg_path = output_root / "run_config.json"
    report_path = output_root / "eval_report.json"
    for path in (run_cfg_path, report_path):
        payload = _load_json_if_possible(path)
        if not isinstance(payload, dict):
            continue
        payload["hparam_search_enabled"] = True
        payload["hparam_search_split"] = str(hparam_search_split)
        payload["best_adaptive_overrides"] = dict(best_adaptive_overrides)
        payload["best_lambdas"] = str(best_lambdas)
        payload["lambda_search_enabled"] = bool(lambda_search_enabled)
        payload["hparam_search_summary_path"] = str(summary_path)
        _write_json(path, payload)


def _run_hparam_search_and_final(args, raw_argv: List[str]) -> None:
    if str(args.chunker).strip().lower() != "adaptive":
        raise ValueError("Hparam search currently requires --chunker adaptive.")

    search_split = str(args.hparam_search_split).strip()
    if search_split == "":
        return

    print(
        f"[hparam-search] enabled dataset={args.dataset} eval_split={args.split} "
        f"search_split={search_split} method={args.hparam_search_method}"
    )

    output_root = _output_root_from_values(
        output_dir=Path(args.output_dir),
        dataset=str(args.dataset),
        model_path=str(args.model_path),
        chunker=str(args.chunker),
        search=str(args.search),
        k=int(args.k),
        lambdas=str(args.lambdas),
        seed=int(args.seed),
        explain_method=str(args.explain_method),
    )
    hparam_root = output_root / "hparam_search"
    ensure_dir(hparam_root)

    print(f"[hparam-search] loading eval split: dataset={args.dataset} split={args.split}")
    eval_bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=None,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    print(f"[hparam-search] eval pool size={len(eval_bundle.samples)}")
    print(f"[hparam-search] loading search split: dataset={args.dataset} split={search_split}")
    search_bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=search_split,
        max_samples=None,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    print(f"[hparam-search] search pool size={len(search_bundle.samples)}")

    same_split = _canonical_hf_split(args.dataset, args.split) == _canonical_hf_split(args.dataset, search_split)
    requested_search_count = int(args.hparam_train_size) + int(args.hparam_dev_size)
    total_search_pool = int(len(search_bundle.samples))

    if total_search_pool < requested_search_count:
        raise ValueError(
            "Search pool is too small for requested train/dev: "
            f"total={total_search_pool}, train={args.hparam_train_size}, dev={args.hparam_dev_size}."
        )

    if same_split:
        remaining = int(len(eval_bundle.samples)) - requested_search_count
        if remaining <= 0:
            raise ValueError(
                "Search split equals eval split but no samples remain for evaluation: "
                f"total={len(eval_bundle.samples)}, train={args.hparam_train_size}, "
                f"dev={args.hparam_dev_size}, remaining_eval={remaining}."
            )

    train_samples, dev_samples = _split_train_dev(
        samples=search_bundle.samples,
        train_size=int(args.hparam_train_size),
        dev_size=int(args.hparam_dev_size),
        seed=int(args.seed),
    )
    print(
        f"[hparam-search] split done same_split={same_split} "
        f"train={len(train_samples)} dev={len(dev_samples)}"
    )

    used_search_ids = {_sample_id(s) for s in train_samples}
    used_search_ids.update(_sample_id(s) for s in dev_samples)

    eval_candidates = [s for s in eval_bundle.samples if _sample_id(s) not in used_search_ids]

    if args.sample_ids_file:
        req_ids = _load_sample_ids(args.sample_ids_file)
        eval_candidates, missing_count = _filter_samples_by_ids(eval_candidates, req_ids)
        print(
            f"[sample-filter] file={args.sample_ids_file} requested={len(req_ids)} "
            f"selected={len(eval_candidates)} missing={missing_count}"
        )

    if args.max_samples is not None:
        eval_candidates = list(eval_candidates[: int(args.max_samples)])

    if len(eval_candidates) == 0:
        raise ValueError("No evaluation samples remain after removing search samples and applying filters.")
    print(f"[hparam-search] eval candidates after disjoint/filter/max={len(eval_candidates)}")

    train_ids_path = hparam_root / "train_ids.json"
    dev_ids_path = hparam_root / "dev_ids.json"
    eval_ids_path = hparam_root / "eval_ids.json"
    _write_json(train_ids_path, {"sample_ids": [_sample_id(s) for s in train_samples]})
    _write_json(dev_ids_path, {"sample_ids": [_sample_id(s) for s in dev_samples]})
    _write_json(eval_ids_path, {"sample_ids": [_sample_id(s) for s in eval_candidates]})

    raw_hparam_space = _load_hparam_space(args.hparam_space_file)
    adaptive_space, lambda_space = _split_hparam_space(
        raw_space=raw_hparam_space,
        enable_lambda_search=bool(args.hparam_enable_lambda_search),
    )
    candidates = _build_candidates(
        method=str(args.hparam_search_method),
        adaptive_space=adaptive_space,
        lambda_space=lambda_space,
        enable_lambda_search=bool(args.hparam_enable_lambda_search),
        random_trials=int(args.hparam_random_trials),
        max_trials=int(args.hparam_max_trials),
        seed=int(args.seed),
    )
    base_lambdas = parse_lambdas(str(args.lambdas))
    print(
        f"[hparam-search] candidates={len(candidates)} "
        f"(includes baseline adaptive={{}} lambdas={args.lambdas})"
    )

    trial_rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        trial_id = f"trial_{idx:03d}"
        trial_root = hparam_root / "trials" / trial_id
        candidate_overrides = dict(candidate.get("adaptive_params") or {})
        candidate_lambda_params = dict(candidate.get("lambda_params") or {})
        trial_lambdas = _resolve_trial_lambdas(
            base_lambdas=base_lambdas,
            lambda_params=candidate_lambda_params,
        )
        print(
            f"[hparam-search] [{idx + 1}/{len(candidates)}] start {trial_id} "
            f"overrides={json.dumps(candidate_overrides, ensure_ascii=False, sort_keys=True)} "
            f"lambdas={trial_lambdas}"
        )
        train_result = _run_subprocess_trial(
            args=args,
            split=search_split,
            output_dir=trial_root / "train",
            sample_ids_file=train_ids_path,
            adaptive_overrides=candidate_overrides,
            lambdas=trial_lambdas,
        )
        print(
            f"[hparam-search] [{idx + 1}/{len(candidates)}] {trial_id} train "
            f"elapsed={train_result['elapsed_seconds']:.2f}s"
        )
        dev_result = _run_subprocess_trial(
            args=args,
            split=search_split,
            output_dir=trial_root / "dev",
            sample_ids_file=dev_ids_path,
            adaptive_overrides=candidate_overrides,
            lambdas=trial_lambdas,
        )
        print(
            f"[hparam-search] [{idx + 1}/{len(candidates)}] {trial_id} dev "
            f"elapsed={dev_result['elapsed_seconds']:.2f}s metrics={json.dumps(dev_result['metrics'], ensure_ascii=False)}"
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "candidate_adaptive_overrides": candidate_overrides,
                "candidate_lambdas": trial_lambdas,
                "adaptive_overrides": dict(candidate_overrides),
                "lambda_params": candidate_lambda_params,
                "train": train_result,
                "dev": dev_result,
            }
        )

    baseline_dev_metrics = trial_rows[0]["dev"]["metrics"]
    for row in trial_rows:
        row["quality_gain_dev_vs_baseline"] = _quality_gain(row["dev"]["metrics"], baseline_dev_metrics)

    ranked = sorted(
        trial_rows,
        key=lambda x: (-float(x.get("quality_gain_dev_vs_baseline", 0.0)), str(x.get("trial_id", ""))),
    )
    best = ranked[0]
    best_overrides = dict(best.get("candidate_adaptive_overrides") or {})
    best_lambdas = str(best.get("candidate_lambdas", str(args.lambdas)))
    print(
        f"[hparam-search] best trial={best.get('trial_id')} "
        f"gain={float(best.get('quality_gain_dev_vs_baseline', 0.0)):.6f} "
        f"lambdas={best_lambdas}"
    )

    final_output_dir = Path(args.output_dir)
    final_result = _run_subprocess_trial(
        args=args,
        split=str(args.split),
        output_dir=final_output_dir,
        sample_ids_file=eval_ids_path,
        adaptive_overrides=best_overrides,
        lambdas=best_lambdas,
    )
    final_run_root = Path(str(final_result["run_root"]))

    summary = {
        "hparam_search_enabled": True,
        "lambda_search_enabled": bool(args.hparam_enable_lambda_search),
        "dataset": args.dataset,
        "eval_split": args.split,
        "hparam_search_split": search_split,
        "same_split": bool(same_split),
        "hparam_train_size": int(args.hparam_train_size),
        "hparam_dev_size": int(args.hparam_dev_size),
        "hparam_search_method": str(args.hparam_search_method),
        "hparam_random_trials": int(args.hparam_random_trials),
        "hparam_max_trials": int(args.hparam_max_trials),
        "hparam_space": raw_hparam_space,
        "adaptive_space": adaptive_space,
        "lambda_space": lambda_space,
        "requested_search_count": int(requested_search_count),
        "search_pool_count": int(total_search_pool),
        "train_count": int(len(train_samples)),
        "dev_count": int(len(dev_samples)),
        "eval_count_after_disjoint_and_filters": int(len(eval_candidates)),
        "candidate_adaptive_overrides": [dict(row.get("adaptive_params") or {}) for row in candidates],
        "candidate_lambdas": [
            _resolve_trial_lambdas(
                base_lambdas=base_lambdas,
                lambda_params=dict(row.get("lambda_params") or {}),
            )
            for row in candidates
        ],
        "best_adaptive_overrides": best_overrides,
        "best_lambdas": best_lambdas,
        "final_run_root": str(final_run_root),
        "trials": trial_rows,
        "best_trial": {
            "trial_id": str(best.get("trial_id", "")),
            "quality_gain_dev_vs_baseline": float(best.get("quality_gain_dev_vs_baseline", 0.0)),
            "adaptive_overrides": best_overrides,
            "lambdas": best_lambdas,
        },
        "command": raw_argv,
    }
    summary_path = hparam_root / "search_summary.json"
    _write_json(summary_path, summary)

    _augment_final_reports_hparam(
        output_root=final_run_root,
        hparam_search_split=search_split,
        best_adaptive_overrides=best_overrides,
        best_lambdas=best_lambdas,
        lambda_search_enabled=bool(args.hparam_enable_lambda_search),
        summary_path=summary_path,
    )
    print(f"[hparam-search] summary={summary_path}")
    print(f"[hparam-search] best_overrides={json.dumps(best_overrides, ensure_ascii=False)}")


def main(argv: List[str] | None = None) -> None:
    run_started = time.time()
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])

    deterministic_info = configure_determinism(args.deterministic)
    if args.deterministic:
        print(f"[deterministic] enabled info={deterministic_info}")
    else:
        print("[deterministic] disabled")

    if str(args.hparam_search_split).strip() != "":
        _run_hparam_search_and_final(args=args, raw_argv=raw_argv)
        return

    weights = ObjectiveWeights(*parse_lambdas(args.lambdas))

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )

    if args.sample_ids_file:
        sample_ids = _load_sample_ids(args.sample_ids_file)
        filtered_samples, missing_count = _filter_samples_by_ids(bundle.samples, sample_ids)
        print(
            f"[sample-filter] file={args.sample_ids_file} requested={len(sample_ids)} "
            f"selected={len(filtered_samples)} missing={missing_count}"
        )
        bundle.samples = filtered_samples
        if len(bundle.samples) == 0:
            raise ValueError("No samples selected by --sample-ids-file")

    adaptive_overrides = _load_adaptive_overrides(args.adaptive_overrides_json)

    backbone = build_backbone(
        model_path=args.model_path,
        device=args.device,
        use_mock_backbone=args.mock_backbone,
        max_length=args.max_length,
        embedding_layer_ratio=args.embedding_layer_ratio,
        dtype=args.dtype,
    )
    set_seed(args.seed)

    chunker = build_chunker(
        method=args.chunker,
        tokenizer=getattr(backbone, "tokenizer", None),
        fixed_token_size=args.fixed_token_size,
        adaptive_profile=args.adaptive_profile,
        adaptive_overrides=adaptive_overrides,
    )

    config = ExplainerConfig(
        dataset_name=bundle.dataset_name,
        split=args.split,
        k=args.k,
        search=args.search,
        weights=weights,
        explain_method=args.explain_method,
        seed=args.seed,
    )
    explainer = TextLIMAExplainer(
        backbone=backbone,
        chunker=chunker,
        verbalizers=bundle.verbalizers,
        config=config,
    )

    if args.dry_run > 0:
        _preview_dataset(bundle, backbone, dry_run=args.dry_run)
        preview_count = min(1, len(bundle.samples))
        for idx in range(preview_count):
            _print_chunk_preview(explainer, bundle.samples[idx])
        return

    output_root = _output_root_from_values(
        output_dir=Path(args.output_dir),
        dataset=str(args.dataset),
        model_path=str(args.model_path),
        chunker=str(args.chunker),
        search=str(args.search),
        k=int(args.k),
        lambdas=str(args.lambdas),
        seed=int(args.seed),
        explain_method=str(args.explain_method),
    )
    ensure_dir(output_root)
    ensure_dir(output_root / "samples")

    run_config_payload = dict(vars(args))
    run_config_payload["provenance"] = build_provenance(
        stage="run_config",
        parsed_args=vars(args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=run_started,
        end_time=time.time(),
        cwd=Path.cwd(),
    )

    config_path = output_root / "run_config.json"
    if config_path.exists():
        existing = _load_json_if_possible(config_path)
        if isinstance(existing, dict) and "provenance" in existing:
            print(f"[config] keep existing run config: {config_path}")
        else:
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(run_config_payload)
            _write_json(config_path, merged)
            print(f"[config] updated run config with provenance: {config_path}")
    else:
        _write_json(config_path, run_config_payload)

    if args.run_eval:
        eval_cfg_path = output_root / "eval_config.json"
        eval_cfg_payload = dict(vars(args))
        eval_cfg_payload["provenance"] = build_provenance(
            stage="eval_config",
            parsed_args=vars(args),
            raw_argv=raw_argv,
            deterministic_info=deterministic_info,
            start_time=run_started,
            end_time=time.time(),
            cwd=Path.cwd(),
        )
        _write_json(eval_cfg_path, eval_cfg_payload)

    pending_samples, _ = _scan_resume(bundle.samples, output_root=output_root, resume_mode=args.resume_check)
    if not pending_samples:
        print("[resume] no pending samples, exiting")
    else:
        begin = time.time()
        for sample in tqdm(pending_samples, desc="lima-llm-v1", dynamic_ncols=True):
            result = explainer.explain_sample(sample, verbose=args.verbose_chunks)
            save_explanation(result, output_root)
        elapsed = time.time() - begin
        print(f"[done] processed={len(pending_samples)} elapsed={elapsed:.2f}s")

    summary_path = output_root / "summary.csv"
    if pending_samples or not summary_path.exists():
        summary_path = rebuild_summary_csv(output_root)
    print(f"[done] summary={summary_path}")

    if args.run_eval:
        from ..eval.evaluate import evaluate_saved_explanations

        q_values = parse_q_values(args.eval_q_values)
        print(f"[eval] running q_values={q_values} method={args.explain_method}")
        eval_started = time.time()
        eval_report = evaluate_saved_explanations(
            output_root=output_root,
            bundle=bundle,
            backbone=backbone,
            verbalizers=bundle.verbalizers,
            q_values=q_values,
            explain_method=args.explain_method,
            eval_granularity=args.eval_granularity,
        )
        eval_report["provenance"] = build_provenance(
            stage="eval_report",
            parsed_args=vars(args),
            raw_argv=raw_argv,
            deterministic_info=deterministic_info,
            start_time=eval_started,
            end_time=time.time(),
            cwd=Path.cwd(),
        )
        report_path = output_root / "eval_report.json"
        _write_json(report_path, eval_report)
        print(f"[eval] report={report_path}")


if __name__ == "__main__":
    main()
