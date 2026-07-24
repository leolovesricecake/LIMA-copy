from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.fit_additive import fit_additive_lasso
from mobius_verify.src.fit_fourier import fit_fourier_lasso
from mobius_verify.src.fit_gbt import fit_sklearn_gbt
from mobius_verify.src.fit_mobius import fit_mobius_lasso
from mobius_verify.src.reconstruction_metrics import auc_logx
from mobius_verify.src.subset_enumeration import all_masks, split_masks
from mobius_verify.src.utils import (
    atomic_write_json,
    load_yaml,
    patch_multiprocess_resource_tracker_shutdown,
    read_json,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run limited-query recovery from saved exact value tables.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "recovery_default.yaml"))
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _iter_tables(results_dir: Path, scopes: List[str]):
    if "exact_global" in scopes:
        for value_file in sorted((results_dir / "values_exact_global").glob("*/*/values.npy")):
            task = value_file.parents[1].name
            sample_id = value_file.parent.name
            yield "exact_global", task, sample_id, "global", value_file
    if "exact_probe" in scopes:
        for value_file in sorted((results_dir / "values_exact_probe").glob("*/*/*/values.npy")):
            task = value_file.parents[2].name
            sample_id = value_file.parents[1].name
            probe_id = value_file.parent.name
            yield "exact_probe", task, sample_id, probe_id, value_file


def _budget_values(n_features: int, alphas: List[float], pool_size: int) -> List[int]:
    denom = max(1.0, float(n_features) * math.log2(max(2, int(n_features))))
    budgets = [int(round(float(alpha) * denom)) for alpha in alphas]
    return sorted(set(max(2, min(int(m), int(pool_size) - 1)) for m in budgets if int(pool_size) > 2))


def _values_for_masks(values: np.ndarray, masks: List[int]) -> np.ndarray:
    return np.asarray([float(values[int(mask)]) for mask in masks], dtype=np.float64)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return values * 0.0, mean, std
    return (values - mean) / std, mean, std


def _run_methods(
    *,
    methods: List[str],
    train_masks: List[int],
    y_train: np.ndarray,
    val_masks: List[int],
    y_val: np.ndarray,
    test_masks: List[int],
    y_test: np.ndarray,
    n_features: int,
    config: Dict,
    seed: int,
) -> Dict[str, Dict]:
    outputs = {}
    alphas = [float(x) for x in config.get("lasso_alphas", [0.0001, 0.001, 0.01, 0.1, 1.0])]
    degrees = [int(x) for x in config.get("degrees", [2, 3, 4])]
    max_candidates = int(config.get("max_candidates", 20000))
    for method in methods:
        if method == "additive_lasso":
            outputs[method] = fit_additive_lasso(
                train_masks=train_masks,
                y_train=y_train,
                validation_masks=val_masks,
                y_validation=y_val,
                test_masks=test_masks,
                y_test=y_test,
                n_features=n_features,
                alphas=alphas,
            )
        elif method == "mobius_lasso":
            outputs[method] = fit_mobius_lasso(
                train_masks=train_masks,
                y_train=y_train,
                validation_masks=val_masks,
                y_validation=y_val,
                test_masks=test_masks,
                y_test=y_test,
                n_features=n_features,
                degrees=degrees,
                alphas=alphas,
                max_candidates=max_candidates,
            )
        elif method == "fourier_lasso":
            outputs[method] = fit_fourier_lasso(
                train_masks=train_masks,
                y_train=y_train,
                validation_masks=val_masks,
                y_validation=y_val,
                test_masks=test_masks,
                y_test=y_test,
                n_features=n_features,
                degrees=degrees,
                alphas=alphas,
                max_candidates=max_candidates,
            )
        elif method == "sklearn_gbt":
            outputs[method] = fit_sklearn_gbt(
                train_masks=train_masks,
                y_train=y_train,
                validation_masks=val_masks,
                y_validation=y_val,
                test_masks=test_masks,
                y_test=y_test,
                n_features=n_features,
                random_state=seed,
                param_grid=config.get("gbt_param_grid"),
            )
        else:
            raise ValueError(f"Unsupported recovery method: {method!r}")
    return outputs


def main() -> None:
    patch_multiprocess_resource_tracker_shutdown()
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    results_dir = resolve_project_path(
        args.results_dir or config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "exact_default",
    )
    scopes = [str(x) for x in config.get("source_scopes", ["exact_global", "exact_probe"])]
    methods = [str(x) for x in config.get("methods", ["additive_lasso", "mobius_lasso", "fourier_lasso", "sklearn_gbt"])]
    seeds = [int(x) for x in config.get("seeds", [0, 1, 2, 3, 4])]
    alphas = [float(x) for x in config.get("budget_alphas", [0.25, 0.5, 1, 2, 4, 8])]

    for scope, task, sample_id, probe_id, value_file in _iter_tables(results_dir, scopes):
        metadata = read_json(value_file.parent / "metadata.json")
        raw_values = np.load(value_file).astype(np.float64)
        y_all, y_mean, y_std = _standardize(raw_values)
        n_features = int(round(math.log2(len(raw_values))))
        if y_std <= 1e-12:
            status_dir = results_dir / "recovery" / scope / "_degenerate" / task / sample_id / probe_id
            atomic_write_json(
                status_dir / "metrics.json",
                {
                    "status": "degenerate_value_function",
                    "experiment_scope": scope,
                    "sample_id": sample_id,
                    "probe_id": probe_id,
                    "n_features": int(n_features),
                    "value_std": float(y_std),
                },
            )
            continue

        masks = all_masks(n_features)
        for seed in seeds:
            train_pool, val_masks, test_masks = split_masks(
                masks,
                seed=int(seed),
                test_fraction=float(config.get("test_fraction", 0.2)),
                validation_fraction=float(config.get("validation_fraction", 0.1)),
            )
            budgets = _budget_values(n_features, alphas, len(train_pool))
            rng = np.random.default_rng(int(seed))
            shuffled_pool = [int(train_pool[idx]) for idx in rng.permutation(len(train_pool))]
            y_val = _values_for_masks(y_all, val_masks)
            y_test = _values_for_masks(y_all, test_masks)

            budget_rows = []
            for budget in budgets:
                train_masks = sorted(shuffled_pool[: int(budget)])
                y_train = _values_for_masks(y_all, train_masks)
                outputs = _run_methods(
                    methods=methods,
                    train_masks=train_masks,
                    y_train=y_train,
                    val_masks=val_masks,
                    y_val=y_val,
                    test_masks=test_masks,
                    y_test=y_test,
                    n_features=n_features,
                    config=config,
                    seed=int(seed),
                )
                normalized_query_budget = float(budget / (n_features * math.log2(max(2, n_features))))
                for method, result in outputs.items():
                    out = {
                        **result,
                        "experiment_scope": scope,
                        "task": task,
                        "sample_id": sample_id,
                        "probe_id": None if probe_id == "global" else probe_id,
                        "probe_strategy": metadata.get("probe_strategy"),
                        "conditioning_mode": metadata.get("conditioning_mode"),
                        "n_features": int(n_features),
                        "n_total_words": metadata.get("n_total_words", n_features),
                        "budget": int(budget),
                        "seed": int(seed),
                        "normalized_query_budget": normalized_query_budget,
                        "m_over_n": float(budget / n_features),
                        "value_mean": float(y_mean),
                        "value_std": float(y_std),
                    }
                    out_dir = (
                        results_dir
                        / "recovery"
                        / scope
                        / method
                        / task
                        / sample_id
                        / probe_id
                        / f"n_{n_features}"
                        / f"budget_{budget}"
                        / f"seed_{seed}"
                    )
                    if (out_dir / "metrics.json").exists() and not args.overwrite:
                        continue
                    atomic_write_json(out_dir / "metrics.json", out)
                    if result.get("status") == "ok":
                        budget_rows.append(
                            {
                                "budget": int(budget),
                                "normalized_query_budget": normalized_query_budget,
                                "method": method,
                                "r2": float(result["test_r2"]),
                            }
                        )
            by_method: Dict[str, List[Dict[str, float]]] = {}
            for row in budget_rows:
                by_method.setdefault(str(row["method"]), []).append(row)
            for method, rows in by_method.items():
                auc = auc_logx(rows, "normalized_query_budget", "r2")
                summary_dir = results_dir / "recovery" / scope / method / task / sample_id / probe_id / f"seed_{seed}"
                atomic_write_json(summary_dir / "query_auc.json", {"auc_r2_log_query": auc, "rows": rows})


if __name__ == "__main__":
    main()
