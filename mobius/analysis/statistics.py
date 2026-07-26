"""Cluster-aware summaries used by paper audit scripts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


def aggregate_numeric(values: Sequence[float | None]) -> Dict[str, Any]:
    """Summarize finite numeric values with population standard deviation."""

    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else None,
        "std": float(np.std(array)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
    }


def clustered_bootstrap_mean(
    values_by_cluster: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_bootstrap: int = 2000,
    level: float = 0.95,
) -> Dict[str, Any]:
    """Bootstrap a mean by resampling clusters and retaining within-cluster rows."""

    clean = {
        str(cluster): [
            float(value)
            for value in values
            if np.isfinite(float(value))
        ]
        for cluster, values in values_by_cluster.items()
    }
    clean = {key: values for key, values in clean.items() if values}
    if not clean:
        return {
            "cluster_count": 0,
            "row_count": 0,
            "mean": None,
            "ci_low": None,
            "ci_high": None,
        }
    keys = sorted(clean)
    point = float(np.mean([value for key in keys for value in clean[key]]))
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        rows = [value for key in sampled for value in clean[str(key)]]
        draws.append(float(np.mean(rows)))
    alpha = (1.0 - float(level)) / 2.0
    return {
        "cluster_count": len(keys),
        "row_count": sum(len(values) for values in clean.values()),
        "mean": point,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
    }


def paired_cluster_summary(
    differences_by_cluster: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    """Summarize paired row differences using sample-cluster bootstrap."""

    summary = clustered_bootstrap_mean(
        differences_by_cluster,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    sample_means = np.asarray(
        [
            float(np.mean(values))
            for values in differences_by_cluster.values()
            if values
        ],
        dtype=np.float64,
    )
    effect = (
        float(np.mean(sample_means) / np.std(sample_means))
        if len(sample_means) > 1 and float(np.std(sample_means)) > 1e-15
        else None
    )
    p_value = None
    if len(sample_means):
        try:
            from scipy.stats import wilcoxon

            p_value = (
                1.0
                if np.allclose(sample_means, 0.0)
                else float(wilcoxon(sample_means).pvalue)
            )
        except ValueError:
            p_value = None
    return {
        **summary,
        "paired_effect_dz": effect,
        "wilcoxon_p_value": p_value,
    }
