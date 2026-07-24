from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, n_bootstrap: int = 2000, level: float = 0.95) -> Dict[str, float | None]:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1.0 - float(level)) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "n": int(len(arr)),
    }


def paired_difference_summary(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> Dict[str, float | None]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b))
    ]
    if not pairs:
        return {"mean_diff": None, "median_diff": None, "low": None, "high": None, "n": 0}
    diffs = np.asarray([a - b for a, b in pairs], dtype=np.float64)
    ci = bootstrap_ci(diffs, seed=seed, n_bootstrap=n_bootstrap)
    ci["mean_diff"] = ci.pop("mean")
    ci["median_diff"] = ci.pop("median")
    return ci

