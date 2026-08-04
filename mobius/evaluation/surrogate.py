"""Shared held-out mask sampling and surrogate reconstruction metrics."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def _mask_from_kept(kept: Iterable[int]) -> int:
    """Encode kept player indices as one arbitrary-width integer mask."""

    mask = 0
    for player in kept:
        mask |= 1 << int(player)
    return mask


HELDOUT_DISTRIBUTIONS = ("bernoulli", "near_full")


def _fill_bernoulli_masks(
    n_features: int,
    count: int,
    *,
    excluded: set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Draw unique Bernoulli-0.5 masks outside an exclusion set."""

    n = int(n_features)
    universe = 1 << n
    target = min(max(0, int(count)), max(0, universe - len(excluded)))
    selected: set[int] = set()
    attempts = 0
    while len(selected) < target and attempts < max(2000, target * 400):
        attempts += 1
        row = rng.random(n) < 0.5
        mask = _mask_from_kept(np.flatnonzero(row))
        if mask not in excluded:
            selected.add(mask)
    if len(selected) < target and n <= 20:
        remaining = [
            mask
            for mask in range(universe)
            if mask not in excluded and mask not in selected
        ]
        rng.shuffle(remaining)
        selected.update(remaining[: target - len(selected)])
    return sorted(selected)


def _near_full_candidates(
    n_features: int,
    deletion_counts: Sequence[int],
) -> Iterable[int]:
    """Yield all masks from configured near-full deletion cardinalities."""

    n = int(n_features)
    full = (1 << n) - 1
    for deletion_count in deletion_counts:
        for deleted in itertools.combinations(range(n), int(deletion_count)):
            mask = full
            for player in deleted:
                mask &= ~(1 << int(player))
            yield mask


def _fill_near_full_masks(
    n_features: int,
    count: int,
    *,
    deletion_counts: Sequence[int],
    excluded: set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Draw unique near-full masks outside an exclusion set."""

    n = int(n_features)
    choices = sorted({int(value) for value in deletion_counts if 0 < int(value) <= n})
    possible = sum(math.comb(n, value) for value in choices)
    target = min(max(0, int(count)), max(0, possible))
    full = (1 << n) - 1
    selected: set[int] = set()
    attempts = 0
    while len(selected) < target and attempts < max(2000, target * 400):
        attempts += 1
        deletion_count = int(rng.choice(choices)) if choices else 0
        deleted = (
            rng.choice(n, size=deletion_count, replace=False)
            if deletion_count
            else []
        )
        mask = full
        for player in deleted:
            mask &= ~(1 << int(player))
        if mask not in excluded:
            selected.add(mask)
    if len(selected) < target and possible <= 1_000_000:
        remaining = [
            mask
            for mask in _near_full_candidates(n, choices)
            if mask not in excluded and mask not in selected
        ]
        rng.shuffle(remaining)
        selected.update(remaining[: target - len(selected)])
    return sorted(selected)


def sample_shared_heldout_masks(
    n_features: int,
    *,
    count_per_distribution: int,
    seed: int,
    excluded_masks: Sequence[int],
    near_full_deletions: Sequence[int] = (1, 2, 3, 5),
    distributions: Sequence[str] = ("bernoulli",),
) -> Dict[str, Any]:
    """Generate deterministic, disjoint masks for requested held-out distributions."""

    excluded = {int(mask) for mask in excluded_masks}
    raw_distributions = (
        distributions.split(",")
        if isinstance(distributions, str)
        else distributions
    )
    requested = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in raw_distributions
            if str(value).strip()
        )
    )
    if not requested:
        raise ValueError("At least one held-out distribution is required.")
    unknown = sorted(set(requested) - set(HELDOUT_DISTRIBUTIONS))
    if unknown:
        raise ValueError(
            f"Unsupported held-out distributions {unknown}; "
            f"expected {list(HELDOUT_DISTRIBUTIONS)}."
        )

    sampled: Dict[str, Any] = {}
    already_selected = set(excluded)
    if "bernoulli" in requested:
        bernoulli = _fill_bernoulli_masks(
            int(n_features),
            int(count_per_distribution),
            excluded=already_selected,
            rng=np.random.default_rng(int(seed)),
        )
        sampled["bernoulli"] = bernoulli
        already_selected.update(bernoulli)
    if "near_full" in requested:
        near_full = _fill_near_full_masks(
            int(n_features),
            int(count_per_distribution),
            deletion_counts=near_full_deletions,
            excluded=already_selected,
            rng=np.random.default_rng(int(seed) + 104729),
        )
        sampled["near_full"] = near_full
        already_selected.update(near_full)

    sampled["diagnostics"] = {
        "n_features": int(n_features),
        "distributions": list(requested),
        "count_per_distribution_requested": int(count_per_distribution),
        "counts": {name: len(sampled[name]) for name in requested},
        "excluded_count": len(excluded),
        "seed": int(seed),
        "near_full_deletions": [
            int(value) for value in near_full_deletions
        ],
    }
    return sampled


def reconstruction_metrics(
    truth: Sequence[float],
    prediction: Sequence[float],
) -> Dict[str, Any]:
    """Compute R2, range-normalized RMSE, and MAE with degenerate handling."""

    y = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError("Surrogate truth and prediction must have equal shape.")
    if len(y) == 0:
        return {
            "count": 0,
            "value_range": None,
            "degenerate": True,
            "r2": None,
            "nrmse_range": None,
            "mae": None,
        }
    residual = y - pred
    value_range = float(np.max(y) - np.min(y))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    degenerate = value_range <= 1e-12 or ss_total <= 1e-15
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "count": len(y),
        "value_range": value_range,
        "degenerate": bool(degenerate),
        "r2": (
            None
            if degenerate
            else float(1.0 - np.sum(residual**2) / ss_total)
        ),
        "nrmse_range": None if degenerate else float(rmse / value_range),
        "mae": float(np.mean(np.abs(residual))),
    }


def aggregate_reconstruction(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-sample reconstruction metrics into macro summaries."""

    output: Dict[str, Any] = {}
    for metric in ("r2", "nrmse_range", "mae"):
        values = np.asarray(
            [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None
                and np.isfinite(float(row[metric]))
            ],
            dtype=np.float64,
        )
        output[metric] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else None,
            "std": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else None)
            ),
            "median": float(np.median(values)) if len(values) else None,
        }
    output["sample_count"] = len(rows)
    output["degenerate_count"] = sum(bool(row.get("degenerate")) for row in rows)
    return output
