"""Unique-mask samplers with exact logical budgets."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mobius.core.schema import SamplingResult


SAMPLERS = {"deletion_mixture", "bernoulli", "uniform_size"}


def normalize_sampler(value: str | None) -> str:
    """Normalize sampler names and aliases."""

    normalized = str(value or "deletion_mixture").strip().lower()
    aliases = {"mixture": "deletion_mixture", "global": "bernoulli"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SAMPLERS:
        raise ValueError(f"Unsupported sampler {value!r}; expected {sorted(SAMPLERS)}")
    return normalized


def _digest(masks: Sequence[int]) -> str:
    """Hash the realized ordered mask list."""

    payload = json.dumps([int(value) for value in masks], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_budget(n_features: int, budget: int) -> Tuple[int, int]:
    """Clamp a logical budget to the finite coalition universe."""

    universe = 1 << int(n_features)
    return min(max(0, int(budget)), universe), universe


def _mask_from_players(players: Sequence[int]) -> int:
    """Encode selected players as an integer keep mask."""

    mask = 0
    for player in players:
        mask |= 1 << int(player)
    return mask


def _finalize(
    masks: Sequence[int],
    sources: Mapping[int, str],
    *,
    name: str,
    n_features: int,
    budget: int,
    seed: int,
    extra: Mapping[str, object] | None = None,
) -> SamplingResult:
    """Sort masks and build common cardinality/source diagnostics."""

    ordered = sorted({int(mask) for mask in masks})
    keep_counts = [mask.bit_count() for mask in ordered]
    diagnostics: Dict[str, object] = {
        "sampler": name,
        "requested_budget": int(budget),
        "realized_budget": len(ordered),
        "universe_size": 1 << int(n_features),
        "seed": int(seed),
        "source_counts": dict(sorted(Counter(sources.get(mask, name) for mask in ordered).items())),
        "keep_cardinality_counts": dict(
            sorted((str(key), value) for key, value in Counter(keep_counts).items())
        ),
        "masks_digest": _digest(ordered),
    }
    diagnostics.update(dict(extra or {}))
    return SamplingResult(ordered, diagnostics)


def _exhaustive(
    n_features: int,
    budget: int,
    seed: int,
    name: str,
) -> SamplingResult | None:
    """Enumerate exactly when the requested budget covers the universe."""

    target, universe = _target_budget(n_features, budget)
    if target != universe:
        return None
    masks = list(range(universe))
    return _finalize(
        masks,
        {mask: "exhaustive" for mask in masks},
        name=name,
        n_features=n_features,
        budget=budget,
        seed=seed,
    )


def _draw_bernoulli_mask(rng: np.random.Generator, n_features: int) -> int:
    """Draw one Bernoulli-0.5 keep mask."""

    return _mask_from_players(np.flatnonzero(rng.random(int(n_features)) < 0.5))


def _fill_unique(
    selected: Dict[int, str],
    *,
    target: int,
    universe: int,
    rng: np.random.Generator,
    draw,
    source: str,
) -> None:
    """Fill a unique-mask dictionary with bounded rejection sampling."""

    attempts = 0
    limit = max(2000, target * 500)
    while len(selected) < target and attempts < limit:
        attempts += 1
        selected.setdefault(int(draw()), source)
    if len(selected) < target and universe <= (1 << 20):
        remaining = [mask for mask in range(universe) if mask not in selected]
        rng.shuffle(remaining)
        for mask in remaining[: target - len(selected)]:
            selected[mask] = "exhaustive_fill"
    if len(selected) != target:
        raise RuntimeError(f"Could not realize exact sampler budget {target}.")


def sample_bernoulli(
    n_features: int,
    budget: int,
    *,
    seed: int,
    include_empty_full: bool = True,
) -> SamplingResult:
    """Sample unique Bernoulli-0.5 masks plus optional anchors."""

    exhaustive = _exhaustive(n_features, budget, seed, "bernoulli")
    if exhaustive is not None:
        return exhaustive
    target, universe = _target_budget(n_features, budget)
    rng = np.random.default_rng(int(seed))
    selected: Dict[int, str] = {}
    if include_empty_full:
        for mask, source in ((0, "anchor_empty"), (universe - 1, "anchor_full")):
            if len(selected) < target:
                selected[mask] = source
    _fill_unique(
        selected,
        target=target,
        universe=universe,
        rng=rng,
        draw=lambda: _draw_bernoulli_mask(rng, n_features),
        source="bernoulli",
    )
    return _finalize(
        selected,
        selected,
        name="bernoulli",
        n_features=n_features,
        budget=budget,
        seed=seed,
        extra={"include_empty_full": bool(include_empty_full)},
    )


def sample_uniform_size(
    n_features: int,
    budget: int,
    *,
    seed: int,
    include_empty_full: bool = True,
) -> SamplingResult:
    """Sample size uniformly, then sample a coalition uniformly at that size."""

    exhaustive = _exhaustive(n_features, budget, seed, "uniform_size")
    if exhaustive is not None:
        return exhaustive
    n = int(n_features)
    target, universe = _target_budget(n, budget)
    rng = np.random.default_rng(int(seed))
    selected: Dict[int, str] = {}
    if include_empty_full:
        for mask, source in ((0, "anchor_empty"), (universe - 1, "anchor_full")):
            if len(selected) < target:
                selected[mask] = source

    def draw() -> int:
        """Draw one coalition under a uniform cardinality prior."""

        size = int(rng.integers(0, n + 1))
        players = rng.choice(n, size=size, replace=False) if size else []
        return _mask_from_players(players)

    _fill_unique(
        selected,
        target=target,
        universe=universe,
        rng=rng,
        draw=draw,
        source="uniform_size",
    )
    return _finalize(
        selected,
        selected,
        name="uniform_size",
        n_features=n,
        budget=budget,
        seed=seed,
        extra={"include_empty_full": bool(include_empty_full)},
    )


def sample_deletion_mixture(
    n_features: int,
    budget: int,
    *,
    seed: int,
    global_fraction: float = 0.5,
    near_full_fraction: float = 0.3,
    fixed_cardinality_fraction: float = 0.2,
    near_full_deletions: Sequence[int] = (1, 2, 3, 5),
    fixed_keep_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    include_empty_full: bool = True,
) -> SamplingResult:
    """Mix global, near-full, and fixed-cardinality observations."""

    exhaustive = _exhaustive(n_features, budget, seed, "deletion_mixture")
    if exhaustive is not None:
        return exhaustive
    n = int(n_features)
    target, universe = _target_budget(n, budget)
    full = universe - 1
    fractions = np.asarray(
        [global_fraction, near_full_fraction, fixed_cardinality_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or float(fractions.sum()) <= 0:
        raise ValueError("Sampler fractions must be nonnegative with positive sum.")
    fractions /= fractions.sum()
    rng = np.random.default_rng(int(seed))
    selected: Dict[int, str] = {}
    if include_empty_full:
        for mask, source in ((0, "anchor_empty"), (full, "anchor_full")):
            if len(selected) < target:
                selected[mask] = source
    remaining = target - len(selected)
    raw = fractions * remaining
    allocation = np.floor(raw).astype(int)
    residual = remaining - int(allocation.sum())
    for index in np.argsort(-(raw - allocation))[:residual]:
        allocation[int(index)] += 1

    def add_from_sizes(count: int, keep_sizes: Sequence[int], source: str) -> None:
        """Add up to count unique masks drawn from selected cardinalities."""

        goal = min(target, len(selected) + int(count))
        choices = sorted({max(0, min(n, int(value))) for value in keep_sizes})
        attempts = 0
        while len(selected) < goal and attempts < max(1000, count * 300):
            attempts += 1
            size = int(rng.choice(choices))
            players = rng.choice(n, size=size, replace=False) if size else []
            selected.setdefault(_mask_from_players(players), source)

    add_from_sizes(
        int(allocation[1]),
        [n - int(value) for value in near_full_deletions],
        "near_full",
    )
    add_from_sizes(
        int(allocation[2]),
        [round(float(value) * n) for value in fixed_keep_fractions],
        "fixed_cardinality",
    )
    global_goal = min(target, len(selected) + int(allocation[0]))
    _fill_unique(
        selected,
        target=global_goal,
        universe=universe,
        rng=rng,
        draw=lambda: _draw_bernoulli_mask(rng, n),
        source="global_bernoulli",
    )
    _fill_unique(
        selected,
        target=target,
        universe=universe,
        rng=rng,
        draw=lambda: _draw_bernoulli_mask(rng, n),
        source="global_fill",
    )
    return _finalize(
        selected,
        selected,
        name="deletion_mixture",
        n_features=n,
        budget=budget,
        seed=seed,
        extra={
            "normalized_fractions": {
                "global": float(fractions[0]),
                "near_full": float(fractions[1]),
                "fixed_cardinality": float(fractions[2]),
            },
            "near_full_deletions": [int(value) for value in near_full_deletions],
            "fixed_keep_fractions": [float(value) for value in fixed_keep_fractions],
            "include_empty_full": bool(include_empty_full),
        },
    )


def sample_masks(
    n_features: int,
    budget: int,
    *,
    seed: int,
    config: Mapping[str, object],
) -> SamplingResult:
    """Dispatch to the configured method-owned coalition sampler."""

    name = normalize_sampler(str(config.get("name", "deletion_mixture")))
    common = {
        "n_features": int(n_features),
        "budget": int(budget),
        "seed": int(seed),
        "include_empty_full": bool(config.get("include_empty_full", True)),
    }
    if name == "bernoulli":
        return sample_bernoulli(**common)
    if name == "uniform_size":
        return sample_uniform_size(**common)
    return sample_deletion_mixture(
        **common,
        global_fraction=float(config.get("global_fraction", 0.5)),
        near_full_fraction=float(config.get("near_full_fraction", 0.3)),
        fixed_cardinality_fraction=float(
            config.get("fixed_cardinality_fraction", 0.2)
        ),
        near_full_deletions=config.get("near_full_deletions", [1, 2, 3, 5]),
        fixed_keep_fractions=config.get(
            "fixed_keep_fractions", [0.25, 0.5, 0.75]
        ),
    )

