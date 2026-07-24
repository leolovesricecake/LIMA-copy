from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DeletionMobiusSamplingResult:
    masks: List[int]
    diagnostics: Dict[str, object]


def _sampling_digest(masks: Sequence[int]) -> str:
    payload = json.dumps([int(mask) for mask in masks], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_deletion_mobius_masks(
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
) -> DeletionMobiusSamplingResult:
    """Sample the method-owned observation design for deletion-Mobius recovery.

    Masks encode kept players. Near-full rows therefore correspond to small
    deletion sets in g(D) = f(N \\ D). The mixture supplies local coefficient
    information without giving up global surrogate coverage.
    """

    n = int(n_features)
    if n < 0:
        raise ValueError("n_features must be non-negative")
    universe_size = 1 << n
    target = min(max(0, int(budget)), universe_size)
    if target == 0:
        return DeletionMobiusSamplingResult(
            masks=[],
            diagnostics={
                "sampler": "deletion_mobius_mixture_v1",
                "requested_budget": int(budget),
                "realized_budget": 0,
                "source_counts": {},
                "masks_digest": _sampling_digest([]),
            },
        )

    fractions = np.asarray(
        [float(global_fraction), float(near_full_fraction), float(fixed_cardinality_fraction)],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or float(np.sum(fractions)) <= 0:
        raise ValueError("Sampler fractions must be non-negative with a positive sum")
    fractions = fractions / float(np.sum(fractions))

    if target == universe_size and n <= 20:
        masks = list(range(universe_size))
        return DeletionMobiusSamplingResult(
            masks=masks,
            diagnostics={
                "sampler": "deletion_mobius_mixture_v1",
                "requested_budget": int(budget),
                "realized_budget": int(len(masks)),
                "universe_size": int(universe_size),
                "source_counts": {"exhaustive": int(len(masks))},
                "normalized_fractions": {
                    "global": float(fractions[0]),
                    "near_full": float(fractions[1]),
                    "fixed_cardinality": float(fractions[2]),
                },
                "masks_digest": _sampling_digest(masks),
            },
        )

    rng = np.random.default_rng(int(seed))
    source_by_mask: Dict[int, str] = {}
    full = universe_size - 1

    def add(mask: int, source: str) -> bool:
        value = int(mask)
        if value < 0 or value >= universe_size or value in source_by_mask:
            return False
        source_by_mask[value] = str(source)
        return True

    if include_empty_full:
        for mask, source in ((full, "anchor_full"), (0, "anchor_empty")):
            if len(source_by_mask) < target:
                add(mask, source)

    remaining = target - len(source_by_mask)
    raw_allocations = fractions * remaining
    allocations = np.floor(raw_allocations).astype(int)
    for index in np.argsort(-(raw_allocations - allocations))[: remaining - int(np.sum(allocations))]:
        allocations[int(index)] += 1

    def draw_by_deletion_count(count: int, deletion_counts: Sequence[int], source: str) -> None:
        choices = sorted({max(0, min(n, int(value))) for value in deletion_counts})
        if not choices or count <= 0:
            return
        goal = min(target, len(source_by_mask) + int(count))
        attempts = 0
        max_attempts = max(1000, int(count) * 200)
        while len(source_by_mask) < goal and attempts < max_attempts:
            attempts += 1
            deletion_count = int(rng.choice(choices))
            deleted = (
                rng.choice(n, size=deletion_count, replace=False).tolist()
                if deletion_count > 0
                else []
            )
            mask = full
            for player in deleted:
                mask &= ~(1 << int(player))
            add(mask, source)

    draw_by_deletion_count(
        int(allocations[1]),
        near_full_deletions,
        "near_full",
    )
    keep_counts = sorted(
        {
            max(0, min(n, int(round(float(fraction) * n))))
            for fraction in fixed_keep_fractions
        }
    )
    draw_by_deletion_count(
        int(allocations[2]),
        [n - keep_count for keep_count in keep_counts],
        "fixed_cardinality",
    )

    global_goal = min(target, len(source_by_mask) + int(allocations[0]))
    attempts = 0
    while len(source_by_mask) < global_goal and attempts < max(1000, target * 200):
        attempts += 1
        row = rng.random(n) < 0.5
        mask = 0
        for player, keep in enumerate(row):
            if bool(keep):
                mask |= 1 << player
        add(mask, "global_bernoulli")

    # Reallocate any source shortfall while preserving the exact logical budget.
    attempts = 0
    while len(source_by_mask) < target and attempts < max(2000, target * 400):
        attempts += 1
        row = rng.random(n) < 0.5
        mask = 0
        for player, keep in enumerate(row):
            if bool(keep):
                mask |= 1 << player
        add(mask, "global_fill")

    if len(source_by_mask) < target and n <= 20:
        remaining_pool = [mask for mask in range(universe_size) if mask not in source_by_mask]
        rng.shuffle(remaining_pool)
        for mask in remaining_pool[: target - len(source_by_mask)]:
            add(mask, "exhaustive_fill")

    if len(source_by_mask) != target:
        raise RuntimeError(
            f"Could not realize deletion-Mobius budget: requested={target}, got={len(source_by_mask)}"
        )

    masks = sorted(source_by_mask)
    source_counts: Dict[str, int] = {}
    for source in source_by_mask.values():
        source_counts[source] = source_counts.get(source, 0) + 1
    popcounts = [int(mask).bit_count() for mask in masks]
    deletion_counts = [n - count for count in popcounts]
    diagnostics = {
        "sampler": "deletion_mobius_mixture_v1",
        "requested_budget": int(budget),
        "realized_budget": int(len(masks)),
        "universe_size": int(universe_size),
        "seed": int(seed),
        "include_empty_full": bool(include_empty_full),
        "normalized_fractions": {
            "global": float(fractions[0]),
            "near_full": float(fractions[1]),
            "fixed_cardinality": float(fractions[2]),
        },
        "near_full_deletions": [int(value) for value in near_full_deletions],
        "fixed_keep_fractions": [float(value) for value in fixed_keep_fractions],
        "source_counts": dict(sorted(source_counts.items())),
        "keep_count_min": int(min(popcounts)),
        "keep_count_max": int(max(popcounts)),
        "deletion_count_min": int(min(deletion_counts)),
        "deletion_count_max": int(max(deletion_counts)),
        "masks_digest": _sampling_digest(masks),
    }
    return DeletionMobiusSamplingResult(masks=masks, diagnostics=diagnostics)


def all_masks(n_features: int) -> List[int]:
    n = int(n_features)
    if n < 0:
        raise ValueError("n_features must be non-negative")
    return list(range(1 << n))


def masks_array(masks: Sequence[int]) -> np.ndarray:
    max_mask = max([0] + [int(mask) for mask in masks])
    dtype = np.uint64 if max_mask < (1 << 63) else object
    return np.asarray([int(mask) for mask in masks], dtype=dtype)


def mask_to_bool(mask: int, n_features: int) -> np.ndarray:
    value = int(mask)
    return np.asarray([(value & (1 << idx)) != 0 for idx in range(int(n_features))], dtype=bool)


def masks_to_matrix(masks: Sequence[int], n_features: int) -> np.ndarray:
    return np.vstack([mask_to_bool(mask, n_features) for mask in masks]).astype(np.float64)


def mask_popcount(mask: int) -> int:
    return int(mask).bit_count()


def popcounts_for_n(n_features: int) -> np.ndarray:
    return np.asarray([int(mask).bit_count() for mask in range(1 << int(n_features))], dtype=np.int16)


def candidate_terms(n_features: int, max_degree: int, *, include_empty: bool = False) -> List[int]:
    n = int(n_features)
    d = int(max_degree)
    terms = []
    start = 0 if include_empty else 1
    for mask in range(start, 1 << n):
        if mask.bit_count() <= d:
            terms.append(mask)
    return terms


def candidate_count(n_features: int, max_degree: int) -> int:
    n = int(n_features)
    return int(sum(math.comb(n, degree) for degree in range(1, int(max_degree) + 1)))


def random_uniform_masks(n_features: int, count: int, seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    n = int(n_features)
    out = set()
    target = min(int(count), 1 << n)
    while len(out) < target:
        rows = rng.random((max(1, target - len(out)), n)) < 0.5
        for row in rows:
            mask = 0
            for idx, keep in enumerate(row):
                if bool(keep):
                    mask |= 1 << idx
            out.add(mask)
            if len(out) >= target:
                break
    return sorted(out)


def sample_attribution_masks(
    n_features: int,
    budget: int,
    *,
    seed: int,
    exclude: Sequence[int] = (),
    include_empty_full: bool = True,
) -> List[int]:
    """Sample unique Bernoulli-0.5 masks under an exact logical budget."""

    n = int(n_features)
    universe_size = 1 << n
    excluded = {int(mask) for mask in exclude if 0 <= int(mask) < universe_size}
    available = universe_size - len(excluded)
    target = min(max(0, int(budget)), available)
    if target == 0:
        return []
    out: set[int] = set()
    if include_empty_full:
        for mask in (0, universe_size - 1):
            if mask not in excluded and len(out) < target:
                out.add(mask)
    rng = np.random.default_rng(int(seed))
    if n <= 20 and target > available // 2:
        pool = np.asarray([mask for mask in range(universe_size) if mask not in excluded], dtype=object)
        chosen = rng.choice(len(pool), size=target, replace=False)
        return sorted(int(pool[idx]) for idx in chosen)
    while len(out) < target:
        needed = target - len(out)
        rows = rng.random((max(8, needed * 2), n)) < 0.5
        for row in rows:
            mask = 0
            for idx, keep in enumerate(row):
                if bool(keep):
                    mask |= 1 << idx
            if mask not in excluded:
                out.add(mask)
            if len(out) >= target:
                break
    return sorted(out)


def sample_evaluation_masks(
    n_features: int,
    *,
    count_per_distribution: int,
    seed: int,
    exclude: Sequence[int] = (),
    near_full_deletions: Sequence[int] = (1, 2, 3, 5),
    fixed_keep_fractions: Sequence[float] = (0.25, 0.5, 0.75),
) -> Dict[str, List[int]]:
    """Build shared held-out masks for uniform, near-full and fixed-cardinality tests."""

    n = int(n_features)
    excluded = {int(mask) for mask in exclude}
    count = max(0, int(count_per_distribution))
    uniform = sample_attribution_masks(
        n,
        count,
        seed=int(seed),
        exclude=sorted(excluded),
        include_empty_full=False,
    )
    used = excluded | set(uniform)
    rng = np.random.default_rng(int(seed) + 104729)

    def _draw_by_cardinality(delete_counts: Sequence[int], target: int) -> List[int]:
        choices = sorted({max(0, min(n, int(value))) for value in delete_counts})
        possible = sum(math.comb(n, deletion) for deletion in choices)
        target = min(int(target), max(0, possible))
        out: set[int] = set()
        attempts = 0
        while len(out) < target and attempts < max(1000, target * 100):
            attempts += 1
            deletion = int(rng.choice(choices))
            deleted = rng.choice(n, size=deletion, replace=False).tolist() if deletion else []
            mask = (1 << n) - 1
            for idx in deleted:
                mask &= ~(1 << int(idx))
            if mask not in used:
                out.add(mask)
        return sorted(out)

    near_full = _draw_by_cardinality(near_full_deletions, count)
    used.update(near_full)
    keep_counts = sorted(
        {
            max(0, min(n, int(round(float(fraction) * n))))
            for fraction in fixed_keep_fractions
        }
    )
    fixed_deletions = [n - keep for keep in keep_counts]
    fixed = _draw_by_cardinality(fixed_deletions, count)
    return {"uniform": uniform, "near_full": near_full, "fixed_cardinality": fixed}


def random_near_full_masks(
    n_features: int,
    count: int,
    seed: int,
    deletion_counts: Sequence[int] = (1, 2, 3, 5, 10),
) -> List[int]:
    rng = np.random.default_rng(int(seed))
    n = int(n_features)
    full = (1 << n) - 1
    choices = [max(0, min(int(x), n)) for x in deletion_counts]
    out = set()
    while len(out) < int(count):
        r = int(rng.choice(choices))
        delete = set(rng.choice(n, size=r, replace=False).tolist()) if r > 0 else set()
        mask = full
        for idx in delete:
            mask &= ~(1 << int(idx))
        out.add(mask)
    return sorted(out)


def random_fixed_cardinality_masks(
    n_features: int,
    count: int,
    seed: int,
    keep_counts: Sequence[int],
) -> List[int]:
    rng = np.random.default_rng(int(seed))
    n = int(n_features)
    choices = [max(0, min(int(x), n)) for x in keep_counts]
    out = set()
    while len(out) < int(count):
        k = int(rng.choice(choices))
        keep = rng.choice(n, size=k, replace=False).tolist() if k > 0 else []
        mask = 0
        for idx in keep:
            mask |= 1 << int(idx)
        out.add(mask)
    return sorted(out)


def split_masks(
    masks: Sequence[int],
    *,
    seed: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(int(seed))
    arr = np.asarray([int(mask) for mask in masks], dtype=object)
    perm = rng.permutation(len(arr))
    shuffled = [int(arr[idx]) for idx in perm]
    n_total = len(shuffled)
    n_test = int(round(float(test_fraction) * n_total))
    n_val = int(round(float(validation_fraction) * n_total))
    test = sorted(shuffled[:n_test])
    validation = sorted(shuffled[n_test : n_test + n_val])
    train_pool = sorted(shuffled[n_test + n_val :])
    return train_pool, validation, test
