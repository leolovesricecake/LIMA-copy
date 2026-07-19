from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


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
