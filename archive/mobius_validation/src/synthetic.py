from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .subset_enumeration import candidate_terms
from .transforms import inverse_fourier_transform, inverse_mobius_transform


@dataclass(frozen=True)
class SyntheticInstance:
    instance_id: str
    family: str
    n_features: int
    values: np.ndarray
    true_basis: str
    true_support: List[int]
    metadata: Dict[str, object]


def _random_terms(rng: np.random.Generator, n: int, degree_min: int, degree_max: int, count: int) -> List[int]:
    terms = [term for term in candidate_terms(n, degree_max) if term.bit_count() >= degree_min]
    if count >= len(terms):
        return terms
    return [int(x) for x in rng.choice(terms, size=count, replace=False).tolist()]


def _values_from_sparse_coeff(n: int, basis: str, coeffs: Dict[int, float]) -> np.ndarray:
    arr = np.zeros(1 << int(n), dtype=np.float64)
    for term, value in coeffs.items():
        arr[int(term)] = float(value)
    if basis == "mobius":
        return inverse_mobius_transform(arr)
    if basis == "fourier":
        return inverse_fourier_transform(arr)
    raise ValueError(f"Unsupported synthetic basis: {basis!r}")


def generate_synthetic_suite(
    *,
    n_features: int = 12,
    instances_per_family: int = 50,
    seed: int = 0,
    noise_std: float = 0.0,
) -> List[SyntheticInstance]:
    rng = np.random.default_rng(int(seed))
    n = int(n_features)
    instances: List[SyntheticInstance] = []

    for idx in range(int(instances_per_family)):
        coeffs = {1 << bit: float(rng.normal()) for bit in range(n)}
        values = _values_from_sparse_coeff(n, "mobius", coeffs)
        instances.append(
            SyntheticInstance(
                instance_id=f"additive-{idx}",
                family="additive",
                n_features=n,
                values=_add_noise(values, rng, noise_std),
                true_basis="mobius",
                true_support=sorted(coeffs),
                metadata={"noise_std": float(noise_std)},
            )
        )

        coeffs = {}
        for chain_start in (0, max(0, n // 2 - 1)):
            term = 0
            for depth in range(1, min(4, n - chain_start) + 1):
                term |= 1 << (chain_start + depth - 1)
                coeffs[term] = float(rng.normal())
        values = _values_from_sparse_coeff(n, "fourier", coeffs)
        instances.append(
            SyntheticInstance(
                instance_id=f"hierarchical_fourier-{idx}",
                family="hierarchical_fourier",
                n_features=n,
                values=_add_noise(values, rng, noise_std),
                true_basis="fourier",
                true_support=sorted(coeffs),
                metadata={"noise_std": float(noise_std)},
            )
        )

        terms = _random_terms(rng, n, 3, 4, count=4)
        coeffs = {term: float(rng.normal()) for term in terms}
        values = _values_from_sparse_coeff(n, "fourier", coeffs)
        instances.append(
            SyntheticInstance(
                instance_id=f"nonhierarchical_fourier-{idx}",
                family="nonhierarchical_fourier",
                n_features=n,
                values=_add_noise(values, rng, noise_std),
                true_basis="fourier",
                true_support=sorted(coeffs),
                metadata={"noise_std": float(noise_std)},
            )
        )

        terms = _random_terms(rng, n, 2, 4, count=8)
        coeffs = {term: float(rng.normal()) for term in terms}
        values = _values_from_sparse_coeff(n, "mobius", coeffs)
        instances.append(
            SyntheticInstance(
                instance_id=f"sparse_mobius-{idx}",
                family="sparse_mobius",
                n_features=n,
                values=_add_noise(values, rng, noise_std),
                true_basis="mobius",
                true_support=sorted(coeffs),
                metadata={"noise_std": float(noise_std)},
            )
        )

        terms = _random_terms(rng, n, 2, 3, count=min(120, len(candidate_terms(n, 3))))
        coeffs = {term: float(rng.normal(scale=0.2)) for term in terms}
        values = _values_from_sparse_coeff(n, "mobius", coeffs)
        instances.append(
            SyntheticInstance(
                instance_id=f"dense_low_degree-{idx}",
                family="dense_low_degree",
                n_features=n,
                values=_add_noise(values, rng, noise_std),
                true_basis="mobius",
                true_support=sorted(coeffs),
                metadata={"noise_std": float(noise_std)},
            )
        )

    return instances


def _add_noise(values: Sequence[float], rng: np.random.Generator, noise_std: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if float(noise_std) <= 0:
        return arr
    return arr + rng.normal(scale=float(noise_std), size=arr.shape)

