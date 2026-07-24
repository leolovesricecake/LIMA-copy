"""Low-degree deletion-Mobius and Fourier basis encoders."""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np


BASES = {"deletion_mobius", "fourier"}


def normalize_basis(value: str | None) -> str:
    """Normalize supported basis aliases."""

    normalized = str(value or "deletion_mobius").strip().lower()
    aliases = {"deletion": "deletion_mobius", "mobius": "deletion_mobius"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in BASES:
        raise ValueError(f"Unsupported basis {value!r}; expected {sorted(BASES)}")
    return normalized


def term_players(term: int) -> Tuple[int, ...]:
    """Decode a bit-mask term into ordered player indices."""

    value = int(term)
    return tuple(
        index for index in range(value.bit_length()) if value & (1 << index)
    )


def low_degree_terms(n_features: int, max_degree: int) -> List[int]:
    """Enumerate all nonempty terms up to the requested degree."""

    n = int(n_features)
    maximum = min(n, int(max_degree))
    terms: List[int] = []
    for degree in range(1, maximum + 1):
        for players in itertools.combinations(range(n), degree):
            term = 0
            for player in players:
                term |= 1 << player
            terms.append(term)
    return terms


def masks_to_matrix(masks: Sequence[int], n_features: int) -> np.ndarray:
    """Decode integer keep masks into a binary observation matrix."""

    return np.asarray(
        [
            [bool(int(mask) & (1 << player)) for player in range(int(n_features))]
            for mask in masks
        ],
        dtype=np.float64,
    )


def design_matrix(
    masks: Sequence[int],
    terms: Sequence[int],
    *,
    n_features: int,
    basis: str,
    dtype=np.float32,
) -> np.ndarray:
    """Encode coalition masks in deletion monomial or parity coordinates."""

    normalized = normalize_basis(basis)
    keep = masks_to_matrix(masks, int(n_features)).astype(dtype, copy=False)
    coordinates = 1.0 - keep if normalized == "deletion_mobius" else keep
    output = np.empty((len(masks), len(terms)), dtype=dtype)
    for column, term in enumerate(terms):
        players = term_players(int(term))
        if normalized == "fourier":
            values = np.ones(len(masks), dtype=dtype)
            for player in players:
                values *= 1.0 - 2.0 * coordinates[:, player]
        else:
            values = np.ones(len(masks), dtype=dtype)
            for player in players:
                values *= coordinates[:, player]
        output[:, column] = values
    return output


def fourier_to_presence(
    intercept: float,
    terms: Sequence[int],
    coefficients: Sequence[float],
) -> Tuple[float, Dict[int, float]]:
    """Expand parity terms into presence-Mobius monomials."""

    baseline = float(intercept)
    presence: Dict[int, float] = {}
    for term, coefficient in zip(terms, coefficients):
        players = term_players(int(term))
        for degree in range(len(players) + 1):
            for subset in itertools.combinations(players, degree):
                subset_term = 0
                for player in subset:
                    subset_term |= 1 << player
                contribution = float(coefficient) * ((-2.0) ** degree)
                if subset_term:
                    presence[subset_term] = presence.get(subset_term, 0.0) + contribution
                else:
                    baseline += contribution
    return baseline, {
        term: value for term, value in presence.items() if abs(value) > 1e-15
    }

