from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from .subset_enumeration import masks_to_matrix


BASES = {"presence_mobius", "deletion_mobius", "fourier"}


def normalize_basis(basis: str) -> str:
    normalized = str(basis).strip().lower()
    aliases = {
        "presence": "presence_mobius",
        "mobius": "presence_mobius",
        "deletion": "deletion_mobius",
        "or": "deletion_mobius",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in BASES:
        raise ValueError(f"Unsupported basis: {basis!r}. Expected one of {sorted(BASES)}")
    return normalized


def low_degree_terms(n_features: int, max_degree: int) -> list[int]:
    n = int(n_features)
    degree = min(n, int(max_degree))
    if n < 0 or degree < 1:
        return []
    terms: list[int] = []
    for order in range(1, degree + 1):
        for indices in itertools.combinations(range(n), order):
            term = 0
            for idx in indices:
                term |= 1 << int(idx)
            terms.append(term)
    return terms


def term_players(term: int) -> tuple[int, ...]:
    value = int(term)
    return tuple(idx for idx in range(value.bit_length()) if value & (1 << idx))


def design_matrix(
    masks: Sequence[int],
    terms: Sequence[int],
    *,
    n_features: int,
    basis: str,
    dtype=np.float32,
) -> np.ndarray:
    normalized = normalize_basis(basis)
    rows = masks_to_matrix(masks, int(n_features)).astype(dtype, copy=False)
    if normalized == "deletion_mobius":
        rows = 1.0 - rows
    out = np.empty((len(masks), len(terms)), dtype=dtype)
    for col, term in enumerate(terms):
        players = term_players(int(term))
        if not players:
            out[:, col] = 1.0
        elif normalized == "fourier":
            values = 1.0 - 2.0 * rows[:, players[0]]
            for player in players[1:]:
                values = values * (1.0 - 2.0 * rows[:, player])
            out[:, col] = values
        else:
            values = rows[:, players[0]].copy()
            for player in players[1:]:
                values *= rows[:, player]
            out[:, col] = values
    return out


def estimated_design_megabytes(n_rows: int, n_terms: int, *, dtype=np.float32) -> float:
    return float(int(n_rows) * int(n_terms) * np.dtype(dtype).itemsize / (1024.0**2))


@dataclass(frozen=True)
class EmpiricalStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    identifiable: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, matrix: np.ndarray, *, eps: float = 1e-8) -> "EmpiricalStandardizer":
        values = np.asarray(matrix, dtype=np.float64)
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        identifiable = scale >= float(eps)
        safe_scale = scale.copy()
        safe_scale[~identifiable] = 1.0
        return cls(mean=mean, scale=safe_scale, identifiable=identifiable, eps=float(eps))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        return ((values[:, self.identifiable] - self.mean[self.identifiable]) / self.scale[self.identifiable]).astype(
            np.float32
        )

    def original_coefficients(
        self,
        standardized_coefficients: Sequence[float],
        standardized_intercept: float,
    ) -> tuple[np.ndarray, float]:
        selected = np.asarray(standardized_coefficients, dtype=np.float64)
        if len(selected) != int(np.sum(self.identifiable)):
            raise ValueError("standardized coefficient count does not match identifiable columns")
        original = np.zeros(len(self.mean), dtype=np.float64)
        original[self.identifiable] = selected / self.scale[self.identifiable]
        intercept = float(
            float(standardized_intercept)
            - np.sum(selected * self.mean[self.identifiable] / self.scale[self.identifiable])
        )
        return original, intercept


def empirical_design_diagnostics(matrix: np.ndarray, *, max_columns: int = 512) -> Dict[str, float | int | None]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.size == 0 or values.shape[1] == 0:
        return {"max_coherence": None, "effective_rank": 0, "diagnostic_column_count": 0}
    count = min(int(max_columns), values.shape[1])
    indices = np.linspace(0, values.shape[1] - 1, num=count, dtype=int)
    sample = values[:, indices]
    sample = sample - np.mean(sample, axis=0, keepdims=True)
    norms = np.linalg.norm(sample, axis=0)
    valid = norms > 1e-12
    sample = sample[:, valid]
    norms = norms[valid]
    max_coherence = None
    if sample.shape[1] >= 2:
        normalized = sample / norms[None, :]
        gram = np.abs(normalized.T @ normalized)
        np.fill_diagonal(gram, 0.0)
        max_coherence = float(np.max(gram))
    effective_rank = 0
    if sample.size:
        singular = np.linalg.svd(sample, compute_uv=False)
        if np.sum(singular) > 0:
            probabilities = singular / np.sum(singular)
            entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
            effective_rank = int(round(float(np.exp(entropy))))
    return {
        "max_coherence": max_coherence,
        "effective_rank": int(effective_rank),
        "diagnostic_column_count": int(sample.shape[1]),
    }


def fourier_to_presence_coefficients(
    *,
    intercept: float,
    terms: Sequence[int],
    coefficients: Sequence[float],
) -> tuple[float, Dict[int, float]]:
    baseline = float(intercept)
    mobius: Dict[int, float] = {}
    for term, coefficient in zip(terms, coefficients):
        players = term_players(int(term))
        coef = float(coefficient)
        for order in range(0, len(players) + 1):
            for subset in itertools.combinations(players, order):
                subset_mask = 0
                for player in subset:
                    subset_mask |= 1 << int(player)
                contribution = coef * ((-2.0) ** order)
                if subset_mask == 0:
                    baseline += contribution
                else:
                    mobius[subset_mask] = mobius.get(subset_mask, 0.0) + contribution
    return baseline, {term: value for term, value in mobius.items() if abs(value) > 1e-15}


def equal_share_node_scores(
    *,
    n_features: int,
    coefficients: Dict[int, float],
    orientation: str,
) -> np.ndarray:
    normalized = normalize_basis(orientation)
    if normalized == "fourier":
        raise ValueError("Convert Fourier coefficients to presence Mobius before node allocation")
    direction = -1.0 if normalized == "deletion_mobius" else 1.0
    scores = np.zeros(int(n_features), dtype=np.float64)
    for term, coefficient in coefficients.items():
        players = term_players(int(term))
        if not players:
            continue
        share = direction * float(coefficient) / float(len(players))
        for player in players:
            scores[player] += share
    return scores
