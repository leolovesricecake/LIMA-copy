"""Hyperedge-to-node policies, including deletion-game Shapley allocation.

For a deletion polynomial ``g(D)=theta_0+sum_T theta_T 1[T subset D]``,
the Shapley value of deletion player ``i`` is ``sum_{T contains i}
theta_T/|T|``. Feature presence has the opposite direction, so
``signed_equal_share`` returns its negative. This allocation is the canonical
Harsanyi-dividend allocation satisfying efficiency, symmetry, linearity, and
the dummy-player property; it is not claimed to optimize a faithfulness metric.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .basis import fourier_to_presence, term_players
from .estimator import SparseModel


PROJECTORS = {"signed_equal_share", "absolute_equal_share", "singleton_only"}


def normalize_projector(value: str | None) -> str:
    """Normalize projector names and aliases."""

    normalized = str(value or "signed_equal_share").strip().lower()
    aliases = {"signed": "signed_equal_share", "absolute": "absolute_equal_share"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROJECTORS:
        raise ValueError(
            f"Unsupported projector {value!r}; expected {sorted(PROJECTORS)}"
        )
    return normalized


def _signed_coefficients(model: SparseModel) -> tuple[Dict[int, float], float]:
    """Convert fitted coefficients to directional node-attribution coordinates."""

    coefficients = model.coefficient_dict()
    if model.basis == "fourier":
        _, presence = fourier_to_presence(
            model.intercept,
            model.terms,
            model.coefficients,
        )
        return presence, 1.0
    if model.basis == "deletion_mobius":
        return coefficients, -1.0
    raise ValueError(f"Unsupported fitted basis {model.basis!r}")


def project_nodes(model: SparseModel, projector: str) -> np.ndarray:
    """Allocate hyperedges, using negated deletion-Shapley for signed mode."""

    mode = normalize_projector(projector)
    scores = np.zeros(model.n_features, dtype=np.float64)
    if mode == "absolute_equal_share":
        coefficients = model.coefficient_dict()
        direction = 1.0
    else:
        coefficients, direction = _signed_coefficients(model)
    for term, coefficient in coefficients.items():
        players = term_players(term)
        if not players:
            continue
        if mode == "singleton_only" and len(players) != 1:
            continue
        contribution = (
            abs(float(coefficient))
            if mode == "absolute_equal_share"
            else direction * float(coefficient)
        )
        share = contribution / len(players)
        for player in players:
            scores[player] += share
    return scores
