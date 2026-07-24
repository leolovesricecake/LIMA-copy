"""Candidate policies for unconstrained and strong-heredity recovery."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .basis import low_degree_terms, term_players
from .estimator import SparseModel, fit_sparse_model


HIERARCHIES = {"none", "strong"}


def normalize_hierarchy(value: str | None) -> str:
    """Normalize hierarchy-policy names."""

    normalized = str(value or "none").strip().lower()
    if normalized not in HIERARCHIES:
        raise ValueError(
            f"Unsupported hierarchy {value!r}; expected {sorted(HIERARCHIES)}"
        )
    return normalized


def _term_from_players(players: Sequence[int]) -> int:
    """Encode one candidate player subset as a bit-mask term."""

    term = 0
    for player in players:
        term |= 1 << int(player)
    return term


def strong_heredity_terms(
    selected_singletons: Sequence[int],
    *,
    max_degree: int,
) -> List[int]:
    """Keep selected singletons and interactions whose every parent is selected."""

    players = sorted(
        term_players(term)[0]
        for term in selected_singletons
        if int(term).bit_count() == 1
    )
    terms: List[int] = []
    for degree in range(1, min(len(players), int(max_degree)) + 1):
        terms.extend(
            _term_from_players(subset)
            for subset in itertools.combinations(players, degree)
        )
    return terms


def choose_candidates(
    policy: str,
    masks: Sequence[int],
    values: Sequence[float],
    *,
    n_features: int,
    max_degree: int,
    basis: str,
    estimator_config: Mapping[str, object],
    random_state: int,
) -> Tuple[List[int], Dict[str, Any], SparseModel | None]:
    """Produce final candidates, fitting singleton screening only when required."""

    normalized = normalize_hierarchy(policy)
    all_terms = low_degree_terms(n_features, max_degree)
    if normalized == "none":
        return all_terms, {
            "policy": "none",
            "initial_candidate_count": len(all_terms),
            "final_candidate_count": len(all_terms),
        }, None
    singleton_terms = low_degree_terms(n_features, 1)
    screening = fit_sparse_model(
        masks,
        values,
        n_features=n_features,
        terms=singleton_terms,
        basis=basis,
        max_degree=1,
        config=estimator_config,
        random_state=random_state,
    )
    selected = sorted(screening.coefficient_dict(selection=True))
    final_terms = strong_heredity_terms(selected, max_degree=max_degree)
    return final_terms, {
        "policy": "strong",
        "screening_candidate_count": len(singleton_terms),
        "selected_singleton_count": len(selected),
        "selected_singleton_terms": selected,
        "initial_candidate_count": len(all_terms),
        "final_candidate_count": len(final_terms),
        "screening_fit": dict(screening.diagnostics),
    }, screening

