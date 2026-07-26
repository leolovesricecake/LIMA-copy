"""Candidate and support policies for hierarchy ablations."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple

from .basis import low_degree_terms, term_players

if TYPE_CHECKING:
    from .estimator import SparseModel


HIERARCHIES = {"none", "strict", "parent_screening"}


def normalize_hierarchy(value: str | None) -> str:
    """Normalize hierarchy-policy names."""

    normalized = str(value or "none").strip().lower()
    if normalized == "strong":
        raise ValueError(
            "hierarchy='strong' was renamed. Use 'parent_screening' for the "
            "legacy two-stage candidate policy or 'strict' for support pruning."
        )
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


def strict_support_terms(
    selected_terms: Sequence[int],
    *,
    max_degree: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Prune orphan pairs so the retained selection support has strong heredity."""

    if int(max_degree) > 2:
        raise ValueError("hierarchy='strict' currently supports max_degree <= 2.")
    selected = sorted({int(term) for term in selected_terms})
    singletons = {term for term in selected if term.bit_count() == 1}
    retained: List[int] = []
    removed: List[Dict[str, Any]] = []
    for term in selected:
        players = term_players(term)
        if len(players) <= 1:
            retained.append(term)
            continue
        parents = [1 << int(player) for player in players]
        present = [parent for parent in parents if parent in singletons]
        if len(present) == len(parents):
            retained.append(term)
        else:
            removed.append(
                {
                    "term": int(term),
                    "players": list(players),
                    "parent_terms": parents,
                    "present_parent_count": len(present),
                    "missing_parent_terms": [
                        parent for parent in parents if parent not in singletons
                    ],
                }
            )
    return retained, {
        "policy": "strict",
        "selection_support_size": len(selected),
        "retained_support_size": len(retained),
        "removed_support_size": len(removed),
        "removed_terms": removed,
    }


def apply_hierarchy_support(
    policy: str,
    selected_terms: Sequence[int],
    *,
    max_degree: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply a post-selection support policy and return retained terms."""

    normalized = normalize_hierarchy(policy)
    selected = sorted({int(term) for term in selected_terms})
    if normalized == "strict":
        return strict_support_terms(selected, max_degree=max_degree)
    return selected, {
        "policy": normalized,
        "selection_support_size": len(selected),
        "retained_support_size": len(selected),
        "removed_support_size": 0,
        "removed_terms": [],
    }


def parent_count(
    term: int,
    singleton_support: Sequence[int],
) -> int:
    """Count singleton parents present for one selected interaction term."""

    support = {int(value) for value in singleton_support}
    return sum((1 << int(player)) in support for player in term_players(int(term)))


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
) -> Tuple[List[int], Dict[str, Any], "SparseModel | None"]:
    """Produce final candidates, fitting singleton screening only when required."""

    from .estimator import fit_sparse_model

    normalized = normalize_hierarchy(policy)
    all_terms = low_degree_terms(n_features, max_degree)
    if normalized in {"none", "strict"}:
        return all_terms, {
            "policy": normalized,
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
        "policy": "parent_screening",
        "screening_candidate_count": len(singleton_terms),
        "selected_singleton_count": len(selected),
        "selected_singleton_terms": selected,
        "initial_candidate_count": len(all_terms),
        "final_candidate_count": len(final_terms),
        "screening_fit": dict(screening.diagnostics),
    }, screening
