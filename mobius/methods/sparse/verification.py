"""Targeted exact verification of fitted deletion-Mobius coefficients."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Sequence

import numpy as np

from mobius.models.oracle import QueryLedger
from mobius.values.classification import attribution_values

from .basis import term_players
from .estimator import SparseModel


def verify_deletion_coefficients(
    model: SparseModel,
    *,
    n_players: int,
    top_k: int,
    score_masks,
    target_class: int,
    value_function: str,
    ledger: QueryLedger,
) -> Dict[str, Any]:
    """Exactly verify top deletion coefficients using all local deletion subsets."""

    if model.basis != "deletion_mobius":
        return {
            "status": "not_applicable",
            "reason": "Exact local deletion verification is defined for deletion_mobius.",
            "verified_count": 0,
        }
    ranked = sorted(
        model.coefficient_dict().items(),
        key=lambda item: (-abs(float(item[1])), int(item[0])),
    )[: max(0, int(top_k))]
    full_mask = (1 << int(n_players)) - 1
    rows = []
    for term, estimate in ranked:
        players = term_players(term)
        masks = []
        signs = []
        for degree in range(len(players) + 1):
            for deleted_players in itertools.combinations(players, degree):
                deleted_mask = 0
                for player in deleted_players:
                    deleted_mask |= 1 << player
                masks.append(full_mask & ~deleted_mask)
                signs.append((-1.0) ** (len(players) - degree))
        label_scores = score_masks(
            masks,
            ledger=ledger,
            category="interaction_verification",
        )
        values = attribution_values(
            label_scores,
            target_class=target_class,
            value_function=value_function,
        )
        truth = float(np.dot(np.asarray(signs, dtype=np.float64), values))
        rows.append(
            {
                "term": int(term),
                "players": list(players),
                "degree": len(players),
                "estimated_coefficient": float(estimate),
                "true_coefficient": truth,
                "absolute_error": abs(float(estimate) - truth),
                "sign_match": bool(np.sign(float(estimate)) == np.sign(truth)),
                "masks": [int(mask) for mask in masks],
                "values": [float(value) for value in values],
            }
        )
    return {
        "status": "ok",
        "definition": "g(D)=f(N\\D); theta(T)=sum_{U subseteq T}(-1)^(|T|-|U|)g(U)",
        "verified_count": len(rows),
        "sign_accuracy": (
            float(np.mean([row["sign_match"] for row in rows])) if rows else None
        ),
        "mean_absolute_error": (
            float(np.mean([row["absolute_error"] for row in rows])) if rows else None
        ),
        "rows": rows,
    }


def verification_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove query-level rows from standard-output verification diagnostics."""

    return {
        key: value
        for key, value in result.items()
        if key != "rows"
    }

