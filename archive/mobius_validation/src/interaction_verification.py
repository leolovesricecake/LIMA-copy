from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np

from .designs import low_degree_terms, normalize_basis, term_players
from .query_ledger import QueryLedger
from .schema import FeatureSpec
from .value_oracle import ValueOracle


def targeted_true_coefficient(
    *,
    oracle: ValueOracle,
    feature_spec: FeatureSpec,
    term: int,
    orientation: str,
    target_class: int,
    value_type: str,
    ledger: QueryLedger,
    operator: str = "delete",
) -> Dict[str, object]:
    basis = normalize_basis(orientation)
    if basis not in {"presence_mobius", "deletion_mobius"}:
        raise ValueError("Targeted Mobius verification requires a presence or deletion orientation")
    players = term_players(int(term))
    full = (1 << feature_spec.n_features) - 1
    masks = []
    signs = []
    subset_sizes = []
    for order in range(len(players) + 1):
        for subset in itertools.combinations(players, order):
            subset_mask = 0
            for player in subset:
                subset_mask |= 1 << int(player)
            mask = subset_mask if basis == "presence_mobius" else full & ~subset_mask
            masks.append(int(mask))
            signs.append((-1.0) ** (len(players) - order))
            subset_sizes.append(int(order))
    values, _ = oracle.values_for_masks(
        feature_spec,
        masks,
        target_class=int(target_class),
        value_type=value_type,
        ledger=ledger,
        category="interaction_verification",
        operator=operator,
    )
    coefficient = float(np.dot(np.asarray(signs), values))
    return {
        "term": int(term),
        "players": [int(player) for player in players],
        "degree": int(len(players)),
        "orientation": basis,
        "masks": [int(mask) for mask in masks],
        "subset_sizes": subset_sizes,
        "values": [float(value) for value in values],
        "true_coefficient": coefficient,
    }


def verify_top_hyperedges(
    *,
    oracle: ValueOracle,
    feature_spec: FeatureSpec,
    estimated_coefficients: Dict[int, float],
    orientation: str,
    target_class: int,
    value_type: str,
    ledger: QueryLedger,
    top_k: int = 5,
    random_state: int = 0,
    operator: str = "delete",
) -> Dict[str, object]:
    ranked = sorted(
        ((int(term), float(value)) for term, value in estimated_coefficients.items() if int(term) != 0),
        key=lambda item: (-abs(item[1]), item[0]),
    )[: max(0, int(top_k))]
    rows = []
    for term, estimate in ranked:
        exact = targeted_true_coefficient(
            oracle=oracle,
            feature_spec=feature_spec,
            term=term,
            orientation=orientation,
            target_class=target_class,
            value_type=value_type,
            ledger=ledger,
            operator=operator,
        )
        true = float(exact["true_coefficient"])
        value_range = float(np.ptp(np.asarray(exact["values"], dtype=np.float64)))
        rows.append(
            {
                **exact,
                "estimated_coefficient": estimate,
                "absolute_error": float(abs(estimate - true)),
                "normalized_error": float(abs(estimate - true) / max(value_range, abs(true), 1e-8)),
                "sign_match": bool(np.sign(estimate) == np.sign(true)),
            }
        )

    rng = np.random.default_rng(int(random_state))
    selected = {term for term, _ in ranked}
    random_rows = []
    by_degree: Dict[int, list[int]] = {}
    for term in low_degree_terms(feature_spec.n_features, 2):
        if term not in selected:
            by_degree.setdefault(int(term.bit_count()), []).append(term)
    for term, _ in ranked:
        candidates = by_degree.get(int(term.bit_count()), [])
        if not candidates:
            continue
        random_term = int(candidates.pop(int(rng.integers(0, len(candidates)))))
        random_rows.append(
            targeted_true_coefficient(
                oracle=oracle,
                feature_spec=feature_spec,
                term=random_term,
                orientation=orientation,
                target_class=target_class,
                value_type=value_type,
                ledger=ledger,
                operator=operator,
            )
        )
    return {
        "orientation": normalize_basis(orientation),
        "top": rows,
        "random_control": random_rows,
        "top_mean_abs_true": float(np.mean([abs(float(row["true_coefficient"])) for row in rows]))
        if rows
        else None,
        "random_mean_abs_true": float(
            np.mean([abs(float(row["true_coefficient"])) for row in random_rows])
        )
        if random_rows
        else None,
        "sign_accuracy": float(np.mean([bool(row["sign_match"]) for row in rows]))
        if rows
        else None,
    }
