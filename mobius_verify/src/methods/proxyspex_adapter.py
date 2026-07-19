from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..designs import equal_share_node_scores
from ..query_ledger import QueryLedger
from ..schema import FeatureSpec
from ..subset_enumeration import masks_to_matrix
from ..value_oracle import ValueOracle


REPO_ROOT = Path(__file__).resolve().parents[3]
SHAPIQ_COPY_SRC = REPO_ROOT / "baselines" / "shapiq-copy" / "src"


def _import_proxyspex():
    if str(SHAPIQ_COPY_SRC) not in sys.path:
        sys.path.insert(0, str(SHAPIQ_COPY_SRC))
    from shapiq.approximator.proxy.proxyspex import ProxySPEX

    if not hasattr(ProxySPEX, "approximate_from_observations"):
        raise RuntimeError(
            "Loaded ProxySPEX does not expose approximate_from_observations; "
            "ensure baselines/shapiq-copy/src is imported before another shapiq installation."
        )
    return ProxySPEX


def tuple_to_term(interaction: Sequence[int]) -> int:
    term = 0
    for player in interaction:
        term |= 1 << int(player)
    return term


@dataclass
class ProxySPEXSurrogateModel:
    n_features: int
    approximator: object
    interaction_values: object
    train_masks: list[int]

    def predict(self, masks: Sequence[int]) -> np.ndarray:
        matrix = masks_to_matrix(masks, self.n_features).astype(bool)
        return np.asarray(self.approximator.predict_refined_fourier(matrix), dtype=np.float64)

    def presence_coefficients(self, *, max_degree: int | None = None) -> Dict[int, float]:
        raw = dict(getattr(self.approximator, "moebius_transform_", {}) or {})
        out = {}
        for interaction, value in raw.items():
            if not interaction:
                continue
            if max_degree is not None and len(interaction) > int(max_degree):
                continue
            out[tuple_to_term(interaction)] = float(value)
        return out

    def node_scores(self, *, max_degree: int | None = None) -> np.ndarray:
        return equal_share_node_scores(
            n_features=self.n_features,
            coefficients=self.presence_coefficients(max_degree=max_degree),
            orientation="presence_mobius",
        )

    def to_dict(self) -> Dict[str, object]:
        refined = dict(getattr(self.approximator, "refined_fourier_", {}) or {})
        mobius = self.presence_coefficients()
        return {
            "basis": "proxyspex_refined_fourier",
            "n_features": int(self.n_features),
            "train_masks": [int(mask) for mask in self.train_masks],
            "coefficient_count": int(len(mobius)),
            "refined_fourier_count": int(len(refined)),
            "refined_fourier": [
                {"players": [int(x) for x in key], "coefficient": float(value)}
                for key, value in sorted(refined.items(), key=lambda item: (len(item[0]), item[0]))
            ],
            "presence_mobius": [
                {
                    "term": int(term),
                    "players": [idx for idx in range(term.bit_length()) if term & (1 << idx)],
                    "coefficient": float(value),
                }
                for term, value in sorted(mobius.items())
            ],
            "node_scores": [float(value) for value in self.node_scores(max_degree=2)],
        }


class _OracleGame:
    def __init__(
        self,
        *,
        oracle: ValueOracle,
        feature_spec: FeatureSpec,
        target_class: int,
        value_type: str,
        ledger: QueryLedger,
        operator: str,
    ) -> None:
        self.oracle = oracle
        self.feature_spec = feature_spec
        self.target_class = int(target_class)
        self.value_type = str(value_type)
        self.ledger = ledger
        self.operator = str(operator)
        self.masks: list[int] = []

    def __call__(self, coalitions_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(coalitions_matrix, dtype=bool)
        masks = []
        for row in matrix:
            mask = 0
            for idx, keep in enumerate(row):
                if bool(keep):
                    mask |= 1 << idx
            masks.append(mask)
        self.masks.extend(int(mask) for mask in masks)
        values, _ = self.oracle.values_for_masks(
            self.feature_spec,
            masks,
            target_class=self.target_class,
            value_type=self.value_type,
            ledger=self.ledger,
            category="training",
            operator=self.operator,
        )
        return values


def _new_approximator(
    *,
    n_features: int,
    max_order: int,
    index: str,
    proxy_model: str,
    hpo: bool,
    expected_observations: int,
    random_state: int,
):
    ProxySPEX = _import_proxyspex()
    n_observations = int(expected_observations)
    effective_hpo = bool(hpo) and n_observations >= 4
    approximator = ProxySPEX(
        n=int(n_features),
        max_order=min(int(max_order), int(n_features)),
        index=str(index),
        proxy_model=str(proxy_model),
        hpo=effective_hpo,
        random_state=int(random_state),
    )
    if effective_hpo:
        from sklearn.model_selection import KFold

        splits = min(5, max(2, n_observations // 2))
        cv = KFold(n_splits=splits, shuffle=True, random_state=int(random_state))
        proxy = approximator.proxy_model
        if hasattr(proxy, "cv"):
            proxy.cv = cv
        elif hasattr(proxy, "search") and hasattr(proxy.search, "cv"):
            proxy.search.cv = cv
    return approximator


def fit_proxyspex_from_observations(
    *,
    masks: Sequence[int],
    values: Sequence[float],
    n_features: int,
    max_order: int = 2,
    index: str = "FBII",
    proxy_model: str = "tree",
    hpo: bool = False,
    random_state: int = 0,
) -> ProxySPEXSurrogateModel:
    approximator = _new_approximator(
        n_features=int(n_features),
        max_order=int(max_order),
        index=index,
        proxy_model=proxy_model,
        hpo=hpo,
        expected_observations=len(masks),
        random_state=int(random_state),
    )
    matrix = masks_to_matrix(masks, int(n_features)).astype(bool)
    interactions = approximator.approximate_from_observations(
        matrix,
        np.asarray(values, dtype=np.float64),
        estimation_budget=len(masks),
    )
    return ProxySPEXSurrogateModel(
        n_features=int(n_features),
        approximator=approximator,
        interaction_values=interactions,
        train_masks=[int(mask) for mask in masks],
    )


def fit_proxyspex_native(
    *,
    oracle: ValueOracle,
    feature_spec: FeatureSpec,
    target_class: int,
    value_type: str,
    ledger: QueryLedger,
    budget: int,
    operator: str = "delete",
    max_order: int = 2,
    index: str = "FBII",
    proxy_model: str = "tree",
    hpo: bool = False,
    random_state: int = 0,
) -> ProxySPEXSurrogateModel:
    n_features = feature_spec.n_features
    expected = min(int(budget), 1 << int(n_features))
    approximator = _new_approximator(
        n_features=n_features,
        max_order=max_order,
        index=index,
        proxy_model=proxy_model,
        hpo=hpo,
        expected_observations=expected,
        random_state=random_state,
    )
    game = _OracleGame(
        oracle=oracle,
        feature_spec=feature_spec,
        target_class=target_class,
        value_type=value_type,
        ledger=ledger,
        operator=operator,
    )
    interactions = approximator.approximate(int(budget), game)
    return ProxySPEXSurrogateModel(
        n_features=n_features,
        approximator=approximator,
        interaction_values=interactions,
        train_masks=sorted(set(game.masks)),
    )
