from __future__ import annotations

import math
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


def resolve_lightgbm_min_child_samples(
    *,
    n_observations: int,
    cv_splits: int | None,
    configured: int | str | None,
) -> tuple[int, int]:
    """Return the leaf minimum and smallest training-fold size used to derive it."""
    count = max(1, int(n_observations))
    if cv_splits is None:
        minimum_train_size = count
    else:
        minimum_train_size = count - int(math.ceil(count / int(cv_splits)))
        minimum_train_size = max(1, minimum_train_size)
    if configured in {None, "", "auto"}:
        min_child_samples = max(1, min(20, minimum_train_size // 4))
    else:
        min_child_samples = max(1, int(configured))
    return int(min_child_samples), int(minimum_train_size)


@dataclass
class ProxySPEXSurrogateModel:
    n_features: int
    approximator: object
    interaction_values: object
    train_masks: list[int]
    diagnostics: Dict[str, object]

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
            "diagnostics": self.diagnostics,
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
    lightgbm_verbosity: int = -1,
    lightgbm_min_child_samples: int | str | None = None,
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
    cv_splits = None
    if effective_hpo:
        from sklearn.model_selection import KFold

        cv_splits = min(5, max(2, n_observations // 2))
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=int(random_state))
        proxy = approximator.proxy_model
        if hasattr(proxy, "cv"):
            proxy.cv = cv
        elif hasattr(proxy, "search") and hasattr(proxy.search, "cv"):
            proxy.search.cv = cv
    proxy = approximator.proxy_model
    base_proxy = getattr(proxy, "estimator", proxy)
    backend = f"{type(base_proxy).__module__}.{type(base_proxy).__name__}"
    lightgbm_settings = None
    if type(base_proxy).__module__.split(".")[0] == "lightgbm":
        min_child_samples, minimum_train_size = resolve_lightgbm_min_child_samples(
            n_observations=n_observations,
            cv_splits=cv_splits,
            configured=lightgbm_min_child_samples,
        )
        base_proxy.set_params(
            verbosity=int(lightgbm_verbosity),
            min_child_samples=int(min_child_samples),
        )
        lightgbm_settings = {
            "verbosity": int(lightgbm_verbosity),
            "min_child_samples": int(min_child_samples),
            "minimum_cv_train_size": int(minimum_train_size),
        }
    approximator._mobius_verify_fit_config = {
        "proxy_backend": backend,
        "hpo_requested": bool(hpo),
        "hpo_effective": bool(effective_hpo and base_proxy is not proxy),
        "cv_splits": cv_splits,
        "expected_observations": n_observations,
        "lightgbm": lightgbm_settings,
    }
    return approximator


def _fit_diagnostics(approximator) -> Dict[str, object]:
    diagnostics = dict(getattr(approximator, "_mobius_verify_fit_config", {}) or {})
    matrix = np.asarray(approximator.coalitions_matrix_, dtype=bool)
    values = np.asarray(approximator.coalition_values_, dtype=np.float64)
    refined_prediction = np.asarray(
        approximator.predict_refined_fourier(matrix), dtype=np.float64
    )
    final_proxy = getattr(approximator, "final_proxy_model_", None)
    proxy_prediction = (
        np.asarray(final_proxy.predict(matrix), dtype=np.float64)
        if final_proxy is not None
        else np.zeros(len(values), dtype=np.float64)
    )
    residual = values - refined_prediction
    denominator = float(np.sum((values - np.mean(values)) ** 2))
    refined_r2 = (
        1.0 - float(np.sum(residual**2)) / denominator
        if denominator > 1e-24
        else float(np.max(np.abs(residual)) <= 1e-12)
    )
    leaf_counts: list[int] = []
    booster = getattr(final_proxy, "booster_", None)
    if booster is not None:
        try:
            leaf_counts = [
                int(tree.get("num_leaves", 0))
                for tree in booster.dump_model().get("tree_info", [])
            ]
        except Exception:
            leaf_counts = []
    proxy_std = float(np.std(proxy_prediction))
    diagnostics.update(
        {
            "final_proxy_backend": (
                f"{type(final_proxy).__module__}.{type(final_proxy).__name__}"
                if final_proxy is not None
                else None
            ),
            "best_params": dict(getattr(approximator.proxy_model, "best_params_", {}) or {}),
            "training_value_std": float(np.std(values)),
            "proxy_training_prediction_std": proxy_std,
            "refined_training_prediction_std": float(np.std(refined_prediction)),
            "refined_training_r2": float(refined_r2),
            "tree_count": int(len(leaf_counts)) if leaf_counts else None,
            "nontrivial_tree_count": (
                int(sum(count > 1 for count in leaf_counts)) if leaf_counts else None
            ),
            "max_tree_leaves": int(max(leaf_counts)) if leaf_counts else None,
            "degenerate_proxy": bool(
                float(np.std(values)) > 1e-12 and proxy_std <= 1e-12
            ),
        }
    )
    return diagnostics


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
    lightgbm_verbosity: int = -1,
    lightgbm_min_child_samples: int | str | None = None,
) -> ProxySPEXSurrogateModel:
    approximator = _new_approximator(
        n_features=int(n_features),
        max_order=int(max_order),
        index=index,
        proxy_model=proxy_model,
        hpo=hpo,
        expected_observations=len(masks),
        random_state=int(random_state),
        lightgbm_verbosity=lightgbm_verbosity,
        lightgbm_min_child_samples=lightgbm_min_child_samples,
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
        diagnostics=_fit_diagnostics(approximator),
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
    lightgbm_verbosity: int = -1,
    lightgbm_min_child_samples: int | str | None = None,
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
        lightgbm_verbosity=lightgbm_verbosity,
        lightgbm_min_child_samples=lightgbm_min_child_samples,
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
        diagnostics=_fit_diagnostics(approximator),
    )
