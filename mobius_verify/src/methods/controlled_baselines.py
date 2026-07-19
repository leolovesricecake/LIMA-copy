from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Sequence

import numpy as np

from ..subset_enumeration import masks_to_matrix


@dataclass
class GBTSurrogateModel:
    n_features: int
    estimator: object
    diagnostics: Dict[str, object]

    def predict(self, masks: Sequence[int]) -> np.ndarray:
        matrix = masks_to_matrix(masks, self.n_features)
        return np.asarray(self.estimator.predict(matrix), dtype=np.float64)

    def node_scores(self) -> np.ndarray:
        values = getattr(self.estimator, "feature_importances_", np.zeros(self.n_features))
        return np.asarray(values, dtype=np.float64)

    def to_dict(self) -> Dict[str, object]:
        return {
            "basis": "tree_proxy",
            "n_features": int(self.n_features),
            "coefficient_count": int(np.sum(np.abs(self.node_scores()) > 1e-12)),
            "diagnostics": self.diagnostics,
            "node_scores": [float(value) for value in self.node_scores()],
        }


def fit_gbt_surrogate(
    *,
    masks: Sequence[int],
    values: Sequence[float],
    n_features: int,
    random_state: int,
    param_grid: Dict[str, Sequence[object]] | None = None,
    cv_folds: int = 3,
) -> GBTSurrogateModel:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    grid = param_grid or {
        "max_depth": [2, 3],
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "subsample": [1.0],
    }
    matrix = masks_to_matrix(masks, int(n_features))
    y = np.asarray(values, dtype=np.float64)
    keys = list(grid)
    if len(y) >= 4:
        splitter = KFold(
            n_splits=min(int(cv_folds), max(2, len(y) // 2)),
            shuffle=True,
            random_state=int(random_state),
        )
        splits = list(splitter.split(matrix))
    else:
        splits = [(np.arange(len(y)), np.arange(len(y)))]
    best = None
    for values_tuple in product(*[grid[key] for key in keys]):
        params = dict(zip(keys, values_tuple))
        losses = []
        for train_idx, val_idx in splits:
            estimator = GradientBoostingRegressor(random_state=int(random_state), **params)
            estimator.fit(matrix[train_idx], y[train_idx])
            prediction = estimator.predict(matrix[val_idx])
            losses.append(float(np.mean((y[val_idx] - prediction) ** 2)))
        candidate = (float(np.mean(losses)), tuple(str(params[key]) for key in keys), params)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    estimator = GradientBoostingRegressor(random_state=int(random_state), **best[2])
    estimator.fit(matrix, y)
    return GBTSurrogateModel(
        n_features=int(n_features),
        estimator=estimator,
        diagnostics={"cv_mse": float(best[0]), "best_params": best[2]},
    )
