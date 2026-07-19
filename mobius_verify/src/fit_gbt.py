from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np

from .reconstruction_metrics import mae, normalized_rmse, r2_score
from .subset_enumeration import masks_to_matrix


def fit_sklearn_gbt(
    *,
    train_masks: Sequence[int],
    y_train: Sequence[float],
    validation_masks: Sequence[int],
    y_validation: Sequence[float],
    test_masks: Sequence[int],
    y_test: Sequence[float],
    n_features: int,
    random_state: int = 0,
    param_grid: Dict[str, Sequence[object]] | None = None,
) -> Dict[str, object]:
    from sklearn.ensemble import GradientBoostingRegressor

    grid = param_grid or {
        "max_depth": [2, 3, 5],
        "n_estimators": [100, 300],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
    }
    X_train = masks_to_matrix(train_masks, n_features)
    X_val = masks_to_matrix(validation_masks, n_features)
    X_test = masks_to_matrix(test_masks, n_features)
    ytr = np.asarray(y_train, dtype=np.float64)
    yv = np.asarray(y_validation, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)

    keys = list(grid)
    best = None
    for values in product(*[grid[key] for key in keys]):
        params = dict(zip(keys, values))
        model = GradientBoostingRegressor(random_state=int(random_state), **params)
        model.fit(X_train, ytr)
        pred_val = model.predict(X_val)
        score = r2_score(yv, pred_val)
        if best is None or score > best["validation_r2"]:
            best = {"model": model, "params": params, "validation_r2": float(score)}

    pred = best["model"].predict(X_test)
    return {
        "method": "sklearn_gbt",
        "status": "ok",
        "best_params": best["params"],
        "validation_r2": float(best["validation_r2"]),
        "test_r2": r2_score(yt, pred),
        "test_normalized_rmse": normalized_rmse(yt, pred),
        "test_mae": mae(yt, pred),
    }

