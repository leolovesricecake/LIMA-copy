from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .reconstruction_metrics import mae, normalized_rmse, r2_score
from .subset_enumeration import masks_to_matrix


def fit_additive_lasso(
    *,
    train_masks: Sequence[int],
    y_train: Sequence[float],
    validation_masks: Sequence[int],
    y_validation: Sequence[float],
    test_masks: Sequence[int],
    y_test: Sequence[float],
    n_features: int,
    alphas: Sequence[float],
) -> Dict[str, object]:
    from sklearn.linear_model import Lasso

    X_train = masks_to_matrix(train_masks, n_features)
    X_val = masks_to_matrix(validation_masks, n_features)
    X_test = masks_to_matrix(test_masks, n_features)
    ytr = np.asarray(y_train, dtype=np.float64)
    yv = np.asarray(y_validation, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)

    best = None
    for alpha in alphas:
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000, tol=1e-5)
        model.fit(X_train, ytr)
        pred_val = model.predict(X_val)
        score = r2_score(yv, pred_val)
        if best is None or score > best["validation_r2"]:
            best = {
                "model": model,
                "alpha": float(alpha),
                "validation_r2": float(score),
            }

    model = best["model"]
    pred = model.predict(X_test)
    selected = [idx for idx, coef in enumerate(model.coef_) if abs(float(coef)) > 1e-9]
    return {
        "method": "additive_lasso",
        "status": "ok",
        "best_alpha": float(best["alpha"]),
        "validation_r2": float(best["validation_r2"]),
        "test_r2": r2_score(yt, pred),
        "test_normalized_rmse": normalized_rmse(yt, pred),
        "test_mae": mae(yt, pred),
        "selected_terms": [int(1 << idx) for idx in selected],
        "coefficient_count": int(len(selected)),
    }

