from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .methods.sparse_surrogate import fit_sparse_surrogate
from .reconstruction_metrics import mae, normalized_rmse, r2_score


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
    model = fit_sparse_surrogate(
        masks=train_masks,
        values=y_train,
        n_features=n_features,
        basis="presence_mobius",
        max_degree=1,
        alphas=alphas,
    )
    validation_prediction = model.predict(validation_masks)
    test_prediction = model.predict(test_masks)
    y_val = np.asarray(y_validation, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64)
    return {
        "method": "additive_lasso",
        "status": "ok",
        "best_alpha": model.diagnostics.get("best_alpha"),
        "validation_r2": r2_score(y_val, validation_prediction),
        "test_r2": r2_score(y_test_arr, test_prediction),
        "test_normalized_rmse": normalized_rmse(y_test_arr, test_prediction),
        "test_mae": mae(y_test_arr, test_prediction),
        "selected_terms": [int(term) for term in model.coefficient_dict()],
        "coefficient_count": int(len(model.coefficient_dict())),
        "fit_diagnostics": model.diagnostics,
    }
