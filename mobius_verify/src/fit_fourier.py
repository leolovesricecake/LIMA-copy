from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .methods.sparse_surrogate import fit_sparse_surrogate
from .reconstruction_metrics import mae, normalized_rmse, r2_score
from .subset_enumeration import candidate_count


def fit_fourier_lasso(
    *,
    train_masks: Sequence[int],
    y_train: Sequence[float],
    validation_masks: Sequence[int],
    y_validation: Sequence[float],
    test_masks: Sequence[int],
    y_test: Sequence[float],
    n_features: int,
    degrees: Sequence[int],
    alphas: Sequence[float],
    max_candidates: int = 20000,
) -> Dict[str, object]:
    """Legacy recovery wrapper backed by the shared empirical-standardized estimator."""

    y_val = np.asarray(y_validation, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64)
    best = None
    infeasible = []
    for degree in degrees:
        if candidate_count(n_features, int(degree)) > int(max_candidates):
            infeasible.append(int(degree))
            continue
        model = fit_sparse_surrogate(
            masks=train_masks,
            values=y_train,
            n_features=n_features,
            basis="fourier",
            max_degree=int(degree),
            alphas=alphas,
        )
        prediction = model.predict(validation_masks)
        score = r2_score(y_val, prediction)
        if best is None or score > best["validation_r2"]:
            best = {"model": model, "degree": int(degree), "validation_r2": float(score)}
    if best is None:
        return {
            "method": "fourier_lasso",
            "status": "infeasible_due_to_candidate_size",
            "infeasible_degrees": infeasible,
            "max_candidates": int(max_candidates),
        }
    model = best["model"]
    prediction = model.predict(test_masks)
    return {
        "method": "fourier_lasso",
        "status": "ok",
        "best_degree": int(best["degree"]),
        "best_alpha": model.diagnostics.get("best_alpha"),
        "validation_r2": float(best["validation_r2"]),
        "test_r2": r2_score(y_test_arr, prediction),
        "test_normalized_rmse": normalized_rmse(y_test_arr, prediction),
        "test_mae": mae(y_test_arr, prediction),
        "selected_terms": [int(term) for term in model.coefficient_dict()],
        "coefficient_count": int(len(model.coefficient_dict())),
        "candidate_count": int(len(model.terms)),
        "condition_number": model.diagnostics.get("selected_support_condition_number"),
        "fit_diagnostics": model.diagnostics,
    }
