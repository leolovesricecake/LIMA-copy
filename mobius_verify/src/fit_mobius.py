from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .reconstruction_metrics import mae, normalized_rmse, r2_score
from .subset_enumeration import candidate_count, candidate_terms
from .transforms import design_mobius


def fit_mobius_lasso(
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
    from sklearn.linear_model import Lasso

    ytr = np.asarray(y_train, dtype=np.float64)
    yv = np.asarray(y_validation, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)
    best = None
    infeasible = []
    for degree in degrees:
        if candidate_count(n_features, int(degree)) > int(max_candidates):
            infeasible.append(int(degree))
            continue
        terms = candidate_terms(n_features, int(degree))
        scales = np.asarray([2.0 ** (-term.bit_count() / 2.0) for term in terms], dtype=np.float64)
        scales[scales <= 1e-12] = 1.0
        X_train = design_mobius(train_masks, terms) / scales[None, :]
        X_val = design_mobius(validation_masks, terms) / scales[None, :]
        for alpha in alphas:
            model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=30000, tol=1e-5)
            model.fit(X_train, ytr)
            pred_val = model.predict(X_val)
            score = r2_score(yv, pred_val)
            if best is None or score > best["validation_r2"]:
                best = {
                    "model": model,
                    "terms": terms,
                    "scales": scales,
                    "degree": int(degree),
                    "alpha": float(alpha),
                    "validation_r2": float(score),
                    "condition_number": _condition_number(X_train),
                }
    if best is None:
        return {
            "method": "mobius_lasso",
            "status": "infeasible_due_to_candidate_size",
            "infeasible_degrees": infeasible,
            "max_candidates": int(max_candidates),
        }

    X_test = design_mobius(test_masks, best["terms"]) / best["scales"][None, :]
    pred = best["model"].predict(X_test)
    coef_original = np.asarray(best["model"].coef_, dtype=np.float64) / best["scales"]
    selected = [
        int(term) for term, coef in zip(best["terms"], coef_original) if abs(float(coef)) > 1e-9
    ]
    return {
        "method": "mobius_lasso",
        "status": "ok",
        "best_degree": int(best["degree"]),
        "best_alpha": float(best["alpha"]),
        "validation_r2": float(best["validation_r2"]),
        "test_r2": r2_score(yt, pred),
        "test_normalized_rmse": normalized_rmse(yt, pred),
        "test_mae": mae(yt, pred),
        "selected_terms": selected,
        "coefficient_count": int(len(selected)),
        "candidate_count": int(len(best["terms"])),
        "condition_number": best["condition_number"],
    }


def _condition_number(matrix: np.ndarray) -> float | None:
    try:
        if matrix.size == 0:
            return None
        return float(np.linalg.cond(matrix))
    except Exception:
        return None

