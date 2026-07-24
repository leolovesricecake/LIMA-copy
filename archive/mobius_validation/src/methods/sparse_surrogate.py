from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence
import warnings

import numpy as np

from ..designs import (
    EmpiricalStandardizer,
    design_matrix,
    empirical_design_diagnostics,
    estimated_design_megabytes,
    low_degree_terms,
    normalize_basis,
)
from ..reconstruction_metrics import mae, normalized_rmse, r2_score


@dataclass
class SparseSurrogateModel:
    basis: str
    n_features: int
    max_degree: int
    intercept: float
    terms: list[int]
    coefficients: np.ndarray
    selection_coefficients: np.ndarray
    diagnostics: Dict[str, object]

    def predict(self, masks: Sequence[int], *, batch_size: int = 2048) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for start in range(0, len(masks), max(1, int(batch_size))):
            batch = list(masks[start : start + max(1, int(batch_size))])
            matrix = design_matrix(
                batch,
                self.terms,
                n_features=self.n_features,
                basis=self.basis,
            )
            predictions.append(self.intercept + matrix @ self.coefficients)
        return np.concatenate(predictions).astype(np.float64) if predictions else np.zeros(0)

    def coefficient_dict(self, *, tolerance: float = 1e-12) -> Dict[int, float]:
        return {
            int(term): float(coefficient)
            for term, coefficient in zip(self.terms, self.coefficients)
            if abs(float(coefficient)) > float(tolerance)
        }

    def to_dict(self) -> Dict[str, object]:
        hyperedges = [
            {
                "term": int(term),
                "players": [idx for idx in range(int(term).bit_length()) if int(term) & (1 << idx)],
                "degree": int(int(term).bit_count()),
                "coefficient": float(coefficient),
                "selection_coefficient": float(selection),
            }
            for term, coefficient, selection in zip(
                self.terms, self.coefficients, self.selection_coefficients
            )
            if abs(float(coefficient)) > 1e-12 or abs(float(selection)) > 1e-12
        ]
        return {
            "basis": self.basis,
            "n_features": int(self.n_features),
            "max_degree": int(self.max_degree),
            "intercept": float(self.intercept),
            "candidate_count": int(len(self.terms)),
            "coefficient_count": int(len(self.coefficient_dict())),
            "hyperedges": hyperedges,
            "diagnostics": self.diagnostics,
        }


def _new_sparse_model(alpha: float, l1_ratio: float, random_state: int):
    if float(l1_ratio) >= 1.0 - 1e-12:
        from sklearn.linear_model import Lasso

        return Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=50000,
            tol=1e-6,
            selection="cyclic",
        )
    from sklearn.linear_model import ElasticNet

    return ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=50000,
        tol=1e-6,
        selection="cyclic",
        random_state=int(random_state),
    )


def _choose_regularization(
    matrix: np.ndarray,
    values: np.ndarray,
    *,
    alphas: Sequence[float],
    l1_ratios: Sequence[float],
    cv_folds: int,
    random_state: int,
) -> tuple[float, float, float | None]:
    from sklearn.model_selection import KFold

    if len(values) < 4:
        return float(min(alphas)), float(max(l1_ratios)), None
    splits = min(int(cv_folds), max(2, len(values) // 2))
    splitter = KFold(n_splits=splits, shuffle=True, random_state=int(random_state))
    best: tuple[float, float, float] | None = None
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            losses = []
            for train_idx, val_idx in splitter.split(matrix):
                model = _new_sparse_model(float(alpha), float(l1_ratio), random_state)
                with warnings.catch_warnings():
                    from sklearn.exceptions import ConvergenceWarning

                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(matrix[train_idx], values[train_idx])
                pred = model.predict(matrix[val_idx])
                losses.append(float(np.mean((values[val_idx] - pred) ** 2)))
            loss = float(np.mean(losses))
            candidate = (loss, float(alpha), float(l1_ratio))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def fit_sparse_surrogate(
    *,
    masks: Sequence[int],
    values: Sequence[float],
    n_features: int,
    basis: str,
    max_degree: int = 2,
    alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1),
    l1_ratios: Sequence[float] = (1.0,),
    cv_folds: int = 5,
    random_state: int = 0,
    coefficient_tolerance: float = 1e-9,
    ridge_alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0, 10.0),
    max_design_mb: float = 2048.0,
) -> SparseSurrogateModel:
    normalized_basis = normalize_basis(basis)
    y = np.asarray(values, dtype=np.float64)
    if len(masks) != len(y):
        raise ValueError("masks and values must have equal length")
    if len(y) < 2:
        raise ValueError("At least two observations are required")
    terms = low_degree_terms(int(n_features), int(max_degree))
    estimated_mb = estimated_design_megabytes(len(masks), len(terms))
    if estimated_mb > float(max_design_mb):
        raise MemoryError(
            f"Design requires approximately {estimated_mb:.1f} MiB, exceeding max_design_mb={max_design_mb}"
        )
    raw = design_matrix(
        masks,
        terms,
        n_features=int(n_features),
        basis=normalized_basis,
    )
    standardizer = EmpiricalStandardizer.fit(raw)
    standardized = standardizer.transform(raw)
    identifiable_terms = [term for term, keep in zip(terms, standardizer.identifiable) if bool(keep)]
    diagnostics: Dict[str, object] = {
        "estimated_design_mb": float(estimated_mb),
        "candidate_count": int(len(terms)),
        "identifiable_candidate_count": int(len(identifiable_terms)),
        "unidentifiable_candidate_count": int(len(terms) - len(identifiable_terms)),
        **empirical_design_diagnostics(standardized),
    }
    if standardized.shape[1] == 0:
        return SparseSurrogateModel(
            basis=normalized_basis,
            n_features=int(n_features),
            max_degree=int(max_degree),
            intercept=float(np.mean(y)),
            terms=terms,
            coefficients=np.zeros(len(terms), dtype=np.float64),
            selection_coefficients=np.zeros(len(terms), dtype=np.float64),
            diagnostics={**diagnostics, "status": "constant_design", "selected_support_size": 0},
        )

    alpha, l1_ratio, cv_mse = _choose_regularization(
        standardized,
        y,
        alphas=[float(value) for value in alphas],
        l1_ratios=[float(value) for value in l1_ratios],
        cv_folds=int(cv_folds),
        random_state=int(random_state),
    )
    selector = _new_sparse_model(alpha, l1_ratio, random_state)
    with warnings.catch_warnings(record=True) as caught_warnings:
        from sklearn.exceptions import ConvergenceWarning

        warnings.simplefilter("always", ConvergenceWarning)
        selector.fit(standardized, y)
    convergence_warning_count = sum(
        issubclass(warning.category, ConvergenceWarning) for warning in caught_warnings
    )
    selection_all, selection_intercept = standardizer.original_coefficients(
        selector.coef_, float(selector.intercept_)
    )
    support = np.flatnonzero(np.abs(selection_all) > float(coefficient_tolerance))

    final_coefficients = np.zeros(len(terms), dtype=np.float64)
    final_intercept = float(np.mean(y))
    refit_kind = "constant"
    ridge_alpha = None
    support_condition = None
    ols_coefficients = None
    if len(support) > 0:
        from sklearn.linear_model import RidgeCV

        selected_raw = raw[:, support].astype(np.float64)
        ridge = RidgeCV(alphas=np.asarray(list(ridge_alphas), dtype=np.float64), fit_intercept=True)
        ridge.fit(selected_raw, y)
        final_coefficients[support] = np.asarray(ridge.coef_, dtype=np.float64)
        final_intercept = float(ridge.intercept_)
        ridge_alpha = float(ridge.alpha_)
        refit_kind = "ridge"
        try:
            centered = selected_raw - np.mean(selected_raw, axis=0, keepdims=True)
            support_condition = float(np.linalg.cond(centered))
        except Exception:
            support_condition = None
        if len(y) > len(support) + 1 and support_condition is not None and support_condition < 1e8:
            augmented = np.column_stack([np.ones(len(y)), selected_raw])
            solution, *_ = np.linalg.lstsq(augmented, y, rcond=None)
            ols_coefficients = {
                "intercept": float(solution[0]),
                "coefficients": [float(value) for value in solution[1:]],
            }

    train_prediction = final_intercept + raw @ final_coefficients
    diagnostics.update(
        {
            "status": "ok",
            "best_alpha": float(alpha),
            "best_l1_ratio": float(l1_ratio),
            "cv_mse": cv_mse,
            "selector_converged": bool(getattr(selector, "n_iter_", 0) < getattr(selector, "max_iter", 50000)),
            "selector_iterations": int(getattr(selector, "n_iter_", 0)),
            "selector_convergence_warning_count": int(convergence_warning_count),
            "selected_support_size": int(len(support)),
            "selected_support_condition_number": support_condition,
            "refit_kind": refit_kind,
            "ridge_alpha": ridge_alpha,
            "ols_debias_available": ols_coefficients is not None,
            "ols_debias": ols_coefficients,
            "selection_intercept": float(selection_intercept),
            "train_r2": r2_score(y, train_prediction),
            "train_normalized_rmse": normalized_rmse(y, train_prediction),
            "train_mae": mae(y, train_prediction),
        }
    )
    return SparseSurrogateModel(
        basis=normalized_basis,
        n_features=int(n_features),
        max_degree=int(max_degree),
        intercept=float(final_intercept),
        terms=terms,
        coefficients=final_coefficients,
        selection_coefficients=selection_all,
        diagnostics=diagnostics,
    )
