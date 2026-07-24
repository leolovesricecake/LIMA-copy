"""Sparse support selection followed by stable ridge coefficient refitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .basis import design_matrix, normalize_basis, term_players


@dataclass(frozen=True)
class EmpiricalStandardizer:
    """Store empirical column scaling and identifiable-column flags."""

    mean: np.ndarray
    scale: np.ndarray
    identifiable: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, epsilon: float = 1e-8) -> "EmpiricalStandardizer":
        """Estimate stable empirical mean and scale per design column."""

        values = np.asarray(matrix, dtype=np.float64)
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        identifiable = scale >= float(epsilon)
        safe = scale.copy()
        safe[~identifiable] = 1.0
        return cls(mean, safe, identifiable)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Standardize only empirically identifiable columns."""

        values = np.asarray(matrix, dtype=np.float64)
        return (
            (values[:, self.identifiable] - self.mean[self.identifiable])
            / self.scale[self.identifiable]
        ).astype(np.float32)

    def restore_coefficients(
        self,
        coefficients: Sequence[float],
        intercept: float,
    ) -> tuple[np.ndarray, float]:
        """Convert standardized coefficients back to raw basis coordinates."""

        selected = np.asarray(coefficients, dtype=np.float64)
        output = np.zeros(len(self.mean), dtype=np.float64)
        output[self.identifiable] = selected / self.scale[self.identifiable]
        restored_intercept = float(
            intercept
            - np.sum(
                selected
                * self.mean[self.identifiable]
                / self.scale[self.identifiable]
            )
        )
        return output, restored_intercept


@dataclass
class SparseModel:
    """Store a fitted sparse surrogate in its original basis coordinates."""

    basis: str
    n_features: int
    max_degree: int
    intercept: float
    terms: List[int]
    coefficients: np.ndarray
    selection_coefficients: np.ndarray
    diagnostics: Dict[str, Any]

    def predict(self, masks: Sequence[int]) -> np.ndarray:
        """Predict scalar coalition values for integer keep masks."""

        if not masks:
            return np.zeros(0, dtype=np.float64)
        matrix = design_matrix(
            masks,
            self.terms,
            n_features=self.n_features,
            basis=self.basis,
        )
        return self.intercept + matrix @ self.coefficients

    def coefficient_dict(
        self,
        *,
        tolerance: float = 1e-12,
        selection: bool = False,
    ) -> Dict[int, float]:
        """Return nonzero fitted or selection coefficients by term."""

        values = self.selection_coefficients if selection else self.coefficients
        return {
            int(term): float(coefficient)
            for term, coefficient in zip(self.terms, values)
            if abs(float(coefficient)) > float(tolerance)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize nonzero hyperedges and compact fitting diagnostics."""

        hyperedges = []
        for term, coefficient, selection in zip(
            self.terms,
            self.coefficients,
            self.selection_coefficients,
        ):
            if abs(float(coefficient)) <= 1e-12 and abs(float(selection)) <= 1e-12:
                continue
            hyperedges.append(
                {
                    "term": int(term),
                    "players": list(term_players(term)),
                    "degree": int(term.bit_count()),
                    "coefficient": float(coefficient),
                    "selection_coefficient": float(selection),
                }
            )
        return {
            "basis": self.basis,
            "n_features": int(self.n_features),
            "max_degree": int(self.max_degree),
            "intercept": float(self.intercept),
            "candidate_count": len(self.terms),
            "coefficient_count": len(self.coefficient_dict()),
            "hyperedges": hyperedges,
            "diagnostics": dict(self.diagnostics),
        }


def _sparse_regressor(alpha: float, l1_ratio: float, random_state: int):
    """Construct the fixed Lasso or ElasticNet support selector."""

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
    """Choose regularization by deterministic shuffled K-fold CV."""

    if len(values) < 4:
        return float(min(alphas)), float(max(l1_ratios)), None
    from sklearn.model_selection import KFold

    splits = min(int(cv_folds), max(2, len(values) // 2), len(values))
    splitter = KFold(n_splits=splits, shuffle=True, random_state=int(random_state))
    best: tuple[float, float, float] | None = None
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            losses: List[float] = []
            for train_indices, validation_indices in splitter.split(matrix):
                model = _sparse_regressor(alpha, l1_ratio, random_state)
                with warnings.catch_warnings():
                    from sklearn.exceptions import ConvergenceWarning

                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(matrix[train_indices], values[train_indices])
                prediction = model.predict(matrix[validation_indices])
                losses.append(
                    float(np.mean((values[validation_indices] - prediction) ** 2))
                )
            candidate = (float(np.mean(losses)), float(alpha), float(l1_ratio))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No regularization candidate was evaluated.")
    return best[1], best[2], best[0]


def _fit_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Compute compact in-sample surrogate reconstruction diagnostics."""

    residual = truth - prediction
    mse = float(np.mean(residual**2))
    variance = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / variance if variance > 1e-15 else float(mse <= 1e-15)
    scale = float(np.max(truth) - np.min(truth))
    nrmse = float(np.sqrt(mse) / scale) if scale > 1e-15 else float(np.sqrt(mse))
    return {
        "train_r2": float(r2),
        "train_normalized_rmse": nrmse,
        "train_mae": float(np.mean(np.abs(residual))),
    }


def fit_sparse_model(
    masks: Sequence[int],
    values: Sequence[float],
    *,
    n_features: int,
    terms: Sequence[int],
    basis: str,
    max_degree: int,
    config: Mapping[str, object],
    random_state: int,
) -> SparseModel:
    """Fit the fixed standardize-select-ridge estimator on explicit candidates."""

    normalized_basis = normalize_basis(basis)
    y = np.asarray(values, dtype=np.float64)
    candidate_terms = [int(term) for term in terms]
    if len(masks) != len(y) or len(y) < 2:
        raise ValueError("Fitting requires aligned masks and at least two values.")
    if not candidate_terms:
        return SparseModel(
            normalized_basis,
            int(n_features),
            int(max_degree),
            float(np.mean(y)),
            [],
            np.zeros(0),
            np.zeros(0),
            {
                "status": "constant_no_candidates",
                "candidate_count": 0,
                "selected_support_size": 0,
                **_fit_metrics(y, np.full(len(y), np.mean(y))),
            },
        )
    estimated_mb = (
        len(masks) * len(candidate_terms) * np.dtype(np.float32).itemsize / 1024**2
    )
    maximum_mb = float(config.get("max_design_mb", 2048))
    if estimated_mb > maximum_mb:
        raise MemoryError(
            f"Design needs {estimated_mb:.1f} MiB, above max_design_mb={maximum_mb}."
        )
    raw = design_matrix(
        masks,
        candidate_terms,
        n_features=int(n_features),
        basis=normalized_basis,
    )
    standardizer = EmpiricalStandardizer.fit(raw)
    standardized = standardizer.transform(raw)
    base_diagnostics: Dict[str, Any] = {
        "estimated_design_mb": estimated_mb,
        "candidate_count": len(candidate_terms),
        "identifiable_candidate_count": int(np.sum(standardizer.identifiable)),
        "unidentifiable_candidate_count": int(np.sum(~standardizer.identifiable)),
    }
    if standardized.shape[1] == 0:
        prediction = np.full(len(y), np.mean(y))
        return SparseModel(
            normalized_basis,
            int(n_features),
            int(max_degree),
            float(np.mean(y)),
            candidate_terms,
            np.zeros(len(candidate_terms)),
            np.zeros(len(candidate_terms)),
            {
                **base_diagnostics,
                "status": "constant_design",
                "selected_support_size": 0,
                **_fit_metrics(y, prediction),
            },
        )
    alphas = [float(value) for value in config.get("alphas", [1e-4, 1e-3, 1e-2, 1e-1])]
    l1_ratios = [float(value) for value in config.get("l1_ratios", [1.0])]
    alpha, l1_ratio, cv_mse = _choose_regularization(
        standardized,
        y,
        alphas=alphas,
        l1_ratios=l1_ratios,
        cv_folds=int(config.get("cv_folds", 3)),
        random_state=int(random_state),
    )
    selector = _sparse_regressor(alpha, l1_ratio, random_state)
    with warnings.catch_warnings(record=True) as caught:
        from sklearn.exceptions import ConvergenceWarning

        warnings.simplefilter("always", ConvergenceWarning)
        selector.fit(standardized, y)
    selection, selection_intercept = standardizer.restore_coefficients(
        selector.coef_,
        float(selector.intercept_),
    )
    tolerance = float(config.get("coefficient_tolerance", 1e-9))
    support = np.flatnonzero(np.abs(selection) > tolerance)
    coefficients = np.zeros(len(candidate_terms), dtype=np.float64)
    intercept = float(np.mean(y))
    ridge_alpha = None
    if len(support):
        from sklearn.linear_model import RidgeCV

        ridge_alphas = np.asarray(
            config.get("ridge_alphas", [1e-6, 1e-4, 1e-2, 1.0, 10.0]),
            dtype=np.float64,
        )
        ridge = RidgeCV(alphas=ridge_alphas, fit_intercept=True)
        ridge.fit(raw[:, support], y)
        coefficients[support] = np.asarray(ridge.coef_, dtype=np.float64)
        intercept = float(ridge.intercept_)
        ridge_alpha = float(ridge.alpha_)
    prediction = intercept + raw @ coefficients
    convergence_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    diagnostics = {
        **base_diagnostics,
        "status": "ok",
        "best_alpha": alpha,
        "best_l1_ratio": l1_ratio,
        "cv_mse": cv_mse,
        "selection_intercept": selection_intercept,
        "selected_support_size": len(support),
        "ridge_alpha": ridge_alpha,
        "selector_iterations": int(getattr(selector, "n_iter_", 0)),
        "selector_convergence_warning_count": convergence_count,
        **_fit_metrics(y, prediction),
    }
    return SparseModel(
        normalized_basis,
        int(n_features),
        int(max_degree),
        intercept,
        candidate_terms,
        coefficients,
        selection,
        diagnostics,
    )

