"""Sparse support selection followed by stable ridge coefficient refitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .basis import design_matrix, normalize_basis, term_players
from .hierarchy import apply_hierarchy_support


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
    selection_support: List[int] = field(default_factory=list)
    hierarchy_support: List[int] = field(default_factory=list)
    refit_support: List[int] = field(default_factory=list)
    refit_mode: str = "ridge_cv"

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

        selected = set(int(term) for term in self.selection_support)
        hierarchy = set(int(term) for term in self.hierarchy_support)
        refit = set(int(term) for term in self.refit_support)
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
                    "degree": bin(int(term)).count("1"),
                    "coefficient": float(coefficient),
                    "selection_coefficient": float(selection),
                    "selected": int(term) in selected,
                    "hierarchy_retained": int(term) in hierarchy,
                    "refit_retained": int(term) in refit,
                }
            )
        return {
            "basis": self.basis,
            "n_features": int(self.n_features),
            "max_degree": int(self.max_degree),
            "intercept": float(self.intercept),
            "candidate_count": len(self.terms),
            "coefficient_count": len(self.coefficient_dict()),
            "candidate_definition": {
                "type": "explicit_low_degree",
                "count": len(self.terms),
                "max_degree": int(self.max_degree),
            },
            "selection_support": _support_payload(self.selection_support),
            "hierarchy_support": _support_payload(self.hierarchy_support),
            "refit_support": _support_payload(self.refit_support),
            "support_counts": {
                "selection": _support_counts(self.selection_support),
                "hierarchy": _support_counts(self.hierarchy_support),
                "refit": _support_counts(self.refit_support),
            },
            "refit_mode": self.refit_mode,
            "hyperedges": hyperedges,
            "diagnostics": dict(self.diagnostics),
        }


def _support_payload(terms: Sequence[int]) -> List[Dict[str, Any]]:
    """Serialize structural support terms independently from fitted coefficients."""

    return [
        {
            "term": int(term),
            "players": list(term_players(int(term))),
            "degree": bin(int(term)).count("1"),
        }
        for term in sorted({int(value) for value in terms})
    ]


def _support_counts(terms: Sequence[int]) -> Dict[str, int]:
    """Count structural support terms overall and by interaction order."""

    values = [int(term) for term in terms]
    counts: Dict[str, int] = {"total": len(values)}
    for term in values:
        key = f"order_{bin(int(term)).count('1')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def sparse_model_from_surrogate(payload: Mapping[str, Any]) -> SparseModel:
    """Reconstruct the fitted sparse predictor needed for offline projection."""

    predictor = dict(payload["predictor"])
    if predictor.get("type") != "sparse_polynomial":
        raise ValueError("Offline sparse projection requires sparse_polynomial.")

    def decode(rows: Sequence[Mapping[str, Any]]) -> List[int]:
        """Decode serialized player lists into integer terms."""

        terms: List[int] = []
        for row in rows:
            term = 0
            for player in row.get("players", []):
                term |= 1 << int(player)
            terms.append(term)
        return terms

    term_rows = list(predictor.get("terms", []))
    terms = decode(term_rows)
    coefficients = np.asarray(
        [float(row["coefficient"]) for row in term_rows],
        dtype=np.float64,
    )
    support = dict(payload.get("support", {}))
    selection_support = decode(support.get("selection", []))
    hierarchy_support = decode(support.get("hierarchy", []))
    refit_support = decode(support.get("refit", []))
    return SparseModel(
        basis=str(predictor["basis"]),
        n_features=int(payload["n_features"]),
        max_degree=int(
            dict(payload.get("candidate_definition", {})).get("max_degree", 0)
        ),
        intercept=float(predictor.get("intercept", 0.0)),
        terms=terms,
        coefficients=coefficients,
        selection_coefficients=np.zeros(len(terms), dtype=np.float64),
        diagnostics=dict(payload.get("fit_diagnostics", {})),
        selection_support=selection_support,
        hierarchy_support=hierarchy_support,
        refit_support=refit_support,
        refit_mode=str(
            dict(payload.get("fit_diagnostics", {})).get(
                "refit_mode",
                "unknown",
            )
        ),
    )


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
    coefficient_tolerance: float,
) -> tuple[float, float, float | None, List[Dict[str, Any]]]:
    """Choose regularization by deterministic shuffled K-fold CV."""

    if len(values) < 4:
        alpha = float(min(alphas))
        ratio = float(max(l1_ratios))
        model = _sparse_regressor(alpha, ratio, random_state)
        model.fit(matrix, values)
        return alpha, ratio, None, [
            {
                "alpha": alpha,
                "l1_ratio": ratio,
                "cv_mse_mean": None,
                "cv_mse_std": None,
                "full_fit_support_size": int(
                    np.sum(np.abs(model.coef_) > float(coefficient_tolerance))
                ),
            }
        ]
    from sklearn.model_selection import KFold

    splits = min(int(cv_folds), max(2, len(values) // 2), len(values))
    splitter = KFold(n_splits=splits, shuffle=True, random_state=int(random_state))
    best: tuple[float, float, float] | None = None
    path: List[Dict[str, Any]] = []
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
            full_model = _sparse_regressor(alpha, l1_ratio, random_state)
            with warnings.catch_warnings():
                from sklearn.exceptions import ConvergenceWarning

                warnings.simplefilter("ignore", ConvergenceWarning)
                full_model.fit(matrix, values)
            path.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "cv_mse_mean": float(np.mean(losses)),
                    "cv_mse_std": float(np.std(losses)),
                    "full_fit_support_size": int(
                        np.sum(
                            np.abs(full_model.coef_)
                            > float(coefficient_tolerance)
                        )
                    ),
                }
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No regularization candidate was evaluated.")
    return best[1], best[2], best[0], path


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
    hierarchy_policy: str = "none",
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
            selection_support=[],
            hierarchy_support=[],
            refit_support=[],
            refit_mode=str(config.get("refit", "ridge_cv")),
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
            selection_support=[],
            hierarchy_support=[],
            refit_support=[],
            refit_mode=str(config.get("refit", "ridge_cv")),
        )
    alphas = [float(value) for value in config.get("alphas", [1e-4, 1e-3, 1e-2, 1e-1])]
    l1_ratios = [float(value) for value in config.get("l1_ratios", [1.0])]
    tolerance = float(config.get("coefficient_tolerance", 1e-9))
    alpha, l1_ratio, cv_mse, regularization_path = _choose_regularization(
        standardized,
        y,
        alphas=alphas,
        l1_ratios=l1_ratios,
        cv_folds=int(config.get("cv_folds", 3)),
        random_state=int(random_state),
        coefficient_tolerance=tolerance,
    )
    alpha_scale = float(config.get("selection_alpha_scale", 1.0))
    if alpha_scale <= 0:
        raise ValueError("selection_alpha_scale must be positive.")
    effective_alpha = float(alpha) * alpha_scale
    for path_entry in regularization_path:
        path_entry["cv_selected"] = bool(
            np.isclose(float(path_entry["alpha"]), float(alpha))
            and np.isclose(float(path_entry["l1_ratio"]), float(l1_ratio))
        )
    selector = _sparse_regressor(effective_alpha, l1_ratio, random_state)
    with warnings.catch_warnings(record=True) as caught:
        from sklearn.exceptions import ConvergenceWarning

        warnings.simplefilter("always", ConvergenceWarning)
        selector.fit(standardized, y)
    selection, selection_intercept = standardizer.restore_coefficients(
        selector.coef_,
        float(selector.intercept_),
    )
    selection_indices = np.flatnonzero(np.abs(selection) > tolerance)
    selection_support = [candidate_terms[int(index)] for index in selection_indices]
    hierarchy_support, hierarchy_diagnostics = apply_hierarchy_support(
        hierarchy_policy,
        selection_support,
        max_degree=int(max_degree),
    )
    hierarchy_set = set(hierarchy_support)
    support = np.asarray(
        [
            index
            for index, term in enumerate(candidate_terms)
            if term in hierarchy_set
        ],
        dtype=int,
    )
    coefficients = np.zeros(len(candidate_terms), dtype=np.float64)
    intercept = float(np.mean(y))
    ridge_alpha = None
    refit_mode = str(config.get("refit", "ridge_cv")).strip().lower()
    if refit_mode not in {"none", "ridge_cv", "ridge_fixed", "ols"}:
        raise ValueError(
            "estimator.refit must be none, ridge_cv, ridge_fixed, or ols."
        )
    refit_diagnostics: Dict[str, Any] = {"refit_mode": refit_mode}
    if len(support) and refit_mode == "none":
        coefficients[support] = selection[support]
        intercept = float(selection_intercept)
    elif len(support) and refit_mode in {"ridge_cv", "ridge_fixed"}:
        from sklearn.linear_model import Ridge, RidgeCV

        if refit_mode == "ridge_cv":
            ridge_alphas = np.asarray(
                config.get("ridge_alphas", [1e-6, 1e-4, 1e-2, 1.0, 10.0]),
                dtype=np.float64,
            )
            ridge = RidgeCV(alphas=ridge_alphas, fit_intercept=True)
        else:
            configured_alpha = config.get("ridge_alpha")
            if configured_alpha is None or float(configured_alpha) <= 0:
                raise ValueError(
                    "estimator.ridge_alpha must be positive for ridge_fixed."
                )
            ridge = Ridge(alpha=float(configured_alpha), fit_intercept=True)
        ridge.fit(raw[:, support], y)
        coefficients[support] = np.asarray(ridge.coef_, dtype=np.float64)
        intercept = float(ridge.intercept_)
        ridge_alpha = float(getattr(ridge, "alpha_", getattr(ridge, "alpha", 0.0)))
    elif len(support) and refit_mode == "ols":
        augmented = np.column_stack(
            [np.ones(len(y), dtype=np.float64), raw[:, support]]
        )
        solution, _, rank, singular_values = np.linalg.lstsq(
            augmented,
            y,
            rcond=None,
        )
        intercept = float(solution[0])
        coefficients[support] = np.asarray(solution[1:], dtype=np.float64)
        condition = (
            float(np.max(singular_values) / np.min(singular_values))
            if len(singular_values) and float(np.min(singular_values)) > 0
            else None
        )
        refit_diagnostics.update(
            {
                "ols_rank": int(rank),
                "ols_column_count": int(augmented.shape[1]),
                "ols_rank_deficient": bool(rank < augmented.shape[1]),
                "ols_condition_number": condition,
            }
        )
    prediction = intercept + raw @ coefficients
    convergence_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    diagnostics = {
        **base_diagnostics,
        "status": "ok",
        "best_alpha": alpha,
        "selection_alpha_scale": alpha_scale,
        "effective_selection_alpha": effective_alpha,
        "best_l1_ratio": l1_ratio,
        "cv_mse": cv_mse,
        "regularization_path": regularization_path,
        "selection_intercept": selection_intercept,
        "selected_support_size": len(selection_support),
        "hierarchy_support_size": len(hierarchy_support),
        "refit_support_size": len(support),
        "hierarchy": hierarchy_diagnostics,
        "ridge_alpha": ridge_alpha,
        **refit_diagnostics,
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
        selection_support=selection_support,
        hierarchy_support=hierarchy_support,
        refit_support=list(hierarchy_support),
        refit_mode=refit_mode,
    )
