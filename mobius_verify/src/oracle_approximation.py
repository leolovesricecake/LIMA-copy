from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .reconstruction_metrics import r2_score, normalized_rmse
from .subset_enumeration import all_masks, candidate_terms
from .transforms import design_fourier, design_mobius, infer_n_from_values


def _design_for_basis(masks: Sequence[int], terms: Sequence[int], basis: str) -> np.ndarray:
    if basis == "mobius":
        return design_mobius(masks, terms)
    if basis == "fourier":
        return design_fourier(masks, terms)
    raise ValueError(f"Unsupported basis: {basis!r}")


def omp_curve(
    values: Sequence[float],
    *,
    basis: str,
    d_max: int,
    max_k: int | None = None,
    k_values: Sequence[int] | None = None,
) -> Dict[str, object]:
    y = np.asarray(values, dtype=np.float64)
    n = infer_n_from_values(y)
    masks = all_masks(n)
    terms = candidate_terms(n, int(d_max), include_empty=False)
    if not terms:
        return {"basis": basis, "selected_terms": [], "curve": []}
    X = _design_for_basis(masks, terms, basis)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms <= 1e-12] = 1.0
    Xn = X / col_norms[None, :]

    limit = min(len(terms), int(max_k) if max_k is not None else len(terms))
    wanted = sorted(set(int(k) for k in (k_values or []) if int(k) > 0))
    if wanted:
        limit = min(limit, max(wanted))

    selected_cols: List[int] = []
    selected_terms: List[int] = []
    available = np.ones(len(terms), dtype=bool)
    pred = np.full_like(y, float(np.mean(y)))
    residual = y - pred
    rows = []
    last_r2 = r2_score(y, pred)

    for step in range(1, limit + 1):
        corr = Xn.T @ residual
        corr[~available] = 0.0
        best_col = int(np.argmax(np.abs(corr)))
        if not available[best_col]:
            break
        selected_cols.append(best_col)
        selected_terms.append(int(terms[best_col]))
        available[best_col] = False

        design = np.column_stack([np.ones(len(y), dtype=np.float64), X[:, selected_cols]])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        residual = y - pred
        cur_r2 = max(float(last_r2), r2_score(y, pred))
        last_r2 = cur_r2
        if not wanted or step in wanted:
            rows.append(
                {
                    "k": int(step),
                    "r2": float(cur_r2),
                    "normalized_rmse": normalized_rmse(y, pred),
                    "selected_term": int(selected_terms[-1]),
                }
            )

    return {
        "basis": str(basis),
        "d_max": int(d_max),
        "candidate_count": int(len(terms)),
        "selected_terms": selected_terms,
        "curve": rows,
    }

