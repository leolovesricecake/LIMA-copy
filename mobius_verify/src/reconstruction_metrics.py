from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .subset_enumeration import candidate_count, popcounts_for_n
from .transforms import (
    infer_n_from_values,
    inverse_fourier_transform,
    inverse_mobius_transform,
    reconstruct_fourier_degree,
    reconstruct_mobius_degree,
)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-24:
        return 1.0 if ss_res <= 1e-24 else 0.0
    return float(1.0 - ss_res / ss_tot)


def normalized_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.std(y))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return float(rmse / denom) if denom > 1e-12 else float(rmse)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def threshold_first_x(records: Sequence[Dict[str, float]], x_key: str, y_key: str, threshold: float) -> int | None:
    for row in records:
        if float(row[y_key]) >= float(threshold):
            return int(row[x_key])
    return None


def _trapezoid_area(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    if len(y_values) < 2:
        return 0.0
    widths = x_values[1:] - x_values[:-1]
    heights = (y_values[1:] + y_values[:-1]) * 0.5
    return float(np.sum(widths * heights))


def auc_logx(records: Sequence[Dict[str, float]], x_key: str, y_key: str) -> float | None:
    clean = [(float(row[x_key]), float(row[y_key])) for row in records if float(row[x_key]) > 0]
    if len(clean) < 2:
        return None
    clean.sort(key=lambda item: item[0])
    xs = np.log2(np.asarray([item[0] for item in clean], dtype=np.float64))
    ys = np.asarray([item[1] for item in clean], dtype=np.float64)
    denom = float(xs[-1] - xs[0])
    if denom <= 0:
        return None
    return float(_trapezoid_area(ys, xs) / denom)


def degree_curves(
    values: Sequence[float],
    mobius_coefficients: Sequence[float],
    fourier_coefficients: Sequence[float],
    *,
    d_max: int,
) -> Dict[str, List[Dict[str, float]]]:
    rows_m = []
    rows_f = []
    for degree in range(1, int(d_max) + 1):
        pred_m = reconstruct_mobius_degree(mobius_coefficients, degree)
        pred_f = reconstruct_fourier_degree(fourier_coefficients, degree)
        rows_m.append(
            {
                "degree": int(degree),
                "r2": r2_score(values, pred_m),
                "normalized_rmse": normalized_rmse(values, pred_m),
            }
        )
        rows_f.append(
            {
                "degree": int(degree),
                "r2": r2_score(values, pred_f),
                "normalized_rmse": normalized_rmse(values, pred_f),
            }
        )
    return {"mobius": rows_m, "fourier": rows_f}


def default_k_grid(n_terms: int) -> List[int]:
    if n_terms <= 0:
        return []
    grid = []
    k = 1
    while k < int(n_terms):
        grid.append(k)
        k *= 2
    grid.append(int(n_terms))
    return sorted(set(grid))


def coefficient_topk_curve(
    values: Sequence[float],
    coefficients: Sequence[float],
    *,
    basis: str,
    d_max: int,
    weighted_mobius: bool = False,
    k_values: Iterable[int] | None = None,
) -> List[Dict[str, float]]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    n = infer_n_from_values(coeff)
    degrees = popcounts_for_n(n)
    terms = [mask for mask in range(1, 1 << n) if int(degrees[mask]) <= int(d_max)]
    if weighted_mobius:
        ranked = sorted(terms, key=lambda mask: (-abs(coeff[mask]) * 2.0 ** (-degrees[mask] / 2.0), mask))
    else:
        ranked = sorted(terms, key=lambda mask: (-abs(coeff[mask]), mask))
    ks = list(k_values) if k_values is not None else default_k_grid(len(ranked))
    rows = []
    for k in ks:
        selected = set(ranked[: min(int(k), len(ranked))])
        kept = np.zeros_like(coeff)
        kept[0] = coeff[0]
        for term in selected:
            kept[term] = coeff[term]
        if basis == "mobius":
            pred = inverse_mobius_transform(kept)
        elif basis == "fourier":
            pred = inverse_fourier_transform(kept)
        else:
            raise ValueError(f"Unsupported basis: {basis!r}")
        rows.append(
            {
                "k": int(min(int(k), len(ranked))),
                "r2": r2_score(values, pred),
                "normalized_rmse": normalized_rmse(values, pred),
                "candidate_count": int(candidate_count(n, d_max)),
            }
        )
    return rows


def summarize_curve(records: Sequence[Dict[str, float]], *, x_key: str = "k") -> Dict[str, float | int | None]:
    return {
        "x80": threshold_first_x(records, x_key, "r2", 0.80),
        "x90": threshold_first_x(records, x_key, "r2", 0.90),
        "x95": threshold_first_x(records, x_key, "r2", 0.95),
        "auc_r2_logx": auc_logx(records, x_key, "r2"),
    }
