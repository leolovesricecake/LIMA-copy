from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .subset_enumeration import popcounts_for_n


def infer_n_from_values(values: Sequence[float]) -> int:
    size = len(values)
    if size <= 0 or size & (size - 1):
        raise ValueError(f"values length must be a positive power of two, got {size}")
    return int(math.log2(size))


def mobius_transform(values: Sequence[float]) -> np.ndarray:
    coeff = np.asarray(values, dtype=np.float64).copy()
    n = infer_n_from_values(coeff)
    for bit in range(n):
        step = 1 << bit
        for mask in range(1 << n):
            if mask & step:
                coeff[mask] -= coeff[mask ^ step]
    return coeff


def inverse_mobius_transform(coefficients: Sequence[float]) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64).copy()
    n = infer_n_from_values(values)
    for bit in range(n):
        step = 1 << bit
        for mask in range(1 << n):
            if mask & step:
                values[mask] += values[mask ^ step]
    return values


def fwht(values: Sequence[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    n_total = len(out)
    if n_total <= 0 or n_total & (n_total - 1):
        raise ValueError(f"FWHT length must be a positive power of two, got {n_total}")
    h = 1
    while h < n_total:
        for start in range(0, n_total, h * 2):
            left = out[start : start + h].copy()
            right = out[start + h : start + 2 * h].copy()
            out[start : start + h] = left + right
            out[start + h : start + 2 * h] = left - right
        h *= 2
    return out


def fourier_transform(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    infer_n_from_values(arr)
    return fwht(arr) / float(len(arr))


def inverse_fourier_transform(coefficients: Sequence[float]) -> np.ndarray:
    arr = np.asarray(coefficients, dtype=np.float64)
    infer_n_from_values(arr)
    return fwht(arr)


def truncate_by_degree(coefficients: Sequence[float], max_degree: int) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float64)
    n = infer_n_from_values(coeff)
    degrees = popcounts_for_n(n)
    out = coeff.copy()
    out[degrees > int(max_degree)] = 0.0
    return out


def reconstruct_mobius_degree(coefficients: Sequence[float], max_degree: int) -> np.ndarray:
    return inverse_mobius_transform(truncate_by_degree(coefficients, max_degree=max_degree))


def reconstruct_fourier_degree(coefficients: Sequence[float], max_degree: int) -> np.ndarray:
    return inverse_fourier_transform(truncate_by_degree(coefficients, max_degree=max_degree))


def design_mobius(masks: Sequence[int], terms: Sequence[int]) -> np.ndarray:
    mask_arr = np.asarray([int(mask) for mask in masks], dtype=np.int64)
    term_arr = np.asarray([int(term) for term in terms], dtype=np.int64)
    return ((mask_arr[:, None] & term_arr[None, :]) == term_arr[None, :]).astype(np.float64)


def design_fourier(masks: Sequence[int], terms: Sequence[int]) -> np.ndarray:
    rows = []
    for mask in masks:
        row = [1.0 if ((int(mask) & int(term)).bit_count() % 2 == 0) else -1.0 for term in terms]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)

