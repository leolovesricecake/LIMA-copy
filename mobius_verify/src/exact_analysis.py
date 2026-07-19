from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .hierarchy_metrics import hierarchy_curve
from .oracle_approximation import omp_curve
from .reconstruction_metrics import (
    coefficient_topk_curve,
    degree_curves,
    default_k_grid,
    summarize_curve,
    threshold_first_x,
)
from .subset_enumeration import candidate_count
from .transforms import (
    fourier_transform,
    infer_n_from_values,
    inverse_fourier_transform,
    inverse_mobius_transform,
    mobius_transform,
)


def analyze_value_table(
    values: Sequence[float],
    *,
    d_max: int = 4,
    omp_max_k: int | None = None,
) -> Dict[str, object]:
    y = np.asarray(values, dtype=np.float64)
    n = infer_n_from_values(y)
    mobius = mobius_transform(y)
    fourier = fourier_transform(y)
    mobius_rebuilt = inverse_mobius_transform(mobius)
    fourier_rebuilt = inverse_fourier_transform(fourier)

    degree = degree_curves(y, mobius, fourier, d_max=int(d_max))
    term_count = candidate_count(n, int(d_max))
    k_grid = default_k_grid(term_count)
    hierarchy_k = [k for k in (8, 16, 32, 64, 128, 256) if k <= max(1, (1 << n) - 1)]

    mobius_weighted = coefficient_topk_curve(
        y,
        mobius,
        basis="mobius",
        d_max=int(d_max),
        weighted_mobius=True,
        k_values=k_grid,
    )
    fourier_topk = coefficient_topk_curve(
        y,
        fourier,
        basis="fourier",
        d_max=int(d_max),
        weighted_mobius=False,
        k_values=k_grid,
    )
    mobius_omp = omp_curve(y, basis="mobius", d_max=int(d_max), max_k=omp_max_k, k_values=k_grid)
    fourier_omp = omp_curve(y, basis="fourier", d_max=int(d_max), max_k=omp_max_k, k_values=k_grid)

    degree_summary = {
        "mobius_d90": threshold_first_x(degree["mobius"], "degree", "r2", 0.90),
        "mobius_d95": threshold_first_x(degree["mobius"], "degree", "r2", 0.95),
        "fourier_d90": threshold_first_x(degree["fourier"], "degree", "r2", 0.90),
        "fourier_d95": threshold_first_x(degree["fourier"], "degree", "r2", 0.95),
    }
    sparse_summary = {
        "mobius_weighted": summarize_curve(mobius_weighted),
        "fourier_topk": summarize_curve(fourier_topk),
        "mobius_omp": summarize_curve(mobius_omp["curve"]),
        "fourier_omp": summarize_curve(fourier_omp["curve"]),
    }
    mobius_k90 = sparse_summary["mobius_omp"]["x90"]
    fourier_k90 = sparse_summary["fourier_omp"]["x90"]
    normalized = {
        "mobius_k90_over_n": float(mobius_k90 / n) if mobius_k90 is not None and n > 0 else None,
        "fourier_k90_over_n": float(fourier_k90 / n) if fourier_k90 is not None and n > 0 else None,
        "mobius_k90_over_candidate_count": float(mobius_k90 / term_count)
        if mobius_k90 is not None and term_count > 0
        else None,
        "fourier_k90_over_candidate_count": float(fourier_k90 / term_count)
        if fourier_k90 is not None and term_count > 0
        else None,
    }

    return {
        "n_features": int(n),
        "value_mean": float(np.mean(y)),
        "value_std": float(np.std(y)),
        "degenerate": bool(float(np.std(y)) < 1e-8),
        "candidate_count_d_max": int(term_count),
        "inverse_errors": {
            "mobius_max_abs": float(np.max(np.abs(y - mobius_rebuilt))),
            "fourier_max_abs": float(np.max(np.abs(y - fourier_rebuilt))),
        },
        "degree_curves": degree,
        "degree_summary": degree_summary,
        "sparsity_curves": {
            "mobius_weighted_topk": mobius_weighted,
            "fourier_exact_topk": fourier_topk,
            "mobius_omp": mobius_omp["curve"],
            "fourier_omp": fourier_omp["curve"],
        },
        "sparsity_summary": sparse_summary,
        "normalized_support": normalized,
        "hierarchy_metrics": hierarchy_curve(fourier, hierarchy_k),
        "oracle_selected_terms": {
            "mobius_omp": [int(x) for x in mobius_omp["selected_terms"]],
            "fourier_omp": [int(x) for x in fourier_omp["selected_terms"]],
        },
    }

