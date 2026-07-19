from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..designs import (
    equal_share_node_scores,
    fourier_to_presence_coefficients,
)
from .sparse_surrogate import SparseSurrogateModel, fit_sparse_surrogate


def fit_sparse_mobius(
    *,
    masks: Sequence[int],
    values: Sequence[float],
    n_features: int,
    orientation: str = "deletion_mobius",
    max_degree: int = 2,
    **kwargs,
) -> SparseSurrogateModel:
    return fit_sparse_surrogate(
        masks=masks,
        values=values,
        n_features=int(n_features),
        basis=str(orientation),
        max_degree=int(max_degree),
        **kwargs,
    )


def model_presence_coefficients(model: SparseSurrogateModel) -> tuple[float, Dict[int, float]]:
    coefficients = model.coefficient_dict()
    if model.basis == "fourier":
        return fourier_to_presence_coefficients(
            intercept=model.intercept,
            terms=model.terms,
            coefficients=model.coefficients,
        )
    if model.basis != "presence_mobius":
        raise ValueError(f"Model basis {model.basis!r} is not a presence representation")
    return float(model.intercept), coefficients


def model_node_scores(model: SparseSurrogateModel) -> np.ndarray:
    if model.basis == "fourier":
        _, coefficients = model_presence_coefficients(model)
        return equal_share_node_scores(
            n_features=model.n_features,
            coefficients=coefficients,
            orientation="presence_mobius",
        )
    return equal_share_node_scores(
        n_features=model.n_features,
        coefficients=model.coefficient_dict(),
        orientation=model.basis,
    )


def ranked_nodes(scores: Sequence[float]) -> Dict[str, list[int]]:
    values = [float(value) for value in scores]
    return {
        "positive": sorted(range(len(values)), key=lambda idx: (-values[idx], idx)),
        "negative": sorted(range(len(values)), key=lambda idx: (values[idx], idx)),
        "absolute": sorted(range(len(values)), key=lambda idx: (-abs(values[idx]), idx)),
    }
