from __future__ import annotations

from typing import Sequence

import numpy as np


ATTRIBUTION_VALUE_FUNCTIONS = {
    "target_probability",
    "predicted_probability",
    "predicted_class_margin",
    "raw_target_score",
}


def normalize_attribution_value_function(value_function: str | None) -> str:
    normalized = str(value_function or "target_probability").strip().lower()
    aliases = {
        "probability": "target_probability",
        "prob": "target_probability",
        "predicted_prob": "predicted_probability",
        "margin": "predicted_class_margin",
        "predicted_margin": "predicted_class_margin",
        "raw": "raw_target_score",
        "raw_target": "raw_target_score",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ATTRIBUTION_VALUE_FUNCTIONS:
        raise ValueError(
            f"Unsupported attribution value function: {value_function!r}. "
            f"Expected one of {sorted(ATTRIBUTION_VALUE_FUNCTIONS)}."
        )
    return normalized


def probabilities_from_label_scores(
    scores: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must have shape [n_samples, n_classes], got {matrix.shape}")
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    denominator = np.sum(probabilities, axis=1, keepdims=True)
    return (probabilities / denominator).astype(np.float64)


def attribution_values_from_label_scores(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    target_class: int,
    value_function: str,
) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must have shape [n_samples, n_classes], got {matrix.shape}")
    target = int(target_class)
    if target < 0 or target >= matrix.shape[1]:
        raise ValueError(f"target_class={target} is outside [0, {matrix.shape[1]})")

    normalized = normalize_attribution_value_function(value_function)
    if normalized in {"target_probability", "predicted_probability"}:
        return probabilities_from_label_scores(matrix)[:, target]
    if normalized == "raw_target_score":
        return matrix[:, target].astype(np.float64)
    if matrix.shape[1] < 2:
        raise ValueError("predicted_class_margin requires at least two classes")
    competitors = np.delete(matrix, target, axis=1)
    return (matrix[:, target] - np.max(competitors, axis=1)).astype(np.float64)


__all__ = [
    "ATTRIBUTION_VALUE_FUNCTIONS",
    "attribution_values_from_label_scores",
    "normalize_attribution_value_function",
    "probabilities_from_label_scores",
]
