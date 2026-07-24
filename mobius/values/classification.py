"""Scalar value functions derived from all-class verbalizer scores."""

from __future__ import annotations

from typing import Sequence

import numpy as np


VALUE_FUNCTIONS = {
    "target_probability",
    "predicted_probability",
    "predicted_class_margin",
    "raw_target_score",
}


def normalize_value_function(value: str | None) -> str:
    """Normalize value-function aliases and reject unknown semantics."""

    normalized = str(value or "target_probability").strip().lower()
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
    if normalized not in VALUE_FUNCTIONS:
        raise ValueError(
            f"Unsupported value function {value!r}; expected {sorted(VALUE_FUNCTIONS)}"
        )
    return normalized


def effective_target_mode(value_function: str, requested: str) -> str:
    """Force predicted target semantics for explicitly predicted value functions."""

    normalized = normalize_value_function(value_function)
    if normalized in {"predicted_probability", "predicted_class_margin"}:
        return "predicted"
    mode = str(requested).strip().lower()
    if mode not in {"predicted", "gold"}:
        raise ValueError("target_mode must be predicted or gold.")
    return mode


def probabilities(scores: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Apply a stable row-wise softmax to raw class scores."""

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be two-dimensional, got {matrix.shape}")
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def attribution_values(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    target_class: int,
    value_function: str,
) -> np.ndarray:
    """Reduce all-class scores to the configured scalar game value."""

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be two-dimensional, got {matrix.shape}")
    target = int(target_class)
    if target < 0 or target >= matrix.shape[1]:
        raise ValueError(f"target_class {target} is outside the score matrix.")
    normalized = normalize_value_function(value_function)
    if normalized in {"target_probability", "predicted_probability"}:
        return probabilities(matrix)[:, target]
    if normalized == "raw_target_score":
        return matrix[:, target].astype(np.float64)
    if matrix.shape[1] < 2:
        raise ValueError("predicted_class_margin requires at least two classes.")
    competitors = np.delete(matrix, target, axis=1)
    return (matrix[:, target] - np.max(competitors, axis=1)).astype(np.float64)

