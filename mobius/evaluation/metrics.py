"""Pure faithfulness metric helpers over probability perturbation curves."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


PRIMARY_Q = 20


def top_count(total_units: int, q_percent: int) -> int:
    """Convert a percentage to the legacy floor-based top-unit count."""

    if int(total_units) <= 0:
        return 0
    return int(math.floor(int(q_percent) * int(total_units) / 100.0))


def per_q_metrics(
    full_probability: float,
    remove_probabilities: Dict[int, float],
    keep_probabilities: Dict[int, float],
    q_values: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Compute comprehensiveness and sufficiency at each perturbation percentage."""

    return {
        int(q): {
            "comprehensiveness": float(
                full_probability - remove_probabilities[int(q)]
            ),
            "sufficiency": float(full_probability - keep_probabilities[int(q)]),
        }
        for q in q_values
    }


def aopc_from_drops(drops: Sequence[float]) -> float:
    """Average probability drops along a complete deletion trajectory."""

    return float(np.mean(np.asarray(drops, dtype=np.float64))) if drops else 0.0


def aupc_from_probabilities(probabilities: Sequence[float]) -> float:
    """Integrate a complete deletion probability curve over normalized progress."""

    values = np.asarray(probabilities, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    interval_count = len(values) - 1
    trapezoid_sum = 0.5 * float(values[0] + values[-1])
    if len(values) > 2:
        trapezoid_sum += float(np.sum(values[1:-1]))
    return float(trapezoid_sum / interval_count)


def aml_aopc(
    per_q: Dict[int, Dict[str, float]],
    q_values: Sequence[int],
    metric: str,
) -> float:
    """Match AML's q-grid AOPC denominator including the implicit zero point."""

    if not q_values:
        return 0.0
    return float(
        sum(float(per_q[int(q)][metric]) for q in q_values)
        / (len(q_values) + 1)
    )


def aggregate(values: Sequence[float]) -> Dict[str, float | int | None]:
    """Summarize a metric with count, mean, and sample standard deviation."""

    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }
