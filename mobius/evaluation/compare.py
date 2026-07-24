"""Compare target-compatible schema-v2 metrics files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _load_metrics(run_dir: str | Path) -> Dict[str, Any]:
    """Load one run's target-explicit metrics report."""

    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def compare_runs(left: str | Path, right: str | Path) -> Dict[str, Any]:
    """Return right-minus-left deltas for shared aggregate metrics and costs."""

    left_metrics = _load_metrics(left)
    right_metrics = _load_metrics(right)
    if left_metrics.get("target") != right_metrics.get("target"):
        raise ValueError("Cannot compare runs evaluated with different targets.")
    faithfulness: Dict[str, float] = {}
    for name in sorted(
        set(left_metrics.get("faithfulness", {}))
        & set(right_metrics.get("faithfulness", {}))
    ):
        left_value = left_metrics["faithfulness"][name].get("mean")
        right_value = right_metrics["faithfulness"][name].get("mean")
        if left_value is not None and right_value is not None:
            faithfulness[name] = float(right_value) - float(left_value)
    costs: Dict[str, float] = {}
    for name in sorted(
        set(left_metrics.get("attribution_cost", {}))
        & set(right_metrics.get("attribution_cost", {}))
    ):
        costs[name] = float(right_metrics["attribution_cost"][name]) - float(
            left_metrics["attribution_cost"][name]
        )
    return {
        "target": left_metrics["target"],
        "left": str(left),
        "right": str(right),
        "delta_right_minus_left": {
            "faithfulness": faithfulness,
            "attribution_cost": costs,
            "accuracy": (
                float(right_metrics["accuracy"]) - float(left_metrics["accuracy"])
                if left_metrics.get("accuracy") is not None
                and right_metrics.get("accuracy") is not None
                else None
            ),
        },
    }

