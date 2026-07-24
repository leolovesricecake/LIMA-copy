from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from .query_ledger import QueryLedger
from .reconstruction_metrics import mae, normalized_rmse, r2_score
from .schema import FeatureSpec
from .value_oracle import ValueOracle


def evaluate_surrogate_distributions(
    model,
    *,
    masks_by_distribution: Mapping[str, Sequence[int]],
    values_by_distribution: Mapping[str, Sequence[float]],
) -> Dict[str, Dict[str, float | int | bool | None]]:
    output: Dict[str, Dict[str, float | int | bool | None]] = {}
    for name, masks in masks_by_distribution.items():
        true = np.asarray(values_by_distribution[name], dtype=np.float64)
        if len(true) == 0:
            output[str(name)] = {
                "count": 0,
                "value_std": None,
                "degenerate": True,
                "r2": None,
                "normalized_rmse": None,
                "mae": None,
            }
            continue
        prediction = np.asarray(model.predict(list(masks)), dtype=np.float64)
        value_std = float(np.std(true))
        degenerate = value_std <= 1e-12
        output[str(name)] = {
            "count": int(len(true)),
            "value_std": value_std,
            "degenerate": bool(degenerate),
            "r2": None if degenerate else r2_score(true, prediction),
            "normalized_rmse": normalized_rmse(true, prediction),
            "mae": mae(true, prediction),
        }
    return output


def _mask_deleting(full_mask: int, players: Sequence[int]) -> int:
    mask = int(full_mask)
    for player in players:
        mask &= ~(1 << int(player))
    return mask


def _mask_keeping(players: Sequence[int]) -> int:
    mask = 0
    for player in players:
        mask |= 1 << int(player)
    return mask


def _curve_auc(rows: Sequence[Dict[str, float]], key: str) -> float | None:
    if len(rows) < 2:
        return None
    x = np.asarray([float(row["fraction"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    denom = float(x[-1] - x[0])
    if denom <= 0:
        return None
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(y, x))
    else:
        area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))
    return float(area / denom)


def evaluate_node_ranking(
    *,
    oracle: ValueOracle,
    feature_spec: FeatureSpec,
    ranking: Sequence[int],
    target_class: int,
    value_type: str,
    ledger: QueryLedger,
    operator: str = "delete",
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.5, 1.0),
) -> Dict[str, object]:
    n = feature_spec.n_features
    normalized_ranking = [int(idx) for idx in ranking if 0 <= int(idx) < n]
    seen = set(normalized_ranking)
    normalized_ranking.extend(idx for idx in range(n) if idx not in seen)
    full = (1 << n) - 1
    rows = []
    required_masks = {full}
    plans = []
    for fraction in sorted(set(float(value) for value in fractions)):
        count = min(n, max(0, int(round(fraction * n))))
        top = normalized_ranking[:count]
        bottom = normalized_ranking[n - count :] if count > 0 else []
        morf_mask = _mask_deleting(full, top)
        lerf_mask = _mask_deleting(full, bottom)
        sufficient_mask = _mask_keeping(top)
        required_masks.update([morf_mask, lerf_mask, sufficient_mask])
        plans.append((fraction, count, morf_mask, lerf_mask, sufficient_mask))

    ordered_masks = sorted(required_masks)
    values, _ = oracle.values_for_masks(
        feature_spec,
        ordered_masks,
        target_class=int(target_class),
        value_type=value_type,
        ledger=ledger,
        category="evaluation",
        operator=operator,
    )
    value_by_mask = {mask: float(value) for mask, value in zip(ordered_masks, values)}
    full_value = value_by_mask[full]
    for fraction, count, morf_mask, lerf_mask, sufficient_mask in plans:
        rows.append(
            {
                "fraction": float(fraction),
                "count": int(count),
                "morf_value": value_by_mask[morf_mask],
                "lerf_value": value_by_mask[lerf_mask],
                "sufficient_value": value_by_mask[sufficient_mask],
                "comprehensiveness": float(full_value - value_by_mask[morf_mask]),
                "lerf_drop": float(full_value - value_by_mask[lerf_mask]),
                "sufficiency_gap": float(full_value - value_by_mask[sufficient_mask]),
            }
        )
    primary = min(rows, key=lambda row: abs(float(row["fraction"]) - 0.2))
    return {
        "full_value": float(full_value),
        "ranking": normalized_ranking,
        "curve": rows,
        "morf_comprehensiveness_auc": _curve_auc(rows, "comprehensiveness"),
        "lerf_drop_auc": _curve_auc(rows, "lerf_drop"),
        "sufficiency_gap_auc": _curve_auc(rows, "sufficiency_gap"),
        "primary_fraction": float(primary["fraction"]),
        "primary_comprehensiveness": float(primary["comprehensiveness"]),
        "primary_sufficiency_gap": float(primary["sufficiency_gap"]),
    }
