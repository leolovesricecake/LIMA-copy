from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .masking import apply_mask
from .models.base import RawTextScorer
from .schema import FeatureSpec


@dataclass(frozen=True)
class ValueFunctionMetadata:
    value_function: str
    target_class: int
    target_label_text: str
    full_scores: List[float]
    target_class_source: str
    verbalizer_length_normalization: str
    full_value: float
    full_competitor_class: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "value_function": self.value_function,
            "target_class": int(self.target_class),
            "target_label_text": self.target_label_text,
            "full_scores": [float(x) for x in self.full_scores],
            "target_class_source": self.target_class_source,
            "verbalizer_length_normalization": self.verbalizer_length_normalization,
            "full_value": float(self.full_value),
            "full_competitor_class": self.full_competitor_class,
        }


VALUE_FUNCTION_TYPES = {"predicted_class_margin", "raw_target_score"}


def normalize_value_function_type(value_type: str) -> str:
    normalized = str(value_type).strip().lower()
    aliases = {
        "margin": "predicted_class_margin",
        "predicted_margin": "predicted_class_margin",
        "raw": "raw_target_score",
        "raw_target": "raw_target_score",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALUE_FUNCTION_TYPES:
        raise ValueError(
            f"Unsupported value function: {value_type!r}. Expected one of {sorted(VALUE_FUNCTION_TYPES)}."
        )
    return normalized


def values_from_score_matrix(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    target_class: int,
    value_type: str = "predicted_class_margin",
) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must have shape [n_samples, n_classes], got {matrix.shape}")
    target = int(target_class)
    if target < 0 or target >= matrix.shape[1]:
        raise ValueError(f"target_class={target} is outside [0, {matrix.shape[1]})")
    normalized = normalize_value_function_type(value_type)
    target_scores = matrix[:, target]
    if normalized == "raw_target_score":
        return target_scores.astype(np.float64)
    if matrix.shape[1] < 2:
        raise ValueError("predicted_class_margin requires at least two classes")
    competitors = np.delete(matrix, target, axis=1)
    return (target_scores - np.max(competitors, axis=1)).astype(np.float64)


def best_competitor(scores: Sequence[float], target_class: int) -> int | None:
    row = np.asarray(scores, dtype=np.float64)
    if row.ndim != 1 or len(row) < 2:
        return None
    target = int(target_class)
    candidates = [(float(value), idx) for idx, value in enumerate(row) if idx != target]
    return int(max(candidates, key=lambda item: (item[0], -item[1]))[1])


class PredictedClassMarginValueFunction:
    def __init__(
        self,
        scorer: RawTextScorer,
        *,
        target_class_source: str = "full_input_prediction",
        verbalizer_length_normalization: str = "mean",
        value_type: str = "predicted_class_margin",
    ) -> None:
        self.scorer = scorer
        self.target_class_source = str(target_class_source)
        self.verbalizer_length_normalization = str(verbalizer_length_normalization)
        self.value_type = normalize_value_function_type(value_type)

    def target_for_text(self, text: str, *, gold_label: Optional[int] = None) -> tuple[int, np.ndarray]:
        scores = np.asarray(self.scorer.score_text(text), dtype=np.float64)
        if self.target_class_source == "gold":
            if gold_label is None:
                raise ValueError("gold_label is required when target_class_source='gold'")
            target = int(gold_label)
        elif self.target_class_source == "full_input_prediction":
            target = int(np.argmax(scores))
        else:
            raise ValueError(f"Unsupported target_class_source: {self.target_class_source!r}")
        return target, scores

    def metadata_for_feature_spec(
        self,
        feature_spec: FeatureSpec,
        *,
        gold_label: Optional[int] = None,
    ) -> ValueFunctionMetadata:
        target, scores = self.target_for_text(feature_spec.normalized_model_text, gold_label=gold_label)
        label_text = str(self.scorer.verbalizers[target]) if target < len(self.scorer.verbalizers) else str(target)
        return ValueFunctionMetadata(
            value_function=self.value_type,
            target_class=int(target),
            target_label_text=label_text,
            full_scores=[float(x) for x in scores.tolist()],
            target_class_source=self.target_class_source,
            verbalizer_length_normalization=self.verbalizer_length_normalization,
            full_value=float(
                values_from_score_matrix(
                    scores.reshape(1, -1),
                    target_class=target,
                    value_type=self.value_type,
                )[0]
            ),
            full_competitor_class=best_competitor(scores, target),
        )

    def evaluate_texts(self, texts: Sequence[str], *, target_class: int) -> np.ndarray:
        scores = np.asarray(self.scorer.score_texts(list(texts)), dtype=np.float64)
        return values_from_score_matrix(
            scores,
            target_class=int(target_class),
            value_type=self.value_type,
        )

    def evaluate_masks(
        self,
        feature_spec: FeatureSpec,
        masks: Sequence[int],
        *,
        target_class: int,
        operator: str = "delete",
        active_feature_ids: Optional[Sequence[int]] = None,
        conditioning_mode: str = "global",
        mask_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        values: List[float] = []
        texts: List[str] = []
        for mask in masks:
            texts.append(
                apply_mask(
                    feature_spec,
                    int(mask),
                    operator=operator,
                    active_feature_ids=active_feature_ids,
                    conditioning_mode=conditioning_mode,
                    mask_token=mask_token,
                    unk_token=unk_token,
                )
            )
        for start in range(0, len(texts), int(batch_size)):
            cur = texts[start : start + int(batch_size)]
            values.extend(self.evaluate_texts(cur, target_class=target_class).tolist())
        return np.asarray(values, dtype=np.float64)
