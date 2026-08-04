"""Task-level diagnostics for a verbalizer-based text classifier."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mobius.core.schema import DatasetBundle
from mobius.models.base import RawTextScorer
from mobius.values.classification import probabilities


def evaluate_text_classifier(
    bundle: DatasetBundle,
    scorer: RawTextScorer,
) -> Dict[str, Any]:
    """Evaluate full-input classification before running attribution experiments."""

    scores = np.asarray(
        scorer.score_texts([sample.text for sample in bundle.samples]),
        dtype=np.float64,
    )
    expected = (len(bundle.samples), len(bundle.verbalizers))
    if scores.shape != expected:
        raise ValueError(
            f"Classifier returned score shape {scores.shape}; expected {expected}."
        )
    probs = probabilities(scores)
    gold = np.asarray([int(sample.label) for sample in bundle.samples], dtype=np.int64)
    predicted = np.argmax(scores, axis=1).astype(np.int64)
    class_count = len(bundle.verbalizers)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, output in zip(gold, predicted):
        confusion[int(target), int(output)] += 1

    per_class = []
    recalls = []
    for class_id, label in enumerate(bundle.verbalizers):
        support = int(np.sum(gold == class_id))
        correct = int(confusion[class_id, class_id])
        recall = float(correct / support) if support else None
        if recall is not None:
            recalls.append(recall)
        per_class.append(
            {
                "class_id": class_id,
                "label": str(label),
                "support": support,
                "correct": correct,
                "recall": recall,
            }
        )

    class_supports = np.bincount(gold, minlength=class_count)
    row_indices = np.arange(len(gold), dtype=np.int64)
    gold_probabilities = probs[row_indices, gold]
    accuracy = float(np.mean(predicted == gold)) if len(gold) else None
    return {
        "dataset": bundle.dataset_name,
        "split": bundle.split,
        "sample_count": len(bundle.samples),
        "class_count": class_count,
        "verbalizers": list(bundle.verbalizers),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "random_baseline_accuracy": float(1.0 / class_count),
        "majority_baseline_accuracy": (
            float(np.max(class_supports) / len(gold)) if len(gold) else None
        ),
        "mean_gold_probability": (
            float(np.mean(gold_probabilities)) if len(gold_probabilities) else None
        ),
        "mean_prediction_confidence": (
            float(np.mean(np.max(probs, axis=1))) if len(probs) else None
        ),
        "negative_log_likelihood": (
            float(-np.mean(np.log(np.clip(gold_probabilities, 1e-12, 1.0))))
            if len(gold_probabilities)
            else None
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "prompt": scorer.scoring_contract().get("prompt"),
        "model_counters": scorer.snapshot_counters(),
    }


__all__ = ["evaluate_text_classifier"]
