from __future__ import annotations

import numpy as np

from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.objective.submodular import ObjectiveWeights, TextSubmodularObjective
from lima_llm.types import TextChunk


def _build_objective(weights: ObjectiveWeights) -> tuple[MockBackbone, TextSubmodularObjective]:
    text = "alpha beta gamma delta"
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=6, text="alpha "),
        TextChunk(chunk_id=1, start_char=6, end_char=11, text="beta "),
        TextChunk(chunk_id=2, start_char=11, end_char=17, text="gamma "),
        TextChunk(chunk_id=3, start_char=17, end_char=22, text="delta"),
    ]

    # Keep this synthetic so objective-side model calls are easy to assert.
    chunk_embeddings = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([0.5, 0.5], dtype=np.float32),
    ]

    backbone = MockBackbone()
    objective = TextSubmodularObjective(
        backbone=backbone,
        text=text,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        target_label=1,
        verbalizers=["NEG", "POS"],
        weights=weights,
        enable_batch_prefetch=True,
    )
    return backbone, objective


def test_zero_lambda_components_are_skipped() -> None:
    backbone, objective = _build_objective(ObjectiveWeights(0.0, 1.0, 0.0, 0.0))
    _ = objective.evaluate_gains([], [0, 1, 2, 3])

    stats = objective.cache_stats()
    enabled = stats.get("component_enabled", {})

    assert enabled == {
        "confidence": False,
        "effectiveness": True,
        "consistency": False,
        "collaboration": False,
    }

    assert int(stats["confidence_compute_calls"]) == 0
    assert int(stats["consistency_compute_calls"]) == 0
    assert int(stats["collaboration_compute_calls"]) == 0

    assert int(stats["confidence_skipped_due_to_zero_lambda"]) > 0
    assert int(stats["consistency_skipped_due_to_zero_lambda"]) > 0
    assert int(stats["collaboration_skipped_due_to_zero_lambda"]) > 0

    assert int(stats["effectiveness_compute_calls"]) > 0
    assert int(stats["effectiveness_skipped_due_to_zero_lambda"]) == 0

    # With only effectiveness enabled, objective should avoid model predict/embed calls.
    assert int(backbone.forward_counters.get("predict_calls", 0)) == 0
    assert int(backbone.forward_counters.get("embed_calls", 0)) == 0


def test_disabled_component_values_are_zero_in_subset_score() -> None:
    _, objective = _build_objective(ObjectiveWeights(1.0, 0.0, 0.0, 0.0))
    score = objective.evaluate_subset([0, 1])

    assert float(score.components.effectiveness) == 0.0
    assert float(score.components.consistency) == 0.0
    assert float(score.components.collaboration) == 0.0
    assert float(score.components.confidence) > 0.0
