import math

import numpy as np

from lima_llm.scoring import (
    collaboration_score,
    confidence_score,
    consistency_score,
    effectiveness_score,
)


def test_confidence_in_valid_range() -> None:
    probs = [0.2, 0.8]
    value = confidence_score(probs, num_classes=2)
    assert 0.0 <= value <= 1.0


def test_effectiveness_singleton_zero() -> None:
    emb = [np.array([1.0, 0.0], dtype=np.float32)]
    assert effectiveness_score(emb, [0]) == 0.0


def test_effectiveness_pair_non_negative() -> None:
    emb = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    value = effectiveness_score(emb, [0, 1])
    assert value >= 0.0


def test_consistency_and_collaboration_no_nan() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)

    cons = consistency_score(a, b)
    col = collaboration_score(c, b)

    assert not math.isnan(cons)
    assert not math.isnan(col)
