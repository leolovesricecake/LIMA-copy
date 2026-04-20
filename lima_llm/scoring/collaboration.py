from __future__ import annotations

import numpy as np

from ..utils import cosine_similarity


def collaboration_score(complement_embedding: np.ndarray, anchor_embedding: np.ndarray) -> float:
    return float(1.0 - cosine_similarity(complement_embedding, anchor_embedding))
