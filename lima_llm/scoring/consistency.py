from __future__ import annotations

import numpy as np

from ..utils import cosine_similarity


def consistency_score(subset_embedding: np.ndarray, anchor_embedding: np.ndarray) -> float:
    return float(cosine_similarity(subset_embedding, anchor_embedding))
