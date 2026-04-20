from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import cosine_similarity


def effectiveness_score(chunk_embeddings: Sequence[np.ndarray], selected_chunk_ids: Sequence[int]) -> float:
    ids = sorted(set(int(i) for i in selected_chunk_ids))
    if len(ids) <= 1:
        return 0.0

    selected = [chunk_embeddings[i] for i in ids]
    m = len(selected)
    dist = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            sim = cosine_similarity(selected[i], selected[j])
            dist[i, j] = 1.0 - sim

    mins = []
    for i in range(m):
        candidates = [dist[i, j] for j in range(m) if j != i]
        mins.append(min(candidates) if candidates else 0.0)
    return float(np.sum(mins))
