from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import cosine_similarity


def effectiveness_score(
    chunk_embeddings: Sequence[np.ndarray],
    selected_chunk_ids: Sequence[int],
) -> float:
    ids = sorted(set(int(i) for i in selected_chunk_ids))
    m = len(ids)

    if m <= 1:
        return 0.0

    mins = np.full(m, np.inf, dtype=np.float64)

    for i in range(m):
        emb_i = chunk_embeddings[ids[i]]

        for j in range(i + 1, m):
            d = 1.0 - cosine_similarity(emb_i, chunk_embeddings[ids[j]])

            if d < mins[i]:
                mins[i] = d
            if d < mins[j]:
                mins[j] = d

    return float(np.sum(mins))