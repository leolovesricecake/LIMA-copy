from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..utils import cosine_similarity


def build_chunk_distance_matrix(chunk_embeddings: Sequence[np.ndarray]) -> np.ndarray:
    n = len(chunk_embeddings)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        emb_i = chunk_embeddings[i]
        for j in range(i + 1, n):
            d = 1.0 - cosine_similarity(emb_i, chunk_embeddings[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def effectiveness_score_from_distance_matrix(
    distance_matrix: np.ndarray,
    selected_chunk_ids: Sequence[int],
) -> float:
    ids = sorted(set(int(i) for i in selected_chunk_ids))
    m = len(ids)

    if m <= 1:
        return 0.0

    mins = np.full(m, np.inf, dtype=np.float64)
    for i in range(m):
        ii = ids[i]
        for j in range(i + 1, m):
            jj = ids[j]
            d = float(distance_matrix[ii, jj])
            if d < mins[i]:
                mins[i] = d
            if d < mins[j]:
                mins[j] = d

    return float(np.sum(mins))


def effectiveness_score(
    chunk_embeddings: Sequence[np.ndarray],
    selected_chunk_ids: Sequence[int],
    *,
    distance_matrix: np.ndarray | None = None,
) -> float:
    if distance_matrix is not None:
        return effectiveness_score_from_distance_matrix(
            distance_matrix=distance_matrix,
            selected_chunk_ids=selected_chunk_ids,
        )

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
