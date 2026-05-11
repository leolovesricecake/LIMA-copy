from __future__ import annotations

import os
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

    # Optional reference path used for strict A/B equivalence checks.
    # default=0 uses the allocation-reduced path below.
    if os.getenv("LIMA_EFFECTIVENESS_REFERENCE", "0").strip().lower() in ("1", "true", "on", "yes"):
        mins = []
        for i in range(m):
            candidates = [dist[i, j] for j in range(m) if j != i]
            mins.append(min(candidates) if candidates else 0.0)
        return float(np.sum(mins))

    total = 0.0
    for i in range(m):
        row = dist[i]
        best = None
        for j in range(m):
            if i == j:
                continue
            cur = float(row[j])
            if best is None or cur < best:
                best = cur
        total += 0.0 if best is None else best
    return float(total)
