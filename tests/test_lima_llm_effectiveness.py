from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from lima_llm.scoring.effectiveness import effectiveness_score
from lima_llm.utils import cosine_similarity


def _reference_effectiveness(chunk_embeddings: Sequence[np.ndarray], selected_chunk_ids: Sequence[int]) -> float:
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


def _build_embeddings(n: int, dim: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vec = rng.normal(size=(dim,)).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        out.append(vec)
    return out


def test_effectiveness_matches_reference_impl() -> None:
    embs = _build_embeddings(n=12, dim=32, seed=20260511)
    rng = np.random.default_rng(42)
    original_flag = os.environ.get("LIMA_EFFECTIVENESS_REFERENCE")

    cases = [
        [],
        [0],
        [1],
        [0, 1],
        [2, 3, 4],
        [4, 4, 3, 2],
        [0, 5, 11, 3, 2],
    ]
    for _ in range(100):
        k = int(rng.integers(0, 8))
        ids = [int(x) for x in rng.integers(0, len(embs), size=(k,)).tolist()]
        cases.append(ids)

    try:
        for ids in cases:
            os.environ["LIMA_EFFECTIVENESS_REFERENCE"] = "1"
            got_ref_path = effectiveness_score(embs, ids)
            os.environ["LIMA_EFFECTIVENESS_REFERENCE"] = "0"
            got_opt_path = effectiveness_score(embs, ids)
            ref = _reference_effectiveness(embs, ids)

            assert abs(float(got_ref_path) - float(ref)) <= 1e-12
            assert abs(float(got_opt_path) - float(ref)) <= 1e-12
            assert abs(float(got_opt_path) - float(got_ref_path)) <= 1e-12
    finally:
        if original_flag is None:
            os.environ.pop("LIMA_EFFECTIVENESS_REFERENCE", None)
        else:
            os.environ["LIMA_EFFECTIVENESS_REFERENCE"] = original_flag
