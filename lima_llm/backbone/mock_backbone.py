from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .base import BaseBackbone


class MockBackbone(BaseBackbone):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def tokenize_len(self, text: str) -> int:
        return max(1, len(text.split()))

    def _hash_vector(self, text: str, dim: int) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=(dim,)).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def predict_label_probs(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        self.forward_counters["predict_calls"] += 1
        scores = []
        for label in verbalizers:
            vec = self._hash_vector(text + "<lbl>" + str(label), 1)
            scores.append(float(vec[0]))
        arr = np.asarray(scores, dtype=np.float64)
        arr = arr - arr.max()
        probs = np.exp(arr)
        probs = probs / probs.sum()
        return probs.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        return self._hash_vector(text, self.embedding_dim)

    def gradient_chunk_importance(self, text: str, chunks, target_label: int, verbalizers: Sequence[str]) -> np.ndarray:
        self.forward_counters["gradient_calls"] += 1
        scores = []
        for chunk in chunks:
            key = f"{text}<chunk>{chunk.chunk_id}<target>{target_label}"
            scores.append(float(abs(self._hash_vector(key, 1)[0])))
        return np.asarray(scores, dtype=np.float32)
