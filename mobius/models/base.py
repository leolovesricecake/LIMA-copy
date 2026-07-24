"""Abstract all-class scorer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np


class RawTextScorer(ABC):
    """Score complete texts against a fixed ordered verbalizer list."""

    verbalizers: Sequence[str]
    tokenizer = None
    max_length = 2048

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return raw class scores with shape [texts, classes]."""

    def score_text(self, text: str) -> np.ndarray:
        """Score one text while preserving the batch implementation."""

        return self.score_texts([text])[0]

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Convert raw scores to class probabilities."""

        scores = np.asarray(self.score_texts(texts), dtype=np.float64)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / np.sum(values, axis=1, keepdims=True)

    def snapshot_counters(self) -> Dict[str, float | int]:
        """Return implementation-specific query and forward counters."""

        return {}

