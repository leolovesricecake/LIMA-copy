from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np


class RawTextScorer(ABC):
    verbalizers: Sequence[str]

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return raw class scores with shape [len(texts), n_classes]."""

    def score_text(self, text: str) -> np.ndarray:
        rows = self.score_texts([text])
        return rows[0]

    def snapshot_counters(self) -> Dict[str, float | int]:
        return {}

