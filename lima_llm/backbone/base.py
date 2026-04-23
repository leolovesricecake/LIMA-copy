from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np


class BaseBackbone(ABC):
    def __init__(self) -> None:
        self.forward_counters = {
            "predict_calls": 0,
            "embed_calls": 0,
            "gradient_calls": 0,
        }

    def snapshot_counters(self) -> Dict[str, int]:
        return dict(self.forward_counters)

    @abstractmethod
    def tokenize_len(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict_label_probs(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_label_probs_batch(self, texts: Sequence[str], verbalizers: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(verbalizers)), dtype=np.float32)
        rows = [self.predict_label_probs(text, verbalizers) for text in texts]
        return np.stack(rows, axis=0).astype(np.float32)

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]

    def gradient_chunk_importance(self, text: str, chunks, target_label: int, verbalizers: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("Gradient baseline is not available for this backbone")
