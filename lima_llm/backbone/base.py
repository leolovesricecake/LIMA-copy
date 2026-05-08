from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np


class BaseBackbone(ABC):
    _VALID_EQUIVALENCE_MODES = {"strict_ref", "optimized_batch"}

    def __init__(self, equivalence_mode: str = "optimized_batch") -> None:
        self.forward_counters = {
            "predict_calls": 0,
            "predict_batch_calls": 0,
            "predict_model_forwards": 0,
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "embed_model_forwards": 0,
            "gradient_calls": 0,
        }
        self.equivalence_mode = "optimized_batch"
        self.set_equivalence_mode(equivalence_mode)

    def snapshot_counters(self) -> Dict[str, int]:
        return dict(self.forward_counters)

    def set_equivalence_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in self._VALID_EQUIVALENCE_MODES:
            raise ValueError(
                "Invalid equivalence mode: {!r}. Expected one of: {}".format(
                    mode, ", ".join(sorted(self._VALID_EQUIVALENCE_MODES))
                )
            )
        self.equivalence_mode = normalized

    @abstractmethod
    def tokenize_len(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict_label_probs(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def _predict_label_probs_batch_impl(self, texts: Sequence[str], verbalizers: Sequence[str]) -> np.ndarray:
        rows = [self.predict_label_probs(text, verbalizers) for text in texts]
        return np.stack(rows, axis=0).astype(np.float32)

    def predict_label_probs_batch(self, texts: Sequence[str], verbalizers: Sequence[str]) -> np.ndarray:
        self.forward_counters["predict_batch_calls"] += 1
        if not texts:
            return np.zeros((0, len(verbalizers)), dtype=np.float32)

        if self.equivalence_mode == "strict_ref":
            rows = [self.predict_label_probs(text, verbalizers) for text in texts]
            return np.stack(rows, axis=0).astype(np.float32)

        return self._predict_label_probs_batch_impl(texts, verbalizers)

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def _embed_texts_impl(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        self.forward_counters["embed_batch_calls"] += 1
        if self.equivalence_mode == "strict_ref":
            return [self.embed_text(text) for text in texts]
        return self._embed_texts_impl(texts)

    def gradient_chunk_importance(self, text: str, chunks, target_label: int, verbalizers: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("Gradient baseline is not available for this backbone")
