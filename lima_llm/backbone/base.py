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
            "batch_calls": 0,
            "batch_rows": 0,
            "model_forward_calls": 0,
            "oom_shrink_events": 0,
            "batch_tokenize_calls": 0,
            "batch_pack_calls": 0,
            "batch_forward_calls": 0,
            "batch_tokenize_seconds": 0.0,
            "batch_pack_seconds": 0.0,
            "batch_forward_seconds": 0.0,
        }

    def snapshot_counters(self) -> Dict[str, float | int]:
        return dict(self.forward_counters)

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
        if not texts:
            return np.zeros((0, len(verbalizers)), dtype=np.float32)
        return self._predict_label_probs_batch_impl(texts, verbalizers)

    def predict_label_scores(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        return self.predict_label_scores_batch([text], verbalizers)[0]

    def predict_label_scores_batch(
        self,
        texts: Sequence[str],
        verbalizers: Sequence[str],
    ) -> np.ndarray:
        """Return pre-softmax label scores.

        Backbones without native score access fall back to log probabilities.
        Causal-LM backbones override this with mean conditional log likelihoods.
        """

        if not texts:
            return np.zeros((0, len(verbalizers)), dtype=np.float32)
        probabilities = np.asarray(
            self.predict_label_probs_batch(texts, verbalizers),
            dtype=np.float64,
        )
        return np.log(np.clip(probabilities, 1e-30, 1.0)).astype(np.float32)

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def _embed_texts_impl(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_text(text) for text in texts]

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return self._embed_texts_impl(texts)

    def gradient_chunk_importance(self, text: str, chunks, target_label: int, verbalizers: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("Gradient baseline is not available for this backbone")
