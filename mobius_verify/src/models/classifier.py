from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from .base import RawTextScorer


class MockSentimentScorer(RawTextScorer):
    """A deterministic raw-score sentiment scorer for fast smoke tests."""

    def __init__(self, verbalizers: Sequence[str] = ("negative", "positive")) -> None:
        self.verbalizers = list(verbalizers)
        self.forward_calls = 0
        self.positive_words = {
            "good",
            "great",
            "excellent",
            "moving",
            "fun",
            "convincing",
            "love",
            "loved",
            "best",
            "bright",
            "smart",
            "joy",
        }
        self.negative_words = {
            "bad",
            "awful",
            "boring",
            "dull",
            "not",
            "worst",
            "hate",
            "hated",
            "flat",
            "mess",
            "poor",
        }

    def _raw_margin(self, text: str) -> float:
        tokens = [piece.strip(".,!?;:\"'()[]{}").lower() for piece in str(text).split()]
        pos = sum(1.0 for token in tokens if token in self.positive_words)
        neg = sum(1.0 for token in tokens if token in self.negative_words)
        not_good = 0.0
        for left, right in zip(tokens, tokens[1:]):
            if left == "not" and right in self.positive_words:
                not_good += 2.0
        length_term = 0.02 * math.tanh(len(tokens) / 12.0)
        return float(pos - neg - not_good + length_term)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.forward_calls += 1
        margins = np.asarray([self._raw_margin(text) for text in texts], dtype=np.float64)
        return np.stack([-margins, margins], axis=1)

    def snapshot_counters(self) -> Dict[str, float | int]:
        return {"mock_forward_calls": int(self.forward_calls)}


class HFVerbalizerScorer(RawTextScorer):
    """Causal-LM verbalizer scorer using mean conditional log probability as raw score."""

    def __init__(
        self,
        *,
        model_path: str,
        verbalizers: Sequence[str],
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 2048,
        trust_remote_code: bool = False,
    ) -> None:
        from lima_llm.backbone.hf_backbone import HFBackbone

        self.verbalizers = list(verbalizers)
        self.backbone = HFBackbone(
            model_path=str(model_path),
            device=str(device),
            dtype=str(dtype),
            max_length=int(max_length),
            trust_remote_code=bool(trust_remote_code),
        )

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.verbalizers)), dtype=np.float64)
        columns = [
            self.backbone._label_conditional_logprob_batch(list(texts), str(label))
            for label in self.verbalizers
        ]
        return np.stack(columns, axis=1).astype(np.float64)

    def snapshot_counters(self) -> Dict[str, float | int]:
        return self.backbone.snapshot_counters()


def build_text_scorer(config: Dict[str, Any], *, verbalizers: Sequence[str]) -> RawTextScorer:
    model_type = str(config.get("type", "mock_sentiment")).strip().lower()
    if model_type in {"mock", "mock_sentiment"}:
        return MockSentimentScorer(verbalizers=verbalizers)
    if model_type in {"hf", "hf_causal_lm", "causal_lm"}:
        return HFVerbalizerScorer(
            model_path=str(config["model_path"]),
            verbalizers=verbalizers,
            device=str(config.get("device", "cuda:0")),
            dtype=str(config.get("dtype", "bfloat16")),
            max_length=int(config.get("max_length", 2048)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unsupported model scorer type: {model_type!r}")

