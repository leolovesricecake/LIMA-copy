"""Deterministic mock scorer for tests and smoke experiments."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .base import RawTextScorer


class MockSentimentScorer(RawTextScorer):
    """Assign deterministic sentiment scores with a small interaction term."""

    def __init__(self, verbalizers: Sequence[str] = ("negative", "positive")) -> None:
        """Initialize vocabulary and counters."""

        self.verbalizers = [str(value) for value in verbalizers]
        self.forward_calls = 0
        self.scored_rows = 0
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
        self.max_length = 2048
        self.tokenizer = None

    def _margin(self, text: str) -> float:
        """Compute a deterministic nonlinear sentiment margin."""

        tokens = [part.strip(".,!?;:\"'()[]{}").lower() for part in str(text).split()]
        positive = sum(token in self.positive_words for token in tokens)
        negative = sum(token in self.negative_words for token in tokens)
        negation = sum(
            2.0
            for left, right in zip(tokens, tokens[1:])
            if left == "not" and right in self.positive_words
        )
        return float(positive - negative - negation + 0.02 * math.tanh(len(tokens) / 12.0))

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Score a batch and count one physical forward per batch."""

        self.forward_calls += 1
        self.scored_rows += len(texts)
        margins = np.asarray([self._margin(text) for text in texts], dtype=np.float64)
        if len(self.verbalizers) == 2:
            return np.stack([-margins, margins], axis=1)
        columns = [
            np.cos(margins + float(index)) + 0.1 * float(index)
            for index in range(len(self.verbalizers))
        ]
        return np.stack(columns, axis=1)

    def snapshot_counters(self) -> Dict[str, int]:
        """Expose mock forward and row counts."""

        return {
            "model_forward_calls": int(self.forward_calls),
            "batch_calls": int(self.forward_calls),
            "batch_rows": int(self.scored_rows),
        }

