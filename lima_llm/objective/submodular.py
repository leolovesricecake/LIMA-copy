from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from ..backbone.base import BaseBackbone
from ..chunking.utils import complement_chunk_ids, compose_text_from_chunk_ids
from ..scoring import (
    collaboration_score,
    consistency_score,
    effectiveness_score,
)
from ..types import ScoreComponents, SubsetScore, TextChunk, normalize_subset


@dataclass(frozen=True)
class ObjectiveWeights:
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    lambda4: float = 1.0


class TextSubmodularObjective:
    def __init__(
        self,
        backbone: BaseBackbone,
        text: str,
        chunks: Sequence[TextChunk],
        chunk_embeddings: Sequence[np.ndarray],
        target_label: int,
        verbalizers: Sequence[str],
        weights: ObjectiveWeights,
        empty_text_token: str = "<EMPTY>",
    ) -> None:
        self.backbone = backbone
        self.text = text
        self.chunks = list(chunks)
        self.chunk_embeddings = list(chunk_embeddings)
        self.target_label = int(target_label)
        self.verbalizers = list(verbalizers)
        self.weights = weights
        self.empty_text_token = empty_text_token

        self.cache: Dict[Tuple[int, ...], SubsetScore] = {}
        self._prob_cache: Dict[str, np.ndarray] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self.anchor_embedding = self._embed_text_cached(self.text if self.text else self.empty_text_token)

    def _subset_text(self, subset: Sequence[int]) -> str:
        text = compose_text_from_chunk_ids(self.chunks, subset)
        return text if text != "" else self.empty_text_token

    def _complement_text(self, subset: Sequence[int]) -> str:
        comp = complement_chunk_ids(self.chunks, subset)
        text = compose_text_from_chunk_ids(self.chunks, comp)
        return text if text != "" else self.empty_text_token

    def _predict_prob_cached(self, text: str) -> np.ndarray:
        if text not in self._prob_cache:
            self._prob_cache[text] = np.asarray(
                self.backbone.predict_label_probs(text, self.verbalizers),
                dtype=np.float32,
            )
        return self._prob_cache[text]

    def _embed_text_cached(self, text: str) -> np.ndarray:
        if text not in self._embed_cache:
            self._embed_cache[text] = np.asarray(self.backbone.embed_text(text), dtype=np.float32)
        return self._embed_cache[text]

    def _materialize_subset(self, subset_key: Tuple[int, ...]) -> None:
        if subset_key in self.cache:
            return

        subset_text = self._subset_text(subset_key)
        complement_text = self._complement_text(subset_key)

        label_probs = self._predict_prob_cached(subset_text)
        target_prob = float(label_probs[self.target_label])

        # Use target-class probability so F(S) is explicitly class-conditional.
        conf = target_prob
        eff = effectiveness_score(self.chunk_embeddings, subset_key)
        cons = consistency_score(self._embed_text_cached(subset_text), self.anchor_embedding)
        col = collaboration_score(self._embed_text_cached(complement_text), self.anchor_embedding)

        components = ScoreComponents(
            confidence=conf,
            effectiveness=eff,
            consistency=cons,
            collaboration=col,
        )

        total = (
            self.weights.lambda1 * conf
            + self.weights.lambda2 * eff
            + self.weights.lambda3 * cons
            + self.weights.lambda4 * col
        )

        self.cache[subset_key] = SubsetScore(
            subset_indices=subset_key,
            total=float(total),
            components=components,
            target_probability=target_prob,
            label_probabilities=tuple(float(x) for x in label_probs.tolist()),
        )

    def evaluate_subset(self, subset: Sequence[int]) -> SubsetScore:
        key = normalize_subset(subset)
        self._materialize_subset(key)
        return self.cache[key]

    def evaluate_gain(self, subset: Sequence[int], candidate: int) -> Tuple[float, SubsetScore, SubsetScore]:
        base_key = normalize_subset(subset)
        aug_key = normalize_subset(tuple(list(base_key) + [candidate]))
        self._materialize_subset(base_key)
        self._materialize_subset(aug_key)
        base = self.cache[base_key]
        augmented = self.cache[aug_key]
        return float(augmented.total - base.total), base, augmented
