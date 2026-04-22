from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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

        self.anchor_embedding = self.backbone.embed_text(self.text if self.text else self.empty_text_token)
        self.cache: Dict[Tuple[int, ...], SubsetScore] = {}

    def _subset_text(self, subset: Sequence[int]) -> str:
        text = compose_text_from_chunk_ids(self.chunks, subset)
        return text if text != "" else self.empty_text_token

    def _complement_text(self, subset: Sequence[int]) -> str:
        comp = complement_chunk_ids(self.chunks, subset)
        text = compose_text_from_chunk_ids(self.chunks, comp)
        return text if text != "" else self.empty_text_token

    def evaluate_subset(self, subset: Sequence[int]) -> SubsetScore:
        key = normalize_subset(subset)
        if key in self.cache:
            return self.cache[key]

        subset_text = self._subset_text(key)
        complement_text = self._complement_text(key)

        label_probs = self.backbone.predict_label_probs(subset_text, self.verbalizers)
        target_prob = float(label_probs[self.target_label])

        # Use target-class probability so F(S) is explicitly class-conditional.
        conf = target_prob
        eff = effectiveness_score(self.chunk_embeddings, key)
        cons = consistency_score(self.backbone.embed_text(subset_text), self.anchor_embedding)
        col = collaboration_score(self.backbone.embed_text(complement_text), self.anchor_embedding)

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

        score = SubsetScore(
            subset_indices=key,
            total=float(total),
            components=components,
            target_probability=target_prob,
            label_probabilities=tuple(float(x) for x in label_probs.tolist()),
        )
        self.cache[key] = score
        return score

    def evaluate_gain(self, subset: Sequence[int], candidate: int) -> Tuple[float, SubsetScore, SubsetScore]:
        base = self.evaluate_subset(subset)
        augmented = self.evaluate_subset(tuple(list(normalize_subset(subset)) + [candidate]))
        return float(augmented.total - base.total), base, augmented
