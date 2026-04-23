from __future__ import annotations

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

    @staticmethod
    def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _predict_probs_cached(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        unique = self._dedupe_keep_order(texts)
        missing = [text for text in unique if text not in self._prob_cache]
        if missing:
            batched = self.backbone.predict_label_probs_batch(missing, self.verbalizers)
            for text, probs in zip(missing, batched):
                self._prob_cache[text] = np.asarray(probs, dtype=np.float32)
        return {text: self._prob_cache[text] for text in unique}

    def _embed_cached(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        unique = self._dedupe_keep_order(texts)
        missing = [text for text in unique if text not in self._embed_cache]
        if missing:
            embeddings = self.backbone.embed_texts(missing)
            for text, emb in zip(missing, embeddings):
                self._embed_cache[text] = np.asarray(emb, dtype=np.float32)
        return {text: self._embed_cache[text] for text in unique}

    def _embed_text_cached(self, text: str) -> np.ndarray:
        return self._embed_cached([text])[text]

    def _materialize_subsets(self, subset_keys: Sequence[Tuple[int, ...]]) -> None:
        missing_keys = [key for key in subset_keys if key not in self.cache]
        if not missing_keys:
            return

        subset_text_by_key: Dict[Tuple[int, ...], str] = {}
        complement_text_by_key: Dict[Tuple[int, ...], str] = {}
        subset_texts: List[str] = []
        embed_texts: List[str] = []

        for key in missing_keys:
            subset_text = self._subset_text(key)
            complement_text = self._complement_text(key)
            subset_text_by_key[key] = subset_text
            complement_text_by_key[key] = complement_text
            subset_texts.append(subset_text)
            embed_texts.append(subset_text)
            embed_texts.append(complement_text)

        self._predict_probs_cached(subset_texts)
        self._embed_cached(embed_texts)

        for key in missing_keys:
            subset_text = subset_text_by_key[key]
            complement_text = complement_text_by_key[key]

            label_probs = self._prob_cache[subset_text]
            target_prob = float(label_probs[self.target_label])

            # Use target-class probability so F(S) is explicitly class-conditional.
            conf = target_prob
            eff = effectiveness_score(self.chunk_embeddings, key)
            cons = consistency_score(self._embed_cache[subset_text], self.anchor_embedding)
            col = collaboration_score(self._embed_cache[complement_text], self.anchor_embedding)

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

    def evaluate_subset(self, subset: Sequence[int]) -> SubsetScore:
        key = normalize_subset(subset)
        self._materialize_subsets([key])
        return self.cache[key]

    def evaluate_subsets(self, subsets: Sequence[Sequence[int]]) -> List[SubsetScore]:
        keys = [normalize_subset(subset) for subset in subsets]
        self._materialize_subsets(keys)
        return [self.cache[key] for key in keys]

    def evaluate_gains(
        self,
        subset: Sequence[int],
        candidates: Sequence[int],
    ) -> Tuple[SubsetScore, Dict[int, float], Dict[int, SubsetScore]]:
        base_key = normalize_subset(subset)
        aug_keys = [normalize_subset(tuple(list(base_key) + [candidate])) for candidate in candidates]
        self._materialize_subsets([base_key, *aug_keys])

        base = self.cache[base_key]
        gains: Dict[int, float] = {}
        scores: Dict[int, SubsetScore] = {}
        for candidate, key in zip(candidates, aug_keys):
            score = self.cache[key]
            gains[int(candidate)] = float(score.total - base.total)
            scores[int(candidate)] = score
        return base, gains, scores

    def evaluate_gain(self, subset: Sequence[int], candidate: int) -> Tuple[float, SubsetScore, SubsetScore]:
        base, gains, scores = self.evaluate_gains(subset=subset, candidates=[candidate])
        augmented = scores[int(candidate)]
        return float(gains[int(candidate)]), base, augmented
