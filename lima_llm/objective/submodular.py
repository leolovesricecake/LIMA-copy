from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..backbone.base import BaseBackbone
from ..chunking.utils import complement_chunk_ids, compose_text_from_chunk_ids
from ..scoring import (
    collaboration_score,
    consistency_score,
)
from ..types import ScoreComponents, SubsetScore, TextChunk, normalize_subset
from ..utils import cosine_similarity


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
        enable_batch_prefetch: bool | None = None,
    ) -> None:
        self.backbone = backbone
        self.text = text
        self.chunks = list(chunks)
        self.chunk_embeddings = list(chunk_embeddings)
        self.target_label = int(target_label)
        self.verbalizers = list(verbalizers)
        self.weights = weights
        self.empty_text_token = empty_text_token
        if enable_batch_prefetch is None:
            env = os.getenv("LIMA_OURS_BATCH_PREFETCH", "1").strip().lower()
            self.enable_batch_prefetch = env not in ("0", "false", "off", "no")
        else:
            self.enable_batch_prefetch = bool(enable_batch_prefetch)

        self.cache: Dict[Tuple[int, ...], SubsetScore] = {}
        self._prob_cache: Dict[str, np.ndarray] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._chunk_distances = self._build_chunk_distance_matrix(self.chunk_embeddings)
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

    def _prefetch_prob_texts(self, texts: Sequence[str]) -> None:
        missing: List[str] = []
        seen = set()
        for text in texts:
            if text in self._prob_cache or text in seen:
                continue
            missing.append(text)
            seen.add(text)
        if not missing:
            return
        if self.enable_batch_prefetch:
            probs = np.asarray(
                self.backbone.predict_label_probs_batch(missing, self.verbalizers),
                dtype=np.float32,
            )
            for idx, text in enumerate(missing):
                self._prob_cache[text] = probs[idx]
            return
        for text in missing:
            self._prob_cache[text] = np.asarray(
                self.backbone.predict_label_probs(text, self.verbalizers),
                dtype=np.float32,
            )

    def _embed_text_cached(self, text: str) -> np.ndarray:
        if text not in self._embed_cache:
            self._embed_cache[text] = np.asarray(self.backbone.embed_text(text), dtype=np.float32)
        return self._embed_cache[text]

    @staticmethod
    def _build_chunk_distance_matrix(chunk_embeddings: Sequence[np.ndarray]) -> np.ndarray:
        n = len(chunk_embeddings)
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = cosine_similarity(chunk_embeddings[i], chunk_embeddings[j])
                dist[i, j] = float(1.0 - sim)
        return dist

    def _effectiveness_from_subset(self, subset: Sequence[int]) -> float:
        ids = sorted(set(int(i) for i in subset))
        if len(ids) <= 1:
            return 0.0

        local = self._chunk_distances[np.ix_(ids, ids)].astype(np.float64, copy=True)
        np.fill_diagonal(local, np.inf)
        mins = np.min(local, axis=1)
        mins = np.where(np.isfinite(mins), mins, 0.0)
        return float(np.sum(mins, dtype=np.float64))

    def _prefetch_embed_texts(self, texts: Sequence[str]) -> None:
        missing: List[str] = []
        seen = set()
        for text in texts:
            if text in self._embed_cache or text in seen:
                continue
            missing.append(text)
            seen.add(text)
        if not missing:
            return
        if self.enable_batch_prefetch:
            vectors = self.backbone.embed_texts(missing)
            for text, vec in zip(missing, vectors):
                self._embed_cache[text] = np.asarray(vec, dtype=np.float32)
            return
        for text in missing:
            self._embed_cache[text] = np.asarray(self.backbone.embed_text(text), dtype=np.float32)

    def _materialize_subsets(self, subset_keys: Sequence[Tuple[int, ...]]) -> None:
        pending: List[Tuple[int, ...]] = []
        for key in subset_keys:
            if key not in self.cache:
                pending.append(key)
        if not pending:
            return

        subset_texts: List[str] = []
        complement_texts: List[str] = []
        text_by_key: Dict[Tuple[int, ...], Tuple[str, str]] = {}
        for key in pending:
            subset_text = self._subset_text(key)
            complement_text = self._complement_text(key)
            subset_texts.append(subset_text)
            complement_texts.append(complement_text)
            text_by_key[key] = (subset_text, complement_text)

        self._prefetch_prob_texts(subset_texts)
        self._prefetch_embed_texts([*subset_texts, *complement_texts])

        for subset_key in pending:
            subset_text, complement_text = text_by_key[subset_key]
            label_probs = self._predict_prob_cached(subset_text)
            target_prob = float(label_probs[self.target_label])

            conf = target_prob
            eff = self._effectiveness_from_subset(subset_key)
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

    def _materialize_subset(self, subset_key: Tuple[int, ...]) -> None:
        self._materialize_subsets([subset_key])

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

    def evaluate_gains(
        self,
        subset: Sequence[int],
        candidates: Sequence[int],
    ) -> Tuple[SubsetScore, List[Tuple[int, float, SubsetScore]]]:
        base_key = normalize_subset(subset)
        augmented_pairs: List[Tuple[int, Tuple[int, ...]]] = []
        for candidate in candidates:
            aug_key = normalize_subset(tuple(list(base_key) + [candidate]))
            augmented_pairs.append((int(candidate), aug_key))

        keys = [base_key, *[key for _, key in augmented_pairs]]
        self._materialize_subsets(keys)

        base = self.cache[base_key]
        gains: List[Tuple[int, float, SubsetScore]] = []
        for candidate, aug_key in augmented_pairs:
            augmented = self.cache[aug_key]
            gains.append((candidate, float(augmented.total - base.total), augmented))
        return base, gains
