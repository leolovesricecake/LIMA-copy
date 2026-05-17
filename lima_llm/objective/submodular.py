from __future__ import annotations

import bisect
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..backbone.base import BaseBackbone
from ..scoring import (
    build_chunk_distance_matrix,
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
        self._subset_text_cache: Dict[Tuple[int, ...], str] = {}
        self._complement_text_cache: Dict[Tuple[int, ...], str] = {}
        self._all_chunk_ids: Tuple[int, ...] = tuple(sorted(int(chunk.chunk_id) for chunk in self.chunks))
        self._chunk_text_by_id: Dict[int, str] = {int(chunk.chunk_id): chunk.text for chunk in self.chunks}
        backbone_name = type(self.backbone).__name__.strip().lower()
        if backbone_name == "mockbackbone":
            self._chunk_distance_matrix = None
        else:
            self._chunk_distance_matrix = build_chunk_distance_matrix(self.chunk_embeddings)

        self._stats: Dict[str, float | int] = {
            "subset_requested": 0,
            "subset_cache_hits": 0,
            "subset_cache_misses": 0,
            "subset_materialized": 0,
            "subset_text_cache_hits": 0,
            "subset_text_cache_misses": 0,
            "complement_text_cache_hits": 0,
            "complement_text_cache_misses": 0,
            "prob_cache_hits": 0,
            "prob_cache_misses": 0,
            "embed_cache_hits": 0,
            "embed_cache_misses": 0,
            "prob_prefetch_calls": 0,
            "prob_prefetch_requested_texts": 0,
            "prob_prefetch_missing_texts": 0,
            "embed_prefetch_calls": 0,
            "embed_prefetch_requested_texts": 0,
            "embed_prefetch_missing_texts": 0,
            "model_prefetch_calls": 0,
            "model_prefetch_seconds": 0.0,
            "evaluate_subset_calls": 0,
            "evaluate_gain_calls": 0,
            "evaluate_gains_calls": 0,
        }

        self.anchor_embedding = self._embed_text_cached(self.text if self.text else self.empty_text_token)

    @staticmethod
    def _augment_subset_key(base_key: Tuple[int, ...], candidate: int) -> Tuple[int, ...]:
        cid = int(candidate)
        pos = bisect.bisect_left(base_key, cid)
        if pos < len(base_key) and base_key[pos] == cid:
            return base_key
        return base_key[:pos] + (cid,) + base_key[pos:]

    def _subset_text(self, subset_key: Tuple[int, ...]) -> str:
        cached = self._subset_text_cache.get(subset_key)
        if cached is not None:
            self._stats["subset_text_cache_hits"] = int(self._stats["subset_text_cache_hits"]) + 1
            return cached
        self._stats["subset_text_cache_misses"] = int(self._stats["subset_text_cache_misses"]) + 1
        if len(subset_key) == 0:
            text = self.empty_text_token
        else:
            text = "".join(self._chunk_text_by_id[idx] for idx in subset_key if idx in self._chunk_text_by_id)
            if text == "":
                text = self.empty_text_token
        self._subset_text_cache[subset_key] = text
        return text

    def _complement_text(self, subset_key: Tuple[int, ...]) -> str:
        cached = self._complement_text_cache.get(subset_key)
        if cached is not None:
            self._stats["complement_text_cache_hits"] = int(self._stats["complement_text_cache_hits"]) + 1
            return cached
        self._stats["complement_text_cache_misses"] = int(self._stats["complement_text_cache_misses"]) + 1

        complement_ids: List[int] = []
        cursor = 0
        subset_len = len(subset_key)
        for cid in self._all_chunk_ids:
            while cursor < subset_len and subset_key[cursor] < cid:
                cursor += 1
            if cursor < subset_len and subset_key[cursor] == cid:
                cursor += 1
                continue
            complement_ids.append(cid)

        text = "".join(self._chunk_text_by_id[idx] for idx in complement_ids)
        if text == "":
            text = self.empty_text_token
        self._complement_text_cache[subset_key] = text
        return text

    def _predict_prob_cached(self, text: str) -> np.ndarray:
        if text in self._prob_cache:
            self._stats["prob_cache_hits"] = int(self._stats["prob_cache_hits"]) + 1
            return self._prob_cache[text]
        self._stats["prob_cache_misses"] = int(self._stats["prob_cache_misses"]) + 1
        self._prob_cache[text] = np.asarray(
            self.backbone.predict_label_probs(text, self.verbalizers),
            dtype=np.float32,
        )
        return self._prob_cache[text]

    def _prefetch_prob_texts(self, texts: Sequence[str]) -> None:
        self._stats["prob_prefetch_calls"] = int(self._stats["prob_prefetch_calls"]) + 1
        self._stats["prob_prefetch_requested_texts"] = int(self._stats["prob_prefetch_requested_texts"]) + len(texts)

        missing: List[str] = []
        seen = set()
        for text in texts:
            if text in self._prob_cache:
                self._stats["prob_cache_hits"] = int(self._stats["prob_cache_hits"]) + 1
                continue
            if text in seen:
                continue
            missing.append(text)
            seen.add(text)

        self._stats["prob_prefetch_missing_texts"] = int(self._stats["prob_prefetch_missing_texts"]) + len(missing)
        if not missing:
            return

        self._stats["prob_cache_misses"] = int(self._stats["prob_cache_misses"]) + len(missing)
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
        if text in self._embed_cache:
            self._stats["embed_cache_hits"] = int(self._stats["embed_cache_hits"]) + 1
            return self._embed_cache[text]
        self._stats["embed_cache_misses"] = int(self._stats["embed_cache_misses"]) + 1
        self._embed_cache[text] = np.asarray(self.backbone.embed_text(text), dtype=np.float32)
        return self._embed_cache[text]

    def _prefetch_embed_texts(self, texts: Sequence[str]) -> None:
        self._stats["embed_prefetch_calls"] = int(self._stats["embed_prefetch_calls"]) + 1
        self._stats["embed_prefetch_requested_texts"] = int(self._stats["embed_prefetch_requested_texts"]) + len(texts)

        missing: List[str] = []
        seen = set()
        for text in texts:
            if text in self._embed_cache:
                self._stats["embed_cache_hits"] = int(self._stats["embed_cache_hits"]) + 1
                continue
            if text in seen:
                continue
            missing.append(text)
            seen.add(text)

        self._stats["embed_prefetch_missing_texts"] = int(self._stats["embed_prefetch_missing_texts"]) + len(missing)
        if not missing:
            return

        self._stats["embed_cache_misses"] = int(self._stats["embed_cache_misses"]) + len(missing)
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
            self._stats["subset_requested"] = int(self._stats["subset_requested"]) + 1
            if key in self.cache:
                self._stats["subset_cache_hits"] = int(self._stats["subset_cache_hits"]) + 1
                continue
            self._stats["subset_cache_misses"] = int(self._stats["subset_cache_misses"]) + 1
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

        t_prefetch0 = time.perf_counter()
        self._prefetch_prob_texts(subset_texts)
        self._prefetch_embed_texts([*subset_texts, *complement_texts])
        self._stats["model_prefetch_calls"] = int(self._stats["model_prefetch_calls"]) + 1
        self._stats["model_prefetch_seconds"] = float(self._stats["model_prefetch_seconds"]) + (
            time.perf_counter() - t_prefetch0
        )

        for subset_key in pending:
            subset_text, complement_text = text_by_key[subset_key]
            label_probs = self._predict_prob_cached(subset_text)
            target_prob = float(label_probs[self.target_label])

            conf = target_prob
            eff = effectiveness_score(
                self.chunk_embeddings,
                subset_key,
                distance_matrix=self._chunk_distance_matrix,
            )
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
            self._stats["subset_materialized"] = int(self._stats["subset_materialized"]) + 1

    def evaluate_subset(self, subset: Sequence[int]) -> SubsetScore:
        self._stats["evaluate_subset_calls"] = int(self._stats["evaluate_subset_calls"]) + 1
        key = normalize_subset(subset)
        self._materialize_subsets([key])
        return self.cache[key]

    def evaluate_gain(self, subset: Sequence[int], candidate: int) -> Tuple[float, SubsetScore, SubsetScore]:
        self._stats["evaluate_gain_calls"] = int(self._stats["evaluate_gain_calls"]) + 1
        base_key = normalize_subset(subset)
        aug_key = self._augment_subset_key(base_key, int(candidate))
        self._materialize_subsets([base_key, aug_key])
        base = self.cache[base_key]
        augmented = self.cache[aug_key]
        return float(augmented.total - base.total), base, augmented

    def evaluate_gains(
        self,
        subset: Sequence[int],
        candidates: Sequence[int],
    ) -> Tuple[SubsetScore, List[Tuple[int, float, SubsetScore]]]:
        self._stats["evaluate_gains_calls"] = int(self._stats["evaluate_gains_calls"]) + 1
        base_key = normalize_subset(subset)
        augmented_pairs: List[Tuple[int, Tuple[int, ...]]] = []
        for candidate in candidates:
            aug_key = self._augment_subset_key(base_key, int(candidate))
            augmented_pairs.append((int(candidate), aug_key))

        keys = [base_key, *[key for _, key in augmented_pairs]]
        self._materialize_subsets(keys)

        base = self.cache[base_key]
        gains: List[Tuple[int, float, SubsetScore]] = []
        for candidate, aug_key in augmented_pairs:
            augmented = self.cache[aug_key]
            gains.append((candidate, float(augmented.total - base.total), augmented))
        return base, gains

    def cache_stats(self) -> Dict[str, float | int]:
        subset_requested = int(self._stats["subset_requested"])
        subset_hits = int(self._stats["subset_cache_hits"])
        prob_total = int(self._stats["prob_cache_hits"]) + int(self._stats["prob_cache_misses"])
        embed_total = int(self._stats["embed_cache_hits"]) + int(self._stats["embed_cache_misses"])
        return {
            **self._stats,
            "subset_cache_entries": len(self.cache),
            "prob_cache_entries": len(self._prob_cache),
            "embed_cache_entries": len(self._embed_cache),
            "subset_cache_hit_rate": (float(subset_hits) / float(subset_requested)) if subset_requested > 0 else 0.0,
            "prob_cache_hit_rate": (
                float(self._stats["prob_cache_hits"]) / float(prob_total) if prob_total > 0 else 0.0
            ),
            "embed_cache_hit_rate": (
                float(self._stats["embed_cache_hits"]) / float(embed_total) if embed_total > 0 else 0.0
            ),
        }
