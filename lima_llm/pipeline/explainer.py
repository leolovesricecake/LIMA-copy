from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..backbone.base import BaseBackbone
from ..chunking.utils import compose_text_from_chunk_ids, validate_chunk_coverage
from ..objective.submodular import ObjectiveWeights, TextSubmodularObjective
from ..search import run_bidirectional_search, run_forward_greedy
from ..types import ExplanationResult, ScoreComponents, ScoreTrace, TextChunk, TextSample


@dataclass(frozen=True)
class ExplainerConfig:
    dataset_name: str
    split: str
    k: int
    search: str
    weights: ObjectiveWeights
    explain_method: str
    seed: int


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _counter_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before.keys()).union(after.keys())
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in sorted(keys)}


def _ranking_digest(ids: Sequence[int]) -> str:
    raw = ",".join(str(int(x)) for x in ids)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _build_rank_trace(
    selected: Sequence[int],
    score_by_chunk: Dict[int, float],
) -> List[ScoreTrace]:
    out: List[ScoreTrace] = []
    running = 0.0
    empty = ScoreComponents(0.0, 0.0, 0.0, 0.0)
    for step, cid in enumerate(selected):
        gain = float(score_by_chunk.get(int(cid), 0.0))
        running += gain
        out.append(
            ScoreTrace(
                step=int(step),
                selected_chunk_id=int(cid),
                marginal_gain=gain,
                total_score=float(running),
                components=empty,
            )
        )
    return out


class TextLIMAExplainer:
    def __init__(
        self,
        backbone: BaseBackbone,
        chunker,
        verbalizers: Sequence[str],
        config: ExplainerConfig,
    ) -> None:
        self.backbone = backbone
        self.chunker = chunker
        self.verbalizers = list(verbalizers)
        self.config = config

    def explain_sample(self, sample: TextSample, verbose: bool = False) -> ExplanationResult:
        t0 = time.time()
        counters_before = self.backbone.snapshot_counters()
        t_chunk0 = time.perf_counter()
        chunks: List[TextChunk] = self.chunker(sample.text)
        ok, msg = validate_chunk_coverage(sample.text, chunks)
        if not ok:
            raise ValueError(f"Chunk coverage invalid for {sample.sample_id}: {msg}")
        chunk_build_seconds = time.perf_counter() - t_chunk0
        chunk_diagnostics = dict(getattr(self.chunker, "last_diagnostics", {}) or {})
        if not chunk_diagnostics:
            lengths = [max(0, int(chunk.end_char) - int(chunk.start_char)) for chunk in chunks]
            chunk_diagnostics = {
                "chunk_strategy_requested": "unknown",
                "chunk_strategy": "unknown",
                "chunk_count": len(chunks),
                "chunk_len_chars_min": float(min(lengths)) if lengths else 0.0,
                "chunk_len_chars_mean": (float(sum(lengths)) / float(len(lengths))) if lengths else 0.0,
                "chunk_len_chars_p90": float(max(lengths)) if lengths else 0.0,
                "chunk_len_chars_max": float(max(lengths)) if lengths else 0.0,
                "singleton_orphan_punctuation_chunks": 0,
                "cross_newline_boundary_chunks": 0,
                "fallback_applied": False,
                "fallback_reason": None,
                "pre_fallback_chunk_count": None,
            }

        if verbose:
            for chunk in chunks:
                preview = chunk.text.replace("\n", " ")[:40]
                print(
                    f"  chunk#{chunk.chunk_id} [{chunk.start_char}:{chunk.end_char}] {preview!r}"
                )

        candidate_ids = [chunk.chunk_id for chunk in chunks]
        max_k = min(max(0, int(self.config.k)), len(candidate_ids))
        method = str(self.config.explain_method).strip().lower()
        t_search0 = time.perf_counter()
        objective_cache_stats: Dict[str, object] = {}
        objective_compute_stats: Dict[str, object] = {}
        component_profile: Dict[str, object] = {}
        search_profile: Dict[str, object] = {}
        model_prefetch_seconds = 0.0

        if method == "ours":
            chunk_embeddings = []
            chunk_embed_cache: Dict[str, Sequence[float]] = {}
            for chunk in chunks:
                chunk_text = chunk.text if chunk.text else "<EMPTY>"
                if chunk_text not in chunk_embed_cache:
                    chunk_embed_cache[chunk_text] = self.backbone.embed_text(chunk_text)
                chunk_embeddings.append(chunk_embed_cache[chunk_text])

            objective = TextSubmodularObjective(
                backbone=self.backbone,
                text=sample.text,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                target_label=sample.label,
                verbalizers=self.verbalizers,
                weights=self.config.weights,
            )

            singleton_gain: Dict[int, float] = {}
            singleton_components: Dict[str, Dict[str, float]] = {}
            _, singleton_batch = objective.evaluate_gains([], candidate_ids)
            for cid, gain, score in singleton_batch:
                singleton_gain[int(cid)] = float(gain)
                singleton_components[str(int(cid))] = {
                    "confidence": float(score.components.confidence),
                    "effectiveness": float(score.components.effectiveness),
                    "consistency": float(score.components.consistency),
                    "collaboration": float(score.components.collaboration),
                    "total": float(score.total),
                }

            if self.config.search == "greedy":
                selected, trace = run_forward_greedy(
                    objective,
                    candidate_ids=candidate_ids,
                    k=max_k,
                    profile=search_profile,
                )
            elif self.config.search == "bidirectional":
                selected, trace = run_bidirectional_search(
                    objective,
                    candidate_ids=candidate_ids,
                    k=max_k,
                    profile=search_profile,
                )
            else:
                raise ValueError(f"Unsupported search method: {self.config.search}")

            if len(singleton_gain) != len(set(int(cid) for cid in candidate_ids)):
                for cid in candidate_ids:
                    if int(cid) in singleton_gain:
                        continue
                    gain, _, _ = objective.evaluate_gain([], int(cid))
                    singleton_gain[int(cid)] = float(gain)

            selected_set = set(selected)
            remaining = [cid for cid in candidate_ids if cid not in selected_set]
            remaining_sorted = sorted(remaining, key=lambda cid: (-singleton_gain[cid], cid))
            chunk_ranking = list(selected) + remaining_sorted
            chunk_scores = [float(singleton_gain.get(chunk.chunk_id, 0.0)) for chunk in chunks]

            final_score = objective.evaluate_subset(selected)
            scores = {
                "total": final_score.total,
                "confidence": final_score.components.confidence,
                "effectiveness": final_score.components.effectiveness,
                "consistency": final_score.components.consistency,
                "collaboration": final_score.components.collaboration,
                "target_probability": final_score.target_probability,
                "label_probabilities": list(final_score.label_probabilities),
            }
            objective_cache_stats = objective.cache_stats()
            model_prefetch_seconds = float(objective_cache_stats.get("model_prefetch_seconds", 0.0))
            component_enabled = objective.component_enabled()
            component_profile = {
                "component_enabled": component_enabled,
                "singleton_components": singleton_components,
            }
            objective_compute_stats = {
                "evaluate_gains_calls": int(objective_cache_stats.get("evaluate_gains_calls", 0)),
                "evaluate_gain_calls": int(objective_cache_stats.get("evaluate_gain_calls", 0)),
                "subset_cache_hit_rate": float(objective_cache_stats.get("subset_cache_hit_rate", 0.0)),
                "prob_cache_hit_rate": float(objective_cache_stats.get("prob_cache_hit_rate", 0.0)),
                "embed_cache_hit_rate": float(objective_cache_stats.get("embed_cache_hit_rate", 0.0)),
                "confidence_compute_calls": int(objective_cache_stats.get("confidence_compute_calls", 0)),
                "effectiveness_compute_calls": int(objective_cache_stats.get("effectiveness_compute_calls", 0)),
                "consistency_compute_calls": int(objective_cache_stats.get("consistency_compute_calls", 0)),
                "collaboration_compute_calls": int(objective_cache_stats.get("collaboration_compute_calls", 0)),
                "confidence_skipped_due_to_zero_lambda": int(
                    objective_cache_stats.get("confidence_skipped_due_to_zero_lambda", 0)
                ),
                "effectiveness_skipped_due_to_zero_lambda": int(
                    objective_cache_stats.get("effectiveness_skipped_due_to_zero_lambda", 0)
                ),
                "consistency_skipped_due_to_zero_lambda": int(
                    objective_cache_stats.get("consistency_skipped_due_to_zero_lambda", 0)
                ),
                "collaboration_skipped_due_to_zero_lambda": int(
                    objective_cache_stats.get("collaboration_skipped_due_to_zero_lambda", 0)
                ),
                "component_enabled": component_enabled,
                "timing": {
                    "text_build_seconds": float(objective_cache_stats.get("text_build_seconds", 0.0)),
                    "prefetch_seconds": float(objective_cache_stats.get("model_prefetch_seconds", 0.0)),
                    "component_compute_seconds": float(objective_cache_stats.get("component_compute_seconds", 0.0)),
                    "confidence_compute_seconds": float(objective_cache_stats.get("confidence_compute_seconds", 0.0)),
                    "effectiveness_compute_seconds": float(
                        objective_cache_stats.get("effectiveness_compute_seconds", 0.0)
                    ),
                    "consistency_compute_seconds": float(objective_cache_stats.get("consistency_compute_seconds", 0.0)),
                    "collaboration_compute_seconds": float(
                        objective_cache_stats.get("collaboration_compute_seconds", 0.0)
                    ),
                },
            }
        elif method == "random":
            seed = int(self.config.seed) + (_stable_hash_int(sample.sample_id) % 10_000)
            rng = random.Random(seed)
            chunk_ranking = list(candidate_ids)
            rng.shuffle(chunk_ranking)
            selected = list(chunk_ranking[:max_k])
            score_by_chunk = {
                int(cid): float(len(chunk_ranking) - idx)
                for idx, cid in enumerate(chunk_ranking)
            }
            chunk_scores = [float(score_by_chunk.get(chunk.chunk_id, 0.0)) for chunk in chunks]
            trace = _build_rank_trace(selected=selected, score_by_chunk=score_by_chunk)
            scores = {
                "total": float(sum(score_by_chunk.get(int(cid), 0.0) for cid in selected)),
                "confidence": 0.0,
                "effectiveness": 0.0,
                "consistency": 0.0,
                "collaboration": 0.0,
                "target_probability": 0.0,
                "label_probabilities": [],
            }
        elif method == "gradient":
            grad_scores = self.backbone.gradient_chunk_importance(
                sample.text,
                chunks,
                sample.label,
                self.verbalizers,
            )
            score_by_chunk: Dict[int, float] = {}
            for idx, chunk in enumerate(chunks):
                value = 0.0
                if chunk.chunk_id < len(grad_scores):
                    value = float(grad_scores[chunk.chunk_id])
                elif idx < len(grad_scores):
                    value = float(grad_scores[idx])
                score_by_chunk[int(chunk.chunk_id)] = value

            chunk_ranking = [
                cid
                for cid, _ in sorted(
                    ((int(chunk.chunk_id), float(score_by_chunk.get(int(chunk.chunk_id), 0.0))) for chunk in chunks),
                    key=lambda x: (-x[1], x[0]),
                )
            ]
            selected = list(chunk_ranking[:max_k])
            chunk_scores = [float(score_by_chunk.get(chunk.chunk_id, 0.0)) for chunk in chunks]
            trace = _build_rank_trace(selected=selected, score_by_chunk=score_by_chunk)
            scores = {
                "total": float(sum(score_by_chunk.get(int(cid), 0.0) for cid in selected)),
                "confidence": 0.0,
                "effectiveness": 0.0,
                "consistency": 0.0,
                "collaboration": 0.0,
                "target_probability": 0.0,
                "label_probabilities": [],
            }
        else:
            raise ValueError(f"Unsupported explain method: {method}")

        search_seconds = time.perf_counter() - t_search0
        selected_text = compose_text_from_chunk_ids(chunks, selected)

        elapsed = time.time() - t0
        counters_after = self.backbone.snapshot_counters()
        counter_delta = _counter_delta(counters_before, counters_after)
        chunk_id_set = set(candidate_ids)
        ranking_set = set(int(x) for x in chunk_ranking)
        metadata = {
            "elapsed_seconds": elapsed,
            "explain_timing_breakdown": {
                "chunk_build_seconds": float(chunk_build_seconds),
                "search_seconds": float(search_seconds),
                "model_prefetch_seconds": float(model_prefetch_seconds),
            },
            "chunk_diagnostics": chunk_diagnostics,
            "objective_cache_stats": objective_cache_stats,
            "objective_compute_stats": objective_compute_stats,
            "component_profile": component_profile,
            "search_profile": search_profile,
            "chunk_count": len(chunks),
            "search": self.config.search,
            "k": self.config.k,
            "explain_method": method,
            "forward_counters": counters_after,
            "forward_counters_before": counters_before,
            "forward_counters_after": counters_after,
            "forward_counters_delta": counter_delta,
            "ranking_digest": _ranking_digest(chunk_ranking),
            "selected_digest": _ranking_digest(selected),
            "ranking_checks": {
                "covers_all_chunks": ranking_set == chunk_id_set,
                "has_duplicates": len(chunk_ranking) != len(ranking_set),
                "selected_is_prefix": list(selected) == list(chunk_ranking[: len(selected)]),
            },
        }

        return ExplanationResult(
            explain_method=method,
            sample_id=sample.sample_id,
            dataset=self.config.dataset_name,
            split=self.config.split,
            label=sample.label,
            label_text=sample.label_text,
            text=sample.text,
            chunks=chunks,
            chunk_ranking=list(chunk_ranking),
            chunk_scores=list(chunk_scores),
            selected_chunk_ids=selected,
            selected_text=selected_text,
            scores=scores,
            trace=trace,
            metadata=metadata,
        )
