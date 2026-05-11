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
        chunks: List[TextChunk] = self.chunker(sample.text)
        ok, msg = validate_chunk_coverage(sample.text, chunks)
        if not ok:
            raise ValueError(f"Chunk coverage invalid for {sample.sample_id}: {msg}")

        if verbose:
            for chunk in chunks:
                preview = chunk.text.replace("\n", " ")[:40]
                print(
                    f"  chunk#{chunk.chunk_id} [{chunk.start_char}:{chunk.end_char}] {preview!r}"
                )

        candidate_ids = [chunk.chunk_id for chunk in chunks]
        max_k = min(max(0, int(self.config.k)), len(candidate_ids))
        method = str(self.config.explain_method).strip().lower()

        if method == "ours":
            chunk_embeddings = [self.backbone.embed_text(chunk.text if chunk.text else "<EMPTY>") for chunk in chunks]
            objective = TextSubmodularObjective(
                backbone=self.backbone,
                text=sample.text,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                target_label=sample.label,
                verbalizers=self.verbalizers,
                weights=self.config.weights,
            )

            if self.config.search == "greedy":
                selected, trace = run_forward_greedy(objective, candidate_ids=candidate_ids, k=max_k)
            elif self.config.search == "bidirectional":
                selected, trace = run_bidirectional_search(objective, candidate_ids=candidate_ids, k=max_k)
            else:
                raise ValueError(f"Unsupported search method: {self.config.search}")

            selected_set = set(selected)
            singleton_gain: Dict[int, float] = {}
            for cid in candidate_ids:
                gain, _, _ = objective.evaluate_gain([], int(cid))
                singleton_gain[int(cid)] = float(gain)

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

        selected_text = compose_text_from_chunk_ids(chunks, selected)

        elapsed = time.time() - t0
        metadata = {
            "elapsed_seconds": elapsed,
            "chunk_count": len(chunks),
            "search": self.config.search,
            "k": self.config.k,
            "explain_method": method,
            "forward_counters": self.backbone.snapshot_counters(),
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
