from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..backbone.base import BaseBackbone
from ..chunking.utils import compose_text_from_chunk_ids, validate_chunk_coverage
from ..objective.submodular import ObjectiveWeights, TextSubmodularObjective
from ..search import run_bidirectional_search, run_forward_greedy
from ..types import ExplanationResult, TextChunk, TextSample


@dataclass(frozen=True)
class ExplainerConfig:
    dataset_name: str
    split: str
    k: int
    search: str
    weights: ObjectiveWeights


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

        candidate_ids = [chunk.chunk_id for chunk in chunks]
        if self.config.search == "greedy":
            selected, trace = run_forward_greedy(objective, candidate_ids=candidate_ids, k=self.config.k)
        elif self.config.search == "bidirectional":
            selected, trace = run_bidirectional_search(objective, candidate_ids=candidate_ids, k=self.config.k)
        else:
            raise ValueError(f"Unsupported search method: {self.config.search}")

        selected_text = compose_text_from_chunk_ids(chunks, selected)
        final_score = objective.evaluate_subset(selected)

        elapsed = time.time() - t0
        metadata = {
            "elapsed_seconds": elapsed,
            "chunk_count": len(chunks),
            "search": self.config.search,
            "k": self.config.k,
            "forward_counters": self.backbone.snapshot_counters(),
        }

        scores = {
            "total": final_score.total,
            "confidence": final_score.components.confidence,
            "effectiveness": final_score.components.effectiveness,
            "consistency": final_score.components.consistency,
            "collaboration": final_score.components.collaboration,
            "target_probability": final_score.target_probability,
            "label_probabilities": list(final_score.label_probabilities),
        }

        return ExplanationResult(
            sample_id=sample.sample_id,
            dataset=self.config.dataset_name,
            split=self.config.split,
            label=sample.label,
            label_text=sample.label_text,
            text=sample.text,
            chunks=chunks,
            selected_chunk_ids=selected,
            selected_text=selected_text,
            scores=scores,
            trace=trace,
            metadata=metadata,
        )
