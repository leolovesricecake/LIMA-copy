from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


CharSpan = Tuple[int, int]


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    text: str
    label: int
    label_text: Optional[str] = None
    rationale_char_spans: Tuple[CharSpan, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    start_char: int
    end_char: int
    text: str
    token_start: Optional[int] = None
    token_end: Optional[int] = None


@dataclass(frozen=True)
class ScoreComponents:
    confidence: float
    effectiveness: float
    consistency: float
    collaboration: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SubsetScore:
    subset_indices: Tuple[int, ...]
    total: float
    components: ScoreComponents
    target_probability: float
    label_probabilities: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["subset_indices"] = list(self.subset_indices)
        payload["label_probabilities"] = list(self.label_probabilities)
        return payload


@dataclass(frozen=True)
class ScoreTrace:
    step: int
    selected_chunk_id: int
    marginal_gain: float
    total_score: float
    components: ScoreComponents

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["components"] = self.components.to_dict()
        return payload


@dataclass
class ExplanationResult:
    sample_id: str
    dataset: str
    split: str
    label: int
    label_text: Optional[str]
    text: str
    chunks: List[TextChunk]
    selected_chunk_ids: List[int]
    selected_text: str
    scores: Dict[str, Any]
    trace: List[ScoreTrace]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "label": self.label,
            "label_text": self.label_text,
            "text": self.text,
            "chunks": [asdict(c) for c in self.chunks],
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "selected_text": self.selected_text,
            "scores": self.scores,
            "trace": [item.to_dict() for item in self.trace],
            "metadata": self.metadata,
        }


def normalize_subset(indices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(i) for i in indices)))
