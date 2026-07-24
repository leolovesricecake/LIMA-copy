"""Small shared dataclasses used across methods and baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


CharSpan = Tuple[int, int]
SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class TextSample:
    """Represent one labeled text example."""

    sample_id: str
    text: str
    label: int
    label_text: Optional[str] = None
    rationale_char_spans: Tuple[CharSpan, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """Represent one explanation player aligned to a character span."""

    chunk_id: int
    start_char: int
    end_char: int
    text: str
    token_start: Optional[int] = None
    token_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the chunk without implementation-specific objects."""

        return asdict(self)


@dataclass
class DatasetBundle:
    """Bundle samples with the label names and verbalizers used for scoring."""

    dataset_name: str
    split: str
    samples: List[TextSample]
    label_names: List[str]
    verbalizers: List[str]


@dataclass(frozen=True)
class SamplingResult:
    """Return unique coalition masks together with compact diagnostics."""

    masks: List[int]
    diagnostics: Dict[str, Any]


@dataclass
class AttributionResult:
    """Represent one schema-v2 explanation before it is written to disk."""

    sample_id: str
    gold_label: int
    predicted_label: int
    target_label: int
    text: str
    chunks: List[TextChunk]
    node_scores: List[float]
    ranking: List[int]
    selected_ids: List[int]
    attribution_cost: Dict[str, Any]
    method_summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, output_level: str = "standard") -> Dict[str, Any]:
        """Serialize fields allowed by the requested output level."""

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": str(self.sample_id),
            "gold_label": int(self.gold_label),
            "predicted_label": int(self.predicted_label),
            "target_label": int(self.target_label),
            "text": str(self.text),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "node_scores": [float(value) for value in self.node_scores],
            "ranking": [int(value) for value in self.ranking],
            "selected_ids": [int(value) for value in self.selected_ids],
            "attribution_cost": dict(self.attribution_cost),
        }
        if output_level in {"standard", "debug"}:
            payload["method_summary"] = dict(self.method_summary)
        return payload

