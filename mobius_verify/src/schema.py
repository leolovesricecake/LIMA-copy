from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WordFeature:
    feature_id: int
    word_text: str
    span_text: str
    start_char: int
    end_char: int
    word_start_char: int
    word_end_char: int
    token_start: Optional[int] = None
    token_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSpec:
    sample_id: str
    original_text: str
    normalized_model_text: str
    tokenizer_name: str
    token_count: int
    punctuation_attachment_policy: str
    features: List[WordFeature]

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_features": self.n_features,
            "word_texts": [feature.word_text for feature in self.features],
            "word_char_spans": [
                [feature.word_start_char, feature.word_end_char] for feature in self.features
            ],
            "feature_char_spans": [
                [feature.start_char, feature.end_char] for feature in self.features
            ],
            "word_token_spans": [
                [feature.token_start, feature.token_end] for feature in self.features
            ],
            "original_text": self.original_text,
            "normalized_model_text": self.normalized_model_text,
            "tokenizer_name": self.tokenizer_name,
            "token_count": self.token_count,
            "punctuation_attachment_policy": self.punctuation_attachment_policy,
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass(frozen=True)
class ProbeSpec:
    sample_id: str
    probe_id: str
    probe_strategy: str
    probe_word_indices: List[int]
    probe_word_texts: List[str]
    k: int
    conditioning_mode: str
    random_seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextRecord:
    sample_id: str
    text: str
    label: Optional[int] = None
    label_text: Optional[str] = None
    task: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata or {})
        return payload

