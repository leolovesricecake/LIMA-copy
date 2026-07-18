from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from lima_llm.chunking.adaptive import adaptive_chunk_with_stats
from lima_llm.chunking.diagnostics import build_chunk_diagnostics
from lima_llm.chunking.utils import validate_chunk_coverage
from lima_llm.eval.units import repair_units_to_full_coverage, token_units_from_text, word_units_from_text
from lima_llm.types import TextChunk


_CHUNKERS = {"token", "word", "adaptive"}


@dataclass(frozen=True)
class ProxySPEXChunkingResult:
    chunks: List[TextChunk]
    diagnostics: Dict[str, object]
    fallback_used: bool
    segmentation_strategy: str


def normalize_proxyspex_chunker(chunker: str | None) -> str:
    value = str(chunker or "word").strip().lower()
    if value not in _CHUNKERS:
        raise ValueError(f"Unsupported ProxySPEX chunker: {chunker!r}. Expected one of {sorted(_CHUNKERS)}.")
    return value


def load_adaptive_overrides(raw: str | None) -> Dict[str, object] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None

    if text.startswith("{"):
        payload = json.loads(text)
    else:
        payload = json.loads(Path(text).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--adaptive-overrides-json must decode to a JSON object")
    return dict(payload)


def build_proxyspex_chunks(
    *,
    text: str,
    chunker: str,
    tokenizer,
    adaptive_profile: str = "balanced",
    adaptive_overrides: Mapping[str, object] | None = None,
) -> ProxySPEXChunkingResult:
    mode = normalize_proxyspex_chunker(chunker)
    if mode == "token":
        return _build_token_chunks(text=text, tokenizer=tokenizer)
    if mode == "word":
        return _build_word_chunks(text=text)
    return _build_adaptive_chunks(
        text=text,
        adaptive_profile=adaptive_profile,
        adaptive_overrides=adaptive_overrides,
    )


def _build_token_chunks(*, text: str, tokenizer) -> ProxySPEXChunkingResult:
    chunks, fallback_used = token_units_from_text(text=text, tokenizer=tokenizer)
    segmentation_strategy = (
        "tokenizer_offset_mapping" if not fallback_used else "whitespace_fallback_without_tokenizer_offsets"
    )
    diagnostics = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="token",
        effective_strategy="token" if not fallback_used else "word_fallback",
        fallback_applied=bool(fallback_used),
        fallback_reason="missing_tokenizer_offsets" if fallback_used else None,
        pre_fallback_chunk_count=None,
        extra_stats={"segmentation_strategy": segmentation_strategy},
    )
    return ProxySPEXChunkingResult(
        chunks=list(chunks),
        diagnostics=diagnostics,
        fallback_used=bool(fallback_used),
        segmentation_strategy=segmentation_strategy,
    )


def _build_word_chunks(*, text: str) -> ProxySPEXChunkingResult:
    chunks = repair_units_to_full_coverage(
        text=text,
        units=word_units_from_text(text),
        keep_token_span=False,
    )
    diagnostics = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="word",
        effective_strategy="word",
        fallback_applied=False,
        fallback_reason=None,
        pre_fallback_chunk_count=None,
        extra_stats={"segmentation_strategy": "word_whitespace_spans"},
    )
    return ProxySPEXChunkingResult(
        chunks=list(chunks),
        diagnostics=diagnostics,
        fallback_used=False,
        segmentation_strategy="word_whitespace_spans",
    )


def _build_adaptive_chunks(
    *,
    text: str,
    adaptive_profile: str,
    adaptive_overrides: Mapping[str, object] | None,
) -> ProxySPEXChunkingResult:
    chunks, adaptive_stats = adaptive_chunk_with_stats(
        text=text,
        profile=str(adaptive_profile or "balanced"),
        overrides=adaptive_overrides,
    )
    ok, msg = validate_chunk_coverage(text, chunks)
    if not ok:
        raise RuntimeError(f"ProxySPEX adaptive chunker returned invalid coverage: {msg}")

    diagnostics = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="adaptive",
        effective_strategy="adaptive",
        fallback_applied=False,
        fallback_reason=None,
        pre_fallback_chunk_count=len(chunks),
        extra_stats={**dict(adaptive_stats), "segmentation_strategy": "adaptive"},
    )
    return ProxySPEXChunkingResult(
        chunks=list(chunks),
        diagnostics=diagnostics,
        fallback_used=False,
        segmentation_strategy="adaptive",
    )
