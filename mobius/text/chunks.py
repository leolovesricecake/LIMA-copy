"""Token, word, and adaptive chunking with complete character coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mobius.core.schema import TextChunk

from .adaptive import adaptive_chunk_with_stats
from .diagnostics import build_chunk_diagnostics


CHUNKERS = {"token", "word", "adaptive"}
EVAL_GRANULARITIES = {"token", "word"}
WORD_PATTERN = re.compile(r"\s*\S+\s*")


@dataclass(frozen=True)
class ChunkingResult:
    """Return chunks plus the effective segmentation strategy."""

    chunks: List[TextChunk]
    diagnostics: Dict[str, Any]
    fallback_used: bool
    strategy: str


def normalize_chunker(value: str | None) -> str:
    """Validate an explanation chunker name."""

    normalized = str(value or "word").strip().lower()
    if normalized not in CHUNKERS:
        raise ValueError(f"Unsupported chunker {value!r}; expected {sorted(CHUNKERS)}")
    return normalized


def normalize_eval_granularity(value: str | None) -> str:
    """Validate an evaluation perturbation unit name."""

    normalized = str(value or "word").strip().lower()
    if normalized not in EVAL_GRANULARITIES:
        raise ValueError(
            f"Unsupported eval granularity {value!r}; expected {sorted(EVAL_GRANULARITIES)}"
        )
    return normalized


def word_units(text: str) -> List[TextChunk]:
    """Split text into whitespace-preserving word spans."""

    chunks = [
        TextChunk(
            chunk_id=index,
            start_char=match.start(),
            end_char=match.end(),
            text=text[match.start() : match.end()],
        )
        for index, match in enumerate(WORD_PATTERN.finditer(text))
    ]
    return chunks or [TextChunk(0, 0, len(text), text)]


def repair_coverage(
    text: str,
    units: Sequence[TextChunk],
    *,
    keep_token_span: bool,
) -> List[TextChunk]:
    """Repair gaps and overlaps so concatenated chunks exactly reconstruct text."""

    if text == "":
        return [TextChunk(0, 0, 0, "")]
    spans: List[Tuple[int, int, int | None, int | None]] = []
    for unit in units:
        start = max(0, min(int(unit.start_char), len(text)))
        end = max(start, min(int(unit.end_char), len(text)))
        spans.append((start, end, unit.token_start, unit.token_end))
    if not spans:
        return [TextChunk(0, 0, len(text), text)]
    spans.sort(key=lambda item: (item[0], item[1]))
    starts: List[int] = []
    previous = 0
    for index, span in enumerate(spans):
        start = 0 if index == 0 else max(previous, span[0])
        starts.append(start)
        previous = start
    repaired: List[TextChunk] = []
    for index, span in enumerate(spans):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        repaired.append(
            TextChunk(
                chunk_id=index,
                start_char=start,
                end_char=end,
                text=text[start:end],
                token_start=int(span[2]) if keep_token_span and span[2] is not None else None,
                token_end=int(span[3]) if keep_token_span and span[3] is not None else None,
            )
        )
    return repaired


def token_units(text: str, tokenizer) -> Tuple[List[TextChunk], bool]:
    """Build tokenizer-offset units and signal a word fallback when unavailable."""

    if text == "":
        return [TextChunk(0, 0, 0, "")], False
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = list(encoded.get("offset_mapping") or [])
            units = [
                TextChunk(
                    chunk_id=index,
                    start_char=int(offset[0]),
                    end_char=int(offset[1]),
                    text=text[int(offset[0]) : int(offset[1])],
                    token_start=index,
                    token_end=index + 1,
                )
                for index, offset in enumerate(offsets)
                if offset is not None and len(offset) >= 2 and int(offset[1]) >= int(offset[0])
            ]
            if units:
                return repair_coverage(text, units, keep_token_span=True), False
        except Exception:
            pass
    return repair_coverage(text, word_units(text), keep_token_span=False), True


def validate_coverage(text: str, chunks: Sequence[TextChunk]) -> Tuple[bool, str]:
    """Check IDs, spans, per-chunk text, and exact full reconstruction."""

    if not chunks:
        return False, "no chunks"
    cursor = 0
    parts: List[str] = []
    for index, chunk in enumerate(chunks):
        if int(chunk.chunk_id) != index:
            return False, f"chunk id {chunk.chunk_id} is not contiguous"
        if int(chunk.start_char) != cursor or int(chunk.end_char) < cursor:
            return False, f"invalid span at chunk {index}"
        expected = text[int(chunk.start_char) : int(chunk.end_char)]
        if chunk.text != expected:
            return False, f"text mismatch at chunk {index}"
        cursor = int(chunk.end_char)
        parts.append(chunk.text)
    if cursor != len(text) or "".join(parts) != text:
        return False, "chunks do not reconstruct the complete text"
    return True, "ok"


def build_chunks(
    text: str,
    chunker: str = "word",
    tokenizer=None,
    *,
    adaptive_profile: str = "balanced",
    adaptive_overrides: Mapping[str, object] | None = None,
) -> ChunkingResult:
    """Build explanation players using one normalized public interface."""

    mode = normalize_chunker(chunker)
    fallback = False
    if mode == "token":
        chunks, fallback = token_units(text, tokenizer)
        strategy = "tokenizer_offset_mapping" if not fallback else "word_fallback"
    elif mode == "word":
        chunks = repair_coverage(text, word_units(text), keep_token_span=False)
        strategy = "word_whitespace_spans"
    else:
        chunks, adaptive_stats = adaptive_chunk_with_stats(
            text=text,
            profile=str(adaptive_profile),
            overrides=adaptive_overrides,
        )
        strategy = "adaptive"
    valid, reason = validate_coverage(text, chunks)
    if not valid:
        raise RuntimeError(f"Chunker returned invalid coverage: {reason}")
    extras = {"segmentation_strategy": strategy}
    if mode == "adaptive":
        extras.update(dict(adaptive_stats))
    diagnostics = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy=mode,
        effective_strategy="word" if fallback else mode,
        fallback_applied=fallback,
        fallback_reason="missing_tokenizer_offsets" if fallback else None,
        pre_fallback_chunk_count=None,
        extra_stats=extras,
    )
    return ChunkingResult(list(chunks), diagnostics, fallback, strategy)


def build_eval_units(
    text: str,
    granularity: str,
    tokenizer=None,
) -> ChunkingResult:
    """Build evaluation units independently from explanation chunks."""

    mode = normalize_eval_granularity(granularity)
    return build_chunks(text, mode, tokenizer)


def content_span(chunk: TextChunk) -> Tuple[int, int]:
    """Return the non-whitespace interior span used for overlap projection."""

    leading = len(chunk.text) - len(chunk.text.lstrip())
    trailing = len(chunk.text) - len(chunk.text.rstrip())
    start = int(chunk.start_char) + leading
    end = int(chunk.end_char) - trailing
    return (int(chunk.start_char), int(chunk.end_char)) if end <= start else (start, end)


def compose_text(chunks: Sequence[TextChunk], chunk_ids: Sequence[int]) -> str:
    """Concatenate selected chunks in original document order."""

    selected = {int(value) for value in chunk_ids}
    return "".join(chunk.text for chunk in chunks if int(chunk.chunk_id) in selected)


def project_ranking(
    eval_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    """Project chunk ranks to eval units by character-overlap-weighted rank."""

    ranks = {int(chunk_id): index for index, chunk_id in enumerate(chunk_ranking)}
    fallback = len(ranks) + len(chunks) + 1
    projected: List[Tuple[int, float]] = []
    for unit in eval_units:
        unit_start, unit_end = content_span(unit)
        weighted = 0.0
        overlap_total = 0
        for chunk in chunks:
            overlap = max(
                0,
                min(unit_end, int(chunk.end_char)) - max(unit_start, int(chunk.start_char)),
            )
            if overlap:
                weighted += overlap * ranks.get(int(chunk.chunk_id), fallback)
                overlap_total += overlap
        rank = weighted / overlap_total if overlap_total else fallback + int(unit.chunk_id)
        projected.append((int(unit.chunk_id), float(rank)))
    return [unit_id for unit_id, _ in sorted(projected, key=lambda item: (item[1], item[0]))]
