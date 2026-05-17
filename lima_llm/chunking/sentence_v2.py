from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from ..types import TextChunk
from .diagnostics import is_orphan_punctuation_chunk_text
from .sentence import sentence_chunk

_LEADING_CLOSE_PUNCT_RE = re.compile(r"^(\s*[\)\]\}]+\s*)(.*)$", flags=re.DOTALL)
_ABBREVIATION_SINGLETON_RE = re.compile(r"^(mr|mrs|ms|dr|prof|st|jr|sr)\s*\.\s*$", flags=re.IGNORECASE)


def _merge_orphan_punctuation_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int, int]:
    if not spans:
        return [], 0, 0
    merged: List[Tuple[int, int]] = []
    orphan_merge_count = 0
    orphan_before_count = 0
    for start, end in spans:
        if end <= start:
            continue
        seg = text[start:end]
        if is_orphan_punctuation_chunk_text(seg):
            orphan_before_count += 1
        if merged and is_orphan_punctuation_chunk_text(seg):
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
            orphan_merge_count += 1
            continue
        merged.append((start, end))
    return merged, orphan_merge_count, orphan_before_count


def _spans_to_chunks(text: str, spans: Sequence[Tuple[int, int]]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for idx, (chunk_start, chunk_end) in enumerate(spans):
        chunks.append(
            TextChunk(
                chunk_id=idx,
                start_char=chunk_start,
                end_char=chunk_end,
                text=text[chunk_start:chunk_end],
            )
        )
    return chunks


def _fix_leading_close_punct_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    if len(spans) <= 1:
        return list(spans), 0

    fixed: List[Tuple[int, int]] = []
    fix_count = 0
    for start, end in spans:
        if not fixed:
            fixed.append((start, end))
            continue

        seg = text[start:end]
        match = _LEADING_CLOSE_PUNCT_RE.match(seg)
        if not match:
            fixed.append((start, end))
            continue

        prefix = match.group(1)
        rest = match.group(2)
        if prefix.strip() == "" or rest.strip() == "":
            fixed.append((start, end))
            continue

        prefix_len = len(prefix)
        prev_start, prev_end = fixed[-1]
        fixed[-1] = (prev_start, prev_end + prefix_len)

        new_start = start + prefix_len
        if new_start < end:
            fixed.append((new_start, end))
        fix_count += 1

    return fixed, fix_count


def _is_abbreviation_singleton_text(text: str) -> bool:
    normalized = text.replace("\n", " ").strip()
    if normalized == "":
        return False
    return _ABBREVIATION_SINGLETON_RE.fullmatch(normalized) is not None


def _merge_abbreviation_singleton_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    if len(spans) <= 1:
        return list(spans), 0

    merged: List[Tuple[int, int]] = []
    merge_count = 0
    i = 0
    while i < len(spans):
        start, end = spans[i]
        seg = text[start:end]
        if i + 1 < len(spans) and _is_abbreviation_singleton_text(seg):
            _, next_end = spans[i + 1]
            merged.append((start, next_end))
            merge_count += 1
            i += 2
            continue
        merged.append((start, end))
        i += 1

    return merged, merge_count


def sentence_chunk_v2_with_stats(text: str) -> Tuple[List[TextChunk], Dict[str, int]]:
    """
    Safe v2: keep sentence boundary behavior aligned with sentence_chunk,
    and only merge orphan punctuation-only chunks into previous chunk.
    """
    base_chunks = sentence_chunk(text)
    spans = [(int(chunk.start_char), int(chunk.end_char)) for chunk in base_chunks]

    merged_spans, orphan_merge_count, orphan_before_count = _merge_orphan_punctuation_spans(text, spans)
    leading_close_fixed_spans, leading_close_fix_count = _fix_leading_close_punct_spans(text, merged_spans)
    abbreviation_merged_spans, abbreviation_merge_count = _merge_abbreviation_singleton_spans(
        text,
        leading_close_fixed_spans,
    )
    if not abbreviation_merged_spans:
        abbreviation_merged_spans = [(0, len(text))]

    chunks = _spans_to_chunks(text, abbreviation_merged_spans)
    orphan_after_count = sum(1 for chunk in chunks if is_orphan_punctuation_chunk_text(chunk.text))
    stats = {
        "orphan_merge_count": int(orphan_merge_count),
        "orphan_chunks_before_merge": int(orphan_before_count),
        "orphan_chunks_after_merge": int(orphan_after_count),
        "leading_close_punct_fix_count": int(leading_close_fix_count),
        "abbreviation_merge_count": int(abbreviation_merge_count),
    }
    return chunks, stats


def sentence_chunk_v2(text: str) -> List[TextChunk]:
    chunks, _ = sentence_chunk_v2_with_stats(text)
    return chunks
