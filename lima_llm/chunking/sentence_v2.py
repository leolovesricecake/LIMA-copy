from __future__ import annotations

from typing import List, Sequence, Tuple

from ..types import TextChunk
from .diagnostics import is_orphan_punctuation_chunk_text

_SENTENCE_PUNCTS = {".", "!", "?"}
_TRAILING_SYMBOLS = set(
    "\"'`“”‘’()[]{}.,!?;:，。！？；：、…-–—"
)


def _merge_orphan_punctuation_spans(text: str, spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if end <= start:
            continue
        seg = text[start:end]
        if merged and is_orphan_punctuation_chunk_text(seg):
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
            continue
        merged.append((start, end))
    return merged


def sentence_chunk_v2(text: str) -> List[TextChunk]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")]

    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in _SENTENCE_PUNCTS:
            i += 1
            continue

        j = i + 1
        while j < n and text[j] in _SENTENCE_PUNCTS:
            j += 1

        k = j
        while k < n:
            t = k
            while t < n and text[t].isspace():
                t += 1
            u = t
            while u < n and text[u] in _TRAILING_SYMBOLS:
                u += 1
            if u > t:
                k = u
                continue
            if t > k:
                k = t
            break

        end = k
        if end > start:
            spans.append((start, end))
            start = end
        i = max(i + 1, end)

    if start < n:
        spans.append((start, n))

    spans = _merge_orphan_punctuation_spans(text, spans)
    if not spans:
        spans = [(0, len(text))]

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
