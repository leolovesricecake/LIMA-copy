from __future__ import annotations

import re
from typing import List

from ..types import TextChunk

_SENTENCE_SPLIT_RE = re.compile(r"([.!?]+[\]\)\"']*)(\s+|$)")


def sentence_chunk(text: str) -> List[TextChunk]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")]

    spans = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        end = match.end()
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))

    cleaned_spans = []
    for start, end in spans:
        if end <= start:
            continue
        cleaned_spans.append((start, end))

    if not cleaned_spans:
        cleaned_spans = [(0, len(text))]

    chunks: List[TextChunk] = []
    for idx, (start, end) in enumerate(cleaned_spans):
        chunks.append(
            TextChunk(
                chunk_id=idx,
                start_char=start,
                end_char=end,
                text=text[start:end],
            )
        )
    return chunks
