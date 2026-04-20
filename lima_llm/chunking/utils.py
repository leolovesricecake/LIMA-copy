from __future__ import annotations

from typing import List, Sequence, Tuple

from ..types import TextChunk


def validate_chunk_coverage(text: str, chunks: Sequence[TextChunk]) -> Tuple[bool, str]:
    if len(chunks) == 0:
        return False, "no chunks"
    chunks_sorted = sorted(chunks, key=lambda c: c.start_char)
    if chunks_sorted[0].start_char != 0:
        return False, f"coverage starts at {chunks_sorted[0].start_char}, expected 0"
    last_end = 0
    for chunk in chunks_sorted:
        if chunk.start_char < last_end:
            return False, f"overlap around chunk {chunk.chunk_id}"
        if chunk.start_char != last_end:
            return False, f"gap [{last_end}, {chunk.start_char})"
        if chunk.end_char < chunk.start_char:
            return False, f"invalid span [{chunk.start_char}, {chunk.end_char})"
        last_end = chunk.end_char
    if last_end != len(text):
        return False, f"coverage ends at {last_end}, expected {len(text)}"
    return True, "ok"


def compose_text_from_chunk_ids(chunks: Sequence[TextChunk], chunk_ids: Sequence[int]) -> str:
    if not chunk_ids:
        return ""
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    selected = [lookup[i] for i in sorted(set(chunk_ids)) if i in lookup]
    return "".join(chunk.text for chunk in selected)


def chunk_char_length(chunks: Sequence[TextChunk], chunk_ids: Sequence[int]) -> int:
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    return sum(max(0, lookup[i].end_char - lookup[i].start_char) for i in set(chunk_ids) if i in lookup)


def complement_chunk_ids(chunks: Sequence[TextChunk], chunk_ids: Sequence[int]) -> List[int]:
    selected = set(int(i) for i in chunk_ids)
    return [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in selected]
