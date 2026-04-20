from __future__ import annotations

from typing import List, Optional

from ..types import TextChunk


def fixed_token_chunk(
    text: str,
    token_size: int = 64,
    tokenizer=None,
) -> List[TextChunk]:
    if token_size <= 0:
        raise ValueError("token_size must be positive")

    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")]

    if tokenizer is None:
        return _whitespace_chunk(text=text, token_size=token_size)

    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping", None)
        if offsets is None:
            return _whitespace_chunk(text=text, token_size=token_size)

        chunks: List[TextChunk] = []
        for idx, tok_start in enumerate(range(0, len(offsets), token_size)):
            tok_end = min(tok_start + token_size, len(offsets))
            char_start = offsets[tok_start][0]
            char_end = offsets[tok_end - 1][1]
            if char_end < char_start:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=idx,
                    start_char=char_start,
                    end_char=char_end,
                    text=text[char_start:char_end],
                    token_start=tok_start,
                    token_end=tok_end,
                )
            )
        if chunks:
            _force_coverage(text, chunks)
            return chunks
    except Exception:
        pass

    return _whitespace_chunk(text=text, token_size=token_size)


def _whitespace_token_spans(text: str):
    spans = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        end = i
        spans.append((start, end))
    if not spans:
        spans = [(0, len(text))]
    return spans


def _whitespace_chunk(text: str, token_size: int) -> List[TextChunk]:
    spans = _whitespace_token_spans(text)
    chunks: List[TextChunk] = []
    for idx, tok_start in enumerate(range(0, len(spans), token_size)):
        tok_end = min(tok_start + token_size, len(spans))
        char_start = spans[tok_start][0]
        char_end = spans[tok_end - 1][1]
        chunks.append(
            TextChunk(
                chunk_id=idx,
                start_char=char_start,
                end_char=char_end,
                text=text[char_start:char_end],
                token_start=tok_start,
                token_end=tok_end,
            )
        )
    _force_coverage(text, chunks)
    return chunks


def _force_coverage(text: str, chunks: List[TextChunk]) -> None:
    if not chunks:
        return
    # Expand chunk boundaries so spans form a full non-overlapping partition [0, len(text)).
    starts = [chunk.start_char for chunk in chunks]
    starts[0] = 0

    rebuilt: List[TextChunk] = []
    for idx, chunk in enumerate(chunks):
        start = starts[idx]
        if idx < len(chunks) - 1:
            end = starts[idx + 1]
        else:
            end = len(text)
        if end < start:
            end = start
        rebuilt.append(
            TextChunk(
                chunk_id=chunk.chunk_id,
                start_char=start,
                end_char=end,
                text=text[start:end],
                token_start=chunk.token_start,
                token_end=chunk.token_end,
            )
        )

    chunks[:] = rebuilt
