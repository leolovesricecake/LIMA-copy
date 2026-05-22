from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from ..types import TextChunk
from .diagnostics import is_leading_close_punct_chunk_text, is_orphan_punctuation_chunk_text

_WORD_RE = re.compile(r"\S+")
_PUNCT_RE = re.compile(r"[.!?,;:，。！？；：]")


def _token_spans_from_whitespace(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
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
        spans.append((start, i))
    return spans


def _token_spans_from_tokenizer(text: str, tokenizer: Any) -> Tuple[List[Tuple[int, int]], str, bool]:
    if tokenizer is None:
        return _token_spans_from_whitespace(text), "whitespace_fallback", True

    text_len = len(text)
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping", None)
        if offsets:
            spans: List[Tuple[int, int]] = []
            for off in offsets:
                if off is None or len(off) < 2:
                    continue
                try:
                    start = max(0, min(int(off[0]), text_len))
                    end = max(0, min(int(off[1]), text_len))
                except Exception:
                    continue
                if end > start:
                    spans.append((start, end))
            if spans:
                return spans, "tokenizer_offset_mapping", False
    except Exception:
        pass

    return _token_spans_from_whitespace(text), "whitespace_fallback", True


def _chunk_token_span(chunk: TextChunk, token_spans: Sequence[Tuple[int, int]]) -> Tuple[int | None, int | None, int, List[int]]:
    overlap_ids: List[int] = []
    c_start = int(chunk.start_char)
    c_end = int(chunk.end_char)
    for idx, (t_start, t_end) in enumerate(token_spans):
        if t_end <= c_start or t_start >= c_end:
            continue
        overlap_ids.append(idx)

    if not overlap_ids:
        return None, None, 0, []

    token_start = int(min(overlap_ids))
    token_end = int(max(overlap_ids)) + 1
    return token_start, token_end, int(len(overlap_ids)), overlap_ids


def build_chunk_feature_payload(
    *,
    text: str,
    chunks: Sequence[TextChunk],
    tokenizer: Any = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    token_spans, token_alignment_mode, fallback_used = _token_spans_from_tokenizer(text=text, tokenizer=tokenizer)

    features: Dict[str, Dict[str, object]] = {}
    covered_token_ids = set()
    aligned_chunk_count = 0
    native_token_span_chunks = 0

    for chunk in chunks:
        leading_ws = len(chunk.text) - len(chunk.text.lstrip())
        trailing_ws = len(chunk.text) - len(chunk.text.rstrip())

        token_start, token_end, token_count, overlap_ids = _chunk_token_span(chunk=chunk, token_spans=token_spans)
        if token_count > 0:
            aligned_chunk_count += 1
            covered_token_ids.update(overlap_ids)

        if chunk.token_start is not None and chunk.token_end is not None:
            native_token_span_chunks += 1

        feature = {
            "char_len": int(max(0, int(chunk.end_char) - int(chunk.start_char))),
            "word_count": int(len(_WORD_RE.findall(chunk.text))),
            "punct_count": int(len(_PUNCT_RE.findall(chunk.text))),
            "newline_count": int(chunk.text.count("\n")),
            "leading_ws_chars": int(leading_ws),
            "trailing_ws_chars": int(trailing_ws),
            "token_start": token_start,
            "token_end": token_end,
            "token_count": int(token_count),
            "orphan_punctuation": bool(is_orphan_punctuation_chunk_text(chunk.text)),
            "leading_close_punct": bool(is_leading_close_punct_chunk_text(chunk.text)),
        }
        features[str(int(chunk.chunk_id))] = feature

    total_chunks = len(chunks)
    total_tokens = len(token_spans)
    covered_tokens = len(covered_token_ids)
    uncovered_tokens = max(0, total_tokens - covered_tokens)

    coverage: Dict[str, object] = {
        "token_alignment_mode": token_alignment_mode,
        "token_alignment_fallback_used": bool(fallback_used),
        "total_chunks": int(total_chunks),
        "aligned_chunks": int(aligned_chunk_count),
        "aligned_chunk_ratio": float(aligned_chunk_count / float(total_chunks)) if total_chunks > 0 else 0.0,
        "total_tokens": int(total_tokens),
        "covered_tokens": int(covered_tokens),
        "uncovered_tokens": int(uncovered_tokens),
        "token_coverage_ratio": float(covered_tokens / float(total_tokens)) if total_tokens > 0 else 1.0,
        "native_token_span_chunks": int(native_token_span_chunks),
    }

    return features, coverage
