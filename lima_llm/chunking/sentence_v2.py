from __future__ import annotations

import importlib
from typing import Any, Dict, List, Sequence, Tuple

from ..types import TextChunk

_PYSBD_INSTALL_HINT = "pip install pysbd==0.3.4"


def _load_pysbd_module() -> Any:
    try:
        return importlib.import_module("pysbd")
    except Exception as exc:
        raise RuntimeError(
            "chunker=sentence_v2 requires optional dependency `pysbd` (MIT). "
            f"Install with `{_PYSBD_INSTALL_HINT}`."
        ) from exc


def _build_segmenter(pysbd_module: Any, *, char_span: bool) -> Any:
    kwargs: Dict[str, Any] = {"language": "en", "clean": False}
    if char_span:
        kwargs["char_span"] = True
    try:
        return pysbd_module.Segmenter(**kwargs)
    except TypeError:
        if char_span:
            return None
        raise


def _repair_spans_to_full_coverage(
    text: str,
    spans: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    text_len = len(text)
    if text_len == 0:
        return [(0, 0)]
    if not spans:
        return [(0, text_len)]

    normalized: List[Tuple[int, int]] = []
    for start, end in spans:
        s = max(0, min(int(start), text_len))
        e = max(0, min(int(end), text_len))
        if e <= s:
            continue
        normalized.append((s, e))
    if not normalized:
        return [(0, text_len)]

    normalized.sort(key=lambda x: (x[0], x[1]))

    starts: List[int] = []
    prev = 0
    for idx, (start, _end) in enumerate(normalized):
        cur = start
        if idx == 0:
            cur = 0
        elif cur < prev:
            cur = prev
        starts.append(cur)
        prev = cur

    repaired: List[Tuple[int, int]] = []
    for idx in range(len(normalized)):
        start = starts[idx]
        end = starts[idx + 1] if idx < len(normalized) - 1 else text_len
        if end > start:
            repaired.append((start, end))
    if not repaired:
        repaired = [(0, text_len)]
    return repaired


def _validate_char_spans(text: str, span_rows: Sequence[Any]) -> List[Tuple[int, int]] | None:
    text_len = len(text)
    spans: List[Tuple[int, int]] = []
    prev_end = 0
    for row in span_rows:
        start = getattr(row, "start", None)
        end = getattr(row, "end", None)
        sent = getattr(row, "sent", None)
        if start is None or end is None:
            return None
        try:
            s = int(start)
            e = int(end)
        except Exception:
            return None
        if s < 0 or e < s or e > text_len:
            return None
        if s < prev_end:
            return None
        if sent is not None:
            seg = text[s:e]
            if str(sent) != seg:
                return None
        if e > s:
            spans.append((s, e))
            prev_end = e
    return spans


def _spans_from_segments_by_cursor(
    text: str,
    segments: Sequence[str],
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for seg in segments:
        sent = str(seg)
        if sent == "":
            continue
        pos = text.find(sent, cursor)
        if pos < 0:
            return [(0, len(text))]
        end = pos + len(sent)
        spans.append((pos, end))
        cursor = end
    return spans


def sentence_chunk_v2_with_stats(text: str) -> Tuple[List[TextChunk], Dict[str, object]]:
    pysbd_module = _load_pysbd_module()
    segment_count = 0
    used_char_span = False
    used_span_rebuild = False

    spans: List[Tuple[int, int]] = []

    char_span_segmenter = _build_segmenter(pysbd_module, char_span=True)
    if char_span_segmenter is not None:
        try:
            char_rows = list(char_span_segmenter.segment(text))
        except Exception:
            char_rows = []
        segment_count = len(char_rows)
        validated = _validate_char_spans(text, char_rows)
        if validated:
            spans = validated
            used_char_span = True

    if not spans:
        segmenter = _build_segmenter(pysbd_module, char_span=False)
        raw_segments = list(segmenter.segment(text))
        segment_count = len(raw_segments)
        spans = _spans_from_segments_by_cursor(text, raw_segments)
        used_span_rebuild = True

    spans = _repair_spans_to_full_coverage(text, spans)

    chunks: List[TextChunk] = []
    for idx, (start, end) in enumerate(spans):
        chunks.append(
            TextChunk(
                chunk_id=idx,
                start_char=int(start),
                end_char=int(end),
                text=text[int(start) : int(end)],
            )
        )

    # Keep legacy keys for compatibility even though v2 no longer uses merge-based post-fixes.
    stats = {
        "orphan_merge_count": 0,
        "orphan_chunks_before_merge": 0,
        "orphan_chunks_after_merge": 0,
        "leading_close_punct_fix_count": 0,
        "abbreviation_merge_count": 0,
        "sentence_backend": "pysbd",
        "sentence_backend_char_span_used": bool(used_char_span),
        "sentence_backend_span_rebuild_used": bool(used_span_rebuild),
        "sentence_backend_segment_count": int(segment_count),
    }
    return chunks, stats


def sentence_chunk_v2(text: str) -> List[TextChunk]:
    chunks, _ = sentence_chunk_v2_with_stats(text)
    return chunks
