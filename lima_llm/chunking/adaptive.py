from __future__ import annotations

import importlib
import re
from typing import Any, Dict, List, Sequence, Tuple

from ..types import TextChunk
from .diagnostics import is_orphan_punctuation_chunk_text

_WORD_RE = re.compile(r"\S+")
_PUNCT_RE = re.compile(r"[.!?,;:，。！？；：]")
_SENT_END_EST_RE = re.compile(r"([.!?]+[\]\)\"']*)(\s+|$)")
_SENT_SPLIT_RE = re.compile(r"([.!?]+[\]\)\"']*)(\s+|$)")
_CLAUSE_SPLIT_RE = re.compile(r"([,;:，；：]+|[—–]+)(\s+|$)")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

_PROFILE_THRESHOLDS = {
    "conservative": {"short_max_words": 64, "medium_max_words": 256, "long_max_words": 768},
    "balanced": {"short_max_words": 96, "medium_max_words": 384, "long_max_words": 960},
    "aggressive": {"short_max_words": 128, "medium_max_words": 512, "long_max_words": 1200},
}

_LONG_SPLIT_MAX_WORDS_BY_BUCKET = {
    "short": 80,
    "medium": 80,
    "long": 120,
    "very_long": 200,
}

_VERY_LONG_SEED_PACK_MAX_WORDS = 180
_VERY_LONG_ADJACENT_PACK_MAX_WORDS = 200


def _resolve_profile(profile: str) -> Tuple[str, Dict[str, int]]:
    key = str(profile).strip().lower()
    if key not in _PROFILE_THRESHOLDS:
        key = "balanced"
    return key, dict(_PROFILE_THRESHOLDS[key])


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_punct(text: str) -> int:
    return len(_PUNCT_RE.findall(text))


def _count_sentence_end_est(text: str) -> int:
    return len(_SENT_END_EST_RE.findall(text))


def _spans_from_regex_boundaries(text: str, pattern: re.Pattern[str]) -> List[Tuple[int, int]]:
    if text == "":
        return [(0, 0)]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        end = int(match.end())
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return _cleanup_spans(text, spans)


def _cleanup_spans(text: str, spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    text_len = len(text)
    if text_len == 0:
        return [(0, 0)]
    cleaned: List[Tuple[int, int]] = []
    for start, end in spans:
        s = max(0, min(int(start), text_len))
        e = max(0, min(int(end), text_len))
        if e <= s:
            continue
        cleaned.append((s, e))
    if not cleaned:
        return [(0, text_len)]
    cleaned.sort(key=lambda x: (x[0], x[1]))
    repaired: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in cleaned:
        if start > cursor:
            repaired.append((cursor, start))
        s = max(start, cursor)
        if end > s:
            repaired.append((s, end))
            cursor = end
    if cursor < text_len:
        repaired.append((cursor, text_len))
    final = [(s, e) for s, e in repaired if e > s]
    return final or [(0, text_len)]


def _spans_to_chunks(text: str, spans: Sequence[Tuple[int, int]]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for idx, (start, end) in enumerate(spans):
        chunks.append(
            TextChunk(
                chunk_id=int(idx),
                start_char=int(start),
                end_char=int(end),
                text=text[int(start) : int(end)],
            )
        )
    return chunks


def _try_load_pysbd() -> Any | None:
    try:
        return importlib.import_module("pysbd")
    except Exception:
        return None


def _build_pysbd_segmenter(pysbd_module: Any, *, char_span: bool) -> Any | None:
    kwargs: Dict[str, Any] = {"language": "en", "clean": False}
    if char_span:
        kwargs["char_span"] = True
    try:
        return pysbd_module.Segmenter(**kwargs)
    except TypeError:
        if char_span:
            return None
        raise


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
        if sent is not None and str(sent) != text[s:e]:
            return None
        if e > s:
            spans.append((s, e))
            prev_end = e
    return spans


def _spans_from_segments_by_cursor(text: str, segments: Sequence[str]) -> List[Tuple[int, int]]:
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


def _sentence_spans_with_backend(text: str) -> Tuple[List[Tuple[int, int]], str, bool]:
    pysbd_module = _try_load_pysbd()
    if pysbd_module is not None:
        char_seg = _build_pysbd_segmenter(pysbd_module, char_span=True)
        if char_seg is not None:
            try:
                char_rows = list(char_seg.segment(text))
            except Exception:
                char_rows = []
            validated = _validate_char_spans(text, char_rows)
            if validated:
                return _cleanup_spans(text, validated), "pysbd_char_span", False
        try:
            seg = _build_pysbd_segmenter(pysbd_module, char_span=False)
            raw = list(seg.segment(text))
            spans = _spans_from_segments_by_cursor(text, raw)
            return _cleanup_spans(text, spans), "pysbd_segment", False
        except Exception:
            pass

    spans = _spans_from_regex_boundaries(text, _SENT_SPLIT_RE)
    return spans, "regex_sentence", True


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    return _spans_from_regex_boundaries(text, _PARAGRAPH_SPLIT_RE)


def _clause_spans(text: str) -> List[Tuple[int, int]]:
    return _spans_from_regex_boundaries(text, _CLAUSE_SPLIT_RE)


def _word_window_spans(text: str, start: int, end: int, max_words: int) -> List[Tuple[int, int]]:
    span_text = text[start:end]
    words = list(_WORD_RE.finditer(span_text))
    if len(words) <= max_words:
        return [(start, end)]
    out: List[Tuple[int, int]] = []
    idx = 0
    cursor = 0
    while idx < len(words):
        end_idx = min(idx + max_words, len(words)) - 1
        rel_end = int(words[end_idx].end())
        out.append((start + cursor, start + rel_end))
        cursor = rel_end
        idx = end_idx + 1
    if out and out[-1][1] < end:
        out[-1] = (out[-1][0], end)
    normalized: List[Tuple[int, int]] = []
    for s, e in out:
        ss = max(start, min(end, int(s)))
        ee = max(start, min(end, int(e)))
        if ee > ss:
            normalized.append((ss, ee))
    if not normalized:
        return [(start, end)]
    normalized.sort(key=lambda x: (x[0], x[1]))
    repaired: List[Tuple[int, int]] = []
    cursor = start
    for s, e in normalized:
        s2 = max(s, cursor)
        if s2 > cursor:
            repaired.append((cursor, s2))
        if e > s2:
            repaired.append((s2, e))
            cursor = e
    if cursor < end:
        repaired.append((cursor, end))
    final = [(s, e) for s, e in repaired if e > s]
    return final or [(start, end)]


def _split_overlong_span(
    text: str,
    start: int,
    end: int,
    max_words: int,
) -> List[Tuple[int, int]]:
    span_text = text[start:end]
    if _count_words(span_text) <= max_words:
        return [(start, end)]

    sentence_spans, _, _ = _sentence_spans_with_backend(span_text)
    if len(sentence_spans) > 1:
        out: List[Tuple[int, int]] = []
        for s_rel, e_rel in sentence_spans:
            piece_start = start + s_rel
            piece_end = start + e_rel
            if _count_words(text[piece_start:piece_end]) > max_words:
                out.extend(_split_overlong_span_clause_first(text, piece_start, piece_end, max_words))
            else:
                out.append((piece_start, piece_end))
        return _cleanup_spans(text, out)

    return _split_overlong_span_clause_first(text, start, end, max_words)


def _split_overlong_span_clause_first(
    text: str,
    start: int,
    end: int,
    max_words: int,
) -> List[Tuple[int, int]]:
    span_text = text[start:end]
    clause_spans = _clause_spans(span_text)
    if len(clause_spans) > 1:
        out: List[Tuple[int, int]] = []
        for s_rel, e_rel in clause_spans:
            piece_start = start + s_rel
            piece_end = start + e_rel
            if _count_words(text[piece_start:piece_end]) > max_words:
                out.extend(_word_window_spans(text, piece_start, piece_end, max_words))
            else:
                out.append((piece_start, piece_end))
        return _cleanup_spans(text, out)
    return _word_window_spans(text, start, end, max_words)


def _pack_adjacent_spans_by_word_budget(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    max_words: int,
) -> Tuple[List[Tuple[int, int]], int]:
    cleaned = _cleanup_spans(text, spans)
    if len(cleaned) <= 1:
        return list(cleaned), 0

    out: List[Tuple[int, int]] = []
    merge_count = 0

    cur_start, cur_end = cleaned[0]
    cur_words = _count_words(text[cur_start:cur_end])

    for start, end in cleaned[1:]:
        next_words = _count_words(text[start:end])
        if cur_words + next_words <= int(max_words):
            cur_end = end
            cur_words += next_words
            merge_count += 1
            continue

        out.append((cur_start, cur_end))
        cur_start, cur_end = start, end
        cur_words = next_words

    out.append((cur_start, cur_end))
    return _cleanup_spans(text, out), int(merge_count)


def _merge_invalid_spans(text: str, spans: Sequence[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    if len(spans) <= 1:
        return list(spans), 0

    merged: List[Tuple[int, int]] = []
    pending_prefix_start: int | None = None
    merge_count = 0

    for start, end in spans:
        seg = text[start:end]
        invalid = seg.strip() == "" or is_orphan_punctuation_chunk_text(seg)
        if invalid:
            merge_count += 1
            if merged:
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                if pending_prefix_start is None:
                    pending_prefix_start = start
            continue

        if pending_prefix_start is not None:
            start = pending_prefix_start
            pending_prefix_start = None
        merged.append((start, end))

    if pending_prefix_start is not None:
        if merged:
            head_start, head_end = merged[0]
            merged[0] = (pending_prefix_start, head_end)
        else:
            merged = [(0, len(text))]

    return _cleanup_spans(text, merged), int(merge_count)


def _merge_short_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    min_words: int,
    min_chars: int,
) -> Tuple[List[Tuple[int, int]], int]:
    if len(spans) <= 1:
        return list(spans), 0

    merged: List[Tuple[int, int]] = []
    pending_prefix_start: int | None = None
    merge_count = 0

    for start, end in spans:
        seg = text[start:end]
        words = _count_words(seg)
        chars = max(0, end - start)
        short = words < min_words or chars < min_chars
        if short:
            merge_count += 1
            if merged:
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                if pending_prefix_start is None:
                    pending_prefix_start = start
            continue

        if pending_prefix_start is not None:
            start = pending_prefix_start
            pending_prefix_start = None
        merged.append((start, end))

    if pending_prefix_start is not None:
        if merged:
            head_start, head_end = merged[0]
            merged[0] = (pending_prefix_start, head_end)
        else:
            merged = [(0, len(text))]

    return _cleanup_spans(text, merged), int(merge_count)


def _split_long_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    max_words: int,
) -> Tuple[List[Tuple[int, int]], int]:
    if len(spans) == 0:
        return [], 0

    out: List[Tuple[int, int]] = []
    split_count = 0
    for start, end in spans:
        seg = text[start:end]
        if _count_words(seg) > max_words:
            split_spans = _split_overlong_span(text, start, end, max_words=max_words)
            if len(split_spans) > 1:
                split_count += 1
            out.extend(split_spans)
        else:
            out.append((start, end))
    return _cleanup_spans(text, out), int(split_count)


def _bucket_for_word_count(word_count: int, thresholds: Dict[str, int]) -> str:
    if word_count <= int(thresholds["short_max_words"]):
        return "short"
    if word_count <= int(thresholds["medium_max_words"]):
        return "medium"
    if word_count <= int(thresholds["long_max_words"]):
        return "long"
    return "very_long"


def _initial_spans_for_bucket(text: str, bucket: str) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    if bucket == "short":
        spans = _clause_spans(text)
        return spans, {
            "adaptive_sentence_backend": "none",
            "adaptive_sentence_backend_fallback_used": False,
            "adaptive_sentence_backend_calls": 0,
            "adaptive_very_long_single_paragraph_fallback_used": False,
            "adaptive_very_long_seed_pack_merge_count": 0,
        }

    if bucket == "medium":
        spans, backend, fallback = _sentence_spans_with_backend(text)
        return spans, {
            "adaptive_sentence_backend": backend,
            "adaptive_sentence_backend_fallback_used": bool(fallback),
            "adaptive_sentence_backend_calls": 1,
            "adaptive_very_long_single_paragraph_fallback_used": False,
            "adaptive_very_long_seed_pack_merge_count": 0,
        }

    if bucket == "long":
        para_spans = _paragraph_spans(text)
        out: List[Tuple[int, int]] = []
        backend_used = "none"
        fallback_used = False
        backend_calls = 0
        for p_start, p_end in para_spans:
            p_text = text[p_start:p_end]
            sent_spans, backend, fallback = _sentence_spans_with_backend(p_text)
            backend_used = backend
            fallback_used = fallback_used or bool(fallback)
            backend_calls += 1
            for s_rel, e_rel in sent_spans:
                out.append((p_start + s_rel, p_start + e_rel))
        return _cleanup_spans(text, out), {
            "adaptive_sentence_backend": backend_used,
            "adaptive_sentence_backend_fallback_used": bool(fallback_used),
            "adaptive_sentence_backend_calls": int(backend_calls),
            "adaptive_very_long_single_paragraph_fallback_used": False,
            "adaptive_very_long_seed_pack_merge_count": 0,
        }

    para_spans = _paragraph_spans(text)
    if len(para_spans) <= 1:
        sentence_seed, backend, fallback = _sentence_spans_with_backend(text)
        packed, pack_merge_count = _pack_adjacent_spans_by_word_budget(
            text,
            sentence_seed,
            max_words=_VERY_LONG_SEED_PACK_MAX_WORDS,
        )
        return packed, {
            "adaptive_sentence_backend": backend,
            "adaptive_sentence_backend_fallback_used": bool(fallback),
            "adaptive_sentence_backend_calls": 1,
            "adaptive_very_long_single_paragraph_fallback_used": True,
            "adaptive_very_long_seed_pack_merge_count": int(pack_merge_count),
        }

    return para_spans, {
        "adaptive_sentence_backend": "none",
        "adaptive_sentence_backend_fallback_used": False,
        "adaptive_sentence_backend_calls": 0,
        "adaptive_very_long_single_paragraph_fallback_used": False,
        "adaptive_very_long_seed_pack_merge_count": 0,
    }


def adaptive_chunk_with_stats(
    text: str,
    *,
    profile: str = "balanced",
) -> Tuple[List[TextChunk], Dict[str, object]]:
    profile_key, thresholds = _resolve_profile(profile)
    word_count = _count_words(text)
    punct_count = _count_punct(text)
    newline_count = int(text.count("\n"))
    sentence_end_est = _count_sentence_end_est(text)
    bucket = _bucket_for_word_count(word_count, thresholds=thresholds)

    raw_spans, backend_stats = _initial_spans_for_bucket(text, bucket=bucket)
    raw_spans = _cleanup_spans(text, raw_spans)
    after_invalid, invalid_merge_count = _merge_invalid_spans(text, raw_spans)
    after_short, short_merge_count = _merge_short_spans(
        text,
        after_invalid,
        min_words=6,
        min_chars=24,
    )
    after_long, long_split_count = _split_long_spans(
        text,
        after_short,
        max_words=int(_LONG_SPLIT_MAX_WORDS_BY_BUCKET[bucket]),
    )
    after_post_long_invalid, post_long_invalid_merge_count = _merge_invalid_spans(text, after_long)
    after_post_long_short, post_long_short_merge_count = _merge_short_spans(
        text,
        after_post_long_invalid,
        min_words=6,
        min_chars=24,
    )

    adjacent_pack_merge_count = 0
    packed_spans = list(after_post_long_short)
    if bucket == "very_long":
        packed_spans, adjacent_pack_merge_count = _pack_adjacent_spans_by_word_budget(
            text,
            after_post_long_short,
            max_words=_VERY_LONG_ADJACENT_PACK_MAX_WORDS,
        )

    final_spans = _cleanup_spans(text, packed_spans)
    chunks = _spans_to_chunks(text, final_spans)

    stats: Dict[str, object] = {
        "adaptive_enabled": True,
        "adaptive_profile": profile_key,
        "adaptive_bucket": bucket,
        "adaptive_features": {
            "word_count": int(word_count),
            "punct_count": int(punct_count),
            "newline_count": int(newline_count),
            "sentence_end_count_est": int(sentence_end_est),
        },
        "adaptive_postprocess": {
            "invalid_merge_count": int(invalid_merge_count),
            "short_merge_count": int(short_merge_count),
            "long_split_count": int(long_split_count),
            "post_long_invalid_merge_count": int(post_long_invalid_merge_count),
            "post_long_short_merge_count": int(post_long_short_merge_count),
            "adjacent_pack_merge_count": int(adjacent_pack_merge_count),
        },
        "adaptive_stage_chunk_counts": {
            "raw": int(len(raw_spans)),
            "after_invalid": int(len(after_invalid)),
            "after_short": int(len(after_short)),
            "after_long": int(len(after_long)),
            "after_post_long_merge": int(len(after_post_long_short)),
            "final": int(len(final_spans)),
        },
        "adaptive_long_split_max_words": int(_LONG_SPLIT_MAX_WORDS_BY_BUCKET[bucket]),
    }
    stats.update(backend_stats)
    return chunks, stats


def adaptive_chunk(text: str, *, profile: str = "balanced") -> List[TextChunk]:
    chunks, _ = adaptive_chunk_with_stats(text=text, profile=profile)
    return chunks
