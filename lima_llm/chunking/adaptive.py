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

_MIN_EFFECTIVE_CHUNKS = 5
_SHORT_FLOOR_MIN_WORDS = 15
_VERY_LONG_FRAGMENTATION_MAX_CHUNKS = 48
_VERY_LONG_FRAGMENTATION_RATIO_THRESHOLD = 24.0
_FRAGMENTATION_TARGET_WORDS = 32


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


def _split_point_from_words_or_chars(text: str, start: int, end: int) -> int | None:
    span_text = text[start:end]
    words = list(_WORD_RE.finditer(span_text))
    if len(words) >= 2:
        mid = len(words) // 2
        left_end = int(words[max(0, mid - 1)].end())
        split = start + left_end
        if start < split < end:
            return split
        right_start = int(words[mid].start())
        split = start + right_start
        if start < split < end:
            return split

    width = end - start
    if width <= 1:
        return None
    split = start + (width // 2)
    if start < split < end:
        return split
    return None


def _pick_preferred_boundary(start: int, end: int, boundaries: Sequence[int]) -> int | None:
    candidates = [int(b) for b in boundaries if start < int(b) < end]
    if not candidates:
        return None
    mid = (start + end) / 2.0
    return int(min(candidates, key=lambda b: (abs(float(b) - mid), abs((end - start) / 2.0 - b), b)))


def _collect_short_bucket_boundaries(text: str) -> List[int]:
    text_len = len(text)
    boundaries = set()
    clause_spans = _clause_spans(text)
    for _start, end in clause_spans[:-1]:
        if 0 < end < text_len:
            boundaries.add(int(end))
    sent_spans, _, _ = _sentence_spans_with_backend(text)
    for _start, end in sent_spans[:-1]:
        if 0 < end < text_len:
            boundaries.add(int(end))
    return sorted(boundaries)


def _apply_effective_chunk_floor(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    target_chunks: int,
) -> Tuple[List[Tuple[int, int]], bool, int]:
    work = [tuple((int(s), int(e))) for s, e in spans if int(e) > int(s)]
    if len(work) >= int(target_chunks):
        return _cleanup_spans(text, work), False, 0

    preferred_boundaries = _collect_short_bucket_boundaries(text)
    split_count = 0
    while len(work) < int(target_chunks):
        idx = max(range(len(work)), key=lambda i: (work[i][1] - work[i][0], -i))
        start, end = work[idx]
        if end - start <= 1:
            break

        split_point = _pick_preferred_boundary(start, end, preferred_boundaries)
        if split_point is None:
            split_point = _split_point_from_words_or_chars(text, start, end)
        if split_point is None or split_point <= start or split_point >= end:
            break

        left = (start, int(split_point))
        right = (int(split_point), end)
        work[idx : idx + 1] = [left, right]
        split_count += 1

    final_spans = _cleanup_spans(text, work)
    return final_spans, bool(split_count > 0), int(split_count)


def _pack_adjacent_spans_by_word_budget(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    max_chunks: int,
    target_words: int,
) -> List[Tuple[int, int]]:
    work = [tuple((int(s), int(e))) for s, e in spans if int(e) > int(s)]
    if len(work) <= int(max_chunks):
        return _cleanup_spans(text, work)

    packed: List[Tuple[int, int]] = []
    cur_start, cur_end = work[0]
    cur_words = _count_words(text[cur_start:cur_end])
    for s, e in work[1:]:
        seg_words = _count_words(text[s:e])
        if cur_words + seg_words <= int(target_words):
            cur_end = e
            cur_words += seg_words
            continue
        packed.append((cur_start, cur_end))
        cur_start, cur_end = s, e
        cur_words = seg_words
    packed.append((cur_start, cur_end))

    packed = _cleanup_spans(text, packed)
    if len(packed) <= int(max_chunks):
        return packed

    # If still too many chunks, keep greedily merging smallest adjacent pairs.
    merged = list(packed)
    while len(merged) > int(max_chunks):
        best_idx = None
        best_cost = None
        for i in range(len(merged) - 1):
            s0, e0 = merged[i]
            s1, e1 = merged[i + 1]
            merged_words = _count_words(text[s0:e1])
            overflow = max(0, merged_words - int(target_words))
            seg_len = e1 - s0
            cost = (overflow, seg_len, i)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = i
        if best_idx is None:
            break
        s0, _e0 = merged[best_idx]
        _s1, e1 = merged[best_idx + 1]
        merged[best_idx : best_idx + 2] = [(s0, e1)]

    return _cleanup_spans(text, merged)


def _apply_fragmentation_guard(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    bucket: str,
    raw_count: int,
    word_count: int,
) -> Tuple[List[Tuple[int, int]], bool, int, int]:
    before = int(len(spans))
    if bucket != "very_long":
        return _cleanup_spans(text, spans), False, before, before

    should_apply = False
    if raw_count <= 1 and before > _VERY_LONG_FRAGMENTATION_MAX_CHUNKS:
        should_apply = True
    elif raw_count > 0 and (float(before) / float(raw_count)) > _VERY_LONG_FRAGMENTATION_RATIO_THRESHOLD:
        should_apply = True

    if not should_apply:
        return _cleanup_spans(text, spans), False, before, before

    target_by_words = max(1, int((int(word_count) + _FRAGMENTATION_TARGET_WORDS - 1) // _FRAGMENTATION_TARGET_WORDS))
    target_chunks = min(_VERY_LONG_FRAGMENTATION_MAX_CHUNKS, target_by_words)
    guarded = _pack_adjacent_spans_by_word_budget(
        text,
        spans,
        max_chunks=target_chunks,
        target_words=_FRAGMENTATION_TARGET_WORDS,
    )
    after = int(len(guarded))
    return guarded, True, before, after


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
        }

    if bucket == "medium":
        spans, backend, fallback = _sentence_spans_with_backend(text)
        return spans, {
            "adaptive_sentence_backend": backend,
            "adaptive_sentence_backend_fallback_used": bool(fallback),
            "adaptive_sentence_backend_calls": 1,
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
        }

    spans = _paragraph_spans(text)
    return spans, {
        "adaptive_sentence_backend": "none",
        "adaptive_sentence_backend_fallback_used": False,
        "adaptive_sentence_backend_calls": 0,
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
    after_effective_floor = _cleanup_spans(text, after_long)
    effective_floor_applied = False
    effective_floor_split_count = 0
    if bucket == "short" and int(word_count) >= _SHORT_FLOOR_MIN_WORDS and len(after_effective_floor) < _MIN_EFFECTIVE_CHUNKS:
        after_effective_floor, effective_floor_applied, effective_floor_split_count = _apply_effective_chunk_floor(
            text,
            after_effective_floor,
            target_chunks=_MIN_EFFECTIVE_CHUNKS,
        )

    after_fragmentation_guard, frag_guard_applied, frag_before_count, frag_after_count = _apply_fragmentation_guard(
        text,
        after_effective_floor,
        bucket=bucket,
        raw_count=len(raw_spans),
        word_count=int(word_count),
    )
    final_spans = _cleanup_spans(text, after_fragmentation_guard)
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
            "adaptive_effective_floor_split_count": int(effective_floor_split_count),
            "adaptive_fragmentation_before_chunks": int(frag_before_count),
            "adaptive_fragmentation_after_chunks": int(frag_after_count),
        },
        "adaptive_stage_chunk_counts": {
            "raw": int(len(raw_spans)),
            "after_invalid": int(len(after_invalid)),
            "after_short": int(len(after_short)),
            "after_long": int(len(after_long)),
            "after_effective_floor": int(len(after_effective_floor)),
            "after_fragmentation_guard": int(len(after_fragmentation_guard)),
            "final": int(len(final_spans)),
        },
        "adaptive_long_split_max_words": int(_LONG_SPLIT_MAX_WORDS_BY_BUCKET[bucket]),
        "adaptive_effective_floor_applied": bool(effective_floor_applied),
        "adaptive_effective_floor_target_chunks": int(_MIN_EFFECTIVE_CHUNKS),
        "adaptive_effective_floor_split_count": int(effective_floor_split_count),
        "adaptive_fragmentation_guard_applied": bool(frag_guard_applied),
        "adaptive_fragmentation_before_chunks": int(frag_before_count),
        "adaptive_fragmentation_after_chunks": int(frag_after_count),
    }
    stats.update(backend_stats)
    return chunks, stats


def adaptive_chunk(text: str, *, profile: str = "balanced") -> List[TextChunk]:
    chunks, _ = adaptive_chunk_with_stats(text=text, profile=profile)
    return chunks
