from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..types import TextChunk

_WORD_UNIT_RE = re.compile(r"\s*\S+\s*")
_EVAL_GRANULARITIES = {"token", "word"}


def word_units_from_text(text: str) -> List[TextChunk]:
    units: List[TextChunk] = []
    for idx, match in enumerate(_WORD_UNIT_RE.finditer(text)):
        units.append(
            TextChunk(
                chunk_id=idx,
                start_char=match.start(),
                end_char=match.end(),
                text=text[match.start() : match.end()],
            )
        )
    if not units:
        return [TextChunk(chunk_id=0, start_char=0, end_char=len(text), text=text)]
    return units


def normalize_eval_granularity(eval_granularity: str | None) -> str:
    value = str(eval_granularity or "token").strip().lower()
    if value not in _EVAL_GRANULARITIES:
        raise ValueError(f"Unsupported eval granularity: {eval_granularity!r}. Expected one of {_EVAL_GRANULARITIES}.")
    return value


def repair_units_to_full_coverage(
    text: str,
    units: Sequence[TextChunk],
    *,
    keep_token_span: bool,
) -> List[TextChunk]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")]

    spans: List[Tuple[int, int, object, object]] = []
    text_len = len(text)
    for unit in units:
        start = max(0, min(int(unit.start_char), text_len))
        end = max(0, min(int(unit.end_char), text_len))
        if end < start:
            continue
        spans.append((start, end, unit.token_start, unit.token_end))

    if not spans:
        return [TextChunk(chunk_id=0, start_char=0, end_char=text_len, text=text)]

    spans.sort(key=lambda item: (item[0], item[1]))

    starts: List[int] = []
    prev = 0
    for idx, (start, _end, _tok_start, _tok_end) in enumerate(spans):
        cur = start
        if idx == 0:
            cur = 0
        elif cur < prev:
            cur = prev
        starts.append(cur)
        prev = cur

    repaired: List[TextChunk] = []
    for idx in range(len(spans)):
        start = starts[idx]
        end = starts[idx + 1] if idx < len(spans) - 1 else text_len
        if end < start:
            end = start
        tok_start = int(spans[idx][2]) if keep_token_span and spans[idx][2] is not None else None
        tok_end = int(spans[idx][3]) if keep_token_span and spans[idx][3] is not None else None
        repaired.append(
            TextChunk(
                chunk_id=idx,
                start_char=start,
                end_char=end,
                text=text[start:end],
                token_start=tok_start,
                token_end=tok_end,
            )
        )
    return repaired


def token_units_from_text(text: str, tokenizer) -> tuple[List[TextChunk], bool]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")], False

    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = encoded.get("offset_mapping", None)
            if offsets:
                token_units: List[TextChunk] = []
                for idx, offset in enumerate(offsets):
                    if offset is None or len(offset) < 2:
                        continue
                    start, end = int(offset[0]), int(offset[1])
                    if end < start:
                        continue
                    token_units.append(
                        TextChunk(
                            chunk_id=idx,
                            start_char=start,
                            end_char=end,
                            text=text[start:end],
                            token_start=idx,
                            token_end=idx + 1,
                        )
                    )
                if token_units:
                    return repair_units_to_full_coverage(text=text, units=token_units, keep_token_span=True), False
        except Exception:
            pass

    fallback_units = word_units_from_text(text)
    return repair_units_to_full_coverage(text=text, units=fallback_units, keep_token_span=False), True


def build_eval_units(text: str, eval_granularity: str, tokenizer) -> tuple[List[TextChunk], bool, str]:
    granularity = normalize_eval_granularity(eval_granularity)
    if granularity == "word":
        word_units = repair_units_to_full_coverage(text=text, units=word_units_from_text(text), keep_token_span=False)
        return word_units, False, "word_whitespace_spans"

    token_units, fallback = token_units_from_text(text=text, tokenizer=tokenizer)
    strategy = "tokenizer_offset_mapping" if not fallback else "whitespace_fallback_without_tokenizer_offsets"
    return token_units, fallback, strategy


def content_span(unit: TextChunk) -> tuple[int, int]:
    leading_len = len(unit.text) - len(unit.text.lstrip())
    trailing_len = len(unit.text) - len(unit.text.rstrip())
    start = unit.start_char + leading_len
    end = unit.end_char - trailing_len
    if end <= start:
        return unit.start_char, unit.end_char
    return start, end


def project_chunk_ranking_to_unit_ranking(
    eval_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    chunk_rank = {int(chunk_id): idx for idx, chunk_id in enumerate(chunk_ranking)}
    fallback_rank = len(chunk_rank) + len(chunks) + 1

    projected = []
    for unit in eval_units:
        unit_start, unit_end = content_span(unit)
        weighted_rank = 0.0
        overlap_total = 0

        for chunk in chunks:
            overlap = max(0, min(unit_end, chunk.end_char) - max(unit_start, chunk.start_char))
            if overlap <= 0:
                continue
            weighted_rank += float(overlap) * float(chunk_rank.get(chunk.chunk_id, fallback_rank))
            overlap_total += int(overlap)

        if overlap_total <= 0:
            projected_rank = float(fallback_rank + unit.chunk_id)
        else:
            projected_rank = weighted_rank / float(overlap_total)
        projected.append((unit.chunk_id, projected_rank))

    return [unit_id for unit_id, _ in sorted(projected, key=lambda item: (item[1], item[0]))]


def count_units_split_across_chunks(eval_units: Sequence[TextChunk], chunks: Sequence[TextChunk]) -> int:
    split_count = 0
    for unit in eval_units:
        unit_start, unit_end = content_span(unit)
        overlaps = 0
        for chunk in chunks:
            if min(unit_end, chunk.end_char) > max(unit_start, chunk.start_char):
                overlaps += 1
                if overlaps > 1:
                    split_count += 1
                    break
    return split_count


def project_chunk_ranking_to_word_ranking(
    word_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    return project_chunk_ranking_to_unit_ranking(
        eval_units=word_units,
        chunks=chunks,
        chunk_ranking=chunk_ranking,
    )


def count_words_split_across_chunks(word_units: Sequence[TextChunk], chunks: Sequence[TextChunk]) -> int:
    return count_units_split_across_chunks(word_units, chunks)


_word_units_from_text = word_units_from_text
_normalize_eval_granularity = normalize_eval_granularity
_repair_units_to_full_coverage = repair_units_to_full_coverage
_token_units_from_text = token_units_from_text
_build_eval_units = build_eval_units
_content_span = content_span
_project_chunk_ranking_to_unit_ranking = project_chunk_ranking_to_unit_ranking
_count_units_split_across_chunks = count_units_split_across_chunks
_project_chunk_ranking_to_word_ranking = project_chunk_ranking_to_word_ranking
_count_words_split_across_chunks = count_words_split_across_chunks

__all__ = [
    "build_eval_units",
    "content_span",
    "count_units_split_across_chunks",
    "count_words_split_across_chunks",
    "normalize_eval_granularity",
    "project_chunk_ranking_to_unit_ranking",
    "project_chunk_ranking_to_word_ranking",
    "repair_units_to_full_coverage",
    "token_units_from_text",
    "word_units_from_text",
]
