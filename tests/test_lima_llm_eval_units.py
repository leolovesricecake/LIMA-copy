from __future__ import annotations

from lima_llm.eval.units import (
    build_eval_units,
    count_units_split_across_chunks,
    normalize_eval_granularity,
    project_chunk_ranking_to_unit_ranking,
    token_units_from_text,
    word_units_from_text,
)
from lima_llm.types import TextChunk


class _TokenizerWithOffsets:
    def __call__(self, text, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        if text == "abcde":
            return {"offset_mapping": [(0, 2), (2, 5)]}
        if text == "A BC?":
            return {"offset_mapping": [(0, 1), (2, 3), (3, 5)]}
        raise AssertionError(text)


def test_word_units_from_text_preserves_whitespace_coverage() -> None:
    units = word_units_from_text("hi there")
    assert [unit.text for unit in units] == ["hi ", "there"]
    assert "".join(unit.text for unit in units) == "hi there"


def test_token_units_from_text_repairs_offset_gaps_to_full_coverage() -> None:
    units, fallback = token_units_from_text("A BC?", _TokenizerWithOffsets())
    assert fallback is False
    assert "".join(unit.text for unit in units) == "A BC?"
    assert units[0].start_char == 0
    assert units[-1].end_char == len("A BC?")


def test_build_eval_units_supports_token_word_and_empty_text() -> None:
    token_units, token_fallback, token_strategy = build_eval_units("abcde", "token", _TokenizerWithOffsets())
    assert token_fallback is False
    assert token_strategy == "tokenizer_offset_mapping"
    assert [(unit.start_char, unit.end_char) for unit in token_units] == [(0, 2), (2, 5)]

    word_units, word_fallback, word_strategy = build_eval_units("abcde", "word", _TokenizerWithOffsets())
    assert word_fallback is False
    assert word_strategy == "word_whitespace_spans"
    assert "".join(unit.text for unit in word_units) == "abcde"

    empty_units, empty_fallback, empty_strategy = build_eval_units("", "token", _TokenizerWithOffsets())
    assert empty_fallback is False
    assert empty_strategy == "tokenizer_offset_mapping"
    assert len(empty_units) == 1
    assert empty_units[0].start_char == 0
    assert empty_units[0].end_char == 0


def test_chunk_projection_from_chunks_to_token_units_uses_overlap_weighting() -> None:
    eval_units, fallback, strategy = build_eval_units("abcde", "token", _TokenizerWithOffsets())
    assert fallback is False
    assert strategy == "tokenizer_offset_mapping"

    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=1, text="a"),
        TextChunk(chunk_id=1, start_char=1, end_char=4, text="bcd"),
        TextChunk(chunk_id=2, start_char=4, end_char=5, text="e"),
    ]
    ranking = project_chunk_ranking_to_unit_ranking(
        eval_units=eval_units,
        chunks=chunks,
        chunk_ranking=[1, 2, 0],
    )
    assert ranking == [1, 0]
    assert count_units_split_across_chunks(eval_units, chunks) == 2


def test_normalize_eval_granularity_rejects_unknown_value() -> None:
    try:
        normalize_eval_granularity("sentence")
    except ValueError as exc:
        assert "Unsupported eval granularity" in str(exc)
    else:
        raise AssertionError("Expected invalid granularity to raise")
