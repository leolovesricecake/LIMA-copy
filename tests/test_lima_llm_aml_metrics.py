import math

import numpy as np

from lima_llm.eval.metrics import (
    AML_AOPC_Q_VALUES,
    AML_PRIMARY_Q_PERCENT,
    aml_faithfulness_metrics,
    top_percent_chunk_count,
)
from lima_llm.eval.evaluate import (
    _build_eval_units,
    _count_words_split_across_chunks,
    _count_units_split_across_chunks,
    _project_chunk_ranking_to_unit_ranking,
    _project_chunk_ranking_to_word_ranking,
    _token_units_from_text,
    _word_units_from_text,
)
from lima_llm.types import TextChunk


def test_aml_top_percent_chunk_count_uses_floor_and_allows_zero() -> None:
    assert top_percent_chunk_count(total_chunks=4, q_percent=20) == 0
    assert top_percent_chunk_count(total_chunks=5, q_percent=20) == 1


def test_aml_faithfulness_metrics_match_reference_settings() -> None:
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=1, text="a"),
        TextChunk(chunk_id=1, start_char=1, end_char=2, text="b"),
        TextChunk(chunk_id=2, start_char=2, end_char=3, text="c"),
        TextChunk(chunk_id=3, start_char=3, end_char=4, text="d"),
        TextChunk(chunk_id=4, start_char=4, end_char=5, text="e"),
    ]
    probs_by_text = {
        "abcde": np.asarray([0.1, 0.9], dtype=np.float32),
        "bcde": np.asarray([0.2, 0.8], dtype=np.float32),
        "a": np.asarray([0.6, 0.4], dtype=np.float32),
        "cde": np.asarray([0.4, 0.6], dtype=np.float32),
        "ab": np.asarray([0.5, 0.5], dtype=np.float32),
        "<UNK>bcde": np.asarray([0.3, 0.7], dtype=np.float32),
    }

    def prob_fn(text, verbalizers):
        return probs_by_text[text]

    metrics, per_q = aml_faithfulness_metrics(
        chunks=chunks,
        ranking=[0, 1, 2, 3, 4],
        target_label=1,
        verbalizers=["NEG", "POS"],
        prob_fn=prob_fn,
        primary_q_percent=AML_PRIMARY_Q_PERCENT,
        aopc_q_values=AML_AOPC_Q_VALUES,
        reference_token_text="<UNK>",
    )

    assert per_q[1]["top_count"] == 0
    assert per_q[1]["comp"] == 0.0
    assert per_q[1]["suff"] == 0.0
    assert math.isclose(metrics["comprehensiveness"], 0.1, rel_tol=1e-6)
    assert math.isclose(metrics["sufficiency"], 0.5, rel_tol=1e-6)
    assert math.isclose(metrics["log_odds"], math.log(0.7) - math.log(0.9), rel_tol=1e-6)
    assert math.isclose(metrics["aopc_comprehensiveness"], (0.1 + 0.3) / 6.0, rel_tol=1e-6)
    assert math.isclose(metrics["aopc_sufficiency"], (0.5 + 0.4) / 6.0, rel_tol=1e-6)


def test_chunk_ranking_projects_to_word_units_when_word_is_split() -> None:
    text = "unbelievable movie"
    word_units = _word_units_from_text(text)
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=2, text="un"),
        TextChunk(chunk_id=1, start_char=2, end_char=13, text="believable "),
        TextChunk(chunk_id=2, start_char=13, end_char=len(text), text="movie"),
    ]

    ranking = _project_chunk_ranking_to_word_ranking(
        word_units=word_units,
        chunks=chunks,
        chunk_ranking=[1, 2, 0],
    )

    assert [unit.text for unit in word_units] == ["unbelievable ", "movie"]
    assert _count_words_split_across_chunks(word_units, chunks) == 1
    assert ranking == [0, 1]


class _DummyTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        return {"offset_mapping": [(0, 1), (2, 3), (3, 5)]}


class _ProjectionTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "abcde"
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        return {"offset_mapping": [(0, 2), (2, 5)]}


def test_token_units_repair_contiguous_coverage() -> None:
    text = "A BC?"
    units, fallback = _token_units_from_text(text=text, tokenizer=_DummyTokenizer())
    assert fallback is False
    assert units[0].start_char == 0
    assert units[-1].end_char == len(text)
    for idx in range(1, len(units)):
        assert units[idx - 1].end_char == units[idx].start_char
    assert "".join(unit.text for unit in units) == text


def test_chunk_ranking_projects_to_token_units() -> None:
    text = "abcde"
    token_units, fallback, strategy = _build_eval_units(
        text=text,
        eval_granularity="token",
        tokenizer=_ProjectionTokenizer(),
    )
    assert fallback is False
    assert strategy == "tokenizer_offset_mapping"
    assert [(unit.start_char, unit.end_char) for unit in token_units] == [(0, 2), (2, 5)]

    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=1, text="a"),
        TextChunk(chunk_id=1, start_char=1, end_char=4, text="bcd"),
        TextChunk(chunk_id=2, start_char=4, end_char=5, text="e"),
    ]

    ranking = _project_chunk_ranking_to_unit_ranking(
        eval_units=token_units,
        chunks=chunks,
        chunk_ranking=[1, 2, 0],
    )

    assert _count_units_split_across_chunks(token_units, chunks) == 2
    assert ranking == [1, 0]


def test_token_granularity_falls_back_without_tokenizer() -> None:
    units, fallback, strategy = _build_eval_units(
        text="hello world",
        eval_granularity="token",
        tokenizer=None,
    )
    assert fallback is True
    assert strategy == "whitespace_fallback_without_tokenizer_offsets"
    assert "".join(unit.text for unit in units) == "hello world"
