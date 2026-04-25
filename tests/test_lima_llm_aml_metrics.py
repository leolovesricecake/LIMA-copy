import math

import numpy as np

from lima_llm.eval.metrics import (
    AML_AOPC_Q_VALUES,
    AML_PRIMARY_Q_PERCENT,
    aml_faithfulness_metrics,
    top_percent_chunk_count,
)
from lima_llm.eval.evaluate import (
    _count_words_split_across_chunks,
    _project_chunk_ranking_to_word_ranking,
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
