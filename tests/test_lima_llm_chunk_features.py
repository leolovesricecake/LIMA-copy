from __future__ import annotations

from lima_llm.chunking.features import build_chunk_feature_payload
from lima_llm.types import TextChunk


def test_chunk_features_whitespace_fallback_payload() -> None:
    text = "Alpha.\n) beta"
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=7, text="Alpha.\n"),
        TextChunk(chunk_id=1, start_char=7, end_char=len(text), text=") beta"),
    ]
    features, coverage = build_chunk_feature_payload(text=text, chunks=chunks, tokenizer=None)

    assert coverage["token_alignment_mode"] == "whitespace_fallback"
    assert coverage["token_alignment_fallback_used"] is True
    assert coverage["total_chunks"] == 2
    assert coverage["aligned_chunks"] == 2
    assert "0" in features
    assert "1" in features
    assert features["0"]["char_len"] == 7
    assert features["0"]["word_count"] == 1
    assert features["1"]["leading_close_punct"] is True


def test_chunk_features_prefers_tokenizer_offsets() -> None:
    class _FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"offset_mapping": [(0, 1), (2, 4), (5, 6)]}

    text = "a bc d"
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=4, text="a bc"),
        TextChunk(chunk_id=1, start_char=4, end_char=len(text), text=" d"),
    ]
    features, coverage = build_chunk_feature_payload(text=text, chunks=chunks, tokenizer=_FakeTokenizer())

    assert coverage["token_alignment_mode"] == "tokenizer_offset_mapping"
    assert coverage["token_alignment_fallback_used"] is False
    assert coverage["total_tokens"] == 3
    assert features["0"]["token_start"] == 0
    assert features["0"]["token_end"] == 2
    assert features["0"]["token_count"] == 2
    assert features["1"]["token_start"] == 2
    assert features["1"]["token_end"] == 3
    assert features["1"]["token_count"] == 1
