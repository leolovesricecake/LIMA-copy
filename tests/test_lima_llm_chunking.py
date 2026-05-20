from __future__ import annotations

import types

import pytest

from lima_llm.chunking.factory import build_chunker
from lima_llm.chunking.sentence_v2 import sentence_chunk_v2_with_stats
from lima_llm.chunking.utils import validate_chunk_coverage


class _FakeTextSpan:
    def __init__(self, sent: str, start: int, end: int) -> None:
        self.sent = sent
        self.start = start
        self.end = end


def _patch_fake_pysbd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    char_span_rows=None,
    segment_rows=None,
    supports_char_span: bool = True,
) -> None:
    if char_span_rows is None:
        char_span_rows = []
    if segment_rows is None:
        segment_rows = []

    class _FakeSegmenter:
        def __init__(self, *, language: str, clean: bool, char_span: bool = False) -> None:
            assert language == "en"
            assert clean is False
            if char_span and not supports_char_span:
                raise TypeError("char_span is unsupported")
            self._char_span = bool(char_span)

        def segment(self, _text: str):
            return list(char_span_rows) if self._char_span else list(segment_rows)

    fake_module = types.SimpleNamespace(Segmenter=_FakeSegmenter)
    monkeypatch.setattr("lima_llm.chunking.sentence_v2._load_pysbd_module", lambda: fake_module)


def test_sentence_chunk_coverage() -> None:
    text = "Sentence one. Sentence two! Sentence three?"
    chunker = build_chunker(method="sentence", tokenizer=None, fixed_token_size=4)
    chunks = chunker(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 2


def test_fixed_token_chunk_coverage_without_tokenizer() -> None:
    text = "a b c d e f g h i j k l"
    chunker = build_chunker(method="fixed_token", tokenizer=None, fixed_token_size=3)
    chunks = chunker(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 3


def test_sentence_v2_chunk_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Sentence one. Sentence two! Sentence three?"
    _patch_fake_pysbd(
        monkeypatch,
        supports_char_span=False,
        segment_rows=["Sentence one. ", "Sentence two! ", "Sentence three?"],
    )
    chunker = build_chunker(method="sentence_v2", tokenizer=None, fixed_token_size=4)
    chunks = chunker(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 2
    assert chunker.last_diagnostics["chunk_strategy_requested"] == "sentence_v2"


def test_sentence_v2_avoids_orphan_punctuation_tail_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_pysbd(
        monkeypatch,
        supports_char_span=False,
        segment_rows=[
            "she still has n't forgiven the guy as they board the terror train\n"
            '( " you asshole , you ca n\'t have a good time without hurting somebody ! " ) .\n'
        ],
    )
    text = (
        "she still has n't forgiven the guy as they board the terror train\n"
        '( " you asshole , you ca n\'t have a good time without hurting somebody ! " ) .\n'
    )
    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 1
    assert not any(chunk.text.strip() == '" ) .' for chunk in chunks)
    assert int(stats.get("sentence_backend_segment_count", 0)) == 1


def test_sentence_v2_prefers_char_span_path(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "First line.\nSecond line!"
    _patch_fake_pysbd(
        monkeypatch,
        char_span_rows=[
            _FakeTextSpan("First line.\n", 0, 12),
            _FakeTextSpan("Second line!", 12, len(text)),
        ],
        segment_rows=["unused"],
    )

    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert [c.text for c in chunks] == ["First line.\n", "Second line!"]
    assert stats["sentence_backend"] == "pysbd"
    assert stats["sentence_backend_char_span_used"] is True
    assert stats["sentence_backend_span_rebuild_used"] is False
    assert int(stats["sentence_backend_segment_count"]) == 2


def test_sentence_v2_rebuilds_spans_when_char_span_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        "in the end , this film reminds us that original approach can not prevent missed opportunities.\n"
        "( special note to readers : this line should start as the next chunk. )"
    )
    _patch_fake_pysbd(
        monkeypatch,
        char_span_rows=[
            _FakeTextSpan("bad-1", 0, 5),
            _FakeTextSpan("bad-2", 0, 5),
        ],
        segment_rows=[
            "in the end , this film reminds us that original approach can not prevent missed opportunities.\n",
            "( special note to readers : this line should start as the next chunk. )",
        ],
    )
    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) == 2
    assert chunks[0].text.endswith("\n")
    assert chunks[1].text.startswith("(")
    assert stats["sentence_backend_char_span_used"] is False
    assert stats["sentence_backend_span_rebuild_used"] is True
    assert int(stats["sentence_backend_segment_count"]) == 2


def test_sentence_v2_fallback_reason_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_pysbd(monkeypatch, supports_char_span=False, segment_rows=["Only one sentence."])
    text = "Only one sentence."
    chunker = build_chunker(method="sentence_v2", tokenizer=None, fixed_token_size=1)
    _ = chunker(text)
    diag = chunker.last_diagnostics
    assert bool(diag.get("fallback_applied", False)) is True
    assert str(diag.get("fallback_reason")) == "single_chunk_fallback"
    assert int(diag.get("chunk_count", 0)) >= 2
    assert "orphan_merge_count" in diag
    assert "leading_close_punct_fix_count" in diag
    assert "abbreviation_merge_count" in diag


def test_sentence_v2_fail_fast_when_pysbd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lima_llm.chunking.sentence_v2.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'pysbd'")),
    )
    with pytest.raises(RuntimeError, match="pip install pysbd==0.3.4"):
        _ = sentence_chunk_v2_with_stats("A. B.")
