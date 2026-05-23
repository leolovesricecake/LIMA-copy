from __future__ import annotations

import types

import pytest

from lima_llm.chunking import adaptive as adaptive_mod
from lima_llm.chunking.diagnostics import is_orphan_punctuation_chunk_text
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


def _make_word_text(word_count: int) -> str:
    return " ".join(f"w{i}" for i in range(word_count)) + "."


def test_adaptive_bucket_routing_balanced_profile() -> None:
    for word_count, expected_bucket in [
        (80, "short"),
        (200, "medium"),
        (500, "long"),
        (1200, "very_long"),
    ]:
        text = _make_word_text(word_count)
        _, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
        assert stats["adaptive_bucket"] == expected_bucket
        assert stats["adaptive_features"]["word_count"] == word_count


def test_adaptive_invalid_orphan_chunk_is_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Alpha beta.\n.\nGamma delta epsilon zeta eta theta iota kappa lambda mu."
    dot_start = text.index("\n.\n") + 1
    dot_end = dot_start + 2

    def _fake_initial(_text: str, bucket: str):
        return (
            [(0, dot_start), (dot_start, dot_end), (dot_end, len(_text))],
            {
                "adaptive_sentence_backend": "none",
                "adaptive_sentence_backend_fallback_used": False,
                "adaptive_sentence_backend_calls": 0,
            },
        )

    monkeypatch.setattr(adaptive_mod, "_initial_spans_for_bucket", _fake_initial)
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert int(stats["adaptive_postprocess"]["invalid_merge_count"]) >= 1
    assert not any(is_orphan_punctuation_chunk_text(chunk.text) for chunk in chunks)


def test_adaptive_long_chunk_is_split_with_word_window() -> None:
    text = _make_word_text(220)
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert int(stats["adaptive_postprocess"]["long_split_count"]) >= 1
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) > 1
    assert all(len(chunk.text.split()) <= 80 for chunk in chunks)


def test_adaptive_sentence_backend_falls_back_to_regex_when_pysbd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adaptive_mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'pysbd'")),
    )
    text = _make_word_text(120)
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "medium"
    assert str(stats["adaptive_sentence_backend"]) == "regex_sentence"
    assert bool(stats["adaptive_sentence_backend_fallback_used"]) is True


def test_adaptive_short_bucket_effective_floor_applies() -> None:
    text = " ".join(f"tok{i}" for i in range(24)) + "."
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "short"
    assert int(stats["adaptive_features"]["word_count"]) >= 15
    assert bool(stats["adaptive_effective_floor_applied"]) is True
    assert int(stats["adaptive_stage_chunk_counts"]["after_effective_floor"]) >= 5
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) >= 5
    assert int(stats["adaptive_effective_floor_split_count"]) >= 1


def test_adaptive_balanced_v2_short_floor_requires_structure_signal() -> None:
    text = " ".join(f"tok{i}" for i in range(24))
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced_v2")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "short"
    assert bool(stats["adaptive_effective_floor_condition_met"]) is False
    assert bool(stats["adaptive_effective_floor_applied"]) is False
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) < 5


def test_adaptive_balanced_v2_short_floor_applies_with_structure_signal() -> None:
    text = " ".join(f"tok{i}" for i in range(24)) + "."
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced_v2")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "short"
    assert bool(stats["adaptive_effective_floor_condition_met"]) is True
    assert bool(stats["adaptive_effective_floor_applied"]) is True
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) >= 5


def test_adaptive_very_long_fragmentation_guard_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _make_word_text(1300)

    def _fake_split_long_spans(_text: str, spans, *, max_words: int):
        start, end = spans[0]
        width = max(1, (end - start) // 96)
        out = []
        cur = start
        for _ in range(95):
            nxt = min(end, cur + width)
            out.append((cur, nxt))
            cur = nxt
        out.append((cur, end))
        return out, 1

    monkeypatch.setattr(adaptive_mod, "_split_long_spans", _fake_split_long_spans)
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "very_long"
    assert bool(stats["adaptive_fragmentation_guard_applied"]) is True
    assert str(stats["adaptive_guard_mode"]) == "hard_cap"
    assert int(stats["adaptive_fragmentation_before_chunks"]) > int(stats["adaptive_fragmentation_after_chunks"])
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) <= 48


def test_adaptive_balanced_v2_very_long_guard_uses_soft_band(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _make_word_text(1300)

    def _fake_split_long_spans(_text: str, spans, *, max_words: int):
        start, end = spans[0]
        width = max(1, (end - start) // 96)
        out = []
        cur = start
        for _ in range(95):
            nxt = min(end, cur + width)
            out.append((cur, nxt))
            cur = nxt
        out.append((cur, end))
        return out, 1

    monkeypatch.setattr(adaptive_mod, "_split_long_spans", _fake_split_long_spans)
    chunks, stats = adaptive_mod.adaptive_chunk_with_stats(text, profile="balanced_v2")
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert stats["adaptive_bucket"] == "very_long"
    assert bool(stats["adaptive_fragmentation_guard_applied"]) is True
    assert str(stats["adaptive_guard_mode"]) == "soft_band"
    assert int(stats["adaptive_fragmentation_before_chunks"]) > int(stats["adaptive_fragmentation_after_chunks"])
    assert int(stats["adaptive_fragmentation_target_min"]) <= int(stats["adaptive_stage_chunk_counts"]["final"])
    assert int(stats["adaptive_stage_chunk_counts"]["final"]) <= int(stats["adaptive_fragmentation_target_max"])
    assert int(stats["adaptive_fragmentation_merge_ops"]) > 0


def test_adaptive_chunker_no_single_chunk_fixed_token_fallback() -> None:
    text = "short text"
    chunker = build_chunker(method="adaptive", tokenizer=None, fixed_token_size=1)
    chunks = chunker(text)
    diag = chunker.last_diagnostics
    assert len(chunks) == 1
    assert bool(diag.get("fallback_applied", False)) is False
    assert str(diag.get("chunk_strategy")) == "adaptive"
