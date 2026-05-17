from lima_llm.chunking.factory import build_chunker
from lima_llm.chunking.diagnostics import (
    is_abbreviation_singleton_chunk_text,
    is_leading_close_punct_chunk_text,
)
from lima_llm.chunking.sentence import sentence_chunk
from lima_llm.chunking.sentence_v2 import sentence_chunk_v2, sentence_chunk_v2_with_stats
from lima_llm.chunking.utils import validate_chunk_coverage


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


def test_sentence_v2_chunk_coverage() -> None:
    text = "Sentence one. Sentence two! Sentence three?"
    chunker = build_chunker(method="sentence_v2", tokenizer=None, fixed_token_size=4)
    chunks = chunker(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 2
    assert chunker.last_diagnostics["chunk_strategy_requested"] == "sentence_v2"


def test_sentence_v2_avoids_orphan_punctuation_tail_chunk() -> None:
    text = (
        "she still has n't forgiven the guy as they board the terror train\n"
        '( " you asshole , you ca n\'t have a good time without hurting somebody ! " ) .\n'
    )
    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 1
    assert not any(chunk.text.strip() == '" ) .' for chunk in chunks)
    assert not any(chunk.text.strip() in {'" ) .', '" ) .'} for chunk in chunks)
    assert int(stats.get("orphan_merge_count", 0)) >= 1
    assert int(stats.get("orphan_chunks_before_merge", 0)) >= int(stats.get("orphan_chunks_after_merge", 0))


def test_sentence_v2_safe_keeps_sentence_boundaries_without_right_shift() -> None:
    text = (
        "in the end , this film reminds us that original approach can not prevent missed opportunities .\n"
        "( special note to readers : this line should start as the next chunk . )"
    )
    baseline_chunks = sentence_chunk(text)
    safe_chunks = sentence_chunk_v2(text)
    baseline_bounds = [chunk.end_char for chunk in baseline_chunks]
    safe_bounds = [chunk.end_char for chunk in safe_chunks]
    assert baseline_bounds[0] == safe_bounds[0]
    assert set(safe_bounds).issubset(set(baseline_bounds))
    assert safe_chunks[0].text.endswith("\n")
    assert safe_chunks[1].text.startswith("(")


def test_sentence_v2_fallback_reason_is_recorded() -> None:
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


def test_sentence_v2_fixes_leading_close_punctuation_chunk() -> None:
    text = "before .\n) after words continue here.\n"
    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert int(stats.get("leading_close_punct_fix_count", 0)) >= 1
    assert not any(is_leading_close_punct_chunk_text(chunk.text) for chunk in chunks)


def test_sentence_v2_merges_abbreviation_singleton_chunk() -> None:
    text = "bond .\nmr .\ntaylor writes well.\n"
    chunks, stats = sentence_chunk_v2_with_stats(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert int(stats.get("abbreviation_merge_count", 0)) >= 1
    assert not any(is_abbreviation_singleton_chunk_text(chunk.text) for chunk in chunks)
