from lima_llm.chunking.factory import build_chunker
from lima_llm.chunking.sentence_v2 import sentence_chunk_v2
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
    chunks = sentence_chunk_v2(text)
    ok, msg = validate_chunk_coverage(text, chunks)
    assert ok, msg
    assert len(chunks) >= 1
    assert not any(chunk.text.strip() == '" ) .' for chunk in chunks)
    assert not any(chunk.text.strip() in {'" ) .', '" ) .'} for chunk in chunks)


def test_sentence_v2_fallback_reason_is_recorded() -> None:
    text = "Only one sentence."
    chunker = build_chunker(method="sentence_v2", tokenizer=None, fixed_token_size=1)
    _ = chunker(text)
    diag = chunker.last_diagnostics
    assert bool(diag.get("fallback_applied", False)) is True
    assert str(diag.get("fallback_reason")) == "single_chunk_fallback"
    assert int(diag.get("chunk_count", 0)) >= 2
