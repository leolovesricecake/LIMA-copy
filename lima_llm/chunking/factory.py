from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from ..types import TextChunk
from .adaptive import adaptive_chunk_with_stats
from .diagnostics import build_chunk_diagnostics
from .fixed_token import fixed_token_chunk
from .sentence import sentence_chunk
from .sentence_v2 import sentence_chunk_v2_with_stats
from .utils import validate_chunk_coverage


Chunker = Callable[[str], List[TextChunk]]


class _ChunkerWithDiagnostics:
    def __init__(self, run_impl):
        self._run_impl = run_impl
        self.last_diagnostics: Dict[str, object] = {}

    def __call__(self, text: str) -> List[TextChunk]:
        chunks, diag = self._run_impl(text)
        self.last_diagnostics = dict(diag)
        return chunks


def _fixed_token_run_impl(text: str, tokenizer, fixed_token_size: int) -> Tuple[List[TextChunk], Dict[str, object]]:
    chunks = fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)
    return chunks, build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="fixed_token",
        effective_strategy="fixed_token",
        fallback_applied=False,
        fallback_reason=None,
        pre_fallback_chunk_count=None,
    )


def _sentence_like_run_impl(
    *,
    text: str,
    requested_strategy: str,
    sentence_impl,
    tokenizer,
    fixed_token_size: int,
) -> Tuple[List[TextChunk], Dict[str, object]]:
    chunks = sentence_impl(text)
    ok, _ = validate_chunk_coverage(text, chunks)
    fallback_applied = False
    fallback_reason = None
    effective_strategy = requested_strategy
    pre_fallback_chunk_count = len(chunks)

    if not ok:
        fallback_applied = True
        fallback_reason = "invalid_coverage_fallback"
    elif len(chunks) <= 1:
        fallback_applied = True
        fallback_reason = "single_chunk_fallback"

    if fallback_applied:
        chunks = fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)
        effective_strategy = "fixed_token"

    diag = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy=requested_strategy,
        effective_strategy=effective_strategy,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        pre_fallback_chunk_count=pre_fallback_chunk_count,
    )
    return chunks, diag


def _sentence_v2_run_impl(
    *,
    text: str,
    tokenizer,
    fixed_token_size: int,
) -> Tuple[List[TextChunk], Dict[str, object]]:
    chunks, v2_stats = sentence_chunk_v2_with_stats(text)
    ok, _ = validate_chunk_coverage(text, chunks)
    fallback_applied = False
    fallback_reason = None
    effective_strategy = "sentence_v2"
    pre_fallback_chunk_count = len(chunks)

    if not ok:
        fallback_applied = True
        fallback_reason = "invalid_coverage_fallback"
    elif len(chunks) <= 1:
        fallback_applied = True
        fallback_reason = "single_chunk_fallback"

    if fallback_applied:
        chunks = fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)
        effective_strategy = "fixed_token"

    diag = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="sentence_v2",
        effective_strategy=effective_strategy,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        pre_fallback_chunk_count=pre_fallback_chunk_count,
        extra_stats=v2_stats,
    )
    return chunks, diag


def _adaptive_run_impl(
    *,
    text: str,
    tokenizer,
    fixed_token_size: int,
    adaptive_profile: str,
) -> Tuple[List[TextChunk], Dict[str, object]]:
    chunks, adaptive_stats = adaptive_chunk_with_stats(text=text, profile=adaptive_profile)
    ok, _ = validate_chunk_coverage(text, chunks)
    fallback_applied = False
    fallback_reason = None
    effective_strategy = "adaptive"
    pre_fallback_chunk_count = len(chunks)

    if not ok:
        fallback_applied = True
        fallback_reason = "invalid_coverage_fallback"
    elif len(chunks) <= 1:
        fallback_applied = True
        fallback_reason = "single_chunk_fallback"

    if fallback_applied:
        chunks = fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)
        effective_strategy = "fixed_token"

    diag = build_chunk_diagnostics(
        chunks=chunks,
        requested_strategy="adaptive",
        effective_strategy=effective_strategy,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        pre_fallback_chunk_count=pre_fallback_chunk_count,
        extra_stats=adaptive_stats,
    )
    return chunks, diag


def build_chunker(
    method: str,
    tokenizer=None,
    fixed_token_size: int = 64,
    adaptive_profile: str = "balanced",
) -> Chunker:
    method = method.lower().strip()

    if method == "fixed_token":
        return _ChunkerWithDiagnostics(
            lambda text: _fixed_token_run_impl(text, tokenizer=tokenizer, fixed_token_size=fixed_token_size)
        )

    if method == "sentence":
        return _ChunkerWithDiagnostics(
            lambda text: _sentence_like_run_impl(
                text=text,
                requested_strategy="sentence",
                sentence_impl=sentence_chunk,
                tokenizer=tokenizer,
                fixed_token_size=fixed_token_size,
            )
        )

    if method == "sentence_v2":
        return _ChunkerWithDiagnostics(
            lambda text: _sentence_v2_run_impl(
                text=text,
                tokenizer=tokenizer,
                fixed_token_size=fixed_token_size,
            )
        )

    if method == "adaptive":
        return _ChunkerWithDiagnostics(
            lambda text: _adaptive_run_impl(
                text=text,
                tokenizer=tokenizer,
                fixed_token_size=fixed_token_size,
                adaptive_profile=adaptive_profile,
            )
        )

    raise ValueError(f"Unsupported chunker: {method}")
