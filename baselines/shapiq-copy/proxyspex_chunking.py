from __future__ import annotations

from typing import Mapping

from lima_llm.chunking.explanation import (
    ExplanationChunkingResult,
    build_explanation_chunks,
    load_adaptive_overrides,
    normalize_explanation_chunker,
)


ProxySPEXChunkingResult = ExplanationChunkingResult


def normalize_proxyspex_chunker(chunker: str | None) -> str:
    return normalize_explanation_chunker(chunker)


def build_proxyspex_chunks(
    *,
    text: str,
    chunker: str,
    tokenizer,
    adaptive_profile: str = "balanced",
    adaptive_overrides: Mapping[str, object] | None = None,
) -> ProxySPEXChunkingResult:
    return build_explanation_chunks(
        text=text,
        chunker=chunker,
        tokenizer=tokenizer,
        adaptive_profile=adaptive_profile,
        adaptive_overrides=adaptive_overrides,
    )


__all__ = [
    "ProxySPEXChunkingResult",
    "build_proxyspex_chunks",
    "load_adaptive_overrides",
    "normalize_proxyspex_chunker",
]
