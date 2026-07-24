from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Mapping, Sequence

from mobius.core.schema import TextChunk

_ORPHAN_PUNCT_RE = re.compile(r'^[\s"\'`“”‘’\(\)\[\]\{\}\.,!?;:，。！？；：、…\-–—]+$')
_CROSS_NEWLINE_RE = re.compile(r"\w[^\n]*\n[^\n]*\w", flags=re.UNICODE)
_LEADING_CLOSE_PUNCT_RE = re.compile(r"^\s*[\)\]\}]+")
_ABBREVIATION_SINGLETON_RE = re.compile(r"^(mr|mrs|ms|dr|prof|st|jr|sr)\s*\.\s*$", flags=re.IGNORECASE)


def _percentile(values: Sequence[float], q: float) -> float:
    """Handle the percentile step of adaptive text chunking."""

    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def is_orphan_punctuation_chunk_text(text: str) -> bool:
    """Handle the is orphan punctuation chunk text step of adaptive text chunking."""

    if text.strip() == "":
        return False
    return _ORPHAN_PUNCT_RE.fullmatch(text) is not None


def _cross_newline_boundary_chunk_text(text: str) -> bool:
    """Handle the cross newline boundary chunk text step of adaptive text chunking."""

    if "\n" not in text:
        return False
    return _CROSS_NEWLINE_RE.search(text) is not None


def is_leading_close_punct_chunk_text(text: str) -> bool:
    """Handle the is leading close punct chunk text step of adaptive text chunking."""

    stripped = text.lstrip()
    if stripped == "":
        return False
    match = _LEADING_CLOSE_PUNCT_RE.match(stripped)
    if match is None:
        return False
    rest = stripped[match.end() :]
    return rest.strip() != ""


def is_abbreviation_singleton_chunk_text(text: str) -> bool:
    """Handle the is abbreviation singleton chunk text step of adaptive text chunking."""

    normalized = text.replace("\n", " ").strip()
    if normalized == "":
        return False
    return _ABBREVIATION_SINGLETON_RE.fullmatch(normalized) is not None


def build_chunk_diagnostics(
    *,
    chunks: Sequence[TextChunk],
    requested_strategy: str,
    effective_strategy: str,
    fallback_applied: bool,
    fallback_reason: str | None,
    pre_fallback_chunk_count: int | None = None,
    extra_stats: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Handle the build chunk diagnostics step of adaptive text chunking."""

    chunk_lengths = [max(0, int(chunk.end_char) - int(chunk.start_char)) for chunk in chunks]
    orphan_count = sum(1 for chunk in chunks if is_orphan_punctuation_chunk_text(chunk.text))
    leading_close_count = sum(1 for chunk in chunks if is_leading_close_punct_chunk_text(chunk.text))
    abbreviation_singleton_count = sum(1 for chunk in chunks if is_abbreviation_singleton_chunk_text(chunk.text))
    cross_newline_count = sum(1 for chunk in chunks if _cross_newline_boundary_chunk_text(chunk.text))

    result = {
        "chunk_strategy_requested": str(requested_strategy),
        "chunk_strategy": str(effective_strategy),
        "chunk_count": int(len(chunks)),
        "chunk_len_chars_min": float(min(chunk_lengths)) if chunk_lengths else 0.0,
        "chunk_len_chars_mean": (float(sum(chunk_lengths)) / float(len(chunk_lengths))) if chunk_lengths else 0.0,
        "chunk_len_chars_p90": _percentile(chunk_lengths, 0.90),
        "chunk_len_chars_max": float(max(chunk_lengths)) if chunk_lengths else 0.0,
        "singleton_orphan_punctuation_chunks": int(orphan_count),
        "leading_close_punct_chunks": int(leading_close_count),
        "abbreviation_singleton_chunks": int(abbreviation_singleton_count),
        "cross_newline_boundary_chunks": int(cross_newline_count),
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason,
        "pre_fallback_chunk_count": (
            int(pre_fallback_chunk_count) if pre_fallback_chunk_count is not None else None
        ),
    }
    if extra_stats:
        result.update(dict(extra_stats))
    return result


def aggregate_chunk_diagnostics(diags: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Handle the aggregate chunk diagnostics step of adaptive text chunking."""

    if not diags:
        return {
            "samples_with_chunk_diagnostics": 0,
            "chunk_strategy_counts": {},
            "fallback_rate": 0.0,
            "orphan_chunks_mean": 0.0,
            "orphan_samples_ratio": 0.0,
            "orphan_merge_count_mean": 0.0,
            "orphan_merge_samples_ratio": 0.0,
            "leading_close_punct_mean": 0.0,
            "leading_close_punct_samples_ratio": 0.0,
            "abbreviation_singleton_mean": 0.0,
            "abbreviation_singleton_samples_ratio": 0.0,
            "cross_newline_chunks_mean": 0.0,
            "cross_newline_samples_ratio": 0.0,
        }

    strategy_counter = Counter(str(item.get("chunk_strategy", "")) for item in diags)
    fallback_count = sum(1 for item in diags if bool(item.get("fallback_applied", False)))
    orphan_values = [float(item.get("singleton_orphan_punctuation_chunks", 0.0)) for item in diags]
    orphan_merge_values = [float(item.get("orphan_merge_count", 0.0)) for item in diags]
    leading_close_values = [float(item.get("leading_close_punct_chunks", 0.0)) for item in diags]
    abbreviation_values = [float(item.get("abbreviation_singleton_chunks", 0.0)) for item in diags]
    cross_values = [float(item.get("cross_newline_boundary_chunks", 0.0)) for item in diags]
    orphan_samples = sum(1 for value in orphan_values if value > 0.0)
    orphan_merge_samples = sum(1 for value in orphan_merge_values if value > 0.0)
    leading_close_samples = sum(1 for value in leading_close_values if value > 0.0)
    abbreviation_samples = sum(1 for value in abbreviation_values if value > 0.0)
    cross_samples = sum(1 for value in cross_values if value > 0.0)

    denom = float(len(diags))
    return {
        "samples_with_chunk_diagnostics": int(len(diags)),
        "chunk_strategy_counts": dict(strategy_counter),
        "fallback_rate": float(fallback_count / denom),
        "orphan_chunks_mean": float(sum(orphan_values) / denom),
        "orphan_samples_ratio": float(orphan_samples / denom),
        "orphan_merge_count_mean": float(sum(orphan_merge_values) / denom),
        "orphan_merge_samples_ratio": float(orphan_merge_samples / denom),
        "leading_close_punct_mean": float(sum(leading_close_values) / denom),
        "leading_close_punct_samples_ratio": float(leading_close_samples / denom),
        "abbreviation_singleton_mean": float(sum(abbreviation_values) / denom),
        "abbreviation_singleton_samples_ratio": float(abbreviation_samples / denom),
        "cross_newline_chunks_mean": float(sum(cross_values) / denom),
        "cross_newline_samples_ratio": float(cross_samples / denom),
    }
