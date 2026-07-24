"""Shared text chunking, coalition composition, and perturbation helpers."""

from .chunks import (
    ChunkingResult,
    build_chunks,
    build_eval_units,
    compose_text,
    project_ranking,
)
from .coalitions import CoalitionGame

__all__ = [
    "ChunkingResult",
    "CoalitionGame",
    "build_chunks",
    "build_eval_units",
    "compose_text",
    "project_ranking",
]

