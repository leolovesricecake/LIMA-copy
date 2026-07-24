"""Evaluation perturbations over independently chosen eval units."""

from __future__ import annotations

from typing import Sequence

from mobius.core.schema import TextChunk

from .chunks import compose_text
from .coalitions import EMPTY_TEXT


def keep_units(units: Sequence[TextChunk], unit_ids: Sequence[int]) -> str:
    """Keep only selected evaluation units."""

    text = compose_text(units, unit_ids)
    return text if text else EMPTY_TEXT


def delete_units(units: Sequence[TextChunk], unit_ids: Sequence[int]) -> str:
    """Delete selected evaluation units and keep their complement."""

    removed = {int(value) for value in unit_ids}
    text = compose_text(
        units,
        [int(unit.chunk_id) for unit in units if int(unit.chunk_id) not in removed],
    )
    return text if text else EMPTY_TEXT

