"""Compatibility-free schema-v2 resume helpers."""

from __future__ import annotations

from pathlib import Path

from .results import ResultStore


def completed_sample_ids(store: ResultStore) -> set[str]:
    """Return validated completed IDs from the current result store."""

    return {
        path.stem
        for path in Path(store.samples_dir).glob("*.json")
        if store.sample_complete(path.stem)
    }

