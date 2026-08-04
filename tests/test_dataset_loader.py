"""Regression tests for dataset identity and configured text schemas."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mobius.data.loader import (
    DATASETS,
    _bundle_from_rows,
    _cached_arrow_files,
    _extract_text,
)


def _make_arrow(path: Path, *, mtime_ns: int) -> Path:
    """Create a cache-shaped placeholder with a deterministic modification time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_sst2_cache_lookup_never_selects_newer_mrpc_config(tmp_path: Path) -> None:
    """Keep cache discovery inside the requested GLUE configuration directory."""

    dataset_root = tmp_path / "datasets" / "nyu-mll___glue"
    sst2_file = _make_arrow(
        dataset_root / "sst2" / "0.0.0" / "sst2-hash" / "glue-validation.arrow",
        mtime_ns=1,
    )
    _make_arrow(
        dataset_root / "mrpc" / "0.0.0" / "mrpc-hash" / "glue-validation.arrow",
        mtime_ns=2,
    )

    files = _cached_arrow_files(
        str(tmp_path),
        DATASETS["sst2"]["cache_names"],
        "validation",
        cache_configs=DATASETS["sst2"]["cache_configs"],
    )

    assert files == [sst2_file.resolve()]


def test_missing_sst2_cache_does_not_fall_back_to_mrpc(tmp_path: Path) -> None:
    """Return no offline match when only another GLUE task is cached."""

    _make_arrow(
        tmp_path
        / "datasets"
        / "nyu-mll___glue"
        / "mrpc"
        / "0.0.0"
        / "mrpc-hash"
        / "glue-validation.arrow",
        mtime_ns=1,
    )

    files = _cached_arrow_files(
        str(tmp_path),
        DATASETS["sst2"]["cache_names"],
        "validation",
        cache_configs=DATASETS["sst2"]["cache_configs"],
    )

    assert files == []


def test_text_extraction_uses_only_the_dataset_declared_field() -> None:
    """Prevent paired-task columns from being guessed as SST-2 input text."""

    mrpc_row = {"sentence1": "first", "sentence2": "second", "label": 1}
    assert _extract_text(mrpc_row, DATASETS["sst2"]["text_field"]) == ""
    assert _extract_text(
        {"sentence": "an SST-2 sentence"},
        DATASETS["sst2"]["text_field"],
    ) == "an SST-2 sentence"

    with pytest.raises(ValueError, match="No valid rows"):
        _bundle_from_rows(
            [mrpc_row],
            dataset_name="sst2",
            split="validation",
            labels=DATASETS["sst2"]["labels"],
            text_field=DATASETS["sst2"]["text_field"],
            max_samples=None,
            source="test",
        )
