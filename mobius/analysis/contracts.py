"""Canonical run identities and comparison contracts for offline analyses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from mobius.core.results import canonical_digest, dataset_slug, model_slug
from mobius.values.classification import (
    effective_target_mode,
    normalize_value_function,
)


def load_json_object(path: str | Path) -> Dict[str, Any]:
    """Load one JSON object with a path-specific validation error."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return dict(payload)


def canonical_value_semantics(config: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize equivalent value-function and target-mode configurations."""

    value_function = normalize_value_function(
        str(config.get("value_function", "target_probability"))
    )
    target_mode = effective_target_mode(
        value_function,
        str(config.get("target_mode", "predicted")),
    )
    if value_function == "target_probability" and target_mode == "predicted":
        value_function = "predicted_probability"
    return {
        "value_function": value_function,
        "target_mode": target_mode,
    }


def dataset_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Select dataset fields that identify one comparable experiment cell."""

    dataset = dict(config.get("dataset", {}))
    return {
        "name": str(dataset.get("name", "dataset")),
        "split": str(dataset.get("split", "validation")),
        "source": dataset.get("source", dataset.get("sst2_source")),
        "verbalizers": (
            [str(value) for value in dataset.get("verbalizers", [])]
            if dataset.get("verbalizers") is not None
            else None
        ),
    }


def model_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Select model fields while excluding runtime-only device placement."""

    model = dict(config.get("model", {}))
    return {
        "type": str(model.get("type", "hf_causal_lm")),
        "model_path": model.get("model_path"),
        "dtype": str(model.get("dtype", "bfloat16")),
        "max_length": int(model.get("max_length", 2048)),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }


def comparison_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the scientific fields that must agree for paired analysis."""

    return {
        "dataset": dataset_contract(config),
        "model": model_contract(config),
        "prompt": dict(config.get("prompt", {})),
        "chunker": str(config.get("chunker", "word")),
        "eval_granularity": str(config.get("eval_granularity", "word")),
        **canonical_value_semantics(config),
    }


def dataset_cell_id(config: Mapping[str, Any]) -> str:
    """Build a readable dataset-and-split path component."""

    dataset = dict(config.get("dataset", {}))
    name = dataset_slug(dataset)
    split = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(dataset.get("split", "validation")).strip(),
    ).strip("_")
    return f"{name}-{split or 'split'}"


def model_cell_id(config: Mapping[str, Any]) -> str:
    """Build a readable model path component with a collision-resistant hash."""

    model = dict(config.get("model", {}))
    readable = model_slug(model)
    fingerprint = canonical_digest(model_contract(config))[:8]
    return f"{readable}-{fingerprint}"


def load_run_identity(run_dir: str | Path) -> Dict[str, Any]:
    """Load one immutable run identity and its canonical comparison contract."""

    root = Path(run_dir).resolve()
    payload = load_json_object(root / "run.json")
    config = dict(payload.get("scientific_config", {}))
    if not config:
        raise ValueError(f"Run has no scientific_config: {root}")
    return {
        "path": str(root),
        "root": root,
        "run_id": payload.get("run_id"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "config": config,
        "method": str(config.get("method", "method")),
        "seed": int(config.get("seed", 0)),
        "dataset_id": dataset_cell_id(config),
        "model_id": model_cell_id(config),
        "contract": comparison_contract(config),
        "payload": payload,
    }


def assert_same_cell(
    identities: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require every run to share dataset, model, target, and evaluation semantics."""

    values = list(identities)
    if not values:
        raise ValueError("At least one run is required.")
    expected = dict(values[0]["contract"])
    mismatches = [
        str(identity["path"])
        for identity in values[1:]
        if dict(identity["contract"]) != expected
    ]
    if mismatches:
        raise ValueError(
            "Runs do not share one dataset/model/target/evaluation cell: "
            + ", ".join(mismatches)
        )
    return expected


def assert_same_seed(identities: Sequence[Mapping[str, Any]]) -> int:
    """Require paired attribution runs to use one attribution random seed."""

    values = list(identities)
    if not values:
        raise ValueError("At least one run is required.")
    expected = int(values[0]["seed"])
    mismatches = [
        str(identity["path"])
        for identity in values[1:]
        if int(identity["seed"]) != expected
    ]
    if mismatches:
        raise ValueError(
            "Runs do not share one attribution seed: "
            + ", ".join(mismatches)
        )
    return expected
