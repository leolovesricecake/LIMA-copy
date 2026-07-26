"""Strict configuration loading and scientific-config normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


TOP_LEVEL_FIELDS = {
    "results_dir",
    "run_suffix",
    "output_level",
    "method",
    "budget",
    "seed",
    "max_degree",
    "k",
    "value_function",
    "target_mode",
    "chunker",
    "adaptive_profile",
    "adaptive_overrides",
    "adaptive_overrides_json",
    "eval_granularity",
    "eval_q_values",
    "min_features",
    "max_features",
    "batch_size",
    "fail_fast",
    "deterministic",
    "dataset",
    "model",
    "sampler",
    "basis",
    "hierarchy",
    "projector",
    "estimator",
    "fit",
}

RUNTIME_ONLY_FIELDS = {
    "results_dir",
    "run_suffix",
    "device",
    "overwrite",
    "command",
    "started_at",
}

NESTED_FIELDS = {
    "dataset": {
        "name",
        "split",
        "max_samples",
        "dataset_cache_dir",
        "sst2_source",
        "eraser_root",
        "source",
        "samples",
        "verbalizers",
    },
    "model": {
        "type",
        "model_path",
        "device",
        "dtype",
        "max_length",
        "trust_remote_code",
    },
    "sampler": {
        "name",
        "global_fraction",
        "near_full_fraction",
        "fixed_cardinality_fraction",
        "near_full_deletions",
        "fixed_keep_fractions",
        "include_empty_full",
    },
    "estimator": {
        "name",
        "alphas",
        "l1_ratios",
        "cv_folds",
        "coefficient_tolerance",
        "ridge_alphas",
        "refit",
        "selection_alpha_scale",
        "ridge_alpha",
        "max_design_mb",
    },
    "fit": {
        "alphas",
        "l1_ratios",
        "cv_folds",
        "coefficient_tolerance",
        "ridge_alphas",
        "refit",
        "selection_alpha_scale",
        "ridge_alpha",
        "max_design_mb",
    },
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration as a mapping."""

    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return dict(payload)


def load_json_object(raw: str | None) -> Dict[str, Any] | None:
    """Decode adaptive overrides from inline JSON or a JSON file."""

    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    payload = json.loads(text) if text.startswith("{") else json.loads(
        Path(text).expanduser().read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("adaptive_overrides_json must decode to an object.")
    return dict(payload)


def _validate_nested_fields(config: Mapping[str, Any]) -> None:
    """Reject misspelled fields in known nested configuration blocks."""

    for block_name, allowed in NESTED_FIELDS.items():
        raw = config.get(block_name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"{block_name} must be a mapping.")
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown fields in {block_name}: {unknown}"
            )


def resolve_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate known fields and fill stable defaults."""

    unknown = sorted(set(config) - TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown top-level configuration fields: {unknown}")
    _validate_nested_fields(config)
    output = copy.deepcopy(dict(config))
    output.setdefault("method", "sparse_mobius")
    if output.get("run_suffix") is not None:
        output["run_suffix"] = str(output["run_suffix"]).strip()
        if not output["run_suffix"]:
            output.pop("run_suffix")
    output.setdefault("output_level", "standard")
    output.setdefault("budget", 512)
    output.setdefault("seed", 42)
    output.setdefault("max_degree", 2)
    output.setdefault("k", 8)
    output.setdefault("value_function", "target_probability")
    output.setdefault("target_mode", "predicted")
    output.setdefault("chunker", "word")
    output.setdefault("adaptive_profile", "balanced")
    output.setdefault("eval_granularity", "token")
    output.setdefault("eval_q_values", [1, 5, 10, 20, 50])
    output.setdefault("min_features", 1)
    output.setdefault("max_features", None)
    output.setdefault("batch_size", 16)
    output.setdefault("fail_fast", False)
    output.setdefault("deterministic", False)
    output.setdefault("basis", "deletion_mobius")
    output.setdefault("hierarchy", "none")
    output.setdefault("projector", "signed_equal_share")
    output.setdefault("sampler", {"name": "uniform_size"})
    output.setdefault(
        "estimator",
        {"name": "lasso_support", "refit": "ridge_cv"},
    )
    output.setdefault("dataset", {})
    output.setdefault("model", {})
    if output["output_level"] not in {"minimal", "standard", "debug"}:
        raise ValueError("output_level must be minimal, standard, or debug.")
    if output.get("adaptive_overrides") is None:
        output["adaptive_overrides"] = load_json_object(
            output.get("adaptive_overrides_json")
        )
    output.pop("adaptive_overrides_json", None)
    return output


def scientific_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove machine-local runtime settings before hashing a run."""

    def clean(value: Any) -> Any:
        """Recursively remove runtime-only mapping keys."""

        if isinstance(value, Mapping):
            return {
                str(key): clean(item)
                for key, item in sorted(value.items())
                if str(key) not in RUNTIME_ONLY_FIELDS
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(dict(config))
