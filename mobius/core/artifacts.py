"""Portable observation and surrogate sidecar artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .runtime import atomic_write_json, ensure_dir


ARTIFACT_SCHEMA_VERSION = "1.0"


def masks_to_bool_matrix(
    masks: Sequence[int] | np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Normalize integer or matrix keep masks to a two-dimensional bool matrix."""

    values = np.asarray(masks)
    if values.ndim == 2:
        if values.shape[1] != int(n_features):
            raise ValueError(
                f"keep_masks has width {values.shape[1]}, expected {n_features}."
            )
        return values.astype(bool, copy=True)
    if values.ndim != 1:
        raise ValueError("keep_masks must be one- or two-dimensional.")
    if len(values) == 0:
        return np.zeros((0, int(n_features)), dtype=bool)
    return np.asarray(
        [
            [
                bool(int(mask) & (1 << player))
                for player in range(int(n_features))
            ]
            for mask in values.tolist()
        ],
        dtype=bool,
    )


def bool_matrix_to_masks(matrix: Sequence[Sequence[bool]] | np.ndarray) -> list[int]:
    """Encode a two-dimensional bool keep-mask matrix as arbitrary-width integers."""

    values = np.asarray(matrix, dtype=bool)
    if values.ndim != 2:
        raise ValueError("keep_masks must be a two-dimensional matrix.")
    masks: list[int] = []
    for row in values:
        mask = 0
        for player in np.flatnonzero(row):
            mask |= 1 << int(player)
        masks.append(mask)
    return masks


def observation_digest(
    keep_masks: np.ndarray,
    label_scores: np.ndarray,
    attribution_values: np.ndarray,
) -> str:
    """Hash observation arrays including shape, dtype, and exact bytes."""

    digest = hashlib.sha256()
    for name, array in (
        ("keep_masks", np.asarray(keep_masks, dtype=bool)),
        ("label_scores", np.asarray(label_scores, dtype=np.float64)),
        ("attribution_values", np.asarray(attribution_values, dtype=np.float64)),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def normalize_observation_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one in-memory observation artifact."""

    n_features = int(payload["n_features"])
    keep_masks = masks_to_bool_matrix(payload["keep_masks"], n_features)
    label_scores = np.asarray(payload["label_scores"], dtype=np.float64)
    values = np.asarray(payload["attribution_values"], dtype=np.float64).reshape(-1)
    if label_scores.ndim != 2:
        raise ValueError("label_scores must be a two-dimensional matrix.")
    if len(keep_masks) != len(label_scores) or len(keep_masks) != len(values):
        raise ValueError("Observation masks, scores, and values must have equal rows.")
    if not np.all(np.isfinite(label_scores)) or not np.all(np.isfinite(values)):
        raise ValueError("Observation scores and values must be finite.")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "sample_id": str(payload["sample_id"]),
        "method": str(payload["method"]),
        "n_features": n_features,
        "mask_semantics": "keep",
        "keep_masks": keep_masks,
        "label_scores": label_scores,
        "attribution_values": values,
        "digest": observation_digest(keep_masks, label_scores, values),
    }


def write_observation_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Atomically write one compressed observation NPZ and return compact metadata."""

    normalized = normalize_observation_artifact(payload)
    destination = Path(path)
    ensure_dir(destination.parent)
    metadata = {
        key: normalized[key]
        for key in (
            "schema_version",
            "sample_id",
            "method",
            "n_features",
            "mask_semantics",
            "digest",
        )
    }
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            keep_masks=normalized["keep_masks"],
            label_scores=normalized["label_scores"],
            attribution_values=normalized["attribution_values"],
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
    os.replace(temporary, destination)
    return metadata


def load_observation_artifact(path: str | Path) -> Dict[str, Any]:
    """Load and validate one observation artifact without pickle support."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        payload = {
            **metadata,
            "keep_masks": np.asarray(archive["keep_masks"], dtype=bool),
            "label_scores": np.asarray(archive["label_scores"], dtype=np.float64),
            "attribution_values": np.asarray(
                archive["attribution_values"],
                dtype=np.float64,
            ),
        }
    normalized = normalize_observation_artifact(payload)
    if normalized["digest"] != metadata.get("digest"):
        raise ValueError(f"Observation digest mismatch in {source}.")
    return normalized


def surrogate_digest(payload: Mapping[str, Any]) -> str:
    """Hash a surrogate payload while excluding its own digest field."""

    clean = {str(key): value for key, value in payload.items() if key != "digest"}
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_surrogate_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate common surrogate fields and attach a canonical digest."""

    output = dict(payload)
    output["schema_version"] = ARTIFACT_SCHEMA_VERSION
    output["sample_id"] = str(output["sample_id"])
    output["method"] = str(output["method"])
    output["n_features"] = int(output["n_features"])
    output["mask_semantics"] = "keep"
    output["player_to_chunk_id"] = [
        int(value) for value in output["player_to_chunk_id"]
    ]
    if len(output["player_to_chunk_id"]) != output["n_features"]:
        raise ValueError("player_to_chunk_id must align with n_features.")
    predictor = dict(output["predictor"])
    predictor["intercept"] = float(predictor.get("intercept", 0.0))
    predictor["terms"] = [
        {
            "players": [int(value) for value in term.get("players", [])],
            "coefficient": float(term["coefficient"]),
        }
        for term in predictor.get("terms", [])
    ]
    output["predictor"] = predictor
    output["digest"] = surrogate_digest(output)
    return output


def write_surrogate_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Atomically write a normalized surrogate JSON artifact."""

    normalized = normalize_surrogate_artifact(payload)
    atomic_write_json(path, normalized)
    return normalized


def load_surrogate_artifact(path: str | Path) -> Dict[str, Any]:
    """Load and validate a surrogate JSON artifact."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a surrogate object in {source}.")
    normalized = normalize_surrogate_artifact(payload)
    if normalized["digest"] != payload.get("digest"):
        raise ValueError(f"Surrogate digest mismatch in {source}.")
    return normalized


def predict_surrogate(
    surrogate: Mapping[str, Any],
    keep_masks: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Predict coalition values from a serialized sparse or Fourier predictor."""

    from mobius.methods.sparse.basis import design_matrix

    payload = normalize_surrogate_artifact(surrogate)
    matrix = masks_to_bool_matrix(keep_masks, int(payload["n_features"]))
    masks = bool_matrix_to_masks(matrix)
    predictor = dict(payload["predictor"])
    basis = str(predictor["basis"])
    terms = []
    coefficients = []
    for row in predictor.get("terms", []):
        term = 0
        for player in row["players"]:
            term |= 1 << int(player)
        terms.append(term)
        coefficients.append(float(row["coefficient"]))
    if not terms:
        return np.full(len(masks), float(predictor.get("intercept", 0.0)))
    design = design_matrix(
        masks,
        terms,
        n_features=int(payload["n_features"]),
        basis=basis,
        dtype=np.float64,
    )
    return float(predictor.get("intercept", 0.0)) + design @ np.asarray(
        coefficients,
        dtype=np.float64,
    )
