"""Build one shared held-out coalition audit for multiple attribution runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.core.artifacts import (
    bool_matrix_to_masks,
    load_observation_artifact,
    masks_to_bool_matrix,
)
from mobius.core.results import canonical_digest
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.evaluation.surrogate import (
    HELDOUT_DISTRIBUTIONS,
    sample_shared_heldout_masks,
)
from mobius.models.hf import build_scorer
from mobius.models.oracle import QueryLedger, ValueOracle, stable_digest
from mobius.text.coalitions import CoalitionGame
from mobius.core.schema import TextChunk


def build_parser() -> argparse.ArgumentParser:
    """Build the shared held-out construction CLI."""

    parser = argparse.ArgumentParser(
        description="Build model-scored held-out masks shared by multiple runs."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--count-per-distribution", type=int, default=64)
    parser.add_argument(
        "--distributions",
        default="bernoulli",
        help="Comma-separated held-out distributions: bernoulli, near_full.",
    )
    parser.add_argument("--seed", type=int, default=260726)
    parser.add_argument("--near-full-deletions", default="1,2,3,5")
    parser.add_argument("--min-count", type=int, default=16)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON object with a useful path-specific error."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _normalize_distributions(values: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize and validate requested held-out sampling distributions."""

    raw_values = values.split(",") if isinstance(values, str) else values
    normalized = tuple(
        dict.fromkeys(str(value).strip() for value in raw_values if str(value).strip())
    )
    if not normalized:
        raise ValueError("At least one held-out distribution is required.")
    unknown = sorted(set(normalized) - set(HELDOUT_DISTRIBUTIONS))
    if unknown:
        raise ValueError(
            f"Unsupported held-out distributions {unknown}; "
            f"expected {list(HELDOUT_DISTRIBUTIONS)}."
        )
    return normalized


def _chunks(payload: Mapping[str, Any]) -> list[TextChunk]:
    """Reconstruct chunks for coalition-to-text conversion."""

    return [
        TextChunk(
            chunk_id=int(row["chunk_id"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            text=str(row["text"]),
            token_start=row.get("token_start"),
            token_end=row.get("token_end"),
        )
        for row in payload["chunks"]
    ]


def _sample_contract(
    sample: Mapping[str, Any],
    surrogate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the exact text/chunk/player contract required for shared masks."""

    return {
        "text": str(sample["text"]),
        "chunks": list(sample["chunks"]),
        "player_to_chunk_id": [
            int(value) for value in surrogate["player_to_chunk_id"]
        ],
        "n_features": int(surrogate["n_features"]),
    }


def _run_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Select run-level fields that determine coalition score semantics."""

    dataset = dict(config.get("dataset", {}))
    model = dict(config.get("model", {}))
    value_function = str(config.get("value_function"))
    target_mode = str(config.get("target_mode"))
    canonical_value = (
        "predicted_probability"
        if value_function == "target_probability" and target_mode == "predicted"
        else value_function
    )
    return {
        "dataset": {
            "name": dataset.get("name"),
            "split": dataset.get("split"),
            "verbalizers": dataset.get("verbalizers"),
        },
        "model": {
            "type": model.get("type", "hf_causal_lm"),
            "model_path": model.get("model_path"),
            "dtype": model.get("dtype", "bfloat16"),
            "max_length": int(model.get("max_length", 2048)),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
        },
        "chunker": config.get("chunker"),
        "value_function": canonical_value,
        "target_mode": target_mode,
    }


def _write_sample_npz(
    path: Path,
    *,
    sample_id: str,
    n_features: int,
    masks_by_distribution: Mapping[str, Sequence[int]],
    scores_by_distribution: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically write one shared held-out sample artifact."""

    ensure_dir(path.parent)
    arrays: Dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **dict(metadata),
                    "sample_id": str(sample_id),
                    "n_features": int(n_features),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    }
    for distribution, scores in scores_by_distribution.items():
        arrays[f"{distribution}_keep_masks"] = masks_to_bool_matrix(
            masks_by_distribution[distribution],
            n_features,
        )
        arrays[f"{distribution}_label_scores"] = np.asarray(
            scores,
            dtype=np.float64,
        )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _audit_id(run_payloads: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> str:
    """Build a stable audit ID from run fingerprints and held-out settings."""

    return canonical_digest(
        {
            "runs": [
                {
                    "run_id": payload.get("run_id"),
                    "config_fingerprint": payload.get("config_fingerprint"),
                }
                for payload in run_payloads
            ],
            "settings": dict(settings),
        }
    )[:12]


def build_shared_holdout(
    run_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path | None,
    device: str | None,
    count_per_distribution: int,
    seed: int,
    near_full_deletions: Sequence[int],
    min_count: int,
    distributions: Sequence[str] = ("bernoulli",),
) -> Path:
    """Align runs, query shared masks, and persist one reusable audit."""

    roots = [Path(value).resolve() for value in run_dirs]
    heldout_distributions = _normalize_distributions(distributions)
    if len(roots) < 2:
        raise ValueError("Shared held-out construction requires at least two runs.")
    run_payloads = [_load_json(root / "run.json") for root in roots]
    configs = [dict(payload["scientific_config"]) for payload in run_payloads]
    contracts = [_run_contract(config) for config in configs]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Runs do not share dataset/model/chunk/value semantics.")
    settings = {
        "count_per_distribution": int(count_per_distribution),
        "distributions": list(heldout_distributions),
        "seed": int(seed),
        "near_full_deletions": [int(value) for value in near_full_deletions],
        "min_count": int(min_count),
        "mask_semantics": "keep",
    }
    audit_id = _audit_id(run_payloads, settings)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path("results/audits/surrogate-heldout") / audit_id
    )
    samples_dir = ensure_dir(destination / "samples")
    model_config = dict(configs[0]["model"])
    if device is not None:
        model_config["device"] = str(device)
    verbalizers = [
        str(value)
        for value in dict(configs[0]["dataset"]).get("verbalizers", [])
    ]
    scorer = build_scorer(
        model_config,
        verbalizers,
        batch_size=int(configs[0].get("batch_size", 16)),
    )
    oracle = ValueOracle(
        scorer,
        destination / ".cache" / "value_oracle.sqlite3",
        model_fingerprint=model_config,
        batch_size=int(configs[0].get("batch_size", 16)),
    )
    sample_sets = [
        {path.stem for path in (root / "samples").glob("*.json")}
        for root in roots
    ]
    common_ids = sorted(set.intersection(*sample_sets))
    ledger = QueryLedger(f"surrogate-heldout/{audit_id}")
    rows: list[Dict[str, Any]] = []
    try:
        for sample_id in common_ids:
            try:
                samples = [
                    _load_json(root / "samples" / f"{sample_id}.json")
                    for root in roots
                ]
                surrogates = [
                    _load_json(root / "surrogates" / f"{sample_id}.json")
                    for root in roots
                ]
                sample_contracts = [
                    _sample_contract(sample, surrogate)
                    for sample, surrogate in zip(samples, surrogates)
                ]
                if any(
                    contract != sample_contracts[0]
                    for contract in sample_contracts[1:]
                ):
                    raise ValueError("Sample text/chunks/active players do not align.")
                n_features = int(sample_contracts[0]["n_features"])
                excluded: set[int] = set()
                observation_digests = []
                for root in roots:
                    observation = load_observation_artifact(
                        root / "observations" / f"{sample_id}.npz"
                    )
                    excluded.update(
                        bool_matrix_to_masks(observation["keep_masks"])
                    )
                    observation_digests.append(observation["digest"])
                sampled = sample_shared_heldout_masks(
                    n_features,
                    count_per_distribution=count_per_distribution,
                    seed=int(seed)
                    + int.from_bytes(
                        hashlib.sha256(sample_id.encode("utf-8")).digest()[:4],
                        "big",
                    ),
                    excluded_masks=sorted(excluded),
                    near_full_deletions=near_full_deletions,
                    distributions=heldout_distributions,
                )
                game = CoalitionGame(
                    _chunks(samples[0]),
                    sample_contracts[0]["player_to_chunk_id"],
                )
                scores_by_distribution: Dict[str, np.ndarray] = {}
                for distribution in heldout_distributions:
                    masks = sampled[distribution]
                    scores_by_distribution[distribution] = oracle.score_texts(
                        game.texts(masks),
                        logical_keys=[
                            stable_digest(
                                {
                                    "audit_id": audit_id,
                                    "sample_id": sample_id,
                                    "distribution": distribution,
                                    "mask": int(mask),
                                }
                            )
                            for mask in masks
                        ],
                        ledger=ledger,
                        category=f"surrogate_audit_{distribution}",
                    )
                counts = {
                    name: len(sampled[name])
                    for name in heldout_distributions
                }
                status = (
                    "ok"
                    if all(value >= int(min_count) for value in counts.values())
                    else "insufficient"
                )
                _write_sample_npz(
                    samples_dir / f"{sample_id}.npz",
                    sample_id=sample_id,
                    n_features=n_features,
                    masks_by_distribution=sampled,
                    scores_by_distribution=scores_by_distribution,
                    metadata={
                        "status": status,
                        "counts": counts,
                        "excluded_training_mask_count": len(excluded),
                        "observation_digests": observation_digests,
                        "audit_id": audit_id,
                        "distributions": list(heldout_distributions),
                    },
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "status": status,
                        "counts": counts,
                        "excluded_training_mask_count": len(excluded),
                        "artifact": f"samples/{sample_id}.npz",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
    finally:
        oracle.close()
    run_metadata = [
        {
            "path": str(root),
            "run_id": payload.get("run_id"),
            "config_fingerprint": payload.get("config_fingerprint"),
            "method": dict(payload.get("scientific_config", {})).get("method"),
        }
        for root, payload in zip(roots, run_payloads)
    ]
    manifest = {
        "schema_version": "1.1",
        "audit_id": audit_id,
        "kind": "shared_surrogate_heldout",
        "metadata": {
            "input_run_dirs": [str(root) for root in roots],
            "runs": run_metadata,
            "dataset": dict(contracts[0]["dataset"]),
            "model": dict(contracts[0]["model"]),
            "chunker": contracts[0].get("chunker"),
            "value_function": contracts[0].get("value_function"),
            "target_mode": contracts[0].get("target_mode"),
        },
        "runs": run_metadata,
        "run_contract": contracts[0],
        "settings": settings,
        "common_sample_count": len(common_ids),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "insufficient_count": sum(
            row["status"] == "insufficient" for row in rows
        ),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "samples": rows,
        "query_cost": ledger.to_dict(),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run shared held-out construction from CLI arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    deletions = [
        int(value)
        for value in str(args.near_full_deletions).split(",")
        if value.strip()
    ]
    destination = build_shared_holdout(
        args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        count_per_distribution=args.count_per_distribution,
        seed=args.seed,
        near_full_deletions=deletions,
        min_count=args.min_count,
        distributions=_normalize_distributions(args.distributions),
    )
    print(f"[heldout-built] audit={destination}")


if __name__ == "__main__":
    main()
