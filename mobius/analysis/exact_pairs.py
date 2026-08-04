"""Exact all-pair deletion oracle for ordinary-length attribution samples."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from mobius.analysis.contracts import (
    assert_same_cell,
    assert_same_seed,
    load_json_object,
    load_run_identity,
)
from mobius.analysis.representation import (
    aggregate_exact_pair_rows,
    pair_ranking_metrics,
    pair_scores_from_proxyspex,
    pair_scores_from_sparse,
    random_pair_ranking,
    rank_all_pairs,
    rank_supported_pairs,
)
from mobius.core.artifacts import load_surrogate_artifact
from mobius.core.results import canonical_digest, dataset_slug
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.core.schema import TextChunk
from mobius.models.base import RawTextScorer
from mobius.models.hf import build_scorer
from mobius.models.oracle import QueryLedger, ValueOracle, stable_digest
from mobius.text.coalitions import CoalitionGame
from mobius.values.classification import attribution_values


Pair = Tuple[int, int]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write deterministic JSONL rows."""

    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _chunks(payload: Mapping[str, Any]) -> list[TextChunk]:
    """Reconstruct text chunks used by coalition masking."""

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


def exact_pair_keep_masks(n_features: int) -> list[int]:
    """List the unique full, singleton-deletion, and pair-deletion masks."""

    n = int(n_features)
    full = (1 << n) - 1
    masks = {full}
    for player in range(n):
        masks.add(full & ~(1 << player))
    for left, right in itertools.combinations(range(n), 2):
        masks.add(full & ~(1 << left) & ~(1 << right))
    return sorted(masks)


def exact_pair_coefficients(
    values_by_keep_mask: Mapping[int, float],
    n_features: int,
) -> Dict[Pair, float]:
    """Compute every exact degree-two deletion-Möbius coefficient."""

    n = int(n_features)
    full = (1 << n) - 1
    return {
        (left, right): float(
            values_by_keep_mask[
                full & ~(1 << left) & ~(1 << right)
            ]
            - values_by_keep_mask[full & ~(1 << left)]
            - values_by_keep_mask[full & ~(1 << right)]
            + values_by_keep_mask[full]
        )
        for left, right in itertools.combinations(range(n), 2)
    }


def _sample_contract(
    sample: Mapping[str, Any],
    surrogate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select fields that must align between C and ProxySPEX."""

    return {
        "text": str(sample["text"]),
        "chunks": list(sample["chunks"]),
        "target_label": int(sample["target_label"]),
        "n_features": int(surrogate["n_features"]),
        "player_to_chunk_id": [
            int(value) for value in surrogate["player_to_chunk_id"]
        ],
    }


def _stable_sample_order(
    rows: Sequence[tuple[str, int]],
    *,
    seed: int,
    maximum: int | None,
) -> list[str]:
    """Select a deterministic random audit subset without length stratification."""

    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row[0]}".encode("utf-8")
        ).hexdigest(),
    )
    sample_ids = [sample_id for sample_id, _ in ordered]
    return (
        sample_ids
        if maximum is None
        else sample_ids[: max(0, int(maximum))]
    )


def _eligible_samples(
    mobius_root: Path,
    proxy_root: Path,
    *,
    min_features: int,
    max_features: int,
    max_samples: int | None,
    seed: int,
) -> list[str]:
    """Find aligned samples and apply one deterministic random limit."""

    common = {
        path.stem for path in (mobius_root / "samples").glob("*.json")
    }.intersection(
        path.stem for path in (proxy_root / "samples").glob("*.json")
    )
    eligible = []
    for sample_id in sorted(common):
        mobius_surrogate_path = (
            mobius_root / "surrogates" / f"{sample_id}.json"
        )
        proxy_surrogate_path = (
            proxy_root / "surrogates" / f"{sample_id}.json"
        )
        if not mobius_surrogate_path.is_file() or not proxy_surrogate_path.is_file():
            continue
        surrogate = load_surrogate_artifact(mobius_surrogate_path)
        n_features = int(surrogate["n_features"])
        if int(min_features) <= n_features <= int(max_features):
            eligible.append((sample_id, n_features))
    return _stable_sample_order(
        eligible,
        seed=int(seed),
        maximum=max_samples,
    )


def _method_metrics(
    exact: Mapping[Pair, float],
    mobius: Mapping[Pair, float],
    proxyspex: Mapping[Pair, float],
    *,
    seed: int,
    sample_id: str,
) -> Dict[str, Dict[str, float | None]]:
    """Evaluate both methods, random control, and oracle on one exact table."""

    all_pairs = sorted(exact)
    mobius_ranking = rank_all_pairs(all_pairs, mobius)
    proxy_ranking = rank_all_pairs(all_pairs, proxyspex)
    random_ranking = random_pair_ranking(
        all_pairs,
        seed=int(seed),
        sample_id=sample_id,
    )
    oracle_ranking = sorted(
        all_pairs,
        key=lambda pair: (-abs(float(exact[pair])), pair),
    )
    return {
        "mobius": pair_ranking_metrics(
            all_pairs,
            exact,
            mobius,
            mobius_ranking,
        ),
        "proxyspex": pair_ranking_metrics(
            all_pairs,
            exact,
            proxyspex,
            proxy_ranking,
        ),
        "random": pair_ranking_metrics(
            all_pairs,
            exact,
            {
                pair: float(len(all_pairs) - index)
                for index, pair in enumerate(random_ranking)
            },
            random_ranking,
        ),
        "oracle": pair_ranking_metrics(
            all_pairs,
            exact,
            exact,
            oracle_ranking,
        ),
    }


def _sample_rows(
    *,
    sample_id: str,
    n_features: int,
    exact: Mapping[Pair, float],
    mobius: Mapping[Pair, float],
    proxyspex: Mapping[Pair, float],
    method_metrics: Mapping[str, Mapping[str, float | None]],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Build aggregate-compatible sample and pair rows."""

    all_pairs = sorted(exact)
    mobius_ranking = rank_all_pairs(all_pairs, mobius)
    proxy_ranking = rank_all_pairs(all_pairs, proxyspex)
    mobius_support = rank_supported_pairs(mobius)
    proxy_support = rank_supported_pairs(proxyspex)
    oracle_ranking = sorted(
        exact,
        key=lambda pair: (-abs(float(exact[pair])), pair),
    )
    mobius_positions = {
        pair: index + 1 for index, pair in enumerate(mobius_ranking)
    }
    proxy_positions = {
        pair: index + 1 for index, pair in enumerate(proxy_ranking)
    }
    oracle_positions = {
        pair: index + 1 for index, pair in enumerate(oracle_ranking)
    }
    selected_errors = [
        float(mobius[pair]) - float(exact[pair])
        for pair in mobius_support
    ]
    selected_signs = [
        float(np.sign(mobius[pair]) == np.sign(exact[pair]))
        for pair in mobius_support
        if abs(float(exact[pair])) > 1e-15
    ]
    all_errors = [
        float(mobius.get(pair, 0.0)) - float(exact[pair])
        for pair in exact
    ]
    all_signs = [
        float(
            np.sign(float(mobius.get(pair, 0.0)))
            == np.sign(float(exact[pair]))
        )
        for pair in exact
        if abs(float(exact[pair])) > 1e-15
    ]
    sample = {
        "sample_id": sample_id,
        "n_features": int(n_features),
        "pair_count": len(exact),
        "method_metrics": {
            method: dict(metrics)
            for method, metrics in method_metrics.items()
        },
        "mobius_selected_pair_count": len(mobius_support),
        "proxyspex_selected_pair_count": len(proxy_support),
        "mobius_selected_coefficient_mae": (
            float(np.mean(np.abs(selected_errors)))
            if selected_errors
            else None
        ),
        "mobius_selected_sign_agreement": (
            float(np.mean(selected_signs)) if selected_signs else None
        ),
        "mobius_all_pair_coefficient_mae": (
            float(np.mean(np.abs(all_errors))) if all_errors else None
        ),
        "mobius_all_pair_sign_agreement": (
            float(np.mean(all_signs)) if all_signs else None
        ),
    }
    pairs = [
        {
            "sample_id": sample_id,
            "n_features": int(n_features),
            "players": list(pair),
            "exact_coefficient": float(exact[pair]),
            "mobius_coefficient": (
                float(mobius[pair]) if pair in mobius else None
            ),
            "proxyspex_score": (
                float(proxyspex[pair]) if pair in proxyspex else None
            ),
            "oracle_rank": oracle_positions[pair],
            "mobius_rank": mobius_positions.get(pair),
            "proxyspex_rank": proxy_positions.get(pair),
        }
        for pair in sorted(exact)
    ]
    return sample, pairs


def _length_bin(n_features: int) -> str:
    """Map one player count to a stable eight-feature interval."""

    lower = (int(n_features) // 8) * 8
    upper = lower + 7
    return f"{lower:02d}-{upper:02d}"


def run_exact_pair_audit(
    mobius_run: str | Path,
    proxyspex_run: str | Path,
    *,
    output_root: str | Path = "results/audits",
    cache_path: str | Path = "results/.cache/value_oracle.sqlite3",
    device: str | None = None,
    min_features: int = 10,
    max_features: int = 32,
    max_samples: int | None = 100,
    seed: int = 260731,
    bootstrap: int = 2000,
    scorer: RawTextScorer | None = None,
) -> Path:
    """Query one exact all-pair table per selected ordinary-length sample."""

    if int(min_features) < 2:
        raise ValueError("min_features must be at least 2.")
    if int(max_features) < int(min_features):
        raise ValueError("max_features must be at least min_features.")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")
    mobius_identity = load_run_identity(mobius_run)
    proxy_identity = load_run_identity(proxyspex_run)
    contract = assert_same_cell([mobius_identity, proxy_identity])
    attribution_seed = assert_same_seed([mobius_identity, proxy_identity])
    mobius_root = Path(mobius_identity["root"])
    proxy_root = Path(proxy_identity["root"])
    mobius_config = dict(mobius_identity["config"])
    proxy_config = dict(proxy_identity["config"])
    if (
        str(mobius_config.get("method")) != "sparse_mobius"
        or str(mobius_config.get("basis")) != "deletion_mobius"
    ):
        raise ValueError("The C run must be a deletion-Möbius run.")
    if (
        str(proxy_config.get("method")) != "proxyspex"
        or int(proxy_config.get("max_order", 0))
        != int(mobius_config.get("max_degree", 0))
    ):
        raise ValueError(
            "The comparison run must be degree-matched native ProxySPEX."
        )
    if int(proxy_config.get("budget", -1)) != int(
        mobius_config.get("budget", -2)
    ):
        raise ValueError(
            "Exact pair audit requires matched C/ProxySPEX requested budgets."
        )
    settings = {
        "oracle_scope": "pair_oracle",
        "min_features": int(min_features),
        "max_features": int(max_features),
        "max_samples": max_samples,
        "seed": int(seed),
        "bootstrap": int(bootstrap),
    }
    audit_id = canonical_digest(
        {
            "mobius": {
                "run_id": mobius_identity["run_id"],
                "fingerprint": mobius_identity["config_fingerprint"],
            },
            "proxyspex": {
                "run_id": proxy_identity["run_id"],
                "fingerprint": proxy_identity["config_fingerprint"],
            },
            "settings": settings,
        }
    )[:12]
    destination = (
        Path(output_root)
        / dataset_slug(dict(mobius_config.get("dataset", {})))
        / "interactions"
        / audit_id
    )
    ensure_dir(destination)
    model_config = dict(mobius_config.get("model", {}))
    if device is not None:
        model_config["device"] = str(device)
    active_scorer = scorer or build_scorer(
        model_config,
        list(dict(mobius_config["dataset"]).get("verbalizers", [])),
        batch_size=int(mobius_config.get("batch_size", 16)),
        dataset_name=str(dict(mobius_config["dataset"]).get("name", "dataset")),
        prompt_config=dict(mobius_config.get("prompt", {})),
    )
    oracle = ValueOracle(
        active_scorer,
        cache_path,
        model_fingerprint=contract["model"],
        batch_size=int(mobius_config.get("batch_size", 16)),
    )
    ledger = QueryLedger(f"exact-pair-oracle/{audit_id}")
    selected_ids = _eligible_samples(
        mobius_root,
        proxy_root,
        min_features=int(min_features),
        max_features=int(max_features),
        max_samples=max_samples,
        seed=int(seed),
    )
    sample_rows: list[Dict[str, Any]] = []
    pair_rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    try:
        for sample_id in selected_ids:
            try:
                mobius_sample = load_json_object(
                    mobius_root / "samples" / f"{sample_id}.json"
                )
                proxy_sample = load_json_object(
                    proxy_root / "samples" / f"{sample_id}.json"
                )
                mobius_surrogate = load_surrogate_artifact(
                    mobius_root / "surrogates" / f"{sample_id}.json"
                )
                proxy_surrogate = load_surrogate_artifact(
                    proxy_root / "surrogates" / f"{sample_id}.json"
                )
                if _sample_contract(
                    mobius_sample,
                    mobius_surrogate,
                ) != _sample_contract(proxy_sample, proxy_surrogate):
                    raise ValueError(
                        "C and ProxySPEX sample/chunk/target contracts differ."
                    )
                n_features = int(mobius_surrogate["n_features"])
                game = CoalitionGame(
                    _chunks(mobius_sample),
                    mobius_surrogate["player_to_chunk_id"],
                )
                masks = exact_pair_keep_masks(n_features)
                label_scores = oracle.score_texts(
                    game.texts(masks),
                    logical_keys=[
                        stable_digest(
                            {
                                "audit_id": audit_id,
                                "sample_id": sample_id,
                                "mask": int(mask),
                                "category": "exact_pair_oracle",
                            }
                        )
                        for mask in masks
                    ],
                    ledger=ledger,
                    category="exact_pair_oracle",
                )
                values = attribution_values(
                    label_scores,
                    target_class=int(mobius_sample["target_label"]),
                    value_function=str(mobius_config["value_function"]),
                )
                exact = exact_pair_coefficients(
                    {
                        int(mask): float(value)
                        for mask, value in zip(masks, values)
                    },
                    n_features,
                )
                mobius = pair_scores_from_sparse(mobius_surrogate)
                proxyspex = pair_scores_from_proxyspex(proxy_surrogate)
                metrics = _method_metrics(
                    exact,
                    mobius,
                    proxyspex,
                    seed=int(seed),
                    sample_id=sample_id,
                )
                sample_row, sample_pairs = _sample_rows(
                    sample_id=sample_id,
                    n_features=n_features,
                    exact=exact,
                    mobius=mobius,
                    proxyspex=proxyspex,
                    method_metrics=metrics,
                )
                sample_row["query_count"] = len(masks)
                sample_rows.append(sample_row)
                pair_rows.extend(sample_pairs)
            except Exception as error:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
    finally:
        oracle_counters = oracle.snapshot_counters()
        oracle.close()
    aggregate = aggregate_exact_pair_rows(
        sample_rows,
        seed=int(seed),
        bootstrap=int(bootstrap),
    )
    by_length_bin = {
        length_bin: aggregate_exact_pair_rows(
            [
                row
                for row in sample_rows
                if _length_bin(int(row["n_features"])) == length_bin
            ],
            seed=int(seed),
            bootstrap=int(bootstrap),
        )
        for length_bin in sorted(
            {
                _length_bin(int(row["n_features"]))
                for row in sample_rows
            }
        )
    }
    summary = {
        "schema_version": "1.0",
        "audit_id": audit_id,
        "kind": "exact_pair_oracle",
        "oracle_scope": "pair_oracle",
        "dataset": contract["dataset"],
        "model": contract["model"],
        "attribution_seed": attribution_seed,
        "selected_count": len(selected_ids),
        "processed_count": len(sample_rows),
        "failed_count": len(failures),
        "aggregate": aggregate,
        "by_length_bin": by_length_bin,
        "query_cost": ledger.to_dict(),
        "oracle_counters": oracle_counters,
        "failures": failures,
    }
    manifest = {
        "schema_version": "1.0",
        "audit_id": audit_id,
        "kind": "exact_pair_oracle",
        "metadata": {
            "dataset": contract["dataset"],
            "model": contract["model"],
            "prompt": dict(contract.get("prompt", {})),
            "value_function": contract["value_function"],
            "target_mode": contract["target_mode"],
            "chunker": contract["chunker"],
            "eval_granularity": contract["eval_granularity"],
            "attribution_seed": attribution_seed,
            "runs": [
                {
                    "role": "C",
                    "path": str(mobius_root),
                    "run_id": mobius_identity["run_id"],
                },
                {
                    "role": "PROXYSPEX",
                    "path": str(proxy_root),
                    "run_id": proxy_identity["run_id"],
                },
            ],
        },
        "settings": settings,
        "files": {
            "summary": "summary.json",
            "samples": "samples.jsonl",
            "pairs": "pairs.jsonl",
        },
        "query_cost": summary["query_cost"],
    }
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_json(destination / "summary.json", summary)
    _write_jsonl(destination / "samples.jsonl", sample_rows)
    _write_jsonl(destination / "pairs.jsonl", pair_rows)
    for root in (mobius_root, proxy_root):
        atomic_write_json(
            ensure_dir(root / "analyses")
            / f"exact-pair-oracle-{audit_id}.json",
            {
                "schema_version": "1.0",
                "audit_id": audit_id,
                "audit_dir": str(destination.resolve()),
                "summary": str((destination / "summary.json").resolve()),
            },
        )
    return destination


__all__ = [
    "exact_pair_coefficients",
    "exact_pair_keep_masks",
    "run_exact_pair_audit",
]
