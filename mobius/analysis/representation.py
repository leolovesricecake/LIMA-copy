"""Offline basis recovery, exact structure, and full-table pair audits."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
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
from mobius.analysis.statistics import aggregate_numeric, paired_cluster_summary
from mobius.core.artifacts import (
    bool_matrix_to_masks,
    load_observation_artifact,
    load_surrogate_artifact,
    predict_surrogate,
)
from mobius.core.results import canonical_digest, dataset_slug
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.evaluation.surrogate import reconstruction_metrics
from mobius.methods.sparse.basis import (
    design_matrix,
    low_degree_terms,
)
from mobius.methods.sparse.estimator import (
    SparseModel,
    fit_sparse_model,
    sparse_model_from_surrogate,
)
from mobius.values.classification import attribution_values


Pair = Tuple[int, int]
RECONSTRUCTION_METRICS = ("r2", "nrmse_range", "mae")
RANKING_METRICS = (
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "recall_at_10",
    "sign_agreement_at_10",
    "spearman",
)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write deterministic JSONL output."""

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


def _estimator_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the estimator block used by the source Möbius run."""

    return dict(config.get("estimator", config.get("fit", {})))


def _load_heldout_manifest(
    audit_dir: str | Path,
    mobius_run: Path,
) -> tuple[Path, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load a held-out audit and require that it excluded the source run masks."""

    root = Path(audit_dir).resolve()
    manifest = load_json_object(root / "manifest.json")
    run_paths = {
        str(Path(row["path"]).resolve())
        for row in manifest.get("runs", [])
        if row.get("path")
    }
    if str(mobius_run.resolve()) not in run_paths:
        raise ValueError(
            "Held-out audit does not include the Möbius source run, so its "
            "masks are not guaranteed to be unseen."
        )
    samples = {
        str(row["sample_id"]): dict(row)
        for row in manifest.get("samples", [])
    }
    return root, manifest, samples


def _validate_heldout_contract(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Require held-out values to share dataset, model, chunk, and target semantics."""

    metadata = dict(manifest.get("metadata", {}))
    dataset = dict(contract["dataset"])
    expected_dataset = {
        "name": dataset.get("name"),
        "split": dataset.get("split"),
        "verbalizers": dataset.get("verbalizers"),
    }
    actual = {
        "dataset": metadata.get("dataset"),
        "model": metadata.get("model"),
        "prompt": dict(metadata.get("prompt", {})),
        "chunker": metadata.get("chunker"),
        "eval_granularity": metadata.get("eval_granularity"),
        "value_function": metadata.get("value_function"),
        "target_mode": metadata.get("target_mode"),
    }
    expected = {
        "dataset": expected_dataset,
        "model": contract["model"],
        "prompt": dict(contract.get("prompt", {})),
        "chunker": contract["chunker"],
        "eval_granularity": contract["eval_granularity"],
        "value_function": contract["value_function"],
        "target_mode": contract["target_mode"],
    }
    if actual != expected:
        raise ValueError(
            "Held-out audit does not match the representation run contract."
        )


def _load_heldout_sample(
    audit_root: Path,
    entry: Mapping[str, Any],
    distributions: Sequence[str],
) -> Dict[str, Any]:
    """Load masks and class scores for one held-out sample."""

    source = audit_root / str(entry["artifact"])
    with np.load(source, allow_pickle=False) as archive:
        output: Dict[str, Any] = {}
        for distribution in distributions:
            output[distribution] = {
                "keep_masks": np.asarray(
                    archive[f"{distribution}_keep_masks"],
                    dtype=bool,
                ),
                "label_scores": np.asarray(
                    archive[f"{distribution}_label_scores"],
                    dtype=np.float64,
                ),
            }
    return output


def _support_size(model: SparseModel) -> int:
    """Count nonzero fitted terms in one sparse model."""

    return len(model.coefficient_dict())


def _model_metrics(
    model: SparseModel,
    masks: Sequence[int],
    truth: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate one sparse model on explicit masks and values."""

    return reconstruction_metrics(truth, model.predict(list(masks)))


def _rank_model_terms(
    model: SparseModel,
    train_masks: Sequence[int],
) -> list[int]:
    """Rank fitted terms by empirical standardized contribution magnitude."""

    coefficients = model.coefficient_dict()
    if not coefficients:
        return []
    terms = sorted(coefficients)
    matrix = design_matrix(
        train_masks,
        terms,
        n_features=model.n_features,
        basis=model.basis,
        dtype=np.float64,
    )
    scales = np.std(matrix, axis=0)
    return [
        term
        for term, _ in sorted(
            zip(
                terms,
                [
                    abs(float(coefficients[term])) * float(scale)
                    for term, scale in zip(terms, scales)
                ],
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _top_k_refit_metrics(
    model: SparseModel,
    train_masks: Sequence[int],
    train_values: Sequence[float],
    eval_masks: Sequence[int],
    eval_values: Sequence[float],
    k: int,
) -> Dict[str, Any]:
    """Refit the top-k empirical-effect terms by OLS and evaluate reconstruction."""

    ranked = _rank_model_terms(model, train_masks)
    selected = ranked[: min(max(0, int(k)), len(ranked))]
    train_y = np.asarray(train_values, dtype=np.float64)
    if not selected:
        prediction = np.full(len(eval_values), float(np.mean(train_y)))
        metrics = reconstruction_metrics(eval_values, prediction)
        return {
            **metrics,
            "requested_k": int(k),
            "realized_k": 0,
        }
    train_matrix = design_matrix(
        train_masks,
        selected,
        n_features=model.n_features,
        basis=model.basis,
        dtype=np.float64,
    )
    augmented = np.column_stack(
        [np.ones(len(train_matrix), dtype=np.float64), train_matrix]
    )
    solution, _, _, _ = np.linalg.lstsq(augmented, train_y, rcond=None)
    eval_matrix = design_matrix(
        eval_masks,
        selected,
        n_features=model.n_features,
        basis=model.basis,
        dtype=np.float64,
    )
    prediction = float(solution[0]) + eval_matrix @ np.asarray(
        solution[1:],
        dtype=np.float64,
    )
    metrics = reconstruction_metrics(eval_values, prediction)
    return {
        **metrics,
        "requested_k": int(k),
        "realized_k": len(selected),
    }


def _refit_selected_terms(
    *,
    terms: Sequence[int],
    basis: str,
    n_features: int,
    train_masks: Sequence[int],
    train_values: Sequence[float],
    eval_masks: Sequence[int],
    eval_values: Sequence[float],
    requested_k: int,
) -> Dict[str, Any]:
    """Refit an explicit term support by OLS and evaluate it."""

    selected = [int(term) for term in terms]
    train_y = np.asarray(train_values, dtype=np.float64)
    if not selected:
        prediction = np.full(len(eval_values), float(np.mean(train_y)))
        return {
            **reconstruction_metrics(eval_values, prediction),
            "requested_k": int(requested_k),
            "realized_k": 0,
        }
    train_matrix = design_matrix(
        train_masks,
        selected,
        n_features=int(n_features),
        basis=basis,
        dtype=np.float64,
    )
    augmented = np.column_stack(
        [np.ones(len(train_matrix), dtype=np.float64), train_matrix]
    )
    solution, _, rank, singular_values = np.linalg.lstsq(
        augmented,
        train_y,
        rcond=None,
    )
    eval_matrix = design_matrix(
        eval_masks,
        selected,
        n_features=int(n_features),
        basis=basis,
        dtype=np.float64,
    )
    prediction = float(solution[0]) + eval_matrix @ np.asarray(
        solution[1:],
        dtype=np.float64,
    )
    return {
        **reconstruction_metrics(eval_values, prediction),
        "requested_k": int(requested_k),
        "realized_k": len(selected),
        "ols_rank": int(rank),
        "ols_column_count": int(augmented.shape[1]),
        "ols_rank_deficient": bool(rank < augmented.shape[1]),
        "ols_condition_number": (
            float(np.max(singular_values) / np.min(singular_values))
            if len(singular_values) and float(np.min(singular_values)) > 0
            else None
        ),
    }


def _omp_support_path(
    masks: Sequence[int],
    values: Sequence[float],
    *,
    n_features: int,
    terms: Sequence[int],
    basis: str,
    max_k: int,
) -> list[int]:
    """Build one deterministic standardized OMP support path."""

    candidates = [int(term) for term in terms]
    if not candidates or int(max_k) <= 0:
        return []
    raw = design_matrix(
        masks,
        candidates,
        n_features=int(n_features),
        basis=basis,
        dtype=np.float64,
    )
    means = np.mean(raw, axis=0)
    scales = np.std(raw, axis=0)
    identifiable = scales >= 1e-8
    if not np.any(identifiable):
        return []
    candidate_indices = np.flatnonzero(identifiable)
    matrix = (
        raw[:, identifiable] - means[identifiable]
    ) / scales[identifiable]
    residual = np.asarray(values, dtype=np.float64)
    residual = residual - float(np.mean(residual))
    selected: list[int] = []
    available = np.ones(matrix.shape[1], dtype=bool)
    limit = min(int(max_k), matrix.shape[1], max(0, len(masks) - 1))
    for _ in range(limit):
        correlations = np.abs(matrix.T @ residual)
        correlations[~available] = -np.inf
        index = int(np.argmax(correlations))
        if not np.isfinite(correlations[index]) or correlations[index] <= 1e-12:
            break
        selected.append(index)
        available[index] = False
        solution, _, _, _ = np.linalg.lstsq(
            matrix[:, selected],
            np.asarray(values, dtype=np.float64)
            - float(np.mean(values)),
            rcond=None,
        )
        residual = (
            np.asarray(values, dtype=np.float64)
            - float(np.mean(values))
            - matrix[:, selected] @ solution
        )
    return [
        candidates[int(candidate_indices[index])]
        for index in selected
    ]


def _nonzero_condition(values: np.ndarray) -> tuple[int, float | None]:
    """Return effective rank and condition over nonzero singular values."""

    if not values.size:
        return 0, None
    singular = np.linalg.svd(values, compute_uv=False)
    if not len(singular):
        return 0, None
    tolerance = float(np.max(singular)) * max(values.shape) * np.finfo(float).eps
    nonzero = singular[singular > tolerance]
    if not len(nonzero):
        return 0, None
    return int(len(nonzero)), float(np.max(nonzero) / np.min(nonzero))


def _design_geometry(
    masks: Sequence[int],
    *,
    n_features: int,
    terms: Sequence[int],
    basis: str,
    max_columns: int,
    seed: int,
) -> Dict[str, Any]:
    """Measure design geometry on a declared deterministic term subsample."""

    candidates = [int(term) for term in terms]
    limit = min(len(candidates), max(1, int(max_columns)))
    if len(candidates) > limit:
        selected_indices = np.sort(
            np.random.default_rng(int(seed)).choice(
                len(candidates),
                size=limit,
                replace=False,
            )
        )
        selected = [candidates[int(index)] for index in selected_indices]
    else:
        selected = candidates
    raw = design_matrix(
        masks,
        selected,
        n_features=int(n_features),
        basis=basis,
        dtype=np.float64,
    )
    means = np.mean(raw, axis=0) if raw.size else np.zeros(0)
    scales = np.std(raw, axis=0) if raw.size else np.zeros(0)
    identifiable = scales >= 1e-8
    centered = raw[:, identifiable] - means[identifiable]
    standardized = (
        centered / scales[identifiable]
        if np.any(identifiable)
        else np.zeros((len(masks), 0), dtype=np.float64)
    )
    raw_rank, raw_condition = _nonzero_condition(centered)
    standardized_rank, standardized_condition = _nonzero_condition(
        standardized
    )
    coherence = None
    if standardized.shape[1] >= 2:
        normalized = standardized / np.linalg.norm(
            standardized,
            axis=0,
            keepdims=True,
        )
        gram = np.abs(normalized.T @ normalized)
        np.fill_diagonal(gram, 0.0)
        coherence = float(np.max(gram))
    return {
        "candidate_count": len(candidates),
        "diagnostic_column_count": len(selected),
        "diagnostic_is_subsample": bool(len(selected) < len(candidates)),
        "identifiable_column_count": int(np.sum(identifiable)),
        "raw_effective_rank": raw_rank,
        "raw_nonzero_condition_number": raw_condition,
        "standardized_effective_rank": standardized_rank,
        "standardized_nonzero_condition_number": standardized_condition,
        "standardized_coherence": coherence,
    }


def _fwht(values: Sequence[float]) -> np.ndarray:
    """Apply the unnormalized Walsh-Hadamard transform."""

    output = np.asarray(values, dtype=np.float64).copy()
    width = 1
    while width < len(output):
        for start in range(0, len(output), 2 * width):
            left = output[start : start + width].copy()
            right = output[start + width : start + 2 * width].copy()
            output[start : start + width] = left + right
            output[start + width : start + 2 * width] = left - right
        width *= 2
    return output


def _mobius_transform(values_by_deletion: Sequence[float]) -> np.ndarray:
    """Compute exact deletion-Möbius coefficients by subset inversion."""

    coefficients = np.asarray(values_by_deletion, dtype=np.float64).copy()
    universe = len(coefficients)
    bit = 1
    while bit < universe:
        for mask in range(universe):
            if mask & bit:
                coefficients[mask] -= coefficients[mask ^ bit]
        bit <<= 1
    return coefficients


def _mobius_reconstruct(coefficients: Sequence[float]) -> np.ndarray:
    """Reconstruct deletion-table values by the subset zeta transform."""

    values = np.asarray(coefficients, dtype=np.float64).copy()
    universe = len(values)
    bit = 1
    while bit < universe:
        for mask in range(universe):
            if mask & bit:
                values[mask] += values[mask ^ bit]
        bit <<= 1
    return values


def _exact_structure_audit(
    *,
    sample_id: str,
    observation: Mapping[str, Any],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Measure intrinsic full-table structure in exact Möbius/Fourier coordinates."""

    n_features = int(observation["n_features"])
    universe = 1 << n_features
    masks = bool_matrix_to_masks(observation["keep_masks"])
    if len(set(masks)) != universe or set(masks) != set(range(universe)):
        raise ValueError("Exact structural audit requires the complete value table.")
    values_by_keep = np.zeros(universe, dtype=np.float64)
    for mask, value in zip(masks, observation["attribution_values"]):
        values_by_keep[int(mask)] = float(value)
    full = universe - 1
    values_by_deletion = np.asarray(
        [values_by_keep[full ^ deletion] for deletion in range(universe)],
        dtype=np.float64,
    )
    coefficients = {
        "mobius": _mobius_transform(values_by_deletion),
        "fourier": _fwht(values_by_keep) / float(universe),
    }
    profiles: Dict[str, Any] = {}
    truncation: Dict[str, Any] = {}
    for basis, basis_coefficients in coefficients.items():
        tolerance = max(
            1e-12,
            float(np.max(np.abs(basis_coefficients))) * 1e-9,
        )
        total_l1 = float(np.sum(np.abs(basis_coefficients[1:])))
        profiles[basis] = {}
        for degree in range(0, n_features + 1):
            indices = [
                term
                for term in range(universe)
                if bin(int(term)).count("1") == degree
            ]
            l1_mass = float(
                np.sum(np.abs(basis_coefficients[indices]))
            )
            profiles[basis][str(degree)] = {
                "term_count": len(indices),
                "nonzero_count": int(
                    np.sum(
                        np.abs(basis_coefficients[indices]) > tolerance
                    )
                ),
                "l1_mass": l1_mass,
                "l1_fraction_nonconstant": (
                    l1_mass / total_l1
                    if degree > 0 and total_l1 > 1e-15
                    else None
                ),
            }
    for degree in range(0, n_features + 1):
        truncation[str(degree)] = {}
        for basis, basis_coefficients in coefficients.items():
            retained = np.asarray(
                [
                    coefficient
                    if bin(int(term)).count("1") <= degree
                    else 0.0
                    for term, coefficient in enumerate(basis_coefficients)
                ],
                dtype=np.float64,
            )
            if basis == "mobius":
                predicted_deletion = _mobius_reconstruct(retained)
                prediction = np.asarray(
                    [
                        predicted_deletion[full ^ keep]
                        for keep in range(universe)
                    ],
                    dtype=np.float64,
                )
            else:
                prediction = _fwht(retained)
            truncation[str(degree)][basis] = reconstruction_metrics(
                values_by_keep,
                prediction,
            )
    top_k: Dict[str, Any] = {}
    all_masks = list(range(universe))
    for k in k_values:
        top_k[str(k)] = {}
        for basis, basis_coefficients in coefficients.items():
            weighted = []
            for term in range(1, universe):
                if basis == "fourier":
                    scale = 1.0
                else:
                    probability = 2.0 ** (-bin(int(term)).count("1"))
                    scale = math.sqrt(probability * (1.0 - probability))
                weighted.append(
                    (
                        int(term),
                        abs(float(basis_coefficients[term])) * scale,
                    )
                )
            ranked = [
                term
                for term, _ in sorted(
                    weighted,
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            top_k[str(k)][basis] = _refit_selected_terms(
                terms=ranked[: min(int(k), len(ranked))],
                basis=(
                    "deletion_mobius"
                    if basis == "mobius"
                    else "fourier"
                ),
                n_features=n_features,
                train_masks=all_masks,
                train_values=values_by_keep,
                eval_masks=all_masks,
                eval_values=values_by_keep,
                requested_k=int(k),
            )
    return {
        "sample_id": sample_id,
        "n_features": n_features,
        "coefficient_profile": profiles,
        "degree_truncation": truncation,
        "top_k": top_k,
    }


def _pair_coefficient(
    values_by_keep_mask: Mapping[int, float],
    full_mask: int,
    pair: Pair,
) -> float:
    """Compute one exact deletion-Möbius pair coefficient."""

    left, right = pair
    keep_left_deleted = int(full_mask) & ~(1 << int(left))
    keep_right_deleted = int(full_mask) & ~(1 << int(right))
    keep_pair_deleted = (
        int(full_mask) & ~(1 << int(left)) & ~(1 << int(right))
    )
    return float(
        values_by_keep_mask[keep_pair_deleted]
        - values_by_keep_mask[keep_left_deleted]
        - values_by_keep_mask[keep_right_deleted]
        + values_by_keep_mask[int(full_mask)]
    )


def pair_scores_from_surrogate(
    surrogate: Mapping[str, Any],
) -> Dict[Pair, float]:
    """Compute deletion-Möbius pair scores by surrogate four-point differences."""

    n_features = int(surrogate["n_features"])
    full = (1 << n_features) - 1
    pairs = list(itertools.combinations(range(n_features), 2))
    masks = {full}
    for player in range(n_features):
        masks.add(full & ~(1 << player))
    for left, right in pairs:
        masks.add(full & ~(1 << left) & ~(1 << right))
    ordered_masks = sorted(masks)
    predictions = predict_surrogate(surrogate, ordered_masks)
    values = {
        int(mask): float(value)
        for mask, value in zip(ordered_masks, predictions)
    }
    scores = {
        (left, right): float(
            values[full & ~(1 << left) & ~(1 << right)]
            - values[full & ~(1 << left)]
            - values[full & ~(1 << right)]
            + values[full]
        )
        for left, right in pairs
    }
    return {
        pair: (0.0 if abs(value) <= 1e-12 else float(value))
        for pair, value in scores.items()
    }


def pair_scores_from_sparse(surrogate: Mapping[str, Any]) -> Dict[Pair, float]:
    """Evaluate a sparse surrogate in deletion-pair coordinates."""

    predictor = dict(surrogate["predictor"])
    if str(predictor.get("basis")) != "deletion_mobius":
        raise ValueError("Exact Möbius pair audit requires deletion_mobius.")
    return pair_scores_from_surrogate(surrogate)


def pair_scores_from_proxyspex(
    surrogate: Mapping[str, Any],
) -> Dict[Pair, float]:
    """Evaluate refined-Fourier ProxySPEX in deletion-pair coordinates."""

    predictor = dict(surrogate["predictor"])
    if (
        str(predictor.get("type")) != "refined_fourier"
        or str(predictor.get("basis")) != "fourier"
    ):
        raise ValueError(
            "ProxySPEX exact-pair audit requires a refined_fourier predictor."
        )
    return pair_scores_from_surrogate(surrogate)


def rank_supported_pairs(scores: Mapping[Pair, float]) -> list[Pair]:
    """Rank explicitly supported pairs by absolute score and stable tie-break."""

    return [
        pair
        for pair, _ in sorted(
            scores.items(),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )
        if abs(float(scores[pair])) > 1e-15
    ]


def rank_all_pairs(
    pairs: Sequence[Pair],
    scores: Mapping[Pair, float],
) -> list[Pair]:
    """Rank every candidate pair by estimated magnitude with stable ties."""

    return sorted(
        [tuple(pair) for pair in pairs],
        key=lambda pair: (-abs(float(scores.get(pair, 0.0))), pair),
    )


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Compute ascending average ranks with deterministic tie handling."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Spearman correlation without requiring SciPy."""

    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.std(left_ranks) <= 1e-15 or np.std(right_ranks) <= 1e-15:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _dcg(ranking: Sequence[Pair], relevance: Mapping[Pair, float], k: int) -> float:
    """Compute discounted cumulative gain for one pair ranking."""

    return float(
        sum(
            float(relevance.get(pair, 0.0)) / math.log2(index + 2)
            for index, pair in enumerate(ranking[: int(k)])
        )
    )


def pair_ranking_metrics(
    all_pairs: Sequence[Pair],
    exact_scores: Mapping[Pair, float],
    method_scores: Mapping[Pair, float],
    ranking: Sequence[Pair],
) -> Dict[str, float | None]:
    """Compare one discovered pair ranking with exact absolute interactions."""

    oracle = sorted(
        all_pairs,
        key=lambda pair: (-abs(float(exact_scores[pair])), pair),
    )
    relevance = {pair: abs(float(exact_scores[pair])) for pair in all_pairs}
    complete_ranking = list(dict.fromkeys(tuple(pair) for pair in ranking))
    seen = set(complete_ranking)
    complete_ranking.extend(
        pair
        for pair in rank_all_pairs(all_pairs, method_scores)
        if pair not in seen
    )
    metrics: Dict[str, float | None] = {}
    for k in (1, 3, 5, 10):
        width = min(int(k), len(all_pairs))
        if width == 0:
            metrics[f"precision_at_{k}"] = None
            metrics[f"ndcg_at_{k}"] = None
            continue
        oracle_set = set(oracle[:width])
        discovered = list(complete_ranking[:width])
        metrics[f"precision_at_{k}"] = float(
            len(oracle_set.intersection(discovered)) / width
        )
        ideal = _dcg(oracle, relevance, width)
        metrics[f"ndcg_at_{k}"] = (
            float(_dcg(discovered, relevance, width) / ideal)
            if ideal > 1e-15
            else None
        )
    width = min(10, len(all_pairs))
    if width:
        exact_top = oracle[:width]
        estimated_top = complete_ranking[:width]
        metrics["recall_at_10"] = float(
            len(set(exact_top).intersection(estimated_top)) / width
        )
        signed_pairs = [
            pair
            for pair in exact_top
            if abs(float(exact_scores[pair])) > 1e-15
        ]
        metrics["sign_agreement_at_10"] = (
            float(
                np.mean(
                    [
                        np.sign(float(method_scores.get(pair, 0.0)))
                        == np.sign(float(exact_scores[pair]))
                        for pair in signed_pairs
                    ]
                )
            )
            if signed_pairs
            else None
        )
    else:
        metrics["recall_at_10"] = None
        metrics["sign_agreement_at_10"] = None
    metrics["spearman"] = _spearman(
        [abs(float(method_scores.get(pair, 0.0))) for pair in all_pairs],
        [abs(float(exact_scores[pair])) for pair in all_pairs],
    )
    return metrics


def random_pair_ranking(
    pairs: Sequence[Pair],
    *,
    seed: int,
    sample_id: str,
) -> list[Pair]:
    """Build a deterministic random control ranking for one sample."""

    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    sample_seed = int(seed) + int.from_bytes(digest[:4], "big")
    order = list(pairs)
    np.random.default_rng(sample_seed).shuffle(order)
    return order


def _rank_positions(ranking: Sequence[Pair]) -> Dict[Pair, int]:
    """Map supported pairs to one-based ranks."""

    return {pair: index + 1 for index, pair in enumerate(ranking)}


def _exact_pair_audit(
    *,
    sample_id: str,
    observation: Mapping[str, Any],
    mobius_surrogate: Mapping[str, Any],
    proxyspex_surrogate: Mapping[str, Any],
    seed: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Evaluate sparse and ProxySPEX pair rankings against a complete value table."""

    n_features = int(observation["n_features"])
    universe = 1 << n_features
    masks = bool_matrix_to_masks(observation["keep_masks"])
    if len(set(masks)) != universe or set(masks) != set(range(universe)):
        raise ValueError("Exact pair audit requires every coalition exactly once.")
    values_by_mask = {
        int(mask): float(value)
        for mask, value in zip(masks, observation["attribution_values"])
    }
    all_pairs = [
        (left, right)
        for left in range(n_features)
        for right in range(left + 1, n_features)
    ]
    full_mask = universe - 1
    exact = {
        pair: _pair_coefficient(values_by_mask, full_mask, pair)
        for pair in all_pairs
    }
    mobius = pair_scores_from_sparse(mobius_surrogate)
    proxyspex = pair_scores_from_proxyspex(proxyspex_surrogate)
    mobius_ranking = rank_all_pairs(all_pairs, mobius)
    proxy_ranking = rank_all_pairs(all_pairs, proxyspex)
    mobius_support = rank_supported_pairs(mobius)
    proxy_support = rank_supported_pairs(proxyspex)
    random_ranking = random_pair_ranking(
        all_pairs,
        seed=int(seed),
        sample_id=sample_id,
    )
    oracle_ranking = sorted(
        all_pairs,
        key=lambda pair: (-abs(float(exact[pair])), pair),
    )
    method_rows = {
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
            {pair: float(len(all_pairs) - index) for index, pair in enumerate(random_ranking)},
            random_ranking,
        ),
        "oracle": pair_ranking_metrics(
            all_pairs,
            exact,
            exact,
            oracle_ranking,
        ),
    }
    selected_errors = [
        float(mobius[pair]) - float(exact[pair])
        for pair in mobius_support
    ]
    sign_rows = [
        float(np.sign(mobius[pair]) == np.sign(exact[pair]))
        for pair in mobius_support
        if abs(float(exact[pair])) > 1e-15
    ]
    all_pair_errors = [
        float(mobius.get(pair, 0.0)) - float(exact[pair])
        for pair in all_pairs
    ]
    all_pair_sign_rows = [
        float(
            np.sign(float(mobius.get(pair, 0.0)))
            == np.sign(float(exact[pair]))
        )
        for pair in all_pairs
        if abs(float(exact[pair])) > 1e-15
    ]
    positions = {
        "oracle": _rank_positions(oracle_ranking),
        "mobius": _rank_positions(mobius_ranking),
        "proxyspex": _rank_positions(proxy_ranking),
        "random": _rank_positions(random_ranking),
    }
    pair_rows = [
        {
            "sample_id": sample_id,
            "players": list(pair),
            "exact_coefficient": float(exact[pair]),
            "mobius_coefficient": (
                float(mobius[pair]) if pair in mobius else None
            ),
            "proxyspex_score": (
                float(proxyspex[pair]) if pair in proxyspex else None
            ),
            "oracle_rank": positions["oracle"][pair],
            "mobius_rank": positions["mobius"].get(pair),
            "proxyspex_rank": positions["proxyspex"].get(pair),
            "random_rank": positions["random"][pair],
        }
        for pair in all_pairs
    ]
    return (
        {
            "sample_id": sample_id,
            "n_features": n_features,
            "pair_count": len(all_pairs),
            "method_metrics": method_rows,
            "mobius_selected_pair_count": len(mobius_support),
            "proxyspex_selected_pair_count": len(proxy_support),
            "mobius_selected_coefficient_mae": (
                float(np.mean(np.abs(selected_errors)))
                if selected_errors
                else None
            ),
            "mobius_selected_sign_agreement": (
                float(np.mean(sign_rows)) if sign_rows else None
            ),
            "mobius_all_pair_coefficient_mae": (
                float(np.mean(np.abs(all_pair_errors)))
                if all_pair_errors
                else None
            ),
            "mobius_all_pair_sign_agreement": (
                float(np.mean(all_pair_sign_rows))
                if all_pair_sign_rows
                else None
            ),
        },
        pair_rows,
    )


def _aggregate_basis_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap: int,
) -> Dict[str, Any]:
    """Aggregate basis reconstruction, support, and top-k compression rows."""

    output: Dict[str, Any] = {
        "sample_count": len(rows),
        "support_size": {
            basis: aggregate_numeric(
                [
                    float(row["models"][basis]["support_size"])
                    for row in rows
                ]
            )
            for basis in ("mobius", "fourier")
        },
        "training": {
            basis: {
                metric: aggregate_numeric(
                    [
                        row["models"][basis]["training"].get(metric)
                        for row in rows
                    ]
                )
                for metric in RECONSTRUCTION_METRICS
            }
            for basis in ("mobius", "fourier")
        },
        "design_geometry": {
            basis: {
                metric: aggregate_numeric(
                    [
                        row["models"][basis]["design_geometry"].get(metric)
                        for row in rows
                    ]
                )
                for metric in (
                    "candidate_count",
                    "diagnostic_column_count",
                    "identifiable_column_count",
                    "raw_effective_rank",
                    "raw_nonzero_condition_number",
                    "standardized_effective_rank",
                    "standardized_nonzero_condition_number",
                    "standardized_coherence",
                )
            }
            for basis in ("mobius", "fourier")
        },
        "heldout": {},
        "compression": {},
        "fixed_k": {},
        "matched_error_support": {},
    }
    distributions = sorted(
        {
            distribution
            for row in rows
            for distribution in row.get("heldout", {})
        }
    )
    for distribution in distributions:
        output["heldout"][distribution] = {}
        for metric in RECONSTRUCTION_METRICS:
            basis_values = {
                basis: [
                    row["heldout"][distribution][basis].get(metric)
                    for row in rows
                    if distribution in row.get("heldout", {})
                ]
                for basis in ("mobius", "fourier")
            }
            differences = {
                str(row["sample_id"]): [
                    (
                        float(row["heldout"][distribution]["mobius"][metric])
                        - float(row["heldout"][distribution]["fourier"][metric])
                    )
                    * (1.0 if metric == "r2" else -1.0)
                ]
                for row in rows
                if distribution in row.get("heldout", {})
                and row["heldout"][distribution]["mobius"].get(metric) is not None
                and row["heldout"][distribution]["fourier"].get(metric) is not None
            }
            output["heldout"][distribution][metric] = {
                "mobius": aggregate_numeric(basis_values["mobius"]),
                "fourier": aggregate_numeric(basis_values["fourier"]),
                "mobius_improvement": paired_cluster_summary(
                    differences,
                    seed=int(seed),
                    n_bootstrap=int(bootstrap),
                ),
                "direction": (
                    "higher_is_better" if metric == "r2" else "lower_is_better"
                ),
            }
    compression_axes = sorted(
        {
            (split, int(k))
            for row in rows
            for split, values in row.get("compression", {}).items()
            for k in values
        }
    )
    for split, k in compression_axes:
        split_output = output["compression"].setdefault(split, {})
        split_output[str(k)] = {
            basis: {
                metric: aggregate_numeric(
                    [
                        row["compression"][split][str(k)][basis].get(metric)
                        for row in rows
                        if split in row.get("compression", {})
                        and str(k) in row["compression"][split]
                    ]
                )
                for metric in RECONSTRUCTION_METRICS
            }
            for basis in ("mobius", "fourier")
        }
    fixed_axes = sorted(
        {
            (split, int(k))
            for row in rows
            for split, values in row.get("fixed_k", {}).items()
            for k in values
        }
    )
    for split, k in fixed_axes:
        metric_output: Dict[str, Any] = {}
        for metric in (
            *RECONSTRUCTION_METRICS,
            "realized_k",
            "ols_rank",
            "ols_column_count",
            "ols_rank_deficient",
            "ols_condition_number",
        ):
            basis_values = {
                basis: [
                    row["fixed_k"][split][str(k)][basis].get(metric)
                    for row in rows
                    if split in row.get("fixed_k", {})
                    and str(k) in row["fixed_k"][split]
                ]
                for basis in ("mobius", "fourier")
            }
            payload: Dict[str, Any] = {
                basis: aggregate_numeric(values)
                for basis, values in basis_values.items()
            }
            if metric in RECONSTRUCTION_METRICS:
                multiplier = 1.0 if metric == "r2" else -1.0
                differences = {
                    str(row["sample_id"]): [
                        multiplier
                        * (
                            float(
                                row["fixed_k"][split][str(k)]["mobius"][
                                    metric
                                ]
                            )
                            - float(
                                row["fixed_k"][split][str(k)]["fourier"][
                                    metric
                                ]
                            )
                        )
                    ]
                    for row in rows
                    if split in row.get("fixed_k", {})
                    and str(k) in row["fixed_k"][split]
                    and row["fixed_k"][split][str(k)]["mobius"].get(metric)
                    is not None
                    and row["fixed_k"][split][str(k)]["fourier"].get(metric)
                    is not None
                }
                payload["mobius_improvement"] = paired_cluster_summary(
                    differences,
                    seed=int(seed),
                    n_bootstrap=int(bootstrap),
                )
                payload["direction"] = (
                    "higher_is_better"
                    if metric == "r2"
                    else "lower_is_better"
                )
            metric_output[metric] = payload
        output["fixed_k"].setdefault(split, {})[str(k)] = metric_output
    fixed_splits = sorted(
        {
            split
            for row in rows
            for split in row.get("fixed_k", {})
            if split != "training"
        }
    )
    targets = (
        ("r2", 0.5, "at_least"),
        ("r2", 0.8, "at_least"),
        ("nrmse_range", 0.2, "at_most"),
        ("nrmse_range", 0.1, "at_most"),
    )
    for split in fixed_splits:
        split_output: Dict[str, Any] = {}
        for metric, threshold, relation in targets:
            target_key = f"{metric}_{relation}_{threshold:g}"
            support_by_basis: Dict[str, Dict[str, float]] = {
                "mobius": {},
                "fourier": {},
            }
            for row in rows:
                if split not in row.get("fixed_k", {}):
                    continue
                for basis in ("mobius", "fourier"):
                    for k in sorted(
                        row["fixed_k"][split],
                        key=lambda value: int(value),
                    ):
                        payload = row["fixed_k"][split][k][basis]
                        value = payload.get(metric)
                        if value is None:
                            continue
                        reached = (
                            float(value) >= float(threshold)
                            if relation == "at_least"
                            else float(value) <= float(threshold)
                        )
                        if reached:
                            support_by_basis[basis][
                                str(row["sample_id"])
                            ] = float(payload["realized_k"])
                            break
            common = sorted(
                set(support_by_basis["mobius"]).intersection(
                    support_by_basis["fourier"]
                )
            )
            differences = {
                sample_id: [
                    support_by_basis["fourier"][sample_id]
                    - support_by_basis["mobius"][sample_id]
                ]
                for sample_id in common
            }
            split_output[target_key] = {
                "metric": metric,
                "threshold": float(threshold),
                "relation": relation,
                "mobius": aggregate_numeric(
                    list(support_by_basis["mobius"].values())
                ),
                "fourier": aggregate_numeric(
                    list(support_by_basis["fourier"].values())
                ),
                "mobius_improvement": paired_cluster_summary(
                    differences,
                    seed=int(seed),
                    n_bootstrap=int(bootstrap),
                ),
                "direction": "lower_support_is_better",
            }
        output["matched_error_support"][split] = split_output
    return output


def _aggregate_exact_structure_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate intrinsic full-table structure without cross-basis energy claims."""

    output: Dict[str, Any] = {
        "sample_count": len(rows),
        "n_features": aggregate_numeric(
            [float(row["n_features"]) for row in rows]
        ),
        "coefficient_profile": {},
        "degree_truncation": {},
        "top_k": {},
    }
    profile_axes = sorted(
        {
            (basis, int(degree))
            for row in rows
            for basis, by_degree in row.get(
                "coefficient_profile",
                {},
            ).items()
            for degree in by_degree
        }
    )
    for basis, degree in profile_axes:
        degree_output = output["coefficient_profile"].setdefault(
            basis,
            {},
        ).setdefault(str(degree), {})
        for metric in (
            "term_count",
            "nonzero_count",
            "l1_mass",
            "l1_fraction_nonconstant",
        ):
            degree_output[metric] = aggregate_numeric(
                [
                    row["coefficient_profile"][basis][str(degree)].get(
                        metric
                    )
                    for row in rows
                    if basis in row.get("coefficient_profile", {})
                    and str(degree)
                    in row["coefficient_profile"][basis]
                ]
            )
    degree_axes = sorted(
        {
            int(degree)
            for row in rows
            for degree in row.get("degree_truncation", {})
        }
    )
    for degree in degree_axes:
        output["degree_truncation"][str(degree)] = {
            basis: {
                metric: aggregate_numeric(
                    [
                        row["degree_truncation"][str(degree)][basis].get(
                            metric
                        )
                        for row in rows
                        if str(degree)
                        in row.get("degree_truncation", {})
                    ]
                )
                for metric in RECONSTRUCTION_METRICS
            }
            for basis in ("mobius", "fourier")
        }
    k_axes = sorted(
        {
            int(k)
            for row in rows
            for k in row.get("top_k", {})
        }
    )
    for k in k_axes:
        output["top_k"][str(k)] = {
            basis: {
                metric: aggregate_numeric(
                    [
                        row["top_k"][str(k)][basis].get(metric)
                        for row in rows
                        if str(k) in row.get("top_k", {})
                    ]
                )
                for metric in (*RECONSTRUCTION_METRICS, "realized_k")
            }
            for basis in ("mobius", "fourier")
        }
    return output


def aggregate_exact_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap: int,
) -> Dict[str, Any]:
    """Aggregate exact pair ranking and coefficient-recovery evidence."""

    output: Dict[str, Any] = {
        "sample_count": len(rows),
        "pair_count": int(sum(int(row["pair_count"]) for row in rows)),
        "n_features": aggregate_numeric(
            [float(row["n_features"]) for row in rows]
        ),
        "methods": {},
        "mobius_selected_coefficient_mae": aggregate_numeric(
            [row.get("mobius_selected_coefficient_mae") for row in rows]
        ),
        "mobius_selected_sign_agreement": aggregate_numeric(
            [row.get("mobius_selected_sign_agreement") for row in rows]
        ),
        "mobius_all_pair_coefficient_mae": aggregate_numeric(
            [row.get("mobius_all_pair_coefficient_mae") for row in rows]
        ),
        "mobius_all_pair_sign_agreement": aggregate_numeric(
            [row.get("mobius_all_pair_sign_agreement") for row in rows]
        ),
        "mobius_vs_proxyspex": {},
        "mobius_vs_random": {},
    }
    for method in ("mobius", "proxyspex", "random", "oracle"):
        output["methods"][method] = {
            metric: aggregate_numeric(
                [
                    row["method_metrics"][method].get(metric)
                    for row in rows
                ]
            )
            for metric in RANKING_METRICS
        }
    for metric in RANKING_METRICS:
        differences = {
            str(row["sample_id"]): [
                float(row["method_metrics"]["mobius"][metric])
                - float(row["method_metrics"]["proxyspex"][metric])
            ]
            for row in rows
            if row["method_metrics"]["mobius"].get(metric) is not None
            and row["method_metrics"]["proxyspex"].get(metric) is not None
        }
        output["mobius_vs_proxyspex"][metric] = paired_cluster_summary(
            differences,
            seed=int(seed),
            n_bootstrap=int(bootstrap),
        )
        random_differences = {
            str(row["sample_id"]): [
                float(row["method_metrics"]["mobius"][metric])
                - float(row["method_metrics"]["random"][metric])
            ]
            for row in rows
            if row["method_metrics"]["mobius"].get(metric) is not None
            and row["method_metrics"]["random"].get(metric) is not None
        }
        output["mobius_vs_random"][metric] = paired_cluster_summary(
            random_differences,
            seed=int(seed) + 1,
            n_bootstrap=int(bootstrap),
        )
    return output


def _sample_ids(root: Path) -> list[str]:
    """List sample IDs with complete source sidecar files."""

    return sorted(
        path.stem
        for path in (root / "samples").glob("*.json")
        if (root / "observations" / f"{path.stem}.npz").is_file()
        and (root / "surrogates" / f"{path.stem}.json").is_file()
    )


def run_representation_audit(
    mobius_run: str | Path,
    proxyspex_run: str | Path,
    heldout_audit: str | Path,
    *,
    output_root: str | Path = "results/audits",
    compression_k: Sequence[int] = (1, 2, 4, 8, 16, 32),
    max_exact_features: int = 9,
    geometry_max_columns: int = 128,
    seed: int = 260730,
    bootstrap: int = 2000,
    max_samples: int | None = None,
) -> Path:
    """Run one content-addressed, zero-query representation audit."""

    if int(geometry_max_columns) <= 0:
        raise ValueError("geometry_max_columns must be positive.")
    if int(max_exact_features) < 0:
        raise ValueError("max_exact_features must be nonnegative.")
    mobius_identity = load_run_identity(mobius_run)
    proxy_identity = load_run_identity(proxyspex_run)
    contract = assert_same_cell([mobius_identity, proxy_identity])
    attribution_seed = assert_same_seed([mobius_identity, proxy_identity])
    mobius_root = Path(mobius_identity["root"])
    proxy_root = Path(proxy_identity["root"])
    config = dict(mobius_identity["config"])
    if (
        str(config.get("method")) != "sparse_mobius"
        or str(config.get("basis")) != "deletion_mobius"
        or str(config.get("hierarchy")) != "none"
        or str(config.get("projector")) != "signed_equal_share"
    ):
        raise ValueError(
            "The source C run must be hierarchy-free signed deletion-Möbius."
        )
    max_degree = int(config.get("max_degree", 2))
    proxy_config = dict(proxy_identity["config"])
    if (
        str(proxy_config.get("method")) != "proxyspex"
        or int(proxy_config.get("max_order", 0)) != max_degree
    ):
        raise ValueError(
            "The ProxySPEX source must use the same maximum interaction order."
        )
    if int(proxy_config.get("budget", -1)) != int(config.get("budget", -2)):
        raise ValueError(
            "Representation audit requires matched C/ProxySPEX requested budgets."
        )
    estimator = _estimator_config(config)
    heldout_root, heldout_manifest, heldout_entries = _load_heldout_manifest(
        heldout_audit,
        mobius_root,
    )
    _validate_heldout_contract(heldout_manifest, contract)
    distributions = [
        str(value)
        for value in dict(heldout_manifest.get("settings", {})).get(
            "distributions",
            ["bernoulli"],
        )
    ]
    settings = {
        "max_degree": max_degree,
        "estimator": estimator,
        "hierarchy": "none",
        "compression_k": sorted(
            {int(value) for value in compression_k if int(value) > 0}
        ),
        "max_exact_features": int(max_exact_features),
        "geometry_max_columns": int(geometry_max_columns),
        "seed": int(seed),
        "bootstrap": int(bootstrap),
        "max_samples": max_samples,
        "heldout_audit_id": heldout_manifest.get("audit_id"),
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
        / dataset_slug(dict(config.get("dataset", {})))
        / "representation"
        / audit_id
    )
    ensure_dir(destination)
    sample_ids = _sample_ids(mobius_root)
    if max_samples is not None:
        sample_ids = sample_ids[: max(0, int(max_samples))]
    basis_rows: list[Dict[str, Any]] = []
    exact_structure_rows: list[Dict[str, Any]] = []
    exact_rows: list[Dict[str, Any]] = []
    pair_rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    for sample_id in sample_ids:
        try:
            sample = load_json_object(
                mobius_root / "samples" / f"{sample_id}.json"
            )
            observation = load_observation_artifact(
                mobius_root / "observations" / f"{sample_id}.npz"
            )
            mobius_surrogate = load_surrogate_artifact(
                mobius_root / "surrogates" / f"{sample_id}.json"
            )
            n_features = int(observation["n_features"])
            train_masks = bool_matrix_to_masks(observation["keep_masks"])
            train_values = np.asarray(
                observation["attribution_values"],
                dtype=np.float64,
            )
            candidate_terms = low_degree_terms(n_features, max_degree)
            mobius_model = sparse_model_from_surrogate(mobius_surrogate)
            fourier_model = fit_sparse_model(
                train_masks,
                train_values,
                n_features=n_features,
                terms=candidate_terms,
                basis="fourier",
                max_degree=max_degree,
                config=estimator,
                random_state=int(config.get("seed", 0)),
                hierarchy_policy="none",
            )
            maximum_k = max(settings["compression_k"], default=0)
            fixed_supports = {
                "mobius": _omp_support_path(
                    train_masks,
                    train_values,
                    n_features=n_features,
                    terms=candidate_terms,
                    basis="deletion_mobius",
                    max_k=maximum_k,
                ),
                "fourier": _omp_support_path(
                    train_masks,
                    train_values,
                    n_features=n_features,
                    terms=candidate_terms,
                    basis="fourier",
                    max_k=maximum_k,
                ),
            }
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "n_features": n_features,
                "training_mask_count": len(train_masks),
                "models": {
                    "mobius": {
                        "support_size": _support_size(mobius_model),
                        "training": _model_metrics(
                            mobius_model,
                            train_masks,
                            train_values,
                        ),
                        "design_geometry": _design_geometry(
                            train_masks,
                            n_features=n_features,
                            terms=candidate_terms,
                            basis="deletion_mobius",
                            max_columns=int(geometry_max_columns),
                            seed=int(seed),
                        ),
                    },
                    "fourier": {
                        "support_size": _support_size(fourier_model),
                        "training": _model_metrics(
                            fourier_model,
                            train_masks,
                            train_values,
                        ),
                        "design_geometry": _design_geometry(
                            train_masks,
                            n_features=n_features,
                            terms=candidate_terms,
                            basis="fourier",
                            max_columns=int(geometry_max_columns),
                            seed=int(seed),
                        ),
                    },
                },
                "heldout": {},
                "compression": {},
                "fixed_k": {
                    "training": {
                        str(k): {
                            basis: _refit_selected_terms(
                                terms=fixed_supports[basis][: int(k)],
                                basis=(
                                    "deletion_mobius"
                                    if basis == "mobius"
                                    else "fourier"
                                ),
                                n_features=n_features,
                                train_masks=train_masks,
                                train_values=train_values,
                                eval_masks=train_masks,
                                eval_values=train_values,
                                requested_k=int(k),
                            )
                            for basis in ("mobius", "fourier")
                        }
                        for k in settings["compression_k"]
                    }
                },
            }
            heldout_entry = heldout_entries.get(sample_id)
            if heldout_entry and heldout_entry.get("status") == "ok":
                heldout = _load_heldout_sample(
                    heldout_root,
                    heldout_entry,
                    distributions,
                )
                for distribution in distributions:
                    values = attribution_values(
                        heldout[distribution]["label_scores"],
                        target_class=int(sample["target_label"]),
                        value_function=str(config["value_function"]),
                    )
                    masks = bool_matrix_to_masks(
                        heldout[distribution]["keep_masks"]
                    )
                    row["heldout"][distribution] = {
                        "mobius": _model_metrics(
                            mobius_model,
                            masks,
                            values,
                        ),
                        "fourier": _model_metrics(
                            fourier_model,
                            masks,
                            values,
                        ),
                    }
                    row["compression"][distribution] = {
                        str(k): {
                            "mobius": _top_k_refit_metrics(
                                mobius_model,
                                train_masks,
                                train_values,
                                masks,
                                values,
                                k,
                            ),
                            "fourier": _top_k_refit_metrics(
                                fourier_model,
                                train_masks,
                                train_values,
                                masks,
                                values,
                                k,
                            ),
                        }
                        for k in settings["compression_k"]
                    }
                    row["fixed_k"][distribution] = {
                        str(k): {
                            basis: _refit_selected_terms(
                                terms=fixed_supports[basis][: int(k)],
                                basis=(
                                    "deletion_mobius"
                                    if basis == "mobius"
                                    else "fourier"
                                ),
                                n_features=n_features,
                                train_masks=train_masks,
                                train_values=train_values,
                                eval_masks=masks,
                                eval_values=values,
                                requested_k=int(k),
                            )
                            for basis in ("mobius", "fourier")
                        }
                        for k in settings["compression_k"]
                    }
            exhaustive = (
                n_features <= int(max_exact_features)
                and len(set(train_masks)) == (1 << n_features)
            )
            row["exact_table_available"] = bool(exhaustive)
            if exhaustive:
                exact_structure_rows.append(
                    _exact_structure_audit(
                        sample_id=sample_id,
                        observation=observation,
                        k_values=settings["compression_k"],
                    )
                )
                row["compression"]["exact_full_table"] = {
                    str(k): {
                        "mobius": _top_k_refit_metrics(
                            mobius_model,
                            train_masks,
                            train_values,
                            train_masks,
                            train_values,
                            k,
                        ),
                        "fourier": _top_k_refit_metrics(
                            fourier_model,
                            train_masks,
                            train_values,
                            train_masks,
                            train_values,
                            k,
                        ),
                    }
                    for k in settings["compression_k"]
                }
                proxy_sample_path = (
                    proxy_root / "samples" / f"{sample_id}.json"
                )
                proxy_surrogate_path = (
                    proxy_root / "surrogates" / f"{sample_id}.json"
                )
                if proxy_sample_path.is_file() and proxy_surrogate_path.is_file():
                    proxy_sample = load_json_object(proxy_sample_path)
                    proxy_surrogate = load_surrogate_artifact(
                        proxy_surrogate_path
                    )
                    if str(proxy_sample.get("text")) != str(sample.get("text")):
                        raise ValueError(
                            "ProxySPEX and Möbius sample text do not align."
                        )
                    if list(proxy_sample.get("chunks", [])) != list(
                        sample.get("chunks", [])
                    ):
                        raise ValueError(
                            "ProxySPEX and Möbius sample chunks do not align."
                        )
                    if list(proxy_surrogate["player_to_chunk_id"]) != list(
                        mobius_surrogate["player_to_chunk_id"]
                    ):
                        raise ValueError(
                            "ProxySPEX and Möbius active players do not align."
                        )
                    if int(proxy_surrogate.get("target_label", -1)) != int(
                        mobius_surrogate.get("target_label", -2)
                    ):
                        raise ValueError(
                            "ProxySPEX and Möbius target labels do not align."
                        )
                    exact_row, sample_pairs = _exact_pair_audit(
                        sample_id=sample_id,
                        observation=observation,
                        mobius_surrogate=mobius_surrogate,
                        proxyspex_surrogate=proxy_surrogate,
                        seed=int(seed),
                    )
                    exact_rows.append(exact_row)
                    pair_rows.extend(sample_pairs)
                else:
                    row["exact_proxy_status"] = "missing_sidecar"
            basis_rows.append(row)
        except Exception as error:
            failures.append(
                {
                    "sample_id": sample_id,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
    summary = {
        "schema_version": "1.1",
        "audit_id": audit_id,
        "kind": "representation",
        "dataset": contract["dataset"],
        "model": contract["model"],
        "attribution_seed": attribution_seed,
        "basis": _aggregate_basis_rows(
            basis_rows,
            seed=int(seed),
            bootstrap=int(bootstrap),
        ),
        "exact_structure": _aggregate_exact_structure_rows(
            exact_structure_rows
        ),
        "exact_interactions": aggregate_exact_pair_rows(
            exact_rows,
            seed=int(seed),
            bootstrap=int(bootstrap),
        ),
        "processed_count": len(basis_rows),
        "failed_count": len(failures),
        "failures": failures,
        "query_cost": {
            "requested_queries": 0,
            "logical_unique_queries": 0,
            "physical_values_scored": 0,
            "attribution_budget_used": 0,
        },
    }
    manifest = {
        "schema_version": "1.1",
        "audit_id": audit_id,
        "kind": "representation",
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
                    "config_fingerprint": mobius_identity[
                        "config_fingerprint"
                    ],
                },
                {
                    "role": "PROXYSPEX",
                    "path": str(proxy_root),
                    "run_id": proxy_identity["run_id"],
                    "config_fingerprint": proxy_identity[
                        "config_fingerprint"
                    ],
                },
            ],
            "heldout_audit": str(heldout_root),
        },
        "settings": settings,
        "files": {
            "summary": "summary.json",
            "representation_rows": "representation-rows.jsonl",
            "exact_structure": "exact-structure.jsonl",
            "exact_samples": "exact-samples.jsonl",
            "exact_pairs": "exact-pairs.jsonl",
        },
        "query_cost": summary["query_cost"],
    }
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_json(destination / "summary.json", summary)
    _write_jsonl(destination / "representation-rows.jsonl", basis_rows)
    _write_jsonl(
        destination / "exact-structure.jsonl",
        exact_structure_rows,
    )
    _write_jsonl(destination / "exact-samples.jsonl", exact_rows)
    _write_jsonl(destination / "exact-pairs.jsonl", pair_rows)
    return destination
