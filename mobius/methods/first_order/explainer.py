"""Word-level Occlusion and LIME over the shared classification value function."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from mobius.core.artifacts import (
    normalize_observation_artifact,
    normalize_surrogate_artifact,
)
from mobius.core.config import resolve_config, scientific_config
from mobius.core.results import ResultStore
from mobius.core.runtime import counter_delta
from mobius.core.schema import AttributionResult, DatasetBundle, TextSample
from mobius.evaluation.evaluator import evaluate_run
from mobius.models.base import RawTextScorer
from mobius.models.oracle import QueryLedger, ValueOracle, stable_digest
from mobius.models.prompting import build_classification_prompt
from mobius.text.chunks import build_chunks, compose_text
from mobius.text.coalitions import CoalitionGame, active_chunk_ids, visible_text_span
from mobius.values.classification import (
    attribution_values,
    effective_target_mode,
    normalize_value_function,
    probabilities,
)


FIRST_ORDER_METHODS = {"word_occlusion", "word_lime"}


class FirstOrderSampleExcludedError(ValueError):
    """Signal a deliberate feature-count or truncation exclusion."""


def normalize_first_order_method(value: str) -> str:
    """Normalize supported first-order method names."""

    normalized = str(value).strip().lower()
    aliases = {
        "occlusion": "word_occlusion",
        "lime": "word_lime",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FIRST_ORDER_METHODS:
        raise ValueError(
            f"Unsupported first-order method {value!r}; "
            f"expected {sorted(FIRST_ORDER_METHODS)}."
        )
    return normalized


def _mask_digest(masks: Sequence[int]) -> str:
    """Hash one deterministic integer-mask sequence."""

    encoded = ",".join(str(int(mask)) for mask in masks).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _lime_masks(n_features: int, budget: int, seed: int) -> list[int]:
    """Sample unique LIME text neighborhoods with uniform deletion counts."""

    n = int(n_features)
    universe = 1 << n
    target = min(max(1, int(budget)), universe)
    full = universe - 1
    selected = {full}
    if target > 1:
        selected.add(0)
    rng = np.random.default_rng(int(seed))
    attempts = 0
    while len(selected) < target and attempts < max(2000, target * 500):
        attempts += 1
        deletion_count = int(rng.integers(1, n + 1))
        deleted = rng.choice(n, size=deletion_count, replace=False)
        mask = full
        for player in deleted:
            mask &= ~(1 << int(player))
        selected.add(mask)
    if len(selected) < target and universe <= (1 << 20):
        remaining = [mask for mask in range(universe) if mask not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[: target - len(selected)])
    if len(selected) != target:
        raise RuntimeError(f"Could not realize LIME budget {target}.")
    return [full] + sorted(selected - {full})


def _mask_matrix(masks: Sequence[int], n_features: int) -> np.ndarray:
    """Decode integer keep masks into a dense binary matrix."""

    return np.asarray(
        [
            [
                float(bool(int(mask) & (1 << player)))
                for player in range(int(n_features))
            ]
            for mask in masks
        ],
        dtype=np.float64,
    )


def _lime_kernel_weights(
    keep_matrix: np.ndarray,
    *,
    kernel_width: float,
) -> np.ndarray:
    """Compute the standard LIME text cosine-distance kernel weights."""

    width = float(kernel_width)
    if width <= 0:
        raise ValueError("first_order.lime_kernel_width must be positive.")
    n_features = int(keep_matrix.shape[1])
    kept = np.sum(keep_matrix, axis=1)
    cosine_similarity = np.sqrt(
        np.clip(kept / max(1, n_features), 0.0, 1.0)
    )
    distances = 100.0 * (1.0 - cosine_similarity)
    return np.sqrt(np.exp(-(distances**2) / (width**2)))


def _weighted_ridge(
    matrix: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, np.ndarray, Dict[str, Any]]:
    """Fit a weighted linear model with an unregularized intercept."""

    ridge_alpha = float(alpha)
    if ridge_alpha < 0:
        raise ValueError("first_order.lime_ridge_alpha must be nonnegative.")
    augmented = np.column_stack(
        [np.ones(len(matrix), dtype=np.float64), matrix]
    )
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted_design = augmented * sqrt_weights[:, None]
    weighted_values = np.asarray(values, dtype=np.float64) * sqrt_weights
    penalty = np.zeros((matrix.shape[1], augmented.shape[1]), dtype=np.float64)
    if matrix.shape[1]:
        penalty[:, 1:] = np.eye(matrix.shape[1]) * np.sqrt(ridge_alpha)
    system = np.vstack([weighted_design, penalty])
    target = np.concatenate(
        [weighted_values, np.zeros(matrix.shape[1], dtype=np.float64)]
    )
    solution, _, rank, singular_values = np.linalg.lstsq(
        system,
        target,
        rcond=None,
    )
    prediction = augmented @ solution
    weighted_error = float(
        np.average((np.asarray(values) - prediction) ** 2, weights=weights)
    )
    return (
        float(solution[0]),
        np.asarray(solution[1:], dtype=np.float64),
        {
            "ridge_alpha": ridge_alpha,
            "weighted_train_mse": weighted_error,
            "linear_system_rank": int(rank),
            "linear_system_column_count": int(system.shape[1]),
            "linear_system_condition_number": (
                float(np.max(singular_values) / np.min(singular_values))
                if len(singular_values) and float(np.min(singular_values)) > 0
                else None
            ),
        },
    )


def _ranking(
    player_to_chunk_id: Sequence[int],
    scores: Sequence[float],
    all_chunk_ids: Sequence[int],
) -> list[int]:
    """Rank active chunks by descending signed importance."""

    active = sorted(
        (
            (int(chunk_id), float(scores[player]))
            for player, chunk_id in enumerate(player_to_chunk_id)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ranked = [chunk_id for chunk_id, _ in active]
    active_set = set(ranked)
    return ranked + [
        int(chunk_id)
        for chunk_id in all_chunk_ids
        if int(chunk_id) not in active_set
    ]


class FirstOrderExplainer:
    """Explain text with shared-value word Occlusion or LIME."""

    def __init__(
        self,
        config: Mapping[str, Any],
        scorer: RawTextScorer,
        oracle: ValueOracle,
        verbalizers: Sequence[str],
    ) -> None:
        """Resolve the first-order protocol and retain shared runtime objects."""

        self.config = resolve_config(config)
        self.method = normalize_first_order_method(str(self.config["method"]))
        self.scorer = scorer
        self.oracle = oracle
        self.verbalizers = [str(value) for value in verbalizers]

    def _score_masks(
        self,
        sample_id: str,
        game: CoalitionGame,
        masks: Sequence[int],
        ledger: QueryLedger,
    ) -> np.ndarray:
        """Score one method-local mask list through the global value cache."""

        return self.oracle.score_texts(
            game.texts(masks),
            logical_keys=[
                stable_digest(
                    {
                        "method": self.method,
                        "sample_id": sample_id,
                        "mask": int(mask),
                        "operator": "keep",
                        "category": "training",
                    }
                )
                for mask in masks
            ],
            ledger=ledger,
            category="training",
        )

    def explain(self, sample: TextSample) -> AttributionResult:
        """Run first-order attribution and serialize a compatible surrogate."""

        started = time.perf_counter()
        counters_before = self.scorer.snapshot_counters()
        value_function = normalize_value_function(self.config["value_function"])
        target_mode = effective_target_mode(
            value_function,
            str(self.config["target_mode"]),
        )
        setup_ledger = QueryLedger(f"{self.method}/{sample.sample_id}/setup")
        full_scores = self.oracle.score_texts(
            [sample.text],
            logical_keys=[
                stable_digest(
                    {"sample_id": sample.sample_id, "category": "full_input"}
                )
            ],
            ledger=setup_ledger,
            category="setup",
        )[0]
        predicted_label = int(np.argmax(full_scores))
        target_label = (
            predicted_label if target_mode == "predicted" else int(sample.label)
        )
        chunking = build_chunks(
            sample.text,
            str(self.config["chunker"]),
            getattr(self.scorer, "tokenizer", None),
            adaptive_profile=str(self.config["adaptive_profile"]),
            adaptive_overrides=self.config.get("adaptive_overrides"),
        )
        truncation = visible_text_span(
            getattr(self.scorer, "tokenizer", None),
            sample.text,
            self.verbalizers[target_label],
            int(dict(self.config.get("model", {})).get("max_length", 2048)),
            prompt_prefix=str(getattr(self.scorer, "prompt_prefix", "Text:\n")),
            prompt_suffix=str(getattr(self.scorer, "prompt_suffix", "\nLabel:")),
            label_token_reserve=(
                int(self.scorer.label_token_reserve())
                if callable(getattr(self.scorer, "label_token_reserve", None))
                else None
            ),
            label_prefix=str(getattr(self.scorer, "label_prefix", " ")),
        )
        players = active_chunk_ids(
            chunking.chunks,
            int(truncation["visible_start_char"]),
            int(truncation["visible_end_char"]),
        )
        if not players:
            raise FirstOrderSampleExcludedError(
                "No explanation chunks remain after truncation."
            )
        minimum = int(self.config["min_features"])
        maximum = self.config.get("max_features")
        if len(players) < minimum:
            raise FirstOrderSampleExcludedError(
                f"active feature count {len(players)} is below min_features={minimum}"
            )
        if maximum is not None and len(players) > int(maximum):
            raise FirstOrderSampleExcludedError(
                f"active feature count {len(players)} exceeds max_features={maximum}"
            )

        game = CoalitionGame(chunking.chunks, players)
        full_mask = (1 << game.n_players) - 1
        if self.method == "word_occlusion":
            masks = [full_mask] + [
                full_mask & ~(1 << player)
                for player in range(game.n_players)
            ]
        else:
            masks = _lime_masks(
                game.n_players,
                int(self.config["budget"]),
                int(self.config["seed"]),
            )
        attribution_before = self.scorer.snapshot_counters()
        ledger = QueryLedger(
            f"{self.method}/{sample.sample_id}/"
            f"{self.config['budget']}/{self.config['seed']}"
        )
        label_scores = self._score_masks(
            sample.sample_id,
            game,
            masks,
            ledger,
        )
        values = attribution_values(
            label_scores,
            target_class=target_label,
            value_function=value_function,
        )
        full_index = masks.index(full_mask)
        fit_diagnostics: Dict[str, Any]
        if self.method == "word_occlusion":
            value_by_mask = {
                int(mask): float(value)
                for mask, value in zip(masks, values)
            }
            active_scores = np.asarray(
                [
                    value_by_mask[full_mask]
                    - value_by_mask[full_mask & ~(1 << player)]
                    for player in range(game.n_players)
                ],
                dtype=np.float64,
            )
            deletion_intercept = float(values[full_index])
            deletion_coefficients = -active_scores
            fit_diagnostics = {
                "estimator": "single_deletion_difference",
                "requested_budget": game.n_players + 1,
                "realized_budget": len(masks),
            }
        else:
            matrix = _mask_matrix(masks, game.n_players)
            first_order = dict(self.config.get("first_order", {}))
            weights = _lime_kernel_weights(
                matrix,
                kernel_width=float(
                    first_order.get("lime_kernel_width", 25.0)
                ),
            )
            keep_intercept, active_scores, fit_diagnostics = _weighted_ridge(
                matrix,
                values,
                weights,
                alpha=float(first_order.get("lime_ridge_alpha", 1.0)),
            )
            deletion_intercept = float(
                keep_intercept + np.sum(active_scores)
            )
            deletion_coefficients = -active_scores
            fit_diagnostics.update(
                {
                    "estimator": "lime_weighted_ridge_all_features",
                    "kernel": "sqrt_exp_negative_cosine_distance_squared",
                    "kernel_width": float(
                        first_order.get("lime_kernel_width", 25.0)
                    ),
                    "requested_budget": int(self.config["budget"]),
                    "realized_budget": len(masks),
                }
            )

        node_scores = [0.0] * len(chunking.chunks)
        for player, chunk_id in enumerate(players):
            node_scores[int(chunk_id)] = float(active_scores[player])
        ranking = _ranking(
            players,
            active_scores,
            [chunk.chunk_id for chunk in chunking.chunks],
        )
        selected = ranking[: min(int(self.config["k"]), len(players))]
        observation = normalize_observation_artifact(
            {
                "sample_id": sample.sample_id,
                "method": self.method,
                "n_features": game.n_players,
                "keep_masks": masks,
                "label_scores": label_scores,
                "attribution_values": values,
            }
        )
        surrogate = normalize_surrogate_artifact(
            {
                "sample_id": sample.sample_id,
                "method": self.method,
                "n_features": game.n_players,
                "player_to_chunk_id": players,
                "observation_file": f"../observations/{sample.sample_id}.npz",
                "observation_digest": observation["digest"],
                "value_function": value_function,
                "target_mode": target_mode,
                "target_label": target_label,
                "predictor": {
                    "type": "sparse_polynomial",
                    "basis": "deletion_mobius",
                    "intercept": deletion_intercept,
                    "terms": [
                        {
                            "players": [player],
                            "coefficient": float(coefficient),
                        }
                        for player, coefficient in enumerate(
                            deletion_coefficients
                        )
                    ],
                },
                "fit_config": dict(self.config.get("first_order", {})),
                "fit_diagnostics": fit_diagnostics,
            }
        )
        counters_after = self.scorer.snapshot_counters()
        attribution_delta = counter_delta(attribution_before, counters_after)
        elapsed = time.perf_counter() - started
        attribution_cost = {
            **ledger.to_dict(),
            "attribution_budget_used": len(masks),
            "setup_physical_values_scored": setup_ledger.physical_values_scored,
            "interaction_verification_queries": 0,
            "model_forward_calls": int(
                attribution_delta.get("model_forward_calls", 0)
            ),
            "batch_calls": int(attribution_delta.get("batch_calls", 0)),
            "batch_rows": int(attribution_delta.get("batch_rows", 0)),
            "model_counter_delta": attribution_delta,
            "total_counter_delta_including_setup": counter_delta(
                counters_before,
                counters_after,
            ),
            "elapsed_seconds": elapsed,
        }
        return AttributionResult(
            sample_id=sample.sample_id,
            gold_label=int(sample.label),
            predicted_label=predicted_label,
            target_label=target_label,
            text=sample.text,
            chunks=chunking.chunks,
            node_scores=node_scores,
            ranking=ranking,
            selected_ids=selected,
            attribution_cost=attribution_cost,
            method_summary={
                "method": self.method,
                "value_function": value_function,
                "target_mode": target_mode,
                "full_label_scores": [
                    float(value) for value in full_scores
                ],
                "full_label_probabilities": [
                    float(value)
                    for value in probabilities(full_scores.reshape(1, -1))[0]
                ],
                "chunking": chunking.diagnostics,
                "truncation": {
                    **truncation,
                    "active_chunk_ids": players,
                    "active_chunk_count": len(players),
                },
                "fit": fit_diagnostics,
                "masks_digest": _mask_digest(masks),
                "selected_text": compose_text(chunking.chunks, selected),
                "observation_digest": observation["digest"],
                "surrogate_digest": surrogate["digest"],
            },
            diagnostics={
                "masks": [int(mask) for mask in masks],
                "values": [float(value) for value in values],
                "query_ledger": ledger.to_dict(),
            },
            observation_artifact=observation,
            surrogate_artifact=surrogate,
        )


def run_first_order(
    config: Mapping[str, Any],
    bundle: DatasetBundle,
    scorer: RawTextScorer,
    *,
    run_dir: str | Path,
    cache_path: str | Path,
    overwrite: bool = False,
    command: str | None = None,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """Run one first-order method over a dataset and optionally evaluate it."""

    resolved = resolve_config(config)
    resolved["method"] = normalize_first_order_method(
        str(resolved.get("method", ""))
    )
    resolved["dataset"] = {
        **dict(resolved.get("dataset", {})),
        "name": bundle.dataset_name,
        "split": bundle.split,
        "verbalizers": list(bundle.verbalizers),
    }
    resolved["prompt"] = build_classification_prompt(
        dataset_name=bundle.dataset_name,
        verbalizers=bundle.verbalizers,
        prompt_config=dict(resolved.get("prompt", {})),
    ).to_config()
    configure_task = getattr(scorer, "configure_task", None)
    if callable(configure_task):
        configure_task(
            verbalizers=bundle.verbalizers,
            dataset_name=bundle.dataset_name,
            prompt_config=resolved["prompt"],
        )
    store = ResultStore(
        run_dir,
        resolved,
        output_level=str(resolved["output_level"]),
        command=command,
        overwrite=overwrite,
        required_artifacts=("observation", "surrogate"),
    )
    oracle = ValueOracle(
        scorer,
        cache_path,
        model_fingerprint=scientific_config(
            {"model": dict(resolved.get("model", {}))}
        )["model"],
        batch_size=int(resolved["batch_size"]),
    )
    explainer = FirstOrderExplainer(
        resolved,
        scorer,
        oracle,
        bundle.verbalizers,
    )
    try:
        for sample in tqdm(
            bundle.samples,
            desc=resolved["method"],
            dynamic_ncols=True,
        ):
            if not overwrite and store.sample_complete(sample.sample_id):
                if sample.sample_id not in store.completed_ids:
                    store.completed_ids.append(sample.sample_id)
                continue
            try:
                store.write_sample(explainer.explain(sample))
            except FirstOrderSampleExcludedError as error:
                store.record_failure(sample.sample_id, error, skipped=True)
                if bool(resolved["fail_fast"]):
                    raise
            except Exception as error:
                store.record_failure(sample.sample_id, error, skipped=False)
                if bool(resolved["fail_fast"]):
                    raise
            store.write_status("running", selected_count=len(bundle.samples))
    finally:
        oracle.close()
    if evaluate and store.completed_ids:
        evaluate_run(
            store.run_dir,
            bundle,
            scorer,
            target=str(resolved["target_mode"]),
            eval_granularity=str(resolved["eval_granularity"]),
            q_values=[int(value) for value in resolved["eval_q_values"]],
        )
    return store.finish(len(bundle.samples))


__all__ = [
    "FIRST_ORDER_METHODS",
    "FirstOrderExplainer",
    "FirstOrderSampleExcludedError",
    "normalize_first_order_method",
    "run_first_order",
]
