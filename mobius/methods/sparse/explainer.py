"""End-to-end modular sparse interaction attribution runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from mobius.core.config import resolve_config, scientific_config
from mobius.core.artifacts import (
    normalize_observation_artifact,
    normalize_surrogate_artifact,
)
from mobius.core.results import ResultStore, canonical_digest
from mobius.core.runtime import counter_delta
from mobius.core.schema import AttributionResult, DatasetBundle, TextSample
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

from .estimator import SparseModel, fit_sparse_model
from .hierarchy import choose_candidates, normalize_hierarchy
from .projector import normalize_projector, project_nodes
from .sampler import normalize_sampler, sample_masks
from .basis import normalize_basis, term_players


METHOD_NAME = "sparse_mobius"


class SampleExcludedError(ValueError):
    """Signal a deliberate feature-count or truncation exclusion."""


def _logical_key(
    method: str,
    sample_id: str,
    players: Sequence[int],
    mask: int,
    category: str,
) -> str:
    """Build a stable method-local logical query identity."""

    return stable_digest(
        {
            "method": method,
            "sample_id": str(sample_id),
            "players": [int(value) for value in players],
            "mask": int(mask),
            "category": str(category),
            "operator": "keep",
        }
    )


def _ranking(
    active_chunk_ids: Sequence[int],
    active_scores: Sequence[float],
    all_chunk_ids: Sequence[int],
) -> list[int]:
    """Rank active players by descending signed score and append inactive chunks."""

    active = [
        (int(chunk_id), float(active_scores[player]))
        for player, chunk_id in enumerate(active_chunk_ids)
    ]
    ranked = [
        chunk_id
        for chunk_id, _ in sorted(active, key=lambda item: (-item[1], item[0]))
    ]
    active_set = set(ranked)
    return ranked + [
        int(chunk_id) for chunk_id in all_chunk_ids if int(chunk_id) not in active_set
    ]


class SparseMobiusExplainer:
    """Explain samples using configurable sampler, basis, hierarchy, and projector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        scorer: RawTextScorer,
        oracle: ValueOracle,
        verbalizers: Sequence[str],
    ) -> None:
        """Resolve component configuration and retain the shared score oracle."""

        self.config = resolve_config(config)
        self.scorer = scorer
        self.oracle = oracle
        self.verbalizers = [str(value) for value in verbalizers]
        self.basis = normalize_basis(str(self.config["basis"]))
        self.hierarchy = normalize_hierarchy(str(self.config["hierarchy"]))
        self.projector = normalize_projector(str(self.config["projector"]))
        self.sampler_config = dict(self.config.get("sampler", {}))
        self.sampler = normalize_sampler(str(self.sampler_config.get("name")))
        self.estimator_config = {
            **dict(self.config.get("fit", {})),
            **dict(self.config.get("estimator", {})),
        }

    def _score_masks(
        self,
        sample_id: str,
        game: CoalitionGame,
        masks: Sequence[int],
        *,
        ledger: QueryLedger,
        category: str,
    ) -> np.ndarray:
        """Score coalition masks through the global cache and local ledger."""

        texts = game.texts(masks)
        logical_keys = [
            _logical_key(
                METHOD_NAME,
                sample_id,
                game.player_to_chunk_id,
                mask,
                category,
            )
            for mask in masks
        ]
        return self.oracle.score_texts(
            texts,
            logical_keys=logical_keys,
            ledger=ledger,
            category=category,
        )

    def explain(self, sample: TextSample) -> AttributionResult:
        """Run attribution, sparse recovery, projection, and artifact export."""

        started = time.perf_counter()
        counters_before = self.scorer.snapshot_counters()
        value_function = normalize_value_function(self.config["value_function"])
        target_mode = effective_target_mode(
            value_function,
            str(self.config["target_mode"]),
        )
        setup_ledger = QueryLedger(f"{METHOD_NAME}/{sample.sample_id}/setup")
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
        full_probabilities = probabilities(full_scores.reshape(1, -1))[0]
        predicted_label = int(np.argmax(full_scores))
        target_label = predicted_label if target_mode == "predicted" else int(sample.label)
        target_label_text = self.verbalizers[target_label]

        chunking = build_chunks(
            sample.text,
            str(self.config["chunker"]),
            getattr(self.scorer, "tokenizer", None),
            adaptive_profile=str(self.config["adaptive_profile"]),
            adaptive_overrides=self.config.get("adaptive_overrides"),
        )
        if str(self.config["chunker"]) == "token" and chunking.fallback_used:
            raise RuntimeError("Token chunking requires tokenizer offset mappings.")
        truncation = visible_text_span(
            getattr(self.scorer, "tokenizer", None),
            sample.text,
            target_label_text,
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
            raise SampleExcludedError("No explanation chunks remain after truncation.")
        minimum = int(self.config["min_features"])
        maximum = self.config.get("max_features")
        if len(players) < minimum:
            raise SampleExcludedError(
                f"active feature count {len(players)} is below min_features={minimum}"
            )
        if maximum is not None and len(players) > int(maximum):
            raise SampleExcludedError(
                f"active feature count {len(players)} exceeds max_features={maximum}"
            )

        attribution_counters_before = self.scorer.snapshot_counters()
        game = CoalitionGame(chunking.chunks, players)
        sampling = sample_masks(
            game.n_players,
            int(self.config["budget"]),
            seed=int(self.config["seed"]),
            config=self.sampler_config,
        )
        ledger = QueryLedger(
            f"{METHOD_NAME}/{sample.sample_id}/{self.config['budget']}/{self.config['seed']}"
        )
        training_scores = self._score_masks(
            sample.sample_id,
            game,
            sampling.masks,
            ledger=ledger,
            category="training",
        )
        training_values = attribution_values(
            training_scores,
            target_class=target_label,
            value_function=value_function,
        )
        candidates, hierarchy_diagnostics, _ = choose_candidates(
            self.hierarchy,
            sampling.masks,
            training_values,
            n_features=game.n_players,
            max_degree=int(self.config["max_degree"]),
            basis=self.basis,
            estimator_config=self.estimator_config,
            random_state=int(self.config["seed"]),
        )
        model = fit_sparse_model(
            sampling.masks,
            training_values,
            n_features=game.n_players,
            terms=candidates,
            basis=self.basis,
            max_degree=int(self.config["max_degree"]),
            config=self.estimator_config,
            random_state=int(self.config["seed"]),
            hierarchy_policy=self.hierarchy,
        )
        active_scores = project_nodes(model, self.projector)
        node_scores = [0.0] * len(chunking.chunks)
        for player, chunk_id in enumerate(players):
            node_scores[int(chunk_id)] = float(active_scores[player])
        ranking = _ranking(
            players,
            active_scores,
            [chunk.chunk_id for chunk in chunking.chunks],
        )
        selected = ranking[: min(int(self.config["k"]), len(players))]

        model_payload = model.to_dict()
        for edge in model_payload["hyperedges"]:
            edge["chunk_ids"] = [
                int(players[int(player)]) for player in edge["players"]
            ]
        observation_artifact = normalize_observation_artifact(
            {
                "sample_id": sample.sample_id,
                "method": METHOD_NAME,
                "n_features": game.n_players,
                "keep_masks": sampling.masks,
                "label_scores": training_scores,
                "attribution_values": training_values,
            }
        )
        predictor_terms = []
        coefficient_by_term = {
            int(term): float(coefficient)
            for term, coefficient in zip(model.terms, model.coefficients)
        }
        for term in model.refit_support:
            predictor_terms.append(
                {
                    "players": list(term_players(int(term))),
                    "coefficient": coefficient_by_term[int(term)],
                }
            )
        surrogate_artifact = normalize_surrogate_artifact(
            {
                "sample_id": sample.sample_id,
                "method": METHOD_NAME,
                "n_features": game.n_players,
                "player_to_chunk_id": players,
                "observation_file": f"../observations/{sample.sample_id}.npz",
                "observation_digest": observation_artifact["digest"],
                "value_function": value_function,
                "target_mode": target_mode,
                "target_label": target_label,
                "predictor": {
                    "type": "sparse_polynomial",
                    "basis": self.basis,
                    "intercept": float(model.intercept),
                    "terms": predictor_terms,
                },
                "candidate_definition": model_payload["candidate_definition"],
                "support": {
                    "selection": model_payload["selection_support"],
                    "hierarchy": model_payload["hierarchy_support"],
                    "refit": model_payload["refit_support"],
                    "counts": model_payload["support_counts"],
                },
                "fit_config": dict(self.estimator_config),
                "fit_diagnostics": dict(model.diagnostics),
            }
        )
        elapsed = time.perf_counter() - started
        counters_after = self.scorer.snapshot_counters()
        attribution_counter_delta = counter_delta(
            attribution_counters_before,
            counters_after,
        )
        attribution_cost = {
            **ledger.to_dict(),
            "setup_physical_values_scored": setup_ledger.physical_values_scored,
            "interaction_verification_queries": 0,
            "model_forward_calls": int(
                attribution_counter_delta.get("model_forward_calls", 0)
            ),
            "batch_calls": int(attribution_counter_delta.get("batch_calls", 0)),
            "batch_rows": int(attribution_counter_delta.get("batch_rows", 0)),
            "model_counter_delta": attribution_counter_delta,
            "total_counter_delta_including_setup": counter_delta(
                counters_before,
                counters_after,
            ),
            "elapsed_seconds": elapsed,
        }
        method_summary = {
            "method": METHOD_NAME,
            "basis": self.basis,
            "deletion_mobius_definition": "g(D)=f(N\\D)",
            "hierarchy": {
                "candidate_policy": hierarchy_diagnostics,
                "support_policy": dict(model.diagnostics.get("hierarchy", {})),
            },
            "sampler": sampling.diagnostics,
            "projector": self.projector,
            "model": model_payload,
            "chunking": chunking.diagnostics,
            "truncation": {
                **truncation,
                "active_chunk_ids": players,
                "active_chunk_count": len(players),
            },
            "player_to_chunk_id": players,
            "value_function": value_function,
            "target_mode": target_mode,
            "full_label_scores": [float(value) for value in full_scores],
            "full_label_probabilities": [
                float(value) for value in full_probabilities
            ],
            "selected_text": compose_text(chunking.chunks, selected),
            "observation_digest": observation_artifact["digest"],
            "surrogate_digest": surrogate_artifact["digest"],
        }
        diagnostics = {
            "masks": [int(mask) for mask in sampling.masks],
            "values": [float(value) for value in training_values],
            "label_scores": [
                [float(value) for value in row] for row in training_scores
            ],
            "setup_ledger": setup_ledger.to_dict(),
            "query_ledger": ledger.to_dict(),
            "observation_digest": canonical_digest(
                {
                    "masks": sampling.masks,
                    "values": [float(value) for value in training_values],
                }
            ),
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
            method_summary=method_summary,
            diagnostics=diagnostics,
            observation_artifact=observation_artifact,
            surrogate_artifact=surrogate_artifact,
        )


def run_sparse_mobius(
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
    """Run schema-v2 attribution over a dataset and optionally evaluate it."""

    resolved = resolve_config(config)
    resolved["method"] = METHOD_NAME
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
    explainer = SparseMobiusExplainer(
        resolved,
        scorer,
        oracle,
        bundle.verbalizers,
    )
    try:
        for sample in tqdm(bundle.samples, desc=METHOD_NAME, dynamic_ncols=True):
            if not overwrite and store.sample_complete(sample.sample_id):
                if sample.sample_id not in store.completed_ids:
                    store.completed_ids.append(sample.sample_id)
                continue
            try:
                store.write_sample(explainer.explain(sample))
            except SampleExcludedError as error:
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
        from mobius.evaluation.evaluator import evaluate_run

        evaluate_run(
            store.run_dir,
            bundle,
            scorer,
            target=str(resolved["target_mode"]),
            eval_granularity=str(resolved["eval_granularity"]),
            q_values=[int(value) for value in resolved["eval_q_values"]],
        )
    return store.finish(len(bundle.samples))
