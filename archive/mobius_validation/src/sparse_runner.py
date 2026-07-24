from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from lima_llm.attribution_text import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    active_chunk_ids_from_visible_span,
    compose_coalition_text,
    prompt_visible_text_span_after_left_truncation,
)
from lima_llm.attribution_values import (
    attribution_values_from_label_scores,
    normalize_attribution_value_function,
    probabilities_from_label_scores,
)
from lima_llm.chunking.explanation import (
    build_explanation_chunks,
    normalize_explanation_chunker,
)
from lima_llm.chunking.utils import compose_text_from_chunk_ids
from lima_llm.eval.evaluate import evaluate_saved_explanations
from lima_llm.eval.metrics import EMPTY_PERTURBATION_TEXT
from lima_llm.eval.units import normalize_eval_granularity
from lima_llm.pipeline.io import rebuild_summary_csv, save_explanation
from lima_llm.pipeline.resume import is_sample_completed, sample_output_paths
from lima_llm.types import ExplanationResult, ScoreComponents, ScoreTrace, TextChunk
from lima_llm.utils import atomic_write_json, ensure_dir

from .designs import term_players
from .methods.sparse_mobius import fit_sparse_mobius, model_node_scores
from .models import BackboneLabelScorer
from .query_ledger import QueryLedger
from .subset_enumeration import sample_deletion_mobius_masks
from .utils import environment_snapshot
from .value_oracle import ValueOracle


METHOD_NAME = "sparse_mobius"
ORIENTATION = "deletion_mobius"


class SampleExcludedError(ValueError):
    """A deliberate sample exclusion, distinct from an implementation failure."""


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _counter_delta(
    before: Mapping[str, float | int],
    after: Mapping[str, float | int],
) -> Dict[str, float | int]:
    output: Dict[str, float | int] = {}
    for key in set(before).union(after):
        previous = before.get(key, 0)
        current = after.get(key, 0)
        if isinstance(previous, float) or isinstance(current, float):
            output[str(key)] = float(current) - float(previous)
        else:
            output[str(key)] = int(current) - int(previous)
    return output


def _effective_target_mode(value_function: str, requested: str) -> str:
    if value_function in {"predicted_probability", "predicted_class_margin"}:
        return "predicted"
    return str(requested)


def _rank_scores(scores: Sequence[float], k: int) -> tuple[list[int], list[int]]:
    ranking = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return ranking, ranking[: min(max(0, int(k)), len(ranking))]


def _build_trace(selected: Sequence[int], scores: Sequence[float]) -> list[ScoreTrace]:
    total = 0.0
    zero = ScoreComponents(0.0, 0.0, 0.0, 0.0)
    trace = []
    for step, chunk_id in enumerate(selected, start=1):
        gain = float(scores[int(chunk_id)])
        total += gain
        trace.append(
            ScoreTrace(
                step=int(step),
                selected_chunk_id=int(chunk_id),
                marginal_gain=gain,
                total_score=float(total),
                components=zero,
            )
        )
    return trace


class SparseMobiusCoalitionGame:
    def __init__(
        self,
        *,
        sample_id: str,
        units: Sequence[TextChunk],
        player_to_chunk_id: Sequence[int],
        oracle: ValueOracle,
        target_class: int,
        value_function: str,
        ledger: QueryLedger,
    ) -> None:
        self.sample_id = str(sample_id)
        self.units = list(units)
        self.player_to_chunk_id = [int(value) for value in player_to_chunk_id]
        self.oracle = oracle
        self.target_class = int(target_class)
        self.value_function = normalize_attribution_value_function(value_function)
        self.ledger = ledger
        self.row_counts: Dict[str, int] = {}
        self.unique_masks: Dict[str, set[int]] = {}
        self.unique_texts: Dict[str, set[str]] = {}

    @property
    def n_players(self) -> int:
        return len(self.player_to_chunk_id)

    def text_for_mask(self, mask: int) -> str:
        coalition = [bool(int(mask) & (1 << index)) for index in range(self.n_players)]
        return compose_coalition_text(
            units=self.units,
            player_to_chunk_id=self.player_to_chunk_id,
            coalition_row=coalition,
        )

    def values_for_masks(
        self,
        masks: Sequence[int],
        *,
        category: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_masks = [int(mask) for mask in masks]
        texts = [self.text_for_mask(mask) for mask in normalized_masks]
        category_name = str(category)
        self.row_counts[category_name] = self.row_counts.get(category_name, 0) + len(texts)
        self.unique_masks.setdefault(category_name, set()).update(normalized_masks)
        self.unique_texts.setdefault(category_name, set()).update(texts)
        logical_keys = [
            _json_digest(
                {
                    "method": METHOD_NAME,
                    "sample_id": self.sample_id,
                    "players": self.player_to_chunk_id,
                    "mask": int(mask),
                    "operator": "delete",
                }
            )
            for mask in normalized_masks
        ]
        label_scores = self.oracle.score_texts(
            texts,
            logical_keys=logical_keys,
            ledger=self.ledger,
            category=category_name,
        )
        values = attribution_values_from_label_scores(
            label_scores,
            target_class=self.target_class,
            value_function=self.value_function,
        )
        return values, label_scores

    def stats(self) -> Dict[str, object]:
        return {
            "row_count": int(sum(self.row_counts.values())),
            "row_count_by_category": dict(sorted(self.row_counts.items())),
            "unique_mask_count_by_category": {
                key: int(len(value)) for key, value in sorted(self.unique_masks.items())
            },
            "unique_text_count_by_category": {
                key: int(len(value)) for key, value in sorted(self.unique_texts.items())
            },
            "value_function": self.value_function,
        }


def _targeted_deletion_verification(
    *,
    model,
    game: SparseMobiusCoalitionGame,
    top_k: int,
) -> Dict[str, object] | None:
    ranked = sorted(
        model.coefficient_dict().items(),
        key=lambda item: (-abs(float(item[1])), int(item[0])),
    )[: max(0, int(top_k))]
    if not ranked:
        return None

    full = (1 << game.n_players) - 1
    rows = []
    for term, estimate in ranked:
        players = term_players(int(term))
        masks = []
        signs = []
        for order in range(len(players) + 1):
            for subset in combinations(players, order):
                deleted = 0
                for player in subset:
                    deleted |= 1 << int(player)
                masks.append(full & ~deleted)
                signs.append((-1.0) ** (len(players) - order))
        true_values, _ = game.values_for_masks(masks, category="interaction_verification")
        truth = float(np.dot(np.asarray(signs, dtype=np.float64), true_values))
        estimate_value = float(estimate)
        rows.append(
            {
                "term": int(term),
                "players": [int(player) for player in players],
                "chunk_ids": [int(game.player_to_chunk_id[player]) for player in players],
                "degree": int(len(players)),
                "estimated_coefficient": estimate_value,
                "true_coefficient": truth,
                "absolute_error": float(abs(estimate_value - truth)),
                "sign_match": bool(np.sign(estimate_value) == np.sign(truth)),
                "masks": [int(mask) for mask in masks],
                "values": [float(value) for value in true_values],
            }
        )
    return {
        "definition": "g(D)=f(N\\D); coefficients are the Mobius transform of g",
        "orientation": ORIENTATION,
        "top": rows,
        "sign_accuracy": float(np.mean([row["sign_match"] for row in rows])),
        "mean_absolute_error": float(np.mean([row["absolute_error"] for row in rows])),
    }


def _truncation_metadata(
    *,
    tokenizer,
    text: str,
    label_text: str,
    max_length: int,
) -> Dict[str, Any]:
    if tokenizer is None:
        return {
            "visible_start_char": 0,
            "visible_end_char": len(text),
            "visible_char_count": len(text),
            "dropped_prompt_token_count": 0,
            "strategy": "no_tokenizer_assume_full_visibility",
        }
    return {
        **prompt_visible_text_span_after_left_truncation(
            tokenizer=tokenizer,
            text=text,
            label_text=label_text,
            max_length=int(max_length),
        ),
        "strategy": "prompt_offset_left_truncation",
    }


def explain_sample(
    *,
    sample,
    bundle,
    backbone,
    oracle: ValueOracle,
    config: Mapping[str, Any],
    observations_dir: Path,
) -> ExplanationResult:
    started = time.time()
    timing: Dict[str, float] = {}
    counter_before = backbone.snapshot_counters()
    value_function = normalize_attribution_value_function(
        config.get("value_function", "predicted_class_margin")
    )
    requested_target_mode = str(config.get("target_mode", "predicted"))
    target_mode = _effective_target_mode(value_function, requested_target_mode)
    tokenizer = getattr(backbone, "tokenizer", None)

    setup_ledger = QueryLedger(f"setup/{sample.sample_id}")
    full_label_scores = oracle.score_texts(
        [sample.text],
        logical_keys=[_json_digest({"sample_id": sample.sample_id, "setup": "full_input"})],
        ledger=setup_ledger,
        category="setup",
    )[0]
    full_probabilities = probabilities_from_label_scores(full_label_scores.reshape(1, -1))[0]
    target_class = int(np.argmax(full_label_scores)) if target_mode == "predicted" else int(sample.label)
    target_label_text = str(bundle.verbalizers[target_class])

    chunk_started = time.time()
    chunker = normalize_explanation_chunker(config.get("chunker", "word"))
    chunking = build_explanation_chunks(
        text=sample.text,
        chunker=chunker,
        tokenizer=tokenizer,
        adaptive_profile=str(config.get("adaptive_profile", "balanced")),
        adaptive_overrides=config.get("adaptive_overrides"),
    )
    units = list(chunking.chunks)
    if chunker == "token" and chunking.fallback_used:
        raise RuntimeError(
            "Sparse Mobius token chunker requires tokenizer offset_mapping support."
        )
    truncation = _truncation_metadata(
        tokenizer=tokenizer,
        text=sample.text,
        label_text=target_label_text,
        max_length=int(config.get("max_length", 2048)),
    )
    player_to_chunk_id = active_chunk_ids_from_visible_span(
        units,
        int(truncation["visible_start_char"]),
        int(truncation["visible_end_char"]),
    )
    timing["chunk_and_truncation_seconds"] = float(time.time() - chunk_started)
    if not player_to_chunk_id:
        raise SampleExcludedError("No explanation chunks remain after prompt left truncation")
    minimum = int(config.get("min_features", 1))
    maximum = config.get("max_features")
    if len(player_to_chunk_id) < minimum:
        raise SampleExcludedError(f"active feature count is below min_features={minimum}")
    if maximum is not None and len(player_to_chunk_id) > int(maximum):
        raise SampleExcludedError(f"active feature count exceeds max_features={int(maximum)}")

    sampler = dict(config.get("sampler", {}))
    sampling = sample_deletion_mobius_masks(
        len(player_to_chunk_id),
        int(config.get("budget", 512)),
        seed=int(config.get("seed", 42)),
        global_fraction=float(sampler.get("global_fraction", 0.5)),
        near_full_fraction=float(sampler.get("near_full_fraction", 0.3)),
        fixed_cardinality_fraction=float(sampler.get("fixed_cardinality_fraction", 0.2)),
        near_full_deletions=sampler.get("near_full_deletions", [1, 2, 3, 5]),
        fixed_keep_fractions=sampler.get("fixed_keep_fractions", [0.25, 0.5, 0.75]),
        include_empty_full=bool(sampler.get("include_empty_full", True)),
    )
    ledger = QueryLedger(
        f"{METHOD_NAME}/{sample.sample_id}/{config.get('budget', 512)}/"
        f"{config.get('seed', 42)}/{value_function}"
    )
    game = SparseMobiusCoalitionGame(
        sample_id=sample.sample_id,
        units=units,
        player_to_chunk_id=player_to_chunk_id,
        oracle=oracle,
        target_class=target_class,
        value_function=value_function,
        ledger=ledger,
    )
    query_started = time.time()
    attribution_counter_before = backbone.snapshot_counters()
    train_values, train_label_scores = game.values_for_masks(sampling.masks, category="training")
    attribution_counter_after = backbone.snapshot_counters()
    attribution_forward_delta = _counter_delta(
        attribution_counter_before,
        attribution_counter_after,
    )
    training_physical_values_scored = int(ledger.physical_forwards_caused)
    timing["attribution_query_seconds"] = float(time.time() - query_started)

    fit = dict(config.get("fit", {}))
    fit_started = time.time()
    model = fit_sparse_mobius(
        masks=sampling.masks,
        values=train_values,
        n_features=len(player_to_chunk_id),
        orientation=ORIENTATION,
        max_degree=int(config.get("max_degree", 2)),
        alphas=[float(value) for value in fit.get("alphas", [1e-4, 1e-3, 1e-2, 1e-1])],
        l1_ratios=[float(value) for value in fit.get("l1_ratios", [1.0])],
        cv_folds=int(fit.get("cv_folds", 3)),
        random_state=int(config.get("seed", 42)),
        max_design_mb=float(fit.get("max_design_mb", 2048)),
    )
    timing["sparse_fit_seconds"] = float(time.time() - fit_started)

    active_scores = model_node_scores(model)
    chunk_scores = [0.0] * len(units)
    for player, chunk_id in enumerate(player_to_chunk_id):
        chunk_scores[int(chunk_id)] = float(active_scores[player])
    ranking, selected = _rank_scores(chunk_scores, int(config.get("k", 8)))

    verify_started = time.time()
    verification = _targeted_deletion_verification(
        model=model,
        game=game,
        top_k=int(config.get("targeted_top_k", 0)),
    )
    timing["targeted_verification_seconds"] = float(time.time() - verify_started)

    model_payload = model.to_dict()
    for hyperedge in model_payload.get("hyperedges", []):
        hyperedge["chunk_ids"] = [
            int(player_to_chunk_id[int(player)]) for player in hyperedge.get("players", [])
        ]
    observation_payload = {
        "method": METHOD_NAME,
        "sample_id": sample.sample_id,
        "n_players": int(len(player_to_chunk_id)),
        "player_to_chunk_id": [int(value) for value in player_to_chunk_id],
        "value_function": value_function,
        "target_class": int(target_class),
        "masks": [int(mask) for mask in sampling.masks],
        "values": [float(value) for value in train_values],
        "label_scores": [[float(value) for value in row] for row in train_label_scores],
        "sampling_diagnostics": sampling.diagnostics,
    }
    observation_path = observations_dir / f"{sample.sample_id}.json"
    atomic_write_json(observation_path, observation_payload)

    counter_after = backbone.snapshot_counters()
    elapsed = float(time.time() - started)
    forward_delta = _counter_delta(counter_before, counter_after)
    game_stats = game.stats()
    inactive = [unit.chunk_id for unit in units if unit.chunk_id not in set(player_to_chunk_id)]
    query_accounting = {
        "logical_attribution_queries": int(game_stats["row_count_by_category"].get("training", 0)),
        "logical_unique_attribution_queries": int(
            game_stats["unique_mask_count_by_category"].get("training", 0)
        ),
        "unique_attribution_texts": int(
            game_stats["unique_text_count_by_category"].get("training", 0)
        ),
        "physical_values_scored": int(training_physical_values_scored),
        "total_physical_values_scored_including_verification": int(
            ledger.physical_forwards_caused
        ),
        "model_forward_calls": int(attribution_forward_delta.get("model_forward_calls", 0)),
        "batch_calls": int(attribution_forward_delta.get("batch_calls", 0)),
        "batch_rows": int(attribution_forward_delta.get("batch_rows", 0)),
        "attribution_forward_counters_delta": attribution_forward_delta,
        "interaction_verification_queries": int(ledger.category_count("interaction_verification")),
        "elapsed_seconds": elapsed,
    }
    metadata = {
        "sparse_mobius_method": METHOD_NAME,
        "orientation": ORIENTATION,
        "deletion_mobius_definition": "g(D)=f(N\\D); fit the Mobius transform of g",
        "or_interaction_relation": "non-empty coefficients differ from OR interaction only by sign convention",
        "budget": int(config.get("budget", 512)),
        "seed": int(config.get("seed", 42)),
        "max_degree": int(config.get("max_degree", 2)),
        "value_function": value_function,
        "target_mode": target_mode,
        "target_mode_requested": requested_target_mode,
        "target_label": int(target_class),
        "target_label_text": target_label_text,
        "full_label_scores": [float(value) for value in full_label_scores],
        "full_label_probabilities": [float(value) for value in full_probabilities],
        "chunker": chunker,
        "eval_granularity": normalize_eval_granularity(config.get("eval_granularity", "token")),
        "chunk_diagnostics": dict(chunking.diagnostics),
        "explain_chunk_strategy": str(
            chunking.diagnostics.get("chunk_strategy", chunking.segmentation_strategy)
        ),
        "explain_tokenizer_fallback_used": bool(chunking.fallback_used),
        "segmentation_strategy": str(chunking.segmentation_strategy),
        "tokenizer_fallback_used": bool(chunking.fallback_used),
        "truncation": {
            **truncation,
            "active_chunk_count": int(len(player_to_chunk_id)),
            "inactive_chunk_count": int(len(inactive)),
            "active_chunk_ids": [int(value) for value in player_to_chunk_id],
            "inactive_chunk_ids": [int(value) for value in inactive],
        },
        "player_to_chunk_id": [int(value) for value in player_to_chunk_id],
        "sampling_diagnostics": sampling.diagnostics,
        "model": model_payload,
        "interaction_verification": verification,
        "observation_table": str(observation_path),
        "observation_table_digest": _json_digest(observation_payload),
        "game_stats": game_stats,
        "setup_query_ledger": setup_ledger.to_dict(),
        "query_ledger": ledger.to_dict(),
        "query_accounting": query_accounting,
        "forward_counters_delta": forward_delta,
        "explain_timing_breakdown": timing,
        "elapsed_seconds": elapsed,
    }
    selected_text = compose_text_from_chunk_ids(units, selected)
    return ExplanationResult(
        explain_method=METHOD_NAME,
        sample_id=str(sample.sample_id),
        dataset=str(bundle.dataset_name),
        split=str(bundle.split),
        label=int(sample.label),
        label_text=sample.label_text,
        text=str(sample.text),
        chunks=units,
        chunk_ranking=ranking,
        chunk_scores=[float(value) for value in chunk_scores],
        selected_chunk_ids=selected,
        selected_text=selected_text,
        scores={
            "total": float(sum(chunk_scores[index] for index in selected)),
            "confidence": 0.0,
            "effectiveness": 0.0,
            "consistency": 0.0,
            "collaboration": 0.0,
            "target_probability": float(full_probabilities[target_class]),
            "label_probabilities": [float(value) for value in full_probabilities],
            "attribution_value": float(
                attribution_values_from_label_scores(
                    full_label_scores.reshape(1, -1),
                    target_class=target_class,
                    value_function=value_function,
                )[0]
            ),
        },
        trace=_build_trace(selected, chunk_scores),
        metadata=metadata,
    )


def comparison_contract(config: Mapping[str, Any], bundle) -> Dict[str, object]:
    model = dict(config.get("model", {}))
    value_function = normalize_attribution_value_function(
        config.get("value_function", "predicted_class_margin")
    )
    target_mode = _effective_target_mode(
        value_function,
        str(config.get("target_mode", "predicted")),
    )
    return {
        "dataset": str(bundle.dataset_name),
        "split": str(bundle.split),
        "model_path": str(model.get("model_path", model.get("type", "mock"))),
        "max_length": int(config.get("max_length", model.get("max_length", 2048))),
        "prompt_template": f"{PROMPT_PREFIX}{{text}}{PROMPT_SUFFIX}",
        "chunker": normalize_explanation_chunker(config.get("chunker", "word")),
        "adaptive_profile": str(config.get("adaptive_profile", "balanced")),
        "adaptive_overrides": config.get("adaptive_overrides"),
        "mask_operator": "delete",
        "empty_perturbation_text": EMPTY_PERTURBATION_TEXT,
        "target_mode": target_mode,
        "value_function": value_function,
        "eval_granularity": normalize_eval_granularity(config.get("eval_granularity", "token")),
        "eval_q_values": [int(value) for value in config.get("eval_q_values", [1, 5, 10, 20, 50])],
        "verbalizers": [str(value) for value in bundle.verbalizers],
    }


def run_sparse_mobius(
    *,
    config: Mapping[str, Any],
    bundle,
    backbone,
    output_root: Path,
    overwrite: bool = False,
    evaluate: bool = True,
) -> Dict[str, object]:
    output_root = Path(output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "samples")
    observations_dir = output_root / "observations"
    failures_dir = output_root / "failures"
    ensure_dir(observations_dir)
    ensure_dir(failures_dir)

    resolved_config = dict(config)
    resolved_config["method"] = METHOD_NAME
    resolved_config["comparison_contract"] = comparison_contract(config, bundle)
    resolved_config["experiment_fingerprint"] = _json_digest(resolved_config)
    existing_config_path = output_root / "run_config.json"
    if existing_config_path.exists() and not overwrite:
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing_config.get("experiment_fingerprint")
        has_samples = any((output_root / "samples").glob("*.json"))
        if has_samples and existing_fingerprint != resolved_config["experiment_fingerprint"]:
            raise ValueError(
                "Output root contains samples from a different Sparse Mobius configuration. "
                "Choose another --output-root or use --overwrite."
            )
    atomic_write_json(output_root / "run_config.json", resolved_config)
    atomic_write_json(output_root / "environment.json", environment_snapshot())

    oracle = ValueOracle(
        scorer=BackboneLabelScorer(backbone, bundle.verbalizers),
        cache_path=output_root / "cache" / "value_oracle.sqlite3",
        model_fingerprint=json.dumps(dict(config.get("model", {})), sort_keys=True),
        prompt_version="lima_hf_label_v1",
        batch_size=int(config.get("batch_size", 16)),
    )
    completed_ids: list[str] = []
    failures: list[Dict[str, object]] = []
    skipped: list[Dict[str, object]] = []
    started = time.time()

    def write_manifest(status: str, oracle_counters: Mapping[str, object] | None = None) -> None:
        atomic_write_json(
            output_root / "manifest.json",
            {
                "status": str(status),
                "method": METHOD_NAME,
                "selected_sample_count": int(len(bundle.samples)),
                "completed_sample_count": int(len(completed_ids)),
                "failed_sample_count": int(len(failures)),
                "skipped_sample_count": int(len(skipped)),
                "completed_sample_ids": list(completed_ids),
                "failures": list(failures),
                "skipped": list(skipped),
                "value_oracle_counters": dict(oracle_counters or {}),
                "elapsed_seconds": float(time.time() - started),
            },
        )

    write_manifest("running")
    try:
        for sample in tqdm(bundle.samples, desc=METHOD_NAME, dynamic_ncols=True):
            paths = sample_output_paths(output_root, sample.sample_id)
            if not overwrite and is_sample_completed(paths, mode="strict"):
                completed_ids.append(str(sample.sample_id))
                write_manifest("running")
                continue
            try:
                result = explain_sample(
                    sample=sample,
                    bundle=bundle,
                    backbone=backbone,
                    oracle=oracle,
                    config=config,
                    observations_dir=observations_dir,
                )
                save_explanation(result, output_root)
                completed_ids.append(str(sample.sample_id))
            except SampleExcludedError as exc:
                payload = {
                    "sample_id": str(sample.sample_id),
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                }
                skipped.append(payload)
                atomic_write_json(failures_dir / f"{sample.sample_id}.json", payload)
                if bool(config.get("fail_fast", False)):
                    raise
            except Exception as exc:
                payload = {
                    "sample_id": str(sample.sample_id),
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                }
                failures.append(payload)
                atomic_write_json(failures_dir / f"{sample.sample_id}.json", payload)
                if bool(config.get("fail_fast", False)):
                    raise
            write_manifest("running")
    finally:
        oracle_counters = oracle.snapshot_counters()
        oracle.close()

    summary_path = rebuild_summary_csv(output_root)
    eval_report_path = None
    if evaluate and completed_ids:
        report = evaluate_saved_explanations(
            output_root=output_root,
            bundle=bundle,
            backbone=backbone,
            verbalizers=bundle.verbalizers,
            q_values=[int(value) for value in config.get("eval_q_values", [1, 5, 10, 20, 50])],
            explain_method=METHOD_NAME,
            eval_granularity=normalize_eval_granularity(config.get("eval_granularity", "token")),
        )
        atomic_write_json(output_root / "eval_report.json", report)
        eval_report_path = output_root / "eval_report.json"

    status = "complete" if not failures else "complete_with_failures"
    write_manifest(status, oracle_counters)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["summary_path"] = str(summary_path)
    manifest["eval_report_path"] = str(eval_report_path) if eval_report_path else None
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


__all__ = [
    "METHOD_NAME",
    "ORIENTATION",
    "SampleExcludedError",
    "SparseMobiusCoalitionGame",
    "comparison_contract",
    "explain_sample",
    "run_sparse_mobius",
]
