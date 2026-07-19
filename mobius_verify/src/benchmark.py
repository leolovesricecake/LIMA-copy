from __future__ import annotations

import hashlib
import math
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .attribution_metrics import evaluate_node_ranking, evaluate_surrogate_distributions
from .designs import equal_share_node_scores, fourier_to_presence_coefficients
from .featureization import build_lexical_word_features, validate_feature_reconstruction
from .interaction_verification import verify_top_hyperedges
from .methods.controlled_baselines import GBTSurrogateModel, fit_gbt_surrogate
from .methods.proxyspex_adapter import (
    ProxySPEXSurrogateModel,
    fit_proxyspex_from_observations,
    fit_proxyspex_native,
)
from .methods.sparse_mobius import model_node_scores, ranked_nodes
from .methods.sparse_surrogate import SparseSurrogateModel, fit_sparse_surrogate
from .models import build_text_scorer
from .query_ledger import QueryLedger
from .schema import FeatureSpec, TextRecord
from .subset_enumeration import sample_attribution_masks, sample_evaluation_masks
from .utils import atomic_write_json, ensure_dir, environment_snapshot
from .value_functions import best_competitor, values_from_score_matrix
from .value_oracle import ValueOracle


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(child) for child in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _mask_digest(masks: Sequence[int] | Mapping[str, Sequence[int]]) -> str:
    if isinstance(masks, Mapping):
        payload = {
            str(name): [int(mask) for mask in values]
            for name, values in sorted(masks.items())
        }
    else:
        payload = [int(mask) for mask in masks]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def budget_values(n_features: int, config: Mapping[str, Any]) -> list[int]:
    explicit = config.get("budgets")
    universe = 1 << int(n_features)
    minimum = 2 if universe >= 2 else 1
    if explicit:
        values = [int(value) for value in explicit]
    else:
        denominator = float(max(1, n_features) * math.log2(max(2, n_features)))
        values = [int(round(float(alpha) * denominator)) for alpha in config.get("budget_alphas", [1, 2, 4])]
    return sorted({max(minimum, min(universe, value)) for value in values})


def _method_result_dir(
    root: Path,
    *,
    protocol: str,
    value_type: str,
    method: str,
    task: str,
    sample_id: str,
    budget: int,
    seed: int,
) -> Path:
    return (
        root
        / "protocols"
        / str(protocol)
        / str(value_type)
        / str(method)
        / str(task)
        / str(sample_id)
        / f"budget_{int(budget)}"
        / f"seed_{int(seed)}"
    )


def _fit_controlled_method(
    method: str,
    *,
    masks: Sequence[int],
    values: Sequence[float],
    n_features: int,
    seed: int,
    config: Mapping[str, Any],
):
    fit_cfg = dict(config.get("fit", {}))
    common = {
        "alphas": [float(value) for value in fit_cfg.get("alphas", [1e-4, 1e-3, 1e-2, 1e-1])],
        "l1_ratios": [float(value) for value in fit_cfg.get("l1_ratios", [1.0])],
        "cv_folds": int(fit_cfg.get("cv_folds", 3)),
        "random_state": int(seed),
        "max_design_mb": float(fit_cfg.get("max_design_mb", 2048)),
    }
    if method == "additive_lasso":
        return fit_sparse_surrogate(
            masks=masks,
            values=values,
            n_features=n_features,
            basis="presence_mobius",
            max_degree=1,
            **common,
        )
    if method in {"presence_mobius", "deletion_mobius", "fourier"}:
        return fit_sparse_surrogate(
            masks=masks,
            values=values,
            n_features=n_features,
            basis=method,
            max_degree=int(config.get("max_degree", 2)),
            **common,
        )
    if method == "sklearn_gbt":
        return fit_gbt_surrogate(
            masks=masks,
            values=values,
            n_features=n_features,
            random_state=seed,
            param_grid=config.get("gbt_param_grid"),
            cv_folds=int(fit_cfg.get("cv_folds", 3)),
        )
    if method == "proxyspex_fixed_observations":
        proxy_cfg = dict(config.get("proxyspex", {}))
        return fit_proxyspex_from_observations(
            masks=masks,
            values=values,
            n_features=n_features,
            max_order=int(proxy_cfg.get("max_order", 2)),
            index=str(proxy_cfg.get("index", "FBII")),
            proxy_model=str(proxy_cfg.get("proxy_model", "tree")),
            hpo=bool(proxy_cfg.get("hpo", False)),
            random_state=seed,
        )
    raise ValueError(f"Unsupported controlled method: {method!r}")


def _model_node_scores(model) -> np.ndarray:
    if isinstance(model, SparseSurrogateModel):
        return model_node_scores(model)
    if isinstance(model, GBTSurrogateModel):
        return model.node_scores()
    if isinstance(model, ProxySPEXSurrogateModel):
        return model.node_scores(max_degree=2)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def _model_verification_coefficients(model) -> tuple[str | None, Dict[int, float]]:
    if isinstance(model, SparseSurrogateModel):
        if model.basis == "deletion_mobius":
            return "deletion_mobius", model.coefficient_dict()
        if model.basis == "presence_mobius":
            return "presence_mobius", model.coefficient_dict()
        if model.basis == "fourier":
            _, coefficients = fourier_to_presence_coefficients(
                intercept=model.intercept,
                terms=model.terms,
                coefficients=model.coefficients,
            )
            return "presence_mobius", coefficients
    if isinstance(model, ProxySPEXSurrogateModel):
        return "presence_mobius", model.presence_coefficients(max_degree=2)
    return None, {}


def _model_payload(model) -> Dict[str, object]:
    return _json_safe(model.to_dict())


def _evaluate_and_save(
    *,
    output_dir: Path,
    model,
    method: str,
    protocol: str,
    value_type: str,
    record: TextRecord,
    feature_spec: FeatureSpec,
    target_class: int,
    full_scores: np.ndarray,
    budget: int,
    seed: int,
    train_masks: Sequence[int],
    oracle: ValueOracle,
    method_ledger: QueryLedger,
    evaluation_masks: Mapping[str, Sequence[int]],
    evaluation_values: Mapping[str, Sequence[float]],
    evaluation_scores: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    fit_elapsed_seconds: float,
) -> None:
    evaluation_started = time.time()
    # The value table is prefetched once per protocol after every method has fitted. Replaying
    # these cached requests here gives each method a complete, order-independent logical ledger.
    for distribution_masks in evaluation_masks.values():
        oracle.score_masks(
            feature_spec,
            list(distribution_masks),
            ledger=method_ledger,
            category="evaluation",
            operator=str(config.get("mask_operator", "delete")),
        )
    node_scores = _model_node_scores(model)
    rankings = ranked_nodes(node_scores)
    attribution = evaluate_node_ranking(
        oracle=oracle,
        feature_spec=feature_spec,
        ranking=rankings["positive"],
        target_class=target_class,
        value_type=value_type,
        ledger=method_ledger,
        operator=str(config.get("mask_operator", "delete")),
        fractions=[float(value) for value in config.get("attribution_fractions", [0, 0.1, 0.2, 0.5, 1])],
    )
    random_ranking = list(range(feature_spec.n_features))
    np.random.default_rng(int(seed) + 7919).shuffle(random_ranking)
    random_attribution = evaluate_node_ranking(
        oracle=oracle,
        feature_spec=feature_spec,
        ranking=random_ranking,
        target_class=target_class,
        value_type=value_type,
        ledger=method_ledger,
        operator=str(config.get("mask_operator", "delete")),
        fractions=[float(value) for value in config.get("attribution_fractions", [0, 0.1, 0.2, 0.5, 1])],
    )
    orientation, coefficients = _model_verification_coefficients(model)
    verification = None
    if orientation and coefficients and int(config.get("targeted_top_k", 5)) > 0:
        verification = verify_top_hyperedges(
            oracle=oracle,
            feature_spec=feature_spec,
            estimated_coefficients=coefficients,
            orientation=orientation,
            target_class=target_class,
            value_type=value_type,
            ledger=method_ledger,
            top_k=int(config.get("targeted_top_k", 5)),
            random_state=int(seed),
            operator=str(config.get("mask_operator", "delete")),
        )
    full_competitor = best_competitor(full_scores, target_class)
    competitor_rows = [
        best_competitor(row, target_class)
        for scores in evaluation_scores.values()
        for row in np.asarray(scores, dtype=np.float64)
    ]
    competitor_switch_rate = (
        float(np.mean([competitor != full_competitor for competitor in competitor_rows]))
        if competitor_rows
        else None
    )
    surrogate_metrics = evaluate_surrogate_distributions(
        model,
        masks_by_distribution=evaluation_masks,
        values_by_distribution=evaluation_values,
    )
    evaluation_elapsed_seconds = float(time.time() - evaluation_started)
    payload = {
        "status": "ok",
        "protocol": str(protocol),
        "method": str(method),
        "value_function": str(value_type),
        "task": record.task,
        "sample_id": record.sample_id,
        "n_features": int(feature_spec.n_features),
        "budget": int(budget),
        "seed": int(seed),
        "target_class": int(target_class),
        "target_label_text": str(oracle.verbalizers[target_class]),
        "full_scores": [float(value) for value in full_scores],
        "full_competitor_class": full_competitor,
        "competitor_switch_rate": competitor_switch_rate,
        "mask_operator": str(config.get("mask_operator", "delete")),
        "train_masks": [int(mask) for mask in train_masks],
        "train_masks_digest": _mask_digest(train_masks),
        "evaluation_masks_digest": _mask_digest(evaluation_masks),
        "timing": {
            "fit_elapsed_seconds": float(fit_elapsed_seconds),
            "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
        },
        "model": _model_payload(model),
        "node_scores": [float(value) for value in node_scores],
        "rankings": rankings,
        "surrogate_metrics": surrogate_metrics,
        "attribution_metrics": attribution,
        "random_ranking_control": random_attribution,
        "interaction_verification": verification,
        "query_ledger": method_ledger.to_dict(),
    }
    atomic_write_json(output_dir / "result.json", _json_safe(payload))


def _evaluation_tables(
    *,
    oracle: ValueOracle,
    feature_spec: FeatureSpec,
    target_class: int,
    value_type: str,
    masks: Mapping[str, Sequence[int]],
    operator: str,
    ledger: QueryLedger,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    values_out: Dict[str, np.ndarray] = {}
    scores_out: Dict[str, np.ndarray] = {}
    for name, distribution_masks in masks.items():
        values, scores = oracle.values_for_masks(
            feature_spec,
            list(distribution_masks),
            target_class=target_class,
            value_type=value_type,
            ledger=ledger,
            category="evaluation",
            operator=operator,
        )
        values_out[str(name)] = values
        scores_out[str(name)] = scores
    return values_out, scores_out


def _write_failure(path: Path, *, context: Mapping[str, Any], exception: Exception) -> None:
    atomic_write_json(
        path / "result.json",
        _json_safe(
            {
                **dict(context),
                "status": "failed",
                "failure_type": type(exception).__name__,
                "failure_reason": str(exception),
            }
        ),
    )


def run_controlled_protocol(
    *,
    root: Path,
    record: TextRecord,
    feature_spec: FeatureSpec,
    oracle: ValueOracle,
    target_class: int,
    full_scores: np.ndarray,
    budget: int,
    seed: int,
    value_type: str,
    config: Mapping[str, Any],
    overwrite: bool,
) -> None:
    methods = [str(value) for value in config.get("controlled_methods", [])]
    pending_methods = []
    for method in methods:
        output_dir = _method_result_dir(
            root,
            protocol="controlled",
            value_type=value_type,
            method=method,
            task=record.task,
            sample_id=record.sample_id,
            budget=budget,
            seed=seed,
        )
        if overwrite or not (output_dir / "result.json").exists():
            pending_methods.append(method)
    if not pending_methods:
        return
    train_masks = sample_attribution_masks(feature_spec.n_features, budget, seed=seed)
    excluded = set(train_masks)
    evaluation_masks = sample_evaluation_masks(
        feature_spec.n_features,
        count_per_distribution=int(config.get("evaluation_masks_per_distribution", 64)),
        seed=int(seed) + 100003,
        exclude=sorted(excluded),
        near_full_deletions=config.get("near_full_deletions", [1, 2, 3, 5]),
        fixed_keep_fractions=config.get("fixed_keep_fractions", [0.25, 0.5, 0.75]),
    )
    fitted: Dict[str, tuple[object, QueryLedger, float]] = {}
    failures: Dict[str, tuple[QueryLedger, Exception]] = {}
    for method in pending_methods:
        ledger = QueryLedger(f"controlled/{method}/{record.sample_id}/{budget}/{seed}/{value_type}")
        fit_started = time.time()
        try:
            scores = oracle.score_masks(
                feature_spec,
                train_masks,
                ledger=ledger,
                category="training",
                operator=str(config.get("mask_operator", "delete")),
            )
            values = values_from_score_matrix(scores, target_class=target_class, value_type=value_type)
            model = _fit_controlled_method(
                method,
                masks=train_masks,
                values=values,
                n_features=feature_spec.n_features,
                seed=seed,
                config=config,
            )
            fitted[method] = (model, ledger, float(time.time() - fit_started))
        except Exception as exc:
            failures[method] = (ledger, exc)
            if bool(config.get("fail_fast", False)):
                raise

    # Evaluation values are not requested until every method has finished fitting.
    eval_ledger = QueryLedger(f"controlled/evaluation/{record.sample_id}/{budget}/{seed}/{value_type}")
    evaluation_values, evaluation_scores = _evaluation_tables(
        oracle=oracle,
        feature_spec=feature_spec,
        target_class=target_class,
        value_type=value_type,
        masks=evaluation_masks,
        operator=str(config.get("mask_operator", "delete")),
        ledger=eval_ledger,
    )
    for method in pending_methods:
        output_dir = _method_result_dir(
            root,
            protocol="controlled",
            value_type=value_type,
            method=method,
            task=record.task,
            sample_id=record.sample_id,
            budget=budget,
            seed=seed,
        )
        if method in fitted:
            model, ledger, fit_elapsed_seconds = fitted[method]
            _evaluate_and_save(
                output_dir=output_dir,
                model=model,
                method=method,
                protocol="controlled",
                value_type=value_type,
                record=record,
                feature_spec=feature_spec,
                target_class=target_class,
                full_scores=full_scores,
                budget=budget,
                seed=seed,
                train_masks=train_masks,
                oracle=oracle,
                method_ledger=ledger,
                evaluation_masks=evaluation_masks,
                evaluation_values=evaluation_values,
                evaluation_scores=evaluation_scores,
                config=config,
                fit_elapsed_seconds=fit_elapsed_seconds,
            )
        else:
            ledger, exc = failures[method]
            _write_failure(
                output_dir,
                context={
                    "protocol": "controlled",
                    "method": method,
                    "value_function": value_type,
                    "task": record.task,
                    "sample_id": record.sample_id,
                    "budget": budget,
                    "seed": seed,
                    "query_ledger": ledger.to_dict(),
                },
                exception=exc,
            )


def run_native_protocol(
    *,
    root: Path,
    record: TextRecord,
    feature_spec: FeatureSpec,
    oracle: ValueOracle,
    target_class: int,
    full_scores: np.ndarray,
    budget: int,
    seed: int,
    value_type: str,
    config: Mapping[str, Any],
    overwrite: bool,
) -> None:
    methods = [str(value) for value in config.get("native_methods", [])]
    existing = [
        (
            _method_result_dir(
                root,
                protocol="native",
                value_type=value_type,
                method=method,
                task=record.task,
                sample_id=record.sample_id,
                budget=budget,
                seed=seed,
            )
            / "result.json"
        ).exists()
        for method in methods
    ]
    if methods and all(existing) and not overwrite:
        return
    fitted: Dict[str, tuple[object, QueryLedger, list[int], float]] = {}
    failures: Dict[str, tuple[QueryLedger, Exception]] = {}
    for method in methods:
        ledger = QueryLedger(f"native/{method}/{record.sample_id}/{budget}/{seed}/{value_type}")
        fit_started = time.time()
        try:
            if method in {"deletion_mobius", "presence_mobius"}:
                masks = sample_attribution_masks(feature_spec.n_features, budget, seed=seed)
                scores = oracle.score_masks(
                    feature_spec,
                    masks,
                    ledger=ledger,
                    category="training",
                    operator=str(config.get("mask_operator", "delete")),
                )
                values = values_from_score_matrix(scores, target_class=target_class, value_type=value_type)
                model = _fit_controlled_method(
                    method,
                    masks=masks,
                    values=values,
                    n_features=feature_spec.n_features,
                    seed=seed,
                    config=config,
                )
            elif method == "proxyspex":
                # Make the mandatory target/full query visible in this method's logical budget.
                oracle.score_masks(
                    feature_spec,
                    [(1 << feature_spec.n_features) - 1],
                    ledger=ledger,
                    category="training",
                    operator=str(config.get("mask_operator", "delete")),
                )
                proxy_cfg = dict(config.get("proxyspex", {}))
                model = fit_proxyspex_native(
                    oracle=oracle,
                    feature_spec=feature_spec,
                    target_class=target_class,
                    value_type=value_type,
                    ledger=ledger,
                    budget=budget,
                    operator=str(config.get("mask_operator", "delete")),
                    max_order=int(proxy_cfg.get("max_order", 2)),
                    index=str(proxy_cfg.get("index", "FBII")),
                    proxy_model=str(proxy_cfg.get("proxy_model", "tree")),
                    hpo=bool(proxy_cfg.get("hpo", False)),
                    random_state=seed,
                )
                masks = model.train_masks
            else:
                raise ValueError(f"Unsupported native method: {method!r}")
            fitted[method] = (
                model,
                ledger,
                [int(mask) for mask in masks],
                float(time.time() - fit_started),
            )
        except Exception as exc:
            failures[method] = (ledger, exc)
            if bool(config.get("fail_fast", False)):
                raise

    excluded = sorted({mask for _, _, masks, _ in fitted.values() for mask in masks})
    evaluation_masks = sample_evaluation_masks(
        feature_spec.n_features,
        count_per_distribution=int(config.get("evaluation_masks_per_distribution", 64)),
        seed=int(seed) + 200003,
        exclude=excluded,
        near_full_deletions=config.get("near_full_deletions", [1, 2, 3, 5]),
        fixed_keep_fractions=config.get("fixed_keep_fractions", [0.25, 0.5, 0.75]),
    )
    eval_ledger = QueryLedger(f"native/evaluation/{record.sample_id}/{budget}/{seed}/{value_type}")
    evaluation_values, evaluation_scores = _evaluation_tables(
        oracle=oracle,
        feature_spec=feature_spec,
        target_class=target_class,
        value_type=value_type,
        masks=evaluation_masks,
        operator=str(config.get("mask_operator", "delete")),
        ledger=eval_ledger,
    )
    for method in methods:
        output_dir = _method_result_dir(
            root,
            protocol="native",
            value_type=value_type,
            method=method,
            task=record.task,
            sample_id=record.sample_id,
            budget=budget,
            seed=seed,
        )
        if method in failures:
            ledger, exc = failures[method]
            _write_failure(
                output_dir,
                context={
                    "protocol": "native",
                    "method": method,
                    "value_function": value_type,
                    "task": record.task,
                    "sample_id": record.sample_id,
                    "budget": budget,
                    "seed": seed,
                    "query_ledger": ledger.to_dict(),
                },
                exception=exc,
            )
            continue
        model, ledger, masks, fit_elapsed_seconds = fitted[method]
        _evaluate_and_save(
            output_dir=output_dir,
            model=model,
            method=method,
            protocol="native",
            value_type=value_type,
            record=record,
            feature_spec=feature_spec,
            target_class=target_class,
            full_scores=full_scores,
            budget=budget,
            seed=seed,
            train_masks=masks,
            oracle=oracle,
            method_ledger=ledger,
            evaluation_masks=evaluation_masks,
            evaluation_values=evaluation_values,
            evaluation_scores=evaluation_scores,
            config=config,
            fit_elapsed_seconds=fit_elapsed_seconds,
        )


def run_benchmark(
    *,
    config: Mapping[str, Any],
    records: Sequence[TextRecord],
    verbalizers: Sequence[str],
    results_dir: Path,
    overwrite: bool = False,
) -> Dict[str, object]:
    ensure_dir(results_dir)
    atomic_write_json(results_dir / "run_config.json", _json_safe(dict(config)))
    atomic_write_json(results_dir / "environment.json", environment_snapshot())
    scorer = build_text_scorer(dict(config.get("model", {"type": "mock_sentiment"})), verbalizers=verbalizers)
    tokenizer = getattr(getattr(scorer, "backbone", None), "tokenizer", None)
    model_cfg = dict(config.get("model", {}))
    model_fingerprint = json.dumps(model_cfg, ensure_ascii=False, sort_keys=True)
    oracle = ValueOracle(
        scorer=scorer,
        cache_path=results_dir / "cache" / "value_oracle.sqlite3",
        model_fingerprint=model_fingerprint,
        prompt_version=str(config.get("prompt_version", "lima_hf_label_v1")),
        batch_size=int(config.get("batch_size", 32)),
    )
    exclusions = []
    completed = 0
    started = time.time()
    try:
        for record in records:
            feature_spec = build_lexical_word_features(
                record.text,
                sample_id=record.sample_id,
                tokenizer=tokenizer,
            )
            ok, reason = validate_feature_reconstruction(feature_spec)
            if not ok or feature_spec.n_features < int(config.get("min_features", 2)):
                exclusions.append(
                    {
                        "sample_id": record.sample_id,
                        "reason": reason if not ok else "too_few_features",
                        "n_features": int(feature_spec.n_features),
                    }
                )
                continue
            max_features = config.get("max_features")
            if max_features is not None and feature_spec.n_features > int(max_features):
                exclusions.append(
                    {
                        "sample_id": record.sample_id,
                        "reason": "too_many_features",
                        "n_features": int(feature_spec.n_features),
                    }
                )
                continue
            atomic_write_json(
                results_dir / "features" / record.task / f"{record.sample_id}.json",
                feature_spec.to_dict(),
            )
            setup_ledger = QueryLedger(f"setup/{record.sample_id}")
            full_mask = (1 << feature_spec.n_features) - 1
            full_scores = oracle.score_masks(
                feature_spec,
                [full_mask],
                ledger=setup_ledger,
                category="setup",
                operator=str(config.get("mask_operator", "delete")),
            )[0]
            target_class = int(np.argmax(full_scores))
            for budget in budget_values(feature_spec.n_features, config):
                for seed in [int(value) for value in config.get("seeds", [0, 1, 2])]:
                    for value_type in [
                        str(value)
                        for value in config.get(
                            "value_functions", ["predicted_class_margin", "raw_target_score"]
                        )
                    ]:
                        if bool(config.get("run_controlled", True)):
                            run_controlled_protocol(
                                root=results_dir,
                                record=record,
                                feature_spec=feature_spec,
                                oracle=oracle,
                                target_class=target_class,
                                full_scores=full_scores,
                                budget=budget,
                                seed=seed,
                                value_type=value_type,
                                config=config,
                                overwrite=overwrite,
                            )
                        if bool(config.get("run_native", True)):
                            run_native_protocol(
                                root=results_dir,
                                record=record,
                                feature_spec=feature_spec,
                                oracle=oracle,
                                target_class=target_class,
                                full_scores=full_scores,
                                budget=budget,
                                seed=seed,
                                value_type=value_type,
                                config=config,
                                overwrite=overwrite,
                            )
            completed += 1
    finally:
        oracle_counters = oracle.snapshot_counters()
        cache_count = int(oracle_counters["cache_entry_count"])
        oracle.close()
    manifest = {
        "status": "complete",
        "record_count": int(len(records)),
        "completed_record_count": int(completed),
        "exclusions": exclusions,
        "cache_entry_count": int(cache_count),
        "value_oracle_counters": oracle_counters,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_write_json(results_dir / "manifest.json", manifest)
    return manifest
