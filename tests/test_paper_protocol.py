"""P0/P1 artifact, estimator, held-out, and interaction protocol tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mobius.core.artifacts import (
    bool_matrix_to_masks,
    load_observation_artifact,
    load_surrogate_artifact,
    masks_to_bool_matrix,
    predict_surrogate,
    write_observation_artifact,
    write_surrogate_artifact,
)
from mobius.core.results import ResultStore
from mobius.core.schema import AttributionResult, DatasetBundle, TextChunk, TextSample
from mobius.evaluation.evaluator import evaluate_run
from mobius.evaluation.surrogate import sample_shared_heldout_masks
from mobius.methods.sparse.estimator import fit_sparse_model
from mobius.methods.sparse.hierarchy import (
    normalize_hierarchy,
    strict_support_terms,
)
from mobius.models.mock import MockSentimentScorer
from scripts.build_surrogate_holdout import build_shared_holdout
from scripts.analyze_hierarchy import analyze_hierarchy
from scripts.derive_projection_run import derive_projection_run
from scripts.evaluate_surrogates import evaluate_audit
from scripts.verify_interactions import (
    _match_random_pairs,
    deletion_pair_coefficient,
    verify_runs,
)
from mobius.methods.sparse.explainer import run_sparse_mobius


def _config(*, degree: int, seed: int = 7) -> dict:
    """Build one deterministic inline paper-protocol configuration."""

    return {
        "method": "sparse_mobius",
        "output_level": "standard",
        "budget": 16,
        "seed": seed,
        "max_degree": degree,
        "k": 3,
        "value_function": "target_probability",
        "target_mode": "predicted",
        "chunker": "word",
        "eval_granularity": "word",
        "eval_q_values": [20, 50],
        "min_features": 1,
        "max_features": None,
        "batch_size": 8,
        "basis": "deletion_mobius",
        "hierarchy": "none",
        "projector": "singleton_only" if degree == 1 else "signed_equal_share",
        "sampler": {"name": "uniform_size"},
        "estimator": {
            "name": "lasso_support",
            "refit": "ridge_cv",
            "alphas": [1e-6, 1e-4],
            "l1_ratios": [1.0],
            "cv_folds": 2,
        },
        "dataset": {"name": "inline", "split": "validation"},
        "model": {"type": "mock_sentiment", "device": "cpu"},
    }


def _bundle() -> DatasetBundle:
    """Build an inline sample with five word players and nonlinear sentiment."""

    return DatasetBundle(
        dataset_name="inline",
        split="validation",
        samples=[
            TextSample(
                sample_id="s0",
                text="not good but finally moving",
                label=1,
                label_text="positive",
            )
        ],
        label_names=["negative", "positive"],
        verbalizers=["negative", "positive"],
    )


def _minimal_result(
    *,
    observation: dict | None = None,
    surrogate: dict | None = None,
) -> AttributionResult:
    """Build the smallest valid attribution result for ResultStore tests."""

    return AttributionResult(
        sample_id="s0",
        gold_label=1,
        predicted_label=1,
        target_label=1,
        text="good",
        chunks=[TextChunk(0, 0, 4, "good")],
        node_scores=[1.0],
        ranking=[0],
        selected_ids=[0],
        attribution_cost={},
        observation_artifact=observation,
        surrogate_artifact=surrogate,
    )


def _sidecars() -> tuple[dict, dict]:
    """Build aligned one-player observation and surrogate artifacts."""

    observation = {
        "sample_id": "s0",
        "method": "sparse_mobius",
        "n_features": 1,
        "keep_masks": [0, 1],
        "label_scores": [[0.2, 0.8], [0.1, 0.9]],
        "attribution_values": [0.8, 0.9],
    }
    surrogate = {
        "sample_id": "s0",
        "method": "sparse_mobius",
        "n_features": 1,
        "player_to_chunk_id": [0],
        "predictor": {
            "type": "sparse_polynomial",
            "basis": "deletion_mobius",
            "intercept": 0.9,
            "terms": [{"players": [0], "coefficient": -0.1}],
        },
    }
    return observation, surrogate


def test_artifact_roundtrip_supports_more_than_64_players(tmp_path: Path) -> None:
    """Round-trip arbitrary-width masks, all scores, digest, and predictions."""

    n_features = 70
    integer_masks = [0, 1 << 69, (1 << 70) - 1]
    matrix = masks_to_bool_matrix(integer_masks, n_features)
    assert bool_matrix_to_masks(matrix) == integer_masks
    observation_path = tmp_path / "sample.npz"
    write_observation_artifact(
        observation_path,
        {
            "sample_id": "wide",
            "method": "sparse_mobius",
            "n_features": n_features,
            "keep_masks": matrix,
            "label_scores": [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6]],
            "attribution_values": [1.0, 0.8, 0.6],
        },
    )
    loaded = load_observation_artifact(observation_path)
    assert bool_matrix_to_masks(loaded["keep_masks"]) == integer_masks
    surrogate_path = tmp_path / "sample.json"
    written = write_surrogate_artifact(
        surrogate_path,
        {
            "sample_id": "wide",
            "method": "sparse_mobius",
            "n_features": n_features,
            "player_to_chunk_id": list(range(n_features)),
            "predictor": {
                "type": "sparse_polynomial",
                "basis": "deletion_mobius",
                "intercept": 1.0,
                "terms": [{"players": [69], "coefficient": -0.2}],
            },
        },
    )
    reloaded = load_surrogate_artifact(surrogate_path)
    assert reloaded["digest"] == written["digest"]
    assert np.allclose(predict_surrogate(reloaded, matrix), [0.8, 1.0, 1.0])


def test_required_sidecars_control_resume_without_affecting_other_methods(
    tmp_path: Path,
) -> None:
    """Require sidecars only for stores that explicitly declare them."""

    observation, surrogate = _sidecars()
    run_dir = tmp_path / "run"
    store = ResultStore(
        run_dir,
        _config(degree=1),
        overwrite=True,
        required_artifacts=("observation", "surrogate"),
    )
    store.write_sample(
        _minimal_result(observation=observation, surrogate=surrogate)
    )
    assert store.sample_complete("s0")
    (run_dir / "surrogates" / "s0.json").unlink()
    resumed = ResultStore(
        run_dir,
        _config(degree=1),
        required_artifacts=("observation", "surrogate"),
    )
    assert not resumed.sample_complete("s0")
    ordinary = ResultStore(run_dir, _config(degree=1))
    assert ordinary.sample_complete("s0")


def test_estimator_supports_every_refit_and_records_cv_path() -> None:
    """Exercise raw, ridge-CV, fixed-ridge, and minimum-norm OLS refits."""

    masks = list(range(8))
    values = [
        1.2 - 0.7 * (not bool(mask & 1)) + 0.4 * (not bool(mask & 2))
        for mask in masks
    ]
    for refit in ("none", "ridge_cv", "ridge_fixed", "ols"):
        config = {
            "alphas": [1e-8, 1e-5],
            "l1_ratios": [1.0],
            "cv_folds": 2,
            "coefficient_tolerance": 1e-10,
            "refit": refit,
            "ridge_alpha": 0.01,
        }
        model = fit_sparse_model(
            masks,
            values,
            n_features=3,
            terms=[1, 2, 4, 3, 5, 6],
            basis="deletion_mobius",
            max_degree=2,
            config=config,
            random_state=3,
        )
        assert model.refit_mode == refit
        assert len(model.diagnostics["regularization_path"]) == 2
        assert (
            sum(
                bool(row["cv_selected"])
                for row in model.diagnostics["regularization_path"]
            )
            == 1
        )
        assert model.diagnostics["refit_support_size"] == len(model.refit_support)
        if refit == "ols":
            assert "ols_rank" in model.diagnostics


def test_strict_prunes_orphans_and_rejects_higher_degree() -> None:
    """Verify exact strong-heredity pruning and the degree-two boundary."""

    retained, diagnostics = strict_support_terms(
        [1, 2, 3, 5],
        max_degree=2,
    )
    assert retained == [1, 2, 3]
    assert diagnostics["removed_terms"][0]["term"] == 5
    assert diagnostics["removed_terms"][0]["present_parent_count"] == 1
    fitted = fit_sparse_model(
        [0, 1, 2, 3],
        [1.0, 0.0, 0.0, 0.0],
        n_features=2,
        terms=[1, 2, 3],
        basis="deletion_mobius",
        max_degree=2,
        config={
            "alphas": [1e-3],
            "l1_ratios": [1.0],
            "cv_folds": 2,
            "refit": "ols",
        },
        random_state=1,
        hierarchy_policy="strict",
    )
    assert fitted.selection_support == [3]
    assert fitted.hierarchy_support == []
    assert fitted.diagnostics["hierarchy"]["removed_support_size"] == 1
    assert normalize_hierarchy("parent_screening") == "parent_screening"
    try:
        normalize_hierarchy("strong")
    except ValueError as error:
        assert "renamed" in str(error)
    else:  # pragma: no cover - supports direct execution without pytest
        raise AssertionError("The removed hierarchy name must not be accepted.")
    try:
        strict_support_terms([1], max_degree=3)
    except ValueError as error:
        assert "max_degree <= 2" in str(error)
    else:  # pragma: no cover - supports direct execution without pytest
        raise AssertionError("Strict degree three must fail explicitly.")


def test_shared_holdout_sampling_is_deterministic_disjoint_and_degrades() -> None:
    """Check shared masks never overlap training or each other, even in small spaces."""

    first = sample_shared_heldout_masks(
        3,
        count_per_distribution=8,
        seed=11,
        excluded_masks=[0, 1, 2, 3],
        near_full_deletions=[1, 2],
        distributions=["bernoulli", "near_full"],
    )
    second = sample_shared_heldout_masks(
        3,
        count_per_distribution=8,
        seed=11,
        excluded_masks=[0, 1, 2, 3],
        near_full_deletions=[1, 2],
        distributions=["bernoulli", "near_full"],
    )
    assert first == second
    assert not (set(first["bernoulli"]) & {0, 1, 2, 3})
    assert not (set(first["near_full"]) & {0, 1, 2, 3})
    assert not (set(first["bernoulli"]) & set(first["near_full"]))
    assert len(first["bernoulli"]) + len(first["near_full"]) <= 4
    default_only = sample_shared_heldout_masks(
        3,
        count_per_distribution=2,
        seed=11,
        excluded_masks=[],
    )
    assert "bernoulli" in default_only
    assert "near_full" not in default_only


def test_four_query_deletion_coefficient_is_exact() -> None:
    """Recover a synthetic pair coefficient from full, singleton, and pair deletions."""

    intercept, left, right, interaction = 0.7, 0.2, -0.1, 0.45
    values = [
        intercept,
        intercept + left,
        intercept + right,
        intercept + left + right + interaction,
    ]
    assert np.isclose(deletion_pair_coefficient(values), interaction)


def test_random_pair_controls_exclude_the_complete_fitted_support() -> None:
    """Ensure matched controls cannot reuse another fitted interaction edge."""

    matches = _match_random_pairs(
        [(0, 2)],
        n_features=5,
        seed=5,
        excluded_pairs=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    random_pair = tuple(matches[0]["random_pair"])
    assert random_pair not in {(0, 1), (0, 2), (1, 3), (2, 4)}


def test_mock_e2_derivation_and_shared_heldout_end_to_end(tmp_path: Path) -> None:
    """Run the complete mock A/B/C, held-out, E3, E4, and paper-table workflow."""

    bundle = _bundle()
    run_a = tmp_path / "A"
    run_c = tmp_path / "C"
    run_strict = tmp_path / "strict"
    cache = tmp_path / "global.sqlite3"
    run_sparse_mobius(
        _config(degree=1),
        bundle,
        MockSentimentScorer(),
        run_dir=run_a,
        cache_path=cache,
        overwrite=True,
        evaluate=True,
    )
    run_sparse_mobius(
        _config(degree=2),
        bundle,
        MockSentimentScorer(),
        run_dir=run_c,
        cache_path=cache,
        overwrite=True,
        evaluate=True,
    )
    strict_config = _config(degree=2)
    strict_config["hierarchy"] = "strict"
    run_sparse_mobius(
        strict_config,
        bundle,
        MockSentimentScorer(),
        run_dir=run_strict,
        cache_path=cache,
        overwrite=True,
        evaluate=True,
    )
    run_b = derive_projection_run(
        run_c,
        projector="singleton_only",
        output_root=tmp_path / "derived",
        run_suffix="B",
        overwrite=True,
    )
    c_surrogate = load_surrogate_artifact(run_c / "surrogates" / "s0.json")
    b_surrogate = load_surrogate_artifact(run_b / "surrogates" / "s0.json")
    c_sample = json.loads(
        (run_c / "samples" / "s0.json").read_text(encoding="utf-8")
    )
    assert c_sample["method_summary"]["surrogate_digest"] == c_surrogate["digest"]
    assert c_surrogate["digest"] == b_surrogate["digest"]
    b_run = json.loads((run_b / "run.json").read_text(encoding="utf-8"))
    assert b_run["provenance"]["model_calls"] == 0
    evaluate_run(
        run_b,
        bundle,
        MockSentimentScorer(),
        target="predicted",
        eval_granularity="word",
        q_values=[20, 50],
    )
    audit = build_shared_holdout(
        [run_a, run_b, run_c, run_strict],
        output_dir=tmp_path / "heldout",
        device="cpu",
        count_per_distribution=4,
        seed=19,
        near_full_deletions=[1, 2, 3, 5],
        min_count=1,
    )
    index = evaluate_audit(audit)
    assert len(index["evaluations"]) == 4
    manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"][0]["status"] == "ok"
    assert manifest["query_cost"]["attribution_budget_used"] == 0
    assert manifest["settings"]["distributions"] == ["bernoulli"]
    assert manifest["metadata"]["input_run_dirs"] == [
        str(path.resolve()) for path in (run_a, run_b, run_c, run_strict)
    ]
    assert manifest["metadata"]["dataset"]["name"] == "inline"
    assert manifest["metadata"]["model"]["type"] == "mock_sentiment"
    heldout = np.load(audit / "samples" / "s0.npz", allow_pickle=False)
    training = set()
    for run_dir in (run_a, run_b, run_c, run_strict):
        observation = load_observation_artifact(
            run_dir / "observations" / "s0.npz"
        )
        training.update(bool_matrix_to_masks(observation["keep_masks"]))
    bernoulli = set(bool_matrix_to_masks(heldout["bernoulli_keep_masks"]))
    assert not training.intersection(bernoulli)
    verification = verify_runs(
        [run_c],
        output_dir=tmp_path / "interactions",
        device="cpu",
        top_k=2,
        top_k_per_parent_group=1,
        seed=23,
        bootstrap=20,
    )
    verification_summary = json.loads(
        (verification / "summary.json").read_text(encoding="utf-8")
    )
    assert verification_summary["query_cost"]["attribution_budget_used"] == 0
    hierarchy = analyze_hierarchy(
        run_c,
        run_strict,
        verification_dir=verification,
        heldout_audit=audit,
        output_dir=tmp_path / "hierarchy",
        device="cpu",
        seed=29,
        bootstrap=20,
    )
    hierarchy_summary = json.loads(
        (hierarchy / "summary.json").read_text(encoding="utf-8")
    )
    assert hierarchy_summary["analysis_query_cost"]["attribution_budget_used"] == 0
