from __future__ import annotations

from pathlib import Path

import numpy as np

from mobius_verify.src.designs import design_matrix, low_degree_terms
from mobius_verify.src.featureization import build_lexical_word_features
from mobius_verify.src.interaction_verification import targeted_true_coefficient
from mobius_verify.src.methods.sparse_mobius import model_node_scores
from mobius_verify.src.methods.proxyspex_adapter import (
    fit_proxyspex_from_observations,
    resolve_lightgbm_min_child_samples,
)
from mobius_verify.src.methods.sparse_surrogate import fit_sparse_surrogate
from mobius_verify.src.models.base import RawTextScorer
from mobius_verify.src.query_ledger import QueryLedger
from mobius_verify.src.subset_enumeration import all_masks
from mobius_verify.src.value_functions import values_from_score_matrix
from mobius_verify.src.value_oracle import ValueOracle


class PairScorer(RawTextScorer):
    verbalizers = ["negative", "positive"]

    def __init__(self) -> None:
        self.scored_text_count = 0

    def score_texts(self, texts):
        self.scored_text_count += len(texts)
        rows = []
        for text in texts:
            effect = 5.0 if "red" in text and "blue" in text else 0.0
            rows.append([0.0, effect])
        return np.asarray(rows, dtype=np.float64)


class FailingScorer(PairScorer):
    def score_texts(self, texts):
        if self.scored_text_count >= 1:
            raise RuntimeError("intentional scoring failure")
        return super().score_texts(texts)


def test_predicted_class_margin_is_not_raw_target_score() -> None:
    scores = np.asarray([[1.0, 3.0, 2.5], [4.0, 2.0, 1.0]])
    margin = values_from_score_matrix(scores, target_class=1, value_type="predicted_class_margin")
    raw = values_from_score_matrix(scores, target_class=1, value_type="raw_target_score")
    assert np.allclose(margin, [0.5, -2.0])
    assert np.allclose(raw, [3.0, 2.0])


def test_low_degree_presence_and_fourier_span_same_space() -> None:
    n = 4
    masks = all_masks(n)
    terms = low_degree_terms(n, 2)
    rng = np.random.default_rng(3)
    values = rng.normal(size=len(masks))
    predictions = []
    for basis in ("presence_mobius", "fourier"):
        matrix = design_matrix(masks, terms, n_features=n, basis=basis).astype(np.float64)
        augmented = np.column_stack([np.ones(len(masks)), matrix])
        coefficients, *_ = np.linalg.lstsq(augmented, values, rcond=None)
        predictions.append(augmented @ coefficients)
    assert np.max(np.abs(predictions[0] - predictions[1])) < 1e-9


def test_sparse_surrogate_recovers_degree_two_function() -> None:
    n = 4
    masks = all_masks(n)
    values = []
    for mask in masks:
        x0 = float(bool(mask & 1))
        x1x2 = float((mask & 0b110) == 0b110)
        values.append(1.0 + 2.0 * x0 - 3.0 * x1x2)
    model = fit_sparse_surrogate(
        masks=masks,
        values=values,
        n_features=n,
        basis="presence_mobius",
        max_degree=2,
        alphas=[1e-8],
        cv_folds=2,
        ridge_alphas=[1e-12],
    )
    assert np.max(np.abs(model.predict(masks) - values)) < 1e-5
    node_scores = model_node_scores(model)
    assert abs(float(np.sum(node_scores)) - float(sum(model.coefficient_dict().values()))) < 1e-8


def test_global_cache_and_method_ledgers_are_separate(tmp_path: Path) -> None:
    scorer = PairScorer()
    spec = build_lexical_word_features("red blue", sample_id="sample")
    oracle = ValueOracle(
        scorer=scorer,
        cache_path=tmp_path / "oracle.sqlite3",
        model_fingerprint="pair",
    )
    first = QueryLedger("first")
    second = QueryLedger("second")
    masks = [0, 3]
    oracle.score_masks(spec, masks, ledger=first, category="training")
    oracle.score_masks(spec, masks, ledger=second, category="training")
    counters = oracle.snapshot_counters()
    oracle.close()
    assert first.logical_unique_queries == 2
    assert second.logical_unique_queries == 2
    assert first.physical_forwards_caused == 2
    assert second.physical_forwards_caused == 0
    assert second.global_cache_hits == 2
    assert counters["physical_values_scored_this_run"] == 2
    assert counters["scorer_batch_calls_this_run"] == 1


def test_deletion_targeted_queries_compute_exact_coefficient(tmp_path: Path) -> None:
    scorer = PairScorer()
    spec = build_lexical_word_features("red blue", sample_id="sample")
    oracle = ValueOracle(
        scorer=scorer,
        cache_path=tmp_path / "oracle.sqlite3",
        model_fingerprint="pair",
    )
    ledger = QueryLedger("verify")
    result = targeted_true_coefficient(
        oracle=oracle,
        feature_spec=spec,
        term=0b11,
        orientation="deletion_mobius",
        target_class=1,
        value_type="predicted_class_margin",
        ledger=ledger,
    )
    oracle.close()
    assert result["true_coefficient"] == 5.0
    assert ledger.category_count("interaction_verification") == 4


def test_query_ledger_survives_partial_scoring_failure(tmp_path: Path) -> None:
    scorer = FailingScorer()
    spec = build_lexical_word_features("red blue", sample_id="sample")
    oracle = ValueOracle(
        scorer=scorer,
        cache_path=tmp_path / "oracle.sqlite3",
        model_fingerprint="failing",
        batch_size=1,
    )
    ledger = QueryLedger("failure")
    try:
        oracle.score_masks(spec, [0, 3], ledger=ledger, category="training")
    except RuntimeError as exc:
        assert str(exc) == "intentional scoring failure"
    else:
        raise AssertionError("Expected the scorer to fail")
    counters = oracle.snapshot_counters()
    oracle.close()
    assert ledger.logical_unique_queries == 2
    assert ledger.physical_forwards_caused == 1
    assert counters["physical_values_scored_this_run"] == 1
    assert counters["scorer_batch_calls_this_run"] == 2


def test_lightgbm_leaf_minimum_scales_with_cv_training_fold() -> None:
    assert resolve_lightgbm_min_child_samples(
        n_observations=4, cv_splits=2, configured="auto"
    ) == (1, 2)
    assert resolve_lightgbm_min_child_samples(
        n_observations=100, cv_splits=5, configured="auto"
    ) == (20, 80)
    assert resolve_lightgbm_min_child_samples(
        n_observations=8, cv_splits=2, configured=3
    ) == (3, 4)


def test_proxyspex_result_records_non_degeneracy_diagnostics() -> None:
    masks = list(range(8))
    values = [
        float(bool(mask & 1)) + 2.0 * float((mask & 0b110) == 0b110)
        for mask in masks
    ]
    model = fit_proxyspex_from_observations(
        masks=masks,
        values=values,
        n_features=3,
        proxy_model="tree",
        hpo=False,
        random_state=0,
    )
    diagnostics = model.to_dict()["diagnostics"]
    assert diagnostics["degenerate_proxy"] is False
    assert diagnostics["proxy_training_prediction_std"] > 0
    assert diagnostics["refined_training_r2"] > 0.99
