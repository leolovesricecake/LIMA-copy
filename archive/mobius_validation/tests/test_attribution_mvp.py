from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lima_llm.types import TextChunk

from mobius_verify.src.designs import design_matrix, low_degree_terms
from mobius_verify.src.featureization import build_lexical_word_features
from mobius_verify.src.interaction_verification import targeted_true_coefficient
from mobius_verify.src.methods.sparse_mobius import model_node_scores
from mobius_verify.src.methods.sparse_surrogate import fit_sparse_surrogate
from mobius_verify.src.models.base import RawTextScorer
from mobius_verify.src.query_ledger import QueryLedger
from mobius_verify.src.sparse_runner import SparseMobiusCoalitionGame
from mobius_verify.src.subset_enumeration import (
    all_masks,
    sample_deletion_mobius_masks,
)
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


def test_predicted_class_margin_and_probability_are_distinct() -> None:
    scores = np.asarray([[1.0, 3.0, 2.5], [4.0, 2.0, 1.0]])
    margin = values_from_score_matrix(scores, target_class=1, value_type="predicted_class_margin")
    probability = values_from_score_matrix(scores, target_class=1, value_type="predicted_probability")
    raw = values_from_score_matrix(scores, target_class=1, value_type="raw_target_score")
    assert np.allclose(margin, [0.5, -2.0])
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.allclose(raw, [3.0, 2.0])


def test_low_degree_presence_and_fourier_span_same_space() -> None:
    masks = all_masks(4)
    terms = low_degree_terms(4, 2)
    values = np.random.default_rng(3).normal(size=len(masks))
    predictions = []
    for basis in ("presence_mobius", "fourier"):
        matrix = design_matrix(masks, terms, n_features=4, basis=basis).astype(np.float64)
        augmented = np.column_stack([np.ones(len(masks)), matrix])
        coefficients, *_ = np.linalg.lstsq(augmented, values, rcond=None)
        predictions.append(augmented @ coefficients)
    assert np.max(np.abs(predictions[0] - predictions[1])) < 1e-9


def test_sparse_surrogate_recovers_degree_two_function() -> None:
    masks = all_masks(4)
    values = [
        1.0
        + 2.0 * float(bool(mask & 1))
        - 3.0 * float((mask & 0b110) == 0b110)
        for mask in masks
    ]
    model = fit_sparse_surrogate(
        masks=masks,
        values=values,
        n_features=4,
        basis="presence_mobius",
        max_degree=2,
        alphas=[1e-8],
        cv_folds=2,
        ridge_alphas=[1e-12],
    )
    assert np.max(np.abs(model.predict(masks) - values)) < 1e-5
    assert abs(float(np.sum(model_node_scores(model))) - sum(model.coefficient_dict().values())) < 1e-8


def test_deletion_sampler_owns_exact_reproducible_budget() -> None:
    first = sample_deletion_mobius_masks(8, 64, seed=7)
    second = sample_deletion_mobius_masks(8, 64, seed=7)
    assert first.masks == second.masks
    assert len(first.masks) == len(set(first.masks)) == 64
    assert 0 in first.masks
    assert (1 << 8) - 1 in first.masks
    assert first.diagnostics["sampler"] == "deletion_mobius_mixture_v1"
    assert first.diagnostics["source_counts"].get("near_full", 0) > 0
    assert first.diagnostics["source_counts"].get("global_bernoulli", 0) > 0


def test_sparse_game_uses_its_own_masks_and_ledger(tmp_path: Path) -> None:
    scorer = PairScorer()
    oracle = ValueOracle(
        scorer=scorer,
        cache_path=tmp_path / "oracle.sqlite3",
        model_fingerprint="pair",
    )
    ledger = QueryLedger("sparse")
    units = [
        TextChunk(0, 0, 4, "red "),
        TextChunk(1, 4, 8, "blue"),
    ]
    game = SparseMobiusCoalitionGame(
        sample_id="sample",
        units=units,
        player_to_chunk_id=[0, 1],
        oracle=oracle,
        target_class=1,
        value_function="predicted_class_margin",
        ledger=ledger,
    )
    values, _ = game.values_for_masks([0, 1, 2, 3], category="training")
    oracle.close()
    assert values.tolist() == [0.0, 0.0, 0.0, 5.0]
    assert ledger.category_count("training") == 4
    assert game.stats()["row_count_by_category"]["training"] == 4


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
    oracle.score_masks(spec, [0, 3], ledger=first, category="training")
    oracle.score_masks(spec, [0, 3], ledger=second, category="training")
    oracle.close()
    assert first.logical_unique_queries == second.logical_unique_queries == 2
    assert first.physical_forwards_caused == 2
    assert second.physical_forwards_caused == 0
    assert second.global_cache_hits == 2


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
    oracle.close()
    assert ledger.logical_unique_queries == 2
    assert ledger.physical_forwards_caused == 1
