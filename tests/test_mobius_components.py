"""Numerical and contract tests for each sparse-method ablation component."""

from __future__ import annotations

import numpy as np

from mobius.methods.sparse.basis import design_matrix, low_degree_terms
from mobius.methods.sparse.estimator import SparseModel
from mobius.methods.sparse.hierarchy import strong_heredity_terms
from mobius.methods.sparse.projector import project_nodes
from mobius.methods.sparse.sampler import sample_masks
from mobius.evaluation.metrics import aupc_from_probabilities


def test_deletion_and_fourier_basis_values() -> None:
    """Check exact columns for all two-player coalition masks."""

    masks = [0, 1, 2, 3]
    terms = [1, 2, 3]
    deletion = design_matrix(
        masks,
        terms,
        n_features=2,
        basis="deletion_mobius",
    )
    fourier = design_matrix(
        masks,
        terms,
        n_features=2,
        basis="fourier",
    )
    assert deletion.tolist() == [
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert fourier.tolist() == [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]


def test_aupc_uses_normalized_trapezoidal_deletion_curve() -> None:
    """Check AUPC from already-scored trajectory points without new queries."""

    assert np.isclose(aupc_from_probabilities([1.0, 0.6, 0.2]), 0.6)
    assert np.isclose(aupc_from_probabilities([0.7]), 0.7)
    assert np.isclose(aupc_from_probabilities([]), 0.0)


def test_strong_hierarchy_only_builds_interactions_with_selected_parents() -> None:
    """Ensure strong heredity changes candidates before final fitting."""

    terms = strong_heredity_terms([1, 4, 8], max_degree=2)
    assert terms == [1, 4, 8, 5, 9, 12]
    assert 3 not in terms


def test_all_samplers_respect_unique_exact_budget_and_anchors() -> None:
    """Check exact budgets and deterministic uniqueness for every sampler."""

    for name in ("deletion_mixture", "bernoulli", "uniform_size"):
        result = sample_masks(
            8,
            40,
            seed=13,
            config={"name": name, "include_empty_full": True},
        )
        assert len(result.masks) == 40
        assert len(set(result.masks)) == 40
        assert 0 in result.masks
        assert (1 << 8) - 1 in result.masks
        assert result.diagnostics["realized_budget"] == 40


def test_sampler_enumerates_when_budget_covers_universe() -> None:
    """Check the shared exhaustive rule for a small coalition universe."""

    for name in ("deletion_mixture", "bernoulli", "uniform_size"):
        result = sample_masks(3, 99, seed=2, config={"name": name})
        assert result.masks == list(range(8))
        assert result.diagnostics["source_counts"] == {"exhaustive": 8}


def _model() -> SparseModel:
    """Build a fixed deletion model for projector conservation tests."""

    return SparseModel(
        basis="deletion_mobius",
        n_features=2,
        max_degree=2,
        intercept=0.0,
        terms=[1, 2, 3],
        coefficients=np.asarray([-2.0, 1.0, -4.0]),
        selection_coefficients=np.asarray([-2.0, 1.0, -4.0]),
        diagnostics={},
    )


def test_projectors_change_only_hyperedge_allocation() -> None:
    """Verify signed, absolute, and singleton projections on one fitted model."""

    signed = project_nodes(_model(), "signed_equal_share")
    absolute = project_nodes(_model(), "absolute_equal_share")
    singleton = project_nodes(_model(), "singleton_only")
    assert np.allclose(signed, [4.0, 1.0])
    assert np.allclose(absolute, [4.0, 3.0])
    assert np.allclose(singleton, [2.0, -1.0])
    assert np.isclose(np.sum(signed), -np.sum(_model().coefficients))
    assert np.isclose(np.sum(absolute), np.sum(np.abs(_model().coefficients)))


def test_low_degree_candidate_count_matches_order_two_formula() -> None:
    """Check candidate enumeration for the normal degree-two experiment."""

    assert len(low_degree_terms(10, 2)) == 10 + 45
