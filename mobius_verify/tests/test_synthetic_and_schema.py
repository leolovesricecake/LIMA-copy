from __future__ import annotations

from mobius_verify.src.exact_analysis import analyze_value_table
from mobius_verify.src.synthetic import generate_synthetic_suite


def test_synthetic_suite_has_expected_families_and_analysis_schema() -> None:
    instances = generate_synthetic_suite(n_features=5, instances_per_family=1, seed=0)
    families = {instance.family for instance in instances}
    assert {
        "additive",
        "hierarchical_fourier",
        "nonhierarchical_fourier",
        "sparse_mobius",
        "dense_low_degree",
    } <= families
    analysis = analyze_value_table(instances[0].values, d_max=3, omp_max_k=8)
    assert analysis["n_features"] == 5
    assert analysis["inverse_errors"]["mobius_max_abs"] < 1e-8
    assert analysis["inverse_errors"]["fourier_max_abs"] < 1e-8
    assert "degree_summary" in analysis
    assert "sparsity_summary" in analysis

