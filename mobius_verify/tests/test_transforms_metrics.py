from __future__ import annotations

import numpy as np

from mobius_verify.src.hierarchy_metrics import hierarchy_at_k
from mobius_verify.src.oracle_approximation import omp_curve
from mobius_verify.src.reconstruction_metrics import degree_curves
from mobius_verify.src.transforms import (
    fourier_transform,
    inverse_fourier_transform,
    inverse_mobius_transform,
    mobius_transform,
)


def test_mobius_and_fourier_inverse_roundtrip() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=16)
    mobius = mobius_transform(values)
    fourier = fourier_transform(values)
    assert np.max(np.abs(inverse_mobius_transform(mobius) - values)) < 1e-10
    assert np.max(np.abs(inverse_fourier_transform(fourier) - values)) < 1e-10


def test_additive_function_has_degree_one_mobius_reconstruction() -> None:
    values = np.zeros(8)
    for mask in range(8):
        values[mask] = (mask & 1) + 2 * ((mask >> 1) & 1) - ((mask >> 2) & 1)
    mobius = mobius_transform(values)
    fourier = fourier_transform(values)
    curves = degree_curves(values, mobius, fourier, d_max=3)
    assert curves["mobius"][0]["r2"] > 0.999999


def test_hierarchy_detects_orphan_fourier_peak() -> None:
    coeff = np.zeros(16)
    coeff[0b111] = 1.0
    metrics = hierarchy_at_k(coeff, k=1)
    assert metrics["high_order_count"] == 1
    assert metrics["orphan_ratio"] == 1.0


def test_omp_curve_is_monotonic() -> None:
    values = np.zeros(16)
    values[0b0011:] = values[0b0011:]
    for mask in range(16):
        values[mask] = 1.0 if (mask & 0b0011) == 0b0011 else 0.0
    curve = omp_curve(values, basis="mobius", d_max=2, max_k=5)["curve"]
    r2_values = [row["r2"] for row in curve]
    assert r2_values == sorted(r2_values)
    assert r2_values[-1] > 0.99

