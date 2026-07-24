from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .subset_enumeration import mask_popcount
from .transforms import infer_n_from_values


def direct_parent_terms(term: int) -> List[int]:
    return [int(term) & ~(1 << bit) for bit in range(int(term).bit_length()) if int(term) & (1 << bit)]


def nonempty_subset_terms(term: int) -> List[int]:
    out = []
    sub = int(term)
    while sub:
        out.append(sub)
        sub = (sub - 1) & int(term)
    return out


def top_fourier_terms(coefficients: Sequence[float], k: int) -> List[int]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    infer_n_from_values(coeff)
    terms = list(range(1, len(coeff)))
    terms.sort(key=lambda mask: (-abs(float(coeff[mask])), mask))
    return terms[: min(int(k), len(terms))]


def hierarchy_at_k(coefficients: Sequence[float], k: int) -> Dict[str, object]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    top = top_fourier_terms(coeff, int(k))
    top_set = set(top)
    high_order = [term for term in top if mask_popcount(term) >= 2]

    dsr_values = []
    shr_values = []
    orphan_terms = []
    by_degree: Dict[int, Dict[str, List[float]]] = {}
    for term in top:
        degree = mask_popcount(term)
        if degree < 2:
            continue
        parents = direct_parent_terms(term)
        parent_hits = sum(1 for parent in parents if parent in top_set)
        dsr = float(parent_hits / degree) if degree > 0 else 0.0
        subsets = [sub for sub in nonempty_subset_terms(term) if sub != term]
        shr = 1.0 if subsets and all(sub in top_set for sub in subsets) else 0.0
        dsr_values.append(dsr)
        shr_values.append(shr)
        by_degree.setdefault(degree, {"dsr": [], "shr": []})
        by_degree[degree]["dsr"].append(dsr)
        by_degree[degree]["shr"].append(shr)
        if parent_hits == 0:
            orphan_terms.append(term)

    denom_energy = float(sum(float(coeff[term]) ** 2 for term in high_order))
    orphan_energy = float(sum(float(coeff[term]) ** 2 for term in orphan_terms))
    degree_payload = {
        str(degree): {
            "dsr": float(np.mean(values["dsr"])) if values["dsr"] else None,
            "shr": float(np.mean(values["shr"])) if values["shr"] else None,
            "count": int(len(values["dsr"])),
        }
        for degree, values in sorted(by_degree.items())
    }
    return {
        "k": int(k),
        "top_term_count": int(len(top)),
        "high_order_count": int(len(high_order)),
        "dsr": float(np.mean(dsr_values)) if dsr_values else None,
        "shr": float(np.mean(shr_values)) if shr_values else None,
        "dsr_by_degree": degree_payload,
        "shr_by_degree": {degree: payload["shr"] for degree, payload in degree_payload.items()},
        "orphan_ratio": float(len(orphan_terms) / len(high_order)) if high_order else None,
        "orphan_spectral_energy_ratio": float(orphan_energy / denom_energy) if denom_energy > 0 else None,
        "orphan_terms": [int(term) for term in orphan_terms],
    }


def hierarchy_curve(coefficients: Sequence[float], k_values: Sequence[int]) -> List[Dict[str, object]]:
    return [hierarchy_at_k(coefficients, int(k)) for k in k_values]

