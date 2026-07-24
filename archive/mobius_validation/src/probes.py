from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .schema import FeatureSpec, ProbeSpec


PROBE_STRATEGIES = {"contiguous_random", "dispersed_random", "stratified_position"}


def build_probe(
    feature_spec: FeatureSpec,
    *,
    k: int,
    strategy: str,
    seed: int,
    probe_index: int,
    conditioning_mode: str = "rest_present",
) -> ProbeSpec:
    n = feature_spec.n_features
    size = int(k)
    if size <= 0 or size > n:
        raise ValueError(f"Invalid probe size k={k} for n={n}")
    mode = str(strategy).strip().lower()
    if mode not in PROBE_STRATEGIES:
        raise ValueError(f"Unsupported probe strategy: {strategy!r}")
    rng = np.random.default_rng(int(seed) + int(probe_index) * 1009 + size)

    if mode == "contiguous_random":
        start = int(rng.integers(0, n - size + 1))
        indices = list(range(start, start + size))
    elif mode == "dispersed_random":
        indices = sorted(int(x) for x in rng.choice(n, size=size, replace=False).tolist())
    else:
        thirds = np.array_split(np.arange(n), 3)
        picks: List[int] = []
        base = size // 3
        remainder = size % 3
        for bucket_idx, bucket in enumerate(thirds):
            take = base + (1 if bucket_idx < remainder else 0)
            if take <= 0:
                continue
            take = min(take, len(bucket))
            picks.extend(int(x) for x in rng.choice(bucket, size=take, replace=False).tolist())
        while len(set(picks)) < size:
            picks.append(int(rng.integers(0, n)))
        indices = sorted(set(picks))[:size]

    words = [feature_spec.features[idx].word_text for idx in indices]
    return ProbeSpec(
        sample_id=feature_spec.sample_id,
        probe_id=f"{feature_spec.sample_id}-probe-{mode}-k{size}-{probe_index}",
        probe_strategy=mode,
        probe_word_indices=[int(x) for x in indices],
        probe_word_texts=words,
        k=int(size),
        conditioning_mode=str(conditioning_mode),
        random_seed=int(seed),
    )


def build_probes(
    feature_spec: FeatureSpec,
    *,
    probe_sizes: Sequence[int],
    strategies: Sequence[str],
    probes_per_sample: int,
    seed: int,
    conditioning_mode: str = "rest_present",
) -> List[ProbeSpec]:
    probes: List[ProbeSpec] = []
    idx = 0
    for size in probe_sizes:
        if int(size) > feature_spec.n_features:
            continue
        for strategy in strategies:
            for _ in range(int(probes_per_sample)):
                probes.append(
                    build_probe(
                        feature_spec,
                        k=int(size),
                        strategy=strategy,
                        seed=int(seed),
                        probe_index=idx,
                        conditioning_mode=conditioning_mode,
                    )
                )
                idx += 1
    return probes

