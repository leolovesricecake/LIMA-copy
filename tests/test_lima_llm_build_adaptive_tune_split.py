from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mk_samples(n: int):
    rows = []
    for idx in range(n):
        rows.append(SimpleNamespace(sample_id=f"s{idx}", label=(idx % 2)))
    return rows


def test_stratified_split_is_reproducible() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_adaptive_tune_split.py"
    mod = _load_module(script, "build_adaptive_tune_split")

    samples = _mk_samples(40)
    a_train, a_dev = mod._stratified_split(samples=samples, train_size=12, dev_size=8, seed=42)
    b_train, b_dev = mod._stratified_split(samples=samples, train_size=12, dev_size=8, seed=42)

    assert a_train == b_train
    assert a_dev == b_dev
    assert len(a_train) == 12
    assert len(a_dev) == 8
    assert set(a_train).isdisjoint(set(a_dev))


def test_load_bundle_with_fallback_uses_requested_split_when_train_missing(monkeypatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_adaptive_tune_split.py"
    mod = _load_module(script, "build_adaptive_tune_split_fallback")

    calls = []

    def _fake_loader(*, dataset_name, split, max_samples, eraser_root, sst2_source, dataset_cache_dir):
        calls.append(split)
        if split == "train":
            raise RuntimeError("missing train")
        return SimpleNamespace(samples=_mk_samples(5), dataset_name=dataset_name, split=split)

    monkeypatch.setattr(mod, "load_dataset_bundle", _fake_loader)
    bundle, source = mod._load_bundle_with_fallback(
        dataset="rotten_tomatoes",
        split="validation",
        prefer_train=True,
        eraser_root=None,
        sst2_source=None,
        dataset_cache_dir=None,
    )
    assert calls[0] == "train"
    assert calls[-1] == "validation"
    assert source == "validation"
    assert len(bundle.samples) == 5
