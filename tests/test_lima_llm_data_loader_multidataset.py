from __future__ import annotations

import sys
import types
from pathlib import Path

from lima_llm.data.loader import _normalize_hf_dataset_ref, load_dataset_bundle


class _FakeLabelFeature:
    def __init__(self, names):
        self.names = list(names)


class _FakeDataset(list):
    def __init__(self, rows, label_names):
        super().__init__(rows)
        self.features = {"label": _FakeLabelFeature(label_names)}


def _install_fake_datasets(monkeypatch, load_dataset_impl) -> None:
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = load_dataset_impl
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_imdb_validation_split_maps_to_test(monkeypatch) -> None:
    calls = []

    def _load_dataset(dataset_id, split):
        calls.append((dataset_id, split))
        return _FakeDataset(
            rows=[
                {"text": "great movie", "label": 1},
                {"text": "bad movie", "label": 0},
            ],
            label_names=["negative", "positive"],
        )

    _install_fake_datasets(monkeypatch, _load_dataset)
    bundle = load_dataset_bundle(dataset_name="imdb", split="validation", max_samples=1)

    assert calls == [("imdb", "test")]
    assert bundle.dataset_name == "imdb"
    assert bundle.split == "validation"
    assert bundle.label_names == ["negative", "positive"]
    assert bundle.verbalizers == ["negative", "positive"]
    assert len(bundle.samples) == 1
    assert bundle.samples[0].label == 1
    assert bundle.samples[0].label_text == "positive"
    assert bundle.samples[0].rationale_char_spans == ()
    assert bundle.samples[0].metadata["hf_split"] == "test"


def test_rotten_tomatoes_hf_loading(monkeypatch) -> None:
    calls = []

    def _load_dataset(dataset_id, split):
        calls.append((dataset_id, split))
        return _FakeDataset(
            rows=[
                {"text": "solid writing", "label": 1},
                {"text": "messy plot", "label": 0},
            ],
            label_names=["negative", "positive"],
        )

    _install_fake_datasets(monkeypatch, _load_dataset)
    bundle = load_dataset_bundle(dataset_name="rotten_tomatoes", split="validation", max_samples=10)

    assert calls == [("cornell-movie-review-data/rotten_tomatoes", "validation")]
    assert bundle.dataset_name == "rotten_tomatoes"
    assert len(bundle.samples) == 2
    assert bundle.samples[0].label_text == "positive"
    assert bundle.samples[1].label_text == "negative"
    assert all(sample.rationale_char_spans == () for sample in bundle.samples)


def test_legacy_rotten_tomatoes_refs_are_namespaced() -> None:
    expected = "cornell-movie-review-data/rotten_tomatoes"
    assert _normalize_hf_dataset_ref("rotten_tomatoes") == expected
    assert _normalize_hf_dataset_ref("hf://datasets/rotten_tomatoes") == expected


def test_rotten_tomatoes_uses_legacy_arrow_cache_without_hub(
    monkeypatch,
    tmp_path: Path,
) -> None:
    arrow_path = (
        tmp_path
        / "datasets"
        / "rotten_tomatoes"
        / "default"
        / "0.0.0"
        / "revision"
        / "rotten_tomatoes-validation.arrow"
    )
    arrow_path.parent.mkdir(parents=True)
    arrow_path.touch()
    calls = []

    class _FakeDatasetClass:
        @staticmethod
        def from_file(path):
            calls.append(path)
            return _FakeDataset(
                rows=[{"text": "cached review", "label": 1}],
                label_names=["negative", "positive"],
            )

    fake_module = types.ModuleType("datasets")
    fake_module.Dataset = _FakeDatasetClass
    fake_module.concatenate_datasets = lambda shards: shards[0]
    fake_module.load_dataset = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Hub loading must not run when the legacy Arrow cache exists")
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bundle = load_dataset_bundle(
        dataset_name="rotten_tomatoes",
        split="validation",
        dataset_cache_dir=str(tmp_path),
    )

    assert calls == [str(arrow_path.resolve())]
    assert len(bundle.samples) == 1
    assert bundle.samples[0].text == "cached review"


def test_emotion_supports_label_name_rows(monkeypatch) -> None:
    def _load_dataset(dataset_id, split):
        assert dataset_id == "dair-ai/emotion"
        assert split == "validation"
        return _FakeDataset(
            rows=[
                {"text": "I feel amazing", "label": "joy"},
                {"text": "I am upset", "label": "sadness"},
                {"text": "skip me", "label": "unknown"},
            ],
            label_names=["sadness", "joy", "love", "anger", "fear", "surprise"],
        )

    _install_fake_datasets(monkeypatch, _load_dataset)
    bundle = load_dataset_bundle(dataset_name="emotion", split="validation", max_samples=10)

    assert bundle.dataset_name == "emotion"
    assert bundle.label_names == ["sadness", "joy", "love", "anger", "fear", "surprise"]
    assert len(bundle.samples) == 2
    assert bundle.samples[0].label == 1
    assert bundle.samples[0].label_text == "joy"
    assert bundle.samples[1].label == 0
    assert bundle.samples[1].label_text == "sadness"
