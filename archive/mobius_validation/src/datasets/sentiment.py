from __future__ import annotations

from typing import Any, Dict, List

from ..schema import TextRecord


_DEFAULT_VERBALIZERS = {
    "sst2": ["negative", "positive"],
    "rotten_tomatoes": ["negative", "positive"],
    "rtn": ["negative", "positive"],
    "eraser": ["negative", "positive"],
    "eraser_movie_reviews": ["negative", "positive"],
    "imdb": ["negative", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "inline_sentiment": ["negative", "positive"],
}


def verbalizers_for_dataset(name: str, config: Dict[str, Any] | None = None) -> List[str]:
    if config and config.get("verbalizers"):
        return [str(x) for x in config["verbalizers"]]
    key = str(name).strip().lower()
    return list(_DEFAULT_VERBALIZERS.get(key, ["negative", "positive"]))


def _records_from_inline(config: Dict[str, Any]) -> List[TextRecord]:
    records = []
    task = str(config.get("task", config.get("name", "inline_sentiment")))
    for idx, row in enumerate(config.get("samples", [])):
        records.append(
            TextRecord(
                sample_id=str(row.get("sample_id", f"{task}-{idx}")),
                text=str(row.get("text", "")),
                label=int(row["label"]) if row.get("label") is not None else None,
                label_text=str(row.get("label_text")) if row.get("label_text") is not None else None,
                task=task,
                metadata={"source": "inline"},
            )
        )
    return [record for record in records if record.text != ""]


def _records_from_lima(config: Dict[str, Any]) -> List[TextRecord]:
    from lima_llm.data import load_dataset_bundle

    bundle = load_dataset_bundle(
        dataset_name=str(config["name"]),
        split=str(config.get("split", "validation")),
        max_samples=config.get("max_samples"),
        eraser_root=config.get("eraser_root"),
        sst2_source=config.get("sst2_source"),
        dataset_cache_dir=config.get("dataset_cache_dir"),
    )
    records = []
    for sample in bundle.samples:
        records.append(
            TextRecord(
                sample_id=str(sample.sample_id),
                text=str(sample.text),
                label=int(sample.label),
                label_text=str(sample.label_text) if sample.label_text is not None else None,
                task=str(bundle.dataset_name),
                metadata=dict(sample.metadata or {}),
            )
        )
    return records


def load_sentiment_records(config: Dict[str, Any]) -> List[TextRecord]:
    source = str(config.get("source", config.get("name", "inline_sentiment"))).strip().lower()
    if source in {"inline", "inline_sentiment"} or config.get("samples") is not None:
        return _records_from_inline(config)
    return _records_from_lima(config)

