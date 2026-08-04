"""Focused text-classification loader with an offline Arrow-cache fallback."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from mobius.core.schema import DatasetBundle, TextSample


DATASETS: Dict[str, Dict[str, Any]] = {
    "sst2": {
        "hf_id": "nyu-mll/glue",
        "hf_config": "sst2",
        "cache_configs": ["sst2"],
        "text_field": "sentence",
        "labels": ["negative", "positive"],
        "cache_names": ["glue", "nyu-mll___glue"],
    },
    "rotten_tomatoes": {
        "hf_id": "cornell-movie-review-data/rotten_tomatoes",
        "hf_config": None,
        "cache_configs": ["default"],
        "text_field": "text",
        "labels": ["negative", "positive"],
        "cache_names": [
            "rotten_tomatoes",
            "cornell-movie-review-data___rotten_tomatoes",
        ],
    },
    "emotion": {
        "hf_id": "dair-ai/emotion",
        "hf_config": None,
        "cache_configs": ["split", "default"],
        "text_field": "text",
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "cache_names": ["emotion", "dair-ai___emotion"],
    },
    "ag_news": {
        "hf_id": "wangrongsheng/ag_news",
        "hf_config": None,
        "cache_configs": ["default"],
        "text_field": "text",
        "labels": ["world", "sports", "business", "technology"],
        "cache_names": ["ag_news", "wangrongsheng___ag_news"],
        "fixed_verbalizers": True,
    },
    "imdb": {
        "hf_id": "imdb",
        "hf_config": None,
        "cache_configs": ["plain_text", "default"],
        "text_field": "text",
        "labels": ["negative", "positive"],
        "cache_names": ["imdb"],
    },
    "eraser_movie_reviews": {
        "hf_id": "eraser-benchmark/movie_rationales",
        "hf_config": None,
        "cache_configs": ["default"],
        "text_field": "review",
        "labels": ["negative", "positive"],
        "cache_names": [
            "movie_rationales",
            "eraser-benchmark___movie_rationales",
        ],
    },
}


def _canonical_name(name: str) -> str:
    """Normalize supported dataset aliases."""

    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "glue_sst2": "sst2",
        "rtn": "rotten_tomatoes",
        "agnews": "ag_news",
        "agn": "ag_news",
        "eraser": "eraser_movie_reviews",
    }
    return aliases.get(normalized, normalized)


def _canonical_split(split: str, dataset_name: str) -> str:
    """Normalize common validation aliases and IMDB's test-only convention."""

    normalized = str(split).strip().lower()
    if normalized in {"dev", "val"}:
        normalized = "validation"
    if dataset_name == "imdb" and normalized == "validation":
        return "test"
    return normalized


def _cache_root(cache_dir: str | None) -> Path | None:
    """Resolve a Hugging Face cache root when one was configured."""

    return Path(cache_dir).expanduser().resolve() if cache_dir else None


def _cached_arrow_files(
    cache_dir: str | None,
    cache_names: Sequence[str],
    split: str,
    *,
    cache_configs: Sequence[str] | None = None,
) -> List[Path]:
    """Find the newest cached split within the requested dataset configuration."""

    root = _cache_root(cache_dir)
    if root is None:
        return []
    roots = [root, root / "datasets"]
    split_pattern = re.compile(rf"(?:^|-){re.escape(split)}(?:-|$)")
    groups: Dict[Path, List[Path]] = {}
    for search_root in roots:
        for cache_name in cache_names:
            candidate = search_root / cache_name
            if not candidate.is_dir():
                continue
            # A Hugging Face cache stores configurations directly below the
            # dataset directory. Restricting this level prevents, for example,
            # SST-2 from accidentally selecting a newer MRPC cache under GLUE.
            config_roots = (
                [candidate / str(config_name) for config_name in cache_configs]
                if cache_configs
                else [candidate]
            )
            for config_root in config_roots:
                if not config_root.is_dir():
                    continue
                for arrow_path in config_root.rglob("*.arrow"):
                    if split_pattern.search(arrow_path.stem):
                        groups.setdefault(arrow_path.parent, []).append(arrow_path)
    if not groups:
        return []
    newest = max(
        groups,
        key=lambda parent: max(path.stat().st_mtime_ns for path in groups[parent]),
    )
    return sorted(groups[newest])


def _load_cached_arrow(
    cache_dir: str | None,
    cache_names: Sequence[str],
    split: str,
    *,
    cache_configs: Sequence[str] | None = None,
    text_field: str,
):
    """Load cached Arrow shards and validate their configured text schema."""

    files = _cached_arrow_files(
        cache_dir,
        cache_names,
        split,
        cache_configs=cache_configs,
    )
    if not files:
        return None
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as exc:
        raise RuntimeError("Loading datasets requires the `datasets` package.") from exc
    shards = [Dataset.from_file(str(path)) for path in files]
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    columns = set(getattr(dataset, "column_names", ()))
    if text_field not in columns:
        raise ValueError(
            "Cached dataset schema does not match its configured text field: "
            f"expected {text_field!r}, found {sorted(columns)!r} under {files[0].parent}."
        )
    print(f"[dataset-cache] loaded {split} from {files[0].parent}")
    return dataset


def _extract_text(row: Mapping[str, Any], text_field: str) -> str:
    """Read text only from the field declared by the dataset specification."""

    value = row.get(text_field)
    return "" if value is None else str(value)


def _label_names(dataset, fallback: Sequence[str]) -> List[str]:
    """Prefer ClassLabel names while retaining configured stable verbalizers."""

    try:
        names = dataset.features["label"].names
    except (AttributeError, KeyError, TypeError):
        names = None
    return [str(value) for value in names] if names else [str(value) for value in fallback]


def _bundle_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    split: str,
    labels: Sequence[str],
    text_field: str,
    max_samples: int | None,
    source: str,
) -> DatasetBundle:
    """Convert row mappings into the shared DatasetBundle schema."""

    samples: List[TextSample] = []
    for row_index, row in enumerate(rows):
        text = _extract_text(row, text_field)
        raw_label = row.get("label", row.get("classification"))
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            key = str(raw_label).strip().lower()
            aliases = {
                "neg": 0,
                "negative": 0,
                "pos": 1,
                "positive": 1,
            }
            label = aliases.get(key, -1)
        if not text or label < 0 or label >= len(labels):
            continue
        samples.append(
            TextSample(
                sample_id=f"{dataset_name}-{split}-{row_index}",
                text=text,
                label=label,
                label_text=str(labels[label]),
                metadata={"source": source},
            )
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
    if not samples:
        raise ValueError(f"No valid rows found for {dataset_name}/{split}.")
    return DatasetBundle(
        dataset_name=dataset_name,
        split=split,
        samples=samples,
        label_names=list(labels),
        verbalizers=list(labels),
    )


def _read_local_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    """Yield rows from JSONL, JSON, CSV, or TSV files."""

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                rows = payload if isinstance(payload, list) else payload.get("data", [])
                yield from rows
            else:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
        return
    raise ValueError(f"Unsupported local dataset file: {path}")


def _resolve_local_split(source: str, split: str) -> Path:
    """Resolve a local file or a split file inside a local directory."""

    path = Path(source).expanduser()
    if path.is_file():
        return path
    aliases = [split] + (["dev", "val"] if split == "validation" else [])
    for alias in aliases:
        for suffix in (".jsonl", ".json", ".tsv", ".csv"):
            candidate = path / f"{alias}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No local split file for {split} under {path}")


def load_dataset_bundle(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    eraser_root: Optional[str] = None,
    sst2_source: Optional[str] = None,
    dataset_cache_dir: Optional[str] = None,
    source: Optional[str] = None,
) -> DatasetBundle:
    """Load a supported dataset, preferring direct local cache access."""

    name = _canonical_name(dataset_name)
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    spec = DATASETS[name]
    resolved_split = _canonical_split(split, name)
    local_source = source or (sst2_source if name == "sst2" else None)
    if name == "eraser_movie_reviews" and eraser_root:
        eraser_value = str(eraser_root)
        if eraser_value.startswith("hf://"):
            spec = {**spec, "hf_id": eraser_value[len("hf://") :]}
        else:
            local_source = eraser_value
    if local_source and Path(str(local_source)).expanduser().exists():
        path = _resolve_local_split(str(local_source), resolved_split)
        return _bundle_from_rows(
            _read_local_rows(path),
            dataset_name=name,
            split=resolved_split,
            labels=spec["labels"],
            text_field=spec["text_field"],
            max_samples=max_samples,
            source=str(path),
        )

    dataset = _load_cached_arrow(
        dataset_cache_dir,
        spec["cache_names"],
        resolved_split,
        cache_configs=spec["cache_configs"],
        text_field=spec["text_field"],
    )
    if dataset is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Loading datasets requires the `datasets` package.") from exc
        kwargs: Dict[str, Any] = {"split": resolved_split}
        if dataset_cache_dir:
            kwargs["cache_dir"] = str(_cache_root(dataset_cache_dir))
        if spec["hf_config"]:
            dataset = load_dataset(spec["hf_id"], spec["hf_config"], **kwargs)
        else:
            dataset = load_dataset(spec["hf_id"], **kwargs)
    labels = (
        [str(value) for value in spec["labels"]]
        if bool(spec.get("fixed_verbalizers"))
        else _label_names(dataset, spec["labels"])
    )
    return _bundle_from_rows(
        dataset,
        dataset_name=name,
        split=resolved_split,
        labels=labels,
        text_field=spec["text_field"],
        max_samples=max_samples,
        source=f"hf://{spec['hf_id']}",
    )
