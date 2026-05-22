from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import TextSample


@dataclass
class DatasetBundle:
    dataset_name: str
    split: str
    samples: List[TextSample]
    label_names: List[str]
    verbalizers: List[str]


_SST2_LABELS = ["negative", "positive"]
_DEFAULT_ERASER_HF_DATASET = "eraser-benchmark/movie_rationales"
_DEFAULT_IMDB_HF_DATASET = "imdb"
_DEFAULT_ROTTEN_TOMATOES_HF_DATASET = "rotten_tomatoes"
_DEFAULT_EMOTION_HF_DATASET = "dair-ai/emotion"


def _canonical_split(split: str) -> str:
    x = split.strip().lower()
    if x in {"val", "dev"}:
        return "validation"
    return x


def _cache_dir(dataset_cache_dir: Optional[str]) -> Path:
    if dataset_cache_dir:
        return Path(dataset_cache_dir).expanduser().resolve()
    return Path("~/.cache/lima_llm/datasets").expanduser().resolve()


def _is_remote_uri(source: str) -> bool:
    scheme = urllib.parse.urlparse(source).scheme.lower()
    return scheme in {"http", "https"}


def _is_hf_dataset_ref(source: str) -> bool:
    text = source.strip()
    if text.startswith("hf://"):
        return True
    if _is_remote_uri(text):
        return False
    maybe_path = Path(text).expanduser()
    if maybe_path.exists():
        return False
    parts = text.split("/")
    if len(parts) >= 2 and all(p.strip() != "" for p in parts):
        return True
    return False


def _normalize_hf_dataset_ref(source: str) -> str:
    text = source.strip()
    if text.startswith("hf://"):
        return text[len("hf://") :]
    return text


def _download_to_cache(source_url: str, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = cache_root / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(source_url)
    filename = Path(parsed.path).name or "download.bin"
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    target = downloads_dir / f"{digest}_{filename}"
    if target.exists():
        return target

    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(source_url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, target)
    return target


def _extract_if_needed(downloaded_file: Path, cache_root: Path) -> Path:
    name_lower = downloaded_file.name.lower()

    extract_key = hashlib.sha256(str(downloaded_file).encode("utf-8")).hexdigest()[:16]
    extract_root = cache_root / "extracted" / extract_key

    if name_lower.endswith(".zip"):
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(downloaded_file, "r") as zf:
                zf.extractall(extract_root)
        return extract_root

    tar_suffixes = (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    )
    if name_lower.endswith(tar_suffixes):
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(downloaded_file, "r:*") as tf:
                tf.extractall(extract_root)
        return extract_root

    if name_lower.endswith(".gz") and not name_lower.endswith((".tar.gz", ".tgz")):
        out_file = extract_root / downloaded_file.name[:-3]
        if not out_file.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with gzip.open(downloaded_file, "rb") as fin, open(out_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return out_file

    return downloaded_file


def _resolve_source_to_local_path(source: str, dataset_cache_dir: Optional[str]) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve()

    if _is_remote_uri(source):
        cache_root = _cache_dir(dataset_cache_dir)
        downloaded = _download_to_cache(source, cache_root)
        resolved = _extract_if_needed(downloaded, cache_root)
        return resolved.resolve()

    raise FileNotFoundError(f"Source not found: {source}")


def _parse_label(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        if raw in {0, 1}:
            return raw
        return None
    text = str(raw).strip().lower()
    if text in {"0", "neg", "negative", "false"}:
        return 0
    if text in {"1", "pos", "positive", "true"}:
        return 1
    return None


def _parse_non_negative_int(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("-"):
        return None
    if text.isdigit():
        return int(text)
    return None


def _hf_label_names(ds) -> List[str]:
    features = getattr(ds, "features", None)
    if features is None:
        return []
    try:
        label_feature = features.get("label")
    except Exception:
        label_feature = None
    if label_feature is None:
        return []
    names = getattr(label_feature, "names", None)
    if not isinstance(names, (list, tuple)) or len(names) == 0:
        return []
    return [str(name) for name in names]


def _canonical_hf_split_for_dataset(dataset_name: str, split: str) -> str:
    canon = _canonical_split(split)
    if dataset_name == "imdb" and canon == "validation":
        return "test"
    return canon


def _load_hf_text_classification_dataset(
    *,
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    dataset_ref: str,
) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            f"datasets package is required for {dataset_name}. Please install `datasets`."
        ) from exc

    dataset_id = _normalize_hf_dataset_ref(dataset_ref)
    hf_split = _canonical_hf_split_for_dataset(dataset_name, split)
    ds = load_dataset(dataset_id, split=hf_split)
    label_names = _hf_label_names(ds)
    label_name_to_id = {name.strip().lower(): idx for idx, name in enumerate(label_names)}
    dynamic_label_id_by_raw: Dict[str, int] = {}
    dynamic_label_text_by_id: Dict[int, str] = {}

    samples: List[TextSample] = []
    for idx, row in enumerate(ds):
        text = _extract_text_from_row(row)
        if text == "":
            continue

        raw_label = row.get("label")
        label: Optional[int] = None
        label_text: Optional[str] = None

        numeric_label = _parse_non_negative_int(raw_label)
        if label_names:
            if numeric_label is not None:
                if 0 <= numeric_label < len(label_names):
                    label = numeric_label
                    label_text = label_names[label]
            else:
                key = str(raw_label).strip().lower()
                if key in label_name_to_id:
                    label = int(label_name_to_id[key])
                    label_text = label_names[label]
        elif numeric_label is not None:
            raw_key = f"num::{numeric_label}"
            if raw_key not in dynamic_label_id_by_raw:
                dynamic_label_id_by_raw[raw_key] = len(dynamic_label_id_by_raw)
            label = dynamic_label_id_by_raw[raw_key]
            label_text = f"label_{numeric_label}"
        elif raw_label is not None:
            key = str(raw_label).strip()
            if key != "":
                raw_key = f"str::{key}"
                if raw_key not in dynamic_label_id_by_raw:
                    dynamic_label_id_by_raw[raw_key] = len(dynamic_label_id_by_raw)
                label = dynamic_label_id_by_raw[raw_key]
                label_text = key

        if label is None:
            continue
        if label_text is None:
            label_text = f"label_{label}"
        dynamic_label_text_by_id.setdefault(label, str(label_text))

        samples.append(
            TextSample(
                sample_id=f"{dataset_name}-hf-{hf_split}-{idx}",
                text=text,
                label=int(label),
                label_text=str(label_text),
                rationale_char_spans=(),
                metadata={
                    "source": f"hf://{dataset_id}",
                    "requested_split": str(split),
                    "hf_split": str(hf_split),
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not label_names:
        label_names = [dynamic_label_text_by_id[i] for i in range(len(dynamic_label_text_by_id))]
    verbalizers = list(label_names)
    if not verbalizers:
        raise ValueError(f"No valid labels found for dataset={dataset_name} split={split}")

    return DatasetBundle(
        dataset_name=dataset_name,
        split=split,
        samples=samples,
        label_names=label_names,
        verbalizers=verbalizers,
    )


def _extract_text_from_row(row: Dict) -> str:
    for key in ["sentence", "text", "review", "document"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def _load_rows_from_plain_file(path: Path) -> Iterable[Dict]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                yield json.loads(line)
        return

    delimiter = "\t" if suffix == ".tsv" else ","
    if suffix in {".csv", ".tsv"}:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                yield dict(row)
        return

    raise ValueError(f"Unsupported local file format: {path}")


def _load_sst2_from_local_source(
    split: str,
    source: str,
    max_samples: Optional[int],
    dataset_cache_dir: Optional[str],
) -> DatasetBundle:
    resolved = _resolve_source_to_local_path(source, dataset_cache_dir=dataset_cache_dir)
    split_name = _canonical_split(split)

    candidate_files: List[Path] = []
    if resolved.is_file():
        candidate_files = [resolved]
    else:
        aliases = [split_name]
        if split_name == "validation":
            aliases += ["dev", "val"]

        for alias in aliases:
            candidate_files.extend(
                [
                    resolved / f"{alias}.tsv",
                    resolved / f"{alias}.csv",
                    resolved / f"{alias}.jsonl",
                    resolved / "sst2" / f"{alias}.tsv",
                    resolved / "sst2" / f"{alias}.csv",
                    resolved / "sst2" / f"{alias}.jsonl",
                ]
            )

        if len(candidate_files) == 0:
            raise FileNotFoundError(f"No SST-2 split files found under: {resolved}")

    existing_files = [p for p in candidate_files if p.exists()]
    if len(existing_files) == 0:
        raise FileNotFoundError(
            "No SST-2 files found for split '{}'. Tried: {}".format(
                split_name, ", ".join(str(p) for p in candidate_files)
            )
        )

    samples: List[TextSample] = []
    for file_path in existing_files:
        for idx, row in enumerate(_load_rows_from_plain_file(file_path)):
            text = _extract_text_from_row(row)
            label = _parse_label(row.get("label"))
            if label is None or text == "":
                continue
            samples.append(
                TextSample(
                    sample_id=f"sst2-local-{split_name}-{len(samples)}",
                    text=text,
                    label=label,
                    label_text=_SST2_LABELS[label],
                    rationale_char_spans=(),
                    metadata={
                        "source": "local/sst2",
                        "path": str(file_path),
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    if len(samples) == 0:
        raise ValueError(f"No valid SST-2 rows parsed from source: {source}")

    return DatasetBundle(
        dataset_name="sst2",
        split=split,
        samples=samples,
        label_names=list(_SST2_LABELS),
        verbalizers=list(_SST2_LABELS),
    )


def _load_sst2_from_hf(
    split: str,
    max_samples: Optional[int],
    dataset_ref: str = "nyu-mll/glue",
) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for SST-2. Please install `datasets`.") from exc

    hf_split = _canonical_split(split)

    # For GLUE wrapper datasets, we need subset/config = sst2.
    dataset_id = _normalize_hf_dataset_ref(dataset_ref)
    if dataset_id in {"glue", "nyu-mll/glue"}:
        ds = load_dataset(dataset_id, "sst2", split=hf_split)
    else:
        ds = load_dataset(dataset_id, split=hf_split)

    samples: List[TextSample] = []
    for idx, row in enumerate(ds):
        text = _extract_text_from_row(row)
        label = _parse_label(row.get("label"))
        if label is None or text == "":
            continue
        samples.append(
            TextSample(
                sample_id=f"sst2-{hf_split}-{idx}",
                text=text,
                label=label,
                label_text=_SST2_LABELS[label],
                rationale_char_spans=(),
                metadata={
                    "source": f"hf://{dataset_id}",
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return DatasetBundle(
        dataset_name="sst2",
        split=split,
        samples=samples,
        label_names=list(_SST2_LABELS),
        verbalizers=list(_SST2_LABELS),
    )


def _candidate_eraser_paths(root: Path, split: str) -> List[Path]:
    split_name = _canonical_split(split)
    split_aliases = [split_name]
    if split_name == "validation":
        split_aliases.append("val")

    candidates: List[Path] = []
    for alias in split_aliases:
        candidates.extend(
            [
                root / f"{alias}.jsonl",
                root / "movie_reviews" / f"{alias}.jsonl",
                root / "data" / f"{alias}.jsonl",
                root / "data" / "movie_reviews" / f"{alias}.jsonl",
            ]
        )
    return candidates


def _find_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_text_from_doc(root: Path, docid: str) -> Optional[str]:
    candidates = [
        root / "docs" / docid,
        root / "docs" / f"{docid}.txt",
        root / "movie_reviews" / "docs" / docid,
        root / "movie_reviews" / "docs" / f"{docid}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _token_spans_from_text(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        spans.append((start, i))
    return spans


def _evidence_to_char_span(evidence: Dict, token_spans: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if "start_char" in evidence and "end_char" in evidence:
        start = int(evidence["start_char"])
        end = int(evidence["end_char"])
        if end > start:
            return (start, end)
    if "start_token" in evidence and "end_token" in evidence and token_spans:
        s_tok = int(evidence["start_token"])
        e_tok = int(evidence["end_token"])
        s_tok = max(0, min(s_tok, len(token_spans) - 1))
        e_tok = max(0, min(e_tok, len(token_spans)))
        if e_tok <= s_tok:
            return None
        start = token_spans[s_tok][0]
        end = token_spans[e_tok - 1][1]
        if end > start:
            return (start, end)
    return None


def _flatten_evidences(raw_evidences: Sequence) -> List[Dict]:
    flattened: List[Dict] = []
    for group in raw_evidences:
        if isinstance(group, dict):
            flattened.append(group)
        elif isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    flattened.append(item)
    return flattened


def _flatten_evidence_strings(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            out.extend(_flatten_evidence_strings(item))
        return out
    return []


def _find_spans_from_evidence_strings(text: str, evidences: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for ev in evidences:
        phrase = str(ev).strip()
        if phrase == "":
            continue
        pos = text.find(phrase)
        if pos >= 0:
            spans.append((pos, pos + len(phrase)))
    return spans


def _load_eraser_movie_reviews_local(
    split: str,
    eraser_root: str,
    max_samples: Optional[int],
) -> DatasetBundle:
    root = Path(eraser_root)
    if not root.exists():
        raise FileNotFoundError(f"ERASER root does not exist: {eraser_root}")

    annotation_path = _find_existing_path(_candidate_eraser_paths(root, split))
    if annotation_path is None:
        raise FileNotFoundError(
            "Cannot find ERASER split file. Tried: "
            + ", ".join(str(p) for p in _candidate_eraser_paths(root, split))
        )

    samples: List[TextSample] = []
    label_names = ["negative", "positive"]
    verbalizers = ["negative", "positive"]

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Compatible with HF-style simplified rows: review/label/evidences
            if "review" in data and "label" in data:
                text = str(data.get("review", ""))
                label = _parse_label(data.get("label"))
                if label is None or text == "":
                    continue
                evidence_strings = _flatten_evidence_strings(data.get("evidences", []))
                rationale_spans = _find_spans_from_evidence_strings(text, evidence_strings)
                samples.append(
                    TextSample(
                        sample_id=f"eraser-mr-local-simple-{line_no}",
                        text=text,
                        label=label,
                        label_text=label_names[label],
                        rationale_char_spans=tuple(rationale_spans),
                        metadata={
                            "source": "local/eraser_movie_reviews_simple",
                            "annotation_path": str(annotation_path),
                        },
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
                continue

            annotation_id = str(data.get("annotation_id", f"line-{line_no}"))
            label = _parse_label(data.get("classification", data.get("label", "")))
            if label is None:
                continue

            evidences = _flatten_evidences(data.get("evidences", []))

            docid = None
            for ev in evidences:
                if "docid" in ev:
                    docid = str(ev["docid"])
                    break
            if docid is None:
                docid = str(data.get("docid", annotation_id))

            text = _read_text_from_doc(root, docid)
            if text is None:
                text = str(data.get("document", data.get("text", "")))
            if text == "":
                continue

            token_spans = _token_spans_from_text(text)
            rationale_spans: List[Tuple[int, int]] = []
            for ev in evidences:
                span = _evidence_to_char_span(ev, token_spans)
                if span is not None:
                    rationale_spans.append(span)

            samples.append(
                TextSample(
                    sample_id=f"eraser-mr-{split}-{annotation_id}",
                    text=text,
                    label=label,
                    label_text=label_names[label],
                    rationale_char_spans=tuple(rationale_spans),
                    metadata={
                        "source": "local/eraser_movie_reviews",
                        "annotation_path": str(annotation_path),
                        "docid": docid,
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break

    return DatasetBundle(
        dataset_name="eraser_movie_reviews",
        split=split,
        samples=samples,
        label_names=label_names,
        verbalizers=verbalizers,
    )


def _load_eraser_movie_reviews_hf(
    split: str,
    max_samples: Optional[int],
    dataset_ref: str = _DEFAULT_ERASER_HF_DATASET,
) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for ERASER HF loading. Please install `datasets`."
        ) from exc

    hf_split = _canonical_split(split)
    dataset_id = _normalize_hf_dataset_ref(dataset_ref)
    ds = load_dataset(dataset_id, split=hf_split)

    samples: List[TextSample] = []
    label_names = ["negative", "positive"]
    verbalizers = ["negative", "positive"]

    for idx, row in enumerate(ds):
        text = _extract_text_from_row(row)
        label = _parse_label(row.get("label"))
        if label is None or text == "":
            continue

        evidence_strings = _flatten_evidence_strings(row.get("evidences", []))
        rationale_spans = _find_spans_from_evidence_strings(text, evidence_strings)

        samples.append(
            TextSample(
                sample_id=f"eraser-mr-hf-{hf_split}-{idx}",
                text=text,
                label=label,
                label_text=label_names[label],
                rationale_char_spans=tuple(rationale_spans),
                metadata={
                    "source": f"hf://{dataset_id}",
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return DatasetBundle(
        dataset_name="eraser_movie_reviews",
        split=split,
        samples=samples,
        label_names=label_names,
        verbalizers=verbalizers,
    )


def load_dataset_bundle(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    eraser_root: Optional[str] = None,
    sst2_source: Optional[str] = None,
    dataset_cache_dir: Optional[str] = None,
) -> DatasetBundle:
    name = dataset_name.lower().strip()

    if name == "sst2":
        if sst2_source:
            if _is_hf_dataset_ref(sst2_source):
                return _load_sst2_from_hf(split=split, max_samples=max_samples, dataset_ref=sst2_source)
            return _load_sst2_from_local_source(
                split=split,
                source=sst2_source,
                max_samples=max_samples,
                dataset_cache_dir=dataset_cache_dir,
            )
        return _load_sst2_from_hf(split=split, max_samples=max_samples, dataset_ref="nyu-mll/glue")

    if name in {"eraser_movie_reviews", "eraser-movie-reviews", "eraser"}:
        if eraser_root:
            if _is_hf_dataset_ref(eraser_root):
                return _load_eraser_movie_reviews_hf(
                    split=split,
                    max_samples=max_samples,
                    dataset_ref=eraser_root,
                )
            resolved = _resolve_source_to_local_path(eraser_root, dataset_cache_dir=dataset_cache_dir)
            return _load_eraser_movie_reviews_local(
                split=split,
                eraser_root=str(resolved),
                max_samples=max_samples,
            )

        # Default to remote HF dataset for easier reproduction.
        return _load_eraser_movie_reviews_hf(
            split=split,
            max_samples=max_samples,
            dataset_ref=_DEFAULT_ERASER_HF_DATASET,
        )

    if name == "imdb":
        return _load_hf_text_classification_dataset(
            dataset_name="imdb",
            split=split,
            max_samples=max_samples,
            dataset_ref=_DEFAULT_IMDB_HF_DATASET,
        )

    if name in {"rotten_tomatoes", "rotten-tomatoes"}:
        return _load_hf_text_classification_dataset(
            dataset_name="rotten_tomatoes",
            split=split,
            max_samples=max_samples,
            dataset_ref=_DEFAULT_ROTTEN_TOMATOES_HF_DATASET,
        )

    if name == "emotion":
        return _load_hf_text_classification_dataset(
            dataset_name="emotion",
            split=split,
            max_samples=max_samples,
            dataset_ref=_DEFAULT_EMOTION_HF_DATASET,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
