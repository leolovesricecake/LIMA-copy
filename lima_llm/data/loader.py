from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import TextSample


@dataclass
class DatasetBundle:
    dataset_name: str
    split: str
    samples: List[TextSample]
    label_names: List[str]
    verbalizers: List[str]


_SST2_LABELS = ["negative", "positive"]


def _load_sst2(split: str, max_samples: Optional[int]) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for SST-2. Please install `datasets`.") from exc

    hf_split = split
    if split == "dev":
        hf_split = "validation"

    ds = load_dataset("glue", "sst2", split=hf_split)
    samples: List[TextSample] = []
    for idx, row in enumerate(ds):
        text = str(row.get("sentence", ""))
        label = int(row.get("label", -1))
        if label < 0:
            continue
        sample = TextSample(
            sample_id=f"sst2-{hf_split}-{idx}",
            text=text,
            label=label,
            label_text=_SST2_LABELS[label] if label < len(_SST2_LABELS) else None,
            rationale_char_spans=(),
            metadata={"source": "glue/sst2"},
        )
        samples.append(sample)
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
    split_aliases = [split]
    if split == "validation":
        split_aliases.append("val")
    if split == "val":
        split_aliases.append("validation")
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


def _load_eraser_movie_reviews(
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

            annotation_id = str(data.get("annotation_id", f"line-{line_no}"))
            raw_label = str(data.get("classification", data.get("label", ""))).lower()
            if raw_label in {"pos", "positive", "1", "true"}:
                label = 1
            elif raw_label in {"neg", "negative", "0", "false"}:
                label = 0
            else:
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

            sample = TextSample(
                sample_id=f"eraser-mr-{split}-{annotation_id}",
                text=text,
                label=label,
                label_text=label_names[label],
                rationale_char_spans=tuple(rationale_spans),
                metadata={
                    "source": "eraser/movie_reviews",
                    "annotation_path": str(annotation_path),
                    "docid": docid,
                },
            )
            samples.append(sample)
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
) -> DatasetBundle:
    name = dataset_name.lower().strip()
    if name == "sst2":
        return _load_sst2(split=split, max_samples=max_samples)
    if name in {"eraser_movie_reviews", "eraser-movie-reviews", "eraser"}:
        if not eraser_root:
            raise ValueError("--eraser-root is required for dataset=eraser_movie_reviews")
        return _load_eraser_movie_reviews(split=split, eraser_root=eraser_root, max_samples=max_samples)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
