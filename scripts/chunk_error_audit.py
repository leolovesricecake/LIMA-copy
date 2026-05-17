#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ORPHAN_PUNCT_RE = re.compile(r'^[\s"\'`“”‘’\(\)\[\]\{\}\.,!?;:，。！？；：、…\-–—]+$')
_LEADING_CLOSE_PUNCT_RE = re.compile(r"^\s*[\)\]\}]+")
_ABBREVIATION_SINGLETON_RE = re.compile(r"^(mr|mrs|ms|dr|prof|st|jr|sr)\s*\.\s*$", flags=re.IGNORECASE)


def is_orphan_punctuation_chunk_text(text: str) -> bool:
    if text.strip() == "":
        return False
    return _ORPHAN_PUNCT_RE.fullmatch(text) is not None


def is_leading_close_punct_chunk_text(text: str) -> bool:
    stripped = text.lstrip()
    if stripped == "":
        return False
    match = _LEADING_CLOSE_PUNCT_RE.match(stripped)
    if match is None:
        return False
    rest = stripped[match.end() :]
    return rest.strip() != ""


def is_abbreviation_singleton_chunk_text(text: str) -> bool:
    normalized = text.replace("\n", " ").strip()
    if normalized == "":
        return False
    return _ABBREVIATION_SINGLETON_RE.fullmatch(normalized) is not None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_samples(run_dir: Path) -> List[Dict[str, Any]]:
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _detect_error_types(text: str) -> List[str]:
    out: List[str] = []
    if is_orphan_punctuation_chunk_text(text):
        out.append("orphan_punctuation_chunk")
    if is_leading_close_punct_chunk_text(text):
        out.append("leading_close_punct_chunk")
    if is_abbreviation_singleton_chunk_text(text):
        out.append("abbreviation_singleton_chunk")
    return out


def build_audit(run_dir: Path, context_window: int = 80) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    samples = _load_samples(run_dir)
    error_counts = Counter()
    samples_with_error_counts = Counter()
    sample_error_distribution: Dict[str, Dict[str, Any]] = {}
    manifest_rows: List[Dict[str, Any]] = []

    total_chunks = 0

    for payload in samples:
        sample_id = str(payload.get("sample_id", "unknown"))
        text = str(payload.get("text", ""))
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            continue

        total_chunks += len(chunks)
        local_counts = Counter()

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_text = str(chunk.get("text", ""))
            chunk_id = int(chunk.get("chunk_id", -1))
            start_char = int(chunk.get("start_char", 0))
            end_char = int(chunk.get("end_char", 0))

            error_types = _detect_error_types(chunk_text)
            if not error_types:
                continue

            ctx_start = max(0, start_char - int(context_window))
            ctx_end = min(len(text), end_char + int(context_window))
            context = text[ctx_start:ctx_end].replace("\n", "\\n")

            for error_type in error_types:
                error_counts[error_type] += 1
                local_counts[error_type] += 1
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "chunk_id": chunk_id,
                        "start_char": start_char,
                        "end_char": end_char,
                        "text": chunk_text.replace("\n", "\\n"),
                        "error_type": error_type,
                        "context": context,
                    }
                )

        if local_counts:
            for key in local_counts:
                samples_with_error_counts[key] += 1

        sample_error_distribution[sample_id] = {
            "chunk_count": int(len(chunks)),
            "total_errors": int(sum(local_counts.values())),
            "error_counts": {k: int(v) for k, v in sorted(local_counts.items())},
        }

    top_samples = sorted(
        (
            {
                "sample_id": sid,
                "total_errors": int(info.get("total_errors", 0)),
                "error_counts": dict(info.get("error_counts", {})),
            }
            for sid, info in sample_error_distribution.items()
            if int(info.get("total_errors", 0)) > 0
        ),
        key=lambda row: (-int(row["total_errors"]), str(row["sample_id"])),
    )

    report = {
        "run_dir": str(run_dir),
        "sample_count": int(len(samples)),
        "chunk_count": int(total_chunks),
        "error_counts": {k: int(v) for k, v in sorted(error_counts.items())},
        "samples_with_error_counts": {k: int(v) for k, v in sorted(samples_with_error_counts.items())},
        "samples_with_any_error": int(sum(1 for x in sample_error_distribution.values() if x["total_errors"] > 0)),
        "top_samples_by_error_count": top_samples,
        "sample_error_distribution": sample_error_distribution,
        "manifest_count": int(len(manifest_rows)),
    }
    return report, manifest_rows


def _write_manifest_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_id",
        "chunk_id",
        "start_char",
        "end_char",
        "text",
        "error_type",
        "context",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit chunk errors from a single run directory")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--context-window", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_json = Path(args.output_json) if args.output_json else (run_dir / "chunk_error_audit.json")
    output_csv = Path(args.output_csv) if args.output_csv else (run_dir / "chunk_error_manifest.csv")

    report, rows = build_audit(run_dir=run_dir, context_window=int(args.context_window))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_manifest_csv(rows, output_csv)

    print(f"[chunk-audit] json={output_json}")
    print(f"[chunk-audit] csv={output_csv}")
    print(f"[chunk-audit] samples={report['sample_count']} chunks={report['chunk_count']} errors={report['manifest_count']}")


if __name__ == "__main__":
    main()
