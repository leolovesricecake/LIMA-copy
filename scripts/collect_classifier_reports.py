"""Collect classifier diagnostics into one paper-ready CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


FIELDS = (
    "dataset",
    "split",
    "model",
    "model_path",
    "prompt_version",
    "sample_count",
    "accuracy",
    "balanced_accuracy",
    "random_baseline_accuracy",
    "majority_baseline_accuracy",
    "per_class_recall_json",
    "confusion_matrix_json",
    "config_fingerprint",
    "source_file",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the classifier-report collection parser."""

    parser = argparse.ArgumentParser(
        description="Collect classifier-check JSON reports into one CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="results/classifier-check",
        help="Directory recursively containing classifier diagnostic JSON files.",
    )
    parser.add_argument(
        "--output",
        default="results/paper/paper-v2.3/aggregate/classifier-diagnostics.csv",
    )
    return parser


def _model_name(model: Mapping[str, Any]) -> str:
    """Return a readable model identifier from one report block."""

    raw = str(model.get("model_path", model.get("type", "unknown"))).rstrip("/")
    return raw.split("/")[-1] or "unknown"


def _row(path: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten one diagnostic report while preserving structured fields as JSON."""

    dataset = dict(report.get("dataset", {}))
    model = dict(report.get("model", {}))
    prompt = dict(report.get("prompt", {}))
    per_class_recall = report.get("per_class_recall")
    if per_class_recall is None:
        per_class_recall = {
            str(row.get("label", row.get("class_id"))): row.get("recall")
            for row in report.get("per_class", [])
        }
    return {
        "dataset": dataset.get("name"),
        "split": dataset.get("split"),
        "model": _model_name(model),
        "model_path": model.get("model_path", model.get("type")),
        "prompt_version": prompt.get("version"),
        "sample_count": report.get("sample_count"),
        "accuracy": report.get("accuracy"),
        "balanced_accuracy": report.get("balanced_accuracy"),
        "random_baseline_accuracy": report.get("random_baseline_accuracy"),
        "majority_baseline_accuracy": report.get("majority_baseline_accuracy"),
        "per_class_recall_json": json.dumps(
            per_class_recall, ensure_ascii=False, sort_keys=True
        ),
        "confusion_matrix_json": json.dumps(
            report.get("confusion_matrix", []), ensure_ascii=False
        ),
        "config_fingerprint": report.get("config_fingerprint"),
        "source_file": str(path.resolve()),
    }


def collect_reports(input_dir: Path) -> List[Dict[str, Any]]:
    """Read valid diagnostic reports below one directory in stable order."""

    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping) or "accuracy" not in report:
            continue
        rows.append(_row(path, report))
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic CSV output with the canonical diagnostic columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def main(argv: list[str] | None = None) -> None:
    """Collect reports and print the resulting artifact path."""

    args = build_parser().parse_args(argv)
    rows = collect_reports(Path(args.input_dir))
    output = Path(args.output)
    write_csv(output, rows)
    print(f"[classifier-reports] rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
