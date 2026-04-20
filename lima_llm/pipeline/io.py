from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from ..types import ExplanationResult
from ..utils import atomic_write_json, atomic_write_text, ensure_dir


def save_explanation(result: ExplanationResult, output_root: Path) -> Dict[str, Path]:
    ensure_dir(output_root / "samples")
    json_path = output_root / "samples" / f"{result.sample_id}.json"
    txt_path = output_root / "samples" / f"{result.sample_id}.txt"

    atomic_write_json(json_path, result.to_dict())
    atomic_write_text(txt_path, result.selected_text)
    return {"json": json_path, "txt": txt_path}


def rebuild_summary_csv(output_root: Path) -> Path:
    sample_dir = output_root / "samples"
    rows: List[Dict] = []
    if sample_dir.exists():
        for path in sorted(sample_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            scores = payload.get("scores", {})
            row = {
                "sample_id": payload.get("sample_id", ""),
                "label": payload.get("label", ""),
                "selected_chunk_count": len(payload.get("selected_chunk_ids", [])),
                "total_chunks": len(payload.get("chunks", [])),
                "score_total": scores.get("total", ""),
                "confidence": scores.get("confidence", ""),
                "effectiveness": scores.get("effectiveness", ""),
                "consistency": scores.get("consistency", ""),
                "collaboration": scores.get("collaboration", ""),
                "target_probability": scores.get("target_probability", ""),
            }
            rows.append(row)

    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "label",
                "selected_chunk_count",
                "total_chunks",
                "score_total",
                "confidence",
                "effectiveness",
                "consistency",
                "collaboration",
                "target_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
