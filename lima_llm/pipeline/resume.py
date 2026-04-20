from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..utils import is_valid_json


def sample_output_paths(output_root: Path, sample_id: str) -> Dict[str, Path]:
    sample_dir = output_root / "samples"
    return {
        "json": sample_dir / f"{sample_id}.json",
        "txt": sample_dir / f"{sample_id}.txt",
    }


def is_sample_completed(paths: Dict[str, Path], mode: str) -> bool:
    json_path = paths["json"]
    txt_path = paths["txt"]

    if mode == "exists-only":
        return json_path.exists() and txt_path.exists()

    if not json_path.exists() or not txt_path.exists():
        return False
    if not is_valid_json(json_path):
        return False

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if "selected_chunk_ids" not in payload:
            return False
        if "trace" not in payload:
            return False
    except Exception:
        return False

    return True
