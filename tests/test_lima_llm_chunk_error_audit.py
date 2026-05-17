from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_sample(run_dir: Path, sample_id: str, chunk_texts: list[str]) -> None:
    text = "".join(chunk_texts)
    chunks = []
    cursor = 0
    for idx, chunk_text in enumerate(chunk_texts):
        start = cursor
        end = start + len(chunk_text)
        cursor = end
        chunks.append(
            {
                "chunk_id": idx,
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            }
        )

    payload = {
        "sample_id": sample_id,
        "text": text,
        "chunks": chunks,
    }
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / f"{sample_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_chunk_error_audit_builds_report_and_manifest(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "chunk_error_audit.py"
    mod = _load_module(script, "chunk_error_audit")

    run_dir = tmp_path / "run"
    _write_sample(
        run_dir,
        sample_id="s1",
        chunk_texts=[
            "good sentence.\n",
            ".\n",
            ") broken clause continues here.\n",
            "mr .\n",
            "taylor writes here.\n",
        ],
    )
    _write_sample(
        run_dir,
        sample_id="s2",
        chunk_texts=[
            "clean sentence.\n",
            "another clean sentence.\n",
        ],
    )

    report, rows = mod.build_audit(run_dir=run_dir, context_window=20)

    assert report["sample_count"] == 2
    assert report["chunk_count"] == 7
    assert report["manifest_count"] == 3
    assert report["error_counts"]["orphan_punctuation_chunk"] == 1
    assert report["error_counts"]["leading_close_punct_chunk"] == 1
    assert report["error_counts"]["abbreviation_singleton_chunk"] == 1
    assert report["samples_with_any_error"] == 1
    assert report["sample_error_distribution"]["s1"]["total_errors"] == 3
    assert report["sample_error_distribution"]["s2"]["total_errors"] == 0
    assert len(rows) == 3

    output_json = tmp_path / "chunk_error_audit.json"
    output_csv = tmp_path / "chunk_error_manifest.csv"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mod._write_manifest_csv(rows, output_csv)

    assert output_json.exists()
    assert output_csv.exists()
    csv_content = output_csv.read_text(encoding="utf-8")
    assert "sample_id,chunk_id,start_char,end_char,text,error_type,context" in csv_content
