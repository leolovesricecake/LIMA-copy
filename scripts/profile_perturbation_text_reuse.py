#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lima_llm.eval.metrics import AML_AOPC_Q_VALUES, AML_PRIMARY_Q_PERCENT, build_perturbation_plan
from lima_llm.types import TextChunk
from lima_llm.utils import parse_q_values


def _to_chunks(raw_chunks: Sequence[Dict]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for item in raw_chunks:
        chunks.append(
            TextChunk(
                chunk_id=int(item["chunk_id"]),
                start_char=int(item["start_char"]),
                end_char=int(item["end_char"]),
                text=str(item["text"]),
                token_start=item.get("token_start"),
                token_end=item.get("token_end"),
            )
        )
    return chunks


def _normalize_chunk_ranking(payload: Dict, chunks: Sequence[TextChunk]) -> List[int]:
    chunk_ids = [int(c.chunk_id) for c in chunks]
    chunk_set = set(chunk_ids)
    ranking_raw = [int(x) for x in payload.get("chunk_ranking", [])]

    seen = set()
    ranking: List[int] = []
    for cid in ranking_raw:
        if cid in chunk_set and cid not in seen:
            ranking.append(cid)
            seen.add(cid)
    for cid in chunk_ids:
        if cid not in seen:
            ranking.append(cid)
    return ranking


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Profile cross-sample perturbation text reuse potential. "
            "This uses saved sample chunks/rankings and estimates upper-bound global-cache gains."
        )
    )
    p.add_argument("--run-dir", type=str, required=True, help="Run directory containing samples/*.json")
    p.add_argument(
        "--q-values",
        type=str,
        default=",".join(str(x) for x in AML_AOPC_Q_VALUES),
        help="Comma-separated q values, default matches AML AOPC percentages.",
    )
    p.add_argument("--top-duplicates", type=int, default=20, help="How many most-frequent texts to report.")
    p.add_argument(
        "--include-sample-summaries",
        action="store_true",
        help="Include per-sample required/unique counts in output json.",
    )
    p.add_argument("--output-json", type=str, default="", help="Optional output json path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    sample_files = sorted(sample_dir.glob("*.json"))
    if not sample_files:
        raise FileNotFoundError(f"No sample json found in: {sample_dir}")

    q_values = parse_q_values(args.q_values)

    sample_count = 0
    total_required_count = 0
    total_unique_per_sample = 0
    global_unique = set()
    global_counter: Counter[str] = Counter()
    sample_summaries = []

    for sample_json in sample_files:
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        chunks = _to_chunks(payload.get("chunks", []))
        if not chunks:
            continue
        ranking = _normalize_chunk_ranking(payload=payload, chunks=chunks)
        plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=q_values,
            primary_q_percent=AML_PRIMARY_Q_PERCENT,
        )
        required_texts = list(plan.get("required_texts", []))
        unique_texts = set(required_texts)

        sample_count += 1
        total_required_count += int(plan.get("required_text_count", len(required_texts)))
        total_unique_per_sample += int(plan.get("unique_required_text_count", len(unique_texts)))
        global_unique.update(unique_texts)
        global_counter.update(unique_texts)
        sample_summaries.append(
            {
                "sample_id": payload.get("sample_id"),
                "required_text_count": int(plan.get("required_text_count", len(required_texts))),
                "unique_required_text_count": int(plan.get("unique_required_text_count", len(unique_texts))),
            }
        )

    global_unique_count = len(global_unique)
    cross_sample_reuse_count = max(0, total_unique_per_sample - global_unique_count)
    cross_sample_reuse_ratio = (
        float(cross_sample_reuse_count / total_unique_per_sample) if total_unique_per_sample > 0 else 0.0
    )
    top_dups = [
        {
            "text": text,
            "occurrence_samples": int(cnt),
            "char_len": len(text),
        }
        for text, cnt in global_counter.most_common(max(0, int(args.top_duplicates)))
    ]

    report = {
        "schema_version": 1,
        "profiling_level": "chunk_plan_estimate",
        "note": (
            "This profile estimates cross-sample reuse from saved chunk-level perturbation plans. "
            "Token-level eval perturbations may differ."
        ),
        "run_dir": str(run_dir.resolve()),
        "sample_count": int(sample_count),
        "q_values": [int(q) for q in q_values],
        "totals": {
            "required_text_count": int(total_required_count),
            "sum_unique_text_count_per_sample": int(total_unique_per_sample),
            "global_unique_text_count": int(global_unique_count),
        },
        "cross_sample_reuse_upper_bound": {
            "reusable_unique_text_count": int(cross_sample_reuse_count),
            "reusable_ratio_vs_sum_unique": float(cross_sample_reuse_ratio),
        },
        "top_duplicate_texts": top_dups,
    }
    if args.include_sample_summaries:
        report["sample_summaries"] = sample_summaries

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
