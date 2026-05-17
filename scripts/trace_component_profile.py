#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_profile(run_dir: Path) -> Dict[str, Any]:
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return {
            "run_dir": str(run_dir),
            "sample_count": 0,
            "step_profile": [],
            "final_component_means": {},
        }

    per_step = defaultdict(lambda: defaultdict(list))
    final_components = defaultdict(list)
    selected_counts: List[float] = []

    sample_count = 0
    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        sample_count += 1
        selected_counts.append(float(len(payload.get("selected_chunk_ids", []))))

        trace = payload.get("trace", [])
        for row in trace:
            step = int(_safe_float(row.get("step"), 0.0))
            comps = row.get("components", {})
            per_step[step]["total_score"].append(_safe_float(row.get("total_score")))
            per_step[step]["marginal_gain"].append(_safe_float(row.get("marginal_gain")))
            per_step[step]["confidence"].append(_safe_float(comps.get("confidence")))
            per_step[step]["effectiveness"].append(_safe_float(comps.get("effectiveness")))
            per_step[step]["consistency"].append(_safe_float(comps.get("consistency")))
            per_step[step]["collaboration"].append(_safe_float(comps.get("collaboration")))

        if trace:
            last = trace[-1]
            comps = last.get("components", {})
            for key in ("confidence", "effectiveness", "consistency", "collaboration"):
                final_components[key].append(_safe_float(comps.get(key)))

    rows: List[Dict[str, Any]] = []
    for step in sorted(per_step.keys()):
        bucket = per_step[step]
        rows.append(
            {
                "step": step,
                "sample_rows": len(bucket["total_score"]),
                "mean_total_score": _mean(bucket["total_score"]),
                "mean_marginal_gain": _mean(bucket["marginal_gain"]),
                "mean_confidence": _mean(bucket["confidence"]),
                "mean_effectiveness": _mean(bucket["effectiveness"]),
                "mean_consistency": _mean(bucket["consistency"]),
                "mean_collaboration": _mean(bucket["collaboration"]),
            }
        )

    return {
        "run_dir": str(run_dir),
        "sample_count": int(sample_count),
        "mean_selected_chunks": _mean(selected_counts),
        "step_profile": rows,
        "final_component_means": {
            key: _mean(values) for key, values in final_components.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-step component traces from samples/*.json")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    profile = build_profile(run_dir)

    out_json = Path(args.output_json) if args.output_json else (run_dir / "trace_component_profile.json")
    out_csv = Path(args.output_csv) if args.output_csv else (run_dir / "trace_component_profile.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "step",
        "sample_rows",
        "mean_total_score",
        "mean_marginal_gain",
        "mean_confidence",
        "mean_effectiveness",
        "mean_consistency",
        "mean_collaboration",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile.get("step_profile", []):
            writer.writerow(row)

    print(f"[trace-profile] samples={profile.get('sample_count', 0)}")
    print(f"[trace-profile] json={out_json}")
    print(f"[trace-profile] csv={out_csv}")


if __name__ == "__main__":
    main()
