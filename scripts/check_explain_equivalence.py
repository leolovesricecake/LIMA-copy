#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_samples(run_dir: Path) -> Dict[str, dict]:
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples dir not found: {samples_dir}")
    payloads: Dict[str, dict] = {}
    for path in sorted(samples_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        sid = str(obj.get("sample_id", path.stem))
        payloads[sid] = obj
    return payloads


def _trace_totals(payload: dict) -> List[float]:
    trace = payload.get("trace", [])
    out: List[float] = []
    for step in trace:
        out.append(float(step.get("total_score", 0.0)))
    return out


def _compare(left: dict, right: dict, tol: float) -> Tuple[bool, dict | None]:
    l_ids = [int(x) for x in left.get("selected_chunk_ids", [])]
    r_ids = [int(x) for x in right.get("selected_chunk_ids", [])]
    if l_ids != r_ids:
        return False, {
            "section": "selected_chunk_ids",
            "left": l_ids,
            "right": r_ids,
        }

    l_trace = _trace_totals(left)
    r_trace = _trace_totals(right)
    if len(l_trace) != len(r_trace):
        return False, {
            "section": "trace.length",
            "left": len(l_trace),
            "right": len(r_trace),
        }
    for i, (lv, rv) in enumerate(zip(l_trace, r_trace)):
        diff = abs(float(lv) - float(rv))
        if diff > tol:
            return False, {
                "section": "trace.total_score",
                "step": i,
                "left": float(lv),
                "right": float(rv),
                "abs_diff": float(diff),
                "tolerance": float(tol),
            }
    return True, None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check sample-level explanation equivalence")
    p.add_argument("--left-run-dir", type=str, required=True)
    p.add_argument("--right-run-dir", type=str, required=True)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--output-json", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    left_dir = Path(args.left_run_dir)
    right_dir = Path(args.right_run_dir)
    tol = float(args.tolerance)

    left = _load_samples(left_dir)
    right = _load_samples(right_dir)

    left_ids = set(left.keys())
    right_ids = set(right.keys())
    only_left = sorted(left_ids - right_ids)
    only_right = sorted(right_ids - left_ids)
    common = sorted(left_ids & right_ids)

    result: dict = {
        "passed": True,
        "tolerance": tol,
        "left_run_dir": str(left_dir),
        "right_run_dir": str(right_dir),
        "sample_count_left": len(left_ids),
        "sample_count_right": len(right_ids),
        "sample_count_common": len(common),
        "only_left": only_left,
        "only_right": only_right,
        "first_failure": None,
    }

    if only_left or only_right:
        result["passed"] = False
        result["first_failure"] = {
            "section": "sample_set_mismatch",
            "only_left_count": len(only_left),
            "only_right_count": len(only_right),
        }
    else:
        for sid in common:
            ok, failure = _compare(left[sid], right[sid], tol)
            if not ok:
                result["passed"] = False
                result["first_failure"] = {"sample_id": sid, **(failure or {})}
                break

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

