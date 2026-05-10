#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _index_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group in snapshot.get("groups", []):
        gid = str(group.get("group_id"))
        out[gid] = {}
        for method, run in group.get("methods", {}).items():
            out[gid][str(method)] = run
    return out


def _metric(run: Dict[str, Any], dotted: str) -> float:
    cur: Any = run
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return 0.0
    return _safe_float(cur)


def build_diff(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    idx_b = _index_snapshot(baseline)
    idx_c = _index_snapshot(current)

    metrics = [
        "timing.explain_seconds_total",
        "timing.eval_seconds",
        "timing.total_seconds",
        "metrics.gold.comprehensiveness",
        "metrics.gold.sufficiency",
        "metrics.gold.aopc",
        "metrics.plausibility_f1",
        "metrics.plausibility_iou",
        "metrics.eval_forward_counters.predict_calls",
        "metrics.eval_forward_counters.gradient_calls",
        "metrics.eval_forward_counters.embed_calls",
    ]

    rows: List[Dict[str, Any]] = []
    for gid in sorted(set(idx_b.keys()).intersection(idx_c.keys())):
        methods = sorted(set(idx_b[gid].keys()).intersection(idx_c[gid].keys()))
        for method in methods:
            rb = idx_b[gid][method]
            rc = idx_c[gid][method]
            for metric in metrics:
                vb = _metric(rb, metric)
                vc = _metric(rc, metric)
                rows.append(
                    {
                        "group_id": gid,
                        "method": method,
                        "metric": metric,
                        "baseline": vb,
                        "current": vc,
                        "delta": vc - vb,
                        "ratio": (vc / vb) if abs(vb) > 1e-12 else 0.0,
                    }
                )

    return {
        "baseline_primary_method": baseline.get("primary_method"),
        "current_primary_method": current.get("primary_method"),
        "baseline_reference_method": baseline.get("reference_method"),
        "current_reference_method": current.get("reference_method"),
        "row_count": len(rows),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diff two analysis_snapshot.json files")
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--current", type=str, required=True)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    baseline = _read_json(Path(args.baseline))
    current = _read_json(Path(args.current))
    diff = build_diff(baseline, current)

    out_json = Path(args.output_json) if args.output_json else (Path(args.current).parent / "analysis_snapshot_diff.json")
    out_csv = Path(args.output_csv) if args.output_csv else (Path(args.current).parent / "analysis_snapshot_diff.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["group_id", "method", "metric", "baseline", "current", "delta", "ratio"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diff["rows"]:
            writer.writerow(row)

    print(f"[snapshot-diff] rows={diff['row_count']}")
    print(f"[snapshot-diff] json={out_json}")
    print(f"[snapshot-diff] csv={out_csv}")


if __name__ == "__main__":
    main()
