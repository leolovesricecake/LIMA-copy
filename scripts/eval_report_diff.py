#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_metric(report: Dict[str, Any], dotted: str) -> float:
    cur: Any = report
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return 0.0
    return _safe_float(cur)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare two eval reports and print key metric deltas")
    p.add_argument("--left", type=str, required=True)
    p.add_argument("--right", type=str, required=True)
    p.add_argument("--left-name", type=str, default="left")
    p.add_argument("--right-name", type=str, default="right")
    p.add_argument("--output", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)
    left = _read_json(left_path)
    right = _read_json(right_path)

    keys = [
        "sample_count",
        "metrics_primary.comprehensiveness",
        "metrics_primary.sufficiency",
        "metrics_primary.log_odds",
        "metrics_primary.aopc",
        "metrics_secondary.runtime_seconds",
        "metrics_secondary.forward_counters_delta.predict_calls",
        "metrics_secondary.forward_counters_delta.gradient_calls",
    ]

    rows = []
    for k in keys:
        lv = _get_metric(left, k)
        rv = _get_metric(right, k)
        rows.append(
            {
                "metric": k,
                args.left_name: lv,
                args.right_name: rv,
                "abs_diff": abs(rv - lv),
                "delta": rv - lv,
                "ratio": (rv / lv) if abs(lv) > 1e-12 else 0.0,
            }
        )

    payload = {
        "left": str(left_path),
        "right": str(right_path),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "rows": rows,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[diff] json={out}")

    print("metric,left,right,abs_diff,delta,ratio")
    for r in rows:
        print(
            f"{r['metric']},{r[args.left_name]:.10f},{r[args.right_name]:.10f},"
            f"{r['abs_diff']:.10f},{r['delta']:.10f},{r['ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
