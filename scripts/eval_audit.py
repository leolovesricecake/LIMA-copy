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


def _get(report: Dict[str, Any], dotted: str) -> float:
    cur: Any = report
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return 0.0
    return _safe_float(cur)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create eval_audit_report.json from legacy/current reports")
    p.add_argument("--legacy-report", type=str, required=True)
    p.add_argument("--current-report", type=str, required=True)
    p.add_argument("--output", type=str, default="eval_audit_report.json")
    p.add_argument("--predict-calls-ratio-upper", type=float, default=1.5)
    p.add_argument("--comp-abs-diff-max", type=float, default=0.01)
    p.add_argument("--suff-abs-diff-max", type=float, default=0.01)
    return p


def main() -> None:
    args = build_parser().parse_args()
    legacy_path = Path(args.legacy_report)
    current_path = Path(args.current_report)
    legacy = _read_json(legacy_path)
    current = _read_json(current_path)

    sample_legacy = int(_safe_float(legacy.get("sample_count", 0)))
    sample_current = int(_safe_float(current.get("sample_count", 0)))
    comp_legacy = _get(legacy, "metrics_primary.comprehensiveness")
    comp_current = _get(current, "metrics_primary.comprehensiveness")
    suff_legacy = _get(legacy, "metrics_primary.sufficiency")
    suff_current = _get(current, "metrics_primary.sufficiency")
    pred_legacy = _get(legacy, "metrics_secondary.forward_counters_delta.predict_calls")
    pred_current = _get(current, "metrics_secondary.forward_counters_delta.predict_calls")
    grad_legacy = _get(legacy, "metrics_secondary.forward_counters_delta.gradient_calls")
    grad_current = _get(current, "metrics_secondary.forward_counters_delta.gradient_calls")
    rt_legacy = _get(legacy, "metrics_secondary.runtime_seconds")
    rt_current = _get(current, "metrics_secondary.runtime_seconds")

    predict_ratio = (pred_current / pred_legacy) if abs(pred_legacy) > 1e-12 else 0.0
    comp_abs_diff = abs(comp_current - comp_legacy)
    suff_abs_diff = abs(suff_current - suff_legacy)

    checks = {
        "sample_count_match": sample_legacy == sample_current,
        "predict_calls_ratio_ok": predict_ratio <= float(args.predict_calls_ratio_upper),
        "comp_abs_diff_ok": comp_abs_diff <= float(args.comp_abs_diff_max),
        "suff_abs_diff_ok": suff_abs_diff <= float(args.suff_abs_diff_max),
    }
    passed = all(bool(v) for v in checks.values())

    payload = {
        "passed": passed,
        "legacy_report": str(legacy_path),
        "current_report": str(current_path),
        "thresholds": {
            "predict_calls_ratio_upper": float(args.predict_calls_ratio_upper),
            "comp_abs_diff_max": float(args.comp_abs_diff_max),
            "suff_abs_diff_max": float(args.suff_abs_diff_max),
        },
        "summary": {
            "sample_count_legacy": sample_legacy,
            "sample_count_current": sample_current,
            "runtime_seconds_legacy": rt_legacy,
            "runtime_seconds_current": rt_current,
            "predict_calls_legacy": pred_legacy,
            "predict_calls_current": pred_current,
            "predict_calls_ratio": predict_ratio,
            "gradient_calls_legacy": grad_legacy,
            "gradient_calls_current": grad_current,
            "comprehensiveness_legacy": comp_legacy,
            "comprehensiveness_current": comp_current,
            "comprehensiveness_abs_diff": comp_abs_diff,
            "sufficiency_legacy": suff_legacy,
            "sufficiency_current": suff_current,
            "sufficiency_abs_diff": suff_abs_diff,
        },
        "checks": checks,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] passed={passed}")
    print(f"[audit] report={out}")


if __name__ == "__main__":
    main()
