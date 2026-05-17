#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

COMPONENTS = ("confidence", "effectiveness", "consistency", "collaboration")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_lambdas(raw: Any) -> Tuple[float, float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(float(x) for x in raw)  # type: ignore[return-value]
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 4:
        return 1.0, 1.0, 1.0, 1.0
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def _load_samples(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    sample_dir = run_dir / "samples"
    out: Dict[str, Dict[str, Any]] = {}
    if not sample_dir.exists():
        return out
    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        sample_id = str(payload.get("sample_id", path.stem))
        out[sample_id] = payload
    return out


def _explain_seconds_total(samples: Dict[str, Dict[str, Any]]) -> float:
    return float(sum(_safe_float(row.get("metadata", {}).get("elapsed_seconds"), 0.0) for row in samples.values()))


def _gold_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    gold = report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    return {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comp": _safe_float(gold.get("comprehensiveness")),
        "suff": _safe_float(gold.get("sufficiency")),
        "a_c": _safe_float(gold.get("aopc_comprehensiveness")),
        "a_s": _safe_float(gold.get("aopc_sufficiency")),
    }


def _directional_gain(metric: str, baseline: float, current: float) -> float:
    if metric in {"suff", "a_s"}:
        return baseline - current
    return current - baseline


def _disabled_components(lambdas: Tuple[float, float, float, float], eps: float = 1e-12) -> List[str]:
    out: List[str] = []
    for name, value in zip(COMPONENTS, lambdas):
        if abs(float(value)) <= eps:
            out.append(name)
    return out


@dataclass
class StabilityDiff:
    compared_samples: int
    selected_changed: int
    ranking_changed: int

    def to_dict(self) -> Dict[str, Any]:
        denom = max(1, self.compared_samples)
        return {
            "compared_samples": int(self.compared_samples),
            "selected_changed": int(self.selected_changed),
            "ranking_changed": int(self.ranking_changed),
            "selected_changed_ratio": float(self.selected_changed / denom),
            "ranking_changed_ratio": float(self.ranking_changed / denom),
        }


def _stability_diff(
    baseline_samples: Dict[str, Dict[str, Any]],
    current_samples: Dict[str, Dict[str, Any]],
) -> StabilityDiff:
    shared_ids = sorted(set(baseline_samples.keys()).intersection(current_samples.keys()))
    selected_changed = 0
    ranking_changed = 0
    for sid in shared_ids:
        left = baseline_samples[sid]
        right = current_samples[sid]
        if left.get("selected_chunk_ids") != right.get("selected_chunk_ids"):
            selected_changed += 1
        if left.get("chunk_ranking") != right.get("chunk_ranking"):
            ranking_changed += 1
    return StabilityDiff(
        compared_samples=len(shared_ids),
        selected_changed=selected_changed,
        ranking_changed=ranking_changed,
    )


def _aggregate_objective_compute_stats(samples: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sums: Dict[str, float] = {
        "evaluate_gains_calls": 0.0,
        "subset_cache_hit_rate": 0.0,
        "prob_cache_hit_rate": 0.0,
        "embed_cache_hit_rate": 0.0,
        "confidence_compute_calls": 0.0,
        "effectiveness_compute_calls": 0.0,
        "consistency_compute_calls": 0.0,
        "collaboration_compute_calls": 0.0,
        "confidence_skipped_due_to_zero_lambda": 0.0,
        "effectiveness_skipped_due_to_zero_lambda": 0.0,
        "consistency_skipped_due_to_zero_lambda": 0.0,
        "collaboration_skipped_due_to_zero_lambda": 0.0,
    }
    timing_sums = {
        "text_build_seconds": 0.0,
        "prefetch_seconds": 0.0,
        "component_compute_seconds": 0.0,
        "confidence_compute_seconds": 0.0,
        "effectiveness_compute_seconds": 0.0,
        "consistency_compute_seconds": 0.0,
        "collaboration_compute_seconds": 0.0,
    }
    component_enabled_seen = {name: None for name in COMPONENTS}

    rows = 0
    for payload in samples.values():
        meta = payload.get("metadata", {})
        stats = meta.get("objective_compute_stats", {}) if isinstance(meta, dict) else {}
        if not isinstance(stats, dict):
            continue
        rows += 1

        for key in sums:
            sums[key] += _safe_float(stats.get(key), 0.0)

        timing = stats.get("timing", {})
        if isinstance(timing, dict):
            for key in timing_sums:
                timing_sums[key] += _safe_float(timing.get(key), 0.0)

        enabled = stats.get("component_enabled", {})
        if isinstance(enabled, dict):
            for comp in COMPONENTS:
                cur = enabled.get(comp)
                if cur is None:
                    continue
                prev = component_enabled_seen.get(comp)
                if prev is None:
                    component_enabled_seen[comp] = bool(cur)
                elif bool(prev) != bool(cur):
                    component_enabled_seen[comp] = "mixed"

    means = {f"{key}_mean": (value / rows if rows > 0 else 0.0) for key, value in sums.items()}
    timing_means = {f"{key}_mean": (value / rows if rows > 0 else 0.0) for key, value in timing_sums.items()}

    return {
        "sample_rows_with_stats": int(rows),
        "sum": sums,
        "mean": means,
        "timing_sum": timing_sums,
        "timing_mean": timing_means,
        "component_enabled_seen": component_enabled_seen,
    }


def _zero_lambda_skip_checks(
    lambdas: Tuple[float, float, float, float],
    agg_stats: Dict[str, Any],
) -> Dict[str, Any]:
    disabled = set(_disabled_components(lambdas))
    checks: Dict[str, Any] = {}
    mean_stats = agg_stats.get("mean", {}) if isinstance(agg_stats, dict) else {}
    for comp in COMPONENTS:
        compute_key = f"{comp}_compute_calls_mean"
        skipped_key = f"{comp}_skipped_due_to_zero_lambda_mean"
        compute_calls = _safe_float(mean_stats.get(compute_key), 0.0)
        skipped_calls = _safe_float(mean_stats.get(skipped_key), 0.0)
        if comp in disabled:
            checks[comp] = {
                "disabled": True,
                "compute_calls_mean": compute_calls,
                "skipped_mean": skipped_calls,
                "pass": bool(abs(compute_calls) <= 1e-12 and skipped_calls > 0.0),
            }
        else:
            checks[comp] = {
                "disabled": False,
                "compute_calls_mean": compute_calls,
                "skipped_mean": skipped_calls,
                "pass": True,
            }
    checks["all_disabled_components_pass"] = bool(
        all(v.get("pass", False) for k, v in checks.items() if isinstance(v, dict) and v.get("disabled", False))
    )
    return checks


def _build_row(
    full_run_dir: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    full_report = _read_json(full_run_dir / "eval_report.json")
    run_report = _read_json(run_dir / "eval_report.json")
    full_cfg = _read_json(full_run_dir / "run_config.json")
    run_cfg = _read_json(run_dir / "run_config.json")

    full_samples = _load_samples(full_run_dir)
    run_samples = _load_samples(run_dir)

    full_metrics = _gold_metrics(full_report)
    run_metrics = _gold_metrics(run_report)
    directional_gain = {
        key: _directional_gain(key, full_metrics[key], run_metrics[key]) for key in full_metrics
    }

    run_lambdas = _parse_lambdas(run_cfg.get("lambdas", "1,1,1,1"))
    disabled = _disabled_components(run_lambdas)

    stability = _stability_diff(full_samples, run_samples).to_dict()
    agg_stats = _aggregate_objective_compute_stats(run_samples)
    skip_checks = _zero_lambda_skip_checks(run_lambdas, agg_stats)

    full_runtime = _safe_float(full_report.get("metrics_secondary", {}).get("runtime_seconds"), 0.0)
    run_runtime = _safe_float(run_report.get("metrics_secondary", {}).get("runtime_seconds"), 0.0)
    full_explain_seconds = _explain_seconds_total(full_samples)
    run_explain_seconds = _explain_seconds_total(run_samples)

    return {
        "run_dir": str(run_dir),
        "lambdas": list(run_lambdas),
        "disabled_components": disabled,
        "metrics": run_metrics,
        "directional_gain_vs_full": directional_gain,
        "timing": {
            "eval_runtime_seconds": run_runtime,
            "eval_runtime_seconds_delta_vs_full": run_runtime - full_runtime,
            "explain_seconds_total": run_explain_seconds,
            "explain_seconds_delta_vs_full": run_explain_seconds - full_explain_seconds,
        },
        "stability_vs_full": stability,
        "objective_compute_stats_agg": agg_stats,
        "zero_lambda_skip_checks": skip_checks,
        "provenance": {
            "device": run_cfg.get("device"),
            "deterministic": run_cfg.get("deterministic"),
            "commit_short": run_report.get("provenance", {}).get("git", {}).get("commit_short"),
        },
    }


def build_report(full_run_dir: Path, ablation_run_dirs: Iterable[Path]) -> Dict[str, Any]:
    full_row = _build_row(full_run_dir=full_run_dir, run_dir=full_run_dir)
    ablation_rows = [_build_row(full_run_dir=full_run_dir, run_dir=run_dir) for run_dir in ablation_run_dirs]

    return {
        "full_run_dir": str(full_run_dir),
        "full_metrics": full_row["metrics"],
        "full_timing": full_row["timing"],
        "rows": [full_row, *ablation_rows],
    }


def _flatten_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        gain = row.get("directional_gain_vs_full", {})
        timing = row.get("timing", {})
        stability = row.get("stability_vs_full", {})
        checks = row.get("zero_lambda_skip_checks", {})
        out.append(
            {
                "run_dir": row.get("run_dir"),
                "lambdas": ",".join(str(x) for x in row.get("lambdas", [])),
                "disabled_components": ",".join(row.get("disabled_components", [])),
                "gain_log_odds": _safe_float(gain.get("log_odds"), 0.0),
                "gain_comp": _safe_float(gain.get("comp"), 0.0),
                "gain_suff": _safe_float(gain.get("suff"), 0.0),
                "gain_a_c": _safe_float(gain.get("a_c"), 0.0),
                "gain_a_s": _safe_float(gain.get("a_s"), 0.0),
                "eval_runtime_delta": _safe_float(timing.get("eval_runtime_seconds_delta_vs_full"), 0.0),
                "explain_seconds_delta": _safe_float(timing.get("explain_seconds_delta_vs_full"), 0.0),
                "selected_changed_ratio": _safe_float(stability.get("selected_changed_ratio"), 0.0),
                "ranking_changed_ratio": _safe_float(stability.get("ranking_changed_ratio"), 0.0),
                "zero_lambda_skip_pass": bool(checks.get("all_disabled_components_pass", True)),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize full + leave-one-out ablation runs on key metrics, timing, stability and zero-lambda skip checks"
    )
    parser.add_argument("--full-run-dir", type=str, required=True)
    parser.add_argument("--ablation-run-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_dir",
        "lambdas",
        "disabled_components",
        "gain_log_odds",
        "gain_comp",
        "gain_suff",
        "gain_a_c",
        "gain_a_s",
        "eval_runtime_delta",
        "explain_seconds_delta",
        "selected_changed_ratio",
        "ranking_changed_ratio",
        "zero_lambda_skip_pass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    full_run_dir = Path(args.full_run_dir)
    ablation_run_dirs = [Path(p) for p in args.ablation_run_dirs]
    report = build_report(full_run_dir, ablation_run_dirs)

    out_json = Path(args.output_json) if args.output_json else (full_run_dir.parent / "ablation_key_metrics_summary.json")
    out_csv = Path(args.output_csv) if args.output_csv else (full_run_dir.parent / "ablation_key_metrics_summary.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, _flatten_rows(report.get("rows", [])))

    print(f"[ablation-summary] rows={len(report.get('rows', []))}")
    print(f"[ablation-summary] json={out_json}")
    print(f"[ablation-summary] csv={out_csv}")


if __name__ == "__main__":
    main()
