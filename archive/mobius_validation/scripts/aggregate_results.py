from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.statistics import bootstrap_ci, paired_difference_summary
from mobius_verify.src.utils import atomic_write_json, atomic_write_text, read_json, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Mobius verification results.")
    parser.add_argument("--results-dir", type=str, required=True)
    return parser


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}{key}." if prefix else f"{key}.", child, out)
    elif isinstance(value, list):
        out[prefix[:-1]] = len(value)
    else:
        out[prefix[:-1]] = value


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _spectra_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for analysis_file in sorted(results_dir.glob("spectra_exact_*/*/*/analysis.json")):
        analysis = read_json(analysis_file)
        metadata = analysis.get("metadata", {})
        row = {
            "result_type": "spectra",
            "experiment_scope": metadata.get("experiment_scope"),
            "task": metadata.get("sample", {}).get("task"),
            "sample_id": metadata.get("sample", {}).get("sample_id"),
            "probe_id": metadata.get("probe_id"),
            "probe_strategy": metadata.get("probe_strategy"),
            "conditioning_mode": metadata.get("conditioning_mode"),
            "n_features": analysis.get("n_features"),
            "n_total_words": metadata.get("n_total_words"),
            "value_std": analysis.get("value_std"),
            "degenerate": analysis.get("degenerate"),
            "mobius_d90": analysis.get("degree_summary", {}).get("mobius_d90"),
            "fourier_d90": analysis.get("degree_summary", {}).get("fourier_d90"),
            "mobius_d95": analysis.get("degree_summary", {}).get("mobius_d95"),
            "fourier_d95": analysis.get("degree_summary", {}).get("fourier_d95"),
            "mobius_k90": analysis.get("sparsity_summary", {}).get("mobius_omp", {}).get("x90"),
            "fourier_k90": analysis.get("sparsity_summary", {}).get("fourier_omp", {}).get("x90"),
            "normalized_support_size": analysis.get("normalized_support", {}).get("mobius_k90_over_candidate_count"),
        }
        rows.append(row)
    for analysis_file in sorted(results_dir.glob("spectra_exact_*/*/*/*/analysis.json")):
        analysis = read_json(analysis_file)
        metadata = analysis.get("metadata", {})
        row = {
            "result_type": "spectra",
            "experiment_scope": metadata.get("experiment_scope"),
            "task": metadata.get("sample", {}).get("task"),
            "sample_id": metadata.get("sample", {}).get("sample_id"),
            "probe_id": metadata.get("probe_id"),
            "probe_strategy": metadata.get("probe_strategy"),
            "conditioning_mode": metadata.get("conditioning_mode"),
            "n_features": analysis.get("n_features"),
            "n_total_words": metadata.get("n_total_words"),
            "value_std": analysis.get("value_std"),
            "degenerate": analysis.get("degenerate"),
            "mobius_d90": analysis.get("degree_summary", {}).get("mobius_d90"),
            "fourier_d90": analysis.get("degree_summary", {}).get("fourier_d90"),
            "mobius_d95": analysis.get("degree_summary", {}).get("mobius_d95"),
            "fourier_d95": analysis.get("degree_summary", {}).get("fourier_d95"),
            "mobius_k90": analysis.get("sparsity_summary", {}).get("mobius_omp", {}).get("x90"),
            "fourier_k90": analysis.get("sparsity_summary", {}).get("fourier_omp", {}).get("x90"),
            "normalized_support_size": analysis.get("normalized_support", {}).get("mobius_k90_over_candidate_count"),
        }
        rows.append(row)
    return rows


def _recovery_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for metrics_file in sorted((results_dir / "recovery").glob("**/metrics.json")):
        data = read_json(metrics_file)
        row: Dict[str, Any] = {"result_type": "recovery"}
        _flatten("", data, row)
        rows.append(row)
    return rows


def _report(rows: List[Dict[str, Any]], recovery_rows: List[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
    nondeg = [row for row in rows if not row.get("degenerate")]
    mobius_d90 = [row.get("mobius_d90") for row in nondeg if row.get("mobius_d90") not in {None, ""}]
    fourier_d90 = [row.get("fourier_d90") for row in nondeg if row.get("fourier_d90") not in {None, ""}]
    mobius_k90 = [row.get("mobius_k90") for row in nondeg if row.get("mobius_k90") not in {None, ""}]
    fourier_k90 = [row.get("fourier_k90") for row in nondeg if row.get("fourier_k90") not in {None, ""}]
    supported_low_degree = sum(1 for x in mobius_d90 if int(x) <= 4)
    # A null d90 means the threshold was not reached and must remain in the denominator.
    low_degree_rate = supported_low_degree / len(nondeg) if nondeg else None
    paired_rows = [
        (row.get("mobius_k90"), row.get("fourier_k90"))
        for row in nondeg
        if row.get("mobius_k90") not in {None, ""}
        and row.get("fourier_k90") not in {None, ""}
    ]
    paired = (
        paired_difference_summary(
            [float(left) for left, _ in paired_rows],
            [float(right) for _, right in paired_rows],
        )
        if paired_rows
        else {}
    )
    recovery_ok = [row for row in recovery_rows if row.get("status") == "ok"]
    mobius_r2 = [
        float(row["test_r2"])
        for row in recovery_ok
        if row.get("method") == "mobius_lasso" and row.get("test_r2") not in {None, ""}
    ]
    gbt_r2 = [
        float(row["test_r2"])
        for row in recovery_ok
        if row.get("method") == "sklearn_gbt" and row.get("test_r2") not in {None, ""}
    ]
    payload = {
        "sample_count": len(nondeg),
        "mobius_d90_ci": bootstrap_ci(mobius_d90) if mobius_d90 else {},
        "fourier_d90_ci": bootstrap_ci(fourier_d90) if fourier_d90 else {},
        "mobius_low_degree_rate_d90_le_4": low_degree_rate,
        "mobius_d90_reached_count": int(len(mobius_d90)),
        "mobius_d90_not_reached_count": int(len(nondeg) - len(mobius_d90)),
        "mobius_minus_fourier_k90": paired,
        "mobius_recovery_r2": bootstrap_ci(mobius_r2) if mobius_r2 else {},
        "gbt_recovery_r2": bootstrap_ci(gbt_r2) if gbt_r2 else {},
        "verdict": "Inconclusive",
    }
    report = f"""# Executive Summary

This report was generated from the currently available result files. The verdict is `Inconclusive` until enough real-task sample-level results are present.

# Exact Structural Results

- Non-degenerate exact/probe spectra: {len(nondeg)}
- Mobius d90 <= 4 rate: {low_degree_rate}
- Mobius d90 reached / not reached: {len(mobius_d90)} / {len(nondeg) - len(mobius_d90)}
- Mobius d90 summary: {payload['mobius_d90_ci']}
- Fourier d90 summary: {payload['fourier_d90_ci']}
- Paired Mobius-Fourier k90 difference: {paired}

# Limited-query Results

- Mobius recovery test R2 summary: {payload['mobius_recovery_r2']}
- GBT recovery test R2 summary: {payload['gbt_recovery_r2']}

# Evidence For the Hypothesis

Generated once exact/probe and recovery rows contain sufficient non-degenerate real samples.

# Evidence Against the Hypothesis

Generated once counterexamples and failed samples are present in the aggregate tables.

# Confounders and Limitations

Feature granularity, mask operator, Mobius non-orthogonality, exact small-n/probe conditioning, and LASSO conditioning all remain explicit confounders. Historical `exact_default` tables were collected from raw target verbalizer scores despite the old `predicted_class_margin` label; they are sensitivity evidence, not the new margin-based primary experiment.

# Final Verdict

Inconclusive
"""
    return payload, report


def main() -> None:
    args = build_parser().parse_args()
    results_dir = resolve_project_path(
        args.results_dir,
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "exact_default",
    )
    aggregate_dir = results_dir / "aggregate"
    spectra_rows = _spectra_rows(results_dir)
    recovery_rows = _recovery_rows(results_dir)
    _write_csv(aggregate_dir / "sample_level_metrics.csv", spectra_rows)
    _write_csv(aggregate_dir / "recovery_metrics.csv", recovery_rows)
    hypothesis, report = _report(spectra_rows, recovery_rows)
    atomic_write_json(aggregate_dir / "hypothesis_results.json", hypothesis)
    atomic_write_text(aggregate_dir / "hypothesis_report.md", report)


if __name__ == "__main__":
    main()
