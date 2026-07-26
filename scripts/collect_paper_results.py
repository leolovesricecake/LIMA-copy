"""Collect P0/P1 run and audit artifacts into paper-ready tidy tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.statistics import paired_cluster_summary
from mobius.core.artifacts import (
    load_observation_artifact,
    load_surrogate_artifact,
)
from mobius.core.runtime import atomic_write_json, ensure_dir


LONG_FIELDS = (
    "experiment",
    "role",
    "dataset",
    "model",
    "method",
    "run_id",
    "seed",
    "sample_id",
    "variant",
    "distribution",
    "metric",
    "value",
    "source",
)

COMPARISON_FIELDS = (
    "experiment",
    "comparison",
    "metric",
    "distribution",
    "left_role",
    "right_role",
    "left_mean",
    "right_mean",
    "mean_difference",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
    "sample_count",
    "pair_count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the paper-result collection CLI."""

    parser = argparse.ArgumentParser(
        description="Collect role-labelled runs and P0/P1 audits."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Role-labelled run in ROLE=PATH form; roles may repeat across seeds.",
    )
    parser.add_argument("--audit-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="docs/paper-results")
    parser.add_argument("--seed", type=int, default=260726)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    """Load a JSONL artifact if it exists."""

    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_runs(raw: Sequence[str]) -> list[Tuple[str, Path]]:
    """Parse repeated ROLE=PATH run specifications."""

    output = []
    for value in raw:
        if "=" not in str(value):
            raise ValueError(f"Run must use ROLE=PATH syntax: {value!r}")
        role, path = str(value).split("=", 1)
        if not role.strip() or not path.strip():
            raise ValueError(f"Run must use nonempty ROLE=PATH syntax: {value!r}")
        output.append((role.strip(), Path(path).resolve()))
    return output


def _run_identity(run_dir: Path) -> Dict[str, Any]:
    """Read common run identity fields for tidy rows."""

    payload = _load_json(run_dir / "run.json")
    config = dict(payload["scientific_config"])
    dataset = dict(config.get("dataset", {}))
    model = dict(config.get("model", {}))
    model_name = str(model.get("model_path", model.get("type", "model"))).rstrip(
        "/"
    ).split("/")[-1]
    return {
        "payload": payload,
        "config": config,
        "dataset": dataset.get("name"),
        "model": model_name,
        "method": config.get("method"),
        "run_id": payload.get("run_id"),
        "seed": config.get("seed"),
    }


def _base_row(
    *,
    experiment: str,
    role: str,
    identity: Mapping[str, Any],
    sample_id: str,
    variant: str,
    distribution: str,
    metric: str,
    value: float,
    source: str,
) -> Dict[str, Any]:
    """Build one normalized long-table metric row."""

    return {
        "experiment": experiment,
        "role": role,
        "dataset": identity["dataset"],
        "model": identity["model"],
        "method": identity["method"],
        "run_id": identity["run_id"],
        "seed": identity["seed"],
        "sample_id": str(sample_id),
        "variant": str(variant),
        "distribution": str(distribution),
        "metric": str(metric),
        "value": float(value),
        "source": str(source),
    }


def _faithfulness_rows(
    role: str,
    run_dir: Path,
    identity: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Collect standard sample-level faithfulness curves."""

    target = str(identity["config"].get("target_mode", "predicted"))
    rows = []
    source = run_dir / f"curves-{target}.jsonl"
    for sample in _load_jsonl(source):
        for metric, value in dict(sample.get("metrics", {})).items():
            rows.append(
                _base_row(
                    experiment="faithfulness",
                    role=role,
                    identity=identity,
                    sample_id=str(sample["sample_id"]),
                    variant=role,
                    distribution="",
                    metric=str(metric),
                    value=float(value),
                    source=str(source),
                )
            )
    return rows


def _heldout_rows(
    audit_dir: Path,
    roles_by_path: Mapping[str, list[Tuple[str, Mapping[str, Any]]]],
) -> list[Dict[str, Any]]:
    """Collect sample-level shared held-out metrics from one audit."""

    index_path = audit_dir / "evaluation-index.json"
    if not index_path.is_file():
        return []
    index = _load_json(index_path)
    rows = []
    for entry in index["evaluations"]:
        resolved = str(Path(entry["run_path"]).resolve())
        for role, identity in roles_by_path.get(resolved, []):
            report = _load_json(audit_dir / entry["report"])
            for sample in report["rows"]:
                for distribution, metrics in sample["distributions"].items():
                    for metric in ("r2", "nrmse_range", "mae"):
                        value = metrics.get(metric)
                        if value is None:
                            continue
                        rows.append(
                            _base_row(
                                experiment="surrogate_heldout",
                                role=role,
                                identity=identity,
                                sample_id=str(sample["sample_id"]),
                                variant=role,
                                distribution=distribution,
                                metric=metric,
                                value=float(value),
                                source=str(index_path),
                            )
                        )
    return rows


def _interaction_rows(
    audit_dir: Path,
    roles_by_path: Mapping[str, list[Tuple[str, Mapping[str, Any]]]],
) -> list[Dict[str, Any]]:
    """Collect edge-level exact interaction verification metrics."""

    source = audit_dir / "rows.jsonl"
    rows = []
    for edge in _load_jsonl(source):
        if "run_path" not in edge or "row_type" not in edge:
            continue
        resolved = str(Path(str(edge["run_path"])).resolve())
        for role, identity in roles_by_path.get(resolved, []):
            variant = str(edge["row_type"])
            if variant == "selected":
                sources = ",".join(edge.get("selection_sources", []))
                variant = f"selected:{sources}"
            for metric in (
                "estimated_coefficient",
                "exact_coefficient",
                "absolute_error",
                "squared_error",
            ):
                rows.append(
                    _base_row(
                        experiment="interaction_verification",
                        role=role,
                        identity=identity,
                        sample_id=str(edge["sample_id"]),
                        variant=variant,
                        distribution=f"parent_count_{edge.get('parent_count')}",
                        metric=metric,
                        value=float(edge[metric]),
                        source=str(source),
                    )
                )
    return rows


def _hierarchy_rows(
    audit_dir: Path,
    roles_by_path: Mapping[str, list[Tuple[str, Mapping[str, Any]]]],
) -> list[Dict[str, Any]]:
    """Collect hierarchy support and removal faithfulness rows."""

    summary_path = audit_dir / "summary.json"
    rows_path = audit_dir / "rows.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        return []
    summary = _load_json(summary_path)
    if "none_run" not in summary or "strict_run" not in summary:
        return []
    none_path = str(Path(summary["none_run"]).resolve())
    strict_path = str(Path(summary["strict_run"]).resolve())
    output = []
    for sample in _load_jsonl(rows_path):
        for path, support_key, variant in (
            (none_path, "none_support", "none"),
            (strict_path, "strict_support", "strict"),
        ):
            for role, identity in roles_by_path.get(path, []):
                for metric, value in sample[support_key].items():
                    output.append(
                        _base_row(
                            experiment="hierarchy_support",
                            role=role,
                            identity=identity,
                            sample_id=str(sample["sample_id"]),
                            variant=variant,
                            distribution="",
                            metric=metric,
                            value=float(value),
                            source=str(rows_path),
                        )
                    )
        for role, identity in roles_by_path.get(none_path, []):
            for group, metrics in sample["faithfulness"][
                "remove_parent_group"
            ].items():
                for metric, value in metrics.items():
                    output.append(
                        _base_row(
                            experiment="hierarchy_removal",
                            role=role,
                            identity=identity,
                            sample_id=str(sample["sample_id"]),
                            variant=f"remove_parent_group_{group}",
                            distribution="",
                            metric=metric,
                            value=float(value),
                            source=str(rows_path),
                        )
                    )
    return output


def _assert_controlled_artifacts(
    runs: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    """Enforce A/C observation and B/C surrogate identity by seed and sample."""

    grouped: Dict[Tuple[str, Any], Path] = {}
    for role, path in runs:
        identity = _run_identity(path)
        grouped[(role.upper(), identity["seed"])] = path
    checks = []
    for seed in sorted({key[1] for key in grouped}, key=str):
        a = grouped.get(("A", seed))
        b = grouped.get(("B", seed))
        c = grouped.get(("C", seed))
        if a is not None and c is not None:
            common = sorted(
                {path.stem for path in (a / "samples").glob("*.json")}
                & {path.stem for path in (c / "samples").glob("*.json")}
            )
            if not common:
                raise ValueError(f"A/C have no common samples for seed={seed}.")
            mismatches = []
            for sample_id in common:
                left = load_observation_artifact(
                    a / "observations" / f"{sample_id}.npz"
                )
                right = load_observation_artifact(
                    c / "observations" / f"{sample_id}.npz"
                )
                if left["digest"] != right["digest"]:
                    mismatches.append(sample_id)
            if mismatches:
                raise ValueError(
                    f"A/C observation mismatch for seed={seed}: {mismatches[:5]}"
                )
            checks.append(
                {
                    "seed": seed,
                    "check": "A_C_observation_digest",
                    "sample_count": len(common),
                    "status": "ok",
                }
            )
        if b is not None and c is not None:
            common = sorted(
                {path.stem for path in (b / "samples").glob("*.json")}
                & {path.stem for path in (c / "samples").glob("*.json")}
            )
            if not common:
                raise ValueError(f"B/C have no common samples for seed={seed}.")
            mismatches = []
            for sample_id in common:
                left = load_surrogate_artifact(
                    b / "surrogates" / f"{sample_id}.json"
                )
                right = load_surrogate_artifact(
                    c / "surrogates" / f"{sample_id}.json"
                )
                if left["digest"] != right["digest"]:
                    mismatches.append(sample_id)
            if mismatches:
                raise ValueError(
                    f"B/C surrogate mismatch for seed={seed}: {mismatches[:5]}"
                )
            checks.append(
                {
                    "seed": seed,
                    "check": "B_C_surrogate_digest",
                    "sample_count": len(common),
                    "status": "ok",
                }
            )
    return {"checks": checks}


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap: int,
) -> list[Dict[str, Any]]:
    """Compute predeclared paired E2/E4 comparisons from tidy rows."""

    specifications = (
        ("E2_A_vs_B_heldout", "surrogate_heldout", "A", "B"),
        ("E2_B_vs_C_faithfulness", "faithfulness", "B", "C"),
        ("E2_A_vs_C_faithfulness", "faithfulness", "A", "C"),
        ("E2_A_vs_C_heldout", "surrogate_heldout", "A", "C"),
        ("E4_none_vs_strict_faithfulness", "faithfulness", "none", "strict"),
        ("E4_none_vs_strict_heldout", "surrogate_heldout", "none", "strict"),
    )
    output = []
    for name, experiment, left_role, right_role in specifications:
        relevant = [
            row
            for row in rows
            if row["experiment"] == experiment
            and str(row["role"]).lower() in {
                left_role.lower(),
                right_role.lower(),
            }
        ]
        axes = sorted(
            {
                (str(row["metric"]), str(row["distribution"]))
                for row in relevant
            }
        )
        for metric, distribution in axes:
            left = {
                (row["seed"], row["sample_id"]): float(row["value"])
                for row in relevant
                if str(row["role"]).lower() == left_role.lower()
                and row["metric"] == metric
                and row["distribution"] == distribution
            }
            right = {
                (row["seed"], row["sample_id"]): float(row["value"])
                for row in relevant
                if str(row["role"]).lower() == right_role.lower()
                and row["metric"] == metric
                and row["distribution"] == distribution
            }
            common = sorted(set(left) & set(right), key=str)
            if not common:
                continue
            differences: Dict[str, list[float]] = {}
            for key in common:
                differences.setdefault(str(key[1]), []).append(
                    right[key] - left[key]
                )
            paired = paired_cluster_summary(
                differences,
                seed=int(seed),
                n_bootstrap=int(bootstrap),
            )
            output.append(
                {
                    "experiment": experiment,
                    "comparison": name,
                    "metric": metric,
                    "distribution": distribution,
                    "left_role": left_role,
                    "right_role": right_role,
                    "left_mean": float(np.mean([left[key] for key in common])),
                    "right_mean": float(
                        np.mean([right[key] for key in common])
                    ),
                    "mean_difference": paired["mean"],
                    "ci_low": paired["ci_low"],
                    "ci_high": paired["ci_high"],
                    "paired_effect_dz": paired["paired_effect_dz"],
                    "wilcoxon_p_value": paired["wilcoxon_p_value"],
                    "sample_count": paired["cluster_count"],
                    "pair_count": len(common),
                }
            )
    return output


def _hierarchy_removal_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap: int,
) -> list[Dict[str, Any]]:
    """Compare each parent-group removal ranking against the none ranking."""

    output = []
    for group in (0, 1, 2):
        removal_rows = [
            row
            for row in rows
            if row["experiment"] == "hierarchy_removal"
            and row["variant"] == f"remove_parent_group_{group}"
        ]
        for metric in sorted({str(row["metric"]) for row in removal_rows}):
            left = {
                (row["seed"], row["sample_id"]): float(row["value"])
                for row in rows
                if row["experiment"] == "faithfulness"
                and str(row["role"]).lower() == "none"
                and row["metric"] == metric
            }
            right = {
                (row["seed"], row["sample_id"]): float(row["value"])
                for row in removal_rows
                if row["metric"] == metric
            }
            common = sorted(set(left) & set(right), key=str)
            if not common:
                continue
            differences: Dict[str, list[float]] = {}
            for key in common:
                differences.setdefault(str(key[1]), []).append(
                    right[key] - left[key]
                )
            paired = paired_cluster_summary(
                differences,
                seed=int(seed) + group,
                n_bootstrap=int(bootstrap),
            )
            output.append(
                {
                    "experiment": "hierarchy_removal",
                    "comparison": f"E4_remove_parent_group_{group}_vs_none",
                    "metric": metric,
                    "distribution": "",
                    "left_role": "none",
                    "right_role": f"remove_parent_group_{group}",
                    "left_mean": float(np.mean([left[key] for key in common])),
                    "right_mean": float(np.mean([right[key] for key in common])),
                    "mean_difference": paired["mean"],
                    "ci_low": paired["ci_low"],
                    "ci_high": paired["ci_high"],
                    "paired_effect_dz": paired["paired_effect_dz"],
                    "wilcoxon_p_value": paired["wilcoxon_p_value"],
                    "sample_count": paired["cluster_count"],
                    "pair_count": len(common),
                }
            )
    return output


def _interaction_comparison_rows(
    audit_dirs: Sequence[str | Path],
) -> list[Dict[str, Any]]:
    """Collect selected-vs-random exact interaction comparisons from audit summaries."""

    output = []
    for raw in audit_dirs:
        summary_path = Path(raw) / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _load_json(summary_path)
        if "selected_minus_random" not in summary:
            continue
        selected = dict(summary.get("selected_abs_exact", {}))
        random_control = dict(summary.get("random_abs_exact", {}))
        paired = dict(summary["selected_minus_random"])
        output.append(
            {
                "experiment": "interaction_verification",
                "comparison": (
                    f"E3_selected_vs_random/{summary.get('audit_id', Path(raw).name)}"
                ),
                "metric": "absolute_exact_coefficient",
                "distribution": "distance_matched",
                "left_role": "random_control",
                "right_role": "selected",
                "left_mean": random_control.get("mean"),
                "right_mean": selected.get("mean"),
                "mean_difference": paired.get("mean"),
                "ci_low": paired.get("ci_low"),
                "ci_high": paired.get("ci_high"),
                "paired_effect_dz": paired.get("paired_effect_dz"),
                "wilcoxon_p_value": paired.get("wilcoxon_p_value"),
                "sample_count": paired.get("cluster_count"),
                "pair_count": paired.get("row_count"),
            }
        )
    return output


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV rows."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def collect_paper_results(
    runs: Sequence[Tuple[str, Path]],
    audit_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    seed: int,
    bootstrap: int,
) -> Path:
    """Collect tidy metrics, enforce controls, and compute paired comparisons."""

    destination = ensure_dir(output_dir)
    identities = [(role, path, _run_identity(path)) for role, path in runs]
    roles_by_path: Dict[str, list[Tuple[str, Mapping[str, Any]]]] = {}
    rows: list[Dict[str, Any]] = []
    for role, path, identity in identities:
        roles_by_path.setdefault(str(path.resolve()), []).append((role, identity))
        rows.extend(_faithfulness_rows(role, path, identity))
    for raw in audit_dirs:
        audit = Path(raw)
        rows.extend(_heldout_rows(audit, roles_by_path))
        rows.extend(_interaction_rows(audit, roles_by_path))
        rows.extend(_hierarchy_rows(audit, roles_by_path))
    integrity = _assert_controlled_artifacts(runs)
    comparisons = _comparison_rows(
        rows,
        seed=int(seed),
        bootstrap=int(bootstrap),
    )
    comparisons.extend(
        _hierarchy_removal_comparison_rows(
            rows,
            seed=int(seed),
            bootstrap=int(bootstrap),
        )
    )
    comparisons.extend(_interaction_comparison_rows(audit_dirs))
    _write_csv(destination / "paper_metrics_long.csv", rows, LONG_FIELDS)
    _write_csv(
        destination / "paper_comparisons.csv",
        comparisons,
        COMPARISON_FIELDS,
    )
    atomic_write_json(destination / "integrity_checks.json", integrity)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run paper result collection from CLI arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    destination = collect_paper_results(
        _parse_runs(args.run),
        args.audit_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    print(f"[paper-results-collected] output={destination}")


if __name__ == "__main__":
    main()
