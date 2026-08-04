"""Paper-ready per-cell result collection and cross-cell aggregation."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from mobius.analysis.contracts import (
    assert_same_cell,
    assert_same_seed,
    load_json_object,
    load_run_identity,
)
from mobius.analysis.statistics import paired_cluster_summary
from mobius.core.artifacts import (
    load_observation_artifact,
    load_surrogate_artifact,
)
from mobius.core.results import canonical_digest
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.evaluation.metrics import aml_aopc


METRIC_DIRECTIONS = {
    "aopc": "higher_is_better",
    "aupc": "lower_is_better",
    "comprehensiveness": "higher_is_better",
    "sufficiency": "lower_is_better",
    "aopc_comprehensiveness": "higher_is_better",
    "aopc_sufficiency": "lower_is_better",
}

OVERALL_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "role",
    "method",
    "run_id",
    "target",
    "metric",
    "direction",
    "count",
    "mean",
    "std",
    "source",
)

PAIRED_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "comparison",
    "left_role",
    "right_role",
    "metric",
    "direction",
    "left_mean",
    "right_mean",
    "difference_mean",
    "improvement_mean",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
    "common_sample_count",
    "left_only_count",
    "right_only_count",
    "metric_source",
)

PAIRED_SAMPLE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "comparison",
    "left_role",
    "right_role",
    "sample_id",
    "metric",
    "direction",
    "left_value",
    "right_value",
    "difference",
    "improvement",
    "metric_source",
)

COST_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "role",
    "method",
    "run_id",
    "budget",
    "order",
    "value_function",
    "basis",
    "hierarchy",
    "sampler",
    "projector",
    "completed_count",
    "failed_count",
    "metric",
    "total",
    "per_completed_sample",
    "source",
)

AUDIT_COST_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "audit_kind",
    "metric",
    "total",
    "source",
)

REPRESENTATION_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "analysis",
    "basis",
    "distribution",
    "k",
    "metric",
    "direction",
    "count",
    "mean",
    "std",
    "median",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
)

EXACT_STRUCTURE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "analysis",
    "basis",
    "degree",
    "k",
    "metric",
    "direction",
    "count",
    "mean",
    "std",
    "median",
)

EXACT_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "oracle_scope",
    "length_bin",
    "analysis",
    "method",
    "metric",
    "direction",
    "count",
    "mean",
    "std",
    "median",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
)

SURROGATE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "analysis",
    "role",
    "comparison",
    "distribution",
    "metric",
    "direction",
    "count",
    "mean",
    "std",
    "median",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
)

ATTRIBUTION_SAMPLE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "role",
    "method",
    "sample_id",
    "metric",
    "direction",
    "value",
    "source",
)

SURROGATE_SAMPLE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "role",
    "sample_id",
    "distribution",
    "metric",
    "direction",
    "value",
)

EXACT_PAIR_SAMPLE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "sample_id",
    "method",
    "metric",
    "direction",
    "value",
)

REPRESENTATION_SAMPLE_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "attribution_seed",
    "budget",
    "audit_id",
    "sample_id",
    "distribution",
    "k",
    "basis",
    "metric",
    "direction",
    "value",
)

CELL_CSV_FILES = (
    "attribution-samples.csv",
    "surrogate-heldout-samples.csv",
    "exact-pair-samples.csv",
    "representation-fixed-k-samples.csv",
    "overall-attribution.csv",
    "paired-effects.csv",
    "costs.csv",
    "audit-costs.csv",
    "surrogate-heldout.csv",
    "representation-recovery.csv",
    "exact-structure.csv",
    "exact-interactions.csv",
    "projection-ablation.csv",
    "figure-data/attribution-paired.csv",
)

E1_SUMMARY_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "budget",
    "role",
    "method",
    "metric",
    "direction",
    "sample_count",
    "seed_count_min",
    "seed_count_max",
    "mean",
    "std",
)

E1_PAIRED_SEED_FIELDS = (
    "paper_version",
    "dataset",
    "split",
    "dataset_id",
    "model",
    "model_id",
    "budget",
    "comparison",
    "left_role",
    "right_role",
    "metric",
    "direction",
    "common_sample_count",
    "seed_count_min",
    "seed_count_max",
    "improvement_mean",
    "ci_low",
    "ci_high",
    "paired_effect_dz",
    "wilcoxon_p_value",
)


def _validate_path_id(value: str, *, name: str) -> str:
    """Validate one readable version or directory identifier."""

    normalized = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", normalized):
        raise ValueError(
            f"{name} must use 1-64 letters, digits, '.', '_' or '-'."
        )
    return normalized


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Atomically write deterministic UTF-8 CSV output."""

    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    """Atomically write one UTF-8 text artifact."""

    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(str(value))
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    """Load and validate one JSONL artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL artifact: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at {path}:{line_number}.")
        rows.append(dict(payload))
    return rows


def _require_matching_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Sequence[str],
    *,
    comparison: str,
) -> None:
    """Require controlled runs to match on every non-ablated field."""

    mismatches = [
        field
        for field in fields
        if left.get(field) != right.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"{comparison} differs on controlled fields: {mismatches}"
        )


def _validate_role_semantics(
    identities: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the fixed A/B/C/ProxySPEX/projection role definitions."""

    configs = {
        role: dict(identity["config"])
        for role, identity in identities.items()
    }
    c_config = configs["C"]
    c_degree = int(c_config.get("max_degree", 0))
    if (
        str(c_config.get("method")) != "sparse_mobius"
        or str(c_config.get("basis")) != "deletion_mobius"
        or str(c_config.get("hierarchy")) != "none"
        or str(c_config.get("projector")) != "signed_equal_share"
        or c_degree < 2
    ):
        raise ValueError(
            "Role C must be hierarchy-free, signed deletion-Möbius with "
            "max_degree >= 2."
        )
    a_config = configs["A"]
    if (
        str(a_config.get("method")) != "sparse_mobius"
        or int(a_config.get("max_degree", 0)) != 1
        or str(a_config.get("projector")) != "singleton_only"
    ):
        raise ValueError(
            "Role A must be the degree-1 sparse Möbius additive run."
        )
    controlled_fields = (
        "budget",
        "k",
        "basis",
        "sampler",
        "estimator",
    )
    _require_matching_fields(
        c_config,
        a_config,
        controlled_fields,
        comparison="C vs A",
    )
    proxy_config = configs["PROXYSPEX"]
    if (
        str(proxy_config.get("method")) != "proxyspex"
        or int(proxy_config.get("max_order", 0)) != c_degree
    ):
        raise ValueError(
            "Role PROXYSPEX must be a matched-order native ProxySPEX run."
        )
    _require_matching_fields(
        c_config,
        proxy_config,
        ("budget", "k"),
        comparison="C vs PROXYSPEX",
    )
    optional_projectors = {
        "B": "singleton_only",
        "ABSOLUTE": "absolute_equal_share",
    }
    for role, projector in optional_projectors.items():
        if role not in configs:
            continue
        config = configs[role]
        if (
            str(config.get("method")) != "sparse_mobius"
            or int(config.get("max_degree", 0)) != c_degree
            or str(config.get("hierarchy")) != "none"
            or str(config.get("projector")) != projector
        ):
            raise ValueError(
                f"Role {role} does not match projector={projector}."
            )
        _require_matching_fields(
            c_config,
            config,
            controlled_fields,
            comparison=f"C vs {role}",
        )
    first_order_roles = {
        "OCCLUSION": "word_occlusion",
        "LIME": "word_lime",
    }
    for role, method in first_order_roles.items():
        if role not in configs:
            continue
        if str(configs[role].get("method")) != method:
            raise ValueError(f"Role {role} must use method={method}.")
    if "STRICT" in configs:
        strict_config = configs["STRICT"]
        if (
            str(strict_config.get("method")) != "sparse_mobius"
            or int(strict_config.get("max_degree", 0)) != c_degree
            or str(strict_config.get("hierarchy")) != "strict"
            or str(strict_config.get("projector")) != "signed_equal_share"
        ):
            raise ValueError(
                "Role STRICT must be C's degree-matched strict-hierarchy run."
            )
        _require_matching_fields(
            c_config,
            strict_config,
            controlled_fields,
            comparison="C vs STRICT",
        )


def _load_roles(
    runs: Mapping[str, str | Path],
) -> Dict[str, Dict[str, Any]]:
    """Load unique role-labelled runs for one dataset/model/seed cell."""

    identities: Dict[str, Dict[str, Any]] = {}
    for raw_role, path in runs.items():
        role = str(raw_role).strip().upper()
        if not role:
            raise ValueError("Run roles must be nonempty.")
        if role in identities:
            raise ValueError(f"Duplicate run role: {role}")
        identities[role] = load_run_identity(path)
    missing = sorted({"A", "C", "PROXYSPEX"} - set(identities))
    if missing:
        raise ValueError(f"Paper cell lacks required roles: {missing}")
    assert_same_cell(list(identities.values()))
    assert_same_seed(list(identities.values()))
    _validate_role_semantics(identities)
    return identities


def _curve_metrics(
    row: Mapping[str, Any],
    *,
    q_values: Sequence[int],
) -> Dict[str, tuple[float, str]]:
    """Read full-trajectory metrics and normalize only q-grid AOPC variants."""

    metrics = {
        str(name): (float(value), "stored_curve")
        for name, value in dict(row.get("metrics", {})).items()
        if value is not None
    }
    per_q = {
        int(q): {
            str(name): float(value)
            for name, value in dict(values).items()
        }
        for q, values in dict(row.get("per_q", {})).items()
    }
    missing = sorted(set(int(value) for value in q_values) - set(per_q))
    if missing:
        raise ValueError(
            f"Curve {row.get('sample_id')} lacks q values {missing}."
        )
    for metric in ("comprehensiveness", "sufficiency"):
        output_name = f"aopc_{metric}"
        metrics[output_name] = (
            aml_aopc(per_q, q_values, metric),
            f"recomputed_q_grid:{','.join(str(value) for value in q_values)}",
        )
    return metrics


def _load_role_curves(
    identity: Mapping[str, Any],
    *,
    q_values: Sequence[int],
) -> Dict[str, Dict[str, tuple[float, str]]]:
    """Load target-aligned sample metrics for one run."""

    target = str(dict(identity["contract"])["target_mode"])
    path = Path(identity["root"]) / f"curves-{target}.jsonl"
    return {
        str(row["sample_id"]): _curve_metrics(row, q_values=q_values)
        for row in _load_jsonl(path)
    }


def _sample_std(values: Sequence[float]) -> float:
    """Compute sample standard deviation with a stable singleton value."""

    return (
        float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
        if len(values) > 1
        else 0.0
    )


def _cell_context(
    paper_version: str,
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build columns shared by every paper table in one cell."""

    contract = dict(identity["contract"])
    dataset = dict(contract["dataset"])
    model = dict(contract["model"])
    return {
        "paper_version": paper_version,
        "dataset": dataset.get("name"),
        "split": dataset.get("split"),
        "dataset_id": identity["dataset_id"],
        "model": model.get("model_path") or model.get("type"),
        "model_id": identity["model_id"],
        "attribution_seed": identity["seed"],
        "budget": int(dict(identity["config"]).get("budget", 0)),
    }


def _attribution_sample_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Mapping[str, tuple[float, str]]]],
) -> list[Dict[str, Any]]:
    """Flatten every successful sample metric before any seed aggregation."""

    context = _cell_context(paper_version, identities["C"])
    rows = []
    for role, identity in sorted(identities.items()):
        for sample_id, metrics in sorted(curves[role].items()):
            for metric, (value, source) in sorted(metrics.items()):
                if metric not in METRIC_DIRECTIONS:
                    continue
                rows.append(
                    {
                        **context,
                        "role": role,
                        "method": identity["method"],
                        "sample_id": sample_id,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "value": float(value),
                        "source": str(source),
                    }
                )
    return rows


def _overall_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Mapping[str, tuple[float, str]]]],
) -> list[Dict[str, Any]]:
    """Aggregate sample faithfulness metrics for the main paper table."""

    context = _cell_context(paper_version, identities["C"])
    rows = []
    for role, identity in sorted(identities.items()):
        metrics = sorted(
            {
                metric
                for sample in curves[role].values()
                for metric in sample
                if metric in METRIC_DIRECTIONS
            }
        )
        for metric in metrics:
            values = [
                float(sample[metric][0])
                for sample in curves[role].values()
                if metric in sample
            ]
            sources = sorted(
                {
                    str(sample[metric][1])
                    for sample in curves[role].values()
                    if metric in sample
                }
            )
            rows.append(
                {
                    **context,
                    "role": role,
                    "method": identity["method"],
                    "run_id": identity["run_id"],
                    "target": dict(identity["contract"])["target_mode"],
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS[metric],
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "std": _sample_std(values),
                    "source": "|".join(sources),
                }
            )
    return rows


def _paired_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Mapping[str, tuple[float, str]]]],
    *,
    analysis_seed: int,
    bootstrap: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Compute C-centered paired summaries and their sample-level figure data."""

    context = _cell_context(paper_version, identities["C"])
    summaries = []
    sample_rows = []
    for right_role in sorted(set(identities) - {"C"}):
        comparison = f"C_vs_{right_role}"
        left_role = "C"
        left_ids = set(curves[left_role])
        right_ids = set(curves[right_role])
        common = sorted(left_ids.intersection(right_ids))
        for metric, direction in METRIC_DIRECTIONS.items():
            differences: Dict[str, list[float]] = {}
            improvements: Dict[str, list[float]] = {}
            left_values = []
            right_values = []
            sources = set()
            for sample_id in common:
                left_metric = curves[left_role][sample_id].get(metric)
                right_metric = curves[right_role][sample_id].get(metric)
                if left_metric is None or right_metric is None:
                    continue
                left_value = float(left_metric[0])
                right_value = float(right_metric[0])
                difference = left_value - right_value
                multiplier = (
                    1.0 if direction == "higher_is_better" else -1.0
                )
                improvement = multiplier * difference
                source = (
                    left_metric[1]
                    if left_metric[1] == right_metric[1]
                    else f"{left_metric[1]}|{right_metric[1]}"
                )
                sources.add(str(source))
                differences[sample_id] = [difference]
                improvements[sample_id] = [improvement]
                left_values.append(left_value)
                right_values.append(right_value)
                sample_rows.append(
                    {
                        **context,
                        "comparison": comparison,
                        "left_role": left_role,
                        "right_role": right_role,
                        "sample_id": sample_id,
                        "metric": metric,
                        "direction": direction,
                        "left_value": left_value,
                        "right_value": right_value,
                        "difference": difference,
                        "improvement": improvement,
                        "metric_source": source,
                    }
                )
            difference_summary = paired_cluster_summary(
                differences,
                seed=int(analysis_seed),
                n_bootstrap=int(bootstrap),
            )
            improvement_summary = paired_cluster_summary(
                improvements,
                seed=int(analysis_seed),
                n_bootstrap=int(bootstrap),
            )
            summaries.append(
                {
                    **context,
                    "comparison": comparison,
                    "left_role": left_role,
                    "right_role": right_role,
                    "metric": metric,
                    "direction": direction,
                    "left_mean": (
                        float(np.mean(left_values))
                        if left_values
                        else None
                    ),
                    "right_mean": (
                        float(np.mean(right_values))
                        if right_values
                        else None
                    ),
                    "difference_mean": difference_summary["mean"],
                    "improvement_mean": improvement_summary["mean"],
                    "ci_low": improvement_summary["ci_low"],
                    "ci_high": improvement_summary["ci_high"],
                    "paired_effect_dz": improvement_summary[
                        "paired_effect_dz"
                    ],
                    "wilcoxon_p_value": improvement_summary[
                        "wilcoxon_p_value"
                    ],
                    "common_sample_count": improvement_summary[
                        "cluster_count"
                    ],
                    "left_only_count": len(left_ids - right_ids),
                    "right_only_count": len(right_ids - left_ids),
                    "metric_source": "|".join(sorted(sources)),
                }
            )
    summaries.sort(
        key=lambda row: (row["comparison"], row["metric"])
    )
    sample_rows.sort(
        key=lambda row: (
            row["comparison"],
            row["metric"],
            row["sample_id"],
        )
    )
    return summaries, sample_rows


def _sampler_name(config: Mapping[str, Any]) -> str:
    """Extract one compact sampler label from a scientific configuration."""

    sampler = config.get("sampler")
    if isinstance(sampler, Mapping):
        return str(sampler.get("name", "sampler"))
    return str(sampler or "")


def _cost_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Mapping[str, tuple[float, str]]]],
) -> list[Dict[str, Any]]:
    """Collect attribution-only model calls, logical values, and runtime costs."""

    context = _cell_context(paper_version, identities["C"])
    output = []
    for role, identity in sorted(identities.items()):
        root = Path(identity["root"])
        report = load_json_object(root / "metrics.json")
        status_path = root / "status.json"
        status = load_json_object(status_path) if status_path.is_file() else {}
        config = dict(identity["config"])
        cost = dict(report.get("attribution_cost", {}))
        completed = int(
            status.get(
                "completed_count",
                report.get("evaluated_count", len(curves[role])),
            )
        )
        failed = int(
            status.get("failed_count", report.get("failed_count", 0))
        )
        for metric in (
            "attribution_budget_used",
            "logical_unique_queries",
            "physical_values_scored",
            "model_forward_calls",
            "batch_calls",
            "batch_rows",
            "elapsed_seconds",
        ):
            value = cost.get(metric)
            if value is None:
                continue
            total = float(value)
            output.append(
                {
                    **context,
                    "role": role,
                    "method": identity["method"],
                    "run_id": identity["run_id"],
                    "budget": config.get("budget"),
                    "order": config.get(
                        "max_degree",
                        config.get("max_order"),
                    ),
                    "value_function": config.get("value_function"),
                    "basis": config.get("basis"),
                    "hierarchy": config.get("hierarchy"),
                    "sampler": _sampler_name(config),
                    "projector": config.get("projector"),
                    "completed_count": completed,
                    "failed_count": failed,
                    "metric": metric,
                    "total": total,
                    "per_completed_sample": (
                        total / completed if completed > 0 else None
                    ),
                    "source": str((root / "metrics.json").resolve()),
                }
            )
    return output


def _audit_cost_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Collect analysis-query costs separately from attribution costs."""

    context = _cell_context(paper_version, identity)
    rows = []
    for audit in audits:
        root = Path(str(audit["path"]))
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.json"
        if summary_path.is_file():
            payload = load_json_object(summary_path)
        elif manifest_path.is_file():
            payload = load_json_object(manifest_path)
        else:
            continue
        query_cost = dict(payload.get("query_cost", {}))
        for metric in (
            "requested_queries",
            "logical_unique_queries",
            "duplicate_logical_requests",
            "global_cache_hits",
            "global_cache_misses",
            "physical_values_scored",
        ):
            value = query_cost.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    **context,
                    "audit_id": audit["audit_id"],
                    "audit_kind": audit["kind"],
                    "metric": metric,
                    "total": float(value),
                    "source": str(root),
                }
            )
    return rows


def _validate_representation_audit(
    root: Path,
    manifest: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate one representation audit against the paper cell inputs."""

    metadata = dict(manifest.get("metadata", {}))
    expected = dict(identities["C"]["contract"])
    actual = {
        "dataset": metadata.get("dataset"),
        "model": metadata.get("model"),
        "prompt": dict(metadata.get("prompt", {})),
        "chunker": metadata.get("chunker"),
        "eval_granularity": metadata.get("eval_granularity"),
        "value_function": metadata.get("value_function"),
        "target_mode": metadata.get("target_mode"),
    }
    if actual != expected:
        raise ValueError(
            f"Representation audit contract does not match: {root}"
        )
    if int(metadata.get("attribution_seed", -1)) != int(
        identities["C"]["seed"]
    ):
        raise ValueError(
            f"Representation audit seed does not match: {root}"
        )
    audit_runs = {
        str(row.get("role", "")).upper(): str(
            Path(row["path"]).resolve()
        )
        for row in metadata.get("runs", [])
        if row.get("role") and row.get("path")
    }
    for role in ("C", "PROXYSPEX"):
        expected_path = str(Path(identities[role]["root"]).resolve())
        if audit_runs.get(role) != expected_path:
            raise ValueError(
                f"Representation audit references a different {role} run."
            )


def _validate_exact_pair_audit(
    root: Path,
    manifest: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate an exact pair oracle against the current C/ProxySPEX cell."""

    metadata = dict(manifest.get("metadata", {}))
    expected = dict(identities["C"]["contract"])
    actual = {
        "dataset": metadata.get("dataset"),
        "model": metadata.get("model"),
        "prompt": dict(metadata.get("prompt", {})),
        "chunker": metadata.get("chunker"),
        "eval_granularity": metadata.get("eval_granularity"),
        "value_function": metadata.get("value_function"),
        "target_mode": metadata.get("target_mode"),
    }
    if actual != expected:
        raise ValueError(f"Exact pair audit contract does not match: {root}")
    if int(metadata.get("attribution_seed", -1)) != int(
        identities["C"]["seed"]
    ):
        raise ValueError(f"Exact pair audit seed does not match: {root}")
    audit_runs = {
        str(row.get("role", "")).upper(): str(
            Path(row["path"]).resolve()
        )
        for row in metadata.get("runs", [])
        if row.get("role") and row.get("path")
    }
    for role in ("C", "PROXYSPEX"):
        expected_path = str(Path(identities[role]["root"]).resolve())
        if audit_runs.get(role) != expected_path:
            raise ValueError(
                f"Exact pair audit references a different {role} run."
            )


def _audit_index(
    audit_dirs: Sequence[str | Path],
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], Path | None, Dict[str, Any]]:
    """Index canonical audits while allowing an E1-only paper cell."""

    output = []
    representation: tuple[Path, Dict[str, Any]] | None = None
    for raw in audit_dirs:
        root = Path(raw).resolve()
        manifest_path = root / "manifest.json"
        manifest = (
            load_json_object(manifest_path)
            if manifest_path.is_file()
            else {}
        )
        summary_path = root / "summary.json"
        summary = (
            load_json_object(summary_path)
            if summary_path.is_file()
            else {}
        )
        kind = str(manifest.get("kind", ""))
        if not kind:
            if "none_run" in summary and "strict_run" in summary:
                kind = "hierarchy"
            elif "selected_minus_random" in summary:
                kind = "interaction_verification"
            elif (root / "evaluation-index.json").is_file():
                kind = "shared_surrogate_heldout"
            else:
                kind = "audit"
        if kind == "representation":
            if not manifest:
                raise ValueError(
                    f"Representation audit has no manifest: {root}"
                )
            _validate_representation_audit(root, manifest, identities)
            if representation is not None:
                raise ValueError(
                    "A paper cell accepts exactly one representation audit."
                )
            if not summary:
                raise ValueError(
                    f"Representation audit has no summary: {root}"
                )
            representation = (root, summary)
        elif kind == "exact_pair_oracle":
            if not manifest or not summary:
                raise ValueError(
                    f"Exact pair audit requires manifest and summary: {root}"
                )
            _validate_exact_pair_audit(root, manifest, identities)
        output.append(
            {
                "kind": kind,
                "audit_id": str(
                    manifest.get(
                        "audit_id",
                        summary.get("audit_id", root.name),
                    )
                ),
                "path": str(root),
                "manifest": (
                    str(manifest_path.resolve())
                    if manifest_path.is_file()
                    else None
                ),
                "summary": (
                    str(summary_path.resolve())
                    if summary_path.is_file()
                    else None
                ),
            }
        )
    return (
        sorted(output, key=lambda row: (row["kind"], row["audit_id"])),
        representation[0] if representation is not None else None,
        representation[1] if representation is not None else {},
    )


def _stat_columns(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Select common aggregate and paired-statistic columns."""

    return {
        "count": payload.get(
            "count",
            payload.get("cluster_count"),
        ),
        "mean": payload.get("mean"),
        "std": payload.get("std"),
        "median": payload.get("median"),
        "ci_low": payload.get("ci_low"),
        "ci_high": payload.get("ci_high"),
        "paired_effect_dz": payload.get("paired_effect_dz"),
        "wilcoxon_p_value": payload.get("wilcoxon_p_value"),
    }


def _representation_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Flatten representation, recovery, support, and compression evidence."""

    context = _cell_context(paper_version, identity)
    audit_id = str(summary["audit_id"])
    basis_summary = dict(summary.get("basis", {}))
    rows = []
    for basis, statistics in dict(
        basis_summary.get("support_size", {})
    ).items():
        rows.append(
            {
                **context,
                "audit_id": audit_id,
                "analysis": "support_size",
                "basis": basis,
                "distribution": "",
                "k": "",
                "metric": "nonzero_support_size",
                "direction": "lower_is_sparser",
                **_stat_columns(dict(statistics)),
            }
        )
    for basis, metrics in dict(
        basis_summary.get("training", {})
    ).items():
        for metric, statistics in dict(metrics).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "analysis": "training_reconstruction",
                    "basis": basis,
                    "distribution": "training",
                    "k": "",
                    "metric": metric,
                    "direction": (
                        "higher_is_better"
                        if metric == "r2"
                        else "lower_is_better"
                    ),
                    **_stat_columns(dict(statistics)),
                }
            )
    for distribution, metrics in dict(
        basis_summary.get("heldout", {})
    ).items():
        for metric, comparison in dict(metrics).items():
            payload = dict(comparison)
            for basis in ("mobius", "fourier"):
                rows.append(
                    {
                        **context,
                        "audit_id": audit_id,
                        "analysis": "heldout_reconstruction",
                        "basis": basis,
                        "distribution": distribution,
                        "k": "",
                        "metric": metric,
                        "direction": payload.get("direction"),
                        **_stat_columns(dict(payload.get(basis, {}))),
                    }
                )
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "analysis": "heldout_basis_comparison",
                    "basis": "mobius_vs_fourier",
                    "distribution": distribution,
                    "k": "",
                    "metric": metric,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(
                        dict(payload.get("mobius_improvement", {}))
                    ),
                }
            )
    for distribution, by_k in dict(
        basis_summary.get("compression", {})
    ).items():
        for k, by_basis in dict(by_k).items():
            for basis, metrics in dict(by_basis).items():
                for metric, statistics in dict(metrics).items():
                    rows.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "analysis": "top_k_refit",
                            "basis": basis,
                            "distribution": distribution,
                            "k": k,
                            "metric": metric,
                            "direction": (
                                "higher_is_better"
                                if metric == "r2"
                                else "lower_is_better"
                            ),
                            **_stat_columns(dict(statistics)),
                        }
                    )
    for basis, metrics in dict(
        basis_summary.get("design_geometry", {})
    ).items():
        for metric, statistics in dict(metrics).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "analysis": "design_geometry",
                    "basis": basis,
                    "distribution": "training",
                    "k": "",
                    "metric": metric,
                    "direction": "diagnostic",
                    **_stat_columns(dict(statistics)),
                }
            )
    for distribution, by_k in dict(
        basis_summary.get("fixed_k", {})
    ).items():
        for k, by_metric in dict(by_k).items():
            for metric, comparison in dict(by_metric).items():
                payload = dict(comparison)
                for basis in ("mobius", "fourier"):
                    rows.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "analysis": "fixed_k_omp",
                            "basis": basis,
                            "distribution": distribution,
                            "k": k,
                            "metric": metric,
                            "direction": payload.get(
                                "direction",
                                "matched_support",
                            ),
                            **_stat_columns(
                                dict(payload.get(basis, {}))
                            ),
                        }
                    )
                if "mobius_improvement" in payload:
                    rows.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "analysis": "fixed_k_basis_comparison",
                            "basis": "mobius_vs_fourier",
                            "distribution": distribution,
                            "k": k,
                            "metric": metric,
                            "direction": "positive_favors_mobius",
                            **_stat_columns(
                                dict(payload["mobius_improvement"])
                            ),
                        }
                    )
    for distribution, targets in dict(
        basis_summary.get("matched_error_support", {})
    ).items():
        for target, comparison in dict(targets).items():
            payload = dict(comparison)
            for basis in ("mobius", "fourier"):
                rows.append(
                    {
                        **context,
                        "audit_id": audit_id,
                        "analysis": "matched_error_support",
                        "basis": basis,
                        "distribution": distribution,
                        "k": "",
                        "metric": target,
                        "direction": "lower_is_better",
                        **_stat_columns(dict(payload.get(basis, {}))),
                    }
                )
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "analysis": "matched_error_basis_comparison",
                    "basis": "mobius_vs_fourier",
                    "distribution": distribution,
                    "k": "",
                    "metric": target,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(
                        dict(payload.get("mobius_improvement", {}))
                    ),
                }
            )
    return rows


def _representation_sample_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    representation_root: Path | None,
) -> list[Dict[str, Any]]:
    """Flatten per-sample fixed-k basis diagnostics before seed aggregation."""

    if representation_root is None:
        return []
    context = _cell_context(paper_version, identity)
    manifest = load_json_object(representation_root / "manifest.json")
    audit_id = str(manifest["audit_id"])
    output = []
    metrics = {
        "r2",
        "nrmse_range",
        "ols_rank",
        "ols_rank_deficient",
        "ols_condition_number",
    }
    for sample in _load_jsonl(representation_root / "representation-rows.jsonl"):
        for distribution, by_k in dict(sample.get("fixed_k", {})).items():
            for k, by_basis in dict(by_k).items():
                for basis, payload in dict(by_basis).items():
                    for metric in metrics:
                        value = dict(payload).get(metric)
                        if value is None:
                            continue
                        output.append(
                            {
                                **context,
                                "audit_id": audit_id,
                                "sample_id": str(sample["sample_id"]),
                                "distribution": str(distribution),
                                "k": str(k),
                                "basis": str(basis),
                                "metric": metric,
                                "direction": (
                                    "higher_is_better"
                                    if metric in {"r2", "ols_rank"}
                                    else "lower_is_better"
                                ),
                                "value": float(value),
                            }
                        )
    return output


def _exact_structure_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Flatten exact full-table coefficient, degree, and top-k structure."""

    context = _cell_context(paper_version, identity)
    audit_id = str(summary["audit_id"])
    exact = dict(summary.get("exact_structure", {}))
    rows = []
    for basis, by_degree in dict(
        exact.get("coefficient_profile", {})
    ).items():
        for degree, metrics in dict(by_degree).items():
            for metric, statistics in dict(metrics).items():
                rows.append(
                    {
                        **context,
                        "audit_id": audit_id,
                        "analysis": "exact_coefficient_profile",
                        "basis": basis,
                        "degree": degree,
                        "k": "",
                        "metric": metric,
                        "direction": "basis_internal_diagnostic",
                        **{
                            key: _stat_columns(dict(statistics)).get(key)
                            for key in ("count", "mean", "std", "median")
                        },
                    }
                )
    for degree, by_basis in dict(
        exact.get("degree_truncation", {})
    ).items():
        for basis, metrics in dict(by_basis).items():
            for metric, statistics in dict(metrics).items():
                rows.append(
                    {
                        **context,
                        "audit_id": audit_id,
                        "analysis": "exact_degree_truncation",
                        "basis": basis,
                        "degree": degree,
                        "k": "",
                        "metric": metric,
                        "direction": (
                            "higher_is_better"
                            if metric == "r2"
                            else "lower_is_better"
                        ),
                        **{
                            key: _stat_columns(dict(statistics)).get(key)
                            for key in ("count", "mean", "std", "median")
                        },
                    }
                )
    for k, by_basis in dict(exact.get("top_k", {})).items():
        for basis, metrics in dict(by_basis).items():
            for metric, statistics in dict(metrics).items():
                rows.append(
                    {
                        **context,
                        "audit_id": audit_id,
                        "analysis": "exact_top_k_refit",
                        "basis": basis,
                        "degree": "",
                        "k": k,
                        "metric": metric,
                        "direction": (
                            "higher_is_better"
                            if metric == "r2"
                            else "lower_is_better"
                        ),
                        **{
                            key: _stat_columns(dict(statistics)).get(key)
                            for key in ("count", "mean", "std", "median")
                        },
                    }
                )
    return rows


def _exact_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Flatten full-table pair-ranking and coefficient-recovery evidence."""

    context = _cell_context(paper_version, identity)
    audit_id = str(summary["audit_id"])
    exact = dict(summary.get("exact_interactions", {}))
    rows = []
    for method, metrics in dict(exact.get("methods", {})).items():
        for metric, statistics in dict(metrics).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "oracle_scope": "full_table",
                    "length_bin": "all",
                    "analysis": "ranking",
                    "method": method,
                    "metric": metric,
                    "direction": "higher_is_better",
                    **_stat_columns(dict(statistics)),
                }
            )
    coefficient_metrics = {
        "mobius_selected_coefficient_mae": "lower_is_better",
        "mobius_selected_sign_agreement": "higher_is_better",
        "mobius_all_pair_coefficient_mae": "lower_is_better",
        "mobius_all_pair_sign_agreement": "higher_is_better",
    }
    for metric, direction in coefficient_metrics.items():
        rows.append(
            {
                **context,
                "audit_id": audit_id,
                "oracle_scope": "full_table",
                "length_bin": "all",
                "analysis": "coefficient_recovery",
                "method": "mobius",
                "metric": metric,
                "direction": direction,
                **_stat_columns(dict(exact.get(metric, {}))),
            }
        )
    comparisons = {
        "mobius_vs_proxyspex": "proxyspex",
        "mobius_vs_random": "random",
    }
    for key, baseline in comparisons.items():
        for metric, statistics in dict(exact.get(key, {})).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "oracle_scope": "full_table",
                    "length_bin": "all",
                    "analysis": "paired_ranking_comparison",
                    "method": f"mobius_vs_{baseline}",
                    "metric": metric,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(dict(statistics)),
                }
            )
    return rows


def _flatten_exact_aggregate(
    context: Mapping[str, Any],
    *,
    audit_id: str,
    oracle_scope: str,
    length_bin: str,
    exact: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Flatten one exact-interaction aggregate and its paired comparisons."""

    rows = []
    for method, metrics in dict(exact.get("methods", {})).items():
        for metric, statistics in dict(metrics).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "oracle_scope": oracle_scope,
                    "length_bin": length_bin,
                    "analysis": "ranking",
                    "method": method,
                    "metric": metric,
                    "direction": "higher_is_better",
                    **_stat_columns(dict(statistics)),
                }
            )
    coefficient_metrics = {
        "mobius_selected_coefficient_mae": "lower_is_better",
        "mobius_selected_sign_agreement": "higher_is_better",
        "mobius_all_pair_coefficient_mae": "lower_is_better",
        "mobius_all_pair_sign_agreement": "higher_is_better",
    }
    for metric, direction in coefficient_metrics.items():
        rows.append(
            {
                **context,
                "audit_id": audit_id,
                "oracle_scope": oracle_scope,
                "length_bin": length_bin,
                "analysis": "coefficient_recovery",
                "method": "mobius",
                "metric": metric,
                "direction": direction,
                **_stat_columns(dict(exact.get(metric, {}))),
            }
        )
    for key, baseline in (
        ("mobius_vs_proxyspex", "proxyspex"),
        ("mobius_vs_random", "random"),
    ):
        for metric, statistics in dict(exact.get(key, {})).items():
            rows.append(
                {
                    **context,
                    "audit_id": audit_id,
                    "oracle_scope": oracle_scope,
                    "length_bin": length_bin,
                    "analysis": "paired_ranking_comparison",
                    "method": f"mobius_vs_{baseline}",
                    "metric": metric,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(dict(statistics)),
                }
            )
    return rows


def _exact_pair_oracle_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Flatten overall and length-stratified exact pair-oracle summaries."""

    context = _cell_context(paper_version, identity)
    rows = []
    for audit in audits:
        if audit.get("kind") != "exact_pair_oracle":
            continue
        summary = load_json_object(Path(str(audit["path"])) / "summary.json")
        audit_id = str(summary.get("audit_id", audit["audit_id"]))
        rows.extend(
            _flatten_exact_aggregate(
                context,
                audit_id=audit_id,
                oracle_scope="pair_oracle",
                length_bin="all",
                exact=dict(summary.get("aggregate", {})),
            )
        )
        for length_bin, aggregate in dict(
            summary.get("by_length_bin", {})
        ).items():
            rows.extend(
                _flatten_exact_aggregate(
                    context,
                    audit_id=audit_id,
                    oracle_scope="pair_oracle",
                    length_bin=str(length_bin),
                    exact=dict(aggregate),
                )
            )
    return rows


def _exact_pair_sample_rows(
    paper_version: str,
    identity: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Flatten per-sample E3 ranking metrics before seed aggregation."""

    context = _cell_context(paper_version, identity)
    output = []
    primary = {"ndcg_at_10", "recall_at_10", "sign_agreement_at_10"}
    for audit in audits:
        if audit.get("kind") != "exact_pair_oracle":
            continue
        root = Path(str(audit["path"]))
        audit_id = str(audit["audit_id"])
        for sample in _load_jsonl(root / "samples.jsonl"):
            for method, metrics in dict(sample.get("method_metrics", {})).items():
                for metric, value in dict(metrics).items():
                    if metric not in primary or value is None:
                        continue
                    output.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "sample_id": str(sample["sample_id"]),
                            "method": str(method),
                            "metric": str(metric),
                            "direction": "higher_is_better",
                            "value": float(value),
                        }
                    )
    return output


def _surrogate_heldout_sample_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Flatten per-sample E2 metrics before seed aggregation."""

    context = _cell_context(paper_version, identities["C"])
    role_by_path = {
        str(Path(identity["root"]).resolve()): role
        for role, identity in identities.items()
    }
    output = []
    for audit in audits:
        if audit.get("kind") != "shared_surrogate_heldout":
            continue
        root = Path(str(audit["path"]))
        index = load_json_object(root / "evaluation-index.json")
        audit_id = str(index.get("audit_id", audit["audit_id"]))
        for entry in index.get("evaluations", []):
            role = role_by_path.get(str(Path(entry["run_path"]).resolve()))
            if role is None:
                continue
            report = load_json_object(root / str(entry["report"]))
            for sample in report.get("rows", []):
                for distribution, metrics in dict(
                    sample.get("distributions", {})
                ).items():
                    for metric in ("r2", "nrmse_range", "mae"):
                        value = dict(metrics).get(metric)
                        if value is None:
                            continue
                        output.append(
                            {
                                **context,
                                "audit_id": audit_id,
                                "role": role,
                                "sample_id": str(sample["sample_id"]),
                                "distribution": str(distribution),
                                "metric": metric,
                                "direction": (
                                    "higher_is_better"
                                    if metric == "r2"
                                    else "lower_is_better"
                                ),
                                "value": float(value),
                            }
                        )
    return output


def _surrogate_heldout_rows(
    paper_version: str,
    identities: Mapping[str, Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    analysis_seed: int,
    bootstrap: int,
) -> list[Dict[str, Any]]:
    """Collect shared held-out reconstruction and C-centered paired effects."""

    context = _cell_context(paper_version, identities["C"])
    role_by_path = {
        str(Path(identity["root"]).resolve()): role
        for role, identity in identities.items()
    }
    output = []
    for audit in audits:
        if audit.get("kind") != "shared_surrogate_heldout":
            continue
        root = Path(str(audit["path"]))
        index = load_json_object(root / "evaluation-index.json")
        audit_id = str(index.get("audit_id", audit["audit_id"]))
        values_by_role: Dict[
            str,
            Dict[str, Dict[str, Dict[str, float]]],
        ] = {}
        for entry in index.get("evaluations", []):
            run_path = str(Path(entry["run_path"]).resolve())
            role = role_by_path.get(run_path)
            if role is None:
                continue
            report = load_json_object(root / str(entry["report"]))
            role_values: Dict[str, Dict[str, Dict[str, float]]] = {}
            for sample in report.get("rows", []):
                sample_id = str(sample["sample_id"])
                for distribution, metrics in dict(
                    sample.get("distributions", {})
                ).items():
                    for metric in ("r2", "nrmse_range", "mae"):
                        value = dict(metrics).get(metric)
                        if value is None:
                            continue
                        role_values.setdefault(distribution, {}).setdefault(
                            metric,
                            {},
                        )[sample_id] = float(value)
            values_by_role[role] = role_values
        for role, distributions in sorted(values_by_role.items()):
            for distribution, metrics in sorted(distributions.items()):
                for metric, by_sample in sorted(metrics.items()):
                    values = list(by_sample.values())
                    output.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "analysis": "run_reconstruction",
                            "role": role,
                            "comparison": "",
                            "distribution": distribution,
                            "metric": metric,
                            "direction": (
                                "higher_is_better"
                                if metric == "r2"
                                else "lower_is_better"
                            ),
                            "count": len(values),
                            "mean": (
                                float(np.mean(values))
                                if values
                                else None
                            ),
                            "std": _sample_std(values),
                            "median": (
                                float(np.median(values))
                                if values
                                else None
                            ),
                            "ci_low": None,
                            "ci_high": None,
                            "paired_effect_dz": None,
                            "wilcoxon_p_value": None,
                        }
                    )
        c_values = values_by_role.get("C", {})
        for role in sorted(set(values_by_role) - {"C"}):
            for distribution in sorted(
                set(c_values).intersection(values_by_role[role])
            ):
                for metric in ("r2", "nrmse_range", "mae"):
                    left = c_values[distribution].get(metric, {})
                    right = values_by_role[role][distribution].get(
                        metric,
                        {},
                    )
                    common = sorted(set(left).intersection(right))
                    multiplier = 1.0 if metric == "r2" else -1.0
                    improvements = {
                        sample_id: [
                            multiplier
                            * (
                                float(left[sample_id])
                                - float(right[sample_id])
                            )
                        ]
                        for sample_id in common
                    }
                    paired = paired_cluster_summary(
                        improvements,
                        seed=int(analysis_seed),
                        n_bootstrap=int(bootstrap),
                    )
                    output.append(
                        {
                            **context,
                            "audit_id": audit_id,
                            "analysis": "paired_comparison",
                            "role": "C",
                            "comparison": f"C_vs_{role}",
                            "distribution": distribution,
                            "metric": metric,
                            "direction": "positive_favors_C",
                            **_stat_columns(paired),
                        }
                    )
    return output


def _common_sidecar_ids(left: Path, right: Path) -> list[str]:
    """List samples with both observation and surrogate sidecars in two runs."""

    left_ids = {
        path.stem for path in (left / "samples").glob("*.json")
    }
    right_ids = {
        path.stem for path in (right / "samples").glob("*.json")
    }
    return sorted(left_ids.intersection(right_ids))


def _integrity_checks(
    identities: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Verify controlled masks and derived projection surrogates."""

    c_root = Path(identities["C"]["root"])
    checks = []
    for role in ("A", "STRICT"):
        if role not in identities:
            continue
        root = Path(identities[role]["root"])
        common = _common_sidecar_ids(c_root, root)
        if not common:
            raise ValueError(f"C/{role} have no common sidecar samples.")
        mismatches = []
        for sample_id in common:
            left = load_observation_artifact(
                c_root / "observations" / f"{sample_id}.npz"
            )
            right = load_observation_artifact(
                root / "observations" / f"{sample_id}.npz"
            )
            if left["digest"] != right["digest"]:
                mismatches.append(sample_id)
        if mismatches:
            raise ValueError(
                f"C/{role} observation mismatch: {mismatches[:5]}"
            )
        checks.append(
            {
                "check": f"C_{role}_observation_digest",
                "sample_count": len(common),
                "status": "ok",
            }
        )
    for role in ("B", "ABSOLUTE"):
        if role not in identities:
            continue
        root = Path(identities[role]["root"])
        common = _common_sidecar_ids(c_root, root)
        if not common:
            raise ValueError(f"C/{role} have no common sidecar samples.")
        mismatches = []
        for sample_id in common:
            left = load_surrogate_artifact(
                c_root / "surrogates" / f"{sample_id}.json"
            )
            right = load_surrogate_artifact(
                root / "surrogates" / f"{sample_id}.json"
            )
            if left["digest"] != right["digest"]:
                mismatches.append(sample_id)
        if mismatches:
            raise ValueError(
                f"C/{role} surrogate mismatch: {mismatches[:5]}"
            )
        checks.append(
            {
                "check": f"C_{role}_surrogate_digest",
                "sample_count": len(common),
                "status": "ok",
            }
        )
    return {"checks": checks}


def collect_paper_cell(
    paper_version: str,
    runs: Mapping[str, str | Path],
    audit_dirs: Sequence[str | Path],
    *,
    output_root: str | Path = "results/paper",
    q_values: Sequence[int] = (5, 10, 20, 50),
    analysis_seed: int = 260730,
    bootstrap: int = 2000,
    overwrite: bool = False,
) -> Path:
    """Generate one immutable dataset/model/seed paper-result cell."""

    version = _validate_path_id(paper_version, name="paper_version")
    identities = _load_roles(runs)
    q_grid = sorted({int(value) for value in q_values})
    if not q_grid or any(value <= 0 or value > 100 for value in q_grid):
        raise ValueError("q_values must contain percentages in 1..100.")
    audits, representation_root, representation_summary = _audit_index(
        audit_dirs,
        identities,
    )
    curves = {
        role: _load_role_curves(identity, q_values=q_grid)
        for role, identity in identities.items()
    }
    attribution_samples = _attribution_sample_rows(
        version,
        identities,
        curves,
    )
    overall = _overall_rows(version, identities, curves)
    paired, paired_samples = _paired_rows(
        version,
        identities,
        curves,
        analysis_seed=int(analysis_seed),
        bootstrap=int(bootstrap),
    )
    costs = _cost_rows(version, identities, curves)
    audit_costs = _audit_cost_rows(
        version,
        identities["C"],
        audits,
    )
    surrogate_heldout = _surrogate_heldout_rows(
        version,
        identities,
        audits,
        analysis_seed=int(analysis_seed),
        bootstrap=int(bootstrap),
    )
    surrogate_samples = _surrogate_heldout_sample_rows(
        version,
        identities,
        audits,
    )
    representation = (
        _representation_rows(
            version,
            identities["C"],
            representation_summary,
        )
        if representation_summary
        else []
    )
    representation_samples = _representation_sample_rows(
        version,
        identities["C"],
        representation_root,
    )
    exact_structure = (
        _exact_structure_rows(
            version,
            identities["C"],
            representation_summary,
        )
        if representation_summary
        else []
    )
    exact = (
        _exact_rows(
            version,
            identities["C"],
            representation_summary,
        )
        if representation_summary
        else []
    )
    exact.extend(
        _exact_pair_oracle_rows(
            version,
            identities["C"],
            audits,
        )
    )
    exact_pair_samples = _exact_pair_sample_rows(
        version,
        identities["C"],
        audits,
    )
    projection = [
        row
        for row in paired
        if row["right_role"] in {"B", "ABSOLUTE"}
    ]
    integrity = _integrity_checks(identities)
    destination = (
        Path(output_root)
        / version
        / "cells"
        / str(identities["C"]["dataset_id"])
        / str(identities["C"]["model_id"])
        / f"budget-{int(dict(identities['C']['config']).get('budget', 0))}"
        / f"seed-{identities['C']['seed']}"
    )
    settings = {
        "q_values": q_grid,
        "analysis_seed": int(analysis_seed),
        "bootstrap": int(bootstrap),
    }
    input_payload = {
        "paper_version": version,
        "runs": [
            {
                "role": role,
                "path": str(identity["root"]),
                "run_id": identity["run_id"],
                "config_fingerprint": identity["config_fingerprint"],
            }
            for role, identity in sorted(identities.items())
        ],
        "audits": audits,
        "settings": settings,
    }
    fingerprint = canonical_digest(input_payload)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = load_json_object(manifest_path)
        if existing.get("input_fingerprint") != fingerprint:
            raise ValueError(
                f"Paper cell already belongs to different inputs: {destination}"
            )
    ensure_dir(destination)
    manifest = {
        "schema_version": "1.0",
        "kind": "paper_result_cell",
        "paper_version": version,
        "dataset_id": identities["C"]["dataset_id"],
        "model_id": identities["C"]["model_id"],
        "budget": int(dict(identities["C"]["config"]).get("budget", 0)),
        "attribution_seed": identities["C"]["seed"],
        "comparison_contract": identities["C"]["contract"],
        "input_fingerprint": fingerprint,
        "representation_audit": (
            str(representation_root)
            if representation_root is not None
            else None
        ),
        **input_payload,
        "files": {
            "attribution_samples": "attribution-samples.csv",
            "surrogate_heldout_samples": "surrogate-heldout-samples.csv",
            "exact_pair_samples": "exact-pair-samples.csv",
            "representation_fixed_k_samples": (
                "representation-fixed-k-samples.csv"
            ),
            "overall_attribution": "overall-attribution.csv",
            "paired_effects": "paired-effects.csv",
            "costs": "costs.csv",
            "audit_costs": "audit-costs.csv",
            "surrogate_heldout": "surrogate-heldout.csv",
            "representation_recovery": "representation-recovery.csv",
            "exact_structure": "exact-structure.csv",
            "exact_interactions": "exact-interactions.csv",
            "projection_ablation": "projection-ablation.csv",
            "attribution_figure_data": (
                "figure-data/attribution-paired.csv"
            ),
            "audit_index": "audit-index.json",
            "integrity_checks": "integrity-checks.json",
        },
    }
    atomic_write_json(manifest_path, manifest)
    _write_csv(
        destination / "attribution-samples.csv",
        attribution_samples,
        ATTRIBUTION_SAMPLE_FIELDS,
    )
    _write_csv(
        destination / "surrogate-heldout-samples.csv",
        surrogate_samples,
        SURROGATE_SAMPLE_FIELDS,
    )
    _write_csv(
        destination / "exact-pair-samples.csv",
        exact_pair_samples,
        EXACT_PAIR_SAMPLE_FIELDS,
    )
    _write_csv(
        destination / "representation-fixed-k-samples.csv",
        representation_samples,
        REPRESENTATION_SAMPLE_FIELDS,
    )
    _write_csv(
        destination / "overall-attribution.csv",
        overall,
        OVERALL_FIELDS,
    )
    _write_csv(
        destination / "paired-effects.csv",
        paired,
        PAIRED_FIELDS,
    )
    _write_csv(destination / "costs.csv", costs, COST_FIELDS)
    _write_csv(
        destination / "audit-costs.csv",
        audit_costs,
        AUDIT_COST_FIELDS,
    )
    _write_csv(
        destination / "surrogate-heldout.csv",
        surrogate_heldout,
        SURROGATE_FIELDS,
    )
    _write_csv(
        destination / "representation-recovery.csv",
        representation,
        REPRESENTATION_FIELDS,
    )
    _write_csv(
        destination / "exact-structure.csv",
        exact_structure,
        EXACT_STRUCTURE_FIELDS,
    )
    _write_csv(
        destination / "exact-interactions.csv",
        exact,
        EXACT_FIELDS,
    )
    _write_csv(
        destination / "projection-ablation.csv",
        projection,
        PAIRED_FIELDS,
    )
    _write_csv(
        destination / "figure-data" / "attribution-paired.csv",
        paired_samples,
        PAIRED_SAMPLE_FIELDS,
    )
    atomic_write_json(destination / "audit-index.json", {"audits": audits})
    atomic_write_json(destination / "integrity-checks.json", integrity)
    return destination


def _cell_manifests(root: Path) -> list[Path]:
    """List completed paper cells in deterministic dataset/model/seed order."""

    return sorted(root.glob("cells/*/*/budget-*/seed-*/manifest.json"))


def _concatenate_cell_csvs(
    manifests: Sequence[Path],
    relative_path: str,
) -> tuple[list[Dict[str, str]], Sequence[str]]:
    """Concatenate one table from all cells while checking identical schemas."""

    rows: list[Dict[str, str]] = []
    fields: Sequence[str] = ()
    for manifest_path in manifests:
        source = manifest_path.parent / relative_path
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current_fields = tuple(reader.fieldnames or ())
            if fields and current_fields != tuple(fields):
                raise ValueError(
                    f"Paper table schema mismatch: {source}"
                )
            fields = current_fields
            rows.extend(dict(row) for row in reader)
    return rows, fields


def _float_field(row: Mapping[str, str], key: str) -> float:
    """Parse one required numeric CSV field."""

    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing numeric field {key!r} in aggregate row.")
    return float(value)


def _cross_seed_e1_tables(
    rows: Sequence[Mapping[str, str]],
    *,
    analysis_seed: int = 260730,
    bootstrap: int = 2000,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Aggregate E1 per sample across seeds before sample-level inference."""

    primary_metrics = {"aupc", "aopc_sufficiency"}
    by_role_sample: Dict[
        tuple[str, ...],
        Dict[str, float],
    ] = {}
    method_by_role: Dict[tuple[str, ...], str] = {}
    context_by_cell: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        metric = str(row["metric"])
        if metric not in primary_metrics:
            continue
        cell = (
            str(row["paper_version"]),
            str(row["dataset"]),
            str(row["split"]),
            str(row["dataset_id"]),
            str(row["model"]),
            str(row["model_id"]),
            str(row["budget"]),
        )
        role = str(row["role"])
        key = (*cell, role, str(row["sample_id"]), metric)
        by_role_sample.setdefault(key, {})[
            str(row["attribution_seed"])
        ] = _float_field(row, "value")
        method_by_role[(*cell, role)] = str(row["method"])
        context_by_cell[cell] = {
            "paper_version": row["paper_version"],
            "dataset": row["dataset"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "budget": int(float(row["budget"])),
        }

    grouped: Dict[tuple[str, ...], list[tuple[str, float, int]]] = {}
    for key, seed_values in by_role_sample.items():
        *cell_and_role, sample_id, metric = key
        group_key = (*cell_and_role, metric)
        grouped.setdefault(group_key, []).append(
            (
                sample_id,
                float(np.mean(list(seed_values.values()))),
                len(seed_values),
            )
        )
    table_rows = []
    for key, sample_values in sorted(grouped.items()):
        *cell, role, metric = key
        context = context_by_cell[tuple(cell)]
        values = [value for _, value, _ in sample_values]
        seed_counts = [count for _, _, count in sample_values]
        table_rows.append(
            {
                **context,
                "role": role,
                "method": method_by_role[(*tuple(cell), role)],
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "sample_count": len(values),
                "seed_count_min": min(seed_counts),
                "seed_count_max": max(seed_counts),
                "mean": float(np.mean(values)),
                "std": _sample_std(values),
            }
        )

    paired_rows = []
    cells = sorted(context_by_cell)
    for cell in cells:
        roles = sorted(
            {
                key[len(cell)]
                for key in by_role_sample
                if tuple(key[: len(cell)]) == cell
            }
        )
        if "C" not in roles:
            continue
        for right_role in [role for role in roles if role != "C"]:
            for metric in sorted(primary_metrics):
                left = {
                    key[-2]: values
                    for key, values in by_role_sample.items()
                    if tuple(key[: len(cell)]) == cell
                    and key[len(cell)] == "C"
                    and key[-1] == metric
                }
                right = {
                    key[-2]: values
                    for key, values in by_role_sample.items()
                    if tuple(key[: len(cell)]) == cell
                    and key[len(cell)] == right_role
                    and key[-1] == metric
                }
                improvements: Dict[str, list[float]] = {}
                seed_counts = []
                multiplier = (
                    1.0
                    if METRIC_DIRECTIONS[metric] == "higher_is_better"
                    else -1.0
                )
                for sample_id in sorted(set(left).intersection(right)):
                    common_seeds = sorted(
                        set(left[sample_id]).intersection(right[sample_id])
                    )
                    if not common_seeds:
                        continue
                    seed_counts.append(len(common_seeds))
                    improvements[sample_id] = [
                        float(
                            np.mean(
                                [
                                    multiplier
                                    * (
                                        left[sample_id][seed]
                                        - right[sample_id][seed]
                                    )
                                    for seed in common_seeds
                                ]
                            )
                        )
                    ]
                if not improvements:
                    continue
                summary = paired_cluster_summary(
                    improvements,
                    seed=int(analysis_seed),
                    n_bootstrap=int(bootstrap),
                )
                paired_rows.append(
                    {
                        **context_by_cell[cell],
                        "comparison": f"C_vs_{right_role}",
                        "left_role": "C",
                        "right_role": right_role,
                        "metric": metric,
                        "direction": "positive_favors_C",
                        "common_sample_count": summary["cluster_count"],
                        "seed_count_min": min(seed_counts),
                        "seed_count_max": max(seed_counts),
                        "improvement_mean": summary["mean"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "paired_effect_dz": summary["paired_effect_dz"],
                        "wilcoxon_p_value": summary["wilcoxon_p_value"],
                    }
                )
    return table_rows, paired_rows


def _cross_seed_e2_table(
    rows: Sequence[Mapping[str, str]],
    *,
    analysis_seed: int = 260730,
    bootstrap: int = 2000,
) -> list[Dict[str, Any]]:
    """Aggregate held-out reconstruction within sample before E2 inference."""

    cell_size = 7
    values_by_key: Dict[tuple[str, ...], Dict[str, float]] = {}
    contexts: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        cell = (
            str(row["paper_version"]),
            str(row["dataset"]),
            str(row["split"]),
            str(row["dataset_id"]),
            str(row["model"]),
            str(row["model_id"]),
            str(row["budget"]),
        )
        contexts[cell] = {
            "paper_version": row["paper_version"],
            "dataset": row["dataset"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "budget": int(float(row["budget"])),
            "attribution_seed": "all",
            "audit_id": "cross-seed",
        }
        key = (
            *cell,
            str(row["role"]),
            str(row["distribution"]),
            str(row["metric"]),
            str(row["sample_id"]),
        )
        seed = str(row["attribution_seed"])
        if seed in values_by_key.setdefault(key, {}):
            raise ValueError(f"Duplicate E2 sample/seed row for {key} seed={seed}.")
        values_by_key[key][seed] = _float_field(row, "value")

    grouped: Dict[tuple[str, ...], list[float]] = {}
    for key, seed_values in values_by_key.items():
        grouped.setdefault(key[:-1], []).append(
            float(np.mean(list(seed_values.values())))
        )
    output = []
    for key, values in sorted(grouped.items()):
        cell = tuple(key[:cell_size])
        role, distribution, metric = key[cell_size:]
        output.append(
            {
                **contexts[cell],
                "analysis": "cross_seed_run_reconstruction",
                "role": role,
                "comparison": "",
                "distribution": distribution,
                "metric": metric,
                "direction": (
                    "higher_is_better" if metric == "r2" else "lower_is_better"
                ),
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": _sample_std(values),
                "median": float(np.median(values)),
                "ci_low": None,
                "ci_high": None,
                "paired_effect_dz": None,
                "wilcoxon_p_value": None,
            }
        )

    for cell in sorted(contexts):
        roles = sorted(
            {
                key[cell_size]
                for key in values_by_key
                if tuple(key[:cell_size]) == cell
            }
        )
        if "C" not in roles:
            continue
        axes = sorted(
            {
                (key[cell_size + 1], key[cell_size + 2])
                for key in values_by_key
                if tuple(key[:cell_size]) == cell
                and key[cell_size] == "C"
            }
        )
        for role in [value for value in roles if value != "C"]:
            for distribution, metric in axes:
                left = {
                    key[-1]: seed_values
                    for key, seed_values in values_by_key.items()
                    if tuple(key[:cell_size]) == cell
                    and key[cell_size : cell_size + 3]
                    == ("C", distribution, metric)
                }
                right = {
                    key[-1]: seed_values
                    for key, seed_values in values_by_key.items()
                    if tuple(key[:cell_size]) == cell
                    and key[cell_size : cell_size + 3]
                    == (role, distribution, metric)
                }
                multiplier = 1.0 if metric == "r2" else -1.0
                differences = {}
                for sample_id in sorted(set(left).intersection(right)):
                    seeds = sorted(set(left[sample_id]).intersection(right[sample_id]))
                    if seeds:
                        differences[sample_id] = [
                            float(
                                np.mean(
                                    [
                                        multiplier
                                        * (
                                            left[sample_id][seed]
                                            - right[sample_id][seed]
                                        )
                                        for seed in seeds
                                    ]
                                )
                            )
                        ]
                if not differences:
                    continue
                summary = paired_cluster_summary(
                    differences,
                    seed=int(analysis_seed),
                    n_bootstrap=int(bootstrap),
                )
                output.append(
                    {
                        **contexts[cell],
                        "analysis": "cross_seed_paired_comparison",
                        "role": "C",
                        "comparison": f"C_vs_{role}",
                        "distribution": distribution,
                        "metric": metric,
                        "direction": "positive_favors_C",
                        **_stat_columns(summary),
                    }
                )
    return output


def _cross_seed_e3_table(
    rows: Sequence[Mapping[str, str]],
    *,
    analysis_seed: int = 260730,
    bootstrap: int = 2000,
) -> list[Dict[str, Any]]:
    """Aggregate exact-pair recovery within sample before E3 inference."""

    cell_size = 7
    values_by_key: Dict[tuple[str, ...], Dict[str, float]] = {}
    contexts: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        cell = (
            str(row["paper_version"]),
            str(row["dataset"]),
            str(row["split"]),
            str(row["dataset_id"]),
            str(row["model"]),
            str(row["model_id"]),
            str(row["budget"]),
        )
        contexts[cell] = {
            "paper_version": row["paper_version"],
            "dataset": row["dataset"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "budget": int(float(row["budget"])),
            "attribution_seed": "all",
            "audit_id": "cross-seed",
            "oracle_scope": "pair_oracle",
            "length_bin": "all",
        }
        key = (
            *cell,
            str(row["method"]),
            str(row["metric"]),
            str(row["sample_id"]),
        )
        seed = str(row["attribution_seed"])
        if seed in values_by_key.setdefault(key, {}):
            raise ValueError(f"Duplicate E3 sample/seed row for {key} seed={seed}.")
        values_by_key[key][seed] = _float_field(row, "value")

    output = []
    grouped: Dict[tuple[str, ...], list[float]] = {}
    for key, seed_values in values_by_key.items():
        grouped.setdefault(key[:-1], []).append(
            float(np.mean(list(seed_values.values())))
        )
    for key, values in sorted(grouped.items()):
        cell = tuple(key[:cell_size])
        method, metric = key[cell_size:]
        output.append(
            {
                **contexts[cell],
                "analysis": "cross_seed_ranking",
                "method": method,
                "metric": metric,
                "direction": "higher_is_better",
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": _sample_std(values),
                "median": float(np.median(values)),
                "ci_low": None,
                "ci_high": None,
                "paired_effect_dz": None,
                "wilcoxon_p_value": None,
            }
        )

    for cell in sorted(contexts):
        metrics = sorted(
            {
                key[cell_size + 1]
                for key in values_by_key
                if tuple(key[:cell_size]) == cell and key[cell_size] == "mobius"
            }
        )
        for metric in metrics:
            left = {
                key[-1]: seed_values
                for key, seed_values in values_by_key.items()
                if tuple(key[:cell_size]) == cell
                and key[cell_size : cell_size + 2] == ("mobius", metric)
            }
            right = {
                key[-1]: seed_values
                for key, seed_values in values_by_key.items()
                if tuple(key[:cell_size]) == cell
                and key[cell_size : cell_size + 2] == ("proxyspex", metric)
            }
            differences = {}
            for sample_id in sorted(set(left).intersection(right)):
                seeds = sorted(set(left[sample_id]).intersection(right[sample_id]))
                if seeds:
                    differences[sample_id] = [
                        float(
                            np.mean(
                                [
                                    left[sample_id][seed] - right[sample_id][seed]
                                    for seed in seeds
                                ]
                            )
                        )
                    ]
            if not differences:
                continue
            summary = paired_cluster_summary(
                differences,
                seed=int(analysis_seed),
                n_bootstrap=int(bootstrap),
            )
            output.append(
                {
                    **contexts[cell],
                    "analysis": "cross_seed_paired_ranking",
                    "method": "mobius_vs_proxyspex",
                    "metric": metric,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(summary),
                }
            )
    return output


def _cross_seed_fixed_k_table(
    rows: Sequence[Mapping[str, str]],
    *,
    analysis_seed: int = 260730,
    bootstrap: int = 2000,
) -> list[Dict[str, Any]]:
    """Aggregate fixed-k basis diagnostics within sample across seeds."""

    cell_size = 7
    values_by_key: Dict[tuple[str, ...], Dict[str, float]] = {}
    contexts: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        cell = (
            str(row["paper_version"]),
            str(row["dataset"]),
            str(row["split"]),
            str(row["dataset_id"]),
            str(row["model"]),
            str(row["model_id"]),
            str(row["budget"]),
        )
        contexts[cell] = {
            "paper_version": row["paper_version"],
            "dataset": row["dataset"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "model": row["model"],
            "model_id": row["model_id"],
            "budget": int(float(row["budget"])),
            "attribution_seed": "all",
            "audit_id": "cross-seed",
        }
        key = (
            *cell,
            str(row["distribution"]),
            str(row["k"]),
            str(row["basis"]),
            str(row["metric"]),
            str(row["sample_id"]),
        )
        seed = str(row["attribution_seed"])
        if seed in values_by_key.setdefault(key, {}):
            raise ValueError(f"Duplicate fixed-k sample/seed row for {key} seed={seed}.")
        values_by_key[key][seed] = _float_field(row, "value")

    grouped: Dict[tuple[str, ...], list[float]] = {}
    for key, seed_values in values_by_key.items():
        grouped.setdefault(key[:-1], []).append(
            float(np.mean(list(seed_values.values())))
        )
    output = []
    for key, values in sorted(grouped.items()):
        cell = tuple(key[:cell_size])
        distribution, k, basis, metric = key[cell_size:]
        output.append(
            {
                **contexts[cell],
                "analysis": "cross_seed_fixed_k_omp",
                "basis": basis,
                "distribution": distribution,
                "k": k,
                "metric": metric,
                "direction": (
                    "higher_is_better"
                    if metric in {"r2", "ols_rank"}
                    else "lower_is_better"
                ),
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": _sample_std(values),
                "median": float(np.median(values)),
                "ci_low": None,
                "ci_high": None,
                "paired_effect_dz": None,
                "wilcoxon_p_value": None,
            }
        )

    for cell in sorted(contexts):
        axes = sorted(
            {
                (key[cell_size], key[cell_size + 1], key[cell_size + 3])
                for key in values_by_key
                if tuple(key[:cell_size]) == cell
                and key[cell_size + 2] == "mobius"
                and key[cell_size + 3] in {"r2", "nrmse_range"}
            }
        )
        for distribution, k, metric in axes:
            mobius = {
                key[-1]: seed_values
                for key, seed_values in values_by_key.items()
                if tuple(key[:cell_size]) == cell
                and key[cell_size : cell_size + 4]
                == (distribution, k, "mobius", metric)
            }
            fourier = {
                key[-1]: seed_values
                for key, seed_values in values_by_key.items()
                if tuple(key[:cell_size]) == cell
                and key[cell_size : cell_size + 4]
                == (distribution, k, "fourier", metric)
            }
            multiplier = 1.0 if metric == "r2" else -1.0
            differences = {}
            for sample_id in sorted(set(mobius).intersection(fourier)):
                seeds = sorted(set(mobius[sample_id]).intersection(fourier[sample_id]))
                if seeds:
                    differences[sample_id] = [
                        float(
                            np.mean(
                                [
                                    multiplier
                                    * (
                                        mobius[sample_id][seed]
                                        - fourier[sample_id][seed]
                                    )
                                    for seed in seeds
                                ]
                            )
                        )
                    ]
            if not differences:
                continue
            summary = paired_cluster_summary(
                differences,
                seed=int(analysis_seed),
                n_bootstrap=int(bootstrap),
            )
            output.append(
                {
                    **contexts[cell],
                    "analysis": "cross_seed_fixed_k_basis_comparison",
                    "basis": "mobius_vs_fourier",
                    "distribution": distribution,
                    "k": k,
                    "metric": metric,
                    "direction": "positive_favors_mobius",
                    **_stat_columns(summary),
                }
            )
    return output


def _write_filtered_csv(
    source: Path,
    destination: Path,
    predicate,
) -> None:
    """Write a schema-preserving filtered copy of one aggregate CSV."""

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader if predicate(row)]
    _write_csv(destination, rows, fields)


def summarize_paper_version(
    paper_version: str,
    *,
    output_root: str | Path = "results/paper",
) -> Path:
    """Aggregate every completed cell into versioned paper tables."""

    version = _validate_path_id(paper_version, name="paper_version")
    root = Path(output_root) / version
    manifests = _cell_manifests(root)
    if not manifests:
        raise ValueError(f"No paper cells found under {root}.")
    aggregate = ensure_dir(root / "aggregate")
    cells = []
    for manifest_path in manifests:
        manifest = load_json_object(manifest_path)
        cells.append(
            {
                "dataset_id": manifest["dataset_id"],
                "model_id": manifest["model_id"],
                "budget": manifest.get("budget"),
                "attribution_seed": manifest["attribution_seed"],
                "path": str(manifest_path.parent.resolve()),
                "input_fingerprint": manifest["input_fingerprint"],
            }
        )
    for relative_path in CELL_CSV_FILES:
        rows, fields = _concatenate_cell_csvs(manifests, relative_path)
        destination_name = relative_path.replace("/", "-")
        _write_csv(aggregate / destination_name, rows, fields)
    attribution_rows, _ = _concatenate_cell_csvs(
        manifests,
        "attribution-samples.csv",
    )
    e1_table, e1_paired = _cross_seed_e1_tables(attribution_rows)
    e1_roles = {"A", "C", "PROXYSPEX", "OCCLUSION", "LIME"}
    _write_csv(
        aggregate / "table-e1-faithfulness.csv",
        [row for row in e1_table if row["role"] in e1_roles],
        E1_SUMMARY_FIELDS,
    )
    _write_csv(
        aggregate / "figure-e1-paired-effects.csv",
        [row for row in e1_paired if row["right_role"] in e1_roles - {"C"}],
        E1_PAIRED_SEED_FIELDS,
    )
    surrogate_sample_rows, _ = _concatenate_cell_csvs(
        manifests,
        "surrogate-heldout-samples.csv",
    )
    _write_csv(
        aggregate / "figure-e2a-budget-recovery.csv",
        [
            row
            for row in _cross_seed_e2_table(surrogate_sample_rows)
            if row["metric"] in {"r2", "nrmse_range"}
            and (
                row["role"] in {"A", "C", "PROXYSPEX"}
                and (
                    not row["comparison"]
                    or row["comparison"] in {"C_vs_A", "C_vs_PROXYSPEX"}
                )
            )
        ],
        SURROGATE_FIELDS,
    )
    representation_sample_rows, _ = _concatenate_cell_csvs(
        manifests,
        "representation-fixed-k-samples.csv",
    )
    _write_csv(
        aggregate / "figure-e2b-fixed-support.csv",
        _cross_seed_fixed_k_table(representation_sample_rows),
        REPRESENTATION_FIELDS,
    )
    exact_pair_sample_rows, _ = _concatenate_cell_csvs(
        manifests,
        "exact-pair-samples.csv",
    )
    _write_csv(
        aggregate / "table-e3-exact-pairs.csv",
        [
            row
            for row in _cross_seed_e3_table(exact_pair_sample_rows)
            if row["method"] in {
                "mobius",
                "proxyspex",
                "mobius_vs_proxyspex",
            }
        ],
        EXACT_FIELDS,
    )
    _write_csv(
        aggregate / "table-e4-projection.csv",
        [
            row
            for row in e1_paired
            if row["right_role"] in {"B", "ABSOLUTE"}
        ],
        E1_PAIRED_SEED_FIELDS,
    )
    costs, cost_fields = _concatenate_cell_csvs(manifests, "costs.csv")
    _write_csv(
        aggregate / "attribution-costs.csv",
        costs,
        cost_fields,
    )
    summary = {
        "schema_version": "1.0",
        "kind": "paper_result_aggregate",
        "paper_version": version,
        "cell_count": len(cells),
        "cells": cells,
        "files": [
            relative_path.replace("/", "-")
            for relative_path in CELL_CSV_FILES
        ],
    }
    canonical_files = [
        "table-e1-faithfulness.csv",
        "figure-e1-paired-effects.csv",
        "figure-e2a-budget-recovery.csv",
        "figure-e2b-fixed-support.csv",
        "table-e3-exact-pairs.csv",
        "table-e4-projection.csv",
        "attribution-costs.csv",
        "audit-costs.csv",
    ]
    if (aggregate / "classifier-diagnostics.csv").is_file():
        canonical_files.insert(0, "classifier-diagnostics.csv")
    summary["canonical_files"] = canonical_files
    atomic_write_json(aggregate / "index.json", summary)
    atomic_write_json(aggregate / "manifest.json", summary)
    lines = [
        f"# Paper Results: {version}",
        "",
        f"- Cell 数量：{len(cells)}",
        "- 内容：论文表格、配对统计及绘图数据",
        "",
        "## Cells",
        "",
    ]
    lines.extend(
        (
            f"- `{row['dataset_id']}` / `{row['model_id']}` / "
            f"`seed-{row['attribution_seed']}`"
        )
        for row in cells
    )
    lines.append("")
    _write_text(aggregate / "README.md", "\n".join(lines))
    return aggregate
