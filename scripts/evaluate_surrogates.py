"""Evaluate serialized surrogates on one shared held-out audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.core.artifacts import load_surrogate_artifact, predict_surrogate
from mobius.core.results import canonical_digest
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.evaluation.surrogate import (
    aggregate_reconstruction,
    reconstruction_metrics,
)
from mobius.values.classification import attribution_values


def build_parser() -> argparse.ArgumentParser:
    """Build the serialized-surrogate evaluator CLI."""

    parser = argparse.ArgumentParser(
        description="Evaluate every run in a shared surrogate held-out audit."
    )
    parser.add_argument("--audit-dir", required=True)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _load_audit_sample(
    path: Path,
    distributions: Sequence[str],
) -> Dict[str, Any]:
    """Load shared held-out masks, scores, and metadata without pickle."""

    with np.load(path, allow_pickle=False) as archive:
        payload: Dict[str, Any] = {
            "metadata": json.loads(str(archive["metadata_json"].item())),
        }
        for distribution in distributions:
            payload[f"{distribution}_keep_masks"] = np.asarray(
                archive[f"{distribution}_keep_masks"],
                dtype=bool,
            )
            payload[f"{distribution}_label_scores"] = np.asarray(
                archive[f"{distribution}_label_scores"],
                dtype=np.float64,
            )
        return payload


def _run_key(run_dir: Path, run_payload: Mapping[str, Any]) -> str:
    """Build a collision-resistant filename key for one evaluated run."""

    method = str(dict(run_payload["scientific_config"]).get("method", "method"))
    suffix = canonical_digest(str(run_dir.resolve()))[:8]
    return f"{method}-{run_payload.get('run_id', run_dir.name)}-{suffix}"


def evaluate_audit(audit_dir: str | Path) -> Dict[str, Any]:
    """Evaluate all manifest runs and write per-run pointers and aggregates."""

    root = Path(audit_dir)
    manifest = _load_json(root / "manifest.json")
    audit_id = str(manifest["audit_id"])
    distributions = tuple(
        str(value) for value in dict(manifest["settings"])["distributions"]
    )
    metrics_dir = ensure_dir(root / "metrics")
    evaluation_rows = []
    for run_entry in manifest["runs"]:
        run_dir = Path(run_entry["path"])
        run_payload = _load_json(run_dir / "run.json")
        config = dict(run_payload["scientific_config"])
        rows: list[Dict[str, Any]] = []
        for sample_entry in manifest["samples"]:
            if sample_entry["status"] != "ok":
                continue
            sample_id = str(sample_entry["sample_id"])
            sample = _load_json(run_dir / "samples" / f"{sample_id}.json")
            surrogate = load_surrogate_artifact(
                run_dir / "surrogates" / f"{sample_id}.json"
            )
            heldout = _load_audit_sample(
                root / sample_entry["artifact"],
                distributions,
            )
            distribution_metrics: Dict[str, Any] = {}
            for distribution in distributions:
                keep_masks = heldout[f"{distribution}_keep_masks"]
                label_scores = heldout[f"{distribution}_label_scores"]
                truth = attribution_values(
                    label_scores,
                    target_class=int(sample["target_label"]),
                    value_function=str(config["value_function"]),
                )
                prediction = predict_surrogate(surrogate, keep_masks)
                distribution_metrics[distribution] = reconstruction_metrics(
                    truth,
                    prediction,
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "target_label": int(sample["target_label"]),
                    "surrogate_digest": surrogate["digest"],
                    "distributions": distribution_metrics,
                }
            )
        aggregate = {
            distribution: aggregate_reconstruction(
                [row["distributions"][distribution] for row in rows]
            )
            for distribution in distributions
        }
        report = {
            "schema_version": "1.1",
            "audit_id": audit_id,
            "run_id": run_payload.get("run_id"),
            "run_path": str(run_dir.resolve()),
            "evaluated_count": len(rows),
            "aggregate": aggregate,
            "rows": rows,
            "audit_query_cost": manifest.get("query_cost", {}),
            "attribution_cost_includes_audit": False,
        }
        key = _run_key(run_dir, run_payload)
        report_path = metrics_dir / f"{key}.json"
        atomic_write_json(report_path, report)
        pointer = {
            "schema_version": "1.1",
            "audit_id": audit_id,
            "audit_dir": str(root.resolve()),
            "report": str(report_path.resolve()),
            "evaluated_count": len(rows),
            "aggregate": aggregate,
        }
        analyses_dir = ensure_dir(run_dir / "analyses")
        atomic_write_json(
            analyses_dir / f"heldout-{audit_id}.json",
            pointer,
        )
        evaluation_rows.append(
            {
                "run_id": run_payload.get("run_id"),
                "run_path": str(run_dir.resolve()),
                "report": str(report_path.relative_to(root)),
            }
        )
    index = {
        "schema_version": "1.1",
        "audit_id": audit_id,
        "evaluations": evaluation_rows,
    }
    atomic_write_json(root / "evaluation-index.json", index)
    return index


def main(argv: Sequence[str] | None = None) -> None:
    """Run held-out surrogate evaluation from CLI arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    index = evaluate_audit(args.audit_dir)
    print(
        f"[surrogates-evaluated] audit={index['audit_id']} "
        f"runs={len(index['evaluations'])}"
    )


if __name__ == "__main__":
    main()
