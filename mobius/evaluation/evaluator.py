"""Evaluate schema-v2 explanations against an explicit target convention."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from mobius.core.runtime import atomic_write_json, counter_delta
from mobius.core.schema import DatasetBundle, SCHEMA_VERSION, TextChunk
from mobius.models.base import RawTextScorer
from mobius.text.chunks import build_eval_units, compose_text, project_ranking
from mobius.text.coalitions import EMPTY_TEXT
from mobius.values.classification import probabilities

from .metrics import (
    PRIMARY_Q,
    aggregate,
    aml_aopc,
    aopc_from_drops,
    aupc_from_probabilities,
    per_q_metrics,
    top_count,
)


def _nonempty(text: str) -> str:
    """Replace an empty perturbation with the shared neutral placeholder."""

    return text if text else EMPTY_TEXT


def _load_sample(path: Path) -> Dict[str, Any]:
    """Load and minimally validate one schema-v2 sample."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported sample schema in {path}")
    return payload


def _chunks(payload: Dict[str, Any]) -> List[TextChunk]:
    """Reconstruct typed chunks from a sample payload."""

    return [
        TextChunk(
            chunk_id=int(row["chunk_id"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            text=str(row["text"]),
            token_start=row.get("token_start"),
            token_end=row.get("token_end"),
        )
        for row in payload["chunks"]
    ]


def _sample_perturbations(
    text: str,
    eval_units: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
) -> tuple[List[str], Dict[str, Any]]:
    """Build all unique faithfulness texts and indices into the score matrix."""

    all_ids = [int(unit.chunk_id) for unit in eval_units]
    texts: List[str] = []
    index_by_text: Dict[str, int] = {}

    def add(value: str) -> int:
        """Deduplicate a perturbation text and return its matrix row."""

        normalized = _nonempty(value)
        if normalized not in index_by_text:
            index_by_text[normalized] = len(texts)
            texts.append(normalized)
        return index_by_text[normalized]

    full_index = add(text)
    per_q: Dict[int, Dict[str, int | List[int]]] = {}
    for q in sorted(set(int(value) for value in q_values)):
        count = top_count(len(eval_units), q)
        selected = [int(value) for value in ranking[:count]]
        selected_set = set(selected)
        if count == 0:
            remove_index = full_index
            keep_index = full_index
        else:
            remove_index = add(
                compose_text(
                    eval_units,
                    [unit_id for unit_id in all_ids if unit_id not in selected_set],
                )
            )
            keep_index = add(compose_text(eval_units, selected))
        per_q[q] = {
            "top_ids": selected,
            "top_count": count,
            "remove_index": remove_index,
            "keep_index": keep_index,
        }
    deletion_indices = []
    for step in range(len(eval_units) + 1):
        deleted = set(int(value) for value in ranking[:step])
        deletion_indices.append(
            add(
                compose_text(
                    eval_units,
                    [unit_id for unit_id in all_ids if unit_id not in deleted],
                )
            )
        )
    return texts, {
        "full_index": full_index,
        "per_q": per_q,
        "deletion_indices": deletion_indices,
    }


def _sum_attribution_costs(sample_payloads: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate common scalar attribution-cost fields across samples."""

    keys = {
        "attribution_budget_used",
        "logical_unique_queries",
        "physical_values_scored",
        "interaction_verification_queries",
        "model_forward_calls",
        "batch_calls",
        "batch_rows",
        "elapsed_seconds",
    }
    return {
        key: float(
            sum(
                float(payload.get("attribution_cost", {}).get(key, 0))
                for payload in sample_payloads
            )
        )
        for key in sorted(keys)
    }


def evaluate_run(
    run_dir: str | Path,
    bundle: DatasetBundle,
    scorer: RawTextScorer,
    *,
    target: str = "predicted",
    eval_granularity: str = "word",
    q_values: Sequence[int] = (5, 10, 20, 50),
) -> Dict[str, Any]:
    """Evaluate all completed samples and write target-explicit metrics.json."""

    mode = str(target).strip().lower()
    if mode not in {"predicted", "gold"}:
        raise ValueError("Evaluation target must be predicted or gold.")
    root = Path(run_dir)
    sample_by_id = {str(sample.sample_id): sample for sample in bundle.samples}
    metric_values: Dict[str, List[float]] = {
        "comprehensiveness": [],
        "sufficiency": [],
        "aopc_comprehensiveness": [],
        "aopc_sufficiency": [],
        "aopc": [],
        "aupc": [],
    }
    per_q_values: Dict[int, Dict[str, List[float]]] = {
        int(q): {"comprehensiveness": [], "sufficiency": []}
        for q in sorted(set(int(value) for value in q_values))
    }
    sample_payloads: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    correct = 0
    counters_before = scorer.snapshot_counters()
    started = time.perf_counter()
    for path in sorted((root / "samples").glob("*.json")):
        try:
            payload = _load_sample(path)
            sample = sample_by_id[str(payload["sample_id"])]
            chunks = _chunks(payload)
            eval_result = build_eval_units(
                sample.text,
                eval_granularity,
                getattr(scorer, "tokenizer", None),
            )
            unit_ranking = project_ranking(
                eval_result.chunks,
                chunks,
                payload["ranking"],
            )
            perturbation_texts, plan = _sample_perturbations(
                sample.text,
                eval_result.chunks,
                unit_ranking,
                q_values,
            )
            score_matrix = scorer.score_texts(perturbation_texts)
            probability_matrix = probabilities(score_matrix)
            full_row = int(plan["full_index"])
            predicted_label = int(np.argmax(score_matrix[full_row]))
            target_label = predicted_label if mode == "predicted" else int(sample.label)
            full_probability = float(probability_matrix[full_row, target_label])
            remove_probabilities = {
                int(q): float(
                    probability_matrix[
                        int(plan["per_q"][int(q)]["remove_index"]),
                        target_label,
                    ]
                )
                for q in q_values
            }
            keep_probabilities = {
                int(q): float(
                    probability_matrix[
                        int(plan["per_q"][int(q)]["keep_index"]),
                        target_label,
                    ]
                )
                for q in q_values
            }
            sample_per_q = per_q_metrics(
                full_probability,
                remove_probabilities,
                keep_probabilities,
                q_values,
            )
            deletion_probabilities = [
                float(probability_matrix[int(index), target_label])
                for index in plan["deletion_indices"]
            ]
            deletion_drops = [
                full_probability - value for value in deletion_probabilities
            ]
            primary = PRIMARY_Q if PRIMARY_Q in sample_per_q else int(q_values[-1])
            sample_metrics = {
                "comprehensiveness": sample_per_q[primary]["comprehensiveness"],
                "sufficiency": sample_per_q[primary]["sufficiency"],
                "aopc_comprehensiveness": aml_aopc(
                    sample_per_q,
                    q_values,
                    "comprehensiveness",
                ),
                "aopc_sufficiency": aml_aopc(
                    sample_per_q,
                    q_values,
                    "sufficiency",
                ),
                "aopc": aopc_from_drops(deletion_drops),
                "aupc": aupc_from_probabilities(deletion_probabilities),
            }
            for name, value in sample_metrics.items():
                metric_values[name].append(float(value))
            for q, values in sample_per_q.items():
                for name, value in values.items():
                    per_q_values[int(q)][name].append(float(value))
            correct += int(predicted_label == int(sample.label))
            sample_payloads.append(payload)
            curves.append(
                {
                    "sample_id": sample.sample_id,
                    "target": mode,
                    "target_label": target_label,
                    "full_probability": full_probability,
                    "unit_count": len(eval_result.chunks),
                    "unit_fallback_used": eval_result.fallback_used,
                    "per_q": {str(key): value for key, value in sample_per_q.items()},
                    "deletion_probabilities": deletion_probabilities,
                    "deletion_drops": deletion_drops,
                    "metrics": sample_metrics,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "sample_id": path.stem,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
    elapsed = time.perf_counter() - started
    counters_after = scorer.snapshot_counters()
    curves_path = root / f"curves-{mode}.jsonl"
    with curves_path.open("w", encoding="utf-8") as handle:
        for row in curves:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": mode,
        "sample_count": len(list((root / "samples").glob("*.json"))),
        "evaluated_count": len(curves),
        "failed_count": len(failures),
        "failures": failures,
        "accuracy": float(correct / len(curves)) if curves else None,
        "faithfulness": {
            name: aggregate(values) for name, values in metric_values.items()
        },
        "per_q": {
            str(q): {
                name: aggregate(values)
                for name, values in metric_map.items()
            }
            for q, metric_map in per_q_values.items()
        },
        "protocol": {
            "prompt": scorer.scoring_contract().get("prompt"),
            "eval_granularity": str(eval_granularity),
            "q_values": [int(value) for value in q_values],
            "primary_q": PRIMARY_Q,
            "percentage_rounding": "floor",
            "aopc_grid_denominator": "len(q_values)+1",
            "aupc_definition": (
                "normalized trapezoidal area under the complete deletion "
                "target-probability curve"
            ),
            "aupc_direction": "lower_is_better",
            "aupc_extra_model_calls": 0,
        },
        "curve_artifact": curves_path.name,
        "evaluation_cost": {
            "elapsed_seconds": elapsed,
            "model_counter_delta": counter_delta(counters_before, counters_after),
        },
        "attribution_cost": _sum_attribution_costs(sample_payloads),
    }
    atomic_write_json(root / "metrics.json", report)
    return report
