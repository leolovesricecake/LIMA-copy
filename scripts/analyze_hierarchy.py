"""Analyze strict hierarchy and the faithfulness contribution of orphan pairs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.statistics import (
    aggregate_numeric,
    paired_cluster_summary,
)
from mobius.core.artifacts import (
    load_observation_artifact,
    load_surrogate_artifact,
)
from mobius.core.results import canonical_digest, default_audit_dir
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.core.schema import TextChunk
from mobius.evaluation.evaluator import _sample_perturbations
from mobius.evaluation.metrics import (
    PRIMARY_Q,
    aml_aopc,
    aopc_from_drops,
    aupc_from_probabilities,
    per_q_metrics,
)
from mobius.methods.sparse.estimator import sparse_model_from_surrogate
from mobius.methods.sparse.projector import project_nodes
from mobius.models.hf import build_scorer
from mobius.models.oracle import QueryLedger, ValueOracle, stable_digest
from mobius.text.chunks import build_eval_units, project_ranking
from mobius.values.classification import probabilities


def build_parser() -> argparse.ArgumentParser:
    """Build the hierarchy analysis CLI."""

    parser = argparse.ArgumentParser(
        description="Compare none/strict hierarchy and remove parent-count groups."
    )
    parser.add_argument("--none-run", required=True)
    parser.add_argument("--strict-run", required=True)
    parser.add_argument("--verification-dir")
    parser.add_argument("--heldout-audit")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
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
    """Load JSON objects from a JSONL artifact."""

    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _chunks(payload: Mapping[str, Any]) -> list[TextChunk]:
    """Reconstruct explanation chunks from a sample payload."""

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


def _term(players: Sequence[int]) -> int:
    """Encode players as one integer interaction term."""

    value = 0
    for player in players:
        value |= 1 << int(player)
    return value


def _support_terms(
    surrogate: Mapping[str, Any],
    support_name: str,
) -> set[int]:
    """Decode one structural support from a surrogate artifact."""

    rows = dict(surrogate.get("support", {})).get(support_name, [])
    return {_term(row.get("players", [])) for row in rows}


def _parent_group(
    players: Sequence[int],
    singleton_support: set[int],
) -> int:
    """Return how many singleton parents are present for one pair."""

    return sum((1 << int(player)) in singleton_support for player in players)


def _support_counts(terms: set[int]) -> Dict[str, int]:
    """Count total, singleton, and pair structural support sizes."""

    return {
        "total_support_size": len(terms),
        "singleton_support_size": sum(term.bit_count() == 1 for term in terms),
        "pair_support_size": sum(term.bit_count() == 2 for term in terms),
    }


def _ranking_from_surrogate(
    surrogate: Mapping[str, Any],
    chunks: Sequence[TextChunk],
) -> list[int]:
    """Project a serialized sparse model to one complete chunk ranking."""

    model = sparse_model_from_surrogate(surrogate)
    scores = project_nodes(model, "signed_equal_share")
    mapping = [int(value) for value in surrogate["player_to_chunk_id"]]
    active = sorted(
        (
            (chunk_id, float(scores[player]))
            for player, chunk_id in enumerate(mapping)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ranked = [chunk_id for chunk_id, _ in active]
    present = set(ranked)
    return ranked + [
        int(chunk.chunk_id)
        for chunk in chunks
        if int(chunk.chunk_id) not in present
    ]


def _without_parent_group(
    surrogate: Mapping[str, Any],
    group: int,
) -> Dict[str, Any]:
    """Remove fitted pair terms from one singleton-parent-count group."""

    output = copy.deepcopy(dict(surrogate))
    singleton_support = {
        term
        for term in _support_terms(surrogate, "selection")
        if term.bit_count() == 1
    }
    terms = []
    for row in dict(output["predictor"]).get("terms", []):
        players = [int(value) for value in row.get("players", [])]
        if len(players) == 2 and _parent_group(players, singleton_support) == int(group):
            continue
        terms.append(row)
    output["predictor"]["terms"] = terms
    output.pop("digest", None)
    return output


def _evaluate_ranking(
    *,
    sample: Mapping[str, Any],
    ranking: Sequence[int],
    config: Mapping[str, Any],
    scorer,
    oracle: ValueOracle,
    ledger: QueryLedger,
    variant: str,
) -> Dict[str, float]:
    """Evaluate one derived ranking using the standard faithfulness protocol."""

    chunks = _chunks(sample)
    eval_result = build_eval_units(
        str(sample["text"]),
        str(config.get("eval_granularity", "word")),
        getattr(scorer, "tokenizer", None),
    )
    unit_ranking = project_ranking(eval_result.chunks, chunks, ranking)
    q_values = [
        int(value) for value in config.get("eval_q_values", [1, 5, 10, 20, 50])
    ]
    texts, plan = _sample_perturbations(
        str(sample["text"]),
        eval_result.chunks,
        unit_ranking,
        q_values,
    )
    score_matrix = oracle.score_texts(
        texts,
        logical_keys=[
            stable_digest(
                {
                    "sample_id": sample["sample_id"],
                    "variant": variant,
                    "text": text,
                    "category": "hierarchy_removal_evaluation",
                }
            )
            for text in texts
        ],
        ledger=ledger,
        category="hierarchy_removal_evaluation",
    )
    probability_matrix = probabilities(score_matrix)
    target = int(sample["target_label"])
    full_probability = float(
        probability_matrix[int(plan["full_index"]), target]
    )
    remove = {
        q: float(
            probability_matrix[
                int(plan["per_q"][q]["remove_index"]),
                target,
            ]
        )
        for q in q_values
    }
    keep = {
        q: float(
            probability_matrix[
                int(plan["per_q"][q]["keep_index"]),
                target,
            ]
        )
        for q in q_values
    }
    per_q = per_q_metrics(full_probability, remove, keep, q_values)
    deletion_probabilities = [
        float(probability_matrix[int(index), target])
        for index in plan["deletion_indices"]
    ]
    primary = PRIMARY_Q if PRIMARY_Q in per_q else q_values[-1]
    return {
        "comprehensiveness": float(per_q[primary]["comprehensiveness"]),
        "sufficiency": float(per_q[primary]["sufficiency"]),
        "aopc_comprehensiveness": float(
            aml_aopc(per_q, q_values, "comprehensiveness")
        ),
        "aopc_sufficiency": float(
            aml_aopc(per_q, q_values, "sufficiency")
        ),
        "aopc": float(
            aopc_from_drops(
                [
                    full_probability - value
                    for value in deletion_probabilities
                ]
            )
        ),
        "aupc": float(aupc_from_probabilities(deletion_probabilities)),
    }


def _curve_metrics(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load standard per-sample faithfulness metrics for one run."""

    run_payload = _load_json(run_dir / "run.json")
    target = str(
        dict(run_payload["scientific_config"]).get("target_mode", "predicted")
    )
    return {
        str(row["sample_id"]): {
            str(name): float(value)
            for name, value in dict(row["metrics"]).items()
        }
        for row in _load_jsonl(run_dir / f"curves-{target}.jsonl")
    }


def _heldout_rows(
    audit_dir: str | Path | None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load held-out rows indexed by absolute run path and sample ID."""

    if audit_dir is None:
        return {}
    root = Path(audit_dir)
    index = _load_json(root / "evaluation-index.json")
    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in index["evaluations"]:
        report = _load_json(root / entry["report"])
        output[str(Path(entry["run_path"]).resolve())] = {
            str(row["sample_id"]): dict(row["distributions"])
            for row in report["rows"]
        }
    return output


def analyze_hierarchy(
    none_run: str | Path,
    strict_run: str | Path,
    *,
    verification_dir: str | Path | None,
    heldout_audit: str | Path | None,
    output_dir: str | Path | None,
    device: str | None,
    seed: int,
    bootstrap: int,
) -> Path:
    """Run structural, held-out, and removal analyses for a hierarchy pair."""

    none_root = Path(none_run).resolve()
    strict_root = Path(strict_run).resolve()
    none_meta = _load_json(none_root / "run.json")
    strict_meta = _load_json(strict_root / "run.json")
    none_config = dict(none_meta["scientific_config"])
    strict_config = dict(strict_meta["scientific_config"])
    if str(none_config.get("hierarchy")) != "none":
        raise ValueError("--none-run must use hierarchy=none.")
    if str(strict_config.get("hierarchy")) != "strict":
        raise ValueError("--strict-run must use hierarchy=strict.")
    comparable_left = {
        key: value for key, value in none_config.items() if key != "hierarchy"
    }
    comparable_right = {
        key: value for key, value in strict_config.items() if key != "hierarchy"
    }
    if comparable_left != comparable_right:
        raise ValueError("Hierarchy runs differ on scientific axes beyond hierarchy.")
    audit_id = canonical_digest(
        {
            "none": none_meta.get("config_fingerprint"),
            "strict": strict_meta.get("config_fingerprint"),
            "seed": int(seed),
        }
    )[:12]
    destination = (
        Path(output_dir)
        if output_dir is not None
        else default_audit_dir(
            dict(none_config["dataset"]),
            "hierarchy",
            audit_id,
        )
    )
    ensure_dir(destination)
    model_config = dict(none_config["model"])
    if device is not None:
        model_config["device"] = str(device)
    verbalizers = [
        str(value)
        for value in dict(none_config["dataset"]).get("verbalizers", [])
    ]
    scorer = build_scorer(
        model_config,
        verbalizers,
        batch_size=int(none_config.get("batch_size", 16)),
    )
    oracle = ValueOracle(
        scorer,
        destination / ".cache" / "value_oracle.sqlite3",
        model_fingerprint=model_config,
        batch_size=int(none_config.get("batch_size", 16)),
    )
    ledger = QueryLedger(f"hierarchy/{audit_id}")
    none_curves = _curve_metrics(none_root)
    strict_curves = _curve_metrics(strict_root)
    heldout = _heldout_rows(heldout_audit)
    verification_rows = (
        _load_jsonl(Path(verification_dir) / "rows.jsonl")
        if verification_dir is not None
        else []
    )
    verification_by_key = {
        (str(row["sample_id"]), tuple(row["players"])): row
        for row in verification_rows
        if row.get("row_type") == "selected"
        and Path(str(row["run_path"])).resolve() == none_root
    }
    common_ids = sorted(
        {path.stem for path in (none_root / "samples").glob("*.json")}
        & {path.stem for path in (strict_root / "samples").glob("*.json")}
    )
    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    try:
        for sample_id in common_ids:
            try:
                none_sample = _load_json(
                    none_root / "samples" / f"{sample_id}.json"
                )
                strict_sample = _load_json(
                    strict_root / "samples" / f"{sample_id}.json"
                )
                none_surrogate = load_surrogate_artifact(
                    none_root / "surrogates" / f"{sample_id}.json"
                )
                strict_surrogate = load_surrogate_artifact(
                    strict_root / "surrogates" / f"{sample_id}.json"
                )
                none_observation = load_observation_artifact(
                    none_root / "observations" / f"{sample_id}.npz"
                )
                strict_observation = load_observation_artifact(
                    strict_root / "observations" / f"{sample_id}.npz"
                )
                if none_observation["digest"] != strict_observation["digest"]:
                    raise ValueError(
                        "none/strict training observations differ for this sample."
                    )
                selection = _support_terms(none_surrogate, "selection")
                singletons = {
                    term for term in selection if term.bit_count() == 1
                }
                group_terms: Dict[int, list[Dict[str, Any]]] = {
                    0: [],
                    1: [],
                    2: [],
                }
                for term_row in dict(none_surrogate["predictor"]).get(
                    "terms",
                    [],
                ):
                    players = [int(value) for value in term_row["players"]]
                    if len(players) != 2:
                        continue
                    group = _parent_group(players, singletons)
                    exact = verification_by_key.get(
                        (sample_id, tuple(players))
                    )
                    group_terms[group].append(
                        {
                            "players": players,
                            "coefficient": float(term_row["coefficient"]),
                            "exact_coefficient": (
                                float(exact["exact_coefficient"])
                                if exact is not None
                                else None
                            ),
                        }
                    )
                removal_metrics = {}
                for group in (0, 1, 2):
                    modified = _without_parent_group(none_surrogate, group)
                    ranking = _ranking_from_surrogate(
                        modified,
                        _chunks(none_sample),
                    )
                    removal_metrics[str(group)] = _evaluate_ranking(
                        sample=none_sample,
                        ranking=ranking,
                        config=none_config,
                        scorer=scorer,
                        oracle=oracle,
                        ledger=ledger,
                        variant=f"remove_parent_group_{group}",
                    )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "observation_digest_match": True,
                        "none_support": _support_counts(
                            _support_terms(none_surrogate, "refit")
                        ),
                        "strict_support": _support_counts(
                            _support_terms(strict_surrogate, "refit")
                        ),
                        "parent_groups": {
                            str(group): {
                                "edge_count": len(group_terms[group]),
                                "edges": group_terms[group],
                            }
                            for group in (0, 1, 2)
                        },
                        "faithfulness": {
                            "none": none_curves.get(sample_id),
                            "strict": strict_curves.get(sample_id),
                            "remove_parent_group": removal_metrics,
                        },
                        "heldout": {
                            "none": heldout.get(str(none_root), {}).get(sample_id),
                            "strict": heldout.get(str(strict_root), {}).get(
                                sample_id
                            ),
                        },
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
    finally:
        oracle.close()
    rows_path = destination / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    metric_names = (
        "comprehensiveness",
        "sufficiency",
        "aopc_comprehensiveness",
        "aopc_sufficiency",
        "aopc",
        "aupc",
    )
    comparisons: Dict[str, Any] = {}
    for metric in metric_names:
        strict_diffs = {
            row["sample_id"]: [
                float(row["faithfulness"]["strict"][metric])
                - float(row["faithfulness"]["none"][metric])
            ]
            for row in rows
            if row["faithfulness"]["none"] is not None
            and row["faithfulness"]["strict"] is not None
        }
        comparisons[f"strict_minus_none/{metric}"] = paired_cluster_summary(
            strict_diffs,
            seed=int(seed),
            n_bootstrap=int(bootstrap),
        )
        for group in (0, 1, 2):
            removal_diffs = {
                row["sample_id"]: [
                    float(
                        row["faithfulness"]["remove_parent_group"][str(group)][
                            metric
                        ]
                    )
                    - float(row["faithfulness"]["none"][metric])
                ]
                for row in rows
                if row["faithfulness"]["none"] is not None
            }
            comparisons[
                f"remove_parent_group_{group}_minus_none/{metric}"
            ] = paired_cluster_summary(
                removal_diffs,
                seed=int(seed) + group + 1,
                n_bootstrap=int(bootstrap),
            )
    parent_group_summary: Dict[str, Any] = {}
    for group in (0, 1, 2):
        edges = [
            edge
            for row in rows
            for edge in row["parent_groups"][str(group)]["edges"]
        ]
        exact_values = [
            abs(float(edge["exact_coefficient"]))
            for edge in edges
            if edge.get("exact_coefficient") is not None
        ]
        parent_group_summary[str(group)] = {
            "sample_count": len(rows),
            "samples_with_edges": sum(
                bool(row["parent_groups"][str(group)]["edge_count"])
                for row in rows
            ),
            "edge_count": len(edges),
            "estimated_coefficient_abs": aggregate_numeric(
                [abs(float(edge["coefficient"])) for edge in edges]
            ),
            "verified_exact_coefficient_abs": aggregate_numeric(exact_values),
        }
    summary = {
        "schema_version": "1.0",
        "audit_id": audit_id,
        "none_run": str(none_root),
        "strict_run": str(strict_root),
        "sample_count": len(rows),
        "failed_count": len(failures),
        "failures": failures,
        "observation_digest_mismatch_count": sum(
            not bool(row["observation_digest_match"]) for row in rows
        ),
        "support": {
            variant: {
                key: aggregate_numeric(
                    [float(row[variant][key]) for row in rows]
                )
                for key in (
                    "total_support_size",
                    "singleton_support_size",
                    "pair_support_size",
                )
            }
            for variant in ("none_support", "strict_support")
        },
        "parent_groups": parent_group_summary,
        "comparisons": comparisons,
        "analysis_query_cost": ledger.to_dict(),
        "attribution_cost_includes_analysis": False,
    }
    atomic_write_json(destination / "summary.json", summary)
    for root in (none_root, strict_root):
        atomic_write_json(
            ensure_dir(root / "analyses") / f"hierarchy-{audit_id}.json",
            {
                "schema_version": "1.0",
                "audit_id": audit_id,
                "audit_dir": str(destination.resolve()),
                "summary": str((destination / "summary.json").resolve()),
                "rows": str(rows_path.resolve()),
            },
        )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run hierarchy analysis from CLI arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    destination = analyze_hierarchy(
        args.none_run,
        args.strict_run,
        verification_dir=args.verification_dir,
        heldout_audit=args.heldout_audit,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    print(f"[hierarchy-analyzed] audit={destination}")


if __name__ == "__main__":
    main()
