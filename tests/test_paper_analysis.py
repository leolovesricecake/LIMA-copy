"""Representation audit and direct paper-result protocol tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mobius.analysis.contracts import assert_same_seed, model_cell_id
from mobius.analysis.exact_pairs import (
    exact_pair_coefficients,
    exact_pair_keep_masks,
    run_exact_pair_audit,
)
from mobius.analysis.paper_results import (
    _cross_seed_e1_tables,
    collect_paper_cell,
    summarize_paper_version,
)
from mobius.analysis.representation import (
    pair_ranking_metrics,
    pair_scores_from_proxyspex,
    rank_all_pairs,
    run_representation_audit,
)
from mobius.core.artifacts import (
    write_observation_artifact,
    write_surrogate_artifact,
)
from mobius.core.config import resolve_config
from mobius.models.mock import MockSentimentScorer


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one compact JSON object for a synthetic run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _scientific_config(role: str) -> dict[str, Any]:
    """Build aligned scientific configs whose method details may differ."""

    config: dict[str, Any] = {
        "method": "proxyspex" if role == "PROXYSPEX" else "sparse_mobius",
        "budget": 32,
        "seed": 42,
        "max_degree": 2,
        "k": 3,
        "value_function": (
            "predicted_probability"
            if role == "PROXYSPEX"
            else "target_probability"
        ),
        "target_mode": "predicted",
        "chunker": "word",
        "eval_granularity": "word",
        "eval_q_values": [1, 5, 10, 20, 50],
        "basis": "deletion_mobius",
        "hierarchy": "none",
        "projector": "signed_equal_share",
        "estimator": {
            "alphas": [1e-8, 1e-5],
            "l1_ratios": [1.0],
            "cv_folds": 2,
            "refit": "ols",
        },
        "dataset": {
            "name": "inline",
            "split": "validation",
            "verbalizers": ["negative", "positive"],
        },
        "model": {
            "type": "mock_sentiment",
            "model_path": "/models/Shared-Model",
            "dtype": "float32",
            "max_length": 128,
        },
    }
    if role == "A":
        config["max_degree"] = 1
        config["projector"] = "singleton_only"
    elif role == "B":
        config["projector"] = "singleton_only"
    elif role == "ABSOLUTE":
        config["projector"] = "absolute_equal_share"
    elif role == "STRICT":
        config["hierarchy"] = "strict"
    elif role == "PROXYSPEX":
        config.pop("max_degree")
        config["max_order"] = 2
        config["basis"] = "fourier"
    return config


def _curve_row(
    sample_id: str,
    *,
    aopc: float,
    aupc: float,
) -> dict[str, Any]:
    """Build one complete legacy curve row including the removable q=1 point."""

    per_q = {
        str(q): {
            "comprehensiveness": float(aopc * q / 50.0),
            "sufficiency": float(aupc * (50 - q) / 50.0),
        }
        for q in (1, 5, 10, 20, 50)
    }
    return {
        "sample_id": sample_id,
        "target": "predicted",
        "per_q": per_q,
        "metrics": {
            "aopc": float(aopc),
            "aupc": float(aupc),
            "comprehensiveness": per_q["20"]["comprehensiveness"],
            "sufficiency": per_q["20"]["sufficiency"],
            "aopc_comprehensiveness": -99.0,
            "aopc_sufficiency": -99.0,
        },
    }


def _write_curves(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write deterministic predicted-target curves for one synthetic role."""

    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    (run_dir / "curves-predicted.jsonl").write_text(text, encoding="utf-8")


def _create_run(
    root: Path,
    role: str,
    *,
    curve_values: Sequence[tuple[float, float]],
) -> Path:
    """Create one immutable synthetic run and its paired curve artifacts."""

    run_dir = root / role.lower()
    run_dir.mkdir(parents=True)
    config = _scientific_config(role)
    _write_json(
        run_dir / "run.json",
        {
            "schema_version": "2.0",
            "run_id": f"synthetic-{role.lower()}",
            "config_fingerprint": f"fingerprint-{role.lower()}",
            "scientific_config": config,
        },
    )
    _write_curves(
        run_dir,
        [
            _curve_row(
                sample_id,
                aopc=values[0],
                aupc=values[1],
            )
            for sample_id, values in zip(("exact", "heldout"), curve_values)
        ],
    )
    _write_json(
        run_dir / "status.json",
        {
            "state": "complete",
            "completed_count": 2,
            "failed_count": 0,
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "target": "predicted",
            "evaluated_count": 2,
            "failed_count": 0,
            "attribution_cost": {
                "attribution_budget_used": 64,
                "logical_unique_queries": 60,
                "physical_values_scored": 58,
                "model_forward_calls": 8,
                "batch_calls": 8,
                "batch_rows": 58,
                "elapsed_seconds": 1.5,
            },
        },
    )
    return run_dir


def _game_value(keep_mask: int, n_features: int) -> float:
    """Evaluate a known low-order deletion-Möbius probability game."""

    full = (1 << int(n_features)) - 1
    deleted = full ^ int(keep_mask)
    value = 0.80
    singleton_coefficients = (-0.04, -0.03, -0.02, -0.01)
    for player in range(int(n_features)):
        if deleted & (1 << player):
            value += singleton_coefficients[player]
    if deleted & 1 and deleted & 2:
        value += 0.06
    return float(value)


def _label_scores(values: Sequence[float]) -> np.ndarray:
    """Encode probabilities as two-class logits with exact softmax recovery."""

    return np.asarray(
        [[math.log(1.0 - value), math.log(value)] for value in values],
        dtype=np.float64,
    )


def _chunks(n_features: int) -> list[dict[str, Any]]:
    """Build character-aligned synthetic word chunks."""

    return [
        {
            "chunk_id": player,
            "start_char": 2 * player,
            "end_char": 2 * player + 1,
            "text": chr(ord("a") + player),
        }
        for player in range(int(n_features))
    ]


def _mobius_surrogate(sample_id: str, n_features: int) -> dict[str, Any]:
    """Serialize the exact sparse deletion polynomial used by the fixture."""

    terms = [
        {
            "players": [player],
            "coefficient": coefficient,
        }
        for player, coefficient in enumerate(
            (-0.04, -0.03, -0.02, -0.01)[:n_features]
        )
    ]
    terms.append({"players": [0, 1], "coefficient": 0.06})
    support = [
        {"players": row["players"]}
        for row in terms
    ]
    return {
        "sample_id": sample_id,
        "method": "sparse_mobius",
        "n_features": n_features,
        "player_to_chunk_id": list(range(n_features)),
        "target_label": 1,
        "candidate_definition": {"max_degree": 2},
        "support": {
            "selection": support,
            "hierarchy": support,
            "refit": support,
        },
        "fit_diagnostics": {"refit_mode": "ols"},
        "predictor": {
            "type": "sparse_polynomial",
            "basis": "deletion_mobius",
            "intercept": 0.80,
            "terms": terms,
        },
    }


def _proxyspex_surrogate(sample_id: str, n_features: int) -> dict[str, Any]:
    """Serialize a ranking-valid ProxySPEX sidecar for exact pair comparison."""

    return {
        "sample_id": sample_id,
        "method": "proxyspex",
        "n_features": n_features,
        "player_to_chunk_id": list(range(n_features)),
        "target_label": 1,
        "predictor": {
            "type": "refined_fourier",
            "basis": "fourier",
            "intercept": 0.75,
            "terms": [],
        },
        "final_interactions": [
            {"players": [0, 2], "coefficient": 0.5},
            {"players": [1, 2], "coefficient": 0.1},
        ],
    }


def _write_sidecars(
    c_run: Path,
    proxy_run: Path,
    *,
    sample_id: str,
    n_features: int,
    train_masks: Sequence[int],
) -> None:
    """Write aligned samples, observations, and surrogate artifacts."""

    text = " ".join(chr(ord("a") + index) for index in range(n_features))
    sample = {
        "sample_id": sample_id,
        "text": text,
        "target_label": 1,
        "chunks": _chunks(n_features),
    }
    _write_json(c_run / "samples" / f"{sample_id}.json", sample)
    _write_json(proxy_run / "samples" / f"{sample_id}.json", sample)
    values = [_game_value(mask, n_features) for mask in train_masks]
    write_observation_artifact(
        c_run / "observations" / f"{sample_id}.npz",
        {
            "sample_id": sample_id,
            "method": "sparse_mobius",
            "n_features": n_features,
            "keep_masks": list(train_masks),
            "label_scores": _label_scores(values),
            "attribution_values": values,
        },
    )
    write_surrogate_artifact(
        c_run / "surrogates" / f"{sample_id}.json",
        _mobius_surrogate(sample_id, n_features),
    )
    write_surrogate_artifact(
        proxy_run / "surrogates" / f"{sample_id}.json",
        _proxyspex_surrogate(sample_id, n_features),
    )


def _create_heldout_audit(
    root: Path,
    runs: Mapping[str, Path],
) -> Path:
    """Create a held-out fixture plus evaluator reports for every paper role."""

    audit = root / "heldout"
    samples = audit / "samples"
    samples.mkdir(parents=True)
    masks = [5, 6, 9, 10]
    values = [_game_value(mask, 4) for mask in masks]
    np.savez_compressed(
        samples / "heldout.npz",
        bernoulli_keep_masks=np.asarray(
            [
                [bool(mask & (1 << player)) for player in range(4)]
                for mask in masks
            ],
            dtype=bool,
        ),
        bernoulli_label_scores=_label_scores(values),
    )
    _write_json(
        audit / "manifest.json",
        {
            "schema_version": "1.1",
            "audit_id": "synthetic-heldout",
            "kind": "shared_surrogate_heldout",
            "metadata": {
                "dataset": {
                    "name": "inline",
                    "split": "validation",
                    "verbalizers": ["negative", "positive"],
                },
                "model": {
                    "type": "mock_sentiment",
                    "model_path": "/models/Shared-Model",
                    "dtype": "float32",
                    "max_length": 128,
                    "trust_remote_code": False,
                },
                "prompt": {},
                "chunker": "word",
                "eval_granularity": "word",
                "value_function": "predicted_probability",
                "target_mode": "predicted",
            },
            "runs": [
                {"path": str(run.resolve())}
                for _, run in sorted(runs.items())
            ],
            "settings": {"distributions": ["bernoulli"]},
            "samples": [
                {
                    "sample_id": "exact",
                    "status": "insufficient",
                },
                {
                    "sample_id": "heldout",
                    "status": "ok",
                    "artifact": "samples/heldout.npz",
                },
            ],
        },
    )
    quality_by_role = {
        "C": (0.90, 0.08, 0.04),
        "A": (0.55, 0.22, 0.14),
        "B": (0.72, 0.15, 0.09),
        "ABSOLUTE": (0.68, 0.17, 0.10),
        "STRICT": (0.63, 0.19, 0.12),
        "PROXYSPEX": (0.70, 0.16, 0.10),
    }
    evaluations = []
    for role, run in sorted(runs.items()):
        r2, nrmse, mae = quality_by_role[role]
        report_path = Path("metrics") / f"{role.lower()}.json"
        _write_json(
            audit / report_path,
            {
                "schema_version": "1.1",
                "audit_id": "synthetic-heldout",
                "run_id": f"synthetic-{role.lower()}",
                "run_path": str(run.resolve()),
                "evaluated_count": 1,
                "rows": [
                    {
                        "sample_id": "heldout",
                        "target_label": 1,
                        "distributions": {
                            "bernoulli": {
                                "r2": r2,
                                "nrmse_range": nrmse,
                                "mae": mae,
                            }
                        },
                    }
                ],
            },
        )
        evaluations.append(
            {
                "run_id": f"synthetic-{role.lower()}",
                "run_path": str(run.resolve()),
                "report": str(report_path),
            }
        )
    _write_json(
        audit / "evaluation-index.json",
        {
            "schema_version": "1.1",
            "audit_id": "synthetic-heldout",
            "evaluations": evaluations,
        },
    )
    return audit


def test_representation_audit_and_paper_results_are_cell_isolated(
    tmp_path: Path,
) -> None:
    """Run the complete audit-to-paper flow over synthetic sidecars."""

    runs_root = tmp_path / "runs"
    c_run = _create_run(
        runs_root,
        "C",
        curve_values=((0.60, 0.20), (0.58, 0.22)),
    )
    a_run = _create_run(
        runs_root,
        "A",
        curve_values=((0.40, 0.40), (0.38, 0.42)),
    )
    b_run = _create_run(
        runs_root,
        "B",
        curve_values=((0.50, 0.30), (0.48, 0.32)),
    )
    absolute_run = _create_run(
        runs_root,
        "ABSOLUTE",
        curve_values=((0.48, 0.32), (0.46, 0.34)),
    )
    strict_run = _create_run(
        runs_root,
        "STRICT",
        curve_values=((0.52, 0.28), (0.50, 0.30)),
    )
    proxy_run = _create_run(
        runs_root,
        "PROXYSPEX",
        curve_values=((0.30, 0.50), (0.28, 0.52)),
    )
    _write_sidecars(
        c_run,
        proxy_run,
        sample_id="exact",
        n_features=3,
        train_masks=list(range(8)),
    )
    _write_sidecars(
        c_run,
        proxy_run,
        sample_id="heldout",
        n_features=4,
        train_masks=[0, 1, 2, 3, 4, 8, 12, 15],
    )
    for derived_run in (b_run, absolute_run):
        for sample_id in ("exact", "heldout"):
            source_sample = c_run / "samples" / f"{sample_id}.json"
            _write_json(
                derived_run / "samples" / f"{sample_id}.json",
                json.loads(source_sample.read_text(encoding="utf-8")),
            )
            source_surrogate = json.loads(
                (
                    c_run / "surrogates" / f"{sample_id}.json"
                ).read_text(encoding="utf-8")
            )
            _write_json(
                derived_run / "surrogates" / f"{sample_id}.json",
                source_surrogate,
            )
    for controlled_run in (a_run, strict_run):
        for sample_id in ("exact", "heldout"):
            source_sample = c_run / "samples" / f"{sample_id}.json"
            _write_json(
                controlled_run / "samples" / f"{sample_id}.json",
                json.loads(source_sample.read_text(encoding="utf-8")),
            )
            source_observation = (
                c_run / "observations" / f"{sample_id}.npz"
            )
            controlled_observation = (
                controlled_run / "observations" / f"{sample_id}.npz"
            )
            controlled_observation.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            controlled_observation.write_bytes(source_observation.read_bytes())
    paper_runs = {
        "A": a_run,
        "B": b_run,
        "ABSOLUTE": absolute_run,
        "C": c_run,
        "PROXYSPEX": proxy_run,
        "STRICT": strict_run,
    }
    heldout = _create_heldout_audit(tmp_path, paper_runs)
    representation = run_representation_audit(
        c_run,
        proxy_run,
        heldout,
        output_root=tmp_path / "audits",
        compression_k=(1, 2, 4),
        max_exact_features=3,
        seed=7,
        bootstrap=50,
    )
    pair_oracle = run_exact_pair_audit(
        c_run,
        proxy_run,
        output_root=tmp_path / "audits",
        cache_path=tmp_path / "pair-cache.sqlite3",
        min_features=2,
        max_features=4,
        max_samples=2,
        seed=7,
        bootstrap=50,
        scorer=MockSentimentScorer(),
    )
    representation_manifest = json.loads(
        (representation / "manifest.json").read_text(encoding="utf-8")
    )
    representation_summary = json.loads(
        (representation / "summary.json").read_text(encoding="utf-8")
    )
    assert representation.parts[-3:-1] == ("inline", "representation")
    assert representation_manifest["query_cost"]["requested_queries"] == 0
    assert representation_summary["basis"]["sample_count"] == 2
    assert representation_summary["failed_count"] == 0
    assert representation_summary["exact_interactions"]["sample_count"] == 1
    assert representation_summary["exact_structure"]["sample_count"] == 1
    assert (
        representation_summary["exact_structure"]["degree_truncation"]["2"][
            "mobius"
        ]["r2"]["mean"]
        == 1.0
    )
    assert (
        representation_summary["exact_structure"]["degree_truncation"]["1"][
            "mobius"
        ]["r2"]["mean"]
        < 1.0
    )
    assert representation_summary["basis"]["fixed_k"]["bernoulli"]
    assert representation_summary["basis"]["design_geometry"]["mobius"]
    assert (
        representation_summary["exact_interactions"]["methods"]["mobius"][
            "ndcg_at_1"
        ]["mean"]
        == 1.0
    )
    assert (
        representation_summary["exact_interactions"]["methods"]["oracle"][
            "ndcg_at_5"
        ]["mean"]
        == 1.0
    )
    assert (
        representation_summary["exact_interactions"][
            "mobius_all_pair_coefficient_mae"
        ]["mean"]
        < 1e-12
    )

    output_root = tmp_path / "paper"
    cell = collect_paper_cell(
        "paper-v1",
        paper_runs,
        [representation, heldout, pair_oracle],
        output_root=output_root,
        q_values=(5, 10, 20, 50),
        analysis_seed=7,
        bootstrap=50,
    )
    manifest = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    with (cell / "paired-effects.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        paired_rows = list(csv.DictReader(handle))
    with (cell / "overall-attribution.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        overall_rows = list(csv.DictReader(handle))
    assert cell.name == "seed-42"
    assert cell.parent.name == "budget-32"
    assert cell.parent.parent.name.startswith("Shared-Model-")
    assert cell.parent.parent.parent.name == "inline-validation"
    assert manifest["dataset_id"] == "inline-validation"
    recomputed = [
        row
        for row in overall_rows
        if row["metric"] == "aopc_comprehensiveness"
    ]
    assert recomputed
    assert all(
        row["source"] == "recomputed_q_grid:5,10,20,50"
        for row in recomputed
    )
    stored = [row for row in overall_rows if row["metric"] == "aopc"]
    assert stored
    assert all(row["source"] == "stored_curve" for row in stored)
    c_vs_a = [
        row
        for row in paired_rows
        if row["comparison"] == "C_vs_A"
        and row["metric"] == "aopc"
    ]
    assert float(c_vs_a[0]["improvement_mean"]) > 0
    assert (cell / "costs.csv").is_file()
    assert (cell / "audit-costs.csv").is_file()
    assert (cell / "surrogate-heldout.csv").is_file()
    assert (cell / "representation-recovery.csv").is_file()
    assert (cell / "exact-structure.csv").is_file()
    assert (cell / "exact-interactions.csv").is_file()
    assert (cell / "projection-ablation.csv").is_file()
    with (cell / "projection-ablation.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        projection_rows = list(csv.DictReader(handle))
    assert {row["right_role"] for row in projection_rows} == {
        "ABSOLUTE",
        "B",
    }
    with (cell / "surrogate-heldout.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        heldout_rows = list(csv.DictReader(handle))
    heldout_c_vs_a = [
        row
        for row in heldout_rows
        if row["comparison"] == "C_vs_A"
        and row["metric"] == "r2"
    ]
    assert float(heldout_c_vs_a[0]["mean"]) > 0
    with (cell / "exact-interactions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        exact_rows = list(csv.DictReader(handle))
    assert {row["oracle_scope"] for row in exact_rows} == {
        "full_table",
        "pair_oracle",
    }
    assert {
        row["length_bin"]
        for row in exact_rows
        if row["oracle_scope"] == "pair_oracle"
    } == {"00-07", "all"}

    summary_root = summarize_paper_version(
        "paper-v1",
        output_root=output_root,
    )
    summary = json.loads(
        (summary_root / "index.json").read_text(encoding="utf-8")
    )
    assert summary["cell_count"] == 1
    assert (summary_root / "overall-attribution.csv").is_file()
    assert (summary_root / "audit-costs.csv").is_file()
    assert (summary_root / "surrogate-heldout.csv").is_file()
    assert (summary_root / "exact-structure.csv").is_file()
    for filename in (
        "table-e1-faithfulness.csv",
        "figure-e2a-budget-recovery.csv",
        "figure-e2b-fixed-support.csv",
        "table-e3-exact-pairs.csv",
        "table-e4-projection.csv",
    ):
        with (summary_root / filename).open(
            encoding="utf-8",
            newline="",
        ) as handle:
            canonical_rows = list(csv.DictReader(handle))
        assert canonical_rows, filename
        if filename.startswith(("figure-e2", "table-e3")):
            assert {row["attribution_seed"] for row in canonical_rows} == {"all"}
        if filename == "table-e1-faithfulness.csv":
            assert {row["role"] for row in canonical_rows} == {
                "A",
                "C",
                "PROXYSPEX",
            }
        if filename == "table-e3-exact-pairs.csv":
            assert {row["method"] for row in canonical_rows} <= {
                "mobius",
                "proxyspex",
                "mobius_vs_proxyspex",
            }
    assert (summary_root / "README.md").is_file()

    e1_only = collect_paper_cell(
        "paper-e1-only",
        {"A": a_run, "C": c_run, "PROXYSPEX": proxy_run},
        [],
        output_root=output_root,
        q_values=(5, 10, 20, 50),
        analysis_seed=7,
        bootstrap=20,
    )
    assert (e1_only / "surrogate-heldout-samples.csv").is_file()
    e1_aggregate = summarize_paper_version(
        "paper-e1-only",
        output_root=output_root,
    )
    with (e1_aggregate / "figure-e2a-budget-recovery.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_model_cell_id_prevents_same_basename_collisions() -> None:
    """Keep model outputs separate even when model directory names coincide."""

    left = {
        "model": {
            "type": "hf_causal_lm",
            "model_path": "/models/team-a/Qwen3-8B",
        }
    }
    right = {
        "model": {
            "type": "hf_causal_lm",
            "model_path": "/models/team-b/Qwen3-8B",
        }
    }
    assert model_cell_id(left).startswith("Qwen3-8B-")
    assert model_cell_id(left) != model_cell_id(right)


def test_default_paper_q_grid_excludes_one_percent() -> None:
    """Make future runs use only q values that perturb short word sequences."""

    config = resolve_config({})
    assert config["eval_q_values"] == [5, 10, 20, 50]


def test_paired_runs_reject_mismatched_attribution_seeds() -> None:
    """Prevent a paper cell from pairing methods produced under different seeds."""

    identities = [
        {"seed": 42, "path": "/runs/c"},
        {"seed": 43, "path": "/runs/proxy"},
    ]
    try:
        assert_same_seed(identities)
    except ValueError as error:
        assert "attribution seed" in str(error)
    else:
        raise AssertionError("Mismatched attribution seeds must be rejected.")


def test_exact_pair_mask_count_is_quadratic() -> None:
    """Require one full, every singleton, and every pair deletion exactly once."""

    for n_features in (2, 5, 12):
        masks = exact_pair_keep_masks(n_features)
        assert len(masks) == 1 + n_features + math.comb(n_features, 2)
        assert len(masks) == len(set(masks))


def test_exact_pair_oracle_recovers_known_deletion_coefficient() -> None:
    """Recover the fixture's only nonzero pair from quadratic query masks."""

    n_features = 4
    masks = exact_pair_keep_masks(n_features)
    exact = exact_pair_coefficients(
        {
            mask: _game_value(mask, n_features)
            for mask in masks
        },
        n_features,
    )
    assert np.isclose(exact[(0, 1)], 0.06)
    assert all(
        np.isclose(value, 0.0)
        for pair, value in exact.items()
        if pair != (0, 1)
    )


def test_proxyspex_pair_scores_use_fourier_predictor_not_fbii() -> None:
    """Derive E3 deletion pairs from the serialized function, never native FBII."""

    surrogate = {
        "sample_id": "proxy",
        "method": "proxyspex",
        "n_features": 2,
        "player_to_chunk_id": [0, 1],
        "predictor": {
            "type": "refined_fourier",
            "basis": "fourier",
            "intercept": 0.5,
            "terms": [
                {"players": [0, 1], "coefficient": 0.25},
            ],
        },
        "final_interactions": [
            {"players": [0, 1], "coefficient": -99.0},
        ],
    }
    assert np.isclose(pair_scores_from_proxyspex(surrogate)[(0, 1)], 1.0)


def test_pair_metrics_use_dynamic_top_ten_and_exact_signs() -> None:
    """Use all available pairs when fewer than ten exist and compare their signs."""

    pairs = [(0, 1), (0, 2), (1, 2)]
    exact = {(0, 1): 3.0, (0, 2): -2.0, (1, 2): 1.0}
    estimated = {(0, 1): 2.5, (0, 2): 1.5, (1, 2): 0.5}
    ranking = rank_all_pairs(pairs, estimated)
    metrics = pair_ranking_metrics(pairs, exact, estimated, ranking)
    assert metrics["recall_at_10"] == 1.0
    assert np.isclose(metrics["sign_agreement_at_10"], 2.0 / 3.0)


def test_cross_seed_e1_aggregates_within_sample_before_inference() -> None:
    """Average attribution seeds per sample before sample-level paired inference."""

    rows = []
    values = {
        ("C", "s1"): [0.2, 0.4],
        ("C", "s2"): [0.6, 0.8],
        ("PROXYSPEX", "s1"): [0.5, 0.7],
        ("PROXYSPEX", "s2"): [0.9, 1.1],
    }
    for (role, sample_id), seed_values in values.items():
        for seed, value in zip((42, 43), seed_values):
            rows.append(
                {
                    "paper_version": "paper-v2.3",
                    "dataset": "sst2",
                    "split": "validation",
                    "dataset_id": "sst2-validation",
                    "model": "Qwen3-8B",
                    "model_id": "Qwen3-8B-test",
                    "attribution_seed": str(seed),
                    "budget": "512",
                    "role": role,
                    "method": "sparse_mobius" if role == "C" else "proxyspex",
                    "sample_id": sample_id,
                    "metric": "aupc",
                    "direction": "lower_is_better",
                    "value": str(value),
                    "source": "stored_curve",
                }
            )
    table, paired = _cross_seed_e1_tables(rows, analysis_seed=7, bootstrap=20)
    c_row = next(row for row in table if row["role"] == "C")
    comparison = next(row for row in paired if row["right_role"] == "PROXYSPEX")
    assert np.isclose(c_row["mean"], 0.5)
    assert np.isclose(c_row["std"], np.std([0.3, 0.7], ddof=1))
    assert c_row["seed_count_min"] == c_row["seed_count_max"] == 2
    assert np.isclose(comparison["improvement_mean"], 0.3)
