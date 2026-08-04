"""Verify selected deletion-Mobius pairs with exact four-query differences."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.analysis.statistics import (
    aggregate_numeric,
    paired_cluster_summary,
)
from mobius.core.artifacts import load_surrogate_artifact
from mobius.core.results import canonical_digest, default_audit_dir
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.core.schema import TextChunk
from mobius.models.hf import build_scorer
from mobius.models.oracle import QueryLedger, ValueOracle, stable_digest
from mobius.text.coalitions import CoalitionGame
from mobius.values.classification import attribution_values


def build_parser() -> argparse.ArgumentParser:
    """Build the targeted interaction verification CLI."""

    parser = argparse.ArgumentParser(
        description="Verify top fitted deletion-Mobius pairs and random controls."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-per-parent-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260726)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _chunks(payload: Mapping[str, Any]) -> list[TextChunk]:
    """Reconstruct text chunks for exact coalition queries."""

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
    """Encode a player tuple as an integer term."""

    value = 0
    for player in players:
        value |= 1 << int(player)
    return value


def _decode_support(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    """Decode support rows to integer terms."""

    return {_term(row.get("players", [])) for row in rows}


def _pair_coefficients(surrogate: Mapping[str, Any]) -> Dict[Tuple[int, int], float]:
    """Extract fitted degree-two coefficients from the final predictor."""

    return {
        tuple(int(value) for value in row["players"]): float(row["coefficient"])
        for row in dict(surrogate["predictor"]).get("terms", [])
        if len(row.get("players", [])) == 2
    }


def _parent_count(pair: Tuple[int, int], singleton_support: set[int]) -> int:
    """Count selected singleton parents for a pair."""

    return sum((1 << int(player)) in singleton_support for player in pair)


def _match_random_pairs(
    selected: Sequence[Tuple[int, int]],
    *,
    n_features: int,
    seed: int,
    excluded_pairs: Sequence[Tuple[int, int]] = (),
) -> list[Dict[str, Any]]:
    """Match random non-selected pairs by exact or nearest player distance."""

    selected_set = {
        tuple(sorted(pair))
        for pair in [*selected, *excluded_pairs]
    }
    candidates = [
        pair
        for pair in itertools.combinations(range(int(n_features)), 2)
        if pair not in selected_set
    ]
    rng = np.random.default_rng(int(seed))
    unused = set(candidates)
    rows = []
    for pair in selected:
        distance = abs(int(pair[1]) - int(pair[0]))
        pool = [candidate for candidate in unused if candidate[1] - candidate[0] == distance]
        fallback = "exact_distance"
        if not pool and unused:
            nearest = min(
                abs((candidate[1] - candidate[0]) - distance)
                for candidate in unused
            )
            pool = [
                candidate
                for candidate in unused
                if abs((candidate[1] - candidate[0]) - distance) == nearest
            ]
            fallback = "nearest_distance"
        if not pool and candidates:
            nearest = min(
                abs((candidate[1] - candidate[0]) - distance)
                for candidate in candidates
            )
            pool = [
                candidate
                for candidate in candidates
                if abs((candidate[1] - candidate[0]) - distance) == nearest
            ]
            fallback = "reused_nearest_distance"
        if not pool:
            rows.append(
                {
                    "selected_pair": list(pair),
                    "random_pair": None,
                    "match_strategy": "unavailable",
                }
            )
            continue
        chosen = tuple(pool[int(rng.integers(0, len(pool)))])
        unused.discard(chosen)
        rows.append(
            {
                "selected_pair": list(pair),
                "random_pair": list(chosen),
                "match_strategy": fallback,
                "selected_distance": distance,
                "random_distance": chosen[1] - chosen[0],
            }
        )
    return rows


def deletion_pair_coefficient(values: Sequence[float]) -> float:
    """Compute one exact deletion-Mobius pair coefficient from full/single/pair values."""

    if len(values) != 4:
        raise ValueError("A deletion pair coefficient requires exactly four values.")
    numeric = [float(value) for value in values]
    return float(numeric[3] - numeric[1] - numeric[2] + numeric[0])


def _verify_pair(
    pair: Tuple[int, int],
    *,
    estimated_coefficient: float,
    sample_id: str,
    game: CoalitionGame,
    oracle: ValueOracle,
    ledger: QueryLedger,
    target_label: int,
    value_function: str,
) -> Dict[str, Any]:
    """Query the exact four deletion masks and compute one pair coefficient."""

    full = (1 << game.n_players) - 1
    left, right = (int(pair[0]), int(pair[1]))
    masks = [
        full,
        full & ~(1 << left),
        full & ~(1 << right),
        full & ~(1 << left) & ~(1 << right),
    ]
    label_scores = oracle.score_texts(
        game.texts(masks),
        logical_keys=[
            stable_digest(
                {
                    "sample_id": sample_id,
                    "pair": list(pair),
                    "mask": int(mask),
                    "category": "interaction_verification",
                }
            )
            for mask in masks
        ],
        ledger=ledger,
        category="interaction_verification",
    )
    values = attribution_values(
        label_scores,
        target_class=int(target_label),
        value_function=value_function,
    )
    exact = deletion_pair_coefficient(values)
    error = float(estimated_coefficient) - exact
    estimated_sign = int(np.sign(float(estimated_coefficient)))
    exact_sign = int(np.sign(exact))
    return {
        "players": [left, right],
        "distance": right - left,
        "estimated_coefficient": float(estimated_coefficient),
        "exact_coefficient": exact,
        "estimated_sign": estimated_sign,
        "exact_sign": exact_sign,
        "sign_correct": bool(estimated_sign == exact_sign),
        "absolute_error": abs(error),
        "squared_error": error**2,
        "keep_masks": [int(mask) for mask in masks],
        "label_scores": [
            [float(value) for value in row] for row in label_scores
        ],
        "attribution_values": [float(value) for value in values],
    }


def _correlation(
    estimated: Sequence[float],
    exact: Sequence[float],
) -> Dict[str, float | None]:
    """Compute Pearson and Spearman correlations when they are defined."""

    if len(estimated) < 2:
        return {"pearson": None, "spearman": None}
    left = np.asarray(estimated, dtype=np.float64)
    right = np.asarray(exact, dtype=np.float64)
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return {"pearson": None, "spearman": None}
    def ranks(values: np.ndarray) -> np.ndarray:
        """Assign stable average ranks to tied values."""

        order = np.argsort(values, kind="mergesort")
        output = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while (
                end < len(order)
                and values[order[end]] == values[order[start]]
            ):
                end += 1
            output[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return output

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": (
            float(np.corrcoef(left_ranks, right_ranks)[0, 1])
            if np.std(left_ranks) > 1e-15
            and np.std(right_ranks) > 1e-15
            else None
        ),
    }


def verify_runs(
    run_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path | None,
    device: str | None,
    top_k: int,
    top_k_per_parent_group: int,
    seed: int,
    bootstrap: int,
) -> Path:
    """Verify top interactions across runs and write one shared audit."""

    roots = [Path(value).resolve() for value in run_dirs]
    run_payloads = [_load_json(root / "run.json") for root in roots]
    configs = [dict(payload["scientific_config"]) for payload in run_payloads]
    first_model = dict(configs[0]["model"])
    first_dataset = dict(configs[0]["dataset"])
    for config in configs:
        if dict(config["model"]).get("model_path") != first_model.get("model_path"):
            raise ValueError("Interaction verification runs must share one model.")
        if dict(config["dataset"]).get("name") != first_dataset.get("name"):
            raise ValueError("Interaction verification runs must share one dataset.")
        if str(config.get("basis")) != "deletion_mobius":
            raise ValueError("Interaction verification requires deletion_mobius.")
    settings = {
        "top_k": int(top_k),
        "top_k_per_parent_group": int(top_k_per_parent_group),
        "seed": int(seed),
        "bootstrap": int(bootstrap),
    }
    audit_id = canonical_digest(
        {
            "runs": [
                {
                    "run_id": payload.get("run_id"),
                    "fingerprint": payload.get("config_fingerprint"),
                }
                for payload in run_payloads
            ],
            "settings": settings,
        }
    )[:12]
    destination = (
        Path(output_dir)
        if output_dir is not None
        else default_audit_dir(first_dataset, "interactions", audit_id)
    )
    ensure_dir(destination)
    if device is not None:
        first_model["device"] = str(device)
    verbalizers = [str(value) for value in first_dataset.get("verbalizers", [])]
    scorer = build_scorer(
        first_model,
        verbalizers,
        batch_size=int(configs[0].get("batch_size", 16)),
        dataset_name=str(first_dataset.get("name", "dataset")),
        prompt_config=dict(configs[0].get("prompt", {})),
    )
    oracle = ValueOracle(
        scorer,
        destination / ".cache" / "value_oracle.sqlite3",
        model_fingerprint=first_model,
        batch_size=int(configs[0].get("batch_size", 16)),
    )
    ledger = QueryLedger(f"interaction-verification/{audit_id}")
    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    try:
        for root, run_payload, config in zip(roots, run_payloads, configs):
            for sample_path in sorted((root / "samples").glob("*.json")):
                sample_id = sample_path.stem
                try:
                    sample = _load_json(sample_path)
                    surrogate = load_surrogate_artifact(
                        root / "surrogates" / f"{sample_id}.json"
                    )
                    coefficients = _pair_coefficients(surrogate)
                    ranked = sorted(
                        coefficients,
                        key=lambda pair: (
                            -abs(coefficients[pair]),
                            pair,
                        ),
                    )
                    global_pairs = ranked[: max(0, int(top_k))]
                    selection_support = _decode_support(
                        dict(surrogate.get("support", {})).get(
                            "selection",
                            [],
                        )
                    )
                    singleton_support = {
                        term
                        for term in selection_support
                        if bin(int(term)).count("1") == 1
                    }
                    selected_sources: Dict[Tuple[int, int], set[str]] = {
                        pair: {"global_top"} for pair in global_pairs
                    }
                    for group in (0, 1, 2):
                        group_pairs = [
                            pair
                            for pair in ranked
                            if _parent_count(pair, singleton_support) == group
                        ][: max(0, int(top_k_per_parent_group))]
                        for pair in group_pairs:
                            selected_sources.setdefault(pair, set()).add(
                                f"parent_group_{group}"
                            )
                    game = CoalitionGame(
                        _chunks(sample),
                        surrogate["player_to_chunk_id"],
                    )
                    verified_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
                    for pair, sources in selected_sources.items():
                        verified = _verify_pair(
                            pair,
                            estimated_coefficient=coefficients[pair],
                            sample_id=sample_id,
                            game=game,
                            oracle=oracle,
                            ledger=ledger,
                            target_label=int(sample["target_label"]),
                            value_function=str(config["value_function"]),
                        )
                        row = {
                            **verified,
                            "row_type": "selected",
                            "run_id": run_payload.get("run_id"),
                            "run_path": str(root),
                            "seed": config.get("seed"),
                            "sample_id": sample_id,
                            "selection_sources": sorted(sources),
                            "parent_count": _parent_count(
                                pair,
                                singleton_support,
                            ),
                        }
                        rows.append(row)
                        verified_by_pair[pair] = row
                    pair_seed = int(seed) + int.from_bytes(
                        hashlib.sha256(
                            f"{run_payload.get('run_id')}:{sample_id}".encode()
                        ).digest()[:4],
                        "big",
                    )
                    matches = _match_random_pairs(
                        global_pairs,
                        n_features=int(surrogate["n_features"]),
                        seed=pair_seed,
                        excluded_pairs=list(coefficients),
                    )
                    for match in matches:
                        random_pair = match["random_pair"]
                        if random_pair is None:
                            continue
                        pair = (int(random_pair[0]), int(random_pair[1]))
                        verified = _verify_pair(
                            pair,
                            estimated_coefficient=coefficients.get(pair, 0.0),
                            sample_id=sample_id,
                            game=game,
                            oracle=oracle,
                            ledger=ledger,
                            target_label=int(sample["target_label"]),
                            value_function=str(config["value_function"]),
                        )
                        selected_pair = tuple(match["selected_pair"])
                        rows.append(
                            {
                                **verified,
                                "row_type": "random_control",
                                "run_id": run_payload.get("run_id"),
                                "run_path": str(root),
                                "seed": config.get("seed"),
                                "sample_id": sample_id,
                                "matched_selected_pair": list(selected_pair),
                                "match_strategy": match["match_strategy"],
                                "parent_count": _parent_count(
                                    pair,
                                    singleton_support,
                                ),
                            }
                        )
                except Exception as error:
                    failures.append(
                        {
                            "run_id": run_payload.get("run_id"),
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
    global_selected = [
        row
        for row in rows
        if row["row_type"] == "selected"
        and "global_top" in row["selection_sources"]
    ]
    random_by_key = {
        (
            row["run_id"],
            row["sample_id"],
            tuple(row["matched_selected_pair"]),
        ): row
        for row in rows
        if row["row_type"] == "random_control"
    }
    paired_differences: Dict[str, list[float]] = {}
    for row in global_selected:
        control = random_by_key.get(
            (row["run_id"], row["sample_id"], tuple(row["players"]))
        )
        if control is not None:
            paired_differences.setdefault(str(row["sample_id"]), []).append(
                abs(float(row["exact_coefficient"]))
                - abs(float(control["exact_coefficient"]))
            )
    correlations = _correlation(
        [float(row["estimated_coefficient"]) for row in global_selected],
        [float(row["exact_coefficient"]) for row in global_selected],
    )
    edge_occurrences: Dict[Tuple[str, Tuple[int, int]], list[Dict[str, Any]]] = {}
    for row in global_selected:
        edge_occurrences.setdefault(
            (str(row["sample_id"]), tuple(row["players"])),
            [],
        ).append(row)
    stability = [
        {
            "sample_id": key[0],
            "players": list(key[1]),
            "selected_run_count": len(values),
            "selection_frequency": len(values) / max(1, len(roots)),
            "estimated_sign_stable": len(
                {int(np.sign(float(row["estimated_coefficient"]))) for row in values}
            )
            <= 1,
            "exact_sign_stable": len(
                {int(np.sign(float(row["exact_coefficient"]))) for row in values}
            )
            <= 1,
            "estimated_exact_sign_agreement_rate": float(
                np.mean([bool(row["sign_correct"]) for row in values])
            ),
            "parent_counts": sorted(
                {int(row["parent_count"]) for row in values}
            ),
        }
        for key, values in sorted(edge_occurrences.items())
    ]
    summary = {
        "schema_version": "1.0",
        "audit_id": audit_id,
        "settings": settings,
        "run_count": len(roots),
        "selected_edge_count": len(global_selected),
        "random_control_count": len(random_by_key),
        "mae": aggregate_numeric(
            [float(row["absolute_error"]) for row in global_selected]
        ),
        "rmse": (
            float(
                np.sqrt(
                    np.mean(
                        [
                            float(row["squared_error"])
                            for row in global_selected
                        ]
                    )
                )
            )
            if global_selected
            else None
        ),
        **correlations,
        "selected_abs_exact": aggregate_numeric(
            [abs(float(row["exact_coefficient"])) for row in global_selected]
        ),
        "random_abs_exact": aggregate_numeric(
            [
                abs(float(row["exact_coefficient"]))
                for row in rows
                if row["row_type"] == "random_control"
            ]
        ),
        "selected_minus_random": paired_cluster_summary(
            paired_differences,
            seed=int(seed),
            n_bootstrap=int(bootstrap),
        ),
        "stability": stability,
        "failures": failures,
        "query_cost": ledger.to_dict(),
        "attribution_cost_includes_verification": False,
    }
    atomic_write_json(destination / "summary.json", summary)
    for root, payload in zip(roots, run_payloads):
        pointer = {
            "schema_version": "1.0",
            "audit_id": audit_id,
            "audit_dir": str(destination.resolve()),
            "summary": str((destination / "summary.json").resolve()),
            "rows": str(rows_path.resolve()),
        }
        atomic_write_json(
            ensure_dir(root / "analyses")
            / f"interaction-verification-{audit_id}.json",
            pointer,
        )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run interaction verification from CLI arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    destination = verify_runs(
        args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        top_k=args.top_k,
        top_k_per_parent_group=args.top_k_per_parent_group,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    print(f"[interactions-verified] audit={destination}")


if __name__ == "__main__":
    main()
