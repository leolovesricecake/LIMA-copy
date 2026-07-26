"""Derive a new sparse-Mobius ranking run without refitting its surrogate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.core.artifacts import (
    load_observation_artifact,
    load_surrogate_artifact,
)
from mobius.core.results import ResultStore, default_run_dir
from mobius.core.schema import AttributionResult, TextChunk
from mobius.methods.sparse.estimator import sparse_model_from_surrogate
from mobius.methods.sparse.projector import normalize_projector, project_nodes
from mobius.text.chunks import compose_text


def build_parser() -> argparse.ArgumentParser:
    """Build the offline projection derivation CLI."""

    parser = argparse.ArgumentParser(
        description="Derive another projector from one fitted sparse surrogate run."
    )
    parser.add_argument("--input-run", required=True)
    parser.add_argument(
        "--projector",
        required=True,
        choices=["signed_equal_share", "absolute_equal_share", "singleton_only"],
    )
    parser.add_argument("--output-root", default="results/mobius-mechanisms")
    parser.add_argument("--run-suffix")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _chunks(payload: Dict[str, Any]) -> list[TextChunk]:
    """Reconstruct explanation chunks from one source sample."""

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


def _ranking(
    player_to_chunk_id: Sequence[int],
    active_scores: Sequence[float],
    all_chunk_ids: Sequence[int],
) -> list[int]:
    """Rank active chunks by score and append inactive chunks stably."""

    active = sorted(
        (
            (int(chunk_id), float(active_scores[player]))
            for player, chunk_id in enumerate(player_to_chunk_id)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ranked = [chunk_id for chunk_id, _ in active]
    present = set(ranked)
    return ranked + [
        int(chunk_id) for chunk_id in all_chunk_ids if int(chunk_id) not in present
    ]


def derive_projection_run(
    input_run: str | Path,
    *,
    projector: str,
    output_root: str | Path,
    run_suffix: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a standard run whose only scientific change is node projection."""

    source = Path(input_run)
    run_payload = json.loads((source / "run.json").read_text(encoding="utf-8"))
    config = dict(run_payload["scientific_config"])
    if str(config.get("method")) != "sparse_mobius":
        raise ValueError("Projection derivation only supports sparse_mobius runs.")
    mode = normalize_projector(projector)
    if str(config.get("projector")) == mode:
        raise ValueError("Derived projector must differ from the source projector.")
    config["projector"] = mode
    config["run_suffix"] = str(
        run_suffix or f"derived-{mode.replace('_equal_share', '')}"
    )
    destination = default_run_dir(output_root, config)
    store = ResultStore(
        destination,
        config,
        output_level=str(config.get("output_level", "standard")),
        overwrite=bool(overwrite),
        required_artifacts=("observation", "surrogate"),
        provenance={
            "kind": "offline_projection_derivation",
            "source_run": str(source.resolve()),
            "source_run_id": run_payload.get("run_id"),
            "model_calls": 0,
        },
    )
    for sample_path in sorted((source / "samples").glob("*.json")):
        sample_id = sample_path.stem
        if not overwrite and store.sample_complete(sample_id):
            continue
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        observation = load_observation_artifact(
            source / "observations" / f"{sample_id}.npz"
        )
        surrogate = load_surrogate_artifact(
            source / "surrogates" / f"{sample_id}.json"
        )
        model = sparse_model_from_surrogate(surrogate)
        active_scores = project_nodes(model, mode)
        mapping = [int(value) for value in surrogate["player_to_chunk_id"]]
        chunks = _chunks(sample)
        node_scores = [0.0] * len(chunks)
        for player, chunk_id in enumerate(mapping):
            node_scores[chunk_id] = float(active_scores[player])
        ranking = _ranking(
            mapping,
            active_scores,
            [chunk.chunk_id for chunk in chunks],
        )
        selected = ranking[: min(int(config.get("k", 8)), len(mapping))]
        method_summary = dict(sample.get("method_summary", {}))
        method_summary.update(
            {
                "projector": mode,
                "selected_text": compose_text(chunks, selected),
                "observation_digest": observation["digest"],
                "surrogate_digest": surrogate["digest"],
                "derivation": {
                    "source_run_id": run_payload.get("run_id"),
                    "source_sample_id": sample_id,
                    "model_calls": 0,
                },
            }
        )
        store.write_sample(
            AttributionResult(
                sample_id=sample_id,
                gold_label=int(sample["gold_label"]),
                predicted_label=int(sample["predicted_label"]),
                target_label=int(sample["target_label"]),
                text=str(sample["text"]),
                chunks=chunks,
                node_scores=node_scores,
                ranking=ranking,
                selected_ids=selected,
                attribution_cost=dict(sample.get("attribution_cost", {})),
                method_summary=method_summary,
                observation_artifact=observation,
                surrogate_artifact=surrogate,
            )
        )
        written_observation = load_observation_artifact(
            store.observations_dir / f"{sample_id}.npz"
        )
        written_surrogate = load_surrogate_artifact(
            store.surrogates_dir / f"{sample_id}.json"
        )
        if written_observation["digest"] != observation["digest"]:
            raise RuntimeError(
                f"Derived observation changed for sample {sample_id}."
            )
        if (
            written_surrogate["digest"] != surrogate["digest"]
            or written_surrogate["predictor"] != surrogate["predictor"]
            or written_surrogate.get("support") != surrogate.get("support")
        ):
            raise RuntimeError(
                f"Derived surrogate changed for sample {sample_id}."
            )
        store.write_status("running")
    store.finish(len(list((source / "samples").glob("*.json"))))
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline projection derivation command."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    destination = derive_projection_run(
        args.input_run,
        projector=args.projector,
        output_root=args.output_root,
        run_suffix=args.run_suffix,
        overwrite=bool(args.overwrite),
    )
    print(f"[derived] run={destination}")


if __name__ == "__main__":
    main()
