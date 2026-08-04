"""Evaluate a completed schema-v2 attribution run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobius.cli.run import _load_bundle
from mobius.evaluation.evaluator import evaluate_run
from mobius.models.hf import build_scorer


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluator parser."""

    parser = argparse.ArgumentParser(description="Evaluate a schema-v2 run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", choices=["predicted", "gold"], default="predicted")
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Reconstruct the run protocol and write metrics.json."""

    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = dict(run_payload["scientific_config"])
    if not config.get("prompt"):
        raise RuntimeError(
            "This run predates the task-aware prompt protocol and cannot be "
            "re-evaluated as a valid classification run. Re-run attribution first."
        )
    dataset_config = dict(config["dataset"])
    model_config = dict(config["model"])
    if args.device:
        model_config["device"] = args.device
    bundle = _load_bundle(dataset_config)
    scorer = build_scorer(
        model_config,
        bundle.verbalizers,
        batch_size=int(config.get("batch_size", 16)),
        dataset_name=bundle.dataset_name,
        prompt_config=dict(config.get("prompt", {})),
    )
    report = evaluate_run(
        run_dir,
        bundle,
        scorer,
        target=args.target,
        eval_granularity=str(config.get("eval_granularity", "word")),
        q_values=[
            int(value)
            for value in config.get("eval_q_values", [5, 10, 20, 50])
        ],
    )
    print(
        f"[evaluated] target={report['target']} samples={report['evaluated_count']} "
        f"metrics={run_dir / 'metrics.json'}"
    )


if __name__ == "__main__":
    main()
