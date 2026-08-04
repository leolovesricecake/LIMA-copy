"""Run shared-value word Occlusion or LIME."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from mobius.cli.run import _load_bundle
from mobius.core.config import load_config, resolve_config
from mobius.core.results import default_run_dir
from mobius.core.runtime import set_seed
from mobius.methods.first_order import (
    FIRST_ORDER_METHODS,
    run_first_order,
)
from mobius.models.hf import build_scorer
from mobius.models.prompting import build_classification_prompt
from mobius.values.classification import effective_target_mode


def build_parser() -> argparse.ArgumentParser:
    """Build the first-order baseline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run shared-value word Occlusion or LIME."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--results-dir")
    parser.add_argument("--device")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--model-path")
    parser.add_argument("--run-suffix")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser


def _apply_overrides(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Apply explicit runtime and first-order scientific overrides."""

    output = dict(config)
    output["dataset"] = dict(output.get("dataset", {}))
    output["model"] = dict(output.get("model", {}))
    if args.results_dir is not None:
        output["results_dir"] = args.results_dir
    if args.device is not None:
        output["model"]["device"] = args.device
    if args.dataset is not None:
        output["dataset"]["name"] = args.dataset
    if args.split is not None:
        output["dataset"]["split"] = args.split
    if args.model_path is not None:
        output["model"]["model_path"] = args.model_path
    if args.run_suffix is not None:
        output["run_suffix"] = args.run_suffix
    if args.max_samples is not None:
        output["dataset"]["max_samples"] = args.max_samples
    if args.budget is not None:
        output["budget"] = args.budget
    if args.seed is not None:
        output["seed"] = args.seed
    resolved = resolve_config(output)
    if str(resolved.get("method")) not in FIRST_ORDER_METHODS:
        raise ValueError(
            f"First-order config method must be one of {sorted(FIRST_ORDER_METHODS)}."
        )
    return resolved


def main(argv: list[str] | None = None) -> None:
    """Resolve one first-order run and execute it end to end."""

    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    bundle = _load_bundle(dict(config["dataset"]))
    config["dataset"] = {
        **dict(config["dataset"]),
        "name": bundle.dataset_name,
        "split": bundle.split,
        "verbalizers": list(bundle.verbalizers),
    }
    config["target_mode"] = effective_target_mode(
        str(config["value_function"]),
        str(config["target_mode"]),
    )
    prompt = build_classification_prompt(
        dataset_name=bundle.dataset_name,
        verbalizers=bundle.verbalizers,
        prompt_config=dict(config.get("prompt", {})),
    )
    config["prompt"] = prompt.to_config()
    scorer = build_scorer(
        dict(config["model"]),
        bundle.verbalizers,
        batch_size=int(config["batch_size"]),
        dataset_name=bundle.dataset_name,
        prompt_config=config["prompt"],
    )
    results_dir = Path(config.get("results_dir", "results/baselines/first-order"))
    run_dir = (
        Path(args.output_root)
        if args.output_root
        else default_run_dir(results_dir, config)
    )
    cache_path = results_dir / ".cache" / "value_oracle.sqlite3"
    status = run_first_order(
        config,
        bundle,
        scorer,
        run_dir=run_dir,
        cache_path=cache_path,
        overwrite=bool(args.overwrite),
        command=" ".join(sys.argv),
        evaluate=not bool(args.no_eval),
    )
    print(
        f"[complete] state={status['state']} "
        f"completed={status['completed_count']} "
        f"failed={status['failed_count']} "
        f"skipped={status['skipped_count']} run={run_dir}"
    )


if __name__ == "__main__":
    main()
