"""Run the modular sparse Mobius attribution method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from mobius.core.config import load_config, resolve_config
from mobius.core.results import default_run_dir
from mobius.core.runtime import set_seed
from mobius.core.schema import DatasetBundle, TextSample
from mobius.data.loader import load_dataset_bundle
from mobius.methods.sparse.explainer import run_sparse_mobius
from mobius.models.hf import build_scorer
from mobius.values.classification import effective_target_mode


def build_parser() -> argparse.ArgumentParser:
    """Build the sparse-method command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run low-order sparse interaction attribution."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--results-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser


def _apply_overrides(config: Mapping[str, Any], args) -> Dict[str, Any]:
    """Apply explicit CLI runtime and scientific overrides."""

    output = dict(config)
    output["dataset"] = dict(output.get("dataset", {}))
    output["model"] = dict(output.get("model", {}))
    if args.results_dir is not None:
        output["results_dir"] = args.results_dir
    if args.device is not None:
        output["model"]["device"] = args.device
    if args.max_samples is not None:
        output["dataset"]["max_samples"] = args.max_samples
    if args.budget is not None:
        output["budget"] = args.budget
    if args.seed is not None:
        output["seed"] = args.seed
    return resolve_config(output)


def _inline_bundle(dataset_config: Mapping[str, Any]) -> DatasetBundle | None:
    """Build a tiny configured dataset when inline samples are present."""

    rows = dataset_config.get("samples")
    if not isinstance(rows, list):
        return None
    verbalizers = [
        str(value)
        for value in dataset_config.get("verbalizers", ["negative", "positive"])
    ]
    samples = [
        TextSample(
            sample_id=str(row.get("sample_id", f"inline-{index}")),
            text=str(row["text"]),
            label=int(row["label"]),
            label_text=verbalizers[int(row["label"])],
            metadata={"source": "inline"},
        )
        for index, row in enumerate(rows)
    ]
    return DatasetBundle(
        dataset_name=str(dataset_config.get("name", "inline")),
        split=str(dataset_config.get("split", "validation")),
        samples=samples,
        label_names=verbalizers,
        verbalizers=verbalizers,
    )


def _load_bundle(dataset_config: Mapping[str, Any]) -> DatasetBundle:
    """Load inline or named text-classification data."""

    inline = _inline_bundle(dataset_config)
    if inline is not None:
        return inline
    return load_dataset_bundle(
        dataset_name=str(dataset_config["name"]),
        split=str(dataset_config.get("split", "validation")),
        max_samples=dataset_config.get("max_samples"),
        sst2_source=dataset_config.get("sst2_source"),
        dataset_cache_dir=dataset_config.get("dataset_cache_dir"),
        source=dataset_config.get("source"),
    )


def main(argv: list[str] | None = None) -> None:
    """Resolve one run and execute it end to end."""

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
    scorer = build_scorer(
        dict(config["model"]),
        bundle.verbalizers,
        batch_size=int(config["batch_size"]),
    )
    results_dir = Path(config.get("results_dir", "results/mobius"))
    run_dir = (
        Path(args.output_root)
        if args.output_root
        else default_run_dir(results_dir, config)
    )
    cache_path = results_dir / ".cache" / "value_oracle.sqlite3"
    status = run_sparse_mobius(
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
        f"[complete] state={status['state']} completed={status['completed_count']} "
        f"failed={status['failed_count']} skipped={status['skipped_count']} "
        f"run={run_dir}"
    )


if __name__ == "__main__":
    main()

