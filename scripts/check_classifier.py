"""Report task accuracy independently of attribution experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.cli.run import _load_bundle
from mobius.core.config import load_config, resolve_config, scientific_config
from mobius.core.results import canonical_digest, dataset_slug, model_slug
from mobius.core.runtime import atomic_write_json, ensure_dir
from mobius.evaluation.classification import evaluate_text_classifier
from mobius.models.hf import build_scorer
from mobius.models.prompting import build_classification_prompt


def build_parser() -> argparse.ArgumentParser:
    """Build the task-validation command-line parser."""

    parser = argparse.ArgumentParser(
        description="Report full-input classifier diagnostics before attribution."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    return parser


def _resolved_config(args) -> Dict[str, Any]:
    """Load configuration and apply lightweight validation overrides."""

    config = load_config(args.config)
    config["dataset"] = dict(config.get("dataset", {}))
    config["model"] = dict(config.get("model", {}))
    if args.device is not None:
        config["model"]["device"] = str(args.device)
    if args.max_samples is not None:
        config["dataset"]["max_samples"] = int(args.max_samples)
    return resolve_config(config)


def _default_output(config: Dict[str, Any], prompt_version: str) -> Path:
    """Build a readable default report path for one dataset and model."""

    fingerprint = canonical_digest(
        scientific_config(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "prompt": config.get("prompt", {}),
            }
        )
    )[:8]
    return (
        Path("results/classifier-check")
        / dataset_slug(dict(config["dataset"]))
        / model_slug(dict(config["model"]))
        / f"{prompt_version}-{fingerprint}.json"
    )


def main(argv: list[str] | None = None) -> None:
    """Score all selected full inputs and write descriptive diagnostics."""

    args = build_parser().parse_args(argv)
    config = _resolved_config(args)
    bundle = _load_bundle(dict(config["dataset"]))
    prompt = build_classification_prompt(
        dataset_name=bundle.dataset_name,
        verbalizers=bundle.verbalizers,
        prompt_config=dict(config.get("prompt", {})),
    )
    scorer = build_scorer(
        dict(config["model"]),
        bundle.verbalizers,
        batch_size=int(config.get("batch_size", 16)),
        dataset_name=bundle.dataset_name,
        prompt_config=prompt.to_config(),
    )
    report = evaluate_text_classifier(bundle, scorer)
    # Preserve enough protocol metadata to aggregate reports without reopening configs.
    report["dataset"] = {
        "name": bundle.dataset_name,
        "split": bundle.split,
    }
    report["model"] = dict(config["model"])
    report["prompt"] = prompt.to_config()
    report["config_fingerprint"] = canonical_digest(scientific_config(config))
    output = Path(args.output) if args.output else _default_output(config, prompt.version)
    ensure_dir(output.parent)
    atomic_write_json(output, report)
    print(
        "[classifier-check] "
        f"dataset={bundle.dataset_name} samples={report['sample_count']} "
        f"accuracy={report['accuracy']:.4f} "
        f"balanced_accuracy={report['balanced_accuracy']:.4f} "
        f"majority={report['majority_baseline_accuracy']:.4f} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
