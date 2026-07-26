"""Collect schema-v2 experiment metrics into one analysis-ready CSV file."""

from __future__ import annotations

import os
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


DEFAULT_OUTPUT = "results_summary.csv"

IDENTITY_FIELDS = (
    "dataset",
    "model",
    "method",
    "config",
    "target",
)

FAITHFULNESS_PRIORITY = (
    "comprehensiveness",
    "sufficiency",
    "aopc",
    "aupc",
    "aopc_comprehensiveness",
    "aopc_sufficiency",
)

COST_FIELDS = (
    "attribution_model_forward_calls",
    "attribution_model_forward_calls_per_sample",
)

EXPERIMENT_FIELDS = (
    "budget",
    "order",
    "seed",
    "value_function",
    "chunker",
    "eval_granularity",
    "basis",
    "hierarchy",
    "sampler",
    "projector",
)

OUTCOME_FIELDS = (
    "sample_count",
    "failed_count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the result-collection command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Collect <dataset>/<model>/<method>/<config>/metrics.json "
            "artifacts into one CSV."
        )
    )
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        "--i",
        required=True,
        dest="input_dir",
        help="Method result root, for example results/mobius.",
    )
    parser.add_argument(
        "-o",
        "--o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def _load_json(path: Path, *, required: bool) -> Dict[str, Any]:
    """Load one JSON object, optionally tolerating a missing artifact."""

    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required result artifact is missing: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _path_identity(input_dir: Path, metrics_path: Path) -> Dict[str, str]:
    """Parse the four experiment identity components from a metrics path."""

    relative = metrics_path.relative_to(input_dir)
    if len(relative.parts) != 5 or relative.name != "metrics.json":
        raise ValueError(
            "Expected metrics path "
            "<dataset>/<model>/<method>/<config>/metrics.json, got "
            f"{relative}."
        )
    dataset, model, method, config, _ = relative.parts
    return {
        "dataset": dataset,
        "model": model,
        "method": method,
        "config": config,
    }


def _scalar(value: Any) -> Any:
    """Return CSV-friendly scalars and stable JSON for structured values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _named_component(config: Mapping[str, Any], key: str) -> Any:
    """Extract a component name from either a scalar or mapping config."""

    value = config.get(key)
    if isinstance(value, Mapping):
        return value.get("name")
    return value


def _scientific_fields(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract comparison-relevant scientific axes from run.json."""

    config = run.get("scientific_config", {})
    if not isinstance(config, Mapping):
        return {}
    return {
        "budget": _scalar(config.get("budget")),
        "order": _scalar(
            config.get("max_degree", config.get("max_order"))
        ),
        "seed": _scalar(config.get("seed")),
        "value_function": _scalar(config.get("value_function")),
        "chunker": _scalar(config.get("chunker")),
        "eval_granularity": _scalar(config.get("eval_granularity")),
        "basis": _scalar(_named_component(config, "basis")),
        "hierarchy": _scalar(_named_component(config, "hierarchy")),
        "sampler": _scalar(_named_component(config, "sampler")),
        "projector": _scalar(_named_component(config, "projector")),
    }


def _faithfulness_fields(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only mean and standard deviation for every faithfulness metric."""

    output: Dict[str, Any] = {}
    faithfulness = metrics.get("faithfulness", {})
    if isinstance(faithfulness, Mapping):
        for metric_name, aggregate in faithfulness.items():
            if not isinstance(aggregate, Mapping):
                continue
            for statistic in ("mean", "std"):
                output[f"faithfulness_{metric_name}_{statistic}"] = _scalar(
                    aggregate.get(statistic)
                )
    return output


def _numeric(mapping: Mapping[str, Any], key: str) -> float | None:
    """Read a finite numeric cost while rejecting booleans and missing values."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _per_sample(value: float | None, sample_count: int) -> float | None:
    """Normalize one aggregate cost by the number of evaluated samples."""

    if value is None or sample_count <= 0:
        return None
    return float(value / sample_count)


def _attribution_call_fields(
    metrics: Mapping[str, Any],
    evaluated_count: int,
) -> Dict[str, Any]:
    """Keep only total and per-sample attribution model forward calls."""

    attribution = metrics.get("attribution_cost", {})
    attribution = attribution if isinstance(attribution, Mapping) else {}
    attribution_forwards = _numeric(attribution, "model_forward_calls")
    return {
        "attribution_model_forward_calls": attribution_forwards,
        "attribution_model_forward_calls_per_sample": _per_sample(
            attribution_forwards,
            evaluated_count,
        ),
    }


def collect_row(input_dir: Path, metrics_path: Path) -> Dict[str, Any]:
    """Collect one run directory into a flat analysis row."""

    identity = _path_identity(input_dir, metrics_path)
    run_dir = metrics_path.parent
    metrics = _load_json(metrics_path, required=True)
    run = _load_json(run_dir / "run.json", required=False)
    evaluated_count = int(metrics.get("evaluated_count", 0) or 0)
    row: Dict[str, Any] = {
        **identity,
        "target": _scalar(metrics.get("target")),
        **_faithfulness_fields(metrics),
        **_attribution_call_fields(metrics, evaluated_count),
        **_scientific_fields(run),
        "sample_count": _scalar(metrics.get("sample_count")),
        "failed_count": _scalar(metrics.get("failed_count")),
    }
    return row


def collect_results(input_dir: str | Path) -> list[Dict[str, Any]]:
    """Collect all conforming metrics files below one method result root."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Input result directory does not exist: {root}")
    metrics_paths = sorted(root.glob("*/*/*/*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(
            "No metrics.json files matched "
            "<dataset>/<model>/<method>/<config>/metrics.json below "
            f"{root}."
        )
    return [collect_row(root, path) for path in metrics_paths]


def _ordered_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Order the compact paper-comparison columns requested by the user."""

    available = {key for row in rows for key in row}
    preferred = [*IDENTITY_FIELDS]
    metric_names = {
        key[len("faithfulness_") :].rsplit("_", 1)[0]
        for key in available
        if key.startswith("faithfulness_")
    }
    ordered_metric_names = [
        name for name in FAITHFULNESS_PRIORITY if name in metric_names
    ]
    ordered_metric_names.extend(sorted(metric_names - set(ordered_metric_names)))
    for metric_name in ordered_metric_names:
        preferred.extend(
            (
                f"faithfulness_{metric_name}_mean",
                f"faithfulness_{metric_name}_std",
            )
        )
    preferred.extend(COST_FIELDS)
    preferred.extend(EXPERIMENT_FIELDS)
    preferred.extend(OUTCOME_FIELDS)
    ordered = [key for key in preferred if key in available]
    unexpected = available - set(ordered)
    if unexpected:
        raise ValueError(f"Unexpected result columns: {sorted(unexpected)}")
    return ordered


def write_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write collected rows to a UTF-8 CSV with deterministic columns."""

    if not rows:
        raise ValueError("Cannot write an empty result summary.")
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = _ordered_fields(rows)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main(argv: Iterable[str] | None = None) -> None:
    """Collect result artifacts and report the generated CSV location."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    rows = collect_results(args.input_dir)

    dest = os.path.join(args.input_dir, args.o)
    destination = write_csv(rows, dest)
    print(f"[collected] runs={len(rows)} output={destination}")


if __name__ == "__main__":
    main()
