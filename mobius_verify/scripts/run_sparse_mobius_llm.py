from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lima_llm.attribution_values import normalize_attribution_value_function
from lima_llm.backbone.hf_backbone import HFBackbone
from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.chunking.explanation import load_adaptive_overrides
from lima_llm.data import DatasetBundle, load_dataset_bundle
from lima_llm.types import TextSample
from lima_llm.utils import configure_determinism, set_seed

from mobius_verify.src.datasets.sentiment import load_sentiment_records, verbalizers_for_dataset
from mobius_verify.src.sparse_runner import run_sparse_mobius
from mobius_verify.src.utils import (
    load_yaml,
    patch_multiprocess_resource_tracker_shutdown,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone Sparse deletion-Mobius attribution.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "sparse_mobius_default.yaml"),
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--value-function",
        choices=[
            "target_probability",
            "predicted_probability",
            "predicted_class_margin",
            "raw_target_score",
        ],
        default=None,
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser


def _apply_overrides(config: dict, args) -> dict:
    output = dict(config)
    output["model"] = dict(output.get("model", {}))
    output["dataset"] = dict(output.get("dataset", {}))
    for key in ("budget", "seed", "value_function"):
        value = getattr(args, key)
        if value is not None:
            output[key] = value
    if args.device is not None:
        output["model"]["device"] = str(args.device)
    if args.max_samples is not None:
        output["dataset"]["max_samples"] = int(args.max_samples)
    output["value_function"] = normalize_attribution_value_function(
        output.get("value_function", "predicted_class_margin")
    )
    output["adaptive_overrides"] = load_adaptive_overrides(
        output.get("adaptive_overrides_json")
    )
    output["max_length"] = int(
        output.get("max_length", output["model"].get("max_length", 2048))
    )
    return output


def _load_bundle(dataset_config: dict) -> DatasetBundle:
    if dataset_config.get("samples") is not None or str(
        dataset_config.get("source", "")
    ).lower() in {"inline", "inline_sentiment"}:
        records = load_sentiment_records(dataset_config)
        name = str(dataset_config.get("name", dataset_config.get("task", "inline_sentiment")))
        split = str(dataset_config.get("split", "validation"))
        verbalizers = verbalizers_for_dataset(name, dataset_config)
        return DatasetBundle(
            dataset_name=name,
            split=split,
            samples=[
                TextSample(
                    sample_id=record.sample_id,
                    text=record.text,
                    label=int(record.label or 0),
                    label_text=record.label_text,
                    metadata=dict(record.metadata or {}),
                )
                for record in records
            ],
            label_names=list(verbalizers),
            verbalizers=list(verbalizers),
        )
    return load_dataset_bundle(
        dataset_name=str(dataset_config["name"]),
        split=str(dataset_config.get("split", "validation")),
        max_samples=dataset_config.get("max_samples"),
        eraser_root=dataset_config.get("eraser_root"),
        sst2_source=dataset_config.get("sst2_source"),
        dataset_cache_dir=dataset_config.get("dataset_cache_dir"),
    )


def _build_backbone(config: dict):
    model = dict(config.get("model", {}))
    model_type = str(model.get("type", "hf_causal_lm")).strip().lower()
    if model_type in {"mock", "mock_sentiment", "mock_backbone"}:
        backbone = MockBackbone()
        backbone.tokenizer = None
        backbone.max_length = int(config.get("max_length", 2048))
        return backbone
    return HFBackbone(
        model_path=str(model["model_path"]),
        device=str(model.get("device", "cuda:0")),
        dtype=str(model.get("dtype", "bfloat16")),
        max_length=int(config.get("max_length", model.get("max_length", 2048))),
        embedding_layer_ratio=0.7,
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )


def _slug(value: object) -> str:
    return str(value).rstrip("/").split("/")[-1].replace(".", "_")


def _default_output_root(config: dict, bundle: DatasetBundle) -> Path:
    model = dict(config.get("model", {}))
    base = resolve_project_path(
        config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "sparse_mobius",
    )
    model_name = _slug(model.get("model_path", model.get("type", "mock")))
    value_function = normalize_attribution_value_function(config.get("value_function"))
    target_mode = (
        "predicted"
        if value_function in {"predicted_probability", "predicted_class_margin"}
        else str(config.get("target_mode", "predicted"))
    )
    adaptive_segment = (
        f"profile-{config.get('adaptive_profile', 'balanced')}_"
        if str(config.get("chunker", "word")) == "adaptive"
        else ""
    )
    if str(config.get("chunker", "word")) == "adaptive" and config.get(
        "adaptive_overrides"
    ):
        encoded = json.dumps(
            config["adaptive_overrides"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        adaptive_segment += f"overrides-{hashlib.sha256(encoded).hexdigest()[:8]}_"
    method_slug = (
        f"chunk-{config.get('chunker', 'word')}_{adaptive_segment}"
        f"eval-{config.get('eval_granularity', 'token')}_"
        f"basis-deletion_mobius_order-{int(config.get('max_degree', 2))}_"
        f"budget-{int(config.get('budget', 512))}_value-{value_function}_"
        f"target-{target_mode}_k-{int(config.get('k', 8))}_seed-{int(config.get('seed', 42))}"
    )
    return base / str(bundle.dataset_name) / f"model-{model_name}" / method_slug


def main(argv=None) -> None:
    patch_multiprocess_resource_tracker_shutdown()
    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_yaml(args.config), args)
    set_seed(int(config.get("seed", 42)))
    configure_determinism(bool(config.get("deterministic", False)))
    bundle = _load_bundle(dict(config.get("dataset", {})))
    backbone = _build_backbone(config)
    output_root = (
        resolve_project_path(
            args.output_root,
            project_root=ROOT,
            repo_root=REPO_ROOT,
            default=ROOT / "results" / "sparse_mobius",
        )
        if args.output_root
        else _default_output_root(config, bundle)
    )
    manifest = run_sparse_mobius(
        config=config,
        bundle=bundle,
        backbone=backbone,
        output_root=output_root,
        overwrite=bool(args.overwrite),
        evaluate=not bool(args.no_eval),
    )
    print(
        f"[complete] method=sparse_mobius completed={manifest['completed_sample_count']} "
        f"failed={manifest['failed_sample_count']} skipped={manifest['skipped_sample_count']} "
        f"results={output_root}"
    )


if __name__ == "__main__":
    main()
