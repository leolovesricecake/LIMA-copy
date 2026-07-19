from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.datasets.sentiment import load_sentiment_records, verbalizers_for_dataset
from mobius_verify.src.featureization import build_lexical_word_features, validate_feature_reconstruction
from mobius_verify.src.models import build_text_scorer
from mobius_verify.src.probes import build_probes
from mobius_verify.src.subset_enumeration import all_masks, masks_array
from mobius_verify.src.utils import (
    atomic_save_npy,
    atomic_write_json,
    environment_snapshot,
    load_yaml,
    resolve_project_path,
)
from mobius_verify.src.value_functions import PredictedClassMarginValueFunction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect exact masked value tables for word features.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "exact_default.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _length_bin(n_features: int, bins: List[List[int]]) -> str | None:
    for low, high in bins:
        if int(low) <= int(n_features) <= int(high):
            return f"{int(low)}-{int(high)}"
    return None


def _save_feature_spec(out_dir: Path, task: str, spec) -> None:
    target = out_dir / "features" / task / f"{spec.sample_id}.json"
    atomic_write_json(target, spec.to_dict())


def _collect_table(
    *,
    target_dir: Path,
    feature_spec,
    value_fn: PredictedClassMarginValueFunction,
    target_class: int,
    masks: List[int],
    operator: str,
    active_feature_ids=None,
    conditioning_mode: str = "global",
    metadata: Dict[str, Any],
    batch_size: int,
    overwrite: bool,
) -> None:
    if (target_dir / "values.npy").exists() and (target_dir / "metadata.json").exists() and not overwrite:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    values = value_fn.evaluate_masks(
        feature_spec,
        masks,
        target_class=target_class,
        operator=operator,
        active_feature_ids=active_feature_ids,
        conditioning_mode=conditioning_mode,
        batch_size=batch_size,
    )
    atomic_save_npy(target_dir / "masks.npy", masks_array(masks))
    atomic_save_npy(target_dir / "values.npy", values.astype("float32"))
    atomic_write_json(target_dir / "metadata.json", metadata)


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    out_dir = resolve_project_path(
        config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "exact_default",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "run_config.yaml.json", config)
    atomic_write_json(out_dir / "environment.json", environment_snapshot())

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_name = str(dataset_cfg.get("name", dataset_cfg.get("source", "inline_sentiment")))
    verbalizers = verbalizers_for_dataset(dataset_name, dataset_cfg)
    scorer = build_text_scorer(dict(config.get("model", {"type": "mock_sentiment"})), verbalizers=verbalizers)
    tokenizer = getattr(getattr(scorer, "backbone", None), "tokenizer", None)
    value_cfg = dict(config.get("value_function", {}))
    value_fn = PredictedClassMarginValueFunction(
        scorer,
        target_class_source=str(value_cfg.get("target_class_source", "full_input_prediction")),
        verbalizer_length_normalization=str(value_cfg.get("verbalizer_length_normalization", "mean")),
    )
    records = load_sentiment_records(dataset_cfg)
    max_records = config.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]

    operator = str(config.get("mask_operator", "delete"))
    batch_size = int(config.get("batch_size", 32))
    exclusions = []

    exact_global = dict(config.get("exact_global", {}))
    min_n = int(exact_global.get("min_n", 6))
    max_n = int(exact_global.get("max_n", 14))
    bins = exact_global.get("length_bins", [[6, 8], [9, 11], [12, 14]])
    exact_probe = dict(config.get("exact_probe", {}))

    for record in records:
        spec = build_lexical_word_features(record.text, sample_id=record.sample_id, tokenizer=tokenizer)
        ok, reason = validate_feature_reconstruction(spec)
        if not ok:
            exclusions.append({"sample_id": record.sample_id, "reason": reason})
            continue
        _save_feature_spec(out_dir, record.task, spec)
        if spec.n_features == 0:
            exclusions.append({"sample_id": record.sample_id, "reason": "no_word_features"})
            continue

        vf_meta = value_fn.metadata_for_feature_spec(spec, gold_label=record.label)
        common_meta = {
            **vf_meta.to_dict(),
            "sample": record.to_dict(),
            "n_total_words": int(spec.n_features),
            "mask_operator": operator,
            "model_counters": scorer.snapshot_counters(),
        }

        if bool(exact_global.get("enabled", True)):
            if min_n <= spec.n_features <= max_n:
                masks = all_masks(spec.n_features)
                table_dir = out_dir / "values_exact_global" / record.task / record.sample_id
                metadata = {
                    **common_meta,
                    "experiment_scope": "exact_global",
                    "n_features": int(spec.n_features),
                    "length_bin": _length_bin(spec.n_features, bins),
                    "probe_id": None,
                    "probe_size": None,
                    "probe_indices": None,
                    "probe_strategy": None,
                    "conditioning_mode": "global",
                }
                _collect_table(
                    target_dir=table_dir,
                    feature_spec=spec,
                    value_fn=value_fn,
                    target_class=vf_meta.target_class,
                    masks=masks,
                    operator=operator,
                    metadata=metadata,
                    batch_size=batch_size,
                    overwrite=args.overwrite,
                )
            else:
                exclusions.append(
                    {
                        "sample_id": record.sample_id,
                        "reason": "outside_exact_global_length_range",
                        "n_features": int(spec.n_features),
                    }
                )

        if bool(exact_probe.get("enabled", True)) and spec.n_features >= min(exact_probe.get("probe_sizes", [8])):
            probes = build_probes(
                spec,
                probe_sizes=[int(x) for x in exact_probe.get("probe_sizes", [8, 10, 12])],
                strategies=[str(x) for x in exact_probe.get("strategies", ["contiguous_random"])],
                probes_per_sample=int(exact_probe.get("probes_per_sample", 1)),
                seed=int(config.get("seed", 0)),
                conditioning_mode=str(exact_probe.get("conditioning_mode", "rest_present")),
            )
            for probe in probes:
                masks = all_masks(probe.k)
                table_dir = out_dir / "values_exact_probe" / record.task / record.sample_id / probe.probe_id
                metadata = {
                    **common_meta,
                    "experiment_scope": "exact_probe",
                    "n_features": int(probe.k),
                    "n_total_words": int(spec.n_features),
                    "probe_id": probe.probe_id,
                    "probe_size": int(probe.k),
                    "probe_indices": [int(x) for x in probe.probe_word_indices],
                    "probe_strategy": probe.probe_strategy,
                    "conditioning_mode": probe.conditioning_mode,
                    "random_seed": int(probe.random_seed),
                }
                _collect_table(
                    target_dir=table_dir,
                    feature_spec=spec,
                    value_fn=value_fn,
                    target_class=vf_meta.target_class,
                    masks=masks,
                    operator=operator,
                    active_feature_ids=probe.probe_word_indices,
                    conditioning_mode=probe.conditioning_mode,
                    metadata=metadata,
                    batch_size=batch_size,
                    overwrite=args.overwrite,
                )

    atomic_write_json(out_dir / "manifests" / "exclusions.json", {"rows": exclusions})


if __name__ == "__main__":
    main()
