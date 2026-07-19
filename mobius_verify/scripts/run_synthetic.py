from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.exact_analysis import analyze_value_table
from mobius_verify.src.subset_enumeration import all_masks, masks_array
from mobius_verify.src.synthetic import generate_synthetic_suite
from mobius_verify.src.utils import (
    atomic_save_npy,
    atomic_write_json,
    environment_snapshot,
    load_yaml,
    patch_multiprocess_resource_tracker_shutdown,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic Mobius/Fourier sanity checks.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "synthetic_default.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    patch_multiprocess_resource_tracker_shutdown()
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    out_dir = resolve_project_path(
        config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "synthetic_default",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "run_config.yaml.json", config)
    atomic_write_json(out_dir / "environment.json", environment_snapshot())

    instances = generate_synthetic_suite(
        n_features=int(config.get("n_features", 12)),
        instances_per_family=int(config.get("instances_per_family", 50)),
        seed=int(config.get("seed", 0)),
        noise_std=float(config.get("noise_std", 0.0)),
    )
    d_max = int(config.get("d_max", 4))
    max_instances = config.get("max_instances")
    if max_instances is not None:
        instances = instances[: int(max_instances)]

    summaries = []
    for instance in instances:
        target = out_dir / "synthetic" / instance.family / instance.instance_id
        done = target / "analysis.json"
        if done.exists() and not args.overwrite:
            continue
        target.mkdir(parents=True, exist_ok=True)
        masks = all_masks(instance.n_features)
        atomic_save_npy(target / "masks.npy", masks_array(masks))
        atomic_save_npy(target / "values.npy", instance.values)
        analysis = analyze_value_table(instance.values, d_max=d_max)
        metadata = {
            "experiment_scope": "synthetic",
            "instance_id": instance.instance_id,
            "family": instance.family,
            "n_features": int(instance.n_features),
            "true_basis": instance.true_basis,
            "true_support": [int(x) for x in instance.true_support],
            **instance.metadata,
        }
        atomic_write_json(target / "metadata.json", metadata)
        atomic_write_json(target / "analysis.json", analysis)
        summaries.append(
            {
                "instance_id": instance.instance_id,
                "family": instance.family,
                "mobius_d90": analysis["degree_summary"]["mobius_d90"],
                "fourier_d90": analysis["degree_summary"]["fourier_d90"],
                "mobius_k90": analysis["sparsity_summary"]["mobius_omp"]["x90"],
                "fourier_k90": analysis["sparsity_summary"]["fourier_omp"]["x90"],
            }
        )
    atomic_write_json(out_dir / "synthetic_summary.json", {"rows": summaries})


if __name__ == "__main__":
    main()
