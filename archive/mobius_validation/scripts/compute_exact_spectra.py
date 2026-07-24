from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.exact_analysis import analyze_value_table
from mobius_verify.src.transforms import fourier_transform, mobius_transform
from mobius_verify.src.utils import atomic_save_npy, atomic_write_json, read_json, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute exact Mobius/Fourier spectra for saved value tables.")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--scope", type=str, default="all", choices=["all", "exact_global", "exact_probe"])
    parser.add_argument("--d-max", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _iter_tables(results_dir: Path, scope: str):
    if scope in {"all", "exact_global"}:
        for value_file in sorted((results_dir / "values_exact_global").glob("*/*/values.npy")):
            yield "exact_global", value_file
    if scope in {"all", "exact_probe"}:
        for value_file in sorted((results_dir / "values_exact_probe").glob("*/*/*/values.npy")):
            yield "exact_probe", value_file


def _target_dir(results_dir: Path, scope: str, value_file: Path) -> Path:
    if scope == "exact_global":
        task = value_file.parents[1].name
        sample_id = value_file.parent.name
        return results_dir / "spectra_exact_global" / task / sample_id
    task = value_file.parents[2].name
    sample_id = value_file.parents[1].name
    probe_id = value_file.parent.name
    return results_dir / "spectra_exact_probe" / task / sample_id / probe_id


def main() -> None:
    args = build_parser().parse_args()
    results_dir = resolve_project_path(
        args.results_dir,
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "exact_default",
    )
    for scope, value_file in _iter_tables(results_dir, args.scope):
        target = _target_dir(results_dir, scope, value_file)
        if (target / "analysis.json").exists() and not args.overwrite:
            continue
        values = np.load(value_file).astype(np.float64)
        metadata = read_json(value_file.parent / "metadata.json")
        analysis = analyze_value_table(values, d_max=int(args.d_max))
        target.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(target / "mobius.npy", mobius_transform(values))
        atomic_save_npy(target / "fourier.npy", fourier_transform(values))
        atomic_write_json(target / "analysis.json", {**analysis, "metadata": metadata})


if __name__ == "__main__":
    main()
