from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.benchmark import run_benchmark
from mobius_verify.src.datasets.sentiment import load_sentiment_records, verbalizers_for_dataset
from mobius_verify.src.utils import (
    load_yaml,
    patch_multiprocess_resource_tracker_shutdown,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sparse Mobius attribution benchmarks.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "attribution_mvp.yaml"),
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    patch_multiprocess_resource_tracker_shutdown()
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    results_dir = resolve_project_path(
        args.results_dir or config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "attribution_mvp",
    )
    dataset_cfg = dict(config.get("dataset", {}))
    dataset_name = str(dataset_cfg.get("name", dataset_cfg.get("source", "inline_sentiment")))
    records = load_sentiment_records(dataset_cfg)
    max_records = config.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]
    verbalizers = verbalizers_for_dataset(dataset_name, dataset_cfg)
    manifest = run_benchmark(
        config=config,
        records=records,
        verbalizers=verbalizers,
        results_dir=results_dir,
        overwrite=bool(args.overwrite),
    )
    print(
        f"[complete] records={manifest['completed_record_count']} "
        f"cache_entries={manifest['cache_entry_count']} results={results_dir}"
    )


if __name__ == "__main__":
    main()
