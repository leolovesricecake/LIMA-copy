from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.utils import atomic_write_json, environment_snapshot, load_yaml, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gated medium-n scalability experiments.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "medium_default.yaml"))
    parser.add_argument("--force", action="store_true", help="Allow running once medium-n is implemented.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    out_dir = resolve_project_path(
        config.get("results_dir"),
        project_root=ROOT,
        repo_root=REPO_ROOT,
        default=ROOT / "results" / "medium_default",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "run_config.yaml.json", config)
    atomic_write_json(out_dir / "environment.json", environment_snapshot())
    if not bool(config.get("enabled", False)) and not args.force:
        atomic_write_json(
            out_dir / "status.json",
            {
                "status": "skipped",
                "reason": "medium_n_is_gated_until_stages_a_b_c_support_the_hypothesis",
            },
        )
        return
    raise NotImplementedError(
        "Medium-n online recovery is gated until exact/probe and limited-query results justify it."
    )


if __name__ == "__main__":
    main()
