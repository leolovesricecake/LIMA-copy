#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename '*.jpg.npy' files to '*.npy' recursively under explain_dir."
    )
    parser.add_argument(
        "explain_dir_pos",
        nargs="?",
        default=None,
        help="Root directory to scan recursively (positional form).",
    )
    parser.add_argument(
        "--explain-dir",
        type=str,
        default=None,
        help="Root directory to scan recursively (flag form).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwrite if target file already exists.",
    )
    return parser.parse_args()


def collect_renames(explain_dir: str) -> List[Tuple[str, str]]:
    renames: List[Tuple[str, str]] = []
    for cur_root, _, files in os.walk(explain_dir):
        for name in files:
            if not name.endswith(".jpg.npy"):
                continue
            src = os.path.join(cur_root, name)
            dst = os.path.join(cur_root, name[: -len(".jpg.npy")] + ".npy")
            renames.append((src, dst))
    return renames


def main():
    args = parse_args()
    explain_dir = args.explain_dir if args.explain_dir else args.explain_dir_pos
    if explain_dir is None or explain_dir.strip() == "":
        raise ValueError("Please provide explain_dir via positional arg or --explain-dir.")
    if not os.path.isdir(explain_dir):
        raise ValueError("Invalid --explain-dir '{}': not a directory.".format(explain_dir))

    plans = collect_renames(explain_dir)
    if len(plans) == 0:
        print("[done] no '*.jpg.npy' files found under '{}'.".format(explain_dir))
        return

    renamed = 0
    skipped_exists = 0
    for src, dst in plans:
        if os.path.exists(dst) and not args.overwrite:
            skipped_exists += 1
            print("[skip-exists] {} -> {}".format(src, dst))
            continue

        if args.dry_run:
            print("[dry-run] {} -> {}".format(src, dst))
            renamed += 1
            continue

        if args.overwrite and os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)
        renamed += 1
        print("[rename] {} -> {}".format(src, dst))

    print(
        "[done] total={} renamed={} skipped_exists={} dry_run={}.".format(
            len(plans), renamed, skipped_exists, args.dry_run
        )
    )


if __name__ == "__main__":
    main()
