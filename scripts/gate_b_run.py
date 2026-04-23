#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from typing import List


def _parse_int_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_lambdas_list(raw: str) -> List[str]:
    # Use ';' between lambda tuples because lambdas itself uses commas.
    return [x.strip() for x in raw.split(";") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate B multi-seed/grid experiments")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--k-values", type=str, default="8")
    parser.add_argument("--chunkers", type=str, default="sentence")
    parser.add_argument("--searches", type=str, default="greedy")
    parser.add_argument("--lambdas-list", type=str, default="1,1,1,1")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    seeds = _parse_int_csv(args.seeds)
    k_values = _parse_int_csv(args.k_values)
    chunkers = _parse_str_csv(args.chunkers)
    searches = _parse_str_csv(args.searches)
    lambdas_values = _parse_lambdas_list(args.lambdas_list)
    base_args = list(extra or [])
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]

    if not seeds or not k_values or not chunkers or not searches or not lambdas_values:
        raise ValueError("seeds/k-values/chunkers/searches/lambdas-list must not be empty")

    jobs = list(itertools.product(seeds, k_values, chunkers, searches, lambdas_values))
    print(f"[gate-b-run] jobs={len(jobs)}")

    for i, (seed, k, chunker, search, lambdas) in enumerate(jobs, start=1):
        cmd = [
            args.python_exe,
            "-m",
            "lima_llm",
            *base_args,
            "--seed",
            str(seed),
            "--k",
            str(k),
            "--chunker",
            chunker,
            "--search",
            search,
            "--lambdas",
            lambdas,
        ]
        print(f"[gate-b-run] ({i}/{len(jobs)}) seed={seed} k={k} chunker={chunker} search={search} lam={lambdas}")
        print("[gate-b-run] cmd:", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
