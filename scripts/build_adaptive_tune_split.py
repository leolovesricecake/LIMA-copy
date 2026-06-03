#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lima_llm.data.loader import DatasetBundle, load_dataset_bundle


def _label_distribution(samples: Sequence) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for sample in samples:
        key = int(getattr(sample, "label", -1))
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def _stratified_split(
    *,
    samples: Sequence,
    train_size: int,
    dev_size: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(int(seed))
    by_label: Dict[int, List] = {}
    for sample in samples:
        key = int(getattr(sample, "label", -1))
        by_label.setdefault(key, []).append(sample)

    for rows in by_label.values():
        rng.shuffle(rows)

    total = int(len(samples))
    if total == 0:
        return [], []
    target = min(total, max(0, int(train_size) + int(dev_size)))
    if target <= 0:
        return [], []
    train_target = min(int(train_size), target)
    dev_target = max(0, target - train_target)

    labels = sorted(by_label.keys())
    counts = {k: len(by_label[k]) for k in labels}
    quotas = {k: int(round(target * float(counts[k]) / float(total))) for k in labels}
    q_sum = sum(quotas.values())
    while q_sum > target:
        k = max(labels, key=lambda x: (quotas[x], counts[x]))
        if quotas[k] > 0:
            quotas[k] -= 1
            q_sum -= 1
        else:
            break
    while q_sum < target:
        k = max(labels, key=lambda x: (counts[x] - quotas[x], counts[x]))
        if quotas[k] < counts[k]:
            quotas[k] += 1
            q_sum += 1
        else:
            break

    picked: Dict[int, List] = {}
    for k in labels:
        picked[k] = by_label[k][: min(quotas[k], len(by_label[k]))]

    train_ids: List[str] = []
    dev_ids: List[str] = []
    for k in labels:
        rows = picked[k]
        if not rows:
            continue
        k_train = int(round(train_target * float(len(rows)) / float(max(1, target))))
        k_train = max(0, min(k_train, len(rows)))
        train_ids.extend(str(s.sample_id) for s in rows[:k_train])
        dev_ids.extend(str(s.sample_id) for s in rows[k_train:])

    rng.shuffle(train_ids)
    rng.shuffle(dev_ids)
    train_ids = train_ids[:train_target]
    dev_ids = dev_ids[:dev_target]
    return train_ids, dev_ids


def _load_bundle_with_fallback(
    *,
    dataset: str,
    split: str,
    prefer_train: bool,
    eraser_root: str | None,
    sst2_source: str | None,
    dataset_cache_dir: str | None,
) -> Tuple[DatasetBundle, str]:
    if prefer_train:
        try:
            bundle = load_dataset_bundle(
                dataset_name=dataset,
                split="train",
                max_samples=None,
                eraser_root=eraser_root,
                sst2_source=sst2_source,
                dataset_cache_dir=dataset_cache_dir,
            )
            if len(bundle.samples) > 0:
                return bundle, "train"
        except Exception:
            pass

    bundle = load_dataset_bundle(
        dataset_name=dataset,
        split=split,
        max_samples=None,
        eraser_root=eraser_root,
        sst2_source=sst2_source,
        dataset_cache_dir=dataset_cache_dir,
    )
    return bundle, str(split)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train/dev sample-id splits for adaptive tuning")
    parser.add_argument("--dataset", required=True, choices=["sst2", "eraser_movie_reviews", "imdb", "rotten_tomatoes", "emotion"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--prefer-train", dest="prefer_train", action="store_true", default=True)
    parser.add_argument("--no-prefer-train", dest="prefer_train", action="store_false")
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--dev-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-root", type=str, default="adaptive_tune_splits")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle, source_split = _load_bundle_with_fallback(
        dataset=args.dataset,
        split=args.split,
        prefer_train=bool(args.prefer_train),
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    train_ids, dev_ids = _stratified_split(
        samples=bundle.samples,
        train_size=int(args.train_size),
        dev_size=int(args.dev_size),
        seed=int(args.seed),
    )

    out_dir = Path(args.save_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_ids.json"
    dev_path = out_dir / "dev_ids.json"
    manifest_path = out_dir / "split_manifest.json"

    train_payload = {
        "dataset": args.dataset,
        "source_split": source_split,
        "seed": int(args.seed),
        "sample_ids": train_ids,
    }
    dev_payload = {
        "dataset": args.dataset,
        "source_split": source_split,
        "seed": int(args.seed),
        "sample_ids": dev_ids,
    }
    train_id_set = set(train_ids)
    dev_id_set = set(dev_ids)
    manifest_payload = {
        "dataset": args.dataset,
        "requested_split": args.split,
        "source_split": source_split,
        "seed": int(args.seed),
        "train_size_requested": int(args.train_size),
        "dev_size_requested": int(args.dev_size),
        "train_size_actual": int(len(train_ids)),
        "dev_size_actual": int(len(dev_ids)),
        "total_source_samples": int(len(bundle.samples)),
        "source_label_distribution": _label_distribution(bundle.samples),
        "train_label_distribution": _label_distribution(
            [s for s in bundle.samples if str(s.sample_id) in train_id_set]
        ),
        "dev_label_distribution": _label_distribution(
            [s for s in bundle.samples if str(s.sample_id) in dev_id_set]
        ),
        "train_ids_file": str(train_path),
        "dev_ids_file": str(dev_path),
    }

    train_path.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dev_path.write_text(json.dumps(dev_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[adaptive-tune-split] dataset={args.dataset} source_split={source_split}")
    print(f"[adaptive-tune-split] train={len(train_ids)} -> {train_path}")
    print(f"[adaptive-tune-split] dev={len(dev_ids)} -> {dev_path}")
    print(f"[adaptive-tune-split] manifest={manifest_path}")


if __name__ == "__main__":
    main()
