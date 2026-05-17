#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

COMPONENTS: Tuple[str, ...] = (
    "confidence",
    "effectiveness",
    "consistency",
    "collaboration",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / float(n - 1)
    return float(math.sqrt(max(0.0, var)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) <= 1:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def _rankdata(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    pairs = sorted((float(v), idx) for idx, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = float(avg_rank)
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) <= 1:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)


def _pair_rows(view: str, data: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, left in enumerate(COMPONENTS):
        for right in COMPONENTS[i + 1 :]:
            xs = data.get(left, [])
            ys = data.get(right, [])
            n = min(len(xs), len(ys))
            xs_cut = xs[:n]
            ys_cut = ys[:n]
            rows.append(
                {
                    "view": view,
                    "left": left,
                    "right": right,
                    "n": int(n),
                    "pearson": _pearson(xs_cut, ys_cut),
                    "spearman": _spearman(xs_cut, ys_cut),
                }
            )
    return rows


def _component_stats(data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in COMPONENTS:
        values = data.get(name, [])
        out[name] = {
            "count": float(len(values)),
            "mean": _mean(values),
            "std": _std(values),
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }
    return out


def _load_component_rows(run_dir: Path) -> Tuple[int, Dict[str, List[float]], Dict[str, List[float]]]:
    sample_dir = run_dir / "samples"
    chunk_level: Dict[str, List[float]] = {name: [] for name in COMPONENTS}
    sample_level: Dict[str, List[float]] = {name: [] for name in COMPONENTS}
    sample_count = 0

    if not sample_dir.exists():
        return 0, chunk_level, sample_level

    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        sample_count += 1

        profile = payload.get("metadata", {}).get("component_profile", {})
        singleton = profile.get("singleton_components", {}) if isinstance(profile, dict) else {}
        if isinstance(singleton, dict):
            for comp_row in singleton.values():
                if not isinstance(comp_row, dict):
                    continue
                for name in COMPONENTS:
                    chunk_level[name].append(_safe_float(comp_row.get(name), 0.0))

        scores = payload.get("scores", {})
        if isinstance(scores, dict):
            for name in COMPONENTS:
                sample_level[name].append(_safe_float(scores.get(name), 0.0))

    return sample_count, chunk_level, sample_level


def build_report(run_dir: Path) -> Dict[str, Any]:
    sample_count, chunk_level, sample_level = _load_component_rows(run_dir)
    rows = _pair_rows("chunk_singleton", chunk_level) + _pair_rows("sample_selected_set", sample_level)

    return {
        "run_dir": str(run_dir),
        "sample_count": int(sample_count),
        "components": list(COMPONENTS),
        "chunk_row_count": int(len(chunk_level["confidence"])),
        "sample_row_count": int(len(sample_level["confidence"])),
        "views": {
            "chunk_singleton": {
                "component_stats": _component_stats(chunk_level),
                "pairs": [row for row in rows if row["view"] == "chunk_singleton"],
            },
            "sample_selected_set": {
                "component_stats": _component_stats(sample_level),
                "pairs": [row for row in rows if row["view"] == "sample_selected_set"],
            },
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile four objective components from a full-method run and compute Pearson/Spearman "
            "correlations at chunk-level(singleton) and sample-level(selected-set)."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = ["view", "left", "right", "n", "pearson", "spearman"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)

    out_json = Path(args.output_json) if args.output_json else (run_dir / "component_correlation_full.json")
    out_csv = Path(args.output_csv) if args.output_csv else (run_dir / "component_correlation_full.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, report.get("rows", []))

    print(f"[component-correlation] samples={report.get('sample_count', 0)}")
    print(f"[component-correlation] chunk_rows={report.get('chunk_row_count', 0)}")
    print(f"[component-correlation] sample_rows={report.get('sample_row_count', 0)}")
    print(f"[component-correlation] json={out_json}")
    print(f"[component-correlation] csv={out_csv}")


if __name__ == "__main__":
    main()
