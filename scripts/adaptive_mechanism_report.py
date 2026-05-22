#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_ORPHAN_PUNCT_RE = re.compile(r'^[\s"\'`“”‘’\(\)\[\]\{\}\.,!?;:，。！？；：、…\-–—]+$')
_LEADING_CLOSE_PUNCT_RE = re.compile(r"^\s*[\)\]\}]+")
_ABBREVIATION_SINGLETON_RE = re.compile(r"^(mr|mrs|ms|dr|prof|st|jr|sr)\s*\.\s*$", flags=re.IGNORECASE)

_DIRECTION = {
    "log_odds": "lower_better",
    "comprehensiveness": "higher_better",
    "sufficiency": "lower_better",
    "aopc": "higher_better",
    "aopc_comprehensiveness": "higher_better",
    "aopc_sufficiency": "lower_better",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    idx = int(round((len(xs) - 1) * float(q)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_samples(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return out
    for path in sorted(sample_dir.glob("*.json")):
        payload = _read_json(path)
        sample_id = str(payload.get("sample_id", path.stem))
        out[sample_id] = payload
    return out


def _top20_count(payload: Mapping[str, Any]) -> int:
    chunks = payload.get("chunks", [])
    total = len(chunks) if isinstance(chunks, list) else 0
    return int(math.floor(0.2 * float(total)))


def _extract_boundaries(payload: Mapping[str, Any]) -> List[int]:
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    boundaries: List[int] = []
    for chunk in chunks[:-1]:
        if not isinstance(chunk, Mapping):
            continue
        try:
            boundaries.append(int(chunk.get("end_char")))
        except Exception:
            continue
    return sorted(set(boundaries))


def _is_orphan(text: str) -> bool:
    return text.strip() != "" and (_ORPHAN_PUNCT_RE.fullmatch(text) is not None)


def _is_leading_close(text: str) -> bool:
    stripped = text.lstrip()
    if stripped == "":
        return False
    match = _LEADING_CLOSE_PUNCT_RE.match(stripped)
    if match is None:
        return False
    return stripped[match.end() :].strip() != ""


def _is_abbreviation_singleton(text: str) -> bool:
    normalized = text.replace("\n", " ").strip()
    if normalized == "":
        return False
    return _ABBREVIATION_SINGLETON_RE.fullmatch(normalized) is not None


def _metric_block(eval_report: Mapping[str, Any]) -> Dict[str, float]:
    gold = eval_report.get("metrics_by_target", {}).get("gold", {}).get("metrics_primary", {})
    out = {
        "log_odds": _safe_float(gold.get("log_odds")),
        "comprehensiveness": _safe_float(gold.get("comprehensiveness")),
        "sufficiency": _safe_float(gold.get("sufficiency")),
        "aopc": _safe_float(gold.get("aopc")),
        "aopc_comprehensiveness": _safe_float(gold.get("aopc_comprehensiveness")),
        "aopc_sufficiency": _safe_float(gold.get("aopc_sufficiency")),
        "runtime_seconds": _safe_float(eval_report.get("metrics_secondary", {}).get("runtime_seconds")),
    }
    return out


def _aggregate_sample_shape(samples: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "chunk_count_mean": 0.0,
            "chunk_count_p90": 0.0,
            "top20_count_mean": 0.0,
            "top20_count_p90": 0.0,
            "top20_count_hist": {},
            "selected_chunk_count_mean": 0.0,
            "selected_char_ratio_mean": 0.0,
        }

    chunk_counts: List[float] = []
    top20_counts: List[float] = []
    selected_counts: List[float] = []
    selected_ratios: List[float] = []

    for payload in samples.values():
        chunks = payload.get("chunks", [])
        selected = set(payload.get("selected_chunk_ids", []))
        if not isinstance(chunks, list):
            continue

        chunk_count = float(len(chunks))
        top20_count = float(_top20_count(payload))
        chunk_counts.append(chunk_count)
        top20_counts.append(top20_count)
        selected_counts.append(float(len(selected)))

        text = str(payload.get("text", ""))
        text_len = len(text)
        if text_len > 0:
            selected_chars = 0
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                try:
                    cid = int(chunk.get("chunk_id"))
                    start = int(chunk.get("start_char"))
                    end = int(chunk.get("end_char"))
                except Exception:
                    continue
                if cid in selected:
                    selected_chars += max(0, end - start)
            selected_ratios.append(float(selected_chars / float(text_len)))

    hist_counter: Counter[int] = Counter(int(x) for x in top20_counts)
    return {
        "sample_count": int(len(chunk_counts)),
        "chunk_count_mean": float(sum(chunk_counts) / len(chunk_counts)) if chunk_counts else 0.0,
        "chunk_count_p90": _percentile(chunk_counts, 0.90),
        "top20_count_mean": float(sum(top20_counts) / len(top20_counts)) if top20_counts else 0.0,
        "top20_count_p90": _percentile(top20_counts, 0.90),
        "top20_count_hist": {str(k): int(v) for k, v in sorted(hist_counter.items())},
        "selected_chunk_count_mean": float(sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
        "selected_char_ratio_mean": float(sum(selected_ratios) / len(selected_ratios)) if selected_ratios else 0.0,
    }


def _aggregate_error_clusters(samples: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    sample_counts = Counter()

    for payload in samples.values():
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        local = Counter()
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            text = str(chunk.get("text", ""))
            if _is_orphan(text):
                counts["orphan_punctuation_chunk"] += 1
                local["orphan_punctuation_chunk"] += 1
            if _is_leading_close(text):
                counts["leading_close_punct_chunk"] += 1
                local["leading_close_punct_chunk"] += 1
            if _is_abbreviation_singleton(text):
                counts["abbreviation_singleton_chunk"] += 1
                local["abbreviation_singleton_chunk"] += 1
        for key, value in local.items():
            if value > 0:
                sample_counts[key] += 1

    sample_total = float(len(samples)) if samples else 1.0
    return {
        "error_counts": {k: int(v) for k, v in sorted(counts.items())},
        "sample_with_error_ratio": {k: float(v / sample_total) for k, v in sorted(sample_counts.items())},
    }


def _aggregate_adaptive_signals(samples: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "adaptive_samples": 0,
            "bucket_counts": {},
            "bucket_ratios": {},
            "postprocess_means": {},
            "stage_means": {},
            "very_long": {
                "samples": 0,
                "raw_eq_one_ratio": 0.0,
                "fragmentation_ratio_mean": 0.0,
                "fragmentation_delta_mean": 0.0,
            },
        }

    bucket_counts: Counter[str] = Counter()
    postprocess_values: Dict[str, List[float]] = {
        "invalid_merge_count": [],
        "short_merge_count": [],
        "long_split_count": [],
        "post_long_invalid_merge_count": [],
        "post_long_short_merge_count": [],
        "adjacent_pack_merge_count": [],
    }
    stage_values: Dict[str, List[float]] = {
        "raw": [],
        "after_invalid": [],
        "after_short": [],
        "after_long": [],
        "after_post_long_merge": [],
        "final": [],
    }

    very_long_samples = 0
    very_long_raw_eq_one = 0
    very_long_ratio: List[float] = []
    very_long_delta: List[float] = []

    adaptive_samples = 0
    for payload in samples.values():
        diag = payload.get("metadata", {}).get("chunk_diagnostics", {})
        if not isinstance(diag, Mapping):
            continue
        if not bool(diag.get("adaptive_enabled", False)):
            continue
        adaptive_samples += 1

        bucket = str(diag.get("adaptive_bucket", "unknown"))
        bucket_counts[bucket] += 1

        post = diag.get("adaptive_postprocess", {})
        if not isinstance(post, Mapping):
            post = {}
        for key in postprocess_values:
            postprocess_values[key].append(_safe_float(post.get(key), 0.0))

        stage = diag.get("adaptive_stage_chunk_counts", {})
        if not isinstance(stage, Mapping):
            stage = {}
        for key in stage_values:
            if key == "final":
                stage_values[key].append(_safe_float(stage.get(key), _safe_float(diag.get("chunk_count"), 0.0)))
            elif key == "after_post_long_merge":
                stage_values[key].append(_safe_float(stage.get(key), _safe_float(stage.get("final"), 0.0)))
            else:
                stage_values[key].append(_safe_float(stage.get(key), 0.0))

        if bucket == "very_long":
            very_long_samples += 1
            raw = _safe_float(stage.get("raw"), 0.0)
            final = _safe_float(stage.get("final"), _safe_float(diag.get("chunk_count"), 0.0))
            if int(raw) == 1:
                very_long_raw_eq_one += 1
            if raw > 0:
                very_long_ratio.append(float(final / raw))
                very_long_delta.append(float(final - raw))

    denom = float(adaptive_samples) if adaptive_samples > 0 else 1.0
    very_denom = float(very_long_samples) if very_long_samples > 0 else 1.0

    return {
        "adaptive_samples": int(adaptive_samples),
        "bucket_counts": {k: int(v) for k, v in sorted(bucket_counts.items())},
        "bucket_ratios": {k: float(v / denom) for k, v in sorted(bucket_counts.items())},
        "postprocess_means": {
            k: (float(sum(v) / len(v)) if v else 0.0) for k, v in postprocess_values.items()
        },
        "stage_means": {
            k: (float(sum(v) / len(v)) if v else 0.0) for k, v in stage_values.items()
        },
        "very_long": {
            "samples": int(very_long_samples),
            "raw_eq_one_ratio": float(very_long_raw_eq_one / very_denom),
            "fragmentation_ratio_mean": (
                float(sum(very_long_ratio) / len(very_long_ratio)) if very_long_ratio else 0.0
            ),
            "fragmentation_delta_mean": (
                float(sum(very_long_delta) / len(very_long_delta)) if very_long_delta else 0.0
            ),
        },
    }


def _run_provenance(eval_report: Mapping[str, Any], run_cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    git = eval_report.get("provenance", {}).get("git", {})
    env = eval_report.get("provenance", {}).get("environment", {})
    deterministic = env.get("deterministic", {}) if isinstance(env, Mapping) else {}
    return {
        "commit_short": git.get("commit_short"),
        "is_dirty": git.get("is_dirty"),
        "device": run_cfg.get("device") if isinstance(run_cfg, Mapping) else None,
        "deterministic": run_cfg.get("deterministic") if isinstance(run_cfg, Mapping) else None,
        "deterministic_runtime": deterministic.get("enabled") if isinstance(deterministic, Mapping) else None,
    }


def _run_summary(run_dir: Path, name: str) -> Dict[str, Any]:
    eval_report = _read_json(run_dir / "eval_report.json")
    run_cfg_path = run_dir / "run_config.json"
    run_cfg = _read_json(run_cfg_path) if run_cfg_path.exists() else None
    samples = _load_samples(run_dir)

    search_profile = eval_report.get("search_profile")
    if not isinstance(search_profile, Mapping):
        search_profile = eval_report.get("metrics_secondary", {}).get("search_profile", {})

    return {
        "name": name,
        "run_dir": str(run_dir),
        "provenance": _run_provenance(eval_report, run_cfg),
        "metrics": _metric_block(eval_report),
        "shape": _aggregate_sample_shape(samples),
        "errors": _aggregate_error_clusters(samples),
        "adaptive": _aggregate_adaptive_signals(samples),
        "search_profile": dict(search_profile) if isinstance(search_profile, Mapping) else {},
        "samples": samples,
    }


def _directional_gain(metric: str, base: float, cand: float) -> float:
    direction = _DIRECTION.get(metric, "higher_better")
    if direction == "higher_better":
        return float(cand - base)
    return float(base - cand)


def _drift_summary(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    if not common_ids:
        return {
            "sample_count_common": 0,
            "selected_changed_ratio": 0.0,
            "ranking_changed_ratio": 0.0,
            "boundary_jaccard_mean": 0.0,
            "boundary_shift_chars_mean": 0.0,
        }

    selected_changed = 0
    ranking_changed = 0
    jaccards: List[float] = []
    shifts: List[float] = []

    for sid in common_ids:
        left = baseline_samples[sid]
        right = candidate_samples[sid]

        if left.get("selected_chunk_ids") != right.get("selected_chunk_ids"):
            selected_changed += 1
        if left.get("chunk_ranking") != right.get("chunk_ranking"):
            ranking_changed += 1

        lb = _extract_boundaries(left)
        rb = _extract_boundaries(right)
        lset = set(lb)
        rset = set(rb)
        union = lset.union(rset)
        if not union:
            jaccards.append(1.0)
        else:
            jaccards.append(float(len(lset.intersection(rset)) / float(len(union))))

        if lb and rb:
            for b in lb:
                nearest = min(abs(b - x) for x in rb)
                shifts.append(float(nearest))

    denom = float(len(common_ids))
    return {
        "sample_count_common": int(len(common_ids)),
        "selected_changed_ratio": float(selected_changed / denom),
        "ranking_changed_ratio": float(ranking_changed / denom),
        "boundary_jaccard_mean": float(statistics.mean(jaccards)) if jaccards else 0.0,
        "boundary_shift_chars_mean": float(statistics.mean(shifts)) if shifts else 0.0,
        "boundary_shift_chars_p90": _percentile(shifts, 0.90),
    }


def _top20_drift(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    common_ids = sorted(set(baseline_samples.keys()).intersection(candidate_samples.keys()))
    if not common_ids:
        return {
            "sample_count_common": 0,
            "baseline_top20_count_mean": 0.0,
            "candidate_top20_count_mean": 0.0,
            "top20_count_delta_mean": 0.0,
            "top20_count_delta_p90": 0.0,
        }

    base_vals = [float(_top20_count(baseline_samples[sid])) for sid in common_ids]
    cand_vals = [float(_top20_count(candidate_samples[sid])) for sid in common_ids]
    deltas = [c - b for b, c in zip(base_vals, cand_vals)]
    return {
        "sample_count_common": int(len(common_ids)),
        "baseline_top20_count_mean": float(sum(base_vals) / len(base_vals)),
        "candidate_top20_count_mean": float(sum(cand_vals) / len(cand_vals)),
        "top20_count_delta_mean": float(sum(deltas) / len(deltas)),
        "top20_count_delta_p90": _percentile(deltas, 0.90),
    }


def _compare(adaptive: Dict[str, Any], baseline: Dict[str, Any], pair_name: str) -> Dict[str, Any]:
    metric_delta: Dict[str, Dict[str, float]] = {}
    directional: Dict[str, float] = {}

    for key, base_val in baseline["metrics"].items():
        if key == "runtime_seconds":
            continue
        cand_val = adaptive["metrics"].get(key, 0.0)
        metric_delta[key] = {
            "baseline": float(base_val),
            "adaptive": float(cand_val),
            "delta": float(cand_val - base_val),
            "abs_diff": abs(float(cand_val - base_val)),
        }
        directional[key] = _directional_gain(key, float(base_val), float(cand_val))

    runtime_base = float(baseline["metrics"].get("runtime_seconds", 0.0))
    runtime_adaptive = float(adaptive["metrics"].get("runtime_seconds", 0.0))
    runtime_improve_ratio = ((runtime_base - runtime_adaptive) / runtime_base) if runtime_base > 0 else 0.0

    drift = _drift_summary(baseline["samples"], adaptive["samples"])
    top20 = _top20_drift(baseline["samples"], adaptive["samples"])

    return {
        "pair": pair_name,
        "metric_delta": metric_delta,
        "directional_gain": directional,
        "runtime": {
            "baseline_seconds": runtime_base,
            "adaptive_seconds": runtime_adaptive,
            "improve_ratio": float(runtime_improve_ratio),
        },
        "drift": drift,
        "top20_count_drift": top20,
    }


def _drop_samples(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(summary)
    out.pop("samples", None)
    return out


def build_report(sentence_dir: Path, sentence_v2_dir: Path, adaptive_dir: Path) -> Dict[str, Any]:
    sentence = _run_summary(sentence_dir, "sentence")
    sentence_v2 = _run_summary(sentence_v2_dir, "sentence_v2")
    adaptive = _run_summary(adaptive_dir, "adaptive")

    return {
        "runs": {
            "sentence": _drop_samples(sentence),
            "sentence_v2": _drop_samples(sentence_v2),
            "adaptive": _drop_samples(adaptive),
        },
        "comparisons": {
            "adaptive_vs_sentence": _compare(adaptive, sentence, "adaptive_vs_sentence"),
            "adaptive_vs_sentence_v2": _compare(adaptive, sentence_v2, "adaptive_vs_sentence_v2"),
        },
    }


def _csv_row(report: Mapping[str, Any]) -> Dict[str, Any]:
    cmp_sn = report["comparisons"]["adaptive_vs_sentence"]
    cmp_sv2 = report["comparisons"]["adaptive_vs_sentence_v2"]
    ad = report["runs"]["adaptive"]

    return {
        "adaptive_run_dir": ad.get("run_dir"),
        "adaptive_commit": ad.get("provenance", {}).get("commit_short"),
        "adaptive_runtime_seconds": ad.get("metrics", {}).get("runtime_seconds"),
        "adaptive_chunk_count_mean": ad.get("shape", {}).get("chunk_count_mean"),
        "adaptive_top20_count_mean": ad.get("shape", {}).get("top20_count_mean"),
        "adaptive_very_long_raw_eq_one_ratio": ad.get("adaptive", {}).get("very_long", {}).get("raw_eq_one_ratio"),
        "adaptive_very_long_fragmentation_ratio_mean": ad.get("adaptive", {}).get("very_long", {}).get(
            "fragmentation_ratio_mean"
        ),
        "adaptive_vs_sentence_log_odds_delta": cmp_sn["metric_delta"]["log_odds"]["delta"],
        "adaptive_vs_sentence_comp_delta": cmp_sn["metric_delta"]["comprehensiveness"]["delta"],
        "adaptive_vs_sentence_suff_delta": cmp_sn["metric_delta"]["sufficiency"]["delta"],
        "adaptive_vs_sentence_runtime_improve_ratio": cmp_sn["runtime"]["improve_ratio"],
        "adaptive_vs_sentence_selected_changed_ratio": cmp_sn["drift"]["selected_changed_ratio"],
        "adaptive_vs_sentence_ranking_changed_ratio": cmp_sn["drift"]["ranking_changed_ratio"],
        "adaptive_vs_sentence_top20_delta_mean": cmp_sn["top20_count_drift"]["top20_count_delta_mean"],
        "adaptive_vs_sentence_v2_log_odds_delta": cmp_sv2["metric_delta"]["log_odds"]["delta"],
        "adaptive_vs_sentence_v2_comp_delta": cmp_sv2["metric_delta"]["comprehensiveness"]["delta"],
        "adaptive_vs_sentence_v2_suff_delta": cmp_sv2["metric_delta"]["sufficiency"]["delta"],
        "adaptive_vs_sentence_v2_runtime_improve_ratio": cmp_sv2["runtime"]["improve_ratio"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze adaptive chunking mechanism against sentence baselines")
    parser.add_argument("--sentence-run-dir", type=str, required=True)
    parser.add_argument("--sentence-v2-run-dir", type=str, required=True)
    parser.add_argument("--adaptive-run-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sentence_dir = Path(args.sentence_run_dir)
    sentence_v2_dir = Path(args.sentence_v2_run_dir)
    adaptive_dir = Path(args.adaptive_run_dir)

    report = build_report(sentence_dir=sentence_dir, sentence_v2_dir=sentence_v2_dir, adaptive_dir=adaptive_dir)

    out_json = Path(args.output_json) if args.output_json else (adaptive_dir / "adaptive_mechanism_report.json")
    out_csv = Path(args.output_csv) if args.output_csv else (adaptive_dir / "adaptive_mechanism_report.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    row = _csv_row(report)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[adaptive-mechanism] json={out_json}")
    print(f"[adaptive-mechanism] csv={out_csv}")


if __name__ == "__main__":
    main()
