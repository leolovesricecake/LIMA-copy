from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from tqdm import tqdm

from ..chunking.utils import chunk_char_length
from ..types import TextChunk
from ..utils import atomic_write_json, f1_iou_from_masks, spans_to_char_mask
from .metrics import (
    AML_AOPC_Q_VALUES,
    AML_PRIMARY_Q_PERCENT,
    EMPTY_PERTURBATION_TEXT,
    deletion_trajectory,
    aml_faithfulness_metrics,
    aopc_metrics,
    build_perturbation_plan,
    top_percent_chunk_count,
)
from .units import (
    build_eval_units as _build_eval_units,
    count_units_split_across_chunks as _count_units_split_across_chunks,
    count_words_split_across_chunks as _count_words_split_across_chunks,
    normalize_eval_granularity as _normalize_eval_granularity,
    project_chunk_ranking_to_unit_ranking as _project_chunk_ranking_to_unit_ranking,
    project_chunk_ranking_to_word_ranking as _project_chunk_ranking_to_word_ranking,
    token_units_from_text as _token_units_from_text,
    word_units_from_text as _word_units_from_text,
)

_FLOAT_FORWARD_COUNTER_KEYS = {
    "batch_tokenize_seconds",
    "batch_pack_seconds",
    "batch_forward_seconds",
}

TRAJECTORY_POINTS_CSV = "trajectory_points.csv"
TRAJECTORY_POINTS_JSONL = "trajectory_points.jsonl"
TRAJECTORY_SUMMARY_CSV = "trajectory_summary.csv"


def _safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _reference_token_text(backbone) -> str:
    tokenizer = getattr(backbone, "tokenizer", None)
    for attr in ("mask_token", "unk_token", "pad_token", "eos_token"):
        token = getattr(tokenizer, attr, None) if tokenizer is not None else None
        if token:
            return str(token)
    return "<UNK>"


def _to_chunks(raw_chunks: Sequence[Dict]) -> List[TextChunk]:
    chunks = []
    for item in raw_chunks:
        chunks.append(
            TextChunk(
                chunk_id=int(item["chunk_id"]),
                start_char=int(item["start_char"]),
                end_char=int(item["end_char"]),
                text=str(item["text"]),
                token_start=item.get("token_start"),
                token_end=item.get("token_end"),
            )
        )
    return chunks


def _empty_mode_state(q_values: Sequence[int]) -> Dict:
    q_ints = [int(q) for q in q_values]
    return {
        "log_odds": [],
        "comp": [],
        "suff": [],
        "aopc_suff": [],
        "aopc_comp": [],
        "aopc": [],
        "per_q": {
            "comprehensiveness": {q: [] for q in q_ints},
            "sufficiency": {q: [] for q in q_ints},
        },
        "evaluated_samples": 0,
        "error_counts": {},
        "error_examples": [],
    }


def _append_mode_metrics(
    mode_state: Dict,
    metrics: Dict[str, float],
    aopc_payload: Dict[str, float],
    per_q: Dict[int, Dict[str, float]],
    q_values: Sequence[int],
) -> None:
    mode_state["log_odds"].append(float(metrics["log_odds"]))
    mode_state["comp"].append(float(metrics["comprehensiveness"]))
    mode_state["suff"].append(float(metrics["sufficiency"]))
    mode_state["aopc_suff"].append(float(metrics["aopc_sufficiency"]))
    mode_state["aopc_comp"].append(float(metrics["aopc_comprehensiveness"]))
    mode_state["aopc"].append(float(aopc_payload["aopc"]))
    mode_state["evaluated_samples"] += 1
    for q in q_values:
        q_int = int(q)
        mode_state["per_q"]["comprehensiveness"][q_int].append(float(per_q[q_int]["comp"]))
        mode_state["per_q"]["sufficiency"][q_int].append(float(per_q[q_int]["suff"]))


def _mean_per_q(per_q_block: Dict) -> Dict:
    return {
        "comprehensiveness": {
            str(int(q)): _safe_mean(values) for q, values in per_q_block["comprehensiveness"].items()
        },
        "sufficiency": {
            str(int(q)): _safe_mean(values) for q, values in per_q_block["sufficiency"].items()
        },
    }


def _mode_metrics_primary(mode_state: Dict) -> Dict[str, float]:
    return {
        "log_odds": _safe_mean(mode_state["log_odds"]),
        "comprehensiveness": _safe_mean(mode_state["comp"]),
        "sufficiency": _safe_mean(mode_state["suff"]),
        "aopc_sufficiency": _safe_mean(mode_state["aopc_suff"]),
        "aopc_comprehensiveness": _safe_mean(mode_state["aopc_comp"]),
        "aopc": _safe_mean(mode_state["aopc"]),
    }


def _mode_report(mode_state: Dict, sample_count: int, explain_method: str) -> Dict:
    evaluated = int(mode_state["evaluated_samples"])
    return {
        "report_method": explain_method,
        "metrics_primary": _mode_metrics_primary(mode_state),
        "method_diagnostics": {
            "evaluated_samples": evaluated,
            "failed_samples": max(0, int(sample_count) - evaluated),
            "error_type_counts": mode_state["error_counts"],
            "error_examples": mode_state["error_examples"],
        },
        "per_q": _mean_per_q(mode_state["per_q"]),
    }


def _normalize_chunk_ranking(payload: Dict, chunks: Sequence[TextChunk]) -> List[int]:
    chunk_ids = [int(c.chunk_id) for c in chunks]
    chunk_set = set(chunk_ids)
    ranking_raw = [int(x) for x in payload.get("chunk_ranking", [])]

    seen = set()
    ranking = []
    for cid in ranking_raw:
        if cid in chunk_set and cid not in seen:
            ranking.append(cid)
            seen.add(cid)

    for cid in chunk_ids:
        if cid not in seen:
            ranking.append(cid)
    return ranking


def _normalize_prefetch_fallback_policy(raw: str | None) -> str:
    policy = str(raw or "warn").strip().lower()
    if policy not in {"warn", "fail"}:
        raise ValueError(
            "Unsupported prefetch fallback policy: "
            f"{raw!r}. Expected one of ('warn', 'fail')."
        )
    return policy


def _model_name_from_output_root(output_root: Path) -> str:
    text = output_root.parent.name
    if text.startswith("model-"):
        return text[len("model-") :]
    return text


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding = "utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii = False) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding = "utf-8", newline = "") as file:
            file.write("")
        return

    normalized_rows = []
    fieldnames = list(rows[0].keys())
    for row in rows:
        normalized = {}
        for field in fieldnames:
            value = row.get(field)
            if isinstance(value, (list, dict)):
                normalized[field] = json.dumps(value, ensure_ascii = False)
            else:
                normalized[field] = value
        normalized_rows.append(normalized)

    with open(path, "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _trajectory_summary_rows(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for point in points:
        key = (
            point["source_family"],
            point["run_id"],
            point["report_stage"],
            point["dataset"],
            point["split"],
            point["model_name"],
            point["method_name"],
            point["step_index"],
            point["total_steps"],
            point["delete_count"],
            point["delete_fraction"],
            point["remaining_fraction"],
        )
        group = grouped.setdefault(
            key,
            {
                "source_family": point["source_family"],
                "run_id": point["run_id"],
                "report_stage": point["report_stage"],
                "dataset": point["dataset"],
                "split": point["split"],
                "model_name": point["model_name"],
                "method_name": point["method_name"],
                "step_index": point["step_index"],
                "total_steps": point["total_steps"],
                "delete_count": point["delete_count"],
                "delete_fraction": point["delete_fraction"],
                "remaining_fraction": point["remaining_fraction"],
                "target_probabilities": [],
                "prob_drops": [],
                "sample_ids": set(),
            },
        )
        group["target_probabilities"].append(float(point["target_probability"]))
        group["prob_drops"].append(float(point["prob_drop_from_full"]))
        group["sample_ids"].add(str(point["sample_id"]))

    rows = []
    for group in grouped.values():
        target_probabilities = np.asarray(group.pop("target_probabilities"), dtype = np.float64)
        prob_drops = np.asarray(group.pop("prob_drops"), dtype = np.float64)
        sample_ids = group.pop("sample_ids")
        group["mean_target_probability"] = float(np.mean(target_probabilities)) if len(target_probabilities) else 0.0
        group["std_target_probability"] = float(np.std(target_probabilities)) if len(target_probabilities) else 0.0
        group["mean_prob_drop_from_full"] = float(np.mean(prob_drops)) if len(prob_drops) else 0.0
        group["std_prob_drop_from_full"] = float(np.std(prob_drops)) if len(prob_drops) else 0.0
        group["sample_count"] = int(len(sample_ids))
        rows.append(group)

    rows.sort(
        key = lambda row: (
            str(row["dataset"]),
            str(row["model_name"]),
            str(row["method_name"]),
            int(row["step_index"]),
        )
    )
    return rows


def _write_trajectory_artifacts(output_root: Path, points: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    points_csv = output_root / TRAJECTORY_POINTS_CSV
    points_jsonl = output_root / TRAJECTORY_POINTS_JSONL
    summary_csv = output_root / TRAJECTORY_SUMMARY_CSV

    _write_csv(points_csv, points)
    _write_jsonl(points_jsonl, points)
    _write_csv(summary_csv, _trajectory_summary_rows(points))

    return {
        "trajectory_points_csv": TRAJECTORY_POINTS_CSV,
        "trajectory_points_jsonl": TRAJECTORY_POINTS_JSONL,
        "trajectory_summary_csv": TRAJECTORY_SUMMARY_CSV,
    }


def _prefetch_prob_cache(
    *,
    backbone,
    verbalizers: Sequence[str],
    texts: Sequence[str],
    cache: Dict[str, np.ndarray],
    enable_batch: bool,
    fallback_policy: str,
) -> Dict[str, Any]:
    missing = []
    seen = set()
    for text in texts:
        if text in cache or text in seen:
            continue
        missing.append(text)
        seen.add(text)

    stats: Dict[str, Any] = {
        "required_text_count": int(len(texts)),
        "missing_text_count": int(len(missing)),
        "prefetched_text_count": 0,
        "batch_attempted": False,
        "batch_succeeded": False,
        "batch_fallback_count": 0,
        "batch_error": None,
    }
    if not missing:
        return stats

    if enable_batch:
        stats["batch_attempted"] = True
        try:
            probs = np.asarray(backbone.predict_label_probs_batch(missing, verbalizers), dtype=np.float32)
            for idx, text in enumerate(missing):
                cache[text] = probs[idx]
            stats["batch_succeeded"] = True
            stats["prefetched_text_count"] = int(len(missing))
            return stats
        except Exception as exc:
            stats["batch_fallback_count"] = 1
            stats["batch_error"] = f"{type(exc).__name__}: {exc}"
            if fallback_policy == "fail":
                raise RuntimeError(
                    "Batch prefetch failed and fallback policy is 'fail'. "
                    f"missing={len(missing)} error={stats['batch_error']}"
                ) from exc
            print(
                "[eval][prefetch] batch prefetch failed, fallback to single-call path. "
                f"missing={len(missing)} error={stats['batch_error']}"
            )

    for text in missing:
        cache[text] = np.asarray(backbone.predict_label_probs(text, verbalizers), dtype=np.float32)
    stats["prefetched_text_count"] = int(len(missing))
    return stats


def evaluate_saved_explanations(
    output_root: Path,
    bundle,
    backbone,
    verbalizers: Sequence[str],
    q_values: Sequence[int],
    explain_method: str,
    eval_granularity: str = "token",
) -> Dict:
    method = str(explain_method).strip().lower()
    granularity = _normalize_eval_granularity(eval_granularity)

    sample_dir = output_root / "samples"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    sample_jsons = sorted(sample_dir.glob("*.json"))
    if not sample_jsons:
        raise FileNotFoundError(f"No sample json found in: {sample_dir}")

    aopc_q_values = tuple(int(q) for q in q_values) if q_values else AML_AOPC_Q_VALUES
    tracked_q_values = tuple(sorted(set((*aopc_q_values, AML_PRIMARY_Q_PERCENT))))
    reference_token_text = _reference_token_text(backbone)
    eval_batch_prefetch = os.getenv("LIMA_EVAL_BATCH_PREFETCH", "1").strip().lower() not in ("0", "false", "off", "no")
    prefetch_fallback_policy = _normalize_prefetch_fallback_policy(os.getenv("LIMA_PREFETCH_FALLBACK_POLICY", "warn"))
    prefetch_length_sort_requested = os.getenv("LIMA_EVAL_PREFETCH_LENGTH_SORT", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )
    prefetch_length_sort = False
    if prefetch_length_sort_requested:
        print(
            "[eval][prefetch] LIMA_EVAL_PREFETCH_LENGTH_SORT is deprecated and ignored. "
            "Evaluation keeps original perturbation text order."
        )
    tokenizer = getattr(backbone, "tokenizer", None)
    model_name = _model_name_from_output_root(output_root)

    sample_map = {s.sample_id: s for s in bundle.samples}
    mode_states = {
        "gold": _empty_mode_state(tracked_q_values),
        "predicted": _empty_mode_state(tracked_q_values),
    }

    sparsity_values: List[float] = []
    plaus_f1_values: List[float] = []
    plaus_iou_values: List[float] = []
    top20_count_zero_samples = 0
    selected_all_samples = 0

    total = 0
    acc_hits = 0

    skipped_missing_sample_id = 0
    skipped_empty_chunks = 0
    empty_text_samples = 0
    no_rationale_samples = 0
    over_max_length_samples = 0
    pred_not_gold_samples = 0
    token_len_eval_errors = 0
    eval_unit_count_total = 0
    units_split_across_chunks_total = 0
    tokenizer_fallback_samples = 0
    tokenizer_fallback_unit_total = 0
    unit_segmentation_counter: Dict[str, int] = {}
    timing_breakdown = {
        "unit_build_seconds": 0.0,
        "text_build_seconds": 0.0,
        "prefetch_seconds": 0.0,
        "metric_compute_seconds": 0.0,
    }
    cache_runtime = {
        "required_text_count": 0,
        "unique_text_count": 0,
        "requests": 0,
        "hits": 0,
        "misses": 0,
    }
    prefetch_runtime = {
        "batch_attempt_count": 0,
        "batch_success_count": 0,
        "batch_fallback_count": 0,
        "prefetched_text_count": 0,
        "missing_text_count": 0,
        "fallback_error_examples": [],
    }
    trajectory_points: List[Dict[str, Any]] = []
    sample_trajectory_ranges: Dict[str, Dict[str, int]] = {}

    max_length = getattr(backbone, "max_length", None)

    t0 = time.time()
    counter_before = backbone.snapshot_counters()
    sample_total = len(sample_jsons)
    print(f"[eval] start method={method} samples={sample_total}")

    progress = tqdm(sample_jsons, desc=f"eval-{method}", unit="sample", dynamic_ncols=True)
    for sample_json in progress:
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        sample_id = payload["sample_id"]

        payload_method = str(payload.get("explain_method", "")).strip().lower()
        if payload_method and payload_method != method:
            raise ValueError(
                f"Mismatched explain method in {sample_json}: payload={payload_method}, expected={method}"
            )

        if sample_id not in sample_map:
            skipped_missing_sample_id += 1
            continue

        sample = sample_map[sample_id]
        t_unit = time.time()
        chunks = _to_chunks(payload["chunks"])
        eval_units, fallback_used, segmentation_strategy = _build_eval_units(
            text=sample.text,
            eval_granularity=granularity,
            tokenizer=tokenizer,
        )
        selected = [int(x) for x in payload.get("selected_chunk_ids", [])]
        if not chunks:
            skipped_empty_chunks += 1
            continue
        chunk_ranking = _normalize_chunk_ranking(payload=payload, chunks=chunks)
        ranking_units = _project_chunk_ranking_to_unit_ranking(
            eval_units=eval_units,
            chunks=chunks,
            chunk_ranking=chunk_ranking,
        )
        timing_breakdown["unit_build_seconds"] += time.time() - t_unit

        total += 1
        eval_unit_count_total += len(eval_units)
        units_split_across_chunks_total += _count_units_split_across_chunks(eval_units, chunks)
        unit_segmentation_counter[segmentation_strategy] = unit_segmentation_counter.get(segmentation_strategy, 0) + 1
        if fallback_used:
            tokenizer_fallback_samples += 1
            tokenizer_fallback_unit_total += len(eval_units)

        if sample.text.strip() == "":
            empty_text_samples += 1
        if not sample.rationale_char_spans:
            no_rationale_samples += 1

        if max_length is not None:
            try:
                if int(backbone.tokenize_len(sample.text)) > int(max_length):
                    over_max_length_samples += 1
            except Exception:
                token_len_eval_errors += 1

        t_text_build = time.time()
        perturbation_plan = build_perturbation_plan(
            chunks=eval_units,
            ranking=ranking_units,
            q_values=tracked_q_values,
            primary_q_percent=AML_PRIMARY_Q_PERCENT,
            reference_token_text=reference_token_text,
        )
        required_texts = list(perturbation_plan.get("required_texts", []))
        cache_runtime["required_text_count"] += int(perturbation_plan.get("required_text_count", len(required_texts)))
        cache_runtime["unique_text_count"] += int(
            perturbation_plan.get("unique_required_text_count", len(required_texts))
        )
        timing_breakdown["text_build_seconds"] += time.time() - t_text_build

        sample_prob_cache: Dict[str, np.ndarray] = {}
        t_prefetch = time.time()
        prefetch_meta = _prefetch_prob_cache(
            backbone=backbone,
            verbalizers=verbalizers,
            texts=required_texts,
            cache=sample_prob_cache,
            enable_batch=eval_batch_prefetch,
            fallback_policy=prefetch_fallback_policy,
        )
        timing_breakdown["prefetch_seconds"] += time.time() - t_prefetch
        prefetch_runtime["batch_attempt_count"] += int(prefetch_meta.get("batch_attempted", False))
        prefetch_runtime["batch_success_count"] += int(prefetch_meta.get("batch_succeeded", False))
        prefetch_runtime["batch_fallback_count"] += int(prefetch_meta.get("batch_fallback_count", 0))
        prefetch_runtime["prefetched_text_count"] += int(prefetch_meta.get("prefetched_text_count", 0))
        prefetch_runtime["missing_text_count"] += int(prefetch_meta.get("missing_text_count", 0))
        batch_error = prefetch_meta.get("batch_error")
        if batch_error and len(prefetch_runtime["fallback_error_examples"]) < 10:
            prefetch_runtime["fallback_error_examples"].append({"sample_id": sample_id, "error": str(batch_error)})

        def _prob_fn(text: str, _verbalizers: Sequence[str]) -> np.ndarray:
            cache_runtime["requests"] += 1
            if text in sample_prob_cache:
                cache_runtime["hits"] += 1
                return sample_prob_cache[text]
            cache_runtime["misses"] += 1
            sample_prob_cache[text] = np.asarray(backbone.predict_label_probs(text, verbalizers), dtype=np.float32)
            return sample_prob_cache[text]

        t_metric_compute = time.time()
        text_for_pred = sample.text if sample.text else EMPTY_PERTURBATION_TEXT
        full_probs = _prob_fn(text_for_pred, verbalizers)
        pred_label = int(np.argmax(full_probs))
        if pred_label == sample.label:
            acc_hits += 1
        else:
            pred_not_gold_samples += 1

        selected_len = chunk_char_length(chunks, selected)
        total_len = max(1, len(sample.text))
        sparsity_values.append(selected_len / total_len)
        top20_count = top_percent_chunk_count(total_chunks=len(chunks), q_percent=AML_PRIMARY_Q_PERCENT)
        if int(top20_count) == 0:
            top20_count_zero_samples += 1
        if len(chunks) > 0 and len(set(int(x) for x in selected)) >= len(chunks):
            selected_all_samples += 1

        if sample.rationale_char_spans:
            chunk_by_id = {c.chunk_id: c for c in chunks}
            pred_mask = spans_to_char_mask(
                text_len=len(sample.text),
                spans=[
                    (chunk_by_id[cid].start_char, chunk_by_id[cid].end_char)
                    for cid in selected
                    if cid in chunk_by_id
                ],
            )
            gold_mask = spans_to_char_mask(
                text_len=len(sample.text),
                spans=sample.rationale_char_spans,
            )
            f1, iou = f1_iou_from_masks(pred_mask, gold_mask)
            plaus_f1_values.append(f1)
            plaus_iou_values.append(iou)

        for mode_name, target_label in (("gold", sample.label), ("predicted", pred_label)):
            mode_state = mode_states[mode_name]
            try:
                metrics, per_q = aml_faithfulness_metrics(
                    chunks=eval_units,
                    ranking=ranking_units,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=_prob_fn,
                    aopc_q_values=aopc_q_values,
                    extra_q_values=tracked_q_values,
                    reference_token_text=reference_token_text,
                    perturbation_plan=perturbation_plan,
                )
                if mode_name == "predicted":
                    trajectory_payload = deletion_trajectory(
                        chunks = eval_units,
                        ranking = ranking_units,
                        target_label = target_label,
                        verbalizers = verbalizers,
                        prob_fn = _prob_fn,
                        perturbation_plan = perturbation_plan,
                    )
                    trajectory_start = len(trajectory_points)
                    target_label_text = None
                    if 0 <= int(target_label) < len(getattr(bundle, "label_names", [])):
                        target_label_text = str(bundle.label_names[int(target_label)])
                    elif 0 <= int(target_label) < len(verbalizers):
                        target_label_text = str(verbalizers[int(target_label)])
                    for point in trajectory_payload["points"]:
                        trajectory_points.append(
                            {
                                "source_family": "lima_llm",
                                "run_id": str(output_root.name),
                                "report_stage": "EVAL",
                                "dataset": str(bundle.dataset_name),
                                "split": str(bundle.split),
                                "model_name": str(model_name),
                                "method_name": str(method),
                                "sample_id": str(sample_id),
                                "target_label_id": int(target_label),
                                "target_label_text": target_label_text,
                                "step_index": int(point["step_index"]),
                                "total_steps": int(point["total_steps"]),
                                "delete_count": int(point["delete_count"]),
                                "delete_fraction": float(point["delete_fraction"]),
                                "remaining_fraction": float(point["remaining_fraction"]),
                                "target_probability": float(point["target_probability"]),
                                "prob_drop_from_full": float(point["prob_drop_from_full"]),
                                "is_full_text_step": bool(point["is_full_text_step"]),
                                "deleted_ids": list(point["deleted_ids"]),
                            }
                        )
                    sample_trajectory_ranges[str(sample_id)] = {
                        "start_row": int(trajectory_start),
                        "end_row_exclusive": int(len(trajectory_points)),
                        "step_count": int(len(trajectory_payload["points"])),
                    }
                    aopc_payload = {"aopc": float(trajectory_payload["aopc"])}
                else:
                    aopc_payload = aopc_metrics(
                        chunks = eval_units,
                        ranking = ranking_units,
                        target_label = target_label,
                        verbalizers = verbalizers,
                        prob_fn = _prob_fn,
                        perturbation_plan = perturbation_plan,
                    )
                _append_mode_metrics(mode_state, metrics, aopc_payload, per_q, tracked_q_values)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                mode_state["error_counts"][err] = mode_state["error_counts"].get(err, 0) + 1
                if len(mode_state["error_examples"]) < 10:
                    mode_state["error_examples"].append({"sample_id": sample_id, "error": err})
        timing_breakdown["metric_compute_seconds"] += time.time() - t_metric_compute

    elapsed = time.time() - t0
    counter_after = backbone.snapshot_counters()
    counter_delta: Dict[str, float | int] = {}
    for key in set(counter_before.keys()).union(counter_after.keys()):
        delta = counter_after.get(key, 0) - counter_before.get(key, 0)
        if key in _FLOAT_FORWARD_COUNTER_KEYS:
            counter_delta[key] = float(delta)
        else:
            counter_delta[key] = int(delta)
    total_cache_requests = int(cache_runtime["requests"])
    cache_hit_rate = float(cache_runtime["hits"] / total_cache_requests) if total_cache_requests > 0 else 0.0
    cache_stats = {
        "required_text_count": int(cache_runtime["required_text_count"]),
        "unique_text_count": int(cache_runtime["unique_text_count"]),
        "cache_requests": total_cache_requests,
        "cache_hits": int(cache_runtime["hits"]),
        "cache_misses": int(cache_runtime["misses"]),
        "cache_hit_rate": cache_hit_rate,
    }
    prefetch_stats = {
        "batch_enabled": bool(eval_batch_prefetch),
        "fallback_policy": prefetch_fallback_policy,
        "length_sort_enabled": bool(prefetch_length_sort),
        "batch_attempt_count": int(prefetch_runtime["batch_attempt_count"]),
        "batch_success_count": int(prefetch_runtime["batch_success_count"]),
        "batch_fallback_count": int(prefetch_runtime["batch_fallback_count"]),
        "prefetched_text_count": int(prefetch_runtime["prefetched_text_count"]),
        "missing_text_count": int(prefetch_runtime["missing_text_count"]),
        "fallback_error_examples": list(prefetch_runtime["fallback_error_examples"]),
    }
    timing_breakdown_payload = {k: float(v) for k, v in timing_breakdown.items()}
    backbone_batch_stats = {
        "batch_calls": int(counter_delta.get("batch_calls", 0)),
        "batch_rows": int(counter_delta.get("batch_rows", 0)),
        "model_forward_calls": int(counter_delta.get("model_forward_calls", 0)),
        "oom_shrink_events": int(counter_delta.get("oom_shrink_events", 0)),
        "batch_tokenize_calls": int(counter_delta.get("batch_tokenize_calls", 0)),
        "batch_pack_calls": int(counter_delta.get("batch_pack_calls", 0)),
        "batch_forward_calls": int(counter_delta.get("batch_forward_calls", 0)),
        "batch_tokenize_seconds": float(counter_delta.get("batch_tokenize_seconds", 0.0)),
        "batch_pack_seconds": float(counter_delta.get("batch_pack_seconds", 0.0)),
        "batch_forward_seconds": float(counter_delta.get("batch_forward_seconds", 0.0)),
    }

    mode_reports = {
        "gold": _mode_report(mode_states["gold"], total, method),
        "predicted": _mode_report(mode_states["predicted"], total, method),
    }
    trajectory_artifacts = _write_trajectory_artifacts(output_root, trajectory_points)

    for sample_json in sample_jsons:
        payload = json.loads(sample_json.read_text(encoding = "utf-8"))
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in sample_trajectory_ranges:
            continue
        row_range = sample_trajectory_ranges[sample_id]
        payload["trajectory_step_count"] = int(row_range["step_count"])
        payload["trajectory_row_range"] = {
            "start_row": int(row_range["start_row"]),
            "end_row_exclusive": int(row_range["end_row_exclusive"]),
        }
        atomic_write_json(sample_json, payload)

    report = {
        "report_method": method,
        "dataset": bundle.dataset_name,
        "split": bundle.split,
        "sample_count": total,
        "metric_settings": {
            "protocol": "AML",
            "primary_top_k_percent": int(AML_PRIMARY_Q_PERCENT),
            "aopc_top_k_percentages": [int(q) for q in aopc_q_values],
            "aopc_average_denominator": "len(top_k_percentages)+1",
            "morf_average_denominator": "num_deletion_steps",
            "perturbation_target": "predicted",
            "log_odds_reference_token": reference_token_text,
            "perturbation_unit": granularity,
            "unit_segmentation": (
                "subword token spans from tokenizer offset_mapping with contiguous coverage repair; fallback to whitespace spans if unavailable"
                if granularity == "token"
                else "whitespace-delimited spans with surrounding whitespace preserved"
            ),
            "chunk_to_unit_projection": "overlap-weighted average chunk rank; lower rank is more important",
            "unit_segmentation_counts": {k: int(v) for k, v in sorted(unit_segmentation_counter.items())},
            "eval_batch_prefetch_enabled": bool(eval_batch_prefetch),
            "prefetch_fallback_policy": prefetch_fallback_policy,
            "prefetch_length_sort_enabled": bool(prefetch_length_sort),
        },
        "metrics_primary": {
            "accuracy_full": (acc_hits / total) if total > 0 else 0.0,
            **mode_reports["gold"]["metrics_primary"],
        },
        "metrics_secondary": {
            "sparsity": _safe_mean(sparsity_values),
            "plausibility_f1": _safe_mean(plaus_f1_values),
            "plausibility_iou": _safe_mean(plaus_iou_values),
            "top20_count_zero_ratio": (float(top20_count_zero_samples) / float(total)) if total > 0 else 0.0,
            "selected_all_ratio": (float(selected_all_samples) / float(total)) if total > 0 else 0.0,
            "plausibility_available": bool((total - no_rationale_samples) > 0),
            "plausibility_coverage_ratio": (
                float(total - no_rationale_samples) / float(total) if total > 0 else 0.0
            ),
            "runtime_seconds": elapsed,
            "forward_counters_delta": counter_delta,
            "method_diagnostics": mode_reports["gold"]["method_diagnostics"],
            "timing_breakdown": timing_breakdown_payload,
            "cache_stats": cache_stats,
            "prefetch_stats": prefetch_stats,
            "backbone_batch_stats": backbone_batch_stats,
        },
        "metrics_by_target": mode_reports,
        "dataset_diagnostics": {
            "samples_with_empty_text": empty_text_samples,
            "samples_without_rationale": no_rationale_samples,
            "samples_over_backbone_max_length": over_max_length_samples,
            "predicted_not_gold_samples": pred_not_gold_samples,
            "token_len_eval_errors": token_len_eval_errors,
            "skipped_missing_sample_id": skipped_missing_sample_id,
            "skipped_empty_chunks": skipped_empty_chunks,
            "backbone_max_length": int(max_length) if max_length is not None else None,
            "eval_unit_count_total": int(eval_unit_count_total),
            "units_split_across_chunks_total": int(units_split_across_chunks_total),
            "units_split_across_chunks_rate": (
                float(units_split_across_chunks_total / eval_unit_count_total) if eval_unit_count_total > 0 else 0.0
            ),
            "tokenizer_fallback_samples": int(tokenizer_fallback_samples),
            "tokenizer_fallback_unit_total": int(tokenizer_fallback_unit_total),
        },
        "q_values": [int(q) for q in aopc_q_values],
        "per_q_values": [int(q) for q in tracked_q_values],
        "timing_breakdown": timing_breakdown_payload,
        "cache_stats": cache_stats,
        "prefetch_stats": prefetch_stats,
        "backbone_batch_stats": backbone_batch_stats,
        "artifacts": {
            "eval_report_json": "eval_report.json",
            **trajectory_artifacts,
        },
    }
    return report
