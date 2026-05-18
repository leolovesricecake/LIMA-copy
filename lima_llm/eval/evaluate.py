from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ..chunking.utils import chunk_char_length
from ..types import TextChunk
from ..utils import f1_iou_from_masks, spans_to_char_mask
from .metrics import (
    AML_AOPC_Q_VALUES,
    AML_PRIMARY_Q_PERCENT,
    EMPTY_PERTURBATION_TEXT,
    aml_faithfulness_metrics,
    aopc_metrics,
    build_perturbation_plan,
)

_WORD_UNIT_RE = re.compile(r"\s*\S+\s*")
_EVAL_GRANULARITIES = {"token", "word"}
_FLOAT_FORWARD_COUNTER_KEYS = {
    "batch_tokenize_seconds",
    "batch_pack_seconds",
    "batch_forward_seconds",
}


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


def _word_units_from_text(text: str) -> List[TextChunk]:
    units: List[TextChunk] = []
    for idx, match in enumerate(_WORD_UNIT_RE.finditer(text)):
        units.append(
            TextChunk(
                chunk_id=idx,
                start_char=match.start(),
                end_char=match.end(),
                text=text[match.start() : match.end()],
            )
        )
    if not units:
        return [TextChunk(chunk_id=0, start_char=0, end_char=len(text), text=text)]
    return units


def _normalize_eval_granularity(eval_granularity: str | None) -> str:
    value = str(eval_granularity or "token").strip().lower()
    if value not in _EVAL_GRANULARITIES:
        raise ValueError(f"Unsupported eval granularity: {eval_granularity!r}. Expected one of {_EVAL_GRANULARITIES}.")
    return value


def _repair_units_to_full_coverage(
    text: str,
    units: Sequence[TextChunk],
    *,
    keep_token_span: bool,
) -> List[TextChunk]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")]

    spans: List[Tuple[int, int, Any, Any]] = []
    text_len = len(text)
    for unit in units:
        start = max(0, min(int(unit.start_char), text_len))
        end = max(0, min(int(unit.end_char), text_len))
        if end < start:
            continue
        spans.append((start, end, unit.token_start, unit.token_end))

    if not spans:
        return [TextChunk(chunk_id=0, start_char=0, end_char=text_len, text=text)]

    spans.sort(key=lambda item: (item[0], item[1]))

    starts: List[int] = []
    prev = 0
    for idx, (start, _end, _tok_start, _tok_end) in enumerate(spans):
        cur = start
        if idx == 0:
            cur = 0
        elif cur < prev:
            cur = prev
        starts.append(cur)
        prev = cur

    repaired: List[TextChunk] = []
    for idx in range(len(spans)):
        start = starts[idx]
        end = starts[idx + 1] if idx < len(spans) - 1 else text_len
        if end < start:
            end = start
        tok_start = int(spans[idx][2]) if keep_token_span and spans[idx][2] is not None else None
        tok_end = int(spans[idx][3]) if keep_token_span and spans[idx][3] is not None else None
        repaired.append(
            TextChunk(
                chunk_id=idx,
                start_char=start,
                end_char=end,
                text=text[start:end],
                token_start=tok_start,
                token_end=tok_end,
            )
        )
    return repaired


def _token_units_from_text(text: str, tokenizer) -> tuple[List[TextChunk], bool]:
    if text == "":
        return [TextChunk(chunk_id=0, start_char=0, end_char=0, text="")], False

    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = encoded.get("offset_mapping", None)
            if offsets:
                token_units: List[TextChunk] = []
                for idx, offset in enumerate(offsets):
                    if offset is None or len(offset) < 2:
                        continue
                    start, end = int(offset[0]), int(offset[1])
                    if end < start:
                        continue
                    token_units.append(
                        TextChunk(
                            chunk_id=idx,
                            start_char=start,
                            end_char=end,
                            text=text[start:end],
                            token_start=idx,
                            token_end=idx + 1,
                        )
                    )
                if token_units:
                    return _repair_units_to_full_coverage(text=text, units=token_units, keep_token_span=True), False
        except Exception:
            pass

    fallback_units = _word_units_from_text(text)
    return _repair_units_to_full_coverage(text=text, units=fallback_units, keep_token_span=False), True


def _build_eval_units(text: str, eval_granularity: str, tokenizer) -> tuple[List[TextChunk], bool, str]:
    granularity = _normalize_eval_granularity(eval_granularity)
    if granularity == "word":
        word_units = _repair_units_to_full_coverage(text=text, units=_word_units_from_text(text), keep_token_span=False)
        return word_units, False, "word_whitespace_spans"

    token_units, fallback = _token_units_from_text(text=text, tokenizer=tokenizer)
    strategy = "tokenizer_offset_mapping" if not fallback else "whitespace_fallback_without_tokenizer_offsets"
    return token_units, fallback, strategy


def _content_span(unit: TextChunk) -> tuple[int, int]:
    leading_len = len(unit.text) - len(unit.text.lstrip())
    trailing_len = len(unit.text) - len(unit.text.rstrip())
    start = unit.start_char + leading_len
    end = unit.end_char - trailing_len
    if end <= start:
        return unit.start_char, unit.end_char
    return start, end


def _project_chunk_ranking_to_unit_ranking(
    eval_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    chunk_rank = {int(chunk_id): idx for idx, chunk_id in enumerate(chunk_ranking)}
    fallback_rank = len(chunk_rank) + len(chunks) + 1

    projected = []
    for unit in eval_units:
        unit_start, unit_end = _content_span(unit)
        weighted_rank = 0.0
        overlap_total = 0

        for chunk in chunks:
            overlap = max(0, min(unit_end, chunk.end_char) - max(unit_start, chunk.start_char))
            if overlap <= 0:
                continue
            weighted_rank += float(overlap) * float(chunk_rank.get(chunk.chunk_id, fallback_rank))
            overlap_total += int(overlap)

        if overlap_total <= 0:
            projected_rank = float(fallback_rank + unit.chunk_id)
        else:
            projected_rank = weighted_rank / float(overlap_total)
        projected.append((unit.chunk_id, projected_rank))

    return [unit_id for unit_id, _ in sorted(projected, key=lambda item: (item[1], item[0]))]


def _count_units_split_across_chunks(eval_units: Sequence[TextChunk], chunks: Sequence[TextChunk]) -> int:
    split_count = 0
    for unit in eval_units:
        unit_start, unit_end = _content_span(unit)
        overlaps = 0
        for chunk in chunks:
            if min(unit_end, chunk.end_char) > max(unit_start, chunk.start_char):
                overlaps += 1
                if overlaps > 1:
                    split_count += 1
                    break
    return split_count


def _project_chunk_ranking_to_word_ranking(
    word_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    return _project_chunk_ranking_to_unit_ranking(
        eval_units=word_units,
        chunks=chunks,
        chunk_ranking=chunk_ranking,
    )


def _count_words_split_across_chunks(word_units: Sequence[TextChunk], chunks: Sequence[TextChunk]) -> int:
    return _count_units_split_across_chunks(word_units, chunks)


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


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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

    sample_map = {s.sample_id: s for s in bundle.samples}
    mode_states = {
        "gold": _empty_mode_state(tracked_q_values),
        "predicted": _empty_mode_state(tracked_q_values),
    }

    sparsity_values: List[float] = []
    plaus_f1_values: List[float] = []
    plaus_iou_values: List[float] = []

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
    search_profile_runtime = {
        "samples_with_profile": 0,
        "missing_profile_samples": 0,
        "steps_completed": 0.0,
        "candidates_evaluated_total": 0.0,
        "remaining_build_calls": 0.0,
        "remaining_build_seconds": 0.0,
        "gains_eval_calls": 0.0,
        "gains_eval_seconds": 0.0,
        "argmax_calls": 0.0,
        "argmax_seconds": 0.0,
        "trace_write_calls": 0.0,
        "trace_write_seconds": 0.0,
        "evaluate_gains_path_calls": 0.0,
        "evaluate_gain_path_calls": 0.0,
        "search_name_counts": {},
    }

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

        search_meta = payload.get("metadata", {}).get("search_profile", {})
        if isinstance(search_meta, dict) and search_meta:
            search_profile_runtime["samples_with_profile"] += 1
            for key in (
                "steps_completed",
                "candidates_evaluated_total",
                "remaining_build_calls",
                "remaining_build_seconds",
                "gains_eval_calls",
                "gains_eval_seconds",
                "argmax_calls",
                "argmax_seconds",
                "trace_write_calls",
                "trace_write_seconds",
                "evaluate_gains_path_calls",
                "evaluate_gain_path_calls",
            ):
                search_profile_runtime[key] = float(search_profile_runtime[key]) + _safe_number(search_meta.get(key), 0.0)
            search_name = str(search_meta.get("search_name", "")).strip().lower()
            if search_name:
                name_counts = search_profile_runtime["search_name_counts"]
                name_counts[search_name] = int(name_counts.get(search_name, 0)) + 1
        else:
            search_profile_runtime["missing_profile_samples"] += 1

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
                aopc_payload = aopc_metrics(
                    chunks=eval_units,
                    ranking=ranking_units,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=_prob_fn,
                    perturbation_plan=perturbation_plan,
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
    profiled_samples = int(search_profile_runtime["samples_with_profile"])
    search_profile_stats = {
        "samples_with_profile": profiled_samples,
        "missing_profile_samples": int(search_profile_runtime["missing_profile_samples"]),
        "search_name_counts": dict(search_profile_runtime["search_name_counts"]),
        "steps_completed_total": int(search_profile_runtime["steps_completed"]),
        "candidates_evaluated_total": int(search_profile_runtime["candidates_evaluated_total"]),
        "remaining_build_calls_total": int(search_profile_runtime["remaining_build_calls"]),
        "remaining_build_seconds_total": float(search_profile_runtime["remaining_build_seconds"]),
        "gains_eval_calls_total": int(search_profile_runtime["gains_eval_calls"]),
        "gains_eval_seconds_total": float(search_profile_runtime["gains_eval_seconds"]),
        "argmax_calls_total": int(search_profile_runtime["argmax_calls"]),
        "argmax_seconds_total": float(search_profile_runtime["argmax_seconds"]),
        "trace_write_calls_total": int(search_profile_runtime["trace_write_calls"]),
        "trace_write_seconds_total": float(search_profile_runtime["trace_write_seconds"]),
        "evaluate_gains_path_calls_total": int(search_profile_runtime["evaluate_gains_path_calls"]),
        "evaluate_gain_path_calls_total": int(search_profile_runtime["evaluate_gain_path_calls"]),
        "steps_completed_mean": (
            float(search_profile_runtime["steps_completed"]) / float(profiled_samples) if profiled_samples > 0 else 0.0
        ),
        "candidates_evaluated_mean": (
            float(search_profile_runtime["candidates_evaluated_total"]) / float(profiled_samples)
            if profiled_samples > 0
            else 0.0
        ),
        "remaining_build_seconds_mean": (
            float(search_profile_runtime["remaining_build_seconds"]) / float(profiled_samples)
            if profiled_samples > 0
            else 0.0
        ),
        "gains_eval_seconds_mean": (
            float(search_profile_runtime["gains_eval_seconds"]) / float(profiled_samples) if profiled_samples > 0 else 0.0
        ),
        "argmax_seconds_mean": (
            float(search_profile_runtime["argmax_seconds"]) / float(profiled_samples) if profiled_samples > 0 else 0.0
        ),
        "trace_write_seconds_mean": (
            float(search_profile_runtime["trace_write_seconds"]) / float(profiled_samples)
            if profiled_samples > 0
            else 0.0
        ),
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
            "runtime_seconds": elapsed,
            "forward_counters_delta": counter_delta,
            "method_diagnostics": mode_reports["gold"]["method_diagnostics"],
            "timing_breakdown": timing_breakdown_payload,
            "cache_stats": cache_stats,
            "prefetch_stats": prefetch_stats,
            "backbone_batch_stats": backbone_batch_stats,
            "search_profile": search_profile_stats,
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
        "search_profile": search_profile_stats,
    }
    return report
