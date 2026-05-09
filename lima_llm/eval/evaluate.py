from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm

from ..chunking.utils import chunk_char_length
from ..types import TextChunk
from ..utils import f1_iou_from_masks, spans_to_char_mask
from .metrics import AML_AOPC_Q_VALUES, AML_PRIMARY_Q_PERCENT, aml_faithfulness_metrics, aopc_metrics

_WORD_UNIT_RE = re.compile(r"\s*\S+\s*")


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


def _content_span(unit: TextChunk) -> tuple[int, int]:
    leading_len = len(unit.text) - len(unit.text.lstrip())
    trailing_len = len(unit.text) - len(unit.text.rstrip())
    start = unit.start_char + leading_len
    end = unit.end_char - trailing_len
    if end <= start:
        return unit.start_char, unit.end_char
    return start, end


def _project_chunk_ranking_to_word_ranking(
    word_units: Sequence[TextChunk],
    chunks: Sequence[TextChunk],
    chunk_ranking: Sequence[int],
) -> List[int]:
    chunk_rank = {int(chunk_id): idx for idx, chunk_id in enumerate(chunk_ranking)}
    fallback_rank = len(chunk_rank) + len(chunks) + 1

    projected = []
    for word in word_units:
        word_start, word_end = _content_span(word)
        weighted_rank = 0.0
        overlap_total = 0

        for chunk in chunks:
            overlap = max(0, min(word_end, chunk.end_char) - max(word_start, chunk.start_char))
            if overlap <= 0:
                continue
            weighted_rank += float(overlap) * float(chunk_rank.get(chunk.chunk_id, fallback_rank))
            overlap_total += int(overlap)

        if overlap_total <= 0:
            projected_rank = float(fallback_rank + word.chunk_id)
        else:
            projected_rank = weighted_rank / float(overlap_total)
        projected.append((word.chunk_id, projected_rank))

    return [word_id for word_id, _ in sorted(projected, key=lambda item: (item[1], item[0]))]


def _count_words_split_across_chunks(word_units: Sequence[TextChunk], chunks: Sequence[TextChunk]) -> int:
    split_count = 0
    for word in word_units:
        word_start, word_end = _content_span(word)
        overlaps = 0
        for chunk in chunks:
            if min(word_end, chunk.end_char) > max(word_start, chunk.start_char):
                overlaps += 1
                if overlaps > 1:
                    split_count += 1
                    break
    return split_count


def _empty_mode_state(q_values: Sequence[int]) -> Dict:
    q_ints = [int(q) for q in q_values]
    return {
        "log_odds": [],
        "comp": [],
        "suff": [],
        "aopc_suff": [],
        "aopc_comp": [],
        "aopc": [],
        "del_auc": [],
        "ins_auc": [],
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
    curve: Dict[str, float],
    per_q: Dict[int, Dict[str, float]],
    q_values: Sequence[int],
) -> None:
    mode_state["log_odds"].append(float(metrics["log_odds"]))
    mode_state["comp"].append(float(metrics["comprehensiveness"]))
    mode_state["suff"].append(float(metrics["sufficiency"]))
    mode_state["aopc_suff"].append(float(metrics["aopc_sufficiency"]))
    mode_state["aopc_comp"].append(float(metrics["aopc_comprehensiveness"]))
    mode_state["aopc"].append(float(curve["aopc"]))
    mode_state["del_auc"].append(float(curve["deletion_auc"]))
    mode_state["ins_auc"].append(float(curve["insertion_auc"]))
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
        "deletion_auc": _safe_mean(mode_state["del_auc"]),
        "insertion_auc": _safe_mean(mode_state["ins_auc"]),
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


def evaluate_saved_explanations(
    output_root: Path,
    bundle,
    backbone,
    verbalizers: Sequence[str],
    q_values: Sequence[int],
    explain_method: str,
) -> Dict:
    method = str(explain_method).strip().lower()

    sample_dir = output_root / "samples"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    sample_jsons = sorted(sample_dir.glob("*.json"))
    if not sample_jsons:
        raise FileNotFoundError(f"No sample json found in: {sample_dir}")

    aopc_q_values = tuple(int(q) for q in q_values) if q_values else AML_AOPC_Q_VALUES
    tracked_q_values = tuple(sorted(set((*aopc_q_values, AML_PRIMARY_Q_PERCENT))))
    reference_token_text = _reference_token_text(backbone)

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
    word_count_total = 0
    words_split_across_chunks_total = 0

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
        chunks = _to_chunks(payload["chunks"])
        word_units = _word_units_from_text(sample.text)
        selected = [int(x) for x in payload.get("selected_chunk_ids", [])]
        if not chunks:
            skipped_empty_chunks += 1
            continue

        total += 1
        word_count_total += len(word_units)
        words_split_across_chunks_total += _count_words_split_across_chunks(word_units, chunks)

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

        chunk_ranking = _normalize_chunk_ranking(payload=payload, chunks=chunks)
        ranking_words = _project_chunk_ranking_to_word_ranking(
            word_units=word_units,
            chunks=chunks,
            chunk_ranking=chunk_ranking,
        )

        sample_prob_cache: Dict[str, np.ndarray] = {}

        def _prob_fn(text: str, _verbalizers: Sequence[str]) -> np.ndarray:
            if text not in sample_prob_cache:
                sample_prob_cache[text] = np.asarray(backbone.predict_label_probs(text, verbalizers), dtype=np.float32)
            return sample_prob_cache[text]

        text_for_pred = sample.text if sample.text else "<EMPTY>"
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
                    chunks=word_units,
                    ranking=ranking_words,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=_prob_fn,
                    aopc_q_values=aopc_q_values,
                    extra_q_values=tracked_q_values,
                    reference_token_text=reference_token_text,
                )
                curve = aopc_metrics(
                    chunks=word_units,
                    ranking=ranking_words,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=_prob_fn,
                )
                _append_mode_metrics(mode_state, metrics, curve, per_q, tracked_q_values)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                mode_state["error_counts"][err] = mode_state["error_counts"].get(err, 0) + 1
                if len(mode_state["error_examples"]) < 10:
                    mode_state["error_examples"].append({"sample_id": sample_id, "error": err})

    elapsed = time.time() - t0
    counter_after = backbone.snapshot_counters()
    counter_delta = {
        k: int(counter_after.get(k, 0) - counter_before.get(k, 0))
        for k in set(counter_before.keys()).union(counter_after.keys())
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
            "perturbation_unit": "word",
            "word_segmentation": "whitespace-delimited spans with surrounding whitespace preserved",
            "chunk_to_word_projection": "overlap-weighted average chunk rank; lower rank is more important",
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
            "word_count_total": int(word_count_total),
            "words_split_across_chunks_total": int(words_split_across_chunks_total),
            "words_split_across_chunks_rate": (
                float(words_split_across_chunks_total / word_count_total) if word_count_total > 0 else 0.0
            ),
        },
        "q_values": [int(q) for q in aopc_q_values],
        "per_q_values": [int(q) for q in tracked_q_values],
    }
    return report
