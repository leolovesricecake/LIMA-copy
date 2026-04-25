from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np

from ..chunking.utils import chunk_char_length
from ..types import TextChunk
from ..utils import f1_iou_from_masks, spans_to_char_mask
from .metrics import AML_AOPC_Q_VALUES, AML_PRIMARY_Q_PERCENT, aml_faithfulness_metrics, aopc_metrics


_WORD_UNIT_RE = re.compile(r"\s*\S+\s*")


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def _random_ranking(chunk_ids: Sequence[int], rng: random.Random) -> List[int]:
    result = list(chunk_ids)
    rng.shuffle(result)
    return result


def _is_oom_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "outofmemory" in type(exc).__name__.lower():
        return True
    return "out of memory" in msg and ("cuda" in msg or "cudnn" in msg or "hip" in msg)


def _clear_cuda_cache(backbone) -> None:
    torch = getattr(backbone, "torch", None)
    if torch is None:
        try:
            import torch as _torch
        except Exception:
            return
        torch = _torch
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gradient_ranking(backbone, text: str, chunks: Sequence[TextChunk], label: int, verbalizers: Sequence[str]) -> List[int]:
    scores = backbone.gradient_chunk_importance(text, chunks, label, verbalizers)
    return [
        cid
        for cid, _ in sorted(
            [(chunk.chunk_id, float(scores[chunk.chunk_id])) for chunk in chunks],
            key=lambda x: (-x[1], x[0]),
        )
    ]


def _gradient_ranking_with_retry(
    backbone,
    text: str,
    chunks: Sequence[TextChunk],
    label: int,
    verbalizers: Sequence[str],
    max_oom_retries: int = 1,
) -> List[int]:
    last_exc: Exception | None = None
    for attempt in range(max_oom_retries + 1):
        try:
            return _gradient_ranking(backbone, text, chunks, label, verbalizers)
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt < max_oom_retries and _is_oom_error(exc):
                _clear_cuda_cache(backbone)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected gradient retry state")


def _init_mode_state(q_values: Sequence[int]) -> Dict:
    q_ints = [int(q) for q in q_values]
    return {
        "ours_log_odds": [],
        "ours_comp": [],
        "ours_suff": [],
        "ours_aopc_suff": [],
        "ours_aopc_comp": [],
        "ours_aopc": [],
        "ours_del_auc": [],
        "ours_ins_auc": [],
        "random_log_odds": [],
        "random_comp": [],
        "random_suff": [],
        "random_aopc_suff": [],
        "random_aopc_comp": [],
        "grad_log_odds": [],
        "grad_comp": [],
        "grad_suff": [],
        "grad_aopc_suff": [],
        "grad_aopc_comp": [],
        "diagnosticity_hits": 0,
        "grad_evaluated": 0,
        "grad_error_counts": {},
        "grad_error_examples": [],
        "per_q": {
            "ours": {
                "comprehensiveness": {q: [] for q in q_ints},
                "sufficiency": {q: [] for q in q_ints},
            },
            "random": {
                "comprehensiveness": {q: [] for q in q_ints},
                "sufficiency": {q: [] for q in q_ints},
            },
            "gradient": {
                "comprehensiveness": {q: [] for q in q_ints},
                "sufficiency": {q: [] for q in q_ints},
            },
        },
    }


def _mean_per_q(per_q_block: Dict) -> Dict:
    return {
        "comprehensiveness": {
            str(int(q)): _safe_mean(values) for q, values in per_q_block["comprehensiveness"].items()
        },
        "sufficiency": {
            str(int(q)): _safe_mean(values) for q, values in per_q_block["sufficiency"].items()
        },
    }


def evaluate_saved_explanations(
    output_root: Path,
    bundle,
    backbone,
    verbalizers: Sequence[str],
    q_values: Sequence[int],
    random_trials: int = 5,
    include_gradient_baseline: bool = False,
) -> Dict:
    sample_dir = output_root / "samples"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")

    aopc_q_values = tuple(int(q) for q in q_values) if q_values else AML_AOPC_Q_VALUES
    tracked_q_values = tuple(sorted(set((*aopc_q_values, AML_PRIMARY_Q_PERCENT))))
    reference_token_text = _reference_token_text(backbone)

    sample_map = {s.sample_id: s for s in bundle.samples}
    mode_states = {
        "gold": _init_mode_state(tracked_q_values),
        "predicted": _init_mode_state(tracked_q_values),
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
    prob_cache_hits_total = 0
    prob_cache_misses_total = 0
    prob_cache_unique_texts_total = 0

    max_length = getattr(backbone, "max_length", None)

    t0 = time.time()
    counter_before = backbone.snapshot_counters()

    for sample_json in sorted(sample_dir.glob("*.json")):
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        sample_id = payload["sample_id"]
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

        chunk_ids = [c.chunk_id for c in chunks]
        chunk_ranking_ours = selected + [cid for cid in chunk_ids if cid not in set(selected)]
        word_ids = [w.chunk_id for w in word_units]
        ranking_ours = _project_chunk_ranking_to_word_ranking(
            word_units=word_units,
            chunks=chunks,
            chunk_ranking=chunk_ranking_ours,
        )

        prob_cache: Dict[str, np.ndarray] = {}
        prob_cache_hits = 0
        prob_cache_misses = 0

        def _cached_prob_fn(text: str, _verbalizers: Sequence[str]) -> np.ndarray:
            nonlocal prob_cache_hits, prob_cache_misses
            if text in prob_cache:
                prob_cache_hits += 1
                return prob_cache[text]
            prob_cache_misses += 1
            probs = backbone.predict_label_probs(text, verbalizers)
            prob_cache[text] = probs
            return probs

        text_for_pred = sample.text if sample.text else "<EMPTY>"
        full_probs = _cached_prob_fn(text_for_pred, verbalizers)
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

        rng = random.Random(2026 + (_stable_hash_int(sample_id) % 10_000))
        random_rankings = [
            _random_ranking(word_ids, rng) for _ in range(max(1, int(random_trials)))
        ]

        grad_rank_cache: Dict[int, List[int]] = {}
        grad_error_cache: Dict[int, str] = {}

        for mode_name, target_label in (("gold", sample.label), ("predicted", pred_label)):
            state = mode_states[mode_name]

            ours_metrics, per_q = aml_faithfulness_metrics(
                chunks=word_units,
                ranking=ranking_ours,
                target_label=target_label,
                verbalizers=verbalizers,
                prob_fn=_cached_prob_fn,
                aopc_q_values=aopc_q_values,
                extra_q_values=tracked_q_values,
                reference_token_text=reference_token_text,
            )
            state["ours_log_odds"].append(float(ours_metrics["log_odds"]))
            state["ours_comp"].append(float(ours_metrics["comprehensiveness"]))
            state["ours_suff"].append(float(ours_metrics["sufficiency"]))
            state["ours_aopc_suff"].append(float(ours_metrics["aopc_sufficiency"]))
            state["ours_aopc_comp"].append(float(ours_metrics["aopc_comprehensiveness"]))
            ours_curve = aopc_metrics(
                chunks=word_units,
                ranking=ranking_ours,
                target_label=target_label,
                verbalizers=verbalizers,
                prob_fn=_cached_prob_fn,
            )
            state["ours_aopc"].append(float(ours_curve["aopc"]))
            state["ours_del_auc"].append(float(ours_curve["deletion_auc"]))
            state["ours_ins_auc"].append(float(ours_curve["insertion_auc"]))
            for q in tracked_q_values:
                q_int = int(q)
                state["per_q"]["ours"]["comprehensiveness"][q_int].append(float(per_q[q_int]["comp"]))
                state["per_q"]["ours"]["sufficiency"][q_int].append(float(per_q[q_int]["suff"]))

            trial_log_odds: List[float] = []
            trial_comp: List[float] = []
            trial_suff: List[float] = []
            trial_aopc_suff: List[float] = []
            trial_aopc_comp: List[float] = []
            trial_per_q_comp = {int(q): [] for q in tracked_q_values}
            trial_per_q_suff = {int(q): [] for q in tracked_q_values}

            for rr in random_rankings:
                rm, pq = aml_faithfulness_metrics(
                    chunks=word_units,
                    ranking=rr,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=_cached_prob_fn,
                    aopc_q_values=aopc_q_values,
                    extra_q_values=tracked_q_values,
                    reference_token_text=reference_token_text,
                )
                trial_log_odds.append(float(rm["log_odds"]))
                trial_comp.append(float(rm["comprehensiveness"]))
                trial_suff.append(float(rm["sufficiency"]))
                trial_aopc_suff.append(float(rm["aopc_sufficiency"]))
                trial_aopc_comp.append(float(rm["aopc_comprehensiveness"]))
                for q in tracked_q_values:
                    q_int = int(q)
                    trial_per_q_comp[q_int].append(float(pq[q_int]["comp"]))
                    trial_per_q_suff[q_int].append(float(pq[q_int]["suff"]))

            avg_rlo = float(mean(trial_log_odds))
            avg_rc = float(mean(trial_comp))
            avg_rs = float(mean(trial_suff))
            avg_ras = float(mean(trial_aopc_suff))
            avg_rac = float(mean(trial_aopc_comp))
            state["random_log_odds"].append(avg_rlo)
            state["random_comp"].append(avg_rc)
            state["random_suff"].append(avg_rs)
            state["random_aopc_suff"].append(avg_ras)
            state["random_aopc_comp"].append(avg_rac)
            for q in tracked_q_values:
                q_int = int(q)
                state["per_q"]["random"]["comprehensiveness"][q_int].append(float(mean(trial_per_q_comp[q_int])))
                state["per_q"]["random"]["sufficiency"][q_int].append(float(mean(trial_per_q_suff[q_int])))

            if ours_metrics["comprehensiveness"] > avg_rc and ours_metrics["sufficiency"] < avg_rs:
                state["diagnosticity_hits"] += 1

            if include_gradient_baseline:
                if target_label not in grad_rank_cache and target_label not in grad_error_cache:
                    try:
                        grad_rank_cache[target_label] = _gradient_ranking_with_retry(
                            backbone=backbone,
                            text=sample.text,
                            chunks=word_units,
                            label=target_label,
                            verbalizers=verbalizers,
                            max_oom_retries=1,
                        )
                    except Exception as exc:
                        grad_error_cache[target_label] = f"{type(exc).__name__}: {exc}"

                if target_label in grad_rank_cache:
                    grad_rank = grad_rank_cache[target_label]
                    gm, gpq = aml_faithfulness_metrics(
                        chunks=word_units,
                        ranking=grad_rank,
                        target_label=target_label,
                        verbalizers=verbalizers,
                        prob_fn=_cached_prob_fn,
                        aopc_q_values=aopc_q_values,
                        extra_q_values=tracked_q_values,
                        reference_token_text=reference_token_text,
                    )
                    state["grad_log_odds"].append(float(gm["log_odds"]))
                    state["grad_comp"].append(float(gm["comprehensiveness"]))
                    state["grad_suff"].append(float(gm["sufficiency"]))
                    state["grad_aopc_suff"].append(float(gm["aopc_sufficiency"]))
                    state["grad_aopc_comp"].append(float(gm["aopc_comprehensiveness"]))
                    state["grad_evaluated"] += 1
                    for q in tracked_q_values:
                        q_int = int(q)
                        state["per_q"]["gradient"]["comprehensiveness"][q_int].append(float(gpq[q_int]["comp"]))
                        state["per_q"]["gradient"]["sufficiency"][q_int].append(float(gpq[q_int]["suff"]))
                else:
                    err = grad_error_cache[target_label]
                    state["grad_error_counts"][err] = state["grad_error_counts"].get(err, 0) + 1
                    if len(state["grad_error_examples"]) < 10:
                        state["grad_error_examples"].append({"sample_id": sample_id, "error": err})

        prob_cache_hits_total += prob_cache_hits
        prob_cache_misses_total += prob_cache_misses
        prob_cache_unique_texts_total += len(prob_cache)

    elapsed = time.time() - t0
    counter_after = backbone.snapshot_counters()
    counter_delta = {
        k: int(counter_after.get(k, 0) - counter_before.get(k, 0))
        for k in set(counter_before.keys()).union(counter_after.keys())
    }

    def _mode_report(mode_name: str) -> Dict:
        state = mode_states[mode_name]
        return {
            "metrics_primary": {
                "log_odds": _safe_mean(state["ours_log_odds"]),
                "comprehensiveness": _safe_mean(state["ours_comp"]),
                "sufficiency": _safe_mean(state["ours_suff"]),
                "aopc_sufficiency": _safe_mean(state["ours_aopc_suff"]),
                "aopc_comprehensiveness": _safe_mean(state["ours_aopc_comp"]),
                "aopc": _safe_mean(state["ours_aopc"]),
                "deletion_auc": _safe_mean(state["ours_del_auc"]),
                "insertion_auc": _safe_mean(state["ours_ins_auc"]),
            },
            "diagnosticity_vs_random": (state["diagnosticity_hits"] / total) if total > 0 else 0.0,
            "baselines": {
                "random": {
                    "log_odds": _safe_mean(state["random_log_odds"]),
                    "comprehensiveness": _safe_mean(state["random_comp"]),
                    "sufficiency": _safe_mean(state["random_suff"]),
                    "aopc_sufficiency": _safe_mean(state["random_aopc_suff"]),
                    "aopc_comprehensiveness": _safe_mean(state["random_aopc_comp"]),
                },
                "gradient": {
                    "enabled": bool(include_gradient_baseline),
                    "log_odds": _safe_mean(state["grad_log_odds"]),
                    "comprehensiveness": _safe_mean(state["grad_comp"]),
                    "sufficiency": _safe_mean(state["grad_suff"]),
                    "aopc_sufficiency": _safe_mean(state["grad_aopc_suff"]),
                    "aopc_comprehensiveness": _safe_mean(state["grad_aopc_comp"]),
                    "evaluated_samples": int(state["grad_evaluated"]),
                    "failed_samples": max(0, total - int(state["grad_evaluated"]))
                    if include_gradient_baseline
                    else 0,
                    "error_type_counts": state["grad_error_counts"],
                    "error_examples": state["grad_error_examples"],
                },
            },
            "per_q": {
                "ours": _mean_per_q(state["per_q"]["ours"]),
                "random": _mean_per_q(state["per_q"]["random"]),
                "gradient": _mean_per_q(state["per_q"]["gradient"]),
            },
        }

    mode_reports = {
        "gold": _mode_report("gold"),
        "predicted": _mode_report("predicted"),
    }

    report = {
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
        # Backward-compatible top-level fields: keep gold-target as default.
        "metrics_primary": {
            "accuracy_full": (acc_hits / total) if total > 0 else 0.0,
            **mode_reports["gold"]["metrics_primary"],
        },
        "metrics_secondary": {
            "diagnosticity_vs_random": mode_reports["gold"]["diagnosticity_vs_random"],
            "sparsity": _safe_mean(sparsity_values),
            "plausibility_f1": _safe_mean(plaus_f1_values),
            "plausibility_iou": _safe_mean(plaus_iou_values),
            "runtime_seconds": elapsed,
            "forward_counters_delta": counter_delta,
            "eval_prob_cache_stats": {
                "hits": int(prob_cache_hits_total),
                "misses": int(prob_cache_misses_total),
                "unique_texts": int(prob_cache_unique_texts_total),
                "hit_rate": (
                    float(prob_cache_hits_total / (prob_cache_hits_total + prob_cache_misses_total))
                    if (prob_cache_hits_total + prob_cache_misses_total) > 0
                    else 0.0
                ),
            },
        },
        "baselines": mode_reports["gold"]["baselines"],
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
