from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np

from ..chunking.utils import chunk_char_length
from ..types import TextChunk
from ..utils import f1_iou_from_masks, spans_to_char_mask
from .metrics import aopc_metrics, comprehensiveness_and_sufficiency


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


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
        "ours_comp": [],
        "ours_suff": [],
        "ours_aopc": [],
        "ours_del_auc": [],
        "ours_ins_auc": [],
        "random_comp": [],
        "random_suff": [],
        "grad_comp": [],
        "grad_suff": [],
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

    sample_map = {s.sample_id: s for s in bundle.samples}
    mode_states = {
        "gold": _init_mode_state(q_values),
        "predicted": _init_mode_state(q_values),
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
        selected = [int(x) for x in payload.get("selected_chunk_ids", [])]
        if not chunks:
            skipped_empty_chunks += 1
            continue

        total += 1

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

        all_ids = [c.chunk_id for c in chunks]
        ranking_ours = selected + [cid for cid in all_ids if cid not in set(selected)]

        text_for_pred = sample.text if sample.text else "<EMPTY>"
        full_probs = backbone.predict_label_probs(text_for_pred, verbalizers)
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
            _random_ranking(all_ids, rng) for _ in range(max(1, int(random_trials)))
        ]

        grad_rank_cache: Dict[int, List[int]] = {}
        grad_error_cache: Dict[int, str] = {}

        for mode_name, target_label in (("gold", sample.label), ("predicted", pred_label)):
            state = mode_states[mode_name]

            comp, suff, per_q = comprehensiveness_and_sufficiency(
                chunks=chunks,
                ranking=ranking_ours,
                q_values=q_values,
                target_label=target_label,
                verbalizers=verbalizers,
                prob_fn=backbone.predict_label_probs,
            )
            aopc = aopc_metrics(
                chunks=chunks,
                ranking=ranking_ours,
                target_label=target_label,
                verbalizers=verbalizers,
                prob_fn=backbone.predict_label_probs,
            )
            state["ours_comp"].append(comp)
            state["ours_suff"].append(suff)
            state["ours_aopc"].append(float(aopc["aopc"]))
            state["ours_del_auc"].append(float(aopc["deletion_auc"]))
            state["ours_ins_auc"].append(float(aopc["insertion_auc"]))
            for q in q_values:
                q_int = int(q)
                state["per_q"]["ours"]["comprehensiveness"][q_int].append(float(per_q[q_int]["comp"]))
                state["per_q"]["ours"]["sufficiency"][q_int].append(float(per_q[q_int]["suff"]))

            trial_comp: List[float] = []
            trial_suff: List[float] = []
            trial_per_q_comp = {int(q): [] for q in q_values}
            trial_per_q_suff = {int(q): [] for q in q_values}

            for rr in random_rankings:
                c, s, pq = comprehensiveness_and_sufficiency(
                    chunks=chunks,
                    ranking=rr,
                    q_values=q_values,
                    target_label=target_label,
                    verbalizers=verbalizers,
                    prob_fn=backbone.predict_label_probs,
                )
                trial_comp.append(c)
                trial_suff.append(s)
                for q in q_values:
                    q_int = int(q)
                    trial_per_q_comp[q_int].append(float(pq[q_int]["comp"]))
                    trial_per_q_suff[q_int].append(float(pq[q_int]["suff"]))

            avg_rc = float(mean(trial_comp))
            avg_rs = float(mean(trial_suff))
            state["random_comp"].append(avg_rc)
            state["random_suff"].append(avg_rs)
            for q in q_values:
                q_int = int(q)
                state["per_q"]["random"]["comprehensiveness"][q_int].append(float(mean(trial_per_q_comp[q_int])))
                state["per_q"]["random"]["sufficiency"][q_int].append(float(mean(trial_per_q_suff[q_int])))

            if comp > avg_rc and suff < avg_rs:
                state["diagnosticity_hits"] += 1

            if include_gradient_baseline:
                if target_label not in grad_rank_cache and target_label not in grad_error_cache:
                    try:
                        grad_rank_cache[target_label] = _gradient_ranking_with_retry(
                            backbone=backbone,
                            text=sample.text,
                            chunks=chunks,
                            label=target_label,
                            verbalizers=verbalizers,
                            max_oom_retries=1,
                        )
                    except Exception as exc:
                        grad_error_cache[target_label] = f"{type(exc).__name__}: {exc}"

                if target_label in grad_rank_cache:
                    grad_rank = grad_rank_cache[target_label]
                    gc, gs, gpq = comprehensiveness_and_sufficiency(
                        chunks=chunks,
                        ranking=grad_rank,
                        q_values=q_values,
                        target_label=target_label,
                        verbalizers=verbalizers,
                        prob_fn=backbone.predict_label_probs,
                    )
                    state["grad_comp"].append(gc)
                    state["grad_suff"].append(gs)
                    state["grad_evaluated"] += 1
                    for q in q_values:
                        q_int = int(q)
                        state["per_q"]["gradient"]["comprehensiveness"][q_int].append(float(gpq[q_int]["comp"]))
                        state["per_q"]["gradient"]["sufficiency"][q_int].append(float(gpq[q_int]["suff"]))
                else:
                    err = grad_error_cache[target_label]
                    state["grad_error_counts"][err] = state["grad_error_counts"].get(err, 0) + 1
                    if len(state["grad_error_examples"]) < 10:
                        state["grad_error_examples"].append({"sample_id": sample_id, "error": err})

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
                "comprehensiveness": _safe_mean(state["ours_comp"]),
                "sufficiency": _safe_mean(state["ours_suff"]),
                "aopc": _safe_mean(state["ours_aopc"]),
                "deletion_auc": _safe_mean(state["ours_del_auc"]),
                "insertion_auc": _safe_mean(state["ours_ins_auc"]),
            },
            "diagnosticity_vs_random": (state["diagnosticity_hits"] / total) if total > 0 else 0.0,
            "baselines": {
                "random": {
                    "comprehensiveness": _safe_mean(state["random_comp"]),
                    "sufficiency": _safe_mean(state["random_suff"]),
                },
                "gradient": {
                    "enabled": bool(include_gradient_baseline),
                    "comprehensiveness": _safe_mean(state["grad_comp"]),
                    "sufficiency": _safe_mean(state["grad_suff"]),
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
        },
        "q_values": [int(q) for q in q_values],
    }
    return report
