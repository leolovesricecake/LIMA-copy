from __future__ import annotations

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


def _gradient_ranking(backbone, text: str, chunks: Sequence[TextChunk], label: int, verbalizers: Sequence[str]) -> List[int]:
    scores = backbone.gradient_chunk_importance(text, chunks, label, verbalizers)
    return [
        cid
        for cid, _ in sorted(
            [(chunk.chunk_id, float(scores[chunk.chunk_id])) for chunk in chunks],
            key=lambda x: (-x[1], x[0]),
        )
    ]


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

    ours_comp: List[float] = []
    ours_suff: List[float] = []
    ours_aopc: List[float] = []
    ours_del_auc: List[float] = []
    ours_ins_auc: List[float] = []
    ours_sparsity: List[float] = []
    ours_plaus_f1: List[float] = []
    ours_plaus_iou: List[float] = []

    random_comp: List[float] = []
    random_suff: List[float] = []

    grad_comp: List[float] = []
    grad_suff: List[float] = []

    acc_hits = 0
    total = 0

    diagnosticity_hits = 0

    t0 = time.time()
    counter_before = backbone.snapshot_counters()

    for sample_json in sorted(sample_dir.glob("*.json")):
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        sample_id = payload["sample_id"]
        if sample_id not in sample_map:
            continue

        sample = sample_map[sample_id]
        chunks = _to_chunks(payload["chunks"])
        selected = [int(x) for x in payload.get("selected_chunk_ids", [])]

        if not chunks:
            continue

        total += 1
        all_ids = [c.chunk_id for c in chunks]
        ranking_ours = selected + [cid for cid in all_ids if cid not in set(selected)]

        # Accuracy on full input.
        full_probs = backbone.predict_label_probs(sample.text if sample.text else "<EMPTY>", verbalizers)
        pred = int(np.argmax(full_probs))
        if pred == sample.label:
            acc_hits += 1

        comp, suff, _ = comprehensiveness_and_sufficiency(
            chunks=chunks,
            ranking=ranking_ours,
            q_values=q_values,
            target_label=sample.label,
            verbalizers=verbalizers,
            prob_fn=backbone.predict_label_probs,
        )
        aopc = aopc_metrics(
            chunks=chunks,
            ranking=ranking_ours,
            target_label=sample.label,
            verbalizers=verbalizers,
            prob_fn=backbone.predict_label_probs,
        )
        ours_comp.append(comp)
        ours_suff.append(suff)
        ours_aopc.append(float(aopc["aopc"]))
        ours_del_auc.append(float(aopc["deletion_auc"]))
        ours_ins_auc.append(float(aopc["insertion_auc"]))

        selected_len = chunk_char_length(chunks, selected)
        total_len = max(1, len(sample.text))
        ours_sparsity.append(selected_len / total_len)

        if sample.rationale_char_spans:
            pred_mask = spans_to_char_mask(
                text_len=len(sample.text),
                spans=[(chunks[i].start_char, chunks[i].end_char) for i in selected if i < len(chunks)],
            )
            gold_mask = spans_to_char_mask(
                text_len=len(sample.text),
                spans=sample.rationale_char_spans,
            )
            f1, iou = f1_iou_from_masks(pred_mask, gold_mask)
            ours_plaus_f1.append(f1)
            ours_plaus_iou.append(iou)

        # Random baseline (average over random trials).
        trial_comp = []
        trial_suff = []
        rng = random.Random(2026 + hash(sample_id) % 10_000)
        for _ in range(max(1, random_trials)):
            rr = _random_ranking(all_ids, rng)
            c, s, _ = comprehensiveness_and_sufficiency(
                chunks=chunks,
                ranking=rr,
                q_values=q_values,
                target_label=sample.label,
                verbalizers=verbalizers,
                prob_fn=backbone.predict_label_probs,
            )
            trial_comp.append(c)
            trial_suff.append(s)

        avg_rc = float(mean(trial_comp))
        avg_rs = float(mean(trial_suff))
        random_comp.append(avg_rc)
        random_suff.append(avg_rs)

        if comp > avg_rc and suff < avg_rs:
            diagnosticity_hits += 1

        if include_gradient_baseline:
            try:
                grad_rank = _gradient_ranking(backbone, sample.text, chunks, sample.label, verbalizers)
                gc, gs, _ = comprehensiveness_and_sufficiency(
                    chunks=chunks,
                    ranking=grad_rank,
                    q_values=q_values,
                    target_label=sample.label,
                    verbalizers=verbalizers,
                    prob_fn=backbone.predict_label_probs,
                )
                grad_comp.append(gc)
                grad_suff.append(gs)
            except Exception:
                pass

    elapsed = time.time() - t0
    counter_after = backbone.snapshot_counters()
    counter_delta = {
        k: int(counter_after.get(k, 0) - counter_before.get(k, 0))
        for k in set(counter_before.keys()).union(counter_after.keys())
    }

    def _safe_mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    report = {
        "dataset": bundle.dataset_name,
        "split": bundle.split,
        "sample_count": total,
        "metrics_primary": {
            "accuracy_full": (acc_hits / total) if total > 0 else 0.0,
            "comprehensiveness": _safe_mean(ours_comp),
            "sufficiency": _safe_mean(ours_suff),
            "aopc": _safe_mean(ours_aopc),
            "deletion_auc": _safe_mean(ours_del_auc),
            "insertion_auc": _safe_mean(ours_ins_auc),
        },
        "metrics_secondary": {
            "diagnosticity_vs_random": (diagnosticity_hits / total) if total > 0 else 0.0,
            "sparsity": _safe_mean(ours_sparsity),
            "plausibility_f1": _safe_mean(ours_plaus_f1),
            "plausibility_iou": _safe_mean(ours_plaus_iou),
            "runtime_seconds": elapsed,
            "forward_counters_delta": counter_delta,
        },
        "baselines": {
            "random": {
                "comprehensiveness": _safe_mean(random_comp),
                "sufficiency": _safe_mean(random_suff),
            },
            "gradient": {
                "enabled": bool(include_gradient_baseline),
                "comprehensiveness": _safe_mean(grad_comp),
                "sufficiency": _safe_mean(grad_suff),
                "evaluated_samples": len(grad_comp),
            },
        },
        "q_values": list(int(q) for q in q_values),
    }
    return report
