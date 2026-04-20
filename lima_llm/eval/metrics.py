from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..chunking.utils import complement_chunk_ids, compose_text_from_chunk_ids
from ..types import TextChunk
from ..utils import trapezoid_auc


def top_percent_chunk_count(total_chunks: int, q_percent: int) -> int:
    if total_chunks <= 0:
        return 0
    return max(1, int(math.ceil((q_percent / 100.0) * total_chunks)))


def ranking_to_top_ids(ranking: Sequence[int], total_chunks: int, q_percent: int) -> List[int]:
    count = top_percent_chunk_count(total_chunks=total_chunks, q_percent=q_percent)
    return list(ranking[:count])


def comprehensiveness_and_sufficiency(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
    full_text = compose_text_from_chunk_ids(chunks, [c.chunk_id for c in chunks])
    full_probs = prob_fn(full_text, verbalizers)
    p_full = float(full_probs[target_label])

    per_q: Dict[int, Dict[str, float]] = {}
    comp_vals: List[float] = []
    suff_vals: List[float] = []

    all_ids = [c.chunk_id for c in chunks]
    for q in q_values:
        top_ids = ranking_to_top_ids(ranking, len(chunks), q)
        removed_ids = [i for i in all_ids if i not in set(top_ids)]

        text_remove_top = compose_text_from_chunk_ids(chunks, removed_ids)
        text_keep_top = compose_text_from_chunk_ids(chunks, top_ids)
        if text_remove_top == "":
            text_remove_top = "<EMPTY>"
        if text_keep_top == "":
            text_keep_top = "<EMPTY>"

        p_remove = float(prob_fn(text_remove_top, verbalizers)[target_label])
        p_keep = float(prob_fn(text_keep_top, verbalizers)[target_label])

        comp = p_full - p_remove
        suff = p_full - p_keep
        per_q[int(q)] = {
            "p_full": p_full,
            "p_remove": p_remove,
            "p_keep": p_keep,
            "comp": comp,
            "suff": suff,
        }
        comp_vals.append(comp)
        suff_vals.append(suff)

    return float(np.mean(comp_vals)), float(np.mean(suff_vals)), per_q


def aopc_metrics(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
) -> Dict[str, float]:
    all_ids = [c.chunk_id for c in chunks]
    m = len(all_ids)
    if m == 0:
        return {
            "deletion_auc": 0.0,
            "insertion_auc": 0.0,
            "aopc": 0.0,
        }

    del_probs = []
    ins_probs = []
    xs = []

    for step in range(0, m + 1):
        frac = step / float(m)
        top = list(ranking[:step])
        keep_after_delete = [i for i in all_ids if i not in set(top)]

        text_del = compose_text_from_chunk_ids(chunks, keep_after_delete)
        text_ins = compose_text_from_chunk_ids(chunks, top)
        if text_del == "":
            text_del = "<EMPTY>"
        if text_ins == "":
            text_ins = "<EMPTY>"

        p_del = float(prob_fn(text_del, verbalizers)[target_label])
        p_ins = float(prob_fn(text_ins, verbalizers)[target_label])

        xs.append(frac)
        del_probs.append(p_del)
        ins_probs.append(p_ins)

    deletion_auc = trapezoid_auc(xs, del_probs)
    insertion_auc = trapezoid_auc(xs, ins_probs)

    p_full = float(prob_fn(compose_text_from_chunk_ids(chunks, all_ids), verbalizers)[target_label])
    aopc = float(np.mean([p_full - p for p in del_probs]))

    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "aopc": aopc,
    }
