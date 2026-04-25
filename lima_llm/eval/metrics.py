from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..chunking.utils import compose_text_from_chunk_ids
from ..types import TextChunk
from ..utils import safe_log, trapezoid_auc


AML_PRIMARY_Q_PERCENT = 20
AML_AOPC_Q_VALUES = (1, 5, 10, 20, 50)
EMPTY_PERTURBATION_TEXT = "<EMPTY>"
DEFAULT_REFERENCE_TOKEN_TEXT = "<UNK>"


def top_percent_chunk_count(total_chunks: int, q_percent: int) -> int:
    if total_chunks <= 0:
        return 0
    return int(math.floor((q_percent / 100.0) * total_chunks))


def ranking_to_top_ids(ranking: Sequence[int], total_chunks: int, q_percent: int) -> List[int]:
    count = top_percent_chunk_count(total_chunks=total_chunks, q_percent=q_percent)
    return list(ranking[:count])


def _nonempty_text(text: str) -> str:
    return text if text != "" else EMPTY_PERTURBATION_TEXT


def _all_chunk_ids(chunks: Sequence[TextChunk]) -> List[int]:
    return [chunk.chunk_id for chunk in chunks]


def _compose_text_replacing_chunk_ids(
    chunks: Sequence[TextChunk],
    chunk_ids: Sequence[int],
    replacement_text: str,
) -> str:
    replace_ids = {int(i) for i in chunk_ids}
    parts: List[str] = []
    for chunk in sorted(chunks, key=lambda c: (c.start_char, c.chunk_id)):
        if chunk.chunk_id not in replace_ids:
            parts.append(chunk.text)
            continue

        leading_len = len(chunk.text) - len(chunk.text.lstrip())
        trailing_len = len(chunk.text) - len(chunk.text.rstrip())
        leading = chunk.text[:leading_len]
        trailing = chunk.text[len(chunk.text) - trailing_len :] if trailing_len > 0 else ""
        parts.append(f"{leading}{replacement_text}{trailing}")
    return "".join(parts)


def perturbation_scores_by_q(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
) -> Dict[int, Dict[str, float]]:
    full_text = _nonempty_text(compose_text_from_chunk_ids(chunks, _all_chunk_ids(chunks)))
    full_probs = prob_fn(full_text, verbalizers)
    p_full = float(full_probs[target_label])

    per_q: Dict[int, Dict[str, float]] = {}

    all_ids = _all_chunk_ids(chunks)
    for q in q_values:
        q_int = int(q)
        top_ids = ranking_to_top_ids(ranking, len(chunks), q_int)
        if len(top_ids) == 0:
            p_remove = p_full
            p_keep = p_full
        else:
            top_set = set(top_ids)
            removed_ids = [i for i in all_ids if i not in top_set]

            text_remove_top = _nonempty_text(compose_text_from_chunk_ids(chunks, removed_ids))
            text_keep_top = _nonempty_text(compose_text_from_chunk_ids(chunks, top_ids))

            p_remove = float(prob_fn(text_remove_top, verbalizers)[target_label])
            p_keep = float(prob_fn(text_keep_top, verbalizers)[target_label])

        comp = p_full - p_remove
        suff = p_full - p_keep
        per_q[q_int] = {
            "p_full": p_full,
            "p_remove": p_remove,
            "p_keep": p_keep,
            "comp": comp,
            "suff": suff,
            "top_count": len(top_ids),
        }

    return per_q


def comprehensiveness_and_sufficiency(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
    per_q = perturbation_scores_by_q(
        chunks=chunks,
        ranking=ranking,
        q_values=q_values,
        target_label=target_label,
        verbalizers=verbalizers,
        prob_fn=prob_fn,
    )
    comp_vals = [float(item["comp"]) for item in per_q.values()]
    suff_vals = [float(item["suff"]) for item in per_q.values()]
    return float(np.mean(comp_vals)) if comp_vals else 0.0, float(np.mean(suff_vals)) if suff_vals else 0.0, per_q


def log_odds(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_percent: int,
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
    reference_token_text: str = DEFAULT_REFERENCE_TOKEN_TEXT,
) -> float:
    full_text = _nonempty_text(compose_text_from_chunk_ids(chunks, _all_chunk_ids(chunks)))
    p_full = float(prob_fn(full_text, verbalizers)[target_label])

    top_ids = ranking_to_top_ids(ranking, len(chunks), int(q_percent))
    if len(top_ids) == 0:
        return 0.0

    perturbed_text = _nonempty_text(
        _compose_text_replacing_chunk_ids(
            chunks=chunks,
            chunk_ids=top_ids,
            replacement_text=reference_token_text or DEFAULT_REFERENCE_TOKEN_TEXT,
        )
    )
    p_perturbed = float(prob_fn(perturbed_text, verbalizers)[target_label])
    return float(safe_log(p_perturbed) - safe_log(p_full))


def aml_aopc_from_per_q(
    per_q: Dict[int, Dict[str, float]],
    q_values: Sequence[int],
    metric_key: str,
) -> float:
    if not q_values:
        return 0.0
    total = sum(float(per_q[int(q)][metric_key]) for q in q_values)
    return float(total / (len(q_values) + 1))


def aopc_metrics(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
) -> Dict[str, float]:
    all_ids = _all_chunk_ids(chunks)
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

        text_del = _nonempty_text(compose_text_from_chunk_ids(chunks, keep_after_delete))
        text_ins = _nonempty_text(compose_text_from_chunk_ids(chunks, top))

        p_del = float(prob_fn(text_del, verbalizers)[target_label])
        p_ins = float(prob_fn(text_ins, verbalizers)[target_label])

        xs.append(frac)
        del_probs.append(p_del)
        ins_probs.append(p_ins)

    deletion_auc = trapezoid_auc(xs, del_probs)
    insertion_auc = trapezoid_auc(xs, ins_probs)

    p_full = float(prob_fn(_nonempty_text(compose_text_from_chunk_ids(chunks, all_ids)), verbalizers)[target_label])
    aopc = float(np.mean([p_full - p for p in del_probs]))

    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "aopc": aopc,
    }


def aml_faithfulness_metrics(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
    primary_q_percent: int = AML_PRIMARY_Q_PERCENT,
    aopc_q_values: Sequence[int] = AML_AOPC_Q_VALUES,
    extra_q_values: Sequence[int] = (),
    reference_token_text: str = DEFAULT_REFERENCE_TOKEN_TEXT,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    tracked_q_values = tuple(sorted(set(int(q) for q in (*aopc_q_values, primary_q_percent, *extra_q_values))))
    per_q = perturbation_scores_by_q(
        chunks=chunks,
        ranking=ranking,
        q_values=tracked_q_values,
        target_label=target_label,
        verbalizers=verbalizers,
        prob_fn=prob_fn,
    )

    return {
        "log_odds": log_odds(
            chunks=chunks,
            ranking=ranking,
            q_percent=primary_q_percent,
            target_label=target_label,
            verbalizers=verbalizers,
            prob_fn=prob_fn,
            reference_token_text=reference_token_text,
        ),
        "sufficiency": float(per_q[int(primary_q_percent)]["suff"]),
        "comprehensiveness": float(per_q[int(primary_q_percent)]["comp"]),
        "aopc_sufficiency": aml_aopc_from_per_q(per_q, aopc_q_values, "suff"),
        "aopc_comprehensiveness": aml_aopc_from_per_q(per_q, aopc_q_values, "comp"),
    }, per_q
