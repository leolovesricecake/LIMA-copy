from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..chunking.utils import compose_text_from_chunk_ids
from ..types import TextChunk
from ..utils import safe_log


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


def build_perturbation_plan(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
    *,
    primary_q_percent: int = AML_PRIMARY_Q_PERCENT,
    reference_token_text: str = DEFAULT_REFERENCE_TOKEN_TEXT,
) -> Dict[str, Any]:
    all_ids = _all_chunk_ids(chunks)
    full_text = _nonempty_text(compose_text_from_chunk_ids(chunks, all_ids))

    required_texts: List[str] = []
    seen = set()
    required_text_count = 0

    def _add_required(text: str) -> None:
        nonlocal required_text_count
        required_text_count += 1
        if text in seen:
            return
        seen.add(text)
        required_texts.append(text)

    _add_required(full_text)

    q_payload: Dict[int, Dict[str, Any]] = {}
    q_ints = tuple(sorted(set(int(q) for q in q_values)))
    for q in q_ints:
        top_ids = ranking_to_top_ids(ranking, len(chunks), q)
        if not top_ids:
            remove_text = full_text
            keep_text = full_text
        else:
            top_set = set(int(x) for x in top_ids)
            removed_ids = [i for i in all_ids if i not in top_set]
            remove_text = _nonempty_text(compose_text_from_chunk_ids(chunks, removed_ids))
            keep_text = _nonempty_text(compose_text_from_chunk_ids(chunks, top_ids))

        q_payload[q] = {
            "top_ids": list(top_ids),
            "top_count": len(top_ids),
            "remove_text": remove_text,
            "keep_text": keep_text,
        }
        _add_required(remove_text)
        _add_required(keep_text)

    log_odds_top_ids = ranking_to_top_ids(ranking, len(chunks), int(primary_q_percent))
    log_odds_text = None
    if log_odds_top_ids:
        log_odds_text = _nonempty_text(
            _compose_text_replacing_chunk_ids(
                chunks=chunks,
                chunk_ids=log_odds_top_ids,
                replacement_text=reference_token_text or DEFAULT_REFERENCE_TOKEN_TEXT,
            )
        )
        _add_required(log_odds_text)

    aopc_deletion_texts: List[str] = []
    m = len(all_ids)
    for step in range(0, m + 1):
        top = list(ranking[:step])
        keep_after_delete = [i for i in all_ids if i not in set(top)]
        text_del = _nonempty_text(compose_text_from_chunk_ids(chunks, keep_after_delete))
        aopc_deletion_texts.append(text_del)
        _add_required(text_del)

    return {
        "all_ids": list(all_ids),
        "full_text": full_text,
        "q_payload": q_payload,
        "primary_q_percent": int(primary_q_percent),
        "log_odds_top_ids": list(log_odds_top_ids),
        "log_odds_text": log_odds_text,
        "aopc_deletion_texts": list(aopc_deletion_texts),
        "required_texts": list(required_texts),
        "required_text_count": int(required_text_count),
        "unique_required_text_count": len(required_texts),
    }


def perturbation_scores_by_q(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    q_values: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
    perturbation_plan: Dict[str, Any] | None = None,
) -> Dict[int, Dict[str, float]]:
    if perturbation_plan is None:
        perturbation_plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=q_values,
            primary_q_percent=AML_PRIMARY_Q_PERCENT,
        )

    full_text = str(
        perturbation_plan.get("full_text")
        if perturbation_plan.get("full_text") is not None
        else _nonempty_text(compose_text_from_chunk_ids(chunks, _all_chunk_ids(chunks)))
    )
    full_probs = prob_fn(full_text, verbalizers)
    p_full = float(full_probs[target_label])

    per_q: Dict[int, Dict[str, float]] = {}
    q_payload = perturbation_plan.get("q_payload", {})
    all_ids = perturbation_plan.get("all_ids") or _all_chunk_ids(chunks)
    for q in q_values:
        q_int = int(q)
        payload = q_payload.get(q_int)
        if payload is not None:
            top_ids = list(payload.get("top_ids", []))
            top_count = int(payload.get("top_count", len(top_ids)))
            text_remove_top = str(payload.get("remove_text", full_text))
            text_keep_top = str(payload.get("keep_text", full_text))
        else:
            top_ids = ranking_to_top_ids(ranking, len(chunks), q_int)
            top_count = len(top_ids)
            if top_count > 0:
                top_set = set(top_ids)
                removed_ids = [i for i in all_ids if i not in top_set]
                text_remove_top = _nonempty_text(compose_text_from_chunk_ids(chunks, removed_ids))
                text_keep_top = _nonempty_text(compose_text_from_chunk_ids(chunks, top_ids))
            else:
                text_remove_top = full_text
                text_keep_top = full_text

        if top_count == 0:
            p_remove = p_full
            p_keep = p_full
        else:
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
            "top_count": top_count,
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
    perturbation_plan: Dict[str, Any] | None = None,
) -> float:
    if perturbation_plan is None:
        perturbation_plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=(),
            primary_q_percent=int(q_percent),
            reference_token_text=reference_token_text,
        )

    full_text = str(
        perturbation_plan.get("full_text")
        if perturbation_plan.get("full_text") is not None
        else _nonempty_text(compose_text_from_chunk_ids(chunks, _all_chunk_ids(chunks)))
    )
    p_full = float(prob_fn(full_text, verbalizers)[target_label])

    top_ids = []
    perturbed_text = None
    if int(perturbation_plan.get("primary_q_percent", q_percent)) == int(q_percent):
        top_ids = list(perturbation_plan.get("log_odds_top_ids", []))
        perturbed_text = perturbation_plan.get("log_odds_text")
    if not top_ids:
        top_ids = ranking_to_top_ids(ranking, len(chunks), int(q_percent))

    if len(top_ids) == 0:
        return 0.0

    if perturbed_text is None:
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
    perturbation_plan: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    if perturbation_plan is None:
        perturbation_plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=(),
            primary_q_percent=AML_PRIMARY_Q_PERCENT,
        )

    all_ids = perturbation_plan.get("all_ids") or _all_chunk_ids(chunks)
    m = len(all_ids)
    if m == 0:
        return {"aopc": 0.0}

    del_texts = perturbation_plan.get("aopc_deletion_texts")
    if not isinstance(del_texts, list) or len(del_texts) != (m + 1):
        del_texts = []
        for step in range(0, m + 1):
            top = list(ranking[:step])
            keep_after_delete = [i for i in all_ids if i not in set(top)]
            del_texts.append(_nonempty_text(compose_text_from_chunk_ids(chunks, keep_after_delete)))

    del_probs = [float(prob_fn(str(text_del), verbalizers)[target_label]) for text_del in del_texts]

    p_full = float(
        prob_fn(
            str(
                perturbation_plan.get("full_text")
                if perturbation_plan.get("full_text") is not None
                else _nonempty_text(compose_text_from_chunk_ids(chunks, all_ids))
            ),
            verbalizers,
        )[target_label]
    )
    aopc = float(np.mean([p_full - p for p in del_probs]))

    return {"aopc": aopc}


def deletion_trajectory(
    chunks: Sequence[TextChunk],
    ranking: Sequence[int],
    target_label: int,
    verbalizers: Sequence[str],
    prob_fn,
    perturbation_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if perturbation_plan is None:
        perturbation_plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=(),
            primary_q_percent=AML_PRIMARY_Q_PERCENT,
        )

    all_ids = [int(x) for x in (perturbation_plan.get("all_ids") or _all_chunk_ids(chunks))]
    deletion_texts = perturbation_plan.get("aopc_deletion_texts")
    m = len(all_ids)

    full_text = str(
        perturbation_plan.get("full_text")
        if perturbation_plan.get("full_text") is not None
        else _nonempty_text(compose_text_from_chunk_ids(chunks, all_ids))
    )
    p_full = float(prob_fn(full_text, verbalizers)[target_label])

    if not isinstance(deletion_texts, list) or len(deletion_texts) != (m + 1):
        deletion_texts = []
        for step in range(0, m + 1):
            deleted_ids = [int(x) for x in ranking[:step]]
            keep_after_delete = [i for i in all_ids if i not in set(deleted_ids)]
            deletion_texts.append(_nonempty_text(compose_text_from_chunk_ids(chunks, keep_after_delete)))

    points: List[Dict[str, Any]] = []
    for step_index, text_after_delete in enumerate(deletion_texts):
        deleted_ids = [int(x) for x in ranking[:step_index]]
        if step_index == 0:
            p_step = p_full
        else:
            p_step = float(prob_fn(str(text_after_delete), verbalizers)[target_label])
        delete_fraction = float(step_index / m) if m > 0 else 0.0
        remaining_fraction = float((m - step_index) / m) if m > 0 else 1.0
        points.append(
            {
                "step_index": int(step_index),
                "total_steps": int(m),
                "delete_count": int(step_index),
                "delete_fraction": delete_fraction,
                "remaining_fraction": remaining_fraction,
                "target_probability": float(p_step),
                "prob_drop_from_full": float(p_full - p_step),
                "is_full_text_step": bool(step_index == 0),
                "deleted_ids": list(deleted_ids),
            }
        )

    aopc = float(np.mean([point["prob_drop_from_full"] for point in points])) if points else 0.0
    return {
        "aopc": aopc,
        "full_probability": p_full,
        "points": points,
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
    perturbation_plan: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    tracked_q_values = tuple(sorted(set(int(q) for q in (*aopc_q_values, primary_q_percent, *extra_q_values))))
    needs_plan = perturbation_plan is None
    if not needs_plan and perturbation_plan is not None:
        q_payload = perturbation_plan.get("q_payload", {})
        if int(perturbation_plan.get("primary_q_percent", primary_q_percent)) != int(primary_q_percent):
            needs_plan = True
        elif any(int(q) not in q_payload for q in tracked_q_values):
            needs_plan = True
    if needs_plan:
        perturbation_plan = build_perturbation_plan(
            chunks=chunks,
            ranking=ranking,
            q_values=tracked_q_values,
            primary_q_percent=primary_q_percent,
            reference_token_text=reference_token_text,
        )

    per_q = perturbation_scores_by_q(
        chunks=chunks,
        ranking=ranking,
        q_values=tracked_q_values,
        target_label=target_label,
        verbalizers=verbalizers,
        prob_fn=prob_fn,
        perturbation_plan=perturbation_plan,
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
            perturbation_plan=perturbation_plan,
        ),
        "sufficiency": float(per_q[int(primary_q_percent)]["suff"]),
        "comprehensiveness": float(per_q[int(primary_q_percent)]["comp"]),
        "aopc_sufficiency": aml_aopc_from_per_q(per_q, aopc_q_values, "suff"),
        "aopc_comprehensiveness": aml_aopc_from_per_q(per_q, aopc_q_values, "comp"),
    }, per_q
