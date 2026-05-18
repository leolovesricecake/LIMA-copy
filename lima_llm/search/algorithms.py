from __future__ import annotations

import time
from typing import List, Sequence, Tuple

from ..objective.submodular import TextSubmodularObjective
from ..types import ScoreTrace


def _argmax_with_tiebreak(items: List[Tuple[int, float]]) -> Tuple[int, float]:
    # Deterministic: maximize value, then minimize chunk id.
    return max(items, key=lambda x: (x[1], -x[0]))


def _argmin_with_tiebreak(items: List[Tuple[int, float]]) -> Tuple[int, float]:
    # Deterministic: minimize value, then minimize chunk id.
    return min(items, key=lambda x: (x[1], x[0]))


def _init_search_profile(
    profile: dict | None,
    *,
    search_name: str,
    candidate_count: int,
    target_k: int,
) -> dict:
    if profile is None:
        profile = {}
    profile.setdefault("search_name", str(search_name))
    profile.setdefault("candidate_count", int(candidate_count))
    profile.setdefault("target_k", int(target_k))
    profile.setdefault("steps_completed", 0)
    profile.setdefault("candidates_evaluated_total", 0)
    profile.setdefault("remaining_build_calls", 0)
    profile.setdefault("remaining_build_seconds", 0.0)
    profile.setdefault("gains_eval_calls", 0)
    profile.setdefault("gains_eval_seconds", 0.0)
    profile.setdefault("argmax_calls", 0)
    profile.setdefault("argmax_seconds", 0.0)
    profile.setdefault("trace_write_calls", 0)
    profile.setdefault("trace_write_seconds", 0.0)
    profile.setdefault("evaluate_gains_path_calls", 0)
    profile.setdefault("evaluate_gain_path_calls", 0)
    return profile


def run_forward_greedy(
    objective: TextSubmodularObjective,
    candidate_ids: Sequence[int],
    k: int,
    profile: dict | None = None,
) -> Tuple[List[int], List[ScoreTrace]]:
    selected: List[int] = []
    traces: List[ScoreTrace] = []
    remaining: List[int] = [int(cid) for cid in candidate_ids]
    max_k = min(max(0, int(k)), len(remaining))
    search_profile = _init_search_profile(
        profile,
        search_name="greedy",
        candidate_count=len(remaining),
        target_k=max_k,
    )

    for step in range(max_k):
        t_remaining0 = time.perf_counter()
        search_profile["remaining_build_calls"] = int(search_profile["remaining_build_calls"]) + 1
        remaining_count = len(remaining)
        search_profile["remaining_build_seconds"] = float(search_profile["remaining_build_seconds"]) + (
            time.perf_counter() - t_remaining0
        )
        if not remaining:
            break

        t_gain0 = time.perf_counter()
        search_profile["gains_eval_calls"] = int(search_profile["gains_eval_calls"]) + 1
        candidate_rows: List[Tuple[int, float, object]] = []
        if hasattr(objective, "evaluate_gains"):
            _, batch = objective.evaluate_gains(selected, remaining)
            search_profile["evaluate_gains_path_calls"] = int(search_profile["evaluate_gains_path_calls"]) + 1
            for cid, gain, aug in batch:
                candidate_rows.append((int(cid), float(gain), aug))
        else:
            search_profile["evaluate_gain_path_calls"] = int(search_profile["evaluate_gain_path_calls"]) + 1
            for cid in remaining:
                gain, _, aug = objective.evaluate_gain(selected, cid)
                candidate_rows.append((int(cid), float(gain), aug))
        search_profile["candidates_evaluated_total"] = int(search_profile["candidates_evaluated_total"]) + int(
            remaining_count
        )
        search_profile["gains_eval_seconds"] = float(search_profile["gains_eval_seconds"]) + (
            time.perf_counter() - t_gain0
        )

        t_argmax0 = time.perf_counter()
        search_profile["argmax_calls"] = int(search_profile["argmax_calls"]) + 1
        best_id = -1
        best_gain = float("-inf")
        best_score = None
        for cid, gain, score in candidate_rows:
            if (gain > best_gain) or (gain == best_gain and (best_id < 0 or cid < best_id)):
                best_id = cid
                best_gain = gain
                best_score = score
        if best_score is None or best_id < 0:
            raise RuntimeError("No valid candidate row found during greedy argmax")
        search_profile["argmax_seconds"] = float(search_profile["argmax_seconds"]) + (time.perf_counter() - t_argmax0)

        t_trace0 = time.perf_counter()
        search_profile["trace_write_calls"] = int(search_profile["trace_write_calls"]) + 1
        selected.append(int(best_id))
        for idx, cid in enumerate(remaining):
            if cid == best_id:
                del remaining[idx]
                break
        traces.append(
            ScoreTrace(
                step=step,
                selected_chunk_id=best_id,
                marginal_gain=float(best_gain),
                total_score=float(best_score.total),
                components=best_score.components,
            )
        )
        search_profile["trace_write_seconds"] = float(search_profile["trace_write_seconds"]) + (
            time.perf_counter() - t_trace0
        )
        search_profile["steps_completed"] = int(search_profile["steps_completed"]) + 1

    return selected, traces


def run_bidirectional_search(
    objective: TextSubmodularObjective,
    candidate_ids: Sequence[int],
    k: int,
    profile: dict | None = None,
) -> Tuple[List[int], List[ScoreTrace]]:
    selected: List[int] = []
    selected_set = set()
    removed: List[int] = []
    removed_set = set()
    traces: List[ScoreTrace] = []

    max_k = min(max(0, int(k)), len(candidate_ids))
    search_profile = _init_search_profile(
        profile,
        search_name="bidirectional",
        candidate_count=len(candidate_ids),
        target_k=max_k,
    )
    step = 0

    while len(selected) < max_k:
        t_remaining0 = time.perf_counter()
        search_profile["remaining_build_calls"] = int(search_profile["remaining_build_calls"]) + 1
        remaining = [idx for idx in candidate_ids if idx not in selected_set and idx not in removed_set]
        search_profile["remaining_build_seconds"] = float(search_profile["remaining_build_seconds"]) + (
            time.perf_counter() - t_remaining0
        )
        if not remaining:
            break

        # Forward add: exact marginal-gain scan.
        t_gain0 = time.perf_counter()
        search_profile["gains_eval_calls"] = int(search_profile["gains_eval_calls"]) + 1
        add_gains = []
        add_scores = {}
        if hasattr(objective, "evaluate_gains"):
            _, batch = objective.evaluate_gains(selected, remaining)
            search_profile["evaluate_gains_path_calls"] = int(search_profile["evaluate_gains_path_calls"]) + 1
            for cid, gain, aug in batch:
                add_gains.append((cid, gain))
                add_scores[cid] = aug
        else:
            search_profile["evaluate_gain_path_calls"] = int(search_profile["evaluate_gain_path_calls"]) + 1
            for cid in remaining:
                gain, _, aug = objective.evaluate_gain(selected, cid)
                add_gains.append((cid, gain))
                add_scores[cid] = aug
        search_profile["candidates_evaluated_total"] = int(search_profile["candidates_evaluated_total"]) + int(
            len(remaining)
        )
        search_profile["gains_eval_seconds"] = float(search_profile["gains_eval_seconds"]) + (
            time.perf_counter() - t_gain0
        )

        t_argmax0 = time.perf_counter()
        search_profile["argmax_calls"] = int(search_profile["argmax_calls"]) + 1
        add_id, add_gain = _argmax_with_tiebreak(add_gains)
        search_profile["argmax_seconds"] = float(search_profile["argmax_seconds"]) + (time.perf_counter() - t_argmax0)
        t_trace0 = time.perf_counter()
        search_profile["trace_write_calls"] = int(search_profile["trace_write_calls"]) + 1
        selected.append(add_id)
        selected_set.add(add_id)
        add_score = add_scores[add_id]
        traces.append(
            ScoreTrace(
                step=step,
                selected_chunk_id=add_id,
                marginal_gain=float(add_gain),
                total_score=float(add_score.total),
                components=add_score.components,
            )
        )
        search_profile["trace_write_seconds"] = float(search_profile["trace_write_seconds"]) + (
            time.perf_counter() - t_trace0
        )
        search_profile["steps_completed"] = int(search_profile["steps_completed"]) + 1
        step += 1
        if len(selected) >= max_k:
            break

        # Reverse prune: exact contribution scan on current universe minus removed.
        remaining_after_add = [idx for idx in candidate_ids if idx not in selected_set and idx not in removed_set]
        if not remaining_after_add:
            break

        universe = [idx for idx in candidate_ids if idx not in removed_set]
        score_universe = objective.evaluate_subset(universe)

        contributions: List[Tuple[int, float]] = []
        for cid in remaining_after_add:
            score_pruned = objective.evaluate_subset([idx for idx in universe if idx != cid])
            contribution = score_universe.total - score_pruned.total
            contributions.append((cid, float(contribution)))

        remove_id, _ = _argmin_with_tiebreak(contributions)
        removed.append(remove_id)
        removed_set.add(remove_id)

    return selected, traces
