from __future__ import annotations

from typing import List, Sequence, Tuple

from ..objective.submodular import TextSubmodularObjective
from ..types import ScoreTrace


def _argmax_with_tiebreak(items: List[Tuple[int, float]]) -> Tuple[int, float]:
    # Deterministic: maximize value, then minimize chunk id.
    return max(items, key=lambda x: (x[1], -x[0]))


def _argmin_with_tiebreak(items: List[Tuple[int, float]]) -> Tuple[int, float]:
    # Deterministic: minimize value, then minimize chunk id.
    return min(items, key=lambda x: (x[1], x[0]))


def run_forward_greedy(
    objective: TextSubmodularObjective,
    candidate_ids: Sequence[int],
    k: int,
) -> Tuple[List[int], List[ScoreTrace]]:
    selected: List[int] = []
    traces: List[ScoreTrace] = []

    max_k = min(max(0, int(k)), len(candidate_ids))
    for step in range(max_k):
        remaining = [idx for idx in candidate_ids if idx not in selected]
        if not remaining:
            break

        gains: List[Tuple[int, float]] = []
        candidate_scores = {}
        for cid in remaining:
            gain, _, aug = objective.evaluate_gain(selected, cid)
            gains.append((cid, gain))
            candidate_scores[cid] = aug

        best_id, best_gain = _argmax_with_tiebreak(gains)
        selected.append(best_id)

        best_score = candidate_scores[best_id]
        traces.append(
            ScoreTrace(
                step=step,
                selected_chunk_id=best_id,
                marginal_gain=float(best_gain),
                total_score=float(best_score.total),
                components=best_score.components,
            )
        )

    return selected, traces


def run_bidirectional_search(
    objective: TextSubmodularObjective,
    candidate_ids: Sequence[int],
    k: int,
) -> Tuple[List[int], List[ScoreTrace]]:
    selected: List[int] = []
    removed: List[int] = []
    traces: List[ScoreTrace] = []

    max_k = min(max(0, int(k)), len(candidate_ids))
    step = 0

    while len(selected) < max_k:
        remaining = [idx for idx in candidate_ids if idx not in selected and idx not in removed]
        if not remaining:
            break

        # Forward add: exact marginal-gain scan.
        add_gains: List[Tuple[int, float]] = []
        add_scores = {}
        for cid in remaining:
            gain, _, aug = objective.evaluate_gain(selected, cid)
            add_gains.append((cid, gain))
            add_scores[cid] = aug

        add_id, add_gain = _argmax_with_tiebreak(add_gains)
        selected.append(add_id)
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
        step += 1
        if len(selected) >= max_k:
            break

        # Reverse prune: exact contribution scan on current universe minus removed.
        remaining_after_add = [
            idx for idx in candidate_ids if idx not in selected and idx not in removed
        ]
        if not remaining_after_add:
            break

        universe = [idx for idx in candidate_ids if idx not in removed]
        score_universe = objective.evaluate_subset(universe)

        contributions: List[Tuple[int, float]] = []
        for cid in remaining_after_add:
            pruned = [idx for idx in universe if idx != cid]
            score_pruned = objective.evaluate_subset(pruned)
            contribution = score_universe.total - score_pruned.total
            contributions.append((cid, float(contribution)))

        remove_id, _ = _argmin_with_tiebreak(contributions)
        removed.append(remove_id)

    return selected, traces
