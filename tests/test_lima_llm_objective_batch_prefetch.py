from __future__ import annotations

from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.objective.submodular import ObjectiveWeights, TextSubmodularObjective
from lima_llm.scoring import effectiveness_score
from lima_llm.search.algorithms import run_forward_greedy
from lima_llm.types import TextChunk


def _build_objective(enable_batch_prefetch: bool) -> TextSubmodularObjective:
    text = "alpha beta gamma delta"
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=6, text="alpha "),
        TextChunk(chunk_id=1, start_char=6, end_char=11, text="beta "),
        TextChunk(chunk_id=2, start_char=11, end_char=17, text="gamma "),
        TextChunk(chunk_id=3, start_char=17, end_char=22, text="delta"),
    ]

    backbone = MockBackbone()
    chunk_embeddings = [backbone.embed_text(c.text) for c in chunks]
    return TextSubmodularObjective(
        backbone=backbone,
        text=text,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        target_label=1,
        verbalizers=["NEG", "POS"],
        weights=ObjectiveWeights(1.0, 1.0, 1.0, 1.0),
        enable_batch_prefetch=enable_batch_prefetch,
    )


def test_evaluate_gains_matches_individual_evaluate_gain() -> None:
    objective = _build_objective(enable_batch_prefetch=True)
    subset = [0]
    candidates = [1, 2, 3]

    _, batched = objective.evaluate_gains(subset, candidates)
    by_cid = {cid: (gain, score.total) for cid, gain, score in batched}

    for cid in candidates:
        gain, _, score = objective.evaluate_gain(subset, cid)
        b_gain, b_total = by_cid[cid]
        assert abs(gain - b_gain) <= 1e-9
        assert abs(score.total - b_total) <= 1e-9


def test_batch_prefetch_toggle_keeps_greedy_selection_identical() -> None:
    obj_batch = _build_objective(enable_batch_prefetch=True)
    obj_single = _build_objective(enable_batch_prefetch=False)

    selected_batch, trace_batch = run_forward_greedy(obj_batch, candidate_ids=[0, 1, 2, 3], k=3)
    selected_single, trace_single = run_forward_greedy(obj_single, candidate_ids=[0, 1, 2, 3], k=3)

    assert selected_batch == selected_single
    assert [x.selected_chunk_id for x in trace_batch] == [x.selected_chunk_id for x in trace_single]
    assert len(trace_batch) == len(trace_single)
    for left, right in zip(trace_batch, trace_single):
        assert abs(float(left.total_score) - float(right.total_score)) <= 1e-6


def test_effectiveness_fast_path_matches_reference() -> None:
    objective = _build_objective(enable_batch_prefetch=True)
    subsets = [
        [],
        [0],
        [1],
        [0, 1],
        [0, 2, 3],
        [3, 2, 0],
        [0, 0, 1, 1, 2],
    ]
    for subset in subsets:
        fast = objective._effectiveness_from_subset(subset)
        ref = effectiveness_score(objective.chunk_embeddings, subset)
        assert abs(float(fast) - float(ref)) <= 1e-12
