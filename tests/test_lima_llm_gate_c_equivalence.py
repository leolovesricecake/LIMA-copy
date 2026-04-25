import numpy as np

from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.objective.submodular import ObjectiveWeights, TextSubmodularObjective
from lima_llm.search.algorithms import run_bidirectional_search, run_forward_greedy
from lima_llm.types import TextChunk


class _NoBatchObjective:
    def __init__(self, objective: TextSubmodularObjective) -> None:
        self.objective = objective

    def evaluate_subset(self, subset):
        return self.objective.evaluate_subset(subset)

    def evaluate_gain(self, subset, candidate):
        return self.objective.evaluate_gain(subset, candidate)


def _build_objective() -> TextSubmodularObjective:
    text = "Alpha clue. Beta bridge. Gamma twist. Delta ending."
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=12, text="Alpha clue. "),
        TextChunk(chunk_id=1, start_char=12, end_char=25, text="Beta bridge. "),
        TextChunk(chunk_id=2, start_char=25, end_char=38, text="Gamma twist. "),
        TextChunk(chunk_id=3, start_char=38, end_char=len(text), text="Delta ending."),
    ]
    backbone = MockBackbone(embedding_dim=32)
    chunk_embeddings = [backbone.embed_text(chunk.text) for chunk in chunks]
    return TextSubmodularObjective(
        backbone=backbone,
        text=text,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        target_label=1,
        verbalizers=["NEG", "POS"],
        weights=ObjectiveWeights(1.0, 1.0, 1.0, 1.0),
    )


def _assert_same_trace(left, right) -> None:
    assert [item.selected_chunk_id for item in left] == [item.selected_chunk_id for item in right]
    assert len(left) == len(right)
    for l_item, r_item in zip(left, right):
        assert np.isclose(l_item.marginal_gain, r_item.marginal_gain)
        assert np.isclose(l_item.total_score, r_item.total_score)
        assert np.isclose(l_item.components.confidence, r_item.components.confidence)
        assert np.isclose(l_item.components.effectiveness, r_item.components.effectiveness)
        assert np.isclose(l_item.components.consistency, r_item.components.consistency)
        assert np.isclose(l_item.components.collaboration, r_item.components.collaboration)


def test_forward_greedy_batched_and_fallback_paths_are_equivalent() -> None:
    candidate_ids = [0, 1, 2, 3]
    batched_selected, batched_trace = run_forward_greedy(
        _build_objective(),
        candidate_ids=candidate_ids,
        k=3,
    )
    fallback_selected, fallback_trace = run_forward_greedy(
        _NoBatchObjective(_build_objective()),
        candidate_ids=candidate_ids,
        k=3,
    )

    assert batched_selected == fallback_selected
    _assert_same_trace(batched_trace, fallback_trace)


def test_bidirectional_batched_and_fallback_paths_are_equivalent() -> None:
    candidate_ids = [0, 1, 2, 3]
    batched_selected, batched_trace = run_bidirectional_search(
        _build_objective(),
        candidate_ids=candidate_ids,
        k=3,
    )
    fallback_selected, fallback_trace = run_bidirectional_search(
        _NoBatchObjective(_build_objective()),
        candidate_ids=candidate_ids,
        k=3,
    )

    assert batched_selected == fallback_selected
    _assert_same_trace(batched_trace, fallback_trace)
