import numpy as np

from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.objective.submodular import ObjectiveWeights, TextSubmodularObjective
from lima_llm.types import TextChunk


def _build_objective() -> TextSubmodularObjective:
    text = "A. B. C."
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=3, text="A. "),
        TextChunk(chunk_id=1, start_char=3, end_char=6, text="B. "),
        TextChunk(chunk_id=2, start_char=6, end_char=8, text="C."),
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


def test_evaluate_gains_matches_single_evaluate_gain() -> None:
    objective = _build_objective()
    selected = [0]
    candidates = [1, 2]

    _, gain_map, score_map = objective.evaluate_gains(selected, candidates)

    for candidate in candidates:
        gain_single, _, score_single = objective.evaluate_gain(selected, candidate)
        assert np.isclose(gain_map[candidate], gain_single)
        assert np.isclose(score_map[candidate].total, score_single.total)


def test_evaluate_subsets_matches_evaluate_subset() -> None:
    objective = _build_objective()
    subsets = [[], [0], [1], [0, 2]]

    batch_scores = objective.evaluate_subsets(subsets)
    single_scores = [objective.evaluate_subset(subset) for subset in subsets]

    assert len(batch_scores) == len(single_scores)
    for batch_item, single_item in zip(batch_scores, single_scores):
        assert np.isclose(batch_item.total, single_item.total)
