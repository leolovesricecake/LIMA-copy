from __future__ import annotations

from lima_llm.backbone.mock_backbone import MockBackbone
from lima_llm.objective.submodular import ObjectiveWeights
from lima_llm.pipeline.explainer import ExplainerConfig, TextLIMAExplainer
from lima_llm.types import TextChunk, TextSample


class _StaticChunker:
    def __init__(self) -> None:
        self.last_diagnostics = {
            "chunk_strategy": "unit_test",
            "chunk_count": 2,
            "chunk_len_chars_min": 1.0,
            "chunk_len_chars_mean": 2.0,
            "chunk_len_chars_p90": 2.0,
            "chunk_len_chars_max": 2.0,
            "singleton_orphan_punctuation_chunks": 0,
            "cross_newline_boundary_chunks": 0,
            "fallback_applied": False,
            "fallback_reason": None,
            "pre_fallback_chunk_count": None,
        }

    def __call__(self, text: str):
        return [
            TextChunk(chunk_id=0, start_char=0, end_char=3, text=text[:3]),
            TextChunk(chunk_id=1, start_char=3, end_char=len(text), text=text[3:]),
        ]


def test_explainer_records_component_profile_and_skip_stats() -> None:
    backbone = MockBackbone()
    chunker = _StaticChunker()
    config = ExplainerConfig(
        dataset_name="dummy",
        split="validation",
        k=1,
        search="greedy",
        weights=ObjectiveWeights(0.0, 1.0, 1.0, 1.0),
        explain_method="ours",
        seed=42,
    )
    explainer = TextLIMAExplainer(
        backbone=backbone,
        chunker=chunker,
        verbalizers=["NEG", "POS"],
        config=config,
    )

    sample = TextSample(sample_id="s1", text="abcdef", label=1)
    result = explainer.explain_sample(sample)

    profile = result.metadata.get("component_profile", {})
    enabled = profile.get("component_enabled", {})
    assert enabled["confidence"] is False

    singleton = profile.get("singleton_components", {})
    assert singleton
    for row in singleton.values():
        assert float(row["confidence"]) == 0.0

    compute = result.metadata.get("objective_compute_stats", {})
    assert int(compute.get("confidence_compute_calls", 0)) == 0
    assert int(compute.get("confidence_skipped_due_to_zero_lambda", 0)) > 0

    search_profile = result.metadata.get("search_profile", {})
    assert search_profile
    assert int(search_profile.get("steps_completed", 0)) >= 1
    assert float(search_profile.get("gains_eval_seconds", 0.0)) >= 0.0
    assert float(search_profile.get("argmax_seconds", 0.0)) >= 0.0

    assert float(result.scores["confidence"]) == 0.0
