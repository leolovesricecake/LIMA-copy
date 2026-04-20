from dataclasses import dataclass

from lima_llm.search.algorithms import run_forward_greedy
from lima_llm.types import ScoreComponents, SubsetScore


@dataclass
class _FakeObjective:
    def evaluate_subset(self, subset):
        total = float(sum(subset))
        comp = ScoreComponents(0.0, 0.0, 0.0, 0.0)
        return SubsetScore(
            subset_indices=tuple(sorted(set(subset))),
            total=total,
            components=comp,
            target_probability=0.0,
            label_probabilities=(0.5, 0.5),
        )

    def evaluate_gain(self, subset, candidate):
        base = self.evaluate_subset(subset)
        aug = self.evaluate_subset(tuple(list(subset) + [candidate]))
        return float(aug.total - base.total), base, aug


def test_greedy_is_deterministic() -> None:
    objective = _FakeObjective()
    selected1, trace1 = run_forward_greedy(objective, candidate_ids=[0, 1, 2, 3], k=2)
    selected2, trace2 = run_forward_greedy(objective, candidate_ids=[0, 1, 2, 3], k=2)

    assert selected1 == selected2
    assert [t.selected_chunk_id for t in trace1] == [t.selected_chunk_id for t in trace2]
