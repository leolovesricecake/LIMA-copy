from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set


@dataclass
class QueryLedger:
    """Logical query accounting for one method run.

    Oracle cache hits still count as logical information requests. Physical
    scoring is tracked separately so experiment order cannot change a method's
    attribution budget.
    """

    name: str
    requested_queries: int = 0
    duplicate_logical_requests: int = 0
    global_cache_hits: int = 0
    global_cache_misses: int = 0
    physical_forwards_caused: int = 0
    _logical_keys: Set[str] = field(default_factory=set, repr=False)
    _category_keys: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    def record_requests(
        self,
        logical_keys: Iterable[str],
        *,
        cache_hits: Iterable[bool],
        category: str,
    ) -> None:
        keys = [str(key) for key in logical_keys]
        hits = [bool(hit) for hit in cache_hits]
        if len(keys) != len(hits):
            raise ValueError("logical_keys and cache_hits must have equal length")
        bucket = self._category_keys.setdefault(str(category), set())
        for key, hit in zip(keys, hits):
            self.requested_queries += 1
            if key in self._logical_keys:
                self.duplicate_logical_requests += 1
            self._logical_keys.add(key)
            bucket.add(key)
            if hit:
                self.global_cache_hits += 1
            else:
                self.global_cache_misses += 1

    def record_physical_values(self, count: int) -> None:
        self.physical_forwards_caused += int(count)

    @property
    def logical_unique_queries(self) -> int:
        return len(self._logical_keys)

    def category_count(self, category: str) -> int:
        return len(self._category_keys.get(str(category), set()))

    def logical_keys(self) -> set[str]:
        return set(self._logical_keys)

    def attribution_keys(self) -> set[str]:
        out: set[str] = set()
        for category, keys in self._category_keys.items():
            if category not in {"evaluation", "interaction_verification"}:
                out.update(keys)
        return out

    def to_dict(self) -> Dict[str, object]:
        categories = {
            str(name): int(len(keys)) for name, keys in sorted(self._category_keys.items())
        }
        return {
            "name": self.name,
            "requested_queries": int(self.requested_queries),
            "logical_unique_queries": int(self.logical_unique_queries),
            "duplicate_logical_requests": int(self.duplicate_logical_requests),
            "global_cache_hits": int(self.global_cache_hits),
            "global_cache_misses": int(self.global_cache_misses),
            "physical_forwards_caused": int(self.physical_forwards_caused),
            "physical_values_scored": int(self.physical_forwards_caused),
            "attribution_budget_used": int(len(self.attribution_keys())),
            "category_unique_queries": categories,
            "training_queries": int(categories.get("training", 0)),
            "candidate_discovery_queries": int(categories.get("candidate_discovery", 0)),
            "adaptive_queries": int(categories.get("adaptive", 0)),
            "interaction_verification_queries": int(
                categories.get("interaction_verification", 0)
            ),
            "evaluation_only_queries": int(categories.get("evaluation", 0)),
        }
