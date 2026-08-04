"""Global physical score cache with method-local logical query ledgers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import numpy as np

from mobius.core.runtime import ensure_dir

from .base import RawTextScorer


def stable_digest(payload: object) -> str:
    """Return a canonical SHA-256 digest."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class QueryLedger:
    """Track logical requests independently from shared-cache physical work."""

    name: str
    requested_queries: int = 0
    duplicate_logical_requests: int = 0
    global_cache_hits: int = 0
    global_cache_misses: int = 0
    physical_values_scored: int = 0
    _logical_keys: Set[str] = field(default_factory=set, repr=False)
    _category_keys: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    def record(
        self,
        logical_keys: Iterable[str],
        cache_hits: Iterable[bool],
        category: str,
    ) -> None:
        """Record one batch of logical requests and cache outcomes."""

        keys = [str(value) for value in logical_keys]
        hits = [bool(value) for value in cache_hits]
        if len(keys) != len(hits):
            raise ValueError("logical keys and cache-hit flags must align.")
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

    def record_physical(self, count: int) -> None:
        """Add physical texts scored because of this ledger."""

        self.physical_values_scored += int(count)

    def category_count(self, category: str) -> int:
        """Return the unique logical count for one query category."""

        return len(self._category_keys.get(str(category), set()))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize compact logical and physical query accounting."""

        categories = {
            name: len(keys) for name, keys in sorted(self._category_keys.items())
        }
        return {
            "name": self.name,
            "requested_queries": int(self.requested_queries),
            "logical_unique_queries": int(len(self._logical_keys)),
            "duplicate_logical_requests": int(self.duplicate_logical_requests),
            "global_cache_hits": int(self.global_cache_hits),
            "global_cache_misses": int(self.global_cache_misses),
            "physical_values_scored": int(self.physical_values_scored),
            "attribution_budget_used": int(categories.get("training", 0)),
            "category_unique_queries": categories,
        }


class ValueOracle:
    """Cache all-class scores by model fingerprint and exact perturbed text."""

    def __init__(
        self,
        scorer: RawTextScorer,
        cache_path: str | Path,
        *,
        model_fingerprint: object,
        batch_size: int = 16,
        prompt_version: str = "task_classification_v1",
    ) -> None:
        """Open a process-safe SQLite cache for one scoring contract."""

        self.scorer = scorer
        self.cache_path = Path(cache_path)
        ensure_dir(self.cache_path.parent)
        self.batch_size = max(1, int(batch_size))
        self.physical_values_scored = 0
        self.scorer_batch_calls = 0
        self.scoring_seconds = 0.0
        self.scoring_fingerprint = stable_digest(
            {
                "model": model_fingerprint,
                "prompt_version": str(prompt_version),
                "verbalizers": [str(value) for value in scorer.verbalizers],
                "score_semantics": "mean_conditional_log_probability",
                "scorer_contract": scorer.scoring_contract(),
            }
        )
        self.connection = sqlite3.connect(str(self.cache_path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS value_cache (
                cache_key TEXT PRIMARY KEY,
                scores_json TEXT NOT NULL,
                scoring_fingerprint TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""

        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "ValueOracle":
        """Return this oracle as a context manager."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the cache regardless of run outcome."""

        self.close()

    def _key(self, text: str) -> str:
        """Build the physical cache key for exact model input text."""

        return stable_digest(
            {"fingerprint": self.scoring_fingerprint, "text": str(text)}
        )

    def _cached(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """Load cached score rows in bounded SQLite IN clauses."""

        output: Dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(str(value) for value in keys))
        for start in range(0, len(unique), 800):
            batch = unique[start : start + 800]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT cache_key, scores_json FROM value_cache "
                f"WHERE cache_key IN ({placeholders})"
            )
            for key, scores_json in self.connection.execute(query, batch):
                output[str(key)] = np.asarray(json.loads(scores_json), dtype=np.float64)
        return output

    def score_texts(
        self,
        texts: Sequence[str],
        *,
        logical_keys: Sequence[str],
        ledger: QueryLedger,
        category: str,
    ) -> np.ndarray:
        """Serve cached scores and physically score only unique misses."""

        clean_texts = [str(value) for value in texts]
        logical = [str(value) for value in logical_keys]
        if len(clean_texts) != len(logical):
            raise ValueError("texts and logical_keys must have equal length.")
        if not clean_texts:
            return np.zeros((0, len(self.scorer.verbalizers)), dtype=np.float64)
        physical_keys = [self._key(text) for text in clean_texts]
        cached = self._cached(physical_keys)
        seen: Set[str] = set()
        hits: list[bool] = []
        missing_order: list[str] = []
        missing_text: Dict[str, str] = {}
        for key, text in zip(physical_keys, clean_texts):
            hits.append(key in cached or key in seen)
            seen.add(key)
            if key not in cached and key not in missing_text:
                missing_order.append(key)
                missing_text[key] = text
        ledger.record(logical, hits, category)
        newly_scored: Dict[str, np.ndarray] = {}
        for start in range(0, len(missing_order), self.batch_size):
            keys = missing_order[start : start + self.batch_size]
            batch_texts = [missing_text[key] for key in keys]
            started = time.perf_counter()
            rows = np.asarray(self.scorer.score_texts(batch_texts), dtype=np.float64)
            self.scoring_seconds += time.perf_counter() - started
            self.scorer_batch_calls += 1
            self.physical_values_scored += len(batch_texts)
            ledger.record_physical(len(batch_texts))
            expected = (len(batch_texts), len(self.scorer.verbalizers))
            if rows.shape != expected or not np.all(np.isfinite(rows)):
                raise ValueError(f"Scorer returned invalid values with shape {rows.shape}.")
            for key, row in zip(keys, rows):
                newly_scored[key] = row
                self.connection.execute(
                    "INSERT OR REPLACE INTO value_cache "
                    "(cache_key, scores_json, scoring_fingerprint) VALUES (?, ?, ?)",
                    (
                        key,
                        json.dumps([float(value) for value in row]),
                        self.scoring_fingerprint,
                    ),
                )
            self.connection.commit()
        values = {**cached, **newly_scored}
        return np.vstack([values[key] for key in physical_keys])

    def snapshot_counters(self) -> Dict[str, Any]:
        """Return cache, scorer, and timing counters."""

        row = self.connection.execute("SELECT COUNT(*) FROM value_cache").fetchone()
        return {
            "physical_values_scored_this_process": self.physical_values_scored,
            "scorer_batch_calls_this_process": self.scorer_batch_calls,
            "scoring_seconds_this_process": self.scoring_seconds,
            "cache_entry_count": int(row[0]) if row else 0,
            "scorer": self.scorer.snapshot_counters(),
        }
