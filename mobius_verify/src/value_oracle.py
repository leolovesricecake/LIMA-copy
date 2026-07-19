from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .masking import apply_mask
from .models.base import RawTextScorer
from .query_ledger import QueryLedger
from .schema import FeatureSpec
from .utils import ensure_dir
from .value_functions import values_from_score_matrix


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_spec_digest(feature_spec: FeatureSpec) -> str:
    return _stable_digest(feature_spec.to_dict())


class ValueOracle:
    """Persistent all-class score cache shared by every attribution method."""

    def __init__(
        self,
        *,
        scorer: RawTextScorer,
        cache_path: str | Path,
        model_fingerprint: str,
        prompt_version: str = "lima_hf_label_v1",
        batch_size: int = 32,
    ) -> None:
        self.scorer = scorer
        self.verbalizers = [str(value) for value in scorer.verbalizers]
        self.cache_path = Path(cache_path)
        ensure_dir(self.cache_path.parent)
        self.batch_size = max(1, int(batch_size))
        self.physical_values_scored = 0
        self.scorer_batch_calls = 0
        self.scoring_elapsed_seconds = 0.0
        self.scoring_fingerprint = _stable_digest(
            {
                "model": str(model_fingerprint),
                "prompt_version": str(prompt_version),
                "verbalizers": self.verbalizers,
                "score_semantics": "all_verbalizer_mean_conditional_logprob",
            }
        )
        self._connection = sqlite3.connect(str(self.cache_path), timeout=60.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS value_cache (
                cache_key TEXT PRIMARY KEY,
                text_digest TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                n_classes INTEGER NOT NULL,
                scoring_fingerprint TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "ValueOracle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _physical_key(self, text: str) -> str:
        return _stable_digest(
            {"scoring_fingerprint": self.scoring_fingerprint, "text": str(text)}
        )

    def _load_cached(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        unique = list(dict.fromkeys(str(key) for key in keys))
        out: Dict[str, np.ndarray] = {}
        for start in range(0, len(unique), 800):
            batch = unique[start : start + 800]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT cache_key, scores_json FROM value_cache WHERE cache_key IN ({placeholders})"
            for key, scores_json in self._connection.execute(query, batch):
                out[str(key)] = np.asarray(json.loads(scores_json), dtype=np.float64)
        return out

    def score_texts(
        self,
        texts: Sequence[str],
        *,
        logical_keys: Sequence[str],
        ledger: QueryLedger,
        category: str,
    ) -> np.ndarray:
        clean_texts = [str(text) for text in texts]
        logical = [str(key) for key in logical_keys]
        if len(clean_texts) != len(logical):
            raise ValueError("texts and logical_keys must have equal length")
        if not clean_texts:
            return np.zeros((0, len(self.verbalizers)), dtype=np.float64)

        physical_keys = [self._physical_key(text) for text in clean_texts]
        cached_before = self._load_cached(physical_keys)
        hits = []
        seen_in_call: set[str] = set()
        for key in physical_keys:
            hits.append(key in cached_before or key in seen_in_call)
            seen_in_call.add(key)

        missing_order: list[str] = []
        missing_text: Dict[str, str] = {}
        for key, text in zip(physical_keys, clean_texts):
            if key not in cached_before and key not in missing_text:
                missing_order.append(key)
                missing_text[key] = text

        ledger.record_requests(logical, cache_hits=hits, category=category)
        newly_scored: Dict[str, np.ndarray] = {}
        for start in range(0, len(missing_order), self.batch_size):
            keys_batch = missing_order[start : start + self.batch_size]
            texts_batch = [missing_text[key] for key in keys_batch]
            scoring_started = time.time()
            self.scorer_batch_calls += 1
            try:
                rows = np.asarray(self.scorer.score_texts(texts_batch), dtype=np.float64)
            finally:
                self.scoring_elapsed_seconds += float(time.time() - scoring_started)
            self.physical_values_scored += len(texts_batch)
            ledger.record_physical_values(len(texts_batch))
            if rows.shape != (len(texts_batch), len(self.verbalizers)):
                raise ValueError(
                    "RawTextScorer returned invalid shape: "
                    f"expected {(len(texts_batch), len(self.verbalizers))}, got {rows.shape}"
                )
            if not np.all(np.isfinite(rows)):
                raise ValueError("RawTextScorer returned NaN or Inf values")
            for key, text, row in zip(keys_batch, texts_batch, rows):
                newly_scored[key] = row.astype(np.float64)
                self._connection.execute(
                    "INSERT OR REPLACE INTO value_cache "
                    "(cache_key, text_digest, scores_json, n_classes, scoring_fingerprint) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        key,
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        json.dumps([float(value) for value in row]),
                        int(len(row)),
                        self.scoring_fingerprint,
                    ),
                )
            self._connection.commit()

        values = {**cached_before, **newly_scored}
        return np.vstack([values[key] for key in physical_keys]).astype(np.float64)

    def score_masks(
        self,
        feature_spec: FeatureSpec,
        masks: Sequence[int],
        *,
        ledger: QueryLedger,
        category: str,
        operator: str = "delete",
        active_feature_ids: Sequence[int] | None = None,
        conditioning_mode: str = "global",
        mask_token: str | None = None,
        unk_token: str | None = None,
    ) -> np.ndarray:
        active = None if active_feature_ids is None else [int(value) for value in active_feature_ids]
        spec_digest = feature_spec_digest(feature_spec)
        texts = [
            apply_mask(
                feature_spec,
                int(mask),
                operator=operator,
                active_feature_ids=active,
                conditioning_mode=conditioning_mode,
                mask_token=mask_token,
                unk_token=unk_token,
            )
            for mask in masks
        ]
        logical_keys = [
            _stable_digest(
                {
                    "sample_id": feature_spec.sample_id,
                    "feature_spec": spec_digest,
                    "mask": int(mask),
                    "operator": str(operator),
                    "active_feature_ids": active,
                    "conditioning_mode": str(conditioning_mode),
                }
            )
            for mask in masks
        ]
        return self.score_texts(
            texts,
            logical_keys=logical_keys,
            ledger=ledger,
            category=category,
        )

    def values_for_masks(
        self,
        feature_spec: FeatureSpec,
        masks: Sequence[int],
        *,
        target_class: int,
        value_type: str,
        ledger: QueryLedger,
        category: str,
        operator: str = "delete",
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_masks(
            feature_spec,
            masks,
            ledger=ledger,
            category=category,
            operator=operator,
        )
        values = values_from_score_matrix(
            scores,
            target_class=int(target_class),
            value_type=value_type,
        )
        return values, scores

    def cache_entry_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM value_cache").fetchone()
        return int(row[0]) if row else 0

    def snapshot_counters(self) -> Dict[str, object]:
        scorer_counters = {}
        snapshot = getattr(self.scorer, "snapshot_counters", None)
        if callable(snapshot):
            scorer_counters = dict(snapshot())
        return {
            "physical_values_scored_this_run": int(self.physical_values_scored),
            "scorer_batch_calls_this_run": int(self.scorer_batch_calls),
            "scoring_elapsed_seconds_this_run": float(self.scoring_elapsed_seconds),
            "cache_entry_count": int(self.cache_entry_count()),
            "scorer_counters": scorer_counters,
        }
