"""Causal-LM verbalizer scoring with batched mean conditional log likelihood."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .base import RawTextScorer


class HFVerbalizerScorer(RawTextScorer):
    """Use a causal LM to score ordered class verbalizers."""

    def __init__(
        self,
        *,
        model_path: str,
        verbalizers: Sequence[str] = (),
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 2048,
        trust_remote_code: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Load the tokenizer and model on one explicit device."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HF scoring requires torch and transformers.") from exc
        self.torch = torch
        self.verbalizers = [str(value) for value in verbalizers]
        self.max_length = int(max_length)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(0 if self.device.index is None else self.device.index)
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=bool(trust_remote_code),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=resolved_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=bool(trust_remote_code),
        )
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = max(
            1,
            int(batch_size or os.getenv("MOBIUS_PREDICT_BATCH_SIZE", "4")),
        )
        self.counters: Dict[str, float | int] = {
            "batch_calls": 0,
            "batch_rows": 0,
            "model_forward_calls": 0,
            "oom_shrink_events": 0,
            "scoring_seconds": 0.0,
        }

    def _token_ids(self, text: str) -> List[int]:
        """Tokenize without adding model-specific special tokens."""

        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)
        return [int(value) for value in encoded["input_ids"]]

    def _label_ids(self) -> Dict[str, List[int]]:
        """Tokenize and trim each verbalizer once per scoring call."""

        maximum = max(1, self.max_length - 1)
        return {
            label: self._token_ids(" " + label)[-maximum:]
            for label in self.verbalizers
        }

    def _prefix_ids(self, texts: Sequence[str]) -> List[List[int]]:
        """Batch-tokenize the fixed classification prompt prefixes."""

        prompts = [f"Text:\n{text}\nLabel:" for text in texts]
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return [[int(value) for value in row] for row in encoded["input_ids"]]

    def _score_label(
        self,
        prefix_rows: Sequence[Sequence[int]],
        label_ids: Sequence[int],
    ) -> np.ndarray:
        """Score one label for a batch of already-tokenized prefixes."""

        torch = self.torch
        packed: List[List[int]] = []
        prefix_lengths: List[int] = []
        label_length = len(label_ids)
        if label_length == 0:
            raise ValueError("A verbalizer produced no tokens.")
        for raw_prefix in prefix_rows:
            maximum_prefix = max(0, self.max_length - label_length)
            prefix = list(raw_prefix[-maximum_prefix:]) if maximum_prefix else []
            packed.append(prefix + list(label_ids))
            prefix_lengths.append(len(prefix))
        width = max(len(row) for row in packed)
        input_ids = torch.full(
            (len(packed), width),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.zeros_like(input_ids)
        for row_index, row in enumerate(packed):
            input_ids[row_index, : len(row)] = torch.as_tensor(row, device=self.device)
            attention[row_index, : len(row)] = 1
        with torch.no_grad():
            self.counters["model_forward_calls"] += 1
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            )
            log_probs = torch.nn.functional.log_softmax(output.logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            token_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        scores = np.zeros(len(packed), dtype=np.float64)
        for row_index, prefix_length in enumerate(prefix_lengths):
            start = prefix_length - 1
            scores[row_index] = float(
                token_scores[row_index, start : start + label_length].mean().item()
            )
        return scores

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        """Recognize CUDA/HIP out-of-memory failures."""

        message = f"{type(error).__name__}: {error}".lower()
        return "out of memory" in message or "outofmemory" in message

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Score texts in adaptive batches and shrink batches after OOM."""

        if not texts:
            return np.zeros((0, len(self.verbalizers)), dtype=np.float64)
        started = time.perf_counter()
        rows: List[np.ndarray] = []
        cursor = 0
        batch_size = min(self.batch_size, len(texts))
        labels = self._label_ids()
        while cursor < len(texts):
            batch = list(texts[cursor : cursor + batch_size])
            try:
                prefixes = self._prefix_ids(batch)
                columns = [
                    self._score_label(prefixes, labels[label])
                    for label in self.verbalizers
                ]
                rows.append(np.stack(columns, axis=1))
                self.counters["batch_calls"] += 1
                self.counters["batch_rows"] += len(batch)
                cursor += len(batch)
            except Exception as error:
                if self._is_oom(error) and batch_size > 1:
                    self.torch.cuda.empty_cache()
                    self.counters["oom_shrink_events"] += 1
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
        self.counters["scoring_seconds"] += time.perf_counter() - started
        return np.concatenate(rows, axis=0)

    def predict_label_scores_batch(
        self,
        texts: Sequence[str],
        verbalizers: Sequence[str],
    ) -> np.ndarray:
        """Provide the historical backbone API for retained baselines."""

        self.verbalizers = [str(value) for value in verbalizers]
        return self.score_texts(texts)

    def predict_label_scores(
        self,
        text: str,
        verbalizers: Sequence[str],
    ) -> np.ndarray:
        """Score one text through the historical backbone API."""

        return self.predict_label_scores_batch([text], verbalizers)[0]

    def predict_label_probs_batch(
        self,
        texts: Sequence[str],
        verbalizers: Sequence[str],
    ) -> np.ndarray:
        """Return probabilities through the historical backbone API."""

        scores = self.predict_label_scores_batch(texts, verbalizers)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / np.sum(values, axis=1, keepdims=True)

    def predict_label_probs(
        self,
        text: str,
        verbalizers: Sequence[str],
    ) -> np.ndarray:
        """Return one probability row through the historical backbone API."""

        return self.predict_label_probs_batch([text], verbalizers)[0]

    def tokenize_len(self, text: str) -> int:
        """Count tokenizer IDs without special tokens."""

        return max(1, len(self._token_ids(text)))

    def snapshot_counters(self) -> Dict[str, float | int]:
        """Return an immutable counter snapshot."""

        return dict(self.counters)


def build_scorer(
    config: Mapping[str, object],
    verbalizers: Sequence[str],
    *,
    batch_size: int = 16,
) -> RawTextScorer:
    """Construct a mock or Hugging Face scorer from resolved config."""

    model_type = str(config.get("type", "hf_causal_lm")).lower()
    if model_type in {"mock", "mock_sentiment"}:
        from .mock import MockSentimentScorer

        return MockSentimentScorer(verbalizers)
    return HFVerbalizerScorer(
        model_path=str(config["model_path"]),
        verbalizers=verbalizers,
        device=str(config.get("device", "cuda:0")),
        dtype=str(config.get("dtype", "bfloat16")),
        max_length=int(config.get("max_length", 2048)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        batch_size=int(batch_size),
    )


HFBackbone = HFVerbalizerScorer


__all__ = ["HFBackbone", "HFVerbalizerScorer", "build_scorer"]
