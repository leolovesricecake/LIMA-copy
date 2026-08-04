"""Causal-LM verbalizer scoring with batched mean conditional log likelihood."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .base import RawTextScorer
from .prompting import build_classification_prompt


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
        dataset_name: str | None = None,
        prompt_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Load the tokenizer and model on one explicit device."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HF scoring requires torch and transformers.") from exc
        self.torch = torch
        self.verbalizers = [str(value) for value in verbalizers]
        self.dataset_name = str(dataset_name or "unknown")
        self.prompt_spec = build_classification_prompt(
            dataset_name=self.dataset_name,
            verbalizers=self.verbalizers,
            prompt_config=prompt_config,
        )
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
        self._configure_prompt_layout()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
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
            label: self._token_ids(self.label_prefix + label)[-maximum:]
            for label in self.verbalizers
        }

    def _prefix_ids(
        self,
        texts: Sequence[str],
        label_length: int,
    ) -> List[List[int]]:
        """Tokenize prompts while truncating only the source text from the left."""

        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        prefix = self._token_ids(self.prompt_prefix)
        suffix = self._token_ids(self.prompt_suffix)
        available_text = self.max_length - int(label_length) - len(prefix) - len(suffix)
        if available_text < 0:
            raise ValueError(
                "max_length is too small for the task prompt, candidate labels, and verbalizer."
            )
        rows = []
        for raw_text_ids in encoded["input_ids"]:
            text_ids = [int(value) for value in raw_text_ids]
            kept = text_ids[-available_text:] if available_text else []
            rows.append(prefix + kept + suffix)
        return rows

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
                label_reserve = max(len(value) for value in labels.values())
                prefixes = self._prefix_ids(batch, label_reserve)
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

        requested = [str(value) for value in verbalizers]
        if requested != self.verbalizers:
            self.configure_task(
                verbalizers=requested,
                dataset_name=self.dataset_name,
            )
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

    def label_token_reserve(self) -> int:
        """Reserve one shared target length so every class sees identical source text."""

        labels = self._label_ids()
        return max(len(value) for value in labels.values())

    def configure_task(
        self,
        *,
        verbalizers: Sequence[str],
        dataset_name: str | None = None,
        prompt_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure one dataset task before scoring or attribution."""

        self.verbalizers = [str(value) for value in verbalizers]
        self.dataset_name = str(dataset_name or self.dataset_name or "unknown")
        self.prompt_spec = build_classification_prompt(
            dataset_name=self.dataset_name,
            verbalizers=self.verbalizers,
            prompt_config=prompt_config,
        )
        self._configure_prompt_layout()

    def _configure_prompt_layout(self) -> None:
        """Render a model-native non-thinking chat prefix with a plain fallback."""

        marker = "__LIMA_CLASSIFICATION_TEXT_SLOT_7F3A9C__"
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if callable(apply_chat_template) and chat_template:
            messages = [
                {
                    "role": "user",
                    "content": f"{self.prompt_spec.prefix}{marker}",
                }
            ]
            try:
                rendered = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                thinking_disabled = True
            except TypeError:
                rendered = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                thinking_disabled = None
            if marker not in rendered:
                raise RuntimeError("Chat template removed the classification text marker.")
            self._prompt_prefix, self._prompt_suffix = rendered.split(marker, 1)
            self._uses_chat_template = True
            self._thinking_disabled = thinking_disabled
            self._label_prefix = ""
            return
        self._prompt_prefix = self.prompt_spec.prefix
        self._prompt_suffix = self.prompt_spec.suffix
        self._uses_chat_template = False
        self._thinking_disabled = None
        self._label_prefix = " "

    @property
    def prompt_prefix(self) -> str:
        """Expose the exact prompt prefix for truncation and attribution alignment."""

        return self._prompt_prefix

    @property
    def prompt_suffix(self) -> str:
        """Expose the exact prompt suffix for truncation and attribution alignment."""

        return self._prompt_suffix

    @property
    def label_prefix(self) -> str:
        """Return the exact separator before a forced class verbalizer."""

        return self._label_prefix

    def render_prompt(self, text: str) -> str:
        """Render the exact classifier prompt used for one text."""

        return f"{self.prompt_prefix}{text}{self.prompt_suffix}"

    def label_completion_text(self, label: str) -> str:
        """Render one forced assistant label exactly as it is scored."""

        return f"{self.label_prefix}{label}"

    def prompt_metadata(self) -> Dict[str, Any]:
        """Serialize the resolved model-specific prompt layout."""

        return {
            **self.prompt_spec.to_config(),
            "dataset_name": self.dataset_name,
            "candidate_labels": list(self.verbalizers),
            "template": (
                f"{self.prompt_prefix}{{text}}{self.prompt_suffix}"
                f"{self.label_prefix}{{class_verbalizer}}"
            ),
            "label_prefix": self.label_prefix,
            "add_special_tokens": False,
            "use_chat_template": bool(self._uses_chat_template),
            "enable_thinking": (
                False if self._thinking_disabled else None
            ),
            "score_semantics": "mean_conditional_log_probability",
        }

    def scoring_contract(self) -> Dict[str, Any]:
        """Return prompt and score semantics that define cached model values."""

        return {
            "scorer": f"{type(self).__module__}.{type(self).__qualname__}",
            "prompt": self.prompt_metadata(),
            "verbalizers": list(self.verbalizers),
            "max_length": int(self.max_length),
            "score_semantics": "mean_conditional_log_probability",
        }

    def snapshot_counters(self) -> Dict[str, float | int]:
        """Return an immutable counter snapshot."""

        return dict(self.counters)


def build_scorer(
    config: Mapping[str, object],
    verbalizers: Sequence[str],
    *,
    batch_size: int = 16,
    dataset_name: str | None = None,
    prompt_config: Mapping[str, Any] | None = None,
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
        dataset_name=dataset_name,
        prompt_config=prompt_config,
    )


HFBackbone = HFVerbalizerScorer


__all__ = ["HFBackbone", "HFVerbalizerScorer", "build_scorer"]
