from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .base import BaseBackbone


class HFBackbone(BaseBackbone):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_length: int = 2048,
        embedding_layer_ratio: float = 0.7,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.max_length = int(max_length)
        self.embedding_layer_ratio = float(embedding_layer_ratio)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "HFBackbone requires torch and transformers. Please install dependencies first."
            ) from exc

        self.torch = torch
        self.nnf = torch.nn.functional

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        self.model.to(device)
        self.model.eval()

    def _token_ids_no_special(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(x) for x in ids]

    def tokenize_len(self, text: str) -> int:
        ids = self._token_ids_no_special(text)
        return max(1, len(ids))

    def _label_conditional_logprob(self, text: str, label_text: str) -> float:
        torch = self.torch
        prefix = f"Text:\n{text}\nLabel:"
        prefix_ids = self._token_ids_no_special(prefix)
        label_ids = self._token_ids_no_special(" " + label_text)

        # Keep label tokens intact and truncate prompt prefix from the left if needed.
        max_total = max(2, int(self.max_length))
        max_label = max_total - 1
        if len(label_ids) > max_label:
            label_ids = label_ids[-max_label:]
        max_prefix = max_total - len(label_ids)
        prefix_ids = prefix_ids[-max_prefix:]

        full_ids = prefix_ids + label_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            logprobs = self.nnf.log_softmax(logits, dim=-1)
            token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        label_token_count = len(label_ids)
        start = len(prefix_ids) - 1
        end = start + label_token_count
        label_lp = token_lp[:, start:end]
        score = float(label_lp.mean().item())
        return score

    def predict_label_probs(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        self.forward_counters["predict_calls"] += 1
        scores = [self._label_conditional_logprob(text, label) for label in verbalizers]
        arr = np.asarray(scores, dtype=np.float64)
        arr = arr - arr.max()
        probs = np.exp(arr)
        probs = probs / probs.sum()
        return probs.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        torch = self.torch
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            effective_layers = len(hidden_states) - 1
            layer_index = max(1, int(math.floor(self.embedding_layer_ratio * effective_layers)))
            hidden = hidden_states[layer_index][0]

            mask = attention_mask[0].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
            pooled = pooled.float().cpu().numpy().astype(np.float32)

        norm = float(np.linalg.norm(pooled))
        if norm > 1e-8:
            pooled = pooled / norm
        return pooled

    def gradient_chunk_importance(self, text: str, chunks, target_label: int, verbalizers: Sequence[str]) -> np.ndarray:
        self.forward_counters["gradient_calls"] += 1
        if target_label < 0 or target_label >= len(verbalizers):
            raise ValueError("target_label out of range for verbalizers")

        torch = self.torch
        label_text = verbalizers[target_label]

        prefix = "Text:\n"
        suffix = "\nLabel:"
        prompt = prefix + text + suffix

        label_ids = self._token_ids_no_special(" " + label_text)
        max_total = max(2, int(self.max_length))
        max_label = max_total - 1
        if len(label_ids) > max_label:
            label_ids = label_ids[-max_label:]

        encoded = self.tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        prompt_ids_list = [int(x) for x in encoded["input_ids"]]
        offsets = encoded["offset_mapping"]

        max_prompt = max_total - len(label_ids)
        if len(prompt_ids_list) > max_prompt:
            prompt_ids_list = prompt_ids_list[-max_prompt:]
            offsets = offsets[-max_prompt:]

        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=self.device)
        label_ids_t = torch.tensor([label_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_ids, label_ids_t], dim=1)

        emb_layer = self.model.get_input_embeddings()
        full_embeds = emb_layer(full_ids).detach().requires_grad_(True)

        outputs = self.model(inputs_embeds=full_embeds, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]
        logprobs = self.nnf.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        label_token_count = len(label_ids)
        start = prompt_ids.shape[1] - 1
        end = start + label_token_count
        score = token_lp[:, start:end].mean()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        grad_norm = full_embeds.grad[0, : prompt_ids.shape[1], :].norm(dim=-1).detach().cpu().numpy()

        text_start = len(prefix)
        text_end = len(prefix) + len(text)

        token_scores = []
        token_char_spans = []
        for idx, (s, e) in enumerate(offsets):
            if e <= s:
                continue
            if s < text_start or e > text_end:
                continue
            token_char_spans.append((s - text_start, e - text_start))
            token_scores.append(float(grad_norm[idx]))

        if not token_scores:
            return np.zeros(len(chunks), dtype=np.float32)

        chunk_scores = np.zeros(len(chunks), dtype=np.float32)
        for token_span, token_score in zip(token_char_spans, token_scores):
            ts, te = token_span
            for chunk in chunks:
                overlap = max(0, min(te, chunk.end_char) - max(ts, chunk.start_char))
                if overlap > 0:
                    chunk_scores[chunk.chunk_id] += float(token_score) * float(overlap)

        return chunk_scores
