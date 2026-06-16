from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Sequence, Tuple

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
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.max_length = int(max_length)
        self.embedding_layer_ratio = float(embedding_layer_ratio)
        self.trust_remote_code = bool(trust_remote_code)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "HFBackbone requires torch and transformers. Please install dependencies first."
            ) from exc

        self.torch = torch
        self.nnf = torch.nn.functional
        self.device = self._prepare_device(torch, device)

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        # Batch knobs for batched objective evaluation.
        # Can be overridden by env vars for quick tuning without code changes.
        self.predict_batch_size = max(1, int(os.getenv("LIMA_PREDICT_BATCH_SIZE", "4")))
        self.embed_batch_size = max(1, int(os.getenv("LIMA_EMBED_BATCH_SIZE", "2")))

    @staticmethod
    def _prepare_device(torch_module, device: str):
        resolved = torch_module.device(device)
        if resolved.type == "cuda":
            torch_module.cuda.set_device(0 if resolved.index is None else resolved.index)
        return resolved

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = f"{type(exc).__name__}: {exc}".lower()
        if "outofmemory" in type(exc).__name__.lower():
            return True
        return "out of memory" in msg and ("cuda" in msg or "cudnn" in msg or "hip" in msg)

    def _clear_cuda_cache(self) -> None:
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _token_ids_no_special(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(x) for x in ids]

    def _add_counter(self, key: str, delta: float = 1.0) -> None:
        self.forward_counters[key] = self.forward_counters.get(key, 0) + delta

    def _truncate_prefix_and_label_ids(
        self,
        prefix_ids: Sequence[int],
        label_ids: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        max_total = max(2, int(self.max_length))
        max_label = max_total - 1
        label_ids_trim = list(int(x) for x in label_ids)
        if len(label_ids_trim) > max_label:
            label_ids_trim = label_ids_trim[-max_label:]
        max_prefix = max_total - len(label_ids_trim)
        if max_prefix <= 0:
            return [], label_ids_trim
        return list(int(x) for x in prefix_ids[-max_prefix:]), label_ids_trim

    def _tokenize_prefix_batch(self, texts: Sequence[str]) -> List[List[int]]:
        if not texts:
            return []
        prefixes = [f"Text:\n{text}\nLabel:" for text in texts]
        encoded = self.tokenizer(
            prefixes,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return [[int(x) for x in ids] for ids in encoded["input_ids"]]

    def _label_token_ids_cache(self, verbalizers: Sequence[str]) -> Dict[str, List[int]]:
        cache: Dict[str, List[int]] = {}
        max_total = max(2, int(self.max_length))
        max_label = max_total - 1
        for label in verbalizers:
            ids = self._token_ids_no_special(" " + str(label))
            if len(ids) > max_label:
                ids = ids[-max_label:]
            cache[str(label)] = [int(x) for x in ids]
        return cache

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
            self.forward_counters["model_forward_calls"] += 1
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

    def _build_full_ids_for_label(
        self,
        text: str,
        label_text: str,
    ) -> Tuple[List[int], int, int]:
        prefix = f"Text:\n{text}\nLabel:"
        prefix_ids_raw = self._token_ids_no_special(prefix)
        label_ids_raw = self._token_ids_no_special(" " + label_text)
        prefix_ids, label_ids = self._truncate_prefix_and_label_ids(prefix_ids_raw, label_ids_raw)
        full_ids = list(prefix_ids) + list(label_ids)
        return full_ids, len(prefix_ids), len(label_ids)

    def _label_conditional_logprob_batch_from_token_ids(
        self,
        prefix_ids_batch: Sequence[Sequence[int]],
        label_ids: Sequence[int],
    ) -> np.ndarray:
        if not prefix_ids_batch:
            return np.zeros((0,), dtype=np.float32)

        t_pack0 = time.perf_counter()
        torch = self.torch
        max_total = max(2, int(self.max_length))
        max_label = max_total - 1
        label_ids_trim = list(int(x) for x in label_ids)
        if len(label_ids_trim) > max_label:
            label_ids_trim = label_ids_trim[-max_label:]

        packed_full_ids: List[List[int]] = []
        prefix_lens: List[int] = []
        label_lens: List[int] = []
        max_seq = 0
        for prefix_ids_raw in prefix_ids_batch:
            prefix_ids, label_ids_cur = self._truncate_prefix_and_label_ids(prefix_ids_raw, label_ids_trim)
            full_ids = list(prefix_ids) + list(label_ids_cur)
            packed_full_ids.append(full_ids)
            prefix_lens.append(len(prefix_ids))
            label_lens.append(len(label_ids_cur))
            if len(full_ids) > max_seq:
                max_seq = len(full_ids)

        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full(
            (len(packed_full_ids), max_seq),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, full_ids in enumerate(packed_full_ids):
            seq_len = len(full_ids)
            input_ids[row, :seq_len] = torch.as_tensor(full_ids, dtype=torch.long, device=self.device)
            attention_mask[row, :seq_len] = 1

        self._add_counter("batch_pack_calls", 1)
        self._add_counter("batch_pack_seconds", time.perf_counter() - t_pack0)

        t_forward0 = time.perf_counter()
        with torch.no_grad():
            self.forward_counters["model_forward_calls"] += 1
            self._add_counter("batch_forward_calls", 1)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            logprobs = self.nnf.log_softmax(logits, dim=-1)
            token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        self._add_counter("batch_forward_seconds", time.perf_counter() - t_forward0)

        scores = np.zeros((len(packed_full_ids),), dtype=np.float32)
        for row, (prefix_len, label_len) in enumerate(zip(prefix_lens, label_lens)):
            start = prefix_len - 1
            end = start + label_len
            label_lp = token_lp[row, start:end]
            scores[row] = float(label_lp.mean().item())
        return scores

    def _label_conditional_logprob_batch(self, texts: Sequence[str], label_text: str) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        t_tokenize0 = time.perf_counter()
        prefix_ids_batch = self._tokenize_prefix_batch(texts)
        label_ids = self._label_token_ids_cache([label_text])[str(label_text)]
        self._add_counter("batch_tokenize_calls", 1)
        self._add_counter("batch_tokenize_seconds", time.perf_counter() - t_tokenize0)
        return self._label_conditional_logprob_batch_from_token_ids(prefix_ids_batch, label_ids)

    def predict_label_probs(self, text: str, verbalizers: Sequence[str]) -> np.ndarray:
        probs = self._predict_label_probs_batch_impl([text], verbalizers, record_batch=False)
        return probs[0]

    def _predict_label_probs_batch_impl(
        self,
        texts: Sequence[str],
        verbalizers: Sequence[str],
        record_batch: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(verbalizers)), dtype=np.float32)
        if record_batch:
            self.forward_counters["batch_calls"] += 1
            self.forward_counters["batch_rows"] += len(texts)
        self.forward_counters["predict_calls"] += len(texts)
        outputs: List[np.ndarray] = []
        idx = 0
        batch_size = min(self.predict_batch_size, len(texts))
        label_ids_cache = self._label_token_ids_cache(verbalizers)
        while idx < len(texts):
            cur = list(texts[idx : idx + batch_size])
            try:
                t_tokenize0 = time.perf_counter()
                prefix_ids_batch = self._tokenize_prefix_batch(cur)
                self._add_counter("batch_tokenize_calls", 1)
                self._add_counter("batch_tokenize_seconds", time.perf_counter() - t_tokenize0)
                score_cols = [
                    self._label_conditional_logprob_batch_from_token_ids(
                        prefix_ids_batch=prefix_ids_batch,
                        label_ids=label_ids_cache[str(label)],
                    )
                    for label in verbalizers
                ]
                score_mat = np.stack(score_cols, axis=1).astype(np.float64)
                score_mat = score_mat - score_mat.max(axis=1, keepdims=True)
                probs = np.exp(score_mat)
                probs = probs / probs.sum(axis=1, keepdims=True)
                outputs.append(probs.astype(np.float32))
                idx += len(cur)
            except Exception as exc:
                if self._is_oom_error(exc) and batch_size > 1:
                    self._clear_cuda_cache()
                    self.forward_counters["oom_shrink_events"] += 1
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(verbalizers)), dtype=np.float32)

    def _embed_text_once(self, text: str) -> np.ndarray:
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
            self.forward_counters["model_forward_calls"] += 1
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

    def embed_text(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        return self._embed_text_once(text)

    def _embed_texts_once(self, texts: Sequence[str]) -> List[np.ndarray]:
        torch = self.torch
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            self.forward_counters["model_forward_calls"] += 1
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            effective_layers = len(hidden_states) - 1
            layer_index = max(1, int(math.floor(self.embedding_layer_ratio * effective_layers)))
            hidden = hidden_states[layer_index]

            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled.float().cpu().numpy().astype(np.float32)

        out: List[np.ndarray] = []
        for row in pooled:
            norm = float(np.linalg.norm(row))
            if norm > 1e-8:
                row = row / norm
            out.append(row.astype(np.float32))
        return out

    def _embed_texts_impl(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not texts:
            return []
        self.forward_counters["embed_calls"] += len(texts)
        outputs: List[np.ndarray] = []
        idx = 0
        batch_size = min(self.embed_batch_size, len(texts))
        while idx < len(texts):
            cur = list(texts[idx : idx + batch_size])
            try:
                outputs.extend(self._embed_texts_once(cur))
                idx += len(cur)
            except Exception as exc:
                if self._is_oom_error(exc) and batch_size > 1:
                    self._clear_cuda_cache()
                    self.forward_counters["oom_shrink_events"] += 1
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
        return outputs

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

        self.forward_counters["model_forward_calls"] += 1
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

        grad_norm = (
            full_embeds.grad[0, : prompt_ids.shape[1], :]
            .norm(dim=-1)
            .float()
            .detach()
            .cpu()
            .numpy()
        )

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
