from __future__ import annotations

import types

import numpy as np

from lima_llm.backbone.hf_backbone import HFBackbone


def _build_stub_hf_backbone() -> HFBackbone:
    backbone = HFBackbone.__new__(HFBackbone)
    backbone.forward_counters = {
        "predict_calls": 0,
        "embed_calls": 0,
        "gradient_calls": 0,
        "batch_calls": 0,
        "batch_rows": 0,
        "model_forward_calls": 0,
        "oom_shrink_events": 0,
        "batch_tokenize_calls": 0,
        "batch_pack_calls": 0,
        "batch_forward_calls": 0,
        "batch_tokenize_seconds": 0.0,
        "batch_pack_seconds": 0.0,
        "batch_forward_seconds": 0.0,
    }
    backbone.predict_batch_size = 8
    backbone._clear_cuda_cache = lambda: None
    backbone._add_counter = types.MethodType(HFBackbone._add_counter, backbone)

    def _fake_tokenize_prefix_batch(self, texts):
        self._add_counter("batch_tokenize_calls", 1)
        self._add_counter("batch_tokenize_seconds", 0.001)
        return [[1, 2, 3] for _ in texts]

    def _fake_label_token_ids_cache(self, verbalizers):
        out = {}
        for label in verbalizers:
            out[str(label)] = [1] if str(label).lower() == "pos" else [0]
        return out

    def _fake_label_conditional_logprob_batch_from_token_ids(self, prefix_ids_batch, label_ids):
        base = 2.0 if list(label_ids) == [1] else 1.0
        return np.asarray([base for _ in prefix_ids_batch], dtype=np.float32)

    backbone._tokenize_prefix_batch = types.MethodType(_fake_tokenize_prefix_batch, backbone)
    backbone._label_token_ids_cache = types.MethodType(_fake_label_token_ids_cache, backbone)
    backbone._label_conditional_logprob_batch_from_token_ids = types.MethodType(
        _fake_label_conditional_logprob_batch_from_token_ids,
        backbone,
    )
    return backbone


def test_predict_label_probs_and_batch_share_same_core_path() -> None:
    backbone = _build_stub_hf_backbone()

    probs_batch = HFBackbone._predict_label_probs_batch_impl(
        backbone,
        texts=["a", "b"],
        verbalizers=["NEG", "POS"],
        record_batch=True,
    )
    assert probs_batch.shape == (2, 2)
    assert backbone.forward_counters["predict_calls"] == 2
    assert backbone.forward_counters["batch_calls"] == 1
    assert backbone.forward_counters["batch_rows"] == 2

    probs_single = HFBackbone.predict_label_probs(backbone, "x", ["NEG", "POS"])
    assert probs_single.shape == (2,)
    assert abs(float(np.sum(probs_single)) - 1.0) <= 1e-6
    assert backbone.forward_counters["predict_calls"] == 3
    assert backbone.forward_counters["batch_calls"] == 1
    assert backbone.forward_counters["batch_rows"] == 2
    assert backbone.forward_counters["batch_tokenize_calls"] >= 2
