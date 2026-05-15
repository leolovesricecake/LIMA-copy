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
    }
    backbone.predict_batch_size = 8
    backbone._clear_cuda_cache = lambda: None

    def _fake_label_conditional_logprob_batch(self, texts, label_text):
        base = 2.0 if str(label_text).lower() == "pos" else 1.0
        return np.asarray([base for _ in texts], dtype=np.float32)

    backbone._label_conditional_logprob_batch = types.MethodType(_fake_label_conditional_logprob_batch, backbone)
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
