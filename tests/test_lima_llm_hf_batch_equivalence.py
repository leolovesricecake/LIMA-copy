import types

import numpy as np

from lima_llm.backbone.base import BaseBackbone
from lima_llm.backbone.hf_backbone import HFBackbone


def _build_stub_hf_backbone() -> HFBackbone:
    backbone = HFBackbone.__new__(HFBackbone)
    BaseBackbone.__init__(backbone, equivalence_mode="optimized_batch")
    backbone.predict_batch_size = 2
    backbone.embed_batch_size = 2
    backbone._is_oom_error = lambda exc: False
    backbone._clear_cuda_cache = lambda: None

    def _score(self, text: str, label: str) -> float:
        raw = sum(ord(ch) for ch in f"{text}|{label}")
        return float((raw % 997) / 997.0)

    def _score_batch(self, texts, label):
        vals = [self._score(text, label) for text in texts]
        return np.asarray(vals, dtype=np.float32)

    def _embed_single(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        vec = np.asarray(
            [
                float((len(text) % 13) + 1),
                float((sum(ord(ch) for ch in text) % 17) + 1),
                1.0,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32)

    def _embed_batch_once(self, texts):
        out = []
        for text in texts:
            vec = np.asarray(
                [
                    float((len(text) % 13) + 1),
                    float((sum(ord(ch) for ch in text) % 17) + 1),
                    1.0,
                ],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            out.append(vec.astype(np.float32))
        return out

    backbone._score = types.MethodType(_score, backbone)
    backbone._label_conditional_logprob = types.MethodType(
        lambda self, text, label: self._score(text, label), backbone
    )
    backbone._label_conditional_logprob_batch = types.MethodType(_score_batch, backbone)
    backbone.embed_text = types.MethodType(_embed_single, backbone)
    backbone._embed_texts_once = types.MethodType(_embed_batch_once, backbone)
    return backbone


def test_hf_predict_single_and_batch_are_equivalent_within_tolerance() -> None:
    backbone = _build_stub_hf_backbone()
    texts = ["alpha text", "beta text", "gamma text"]
    verbalizers = ["NEG", "POS"]

    backbone.set_equivalence_mode("strict_ref")
    expected = np.stack([backbone.predict_label_probs(text, verbalizers) for text in texts], axis=0)

    backbone.set_equivalence_mode("optimized_batch")
    got = backbone.predict_label_probs_batch(texts, verbalizers)

    assert expected.shape == got.shape
    assert np.max(np.abs(expected - got)) <= 1e-6


def test_hf_embed_single_and_batch_are_equivalent_within_tolerance() -> None:
    backbone = _build_stub_hf_backbone()
    texts = ["alpha text", "beta text", "gamma text"]

    backbone.set_equivalence_mode("strict_ref")
    expected = [backbone.embed_text(text) for text in texts]

    backbone.set_equivalence_mode("optimized_batch")
    got = backbone.embed_texts(texts)

    assert len(expected) == len(got)
    for left, right in zip(expected, got):
        assert np.max(np.abs(left - right)) <= 1e-6
