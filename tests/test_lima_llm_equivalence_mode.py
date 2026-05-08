import numpy as np

from lima_llm.backbone.base import BaseBackbone


class _ModeAwareBackbone(BaseBackbone):
    def __init__(self, equivalence_mode: str = "optimized_batch") -> None:
        super().__init__(equivalence_mode=equivalence_mode)
        self.batch_predict_impl_calls = 0
        self.batch_embed_impl_calls = 0

    def tokenize_len(self, text: str) -> int:
        return max(1, len(text))

    def _base_probs(self, text: str, verbalizers) -> np.ndarray:
        raw = float(sum(ord(ch) for ch in text) % 997) / 997.0
        if len(verbalizers) == 1:
            return np.asarray([1.0], dtype=np.float32)
        p0 = min(0.99, max(0.01, raw))
        p1 = 1.0 - p0
        return np.asarray([p0, p1], dtype=np.float32)

    def predict_label_probs(self, text: str, verbalizers) -> np.ndarray:
        self.forward_counters["predict_calls"] += 1
        return self._base_probs(text, verbalizers)

    def _predict_label_probs_batch_impl(self, texts, verbalizers):
        self.batch_predict_impl_calls += 1
        self.forward_counters["predict_calls"] += len(texts)
        rows = []
        for text in texts:
            probs = self._base_probs(text, verbalizers).astype(np.float32)
            if len(probs) >= 2:
                shift = min(0.02, probs[1] - 1e-6)
                probs = np.asarray([probs[0] + shift, probs[1] - shift], dtype=np.float32)
            rows.append(probs)
        return np.stack(rows, axis=0).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        base = float((sum(ord(ch) for ch in text) % 19) + 1)
        return np.asarray([base, base + 1.0], dtype=np.float32)

    def _embed_texts_impl(self, texts):
        self.batch_embed_impl_calls += 1
        self.forward_counters["embed_calls"] += len(texts)
        out = []
        for text in texts:
            vec = self.embed_text(text).copy()
            vec[0] += 0.5
            out.append(vec.astype(np.float32))
        return out


def test_strict_ref_forces_single_path_for_batch_predict_and_embed() -> None:
    backbone = _ModeAwareBackbone(equivalence_mode="strict_ref")
    text = "hello"
    labels = ["NEG", "POS"]

    probs_single = backbone.predict_label_probs(text, labels)
    probs_batch = backbone.predict_label_probs_batch([text], labels)[0]
    assert np.allclose(probs_single, probs_batch)
    assert backbone.batch_predict_impl_calls == 0

    emb_single = backbone.embed_text(text)
    emb_batch = backbone.embed_texts([text])[0]
    assert np.allclose(emb_single, emb_batch)
    assert backbone.batch_embed_impl_calls == 0


def test_optimized_batch_uses_batch_impl_paths() -> None:
    backbone = _ModeAwareBackbone(equivalence_mode="optimized_batch")
    text = "hello"
    labels = ["NEG", "POS"]

    probs_single = backbone.predict_label_probs(text, labels)
    probs_batch = backbone.predict_label_probs_batch([text], labels)[0]
    assert not np.allclose(probs_single, probs_batch)
    assert backbone.batch_predict_impl_calls == 1
    assert backbone.snapshot_counters()["predict_batch_calls"] == 1

    emb_single = backbone.embed_text(text)
    emb_batch = backbone.embed_texts([text])[0]
    assert not np.allclose(emb_single, emb_batch)
    assert backbone.batch_embed_impl_calls == 1
    assert backbone.snapshot_counters()["embed_batch_calls"] == 1


def test_invalid_equivalence_mode_raises() -> None:
    try:
        _ModeAwareBackbone(equivalence_mode="bad_mode")
        assert False, "expected ValueError"
    except ValueError:
        pass

