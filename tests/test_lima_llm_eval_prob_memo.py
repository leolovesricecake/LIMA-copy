import hashlib
import json
from pathlib import Path

import numpy as np

from lima_llm.backbone.base import BaseBackbone
from lima_llm.eval.evaluate import evaluate_saved_explanations
from lima_llm.types import TextSample


class _CountingBackbone(BaseBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.max_length = 2048

    def tokenize_len(self, text: str) -> int:
        return max(1, len(text))

    def predict_label_probs(self, text: str, verbalizers) -> np.ndarray:
        self.forward_counters["predict_calls"] += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = int(digest[:8], 16) / float(16**8 - 1)
        p_pos = 0.2 + 0.6 * raw
        probs = np.asarray([1.0 - p_pos, p_pos], dtype=np.float32)
        probs = probs / probs.sum()
        return probs

    def embed_text(self, text: str) -> np.ndarray:
        self.forward_counters["embed_calls"] += 1
        return np.zeros((8,), dtype=np.float32)


class _Bundle:
    dataset_name = "eraser_movie_reviews"
    split = "validation"

    def __init__(self, samples):
        self.samples = samples


def test_eval_probability_cache_reduces_duplicate_predict_calls(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    sample_dir = output_root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text = "good movie. bad ending."
    split_idx = text.index("bad")
    payload = {
        "sample_id": "s1",
        "chunks": [
            {"chunk_id": 0, "start_char": 0, "end_char": split_idx, "text": text[:split_idx]},
            {"chunk_id": 1, "start_char": split_idx, "end_char": len(text), "text": text[split_idx:]},
        ],
        "selected_chunk_ids": [0],
    }
    (sample_dir / "s1.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    sample = TextSample(
        sample_id="s1",
        text=text,
        label=1,
        label_text="POS",
        rationale_char_spans=((0, split_idx),),
    )
    bundle = _Bundle(samples=[sample])
    backbone = _CountingBackbone()
    report = evaluate_saved_explanations(
        output_root=output_root,
        bundle=bundle,
        backbone=backbone,
        verbalizers=["NEG", "POS"],
        q_values=[50],
        random_trials=1,
        include_gradient_baseline=False,
    )

    cache_stats = report["metrics_secondary"]["eval_prob_cache_stats"]
    predict_calls = backbone.snapshot_counters()["predict_calls"]
    assert report["metric_settings"]["perturbation_unit"] == "word"
    assert cache_stats["hits"] > 0
    assert cache_stats["misses"] == predict_calls
    assert predict_calls <= 10
