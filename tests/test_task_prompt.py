"""Task-aware prompt and classifier-diagnostic regression tests."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from mobius.core.schema import DatasetBundle, TextSample
from mobius.evaluation.classification import evaluate_text_classifier
from mobius.models.base import RawTextScorer
from mobius.models.hf import HFVerbalizerScorer
from mobius.models.oracle import ValueOracle
from mobius.models.prompting import (
    TASK_PROMPT_VERSION,
    build_classification_prompt,
)
from mobius.text.coalitions import visible_text_span
from scripts.collect_classifier_reports import collect_reports, write_csv


class _CharTokenizer:
    """Tokenize every character to make prompt truncation deterministic."""

    pad_token_id = 0

    def __call__(self, text, **kwargs):
        """Return character IDs and optional character offsets."""

        ids = list(range(len(text)))
        payload = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return payload


class _ChatTokenizer:
    """Render a deterministic chat wrapper for layout tests."""

    chat_template = "configured"

    def apply_chat_template(self, messages, **kwargs):
        """Wrap one user message and expose an assistant generation prefix."""

        assert kwargs["enable_thinking"] is False
        return f"<user>{messages[0]['content']}</user><assistant>"


class _FixedScorer(RawTextScorer):
    """Return deterministic binary scores for classification tests."""

    def __init__(self, prompt_version: str = TASK_PROMPT_VERSION) -> None:
        """Store verbalizers and one cache-sensitive prompt version."""

        self.verbalizers = ["negative", "positive"]
        self.prompt_version = prompt_version

    def score_texts(self, texts):
        """Predict positive exactly when the text contains good."""

        rows = [(-1.0, 1.0) if "good" in text else (1.0, -1.0) for text in texts]
        return np.asarray(rows, dtype=np.float64)

    def scoring_contract(self):
        """Expose a minimal prompt-dependent cache contract."""

        return {
            "verbalizers": list(self.verbalizers),
            "prompt": {"version": self.prompt_version},
        }


def test_sentiment_prompt_states_task_candidates_and_output_constraint() -> None:
    """Require the default sentiment prompt to define a valid classification task."""

    spec = build_classification_prompt(
        dataset_name="rotten_tomatoes",
        verbalizers=["negative", "positive"],
    )
    rendered = spec.render("A good film.")
    assert "sentiment" in rendered
    assert "Candidate labels: negative | positive" in rendered
    assert "Return exactly one candidate label" in rendered
    assert rendered.endswith("\nLabel:")


def test_emotion_prompt_names_all_candidate_labels() -> None:
    """Require the emotion prompt to expose the complete closed label set."""

    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    spec = build_classification_prompt(
        dataset_name="emotion",
        verbalizers=labels,
    )
    assert "primary emotion" in spec.task_description
    assert list(spec.to_metadata()["candidate_labels"]) == labels


def test_ag_news_prompt_states_topic_task_and_four_candidates() -> None:
    """Require AG News prompts to expose the closed four-topic task."""

    labels = ["world", "sports", "business", "technology"]
    spec = build_classification_prompt(
        dataset_name="ag_news",
        verbalizers=labels,
    )
    assert "news topic" in spec.task_description
    assert list(spec.to_metadata()["candidate_labels"]) == labels


def test_classifier_reports_collect_scientific_metadata(tmp_path: Path) -> None:
    """Flatten classifier diagnostics without reopening machine-local configs."""

    source = tmp_path / "reports" / "sst2" / "report.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "dataset": {"name": "sst2", "split": "validation"},
                "model": {"model_path": "/models/Qwen3-8B"},
                "prompt": {"version": "task_classification_v1"},
                "sample_count": 2,
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "random_baseline_accuracy": 0.5,
                "majority_baseline_accuracy": 0.5,
                "per_class_recall": {"negative": 1.0, "positive": 1.0},
                "confusion_matrix": [[1, 0], [0, 1]],
                "config_fingerprint": "abc",
            }
        ),
        encoding="utf-8",
    )
    rows = collect_reports(tmp_path / "reports")
    output = tmp_path / "classifier.csv"
    write_csv(output, rows)
    assert rows[0]["dataset"] == "sst2"
    assert rows[0]["model"] == "Qwen3-8B"
    assert output.read_text(encoding="utf-8").startswith("dataset,split,model,")


def test_hf_prompt_layout_uses_chat_template_and_disables_thinking() -> None:
    """Use the model-native assistant prefix for instruction-tuned classifiers."""

    scorer = HFVerbalizerScorer.__new__(HFVerbalizerScorer)
    scorer.tokenizer = _ChatTokenizer()
    scorer.prompt_spec = build_classification_prompt(
        dataset_name="sst2",
        verbalizers=["negative", "positive"],
    )
    scorer._configure_prompt_layout()
    assert scorer.prompt_prefix.startswith("<user>Task:")
    assert scorer.prompt_suffix == "</user><assistant>"
    assert scorer.label_prefix == ""
    assert scorer.render_prompt("good").endswith("good</user><assistant>")


def test_visible_span_uses_dynamic_task_prompt_boundaries() -> None:
    """Align truncation to the source span inside the expanded task prompt."""

    spec = build_classification_prompt(
        dataset_name="sst2",
        verbalizers=["negative", "positive"],
    )
    text = "good movie"
    result = visible_text_span(
        _CharTokenizer(),
        text,
        "positive",
        len(spec.render(text)) + len(" positive"),
        prompt_prefix=spec.prefix,
        prompt_suffix=spec.suffix,
    )
    assert result["visible_start_char"] == 0
    assert result["visible_end_char"] == len(text)


def test_classifier_diagnostic_reports_accuracy_and_baselines() -> None:
    """Summarize task validity independently of attribution faithfulness."""

    bundle = DatasetBundle(
        dataset_name="sst2",
        split="validation",
        samples=[
            TextSample("n", "bad movie", 0),
            TextSample("p", "good movie", 1),
        ],
        label_names=["negative", "positive"],
        verbalizers=["negative", "positive"],
    )
    report = evaluate_text_classifier(bundle, _FixedScorer())
    assert report["accuracy"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["random_baseline_accuracy"] == 0.5


def test_value_oracle_cache_fingerprint_changes_with_prompt(tmp_path: Path) -> None:
    """Prevent task-prompt changes from reusing stale physical model values."""

    first = ValueOracle(
        _FixedScorer("prompt-a"),
        tmp_path / "cache.sqlite3",
        model_fingerprint={"model": "fixed"},
    )
    second = ValueOracle(
        _FixedScorer("prompt-b"),
        tmp_path / "cache.sqlite3",
        model_fingerprint={"model": "fixed"},
    )
    try:
        assert first.scoring_fingerprint != second.scoring_fingerprint
    finally:
        first.close()
        second.close()
