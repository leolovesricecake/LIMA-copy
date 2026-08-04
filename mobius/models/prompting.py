"""Versioned task-aware prompts for verbalizer-based text classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


TASK_PROMPT_VERSION = "task_classification_v1"

_SENTIMENT_DATASETS = {
    "sst2",
    "glue_sst2",
    "rotten_tomatoes",
    "rtn",
    "rt",
    "imdb",
    "eraser_movie_reviews",
    "eraser",
}

_TASK_DESCRIPTIONS = {
    "emotion": "Identify the primary emotion expressed in the text.",
    "ag_news": "Identify the main news topic of the text.",
    "agnews": "Identify the main news topic of the text.",
}


def default_task_description(dataset_name: str | None) -> str:
    """Return a concise task instruction for a supported dataset."""

    normalized = str(dataset_name or "").strip().lower().replace("-", "_")
    if normalized in _SENTIMENT_DATASETS:
        return "Determine whether the sentiment of the text is negative or positive."
    return _TASK_DESCRIPTIONS.get(
        normalized,
        "Classify the text according to the candidate labels.",
    )


@dataclass(frozen=True)
class ClassificationPromptSpec:
    """Store the exact fixed context wrapped around every perturbed text."""

    version: str
    dataset_name: str
    task_description: str
    verbalizers: tuple[str, ...]

    @property
    def candidate_labels(self) -> str:
        """Render candidate labels in their scoring order without output quotes."""

        return " | ".join(self.verbalizers)

    @property
    def prefix(self) -> str:
        """Return the fixed task context before the source text."""

        return (
            f"Task: {self.task_description}\n"
            f"Candidate labels: {self.candidate_labels}\n"
            "Return exactly one candidate label without quotation marks or explanation.\n"
            "Text:\n"
        )

    @property
    def suffix(self) -> str:
        """Return the fixed context after the source text."""

        return "\nLabel:"

    def render(self, text: str) -> str:
        """Wrap one complete or perturbed text in the classification prompt."""

        return f"{self.prefix}{text}{self.suffix}"

    def to_config(self) -> Dict[str, str]:
        """Serialize the minimal scientific configuration needed to rebuild the prompt."""

        return {
            "version": self.version,
            "task_description": self.task_description,
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the complete scoring prompt contract for provenance."""

        return {
            **self.to_config(),
            "dataset_name": self.dataset_name,
            "candidate_labels": list(self.verbalizers),
            "template": f"{self.prefix}{{text}}{self.suffix} {{class_verbalizer}}",
            "label_prefix": " ",
            "add_special_tokens": False,
            "use_chat_template": False,
            "score_semantics": "mean_conditional_log_probability",
        }


def build_classification_prompt(
    *,
    dataset_name: str | None,
    verbalizers: Sequence[str],
    prompt_config: Mapping[str, Any] | None = None,
) -> ClassificationPromptSpec:
    """Resolve a versioned task prompt from dataset defaults and optional overrides."""

    config = dict(prompt_config or {})
    version = str(config.get("version", TASK_PROMPT_VERSION)).strip()
    if version != TASK_PROMPT_VERSION:
        raise ValueError(
            f"Unsupported prompt version {version!r}; expected {TASK_PROMPT_VERSION!r}."
        )
    labels = tuple(str(value).strip() for value in verbalizers)
    if not labels or any(not value for value in labels):
        raise ValueError("Task-aware classification requires non-empty verbalizers.")
    if len(set(labels)) != len(labels):
        raise ValueError("Classification verbalizers must be unique.")
    description = str(
        config.get("task_description") or default_task_description(dataset_name)
    ).strip()
    if not description:
        raise ValueError("task_description must not be empty.")
    return ClassificationPromptSpec(
        version=version,
        dataset_name=str(dataset_name or "unknown"),
        task_description=description,
        verbalizers=labels,
    )


__all__ = [
    "TASK_PROMPT_VERSION",
    "ClassificationPromptSpec",
    "build_classification_prompt",
    "default_task_description",
]
