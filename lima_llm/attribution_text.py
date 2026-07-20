from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .chunking.utils import compose_text_from_chunk_ids
from .eval.metrics import EMPTY_PERTURBATION_TEXT
from .eval.units import content_span
from .types import TextChunk


PROMPT_PREFIX = "Text:\n"
PROMPT_SUFFIX = "\nLabel:"


def _token_ids_no_special(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _target_label_ids(tokenizer, label_text: str, max_length: int) -> List[int]:
    ids = _token_ids_no_special(tokenizer, " " + str(label_text))
    max_label = max(1, int(max_length) - 1)
    if len(ids) > max_label:
        ids = ids[-max_label:]
    if not ids:
        raise ValueError(f"Label text is not tokenizable: {label_text!r}")
    return ids


def prompt_visible_text_span_after_left_truncation(
    *,
    tokenizer,
    text: str,
    label_text: str,
    max_length: int,
) -> Dict[str, Any]:
    prompt = f"{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}"
    target_ids = _target_label_ids(tokenizer, label_text, max_length=max_length)
    try:
        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Attribution requires tokenizer offset_mapping support to align prompt truncation "
            "with explanation chunks."
        ) from exc

    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = list(encoded.get("offset_mapping") or [])
    if len(offsets) != len(prompt_ids):
        raise RuntimeError("Tokenizer returned mismatched input_ids and offset_mapping lengths.")

    max_total = max(2, int(max_length))
    max_prompt = max(0, max_total - len(target_ids))
    kept_offsets = offsets[-max_prompt:] if max_prompt > 0 else []
    dropped_prompt_token_count = max(0, len(prompt_ids) - len(kept_offsets))

    text_start = len(PROMPT_PREFIX)
    text_end = text_start + len(text)
    visible_spans: List[Tuple[int, int]] = []
    for start, end in kept_offsets:
        visible_start = max(int(start), text_start)
        visible_end = min(int(end), text_end)
        if visible_end > visible_start:
            visible_spans.append((visible_start - text_start, visible_end - text_start))

    if visible_spans:
        visible_start = min(start for start, _ in visible_spans)
        visible_end = max(end for _, end in visible_spans)
    else:
        visible_start = len(text)
        visible_end = len(text)

    return {
        "visible_start_char": int(visible_start),
        "visible_end_char": int(visible_end),
        "visible_char_count": int(max(0, visible_end - visible_start)),
        "prompt_token_count": int(len(prompt_ids)),
        "kept_prompt_token_count": int(len(kept_offsets)),
        "dropped_prompt_token_count": int(dropped_prompt_token_count),
        "target_token_count": int(len(target_ids)),
        "max_prompt_token_budget": int(max_prompt),
    }


def active_chunk_ids_from_visible_span(
    units: Sequence[TextChunk],
    visible_start: int,
    visible_end: int,
) -> List[int]:
    active = []
    for unit in units:
        start, end = content_span(unit)
        if min(int(end), int(visible_end)) > max(int(start), int(visible_start)):
            active.append(int(unit.chunk_id))
    return active


def compose_coalition_text(
    *,
    units: Sequence[TextChunk],
    player_to_chunk_id: Sequence[int],
    coalition_row: Sequence[bool],
) -> str:
    selected = [
        int(player_to_chunk_id[index])
        for index, keep in enumerate(coalition_row)
        if bool(keep)
    ]
    text = compose_text_from_chunk_ids(units, selected)
    return text if text != "" else EMPTY_PERTURBATION_TEXT


__all__ = [
    "PROMPT_PREFIX",
    "PROMPT_SUFFIX",
    "active_chunk_ids_from_visible_span",
    "compose_coalition_text",
    "prompt_visible_text_span_after_left_truncation",
]
