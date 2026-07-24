"""Coalition-to-text mapping and truncation-aware active-player selection."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from mobius.core.schema import TextChunk

from .chunks import compose_text, content_span


PROMPT_PREFIX = "Text:\n"
PROMPT_SUFFIX = "\nLabel:"
EMPTY_TEXT = " "


def _token_ids(tokenizer, text: str) -> List[int]:
    """Tokenize text without special tokens."""

    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return [int(value) for value in encoded["input_ids"]]


def visible_text_span(
    tokenizer,
    text: str,
    label_text: str,
    max_length: int,
) -> Dict[str, Any]:
    """Map left prompt truncation back to a visible source-character span."""

    if tokenizer is None:
        return {
            "visible_start_char": 0,
            "visible_end_char": len(text),
            "visible_char_count": len(text),
            "dropped_prompt_token_count": 0,
            "strategy": "no_tokenizer_assume_full_visibility",
        }
    prompt = f"{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}"
    label_ids = _token_ids(tokenizer, " " + str(label_text))[-max(1, max_length - 1) :]
    try:
        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
    except Exception as error:
        raise RuntimeError("Tokenizer offsets are required for truncation alignment.") from error
    prompt_ids = list(encoded["input_ids"])
    offsets = list(encoded.get("offset_mapping") or [])
    if len(prompt_ids) != len(offsets):
        raise RuntimeError("Tokenizer returned mismatched IDs and offsets.")
    prompt_budget = max(0, int(max_length) - len(label_ids))
    kept = offsets[-prompt_budget:] if prompt_budget else []
    source_start = len(PROMPT_PREFIX)
    source_end = source_start + len(text)
    spans = [
        (max(int(start), source_start) - source_start, min(int(end), source_end) - source_start)
        for start, end in kept
        if min(int(end), source_end) > max(int(start), source_start)
    ]
    visible_start = min((span[0] for span in spans), default=len(text))
    visible_end = max((span[1] for span in spans), default=len(text))
    return {
        "visible_start_char": int(visible_start),
        "visible_end_char": int(visible_end),
        "visible_char_count": int(max(0, visible_end - visible_start)),
        "prompt_token_count": len(prompt_ids),
        "kept_prompt_token_count": len(kept),
        "dropped_prompt_token_count": max(0, len(prompt_ids) - len(kept)),
        "target_token_count": len(label_ids),
        "strategy": "prompt_offset_left_truncation",
    }


def active_chunk_ids(
    chunks: Sequence[TextChunk],
    visible_start: int,
    visible_end: int,
) -> List[int]:
    """Select chunks whose non-whitespace content intersects the visible span."""

    active: List[int] = []
    for chunk in chunks:
        start, end = content_span(chunk)
        if min(end, int(visible_end)) > max(start, int(visible_start)):
            active.append(int(chunk.chunk_id))
    return active


class CoalitionGame:
    """Map integer keep masks over active players to perturbed texts."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        player_to_chunk_id: Sequence[int],
    ) -> None:
        """Store the complete chunks and active-player mapping."""

        self.chunks = list(chunks)
        self.player_to_chunk_id = [int(value) for value in player_to_chunk_id]

    @property
    def n_players(self) -> int:
        """Return the number of active explanation players."""

        return len(self.player_to_chunk_id)

    def text(self, mask: int) -> str:
        """Compose the text for an integer keep mask."""

        selected = [
            chunk_id
            for player, chunk_id in enumerate(self.player_to_chunk_id)
            if int(mask) & (1 << player)
        ]
        output = compose_text(self.chunks, selected)
        return output if output else EMPTY_TEXT

    def texts(self, masks: Sequence[int]) -> List[str]:
        """Compose texts for a sequence of keep masks."""

        return [self.text(int(mask)) for mask in masks]

