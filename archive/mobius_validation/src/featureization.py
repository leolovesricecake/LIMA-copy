from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from .schema import FeatureSpec, WordFeature


_WORD_RE = re.compile(r"\w+(?:[’']\w+)*(?:-\w+)*", flags=re.UNICODE)


def _tokenizer_name(tokenizer: Any) -> str:
    if tokenizer is None:
        return "none"
    name = getattr(tokenizer, "name_or_path", None)
    if name:
        return str(name)
    return tokenizer.__class__.__name__


def _token_offsets(tokenizer: Any, text: str) -> tuple[List[Tuple[int, int]], str, int]:
    if tokenizer is None:
        return [], "none", 0
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping", []) or []
        clean_offsets = []
        for item in offsets:
            if item is None or len(item) < 2:
                continue
            start, end = int(item[0]), int(item[1])
            if end > start:
                clean_offsets.append((start, end))
        input_ids = encoded.get("input_ids", [])
        return clean_offsets, _tokenizer_name(tokenizer), len(input_ids)
    except Exception:
        return [], _tokenizer_name(tokenizer), 0


def _overlapping_token_span(
    offsets: Sequence[Tuple[int, int]],
    start: int,
    end: int,
) -> tuple[Optional[int], Optional[int]]:
    indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if min(tok_end, end) > max(tok_start, start)
    ]
    if not indices:
        return None, None
    return min(indices), max(indices) + 1


def build_lexical_word_features(
    text: str,
    *,
    sample_id: str = "",
    tokenizer: Any = None,
    normalized_model_text: Optional[str] = None,
    punctuation_policy: str = "attach_to_neighbor",
) -> FeatureSpec:
    """Build word-level features without merging multiple lexical words.

    The feature span covers the full text through neighboring whitespace and
    attached punctuation, while word_start/word_end marks the lexical core.
    """

    model_text = str(text if normalized_model_text is None else normalized_model_text)
    core_spans = [(match.start(), match.end()) for match in _WORD_RE.finditer(model_text)]
    offsets, tok_name, tok_count = _token_offsets(tokenizer, model_text)

    features: List[WordFeature] = []
    for idx, (word_start, word_end) in enumerate(core_spans):
        start = 0 if idx == 0 else word_start
        end = core_spans[idx + 1][0] if idx + 1 < len(core_spans) else len(model_text)
        token_start, token_end = _overlapping_token_span(offsets, word_start, word_end)
        features.append(
            WordFeature(
                feature_id=idx,
                word_text=model_text[word_start:word_end],
                span_text=model_text[start:end],
                start_char=start,
                end_char=end,
                word_start_char=word_start,
                word_end_char=word_end,
                token_start=token_start,
                token_end=token_end,
            )
        )

    return FeatureSpec(
        sample_id=str(sample_id),
        original_text=str(text),
        normalized_model_text=model_text,
        tokenizer_name=tok_name,
        token_count=int(tok_count),
        punctuation_attachment_policy=str(punctuation_policy),
        features=features,
    )


def validate_feature_reconstruction(spec: FeatureSpec) -> tuple[bool, str]:
    if spec.n_features == 0:
        return spec.normalized_model_text == "", "ok" if spec.normalized_model_text == "" else "no word features"
    spans = [(feature.start_char, feature.end_char) for feature in spec.features]
    if spans[0][0] != 0:
        return False, f"coverage starts at {spans[0][0]}"
    last = 0
    for idx, (start, end) in enumerate(spans):
        if start != last:
            return False, f"gap before feature {idx}: [{last}, {start})"
        if end < start:
            return False, f"invalid feature span {idx}: [{start}, {end})"
        last = end
    if last != len(spec.normalized_model_text):
        return False, f"coverage ends at {last}, expected {len(spec.normalized_model_text)}"
    rebuilt = "".join(feature.span_text for feature in spec.features)
    if rebuilt != spec.normalized_model_text:
        return False, "span_text does not reconstruct normalized_model_text"
    return True, "ok"

