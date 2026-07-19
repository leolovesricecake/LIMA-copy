from __future__ import annotations

from mobius_verify.src.featureization import build_lexical_word_features, validate_feature_reconstruction
from mobius_verify.src.masking import apply_mask


class ToyTokenizer:
    name_or_path = "toy"

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        offsets = []
        ids = []
        start = None
        for idx, ch in enumerate(text):
            if ch.isspace():
                if start is not None:
                    offsets.append((start, idx))
                    ids.append(len(ids))
                    start = None
            elif start is None:
                start = idx
        if start is not None:
            offsets.append((start, len(text)))
            ids.append(len(ids))
        return {"input_ids": ids, "offset_mapping": offsets}


def test_lexical_words_preserve_text_and_attach_punctuation() -> None:
    spec = build_lexical_word_features(
        '"Not good," she said.',
        sample_id="x",
        tokenizer=ToyTokenizer(),
    )
    ok, reason = validate_feature_reconstruction(spec)
    assert ok, reason
    assert [feature.word_text for feature in spec.features] == ["Not", "good", "she", "said"]
    assert "".join(feature.span_text for feature in spec.features) == spec.normalized_model_text
    assert spec.features[1].span_text == "good,\" "
    assert spec.features[1].token_start is not None


def test_delete_and_replace_mask_whole_word_feature() -> None:
    spec = build_lexical_word_features("A good film", sample_id="x")
    # Keep only the first and last active words.
    text = apply_mask(spec, 0b101, operator="delete")
    assert text == "A film"
    text = apply_mask(spec, 0b101, operator="replace", mask_token="[MASK]")
    assert text == "A [MASK] film"


def test_probe_rest_present_keeps_non_probe_words() -> None:
    spec = build_lexical_word_features("A good but dull film", sample_id="x")
    # Active features are good and dull. Keep only dull; non-probe words remain.
    text = apply_mask(
        spec,
        0b10,
        operator="delete",
        active_feature_ids=[1, 3],
        conditioning_mode="rest_present",
    )
    assert text == "A but dull film"

