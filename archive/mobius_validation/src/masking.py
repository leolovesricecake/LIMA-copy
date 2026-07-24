from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .schema import FeatureSpec


MASK_OPERATORS = {"delete", "replace"}
CONDITIONING_MODES = {"global", "rest_present", "rest_absent"}


def replacement_token(*, mask_token: Optional[str] = None, unk_token: Optional[str] = None) -> str:
    token = mask_token if mask_token not in {None, ""} else unk_token
    return str(token if token not in {None, ""} else "[UNK]")


def _space_prefix(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]


def _space_suffix(text: str) -> str:
    return text[len(text.rstrip()) :]


def _masked_span_text(span_text: str, repl: str) -> str:
    return f"{_space_prefix(span_text)}{repl}{_space_suffix(span_text)}"


def selected_feature_ids_from_bitmask(
    bitmask: int,
    active_feature_ids: Sequence[int],
) -> set[int]:
    selected = set()
    mask = int(bitmask)
    for bit_pos, feature_id in enumerate(active_feature_ids):
        if mask & (1 << bit_pos):
            selected.add(int(feature_id))
    return selected


def apply_mask(
    feature_spec: FeatureSpec,
    bitmask: int,
    *,
    operator: str = "delete",
    active_feature_ids: Optional[Sequence[int]] = None,
    conditioning_mode: str = "global",
    mask_token: Optional[str] = None,
    unk_token: Optional[str] = None,
) -> str:
    op = str(operator).strip().lower()
    if op not in MASK_OPERATORS:
        raise ValueError(f"Unsupported mask operator: {operator!r}")
    mode = str(conditioning_mode).strip().lower()
    if mode not in CONDITIONING_MODES:
        raise ValueError(f"Unsupported conditioning mode: {conditioning_mode!r}")

    active = list(range(feature_spec.n_features)) if active_feature_ids is None else [int(x) for x in active_feature_ids]
    active_set = set(active)
    selected_active = selected_feature_ids_from_bitmask(int(bitmask), active)
    repl = replacement_token(mask_token=mask_token, unk_token=unk_token)

    parts: List[str] = []
    for feature in feature_spec.features:
        fid = int(feature.feature_id)
        if fid in active_set:
            keep = fid in selected_active
        elif mode == "rest_present":
            keep = True
        elif mode == "rest_absent":
            keep = False
        else:
            keep = True

        if keep:
            parts.append(feature.span_text)
        elif op == "replace":
            parts.append(_masked_span_text(feature.span_text, repl))

    return "".join(parts).strip()


def compose_kept_words(feature_spec: FeatureSpec, feature_ids: Iterable[int]) -> str:
    wanted = set(int(x) for x in feature_ids)
    return "".join(
        feature.span_text for feature in feature_spec.features if int(feature.feature_id) in wanted
    ).strip()

