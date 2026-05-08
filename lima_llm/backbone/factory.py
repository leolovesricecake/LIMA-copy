from __future__ import annotations

from .base import BaseBackbone
from .hf_backbone import HFBackbone
from .mock_backbone import MockBackbone


def build_backbone(
    model_path: str,
    device: str,
    use_mock_backbone: bool,
    max_length: int,
    embedding_layer_ratio: float,
    dtype: str,
) -> BaseBackbone:
    if use_mock_backbone:
        return MockBackbone()
    return HFBackbone(
        model_path=model_path,
        device=device,
        max_length=max_length,
        embedding_layer_ratio=embedding_layer_ratio,
        dtype=dtype,
    )
