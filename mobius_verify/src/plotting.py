from __future__ import annotations

from pathlib import Path
from typing import Sequence


def ensure_figure_dir(results_dir: str | Path, *parts: str) -> Path:
    target = Path(results_dir) / "figures"
    for part in parts:
        target = target / str(part)
    target.mkdir(parents=True, exist_ok=True)
    return target


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for figure generation") from exc
    return plt

