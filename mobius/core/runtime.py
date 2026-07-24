"""Runtime helpers shared by command-line entry points."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Write JSON via an atomic replacement in the destination directory."""

    destination = Path(path)
    ensure_dir(destination.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        return


def git_commit(root: str | Path | None = None) -> str | None:
    """Return the current Git commit when the checkout is available."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_summary() -> Dict[str, Any]:
    """Capture a compact environment summary for run provenance."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def counter_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Subtract numeric model counters while preserving integer types."""

    delta: Dict[str, Any] = {}
    for key in sorted(set(before).union(after)):
        old = before.get(key, 0)
        new = after.get(key, 0)
        if isinstance(old, float) or isinstance(new, float):
            delta[key] = float(new) - float(old)
        elif isinstance(old, (int, bool)) and isinstance(new, (int, bool)):
            delta[key] = int(new) - int(old)
    return delta

