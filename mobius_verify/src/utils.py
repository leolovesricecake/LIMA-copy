from __future__ import annotations

import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        with open(tmp_name, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read config files.") from exc
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must decode to a mapping: {path}")
    return payload


def resolve_project_path(
    raw_path: str | Path | None,
    *,
    project_root: str | Path,
    default: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve paths consistently from either repo root or project root.

    Configs in this project may use `mobius_verify/results/...` for commands
    run at repo root, or `results/...` for commands run inside mobius_verify.
    """

    if raw_path in {None, ""}:
        return Path(default).expanduser().resolve()
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    project = Path(project_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else project.parent
    parts = path.parts
    if parts and parts[0] == project.name:
        return (repo / path).resolve()
    return (project / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def environment_snapshot(extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    modules = ["numpy", "scipy", "sklearn", "pandas", "torch", "transformers", "datasets", "yaml"]
    versions: Dict[str, str] = {}
    for module_name in modules:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "ok"))
        except Exception as exc:
            versions[module_name] = f"missing:{type(exc).__name__}"

    gpu_info = []
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        gpu_info = []

    payload: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "package_versions": versions,
        "gpu": gpu_info,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def iter_existing_files(root: str | Path, pattern: str) -> Iterable[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(base.glob(pattern))
