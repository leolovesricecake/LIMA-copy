from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def is_valid_json(path: str | Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def format_label_distribution(labels: Iterable[int]) -> Dict[int, int]:
    return dict(Counter(labels))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < eps or b_norm < eps:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def entropy_from_probs(probs: Sequence[float], eps: float = 1e-12) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def trapezoid_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    return float(np.trapz(y, x))


def spans_to_char_mask(text_len: int, spans: Sequence[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(text_len, dtype=np.int32)
    for start, end in spans:
        s = max(0, int(start))
        e = min(text_len, int(end))
        if e > s:
            mask[s:e] = 1
    return mask


def f1_iou_from_masks(pred: np.ndarray, gold: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gold = gold.astype(bool)

    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, np.logical_not(gold)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gold).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    union = int(np.logical_or(pred, gold).sum())
    intersection = int(np.logical_and(pred, gold).sum())
    iou = intersection / union if union > 0 else 0.0
    return f1, iou


def parse_lambdas(raw: str) -> Tuple[float, float, float, float]:
    parts = [x.strip() for x in raw.split(",") if x.strip() != ""]
    if len(parts) != 4:
        raise ValueError("--lambdas expects 4 comma-separated values, e.g. 1,1,1,1")
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def parse_q_values(raw: str) -> Tuple[int, ...]:
    items = [x.strip() for x in raw.split(",") if x.strip() != ""]
    if not items:
        raise ValueError("--q-values must not be empty")
    values = tuple(sorted(set(int(x) for x in items)))
    if values[0] <= 0:
        raise ValueError("q-values must be positive percentages")
    if values[-1] > 100:
        raise ValueError("q-values must be <= 100")
    return values


def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))
