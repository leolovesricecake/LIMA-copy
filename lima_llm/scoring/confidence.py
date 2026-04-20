from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..utils import entropy_from_probs


def confidence_score(label_probabilities: Sequence[float], num_classes: int) -> float:
    if num_classes <= 1:
        return 1.0
    entropy = entropy_from_probs(label_probabilities)
    max_entropy = math.log(float(num_classes))
    if max_entropy <= 1e-12:
        return 1.0
    conf = 1.0 - (entropy / max_entropy)
    return float(max(0.0, min(1.0, conf)))
