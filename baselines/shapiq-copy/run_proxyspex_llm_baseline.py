from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BASELINE_DIR = Path(__file__).resolve().parent
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

_LOCAL_SHAPIQ_SRC = _BASELINE_DIR / "src"
_PROXY_FALLBACK_WARNED: set[tuple[str, str]] = set()
_PROXY_HPO_FALLBACK_WARNED: set[tuple[str, int]] = set()
_MAX_PROXY_HPO_CV_SPLITS = 5
_MIN_PROXY_HPO_TEST_FOLD_SIZE = 2

from mobius.core.config import load_json_object
from mobius.core.artifacts import (
    normalize_observation_artifact,
    normalize_surrogate_artifact,
    predict_surrogate,
)
from mobius.core.results import ResultStore, default_run_dir
from mobius.core.runtime import set_seed
from mobius.core.schema import AttributionResult, TextChunk
from mobius.data import load_dataset_bundle
from mobius.evaluation.evaluator import evaluate_run
from mobius.models.hf import HFBackbone
from mobius.models.prompting import build_classification_prompt
from mobius.text.chunks import (
    build_chunks,
    compose_text,
    normalize_chunker,
    normalize_eval_granularity,
)
from mobius.text.coalitions import (
    EMPTY_TEXT,
    active_chunk_ids as _active_unit_ids_from_visible_span,
    visible_text_span as _prompt_visible_text_span_after_left_truncation,
)
from mobius.values.classification import (
    attribution_values as attribution_values_from_label_scores,
    normalize_value_function as normalize_attribution_value_function,
    probabilities as probabilities_from_label_scores,
)


METHOD_NAME = "proxyspex"


def parse_q_values(raw: str) -> List[int]:
    """Parse comma-separated evaluation percentages."""

    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def load_adaptive_overrides(raw: str | None) -> Dict[str, Any] | None:
    """Load adaptive chunk overrides from inline JSON or a file."""

    return load_json_object(raw)


def _compose_coalition_text(
    *,
    units: Sequence[TextChunk],
    player_to_chunk_id: Sequence[int],
    coalition_row: Sequence[bool],
) -> str:
    """Compose a ProxySPEX keep coalition over active explanation chunks."""

    selected = [
        int(player_to_chunk_id[index])
        for index, keep in enumerate(coalition_row)
        if bool(keep)
    ]
    text = compose_text(units, selected)
    return text if text else EMPTY_TEXT


def build_parser() -> argparse.ArgumentParser:
    """Handle the build parser step in this retained baseline."""

    parser = argparse.ArgumentParser(description="ProxySPEX baseline for HF causal language models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "sst2",
            "eraser_movie_reviews",
            "imdb",
            "rotten_tomatoes",
            "emotion",
            "ag_news",
        ],
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="predicted", choices=["gold", "predicted"])
    parser.add_argument(
        "--value-function",
        type=str,
        default="predicted_probability",
        choices=["target_probability", "predicted_probability", "predicted_class_margin"],
        help=(
            "Coalition payoff. target_probability follows --target-mode and preserves the "
            "original baseline with --target-mode gold; predicted_* always explain the "
            "full-input predicted class."
        ),
    )
    parser.add_argument("--eval-q-values", type=str, default="5,10,20,50")
    parser.add_argument("--eval-granularity", type=str, default="word", choices=["token", "word"])
    parser.add_argument(
        "--chunker",
        type=str,
        default="word",
        choices=["token", "word", "adaptive"],
        help="Explanation chunker used as ProxySPEX players. Evaluation remains controlled by --eval-granularity.",
    )
    parser.add_argument(
        "--adaptive-profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Profile for --chunker adaptive.",
    )
    parser.add_argument(
        "--adaptive-overrides-json",
        type=str,
        default=None,
        help="Path to a JSON object or inline JSON object with adaptive chunker overrides.",
    )

    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--index", type=str, default="FBII")
    parser.add_argument("--proxy-model", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "tree"])
    parser.add_argument(
        "--proxy-n-jobs",
        type=int,
        default=1,
        help="CPU workers used by LightGBM/XGBoost proxies. Keep 1 for responsive Ctrl+C and stable memory.",
    )
    parser.add_argument(
        "--quiet-proxy",
        dest="quiet_proxy",
        action="store_true",
        default=True,
        help="Suppress native stdout/stderr emitted by LightGBM/XGBoost while each proxy is fitted.",
    )
    parser.add_argument(
        "--no-quiet-proxy",
        dest="quiet_proxy",
        action="store_false",
        help="Do not suppress native proxy logs. Useful only when debugging the proxy backend itself.",
    )
    parser.add_argument(
        "--sampling-weight-mode",
        type=str,
        default="uniform_coalition",
        choices=["uniform_coalition", "uniform_size"],
        help=(
            "Coalition-size sampling weights. uniform_coalition is a stable log-comb version of "
            "ProxySPEX's default comb(n, k) weights."
        ),
    )
    parser.add_argument("--hpo", dest="hpo", action="store_true", default=True)
    parser.add_argument("--no-hpo", dest="hpo", action="store_false")
    parser.add_argument("--pairing-trick", action="store_true")
    parser.add_argument("--top-order", action="store_true")
    parser.add_argument("--interaction-metadata-limit", type=int, default=256)

    parser.add_argument("--base-save-dir", type=str, default="results")
    parser.add_argument("--save-dir", type=str, default="baselines/proxyspex")
    parser.add_argument(
        "--output-level",
        type=str,
        default="standard",
        choices=["minimal", "standard", "debug"],
    )
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _require_runtime_dependencies():
    """Handle the require runtime dependencies step in this retained baseline."""

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "ProxySPEX LLM runner requires torch and transformers for HFBackbone. "
            "Please install the dependencies listed in baselines/shapiq-copy/README_zh.md."
        ) from exc

    def _import_proxyspex():
        """Handle the import proxyspex step in this retained baseline."""

        from shapiq.approximator.proxy.proxyspex import ProxySPEX

        return ProxySPEX

    # The paper artifact protocol depends on fields added to this vendored ProxySPEX,
    # so an unrelated site-packages release must not be selected silently.
    if _LOCAL_SHAPIQ_SRC.exists():
        local_source = str(_LOCAL_SHAPIQ_SRC)
        if local_source in sys.path:
            sys.path.remove(local_source)
        sys.path.insert(0, local_source)
        for module_name in list(sys.modules):
            if module_name == "shapiq" or module_name.startswith("shapiq."):
                del sys.modules[module_name]
    try:
        return _import_proxyspex()
    except Exception as first_exc:
        missing_module = (
            first_exc.name
            if isinstance(first_exc, ModuleNotFoundError)
            else None
        )
        missing_detail = (
            f" Missing Python module: {missing_module!r}."
            if missing_module
            else ""
        )
        raise RuntimeError(
            "Failed to import ProxySPEX from the vendored shapiq source at "
            f"{_LOCAL_SHAPIQ_SRC}.{missing_detail} Ensure the checkout is complete, then install "
            "its proxy dependencies with: pip install -e 'baselines/shapiq-copy[proxy]'."
        ) from first_exc


def _effective_target_mode(value_function: str, requested_target_mode: str) -> str:
    """Handle the effective target mode step in this retained baseline."""

    normalized = normalize_attribution_value_function(value_function)
    if normalized in {"predicted_probability", "predicted_class_margin"}:
        return "predicted"
    return str(requested_target_mode)


def _result_config(args, bundle=None) -> Dict[str, Any]:
    """Build the scientific ProxySPEX configuration written once per run."""

    verbalizers = list(bundle.verbalizers) if bundle is not None else []
    prompt = (
        build_classification_prompt(
            dataset_name=str(args.dataset),
            verbalizers=verbalizers,
        ).to_config()
        if verbalizers
        else {}
    )
    return {
        "method": METHOD_NAME,
        "output_level": str(getattr(args, "output_level", "standard")),
        "dataset": {
            "name": str(args.dataset),
            "split": str(args.split),
            "max_samples": args.max_samples,
            "dataset_cache_dir": args.dataset_cache_dir,
            "verbalizers": verbalizers,
        },
        "model": {
            "type": "hf_causal_lm",
            "model_path": str(args.model_path),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "max_length": int(args.max_length),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "prompt": prompt,
        "budget": int(args.budget),
        "seed": int(args.seed),
        "max_order": int(args.max_order),
        "k": int(args.k),
        "value_function": normalize_attribution_value_function(args.value_function),
        "target_mode": _effective_target_mode(args.value_function, args.target_mode),
        "chunker": normalize_chunker(args.chunker),
        "adaptive_profile": str(args.adaptive_profile),
        "adaptive_overrides": getattr(args, "adaptive_overrides", None),
        "eval_granularity": normalize_eval_granularity(args.eval_granularity),
        "eval_q_values": parse_q_values(args.eval_q_values),
        "index": str(args.index),
        "proxy_model": str(args.proxy_model),
        "hpo": bool(args.hpo),
        "sampling_weight_mode": str(args.sampling_weight_mode),
        "pairing_trick": bool(args.pairing_trick),
        "top_order": bool(args.top_order),
        "projector": "signed_equal_share",
    }


def _output_root(args, bundle=None) -> Path:
    """Resolve the schema-v2 directory below the requested baseline root."""

    return default_run_dir(
        Path(args.base_save_dir) / str(args.save_dir),
        _result_config(args, bundle),
    )


def _stable_uniform_coalition_sampling_weights(n_players: int) -> np.ndarray:
    """Handle the stable uniform coalition sampling weights step in this retained baseline."""

    n = int(n_players)
    if n < 0:
        raise ValueError("n_players must be non-negative")
    if n == 0:
        return np.ones(1, dtype=np.float64)

    log_weights = np.asarray(
        [
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            for k in range(n + 1)
        ],
        dtype=np.float64,
    )
    log_weights = log_weights - float(np.max(log_weights))
    weights = np.exp(log_weights)
    tiny = np.finfo(np.float64).tiny
    weights[~np.isfinite(weights)] = 0.0
    weights = np.maximum(weights, tiny)
    return weights.astype(np.float64)


def _sampling_weights(n_players: int, mode: str) -> np.ndarray:
    """Handle the sampling weights step in this retained baseline."""

    value = str(mode).strip().lower()
    if value == "uniform_size":
        return np.ones(int(n_players) + 1, dtype=np.float64)
    if value == "uniform_coalition":
        return _stable_uniform_coalition_sampling_weights(int(n_players))
    raise ValueError(f"Unsupported sampling weight mode: {mode!r}")


def _warn_proxy_fallback(requested: str, effective: str, reason: str) -> None:
    """Handle the warn proxy fallback step in this retained baseline."""

    key = (str(requested), str(effective))
    if key in _PROXY_FALLBACK_WARNED:
        return
    _PROXY_FALLBACK_WARNED.add(key)
    warnings.warn(
        f"Requested ProxySPEX proxy_model={requested!r}, using {effective!r} instead. {reason}",
        stacklevel=2,
    )


def _expected_proxy_fit_sample_count(n_players: int, budget: int) -> int:
    """Return the number of distinct coalition rows ProxySPEX will fit the proxy on."""

    n = max(0, int(n_players))
    requested_budget = max(0, int(budget))
    if n == 0 or requested_budget == 0:
        return 0
    if n >= max(0, (requested_budget - 1).bit_length()):
        return requested_budget
    return min(requested_budget, 1 << n)


def _proxy_hpo_cv_splits(n_fit_samples: int | None) -> int | None:
    """Handle the proxy hpo cv splits step in this retained baseline."""

    if n_fit_samples is None:
        return _MAX_PROXY_HPO_CV_SPLITS
    # R2 scoring is undefined for a one-sample test fold. Keep every fold at size >= 2.
    cv = min(_MAX_PROXY_HPO_CV_SPLITS, int(n_fit_samples) // _MIN_PROXY_HPO_TEST_FOLD_SIZE)
    return cv if cv >= 2 else None


def _warn_proxy_hpo_fallback(proxy_name: str, n_fit_samples: int) -> None:
    """Handle the warn proxy hpo fallback step in this retained baseline."""

    key = (str(proxy_name), int(n_fit_samples))
    if key in _PROXY_HPO_FALLBACK_WARNED:
        return
    _PROXY_HPO_FALLBACK_WARNED.add(key)
    warnings.warn(
        "Requested ProxySPEX proxy HPO, but only "
        f"{int(n_fit_samples)} coalition samples are available; using the bare "
        f"{proxy_name!r} proxy for this short-sample case.",
        stacklevel=2,
    )


def _build_proxy_model(args, *, n_fit_samples: int | None = None):
    """Handle the build proxy model step in this retained baseline."""

    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeRegressor

    requested = str(args.proxy_model)
    seed = int(args.seed)
    n_jobs = int(args.proxy_n_jobs)
    hpo_cv_splits = _proxy_hpo_cv_splits(n_fit_samples) if bool(args.hpo) else None

    if requested == "tree":
        return DecisionTreeRegressor(random_state=seed), "tree"

    if requested == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=-1,
                verbose=-1,
            )
            if bool(args.hpo) and hpo_cv_splits is not None:
                return (
                    GridSearchCV(
                        estimator=estimator,
                        param_grid={
                            "max_depth": [3, 5],
                            "max_iter": [500, 1000],
                            "learning_rate": [0.01, 0.1],
                        },
                        scoring="r2",
                        cv=int(hpo_cv_splits),
                        verbose=0,
                        n_jobs=1,
                    ),
                    f"lightgbm_gridsearch_quiet_cv-{int(hpo_cv_splits)}",
                )
            if bool(args.hpo) and n_fit_samples is not None:
                _warn_proxy_hpo_fallback("lightgbm", int(n_fit_samples))
                return estimator, "lightgbm_quiet_hpo_disabled_small_sample"
            return estimator, "lightgbm_quiet"
        except ImportError:
            requested = "xgboost"
            _warn_proxy_fallback("lightgbm", "xgboost/tree", "LightGBM is not installed.")

    if requested == "xgboost":
        try:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=0,
            )
            if bool(args.hpo) and hpo_cv_splits is not None:
                return (
                    GridSearchCV(
                        estimator=estimator,
                        param_grid={
                            "max_depth": [3, 5],
                            "n_estimators": [500, 1000],
                            "learning_rate": [0.01, 0.1],
                        },
                        scoring="r2",
                        cv=int(hpo_cv_splits),
                        verbose=0,
                        n_jobs=1,
                    ),
                    f"xgboost_gridsearch_quiet_cv-{int(hpo_cv_splits)}",
                )
            if bool(args.hpo) and n_fit_samples is not None:
                _warn_proxy_hpo_fallback("xgboost", int(n_fit_samples))
                return estimator, "xgboost_quiet_hpo_disabled_small_sample"
            return estimator, "xgboost_quiet"
        except ImportError:
            _warn_proxy_fallback(str(args.proxy_model), "tree", "Boosting backend is not installed.")

    return DecisionTreeRegressor(random_state=seed), "tree_fallback"


class _SuppressNativeOutput:
    """Temporarily redirects process-level stdout/stderr, including C/C++ library logs."""

    def __init__(self) -> None:
        """Handle the init step in this retained baseline."""

        self._null_fd: int | None = None
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None

    def __enter__(self) -> "_SuppressNativeOutput":
        """Handle the enter step in this retained baseline."""

        sys.stdout.flush()
        sys.stderr.flush()
        self._null_fd = os.open(os.devnull, os.O_WRONLY)
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Handle the exit step in this retained baseline."""

        try:
            sys.stdout.flush()
            sys.stderr.flush()
            if self._stdout_fd is not None:
                os.dup2(self._stdout_fd, 1)
            if self._stderr_fd is not None:
                os.dup2(self._stderr_fd, 2)
        finally:
            for fd in (self._stdout_fd, self._stderr_fd, self._null_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            self._stdout_fd = None
            self._stderr_fd = None
            self._null_fd = None
        return False


@contextmanager
def _maybe_suppress_native_output(enabled: bool):
    """Handle the maybe suppress native output step in this retained baseline."""

    with (_SuppressNativeOutput() if enabled else nullcontext()):
        yield


class ProxySPEXCoalitionGame:
    def __init__(
        self,
        *,
        units: Sequence[TextChunk],
        player_to_chunk_id: Sequence[int],
        backbone: HFBackbone,
        verbalizers: Sequence[str],
        target_label: int,
        value_function: str = "target_probability",
    ) -> None:
        """Handle the init step in this retained baseline."""

        self.units = list(units)
        self.player_to_chunk_id = [int(x) for x in player_to_chunk_id]
        self.backbone = backbone
        self.verbalizers = list(verbalizers)
        self.target_label = int(target_label)
        self.value_function = normalize_attribution_value_function(value_function)
        self.score_cache: Dict[str, np.ndarray] = {}
        self.observed_rows: List[np.ndarray] = []
        self.call_count = 0
        self.row_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.unique_text_count = 0

    def __call__(self, coalitions_matrix: np.ndarray) -> np.ndarray:
        """Handle the call step in this retained baseline."""

        matrix = np.asarray(coalitions_matrix, dtype=bool)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self.observed_rows.extend(np.asarray(row, dtype=bool).copy() for row in matrix)

        self.call_count += 1
        self.row_count += int(matrix.shape[0])
        texts = [
            _compose_coalition_text(
                units=self.units,
                player_to_chunk_id=self.player_to_chunk_id,
                coalition_row=row,
            )
            for row in matrix
        ]

        missing = []
        seen_missing = set()
        for text in texts:
            if text in self.score_cache:
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)

        if missing:
            score_fn = getattr(self.backbone, "predict_label_scores_batch", None)
            if callable(score_fn):
                label_scores = np.asarray(score_fn(missing, self.verbalizers), dtype=np.float64)
            else:
                probabilities = np.asarray(
                    self.backbone.predict_label_probs_batch(missing, self.verbalizers),
                    dtype=np.float64,
                )
                label_scores = np.log(np.clip(probabilities, 1e-30, 1.0))
            values = attribution_values_from_label_scores(
                label_scores,
                target_class=self.target_label,
                value_function=self.value_function,
            )
            for idx, text in enumerate(missing):
                self.score_cache[text] = np.asarray(
                    label_scores[idx],
                    dtype=np.float64,
                )
            self.unique_text_count = int(len(self.score_cache))

        scores = np.vstack([self.score_cache[text] for text in texts])
        return attribution_values_from_label_scores(
            scores,
            target_class=self.target_label,
            value_function=self.value_function,
        )

    def label_scores_for(self, coalitions_matrix: np.ndarray) -> np.ndarray:
        """Return cached all-class scores for already evaluated coalitions."""

        matrix = np.asarray(coalitions_matrix, dtype=bool)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        texts = [
            _compose_coalition_text(
                units=self.units,
                player_to_chunk_id=self.player_to_chunk_id,
                coalition_row=row,
            )
            for row in matrix
        ]
        missing = [text for text in texts if text not in self.score_cache]
        if missing:
            raise RuntimeError(
                "ProxySPEX observation export requested coalitions that were not scored."
            )
        return np.vstack([self.score_cache[text] for text in texts])

    def observed_matrix(self) -> np.ndarray:
        """Return coalition rows requested from the game in their original order."""

        if not self.observed_rows:
            return np.zeros((0, len(self.player_to_chunk_id)), dtype=bool)
        return np.vstack(self.observed_rows).astype(bool, copy=False)

    def stats(self) -> Dict[str, Any]:
        """Handle the stats step in this retained baseline."""

        return {
            "call_count": int(self.call_count),
            "row_count": int(self.row_count),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "unique_text_count": int(len(self.score_cache)),
            "value_function": self.value_function,
        }


def _interaction_items(interaction_values) -> List[Tuple[Tuple[int, ...], float]]:
    """Handle the interaction items step in this retained baseline."""

    raw = getattr(interaction_values, "dict_values", None)
    if raw is None:
        raw = getattr(interaction_values, "interactions", {})
    return [
        (tuple(int(i) for i in interaction), float(value))
        for interaction, value in dict(raw).items()
    ]


def _project_interactions_to_chunk_scores(
    *,
    interaction_items: Sequence[Tuple[Tuple[int, ...], float]],
    player_to_chunk_id: Sequence[int],
    total_chunk_count: int,
) -> List[float]:
    """Handle the project interactions to chunk scores step in this retained baseline."""

    scores = [0.0] * int(total_chunk_count)
    for interaction, value in interaction_items:
        if len(interaction) == 0:
            continue
        share = float(value) / float(len(interaction))
        for player_idx in interaction:
            if 0 <= int(player_idx) < len(player_to_chunk_id):
                chunk_id = int(player_to_chunk_id[int(player_idx)])
                if 0 <= chunk_id < len(scores):
                    scores[chunk_id] += share
    return [float(x) for x in scores]


def _summarize_interactions(
    interaction_items: Sequence[Tuple[Tuple[int, ...], float]],
    *,
    limit: int,
) -> Dict[str, Any]:
    """Handle the summarize interactions step in this retained baseline."""

    order_counts: Dict[str, int] = {}
    order_signed_sums: Dict[str, float] = {}
    order_abs_sums: Dict[str, float] = {}
    for interaction, value in interaction_items:
        order = str(len(interaction))
        order_counts[order] = order_counts.get(order, 0) + 1
        order_signed_sums[order] = order_signed_sums.get(order, 0.0) + float(value)
        order_abs_sums[order] = order_abs_sums.get(order, 0.0) + abs(float(value))

    non_empty = [(interaction, value) for interaction, value in interaction_items if len(interaction) > 0]
    top = sorted(non_empty, key=lambda item: (-abs(float(item[1])), item[0]))[: max(0, int(limit))]
    return {
        "total_interaction_count": int(len(interaction_items)),
        "non_empty_interaction_count": int(len(non_empty)),
        "order_counts": order_counts,
        "order_signed_sums": order_signed_sums,
        "order_abs_sums": order_abs_sums,
        "top_by_abs_value": [
            {
                "players": [int(i) for i in interaction],
                "value": float(value),
            }
            for interaction, value in top
        ],
        "top_by_abs_value_limit": int(limit),
    }


def _spectral_terms_payload(
    coefficients: Mapping[Tuple[int, ...], float] | None,
) -> Dict[str, Any]:
    """Serialize one Fourier dictionary with its baseline separated."""

    values = dict(coefficients or {})
    return {
        "intercept": float(values.get((), 0.0)),
        "terms": [
            {
                "players": [int(player) for player in interaction],
                "coefficient": float(value),
            }
            for interaction, value in sorted(
                values.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
            if interaction
        ],
        "support_size": int(sum(bool(interaction) for interaction in values)),
    }


def _rank_desc_scores(scores: Sequence[float], k: int) -> tuple[List[int], List[int]]:
    """Handle the rank desc scores step in this retained baseline."""

    ranking = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))]
    return ranking, list(ranking[: max(0, int(k))])


def _counter_delta(before: Mapping[str, float | int], after: Mapping[str, float | int]) -> Dict[str, float | int]:
    """Subtract numeric scorer counters for one attribution sample."""

    out: Dict[str, float | int] = {}
    for key, value in after.items():
        prev = before.get(key, 0)
        if isinstance(value, float) or isinstance(prev, float):
            out[key] = float(value) - float(prev)
        else:
            out[key] = int(value) - int(prev)
    return out


def _sample_target_label(
    sample,
    bundle,
    backbone: HFBackbone,
    target_mode: str,
    value_function: str,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Handle the sample target label step in this retained baseline."""

    score_fn = getattr(backbone, "predict_label_scores", None)
    if callable(score_fn):
        label_scores = np.asarray(score_fn(sample.text, bundle.verbalizers), dtype=np.float64)
        probs = probabilities_from_label_scores(label_scores.reshape(1, -1))[0].astype(np.float32)
    else:
        probs = np.asarray(backbone.predict_label_probs(sample.text, bundle.verbalizers), dtype=np.float32)
        label_scores = np.log(np.clip(probs.astype(np.float64), 1e-30, 1.0))
    effective_target_mode = _effective_target_mode(value_function, target_mode)
    if effective_target_mode == "predicted":
        return int(np.argmax(probs)), probs, label_scores
    return int(sample.label), probs, label_scores


def _explain_sample(
    *,
    sample,
    bundle,
    backbone: HFBackbone,
    args,
    ProxySPEX,
) -> AttributionResult:
    """Run native ProxySPEX and adapt its output to schema v2."""
    t0 = time.time()
    timing: Dict[str, float] = {}
    counter_before = backbone.snapshot_counters()

    value_function = normalize_attribution_value_function(
        getattr(args, "value_function", "predicted_probability")
    )
    target_label, full_probs, full_label_scores = _sample_target_label(
        sample,
        bundle,
        backbone,
        args.target_mode,
        value_function,
    )
    target_label_text = str(bundle.verbalizers[target_label])

    t_units = time.time()
    adaptive_overrides = getattr(args, "adaptive_overrides", None)
    if adaptive_overrides is None and getattr(args, "adaptive_overrides_json", None):
        adaptive_overrides = load_adaptive_overrides(getattr(args, "adaptive_overrides_json"))
    chunking = build_chunks(
        text=sample.text,
        chunker=getattr(args, "chunker", "word"),
        tokenizer=backbone.tokenizer,
        adaptive_profile=getattr(args, "adaptive_profile", "balanced"),
        adaptive_overrides=adaptive_overrides,
    )
    units = list(chunking.chunks)
    fallback_used = bool(chunking.fallback_used)
    segmentation_strategy = str(chunking.strategy)
    chunk_diagnostics = dict(chunking.diagnostics)
    if normalize_chunker(getattr(args, "chunker", "word")) == "token" and fallback_used:
        raise RuntimeError(
            "ProxySPEX token chunker requires tokenizer offset_mapping support. "
            "Use a fast tokenizer or choose --chunker word/adaptive."
        )
    timing["unit_build_seconds"] = time.time() - t_units

    truncation = _prompt_visible_text_span_after_left_truncation(
        tokenizer=backbone.tokenizer,
        text=sample.text,
        label_text=target_label_text,
        max_length=int(args.max_length),
        prompt_prefix=str(getattr(backbone, "prompt_prefix", "Text:\n")),
        prompt_suffix=str(getattr(backbone, "prompt_suffix", "\nLabel:")),
        label_token_reserve=(
            int(backbone.label_token_reserve())
            if callable(getattr(backbone, "label_token_reserve", None))
            else None
        ),
        label_prefix=str(getattr(backbone, "label_prefix", " ")),
    )
    active_chunk_ids = _active_unit_ids_from_visible_span(
        units,
        int(truncation["visible_start_char"]),
        int(truncation["visible_end_char"]),
    )
    player_to_chunk_id = list(active_chunk_ids)
    proxy_fit_sample_count = 0
    effective_proxy_hpo = False
    proxy_hpo_cv_splits = None
    tree_conversion_backends: List[str] = []
    tree_fourier_validation_max_abs_error = None
    serialized_fourier_validation_max_abs_error = None
    attribution_counter_before = backbone.snapshot_counters()

    if len(player_to_chunk_id) == 0:
        interaction_values = None
        interaction_items: List[Tuple[Tuple[int, ...], float]] = []
        game_stats: Dict[str, Any] = {
            "call_count": 0,
            "row_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "unique_text_count": 0,
        }
        chunk_scores = [0.0] * len(units)
        timing["proxyspex_seconds"] = 0.0
        training_matrix = np.zeros((0, 0), dtype=bool)
        training_label_scores = np.zeros(
            (0, len(bundle.verbalizers)),
            dtype=np.float64,
        )
        training_values = np.zeros(0, dtype=np.float64)
        unrefined_fourier: Dict[Tuple[int, ...], float] = {}
        refined_fourier: Dict[Tuple[int, ...], float] = {}
    else:
        game = ProxySPEXCoalitionGame(
            units=units,
            player_to_chunk_id=player_to_chunk_id,
            backbone=backbone,
            verbalizers=bundle.verbalizers,
            target_label=target_label,
            value_function=value_function,
        )
        effective_max_order = min(int(args.max_order), len(player_to_chunk_id))
        sampling_weights = _sampling_weights(
            n_players=len(player_to_chunk_id),
            mode=str(args.sampling_weight_mode),
        )
        proxy_fit_sample_count = _expected_proxy_fit_sample_count(
            n_players=len(player_to_chunk_id),
            budget=int(args.budget),
        )
        proxy_model, effective_proxy_model = _build_proxy_model(
            args,
            n_fit_samples=proxy_fit_sample_count,
        )
        proxy_hpo_cv_splits = getattr(proxy_model, "cv", None)
        effective_proxy_hpo = proxy_hpo_cv_splits is not None and hasattr(proxy_model, "param_grid")
        approximator = ProxySPEX(
            n=len(player_to_chunk_id),
            max_order=effective_max_order,
            index=str(args.index),
            proxy_model=proxy_model,
            hpo=False,
            sampling_weights=sampling_weights,
            pairing_trick=bool(args.pairing_trick),
            top_order=bool(args.top_order),
            random_state=int(args.seed),
        )
        t_proxy = time.time()
        with _maybe_suppress_native_output(bool(args.quiet_proxy)):
            interaction_values = approximator.approximate(budget=int(args.budget), game=game)
        timing["proxyspex_seconds"] = time.time() - t_proxy
        tree_conversion_backends = [
            str(value)
            for value in (getattr(approximator, "tree_conversion_backends_", None) or ())
        ]
        raw_validation_error = getattr(
            approximator,
            "tree_fourier_validation_max_abs_error_",
            None,
        )
        if raw_validation_error is not None:
            tree_fourier_validation_max_abs_error = float(raw_validation_error)
        interaction_items = _interaction_items(interaction_values)
        game_stats = game.stats()
        chunk_scores = _project_interactions_to_chunk_scores(
            interaction_items=interaction_items,
            player_to_chunk_id=player_to_chunk_id,
            total_chunk_count=len(units),
        )
        raw_training_matrix = getattr(approximator, "coalitions_matrix_", None)
        raw_unrefined_fourier = getattr(
            approximator,
            "unrefined_fourier_",
            None,
        )
        raw_refined_fourier = getattr(
            approximator,
            "refined_fourier_",
            None,
        )
        if (
            raw_training_matrix is None
            or raw_unrefined_fourier is None
            or raw_refined_fourier is None
        ):
            raise RuntimeError(
                "The loaded ProxySPEX implementation does not expose the "
                "training coalitions and refined Fourier representation required "
                "by the paper artifact protocol. Install the local shapiq-copy."
            )
        training_matrix = np.asarray(raw_training_matrix, dtype=bool)
        training_label_scores = game.label_scores_for(training_matrix)
        training_values = attribution_values_from_label_scores(
            training_label_scores,
            target_class=target_label,
            value_function=value_function,
        )
        unrefined_fourier = dict(raw_unrefined_fourier)
        refined_fourier = dict(raw_refined_fourier)
    attribution_counter_after = backbone.snapshot_counters()
    attribution_forward_delta = _counter_delta(
        attribution_counter_before,
        attribution_counter_after,
    )

    chunk_scores_by_id = {idx: float(score) for idx, score in enumerate(chunk_scores)}
    active_ranking = sorted(
        player_to_chunk_id,
        key=lambda chunk_id: (-float(chunk_scores[int(chunk_id)]), int(chunk_id)),
    )
    inactive_chunk_ids = [
        unit.chunk_id for unit in units if unit.chunk_id not in set(player_to_chunk_id)
    ]
    chunk_ranking = [int(value) for value in active_ranking + inactive_chunk_ids]
    selected = chunk_ranking[: min(int(args.k), len(player_to_chunk_id))]
    selected_text = compose_text(units, selected)
    counter_after = backbone.snapshot_counters()
    elapsed = time.time() - t0

    interaction_summary = _summarize_interactions(
        interaction_items,
        limit=int(args.interaction_metadata_limit),
    )
    observation_artifact = normalize_observation_artifact(
        {
            "sample_id": sample.sample_id,
            "method": METHOD_NAME,
            "n_features": len(player_to_chunk_id),
            "keep_masks": training_matrix,
            "label_scores": training_label_scores,
            "attribution_values": training_values,
        }
    )
    unrefined_payload = _spectral_terms_payload(unrefined_fourier)
    refined_payload = _spectral_terms_payload(refined_fourier)
    surrogate_payload = {
            "sample_id": sample.sample_id,
            "method": METHOD_NAME,
            "n_features": len(player_to_chunk_id),
            "player_to_chunk_id": player_to_chunk_id,
            "observation_file": f"../observations/{sample.sample_id}.npz",
            "observation_digest": observation_artifact["digest"],
            "value_function": value_function,
            "target_mode": _effective_target_mode(
                value_function,
                args.target_mode,
            ),
            "target_label": target_label,
            "predictor": {
                "type": "refined_fourier",
                "basis": "fourier",
                "intercept": refined_payload["intercept"],
                "terms": refined_payload["terms"],
            },
            "unrefined_fourier": unrefined_payload,
            "refined_fourier": refined_payload,
            "final_interactions": [
                {
                    "players": [int(player) for player in interaction],
                    "coefficient": float(value),
                }
                for interaction, value in interaction_items
            ],
            "fit_config": {
                "proxy_model": str(args.proxy_model),
                "hpo": bool(args.hpo),
                "max_order": int(args.max_order),
                "index": str(args.index),
                "sampling_weight_mode": str(args.sampling_weight_mode),
                "tree_conversion_backends": tree_conversion_backends,
                "tree_fourier_validation_max_abs_error": (
                    tree_fourier_validation_max_abs_error
                ),
            },
        }
    if player_to_chunk_id:
        # Validate the portable predictor on native training and local anchor masks.
        n_players = len(player_to_chunk_id)
        full_row = np.ones((1, n_players), dtype=bool)
        empty_row = np.zeros((1, n_players), dtype=bool)
        singleton_deletions = np.repeat(full_row, n_players, axis=0)
        singleton_deletions[np.arange(n_players), np.arange(n_players)] = False
        validation_matrix = np.unique(
            np.vstack(
                [
                    np.asarray(training_matrix, dtype=bool),
                    empty_row,
                    full_row,
                    singleton_deletions,
                ]
            ),
            axis=0,
        )
        portable = normalize_surrogate_artifact(surrogate_payload)
        native_predictions = np.asarray(
            approximator.predict_refined_fourier(validation_matrix),
            dtype=np.float64,
        )
        serialized_predictions = np.asarray(
            predict_surrogate(portable, validation_matrix),
            dtype=np.float64,
        )
        serialized_fourier_validation_max_abs_error = float(
            np.max(np.abs(native_predictions - serialized_predictions))
        )
        if serialized_fourier_validation_max_abs_error > 1e-9:
            raise RuntimeError(
                "Serialized refined Fourier predictor differs from native "
                "ProxySPEX prediction: max_abs_error="
                f"{serialized_fourier_validation_max_abs_error:.3e}."
            )
    surrogate_payload["fit_config"][
        "serialized_fourier_validation_max_abs_error"
    ] = serialized_fourier_validation_max_abs_error
    surrogate_payload["fit_config"][
        "serialized_fourier_validation_tolerance"
    ] = 1e-9
    surrogate_artifact = normalize_surrogate_artifact(surrogate_payload)

    metadata = {
        "proxyspex_method": METHOD_NAME,
        "proxyspex_index": str(args.index),
        "proxyspex_budget": int(args.budget),
        "proxyspex_requested_max_order": int(args.max_order),
        "proxyspex_effective_max_order": int(min(int(args.max_order), len(player_to_chunk_id)))
        if player_to_chunk_id
        else 0,
        "proxyspex_proxy_model": str(args.proxy_model),
        "proxyspex_effective_proxy_model": str(effective_proxy_model) if player_to_chunk_id else "none",
        "proxyspex_proxy_fit_sample_count": int(proxy_fit_sample_count),
        "proxyspex_sampling_weight_mode": str(args.sampling_weight_mode),
        "proxyspex_proxy_n_jobs": int(args.proxy_n_jobs),
        "proxyspex_quiet_proxy": bool(args.quiet_proxy),
        "proxyspex_hpo": bool(args.hpo),
        "proxyspex_effective_hpo": bool(effective_proxy_hpo),
        "proxyspex_proxy_hpo_cv_splits": int(proxy_hpo_cv_splits)
        if proxy_hpo_cv_splits is not None
        else None,
        "proxyspex_tree_conversion_backends": tree_conversion_backends,
        "proxyspex_tree_fourier_validation_max_abs_error": (
            tree_fourier_validation_max_abs_error
        ),
        "proxyspex_serialized_fourier_validation_max_abs_error": (
            serialized_fourier_validation_max_abs_error
        ),
        "proxyspex_pairing_trick": bool(args.pairing_trick),
        "proxyspex_top_order": bool(args.top_order),
        "value_function": value_function,
        "projection_strategy": "signed_equal_share",
        "target_mode": _effective_target_mode(value_function, args.target_mode),
        "target_mode_requested": str(args.target_mode),
        "target_label": int(target_label),
        "target_label_text": target_label_text,
        "full_label_probabilities": [float(x) for x in full_probs.tolist()],
        "full_label_scores": [float(x) for x in full_label_scores.tolist()],
        "eval_granularity": str(args.eval_granularity),
        "proxyspex_chunker": normalize_chunker(getattr(args, "chunker", "word")),
        "chunk_diagnostics": chunk_diagnostics,
        "explain_chunk_strategy": str(chunk_diagnostics.get("chunk_strategy", segmentation_strategy)),
        "explain_tokenizer_fallback_used": bool(fallback_used),
        "segmentation_strategy": segmentation_strategy,
        "tokenizer_fallback_used": bool(fallback_used),
        "truncation": {
            **truncation,
            "active_chunk_count": int(len(player_to_chunk_id)),
            "inactive_chunk_count": int(len(inactive_chunk_ids)),
            "active_chunk_ids": [int(x) for x in player_to_chunk_id],
            "inactive_chunk_ids": [int(x) for x in inactive_chunk_ids],
        },
        "player_to_chunk_id": [int(x) for x in player_to_chunk_id],
        "interaction_summary": interaction_summary,
        "interaction_baseline_value": float(getattr(interaction_values, "baseline_value", 0.0))
        if interaction_values is not None
        else 0.0,
        "game_stats": game_stats,
        "forward_counters_delta": _counter_delta(counter_before, counter_after),
        "explain_timing_breakdown": timing,
        "elapsed_seconds": float(elapsed),
        "observation_digest": observation_artifact["digest"],
        "surrogate_digest": surrogate_artifact["digest"],
    }
    metadata["query_accounting"] = {
        "logical_attribution_queries": int(game_stats.get("row_count", 0)),
        "logical_unique_attribution_queries": int(game_stats.get("unique_text_count", 0)),
        "unique_attribution_texts": int(game_stats.get("unique_text_count", 0)),
        "physical_values_scored": int(game_stats.get("unique_text_count", 0)),
        "total_predict_rows_including_target_setup": int(
            metadata["forward_counters_delta"].get("predict_calls", 0)
        ),
        "model_forward_calls": int(attribution_forward_delta.get("model_forward_calls", 0)),
        "batch_calls": int(attribution_forward_delta.get("batch_calls", 0)),
        "batch_rows": int(attribution_forward_delta.get("batch_rows", 0)),
        "attribution_forward_counters_delta": attribution_forward_delta,
        "elapsed_seconds": float(elapsed),
    }

    predicted_label = int(np.argmax(full_probs))
    return AttributionResult(
        sample_id=str(sample.sample_id),
        gold_label=int(sample.label),
        predicted_label=predicted_label,
        target_label=int(target_label),
        text=sample.text,
        chunks=list(units),
        node_scores=list(chunk_scores),
        ranking=list(chunk_ranking),
        selected_ids=list(selected),
        attribution_cost={
            "attribution_budget_used": int(game_stats.get("row_count", 0)),
            "logical_unique_queries": int(game_stats.get("unique_text_count", 0)),
            "physical_values_scored": int(game_stats.get("unique_text_count", 0)),
            "interaction_verification_queries": 0,
            "model_forward_calls": int(
                attribution_forward_delta.get("model_forward_calls", 0)
            ),
            "batch_calls": int(attribution_forward_delta.get("batch_calls", 0)),
            "batch_rows": int(attribution_forward_delta.get("batch_rows", 0)),
            "model_counter_delta": attribution_forward_delta,
            "elapsed_seconds": float(elapsed),
        },
        method_summary={
            **metadata,
            "selected_text": selected_text,
            "selected_score": float(
                sum(chunk_scores_by_id.get(int(chunk_id), 0.0) for chunk_id in selected)
            ),
            "attribution_value": float(
                attribution_values_from_label_scores(
                    full_label_scores.reshape(1, -1),
                    target_class=target_label,
                    value_function=value_function,
                )[0]
            ),
        },
        diagnostics={
            "interaction_items": [
                {
                    "players": [int(value) for value in interaction],
                    "chunk_ids": [
                        int(player_to_chunk_id[int(value)])
                        for value in interaction
                        if 0 <= int(value) < len(player_to_chunk_id)
                    ],
                    "value": float(value),
                }
                for interaction, value in interaction_items
            ],
            "game_stats": game_stats,
        },
        observation_artifact=observation_artifact,
        surrogate_artifact=surrogate_artifact,
    )


def _validate_args(args) -> None:
    """Validate ProxySPEX-specific runtime and algorithm options."""

    normalize_eval_granularity(args.eval_granularity)
    normalize_chunker(getattr(args, "chunker", "word"))
    value_function = normalize_attribution_value_function(
        getattr(args, "value_function", "predicted_probability")
    )
    if value_function == "raw_target_score":
        raise ValueError(
            "ProxySPEX supports target_probability, predicted_probability, and "
            "predicted_class_margin."
        )
    if int(args.budget) < 2:
        raise ValueError("--budget must be at least 2 because ProxySPEX evaluates empty and grand coalitions")
    if int(args.max_order) <= 0:
        raise ValueError("--max-order must be positive")
    if int(args.k) < 0:
        raise ValueError("--k must be non-negative")
    if int(args.interaction_metadata_limit) < 0:
        raise ValueError("--interaction-metadata-limit must be non-negative")
    if int(args.proxy_n_jobs) == 0:
        raise ValueError("--proxy-n-jobs must not be 0")
    _sampling_weights(1, str(args.sampling_weight_mode))


def run(args, raw_argv: Sequence[str]) -> None:
    """Run native ProxySPEX with shared schema-v2 data and evaluation APIs."""

    ProxySPEX = _require_runtime_dependencies()
    _validate_args(args)
    args.adaptive_overrides = load_adaptive_overrides(getattr(args, "adaptive_overrides_json", None))
    set_seed(int(args.seed), bool(args.deterministic))
    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    output_root = _output_root(args, bundle)
    config = _result_config(args, bundle)
    store = ResultStore(
        output_root,
        config,
        output_level=str(args.output_level),
        command=" ".join([sys.executable, __file__, *raw_argv]),
        required_artifacts=("observation", "surrogate"),
    )
    backbone = HFBackbone(
        model_path=args.model_path,
        verbalizers=bundle.verbalizers,
        device=args.device,
        max_length=args.max_length,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        batch_size=16,
        dataset_name=bundle.dataset_name,
        prompt_config=dict(config.get("prompt", {})),
    )

    pending_samples = [
        sample for sample in bundle.samples if not store.sample_complete(sample.sample_id)
    ]
    print(
        f"[resume] run={output_root.name} selected={len(bundle.samples)} "
        f"completed={len(bundle.samples) - len(pending_samples)} pending={len(pending_samples)}"
    )
    if pending_samples:
        start = time.time()
        for sample in tqdm(pending_samples, desc="proxyspex", dynamic_ncols=True):
            try:
                result = _explain_sample(
                    sample=sample,
                    bundle=bundle,
                    backbone=backbone,
                    args=args,
                    ProxySPEX=ProxySPEX,
                )
                store.write_sample(result)
            except Exception as error:
                store.record_failure(sample.sample_id, error)
                store.write_status("running", selected_count=len(bundle.samples))
                raise
            store.write_status("running", selected_count=len(bundle.samples))
        print(f"[done] method={METHOD_NAME} processed={len(pending_samples)} elapsed={time.time() - start:.2f}s")
    else:
        print(f"[resume] method={METHOD_NAME} no pending samples")

    q_values = parse_q_values(args.eval_q_values)
    eval_report = evaluate_run(
        output_root,
        bundle=bundle,
        scorer=backbone,
        target=_effective_target_mode(args.value_function, args.target_mode),
        eval_granularity=args.eval_granularity,
        q_values=q_values,
    )
    status = store.finish(len(bundle.samples))
    print(
        f"[eval] method={METHOD_NAME} target={eval_report['target']} "
        f"report={output_root / 'metrics.json'} state={status['state']}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and launch one ProxySPEX run."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    run(args, raw_argv=raw_argv)


if __name__ == "__main__":
    main()
