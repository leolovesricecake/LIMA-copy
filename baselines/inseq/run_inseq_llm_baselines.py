from __future__ import annotations

"""
Run Inseq LLM attribution baselines inside LIMA-copy's LLM-transfer pipeline.

Intended location in the repository:
    baselines/inseq/run_inseq_llm_baselines.py

Supported methods:
    saliency, input_x_gradient, integrated_gradients,
    sequential_integrated_gradients, occlusion, reagent, lime

The runner saves one ExplanationResult JSON per sample, rebuilds summary.csv,
runs the existing LIMA evaluation pipeline, and stores run_config.json,
eval_config.json, eval_report.json, plus an aggregate_metrics.csv/jsonl file.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except Exception:  # pragma: no cover - fail-fast happens in _require_runtime_dependencies
    torch = None

from lima_llm.backbone.hf_backbone import HFBackbone
from lima_llm.chunking.utils import compose_text_from_chunk_ids
from lima_llm.data import load_dataset_bundle
from lima_llm.eval.evaluate import evaluate_saved_explanations

# Some historical versions of this branch have build_eval_units in lima_llm.eval.units,
# while the current public tree exposes the same logic from evaluate.py. Support both.
try:  # pragma: no cover - repository-version compatibility shim
    from lima_llm.eval.units import build_eval_units  # type: ignore
except Exception:  # pragma: no cover
    from lima_llm.eval.evaluate import _build_eval_units as build_eval_units  # type: ignore

from lima_llm.pipeline.io import rebuild_summary_csv, save_explanation
from lima_llm.pipeline.resume import is_sample_completed, sample_output_paths
from lima_llm.types import ExplanationResult, ScoreComponents, ScoreTrace, TextChunk
from lima_llm.utils import (
    atomic_write_json,
    build_provenance,
    configure_determinism,
    ensure_dir,
    parse_q_values,
    set_seed,
)

PROMPT_PREFIX = "Text:\n"
PROMPT_SUFFIX = "\nLabel:"

DEFAULT_METHODS = (
    "saliency",
    "input_x_gradient",
    "integrated_gradients",
    "sequential_integrated_gradients",
    "occlusion",
    "reagent",
)

INSEQ_NATIVE_METHODS = (
    *DEFAULT_METHODS,
    "lime",
)

SUPPORTED_METHODS = INSEQ_NATIVE_METHODS

DATASET_ALIASES = {
    "sst2": "sst2",
    "eraser": "eraser_movie_reviews",
    "eraser_movie_reviews": "eraser_movie_reviews",
    "emotion": "emotion",
    "imdb": "imdb",
    "rtn": "rotten_tomatoes",
    "rt": "rotten_tomatoes",
    "rotten_tomatoes": "rotten_tomatoes",
}

DEFAULT_DATASETS = (
    "eraser_movie_reviews",
    "emotion",
    "imdb",
    "rotten_tomatoes",
    "sst2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inseq LLM attribution baselines for LIMA-copy LLM transfer experiments"
    )

    # Dataset selection. --dataset is kept as a single-dataset alias for convenience.
    parser.add_argument(
        "--datasets",
        type=str,
        default="eraser,emotion,imdb,rtn,sst2",
        help="Comma-separated datasets or 'all'. Aliases: eraser->eraser_movie_reviews, rtn->rotten_tomatoes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional single-dataset shortcut; overrides --datasets when provided.",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    # Model/runtime.
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Passed through HFBackbone only if your branch supports it; Inseq defaults to eager internally.",
    )

    # Attribution methods/settings.
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="gold", choices=["gold", "predicted"])
    parser.add_argument(
        "--attributed-fn",
        type=str,
        default="probability",
        choices=["probability", "logits", "crossentropy"],
        help="Inseq step function used as attribution target. probability with logprob=True is the default.",
    )
    parser.add_argument(
        "--no-logprob",
        action="store_true",
        help="When --attributed-fn probability, use probability instead of log probability.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="abs",
        choices=["abs", "positive", "signed"],
        help="How to convert signed token-target attributions to ranking scores.",
    )
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--internal-batch-size", type=int, default=8)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Perturbation sample count for Inseq LIME.",
    )
    parser.add_argument(
        "--attr-pos-start",
        type=int,
        default=None,
        help="Optional Inseq attr_pos_start. Usually leave unset for label-only forced targets.",
    )
    parser.add_argument(
        "--attr-pos-end",
        type=int,
        default=None,
        help="Optional Inseq attr_pos_end. Usually leave unset for label-only forced targets.",
    )
    parser.add_argument(
        "--save-inseq-json",
        action="store_true",
        help="Also save raw Inseq FeatureAttributionOutput per sample. This can be very large.",
    )

    # ReAGent-specific knobs. Kept conservative for long-text experiments.
    parser.add_argument("--reagent-keep-top-n", type=int, default=5)
    parser.add_argument("--reagent-stopping-condition-top-k", type=int, default=3)
    parser.add_argument("--reagent-replacing-ratio", type=float, default=0.3)
    parser.add_argument("--reagent-max-probe-steps", type=int, default=3000)
    parser.add_argument("--reagent-num-probes", type=int, default=8)

    # Evaluation/output.
    parser.add_argument("--eval-q-values", type=str, default="1,5,10,20,50")
    parser.add_argument("--eval-granularity", type=str, default="token", choices=["token", "word"])
    parser.add_argument("--base-save-dir", type=str, default="results")
    parser.add_argument("--save-dir", type=str, default="baselines/inseq")
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])
    parser.add_argument("--deterministic", action="store_true")

    return parser


def _require_runtime_dependencies() -> None:
    if torch is None:
        raise RuntimeError("run_inseq_llm_baselines.py requires torch, transformers and inseq.")
    try:
        import inseq  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Inseq baselines require inseq and transformers. Install with: pip install inseq transformers"
        ) from exc


def _parse_csv(raw: str) -> Tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in raw.split(",") if str(item).strip())


def _parse_methods(raw: str) -> Tuple[str, ...]:
    methods = _parse_csv(raw)
    if not methods:
        raise ValueError("--methods must not be empty")
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported Inseq methods requested: {unknown}. Supported: {SUPPORTED_METHODS}")
    return methods


def _validate_args(args) -> None:
    if int(args.k) < 0:
        raise ValueError("--k must be non-negative")
    if int(args.n_steps) <= 0:
        raise ValueError("--n-steps must be positive")
    if int(args.internal_batch_size) <= 0:
        raise ValueError("--internal-batch-size must be positive")
    if int(args.n_samples) <= 0:
        raise ValueError("--n-samples must be positive")


def _parse_datasets(args) -> Tuple[str, ...]:
    raw = args.dataset if args.dataset else args.datasets
    items = _parse_csv(raw)
    if len(items) == 1 and items[0] == "all":
        return DEFAULT_DATASETS
    if not items:
        raise ValueError("--datasets must not be empty")
    normalized = []
    for item in items:
        if item not in DATASET_ALIASES:
            raise ValueError(f"Unsupported dataset alias: {item!r}. Supported aliases: {sorted(DATASET_ALIASES)}")
        normalized.append(DATASET_ALIASES[item])
    # Preserve order but deduplicate.
    seen = set()
    out = []
    for item in normalized:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return tuple(out)


def _model_slug(model_path: str) -> str:
    return str(model_path).rstrip("/").split("/")[-1].replace(".", "_").replace(":", "_")


def _collection_root(args) -> Path:
    return Path(args.base_save_dir) / str(args.save_dir)


def _method_output_root(args, dataset_name: str, method_name: str) -> Path:
    return (
        _collection_root(args)
        / str(dataset_name)
        / f"model-{_model_slug(args.model_path)}"
        / f"method-{method_name}_target-{args.target_mode}_k-{int(args.k)}_seed-{int(args.seed)}"
    )


def _label_target_text(label_text: str) -> str:
    return " " + str(label_text)


def _tokenize_text_only(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _encode_target_token_ids(tokenizer, label_text: str, max_length: int) -> List[int]:
    target_ids = _tokenize_text_only(tokenizer, _label_target_text(label_text))
    max_total = max(2, int(max_length))
    max_label = max_total - 1
    if len(target_ids) > max_label:
        target_ids = target_ids[-max_label:]
    if not target_ids:
        raise ValueError(f"Label text is not tokenizable for attribution target: {label_text!r}")
    return target_ids


def _truncate_text_for_prompt(
    *,
    tokenizer,
    text: str,
    chunks: Sequence[TextChunk],
    target_token_ids: Sequence[int],
    max_length: int,
) -> Dict[str, Any]:
    full_text_token_ids = _tokenize_text_only(tokenizer, text)
    prefix_token_ids = _tokenize_text_only(tokenizer, PROMPT_PREFIX)
    suffix_token_ids = _tokenize_text_only(tokenizer, PROMPT_SUFFIX)
    max_total = max(2, int(max_length))
    available_text = max(0, max_total - len(prefix_token_ids) - len(suffix_token_ids) - len(target_token_ids))
    kept_text_token_count = min(len(full_text_token_ids), available_text)
    dropped_left = max(0, len(full_text_token_ids) - kept_text_token_count)

    if dropped_left <= 0:
        kept_start_char = 0
    elif dropped_left < len(chunks):
        kept_start_char = int(chunks[dropped_left].start_char)
    else:
        kept_start_char = len(text)
    kept_text = text[kept_start_char:]

    return {
        "kept_text": kept_text,
        "kept_start_char": int(kept_start_char),
        "dropped_left_token_count": int(dropped_left),
        "kept_text_token_count": int(kept_text_token_count),
        "full_text_token_count": int(len(full_text_token_ids)),
        "available_text_token_budget": int(available_text),
        "prefix_token_count": int(len(prefix_token_ids)),
        "suffix_token_count": int(len(suffix_token_ids)),
        "target_token_count": int(len(target_token_ids)),
    }


def _rank_desc_scores(scores: Sequence[float], k: int) -> tuple[List[int], List[int]]:
    ranking = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))]
    return ranking, list(ranking[: max(0, int(k))])


def _build_rank_trace(selected: Sequence[int], chunk_scores: Mapping[int, float]) -> List[ScoreTrace]:
    zero = ScoreComponents(0.0, 0.0, 0.0, 0.0)
    total = 0.0
    trace: List[ScoreTrace] = []
    for step, chunk_id in enumerate(selected):
        gain = float(chunk_scores.get(int(chunk_id), 0.0))
        total += gain
        trace.append(
            ScoreTrace(
                step=int(step),
                selected_chunk_id=int(chunk_id),
                marginal_gain=gain,
                total_score=float(total),
                components=zero,
            )
        )
    return trace


def _sample_target_label(sample, bundle, backbone: HFBackbone, target_mode: str) -> tuple[int, np.ndarray]:
    probs = np.asarray(backbone.predict_label_probs(sample.text, bundle.verbalizers), dtype=np.float32)
    if target_mode == "predicted":
        return int(np.argmax(probs)), probs
    return int(sample.label), probs


def _chunks_for_text(tokenizer, text: str) -> tuple[List[TextChunk], str]:
    chunks, fallback_used, segmentation_strategy = build_eval_units(
        text=text,
        eval_granularity="token",
        tokenizer=tokenizer,
    )
    if fallback_used:
        raise RuntimeError(
            "Inseq baseline runner requires tokenizer offset_mapping support to preserve token/span alignment. "
            "Use a fast tokenizer compatible with return_offsets_mapping."
        )
    return list(chunks), str(segmentation_strategy)


def _torch_dtype_from_name(name: str):
    if torch is None:
        return None
    lowered = str(name or "").lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(lowered, None)


def _load_backbone(args) -> HFBackbone:
    # Keep the call aligned with the existing Captum runner. Some branch revisions may not
    # accept attn_implementation in HFBackbone, so we do not pass it here.
    return HFBackbone(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        embedding_layer_ratio=0.7,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )


def _reagent_load_kwargs(args) -> Dict[str, Any]:
    return {
        "keep_top_n": int(args.reagent_keep_top_n),
        "stopping_condition_top_k": int(args.reagent_stopping_condition_top_k),
        "replacing_ratio": float(args.reagent_replacing_ratio),
        "max_probe_steps": int(args.reagent_max_probe_steps),
        "num_probes": int(args.reagent_num_probes),
    }


def _build_inseq_model(backbone: HFBackbone, method_name: str, args):
    import inseq

    # Reuse the already-loaded HF model/tokenizer to avoid loading a second copy of the LLM.
    # Inseq accepts PreTrainedModel and PreTrainedTokenizerBase objects.
    method_kwargs = _reagent_load_kwargs(args) if method_name == "reagent" else {}
    model = inseq.load_model(
        backbone.model,
        method_name,
        tokenizer=backbone.tokenizer,
        device=args.device,
        **method_kwargs,
    )
    try:
        model.model.config.use_cache = False
    except Exception:
        pass
    return model


def _method_attr_kwargs(args, method_name: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "attributed_fn": str(args.attributed_fn),
        "step_scores": [str(args.attributed_fn)],
        "output_step_attributions": False,
    }
    if str(args.attributed_fn) == "probability":
        kwargs["attributed_fn_args"] = {"logprob": not bool(args.no_logprob)}
        kwargs["step_scores_args"] = {"probability": {"logprob": not bool(args.no_logprob)}}
    if method_name in {"integrated_gradients", "sequential_integrated_gradients"}:
        kwargs["n_steps"] = int(args.n_steps)
        kwargs["internal_batch_size"] = int(args.internal_batch_size)
    if method_name == "lime":
        kwargs["n_samples"] = int(args.n_samples)
    if args.attr_pos_start is not None:
        kwargs["attr_pos_start"] = int(args.attr_pos_start)
    if args.attr_pos_end is not None:
        kwargs["attr_pos_end"] = int(args.attr_pos_end)
    return kwargs


def _run_inseq_attribute(
    *,
    inseq_model,
    method_name: str,
    prompt_text: str,
    target_text: str,
    args,
):
    generated_text = target_text if getattr(inseq_model, "is_encoder_decoder", False) else prompt_text + target_text
    return inseq_model.attribute(
        input_texts=prompt_text,
        generated_texts=generated_text,
        method=method_name,
        **_method_attr_kwargs(args, method_name),
    )


def _sequence_output(inseq_output):
    if hasattr(inseq_output, "sequence_attributions"):
        return inseq_output.sequence_attributions[0]
    return inseq_output[0]


def _as_float_tensor(attr):
    if hasattr(attr, "detach"):
        return attr.detach().float().cpu()
    return torch.tensor(np.asarray(attr), dtype=torch.float32)


def _aggregate_inseq_attr_tensor(
    attr,
    *,
    target_token_count: int,
    score_mode: str,
) -> List[float]:
    attr = _as_float_tensor(attr)

    # Raw gradient methods often return [attributed_len, generated_len, hidden_dim].
    # Inseq's default visualization first reduces hidden dimensions by vector norm;
    # we reproduce that before aggregating over the generated label tokens.
    while attr.dim() > 2:
        attr = torch.linalg.vector_norm(attr, ord=2, dim=-1)
    if attr.dim() == 0:
        return [float(attr.item())]
    if attr.dim() == 1:
        vec = attr
    else:
        tgt_cols = max(1, min(int(target_token_count), int(attr.shape[1])))
        mat = attr[:, -tgt_cols:]
        if score_mode == "abs":
            vec = mat.abs().sum(dim=1)
        elif score_mode == "positive":
            vec = mat.clamp_min(0).sum(dim=1)
        elif score_mode == "signed":
            vec = mat.sum(dim=1)
        else:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
    return [float(x) for x in vec.reshape(-1).tolist()]


def _scores_from_inseq_attributions(
    *,
    inseq_output,
    target_token_count: int,
    score_mode: str,
) -> Tuple[List[float], str]:
    """Return token scores and the attribution side used.

    Encoder-decoder models usually populate ``source_attributions``. Decoder-only
    LMs have no separate source sequence in Inseq, so input/prefix scores are
    stored in ``target_attributions`` instead. Supporting both is necessary for
    LLaMA/Qwen/GPT-style models.
    """
    seq = _sequence_output(inseq_output)

    source_attr = getattr(seq, "source_attributions", None)
    if source_attr is not None:
        return (
            _aggregate_inseq_attr_tensor(
                source_attr,
                target_token_count=target_token_count,
                score_mode=score_mode,
            ),
            "source",
        )

    target_attr = getattr(seq, "target_attributions", None)
    if target_attr is not None:
        return (
            _aggregate_inseq_attr_tensor(
                target_attr,
                target_token_count=target_token_count,
                score_mode=score_mode,
            ),
            "target",
        )

    details = {
        "seq_type": type(seq).__name__,
        "has_source": hasattr(seq, "source"),
        "has_target": hasattr(seq, "target"),
        "source_len": len(getattr(seq, "source", []) or []),
        "target_len": len(getattr(seq, "target", []) or []),
        "has_sequence_scores": getattr(seq, "sequence_scores", None) is not None,
        "has_step_scores": getattr(seq, "step_scores", None) is not None,
    }
    raise RuntimeError(
        "Inseq output contains neither source_attributions nor target_attributions. "
        f"Output summary: {details}"
    )


def _offset_mapping_candidates(tokenizer, text: str) -> List[List[Tuple[int, int]]]:
    candidates: List[List[Tuple[int, int]]] = []
    for add_special in (False, True):
        try:
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=add_special,
                truncation=False,
            )
            offsets = enc.get("offset_mapping", None)
            if offsets is not None:
                candidates.append([(int(s), int(e)) for s, e in offsets])
        except Exception:
            continue
    return candidates


def _choose_offsets_for_scores(tokenizer, prompt_text: str, score_len: int) -> List[Tuple[int, int]]:
    candidates = _offset_mapping_candidates(tokenizer, prompt_text)
    if not candidates:
        # Very conservative fallback: assign empty offsets, resulting in zero text scores.
        return [(0, 0)] * int(score_len)
    offsets = min(candidates, key=lambda xs: abs(len(xs) - int(score_len)))
    if len(offsets) < score_len:
        offsets = offsets + [(0, 0)] * (score_len - len(offsets))
    elif len(offsets) > score_len:
        offsets = offsets[:score_len]
    return offsets


def _project_prompt_scores_to_text_chunks(
    *,
    source_scores: Sequence[float],
    prompt_offsets: Sequence[Tuple[int, int]],
    prompt_text_start: int,
    prompt_text_end: int,
    kept_start_char: int,
    chunks: Sequence[TextChunk],
) -> List[float]:
    scores = np.zeros(len(chunks), dtype=np.float64)
    weights = np.zeros(len(chunks), dtype=np.float64)

    # Only chunks that remain after left truncation can receive non-zero scores.
    for src_score, (start, end) in zip(source_scores, prompt_offsets):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        overlap_start = max(start, int(prompt_text_start))
        overlap_end = min(end, int(prompt_text_end))
        if overlap_end <= overlap_start:
            continue
        sample_start = int(kept_start_char) + (overlap_start - int(prompt_text_start))
        sample_end = int(kept_start_char) + (overlap_end - int(prompt_text_start))
        if sample_end <= sample_start:
            continue

        for chunk in chunks:
            cid = int(chunk.chunk_id)
            ov = max(0, min(sample_end, int(chunk.end_char)) - max(sample_start, int(chunk.start_char)))
            if ov <= 0:
                continue
            scores[cid] += float(src_score) * float(ov)
            weights[cid] += float(ov)

    out = []
    for idx in range(len(chunks)):
        if weights[idx] > 0:
            out.append(float(scores[idx] / weights[idx]))
        else:
            out.append(0.0)
    return out


def _summarize_inseq_output(inseq_output) -> Dict[str, Any]:
    seq = _sequence_output(inseq_output)
    payload: Dict[str, Any] = {}
    try:
        payload["source_tokens"] = [str(tok) for tok in getattr(seq, "source", [])]
    except Exception:
        payload["source_tokens"] = []
    try:
        payload["target_tokens"] = [str(tok) for tok in getattr(seq, "target", [])]
    except Exception:
        payload["target_tokens"] = []
    attr = getattr(seq, "source_attributions", None)
    if attr is not None and hasattr(attr, "shape"):
        payload["source_attributions_shape"] = [int(x) for x in list(attr.shape)]
    tgt_attr = getattr(seq, "target_attributions", None)
    if tgt_attr is not None and hasattr(tgt_attr, "shape"):
        payload["target_attributions_shape"] = [int(x) for x in list(tgt_attr.shape)]
    seq_scores = getattr(seq, "sequence_scores", None)
    if isinstance(seq_scores, dict):
        payload["sequence_score_keys"] = sorted(str(k) for k in seq_scores.keys())
    step_scores = getattr(seq, "step_scores", None)
    if isinstance(step_scores, dict):
        payload["step_score_keys"] = sorted(str(k) for k in step_scores.keys())
    payload["attr_pos_start"] = int(getattr(seq, "attr_pos_start", 0) or 0)
    payload["attr_pos_end"] = int(getattr(seq, "attr_pos_end", 0) or 0)
    try:
        payload["info"] = dict(getattr(inseq_output, "info", {}) or {})
    except Exception:
        payload["info"] = {}
    return payload


def _explain_sample(
    *,
    sample,
    bundle,
    backbone: HFBackbone,
    inseq_model,
    args,
    method_name: str,
    output_root: Path | None = None,
) -> ExplanationResult:
    tokenizer = backbone.tokenizer
    chunks, segmentation_strategy = _chunks_for_text(tokenizer=tokenizer, text=sample.text)

    target_label, full_probs = _sample_target_label(sample, bundle, backbone, args.target_mode)
    target_label_text = str(bundle.verbalizers[target_label])
    target_text = _label_target_text(target_label_text)
    target_token_ids = _encode_target_token_ids(tokenizer, target_label_text, max_length=int(args.max_length))

    truncation = _truncate_text_for_prompt(
        tokenizer=tokenizer,
        text=sample.text,
        chunks=chunks,
        target_token_ids=target_token_ids,
        max_length=int(args.max_length),
    )
    kept_text = str(truncation["kept_text"])
    kept_start_char = int(truncation["kept_start_char"])
    prompt_text = PROMPT_PREFIX + kept_text + PROMPT_SUFFIX

    inseq_output = _run_inseq_attribute(
        inseq_model=inseq_model,
        method_name=method_name,
        prompt_text=prompt_text,
        target_text=target_text,
        args=args,
    )

    source_scores, attribution_side = _scores_from_inseq_attributions(
        inseq_output=inseq_output,
        target_token_count=len(target_token_ids),
        score_mode=str(args.score_mode),
    )

    # Encoder-decoder outputs align scores to input_texts/source tokens. Decoder-only
    # outputs align scores to the single causal target stream, i.e. prompt + label.
    # We therefore choose offset mappings on the same string that the attribution
    # tensor is indexed over, then project only the review/text span back to chunks.
    score_text = prompt_text if attribution_side == "source" else prompt_text + target_text
    prompt_offsets = _choose_offsets_for_scores(tokenizer, score_text, score_len=len(source_scores))
    prompt_text_start = len(PROMPT_PREFIX)
    prompt_text_end = prompt_text_start + len(kept_text)
    full_scores = _project_prompt_scores_to_text_chunks(
        source_scores=source_scores,
        prompt_offsets=prompt_offsets,
        prompt_text_start=prompt_text_start,
        prompt_text_end=prompt_text_end,
        kept_start_char=kept_start_char,
        chunks=chunks,
    )

    if len(full_scores) != len(chunks):
        raise RuntimeError(f"Unexpected score length: got={len(full_scores)} expected={len(chunks)}")

    chunk_scores_by_id = {int(chunk.chunk_id): float(full_scores[int(chunk.chunk_id)]) for chunk in chunks}
    chunk_ranking, selected = _rank_desc_scores(full_scores, int(args.k))
    selected_text = compose_text_from_chunk_ids(chunks, selected)

    if bool(args.save_inseq_json) and output_root is not None:
        _maybe_save_raw_inseq(output_root, str(sample.sample_id), inseq_output)

    inseq_summary = _summarize_inseq_output(inseq_output)
    dropped_left_char_count = int(kept_start_char)
    metadata = {
        "inseq_method": method_name,
        "inseq_attributed_fn": str(args.attributed_fn),
        "inseq_probability_logprob": bool(not args.no_logprob) if str(args.attributed_fn) == "probability" else None,
        "score_mode": str(args.score_mode),
        "inseq_n_steps": int(args.n_steps),
        "inseq_internal_batch_size": int(args.internal_batch_size),
        "target_mode": str(args.target_mode),
        "target_label": int(target_label),
        "target_label_text": target_label_text,
        "target_text": target_text,
        "target_token_ids": [int(token_id) for token_id in target_token_ids],
        "full_label_probabilities": [float(x) for x in full_probs.tolist()],
        "truncation": {**truncation, "dropped_left_char_count": dropped_left_char_count},
        "prompt_template": "Text:\\n{text}\\nLabel:",
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
        "segmentation_strategy": segmentation_strategy,
        "raw_seq_attr": list(float(x) for x in full_scores),
        "source_score_count": int(len(source_scores)),
        "prompt_offset_count": int(len(prompt_offsets)),
        "attribution_side": attribution_side,
        "score_text_length": int(len(score_text)),
        "inseq_summary": inseq_summary,
        "reagent_settings": _reagent_load_kwargs(args) if method_name == "reagent" else None,
    }

    return ExplanationResult(
        explain_method=method_name,
        sample_id=sample.sample_id,
        dataset=bundle.dataset_name,
        split=bundle.split,
        label=sample.label,
        label_text=sample.label_text,
        text=sample.text,
        chunks=list(chunks),
        chunk_ranking=list(chunk_ranking),
        chunk_scores=list(full_scores),
        selected_chunk_ids=list(selected),
        selected_text=selected_text,
        scores={
            "total": float(sum(chunk_scores_by_id.get(int(chunk_id), 0.0) for chunk_id in selected)),
            "confidence": 0.0,
            "effectiveness": 0.0,
            "consistency": 0.0,
            "collaboration": 0.0,
            "target_probability": float(full_probs[target_label]),
            "label_probabilities": [float(x) for x in full_probs.tolist()],
        },
        trace=_build_rank_trace(selected=selected, chunk_scores=chunk_scores_by_id),
        metadata=metadata,
    )


def _scan_resume(samples, output_root: Path, resume_mode: str):
    pending = []
    completed = 0
    for sample in samples:
        paths = sample_output_paths(output_root, sample.sample_id)
        if is_sample_completed(paths, mode=resume_mode):
            completed += 1
        else:
            pending.append(sample)
    print(
        f"[resume] method-root={output_root.name} mode={resume_mode} "
        f"selected={len(samples)} completed={completed} pending={len(pending)}"
    )
    return pending


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))


def _method_cli_args(args, dataset_name: str, method_name: str) -> Dict[str, Any]:
    payload = dict(vars(args))
    payload["dataset"] = dataset_name
    payload["method"] = method_name
    return payload


def _write_configs(
    *,
    output_root: Path,
    args,
    dataset_name: str,
    method_name: str,
    raw_argv: Sequence[str],
    deterministic_info: Dict[str, Any],
) -> argparse.Namespace:
    method_args = argparse.Namespace(**_method_cli_args(args, dataset_name, method_name))
    now = time.time()
    run_config_payload = dict(vars(method_args))
    run_config_payload["provenance"] = build_provenance(
        stage="run_config",
        parsed_args=vars(method_args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=now,
        end_time=time.time(),
        cwd=Path.cwd(),
    )
    eval_config_payload = dict(vars(method_args))
    eval_config_payload["provenance"] = build_provenance(
        stage="eval_config",
        parsed_args=vars(method_args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=now,
        end_time=time.time(),
        cwd=Path.cwd(),
    )
    _write_json(output_root / "run_config.json", run_config_payload)
    _write_json(output_root / "eval_config.json", eval_config_payload)
    return method_args


def _maybe_save_raw_inseq(output_root: Path, sample_id: str, inseq_output) -> None:
    raw_dir = output_root / "inseq_raw"
    ensure_dir(raw_dir)
    try:
        inseq_output.save(str(raw_dir / f"{sample_id}.json"))
    except Exception as exc:
        print(f"[warn] failed to save raw Inseq output for sample={sample_id}: {type(exc).__name__}: {exc}")


def _run_method_dataset(
    *,
    args,
    raw_argv: Sequence[str],
    bundle,
    backbone: HFBackbone,
    method_name: str,
) -> Dict[str, Any]:
    dataset_name = bundle.dataset_name
    output_root = _method_output_root(args, dataset_name, method_name)
    ensure_dir(output_root)
    ensure_dir(output_root / "samples")

    deterministic_info = configure_determinism(bool(args.deterministic))
    method_args = _write_configs(
        output_root=output_root,
        args=args,
        dataset_name=dataset_name,
        method_name=method_name,
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
    )

    pending_samples = _scan_resume(bundle.samples, output_root=output_root, resume_mode=args.resume_check)
    inseq_model = None
    try:
        if pending_samples:
            inseq_model = _build_inseq_model(backbone, method_name, args)
            start = time.time()
            for sample in tqdm(pending_samples, desc=f"inseq-{dataset_name}-{method_name}", dynamic_ncols=True):
                result = _explain_sample(
                    sample=sample,
                    bundle=bundle,
                    backbone=backbone,
                    inseq_model=inseq_model,
                    args=args,
                    method_name=method_name,
                    output_root=output_root,
                )
                save_explanation(result, output_root)
            print(
                f"[done] dataset={dataset_name} method={method_name} "
                f"processed={len(pending_samples)} elapsed={time.time() - start:.2f}s"
            )
        else:
            print(f"[resume] dataset={dataset_name} method={method_name} no pending samples")
    finally:
        if inseq_model is not None:
            try:
                inseq_model.unhook()
            except Exception:
                pass
            del inseq_model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = rebuild_summary_csv(output_root)
    print(f"[done] dataset={dataset_name} method={method_name} summary={summary_path}")

    q_values = parse_q_values(args.eval_q_values)
    eval_started = time.time()
    eval_report = evaluate_saved_explanations(
        output_root=output_root,
        bundle=bundle,
        backbone=backbone,
        verbalizers=bundle.verbalizers,
        q_values=q_values,
        explain_method=method_name,
        eval_granularity=args.eval_granularity,
    )
    eval_report["provenance"] = build_provenance(
        stage="eval_report",
        parsed_args=vars(method_args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=eval_started,
        end_time=time.time(),
        cwd=Path.cwd(),
    )
    _write_json(output_root / "eval_report.json", eval_report)
    print(f"[eval] dataset={dataset_name} method={method_name} report={output_root / 'eval_report.json'}")
    return {
        "dataset": dataset_name,
        "method": method_name,
        "output_root": str(output_root),
        "eval_report": eval_report,
    }


def _flatten_metric_row(args, payload: Mapping[str, Any]) -> Dict[str, Any]:
    report = dict(payload["eval_report"])
    row: Dict[str, Any] = {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "model_path": args.model_path,
        "model_slug": _model_slug(args.model_path),
        "split": report.get("split"),
        "sample_count": report.get("sample_count"),
        "output_root": payload["output_root"],
    }
    for block_name in ("metrics_primary", "metrics_secondary"):
        block = report.get(block_name, {}) or {}
        for key, value in block.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[f"{block_name}.{key}"] = value
    # Convenient aliases for paper tables.
    primary = report.get("metrics_primary", {}) or {}
    row["LO"] = primary.get("log_odds")
    row["Sufficiency"] = primary.get("sufficiency")
    row["Comprehensiveness"] = primary.get("comprehensiveness")
    row["A-S"] = primary.get("aopc_sufficiency")
    row["A-C"] = primary.get("aopc_comprehensiveness")
    return row


def _write_aggregate_reports(args, rows: Sequence[Mapping[str, Any]]) -> None:
    root = _collection_root(args)
    ensure_dir(root)
    jsonl_path = root / "aggregate_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    csv_path = root / "aggregate_metrics.csv"
    if rows:
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
    print(f"[aggregate] wrote {jsonl_path}")
    print(f"[aggregate] wrote {csv_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    methods = _parse_methods(args.methods)
    _validate_args(args)
    _require_runtime_dependencies()
    datasets = _parse_datasets(args)
    set_seed(int(args.seed))

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    print(f"[config] datasets={datasets}")
    print(f"[config] methods={methods}")
    print(f"[config] model={args.model_path}")

    backbone = _load_backbone(args)

    aggregate_rows: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        bundle = load_dataset_bundle(
            dataset_name=dataset_name,
            split=args.split,
            max_samples=args.max_samples,
            eraser_root=args.eraser_root,
            sst2_source=args.sst2_source,
            dataset_cache_dir=args.dataset_cache_dir,
        )
        for method_name in methods:
            payload = _run_method_dataset(
                args=args,
                raw_argv=raw_argv,
                bundle=bundle,
                backbone=backbone,
                method_name=method_name,
            )
            aggregate_rows.append(_flatten_metric_row(args, payload))
            _write_aggregate_reports(args, aggregate_rows)

    _write_aggregate_reports(args, aggregate_rows)


if __name__ == "__main__":
    main()
