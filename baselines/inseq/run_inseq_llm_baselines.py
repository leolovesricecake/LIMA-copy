from __future__ import annotations

"""
Run Inseq LLM attribution baselines with the shared Mobius experiment protocol.

Intended location in the repository:
    baselines/inseq/run_inseq_llm_baselines.py

Supported methods:
    saliency, input_x_gradient, integrated_gradients,
    sequential_integrated_gradients, occlusion, reagent, lime

The runner preserves Inseq's native attribution calls and writes schema-v2
sample, status, metrics, curve, and aggregate collection artifacts.
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

from mobius.core.results import ResultStore, default_run_dir
from mobius.core.runtime import ensure_dir, set_seed
from mobius.core.schema import AttributionResult, TextChunk
from mobius.data import load_dataset_bundle
from mobius.evaluation.evaluator import evaluate_run
from mobius.models.hf import HFBackbone
from mobius.text.chunks import build_eval_units, compose_text

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
_INSEQ_LIME_BFLOAT16_PATCHED = False

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


def parse_q_values(raw: str) -> List[int]:
    """Parse comma-separated evaluation percentages."""

    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Handle the build parser step in this retained baseline."""

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
    parser.add_argument(
        "--output-level",
        choices=["minimal", "standard", "debug"],
        default="standard",
    )
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])
    parser.add_argument("--deterministic", action="store_true")

    return parser


def _require_runtime_dependencies() -> None:
    """Handle the require runtime dependencies step in this retained baseline."""

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
    """Handle the parse csv step in this retained baseline."""

    return tuple(str(item).strip().lower() for item in raw.split(",") if str(item).strip())


def _parse_methods(raw: str) -> Tuple[str, ...]:
    """Handle the parse methods step in this retained baseline."""

    methods = _parse_csv(raw)
    if not methods:
        raise ValueError("--methods must not be empty")
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported Inseq methods requested: {unknown}. Supported: {SUPPORTED_METHODS}")
    return methods


def _validate_args(args) -> None:
    """Handle the validate args step in this retained baseline."""

    if int(args.k) < 0:
        raise ValueError("--k must be non-negative")
    if int(args.n_steps) <= 0:
        raise ValueError("--n-steps must be positive")
    if int(args.internal_batch_size) <= 0:
        raise ValueError("--internal-batch-size must be positive")
    if int(args.n_samples) <= 0:
        raise ValueError("--n-samples must be positive")


def _parse_datasets(args) -> Tuple[str, ...]:
    """Handle the parse datasets step in this retained baseline."""

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
    """Handle the model slug step in this retained baseline."""

    return str(model_path).rstrip("/").split("/")[-1].replace(".", "_").replace(":", "_")


def _collection_root(args) -> Path:
    """Return the root shared by all requested Inseq runs."""

    return Path(args.base_save_dir) / str(args.save_dir)


def _method_config(args, bundle, method_name: str) -> Dict[str, Any]:
    """Build one scientific configuration for an Inseq method run."""

    return {
        "method": f"inseq_{method_name}",
        "output_level": str(args.output_level),
        "dataset": {
            "name": bundle.dataset_name,
            "split": bundle.split,
            "max_samples": args.max_samples,
            "dataset_cache_dir": args.dataset_cache_dir,
            "verbalizers": list(bundle.verbalizers),
        },
        "model": {
            "type": "hf_causal_lm",
            "model_path": str(args.model_path),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "max_length": int(args.max_length),
        },
        "budget": int(args.n_steps if method_name != "lime" else args.n_samples),
        "seed": int(args.seed),
        "max_order": 1,
        "k": int(args.k),
        "target_mode": str(args.target_mode),
        "chunker": "token",
        "eval_granularity": str(args.eval_granularity),
        "eval_q_values": parse_q_values(args.eval_q_values),
        "inseq": {
            "method": method_name,
            "attributed_fn": str(args.attributed_fn),
            "score_mode": str(args.score_mode),
            "n_steps": int(args.n_steps),
            "internal_batch_size": int(args.internal_batch_size),
            "n_samples": int(args.n_samples),
        },
    }


def _method_output_root(args, bundle, method_name: str) -> Path:
    """Resolve the canonical schema-v2 path for one Inseq method."""

    return default_run_dir(
        _collection_root(args),
        _method_config(args, bundle, method_name),
    )


def _label_target_text(label_text: str) -> str:
    """Handle the label target text step in this retained baseline."""

    return " " + str(label_text)


def _tokenize_text_only(tokenizer, text: str) -> List[int]:
    """Handle the tokenize text only step in this retained baseline."""

    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _encode_target_token_ids(tokenizer, label_text: str, max_length: int) -> List[int]:
    """Handle the encode target token ids step in this retained baseline."""

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
    """Handle the truncate text for prompt step in this retained baseline."""

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
    """Handle the rank desc scores step in this retained baseline."""

    ranking = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))]
    return ranking, list(ranking[: max(0, int(k))])


def _sample_target_label(sample, bundle, backbone: HFBackbone, target_mode: str) -> tuple[int, np.ndarray]:
    """Choose the gold or full-input predicted attribution target."""

    probs = np.asarray(backbone.predict_label_probs(sample.text, bundle.verbalizers), dtype=np.float32)
    if target_mode == "predicted":
        return int(np.argmax(probs)), probs
    return int(sample.label), probs


def _chunks_for_text(tokenizer, text: str) -> tuple[List[TextChunk], str]:
    """Build tokenizer-aligned player chunks for Inseq attribution."""

    result = build_eval_units(text, "token", tokenizer)
    if result.fallback_used:
        raise RuntimeError(
            "Inseq baseline runner requires tokenizer offset_mapping support to preserve token/span alignment. "
            "Use a fast tokenizer compatible with return_offsets_mapping."
        )
    return list(result.chunks), str(result.strategy)


def _torch_dtype_from_name(name: str):
    """Handle the torch dtype from name step in this retained baseline."""

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
    """Load the shared Hugging Face model wrapper used by Inseq."""

    # Keep the call aligned with the existing Captum runner. Some branch revisions may not
    # accept attn_implementation in HFBackbone, so we do not pass it here.
    return HFBackbone(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )


def _reagent_load_kwargs(args) -> Dict[str, Any]:
    """Handle the reagent load kwargs step in this retained baseline."""

    return {
        "keep_top_n": int(args.reagent_keep_top_n),
        "stopping_condition_top_k": int(args.reagent_stopping_condition_top_k),
        "replacing_ratio": float(args.reagent_replacing_ratio),
        "max_probe_steps": int(args.reagent_max_probe_steps),
        "num_probes": int(args.reagent_num_probes),
    }


def _tensor_to_numpy_list_for_inseq_lime(value):
    """Handle the tensor to numpy list for inseq lime step in this retained baseline."""

    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy().tolist()
    return value


def _patch_inseq_lime_bfloat16_numpy() -> None:
    """Patch Inseq 0.7.x LIME so bf16 tensors are cast before numpy conversion."""
    global _INSEQ_LIME_BFLOAT16_PATCHED
    if _INSEQ_LIME_BFLOAT16_PATCHED:
        return
    if torch is None:
        return

    from inseq.attr.feat.ops.lime import Lime

    current = getattr(Lime, "perturb_func", None)
    if getattr(current, "_lima_bfloat16_safe", False):
        _INSEQ_LIME_BFLOAT16_PATCHED = True
        return

    def perturb_func(
        self,
        original_input_tuple: tuple = (),
        mask_prob: float = 0.3,
        mask_token: str = "unk",
        **kwargs: Any,
    ) -> tuple:
        """Handle the perturb func step in this retained baseline."""

        perturbed_inputs = []
        for original_input_tensor in original_input_tuple:
            mask_value_probs = torch.tensor([mask_prob, 1 - mask_prob])
            mask_multinomial_binary = torch.multinomial(
                mask_value_probs, len(original_input_tensor[0]), replacement=True
            )

            mask_special_token_ids = torch.Tensor(
                [
                    1 if id_ in self.attribution_model.special_tokens_ids else 0
                    for id_ in _tensor_to_numpy_list_for_inseq_lime(original_input_tensor[0])
                ]
            ).int()

            mask = (
                torch.tensor(
                    [
                        m + s if s == 0 else s
                        for m, s in zip(mask_multinomial_binary, mask_special_token_ids, strict=False)
                    ]
                )
                .to(self.attribution_model.device)
                .unsqueeze(-1)
            )

            if mask_token == "unk":
                tokenizer_mask_token = self.attribution_model.tokenizer.unk_token_id
            elif mask_token == "pad":
                tokenizer_mask_token = self.attribution_model.tokenizer.pad_token_id
            else:
                raise ValueError(f"Invalid mask token {mask_token} for tokenizer: {self.attribution_model.tokenizer}")
            if tokenizer_mask_token is None:
                tokenizer_mask_token = self.attribution_model.tokenizer.eos_token_id
            if tokenizer_mask_token is None:
                raise ValueError(f"Tokenizer has no {mask_token!r}, pad, or eos token id for LIME masking.")

            perturbed_inputs.append(original_input_tensor * mask + (1 - mask) * int(tokenizer_mask_token))

        return tuple(perturbed_inputs)

    perturb_func._lima_bfloat16_safe = True  # type: ignore[attr-defined]
    Lime.perturb_func = perturb_func
    _INSEQ_LIME_BFLOAT16_PATCHED = True


def _build_inseq_model(backbone: HFBackbone, method_name: str, args):
    """Handle the build inseq model step in this retained baseline."""

    import inseq

    # Reuse the already-loaded HF model/tokenizer to avoid loading a second copy of the LLM.
    # Inseq accepts PreTrainedModel and PreTrainedTokenizerBase objects.
    if method_name == "lime":
        _patch_inseq_lime_bfloat16_numpy()
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
    """Handle the method attr kwargs step in this retained baseline."""

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
    """Handle the run inseq attribute step in this retained baseline."""

    generated_text = target_text if getattr(inseq_model, "is_encoder_decoder", False) else prompt_text + target_text
    return inseq_model.attribute(
        input_texts=prompt_text,
        generated_texts=generated_text,
        method=method_name,
        **_method_attr_kwargs(args, method_name),
    )


def _sequence_output(inseq_output):
    """Handle the sequence output step in this retained baseline."""

    if hasattr(inseq_output, "sequence_attributions"):
        return inseq_output.sequence_attributions[0]
    return inseq_output[0]


def _as_float_tensor(attr):
    """Handle the as float tensor step in this retained baseline."""

    if hasattr(attr, "detach"):
        return attr.detach().float().cpu()
    return torch.tensor(np.asarray(attr), dtype=torch.float32)


def _aggregate_inseq_attr_tensor(
    attr,
    *,
    target_token_count: int,
    score_mode: str,
) -> List[float]:
    """Handle the aggregate inseq attr tensor step in this retained baseline."""

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
    """Handle the offset mapping candidates step in this retained baseline."""

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
    """Handle the choose offsets for scores step in this retained baseline."""

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
    """Handle the project prompt scores to text chunks step in this retained baseline."""

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
    """Handle the summarize inseq output step in this retained baseline."""

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
) -> AttributionResult:
    """Run one native Inseq explanation and adapt it to schema v2."""

    started = time.perf_counter()
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

    forward_calls = 0

    def count_forward(_module, _inputs, _output) -> None:
        """Count direct model forwards made internally by Inseq."""

        nonlocal forward_calls
        forward_calls += 1

    hook = backbone.model.register_forward_hook(count_forward)
    try:
        inseq_output = _run_inseq_attribute(
            inseq_model=inseq_model,
            method_name=method_name,
            prompt_text=prompt_text,
            target_text=target_text,
            args=args,
        )
    finally:
        hook.remove()

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

    chunk_scores_by_id = {
        int(chunk.chunk_id): float(full_scores[int(chunk.chunk_id)])
        for chunk in chunks
    }
    active_ids = [
        int(chunk.chunk_id)
        for chunk in chunks
        if int(chunk.end_char) > kept_start_char
    ]
    active_set = set(active_ids)
    active_ranking = sorted(
        active_ids,
        key=lambda chunk_id: (-chunk_scores_by_id[chunk_id], chunk_id),
    )
    inactive_ids = [
        int(chunk.chunk_id)
        for chunk in chunks
        if int(chunk.chunk_id) not in active_set
    ]
    chunk_ranking = active_ranking + inactive_ids
    selected = chunk_ranking[: min(int(args.k), len(active_ids))]
    selected_text = compose_text(chunks, selected)

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
        "active_chunk_ids": active_ids,
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

    predicted_label = int(np.argmax(full_probs))
    return AttributionResult(
        sample_id=str(sample.sample_id),
        gold_label=int(sample.label),
        predicted_label=predicted_label,
        target_label=int(target_label),
        text=sample.text,
        chunks=list(chunks),
        node_scores=list(full_scores),
        ranking=list(chunk_ranking),
        selected_ids=list(selected),
        attribution_cost={
            "attribution_budget_used": int(forward_calls),
            "logical_unique_queries": int(forward_calls),
            "physical_values_scored": int(forward_calls),
            "interaction_verification_queries": 0,
            "model_forward_calls": int(forward_calls),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        method_summary={
            **metadata,
            "selected_text": selected_text,
            "selected_score": float(
                sum(chunk_scores_by_id.get(int(chunk_id), 0.0) for chunk_id in selected)
            ),
        },
        diagnostics={
            "raw_sequence_attributions": [float(value) for value in full_scores],
            "inseq_summary": inseq_summary,
        },
    )


def _maybe_save_raw_inseq(output_root: Path, sample_id: str, inseq_output) -> None:
    """Optionally persist Inseq's native detailed attribution object."""

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
    """Run one Inseq method on one dataset with schema-v2 persistence."""

    dataset_name = bundle.dataset_name
    output_root = _method_output_root(args, bundle, method_name)
    config = _method_config(args, bundle, method_name)
    store = ResultStore(
        output_root,
        config,
        output_level=str(args.output_level),
        command=" ".join([sys.executable, __file__, *raw_argv]),
    )
    pending_samples = [
        sample for sample in bundle.samples if not store.sample_complete(sample.sample_id)
    ]
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
                store.write_sample(result)
                store.write_status("running", selected_count=len(bundle.samples))
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

    q_values = parse_q_values(args.eval_q_values)
    backbone.verbalizers = list(bundle.verbalizers)
    eval_report = evaluate_run(
        output_root,
        bundle,
        backbone,
        target=str(args.target_mode),
        eval_granularity=args.eval_granularity,
        q_values=q_values,
    )
    store.finish(len(bundle.samples))
    print(f"[eval] dataset={dataset_name} method={method_name} report={output_root / 'metrics.json'}")
    return {
        "dataset": dataset_name,
        "method": method_name,
        "output_root": str(output_root),
        "eval_report": eval_report,
    }


def _flatten_metric_row(args, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten schema-v2 aggregate means for the Inseq collection table."""

    report = dict(payload["eval_report"])
    row: Dict[str, Any] = {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "model_path": args.model_path,
        "model_slug": _model_slug(args.model_path),
        "split": args.split,
        "sample_count": report.get("evaluated_count"),
        "target": report.get("target"),
        "output_root": payload["output_root"],
    }
    faithfulness = report.get("faithfulness", {}) or {}
    for key, summary in faithfulness.items():
        row[str(key)] = summary.get("mean")
    row["Sufficiency"] = row.get("sufficiency")
    row["Comprehensiveness"] = row.get("comprehensiveness")
    row["A-S"] = row.get("aopc_sufficiency")
    row["A-C"] = row.get("aopc_comprehensiveness")
    return row


def _write_aggregate_reports(args, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write collection-level CSV and JSONL convenience tables."""

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
    """Run requested Inseq methods and datasets sequentially."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    methods = _parse_methods(args.methods)
    _validate_args(args)
    _require_runtime_dependencies()
    datasets = _parse_datasets(args)
    set_seed(int(args.seed), bool(args.deterministic))

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
