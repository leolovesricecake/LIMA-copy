from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from lima_llm.backbone.hf_backbone import HFBackbone
from lima_llm.chunking.utils import compose_text_from_chunk_ids
from lima_llm.data import load_dataset_bundle
from lima_llm.eval.evaluate import evaluate_saved_explanations
from lima_llm.eval.units import build_eval_units
from lima_llm.pipeline.io import rebuild_summary_csv, save_explanation
from lima_llm.pipeline.resume import is_sample_completed, sample_output_paths
from lima_llm.types import ExplanationResult, ScoreComponents, ScoreTrace
from lima_llm.utils import (
    atomic_write_json,
    build_provenance,
    configure_determinism,
    ensure_dir,
    parse_q_values,
    set_seed,
)

DEFAULT_METHODS = (
    "feature_ablation",
    "layer_integrated_gradients",
    "kernel_shap",
    "lime",
)
SUPPORTED_METHODS = (*DEFAULT_METHODS, "shapley_value_sampling")
UNSUPPORTED_METHODS = {"shapley_values"}
PROMPT_PREFIX = "Text:\n"
PROMPT_SUFFIX = "\nLabel:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Captum LLM baselines for HF causal language models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "eraser_movie_reviews", "imdb", "rotten_tomatoes", "emotion"],
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

    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="gold", choices=["gold", "predicted"])
    parser.add_argument("--eval-q-values", type=str, default="1,5,10,20,50")
    parser.add_argument("--eval-granularity", type=str, default="token", choices=["token", "word"])

    parser.add_argument("--attr-target", type=str, default="log_prob", choices=["log_prob", "prob"])
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--forward-in-tokens",
        type=int,
        default=1,
        help="Perturbation-only knob: 1 runs token-by-token target scoring, 0 runs sequence-level target scoring.",
    )

    parser.add_argument("--base-save-dir", type=str, default="results")
    parser.add_argument("--save-dir", type=str, default="baselines/captum")
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _require_runtime_dependencies() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "run_captum_llm_baselines.py requires torch. Please install torch, transformers, and captum first."
        )
    try:
        import captum  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Captum baselines require captum and transformers. Please install the dependencies listed in "
            "baselines/captum/README_zh.md."
        ) from exc


def _load_captum_symbols() -> Dict[str, Any]:
    _require_runtime_dependencies()
    from captum.attr import (
        FeatureAblation,
        KernelShap,
        LLMAttribution,
        LLMGradientAttribution,
        LayerIntegratedGradients,
        Lime,
        ShapleyValueSampling,
        TextTokenInput,
    )

    return {
        "FeatureAblation": FeatureAblation,
        "KernelShap": KernelShap,
        "LLMAttribution": LLMAttribution,
        "LLMGradientAttribution": LLMGradientAttribution,
        "LayerIntegratedGradients": LayerIntegratedGradients,
        "Lime": Lime,
        "ShapleyValueSampling": ShapleyValueSampling,
        "TextTokenInput": TextTokenInput,
    }


def _parse_methods(raw: str) -> Tuple[str, ...]:
    methods = tuple(str(item).strip().lower() for item in raw.split(",") if str(item).strip() != "")
    if not methods:
        raise ValueError("--methods must not be empty")
    unknown = [method for method in methods if method not in SUPPORTED_METHODS and method not in UNSUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported Captum methods requested: {unknown}. Supported: {sorted(SUPPORTED_METHODS)}")
    unsupported = [method for method in methods if method in UNSUPPORTED_METHODS]
    if unsupported:
        raise ValueError(
            f"Unsupported exact methods requested: {unsupported}. "
            "This runner intentionally excludes exact ShapleyValues due to long-text cost."
        )
    return methods


def _model_slug(model_path: str) -> str:
    return str(model_path).rstrip("/").split("/")[-1].replace(".", "_")


def _method_output_root(args, method_name: str) -> Path:
    return (
        Path(args.base_save_dir)
        / str(args.save_dir)
        / str(args.dataset)
        / f"model-{_model_slug(args.model_path)}"
        / f"method-{method_name}_target-{args.target_mode}_k-{int(args.k)}_seed-{int(args.seed)}"
    )


def _label_target_text(label_text: str) -> str:
    return " " + str(label_text)


def _rank_desc_scores(scores: Sequence[float], k: int) -> tuple[List[int], List[int]]:
    ranking = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))]
    return ranking, list(ranking[: max(0, int(k))])


def _default_baseline_token_id(tokenizer) -> int:
    for token_id in (
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if token_id is not None:
            return int(token_id)
    return 0


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


def _truncate_text_token_ids(
    *,
    text_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    max_length: int,
) -> Dict[str, Any]:
    max_total = max(2, int(max_length))
    available_prompt = max_total - len(target_token_ids)
    available_text = max(0, available_prompt - len(prefix_token_ids) - len(suffix_token_ids))

    kept_text_token_ids = list(int(token_id) for token_id in text_token_ids[-available_text:]) if available_text > 0 else []
    dropped_left = max(0, len(text_token_ids) - len(kept_text_token_ids))

    return {
        "kept_text_token_ids": kept_text_token_ids,
        "dropped_left_token_count": int(dropped_left),
        "kept_text_token_count": int(len(kept_text_token_ids)),
        "full_text_token_count": int(len(text_token_ids)),
        "available_text_token_budget": int(available_text),
        "prefix_token_count": int(len(prefix_token_ids)),
        "suffix_token_count": int(len(suffix_token_ids)),
        "target_token_count": int(len(target_token_ids)),
    }


def _compose_prompt_input_ids(
    *,
    text_token_ids: Sequence[int],
    continuation_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
) -> List[int]:
    return [
        *(int(token_id) for token_id in prefix_token_ids),
        *(int(token_id) for token_id in text_token_ids),
        *(int(token_id) for token_id in suffix_token_ids),
        *(int(token_id) for token_id in continuation_token_ids),
    ]


def _build_token_chunks(tokenizer, text: str):
    chunks, fallback_used, segmentation_strategy = build_eval_units(
        text=text,
        eval_granularity="token",
        tokenizer=tokenizer,
    )
    if fallback_used:
        raise RuntimeError(
            "Captum baseline runner requires tokenizer offset_mapping support to guarantee token/span alignment. "
            "Please use a fast tokenizer compatible with return_offsets_mapping."
        )
    return chunks, segmentation_strategy


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


def _summarize_token_attr(token_attr) -> Dict[str, Any] | None:
    if token_attr is None:
        return None
    if hasattr(token_attr, "detach"):
        token_attr = token_attr.detach().cpu().numpy()
    arr = np.asarray(token_attr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {
        "shape": [int(dim) for dim in arr.shape],
        "row_sums": [float(x) for x in arr.sum(axis=1).tolist()],
        "row_abs_sums": [float(x) for x in np.abs(arr).sum(axis=1).tolist()],
    }


def _build_captum_input_class():
    symbols = _load_captum_symbols()
    base_cls = symbols["TextTokenInput"]

    class PreTokenizedTextTokenInput(base_cls):
        def __init__(self, token_ids: Sequence[int], tokenizer, baselines: int):
            if torch is None:
                raise RuntimeError("torch is required for PreTokenizedTextTokenInput")
            inp_tensor = torch.tensor([list(int(token_id) for token_id in token_ids)], dtype=torch.long)
            self.inp_tensor = inp_tensor
            self.itp_tensor = inp_tensor
            self.itp_mask = None
            self.skip_tokens = None
            self.values = tokenizer.convert_ids_to_tokens(self.itp_tensor[0].tolist())
            self.tokenizer = tokenizer
            self.n_itp_features = len(self.values)
            self.baselines = int(baselines)

    return PreTokenizedTextTokenInput


class PromptedCausalLMAdapter(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        base_model,
        prefix_token_ids: Sequence[int],
        suffix_token_ids: Sequence[int],
        text_token_count: int,
        device,
    ) -> None:
        if nn is not None:
            super().__init__()
        self.base_model = base_model
        self.prefix_token_ids = [int(token_id) for token_id in prefix_token_ids]
        self.suffix_token_ids = [int(token_id) for token_id in suffix_token_ids]
        self.text_token_count = int(text_token_count)
        self.device = device

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _split_text_and_continuation(self, input_ids):
        text_ids = input_ids[:, : self.text_token_count]
        continuation_ids = input_ids[:, self.text_token_count :]
        return text_ids, continuation_ids

    def _compose_full_input_ids(self, input_ids):
        if torch is None:
            raise RuntimeError("torch is required for PromptedCausalLMAdapter")
        batch_size = int(input_ids.size(0))
        text_ids, continuation_ids = self._split_text_and_continuation(input_ids)
        pieces = []
        if self.prefix_token_ids:
            prefix = torch.tensor([self.prefix_token_ids], dtype=torch.long, device=input_ids.device)
            pieces.append(prefix.expand(batch_size, -1))
        pieces.append(text_ids)
        if self.suffix_token_ids:
            suffix = torch.tensor([self.suffix_token_ids], dtype=torch.long, device=input_ids.device)
            pieces.append(suffix.expand(batch_size, -1))
        if continuation_ids.numel() > 0:
            pieces.append(continuation_ids)
        return torch.cat(pieces, dim=1)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        full_input_ids = self._compose_full_input_ids(input_ids.to(self.device))
        full_attention_mask = torch.ones_like(full_input_ids)
        return self.base_model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=use_cache,
            **kwargs,
        )


def _build_attr_runner(method_name: str, adapter, tokenizer, attr_target: str):
    symbols = _load_captum_symbols()
    if method_name == "feature_ablation":
        return symbols["LLMAttribution"](symbols["FeatureAblation"](adapter), tokenizer, attr_target=attr_target)
    if method_name == "kernel_shap":
        return symbols["LLMAttribution"](symbols["KernelShap"](adapter), tokenizer, attr_target=attr_target)
    if method_name == "lime":
        return symbols["LLMAttribution"](symbols["Lime"](adapter), tokenizer, attr_target=attr_target)
    if method_name == "shapley_value_sampling":
        return symbols["LLMAttribution"](
            symbols["ShapleyValueSampling"](adapter),
            tokenizer,
            attr_target=attr_target,
        )
    if method_name == "layer_integrated_gradients":
        if attr_target != "log_prob":
            raise ValueError("layer_integrated_gradients only supports --attr-target log_prob")
        lig = symbols["LayerIntegratedGradients"](adapter, adapter.get_input_embeddings())
        return symbols["LLMGradientAttribution"](lig, tokenizer)
    raise ValueError(f"Unsupported method: {method_name}")


def _method_attr_kwargs(args, method_name: str) -> Dict[str, Any]:
    perturbation_kwargs = {
        "forward_in_tokens": bool(int(args.forward_in_tokens)),
        "use_cached_outputs": False,
    }
    if method_name == "layer_integrated_gradients":
        return {"n_steps": int(args.n_steps)}
    if method_name in {"kernel_shap", "lime", "shapley_value_sampling"}:
        return {
            **perturbation_kwargs,
            "num_trials": int(args.num_trials),
            "n_samples": int(args.n_samples),
        }
    if method_name == "feature_ablation":
        return {
            **perturbation_kwargs,
            "num_trials": int(args.num_trials),
        }
    raise ValueError(f"Unsupported method: {method_name}")


def _run_method_attribute(
    *,
    method_name: str,
    adapter,
    tokenizer,
    input_token_ids: Sequence[int],
    baseline_token_id: int,
    target_token_ids: Sequence[int],
    args,
):
    PreTokenizedTextTokenInput = _build_captum_input_class()
    captum_input = PreTokenizedTextTokenInput(
        token_ids=input_token_ids,
        tokenizer=tokenizer,
        baselines=baseline_token_id,
    )
    runner = _build_attr_runner(method_name, adapter, tokenizer, args.attr_target)
    attr_kwargs = _method_attr_kwargs(args, method_name)
    target_tensor = torch.tensor(list(int(token_id) for token_id in target_token_ids), dtype=torch.long)
    return runner.attribute(captum_input, target=target_tensor, **attr_kwargs)


def _sample_target_label(sample, bundle, backbone: HFBackbone, target_mode: str) -> tuple[int, np.ndarray]:
    probs = np.asarray(backbone.predict_label_probs(sample.text, bundle.verbalizers), dtype=np.float32)
    if target_mode == "predicted":
        return int(np.argmax(probs)), probs
    return int(sample.label), probs


def _explain_sample(
    *,
    sample,
    bundle,
    backbone: HFBackbone,
    args,
    method_name: str,
) -> ExplanationResult:
    tokenizer = backbone.tokenizer
    chunks, segmentation_strategy = _build_token_chunks(tokenizer=tokenizer, text=sample.text)
    full_text_token_ids = _tokenize_text_only(tokenizer, sample.text)
    if len(full_text_token_ids) != len(chunks):
        raise RuntimeError(
            "Tokenizer offset_mapping length does not match tokenized input length, cannot preserve chunk alignment."
        )

    target_label, full_probs = _sample_target_label(sample, bundle, backbone, args.target_mode)
    target_label_text = str(bundle.verbalizers[target_label])
    target_token_ids = _encode_target_token_ids(tokenizer, target_label_text, max_length=int(args.max_length))

    prefix_token_ids = _tokenize_text_only(tokenizer, PROMPT_PREFIX)
    suffix_token_ids = _tokenize_text_only(tokenizer, PROMPT_SUFFIX)
    truncation = _truncate_text_token_ids(
        text_token_ids=full_text_token_ids,
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
        target_token_ids=target_token_ids,
        max_length=int(args.max_length),
    )
    kept_text_token_ids = truncation["kept_text_token_ids"]
    dropped_left = int(truncation["dropped_left_token_count"])

    baseline_token_id = _default_baseline_token_id(tokenizer)
    adapter = PromptedCausalLMAdapter(
        base_model=backbone.model,
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
        text_token_count=len(kept_text_token_ids),
        device=backbone.device,
    )
    attr_result = _run_method_attribute(
        method_name=method_name,
        adapter=adapter,
        tokenizer=tokenizer,
        input_token_ids=kept_text_token_ids,
        baseline_token_id=baseline_token_id,
        target_token_ids=target_token_ids,
        args=args,
    )

    seq_attr_tensor = attr_result.seq_attr.detach().cpu().numpy()
    kept_scores = [float(x) for x in np.asarray(seq_attr_tensor, dtype=np.float32).reshape(-1).tolist()]
    if len(kept_scores) != len(kept_text_token_ids):
        raise RuntimeError(
            f"Unexpected attribution length for {method_name}: got={len(kept_scores)} expected={len(kept_text_token_ids)}"
        )

    full_scores = [0.0] * len(full_text_token_ids)
    for idx, score in enumerate(kept_scores):
        full_scores[dropped_left + idx] = float(score)

    chunk_scores_by_id = {chunk.chunk_id: float(full_scores[chunk.chunk_id]) for chunk in chunks}
    chunk_ranking, selected = _rank_desc_scores(full_scores, int(args.k))
    selected_text = compose_text_from_chunk_ids(chunks, selected)
    token_attr_summary = _summarize_token_attr(getattr(attr_result, "token_attr", None))

    dropped_left_char_count = 0
    if dropped_left > 0 and dropped_left <= len(chunks):
        dropped_left_char_count = int(chunks[dropped_left - 1].end_char)

    metadata = {
        "captum_method": method_name,
        "captum_attr_target": str(args.attr_target),
        "captum_num_trials": int(args.num_trials),
        "captum_n_samples": int(args.n_samples),
        "captum_n_steps": int(args.n_steps),
        "target_mode": str(args.target_mode),
        "target_label": int(target_label),
        "target_label_text": target_label_text,
        "target_token_ids": [int(token_id) for token_id in target_token_ids],
        "target_tokens": list(getattr(attr_result, "output_tokens", [])),
        "full_label_probabilities": [float(x) for x in full_probs.tolist()],
        "truncation": {
            **truncation,
            "dropped_left_char_count": int(dropped_left_char_count),
        },
        "prompt_template": "Text:\\n{text}\\nLabel:",
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
        "segmentation_strategy": segmentation_strategy,
        "raw_seq_attr": list(full_scores),
        "token_attr_summary": token_attr_summary,
        "forward_in_tokens_requested": int(args.forward_in_tokens),
        "forward_in_tokens_effective": int(bool(args.forward_in_tokens)),
        "use_cached_outputs_effective": False,
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


def _write_configs(output_root: Path, args, raw_argv: Sequence[str], deterministic_info: Dict[str, Any]) -> None:
    now = time.time()
    run_config_payload = dict(vars(args))
    run_config_payload["provenance"] = build_provenance(
        stage="run_config",
        parsed_args=vars(args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=now,
        end_time=time.time(),
        cwd=Path.cwd(),
    )
    eval_config_payload = dict(vars(args))
    eval_config_payload["provenance"] = build_provenance(
        stage="eval_config",
        parsed_args=vars(args),
        raw_argv=raw_argv,
        deterministic_info=deterministic_info,
        start_time=now,
        end_time=time.time(),
        cwd=Path.cwd(),
    )
    _write_json(output_root / "run_config.json", run_config_payload)
    _write_json(output_root / "eval_config.json", eval_config_payload)


def _method_cli_args(args, method_name: str) -> Dict[str, Any]:
    payload = dict(vars(args))
    payload["method"] = method_name
    return payload


def _run_method(args, raw_argv: Sequence[str], bundle, backbone: HFBackbone, method_name: str) -> None:
    output_root = _method_output_root(args, method_name)
    ensure_dir(output_root)
    ensure_dir(output_root / "samples")

    deterministic_info = configure_determinism(bool(args.deterministic))
    method_args = argparse.Namespace(**_method_cli_args(args, method_name))
    _write_configs(output_root=output_root, args=method_args, raw_argv=raw_argv, deterministic_info=deterministic_info)

    pending_samples = _scan_resume(bundle.samples, output_root=output_root, resume_mode=args.resume_check)
    if pending_samples:
        start = time.time()
        for sample in tqdm(pending_samples, desc=f"captum-{method_name}", dynamic_ncols=True):
            result = _explain_sample(
                sample=sample,
                bundle=bundle,
                backbone=backbone,
                args=args,
                method_name=method_name,
            )
            save_explanation(result, output_root)
        print(f"[done] method={method_name} processed={len(pending_samples)} elapsed={time.time() - start:.2f}s")
    else:
        print(f"[resume] method={method_name} no pending samples")

    summary_path = rebuild_summary_csv(output_root)
    print(f"[done] method={method_name} summary={summary_path}")

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
    print(f"[eval] method={method_name} report={output_root / 'eval_report.json'}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _require_runtime_dependencies()
    methods = _parse_methods(args.methods)
    if int(args.forward_in_tokens) not in {0, 1}:
        raise ValueError("--forward-in-tokens must be 0 or 1")
    set_seed(int(args.seed))

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    backbone = HFBackbone(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        embedding_layer_ratio=0.7,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    for method_name in methods:
        _run_method(args=args, raw_argv=raw_argv, bundle=bundle, backbone=backbone, method_name=method_name)


if __name__ == "__main__":
    main()
