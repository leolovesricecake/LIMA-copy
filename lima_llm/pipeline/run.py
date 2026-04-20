from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from ..backbone import build_backbone
from ..chunking import build_chunker
from ..data import load_dataset_bundle
from ..objective.submodular import ObjectiveWeights
from ..utils import (
    ensure_dir,
    format_label_distribution,
    parse_lambdas,
    parse_q_values,
    set_seed,
)
from .explainer import ExplainerConfig, TextLIMAExplainer
from .io import rebuild_summary_csv, save_explanation
from .resume import is_sample_completed, sample_output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIMA LLM v1 pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=["sst2", "eraser_movie_reviews"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eraser-root", type=str, default=None)

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--embedding-layer-ratio", type=float, default=0.7)
    parser.add_argument("--mock-backbone", action="store_true")

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambdas", type=str, default="1,1,1,1")
    parser.add_argument("--chunker", type=str, default="sentence", choices=["sentence", "fixed_token"])
    parser.add_argument("--fixed-token-size", type=int, default=64)
    parser.add_argument("--search", type=str, default="greedy", choices=["greedy", "bidirectional"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--verbose-chunks", action="store_true")

    parser.add_argument("--output-dir", type=str, default="lima_llm_results")
    parser.add_argument("--resume-check", type=str, default="strict", choices=["strict", "exists-only"])

    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-q-values", type=str, default="1,5,10,20,50")
    parser.add_argument("--eval-random-trials", type=int, default=5)
    parser.add_argument("--eval-gradient-baseline", action="store_true")
    return parser


def _preview_dataset(bundle, backbone, dry_run: int) -> None:
    print(f"[dry-run] dataset={bundle.dataset_name} split={bundle.split} size={len(bundle.samples)}")
    print(f"[dry-run] label_distribution={format_label_distribution(s.label for s in bundle.samples)}")

    n = min(dry_run, len(bundle.samples))
    for idx in range(n):
        sample = bundle.samples[idx]
        token_len = backbone.tokenize_len(sample.text)
        rationale_chars = sum(end - start for start, end in sample.rationale_char_spans)
        coverage = rationale_chars / max(1, len(sample.text))
        snippet = sample.text.replace("\n", " ")[:120]
        print(
            f"[dry-run] #{idx} id={sample.sample_id} label={sample.label} "
            f"chars={len(sample.text)} tokens={token_len} rationale_cov={coverage:.4f} text={snippet!r}"
        )


def _print_chunk_preview(explainer: TextLIMAExplainer, sample) -> None:
    chunks = explainer.chunker(sample.text)
    print(f"[chunk-preview] sample={sample.sample_id} chunk_count={len(chunks)}")
    for chunk in chunks:
        preview = chunk.text.replace("\n", " ")[:40]
        print(f"  chunk#{chunk.chunk_id} span=[{chunk.start_char},{chunk.end_char}) text={preview!r}")


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
        f"[resume] mode={resume_mode} selected={len(samples)} completed={completed} pending={len(pending)}"
    )
    return pending, completed


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    weights = ObjectiveWeights(*parse_lambdas(args.lambdas))

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        eraser_root=args.eraser_root,
    )

    backbone = build_backbone(
        model_path=args.model_path,
        device=args.device,
        use_mock_backbone=args.mock_backbone,
        max_length=args.max_length,
        embedding_layer_ratio=args.embedding_layer_ratio,
        dtype=args.dtype,
    )

    chunker = build_chunker(
        method=args.chunker,
        tokenizer=getattr(backbone, "tokenizer", None),
        fixed_token_size=args.fixed_token_size,
    )

    config = ExplainerConfig(
        dataset_name=bundle.dataset_name,
        split=args.split,
        k=args.k,
        search=args.search,
        weights=weights,
    )
    explainer = TextLIMAExplainer(
        backbone=backbone,
        chunker=chunker,
        verbalizers=bundle.verbalizers,
        config=config,
    )

    if args.dry_run > 0:
        _preview_dataset(bundle, backbone, dry_run=args.dry_run)
        preview_count = min(1, len(bundle.samples))
        for idx in range(preview_count):
            _print_chunk_preview(explainer, bundle.samples[idx])
        return

    output_root = (
        Path(args.output_dir)
        / args.dataset
        / f"model-{args.model_path.split('/')[-1].replace('.', '_')}"
        / f"chunk-{args.chunker}_search-{args.search}_k-{args.k}_lam-{args.lambdas.replace(',', '-') }"
    )
    ensure_dir(output_root)
    ensure_dir(output_root / "samples")

    config_path = output_root / "run_config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    pending_samples, _ = _scan_resume(bundle.samples, output_root=output_root, resume_mode=args.resume_check)
    if not pending_samples:
        print("[resume] no pending samples, exiting")
    else:
        begin = time.time()
        for sample in tqdm(pending_samples, desc="lima-llm-v1", dynamic_ncols=True):
            result = explainer.explain_sample(sample, verbose=args.verbose_chunks)
            save_explanation(result, output_root)
        elapsed = time.time() - begin
        print(f"[done] processed={len(pending_samples)} elapsed={elapsed:.2f}s")

    summary_path = rebuild_summary_csv(output_root)
    print(f"[done] summary={summary_path}")

    if args.run_eval:
        from ..eval.evaluate import evaluate_saved_explanations

        q_values = parse_q_values(args.eval_q_values)
        eval_report = evaluate_saved_explanations(
            output_root=output_root,
            bundle=bundle,
            backbone=backbone,
            verbalizers=bundle.verbalizers,
            q_values=q_values,
            random_trials=args.eval_random_trials,
            include_gradient_baseline=args.eval_gradient_baseline,
        )
        report_path = output_root / "eval_report.json"
        report_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] report={report_path}")


if __name__ == "__main__":
    main()
