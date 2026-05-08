#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lima_llm.backbone import build_backbone
from lima_llm.chunking import build_chunker
from lima_llm.data import load_dataset_bundle
from lima_llm.eval.equivalence import compare_trace_dicts
from lima_llm.objective.submodular import ObjectiveWeights, TextSubmodularObjective
from lima_llm.search import run_bidirectional_search, run_forward_greedy
from lima_llm.utils import parse_lambdas, set_seed


def _abs_max(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    return float(np.max(np.abs(left_arr - right_arr))) if left_arr.size > 0 else 0.0


def _append_failure(report: Dict[str, Any], section: str, payload: Dict[str, Any]) -> None:
    report["passed"] = False
    if report.get("first_failure") is None:
        report["first_failure"] = {"section": section, **payload}


def _check_backbone_step_equivalence(
    report: Dict[str, Any],
    backbone,
    verbalizers: Sequence[str],
    texts: Sequence[str],
    tolerance: float,
    max_batch_size: int,
) -> None:
    max_prob_diff = 0.0
    max_embed_diff = 0.0

    # 1) Single-text strict vs optimized path (sanity baseline).
    for text in texts:
        safe_text = text if text else "<EMPTY>"

        backbone.set_equivalence_mode("strict_ref")
        probs_ref = backbone.predict_label_probs(safe_text, verbalizers)
        emb_ref = backbone.embed_text(safe_text)

        backbone.set_equivalence_mode("optimized_batch")
        probs_opt = backbone.predict_label_probs_batch([safe_text], verbalizers)[0]
        emb_opt = backbone.embed_texts([safe_text])[0]

        prob_diff = _abs_max(probs_ref.tolist(), probs_opt.tolist())
        embed_diff = _abs_max(emb_ref.tolist(), emb_opt.tolist())
        max_prob_diff = max(max_prob_diff, prob_diff)
        max_embed_diff = max(max_embed_diff, embed_diff)

        if prob_diff > tolerance:
            _append_failure(
                report,
                "backbone.predict",
                {
                    "text_preview": safe_text[:120],
                    "abs_diff": prob_diff,
                    "left": probs_ref.tolist(),
                    "right": probs_opt.tolist(),
                },
            )
            return
        if embed_diff > tolerance:
            _append_failure(
                report,
                "backbone.embed",
                {
                    "text_preview": safe_text[:120],
                    "abs_diff": embed_diff,
                },
            )
            return

    # 2) Multi-text strict-single vs optimized-batch (the key Gate C contract).
    batch_size = max(2, int(max_batch_size))
    for start in range(0, len(texts), batch_size):
        batch_raw = list(texts[start : start + batch_size])
        batch = [item if item else "<EMPTY>" for item in batch_raw]
        if len(batch) < 2:
            continue

        backbone.set_equivalence_mode("strict_ref")
        probs_ref_rows = [backbone.predict_label_probs(text, verbalizers) for text in batch]
        emb_ref_rows = [backbone.embed_text(text) for text in batch]

        backbone.set_equivalence_mode("optimized_batch")
        probs_opt_mat = backbone.predict_label_probs_batch(batch, verbalizers)
        emb_opt_rows = backbone.embed_texts(batch)

        for row_idx, text in enumerate(batch):
            prob_diff = _abs_max(probs_ref_rows[row_idx].tolist(), probs_opt_mat[row_idx].tolist())
            embed_diff = _abs_max(emb_ref_rows[row_idx].tolist(), emb_opt_rows[row_idx].tolist())
            max_prob_diff = max(max_prob_diff, prob_diff)
            max_embed_diff = max(max_embed_diff, embed_diff)

            if prob_diff > tolerance:
                _append_failure(
                    report,
                    "backbone.predict_batch",
                    {
                        "batch_start": start,
                        "batch_size": len(batch),
                        "row": row_idx,
                        "text_preview": text[:120],
                        "abs_diff": prob_diff,
                        "left": probs_ref_rows[row_idx].tolist(),
                        "right": probs_opt_mat[row_idx].tolist(),
                    },
                )
                return
            if embed_diff > tolerance:
                _append_failure(
                    report,
                    "backbone.embed_batch",
                    {
                        "batch_start": start,
                        "batch_size": len(batch),
                        "row": row_idx,
                        "text_preview": text[:120],
                        "abs_diff": embed_diff,
                    },
                )
                return

    report["sections"]["backbone"] = {
        "passed": True,
        "max_prob_abs_diff": max_prob_diff,
        "max_embed_abs_diff": max_embed_diff,
        "checked_texts": len(texts),
        "checked_batch_size": batch_size,
    }


def _build_objective(backbone, sample, chunker, verbalizers, weights: ObjectiveWeights):
    chunks = chunker(sample.text)
    chunk_embeddings = [backbone.embed_text(chunk.text if chunk.text else "<EMPTY>") for chunk in chunks]
    objective = TextSubmodularObjective(
        backbone=backbone,
        text=sample.text,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        target_label=sample.label,
        verbalizers=verbalizers,
        weights=weights,
    )
    return chunks, objective


def _check_objective_step_equivalence(
    report: Dict[str, Any],
    backbone,
    samples,
    chunker,
    verbalizers: Sequence[str],
    weights: ObjectiveWeights,
    tolerance: float,
    max_candidates: int,
) -> None:
    max_abs_diff = 0.0
    checked = 0

    for sample in samples:
        # Cross-mode strict vs optimized equivalence on the same subset/candidate set.
        backbone.set_equivalence_mode("strict_ref")
        chunks_ref, objective_ref = _build_objective(backbone, sample, chunker, verbalizers, weights)
        backbone.set_equivalence_mode("optimized_batch")
        chunks_opt, objective_opt = _build_objective(backbone, sample, chunker, verbalizers, weights)
        if len(chunks_ref) != len(chunks_opt):
            _append_failure(
                report,
                "objective.chunk_count",
                {
                    "sample_id": sample.sample_id,
                    "strict_ref_count": len(chunks_ref),
                    "optimized_count": len(chunks_opt),
                },
            )
            return

        candidate_ids = [chunk.chunk_id for chunk in chunks_ref][: max(1, int(max_candidates))]
        subset = []
        base_ref, gain_map_ref, score_map_ref = objective_ref.evaluate_gains(subset, candidate_ids)
        base_opt, gain_map_opt, score_map_opt = objective_opt.evaluate_gains(subset, candidate_ids)
        base_diff = abs(float(base_ref.total) - float(base_opt.total))
        max_abs_diff = max(max_abs_diff, base_diff)
        if base_diff > tolerance:
            _append_failure(
                report,
                "objective.cross_mode.base",
                {
                    "sample_id": sample.sample_id,
                    "subset": subset,
                    "left_total": float(base_ref.total),
                    "right_total": float(base_opt.total),
                    "abs_diff": base_diff,
                },
            )
            return
        for candidate in candidate_ids:
            diff_gain = abs(float(gain_map_ref[candidate]) - float(gain_map_opt[candidate]))
            diff_score = abs(float(score_map_ref[candidate].total) - float(score_map_opt[candidate].total))
            max_abs_diff = max(max_abs_diff, diff_gain, diff_score)
            if diff_gain > tolerance or diff_score > tolerance:
                _append_failure(
                    report,
                    "objective.cross_mode.gains",
                    {
                        "sample_id": sample.sample_id,
                        "candidate": candidate,
                        "gain_left": float(gain_map_ref[candidate]),
                        "gain_right": float(gain_map_opt[candidate]),
                        "score_left": float(score_map_ref[candidate].total),
                        "score_right": float(score_map_opt[candidate].total),
                        "abs_diff_gain": diff_gain,
                        "abs_diff_score": diff_score,
                    },
                )
                return

        subsets = [[], candidate_ids[:1], candidate_ids[:2], candidate_ids[:3]]
        scores_ref = objective_ref.evaluate_subsets(subsets)
        scores_opt = objective_opt.evaluate_subsets(subsets)
        for left_item, right_item in zip(scores_ref, scores_opt):
            diff_total = abs(float(left_item.total) - float(right_item.total))
            max_abs_diff = max(max_abs_diff, diff_total)
            if diff_total > tolerance:
                _append_failure(
                    report,
                    "objective.cross_mode.subsets",
                    {
                        "sample_id": sample.sample_id,
                        "subset": list(left_item.subset_indices),
                        "left_total": float(left_item.total),
                        "right_total": float(right_item.total),
                        "abs_diff": diff_total,
                    },
                )
                return

        # Intra-mode API consistency: evaluate_gains/subsets vs evaluate_gain/subset.
        for mode in ("strict_ref", "optimized_batch"):
            backbone.set_equivalence_mode(mode)
            chunks, objective = _build_objective(backbone, sample, chunker, verbalizers, weights)
            mode_candidate_ids = [chunk.chunk_id for chunk in chunks][: max(1, int(max_candidates))]
            subset = []

            _, gain_map, score_map = objective.evaluate_gains(subset, mode_candidate_ids)
            for candidate in mode_candidate_ids:
                gain_single, _, score_single = objective.evaluate_gain(subset, candidate)
                diff_gain = abs(float(gain_map[candidate]) - float(gain_single))
                diff_score = abs(float(score_map[candidate].total) - float(score_single.total))
                max_abs_diff = max(max_abs_diff, diff_gain, diff_score)
                if diff_gain > tolerance or diff_score > tolerance:
                    _append_failure(
                        report,
                        "objective.gains",
                        {
                            "sample_id": sample.sample_id,
                            "mode": mode,
                            "candidate": candidate,
                            "gain_batch": float(gain_map[candidate]),
                            "gain_single": float(gain_single),
                            "score_batch": float(score_map[candidate].total),
                            "score_single": float(score_single.total),
                        },
                    )
                    return

            mode_subsets = [[], mode_candidate_ids[:1], mode_candidate_ids[:2], mode_candidate_ids[:3]]
            batch_scores = objective.evaluate_subsets(mode_subsets)
            single_scores = [objective.evaluate_subset(subset_item) for subset_item in mode_subsets]
            for batch_item, single_item in zip(batch_scores, single_scores):
                diff_total = abs(float(batch_item.total) - float(single_item.total))
                max_abs_diff = max(max_abs_diff, diff_total)
                if diff_total > tolerance:
                    _append_failure(
                        report,
                        "objective.subsets",
                        {
                            "sample_id": sample.sample_id,
                            "mode": mode,
                            "subset": list(batch_item.subset_indices),
                            "total_batch": float(batch_item.total),
                            "total_single": float(single_item.total),
                        },
                    )
                    return
            checked += 1

    report["sections"]["objective"] = {
        "passed": True,
        "max_abs_diff": max_abs_diff,
        "checked_cases": checked,
    }


def _search_once(search: str, objective, candidate_ids: Sequence[int], k: int):
    if search == "greedy":
        return run_forward_greedy(objective, candidate_ids=candidate_ids, k=k)
    if search == "bidirectional":
        return run_bidirectional_search(objective, candidate_ids=candidate_ids, k=k)
    raise ValueError(f"Unsupported search: {search}")


def _trace_as_dicts(items) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def _check_search_step_equivalence(
    report: Dict[str, Any],
    backbone,
    samples,
    chunker,
    verbalizers: Sequence[str],
    weights: ObjectiveWeights,
    tolerance: float,
    k: int,
    searches: Sequence[str],
) -> None:
    max_abs_diff = 0.0
    checked = 0

    for sample in samples:
        backbone.set_equivalence_mode("strict_ref")
        chunks_ref, objective_ref = _build_objective(backbone, sample, chunker, verbalizers, weights)
        candidate_ids = [chunk.chunk_id for chunk in chunks_ref]
        run_k = min(max(0, int(k)), len(candidate_ids))

        backbone.set_equivalence_mode("optimized_batch")
        chunks_opt, objective_opt = _build_objective(backbone, sample, chunker, verbalizers, weights)
        if len(chunks_ref) != len(chunks_opt):
            _append_failure(
                report,
                "search.chunk_count",
                {
                    "sample_id": sample.sample_id,
                    "strict_ref_count": len(chunks_ref),
                    "optimized_count": len(chunks_opt),
                },
            )
            return

        for search in searches:
            backbone.set_equivalence_mode("strict_ref")
            selected_ref, trace_ref = _search_once(search, objective_ref, candidate_ids=candidate_ids, k=run_k)
            backbone.set_equivalence_mode("optimized_batch")
            selected_opt, trace_opt = _search_once(search, objective_opt, candidate_ids=candidate_ids, k=run_k)

            if selected_ref != selected_opt:
                mismatch_step = None
                for idx, (left_id, right_id) in enumerate(zip(selected_ref, selected_opt)):
                    if left_id != right_id:
                        mismatch_step = idx
                        break
                if mismatch_step is None and len(selected_ref) != len(selected_opt):
                    mismatch_step = min(len(selected_ref), len(selected_opt))

                _append_failure(
                    report,
                    "search.selected_chunk_ids",
                    {
                        "sample_id": sample.sample_id,
                        "search": search,
                        "first_mismatch_step": mismatch_step,
                        "left_prefix": selected_ref[: min(len(selected_ref), 10)],
                        "right_prefix": selected_opt[: min(len(selected_opt), 10)],
                        "left_trace_preview": _trace_as_dicts(trace_ref)[: min(len(trace_ref), 5)],
                        "right_trace_preview": _trace_as_dicts(trace_opt)[: min(len(trace_opt), 5)],
                        "left": selected_ref,
                        "right": selected_opt,
                    },
                )
                return

            trace_result = compare_trace_dicts(
                left_trace=_trace_as_dicts(trace_ref),
                right_trace=_trace_as_dicts(trace_opt),
                tolerance=tolerance,
            )
            if trace_result["max_abs_diff"] is not None:
                max_abs_diff = max(max_abs_diff, float(trace_result["max_abs_diff"]))
            if not trace_result["passed"]:
                _append_failure(
                    report,
                    "search.trace",
                    {
                        "sample_id": sample.sample_id,
                        "search": search,
                        "details": trace_result["first_failure"],
                    },
                )
                return
            checked += 1

    report["sections"]["search"] = {
        "passed": True,
        "max_abs_diff": max_abs_diff,
        "checked_cases": checked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate C-0 key-step equivalence checker")
    parser.add_argument("--dataset", type=str, required=True, choices=["sst2", "eraser_movie_reviews"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eraser-root", type=str, default=None)
    parser.add_argument("--sst2-source", type=str, default=None)
    parser.add_argument("--dataset-cache-dir", type=str, default=None)

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--embedding-layer-ratio", type=float, default=0.7)
    parser.add_argument("--mock-backbone", action="store_true")

    parser.add_argument("--chunker", type=str, default="sentence", choices=["sentence", "fixed_token"])
    parser.add_argument("--fixed-token-size", type=int, default=64)
    parser.add_argument("--lambdas", type=str, default="1,1,1,1")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--searches", type=str, default="greedy,bidirectional")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-batch-size-check", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(int(args.seed))

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        eraser_root=args.eraser_root,
        sst2_source=args.sst2_source,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    backbone = build_backbone(
        model_path=args.model_path,
        device=args.device,
        use_mock_backbone=args.mock_backbone,
        max_length=args.max_length,
        embedding_layer_ratio=args.embedding_layer_ratio,
        dtype=args.dtype,
        equivalence_mode="strict_ref",
    )
    chunker = build_chunker(
        method=args.chunker,
        tokenizer=getattr(backbone, "tokenizer", None),
        fixed_token_size=args.fixed_token_size,
    )
    weights = ObjectiveWeights(*parse_lambdas(args.lambdas))
    searches = [item.strip() for item in args.searches.split(",") if item.strip()]
    if not searches:
        raise ValueError("--searches must not be empty")

    sample_list = list(bundle.samples)
    texts_to_check: List[str] = []
    for sample in sample_list:
        texts_to_check.append(sample.text)
        for chunk in chunker(sample.text)[: min(8, max(1, int(args.max_candidates)))]:
            texts_to_check.append(chunk.text)
    dedup_texts = []
    seen = set()
    for text in texts_to_check:
        key = text if text else "<EMPTY>"
        if key in seen:
            continue
        seen.add(key)
        dedup_texts.append(text)

    report: Dict[str, Any] = {
        "passed": True,
        "reason": "ok",
        "tolerance": float(args.tolerance),
        "sample_count": len(sample_list),
        "sections": {},
        "first_failure": None,
    }

    _check_backbone_step_equivalence(
        report=report,
        backbone=backbone,
        verbalizers=bundle.verbalizers,
        texts=dedup_texts,
        tolerance=float(args.tolerance),
        max_batch_size=int(args.max_batch_size_check),
    )
    if report["passed"]:
        _check_objective_step_equivalence(
            report=report,
            backbone=backbone,
            samples=sample_list,
            chunker=chunker,
            verbalizers=bundle.verbalizers,
            weights=weights,
            tolerance=float(args.tolerance),
            max_candidates=int(args.max_candidates),
        )
    if report["passed"]:
        _check_search_step_equivalence(
            report=report,
            backbone=backbone,
            samples=sample_list,
            chunker=chunker,
            verbalizers=bundle.verbalizers,
            weights=weights,
            tolerance=float(args.tolerance),
            k=int(args.k),
            searches=searches,
        )

    if not report["passed"]:
        report["reason"] = "key_step_mismatch"

    output_payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_payload, encoding="utf-8")
        print(f"[gate-c-key-steps] wrote {output_path}")

    print(output_payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
