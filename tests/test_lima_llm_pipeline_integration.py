import json
from pathlib import Path

import pandas as pd
import pytest

from lima_llm.pipeline.run import main


def _build_tiny_eraser(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc-1.txt").write_text("This movie is great. I loved the acting.", encoding="utf-8")
    (docs / "doc-2.txt").write_text("This movie is terrible. Waste of time.", encoding="utf-8")

    rows = [
        {
            "annotation_id": "1",
            "classification": "POS",
            "evidences": [[{"docid": "doc-1", "start_char": 0, "end_char": 19}]],
        },
        {
            "annotation_id": "2",
            "classification": "NEG",
            "evidences": [[{"docid": "doc-2", "start_char": 0, "end_char": 23}]],
        },
    ]
    with open(root / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_tiny_eraser_three(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc-1.txt").write_text("This movie is great. I loved the acting.", encoding="utf-8")
    (docs / "doc-2.txt").write_text("This movie is terrible. Waste of time.", encoding="utf-8")
    (docs / "doc-3.txt").write_text("The film was average but watchable.", encoding="utf-8")

    rows = [
        {
            "annotation_id": "1",
            "classification": "POS",
            "evidences": [[{"docid": "doc-1", "start_char": 0, "end_char": 19}]],
        },
        {
            "annotation_id": "2",
            "classification": "NEG",
            "evidences": [[{"docid": "doc-2", "start_char": 0, "end_char": 23}]],
        },
        {
            "annotation_id": "3",
            "classification": "POS",
            "evidences": [[{"docid": "doc-3", "start_char": 0, "end_char": 17}]],
        },
    ]
    with open(root / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_tiny_eraser_four(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc-1.txt").write_text("This movie is great. I loved the acting.", encoding="utf-8")
    (docs / "doc-2.txt").write_text("This movie is terrible. Waste of time.", encoding="utf-8")
    (docs / "doc-3.txt").write_text("The film was average but watchable.", encoding="utf-8")
    (docs / "doc-4.txt").write_text("An excellent and thoughtful drama.", encoding="utf-8")

    rows = [
        {
            "annotation_id": "1",
            "classification": "POS",
            "evidences": [[{"docid": "doc-1", "start_char": 0, "end_char": 19}]],
        },
        {
            "annotation_id": "2",
            "classification": "NEG",
            "evidences": [[{"docid": "doc-2", "start_char": 0, "end_char": 23}]],
        },
        {
            "annotation_id": "3",
            "classification": "POS",
            "evidences": [[{"docid": "doc-3", "start_char": 0, "end_char": 17}]],
        },
        {
            "annotation_id": "4",
            "classification": "POS",
            "evidences": [[{"docid": "doc-4", "start_char": 0, "end_char": 22}]],
        },
    ]
    with open(root / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_args(output_root: Path) -> list[str]:
    return ["--base-save-dir", str(output_root.parent), "--save-dir", output_root.name]


def test_pipeline_mock_backbone_end_to_end(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out = tmp_path / "results"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--search",
        "greedy",
        "--k",
        "2",
        *_save_args(out),
        "--run-eval",
    ]
    main(argv)

    sample_jsons = list(out.glob("**/samples/*.json"))
    assert len(sample_jsons) == 2
    sample_payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
    assert "explain_timing_breakdown" in sample_payload.get("metadata", {})
    assert "objective_cache_stats" in sample_payload.get("metadata", {})
    assert "chunk_diagnostics" in sample_payload.get("metadata", {})
    assert "chunk_features_by_id" in sample_payload.get("metadata", {})
    assert "chunk_feature_coverage" in sample_payload.get("metadata", {})
    chunk_diag = sample_payload["metadata"]["chunk_diagnostics"]
    chunk_features = sample_payload["metadata"]["chunk_features_by_id"]
    chunk_cov = sample_payload["metadata"]["chunk_feature_coverage"]
    assert "chunk_strategy" in chunk_diag
    assert "chunk_count" in chunk_diag
    assert "singleton_orphan_punctuation_chunks" in chunk_diag
    assert "leading_close_punct_chunks" in chunk_diag
    assert "abbreviation_singleton_chunks" in chunk_diag
    assert "cross_newline_boundary_chunks" in chunk_diag
    assert "fallback_applied" in chunk_diag
    assert "fallback_reason" in chunk_diag
    assert isinstance(chunk_features, dict)
    assert len(chunk_features) == len(sample_payload.get("chunks", []))
    first_feat = next(iter(chunk_features.values()))
    assert "char_len" in first_feat
    assert "word_count" in first_feat
    assert "orphan_punctuation" in first_feat
    assert "token_alignment_mode" in chunk_cov
    assert "token_coverage_ratio" in chunk_cov

    summary = list(out.glob("**/summary.csv"))
    report = list(out.glob("**/eval_report.json"))
    trajectory_points_csv = list(out.glob("**/trajectory_points.csv"))
    trajectory_points_jsonl = list(out.glob("**/trajectory_points.jsonl"))
    trajectory_summary_csv = list(out.glob("**/trajectory_summary.csv"))
    assert summary
    assert report
    assert trajectory_points_csv
    assert trajectory_points_jsonl
    assert trajectory_summary_csv

    run_cfg_paths = list(out.glob("**/run_config.json"))
    eval_cfg_paths = list(out.glob("**/eval_config.json"))
    assert len(run_cfg_paths) == 1
    assert len(eval_cfg_paths) == 1

    run_cfg = json.loads(run_cfg_paths[0].read_text(encoding="utf-8"))
    eval_cfg = json.loads(eval_cfg_paths[0].read_text(encoding="utf-8"))
    eval_report = json.loads(report[0].read_text(encoding="utf-8"))
    assert run_cfg["eval_granularity"] == "token"
    assert eval_cfg["eval_granularity"] == "token"
    assert eval_report["metric_settings"]["perturbation_unit"] == "token"
    assert "timing_breakdown" in eval_report
    assert "cache_stats" in eval_report
    assert "prefetch_stats" in eval_report
    assert "backbone_batch_stats" in eval_report
    assert "timing_breakdown" in eval_report["metrics_secondary"]
    assert "cache_stats" in eval_report["metrics_secondary"]
    assert "prefetch_stats" in eval_report["metrics_secondary"]
    assert "backbone_batch_stats" in eval_report["metrics_secondary"]
    assert "eval_forward_counters_delta" in eval_report["metrics_secondary"]
    assert "explain_forward_counters_total" in eval_report["metrics_secondary"]
    assert "explain_forward_counters_mean_per_sample" in eval_report["metrics_secondary"]
    assert "explain_diagnostics" in eval_report
    assert int(eval_report["explain_diagnostics"]["sample_count"]) == len(sample_jsons)
    assert eval_report["prefetch_stats"]["batch_fallback_count"] == 0
    assert eval_report["artifacts"]["trajectory_points_csv"] == "trajectory_points.csv"
    assert eval_report["artifacts"]["trajectory_points_jsonl"] == "trajectory_points.jsonl"
    assert eval_report["artifacts"]["trajectory_summary_csv"] == "trajectory_summary.csv"

    predicted_aopc = float(eval_report["metrics_by_target"]["predicted"]["metrics_primary"]["aopc"])
    trajectory_rows = pd.read_csv(trajectory_points_csv[0])
    recomputed_aopc = float(trajectory_rows.groupby("sample_id")["prob_drop_from_full"].mean().mean())
    assert predicted_aopc == pytest.approx(recomputed_aopc)

    for sample_json in sample_jsons:
        sample_payload = json.loads(sample_json.read_text(encoding="utf-8"))
        assert "trajectory_step_count" in sample_payload
        assert "trajectory_row_range" in sample_payload
        assert "trajectory_target" not in sample_payload
        assert "trajectory_artifact" not in sample_payload

    for payload in (run_cfg, eval_cfg, eval_report):
        assert "provenance" in payload
        prov = payload["provenance"]
        assert isinstance(prov, dict)
        assert "git" in prov
        assert "command" in prov
        assert "environment" in prov
        assert "timing" in prov


def test_pipeline_adaptive_chunker_emits_adaptive_diagnostics(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out = tmp_path / "results_adaptive"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        "balanced",
        "--search",
        "greedy",
        "--k",
        "2",
        *_save_args(out),
    ]
    main(argv)

    sample_jsons = list(out.glob("**/samples/*.json"))
    assert len(sample_jsons) == 2
    sample_payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
    chunk_diag = sample_payload.get("metadata", {}).get("chunk_diagnostics", {})
    assert chunk_diag.get("chunk_strategy_requested") == "adaptive"
    assert chunk_diag.get("adaptive_enabled") is True
    assert chunk_diag.get("adaptive_profile") == "balanced"
    assert chunk_diag.get("adaptive_bucket") in {"short", "medium", "long", "very_long"}
    assert "adaptive_features" in chunk_diag
    assert "adaptive_postprocess" in chunk_diag
    assert "adaptive_stage_chunk_counts" in chunk_diag


def test_pipeline_adaptive_overrides_json_is_applied(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)
    override_path = tmp_path / "adaptive_overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "min_effective_chunks": 7,
                "short_floor_min_words": 12,
                "short_floor_signal_mode": "always",
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "results_adaptive_override"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        "balanced",
        "--adaptive-overrides-json",
        str(override_path),
        "--search",
        "greedy",
        "--k",
        "2",
        *_save_args(out),
    ]
    main(argv)

    sample_payload = json.loads(next(out.glob("**/samples/*.json")).read_text(encoding="utf-8"))
    chunk_diag = sample_payload.get("metadata", {}).get("chunk_diagnostics", {})
    applied = chunk_diag.get("adaptive_overrides_applied", {})
    assert isinstance(applied, dict)
    assert int(applied.get("min_effective_chunks", 0)) == 7
    assert int(applied.get("short_floor_min_words", 0)) == 12
    assert str(applied.get("short_floor_signal_mode")) == "always"


def test_pipeline_sample_ids_file_filters_samples(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text(json.dumps(["eraser-mr-validation-1"]), encoding="utf-8")

    out = tmp_path / "results_filtered"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--search",
        "greedy",
        "--k",
        "2",
        "--sample-ids-file",
        str(sample_ids_path),
        *_save_args(out),
    ]
    main(argv)

    sample_jsons = list(out.glob("**/samples/*.json"))
    assert len(sample_jsons) == 1
    payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
    assert payload.get("sample_id") == "eraser-mr-validation-1"


def test_pipeline_hparam_search_inline_and_final_eval(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_three"
    _build_tiny_eraser_three(eraser_root)
    hparam_space = tmp_path / "hparam_space.json"
    hparam_space.write_text(
        json.dumps(
            {
                "parameters": {
                    "min_effective_chunks": [4, 5],
                    "guard_mode": ["hard_cap"],
                }
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "results_hparam"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        "balanced",
        "--search",
        "greedy",
        "--k",
        "2",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "random",
        "--hparam-random-trials",
        "1",
        "--hparam-space-file",
        str(hparam_space),
        *_save_args(out),
        "--run-eval",
    ]
    main(argv)

    run_cfg_path = next(out.glob("**/run_config.json"))
    report_path = next(out.glob("**/eval_report.json"))
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert run_cfg.get("hparam_search_enabled") is True
    assert report.get("hparam_search_enabled") is True
    assert run_cfg.get("hparam_search_split") == "validation"
    assert int(run_cfg.get("hparam_tune_size")) == 2
    assert run_cfg.get("lambda_search_enabled") is False
    assert isinstance(run_cfg.get("best_lambdas"), str)
    assert run_cfg.get("best_lambdas") == "1,1,1,1"
    summary_path = Path(str(run_cfg.get("hparam_search_summary_path")))
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("lambda_search_enabled") is False
    assert isinstance(summary.get("candidate_adaptive_overrides"), list)
    assert isinstance(summary.get("candidate_lambdas"), list)
    assert isinstance(summary.get("best_lambdas"), str)
    assert summary.get("best_lambdas") == "1,1,1,1"
    assert summary.get("best_trial", {}).get("adaptive_overrides") is not None


def test_pipeline_hparam_search_same_split_requires_eval_remainder(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_two"
    _build_tiny_eraser(eraser_root)
    out = tmp_path / "results_hparam_fail"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "random",
        "--hparam-random-trials",
        "1",
        *_save_args(out),
    ]
    with pytest.raises(ValueError, match="remaining_eval"):
        main(argv)


def test_pipeline_hparam_search_default_space_uses_implicit_budget_16(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_default_budget"
    _build_tiny_eraser_three(eraser_root)

    out = tmp_path / "results_hparam_default_budget"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        "balanced",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "grid+random",
        "--hparam-random-trials",
        "4",
        *_save_args(out),
        "--run-eval",
    ]
    main(argv)

    run_cfg_path = next(out.glob("**/run_config.json"))
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    summary_path = Path(str(run_cfg.get("hparam_search_summary_path")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert int(summary.get("hparam_max_trials", 0)) == 16
    assert int(summary.get("effective_trial_budget", 0)) == 16
    assert len(summary.get("trials", [])) == 16
    assert len(summary.get("candidate_adaptive_overrides", [])) == 16
    trial_dirs = [p for p in summary_path.parent.glob("trials/trial_*") if p.is_dir()]
    assert len(trial_dirs) == 16


def test_pipeline_hparam_search_with_lambda_enabled(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_three_lambda"
    _build_tiny_eraser_three(eraser_root)
    hparam_space = tmp_path / "hparam_space_lambda.json"
    hparam_space.write_text(
        json.dumps(
            {
                "parameters": {
                    "min_effective_chunks": [4],
                    "lambda1": [0.5, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "results_hparam_lambda"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--adaptive-profile",
        "balanced",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "grid",
        "--hparam-max-trials",
        "3",
        "--hparam-enable-lambda-search",
        "--hparam-space-file",
        str(hparam_space),
        *_save_args(out),
        "--run-eval",
    ]
    main(argv)

    run_cfg_path = next(out.glob("**/run_config.json"))
    report_path = next(out.glob("**/eval_report.json"))
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert run_cfg.get("hparam_search_enabled") is True
    assert report.get("hparam_search_enabled") is True
    assert run_cfg.get("lambda_search_enabled") is True
    assert report.get("lambda_search_enabled") is True
    best_lambdas = str(run_cfg.get("best_lambdas"))
    assert isinstance(best_lambdas, str)
    assert len(best_lambdas.split(",")) == 4

    summary_path = Path(str(run_cfg.get("hparam_search_summary_path")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("lambda_search_enabled") is True
    assert isinstance(summary.get("candidate_lambdas"), list)
    assert isinstance(summary.get("candidate_adaptive_overrides"), list)
    assert summary.get("best_lambdas") == best_lambdas
    assert summary.get("best_trial", {}).get("lambdas") == best_lambdas
    assert "candidate_lambdas" in summary.get("trials", [])[0]


def test_pipeline_hparam_search_requires_adaptive_chunker(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_two_non_adaptive"
    _build_tiny_eraser(eraser_root)
    out = tmp_path / "results_hparam_non_adaptive"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--hparam-search-split",
        "validation",
        *_save_args(out),
    ]
    with pytest.raises(ValueError, match="requires --chunker adaptive"):
        main(argv)


def test_pipeline_hparam_search_rejects_lambda_keys_when_disabled(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_lambda_disabled"
    _build_tiny_eraser_three(eraser_root)
    hparam_space = tmp_path / "hparam_space_lambda_disabled.json"
    hparam_space.write_text(
        json.dumps(
            {
                "parameters": {
                    "min_effective_chunks": [4],
                    "lambda1": [0.5, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "results_hparam_lambda_disabled"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "random",
        "--hparam-random-trials",
        "1",
        "--hparam-space-file",
        str(hparam_space),
        *_save_args(out),
    ]
    with pytest.raises(ValueError, match="--hparam-enable-lambda-search"):
        main(argv)


def test_pipeline_hparam_search_applies_max_samples_after_disjoint(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser_four"
    _build_tiny_eraser_four(eraser_root)
    out = tmp_path / "results_hparam_max_samples"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "adaptive",
        "--hparam-search-split",
        "validation",
        "--hparam-tune-size",
        "2",
        "--hparam-search-method",
        "random",
        "--hparam-random-trials",
        "1",
        "--max-samples",
        "1",
        *_save_args(out),
        "--run-eval",
    ]
    main(argv)
    run_cfg_path = next(out.glob("**/run_config.json"))
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    summary_path = Path(str(run_cfg.get("hparam_search_summary_path")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary.get("eval_count_after_disjoint_and_filters", 0)) == 1


def test_pipeline_eval_granularity_word_override(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out = tmp_path / "results"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--search",
        "greedy",
        "--k",
        "2",
        *_save_args(out),
        "--run-eval",
        "--eval-granularity",
        "word",
    ]
    main(argv)

    run_cfg_path = next(out.glob("**/run_config.json"))
    eval_cfg_path = next(out.glob("**/eval_config.json"))
    report_path = next(out.glob("**/eval_report.json"))

    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    eval_cfg = json.loads(eval_cfg_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert run_cfg["eval_granularity"] == "word"
    assert eval_cfg["eval_granularity"] == "word"
    assert report["metric_settings"]["perturbation_unit"] == "word"


def test_pipeline_deterministic_mode_is_recorded(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out = tmp_path / "results"
    argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--search",
        "greedy",
        "--k",
        "2",
        *_save_args(out),
        "--run-eval",
        "--deterministic",
    ]
    main(argv)

    run_cfg_path = next(out.glob("**/run_config.json"))
    eval_cfg_path = next(out.glob("**/eval_config.json"))
    report_path = next(out.glob("**/eval_report.json"))

    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    eval_cfg = json.loads(eval_cfg_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert run_cfg["deterministic"] is True
    assert eval_cfg["deterministic"] is True
    assert report["provenance"]["environment"]["deterministic"]["enabled"] is True
    assert run_cfg["provenance"]["environment"]["deterministic"]["enabled"] is True


def test_deterministic_mode_keeps_sample_outputs_reproducible(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out_a = tmp_path / "results_a"
    out_b = tmp_path / "results_b"
    common_argv = [
        "--dataset",
        "eraser_movie_reviews",
        "--split",
        "validation",
        "--eraser-root",
        str(eraser_root),
        "--mock-backbone",
        "--chunker",
        "sentence",
        "--search",
        "greedy",
        "--k",
        "2",
        "--deterministic",
    ]
    main([*common_argv, *_save_args(out_a)])
    main([*common_argv, *_save_args(out_b)])

    payloads_a = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in out_a.glob("**/samples/*.json")
    }
    payloads_b = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in out_b.glob("**/samples/*.json")
    }
    assert set(payloads_a.keys()) == set(payloads_b.keys())

    for sid in payloads_a:
        left = payloads_a[sid]
        right = payloads_b[sid]
        assert left["selected_chunk_ids"] == right["selected_chunk_ids"]
        assert left["chunk_ranking"] == right["chunk_ranking"]
        assert [x["total_score"] for x in left["trace"]] == [x["total_score"] for x in right["trace"]]
