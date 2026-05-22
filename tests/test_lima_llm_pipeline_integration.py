import json
from pathlib import Path

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
        "--output-dir",
        str(out),
        "--run-eval",
    ]
    main(argv)

    sample_jsons = list(out.glob("**/samples/*.json"))
    assert len(sample_jsons) == 2
    sample_payload = json.loads(sample_jsons[0].read_text(encoding="utf-8"))
    assert "explain_timing_breakdown" in sample_payload.get("metadata", {})
    assert "objective_cache_stats" in sample_payload.get("metadata", {})
    assert "chunk_diagnostics" in sample_payload.get("metadata", {})
    chunk_diag = sample_payload["metadata"]["chunk_diagnostics"]
    assert "chunk_strategy" in chunk_diag
    assert "chunk_count" in chunk_diag
    assert "singleton_orphan_punctuation_chunks" in chunk_diag
    assert "leading_close_punct_chunks" in chunk_diag
    assert "abbreviation_singleton_chunks" in chunk_diag
    assert "cross_newline_boundary_chunks" in chunk_diag
    assert "fallback_applied" in chunk_diag
    assert "fallback_reason" in chunk_diag

    summary = list(out.glob("**/summary.csv"))
    report = list(out.glob("**/eval_report.json"))
    assert summary
    assert report

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
    assert eval_report["prefetch_stats"]["batch_fallback_count"] == 0

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
        "--output-dir",
        str(out),
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
        "--output-dir",
        str(out),
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
        "--output-dir",
        str(out),
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
    main([*common_argv, "--output-dir", str(out_a)])
    main([*common_argv, "--output-dir", str(out_b)])

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
