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


def _sample_payloads(method_root: Path) -> dict[str, dict]:
    payloads = {}
    for path in sorted((method_root / "samples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[str(payload["sample_id"])] = payload
    return payloads


def _method_dir(root: Path, method: str) -> Path:
    matches = list(root.glob(f"**/*_method-{method}"))
    assert len(matches) == 1
    return matches[0]


def test_method_level_reports_and_shared_chunk_partition(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out = tmp_path / "results"
    base_argv = [
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
        "--seed",
        "42",
    ]

    for method in ("ours", "random", "gradient"):
        main([*base_argv, "--explain-method", method])

    method_payloads: dict[str, dict[str, dict]] = {}
    for method in ("ours", "random", "gradient"):
        mdir = _method_dir(out, method)
        report = json.loads((mdir / "eval_report.json").read_text(encoding="utf-8"))
        assert report["report_method"] == method

        payloads = _sample_payloads(mdir)
        assert len(payloads) == 2
        for payload in payloads.values():
            assert payload["explain_method"] == method
            assert len(payload["chunk_ranking"]) == len(payload["chunks"])
            assert len(payload["chunk_scores"]) == len(payload["chunks"])
            assert payload["selected_chunk_ids"] == payload["chunk_ranking"][:2]
        method_payloads[method] = payloads

    # Shared chunk partition must be identical across methods.
    for sample_id in method_payloads["ours"].keys():
        chunks_ours = method_payloads["ours"][sample_id]["chunks"]
        chunks_random = method_payloads["random"][sample_id]["chunks"]
        chunks_gradient = method_payloads["gradient"][sample_id]["chunks"]
        assert chunks_ours == chunks_random == chunks_gradient


def test_random_method_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    eraser_root = tmp_path / "eraser"
    _build_tiny_eraser(eraser_root)

    out_a = tmp_path / "results_a"
    out_b = tmp_path / "results_b"
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
        "--run-eval",
        "--explain-method",
        "random",
        "--seed",
        "42",
    ]

    main([*argv, "--output-dir", str(out_a)])
    main([*argv, "--output-dir", str(out_b)])

    payloads_a = _sample_payloads(_method_dir(out_a, "random"))
    payloads_b = _sample_payloads(_method_dir(out_b, "random"))
    assert set(payloads_a.keys()) == set(payloads_b.keys())
    for sid in payloads_a:
        assert payloads_a[sid]["chunk_ranking"] == payloads_b[sid]["chunk_ranking"]
