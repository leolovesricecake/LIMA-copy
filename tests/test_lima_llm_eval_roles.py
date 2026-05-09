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


def _single_path(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    assert len(matches) == 1
    return matches[0]


def _collect_keys(obj, prefix="") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            cur = f"{prefix}.{k}" if prefix else str(k)
            keys.add(cur)
            keys.update(_collect_keys(v, cur))
    elif isinstance(obj, list):
        keys.add(prefix + "[]")
    return keys


def _assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    assert abs(float(a) - float(b)) <= tol


def test_eval_roles_reports_are_independent_and_consistent(tmp_path: Path) -> None:
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
        "--eval-random-trials",
        "3",
    ]

    main([*base_argv, "--eval-role", "full"])
    main([*base_argv, "--eval-role", "ours"])
    main([*base_argv, "--eval-role", "random"])
    main([*base_argv, "--eval-role", "gradient"])

    full_path = _single_path(out, "**/eval_report.json")
    ours_path = _single_path(out, "**/eval_report.ours.json")
    random_path = _single_path(out, "**/eval_report.random.json")
    gradient_path = _single_path(out, "**/eval_report.gradient.json")

    full = json.loads(full_path.read_text(encoding="utf-8"))
    ours = json.loads(ours_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))

    assert ours["report_role"] == "ours"
    assert random["report_role"] == "random"
    assert gradient["report_role"] == "gradient"

    ours_keys = _collect_keys(ours)
    random_keys = _collect_keys(random)
    gradient_keys = _collect_keys(gradient)
    assert ours_keys == random_keys == gradient_keys

    for mode in ("gold", "predicted"):
        full_ours = full["metrics_by_target"][mode]["metrics_primary"]
        split_ours = ours["metrics_by_target"][mode]["metrics_primary"]
        _assert_close(full_ours["comprehensiveness"], split_ours["comprehensiveness"])
        _assert_close(full_ours["sufficiency"], split_ours["sufficiency"])

        full_random = full["metrics_by_target"][mode]["baselines"]["random"]
        split_random = random["metrics_by_target"][mode]["metrics_primary"]
        _assert_close(full_random["comprehensiveness"], split_random["comprehensiveness"])
        _assert_close(full_random["sufficiency"], split_random["sufficiency"])

        full_grad = full["metrics_by_target"][mode]["baselines"]["gradient"]
        split_grad = gradient["metrics_by_target"][mode]["metrics_primary"]
        _assert_close(full_grad["comprehensiveness"], split_grad["comprehensiveness"])
        _assert_close(full_grad["sufficiency"], split_grad["sufficiency"])
