from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from lima_llm.pipeline import run as run_mod


def test_load_adaptive_overrides_inline_and_file(tmp_path: Path) -> None:
    inline = run_mod._load_adaptive_overrides('{"min_effective_chunks": 6}')
    assert isinstance(inline, dict)
    assert int(inline["min_effective_chunks"]) == 6

    p = tmp_path / "ov.json"
    p.write_text(json.dumps({"guard_mode": "soft_band"}), encoding="utf-8")
    from_file = run_mod._load_adaptive_overrides(str(p))
    assert isinstance(from_file, dict)
    assert str(from_file["guard_mode"]) == "soft_band"


def test_load_sample_ids_from_json_and_csv(tmp_path: Path) -> None:
    p_json = tmp_path / "ids.json"
    p_json.write_text(json.dumps(["a", "b", "b"]), encoding="utf-8")
    ids_json = run_mod._load_sample_ids(str(p_json))
    assert ids_json == {"a", "b"}

    p_txt = tmp_path / "ids.txt"
    p_txt.write_text("x,y\n#comment\nz\n", encoding="utf-8")
    ids_txt = run_mod._load_sample_ids(str(p_txt))
    assert ids_txt == {"x", "y", "z"}


def test_load_hparam_space_and_candidates(tmp_path: Path) -> None:
    p = tmp_path / "space.json"
    p.write_text(
        json.dumps(
            {
                "parameters": {
                    "short_max_words": [96, 120],
                    "min_effective_chunks": [4, 5],
                    "guard_mode": ["hard_cap", "soft_band"],
                }
            }
        ),
        encoding="utf-8",
    )
    raw_space = run_mod._load_hparam_space(str(p))
    adaptive_space, lambda_space = run_mod._split_hparam_space(
        raw_space=raw_space,
        enable_lambda_search=False,
    )
    assert sorted(adaptive_space.keys()) == ["guard_mode", "min_effective_chunks", "short_max_words"]
    assert lambda_space == {}

    grid = run_mod._build_candidates(
        method="grid",
        adaptive_space=adaptive_space,
        lambda_space=lambda_space,
        enable_lambda_search=False,
        random_trials=8,
        max_trials=32,
        seed=42,
    )
    # baseline + 2x2x2 grid
    assert len(grid) == 9
    random_rows = run_mod._build_candidates(
        method="random",
        adaptive_space=adaptive_space,
        lambda_space=lambda_space,
        enable_lambda_search=False,
        random_trials=2,
        max_trials=32,
        seed=42,
    )
    assert len(random_rows) == 3


def test_hparam_space_rejects_lambda_keys_when_disabled(tmp_path: Path) -> None:
    p = tmp_path / "space_lambda.json"
    p.write_text(
        json.dumps(
            {
                "parameters": {
                    "min_effective_chunks": [4],
                    "lambda1": [0.8, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    raw_space = run_mod._load_hparam_space(str(p))
    try:
        run_mod._split_hparam_space(raw_space=raw_space, enable_lambda_search=False)
    except ValueError as exc:
        assert "--hparam-enable-lambda-search" in str(exc)
    else:
        raise AssertionError("Expected lambda keys to be rejected when lambda search is disabled")


def test_hparam_space_accepts_lambda_keys_when_enabled_and_builds_candidates(tmp_path: Path) -> None:
    p = tmp_path / "space_lambda_enabled.json"
    p.write_text(
        json.dumps(
            {
                "parameters": {
                    "min_effective_chunks": [4, 5],
                    "lambda1": [0.8, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    raw_space = run_mod._load_hparam_space(str(p))
    adaptive_space, lambda_space = run_mod._split_hparam_space(
        raw_space=raw_space,
        enable_lambda_search=True,
    )
    assert sorted(adaptive_space.keys()) == ["min_effective_chunks"]
    assert sorted(lambda_space.keys()) == ["lambda1"]

    rows = run_mod._build_candidates(
        method="grid",
        adaptive_space=adaptive_space,
        lambda_space=lambda_space,
        enable_lambda_search=True,
        random_trials=0,
        max_trials=32,
        seed=42,
    )
    assert len(rows) == 5
    assert rows[0]["adaptive_params"] == {}
    assert rows[0]["lambda_params"] == {}
    # at least one candidate should carry lambda override
    assert any("lambda1" in dict(row.get("lambda_params") or {}) for row in rows[1:])


def test_hparam_candidate_budget_is_deterministic() -> None:
    adaptive_space = {
        "short_max_words": [96, 120, 144],
        "min_effective_chunks": [4, 5, 6],
    }
    rows_a = run_mod._build_candidates(
        method="grid",
        adaptive_space=adaptive_space,
        lambda_space={},
        enable_lambda_search=False,
        random_trials=0,
        max_trials=4,
        seed=42,
    )
    rows_b = run_mod._build_candidates(
        method="grid",
        adaptive_space=adaptive_space,
        lambda_space={},
        enable_lambda_search=False,
        random_trials=0,
        max_trials=4,
        seed=42,
    )
    assert len(rows_a) == 4
    assert rows_a == rows_b


def test_resolve_hparam_tune_size_prefers_new_flag() -> None:
    args = SimpleNamespace(
        hparam_tune_size=88,
        hparam_train_size=None,
        hparam_dev_size=None,
    )
    assert run_mod._resolve_hparam_tune_size(args) == 88


def test_resolve_hparam_tune_size_accepts_legacy_sum() -> None:
    args = SimpleNamespace(
        hparam_tune_size=999,
        hparam_train_size=60,
        hparam_dev_size=40,
    )
    assert run_mod._resolve_hparam_tune_size(args) == 100


def test_parser_rejects_balanced_v2_profile() -> None:
    parser = run_mod.build_parser()
    # argparse exits with code 2 for invalid choice.
    try:
        _ = parser.parse_args(
            [
                "--dataset",
                "sst2",
                "--adaptive-profile",
                "balanced_v2",
            ]
        )
    except SystemExit as exc:
        assert int(exc.code) == 2
    else:
        raise AssertionError("Expected parser to reject balanced_v2")
