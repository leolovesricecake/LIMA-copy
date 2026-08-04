from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "baselines" / "inseq" / "run_inseq_llm_baselines.py"
    spec = importlib.util.spec_from_file_location("inseq_runner_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def test_supported_methods_include_lime_without_changing_default() -> None:
    assert "lime" in RUNNER.SUPPORTED_METHODS
    assert "lime" not in RUNNER.DEFAULT_METHODS


def test_parse_methods_accepts_lime() -> None:
    assert RUNNER._parse_methods("lime") == ("lime",)


def test_parse_methods_rejects_unknown_method() -> None:
    try:
        RUNNER._parse_methods("not_a_method")
    except ValueError as exc:
        assert "Unsupported Inseq methods" in str(exc)
    else:  # pragma: no cover - defensive assertion for direct execution without pytest
        raise AssertionError("unknown methods should not be accepted by the Inseq runner")


def test_lime_attr_kwargs_forward_n_samples_to_inseq() -> None:
    args = argparse.Namespace(
        attributed_fn="probability",
        no_logprob=False,
        n_steps=7,
        internal_batch_size=2,
        n_samples=11,
        attr_pos_start=None,
        attr_pos_end=None,
    )
    kwargs = RUNNER._method_attr_kwargs(args, "lime")
    assert kwargs["n_samples"] == 11
    assert kwargs["attributed_fn"] == "probability"
    assert kwargs["attributed_fn_args"] == {"logprob": True}


def test_method_config_records_dataset_method_and_schema_axes() -> None:
    args = RUNNER.build_parser().parse_args(
        ["--model-path", "tiny", "--dataset", "sst2", "--methods", "lime"]
    )
    bundle = SimpleNamespace(
        dataset_name="sst2",
        split="validation",
        verbalizers=["negative", "positive"],
    )
    payload = RUNNER._method_config(args, bundle, "lime")
    assert payload["dataset"]["name"] == "sst2"
    assert payload["method"] == "inseq_lime"
    assert payload["chunker"] == "token"
    assert payload["eval_granularity"] == "token"
    assert payload["prompt"]["version"] == "task_classification_v1"
    assert "sentiment" in payload["prompt"]["task_description"]


def test_method_output_root_uses_schema_v2_run_id() -> None:
    args = RUNNER.build_parser().parse_args(
        ["--model-path", "tiny", "--dataset", "sst2", "--methods", "lime"]
    )
    bundle = SimpleNamespace(
        dataset_name="sst2",
        split="validation",
        verbalizers=["negative", "positive"],
    )
    path = RUNNER._method_output_root(args, bundle, "lime")
    assert path.parts[-4:-1] == ("sst2", "tiny", "inseq_lime")
    assert path.name.startswith("b32-o1-s42-")


def test_lime_perturbation_positions_exclude_fixed_prompt_tokens() -> None:
    """Keep instructions and candidate labels fixed in Inseq LIME samples."""

    model = SimpleNamespace(
        _lima_text_perturbation_contract={
            "prompt_token_ids": [10, 11, 12, 13],
            "text_token_positions": [1, 2],
        }
    )
    positions = RUNNER._lime_perturbable_positions(
        model,
        [99, 10, 11, 12, 13, 77],
    )
    assert positions == {2, 3}
