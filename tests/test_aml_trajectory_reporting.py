from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


AML_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "aml-main_copy"
if str(AML_ROOT) not in sys.path:
    sys.path.insert(0, str(AML_ROOT))


class _FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def convert_ids_to_tokens(self, token_id):
        return f"tok-{int(token_id)}"


class _FakeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids = None, attention_mask = None, inputs_embeds = None):
        effective = float(max(int(input_ids.shape[-1]) - 2, 0))
        logits = torch.tensor([[0.0, effective]], dtype = torch.float32, device = input_ids.device)
        return SimpleNamespace(logits = logits)


def test_aml_plain_aopc_counts_full_step() -> None:
    from config.config import ExpArgs
    from config.types_enums import EvalTokens, ModelBackboneTypes
    from evaluations.metrics.metrics_utils import MetricsFunctions
    from runs.runs_utils import get_task
    from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs

    ExpArgs.task = get_task("sst2")
    ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
    ExpArgs.eval_tokens = EvalTokens.NO_SPECIAL_TOKENS.value

    tokenizer = _FakeTokenizer()
    model = _FakeClassifier()
    special_tokens = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])
    metric_functions = MetricsFunctions(model, tokenizer, ref_token_id = tokenizer.pad_token_id, special_tokens = special_tokens)

    input_ids = torch.tensor([[101, 10, 11, 102]], dtype = torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype = torch.long)
    full_logits = model(input_ids = input_ids, attention_mask = attention_mask).logits.squeeze(0)
    evaluation_data = DataForEvaluation(
        tokens_attr = torch.tensor([0.0, 0.9, 0.8, 0.0], dtype = torch.float32),
        explained_model_predicted_class = torch.tensor(1),
        explained_model_predicted_logits = full_logits,
        input = DataForEvaluationInputs(
            input_ids = input_ids,
            attention_mask = attention_mask,
            task_prompt_input_ids = None,
            label_prompt_input_ids = None,
            task_prompt_attention_mask = None,
            label_prompt_attention_mask = None,
        ),
    )

    payload = metric_functions.deletion_trajectory(evaluation_data)

    assert [point["step_index"] for point in payload["points"]] == [0, 1, 2]
    assert payload["points"][0]["deleted_ids"] == []
    assert payload["points"][1]["deleted_ids"] == [1]
    assert payload["points"][2]["deleted_ids"] == [1, 2]

    p_full = math.exp(2.0) / (1.0 + math.exp(2.0))
    p_step_1 = math.exp(1.0) / (1.0 + math.exp(1.0))
    p_step_2 = 0.5
    expected_aopc = ((p_full - p_full) + (p_full - p_step_1) + (p_full - p_step_2)) / 3.0
    assert payload["aopc"] == pytest.approx(expected_aopc)


def test_aml_eval_report_and_trajectory_artifacts_are_written(tmp_path: Path) -> None:
    from config.config import ExpArgs
    from config.types_enums import EvalMetric, EvalTokens, ModelBackboneTypes
    from evaluations.results_reporting import save_all_metrics_report
    from runs.runs_utils import get_task

    ExpArgs.task = get_task("sst2")
    ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
    ExpArgs.interpreter_model_backbone = ModelBackboneTypes.BERT.value
    ExpArgs.eval_tokens = EvalTokens.NO_SPECIAL_TOKENS.value
    ExpArgs.eval_metric = EvalMetric.COMPREHENSIVENESS.value
    ExpArgs.target_eval_metric = EvalMetric.COMPREHENSIVENESS.value
    ExpArgs.default_root_dir = str(tmp_path / "out")

    rows = [
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.SUFFICIENCY.value,
            metric_result = 0.2,
            metric_result_str = "0.200000",
            metric_steps_result = [0.2],
            steps_k = [20],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.COMPREHENSIVENESS.value,
            metric_result = 0.4,
            metric_result_str = "0.400000",
            metric_steps_result = [0.4],
            steps_k = [20],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.EVAL_LOG_ODDS.value,
            metric_result = -0.1,
            metric_result_str = "-0.100000",
            metric_steps_result = [-0.1],
            steps_k = [20],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.AOPC.value,
            metric_result = 0.15,
            metric_result_str = "0.150000",
            metric_steps_result = [0.0, 0.1, 0.2, 0.3],
            steps_k = [0, 1, 2, 3],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.AOPC_SUFFICIENCY.value,
            metric_result = 0.03,
            metric_result_str = "0.030000",
            metric_steps_result = [0.0, 0.01, 0.02, 0.03, 0.04],
            steps_k = [1, 5, 10, 20, 50],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
        dict(
            result_row_id = "sample-0",
            item_index = "sample-0",
            epoch = -1,
            step = -1,
            explained_model_predicted_class = 1,
            evaluation_metric = EvalMetric.AOPC_COMPREHENSIVENESS.value,
            metric_result = 0.07,
            metric_result_str = "0.070000",
            metric_steps_result = [0.0, 0.02, 0.04, 0.08, 0.12],
            steps_k = [1, 5, 10, 20, 50],
            token_evaluation_option = ExpArgs.eval_tokens,
            report_stage = "INFERENCE_PRETRAIN",
        ),
    ]
    all_metrics_results = pd.DataFrame(rows)
    primary_results = pd.DataFrame([rows[1]])
    trajectory_points = [
        dict(
            source_family = "aml",
            run_id = "run-1",
            report_stage = "INFERENCE_PRETRAIN",
            dataset = "sst2",
            split = "validation",
            model_name = "bert-base-uncased-sst-2",
            method_name = "aml",
            sample_id = "sample-0",
            target_label_id = 1,
            target_label_text = "positive",
            step_index = 0,
            total_steps = 2,
            delete_count = 0,
            delete_fraction = 0.0,
            remaining_fraction = 1.0,
            target_probability = 0.9,
            prob_drop_from_full = 0.0,
            is_full_text_step = True,
            deleted_ids = [],
        ),
        dict(
            source_family = "aml",
            run_id = "run-1",
            report_stage = "INFERENCE_PRETRAIN",
            dataset = "sst2",
            split = "validation",
            model_name = "bert-base-uncased-sst-2",
            method_name = "aml",
            sample_id = "sample-0",
            target_label_id = 1,
            target_label_text = "positive",
            step_index = 1,
            total_steps = 2,
            delete_count = 1,
            delete_fraction = 0.5,
            remaining_fraction = 0.5,
            target_probability = 0.7,
            prob_drop_from_full = 0.2,
            is_full_text_step = False,
            deleted_ids = [3],
        ),
        dict(
            source_family = "aml",
            run_id = "run-1",
            report_stage = "INFERENCE_PRETRAIN",
            dataset = "sst2",
            split = "validation",
            model_name = "bert-base-uncased-sst-2",
            method_name = "aml",
            sample_id = "sample-0",
            target_label_id = 1,
            target_label_text = "positive",
            step_index = 2,
            total_steps = 2,
            delete_count = 2,
            delete_fraction = 1.0,
            remaining_fraction = 0.0,
            target_probability = 0.5,
            prob_drop_from_full = 0.4,
            is_full_text_step = False,
            deleted_ids = [3, 4],
        ),
    ]

    save_all_metrics_report(
        all_metrics_results = all_metrics_results,
        experiment_path = str(tmp_path / "aml-run"),
        experiment_name = "run-1",
        report_stage = "INFERENCE_PRETRAIN",
        primary_results = primary_results,
        selected_hyperparameters = {"lr": 1e-4},
        extra_metadata = {"selection_mode": "pretrained_interpreter_single_pass"},
        trajectory_points = trajectory_points,
        dataset_name = "sst2",
        split_name = "validation",
        model_name = "bert-base-uncased-sst-2",
        log_odds_reference_token = "[MASK]",
    )

    report_path = tmp_path / "aml-run" / "eval_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding = "utf-8"))

    assert report["metrics_primary"]["aopc"] == 0.15
    assert report["metrics_by_target"]["predicted"]["per_q"]["comprehensiveness"]["20"] == 0.08
    assert report["artifacts"]["trajectory_points_csv"] == "trajectory_points.csv"
    assert report["artifacts"]["trajectory_points_jsonl"] == "trajectory_points.jsonl"
    assert report["artifacts"]["trajectory_summary_csv"] == "trajectory_summary.csv"
    assert report["metric_settings"]["perturbation_target"] == "predicted"
    assert report["metric_settings"]["log_odds_reference_token"] == "[MASK]"

    points_csv = tmp_path / "aml-run" / "trajectory_points.csv"
    summary_csv = tmp_path / "aml-run" / "trajectory_summary.csv"
    assert points_csv.exists()
    assert summary_csv.exists()

    points_df = pd.read_csv(points_csv)
    summary_df = pd.read_csv(summary_csv)
    assert len(points_df) == 3
    assert summary_df["sample_count"].tolist() == [1, 1, 1]
