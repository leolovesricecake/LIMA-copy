from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, MetricsMetaData
from config.types_enums import EvalMetric
from evaluations.metrics.metrics_utils import MetricsFunctions
from utils.dataclasses.evaluations import DataForEvaluation
from utils.dataclasses.metric_results import MetricResults
from utils.utils_functions import get_device, get_model_special_tokens


class Metrics:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation, item_index: str,
                 step: int, epoch: int, experiment_path: str):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.device = get_device()
        self.special_tokens = torch.tensor(
            get_model_special_tokens(ExpArgs.explained_model_backbone, self.explained_tokenizer)).to(self.device)
        self.data: DataForEvaluation = data
        self.item_index = item_index
        self.step = step
        self.epoch = epoch
        self.experiment_path = experiment_path
        self.metric_functions = MetricsFunctions(model, explained_tokenizer, ref_token_id, self.special_tokens)
        self.pretu_steps = MetricsMetaData.top_k[ExpArgs.eval_metric]
        self.output_path = Path(experiment_path, "support_results_df.csv")

    def run_perturbation_test(self):
        results_steps: List[float] = []
        for idx, k in enumerate(self.pretu_steps):
            self.data.k = k
            step_metric_result = self.run_metric(self.data)
            results_steps.append(step_metric_result)

        # AOPC or one step only
        if ExpArgs.eval_metric in [EvalMetric.AOPC_SUFFICIENCY.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            metric_res = sum(results_steps) / (len(self.pretu_steps) + 1)
        elif ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                     EvalMetric.EVAL_LOG_ODDS.value]:
            if len(results_steps) > 1:
                raise ValueError("has more than 1 value without AOPC calc")
            metric_res = results_steps[0]
        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected - run_perturbation_test")
        # results_mean = torch.tensor(results).mean()
        results_item = self.transform_results(metric_res)
        self.save_results(results_item)
        return metric_res, results_item

    def run_metric(self, item_args):
        if ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.AOPC_SUFFICIENCY.value]:
            return self.metric_functions.sufficiency(item_args)

        elif ExpArgs.eval_metric in [EvalMetric.COMPREHENSIVENESS.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            return self.metric_functions.comprehensiveness(item_args)

        elif ExpArgs.eval_metric == EvalMetric.EVAL_LOG_ODDS.value:
            return self.metric_functions.log_odds(item_args)
        else:
            raise ValueError("unsupported metric_functions selected")

    def save_results(self, results_item):
        if ExpArgs.is_save_support_results:
            with open(self.output_path, 'a', newline = '', encoding = 'utf-8-sig') as f:
                results_item.to_csv(f, header = f.tell() == 0, index = False)

    def transform_results(self, metric_result):
        res = MetricResults(item_index = self.item_index, task = ExpArgs.task.name, evaluation_metric = ExpArgs.eval_metric,
                            explained_model_backbone = ExpArgs.explained_model_backbone,
                            interpreter_model_backbone = ExpArgs.interpreter_model_backbone,
                            metric_result = metric_result,
                            metric_result_str = "{:.6f}".format(metric_result),
                            metric_steps_result = None, steps_k = self.pretu_steps,
                            explained_model_predicted_class = self.data.explained_model_predicted_class.squeeze().item(),
                            step = self.step, epoch = self.epoch, token_evaluation_option = ExpArgs.eval_tokens)
        return pd.DataFrame([res])
