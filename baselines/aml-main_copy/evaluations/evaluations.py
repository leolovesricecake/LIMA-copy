import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, EvalMetric
from evaluations.metrics.metrics import Metrics
from utils.dataclasses.evaluations import DataForEvaluation


def evaluate_tokens_attributions(model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation,
                                 experiment_path: str, step: int, epoch: int, item_index: str):
    if ExpArgs.eval_metric == EvalMetric.AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY.value:
        return combined_eval(combined_metric = EvalMetric.AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY.value,
            comprehensiveness_metric = EvalMetric.AOPC_COMPREHENSIVENESS.value,
            sufficiency_metric = EvalMetric.AOPC_SUFFICIENCY.value, model = model,
            explained_tokenizer = explained_tokenizer, ref_token_id = ref_token_id, data = data,
            experiment_path = experiment_path, step = step, epoch = epoch, item_index = item_index)
    elif ExpArgs.eval_metric == EvalMetric.COMPREHENSIVENESS_SUFFICIENCY.value:
        return combined_eval(combined_metric = EvalMetric.COMPREHENSIVENESS_SUFFICIENCY.value,
            comprehensiveness_metric = EvalMetric.COMPREHENSIVENESS.value,
            sufficiency_metric = EvalMetric.SUFFICIENCY.value, model = model, explained_tokenizer = explained_tokenizer,
            ref_token_id = ref_token_id, data = data, experiment_path = experiment_path, step = step, epoch = epoch,
            item_index = item_index)
    else:
        return evaluate_tokens_attr_handler(model = model, explained_tokenizer = explained_tokenizer,
                                            ref_token_id = ref_token_id, data = data, experiment_path = experiment_path,
                                            step = step, epoch = epoch, item_index = item_index)


def evaluate_tokens_attr_handler(model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation,
                                 experiment_path: str, step: int, epoch: int, item_index: str):
    with torch.no_grad():
        if ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                   EvalMetric.EVAL_LOG_ODDS.value, EvalMetric.AOPC_SUFFICIENCY.value,
                                   EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            eval_class = Metrics(model = model, explained_tokenizer = explained_tokenizer, ref_token_id = ref_token_id,
                                 data = data, experiment_path = experiment_path, item_index = item_index, step = step,
                                 epoch = epoch)
            return eval_class.run_perturbation_test()
        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected")


def combined_eval(combined_metric: str, comprehensiveness_metric: str, sufficiency_metric: str, model,
                  explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation, experiment_path: str, step: int,
                  epoch: int, item_index: str):
    ExpArgs.eval_metric = comprehensiveness_metric
    evaluation_result_comp, evaluation_item_comp = evaluate_tokens_attr_handler(model, explained_tokenizer,
                                                                                ref_token_id, data, experiment_path,
                                                                                step, epoch, item_index)

    ExpArgs.eval_metric = sufficiency_metric
    evaluation_result_suff, evaluation_item_suff = evaluate_tokens_attr_handler(model, explained_tokenizer,
                                                                                ref_token_id, data, experiment_path,
                                                                                step, epoch, item_index)

    item = evaluation_item_comp
    item["metric_result"] = evaluation_result_comp - evaluation_result_suff
    item["eval_metric"] = combined_metric
    ExpArgs.eval_metric = combined_metric
    return item["metric_result"].item(), item
