from typing import List

import torch

from config.config import ExpArgs, MetricsMetaData
from config.constants import (LABEL_PROMPT_ATTENTION_MASK, TASK_PROMPT_ATTENTION_MASK,
                              LABEL_PROMPT_INPUT_IDS, TASK_PROMPT_INPUT_IDS, EXPLAINED_INPUT_IDS_NAME,
                              EXPLAINED_ATTENTION_MASK_NAME)
from config.types_enums import DirectionTypes
from evaluations.evaluations import evaluate_tokens_attributions
from models.aml_model import AmlModel
from utils.dataclasses import StepOutput
from utils.dataclasses.evaluations import DataForEvaluationInputs, DataForEvaluation


class AmlModelFineTune(AmlModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.MAX_EXAMPLES_TO_PRINT = 5
        self.item_idx = ""
        self.training_step_outputs: List[StepOutput] = []
        self.val_step_outputs: List[StepOutput] = []
        self.best_metric_result, self.best_item, self.best_tokens_attr = None, None, None

    def set_index(self, item_idx: str):
        self.item_idx = item_idx

    def training_step(self, batch, batch_idx):
        self.explained_model.eval()
        if ExpArgs.is_interpreter_model_train_mode_in_fine_tune:
            self.interpreter_model.train()
        else:
            self.interpreter_model.eval()
        output = self.forward(explained_model_inputs = batch)

        results = StepOutput(loss = output.loss_output.loss, prediction_loss = output.loss_output.prediction_loss,
                             prediction_loss_weighted = output.loss_output.prediction_loss_weighted,
                             regularization_loss = output.loss_output.regularization_loss,
                             explained_model_predicted_class = output.explained_model_predicted_class,
                             explained_model_logits = output.explained_model_predicted_logits,
                             regularization_weighted = output.loss_output.regularization_loss_weighted,
                             tokens_attr = output.tokens_attr, input = batch)
        self.training_step_outputs.append(results)
        return dict(loss = results.loss)

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.explained_model.eval()
            self.interpreter_model.eval()
            with torch.no_grad():
                output = self.forward(explained_model_inputs = batch)

            results = StepOutput(loss = output.loss_output.loss, prediction_loss = output.loss_output.prediction_loss_weighted,
                                 prediction_loss_weighted = output.loss_output.prediction_loss_weighted,
                                 regularization_loss = output.loss_output.regularization_loss,
                                 explained_model_predicted_class = output.explained_model_predicted_class,
                                 explained_model_logits = output.explained_model_predicted_logits,
                                 regularization_weighted = output.loss_output.regularization_loss_weighted,
                                 tokens_attr = output.tokens_attr, input = batch)
            self.val_step_outputs.append(results)
            return dict(loss = results.loss)

    def on_train_epoch_end(self):
        if not ExpArgs.is_inference:
            loss = torch.mean(torch.stack([output.loss for output in self.training_step_outputs]))
            if self.current_epoch >= ExpArgs.start_epoch_to_evaluate:
                self.run_perturbation_test(self.training_step_outputs)
            self.training_step_outputs.clear()
            return dict(loss = loss)

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            loss = torch.mean(torch.stack([output.loss for output in self.val_step_outputs]))
            if self.current_epoch >= ExpArgs.start_epoch_to_evaluate:
                self.run_perturbation_test(self.val_step_outputs)
            self.val_step_outputs.clear()
            return dict(loss = loss)

    def run_perturbation_test(self, outputs: List[StepOutput]):
        if not ExpArgs.is_inference:
            # fine tune run on one time only each time
            if len(outputs[0].tokens_attr) != 1:
                raise ValueError(f"val use batch_size 1 only!")
            step_output = outputs[0]
            item_data = DataForEvaluation(tokens_attr = step_output.tokens_attr[0].detach().squeeze(),
                                          explained_model_predicted_class = step_output.explained_model_predicted_class.squeeze(),
                                          explained_model_predicted_logits = step_output.explained_model_logits.squeeze(),
                                          input = DataForEvaluationInputs(  #
                                              input_ids = step_output.input[EXPLAINED_INPUT_IDS_NAME],  #
                                              attention_mask = step_output.input[EXPLAINED_ATTENTION_MASK_NAME],  #
                                              task_prompt_input_ids = step_output.input[TASK_PROMPT_INPUT_IDS],  #
                                              label_prompt_input_ids = step_output.input[LABEL_PROMPT_INPUT_IDS],  #
                                              task_prompt_attention_mask = step_output.input[
                                                  TASK_PROMPT_ATTENTION_MASK],
                                              label_prompt_attention_mask = step_output.input[
                                                  LABEL_PROMPT_ATTENTION_MASK]))

            evaluation_result, evaluation_item = evaluate_tokens_attributions(model = self.explained_model,
                                                                              explained_tokenizer = self.explained_tokenizer,
                                                                              ref_token_id = self.ref_token_id,
                                                                              data = item_data, step = self.global_step,
                                                                              epoch = self.current_epoch,
                                                                              item_index = self.item_idx,
                                                                              experiment_path = self.experiment_path)

            metric_results_dict = {ExpArgs.eval_metric: evaluation_result}
            self.set_best_results(evaluation_result, evaluation_item,
                                  best_tokens_attr = step_output.tokens_attr[0].detach().squeeze())
            new_logs = {f"Val_metric/{k}": v for k, v in metric_results_dict.items()}
            self.log_dict(new_logs, on_step = False, on_epoch = True)

    def set_best_results(self, metric_result, item, best_tokens_attr):
        if self.best_metric_result is None:
            self.best_metric_result, self.best_item, self.best_tokens_attr = metric_result, item, best_tokens_attr
        elif (self.best_metric_result < metric_result) and (
                MetricsMetaData.directions[ExpArgs.eval_metric] == DirectionTypes.MAX.value):
            self.best_metric_result, self.best_item, self.best_tokens_attr = metric_result, item, best_tokens_attr
        elif (self.best_metric_result > metric_result) and (
                MetricsMetaData.directions[ExpArgs.eval_metric] == DirectionTypes.MIN.value):
            self.best_metric_result, self.best_item, self.best_tokens_attr = metric_result, item, best_tokens_attr
