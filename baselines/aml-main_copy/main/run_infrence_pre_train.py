import copy
import os
import time
from pathlib import Path

import pandas as pd

from config.config import ExpArgs
from config.constants import INPUT_TXT
from config.types_enums import ValidationType
from evaluations.results_reporting import evaluate_all_metrics, normalize_model_name, save_all_metrics_report
from main.data_module import DataModule
from main.explained_model_utils import init_exp, set_hp, save_running_time
from models.aml_model_fine_tune import \
    AmlModelFineTune
from models.train_models_utils import (load_interpreter_model, init_trainable_embeddings, load_trainable_embeddings,
                                       get_explained_ref_token_name, resolve_explained_model_path)
from utils.utils_functions import get_device


def set_config():
    ExpArgs.is_save_model = False
    ExpArgs.is_save_results = True
    ExpArgs.is_save_support_results = True


class InferencePretrain:

    def __init__(self, hp: dict, experiment_name: str, explained_model):
        init_exp()
        set_config()
        set_hp(hp)
        self.hp = hp
        self.explained_model = explained_model
        self.experiment_name = experiment_name
        self.pretrain_path = f"{ExpArgs.default_root_dir}/INFERENCE_PRETRAIN"
        self.trainable_embeddings, self.label_embedding_index = init_trainable_embeddings()
        load_trainable_embeddings(self.trainable_embeddings)

    @staticmethod
    def _build_result_row_id(item, idx: int) -> str:
        for key in ("idx", "id"):
            if key in item:
                value = item[key]
                if hasattr(value, "item"):
                    value = value.item()
                return f"{idx}_{value}"
        return str(idx)

    @staticmethod
    def _resolve_reference_token_text(tokenizer, ref_token_id) -> str:
        if ref_token_id is None:
            return ""
        try:
            return str(tokenizer.convert_ids_to_tokens(int(ref_token_id)))
        except Exception:
            return str(ref_token_id)

    def run(self):

        begin = time.time()
        ExpArgs.scheduler_type = ExpArgs.fine_tune_scheduler_type

        interpreter_model = load_interpreter_model()

        data_module = DataModule(train_sample = ExpArgs.task.train_sample, test_sample = ExpArgs.task.test_sample,
                                 val_type = ValidationType.TEST)

        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        inference__results_path = str(Path(self.pretrain_path, "RESULTS_DF", self.experiment_name))
        os.makedirs(inference__results_path, exist_ok = True)
        primary_results, all_metrics_results, trajectory_points = [], [], []
        model_name = normalize_model_name(resolve_explained_model_path(ExpArgs.task))
        split_name = str(ExpArgs.task.dataset_test)

        aml_model = AmlModelFineTune(explained_model = self.explained_model,
                                     interpreter_model = interpreter_model,
                                     explained_tokenizer = data_module.explained_tokenizer,
                                     interpreter_tokenizer = data_module.interpreter_tokenizer,
                                     total_training_steps = 0, # No training
                                     experiment_path = inference__results_path,
                                     checkpoints_path = "", warmup_steps = 0,
                                     trainable_embeddings = self.trainable_embeddings,
                                     label_embedding_index = self.label_embedding_index,
                                     ref_token_id = ref_token_id)

        aml_model = aml_model.to(get_device())
        self.freeze_model(aml_model)

        for idx, item in enumerate(data_module.val_dataset):
            result_row_id = self._build_result_row_id(item, idx)
            item = data_module.collate_fn([item])
            # print(f"inf item: {item}")
            tokens_attribution, evaluation_item, duration, evaluation_data = aml_model.forwad_paml_inference(
                item, is_evaluate = True)
            primary_results.append(evaluation_item.copy())
            if ExpArgs.is_save_results:
                save_to = Path(inference__results_path, "results.csv")
                evaluation_item[INPUT_TXT] = item[INPUT_TXT]
                with open(save_to, 'a', newline = '', encoding = 'utf-8-sig') as f:
                    evaluation_item.to_csv(f, header = f.tell() == 0, index = False)

            metrics_frame, sample_trajectory_points = evaluate_all_metrics(
                model = self.explained_model,
                explained_tokenizer = aml_model.explained_tokenizer,
                ref_token_id = aml_model.ref_token_id,
                data = evaluation_data,
                experiment_path = inference__results_path,
                step = -1,
                epoch = -1,
                item_index = result_row_id,
                save_support_results = False,
                experiment_name = self.experiment_name,
                report_stage = "INFERENCE_PRETRAIN",
                input_text = item[INPUT_TXT],
                result_row_id = result_row_id,
                selection_metric = ExpArgs.target_eval_metric,
                selection_metric_result = float(evaluation_item["metric_result"].iloc[0]),
                selection_mode = "pretrained_interpreter_single_pass",
                return_trajectory = True,
                trajectory_context = dict(
                    run_id = self.experiment_name,
                    report_stage = "INFERENCE_PRETRAIN",
                    dataset = ExpArgs.task.name,
                    split = split_name,
                    model_name = model_name,
                    method_name = "aml",
                    sample_id = result_row_id))
            all_metrics_results.append(metrics_frame)
            trajectory_points.extend(sample_trajectory_points)

        end = time.time()
        log_odds_reference_token = self._resolve_reference_token_text(
            aml_model.explained_tokenizer,
            aml_model.ref_token_id)

        del aml_model

        if not primary_results:
            raise ValueError(
                f"No primary inference results were produced for task={ExpArgs.task.name} "
                f"during experiment={self.experiment_name}."
            )
        if not all_metrics_results:
            raise ValueError(
                f"No all-metrics inference results were produced for task={ExpArgs.task.name} "
                f"during experiment={self.experiment_name}."
            )

        save_all_metrics_report(all_metrics_results = pd.concat(all_metrics_results, ignore_index = True),
                                experiment_path = inference__results_path,
                                experiment_name = self.experiment_name,
                                report_stage = "INFERENCE_PRETRAIN",
                                primary_results = pd.concat(primary_results, ignore_index = True),
                                selected_hyperparameters = self.hp,
                                extra_metadata = dict(selection_mode = "pretrained_interpreter_single_pass"),
                                trajectory_points = trajectory_points,
                                dataset_name = ExpArgs.task.name,
                                split_name = split_name,
                                model_name = model_name,
                                log_odds_reference_token = log_odds_reference_token)

        save_running_time(end, begin, self.experiment_name, file_type = "InferencePretrain")

    def freeze_model(self, _model):
        for param in _model.trainable_embeddings.parameters():
            param.requires_grad = False
        for param in _model.explained_model.parameters():
            param.requires_grad = False
        for param in _model.interpreter_model.parameters():
            param.requires_grad = False
        for param in _model.parameters():
            param.requires_grad = False
