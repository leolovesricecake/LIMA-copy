import copy
import os
import time
from pathlib import Path

import pandas as pd

from config.config import ExpArgs
from config.constants import INPUT_TXT
from config.types_enums import ValidationType
from evaluations.results_reporting import evaluate_all_metrics, save_all_metrics_report
from main.data_module import DataModule
from main.explained_model_utils import init_exp, set_hp, save_running_time
from models.aml_model_fine_tune import \
    AmlModelFineTune
from models.train_models_utils import (load_interpreter_model, init_trainable_embeddings, load_trainable_embeddings,
                                       get_explained_ref_token_name)
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

    def run(self):

        begin = time.time()
        ExpArgs.scheduler_type = ExpArgs.fine_tune_scheduler_type

        interpreter_model = load_interpreter_model()

        data_module = DataModule(train_sample = ExpArgs.task.train_sample, test_sample = ExpArgs.task.test_sample,
                                 val_type = ValidationType.TEST)

        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        inference__results_path = str(Path(self.pretrain_path, "RESULTS_DF", self.experiment_name))
        os.makedirs(inference__results_path, exist_ok = True)
        primary_results, all_metrics_results = [], []

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

            all_metrics_results.append(evaluate_all_metrics(
                model = self.explained_model,
                explained_tokenizer = aml_model.explained_tokenizer,
                ref_token_id = aml_model.ref_token_id,
                data = evaluation_data,
                experiment_path = inference__results_path,
                step = -1,
                epoch = -1,
                item_index = "",
                save_support_results = False,
                experiment_name = self.experiment_name,
                report_stage = "INFERENCE_PRETRAIN",
                input_text = item[INPUT_TXT],
                result_row_id = str(idx),
                selection_metric = ExpArgs.target_eval_metric,
                selection_metric_result = float(evaluation_item["metric_result"].iloc[0]),
                selection_mode = "pretrained_interpreter_single_pass"))

        end = time.time()

        del aml_model

        save_all_metrics_report(all_metrics_results = pd.concat(all_metrics_results, ignore_index = True),
                                experiment_path = inference__results_path,
                                experiment_name = self.experiment_name,
                                report_stage = "INFERENCE_PRETRAIN",
                                primary_results = pd.concat(primary_results, ignore_index = True),
                                selected_hyperparameters = self.hp,
                                extra_metadata = dict(selection_mode = "pretrained_interpreter_single_pass"))

        save_running_time(end, begin, self.experiment_name, file_type = "FineTune")

    def freeze_model(self, _model):
        for param in _model.trainable_embeddings.parameters():
            param.requires_grad = False
        for param in _model.explained_model.parameters():
            param.requires_grad = False
        for param in _model.interpreter_model.parameters():
            param.requires_grad = False
        for param in _model.parameters():
            param.requires_grad = False
