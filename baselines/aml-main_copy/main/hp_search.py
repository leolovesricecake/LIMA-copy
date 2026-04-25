import copy
import os
import pickle
import time
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs, MetricsMetaData, LossCoefficients
from config.types_enums import (DirectionTypes, ValidationType)
from main.data_module import DataModule
from main.explained_model_utils import init_exp, save_running_time
from models.aml_model import (AmlModel)
from models.train_models_utils import (load_interpreter_model, get_warmup_steps_and_total_training_steps,
                                       init_trainable_embeddings, get_explained_ref_token_name, run_trainer)
from utils.utils_functions import is_model_encoder_only


def set_config():
    ExpArgs.is_save_model = False
    ExpArgs.is_save_results = False
    ExpArgs.is_save_support_results = False


class HpSearch:

    def __init__(self, experiment_name: str, explained_model):
        init_exp()
        set_config()
        self.explained_model = explained_model
        self.interpreter_model = load_interpreter_model()
        self.trainable_embeddings, self.label_embedding_index = init_trainable_embeddings()
        self.experiment_name = experiment_name
        self.conf_path = f"{ExpArgs.default_root_dir}/CONFIG"
        self.monitor = f"Val_metric/{ExpArgs.eval_metric}"
        is_direction_max = MetricsMetaData.directions[ExpArgs.eval_metric] == DirectionTypes.MAX.value
        self.direction = "maximize" if is_direction_max else "minimize"
        self.data_module = DataModule(train_sample = ExpArgs.task.hp_search_train_sample,
                                      test_sample = ExpArgs.task.val_sample, val_type = ValidationType.VAL)

    def objective(self, trial):
        # try:
        LossCoefficients.prediction_loss_weight = trial.suggest_float('prediction_loss_weight', low = 0, high = 1)
        LossCoefficients.regularization_loss_weight = trial.suggest_float('regularization_loss_weight', low = 0,
                                                                          high = 1)
        LossCoefficients.inverse_loss_weight = trial.suggest_float('inverse_loss_weight', low = 0, high = 1)

        data_module = copy.deepcopy(self.data_module)

        warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
            n_epochs = ExpArgs.num_epochs_for_pre_train, train_samples_length = len(data_module.train_dataset),
            batch_size = ExpArgs.batch_size, warmup_ratio = ExpArgs.warmup_ratio,
            accumulate_grad_batches = ExpArgs.accumulate_grad_batches)

        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        model = AmlModel(explained_model = self.explained_model,
                         interpreter_model = copy.deepcopy(self.interpreter_model),
                         explained_tokenizer = data_module.explained_tokenizer,
                         interpreter_tokenizer = data_module.interpreter_tokenizer,
                         total_training_steps = total_training_steps,
                         experiment_path = "", checkpoints_path = "", warmup_steps = warmup_steps,
                         trainable_embeddings = copy.deepcopy(self.trainable_embeddings), label_embedding_index = self.label_embedding_index,
                         ref_token_id = ref_token_id)

        tb_logger = TensorBoardLogger(Path(self.conf_path, "TB_LOGS", self.experiment_name))
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(accelerator = device, max_epochs = ExpArgs.hp_search_max_epochs, logger = tb_logger, enable_progress_bar = False,
                             enable_model_summary = False, default_root_dir = self.conf_path, num_sanity_val_steps = 0,
                             val_check_interval = ExpArgs.val_check_interval,
                             accumulate_grad_batches = ExpArgs.accumulate_grad_batches,
                             #
                             # gradient_clip_val = 1.0,
                             #
                             callbacks = [ModelCheckpoint(save_top_k = 0, monitor = self.monitor),
                                          EarlyStopping(monitor = self.monitor, patience = 2)])

        run_trainer(trainer, model = model, data_module = data_module, explained_model = self.explained_model)

        return trainer.callback_metrics[self.monitor].item()


    def run(self):

        begin = time.time()

        ExpArgs.lr = ExpArgs.task.default_lr
        if not is_model_encoder_only(ExpArgs.explained_model_backbone):
            ExpArgs.lr = ExpArgs.task.llm_lr

        os.makedirs(self.conf_path, exist_ok = True)
        result_path = f"{self.conf_path}/OPTUNA_RESULTS"
        os.makedirs(result_path, exist_ok = True)

        exp_args_path = f"{self.conf_path}/EXPERIMENT_ARGUMENTS"
        os.makedirs(exp_args_path, exist_ok = True)

        study = optuna.create_study(direction = self.direction,
                                    sampler = optuna.samplers.TPESampler(seed = ExpArgs.seed))
        study.optimize(self.objective, n_trials = ExpArgs.task.hp_search_n_trials)

        best_trial = study.best_trial

        file_name = f'{self.experiment_name}.pkl'
        with open(f"{result_path}/{file_name}", 'wb') as file:
            pickle.dump({**{"_best_trial_value_": best_trial.value}, **best_trial.params,
                         **{"_eval_metric": ExpArgs.eval_metric,
                            "_interpreter_model_backbone": ExpArgs.interpreter_model_backbone,
                            "_explained_model_backbone": ExpArgs.explained_model_backbone, }}, file)

        end = time.time()

        save_running_time(end, begin, self.experiment_name, file_type = "HpSearch")

        return best_trial.params
