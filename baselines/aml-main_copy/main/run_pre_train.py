import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs
from config.types_enums import ValidationType
from main.data_module import DataModule
from main.explained_model_utils import init_exp, set_hp, save_running_time
from models.aml_model import (AmlModel)
from models.train_models_utils import (load_interpreter_model, get_warmup_steps_and_total_training_steps,
                                       init_trainable_embeddings, get_explained_ref_token_name, run_trainer)
from utils.utils_functions import is_model_encoder_only


def set_config():
    ExpArgs.is_save_model = True
    ExpArgs.is_save_results = True
    ExpArgs.is_save_support_results = True


class PreTrain:

    def __init__(self, hp: dict, experiment_name: str, explained_model):
        init_exp()
        set_config()
        set_hp(hp)

        ExpArgs.lr = ExpArgs.task.default_lr
        if not is_model_encoder_only(ExpArgs.explained_model_backbone):
            ExpArgs.lr = ExpArgs.task.llm_lr

        self.trainable_embeddings, self.label_embedding_index = init_trainable_embeddings()
        self.explained_model = explained_model

        self.experiment_name = experiment_name
        self.pretrain_path = f"{ExpArgs.default_root_dir}/PRE_TRAIN"

    def run(self):
        begin = time.time()

        os.makedirs(self.pretrain_path, exist_ok = True)

        interpreter_model = load_interpreter_model()

        data_module = DataModule(train_sample = ExpArgs.task.train_sample,
                                 test_sample = ExpArgs.task.val_sample, val_type = ValidationType.VAL)

        warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
            n_epochs = ExpArgs.num_epochs_for_pre_train, train_samples_length = len(data_module.train_dataset),
            batch_size = ExpArgs.batch_size, warmup_ratio = ExpArgs.warmup_ratio,
            accumulate_grad_batches = ExpArgs.accumulate_grad_batches)

        pretrain_results_path = Path(self.pretrain_path, "RESULTS_DF", self.experiment_name).__str__()
        os.makedirs(pretrain_results_path, exist_ok = True)
        checkpoints_path = Path(self.pretrain_path, "CHECKPOINTS", self.experiment_name).__str__()
        tb_logger = TensorBoardLogger(Path(self.pretrain_path, "TB_LOGS", self.experiment_name))

        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        model = AmlModel(explained_model = self.explained_model,
                         interpreter_model = interpreter_model, explained_tokenizer = data_module.explained_tokenizer,
                         interpreter_tokenizer = data_module.interpreter_tokenizer, total_training_steps = total_training_steps,
                         experiment_path = pretrain_results_path, checkpoints_path = checkpoints_path,
                         warmup_steps = warmup_steps, trainable_embeddings = self.trainable_embeddings, label_embedding_index = self.label_embedding_index,
                         ref_token_id = ref_token_id)


        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(accelerator = device, max_epochs = ExpArgs.num_epochs_for_pre_train, logger = tb_logger,
                             enable_progress_bar = False, num_sanity_val_steps = 0,
                             default_root_dir = ExpArgs.default_root_dir,
                             val_check_interval = ExpArgs.val_check_interval,
                             log_every_n_steps = ExpArgs.log_every_n_steps,
                             accumulate_grad_batches = ExpArgs.accumulate_grad_batches,
                             enable_checkpointing = ExpArgs.enable_checkpointing
                             )

        run_trainer(trainer, model = model, data_module = data_module,
                    explained_model = self.explained_model)

        end = time.time()

        save_running_time(end, begin, self.experiment_name, file_type = "PreTrain")

        return checkpoints_path
