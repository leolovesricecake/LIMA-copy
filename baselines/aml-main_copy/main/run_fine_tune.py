import copy
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs
from config.constants import INPUT_TXT
from config.types_enums import ValidationType
from main.data_module import DataModule
from main.explained_model_utils import init_exp, set_hp, save_running_time
from models.aml_model_fine_tune import \
    AmlModelFineTune
from models.train_models_utils import (load_interpreter_model, init_trainable_embeddings,
                                       load_trainable_embeddings,
                                       get_warmup_steps_and_total_training_steps, get_explained_ref_token_name,
                                       run_trainer)
from utils.utils_functions import is_model_encoder_only


def set_config():
    ExpArgs.is_save_model = False
    ExpArgs.is_save_results = True
    ExpArgs.is_save_support_results = True


class FineTune:

    def __init__(self, hp: dict, experiment_name: str, explained_model):
        init_exp()
        set_config()
        set_hp(hp)

        ExpArgs.lr = ExpArgs.task.default_lr
        if not is_model_encoder_only(ExpArgs.explained_model_backbone):
            ExpArgs.lr = ExpArgs.task.llm_lr

        self.explained_model = explained_model
        self.experiment_name = experiment_name
        self.pretrain_path = f"{ExpArgs.default_root_dir}/FINE_TUNE"
        self.trainable_embeddings, self.label_embedding_index = init_trainable_embeddings()
        load_trainable_embeddings(self.trainable_embeddings)

    def run(self):

        begin = time.time()
        ExpArgs.scheduler_type = ExpArgs.fine_tune_scheduler_type

        interpreter_model = load_interpreter_model()

        data_module = DataModule(train_sample = ExpArgs.task.train_sample,
                                 test_sample = ExpArgs.task.test_sample, val_type = ValidationType.TEST)

        ref_token_id = get_explained_ref_token_name(data_module.explained_tokenizer)

        warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
            n_epochs = ExpArgs.num_epochs_for_fine_tune, train_samples_length = 1, batch_size = ExpArgs.eval_batch_size,
            warmup_ratio = ExpArgs.warmup_ratio, accumulate_grad_batches = ExpArgs.accumulate_grad_batches)

        fine_tuned_results_path = str(Path(self.pretrain_path, "RESULTS_DF", self.experiment_name))
        os.makedirs(fine_tuned_results_path, exist_ok = True)
        tb_logger = TensorBoardLogger(Path(self.pretrain_path, "TB_LOGS", self.experiment_name))

        device = "gpu" if torch.cuda.is_available() else "cpu"

        for idx, item in enumerate(data_module.val_dataset):
            # print("^"*100)
            # print(f"idx: {idx}. item: {item}")
            # print("^"*100)
            current_model = AmlModelFineTune(
                explained_model = self.explained_model,
                interpreter_model = copy.deepcopy(interpreter_model),
                explained_tokenizer = data_module.explained_tokenizer,
                interpreter_tokenizer = data_module.interpreter_tokenizer, total_training_steps = total_training_steps,
                experiment_path = fine_tuned_results_path, checkpoints_path = "",
                warmup_steps = warmup_steps, trainable_embeddings = copy.deepcopy(self.trainable_embeddings),
                label_embedding_index = self.label_embedding_index, ref_token_id = ref_token_id)

            item_module = DataModule(data = item, explained_tokenizer = data_module.explained_tokenizer,
                                     interpreter_tokenizer = data_module.interpreter_tokenizer,
                                     task_prompt_input_ids = data_module.task_prompt_input_ids,
                                     label_prompt_input_ids = data_module.label_prompt_input_ids,
                                     task_prompt_attention_mask = data_module.task_prompt_attention_mask,
                                     label_prompt_attention_mask = data_module.label_prompt_attention_mask,
                                     val_type = ValidationType.TEST)

            item_id = f"{idx}_{item['idx'].item()}" if "idx" in item else f"{idx}_{item['id'].item()}"
            current_model.set_index(item_idx = f"{idx}_{item_id}")
            trainer = pl.Trainer(accelerator = device, max_epochs = ExpArgs.num_epochs_for_fine_tune, logger = tb_logger,
                                 num_sanity_val_steps = -1, enable_progress_bar = False,
                                 #
                                 # gradient_clip_val = 1.0,
                                 #
                                 enable_checkpointing = False)
            run_trainer(trainer, model = current_model, data_module = item_module,
                        explained_model = self.explained_model)

            if ExpArgs.is_save_results:
                save_to = Path(fine_tuned_results_path, "results.csv")
                evaluation_item = current_model.best_item
                evaluation_item[INPUT_TXT] = item[INPUT_TXT]
                with open(save_to, 'a', newline = '', encoding = 'utf-8-sig') as f:
                    evaluation_item.to_csv(f, header = f.tell() == 0, index = False)

            del current_model

        end = time.time()

        save_running_time(end, begin, self.experiment_name, file_type = "FineTune")
