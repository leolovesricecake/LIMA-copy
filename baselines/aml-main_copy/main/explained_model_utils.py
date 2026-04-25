import gc
import os
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch import Tensor, nn

from config.config import ExpArgs, LossCoefficients
from config.types_enums import InverseLossTypes

ce_loss = nn.CrossEntropyLoss(reduction = "mean")


def l1_loss(tokens_attr) -> Tensor:
    return torch.abs(tokens_attr).mean()


def calculate_prediction_loss(output, target):
    # soft cross entropy
    # target_class_to_compare = torch.argmax(target, dim=1)
    return ce_loss(output, target)


def calculate_inverse_loss(logits: Tensor, target: Tensor):
    if ExpArgs.inverse_token_attr_function == InverseLossTypes.NEGATIVE_PROB_LOSS.value:
        eps = 1e-30
        probabilities = F.softmax(logits, dim = -1)
        target_categories = torch.argmax(target, dim = -1)
        inverse_probabilities = 1 - probabilities
        inverse_probabilities = inverse_probabilities + eps
        return -torch.log(inverse_probabilities[range(len(target)), target_categories]).mean()
    else:
        raise ValueError("unsupported inverse_token_attr_function selected")


def encourage_token_attr_to_prior_loss(tokens_attr: Tensor, prior: int = 0):
    if prior == 0:
        target = torch.zeros_like(tokens_attr)
    elif prior == 1:
        target = torch.ones_like(tokens_attr)
    else:
        raise NotImplementedError
    bce_encourage_prior_loss = F.binary_cross_entropy(tokens_attr, target)
    return bce_encourage_prior_loss


def save_config_to_root_dir(conf_path_dir, experiment_name):
    os.makedirs(conf_path_dir, exist_ok = True)
    h_params = {key: val for key, val in vars(ExpArgs).items() if not key.startswith("__")}
    conf_df = pd.DataFrame([h_params])
    conf_df.to_pickle(f"{conf_path_dir}/{experiment_name}.pkl")


def save_checkpoint(model, tokenizer, path_dir):
    os.makedirs(path_dir, exist_ok = True)
    model.save_pretrained(path_dir)
    tokenizer.save_pretrained(path_dir)


def save_running_time(end, begin, experiment_name, file_type):
    running_times_conf = f"{ExpArgs.default_root_dir}/RUNNING_TIMES"
    os.makedirs(running_times_conf, exist_ok = True)
    exec_time = end - begin
    with open(f"{running_times_conf}/{experiment_name}_{file_type}.pkl", 'wb') as file:
        pickle.dump({"time": exec_time, "experiment_name": experiment_name, "eval_metric": ExpArgs.eval_metric,
                     "explained_model_backbone": ExpArgs.explained_model_backbone,
                     "interpreter_model_backbone": ExpArgs.interpreter_model_backbone, "run_type": ExpArgs.run_type}, file)


def init_exp():
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(ExpArgs.seed)


def set_hp(hp: dict):
    LossCoefficients.prediction_loss_weight = hp["prediction_loss_weight"]
    LossCoefficients.regularization_loss_weight = hp["regularization_loss_weight"]
    LossCoefficients.inverse_loss_weight = hp["inverse_loss_weight"]
