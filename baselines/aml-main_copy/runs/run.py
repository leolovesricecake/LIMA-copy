import sys



sys.path.append("../..")

from main.hp_search import HpSearch
from main.run_fine_tune import FineTune
from main.run_pre_train import PreTrain

from runs.runs_utils import get_task

import argparse

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, ModelBackboneTypes
from utils.utils_functions import get_current_time, is_model_encoder_only
from models.train_models_utils import load_explained_model
from main.run_infrence_pre_train import InferencePretrain

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('task', type = str, help = '')
parser.add_argument('explained_model_backbone', type = str, help = '')
parser.add_argument('interpreter_model_backbone', type = str, help = '')
parser.add_argument('metric', type = str, help = '')

args = parser.parse_args()

arg_task = args.task
arg_explained_model_backbone = args.explained_model_backbone
arg_interpreter_model_backbone = args.interpreter_model_backbone
arg_metric = args.metric

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone
ExpArgs.interpreter_model_backbone = arg_interpreter_model_backbone
ExpArgs.eval_metric = arg_metric

is_llm = not is_model_encoder_only(ExpArgs.explained_model_backbone)

if is_llm:
    ExpArgs.ref_token_name = RefTokenNameTypes.UNK.value
    ExpArgs.accumulate_grad_batches = 5
    ExpArgs.batch_size = 4

# START AML

print("*" * 20, arg_task, arg_explained_model_backbone, arg_interpreter_model_backbone, arg_metric, "*" * 20,
      flush = True)

time_str = get_current_time()
experiment_name_prefix = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{ExpArgs.interpreter_model_backbone}_{ExpArgs.eval_metric}"

# ------------------------------------------------

explained_model = load_explained_model()


# Run hyper params search
hp_experiment_name = f"HP_{experiment_name_prefix}_{time_str}"
hp = HpSearch(hp_experiment_name, explained_model = explained_model).run()

# Run pretrain
pre_train_experiment_name = f"PRETRAIN_{experiment_name_prefix}_{time_str}"
pretrain_model_path = PreTrain(hp, pre_train_experiment_name,
                               explained_model = explained_model).run()

ExpArgs.fine_tuned_interpreter_model_path = pretrain_model_path

inference_pretrain_experiment_name = f"INFERENCE_PRETRAIN_{experiment_name_prefix}_{time_str}"
InferencePretrain(hp, inference_pretrain_experiment_name, explained_model = explained_model).run()

# Run finetune
fine_tune_exp_name = f"FINE_TUNE_{experiment_name_prefix}_{time_str}"
FineTune(hp, fine_tune_exp_name, explained_model = explained_model).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_interpreter_model_backbone, arg_metric, "*" * 20)
