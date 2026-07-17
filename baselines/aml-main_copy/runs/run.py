import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes
from main.hp_search import HpSearch
from main.run_fine_tune import FineTune
from main.run_infrence_pre_train import InferencePretrain
from main.run_pre_train import PreTrain
from models.train_models_utils import load_explained_model
from runs.runs_utils import get_task
from utils.utils_functions import build_path_run_tag, get_current_time, is_model_encoder_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "AML full training pipeline")
    parser.add_argument("task", type = str, help = "")
    parser.add_argument("explained_model_backbone", type = str, help = "")
    parser.add_argument("interpreter_model_backbone", type = str, help = "")
    parser.add_argument("metric", type = str, help = "")
    parser.add_argument("--explained_model_path", type = str, default = None, help = "")
    parser.add_argument("--eraser-root", type = str, default = None, help = "")
    parser.add_argument("--sst2-source", type = str, default = None, help = "")
    parser.add_argument("--dataset-cache-dir", type = str, default = None, help = "")
    return parser


def main(argv = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    ExpArgs.requested_task_name = args.task
    ExpArgs.task = get_task(args.task)
    ExpArgs.explained_model_backbone = args.explained_model_backbone
    ExpArgs.explained_model_path = args.explained_model_path
    ExpArgs.interpreter_model_backbone = args.interpreter_model_backbone
    ExpArgs.eval_metric = args.metric
    ExpArgs.target_eval_metric = args.metric
    ExpArgs.eraser_root = args.eraser_root
    ExpArgs.sst2_source = args.sst2_source
    ExpArgs.dataset_cache_dir = args.dataset_cache_dir

    is_llm = not is_model_encoder_only(ExpArgs.explained_model_backbone)
    if is_llm:
        ExpArgs.ref_token_name = RefTokenNameTypes.UNK.value
        ExpArgs.accumulate_grad_batches = 5
        ExpArgs.batch_size = 4

    print(
        "*" * 20,
        args.task,
        args.explained_model_backbone,
        args.interpreter_model_backbone,
        args.metric,
        "*" * 20,
        flush = True,
    )
    if ExpArgs.explained_model_path is not None:
        print(f"Explained model override path: {ExpArgs.explained_model_path}", flush = True)
    if ExpArgs.eraser_root is not None:
        print(f"ERASER source: {ExpArgs.eraser_root}", flush = True)
    if ExpArgs.sst2_source is not None:
        print(f"SST-2 source: {ExpArgs.sst2_source}", flush = True)

    time_str = get_current_time()
    experiment_name_prefix = (
        f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{ExpArgs.interpreter_model_backbone}_{ExpArgs.eval_metric}"
    )
    if ExpArgs.explained_model_path is not None:
        experiment_name_prefix = (
            f"{experiment_name_prefix}_PATH_{build_path_run_tag(ExpArgs.explained_model_path)}"
        )

    explained_model = load_explained_model()

    hp_experiment_name = f"HP_{experiment_name_prefix}_{time_str}"
    hp = HpSearch(hp_experiment_name, explained_model = explained_model).run()

    pre_train_experiment_name = f"PRETRAIN_{experiment_name_prefix}_{time_str}"
    pretrain_model_path = PreTrain(hp, pre_train_experiment_name, explained_model = explained_model).run()

    ExpArgs.fine_tuned_interpreter_model_path = pretrain_model_path

    inference_pretrain_experiment_name = f"INFERENCE_PRETRAIN_{experiment_name_prefix}_{time_str}"
    InferencePretrain(hp, inference_pretrain_experiment_name, explained_model = explained_model).run()

    # fine_tune_exp_name = f"FINE_TUNE_{experiment_name_prefix}_{time_str}"
    # FineTune(hp, fine_tune_exp_name, explained_model = explained_model).run()
    print(
        "*" * 20,
        "END OF",
        args.task,
        args.explained_model_backbone,
        args.interpreter_model_backbone,
        args.metric,
        "*" * 20,
    )


if __name__ == "__main__":
    main()
