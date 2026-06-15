import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes
from main.shared_task_data import load_task_split_dataset
from models.train_models_utils import get_task_base_model_path
from runs.runs_utils import get_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Train an encoder explained-model checkpoint for AML")
    parser.add_argument("task", type = str)
    parser.add_argument("backbone", type = str, choices = [
        ModelBackboneTypes.BERT.value,
        ModelBackboneTypes.ROBERTA.value,
        ModelBackboneTypes.DISTILBERT.value,
    ])
    parser.add_argument("--output-dir", type = str, required = True)
    parser.add_argument("--model-name-or-path", type = str, default = None)
    parser.add_argument("--num-train-epochs", type = float, default = 1.0)
    parser.add_argument("--learning-rate", type = float, default = 4e-5)
    parser.add_argument("--weight-decay", type = float, default = 0.0)
    parser.add_argument("--per-device-train-batch-size", type = int, default = 8)
    parser.add_argument("--per-device-eval-batch-size", type = int, default = 8)
    parser.add_argument("--max-length", type = int, default = 512)
    parser.add_argument("--max-train-samples", type = int, default = None)
    parser.add_argument("--max-eval-samples", type = int, default = None)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--eraser-root", type = str, default = None)
    parser.add_argument("--sst2-source", type = str, default = None)
    parser.add_argument("--dataset-cache-dir", type = str, default = None)
    return parser


def _resolve_model_name_or_path(task, backbone: str, explicit_model_name_or_path: str = None) -> str:
    if explicit_model_name_or_path is not None:
        return explicit_model_name_or_path
    return get_task_base_model_path(task, backbone)


def _tokenize_dataset(dataset, tokenizer, text_column: str, label_column: str, max_length: int):
    def _tokenize(batch):
        return tokenizer(batch[text_column], truncation = True, max_length = max_length)

    encoded = dataset.map(_tokenize, batched = True)
    if label_column != "labels":
        encoded = encoded.rename_column(label_column, "labels")
    return encoded


def _compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    accuracy = float((predictions == labels).mean()) if len(labels) > 0 else 0.0
    return {"accuracy": accuracy}


def _write_training_report(args, task, model_name_or_path: str, output_dir: Path, metrics: dict) -> None:
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": task.name,
        "requested_task_name": args.task,
        "task_paper_name": task.paper_name,
        "backbone": args.backbone,
        "model_name_or_path": model_name_or_path,
        "output_dir": str(output_dir),
        "num_labels": len(task.labels_str_int_maps),
        "label_names": list(task.labels_str_int_maps.keys()),
        "label_verbalizers": task.labels_str_int_maps,
        "dataset": {
            "train_split": task.dataset_train,
            "eval_split": task.dataset_val,
            "text_column": task.dataset_column_text,
            "label_column": task.dataset_column_label,
            "eraser_root": args.eraser_root,
            "sst2_source": args.sst2_source,
            "dataset_cache_dir": args.dataset_cache_dir,
        },
        "training_config": {
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "max_length": args.max_length,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
        },
        "metrics": metrics,
        "artifacts": {
            "checkpoint_dir": str(output_dir),
            "metrics_json": "aml_explained_model_metrics.json",
            "training_report_json": "aml_explained_model_report.json",
        },
    }
    with open(output_dir / "aml_explained_model_report.json", "w", encoding = "utf-8") as file:
        json.dump(report, file, indent = 2, ensure_ascii = False)


def train_explained_model(args) -> Path:
    task = get_task(args.task)
    ExpArgs.requested_task_name = args.task
    ExpArgs.task = task
    ExpArgs.eraser_root = args.eraser_root
    ExpArgs.sst2_source = args.sst2_source
    ExpArgs.dataset_cache_dir = args.dataset_cache_dir

    model_name_or_path = _resolve_model_name_or_path(task, args.backbone, args.model_name_or_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents = True, exist_ok = True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = load_task_split_dataset(task, task.dataset_train, max_samples = args.max_train_samples)
    eval_dataset = load_task_split_dataset(task, task.dataset_val, max_samples = args.max_eval_samples)

    train_dataset = _tokenize_dataset(
        dataset = train_dataset,
        tokenizer = tokenizer,
        text_column = task.dataset_column_text,
        label_column = task.dataset_column_label,
        max_length = args.max_length,
    )
    eval_dataset = _tokenize_dataset(
        dataset = eval_dataset,
        tokenizer = tokenizer,
        text_column = task.dataset_column_text,
        label_column = task.dataset_column_label,
        max_length = args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels = len(task.labels_str_int_maps.keys()),
    )
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    training_args = TrainingArguments(
        output_dir = str(output_dir),
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        num_train_epochs = args.num_train_epochs,
        seed = args.seed,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 1,
        logging_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        report_to = [],
        remove_unused_columns = True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = _compute_accuracy,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    with open(output_dir / "aml_explained_model_metrics.json", "w", encoding = "utf-8") as file:
        json.dump(metrics, file, indent = 2, ensure_ascii = False)
    _write_training_report(args = args, task = task, model_name_or_path = model_name_or_path, output_dir = output_dir,
                           metrics = metrics)

    return output_dir


def main(argv = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = train_explained_model(args)
    print(f"Saved explained-model checkpoint to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
