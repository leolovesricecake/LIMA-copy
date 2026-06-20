from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


AML_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "aml-main_copy"
if str(AML_ROOT) not in sys.path:
    sys.path.insert(0, str(AML_ROOT))


class _DummyTokenizer:
    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102
    model_max_length = 512

    def encode_plus(self, text, truncation = True, add_special_tokens = True, return_tensors = None):
        tokens = [self.cls_token_id, 200, self.sep_token_id] if add_special_tokens else [200]
        attention = [1] * len(tokens)
        if return_tensors == "pt":
            import torch

            return SimpleNamespace(
                input_ids = torch.tensor(tokens).unsqueeze(0),
                attention_mask = torch.tensor(attention).unsqueeze(0),
            )
        return SimpleNamespace(input_ids = tokens, attention_mask = attention)

    def convert_ids_to_tokens(self, val):
        return str(int(val))


def test_aml_task_aliases_and_ambiguous_emr() -> None:
    from runs.runs_utils import get_task

    assert get_task("sst2").name == "sst2"
    assert get_task("sst").name == "sst2"
    assert get_task("rotten_tomatoes").name == "rotten_tomatoes"
    assert get_task("rtn").name == "rotten_tomatoes"
    assert get_task("emotion").name == "emotion"
    assert get_task("emotions").name == "emotion"
    assert get_task("eraser_movie_reviews").name == "eraser_movie_reviews"
    assert get_task("eraser").name == "eraser_movie_reviews"
    assert get_task("imdb").name == "imdb"

    with pytest.raises(ValueError, match = "ambiguous"):
        get_task("emr")


def test_sst2_aml_test_split_uses_validation() -> None:
    from runs.runs_utils import get_task

    assert get_task("sst2").dataset_test == "validation"


def test_shared_loader_passes_dataset_source_args(monkeypatch) -> None:
    from config.config import ExpArgs
    from main import shared_task_data
    from runs.runs_utils import get_task

    calls = []

    def _fake_load_dataset_bundle(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(label_names = ["negative", "positive"], samples = [])

    monkeypatch.setattr(shared_task_data, "load_dataset_bundle", _fake_load_dataset_bundle)

    ExpArgs.eraser_root = "hf://eraser-benchmark/movie_rationales"
    ExpArgs.sst2_source = "hf://nyu-mll/glue"
    ExpArgs.dataset_cache_dir = "/tmp/aml-cache"

    shared_task_data.load_task_dataset_bundle_for_split(
        get_task("eraser_movie_reviews"),
        split = "validation",
        max_samples = 5,
    )
    shared_task_data.load_task_dataset_bundle_for_split(
        get_task("sst2"),
        split = "train",
        max_samples = 7,
    )

    assert calls[0]["dataset_name"] == "eraser_movie_reviews"
    assert calls[0]["split"] == "validation"
    assert calls[0]["max_samples"] == 5
    assert calls[0]["eraser_root"] == "hf://eraser-benchmark/movie_rationales"
    assert calls[0]["dataset_cache_dir"] == "/tmp/aml-cache"

    assert calls[1]["dataset_name"] == "sst2"
    assert calls[1]["split"] == "train"
    assert calls[1]["max_samples"] == 7
    assert calls[1]["sst2_source"] == "hf://nyu-mll/glue"


def test_data_module_uses_shared_loader_for_mainline_tasks(monkeypatch) -> None:
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("datasets")
    pytest.importorskip("tokenizations")

    from datasets import ClassLabel, Dataset, Features, Value

    from config.config import ExpArgs
    from config.types_enums import ModelBackboneTypes, ValidationType
    import main.data_module as data_module_mod
    from runs.runs_utils import get_task

    calls = []

    def _fake_load_task_split_dataset(task, split, max_samples = None):
        calls.append((task.name, split, max_samples))
        return Dataset.from_dict(
            {
                task.dataset_column_text: ["good", "bad"],
                task.dataset_column_label: [1, 0],
                "id": [0, 1],
            },
            features = Features(
                {
                    task.dataset_column_text: Value("string"),
                    task.dataset_column_label: ClassLabel(names = ["negative", "positive"]),
                    "id": Value("int64"),
                }
            ),
        )

    monkeypatch.setattr(data_module_mod, "load_task_split_dataset", _fake_load_task_split_dataset)

    ExpArgs.task = get_task("sst2")
    ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
    ExpArgs.interpreter_model_backbone = ModelBackboneTypes.BERT.value

    data_module = data_module_mod.DataModule(
        val_type = ValidationType.VAL,
        train_sample = 0,
        test_sample = 0,
        explained_tokenizer = _DummyTokenizer(),
        interpreter_tokenizer = _DummyTokenizer(),
    )

    assert calls == [("sst2", "train", None), ("sst2", "validation", None)]
    assert len(data_module.train_dataset) == 2
    assert len(data_module.val_dataset) == 2


def test_data_module_uses_validation_for_sst2_test_mode(monkeypatch) -> None:
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("datasets")
    pytest.importorskip("tokenizations")

    from datasets import ClassLabel, Dataset, Features, Value

    from config.config import ExpArgs
    from config.types_enums import ModelBackboneTypes, ValidationType
    import main.data_module as data_module_mod
    from runs.runs_utils import get_task

    calls = []

    def _fake_load_task_split_dataset(task, split, max_samples = None):
        calls.append((task.name, split, max_samples))
        return Dataset.from_dict(
            {
                task.dataset_column_text: ["good", "bad"],
                task.dataset_column_label: [1, 0],
                "id": [0, 1],
            },
            features = Features(
                {
                    task.dataset_column_text: Value("string"),
                    task.dataset_column_label: ClassLabel(names = ["negative", "positive"]),
                    "id": Value("int64"),
                }
            ),
        )

    monkeypatch.setattr(data_module_mod, "load_task_split_dataset", _fake_load_task_split_dataset)

    ExpArgs.task = get_task("sst2")
    ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
    ExpArgs.interpreter_model_backbone = ModelBackboneTypes.BERT.value

    data_module = data_module_mod.DataModule(
        val_type = ValidationType.TEST,
        train_sample = 0,
        test_sample = 0,
        explained_tokenizer = _DummyTokenizer(),
        interpreter_tokenizer = _DummyTokenizer(),
    )

    assert calls == [("sst2", "train", None), ("sst2", "validation", None)]
    assert len(data_module.val_dataset) == 2


def test_prompt_label_validation_reports_task_model_and_label() -> None:
    from models.train_models_utils import build_prompt_label_vocab_tokens
    from runs.runs_utils import get_task

    class _BadTokenizer:
        def encode(self, text, add_special_tokens = False):
            if text == "E":
                return [7, 8]
            return [3]

    with pytest.raises(ValueError) as exc:
        build_prompt_label_vocab_tokens(
            task = get_task("emotion"),
            tokenizer = _BadTokenizer(),
            explained_model_path = "/tmp/qwen",
        )

    message = str(exc.value)
    assert "task=emotion" in message
    assert "explained_model_path=/tmp/qwen" in message
    assert "E -> [7, 8]" in message


def test_train_explained_model_smoke_run_writes_checkpoint(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("datasets")

    from datasets import ClassLabel, Dataset, Features, Value

    import runs.train_explained_model as train_mod

    def _fake_load_task_split_dataset(task, split, max_samples = None):
        text_column = task.dataset_column_text
        label_column = task.dataset_column_label
        return Dataset.from_dict(
            {
                text_column: ["a happy sample", "a sad sample"],
                label_column: [1, 0],
                "id": [0, 1],
            },
            features = Features(
                {
                    text_column: Value("string"),
                    label_column: ClassLabel(names = ["negative", "positive"]),
                    "id": Value("int64"),
                }
            ),
        )

    class _FakeTokenizer:
        def __call__(self, texts, truncation = True, max_length = 512):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2, 3] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

        def save_pretrained(self, output_dir):
            Path(output_dir, "tokenizer.json").write_text("{}", encoding = "utf-8")

    class _FakeModel:
        pass

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]

    class _FakeTrainer:
        def __init__(self, **kwargs):
            self.args = kwargs["args"]

        def train(self):
            return None

        def save_model(self, output_dir):
            Path(output_dir, "pytorch_model.bin").write_text("fake", encoding = "utf-8")

        def evaluate(self):
            return {"eval_accuracy": 1.0}

    monkeypatch.setattr(train_mod, "load_task_split_dataset", _fake_load_task_split_dataset)
    monkeypatch.setattr(
        train_mod.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        train_mod.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeModel()),
    )
    monkeypatch.setattr(train_mod, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(train_mod, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_mod, "DataCollatorWithPadding", lambda tokenizer: object())

    output_dir = tmp_path / "eraser_bert"
    rc = train_mod.main(
        [
            "eraser_movie_reviews",
            "BERT",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert (output_dir / "pytorch_model.bin").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "aml_explained_model_metrics.json").exists()
    assert (output_dir / "aml_explained_model_report.json").exists()
