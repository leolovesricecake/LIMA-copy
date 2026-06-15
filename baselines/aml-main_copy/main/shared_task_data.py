import sys
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Value, load_dataset

from config.config import ExpArgs


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lima_llm.data import load_dataset_bundle


MAINLINE_TASK_NAMES = {
    "sst2",
    "eraser_movie_reviews",
    "imdb",
    "rotten_tomatoes",
    "emotion",
}


def uses_shared_lima_loader(task) -> bool:
    return task.name in MAINLINE_TASK_NAMES


def load_task_dataset_bundle_for_split(task, split: str, max_samples = None):
    if not uses_shared_lima_loader(task):
        raise ValueError(f"Task '{task.name}' does not use the shared lima_llm loader")
    return load_dataset_bundle(
        dataset_name = task.name,
        split = split,
        max_samples = max_samples,
        eraser_root = ExpArgs.eraser_root,
        sst2_source = ExpArgs.sst2_source,
        dataset_cache_dir = ExpArgs.dataset_cache_dir,
    )


def dataset_from_bundle(task, bundle) -> Dataset:
    label_names = [str(name) for name in bundle.label_names]
    features = Features(
        {
            task.dataset_column_text: Value("string"),
            task.dataset_column_label: ClassLabel(names = label_names),
            "id": Value("int64"),
        }
    )
    data = {
        task.dataset_column_text: [sample.text for sample in bundle.samples],
        task.dataset_column_label: [int(sample.label) for sample in bundle.samples],
        "id": list(range(len(bundle.samples))),
    }
    return Dataset.from_dict(data, features = features)


def load_task_split_dataset(task, split: str, max_samples = None) -> Dataset:
    if uses_shared_lima_loader(task):
        bundle = load_task_dataset_bundle_for_split(task, split = split, max_samples = max_samples)
        return dataset_from_bundle(task, bundle)

    dataset = load_dataset(task.dataset_name)[split]
    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    return dataset
