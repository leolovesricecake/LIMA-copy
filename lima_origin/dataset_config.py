import os
from typing import Dict, List, Tuple


_DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "imagenet": {
        "eval_list": "../datasets/imagenet/val_clip_vitl_5k_true.txt",
        "image_root_path": "../datasets/imagenet/ILSVRC2012_img_val",
        "save_doc": "imagenet-clip-vitl-true",
    },
    "imagenet-false": {
        "eval_list": "../datasets/imagenet/val_clip_vitl_2k_false.txt",
        "image_root_path": "../datasets/imagenet/ILSVRC2012_img_val",
        "save_doc": "imagenet-clip-vitl-false",
    },
    "imagenet-a": {
        "eval_list": "../datasets/ImageNet-A/imagenet-a_list.txt",
        "image_root_path": "../datasets/ImageNet-A/sample/image",
        "save_doc": "imagenet-a-clip-vitl",
    },
    "imagenet-o": {
        "eval_list": "../datasets/imagenet-o/imagenet-o_list.txt",
        "image_root_path": "../datasets/imagenet-o/samples",
        "save_doc": "imagenet-o-clip-vitl",
    },
}

_ALIASES: Dict[str, str] = {
    "imagenet_a": "imagenet-a",
    "imageneta": "imagenet-a",
    "imagenet_o": "imagenet-o",
    "imageneto": "imagenet-o",
    "imagenet_false": "imagenet-false",
    "imagenet-false": "imagenet-false",
    "imagenet_true": "imagenet",
}


def normalize_dataset_name(dataset_name: str) -> str:
    if dataset_name is None:
        raise ValueError("dataset_name cannot be None.")
    key = dataset_name.strip().lower()
    if key in _DATASET_CONFIGS:
        return key
    return _ALIASES.get(key, key)


def resolve_dataset_config(
    dataset_name: str,
    eval_list: str = None,
    image_root_path: str = None,
    save_doc: str = None,
) -> Dict[str, str]:
    key = normalize_dataset_name(dataset_name)
    if key not in _DATASET_CONFIGS:
        raise ValueError(
            "Unsupported dataset_name '{}'. Available: {}.".format(
                dataset_name, sorted(_DATASET_CONFIGS.keys())
            )
        )

    cfg = dict(_DATASET_CONFIGS[key])
    cfg["dataset_name"] = key
    if eval_list is not None and eval_list.strip() != "":
        cfg["eval_list"] = eval_list
    if image_root_path is not None and image_root_path.strip() != "":
        cfg["image_root_path"] = image_root_path
    if save_doc is not None and save_doc.strip() != "":
        cfg["save_doc"] = save_doc
    return cfg


def parse_eval_list(eval_list_path: str, require_label: bool = True) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(eval_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) == 0:
                continue
            image_rel_path = parts[0]
            if require_label:
                if len(parts) < 2:
                    raise ValueError(
                        "Malformed eval-list line without label: '{}' in '{}'.".format(
                            line, eval_list_path
                        )
                    )
                label = int(parts[1])
            else:
                label = -1
            samples.append((image_rel_path, label))
    return samples


def rel_no_ext(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root.replace("\\", "/")


def build_result_file_path(root_dir: str, image_rel_path: str, ext: str) -> str:
    rel = rel_no_ext(image_rel_path) + ext
    path = os.path.join(root_dir, rel)
    parent = os.path.dirname(path)
    if parent != "":
        os.makedirs(parent, exist_ok=True)
    return path

