# -- coding: utf-8 --**
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate highest confidence statistics for submodular (ours) results."
    )
    parser.add_argument(
        "--explanation-method",
        type=str,
        default="submodular_results/imagenet-clip-vitl-efficientv2/seeds-0.0-0.05-20.0-1.0-pending-samples-8",
        help="Root directory of ours results, expects '<root>/{json,npy}/...'.",
    )
    parser.add_argument(
        "--eval-list",
        type=str,
        default="datasets/imagenet/val_clip_vitl_5k_true.txt",
        help="Evaluation list file.",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Comma separated percentages, e.g. '0.25,0.5,0.75,1.0'.",
    )
    return parser.parse_args()


def _replace_ext(path: str, ext: str) -> str:
    return path.replace(".jpg", ext).replace(".jpeg", ext).replace(".JPEG", ext)


def _parse_eval_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _build_name_index(root_dir: str, suffix: str) -> Dict[str, str]:
    index = {}
    for cur_root, _, files in os.walk(root_dir):
        for name in files:
            if not name.endswith(suffix):
                continue
            if name not in index:
                index[name] = os.path.join(cur_root, name)
    return index


def _resolve_paths(
    json_root: str,
    npy_root: str,
    image_rel_path: str,
    gt_id: str,
    json_name_index: Dict[str, str],
    npy_name_index: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    json_name = _replace_ext(image_rel_path, ".json")
    npy_name = _replace_ext(image_rel_path, ".npy")

    # Preferred path pattern used by current efficientv2 scripts:
    #   <root>/json/<gt_id>/<image>.json
    #   <root>/npy/<gt_id>/<image>.npy
    json_path = os.path.join(json_root, gt_id, json_name)
    npy_path = os.path.join(npy_root, gt_id, npy_name)
    if os.path.exists(json_path) and os.path.exists(npy_path):
        return json_path, npy_path

    # Backward compatible fallback.
    json_path = json_name_index.get(os.path.basename(json_name))
    npy_path = npy_name_index.get(os.path.basename(npy_name))
    if json_path is None or npy_path is None:
        return None, None
    return json_path, npy_path


def _collect_samples(explanation_method: str, eval_list: str):
    json_root = os.path.join(explanation_method, "json")
    npy_root = os.path.join(explanation_method, "npy")
    if not (os.path.isdir(json_root) and os.path.isdir(npy_root)):
        raise ValueError(
            "Invalid explanation_method '{}': expect '{}/json' and '{}/npy' directories.".format(
                explanation_method, explanation_method, explanation_method
            )
        )

    json_name_index = _build_name_index(json_root, ".json")
    npy_name_index = _build_name_index(npy_root, ".npy")

    with open(eval_list, "r") as f:
        infos = [line.strip() for line in f if line.strip() != ""]

    samples = []
    malformed = 0
    missing = 0
    broken = 0
    missing_key = 0

    for info in tqdm(infos, desc="scan", dynamic_ncols=True):
        image_rel_path, gt_id = _parse_eval_line(info)
        if image_rel_path is None:
            malformed += 1
            continue

        json_path, npy_path = _resolve_paths(
            json_root,
            npy_root,
            image_rel_path,
            gt_id,
            json_name_index,
            npy_name_index,
        )
        if json_path is None or npy_path is None:
            missing += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                f_data = json.load(f)
            submodular_image_set = np.load(npy_path)
        except Exception:
            broken += 1
            continue

        if "consistency_score" not in f_data:
            missing_key += 1
            continue

        insertion_area = []
        insertion_ours_image = submodular_image_set[0] - submodular_image_set[0]
        for smdl_sub_mask in submodular_image_set:
            insertion_ours_image += smdl_sub_mask
            insertion_area.append(
                (insertion_ours_image.sum(-1) != 0).sum()
                / (insertion_ours_image.shape[0] * insertion_ours_image.shape[1])
            )

        samples.append(
            {
                "insertion_area": insertion_area,
                "consistency_score": f_data["consistency_score"],
            }
        )

    summary = {
        "total_eval_list": len(infos),
        "valid_samples": len(samples),
        "malformed": malformed,
        "missing": missing,
        "broken": broken,
        "missing_key": missing_key,
    }
    return samples, summary


def _evaluate_for_percentage(samples: List[dict], percentage: float):
    highest_acc = []
    region_area = []
    empty_curve_count = 0

    for item in samples:
        insertion_area = item["insertion_area"]
        consistency_score = item["consistency_score"]

        number = (np.array(insertion_area) <= percentage).sum()
        data = consistency_score[:number]
        if len(data) == 0:
            empty_curve_count += 1
            continue

        highest_conf = max(data)
        highest_acc.append(highest_conf)
        region_area.append(insertion_area[data.index(highest_conf)])

    if len(highest_acc) == 0:
        raise RuntimeError(
            "No valid samples for percentage={}. Check explanation path or eval list.".format(
                percentage
            )
        )

    mean_highest_acc = np.array(highest_acc).mean()
    std_highest_acc = np.array(highest_acc).std()
    mean_region_area = np.array(region_area).mean()
    std_region_area = np.array(region_area).std()
    return {
        "percentage": percentage,
        "mean_highest_acc": mean_highest_acc,
        "std_highest_acc": std_highest_acc,
        "mean_region_area": mean_region_area,
        "std_region_area": std_region_area,
        "used": len(highest_acc),
        "empty_curve_count": empty_curve_count,
    }


def main(args):
    percentages = []
    for x in args.percentages.split(","):
        x = x.strip()
        if x == "":
            continue
        percentages.append(float(x))
    if len(percentages) == 0:
        raise ValueError("--percentages must contain at least one value.")

    samples, summary = _collect_samples(args.explanation_method, args.eval_list)
    print(
        "[scan] eval_list={} valid={} missing={} broken={} missing_key={} malformed={}.".format(
            summary["total_eval_list"],
            summary["valid_samples"],
            summary["missing"],
            summary["broken"],
            summary["missing_key"],
            summary["malformed"],
        )
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No valid sample matched between eval list and '{}'.".format(args.explanation_method)
        )

    for percentage in percentages:
        ret = _evaluate_for_percentage(samples, percentage)
        print(
            "When percentage is {}, the avg. highest confidence is {}, std:{}, "
            "the retention percentage at highest confidence is {}, std:{}, used={}, empty_curve={}.".format(
                ret["percentage"],
                ret["mean_highest_acc"],
                ret["std_highest_acc"],
                ret["mean_region_area"],
                ret["std_region_area"],
                ret["used"],
                ret["empty_curve_count"],
            )
        )


if __name__ == "__main__":
    main(parse_args())
