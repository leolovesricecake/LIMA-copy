# -- coding: utf-8 --**
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate highest confidence statistics from insertion json curves."
    )
    parser.add_argument(
        "--explanation-method",
        type=str,
        default="explanation_insertion_results/imagenet-clip-vitl-true/HsicAttributionMethod",
        help="Path to baseline json root, or ours root containing '<root>/json'.",
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
    if len(parts) < 1:
        return None, None
    image_rel_path = parts[0]
    gt_id = parts[1] if len(parts) > 1 else None
    return image_rel_path, gt_id


def _build_json_name_index(root_dir: str) -> Dict[str, str]:
    index = {}
    for cur_root, _, files in os.walk(root_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if name not in index:
                index[name] = os.path.join(cur_root, name)
    return index


def _resolve_json_path(
    explanation_method: str,
    image_rel_path: str,
    gt_id: Optional[str],
    has_json_subdir: bool,
    json_name_index: Dict[str, str],
) -> Optional[str]:
    json_name = _replace_ext(image_rel_path, ".json")

    if has_json_subdir:
        json_root = os.path.join(explanation_method, "json")
        if gt_id is not None:
            by_class = os.path.join(json_root, gt_id, json_name)
            if os.path.exists(by_class):
                return by_class
        by_flat = os.path.join(json_root, json_name)
        if os.path.exists(by_flat):
            return by_flat
    else:
        by_flat = os.path.join(explanation_method, json_name)
        if os.path.exists(by_flat):
            return by_flat

    return json_name_index.get(os.path.basename(json_name))


def _collect_scores(explanation_method: str, eval_list: str):
    has_json_subdir = os.path.isdir(os.path.join(explanation_method, "json"))
    json_root_for_index = (
        os.path.join(explanation_method, "json")
        if has_json_subdir
        else explanation_method
    )
    json_name_index = _build_json_name_index(json_root_for_index)

    with open(eval_list, "r") as f:
        infos = [line.strip() for line in f if line.strip() != ""]

    scores = []
    malformed = 0
    missing = 0
    broken = 0
    missing_key = 0
    empty = 0

    for info in tqdm(infos, desc="scan", dynamic_ncols=True):
        image_rel_path, gt_id = _parse_eval_line(info)
        if image_rel_path is None:
            malformed += 1
            continue

        json_path = _resolve_json_path(
            explanation_method,
            image_rel_path,
            gt_id,
            has_json_subdir,
            json_name_index,
        )
        if json_path is None:
            missing += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                f_data = json.load(f)
        except Exception:
            broken += 1
            continue

        if "recognition_score" in f_data:
            score_key = "recognition_score"
        elif "consistency_score" in f_data:
            score_key = "consistency_score"
        else:
            missing_key += 1
            continue

        curve = f_data[score_key]
        if len(curve) == 0:
            empty += 1
            continue
        scores.append(curve)

    summary = {
        "total_eval_list": len(infos),
        "valid_samples": len(scores),
        "malformed": malformed,
        "missing": missing,
        "broken": broken,
        "missing_key": missing_key,
        "empty": empty,
        "mode": "format-1-with-json-subdir" if has_json_subdir else "format-2-flat-json",
    }
    return scores, summary


def _evaluate_for_percentage(curves: List[list], percentage: float):
    highest_acc = []
    region_area = []
    empty_curve_count = 0

    for curve in curves:
        steps = len(curve)
        number = int(percentage * steps)
        data = curve[:number]
        if len(data) == 0:
            empty_curve_count += 1
            continue

        highest_conf = max(data)
        highest_acc.append(highest_conf)
        region_area.append((data.index(highest_conf) + 1) / steps)

    if len(highest_acc) == 0:
        raise RuntimeError(
            "No valid samples for percentage={}. Please check input path/config.".format(
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

    curves, summary = _collect_scores(args.explanation_method, args.eval_list)
    print(
        "[scan] mode={} eval_list={} valid={} missing={} broken={} missing_key={} empty={} malformed={}.".format(
            summary["mode"],
            summary["total_eval_list"],
            summary["valid_samples"],
            summary["missing"],
            summary["broken"],
            summary["missing_key"],
            summary["empty"],
            summary["malformed"],
        )
    )

    if len(curves) == 0:
        raise RuntimeError(
            "No valid json curves matched between eval list and '{}'.".format(
                args.explanation_method
            )
        )

    for percentage in percentages:
        ret = _evaluate_for_percentage(curves, percentage)
        print(
            "percentage: {}, avg. highest confidence: {}, std:{}, "
            "retention percentage at highest confidence: {}, std: {}, used={}, empty_curve={}.".format(
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
