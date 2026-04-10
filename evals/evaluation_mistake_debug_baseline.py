# -- coding: utf-8 --**
import os
import json
from tqdm import tqdm
import numpy as np


# explanation_method = "explanation_insertion_results/cub-fair-efficientnet/KernelShap"
# eval_list = "datasets/CUB/eval_fair-efficientnet.txt"
# TODO 手动修改
explanation_method = "explanation_insertion_results/imagenet-clip-vitl-true/HsicAttributionMethod"
# explanation_method = "explanation_insertion_results/imagenet-clip-vitl-true/GradECLIP"
eval_list = "datasets/imagenet/val_clip_vitl_5k_true.txt"

# steps = 49
# percentage = 0.25
# number = int(percentage * steps)
# 

def main(percentage):
    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    highest_acc = []
    region_area = []

    for info in tqdm(infos[:]):
        # if "CUB" in eval_list:
        #     json_file_path = os.path.join(explanation_method, info.split(" ")[0].split("/")[-1].replace(".jpg", ".json").replace(".JPEG", ".json").replace(".jpeg", ".json"))
        # else:
        json_file_path = os.path.join(explanation_method, info.split(" ")[0].replace(".jpg", ".json").replace(".JPEG", ".json").replace(".jpeg", ".json"))

        
        
        if not os.path.exists(json_file_path):
            continue

        with open(json_file_path, 'r', encoding='utf-8') as f:
            f_data = json.load(f)

        if "recognition_score" in f_data:
            score_key = "recognition_score"
        elif "consistency_score" in f_data:
            score_key = "consistency_score"
        else:
            continue

        steps = len(f_data[score_key])
        if steps == 0:
            continue
        number = int(percentage * steps)
        
        data = f_data[score_key][:number]
        if len(data) == 0:
            continue

        highest_conf = max(data)
        highest_acc.append(highest_conf)

        area = (data.index(highest_conf) + 1) / steps
        region_area.append(area)

    mean_highest_acc = np.array(highest_acc).mean()
    std_highest_acc = np.array(highest_acc).std()

    mean_region_area = np.array(region_area).mean()
    std_region_area = np.array(region_area).std()
    print("percentage: {}, avg. highest confidence: {}, std:{}, retention percentage at highest confidence: {}, std: {}".format(
        percentage, mean_highest_acc, std_highest_acc, mean_region_area, std_region_area
    ))
    return

main(0.25)
main(0.5)
main(0.75)
main(1.0)
