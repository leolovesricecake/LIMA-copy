"""
Created on 2024/8/19

@author: Ruoyu Chen
"""
import os
# Keep CUDA runtime device order consistent with nvidia-smi indices.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import subprocess
from torch.autograd import Variable
from .Grad_Eclip.grad_eclip import *

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms

# import tensorflow as tf
from utils import *

import clip


def _query_physical_gpu_indices_from_nvidia_smi():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
        )
        lines = result.decode("utf-8").strip().splitlines()
        indices = [int(x.strip()) for x in lines if x.strip() != ""]
        return indices
    except Exception:
        return []


def _bootstrap_device_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=int, default=0)
    args, _ = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if args.device >= 0:
        all_physical = _query_physical_gpu_indices_from_nvidia_smi()
        if len(all_physical) > 0 and args.device in all_physical:
            reordered = [args.device] + [idx for idx in all_physical if idx != args.device]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in reordered)
            if inherited != "" and inherited != os.environ["CUDA_VISIBLE_DEVICES"]:
                print(
                    "[bootstrap] Override inherited CUDA_VISIBLE_DEVICES={} -> {} (requested physical --device={}).".format(
                        inherited, os.environ["CUDA_VISIBLE_DEVICES"], args.device
                    )
                )
            else:
                print(
                    "[bootstrap] Set CUDA_VISIBLE_DEVICES={} (requested physical --device={}).".format(
                        os.environ["CUDA_VISIBLE_DEVICES"], args.device
                    )
                )
        elif inherited == "":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return args.device


_BOOTSTRAP_DEVICE = _bootstrap_device_from_argv()

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

img_size = 224

# dataset_index = "datasets/imagenet/val_clip_vitl_5k_true.txt"
# SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-true")

dataset_index = "datasets/imagenet/val_clip_vitl_2k_false.txt"
SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-false")
    
dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
class_number = 1000
batch = 1
mkdir(SAVE_PATH)


def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights * 100


def load_or_build_semantic_feature(model, device, semantic_path):
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
        return semantic_feature

    print("semantic feature not found, generating from ImageNet classes and templates...")
    semantic_feature = zeroshot_classifier(
        model, imagenet_classes, imagenet_templates, device
    )
    os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
    torch.save(semantic_feature.cpu(), semantic_path)
    print("semantic feature saved to {}".format(semantic_path))
    return semantic_feature

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-ECLIP attribution maps.")
    parser.add_argument(
        "--device",
        type=int,
        default=_BOOTSTRAP_DEVICE,
        help="Physical GPU index from nvidia-smi. Set -1 for CPU.",
    )
    return parser.parse_args()


def map_physical_to_visible_device_index(requested_device_index):
    if requested_device_index < 0:
        return requested_device_index

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible == "":
        return requested_device_index

    visible_tokens = [x.strip() for x in cuda_visible.split(",") if x.strip() != ""]
    if len(visible_tokens) == 0:
        return requested_device_index

    if all(token.isdigit() for token in visible_tokens):
        requested_str = str(requested_device_index)
        if requested_str not in visible_tokens:
            raise ValueError(
                "Requested physical --device={} is not in CUDA_VISIBLE_DEVICES={}".format(
                    requested_device_index, cuda_visible
                )
            )
        logical_index = visible_tokens.index(requested_str)
        print(
            "Detected CUDA_VISIBLE_DEVICES={}, map physical GPU {} -> visible logical GPU {}".format(
                cuda_visible, requested_device_index, logical_index
            )
        )
        return logical_index

    print(
        "Detected non-numeric CUDA_VISIBLE_DEVICES={}, treat --device as visible logical index.".format(
            cuda_visible
        )
    )
    return requested_device_index


def resolve_device(device_index):
    if device_index < 0 or not torch.cuda.is_available():
        return "cpu"
    gpu_count = torch.cuda.device_count()
    if device_index >= gpu_count:
        raise ValueError(
            "Invalid --device {}. Available CUDA devices: 0..{}.".format(
                device_index, gpu_count - 1
            )
        )
    torch.cuda.set_device(device_index)
    return "cuda:{}".format(device_index)


def main(args):
    print("Requested physical --device={}".format(args.device))
    print("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER", "<not set>")))
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")))
    visible_device_index = map_physical_to_visible_device_index(args.device)
    device = resolve_device(visible_device_index)
    print("Selected visible logical GPU index={}".format(visible_device_index))
    print("Torch device={}".format(device))
    # Load model
    clipmodel, preprocess = clip.load("ViT-L/14", device=device, download_root=".checkpoints/CLIP")
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    text_embedding = load_or_build_semantic_feature(
        clipmodel, device, semantic_path
    )
    
    explainer = Grad_ECLIP(clipmodel, text_embedding)
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        label.append(int(data.strip().split(" ")[-1]))
        input_data.append(
            os.path.join(dataset_path, data.split(" ")[0])
        )
    
    total_steps = math.ceil(len(input_data) / batch)
    
    explainer_method_name = "GradECLIP"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for step in tqdm(range(total_steps), desc=explainer_method_name):
        img_path = input_data[step]
        X_raw = Image.open(img_path).convert("RGB").resize((224,224))
        X_raw = imgprocess(X_raw).to(device).unsqueeze(0)

        Y_true = label[step]
        
        if device != "cpu":
            torch.cuda.empty_cache()
        explanation = explainer(
            X_raw,
            Y_true
            )
        if device != "cpu":
            torch.cuda.empty_cache()
        
        np.save(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".JPEG", "")),
                cv2.resize(explanation.astype(np.float32), (224,224)))

if __name__ == "__main__":
    args = parse_args()
    main(args)
