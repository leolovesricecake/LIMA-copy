# -*- coding: utf-8 -*-  

"""
Created on 2024/6/3

@author: Ruoyu Chen
"""

import os
# NOTE:
# Do not hardcode CUDA_VISIBLE_DEVICES here.
# Select GPU by --device, e.g.:
# python -m baseline_attribution.generate_explanation_maps_clip_vitl --device 1
# Keep CUDA runtime device order consistent with nvidia-smi indices.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import subprocess


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

    # Align CUDA index order with nvidia-smi (physical index order).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Reorder visible devices so requested physical GPU appears as logical GPU 0.
    # This avoids runtime enumeration mismatch across frameworks.
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
        else:
            if inherited != "":
                print(
                    "[bootstrap] Keep inherited CUDA_VISIBLE_DEVICES={} (nvidia-smi query unavailable).".format(
                        inherited
                    )
                )
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return args.device


_BOOTSTRAP_DEVICE = _bootstrap_device_from_argv()

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import clip

from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import torch
from torchvision import transforms

import tensorflow as tf
from utils import *

tf.config.run_functions_eagerly(True)

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CLIP"
net_mode  = "CLIP" # "resnet", vgg

if mode == "CLIP":
    if net_mode == "CLIP":
        img_size = 224
        dataset_index = "datasets/imagenet/val_clip_vitl_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-true")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 100
    mkdir(SAVE_PATH)

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def equip_semantic_modal(self, modal_list):
        text = clip.tokenize(modal_list).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
    def forward(self, vision_inputs):
        
        # with torch.no_grad():
        image_features = self.model.encode_image(vision_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.text_features.T).softmax(dim=-1)
        return scores.float()

data_transform = transforms.Compose(
        [
            transforms.Resize(
                (224,224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def load_and_transform_vision_data(image_paths, device, channel_first=False):
    if image_paths is None:
        return None

    image_outputs = []
    
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    image_outputs = torch.stack(image_outputs, dim=0)
    if channel_first:
        pass
    else:
        image_outputs = image_outputs.permute(0,2,3,1)
    return image_outputs.cpu().numpy()   

def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            
            with torch.no_grad():
                class_embeddings = model.model.encode_text(texts)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights*100


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
    parser = argparse.ArgumentParser(description="Generate CLIP attribution maps.")
    parser.add_argument(
        "--device",
        type=int,
        default=_BOOTSTRAP_DEVICE,
        help="Physical GPU index from nvidia-smi. Set -1 for CPU.",
    )
    parser.add_argument(
        "--tf-memory-limit",
        type=int,
        default=2048,
        help="TensorFlow per-process GPU memory limit in MB.",
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


def configure_tensorflow_gpu(device_index, tf_memory_limit):
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if len(gpus) == 0:
        print("Warning: no TensorFlow GPU device found, skip TF GPU config.")
        return

    if device_index < 0:
        tf.config.set_visible_devices([], "GPU")
        print("TensorFlow GPU disabled (CPU mode).")
        return

    if device_index >= len(gpus):
        raise ValueError(
            "TensorFlow sees {} GPUs, but --device={} was requested.".format(
                len(gpus), device_index
            )
        )

    target_gpu = gpus[device_index]
    try:
        tf.config.set_visible_devices(target_gpu, "GPU")
        tf.config.experimental.set_virtual_device_configuration(
            target_gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=tf_memory_limit)],
        )
        print("TensorFlow target GPU index: {}".format(device_index))
    except RuntimeError as err:
        print("Warning: TensorFlow GPU visibility was already initialized: {}".format(err))


def main(args):
    print("Requested physical --device={}".format(args.device))
    print("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER", "<not set>")))
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")))
    visible_device_index = map_physical_to_visible_device_index(args.device)
    device = resolve_device(visible_device_index)
    configure_tensorflow_gpu(visible_device_index, args.tf_memory_limit)
    print("Selected visible logical GPU index={}".format(visible_device_index))
    print("Torch device={}".format(device))
    # Load model
    vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP", device=device)
    vis_model.eval()
    vis_model.to(device)
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    semantic_feature = load_or_build_semantic_feature(
        vis_model, device, semantic_path
    )

    vis_model.text_features = semantic_feature
    
    wrapped_model = TorchWrapper(vis_model.eval(), device)
    
    batch_size = 32
    
    # define explainers
    explainers = [
        # Saliency(model),
        # GradientInput(model),
        # GuidedBackprop(model),
        # IntegratedGradients(wrapped_model, steps=80, batch_size=batch_size),
        # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(model, nb_samples=80, batch_size=batch_size),
        # VarGrad(model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # GradCAMPP(model),
        # Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # Rise(model, nb_samples=500, batch_size=batch_size),
        # SobolAttributionMethod(model, batch_size=batch_size),
        HsicAttributionMethod(wrapped_model, batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        # Lime(model, nb_samples = 1000),
        # KernelShap(wrapped_model, nb_samples = 1000, batch_size=32)
    ]
    
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
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            image_names = input_data[step * batch : step * batch + batch]
            X_raw = load_and_transform_vision_data(image_names, device)

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = np.eye(class_number)[Y_true]
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name in zip(explanations, image_names):
                mkdir(exp_save_path)
                np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".JPEG", "")), explanation)
    
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
