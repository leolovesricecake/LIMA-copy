# -*- coding: utf-8 -*-  

"""
Created on 2024/6/3

@author: Ruoyu Chen
"""

import os
# Keep CUDA runtime device order consistent with nvidia-smi indices.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU by --device physical index while keeping all GPUs visible.
# Clear inherited masks to avoid visible-index remapping ambiguity.
_inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _inherited_visible not in ("", "-1"):
    print("[bootstrap] Clear inherited CUDA_VISIBLE_DEVICES={}.".format(_inherited_visible))
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import argparse
import subprocess
import ctypes
import ctypes.util
import re

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
from dataset_config import (
    build_result_file_path,
    parse_eval_list,
    resolve_dataset_config,
)

tf.config.run_functions_eagerly(True)

mode = "CLIP"
net_mode  = "CLIP" # "resnet", vgg
if mode == "CLIP":
    if net_mode == "CLIP":
        img_size = 224
    # elif net_mode == "languagebind":
    else:
        raise ValueError("Unsupported net_mode: {}".format(net_mode))

        
    class_number = 1000
    batch = 100


DEFAULT_SAVE_ROOT = "explanation_results"


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
        default=0,
        help="Physical GPU index from nvidia-smi. Set -1 for CPU.",
    )
    parser.add_argument(
        "--tf-memory-limit",
        type=int,
        default=2048,
        help="TensorFlow per-process GPU memory limit in MB.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="imagenet",
        help="Dataset preset name: imagenet / imagenet-false / imagenet-a / imagenet-o.",
    )
    parser.add_argument(
        "--dataset-index",
        type=str,
        default=None,
        help="Optional override for eval-list path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for dataset image root path.",
    )
    parser.add_argument(
        "--save-doc",
        type=str,
        default=None,
        help="Optional override for dataset save-doc name.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=DEFAULT_SAVE_ROOT,
        help="Root output directory for explanation maps.",
    )
    return parser.parse_args()


def _normalize_pci_bus_id(bus_id):
    if bus_id is None:
        return None
    text = bus_id.strip().upper()
    match = re.match(r"^([0-9A-F]+):([0-9A-F]+):([0-9A-F]+)\.([0-9A-F]+)$", text)
    if match is None:
        return text
    domain, bus, device, function = match.groups()
    return "{:04X}:{:02X}:{:02X}.{:1X}".format(
        int(domain, 16), int(bus, 16), int(device, 16), int(function, 16)
    )


def _query_nvidia_smi_bus_map():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except Exception:
        return {}

    result = {}
    for line in output.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        if not parts[0].isdigit():
            continue
        result[int(parts[0])] = _normalize_pci_bus_id(parts[1])
    return result


def _load_cudart():
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _query_cuda_ordinal_bus_map():
    cudart = _load_cudart()
    if cudart is None:
        return {}

    device_count = ctypes.c_int(0)
    if cudart.cudaGetDeviceCount(ctypes.byref(device_count)) != 0:
        return {}

    mapping = {}
    get_bus_id = cudart.cudaDeviceGetPCIBusId
    get_bus_id.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    get_bus_id.restype = ctypes.c_int

    for ordinal in range(device_count.value):
        buffer = ctypes.create_string_buffer(64)
        if get_bus_id(buffer, 64, ordinal) == 0:
            mapping[ordinal] = _normalize_pci_bus_id(buffer.value.decode("utf-8"))
    return mapping


def resolve_cuda_ordinal(physical_device_index):
    if physical_device_index < 0:
        return -1

    nvidia_bus = _query_nvidia_smi_bus_map()
    cuda_bus = _query_cuda_ordinal_bus_map()
    target_bus = nvidia_bus.get(physical_device_index)

    mapped_ordinal = None
    if target_bus is not None and len(cuda_bus) > 0:
        for ordinal, bus in cuda_bus.items():
            if bus == target_bus:
                mapped_ordinal = ordinal
                break

    if mapped_ordinal is None:
        mapped_ordinal = physical_device_index
        print(
            "Warning: failed to map by PCI bus-id; fallback to ordinal mapping physical {} -> ordinal {}.".format(
                physical_device_index, mapped_ordinal
            )
        )
    else:
        print(
            "PCI bus-id mapping: physical GPU {} (bus {}) -> CUDA ordinal {}.".format(
                physical_device_index, target_bus, mapped_ordinal
            )
        )
    return mapped_ordinal


def resolve_device(cuda_ordinal):
    if cuda_ordinal < 0 or not torch.cuda.is_available():
        return "cpu"
    gpu_count = torch.cuda.device_count()
    if cuda_ordinal >= gpu_count:
        raise ValueError(
            "Invalid mapped CUDA ordinal {}. Available ordinals: 0..{}.".format(
                cuda_ordinal, gpu_count - 1
            )
        )
    torch.cuda.set_device(cuda_ordinal)
    return "cuda:{}".format(cuda_ordinal)


def configure_tensorflow_gpu(cuda_ordinal, tf_memory_limit):
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if len(gpus) == 0:
        print("Warning: no TensorFlow GPU device found, skip TF GPU config.")
        return

    if cuda_ordinal < 0:
        tf.config.set_visible_devices([], "GPU")
        print("TensorFlow GPU disabled (CPU mode).")
        return

    if cuda_ordinal >= len(gpus):
        raise ValueError(
            "TensorFlow sees {} GPUs, but mapped CUDA ordinal={} was requested.".format(
                len(gpus), cuda_ordinal
            )
        )

    target_gpu = gpus[cuda_ordinal]
    try:
        tf.config.set_visible_devices(target_gpu, "GPU")
        tf.config.experimental.set_virtual_device_configuration(
            target_gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=tf_memory_limit)],
        )
        print("TensorFlow target CUDA ordinal: {}".format(cuda_ordinal))
    except RuntimeError as err:
        print("Warning: TensorFlow GPU visibility was already initialized: {}".format(err))


def main(args):
    dataset_cfg = resolve_dataset_config(
        dataset_name=args.dataset_name,
        eval_list=args.dataset_index,
        image_root_path=args.dataset_path,
        save_doc=args.save_doc,
    )
    save_path = os.path.join(args.save_root, dataset_cfg["save_doc"])
    mkdir(save_path)
    print(
        "dataset_name: {}\n.  dataset_index: {}\n.  dataset_path: {}\n.  SAVE_PATH: {}".format(
            dataset_cfg["dataset_name"],
            dataset_cfg["eval_list"],
            dataset_cfg["image_root_path"],
            save_path,
        )
    )

    print("Requested physical --device={}".format(args.device))
    print("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER", "<not set>")))
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")))
    cuda_ordinal = resolve_cuda_ordinal(args.device)
    device = resolve_device(cuda_ordinal)
    configure_tensorflow_gpu(cuda_ordinal, args.tf_memory_limit)
    print("Selected CUDA ordinal={}".format(cuda_ordinal))
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
    samples = parse_eval_list(dataset_cfg["eval_list"], require_label=True)
    total_steps = math.ceil(len(samples) / batch)
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(save_path, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            batch_samples = samples[step * batch : step * batch + batch]
            image_names = [
                os.path.join(dataset_cfg["image_root_path"], image_rel_path)
                for image_rel_path, _ in batch_samples
            ]
            X_raw = load_and_transform_vision_data(image_names, device)

            Y_true = np.array([label for _, label in batch_samples])
            labels_ohe = np.eye(class_number)[Y_true]
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, (image_rel_path, _) in zip(explanations, batch_samples):
                save_file = build_result_file_path(exp_save_path, image_rel_path, ".npy")
                np.save(save_file, explanation)
    
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
