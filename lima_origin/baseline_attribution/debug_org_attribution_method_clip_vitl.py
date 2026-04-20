import os
# Use PCI_BUS_ID order to align CUDA ordinal with nvidia-smi bus ordering as much as possible.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The user wants to select physical GPU by --device while seeing all devices.
# Remove inherited visibility masks before importing torch/CUDA libs.
_inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _inherited_visible not in ("", "-1"):
    print("[bootstrap] Clear inherited CUDA_VISIBLE_DEVICES={}.".format(_inherited_visible))
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import argparse
import subprocess
import ctypes
import ctypes.util
import re
import cv2
import math
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt

from tqdm import tqdm
import json
from utils import *

import torch
from torchvision import transforms
import clip

results_save_root = "./explanation_insertion_results"
# TODO 手动修改explanation_method为需要调试的解释方法
# explanation_method = "explanation_results/imagenet-clip-vitl-true/HsicAttributionMethod"
explanation_method = "explanation_results/imagenet-clip-vitl-true/GradECLIP"
image_root_path = "datasets/imagenet/ILSVRC2012_img_val"
eval_list = "datasets/imagenet/val_clip_vitl_5k_true.txt"
save_doc = "imagenet-clip-vitl-true"
steps = 50
batch_size = 10
image_size_ = 224

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
        
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.text_features.T).softmax(dim=-1)
        return scores.float()


def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = model.model.encode_text(texts)
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

def preprocess_input(images):
    """
    Input:
        image: An image read by opencv [b,w,h,c]
    Output:
        outputs: After preproccessing, is a tensor [c,w,h]
    """
    outputs = []
    for image in images:
        image = Image.fromarray(image)
        image = data_transform(image)
        outputs.append(image)
    return torch.stack(outputs)

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Debug insertion/deletion curves for CLIP ViT-L.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Physical GPU index from nvidia-smi. Set -1 for CPU.",
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


def resolve_device(physical_device_index):
    if physical_device_index < 0 or not torch.cuda.is_available():
        return "cpu"

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

    gpu_count = torch.cuda.device_count()
    if mapped_ordinal >= gpu_count:
        raise ValueError(
            "Mapped CUDA ordinal {} out of range, available ordinals: 0..{}.".format(
                mapped_ordinal, gpu_count - 1
            )
        )

    torch.cuda.set_device(mapped_ordinal)
    return "cuda:{}".format(mapped_ordinal)


def main(args):
    print("Requested physical --device={}".format(args.device))
    print("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER", "<not set>")))
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")))
    device = resolve_device(args.device)
    print("Torch device={}".format(device))
    
    mkdir(results_save_root)
    save_dir = os.path.join(results_save_root, save_doc)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, explanation_method.split("/")[-1])
    mkdir(save_dir)

    model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP", device=device)
    model.eval()
    model.to(device)
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    semantic_feature = load_or_build_semantic_feature(
        model, device, semantic_path
    )

    model.text_features = semantic_feature

    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    for info in tqdm(infos[:]):
        json_file = {}
        class_index = int(info.split(" ")[-1])
        # if class_index!=408:
        #     continue
        image_path = os.path.join(image_root_path, info.split(" ")[0])

        mask_path = os.path.join(explanation_method, info.split(" ")[0].replace(".JPEG", ".npy"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size_, image_size_))
        try:
            explanation = np.load(mask_path)
        except:
            continue

        insertion_explanation_images = []
        deletion_explanation_images = []
        for i in range(1, steps+1):
            perturbed_rate = i / steps
            insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
            deletion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "deletion"))
        
        insertion_explanation_images_input = preprocess_input(
            np.array(insertion_explanation_images)
        ).to(device)
        deletion_explanation_images_input = preprocess_input(
            np.array(deletion_explanation_images)
        ).to(device)

        batch_step = math.ceil(
            insertion_explanation_images_input.shape[0] / batch_size)
        
        insertion_data = []
        deletion_data = []
        for j in range(batch_step):
            insertion_explanation_images_input_results = model(
                insertion_explanation_images_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            insertion_data += insertion_explanation_images_input_results.cpu().numpy().tolist()
            
            deletion_explanation_images_input_results = model(
                deletion_explanation_images_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            deletion_data += deletion_explanation_images_input_results.cpu().numpy().tolist()
        
        json_file["consistency_score"] = insertion_data
        json_file["collaboration_score"] = deletion_data
        
        save_path = os.path.join(
            save_dir, info.split(" ")[0].replace(".JPEG", ".json")
        )
        with open(save_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
