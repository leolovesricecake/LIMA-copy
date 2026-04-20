# -*- coding: utf-8 -*-

"""
Created on 2024/6/3

@author: Ruoyu Chen
CLIP ViT version
"""

import argparse

import scipy
import os
# Keep CUDA runtime device order consistent with nvidia-smi indices.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU by --device physical index while keeping all GPUs visible.
# Clear inherited masks to avoid visible-index remapping ambiguity.
_inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _inherited_visible not in ("", "-1"):
    print("[bootstrap] Clear inherited CUDA_VISIBLE_DEVICES={}.".format(_inherited_visible))
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import subprocess
import ctypes
import ctypes.util
import re
import cv2
import json
import imageio
import numpy as np
import uuid
from PIL import Image

from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
# plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

red_tr = get_alpha_cmap('Reds')

import torch
from torchvision import transforms
import clip
from models.submodular_vit_efficient import MultiModalSubModularExplanationEfficientV2

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

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_clip_vitl_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=0.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=0.05,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda4', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--pending-samples',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=None,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default=None,
                        help='output directory to save results')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='Physical GPU index from nvidia-smi. Set -1 for CPU.')
    parser.add_argument('--resume-check',
                        type=str,
                        default='strict',
                        choices=['strict', 'exists-only'],
                        help="Resume mode: strict checks json+npy readability, exists-only only checks json existence.")
    parser.add_argument('--allow-device-fallback',
                        action='store_true',
                        help='Allow fallback to ordinal==physical index when PCI bus-id mapping is unavailable.')
    parser.add_argument('--num-shards',
                        type=int,
                        default=1,
                        help='Total number of workers for data-level sharding.')
    parser.add_argument('--shard-id',
                        type=int,
                        default=-1,
                        help='Current worker id in [0, num-shards). -1 disables sharding.')
    args = parser.parse_args()
    return args

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features


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

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image = data_transform(image)
    return image


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


def resolve_cuda_ordinal(physical_device_index, allow_fallback=False):
    if physical_device_index < 0:
        return -1

    nvidia_bus = _query_nvidia_smi_bus_map()
    cuda_bus = _query_cuda_ordinal_bus_map()
    available_physical = sorted(list(nvidia_bus.keys()))
    if physical_device_index not in nvidia_bus:
        raise ValueError(
            "Requested physical GPU {} not found. nvidia-smi available indices: {}.".format(
                physical_device_index, available_physical
            )
        )

    target_bus = nvidia_bus.get(physical_device_index)
    if len(cuda_bus) == 0:
        if not allow_fallback:
            raise RuntimeError(
                "Failed to query CUDA runtime PCI bus map; refuse fallback to avoid wrong GPU binding. "
                "Use --allow-device-fallback only if you accept ordinal-based best effort."
            )
        mapped_ordinal = physical_device_index
        print(
            "Warning: CUDA runtime PCI bus map unavailable; fallback physical {} -> ordinal {}.".format(
                physical_device_index, mapped_ordinal
            )
        )
        return mapped_ordinal

    mapped_ordinal = None
    for ordinal, bus in cuda_bus.items():
        if bus == target_bus:
            mapped_ordinal = ordinal
            break

    if mapped_ordinal is None and not allow_fallback:
        raise RuntimeError(
            "No CUDA ordinal matches physical GPU {} (bus {}). CUDA ordinal->bus map: {}. "
            "Refuse fallback to avoid wrong GPU binding. Use --allow-device-fallback if needed.".format(
                physical_device_index, target_bus, cuda_bus
            )
        )
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


def probe_device_ready(cuda_ordinal):
    if cuda_ordinal < 0:
        return
    try:
        torch.cuda.set_device(cuda_ordinal)
        _ = torch.empty((1,), device="cuda:{}".format(cuda_ordinal))
        torch.cuda.synchronize(cuda_ordinal)
    except Exception as exc:
        raise RuntimeError(
            "CUDA ordinal {} is not ready (busy/unavailable/runtime mismatch): {}".format(
                cuda_ordinal, repr(exc)
            )
        ) from exc


def _build_output_paths(save_npy_root_path, save_json_root_path, gt_id, image_relative_path):
    base_name = os.path.splitext(image_relative_path)[0]
    npy_path = os.path.join(
        os.path.join(save_npy_root_path, gt_id), f"{base_name}.npy"
    )
    json_path = os.path.join(
        os.path.join(save_json_root_path, gt_id), f"{base_name}.json"
    )
    return npy_path, json_path


def _parse_eval_info_line(info_line):
    parts = info_line.split()
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _is_valid_json(json_path):
    try:
        with open(json_path, "r") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _is_valid_npy(npy_path):
    try:
        arr = np.load(npy_path, mmap_mode="r")
        _ = arr.shape
        return True
    except Exception:
        return False


def is_sample_completed(npy_path, json_path, resume_check_mode):
    if resume_check_mode == "exists-only":
        return os.path.exists(json_path)

    if not (os.path.exists(npy_path) and os.path.exists(json_path)):
        return False
    if not _is_valid_json(json_path):
        return False
    if not _is_valid_npy(npy_path):
        return False
    return True


def atomic_save_json(json_path, payload):
    parent = os.path.dirname(json_path)
    mkdir(parent)
    tmp_path = "{}.tmp.{}".format(json_path, uuid.uuid4().hex)
    try:
        with open(tmp_path, "w") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=4, separators=(',', ':')))
        os.replace(tmp_path, json_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_save_npy(npy_path, array):
    parent = os.path.dirname(npy_path)
    mkdir(parent)
    tmp_path = "{}.tmp.{}".format(npy_path, uuid.uuid4().hex)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, npy_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main(args):
    start_time_all = time.time()
    # Model Init
    print("Requested physical --device={}".format(args.device))
    print("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER", "<not set>")))
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")))
    cuda_ordinal = resolve_cuda_ordinal(args.device, allow_fallback=args.allow_device_fallback)
    device = resolve_device(cuda_ordinal)
    probe_device_ready(cuda_ordinal)
    print("Selected CUDA ordinal={}".format(cuda_ordinal))
    print("Torch device={}".format(device))
    # Instantiate model
    vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP", device=device)
    vis_model.eval()
    vis_model.to(device)
    print("load CLIP model")
    model_ready_time = time.time()
    print("[time] model_init={:.2f}s".format(model_ready_time - start_time_all))
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    semantic_feature = load_or_build_semantic_feature(
        vis_model, device, semantic_path
    )
    semantic_ready_time = time.time()
    print("[time] semantic_ready={:.2f}s".format(semantic_ready_time - model_ready_time))
    
    smdl = MultiModalSubModularExplanationEfficientV2(
        vis_model, semantic_feature, transform_vision_data, device=device, 
        lambda1=args.lambda1, 
        lambda2=args.lambda2, 
        lambda3=args.lambda3, 
        lambda4=args.lambda4,
        pending_samples=args.pending_samples)
    
    with open(args.eval_list, "r") as f:
        infos = [line.strip() for line in f if line.strip() != ""]
    
    save_dir = args.save_dir
    if not save_dir:
        dataset_name = os.path.basename(args.eval_list).split(".")[0]
        save_dir = f'./submodular_results/{dataset_name}-clip-vitl-efficientv2'
    save_dir = os.path.join(save_dir, "{}-{}-{}-{}-{}-pending-samples-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.lambda4, args.pending_samples))  
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_infos = infos[args.begin : end]

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1.")
    if args.shard_id >= 0:
        if args.shard_id >= args.num_shards:
            raise ValueError("--shard-id must be in [0, --num-shards).")
        sharded_infos = []
        for idx, info in enumerate(select_infos):
            if idx % args.num_shards == args.shard_id:
                sharded_infos.append(info)
        select_infos = sharded_infos
        print(
            "[shard] shard-id={}/{} selected {} samples after sharding.".format(
                args.shard_id, args.num_shards, len(select_infos)
            )
        )
    else:
        print("[shard] disabled, selected {} samples.".format(len(select_infos)))

    pending_infos = []
    completed_count = 0
    malformed_count = 0
    for info in select_infos:
        image_relative_path, gt_id = _parse_eval_info_line(info)
        if image_relative_path is None:
            malformed_count += 1
            continue
        npy_path, json_path = _build_output_paths(
            save_npy_root_path, save_json_root_path, gt_id, image_relative_path
        )
        if is_sample_completed(npy_path, json_path, args.resume_check):
            completed_count += 1
        else:
            pending_infos.append(info)

    print(
        "[resume] mode={} selected={} completed={} pending={} malformed={}.".format(
            args.resume_check,
            len(select_infos),
            completed_count,
            len(pending_infos),
            malformed_count,
        )
    )
    resume_scan_done_time = time.time()
    print("[time] resume_scan={:.2f}s".format(resume_scan_done_time - semantic_ready_time))

    if len(pending_infos) == 0:
        print("[resume] no pending samples, worker exits.")
        return

    processed_count = 0
    for info in tqdm(
        pending_infos,
        desc="gpu{}-pending".format(args.device),
        dynamic_ncols=True,
    ):
        image_relative_path, gt_id = _parse_eval_info_line(info)
        if image_relative_path is None:
            print("Warning: malformed eval-list line, skip: {}".format(info))
            continue

        npy_path, json_path = _build_output_paths(
            save_npy_root_path, save_json_root_path, gt_id, image_relative_path
        )
        
        # Ground Truth Label
        gt_label = int(gt_id)
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
        smdl.k = len(element_sets_V)

    #     start = time.time()
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, gt_label)
    #     end = time.time()
    #     # print('程序执行时间: ',end - start)
        
        # Save the final image
        # save_image_root_path = os.path.join(save_dir, "image")
        # mkdir(save_image_root_path)
        # mkdir(os.path.join(save_image_root_path, gt_id))
        # save_image_path = os.path.join(
        #     save_image_root_path, image_relative_path)
        # cv2.imwrite(save_image_path, submodular_image)

        # Save npy file
        atomic_save_npy(npy_path, np.array(submodular_image_set))

        # Save json file
        atomic_save_json(json_path, saved_json_file)
        processed_count += 1

    #     # Save GIF
    #     save_gif_root_path = os.path.join(save_dir, "gif")
    #     mkdir(save_gif_root_path)
    #     save_gif_path = os.path.join(save_gif_root_path, gt_id)
    #     mkdir(save_gif_path)

        # img_frame = submodular_image_set[0][..., ::-1]
        # frames = []
        # frames.append(img_frame)
        # for fps in range(1, submodular_image_set.shape[0]):
        #     img_frame = img_frame.copy() + submodular_image_set[fps][..., ::-1]
        #     frames.append(img_frame)

        # imageio.mimsave(os.path.join(save_gif_root_path, image_relative_path.replace(".jpg", ".gif")), 
        #                       frames, 'GIF', duration=0.0085)

    print(
        "[done] shard-id={} processed={} samples (pending at start={}).".format(
            args.shard_id, processed_count, len(pending_infos)
        )
    )
    print("[time] main_loop={:.2f}s total={:.2f}s".format(
        time.time() - resume_scan_done_time, time.time() - start_time_all
    ))


if __name__ == "__main__":
    args = parse_args()
    
    main(args)
