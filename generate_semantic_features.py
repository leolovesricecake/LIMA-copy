import argparse
import os

import clip
import torch

from utils import imagenet_classes, imagenet_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CLIP semantic features for ImageNet.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-L/14",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt",
        help="Output path for semantic feature tensor.",
    )
    parser.add_argument(
        "--download-root",
        type=str,
        default=".checkpoints/CLIP",
        help="CLIP checkpoint cache directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to encode text features.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Final scale factor for semantic features.",
    )
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        return "cpu"
    return device


def zeroshot_classifier(clip_model, classnames, templates, device, scale=100.0):
    with torch.no_grad():
        zeroshot_weights = []
        total = len(classnames)
        for idx, classname in enumerate(classnames, start=1):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

            if idx % 100 == 0 or idx == total:
                print("Encoded {}/{} classes".format(idx, total))

        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights * scale


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print("Using device: {}".format(device))
    print("Loading CLIP model: {}".format(args.model_name))

    model, _ = clip.load(args.model_name, device=device, download_root=args.download_root)
    model.eval()

    print(
        "Building semantic features with {} classes and {} templates...".format(
            len(imagenet_classes), len(imagenet_templates)
        )
    )
    semantic_feature = zeroshot_classifier(
        model,
        imagenet_classes,
        imagenet_templates,
        device=device,
        scale=args.scale,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(semantic_feature.cpu(), args.output_path)
    print("Saved semantic feature: {}".format(args.output_path))
    print("Shape: {}".format(tuple(semantic_feature.shape)))


if __name__ == "__main__":
    main()
