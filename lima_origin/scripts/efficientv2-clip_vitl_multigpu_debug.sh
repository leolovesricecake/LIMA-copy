#!/bin/bash
set -euo pipefail

# Debug preset for ImageNet false subset.
# All GPU binding/runtime safeguards are delegated to the unified launcher.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/efficientv2-clip_vitl_multigpu.sh" \
  --dataset "${LIMA_DATASET:-/dev/shm/imagenet_val_fast}" \
  --eval-list "${LIMA_EVAL_LIST:-datasets/imagenet/val_clip_vitl_2k_false.txt}" \
  --save-dir "${LIMA_SAVE_DIR:-submodular_results/imagenet-clip-vitl-efficientv2-debug}" \
  --superpixel-algorithm "${LIMA_SUPERPIXEL:-slico}" \
  --lambda1 "${LIMA_LAMBDA1:-0}" \
  --lambda2 "${LIMA_LAMBDA2:-0.05}" \
  --lambda3 "${LIMA_LAMBDA3:-10}" \
  --lambda4 "${LIMA_LAMBDA4:-1}" \
  --pending-samples "${LIMA_PENDING_SAMPLES:-8}" \
  --resume-check "${LIMA_RESUME_CHECK:-strict}" \
  --cuda-devices "${LIMA_CUDA_DEVICES:-0,1,3,6,7}" \
  "$@"
