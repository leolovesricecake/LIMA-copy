#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash baselines/inseq/run_inseq_llm_baselines.sh /path/to/hf-model cuda:0
# Optional environment overrides:
#   DATASETS="eraser,emotion,imdb,rtn,sst2"
#   METHODS="saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,reagent"
#   METHODS="lime"  # Inseq native LIME; slower
#   MAX_SAMPLES=100
#   N_SAMPLES=32
#   SPLIT=validation
#   DTYPE=bfloat16
#   MAX_LENGTH=2048
#   K=8
#   BASE_SAVE_DIR=results
#   SAVE_DIR=baselines/inseq

MODEL_PATH="${1:?MODEL_PATH is required}"
DEVICE="${2:-cuda:0}"

cmd=(
  python baselines/inseq/run_inseq_llm_baselines.py
  --datasets "${DATASETS:-eraser,emotion,imdb,rtn,sst2}"
  --split "${SPLIT:-validation}"
  --model-path "${MODEL_PATH}"
  --device "${DEVICE}"
  --dtype "${DTYPE:-bfloat16}"
  --max-length "${MAX_LENGTH:-2048}"
  --methods "${METHODS:-saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,reagent}"
  --n-samples "${N_SAMPLES:-32}"
  --k "${K:-8}"
  --base-save-dir "${BASE_SAVE_DIR:-results}"
  --save-dir "${SAVE_DIR:-baselines/inseq}"
  --deterministic
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi

"${cmd[@]}"
