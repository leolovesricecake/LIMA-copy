#!/usr/bin/env bash
set -euo pipefail

# Example 1: SST-2 smoke run with mock backbone (fast CI/debug)
# python -m lima_llm \
#   --dataset sst2 \
#   --split validation \
#   --mock-backbone \
#   --k 8 \
#   --chunker sentence \
#   --search greedy \
#   --max-samples 100 \
#   --output-dir lima_llm_results \
#   --run-eval

# Example 2: ERASER Movie Reviews with local HF 7B model
# python -m lima_llm \
#   --dataset eraser_movie_reviews \
#   --split validation \
#   --eraser-root /path/to/eraser \
#   --model-path /path/to/Qwen2.5-7B-Instruct \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --k 8 \
#   --chunker sentence \
#   --search greedy \
#   --output-dir lima_llm_results \
#   --run-eval --eval-gradient-baseline

python -m lima_llm "$@"
