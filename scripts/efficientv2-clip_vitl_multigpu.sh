#!/bin/bash

dataset="datasets/imagenet/ILSVRC2012_img_val"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"
lambda1=0
lambda2=0.05
lambda3=20
lambda4=1
superpixel_algorithm="seeds"
pending_samples=8

# Respect external CUDA_VISIBLE_DEVICES if provided.
# Example:
# CUDA_VISIBLE_DEVICES=1 bash scripts/efficientv2-clip_vitl_multigpu.sh
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
else
    declare -a cuda_devices=("0" "1")
fi

# trim spaces and drop empty items
filtered_devices=()
for device in "${cuda_devices[@]}"
do
    device="${device//[[:space:]]/}"
    if [[ -n "$device" ]]; then
        filtered_devices+=("$device")
    fi
done
cuda_devices=("${filtered_devices[@]}")

if [[ ${#cuda_devices[@]} -eq 0 ]]; then
    echo "No valid CUDA devices found."
    exit 1
fi

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."
echo "CUDA device list: ${cuda_devices[*]}"

# text length
line_count=$(wc -l < "$eval_list")
echo "Evaluation on $line_count instances."

line_count_per_gpu=$(( (line_count + gpu_numbers - 1) / gpu_numbers ))
echo "Each GPU should process at least $line_count_per_gpu lines."

gpu_index=0
for device in "${cuda_devices[@]}"
do
    begin=$((gpu_index * line_count_per_gpu))
    if [ $gpu_index -eq $((gpu_numbers - 1)) ]; then
        end=-1  # 最后一个 GPU，设置 end 为 -1
    else
        end=$((begin + line_count_per_gpu))
    fi

    CUDA_VISIBLE_DEVICES=$device python -m submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --lambda3 $lambda3 \
    --lambda4 $lambda4 \
    --superpixel-algorithm $superpixel_algorithm \
    --pending-samples $pending_samples \
    --begin $begin \
    --end $end &

    gpu_index=$((gpu_index + 1))
done

wait
echo "All processes have completed."
