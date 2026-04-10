#!/bin/bash
set -u

dataset="datasets/imagenet/ILSVRC2012_img_val"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"
lambda1=0
lambda2=0.05
lambda3=20
lambda4=1
superpixel_algorithm="seeds"
pending_samples=8

# Select target GPU indices for subprocesses.
# 1) default: declare -a cuda_devices=("0" "1")
# 2) override by env: LIMA_CUDA_DEVICES="0 1 2 3" bash scripts/efficientv2-clip_vitl_multigpu.sh
declare -a cuda_devices=("0" "1" "4" "6")
if [[ -n "${LIMA_CUDA_DEVICES:-}" ]]; then
    echo "Use devices from env LIMA_CUDA_DEVICES=${LIMA_CUDA_DEVICES}"
    read -r -a cuda_devices <<< "${LIMA_CUDA_DEVICES}"
else
    echo "Use devices from script default cuda_devices: ${cuda_devices[*]}"
fi

if [[ ${#cuda_devices[@]} -eq 0 ]]; then
    echo "No valid CUDA devices found."
    exit 1
fi

declare -a worker_pids=()

cleanup_workers() {
    if [[ ${#worker_pids[@]} -eq 0 ]]; then
        return
    fi
    echo "Stopping ${#worker_pids[@]} worker process(es)..."
    for pid in "${worker_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    wait "${worker_pids[@]}" 2>/dev/null || true
}

on_interrupt() {
    echo "Interrupt received, terminating workers."
    cleanup_workers
    exit 130
}

trap on_interrupt INT TERM

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."
echo "Device list: ${cuda_devices[*]}"

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

    echo "Launch worker on physical GPU ${device}, begin=${begin}, end=${end}"
    setsid python -m submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --lambda3 $lambda3 \
    --lambda4 $lambda4 \
    --superpixel-algorithm $superpixel_algorithm \
    --pending-samples $pending_samples \
    --device $device \
    --resume-check strict \
    --begin $begin \
    --end $end &
    worker_pid="$!"
    worker_pids+=("${worker_pid}")
    echo "  worker pid=${worker_pid}"

    gpu_index=$((gpu_index + 1))
done

wait "${worker_pids[@]}"
trap - INT TERM
echo "All processes have completed."
