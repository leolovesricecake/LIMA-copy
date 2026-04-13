#!/bin/bash
set -u

dataset="datasets/imagenet/ILSVRC2012_img_val"
eval_list="datasets/imagenet/val_clip_vitl_2k_false.txt"
save_dir="submodular_results/imagenet-clip-vitl-efficientv2-debug/"
lambda1=0
lambda2=0.05
lambda3=10
lambda4=1
pending_samples=8

# Select target GPU indices for subprocesses.
# 1) default: declare -a cuda_devices=("0" "1")
# 2) override by env: LIMA_CUDA_DEVICES="0 1 2 3" bash scripts/efficientv2-clip_vitl_multigpu_debug.sh
declare -a cuda_devices=("0" "1")
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
declare -a worker_logs=()

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

timestamp=$(date +"%Y%m%d_%H%M%S")
log_root=${LIMA_LOG_DIR:-"./logs/efficientv2_clip_vitl_debug_${timestamp}"}
mkdir -p "$log_root"
echo "Log directory: $log_root"

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."
echo "Device list: ${cuda_devices[*]}"

# text length
line_count=$(wc -l < "$eval_list")
echo "Evaluation on $line_count instances."

line_count_per_gpu=$(( (line_count + gpu_numbers - 1) / gpu_numbers ))
echo "Each GPU shard has about $line_count_per_gpu lines before resume filtering."

gpu_index=0
for device in "${cuda_devices[@]}"
do
    begin=0
    end=-1
    log_file="${log_root}/worker_shard${gpu_index}_gpu${device}.log"

    echo "Launch worker shard=${gpu_index}/${gpu_numbers} on physical GPU ${device}"
    echo "  log: ${log_file}"
    setsid python -m submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --lambda3 $lambda3 \
    --lambda4 $lambda4 \
    --pending-samples $pending_samples \
    --device $device \
    --resume-check strict \
    --num-shards $gpu_numbers \
    --shard-id $gpu_index \
    --begin $begin \
    --end $end >"$log_file" 2>&1 &
    worker_pid="$!"
    worker_pids+=("${worker_pid}")
    worker_logs+=("${log_file}")
    echo "  worker pid=${worker_pid}"

    gpu_index=$((gpu_index + 1))
done

failed=0
for idx in "${!worker_pids[@]}"; do
    pid="${worker_pids[$idx]}"
    log_file="${worker_logs[$idx]}"
    if ! wait "$pid"; then
        failed=1
        echo "Worker pid=${pid} failed. See log: ${log_file}"
    fi
done
trap - INT TERM
if [[ "$failed" -eq 0 ]]; then
    echo "All processes have completed."
else
    echo "Some workers failed. Check logs in ${log_root}"
    exit 1
fi
