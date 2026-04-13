#!/bin/bash
set -u

dataset="/dev/shm/imagenet_val_fast"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"
lambda1=0
lambda2=0.05
lambda3=20
lambda4=1
superpixel_algorithm="seeds"
pending_samples=8
resume_check=${LIMA_RESUME_CHECK:-strict}

# Select target GPU indices for subprocesses.
# 1) default: declare -a cuda_devices=("0" "1")
# 2) override by env: LIMA_CUDA_DEVICES="0 1 2 3" bash scripts/efficientv2-clip_vitl_multigpu.sh
declare -a cuda_devices=("0" "1" "3" "6" "7")
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
THREADS_PER_WORKER=${LIMA_THREADS_PER_WORKER:-1}
ENABLE_NUMA_BIND=${LIMA_ENABLE_NUMA_BIND:-1}

find_numa_node_for_gpu() {
    local gpu_index="$1"
    local bus_id
    bus_id=$(
        nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v gpu="$gpu_index" '
            {
                gsub(/^[ \t]+|[ \t]+$/, "", $1)
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                if ($1 == gpu) {
                    print tolower($2)
                    exit
                }
            }'
    )
    if [[ -z "${bus_id}" ]]; then
        return 1
    fi
    local numa_file="/sys/bus/pci/devices/${bus_id}/numa_node"
    if [[ ! -r "${numa_file}" ]]; then
        return 1
    fi
    local node
    node=$(cat "${numa_file}" 2>/dev/null || true)
    if [[ -z "${node}" || "${node}" == "-1" ]]; then
        return 1
    fi
    echo "${node}"
}

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
log_root=${LIMA_LOG_DIR:-"./logs/efficientv2_clip_vitl_${timestamp}"}
mkdir -p "$log_root"
echo "Log directory: $log_root"

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."
echo "Device list: ${cuda_devices[*]}"
echo "Resume check mode: ${resume_check}"
echo "Threads per worker: ${THREADS_PER_WORKER}"
echo "NUMA bind enabled: ${ENABLE_NUMA_BIND}"

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
    launch_prefix=()
    if [[ "${ENABLE_NUMA_BIND}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
        numa_node=$(find_numa_node_for_gpu "$device" || true)
        if [[ -n "${numa_node}" ]]; then
            launch_prefix=(numactl --cpunodebind="${numa_node}" --membind="${numa_node}")
            echo "  numa bind: node ${numa_node}"
        else
            echo "  numa bind: skipped (numa node not found for gpu ${device})"
        fi
    else
        echo "  numa bind: disabled"
    fi

    setsid env \
    OMP_NUM_THREADS="${THREADS_PER_WORKER}" \
    MKL_NUM_THREADS="${THREADS_PER_WORKER}" \
    OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}" \
    NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}" \
    VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}" \
    PYTHONUNBUFFERED=1 \
    "${launch_prefix[@]}" \
    python -m submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --lambda3 $lambda3 \
    --lambda4 $lambda4 \
    --superpixel-algorithm $superpixel_algorithm \
    --pending-samples $pending_samples \
    --device $device \
    --resume-check $resume_check \
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
