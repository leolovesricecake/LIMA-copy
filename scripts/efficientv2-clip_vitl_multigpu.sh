#!/bin/bash
set -u

# -----------------------------------------------------------------------------
# Examples:
# 1) ImageNet true subset (default)
#    bash scripts/efficientv2-clip_vitl_multigpu.sh
#      --cuda-devices "0,1,6,7"
#
# 2) ImageNet false subset
#    bash scripts/efficientv2-clip_vitl_multigpu.sh \
#      --eval-list datasets/imagenet/val_clip_vitl_2k_false.txt \
#      --save-dir submodular_results/imagenet-clip-vitl-efficientv2-debug \
#      --superpixel-algorithm slico
#      --lambda3 10
#      --cuda-devices "0,1,6,7"
# 
# 3) ImageNet-A
#    bash scripts/efficientv2-clip_vitl_multigpu.sh \
#      --dataset datasets/ImageNet-A/sample/image \
#      --eval-list datasets/ImageNet-A/imagenet-a_list.txt \
#      --save-dir submodular_results/imagenet-a-clip-vitl-efficientv2 \
#      --cuda-devices "0,1,6,7"
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  bash scripts/efficientv2-clip_vitl_multigpu.sh [OPTIONS]

Options:
  --dataset PATH                 Dataset root path.
  --eval-list PATH               Eval list path.
  --save-dir PATH                Result root path.
  --lambda1 FLOAT                Submodular lambda1.
  --lambda2 FLOAT                Submodular lambda2.
  --lambda3 FLOAT                Submodular lambda3.
  --lambda4 FLOAT                Submodular lambda4.
  --superpixel-algorithm STR     "seeds" or "slico".
  --pending-samples INT          pending_samples for efficient-v2.
  --resume-check STR             strict or exists-only.
  --cuda-devices STR             GPU list, comma/space separated, e.g. "0,1,3,6".
  --threads-per-worker INT       OMP/MKL/BLAS threads per worker.
  --enable-numa-bind INT         1 to enable numactl binding, 0 to disable.
  --log-root PATH                Override log root dir.
  --python-bin BIN               Python executable, default "python".
  --module NAME                  Python module to run.
  --allow-device-fallback        Pass through to python worker.
  --begin INT                    Begin index before sharding.
  --end INT                      End index (exclusive). Use -1 for all.
  -h, --help                     Show this message.
EOF
}

dataset="/dev/shm/imagenet_val_fast"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"
save_dir="submodular_results/imagenet-clip-vitl-efficientv2"
lambda1=0
lambda2=0.05
lambda3=20
lambda4=1
superpixel_algorithm="seeds"
pending_samples=8
resume_check=${LIMA_RESUME_CHECK:-strict}
threads_per_worker=${LIMA_THREADS_PER_WORKER:-1}
enable_numa_bind=${LIMA_ENABLE_NUMA_BIND:-1}
python_bin="python"
module_name="submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel"
allow_device_fallback=0
begin=0
end=-1
log_root=""
cuda_devices_raw=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            dataset="$2"; shift 2 ;;
        --eval-list)
            eval_list="$2"; shift 2 ;;
        --save-dir)
            save_dir="$2"; shift 2 ;;
        --lambda1)
            lambda1="$2"; shift 2 ;;
        --lambda2)
            lambda2="$2"; shift 2 ;;
        --lambda3)
            lambda3="$2"; shift 2 ;;
        --lambda4)
            lambda4="$2"; shift 2 ;;
        --superpixel-algorithm)
            superpixel_algorithm="$2"; shift 2 ;;
        --pending-samples)
            pending_samples="$2"; shift 2 ;;
        --resume-check)
            resume_check="$2"; shift 2 ;;
        --cuda-devices)
            cuda_devices_raw="$2"; shift 2 ;;
        --threads-per-worker)
            threads_per_worker="$2"; shift 2 ;;
        --enable-numa-bind)
            enable_numa_bind="$2"; shift 2 ;;
        --log-root)
            log_root="$2"; shift 2 ;;
        --python-bin)
            python_bin="$2"; shift 2 ;;
        --module)
            module_name="$2"; shift 2 ;;
        --allow-device-fallback)
            allow_device_fallback=1; shift ;;
        --begin)
            begin="$2"; shift 2 ;;
        --end)
            end="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1 ;;
    esac
done

if [[ ! -f "$eval_list" ]]; then
    echo "Eval list not found: $eval_list"
    exit 1
fi
if [[ ! -d "$dataset" ]]; then
    echo "Dataset root not found: $dataset"
    exit 1
fi

# Select target GPU indices for subprocesses.
# Priority:
# 1) --cuda-devices
# 2) env LIMA_CUDA_DEVICES
# 3) script default
declare -a cuda_devices=("0" "1" "3" "6" "7")
if [[ -n "${cuda_devices_raw}" ]]; then
    cuda_devices_raw="${cuda_devices_raw//,/ }"
    read -r -a cuda_devices <<< "${cuda_devices_raw}"
    echo "Use devices from --cuda-devices: ${cuda_devices[*]}"
elif [[ -n "${LIMA_CUDA_DEVICES:-}" ]]; then
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
if [[ -z "${log_root}" ]]; then
    log_root=${LIMA_LOG_DIR:-"./logs/efficientv2_clip_vitl_${timestamp}"}
fi
mkdir -p "$log_root"
echo "Log directory: $log_root"

gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."
echo "Device list: ${cuda_devices[*]}"
echo "Dataset root: $dataset"
echo "Eval list: $eval_list"
echo "Save dir: $save_dir"
echo "Hyper-params: superpixel=${superpixel_algorithm}, lambda=[${lambda1},${lambda2},${lambda3},${lambda4}], pending_samples=${pending_samples}"
echo "Resume check mode: ${resume_check}"
echo "Threads per worker: ${threads_per_worker}"
echo "NUMA bind enabled: ${enable_numa_bind}"

line_count=$(wc -l < "$eval_list")
echo "Evaluation on $line_count instances."

line_count_per_gpu=$(( (line_count + gpu_numbers - 1) / gpu_numbers ))
echo "Each GPU shard has about $line_count_per_gpu lines before resume filtering."

gpu_index=0
for device in "${cuda_devices[@]}"
do
    log_file="${log_root}/worker_shard${gpu_index}_gpu${device}.log"

    echo "Launch worker shard=${gpu_index}/${gpu_numbers} on physical GPU ${device}"
    echo "  log: ${log_file}"
    launch_prefix=()
    if [[ "${enable_numa_bind}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
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

    allow_fallback_flag=()
    if [[ "$allow_device_fallback" == "1" ]]; then
        allow_fallback_flag=(--allow-device-fallback)
    fi

    setsid env \
    OMP_NUM_THREADS="${threads_per_worker}" \
    MKL_NUM_THREADS="${threads_per_worker}" \
    OPENBLAS_NUM_THREADS="${threads_per_worker}" \
    NUMEXPR_NUM_THREADS="${threads_per_worker}" \
    VECLIB_MAXIMUM_THREADS="${threads_per_worker}" \
    PYTHONUNBUFFERED=1 \
    "${launch_prefix[@]}" \
    "${python_bin}" -m "${module_name}" \
    --Datasets "$dataset" \
    --eval-list "$eval_list" \
    --save-dir "$save_dir" \
    --lambda1 "$lambda1" \
    --lambda2 "$lambda2" \
    --lambda3 "$lambda3" \
    --lambda4 "$lambda4" \
    --superpixel-algorithm "$superpixel_algorithm" \
    --pending-samples "$pending_samples" \
    --device "$device" \
    --resume-check "$resume_check" \
    --num-shards "$gpu_numbers" \
    --shard-id "$gpu_index" \
    --begin "$begin" \
    --end "$end" \
    "${allow_fallback_flag[@]}" >"$log_file" 2>&1 &
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
