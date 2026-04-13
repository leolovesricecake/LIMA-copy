#!/bin/bash
echo "-----------------"
echo "running LIMA"
echo -e "-----------------\n"

result_dir="submodular_results/imagenet-clip-vitl-efficientv2/seeds-0.0-0.05-20.0-1.0-pending-samples-8"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"

python_commands=(
    "scripts/efficientv2-clip_vitl_multigpu.sh"
    "python -m evals.eval_AUC_faithfulness --explanation-dir ${result_dir}"
    "python -m evals.evaluation_mistake_debug_ours --explanation-method ${result_dir} --eval-list ${eval_list}"
)

# Execute commands in a loop
for cmd in "${python_commands[@]}"; do
    echo "- - > Executing: $cmd"
    $cmd
    echo ""
done
