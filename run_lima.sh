#!/bin/bash
echo "-----------------"
echo "running GradECLIP"
echo -e "-----------------\n"

# python -m baseline_attribution.generate_explanation_maps_Grad-ECLIP --device 0
# python -m baseline_attribution.debug_org_attribution_method_clip_vitl --device 0
# python -m evals.eval_AUC_faithfulness --explanation-dir explanation_insertion_results/imagenet-clip-vitl-true/GradECLIP
# python -m evals.evaluation_mistake_debug_baseline

python_commands=(
    "scripts/efficientv2-clip_vitl_multigpu.sh"
    "python -m evals.eval_AUC_faithfulness --explanation-dir explanation_insertion_results/imagenet-clip-vitl-true/GradECLIP"
    "python -m evals.evaluation_mistake_debug_baseline"
)

# Execute commands in a loop
for cmd in "${python_commands[@]}"; do
    echo "- - > Executing: $cmd"
    $cmd
    echo ""
done