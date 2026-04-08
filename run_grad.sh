python -m baseline_attribution.generate_explanation_maps_Grad-ECLIP --device 1
python -m baseline_attribution.debug_org_attribution_method_clip_vitl --device 1
python -m evals.eval_AUC_faithfulness --explanation-dir explanation_insertion_results/imagenet-clip-vitl-true/GradECLIP