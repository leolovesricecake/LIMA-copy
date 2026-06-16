

```bash
pip install inseq transformers captum


python baselines/inseq/run_inseq_llm_baselines.py \
  --datasets all \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-length 2048 \
  --methods saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,reagent \
  --k 8 \
  --target-mode gold \
  --eval-q-values 1,5,10,20,50 \
  --eval-granularity token \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda:0 \
  --deterministic
```