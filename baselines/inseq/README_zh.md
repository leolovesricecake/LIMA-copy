

```bash
pip install inseq transformers captum

# todo: eraser-occlusion
# failed: eraser-ig,sig
CUDA_VISIBLE_DEVICES=3 python baselines/inseq/run_inseq_llm_baselines.py \
  --datasets imdb \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-length 2048 \
  --methods saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion \
  --k 8 \
  --target-mode gold \
  --eval-q-values 1,5,10,20,50 \
  --eval-granularity token \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda \
  --deterministic

# smoke
CUDA_VISIBLE_DEVICES=4 python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset all \
  --max-samples 3 \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --methods saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,reagent \
  --base-save-dir results \
  --save-dir smoke-inseq-v2
```