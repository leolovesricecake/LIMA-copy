# Inseq LLM 基线

Runner `run_inseq_llm_baselines.py` 保留 Inseq 原生 attribution 调用，并使用 `mobius` 的数据 loader、schema-v2 结果和统一 evaluator。

## 支持方法

- 梯度：`saliency`、`input_x_gradient`、`integrated_gradients`、`sequential_integrated_gradients`
- 扰动：`occlusion`、`lime`
- ReAGent：`reagent`

默认不包含较慢的 `lime`，需要显式写入 `--methods`。

## 安装

```bash
pip install inseq transformers captum
```

## sst2

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset sst2 \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --methods saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,lime \
  --dtype bfloat16 \
  --max-length 2048 \
  --target-mode predicted \
  --eval-granularity word \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda
```

## rtn

```bash
CUDA_VISIBLE_DEVICES=3 python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset rotten_tomatoes \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --methods saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,lime \
  --dtype bfloat16 \
  --max-length 2048 \
  --target-mode predicted \
  --eval-granularity word \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda
```


## 结果与成本

每个 dataset/method 生成独立 run：

```text
results/baselines/inseq/<dataset>/<model>/inseq_<method>/<run-id>/
```

输出包含 `run.json`、`status.json`、`samples/`、`metrics.json` 和 deletion curves。Runner 使用 PyTorch forward hook 统计 Inseq attribution 内部直接调用模型的 forward 次数，避免用 `n_steps` 粗略代替真实成本。

断点恢复以 schema-v2 `samples/<sample-id>.json` 为准；科学配置改变时 run ID 也会改变。
