# Inseq LLM 基线

Runner `run_inseq_llm_baselines.py` 保留 Inseq 原生 attribution 调用，并使用 `mobius` 的数据 loader、schema-v2 结果和统一 evaluator。

## 支持方法

- 梯度：`saliency`、`input_x_gradient`、`integrated_gradients`、`sequential_integrated_gradients`
- 扰动：`occlusion`、`lime`
- ReAGent：`reagent`

默认不包含较慢的 `lime`，需要显式写入 `--methods`。

所有方法使用与主方法一致的 `task_classification_v1` prompt，包含任务说明和完整候选标签。最终只将原始文本字符区间上的 token attributions 投影为 ranking，不把 instruction、候选标签或 `Label:` 当作解释特征。Occlusion 在计算每个文本 token 的遮挡效应时保持其余 prompt 不变；LIME 的扰动器会显式冻结全部非文本 prompt tokens，只采样原文 token 的开关。若 tokenizer 无法可靠对齐该区域，LIME 会直接报错。

需要区分两种目标语义：Inseq 原生 Occlusion/LIME 解释的是强制生成目标 verbalizer 时的 token probability/log-probability，而 Sparse Möbius 和 ProxySPEX 拟合的是候选 verbalizers 间归一化后的 predicted-class probability。因此 Inseq 方法可以作为原生生成归因基线，并使用统一 evaluator 检查其 ranking；不能把它们的内部 surrogate 数值当成与 Möbius/ProxySPEX 相同的 masked-function reconstruction 结果。

运行任何归因方法前，先使用 `scripts/check_classifier.py` 验证完整输入分类准确率。

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
  --eval-q-values 5,10,20,50 \
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
  --eval-q-values 5,10,20,50 \
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
