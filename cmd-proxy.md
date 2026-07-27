## ProxySPEX

ProxySPEX 保留其原生 sampler、tree proxy、Fourier 提取、refinement 和 FBII 转换，只统一文本粒度、value function 与 ranking evaluator：

### sst2

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --chunker word \
  --eval-granularity word \
  --value-function predicted_probability \
  --target-mode predicted \
  --budget 512 \
  --max-order 2 \
  --seed 42 \
  --device cuda:0 \
  --base-save-dir results \
  --save-dir baselines/proxyspex
```

### rtn

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset rotten_tomatoes \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --chunker word \
  --eval-granularity word \
  --value-function predicted_probability \
  --target-mode predicted \
  --budget 512 \
  --max-order 2 \
  --seed 42 \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

### emotion

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset emotion \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --chunker word \
  --eval-granularity word \
  --value-function predicted_probability \
  --target-mode predicted \
  --budget 512 \
  --max-order 2 \
  --seed 42 \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```
