# Inseq 基线说明

## 1. Inseq 官方支持的方法

本地 `baselines/inseq/README.md` 对应 Inseq 0.7.1。官方方法大致分为三类：

| 类别 | 方法 |
| --- | --- |
| 梯度类 | `saliency`, `input_x_gradient`, `integrated_gradients`, `deeplift`, `gradient_shap`, `discretized_integrated_gradients`, `sequential_integrated_gradients` |
| 内部状态类 | `attention` |
| 扰动类 | `occlusion`, `lime`, `value_zeroing`, `reagent` |

官方 README 明确列出了 `lime`，但没有列出 Kernel-SHAP。因此本目录只接入 Inseq 原生方法：`LIME` 支持，Kernel-SHAP 不在 `baselines/inseq` 中支持。若后续确实需要 Kernel-SHAP，请使用 `baselines/captum` 里的 Captum baseline，而不要在 Inseq runner 中混入 fallback。

## 2. 本项目 runner 已接入的方法

| 方法 | 后端 | 默认运行 | 说明 |
| --- | --- | --- | --- |
| `saliency` | Inseq native | 是 | 梯度 saliency |
| `input_x_gradient` | Inseq native | 是 | input x gradient |
| `integrated_gradients` | Inseq native | 是 | 使用 `--n-steps` 和 `--internal-batch-size` |
| `sequential_integrated_gradients` | Inseq native | 是 | 使用 `--n-steps` 和 `--internal-batch-size` |
| `occlusion` | Inseq native | 是 | 扰动类，通常慢于梯度类 |
| `reagent` | Inseq native | 是 | ReAGent，使用 `--reagent-*` 参数 |
| `lime` | Inseq native | 否 | Inseq 原生 LIME，使用 `--n-samples` |

默认方法保持旧集合，不自动加入 `lime`，避免一次运行意外变慢。需要时请显式传：

```bash
--methods lime
```

## 3. 依赖

```bash
pip install inseq transformers captum
```

Inseq 的 LIME 实现依赖 Captum，因此建议按上面的命令安装。

## 4. 常用命令

默认旧方法集合：

```bash
CUDA_VISIBLE_DEVICES=3 python baselines/inseq/run_inseq_llm_baselines.py \
  --datasets imdb \
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
  --device cuda \
  --deterministic
```

只跑 LIME：

```bash
CUDA_VISIBLE_DEVICES=3 python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset sst2 \
  --max-samples 20 \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-length 2048 \
  --methods lime \
  --n-samples 32 \
  --k 8 \
  --target-mode gold \
  --eval-q-values 1,5,10,20,50 \
  --eval-granularity token \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda
```

快速 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset all \
  --max-samples 3 \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --methods lime \
  --n-samples 8 \
  --base-save-dir results \
  --save-dir smoke-inseq-lime
```

## 5. 输出

每个方法单独一个目录：

```text
results/baselines/inseq/<dataset>/model-<model>/method-<method>_target-<mode>_k-<k>_seed-<seed>/
```

主要产物：

- `samples/*.json`
- `samples/*.txt`
- `summary.csv`
- `run_config.json`
- `eval_config.json`
- `eval_report.json`
- 聚合文件：`aggregate_metrics.csv`, `aggregate_metrics.jsonl`
