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

Inseq 的 LIME 实现依赖 Captum，因此建议按上面的命令安装。LIME 与 `--dtype bfloat16` 一起使用时，Inseq 0.7.x 原生实现会在内部把 bf16 tensor 直接转为 numpy，触发 `TypeError: Got unsupported ScalarType BFloat16`。本 runner 已在 LIME 加载时做局部兼容补丁：只在 Inseq LIME 的扰动函数里把用于 numpy 转换的 bf16 tensor 临时转成 float32，不改变模型本身的 dtype。

## 4. 常用命令

```bash
<!-- saliency,input_x_gradient,integrated_gradients,sequential_integrated_gradients,occlusion,lime -->
# todo: eraser-occlusion
# failed: eraser-ig,sig
CUDA_VISIBLE_DEVICES=7 python baselines/inseq/run_inseq_llm_baselines.py \
  --datasets eraser \
  --model-path /mnt/huawei/nsq/models/meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --max-length 2048 \
  --methods occlusion,lime \
  --k 8 \
  --target-mode gold \
  --eval-q-values 1,5,10,20,50 \
  --eval-granularity token \
  --base-save-dir results \
  --save-dir baselines/inseq \
  --device cuda \
  --deterministic
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

## 6. 断点恢复

runner 默认支持断点恢复。每次运行某个 dataset/method 时，会先扫描当前方法目录下的 `samples/`：

- 默认 `--resume-check strict`：只有同时存在 `samples/<sample_id>.json` 和 `samples/<sample_id>.txt`，且 JSON 可解析并包含 `explain_method`, `chunk_ranking`, `chunk_scores`, `selected_chunk_ids`, `trace` 等字段时，才认为该样本已完成并跳过解释计算。
- 可选 `--resume-check exists-only`：只检查 `.json` 和 `.txt` 是否存在，不校验 JSON 内容，速度稍快但不防损坏文件。
- 已完成样本不会重新跑 Inseq attribution；脚本仍会重建 `summary.csv` 并重新生成 `eval_report.json`。

注意：输出目录名只包含 method、target、k、seed。若你改变了 `--n-samples`, `--score-mode`, `--attributed-fn` 等解释参数但复用同一个输出目录，已有样本仍会被当作已完成并跳过。需要重新计算时，请删除对应方法目录或换 `--save-dir` / `--seed`。
