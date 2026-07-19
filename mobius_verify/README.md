# Mobius Verify

`mobius_verify` 是一个用于验证 LLM / 神经模型局部 masked value function 结构的可复现实验工程。核心问题是：

> 对文本局部掩码函数 `f(S)`，低阶稀疏 Möbius 超图结构是否比 ProxySPEX 所依赖的 Fourier hierarchy / staircase 结构更普遍、更稳定，并且是否更容易用有限黑盒查询恢复？

主实验的 sentiment 特征粒度是 **真实 lexical word**。每个样本允许有自己的 `n_i`，不会为了固定 `n` 把多个词合并成 block。固定大小只用于短文本 exact 筛选或长文本 probe set。

## 环境

最小 smoke 只需要：

```bash
pip install -r mobius_verify/requirements.txt
```

当前仓库机器上可能没有 `python` 命令，可使用 `python3` 或显式解释器路径。例如：

```bash
/opt/miniconda3/bin/python mobius_verify/scripts/run_synthetic.py --config mobius_verify/configs/synthetic_default.yaml
```

如果要运行真实 HF causal LM，还需要可用的 `torch`、`transformers`、模型权重和设备配置。

## 快速 Smoke

默认配置使用 mock sentiment scorer 和内联样本，不会下载模型或数据集。

```bash
python3 mobius_verify/scripts/run_synthetic.py \
  --config mobius_verify/configs/synthetic_default.yaml \
  --overwrite

python3 mobius_verify/scripts/collect_exact_values.py \
  --config mobius_verify/configs/exact_default.yaml \
  --overwrite

python3 mobius_verify/scripts/compute_exact_spectra.py \
  --results-dir mobius_verify/results/exact_default \
  --d-max 3 \
  --overwrite

python3 mobius_verify/scripts/run_limited_query_recovery.py \
  --config mobius_verify/configs/recovery_default.yaml \
  --overwrite

python3 mobius_verify/scripts/aggregate_results.py \
  --results-dir mobius_verify/results/exact_default
```

运行后主要结果在：

```text
mobius_verify/results/exact_default/
├── features/
├── values_exact_global/
├── values_exact_probe/
├── spectra_exact_global/
├── spectra_exact_probe/
├── recovery/
└── aggregate/
```

最终摘要报告：

```text
mobius_verify/results/exact_default/aggregate/hypothesis_report.md
```

## 配置文件

- `configs/synthetic_default.yaml`：合成函数 sanity check。
- `configs/exact_default.yaml`：词级 featureization、短文本 exact global、长文本 exact probe。
- `configs/recovery_default.yaml`：有限查询恢复实验。
- `configs/medium_default.yaml`：medium-n 实验占位，默认 gated，不会误运行。

## 真实数据和模型

默认数据配置可以替换为 LIMA 已支持的 sentiment 数据集，例如 `sst2`、`rotten_tomatoes`、`emotion`、`eraser_movie_reviews`。

示例：

```yaml
dataset:
  name: sst2
  split: validation
  max_samples: 20
  dataset_cache_dir: /path/to/hf_cache
  verbalizers: [negative, positive]

model:
  type: hf_causal_lm
  model_path: /path/to/Qwen2.5-7B-Instruct
  device: cuda:0
  dtype: bfloat16
  max_length: 2048
```

真实模型命令：

```bash
python3 mobius_verify/scripts/collect_exact_values.py \
  --config mobius_verify/configs/exact_default.yaml

python3 mobius_verify/scripts/compute_exact_spectra.py \
  --results-dir mobius_verify/results/exact_default \
  --d-max 4

python3 mobius_verify/scripts/run_limited_query_recovery.py \
  --config mobius_verify/configs/recovery_default.yaml

python3 mobius_verify/scripts/aggregate_results.py \
  --results-dir mobius_verify/results/exact_default
```

## Value Function

Sentiment 主 value function 是：

```yaml
value_function:
  type: predicted_class_margin
  verbalizer_length_normalization: mean
  target_class_source: full_input_prediction
```

含义：

- 先在完整输入上确定 predicted class；
- 对每个 masked input，只取该目标类别的 raw score；
- 对 causal LM，raw score 实现为目标 verbalizer token 的平均 conditional log probability；
- 不使用 softmax probability，避免概率饱和影响结构分析。

## 实验范围

`exact_global`：

对自然短文本完整枚举 `2^n_i` 个词子集，得到全局词级 value table 和精确 Möbius / Fourier 谱。

`exact_probe`：

对长文本选择大小为 `k` 的词级 probe set，只枚举 probe 内组合；probe 外词默认保持存在，即 `rest_present`。该结果是条件局部谱，不是完整长文本全局谱。

`limited-query recovery`：

从保存的 exact table 中抽取同一组 train / validation / test masks，对比：

- Additive LASSO
- Low-degree Möbius LASSO
- Low-degree Fourier LASSO
- sklearn `GradientBoostingRegressor`

查询预算按 `m = α n_i log2(n_i)` 缩放。

## 测试

编译检查：

```bash
python3 -m py_compile $(find mobius_verify -name '*.py' -not -path '*/results/*')
```

单元测试：

```bash
python3 -m pytest mobius_verify/tests -q
```

如果本机 `pytest` 因 native 依赖段错误，可先用 smoke CLI 验证核心链路。

## 重要限制

Möbius 基底非正交，Fourier 基底正交，因此不能直接比较两者原始 coefficient energy。主比较必须使用同一输出空间里的 reconstruction R² / normalized RMSE。

短文本 exact global 只能说明自然短句的全局词级结构；长文本 exact probe 只能说明“在其他词保持存在时”的条件局部结构。最终结论需要按任务、长度区间、样本级 paired comparison 分别报告。
