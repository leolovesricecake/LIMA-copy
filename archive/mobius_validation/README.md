# Sparse Möbius LLM 归因

`mobius_verify` 现在只实现一个独立方法：低阶稀疏 deletion-Möbius 超图归因。

ProxySPEX 不再由本目录包装或调度，而是直接运行：

```text
baselines/shapiq-copy/run_proxyspex_llm_baseline.py
```

两种方法各自决定 attribution masks、拟合器和交互恢复过程。运行完成后，使用同一个 LIMA evaluator 计算 faithfulness，再通过纯离线脚本比较结果。

## 方法定义

令 `N` 为全部 players，`S` 为保留集合：

```text
f(S) = model_value(x_S)
```

对删除集合 `D` 定义：

```text
g(D) = f(N \ D)
```

Sparse Möbius 在低阶 deletion-Möbius 字典中拟合：

```text
g_hat(D) = beta_0 + sum_T theta[T] * 1{T subseteq D}
```

非空 deletion-Möbius coefficient 与 OR interaction 只差符号约定，本项目不把它包装成新指标。`theta[T]` 表示同时删除 `T` 对模型 value 产生的不可由其真子集完全解释的联合效应。

节点归因使用有符号均分：

```text
score_i = -sum_{T contains i} theta[T] / |T|
```

负号来自删除坐标：删除后 value 下降对应正向重要性。

## 独立采样

Sparse Möbius 不与 ProxySPEX 或其他方法共享训练 masks。它在严格逻辑预算内混合：

- 完整集合与空集锚点；
- Bernoulli-0.5 全局 masks；
- 小删除集合，也就是完整输入附近的 masks；
- 固定保留比例 masks。

默认比例：

```yaml
sampler:
  global_fraction: 0.5
  near_full_fraction: 0.3
  fixed_cardinality_fraction: 0.2
```

每个样本会保存实际 masks、values、全类别 raw verbalizer scores、来源计数和 digest。

## 公平比较契约

方法可以独立选择训练 masks，但比较时以下条件必须一致：

- dataset、split 和 sample ID；
- 模型、prompt、max length 和 verbalizers；
- token/word/adaptive chunker；
- 删除文本的组合方式和空文本 `<EMPTY>`；
- target class 来源和 value function；
- evaluation granularity、q values 和 evaluator。

比较脚本会检查 `run_config.json` 中的 `comparison_contract`，还会逐样本核对原文和 chunk spans。不一致时默认拒绝比较。

## Value Function

模型首先返回每个 verbalizer 的平均 conditional log probability：

```text
z_c(S)
```

支持：

```text
target_probability:
    softmax(z(S))[target]

predicted_probability:
    softmax(z(S))[c_star]

predicted_class_margin:
    z_c_star(S) - max_{c != c_star} z_c(S)

raw_target_score:
    z_target(S)
```

其中：

```text
c_star = argmax_c z_c(N)
```

`predicted_probability` 和 `predicted_class_margin` 始终固定解释完整输入的预测类别。默认 Sparse 主实验使用 `predicted_class_margin`；`raw_target_score` 只用于敏感性分析。

ProxySPEX 支持前三种 value function，默认 `predicted_probability`。严格复现旧 ProxySPEX LLM 设置时使用：

```bash
--value-function target_probability --target-mode gold
```

## 安装

```bash
python3 -m pip install -r mobius_verify/requirements.txt
python3 -m pip install torch transformers datasets
python3 -m pip install -e "baselines/shapiq-copy[proxy]"
```

## 运行 Sparse Möbius

先修改：

```text
mobius_verify/configs/sparse_mobius_default.yaml
```

重点确认模型路径、GPU 和数据集 cache：

```yaml
dataset:
  name: sst2
  split: validation
  dataset_cache_dir: /mnt/huawei/nsq/temp/hf

model:
  model_path: /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct
  device: cuda:0
```

运行一个独立实验：

```bash
python3 mobius_verify/scripts/run_sparse_mobius_llm.py \
  --config mobius_verify/configs/sparse_mobius_
```

覆盖单个参数：

```bash
python3 mobius_verify/scripts/run_sparse_mobius_llm.py \
  --config mobius_verify/configs/sparse_mobius_default.yaml \
  --budget 512 \
  --seed 42 \
  --value-function predicted_class_margin \
  --device cuda:7 \
  --max-samples 50
```

一次命令只运行一个 budget、seed 和 value function。不同实验应使用独立命令：

```bash
for budget in 128 256 512; do
  for seed in 0 1 2; do
    python3 mobius_verify/scripts/run_sparse_mobius_llm.py \
      --config mobius_verify/configs/sparse_mobius_default.yaml \
      --budget "$budget" \
      --seed "$seed"
  done
done
```

已有完整 sample JSON 会被 resume。强制重算：

```bash
--overwrite
```

只生成归因、不立即评估：

```bash
--no-eval
```

## 运行 ProxySPEX

概率 value 的公平比较：

```bash
python3 baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda:7 \
  --chunker word \
  --eval-granularity token \
  --value-function predicted_probability \
  --budget 512 \
  --seed 42 \
  --base-save-dir results \
  --save-dir baselines/proxyspex
```

Margin value 的公平比较：

```bash
python3 baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda:7 \
  --chunker word \
  --eval-granularity token \
  --value-function predicted_class_margin \
  --budget 512 \
  --seed 42 \
  --base-save-dir results \
  --save-dir baselines/proxyspex
```

ProxySPEX 的 coalition sampler、LightGBM proxy、Fourier 提取、refinement 和 interaction conversion 都继续由原始 `shapiq-copy` 实现。

## 对比结果

两种方法都完成并生成 `eval_report.json` 后运行：

```bash
python3 mobius_verify/scripts/compare_method_runs.py \
  --left-run /path/to/sparse_run_root \
  --right-run /path/to/proxyspex_run_root \
  --output-dir mobius_verify/results/comparisons/margin_b512_s42 \
  --target predicted
```

比较脚本不会加载模型，也不会发起 LLM 请求。输出：

```text
comparison_report.json
comparison_report.md
paired_sample_metrics.csv
```

报告包含：

- AOPC；
- comprehensiveness；
- sufficiency；
- AOPC-comprehensiveness；
- AOPC-sufficiency；
- log-odds；
- 逻辑 attribution queries；
- 唯一 attribution 文本数；
- 物理评分文本数；
- model forward、batch 数与解释耗时；
- 按 sample ID 配对的 bootstrap 置信区间；
- 两边样本覆盖率和被排除样本。

注意：sufficiency 和 AOPC-sufficiency 是 gap，越低越好；AOPC 和 comprehensiveness 通常越高越好。

## 输出结构

Sparse 单次 run：

```text
<run-root>/
├── run_config.json
├── environment.json
├── manifest.json
├── cache/value_oracle.sqlite3
├── samples/<sample_id>.json
├── samples/<sample_id>.txt
├── observations/<sample_id>.json
├── failures/<sample_id>.json
├── summary.csv
├── eval_report.json
├── eval_sample_metrics.jsonl
└── curve_summary.csv
```

`manifest.json` 在每个样本结束后更新。程序中断时，已经完成的 sample 不会丢失。

`samples/*.json -> metadata` 记录：

- deletion-Möbius 定义和符号约定；
- sampler diagnostics；
- 稀疏超边和映射后的 chunk IDs；
- targeted exact coefficient verification；
- ValueOracle 与 query ledger；
- 统一 `query_accounting`；
- 解释耗时和 backbone forward counters。

## 查询口径

主要比较：

```text
query_accounting.logical_attribution_queries
```

它表示方法请求的 coalition rows，不因 cache 命中而减少。

同时报告：

- `logical_unique_attribution_queries`：唯一方法 masks；
- `unique_attribution_texts`：mask 后实际唯一文本；
- `physical_values_scored`：本次真正触发评分的文本；
- `model_forward_calls`：底层模型 forward 次数；
- `interaction_verification_queries`：额外验证查询；
- evaluator 的 forward counters：仅属于评估，不计入归因预算。

## 测试

```bash
python3 -m py_compile $(find mobius_verify -name '*.py' -not -path '*/results/*')
python3 -m pytest mobius_verify/tests -q
python3 -m pytest tests/test_proxyspex_copy_llm_baseline.py -q
```

完整无需 GPU 的端到端检查：

```bash
python3 mobius_verify/scripts/run_sparse_mobius_llm.py \
  --config mobius_verify/configs/sparse_mobius_smoke.yaml \
  --output-root /tmp/sparse-mobius-smoke

python3 mobius_verify/scripts/compare_method_runs.py \
  --left-run /tmp/sparse-mobius-smoke \
  --right-run /tmp/sparse-mobius-smoke \
  --output-dir /tmp/sparse-mobius-compare-smoke
```

## 已删除的旧流程

以下内容不再属于主实验：

- controlled protocol；
- native protocol；
- ProxySPEX fixed observations；
- 由 `mobius_verify` 包装启动 ProxySPEX；
- 所有方法共享 attribution masks/values；
- 一个进程展开 methods × budgets × seeds × values 的笛卡尔积。

旧 exact/conditional-probe 脚本仍保留用于历史结果复核和小规模机制诊断，但不参与端到端方法比较。
