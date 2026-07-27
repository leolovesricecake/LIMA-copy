# 低阶稀疏 Möbius LLM 归因

本项目研究低阶稀疏 deletion-Möbius 超图能否作为一种可恢复、可验证且有竞争力的 LLM 交互归因表示。主方法位于 `mobius/`；ProxySPEX、Inseq 和 AML 保留为独立 baseline。

当前实现覆盖论文计划的 P0 协议基础和 P1 SST-2 核心机制实验，即 E2 交互必要性、E3 精确交互验证和 E4 hierarchy 分析。E5 可压缩性、跨数据集扩展及敏感性实验尚未实现。

## 安装

```bash
pip install -e .
pip install torch transformers datasets
pip install -e "baselines/shapiq-copy[proxy]"
pip install inseq captum
```

主协议默认使用：

- word explanation chunks；
- word faithfulness evaluation；
- `target_mode=predicted`；
- predicted-class probability；
- attribution budget 512；
- degree 2 deletion-Möbius；
- `uniform_size` attribution sampler；
- 3 个 attribution seeds。

## 主方法

运行 SST-2 主配置：

```bash
python -m mobius.cli.run \
  --config configs/qwen3-8b/sst2.yaml \
  --device cuda:0
```

临时覆盖 seed、预算或样本数：

```bash
python -m mobius.cli.run \
  --config configs/qwen3-8b/sst2.yaml \
  --seed 43 \
  --budget 512 \
  --max-samples 20 \
  --device cuda:0
```

每个 Sparse Möbius 样本必须同时写入：

```text
<run>/
├── run.json
├── status.json
├── metrics.json
├── curves-predicted.jsonl
├── samples/<sample-id>.json
├── observations/<sample-id>.npz
├── surrogates/<sample-id>.json
└── analyses/
```

`observations/` 保存 attribution training keep masks、全部类别 label scores 和 attribution values；`surrogates/` 保存 active-player 映射、support、系数、拟合协议和可对任意 mask 预测的表示。Sparse Möbius 和 ProxySPEX 缺少任一 sidecar 时，该样本不会被 resume 为已完成。

## P1 运行顺序

### 1. 运行 A、C 和 strict

A 是 degree-1 additive；C 是 degree-2 signed interaction。B 不重新训练，而是从 C 离线派生。

```bash
for seed in 42 43 44; do
  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-sst2/e2_a_additive.yaml \
    --seed "$seed" \
    --device cuda:0

  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-sst2/e2_c_interaction.yaml \
    --seed "$seed" \
    --device cuda:0

  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-sst2/e4_strict.yaml \
    --seed "$seed" \
    --device cuda:0
done
```

### 2. 从 C 派生 B

```bash
python scripts/derive_projection_run.py \
  --input-run results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-c-interaction \
  --projector singleton_only \
  --output-root results/mobius-mechanisms \
  --run-suffix e2-b-fit-only
```

派生步骤复用 C 的 observations 和 surrogate，不产生模型调用，并保留 C 的 attribution cost。B 的 ranking faithfulness 需要单独评估，调用计入 evaluation cost：

```bash
python -m mobius.cli.evaluate \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --target predicted \
  --device cuda:0
```

### 3. 构建 shared held-out

建议每个 attribution seed 分别将 A/B/C/strict 放入同一 audit：

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o1-s42-e2-a-additive \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-c-interaction \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e4-strict \
  --output-dir results/audits/surrogate-heldout/s42 \
  --count-per-distribution 64 \
  --min-count 16 \
  --seed 42 \
  --device cuda:0

python scripts/evaluate_surrogates.py \
  --audit-dir results/audits/surrogate-heldout/s42
```

held-out masks 会排除所有输入 run 的 attribution training masks。默认 `bernoulli` 对每个词独立执行 0.5 概率的保留采样，因此对全部 coalition 等概率；它不同于训练配置中的 `uniform_size`。若共同未见空间不足，则使用全部可用 masks；少于 16 个时样本标记为 `insufficient`。

需要额外检查完整输入附近的删除行为时，可同时启用 `near_full`；它先从删除数 `{1,2,3,5}` 中均匀选一层，再在该层均匀选被删除词：

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir <run-1> \
  --run-dir <run-2> \
  --distributions bernoulli,near_full \
  --near-full-deletions 1,2,3,5 \
  --device cuda:0
```

`manifest.json` 的 `metadata` 会记录全部输入 run 目录、run ID、method、dataset、model、chunker 和 value semantics；`settings` 记录 held-out 分布、数量与 seed。

### 4. E3 精确验证

```bash
python scripts/verify_interactions.py \
  --run-dir results/mobius-mechanisms/sst2/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-c-interaction \
  --top-k 5 \
  --top-k-per-parent-group 3 \
  --seed 42 \
  --device cuda:0
```

该 audit 对 selected pair 和距离匹配的 random pair 查询四个完整输入附近的删除组合，精确计算 deletion-Möbius 二阶系数。查询不计入 attribution cost。

### 5. E4 hierarchy 分析

对每个 seed 分别运行：

```bash
python scripts/analyze_hierarchy.py \
  --none-run <C-run-dir> \
  --strict-run <strict-run-dir> \
  --verification-dir <interaction-audit-dir> \
  --heldout-audit <heldout-audit-dir> \
  --seed 260726 \
  --device cuda:0
```

脚本比较 none/strict support、held-out 和原始 faithfulness，并将 none 的 pair 按 0/1/2 个 singleton parent 分组。离线删边后的 ranking evaluation 会产生独立 analysis query cost。

### 6. 生成论文长表

```bash
python scripts/collect_paper_results.py \
  --run A=<A-seed42-run> \
  --run B=<B-seed42-run> \
  --run C=<C-seed42-run> \
  --run none=<C-seed42-run> \
  --run strict=<strict-seed42-run> \
  --audit-dir <heldout-audit-dir> \
  --audit-dir <interaction-audit-dir> \
  --audit-dir <hierarchy-audit-dir> \
  --output-dir docs/paper-results
```

三个 seed 的同一 role 可重复传入。输出包括 `paper_metrics_long.csv`、`paper_comparisons.csv` 和 `integrity_checks.json`。收集器会硬校验同 seed A/C 的 observation digest 相同，以及 B/C 的 surrogate digest 相同。

## ProxySPEX

ProxySPEX 保留其原生 sampler、tree proxy、Fourier 提取、refinement 和 FBII 转换，只统一文本粒度、value function 与 ranking evaluator：

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

ProxySPEX 也保存原生 training coalitions、全部 label scores、unrefined/refined Fourier support 和最终 interactions。shared held-out 的 surrogate 预测固定使用 `refined_fourier`。

## 常规结果整理

将一个方法根目录整理为紧凑 CSV：

```bash
python scripts/collect_results.py \
  --input_dir results/mobius \
  --o results_summary.csv
```

`--o` 默认值为 `results_summary.csv`。该脚本用于普通 run 汇总；论文样本级配对分析使用 `collect_paper_results.py`。

完整配置语义、消融解释、指标方向和查询分账见 [使用与论文实验说明](docs/mobius_usage.md)。
