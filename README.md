# 低阶稀疏 Möbius LLM 归因

本仓库研究一个问题：LLM 局部 value function 的交互结构，是否比带 hierarchy 假设的稀疏 Fourier 表示更适合用低阶稀疏 Möbius 超图刻画，并能否在有限模型查询下恢复为可用的归因方法。

当前主方法位于 `mobius/`。ProxySPEX、Inseq 和 AML 是保留的独立 baseline；旧 exact spectrum、conditional probe、recovery 验证代码已移入 `archive/mobius_validation/`。

## 安装

```bash
pip install -e .
pip install torch transformers datasets
```

运行 ProxySPEX 或 Inseq 时还需安装对应依赖：

```bash
pip install -e "baselines/shapiq-copy[proxy]"
pip install inseq captum
```

## 快速验证

下面的命令只使用 mock scorer，不下载模型或数据：

```bash
python -m mobius.cli.run \
  --config configs/mobius/smoke.yaml \
  --overwrite
```

## 正式运行

```bash
# SST-2
python -m mobius.cli.run \
  --config configs/mobius/sst2.yaml \
  --device cuda:0

# Emotion
python -m mobius.cli.run \
  --config configs/mobius/emotion.yaml \
  --device cuda:1

# Rotten Tomatoes
python -m mobius.cli.run \
  --config configs/mobius/rotten_tomatoes.yaml \
  --device cuda:2
```

临时限制样本数或更换预算：

```bash
python -m mobius.cli.run \
  --config configs/mobius/sst2.yaml \
  --max-samples 20 \
  --budget 128 \
  --device cuda:0
```

## 评估与比较

归因结束时默认自动评估。也可以单独重跑评估：

```bash
python -m mobius.cli.evaluate \
  --run-dir results/mobius/<dataset>/<model>/<method>/<run-id> \
  --target predicted \
  --device cuda:0
```

对比两个已评估 run：

```bash
python -m mobius.cli.compare \
  --left <proxyspex-run-dir> \
  --right <sparse-mobius-run-dir> \
  --output comparison.json
```

`metrics.json` 只包含一个平铺的显式 `target` 和它对应的一套指标，不会保存多个 target block，也不再使用含义模糊的 `metrics_primary`。默认 target 是 `predicted`；用另一个 target 重新评估会覆盖 `metrics.json`。主实验应读取 `target: predicted` 下的 `faithfulness`、`per_q`、`attribution_cost` 和 `evaluation_cost`。

将一个方法根目录下的所有 run 整理为 CSV：

```bash
python scripts/collect_results.py \
  --input_dir results/mobius \
  --o mobius_results.csv
```

省略 `--o` 时默认写入当前目录的 `results_summary.csv`。CSV 仅包含运行身份、所有 faithfulness 的 mean/std、归因模型调用及每样本值、核心实验配置、样本数和失败数。

## 消融实验

六个单轴消融配置位于 `configs/ablations/`：

```bash
python -m mobius.cli.run --config configs/ablations/basis_fourier.yaml --device cuda:0
python -m mobius.cli.run --config configs/ablations/hierarchy_strong.yaml --device cuda:0
python -m mobius.cli.run --config configs/ablations/sampler_bernoulli.yaml --device cuda:0
python -m mobius.cli.run --config configs/ablations/sampler_uniform_size.yaml --device cuda:0
python -m mobius.cli.run --config configs/ablations/projector_absolute.yaml --device cuda:0
python -m mobius.cli.run --config configs/ablations/projector_singleton.yaml --device cuda:0
```

每一项的数学含义、控制变量和当前旧结果配置见 [使用与消融说明](docs/mobius_usage.md)。

## Baseline

ProxySPEX 保持原生 sampler、树 proxy、Fourier 提取、refinement 和 FBII 转换：

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --chunker word \
  --eval-granularity token \
  --value-function predicted_probability \
  --target-mode predicted \
  --budget 512 \
  --max-order 2 \
  --device cuda:0 \
  --base-save-dir results \
  --save-dir baselines/proxyspex-copy
```

Inseq 示例：

```bash
python baselines/inseq/run_inseq_llm_baselines.py \
  --dataset sst2 \
  --methods integrated_gradients,lime \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --target-mode predicted \
  --device cuda:0
```

AML 使用方式见 `baselines/aml-main_copy/README.md`。

## 结果结构

```text
results/<dataset>/<model>/<method>/<run-id>/
├── run.json
├── status.json
├── metrics.json
├── curves-<target>.jsonl
├── samples/
└── diagnostics/        # 仅 output_level=debug
```

`minimal` 只保存排名与成本，`standard` 额外保存超边和拟合摘要，`debug` 再保存 masks、coalition values 和验证细节。旧结果协议不兼容，需要重新运行。

run ID 默认为 `b<budget>-o<order>-s<seed>-<hash8>`。配置中可设置可读后缀：

```yaml
run_suffix: paper-main
```

此时 run ID 为 `b<budget>-o<order>-s<seed>-paper-main`；字段缺失或为空时才使用配置哈希。
