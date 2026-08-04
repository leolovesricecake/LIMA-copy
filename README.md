# 低阶稀疏 Möbius LLM 归因

本项目研究低阶稀疏 deletion-Möbius 超图能否成为一种可恢复、可验证且有竞争力的 LLM 交互归因表示。主方法位于 `mobius/`；ProxySPEX、Inseq 和 AML 保留为独立 baseline。

当前论文协议与完整命令见：

- [论文实验中文使用指引](docs/paper_experiment_usage.md)
- [论文实验执行计划 v2.3](docs/paper-execution-plan-v2.3.md)
- [论文大纲 v2.3](docs/paper-outline-v2.3.md)

## 安装

```bash
pip install -e .
pip install torch transformers datasets scikit-learn scipy
pip install -e "baselines/shapiq-copy[proxy]"
```

ProxySPEX 的本地 `shapiq-copy` 要求 Python 3.12 或更高版本。

## 正式协议

- 数据集：SST-2 validation、Rotten Tomatoes validation、AG News test；
- 模型：Qwen3-8B、Llama-3.1-8B-Instruct；
- word explanation / word evaluation；
- 完整输入 predicted class，扰动期间固定；
- predicted-class probability value function；
- 主方法 degree 2、`uniform_size`、无 hierarchy、signed projection；
- seeds 42/43/44，主预算 512；E2 预算 64/128/256/512；
- 正式 split 全集评估，不设置分类准确率门槛。

所有方法使用 `task_classification_v1` prompt：明确任务、列出候选标签，并只扰动 prompt 中的原始文本区域。prompt 写入 run contract 与 value-cache 指纹，旧 prompt 结果不会被误复用。

## 最短运行示例

先修改配置中的模型和缓存路径。

```bash
# 分类诊断，不作为筛选门槛
python scripts/check_classifier.py \
  --config configs/paper-v2.3/qwen3-8b/sst2.yaml \
  --device cuda:0

# 主方法 C
python -m mobius.cli.run \
  --config configs/paper-v2.3/qwen3-8b/sst2.yaml \
  --budget 512 \
  --seed 42 \
  --run-suffix paper-v2.3-main \
  --device cuda:0

# Additive Deletion A
python -m mobius.cli.run \
  --config configs/paper-v2.3/qwen3-8b/sst2.yaml \
  --budget 512 \
  --seed 42 \
  --max-degree 1 \
  --projector singleton_only \
  --run-suffix paper-v2.3-additive \
  --device cuda:0

# 共享 value function 的 word LIME
python scripts/run_first_order_llm.py \
  --config configs/paper-v2.3/first-order/lime-sst2.yaml \
  --model-path <Qwen3-8B-path> \
  --budget 512 \
  --seed 42 \
  --device cuda:0
```

ProxySPEX：

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --dataset-cache-dir <hf-cache> \
  --model-path <Qwen3-8B-path> \
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

## 结果协议

每个可审计 attribution run 至少包含：

```text
<run>/
├── run.json
├── status.json
├── metrics.json
├── curves-predicted.jsonl
├── samples/<sample-id>.json
├── observations/<sample-id>.npz
└── surrogates/<sample-id>.json
```

`observations/` 保存训练 keep masks、全部类别 scores 和 attribution values；`surrogates/` 保存可离线预测任意 mask 的模型。论文 collector 输出到：

```text
results/paper/paper-v2.3/
├── cells/<dataset-split>/<model-id>/budget-<B>/seed-<seed>/
└── aggregate/
```

主表指标是 AUPC 与 AOPC-Sufficiency；E2 使用 held-out R² 与 NRMSE-range；E3 使用 NDCG@10、Recall@10 与 Sign Agreement@10。attribution、evaluation 和 audit 查询成本分别记录。

## 测试

```bash
python -m pytest -q \
  tests/test_first_order_baselines.py \
  tests/test_paper_analysis.py \
  tests/test_paper_protocol.py \
  tests/test_proxyspex_copy_llm_baseline.py \
  tests/test_task_prompt.py
```
