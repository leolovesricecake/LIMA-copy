# 论文实验中文使用指引 v2.3

本文档对应 `docs/paper-outline-v2.3.md` 与 `docs/paper-execution-plan-v2.3.md`。正式协议固定为：

- 数据集：SST-2 validation、Rotten Tomatoes validation、AG News test；
- 模型：Qwen3-8B、Llama-3.1-8B-Instruct；
- player 与评估粒度：word / word；
- target：完整输入的 predicted class，扰动后保持不变；
- value：候选 verbalizers 间归一化的 predicted-class probability；
- 主方法：degree 2、`uniform_size`、无 hierarchy、signed projection；
- attribution seeds：42、43、44；主预算 512；E2 预算 64、128、256、512；
- 正式 split 使用全集，不设置分类准确率门槛。

以下命令都从仓库根目录执行。`<...>` 表示需要替换的真实路径。

## 1. 环境与目录

```bash
pip install -e .
pip install torch transformers datasets scikit-learn scipy
pip install -e "baselines/shapiq-copy[proxy]"
```

正式配置位于：

```text
configs/v2-3/
├── qwen3-8b/{sst2,rotten_tomatoes,ag_news}.yaml
├── llama-3.1-8b-instruct/{sst2,rotten_tomatoes,ag_news}.yaml
└── first-order/{occlusion-sst2,lime-sst2}.yaml
```

先按机器修改配置中的 `model_path`、`dataset_cache_dir` 和 `device`。科学配置可通过 CLI 覆盖，但正式运行必须保留上面的冻结口径。

## 2. 分类诊断

分类诊断只说明模型是否能完成任务，不筛除 dataset-model cell，也不影响后续归因：

```bash
python scripts/check_classifier.py \
  --config configs/v2-3/qwen3-8b/sst2.yaml \
  --device cuda:0

python scripts/check_classifier.py \
  --config configs/v2-3/qwen3-8b/rotten_tomatoes.yaml \
  --device cuda:0

python scripts/check_classifier.py \
  --config configs/v2-3/qwen3-8b/ag_news.yaml \
  --device cuda:0
```

对另一个模型重复三次。报告包含 accuracy、balanced accuracy、per-class recall 和 confusion matrix。汇总为论文产物：

```bash
python scripts/collect_classifier_reports.py \
  --input-dir results/classifier-check \
  --output results/paper/v2-3/aggregate/classifier-diagnostics.csv
```

## 3. E1/E2 归因运行

### 3.1 Sparse Deletion-Mobius 主方法 C

主预算与三个 seed：

```bash
for seed in 42 43 44; do
  python -m mobius.cli.run \
    --config configs/v2-3/qwen3-8b/sst2.yaml \
    --budget 512 \
    --seed "$seed" \
    --run-suffix v2-3-main \
    --device cuda:0
done
```

将配置路径替换为 `rotten_tomatoes.yaml` 和 `ag_news.yaml`，完成 Qwen 三数据集；再使用 `llama-3.1-8b-instruct/` 下的配置完成 Llama E1。

E2 在 Qwen 上增加预算曲线：

```bash
for budget in 64 128 256 512; do
  for seed in 42 43 44; do
    python -m mobius.cli.run \
      --config configs/v2-3/qwen3-8b/sst2.yaml \
      --budget "$budget" \
      --seed "$seed" \
      --run-suffix v2-3-main \
      --device cuda:0
  done
done
```

### 3.2 Additive Deletion A

A 与 C 使用相同 sampler、seed、budget 和估计协议，只把候选项限制为 degree 1，并使用 singleton projection：

```bash
python -m mobius.cli.run \
  --config configs/v2-3/qwen3-8b/sst2.yaml \
  --budget 512 \
  --seed 42 \
  --max-degree 1 \
  --projector singleton_only \
  --device cuda:0
```

对三个 seed、三个数据集及 E2 的四个预算重复。由于 A/C 的采样只由样本、sampler、budget 和 seed 决定，collector 会检查二者 observation digest 完全相同。

### 3.3 ProxySPEX

ProxySPEX 保留原生 coalition sampler、GBT proxy、tree Fourier extraction、refinement、FBII conversion 与 ranking，仅统一 prompt、word players、target/value 和 evaluator：

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

AG News 使用 `--dataset ag_news --split test`。对所有数据集、模型、预算和 seed 重复。runner 会校验序列化 refined-Fourier predictor 与原生 ProxySPEX 预测一致；该检查不调用 LLM，也不改变原生 attribution。

### 3.4 Word Occlusion 与 Word LIME

两者与主方法共享 task-aware prompt、word players 和 value function。Occlusion 精确查询 full 与每个 singleton deletion；LIME 使用配置预算采样文本邻域并拟合加权 Ridge：

```bash
python scripts/run_first_order_llm.py \
  --config configs/v2-3/first-order/occlusion-sst2.yaml \
  --dataset sst2 \
  --split validation \
  --model-path <Qwen3-8B-path> \
  --seed 42 \
  --run-suffix v2-3-occlusion \
  --device cuda:0

python scripts/run_first_order_llm.py \
  --config configs/v2-3/first-order/lime-sst2.yaml \
  --dataset sst2 \
  --split validation \
  --model-path <Qwen3-8B-path> \
  --budget 512 \
  --seed 42 \
  --run-suffix v2-3-lime \
  --device cuda:0
```

对 AG News 改为 `--dataset ag_news --split test`。两个模型均运行 E1。

## 4. Shared Held-out 与 E2

每个 dataset/model/budget/seed 建立一个 audit。至少登记 A、C、ProxySPEX；它们保留各自训练 masks，held-out 会排除所有登记 run 的训练 masks。默认分布是每个词独立以 0.5 概率保留的 Bernoulli-0.5：

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir <A-run> \
  --run-dir <C-run> \
  --run-dir <ProxySPEX-run> \
  --output-dir results/audits/sst2/surrogate-heldout/b512-s42 \
  --cache-path results/.cache/value_oracle.sqlite3 \
  --distributions bernoulli \
  --count-per-distribution 64 \
  --min-count 16 \
  --seed 42 \
  --device cuda:0

python scripts/evaluate_surrogates.py \
  --audit-dir results/audits/sst2/surrogate-heldout/b512-s42
```

`manifest.json` 记录全部 run-dir、run ID、method、requested budget、realized logical unique queries、模型、数据集、prompt、value semantics、排除 mask 数和 audit query cost。

受控 basis 比较从 C 的 observations 离线完成：

```bash
python scripts/audit_representation.py \
  --mobius-run <C-run> \
  --proxyspex-run <ProxySPEX-run> \
  --heldout-audit results/audits/sst2/surrogate-heldout/b512-s42 \
  --output-root results/audits \
  --compression-k 1,2,4,8,16,32 \
  --seed 42
```

输出自动位于 `results/audits/<dataset>/representation/<audit-id>/`。其中：

- actual-estimator：两种 basis 使用相同 degree、标准化、Lasso support selection 和 RidgeCV refit；
- fixed-k：两种 basis 使用相同 standardized OMP support path 和 OLS refit；
- OMP+OLS 仅是坐标压缩性诊断，不是主方法估计器；
- 每个 fixed-k 结果记录 OLS rank、rank-deficient 标记和 condition number。

## 5. E3 Exact Pair Audit

每个数据集、seed 在 `10 <= n_features <= 32` 的共同成功样本中按固定 seed 随机选 100 条。真实模型只查询 full、全部 singleton deletion 和全部 pair deletion：

```bash
python scripts/audit_exact_pairs.py \
  --mobius-run <C-run> \
  --proxyspex-run <ProxySPEX-run> \
  --output-root results/audits \
  --cache-path results/.cache/value_oracle.sqlite3 \
  --min-features 10 \
  --max-features 32 \
  --max-samples 100 \
  --seed 42 \
  --device cuda:0
```

输出自动位于 `results/audits/<dataset>/interactions/<audit-id>/`。Sparse Mobius 和 ProxySPEX 都通过各自序列化 surrogate 的 full/singleton/pair 四点差分得到 deletion-Mobius pair score；E1 仍使用 ProxySPEX 原生 FBII ranking。主指标为 NDCG@10、Recall@10、Sign Agreement@10。

## 6. E4 Projection

从 C 离线派生 singleton-only 与 sign-agnostic。派生过程复用 observations、support、coefficients 和 surrogate，不产生 LLM attribution query：

```bash
python scripts/derive_projection_run.py \
  --input-run <C-run> \
  --projector singleton_only \
  --output-root results/mobius-derived \
  --run-suffix v2-3-singleton

python scripts/derive_projection_run.py \
  --input-run <C-run> \
  --projector absolute_equal_share \
  --output-root results/mobius-derived \
  --run-suffix v2-3-sign-agnostic
```

派生 ranking 的 faithfulness evaluation 仍会调用模型，必须单独执行：

```bash
python -m mobius.cli.evaluate \
  --run-dir <derived-run> \
  --target predicted \
  --device cuda:0
```

## 7. 论文结果收集

每次收集一个 dataset/model/budget/seed cell。主预算 Qwen cell 示例：

```bash
python scripts/collect_paper_results.py \
  --paper-version v2-3 \
  --run A=<A-run> \
  --run B=<singleton-derived-run> \
  --run ABSOLUTE=<sign-agnostic-derived-run> \
  --run C=<C-run> \
  --run PROXYSPEX=<ProxySPEX-run> \
  --run OCCLUSION=<Occlusion-run> \
  --run LIME=<LIME-run> \
  --audit-dir <heldout-audit> \
  --audit-dir <representation-audit> \
  --audit-dir <exact-pair-audit> \
  --overwrite
```

E2 的 64/128/256 cell 只需 A、C、ProxySPEX 及对应 held-out/representation audits。Llama E1-only cell 不传 audit，仍传 A、C、ProxySPEX、Occlusion 和 LIME。

cell 路径包含预算，避免覆盖：

```text
results/paper/v2-3/cells/
  <dataset-split>/<model-id>/budget-<B>/seed-<seed>/
```

所有 cell 完成后：

```bash
python scripts/collect_paper_results.py \
  --paper-version v2-3 \
  --summarize
```

canonical outputs 位于 `results/paper/v2-3/aggregate/`：

```text
classifier-diagnostics.csv
table-e1-faithfulness.csv
figure-e1-paired-effects.csv
figure-e2a-budget-recovery.csv
figure-e2b-fixed-support.csv
table-e3-exact-pairs.csv
table-e4-projection.csv
attribution-costs.csv
audit-costs.csv
manifest.json
```

E1-E4 都先在同一样本内跨 attribution seed 求平均，再进行样本级 mean/std 和配对统计。配对效应使用共同成功样本、bootstrap confidence interval、paired effect size 与 Wilcoxon signed-rank test。

## 8. 查询成本边界

- attribution cost：方法构建解释所用的 logical unique queries、physical values 和 model forward calls；
- evaluation cost：AUPC 与 deletion/retention faithfulness 所需调用，不计入 attribution budget；
- audit cost：shared held-out 和 exact-pair oracle 的额外调用，单独报告；
- representation audit、surrogate 四点差分和 projection 派生为离线操作，不调用 LLM；
- AOPC-Sufficiency 复用已有四个 retention 比例，不增加调用；完整 AUSufficiency 未实现，因为需要额外 retention trajectory。

## 9. 快速验收

```bash
python -m py_compile \
  mobius/analysis/paper_results.py \
  mobius/methods/first_order/explainer.py \
  baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  scripts/audit_exact_pairs.py \
  scripts/collect_paper_results.py

python -m pytest -q \
  tests/test_first_order_baselines.py \
  tests/test_paper_analysis.py \
  tests/test_paper_protocol.py \
  tests/test_proxyspex_copy_llm_baseline.py \
  tests/test_task_prompt.py
```
