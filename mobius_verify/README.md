# Sparse Möbius Attribution

`mobius_verify` 现在是一个方法优先的 LLM 交互归因实验工程。它直接在 presence/deletion-Möbius 字典中估计稀疏超边，并在相同 word players、value function、掩码方式和逻辑查询预算下，与 Fourier、GBT 和完整 ProxySPEX 比较。

完整研究定义、实验约束和实现门控见 [docs/plan_tree.md](docs/plan_tree.md)。

## 核心问题

给定保留词集合 `S` 和全集 `N`：

```text
f(S) = model_value(x_S)
```

Presence-Möbius 使用：

```text
f_hat(S) = beta_0 + sum_T theta_pre[T] * 1{T subseteq S}
```

Deletion-Möbius 先定义删除集合：

```text
g(D) = f(N \ D)
```

再拟合：

```text
g_hat(D) = beta_0 + sum_T theta_del[T] * 1{T subseteq D}
```

对非空 `T`，常见符号约定下 `OR interaction = -m_del(T)`。本项目不把 deletion-Möbius/OR 当作新指标；研究重点是直接 Möbius 参数化能否带来更稀疏、稳定、查询高效且可校准的归因。

低阶 Möbius 与低阶 Fourier 张成相同函数空间，因此实验比较的是坐标稀疏性和恢复过程，不声称 Möbius 的低阶表达能力更强。

## 当前实现

- lexical-word players，保留 char/token spans；
- delete mask operator；
- predicted-class margin 主 value function；
- raw target verbalizer score 敏感性实验；
- 全局 SQLite `ValueOracle` cache；
- 每方法独立 logical query ledger；
- degree-1 和 degree-2 全候选；
- empirical centering/scaling；
- LASSO/ElasticNet support selection；
- selected support 上 ridge refit，可识别时附带 OLS debias；
- controlled-query comparison；
- native-method comparison；
- 完整 ProxySPEX tree/Fourier/refinement/Möbius 链路；
- uniform、near-full、fixed-cardinality held-out faithfulness；
- comprehensiveness、sufficiency、MoRF/LeRF 删除曲线；
- deletion/presence targeted exact coefficient verification；
- random ranking 和 random hyperedge controls；
- 多预算、seed、失败记录、resume 和聚合报告；
- 仅用 predicted-class margin 执行 sample-level paired bootstrap 与 degree-2 门控。

## Value Function

对每个 masked input，模型返回所有 verbalizer 的平均 conditional log probability `z_c(S)`。

完整输入决定固定目标类别：

```text
c_star = argmax_c z_c(N)
```

主实验：

```text
f_margin(S) = z_c_star(S) - max_{c != c_star} z_c(S)
```

敏感性实验：

```text
f_raw_target(S) = z_c_star(S)
```

ValueOracle 缓存完整类别分数向量，因此两种 value 不会重复请求 LLM。

## 两种比较协议

### Controlled

所有方法共享完全相同的 attribution masks 和 values：

- Additive LASSO；
- Presence-Möbius LASSO；
- Deletion-Möbius LASSO；
- Fourier LASSO；
- sklearn GBT；
- ProxySPEX fixed observations。

该协议用于区分坐标表示与估计器的影响。

### Native

每种方法使用自己的采样/拟合过程，但总 logical attribution budget 相同：

- Sparse deletion-Möbius；
- 完整 ProxySPEX。

所有方法拟合完成后，才生成并查询共同的 held-out evaluation masks。Evaluation 和 targeted verification 不计入 attribution budget，但单独记录真实前向数量。

## 环境

基础依赖：

```bash
python3 -m pip install -r mobius_verify/requirements.txt
```

真实 LLM 实验还需要：

- `torch`；
- `transformers`；
- 可访问的模型权重；
- ProxySPEX 主配置所用的 `lightgbm`。

Smoke 使用 sklearn decision tree 作为 ProxySPEX proxy，不需要 LightGBM、GPU、模型下载或 Hugging Face 数据下载。

ProxySPEX HPO 会按实际 attribution observation 数调整 CV：4 个样本使用 2-fold，少于 4 个样本对该次拟合关闭 HPO，避免 `n_splits > n_samples`。

LightGBM 的叶子约束也会按最小 CV 训练折自动调整，默认关闭重复训练日志：

```yaml
proxyspex:
  proxy_model: lightgbm
  hpo: true
  lightgbm_verbosity: -1
  lightgbm_min_child_samples: auto
```

`auto` 使用 `min(20, minimum_cv_train_size // 4)`，下限为 1。也可以填写固定正整数。每次 ProxySPEX 拟合会在结果的 `model.diagnostics` 中保存实际取值、树叶数量、训练预测方差和 `degenerate_proxy`；因此日志被关闭后仍能识别代理模型是否退化。

修改配置后，已有 `result.json` 会被 resume 跳过。重新拟合可在原命令后添加 `--overwrite`；同一结果目录中的 ValueOracle cache 仍会复用。


## 真实 SST-2 + Qwen

先在 `mobius_verify/configs/attribution_mvp.yaml` 中确认：

```yaml
dataset:
  dataset_cache_dir: /path/to/hf_cache

model:
  model_path: /path/to/Qwen2.5-7B-Instruct
  device: cuda:0
```

运行：

```bash
python3 mobius_verify/scripts/run_attribution_benchmark.py \
  --config mobius_verify/configs/attribution_mvp.yaml

python3 mobius_verify/scripts/aggregate_attribution.py \
  --results-dir mobius_verify/results/attribution_mvp
```

使用其他输出目录：

```bash
python3 mobius_verify/scripts/run_attribution_benchmark.py \
  --config mobius_verify/configs/attribution_mvp.yaml \
  --results-dir mobius_verify/results/my_experiment
```

已有完整 `result.json` 会被跳过。Native 协议若只完成了部分方法，会重新运行该比较单元，以保证各方法使用同一组、且未被训练阶段见过的 evaluation masks。

## 主要配置

```yaml
seeds: [0, 1, 2]
budget_alphas: [1, 2, 4, 8]
max_degree: 2
value_functions: [predicted_class_margin, raw_target_score]

controlled_methods:
  - additive_lasso
  - presence_mobius
  - deletion_mobius
  - fourier
  - sklearn_gbt
  - proxyspex_fixed_observations

native_methods:
  - deletion_mobius
  - proxyspex

evaluation_masks_per_distribution: 128
targeted_top_k: 5
decision_min_samples: 10
proxyspex_noninferiority_margin: -0.02
hyperedge_stability_floor: 0.1
```

每个样本的预算按下式生成：

```text
B = alpha * n * log2(n)
```

也可以用 `budgets: [128, 256, 512]` 指定固定预算。

## 查询预算与 Cache

全局 cache 位于：

```text
results/<experiment>/cache/value_oracle.sqlite3
```

公平比较使用每个方法的：

```text
query_ledger.attribution_budget_used
```

即使某个 mask 已被其他方法缓存，该方法访问它时仍计一次 logical query。Cache 只减少实际 GPU 前向，不会给后运行的方法免费预算。

主要账本字段：

- `attribution_budget_used`；
- `logical_unique_queries`；
- `physical_forwards_caused`；
- `global_cache_hits/misses`；
- `evaluation_only_queries`；
- `interaction_verification_queries`。

`physical_forwards_caused`/`physical_values_scored` 表示该方法触发的未缓存文本评分数，不是 batch 数。全局实际评分文本数、scorer batch 次数和模型自身 counters 记录在 `manifest.json` 的 `value_oracle_counters` 中。使用已有 cache 重新运行时，这些本轮计数可以为 0。

## 输出目录

```text
mobius_verify/results/<experiment>/
├── run_config.json
├── environment.json
├── manifest.json
├── cache/value_oracle.sqlite3
├── features/<task>/<sample_id>.json
├── protocols/
│   ├── controlled/<value>/<method>/<task>/<sample>/budget_*/seed_*/result.json
│   └── native/<value>/<method>/<task>/<sample>/budget_*/seed_*/result.json
└── aggregate/
    ├── sample_metrics.csv
    ├── method_summary.csv
    ├── query_metrics.csv
    ├── stability_metrics.csv
    ├── interaction_metrics.csv
    ├── paired_differences.csv
    ├── paired_query_auc.csv
    ├── hypothesis_results.json
    └── hypothesis_report.md
```

单个结果保存：

- surrogate coefficients/hyperedges；
- node scores 和正向/负向/绝对值 ranking；
- 三种 held-out 分布的 reconstruction metrics；
- attribution 删除曲线；
- targeted coefficient 校准；
- query ledger、fit diagnostics 和失败原因。

## 如何解读结果

- `controlled` 中 Möbius 与 Fourier 的差异主要反映坐标稀疏性和正则化几何。
- `native` 才回答 Sparse Möbius 与完整 ProxySPEX 哪个归因方法更有效。
- Deletion targeted verification 给出真实 deletion-Möbius coefficient。
- Presence targeted verification 使用 near-empty 输入，分布外风险更高。
- 高 held-out R² 不自动证明 hyperedge 是真实模型机制，仍需 targeted calibration 和多 seed stability。
- `predicted_class_margin` 是主结论；`raw_target_score` 只能作为敏感性证据。

主门控使用 near-full held-out R² 对归一化逻辑预算的 AUC，并且只使用 `predicted_class_margin`：

- controlled deletion-Möbius 相对 additive 的 sample-level bootstrap CI 下界大于 0；
- targeted top-edge magnitude 高于 matched random edge；
- 跨 seed hyperedge support 不完全不稳定；
- native deletion-Möbius 相对 ProxySPEX 的 CI 下界不低于 `-0.02`。

样本不足、只有一个预算点或证据互相冲突时，报告会保持 `Inconclusive`，不会用 raw-target 结果替主实验作结论。

## 测试

```bash
python3 -m py_compile $(find mobius_verify -name '*.py' -not -path '*/results/*')
python3 -m pytest mobius_verify/tests -q
```

ProxySPEX copy 测试：

```bash
PYTHONPATH=baselines/shapiq-copy/src \
python3 -m pytest \
  baselines/shapiq-copy/tests/shapiq/tests_unit/tests_approximators/test_approximator_proxyspex.py \
  -q
```

某些本地 Python/native 组合会让 pytest 在收集阶段段错误；此时应至少运行快速 Smoke，它覆盖真实的完整实验链路。

## 旧结构审计

旧的 exact-global、conditional-probe 和 limited-query recovery 不再是推荐主流程。相关脚本和数值模块暂时保留，用于 synthetic sanity、少量短文本 exact calibration 和历史结果复核。

特别注意：现有 `results/exact_default` 虽曾标记为 `predicted_class_margin`，实际收集的是 raw target verbalizer score。它只能作为 sensitivity/失败案例数据，不能与新的 margin 主实验直接合并。
