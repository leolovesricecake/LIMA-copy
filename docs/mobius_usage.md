# Möbius 使用与论文实验说明

> 本文档说明通用方法接口。论文 v2.3 的冻结数据、模型、指标、审计顺序与
> 完整命令以 `docs/paper_experiment_usage.md` 为准。

## 1. 方法与 value function

文本先被切成 word chunks，active chunks 作为 players。keep mask `S` 表示保留哪些 players，模型的局部 masked function 为 `f(S)`。

Deletion-Möbius 固定定义为：

```text
g(D) = f(N \ D)
theta(T) = sum_{U subseteq T} (-1)^(|T|-|U|) g(U)
```

其中 `D` 是删除集合。它与 OR interaction 只差符号约定，不作为新指标。

每个样本的工作流为：

1. sampler 生成 attribution training keep masks；
2. LLM 输出每个扰动文本的全部 verbalizer label scores；
3. value function 将 label scores 转成目标类别标量值；
4. basis 构造低阶设计矩阵；
5. selector 使用 Lasso/ElasticNet CV 选择 support；
6. hierarchy 对 selection support 施加结构策略；
7. refit 在保留 support 上重新估计系数；
8. projector 将超边系数分配到节点并形成 ranking；
9. evaluator 按 word ranking 计算 faithfulness。

主配置中的 `value_function: target_probability` 与 `target_mode: predicted` 表示完整输入预测类别的概率，等价于 ProxySPEX 的 `predicted_probability`。

### 任务 prompt

`task_classification_v1` 使用如下协议，其中任务描述由数据集确定，候选标签来自 dataset verbalizers：

```text
Task: {task_description}
Candidate labels: label_1 | label_2 | ...
Return exactly one candidate label without quotation marks or explanation.
Text:
{text}
Label: {class_verbalizer}
```

label score 是 class verbalizer tokens 的平均条件 log probability，随后在全部候选标签间做 softmax。超过 `max_length` 时固定保留任务说明、候选标签、assistant/`Label:` prefix 和 verbalizer，只从原文左侧截断。实验前使用 `python scripts/check_classifier.py --config <config> --device <device>` 检查任务准确率；没有通过预先规定的准确率标准时，不应继续解释或报告 faithfulness。

对于带 `chat_template` 的 instruction model，上述内容作为 user message，并通过 tokenizer 的原生 chat template 生成 assistant prefix；Qwen3 显式设置 `enable_thinking=False`，使被评分的第一个输出就是类别 verbalizer。没有 chat template 的模型才使用包含 `Label:` 的 plain-text fallback。实际采用的完整布局、`use_chat_template` 和 thinking 开关写入 classifier report 与 `metrics.json`。

## 2. 三类 masks 与查询分账

| 类型 | 作用 | 是否共享 |
|---|---|---|
| attribution training masks | 拟合每种方法的归因表示 | 方法原生，不跨方法共享 |
| ranking evaluation masks | 评估各方法自己的 ranking | 只共享比例和算子，不共享具体 mask |
| surrogate held-out masks | 测试未查询 coalition 上的 surrogate | 待比较 runs 完全共享 |

成本分别记录为 attribution、ranking evaluation、surrogate audit、interaction verification 和 hierarchy analysis。后四类查询不得混入 attribution cost。

`observations/<sample-id>.npz` 保存二维 bool keep masks、全类别 scores、attribution values 和 digest。`surrogates/<sample-id>.json` 保存 predictor、candidate 定义、selection/hierarchy/refit support、各阶数量和 observation digest。

## 3. E2 的 A/B/C

| 版本 | 拟合模型 | 节点投影 | 主要比较 |
|---|---|---|---|
| A additive | degree 1 | singleton | A/B shared held-out |
| B interaction-fit-only | 与 C 相同的 degree 2 | singleton-only | B/C ranking faithfulness |
| C full interaction | degree 2 | signed equal share | A/C held-out 与 faithfulness |

A 和 C 使用相同 sampler、budget 和 seed，因此 training mask digest 必须相同。B 从 C 离线派生，predictor 和 surrogate digest 必须与 C 完全相同，只有节点投影和 ranking 不同。

运行 A/C：

```bash
python -m mobius.cli.run \
  --config configs/qwen3-8b/mechanisms-sst2/e2_a_additive.yaml \
  --seed 42 \
  --device cuda:0

python -m mobius.cli.run \
  --config configs/qwen3-8b/mechanisms-sst2/e2_c_interaction.yaml \
  --seed 42 \
  --device cuda:0
```

派生并评估 B：

```bash
python scripts/derive_projection_run.py \
  --input-run <C-run-dir> \
  --projector singleton_only \
  --output-root results/mobius-mechanisms \
  --run-suffix e2-b-fit-only

python -m mobius.cli.evaluate \
  --run-dir <B-run-dir> \
  --target predicted \
  --device cuda:0
```

## 4. Hierarchy 配置

### `none`

所有一阶和二阶候选共同进入 selector。任何 pair 都不要求 singleton 父项存在。

### `strict`

先在与 `none` 相同的完整候选上执行 selector，然后删除缺少任一 singleton 父项的二阶 selection term，最后使用相同 refit。第一版只允许 `max_degree <= 2`。这是 post-selection support-pruning 消融，不宣称为全局最优 hierarchical regression。

### `parent_screening`

先拟合 singleton，再只为选中的 singleton 生成 pair candidates。这是旧实现的准确名称；它改变候选空间，不能称为 strict。

`hierarchy: strong` 已被删除。程序会给出迁移提示，要求明确选择 `strict` 或 `parent_screening`。

配置：

- `configs/qwen3-8b/mechanisms-sst2/e4_strict.yaml`
- `configs/qwen3-8b/mechanisms-sst2/e4_parent_screening.yaml`
- `configs/qwen3-8b/ablations-sst2/hierarchy_strict.yaml`
- `configs/qwen3-8b/ablations-sst2/hierarchy_parent_screening.yaml`

## 5. Estimator 与 refit 消融

统一拟合过程分为标准化、CV selector、hierarchy support、refit 四步。

| `refit` | 含义 |
|---|---|
| `none` | 保留 selector 原始系数，不重新拟合 |
| `ridge_cv` | 在最终 support 上用 `ridge_alphas` 自动选择 Ridge alpha |
| `ridge_fixed` | 使用显式 `ridge_alpha` |
| `ols` | 使用最小范数最小二乘；保存 rank 和 condition diagnostics |

其他参数：

- `alphas`：selector 的 CV alpha 网格；
- `l1_ratios`：1.0 为 Lasso，小于 1.0 为 ElasticNet；
- `cv_folds`：selector CV 折数；
- `selection_alpha_scale`：将 CV 最优 alpha 乘以该比例，默认 1.0；
- `coefficient_tolerance`：定义 selection support 的非零阈值；
- `ridge_alphas`：`ridge_cv` 网格；
- `ridge_alpha`：仅供 `ridge_fixed` 使用。

`regularization_path` 保存每个 alpha/l1 ratio 的 CV MSE mean/std、完整拟合 support size，并标记 `cv_selected`。

## 6. 其他单轴消融

### Basis

- `deletion_mobius`：设计列为 `1{T subseteq D}`；
- `fourier`：相同 masks、values、degree 和 estimator，仅替换为 parity columns。

配置：`configs/qwen3-8b/ablations-sst2/basis_fourier.yaml`。

### Sampler

- `deletion_mixture`：anchors、global Bernoulli、near-full 和固定保留比例的混合；
- `bernoulli`：每个 player 独立以 0.5 概率保留；
- `uniform_size`：先均匀选 coalition size，再在该 size 内均匀选 coalition；这是主配置和代码默认值。

配置：`sampler_bernoulli.yaml`、`sampler_deletion_mixture.yaml`、`sampler_uniform_size.yaml`。

### Projector

- `signed_equal_share`：对 deletion Harsanyi dividend 做 Shapley 等分并取反，
  对应完整输入中的 feature-presence 方向；
- `absolute_equal_share`：均分绝对系数；
- `singleton_only`：只使用一阶项，忽略 pair 对 ranking 的贡献。

配置：`projector_absolute.yaml`、`projector_singleton.yaml`。

## 7. Shared held-out

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir <run-1> \
  --run-dir <run-2> \
  --output-dir results/audits/<dataset>/surrogate-heldout/<audit-name> \
  --cache-path results/.cache/value_oracle.sqlite3 \
  --count-per-distribution 64 \
  --min-count 16 \
  --seed 260726 \
  --device cuda:0

python scripts/evaluate_surrogates.py \
  --audit-dir results/audits/<dataset>/surrogate-heldout/<audit-name>
```

脚本严格校验 dataset、model、value semantics、text、chunks 和 active-player 映射。它合并所有 training masks 后抽取共同未见 masks，不做跨 run target-label 检查。

默认 `--distributions bernoulli`：每个 player 独立以 0.5 概率保留，所以每个 coalition 等概率。它是 held-out 采样策略，不是训练侧的 `uniform_size` sampler。可用 `--distributions bernoulli,near_full` 加入局部敏感性分布；`near_full` 按 `--near-full-deletions` 指定的删除数靠近完整输入采样。

审计目录的 `manifest.json` 在 `metadata` 中记录全部输入 run 目录、run ID、method、dataset、model、chunker 和 value semantics，在 `settings` 中记录分布、样本数、seed 与 mask 语义。
省略 `--output-dir` 时，默认目录为
`results/audits/<dataset>/surrogate-heldout/<audit-id>`。

每个分布输出每样本：

- R2；
- `RMSE / value_range`；
- MAE；
- value-range 退化标记。

当 value range 退化时 R2 和 NRMSE 为 `null`。run 的 `analyses/heldout-<audit-id>.json` 只保存指标摘要和公共 audit 引用。

## 8. E3 精确交互验证

### 8.1 普通长度 all-pair oracle

```bash
python scripts/audit_exact_pairs.py \
  --mobius-run <C-run> \
  --proxyspex-run <proxyspex-run> \
  --cache-path results/.cache/value_oracle.sqlite3 \
  --min-features 10 \
  --max-features 32 \
  --max-samples 50 \
  --seed 42 \
  --device cuda:0
```

每个样本统一查询 full、全部 singleton deletion 和全部 pair deletion，共
`1+n+C(n,2)` 个唯一 masks。输出 C/ProxySPEX/random/oracle 的完整 pair
ranking 与 coefficient error，论文中标记为 `pair_oracle`。

### 8.2 Targeted selected/random 诊断

对 pair `{i,j}` 查询：

```text
f(N)
f(N \ {i})
f(N \ {j})
f(N \ {i,j})
```

精确系数为：

```text
f(N \ {i,j}) - f(N \ {i}) - f(N \ {j}) + f(N)
```

运行：

```bash
python scripts/verify_interactions.py \
  --run-dir <seed-42-run> \
  --run-dir <seed-43-run> \
  --run-dir <seed-44-run> \
  --top-k 5 \
  --top-k-per-parent-group 3 \
  --seed 260726 \
  --device cuda:0
```

默认验证最终 refit 绝对系数最大的 5 条 pair，并为 0/1/2-parent 组各验证 top-3。随机对照在非 selected pairs 中精确匹配 player-index 距离；无精确候选时使用最近距离并记录 fallback。
省略 `--output-dir` 时，默认目录为
`results/audits/<dataset>/interactions/<audit-id>`。

## 9. E4 分析

```bash
python scripts/analyze_hierarchy.py \
  --none-run <none-run> \
  --strict-run <strict-run> \
  --verification-dir <interaction-audit> \
  --heldout-audit <heldout-audit> \
  --seed 260726 \
  --device cuda:0
```

脚本要求两者除 hierarchy 外所有科学配置一致，并硬校验每个样本的 observation digest。输出 support、held-out、原始 faithfulness、0/1/2-parent edge 组，以及移除每组边后的 faithfulness。新增模型调用仅写入 analysis query ledger。
省略 `--output-dir` 时，默认目录为
`results/audits/<dataset>/hierarchy/<audit-id>`。

## 10. 指标方向

- comprehensiveness：越高越好；
- sufficiency：越低越好；
- AOPC：越高越好；
- AUPC：越低越好；
- AOPC comprehensiveness：越高越好；
- AOPC sufficiency：越低越好；
- held-out R2：越高越好；
- held-out NRMSE、MAE：越低越好。

AUPC 复用完整 deletion trajectory，不增加模型调用。

## 11. 论文汇总

```bash
python scripts/collect_paper_results.py \
  --paper-version paper-v2.3 \
  --run A=<A-run> \
  --run B=<B-run> \
  --run C=<C-run> \
  --run PROXYSPEX=<proxyspex-run> \
  --run STRICT=<strict-run> \
  --audit-dir <representation-audit> \
  --audit-dir <heldout-audit> \
  --audit-dir <exact-pair-audit> \
  --audit-dir <interaction-audit> \
  --audit-dir <hierarchy-audit> \
  --bootstrap 2000

python scripts/collect_paper_results.py \
  --paper-version paper-v2.3 \
  --summarize
```

输出：

- `overall-attribution.csv`：各方法 faithfulness mean/std；
- `paired-effects.csv`：C 对其他方法的逐样本配对统计；
- `costs.csv`：attribution 查询与 forward 成本；
- `audit-costs.csv`：held-out、pair oracle 等分析查询成本；
- `surrogate-heldout.csv`：shared held-out 重构与 C 对基线的配对统计；
- `representation-recovery.csv`：Möbius/Fourier 恢复与 compression；
- `exact-structure.csv`：完整 value table 的 degree/top-k 结构；
- `exact-interactions.csv`：区分 full-table/pair-oracle 的 interaction 指标；
- `projection-ablation.csv`：C vs B/ABSOLUTE；
- `integrity-checks.json`：受控 masks 与 derived surrogate 检查。

详细路径与字段见 `docs/paper_experiment_usage.md`。

常规紧凑 run 汇总仍使用：

```bash
python scripts/collect_results.py \
  --input_dir results/mobius \
  --o results_summary.csv
```
