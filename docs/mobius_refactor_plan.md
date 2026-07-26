# Möbius 主线重构实施计划

## 1. 目标

本轮把项目从“LIMA 的 LLM 迁移实验”重构为“低阶稀疏 Möbius LLM 归因研究代码库”。

完成后：

- 删除 `lima_llm/` 命名空间和 LIMA 方法专属代码；
- 新建统一的 `mobius/` Python 包；
- Möbius 主方法、公共 LLM attribution 基础设施和 evaluator 都位于 `mobius/`；
- 保留并迁移 AML、Inseq、`shapiq-copy` 三个 baseline；
- 旧 exact/probe/recovery 验证流程移动到 `archive/mobius_validation/`；
- 不兼容旧结果，所有新实验使用精简的 result schema v2；
- 完整实现 basis、hierarchy、sampler、projector 四个消融轴；
- 根 README、配置、命令和测试都以 Möbius 为主线。

## 2. 目录结构

目标结构：

```text
mobius/
├── __init__.py
├── core/
│   ├── config.py
│   ├── schema.py
│   ├── results.py
│   ├── resume.py
│   └── runtime.py
├── data/
│   └── loader.py
├── models/
│   ├── base.py
│   ├── hf.py
│   ├── mock.py
│   └── oracle.py
├── text/
│   ├── chunks.py
│   ├── coalitions.py
│   └── masking.py
├── values/
│   └── classification.py
├── methods/
│   └── sparse/
│       ├── basis.py
│       ├── hierarchy.py
│       ├── sampler.py
│       ├── estimator.py
│       ├── projector.py
│       └── explainer.py
├── evaluation/
│   ├── metrics.py
│   ├── units.py
│   ├── evaluator.py
│   └── compare.py
└── cli/
    ├── run.py
    ├── evaluate.py
    └── compare.py

configs/
├── mobius/
│   ├── sst2.yaml
│   ├── emotion.yaml
│   ├── rotten_tomatoes.yaml
│   └── smoke.yaml
└── ablations/
    ├── basis_fourier.yaml
    ├── hierarchy_parent_screening.yaml
    ├── hierarchy_strict.yaml
    ├── sampler_bernoulli.yaml
    ├── sampler_deletion_mixture.yaml
    ├── sampler_uniform_size.yaml
    ├── projector_absolute.yaml
    └── projector_singleton.yaml

archive/
├── mobius_validation/
└── baselines/
    ├── captum/
    └── shapiq-main/
```

## 3. `lima_llm` 迁移映射

保留并改名：

| 原模块 | 新模块 | 处理 |
|---|---|---|
| `attribution_text.py` | `mobius/text/coalitions.py` | 保留 prompt、截断与 coalition 文本协议 |
| `attribution_values.py` | `mobius/values/classification.py` | 保留统一 value function |
| `backbone/base.py` | `mobius/models/base.py` | 精简为分类 attribution 所需接口 |
| `backbone/hf_backbone.py` | `mobius/models/hf.py` | 保留 verbalizer scorer 和计数器 |
| `backbone/mock_backbone.py` | `mobius/models/mock.py` | 保留 smoke tests |
| `chunking/adaptive.py` | `mobius/text/chunks.py` | 与 token/word chunk 统一公共接口 |
| `chunking/explanation.py` | `mobius/text/chunks.py` | 合并，删除重复 wrapper |
| `chunking/utils.py` | `mobius/text/chunks.py` | 只保留 coverage、compose 等公共函数 |
| `data/loader.py` | `mobius/data/loader.py` | 保留数据集和离线 cache 支持 |
| `eval/metrics.py` | `mobius/evaluation/metrics.py` | 保留论文 faithfulness 指标 |
| `eval/units.py` | `mobius/evaluation/units.py` | 保留 chunk 到 eval unit 投影 |
| `eval/evaluate.py` | `mobius/evaluation/evaluator.py` | 按 schema v2 重写输出层 |
| `pipeline/io.py` | `mobius/core/results.py` | 按 schema v2 重写 |
| `pipeline/resume.py` | `mobius/core/resume.py` | 简化为 sample completion 检查 |
| `types.py` | `mobius/core/schema.py` | 删除 LIMA score components |
| `utils.py` | `mobius/core/runtime.py` 等 | 按职责拆分 |

删除：

- `lima_llm/objective/`
- `lima_llm/scoring/`
- `lima_llm/search/`
- `lima_llm/pipeline/explainer.py`
- `lima_llm/pipeline/run.py`
- sentence/fixed-token 等仅服务旧 LIMA 的 chunker
- LIMA lambda、component、search、submodular 等脚本和测试

迁移完成后执行全仓检查，活动代码中不得再出现 `import lima_llm`。

## 4. `mobius_verify` 拆分

迁入 Möbius 主线：

- `sparse_runner.py`
- `methods/sparse_mobius.py`
- `methods/sparse_surrogate.py`
- `designs.py`
- `subset_enumeration.py` 中 attribution sampler
- `query_ledger.py`
- `value_oracle.py`
- `interaction_verification.py` 中 targeted deletion coefficient
- `statistics.py`
- `compare_method_runs.py`
- 三个正式数据集配置和 smoke 配置

移动到 `archive/mobius_validation/`：

- exact spectra
- conditional probes
- synthetic suite
- medium-n
- limited-query recovery
- oracle approximation
- 旧 figures/aggregate 脚本
- 对应旧配置、测试和研究过程文档

归档代码只用于追溯，不要求继续通过主线测试，也不得被活动代码 import。

## 5. 可配置消融架构

### 5.1 公共流水线

统一流程：

```text
text
 -> chunks/players
 -> sampler.sample(n, budget)
 -> ValueOracle(masks)
 -> basis.design(masks, terms)
 -> hierarchy.candidates(...)
 -> sparse estimator
 -> projector.node_scores(hyperedges)
 -> ranking
 -> schema v2 result
```

公共协议：

```text
BasisEncoder
HierarchyPolicy
CoalitionSampler
SparseEstimator
InteractionProjector
```

每个组件必须：

- 有稳定字符串 ID；
- 将解析后的配置写入 `run.json`；
- 将必要诊断写入 sample 的 `method_summary`；
- 能单独做 deterministic unit test。

### 5.2 Basis

```yaml
basis: deletion_mobius
basis: fourier
```

- `deletion_mobius`：`g(D)=f(N\D)`，列为 `1{T subseteq D}`。
- `fourier`：同一 keep/delete 二值域上的 parity basis。
- 两者使用相同 observations、candidate degree、hierarchy、estimator 和 projector。

暂不把 presence-Möbius 列入正式消融，避免增加与核心假设无关的实验轴。

### 5.3 Hierarchy

```yaml
hierarchy: none
hierarchy: parent_screening
```

- `none`：当前行为，全部一阶和二阶候选共同进入稀疏选择。
- `strong`：两阶段 strong-heredity screening。
  1. 在相同 observations 上拟合一阶 sparse model；
  2. 保留入选一阶项；
  3. 最终候选仅包含入选一阶项，以及两个父项均入选的二阶项；
  4. 使用与 `none` 相同的 estimator 重新拟合。

禁止使用“先拟合完整模型，再事后删除不满足 hierarchy 的边”，因为那不能隔离 hierarchy 对恢复过程的影响。

### 5.4 Sampler

```yaml
sampler:
  name: deletion_mixture

sampler:
  name: bernoulli

sampler:
  name: uniform_size
```

- `deletion_mixture`：sampler 消融，full/empty anchors + Bernoulli 全局 + near-full + fixed-cardinality。
- `bernoulli`：除 anchors 外，每个 player 以 0.5 独立保留。
- `uniform_size`：先均匀采样 coalition size，再在该 size 内均匀采样 coalition；当前主配置与代码默认值。

三者都必须：

- 产生唯一 masks；
- 使用严格逻辑 budget；
- 在 `2^n < budget` 时精确枚举；
- 输出最小必要的 cardinality/source diagnostics。

### 5.5 Projector

```yaml
projector: signed_equal_share
projector: absolute_equal_share
projector: singleton_only
```

- `signed_equal_share`：当前主方法；deletion 坐标包含方向负号。
- `absolute_equal_share`：按 `|theta_T|/|T|` 分配，用于检查符号是否影响 ranking。
- `singleton_only`：只使用一阶 coefficient，用于检查 faithfulness 增益是否真正来自交互。

Projector 只改变 hyperedge 到 node score/ranking 的映射，不改变 observations 或拟合结果。

### 5.6 Estimator

本轮 estimator 固定为当前的：

```text
standardize -> Lasso/ElasticNet CV -> selected-support Ridge refit
```

Estimator 不是本轮正式消融轴，但实现为独立模块，以便后续替换。所有正式消融使用相同 alpha grid、CV、refit 和随机种子。

## 6. 当前已有结果对应的配置

### 6.1 Sparse deletion-Möbius 主结果

重跑后的 SST-2、Emotion、Rotten Tomatoes 主配置对应：

```yaml
method: sparse
basis: deletion_mobius
max_degree: 2
hierarchy: none
sampler:
  name: uniform_size
projector: signed_equal_share
estimator: lasso_cv_ridge_refit
budget: 512
seed: 42
chunker: word
eval_granularity: token
value_function: target_probability
target_mode: predicted
k: 8
# 交互验证已迁移到 scripts/verify_interactions.py。
```

SST-2 和 Emotion 另有 sensitivity run，仅把：

```yaml
value_function: predicted_class_margin
```

其余结构配置不变。

所以当前已有结果就是新消融空间中的 baseline cell：

```text
deletion_mobius × hierarchy_none × uniform_size × signed_equal_share
```

### 6.2 ProxySPEX 主结果

当前三数据集 ProxySPEX run 对应：

```yaml
method: proxyspex
index: FBII
max_order: 2
budget: 512
proxy_model: lightgbm
hpo: true
sampling_weight_mode: uniform_coalition
pairing_trick: false
top_order: false
chunker: word
eval_granularity: token
value_function: predicted_probability
target_mode: predicted
k: 8
seed: 42
projector: signed_equal_share
```

ProxySPEX 保持 native sampler、tree proxy、Fourier extraction、refinement 和 FBII conversion，不接入 Möbius 的训练 masks。

## 7. Result schema v2

### 7.1 目录

```text
results/<dataset>/<model>/<method>/<run-id>/
├── run.json
├── status.json
├── metrics.json
├── samples/
│   └── <sample-id>.json
└── diagnostics/                 # standard/debug 按需创建
```

`run-id`：

```text
b<budget>-o<order>-s<seed>-<config-hash-8>
```

method、dataset、model 已由上级目录表达，不再重复进入 run ID。config hash 基于去除 device、output path 等运行位置字段后的 canonical scientific config。

### 7.2 `run.json`

只保存一次：

- `schema_version`
- `run_id`
- dataset/split/sample selection
- model path、dtype、max length、prompt、verbalizers
- chunk/eval unit 配置
- value function 和 target mode
- method 及四个消融组件的完整配置
- budget、seed
- evaluator 协议和 q values
- command、git commit、时间和环境摘要

### 7.3 `status.json`

- `state`: running/complete/complete_with_failures
- selected/completed/failed/skipped counts
- completed IDs
- failure 摘要
- run-level wall time

它是 resume 唯一的 run-level 状态来源，不再重复 method config。

### 7.4 Sample result

`minimal`：

- schema version、sample ID、gold label
- predicted/target label
- chunks 的 ID、span、text
- node scores、完整 ranking、selected IDs
- attribution logical queries、physical texts、model forwards、elapsed time

`standard` 默认额外保存：

- 非零 hyperedges：players、degree、coefficient
- fit 摘要：candidate/support count、alpha、CV loss、train R²/NRMSE、convergence
- sampler 摘要：realized budget、cardinality/source counts
- targeted verification 聚合摘要

`debug` 额外保存到 `diagnostics/<sample-id>.json`：

- masks
- raw coalition values
- all-class raw scores
- per-term selection coefficient
- verification 的具体 masks/values
- 完整 counter/timing breakdown

删除：

- 旧 LIMA `confidence/effectiveness/consistency/collaboration`
- 重复的 segmentation/fallback 字段
- 同一配置在每个 sample 中的重复副本
- trace 中对静态 node score 的机械累加
- provenance 在 run/eval/sample 中重复出现

### 7.5 `metrics.json`

- `target: predicted|gold`
- sample/evaluated/failed count
- accuracy
- faithfulness aggregate
- per-q aggregate
- curve artifact 引用
- evaluation model forward/time
- method attribution cost aggregate

不再保留含义模糊的顶层 `metrics_primary`。所有指标只属于一个显式 target；如需同时计算 gold 和 predicted，写入两个明确命名的 target block。

## 8. Baseline 迁移

### 8.1 ProxySPEX

- 保留 `baselines/shapiq-copy`；
- runner 改用 `mobius` 的 data/model/chunk/value/evaluation/result API；
- ProxySPEX 原生算法不变；
- 输出 schema v2；
- 删除 `proxyspex_chunking.py` wrapper，直接用公共 chunk API；
- `baselines/shapiq-main` 移到 `archive/baselines/shapiq-main`。

### 8.2 Inseq

- 保留 `baselines/inseq`；
- 改用 schema v2；
- 只迁移当前支持且能测试的 LLM attribution runner；
- 不复制公共 evaluator 或数据逻辑。

### 8.3 AML

- 保留 `baselines/aml-main_copy` 原始实现；
- `shared_task_data.py` 改为导入 `mobius.data`；
- 如 AML 结果要进入统一论文表格，提供离线 adapter，不侵入原训练代码。

### 8.4 不保留的 baseline

- Captum 移到 `archive/baselines/captum`；
- `shapiq-main` 移到归档；
- 活动测试不再覆盖它们。

## 9. 配置与 CLI

主命令：

```bash
python -m mobius.cli.run --config configs/mobius/sst2.yaml
python -m mobius.cli.evaluate --run-dir <run-dir> --target predicted
python -m mobius.cli.compare --left <run-a> --right <run-b>
```

配置加载后必须：

- 校验未知字段；
- 规范化 aliases；
- 输出 resolved scientific config；
- 生成短 run ID；
- 不允许不同 scientific config resume 到同一目录。

## 10. 提交与分支

1. 将当前工作树中已完成的 LIMA 清理、target 指标修复、Möbius/Proxy 改造和本计划提交到当前分支。
2. 不提交 `.DS_Store` 和 `docs/lima.zip`。
3. 从该提交创建并切换到新分支：

```bash
git switch -c mobius
```

4. 所有重构修改只发生在 `mobius` 分支。

## 11. 实施任务树

### Phase A：建立新主线

- 创建 `mobius/` 包骨架和 root `pyproject.toml`；
- 迁移 schema、models、data、text、values；
- 迁移 evaluator 核心；
- 添加 import/compile smoke；
- 暂时保留旧目录，直到消费者迁移完成。

### Phase B：结果协议 v2

- 实现 canonical config、short run ID；
- 实现 `run.json`、`status.json`、sample writer；
- 实现 output levels；
- 重写 evaluator 输出 `metrics.json`；
- 添加 schema snapshot 和 resume tests。

### Phase C：Sparse 方法模块化

- 迁移 ValueOracle、query ledger；
- 抽取五个组件协议；
- 实现两个 basis；
- 实现两种 hierarchy policy；
- 实现三个 sampler；
- 实现三个 projector；
- 迁移 estimator 和 targeted verification；
- 提供当前主配置与六个单轴消融配置。

### Phase D：Baseline 迁移

- 迁移 ProxySPEX runner；
- 迁移 Inseq runner；
- 迁移 AML data bridge；
- 归档 Captum 和 `shapiq-main`；
- 验证活动 baseline 不再 import `lima_llm`。

### Phase E：归档与删除

- 把旧 Möbius 验证流程移动到 archive；
- 删除 `mobius_verify/` 活动目录；
- 删除 `lima_llm/`；
- 删除/归档旧 LIMA scripts、tests、README 内容；
- 清理 pycache、DS_Store 和失效文档引用。

### Phase F：验证

- unit：basis 数值、hierarchy candidates、sampler budget、projector conservation；
- unit：schema v2、run ID、resume、output levels；
- unit：target-specific evaluator；
- integration：mock Sparse end-to-end；
- integration：ProxySPEX fake approximator；
- smoke：Inseq/AML import；
- static：全仓无活动 `lima_llm` import；
- static：`py_compile`、`git diff --check`；
- 若 pytest 环境仍在 capture 初始化段错误，直接执行测试函数并记录。

## 12. 完成标准

- 当前分支存在重构前基线提交，重构位于 `mobius` 分支；
- 活动代码只使用 `mobius` 公共包；
- `lima_llm/` 和 `mobius_verify/` 不再作为活动目录存在；
- 三个保留 baseline 不依赖旧命名空间；
- schema v2 的 minimal/standard/debug 都有测试；
- 当前主方法可由新配置完整表达；
- 四个消融轴均可运行 mock end-to-end；
- 根 README 能从安装到运行、评估、对比和消融完整复现。
