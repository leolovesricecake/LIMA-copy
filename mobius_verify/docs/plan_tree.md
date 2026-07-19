# Sparse Mobius Attribution：完整方案与实现计划

本文是 `mobius_verify` 后续实现的规范文件。研究主线从“大规模 exact structural audit”调整为：

> 先实现一个直接估计稀疏 Mobius 字典的 LLM 归因方法，在严格相同的 value function、特征、掩码和查询预算下，与 Fourier、GBT 和完整 ProxySPEX 比较；只保留足以校准系数和定位失败原因的最小 exact 实验。

实现主目录为 `mobius_verify`。现有代码优先复用和改造，不重新搭建另一套实验工程。若需要扩展 ProxySPEX，只修改 `baselines/shapiq-copy`，不修改 `baselines/shapiq-main`。

## 1. 研究对象与数学定义

### 1.1 Presence set function

把文本划分成 `n` 个 lexical-word players，记全集为：

```text
N = {0, 1, ..., n - 1}
```

对保留词集合 `S` 构造 masked input，定义：

```text
f(S) = model_value(x_S),  S subseteq N
```

Presence-Mobius coefficient 定义为：

```text
m_pre(T) = sum_{U subseteq T} (-1)^(|T|-|U|) f(U)
```

对应字典模型：

```text
f_hat(S) = beta_0 + sum_T theta_pre[T] * 1{T subseteq S}
```

精确、完整表示下 `theta_pre[T] = m_pre(T)`；低阶截断、候选裁剪或正则回归下，只称为：

```text
presence-Mobius surrogate coefficient
```

### 1.2 Deletion set function

令 `D` 表示被删除的特征集合，定义：

```text
g(D) = f(N \ D)
```

对 `g` 做普通 Mobius transform：

```text
m_del(T)
  = sum_{U subseteq T} (-1)^(|T|-|U|) g(U)
  = sum_{U subseteq T} (-1)^(|T|-|U|) f(N \ U)
```

对应字典模型：

```text
g_hat(D) = beta_0 + sum_T theta_del[T] * 1{T subseteq D}
```

在常见 AND/OR interaction 符号约定下，对非空集合 `T`：

```text
I_OR(T) = -m_del(T)
```

因此 deletion-Mobius 不是新 interaction index，也不作为额外 baseline；它只是删除变量坐标下的 Mobius/OR 表示。代码保存 `deletion_mobius`，需要兼容 OR 命名时只做显式符号转换。

### 1.3 与 Fourier 的关系

令 `x_i = 1{i in S}`，Fourier character 为：

```text
chi_T(x) = (-1)^|S intersect T| = product_{i in T} (1 - 2 x_i)
```

因此，不超过 `d` 阶的 Fourier 字典与不超过 `d` 阶的 Mobius 单项式张成同一个函数空间。由此固定以下表述边界：

- 不主张“低阶 Mobius 比低阶 Fourier 表达能力更强”。
- 无正则、全候选、同一数据上的最优 degree-`d` 拟合应具有相同函数空间。
- 比较重点是坐标稀疏性、正则化偏差、有限查询恢复、稳定性和归因语义。
- 孤立 Mobius 超边会展开为包含其子集的 Fourier support；“非层级 Mobius”不必然违反 Fourier hierarchy。

### 1.4 与 ProxySPEX 的关系

ProxySPEX 的完整链路是：

```text
sample coalitions
  -> query LLM
  -> fit tree proxy
  -> extract Fourier support from trees
  -> ridge-refine Fourier coefficients
  -> convert Fourier to Mobius
  -> convert Mobius to FBII/k-SII/etc.
```

所以本研究的区别不是“ProxySPEX 没有 Mobius interaction”，而是：

```text
ProxySPEX: tree-guided Fourier support discovery -> Mobius/index conversion
本方法:     direct sparse regression in a Mobius dictionary
```

## 2. 修订后的研究问题

### RQ1：坐标稀疏性

在相同低阶函数空间和相同观测数据上，presence/deletion-Mobius 是否能以更少的非零项达到与 Fourier 相同的 held-out faithfulness？

### RQ2：有限查询恢复

在相同黑盒查询预算下，直接 Mobius 回归是否比 additive、Fourier regression 和 GBT 更准确、更稳定？

### RQ3：删除坐标是否更符合 LLM 归因

相较 presence-Mobius，锚定完整输入的 deletion-Mobius 是否在 near-full masks、删除曲线和交互校准上更有优势？

### RQ4：交互是否可信

估计出的 top hyperedges 是否具有稳定 support、稳定符号，并与 exact targeted Mobius coefficients 一致？

### RQ5：完整方法是否有竞争力

在相同逻辑查询预算下，Sparse Mobius Attribution 是否能达到或超过完整 ProxySPEX 的 surrogate faithfulness、归因效果和查询效率？

## 3. 实验契约

### 3.1 Players

主实验继续使用现有 lexical-word featureization：

- 一个自然词是一个 player；
- subword 合并回原词；
- 标点附着到相邻词；
- 不把任意文本强行合并成固定数量的 blocks；
- 每个样本允许自己的 `n_i`；
- 保存 char spans、token spans 和可重建文本。

首轮只使用 word players。phrase/adaptive players 可在方法成立后作为粒度敏感性实验，不进入 MVP。

### 3.2 Mask operator

主实验使用 `delete`，未保留的完整 lexical word 及其附着空白/标点按现有 `FeatureSpec` 删除。

`replace` 只作为敏感性实验。所有方法在同一比较单元中必须使用相同 operator。

### 3.3 主 value function

对完整输入的所有 verbalizer 计算平均 conditional log probability：

```text
z_c(S) = mean conditional log probability of verbalizer c on x_S
```

目标类别固定为完整输入预测：

```text
c_star = argmax_c z_c(N)
```

主 value function 是 predicted-class margin：

```text
f_margin(S) = z_c_star(S) - max_{c != c_star} z_c(S)
```

`c_star` 在所有 masks 上固定，竞争类别允许随 mask 改变，并记录 competitor-switch rate。

Sensitivity analysis 使用：

```text
f_raw_target(S) = z_c_star(S)
```

物理 cache 保存完整 raw score vector `z(S)`，margin 与 raw-target 从同一次模型输出派生，不重复前向。

现有 `results/exact_default` 实际保存的是 raw target score，必须标记为 legacy sensitivity evidence，不能继续作为 margin 主实验。

### 3.4 默认阶数

MVP 只实现并主比较：

```text
degree = 1
degree = 2, all singleton and pair candidates
```

不使用 adjacency、句法局部性或父项显著性筛选。任何 pair 都可以在 singleton 未入选时独立进入 support。

Degree 3 只有通过阶段门控后才实现。

## 4. 查询、缓存与预算

### 4.1 Global ValueOracle cache

实现一个实验级全局 `ValueOracle`：

```text
logical mask request
  -> construct masked text
  -> look up global cache
  -> batch-score cache misses
  -> persist all-label raw scores
  -> derive requested value function
```

推荐使用 SQLite + WAL 保存实验级 cache：

```text
results/<experiment>/cache/value_oracle.sqlite3
```

物理 cache key 至少包含：

- model fingerprint；
- tokenizer 与 prompt-template version；
- verbalizers 及其顺序；
- normalized masked text；
- scorer normalization；
- dtype/关键推理语义配置。

Feature-spec digest、operator 和 bitmask 保存在逻辑查询记录中。物理 cache 以真实模型输入文本为核心，使不同 mask 产生相同文本时也能去重。

### 4.2 Per-method query ledger

每个 `protocol/method/sample/seed/budget` 有独立账本，至少记录：

```text
requested_queries
logical_unique_queries
duplicate_logical_requests
global_cache_hits
global_cache_misses
physical_forwards_caused
training_queries
candidate_discovery_queries
adaptive_queries
interaction_verification_queries
evaluation_only_queries
```

公平预算使用 `logical_unique_queries`：即使请求被其他方法预先缓存，也计入该方法预算。`physical_forwards_caused` 只用于实际运行成本分析，不能决定方法是否超预算。

完整输入查询同时用于确定 `c_star` 和 full coalition value，只计一个逻辑查询。Empty/full masks 均包含在 attribution budget 内。

### 4.3 Evaluation 隔离

- 方法拟合前不可访问 evaluation values。
- Controlled 协议预先划分 attribution/evaluation masks，严格不重叠。
- Native 协议先收集各方法 attribution masks，再从它们的并集之外生成共享 evaluation masks。
- Evaluation 可使用同一物理 cache，但必须在所有方法拟合完成后执行。
- 论文中单独报告 evaluation-only 前向，不把它们算进 attribution budget。

## 5. Sparse Mobius Attribution 算法

### 5.1 初始采样

MVP sampler：

- 强制加入 empty/full；
- 其余 mask 使用无放回 Bernoulli-0.5；
- 去重后必须精确满足逻辑预算，除非 `2^n` 已全部枚举；
- 所有随机性由 sample/seed 派生；
- 不在 MVP 中加入 adaptive querying。

Static mixture 是后续 ablation：

```text
50% Bernoulli-0.5
30% near-full
20% fixed-cardinality
```

是否使用 mixture 不能与 basis 选择同时改变。

### 5.2 候选字典

对 `degree=2`，完整生成：

```text
C = {{i}} union {{i,j}: i < j}
|C| = n + C(n,2)
```

Presence design：

```text
Phi_pre[S,T] = 1{T subseteq S}
```

Deletion design，令 `D = N \ S`：

```text
Phi_del[S,T] = 1{T subseteq D}
```

Fourier design：

```text
Phi_fourier[S,T] = (-1)^|S intersect T|
```

所有 design 支持 `float32`、分批构造和分批预测。运行前估算峰值内存；资源不足时记录 `infeasible_due_to_memory`，不静默裁剪候选或合并 words。

### 5.3 经验标准化

对实际 attribution masks 构成的每一列，在训练数据上计算：

```text
mu_T = mean(Phi[:, T])
sigma_T = std(Phi[:, T])
Phi_std[:, T] = (Phi[:, T] - mu_T) / sigma_T
```

- `sigma_T < eps` 的列标记为 `unidentifiable` 并删除；
- validation/evaluation 只使用训练期的 `mu_T/sigma_T`；
- Presence、deletion 和 Fourier 都执行相同经验标准化；
- 最终系数必须反变换回原始字典尺度。

### 5.4 Support selection 与 refit

首版估计流程：

1. 在 budget 内已查询的数据上做固定 folds 的交叉验证，不新增模型查询；
2. 用 LASSO 为主、ElasticNet 为配置项选择 support；
3. 超参数选择后在全部 attribution observations 上重新拟合；
4. 在 selected support 上做 ridge refit；
5. 若 `m > |support|` 且 support design 条件良好，补充 OLS debias 结果；
6. 主预测使用 validation 选中的 refit variant。

输出区分：

```text
selection_coefficients
refit_coefficients
original_basis_coefficients
```

所有 `theta` 默认称为 surrogate coefficients。只有 exact/targeted transform 得到的量称为 true Mobius coefficients。

### 5.5 诊断

保存：

- empirical max coherence；
- empirical effective rank；
- selected-support condition number；
- unidentifiable candidate count；
- convergence status 与迭代数；
- support size、degree distribution；
- 多 query seed 的 top-k Jaccard；
- selection frequency、sign consistency、coefficient variation。

不把全设计矩阵 condition number 当主指标，因为 `p >= m` 时它通常必然奇异。MVP 不做昂贵的完整 bootstrap confidence interval。

### 5.6 Hyperedge 与 node 输出

Primary output 是 hyperedges：

```text
players
degree
basis_orientation
surrogate_coefficient
selection_coefficient
refit_coefficient
sign
selection_frequency
```

Presence equal-share node score：

```text
score_pre[i] = sum_{T contains i} theta_pre[T] / |T|
```

Deletion score 使用“删除导致目标 margin 下降为正”的方向：

```text
score_del[i] = -sum_{T contains i} theta_del[T] / |T|
```

保存 positive、negative 和 absolute ranking。统一称为 `equal_share_node_score`；如转换为 Shapley/FBII，必须注明它是 surrogate set function 上的 interaction index。

## 6. Targeted exact verification

### 6.1 Deletion-Mobius

对一个 top hyperedge `T`，查询：

```text
{f(N \ U): U subseteq T}
```

共 `2^|T|` 个完整输入附近的删除组合，可精确计算：

```text
m_del_true(T)
```

直接比较：

```text
theta_del_hat(T) vs m_del_true(T)
```

报告绝对误差、归一化误差、符号一致和 rank agreement。这是 deletion coefficient 的精确 targeted calibration，不只是局部非加性 proxy。

### 6.2 Presence-Mobius

精确计算 `m_pre_true(T)` 需要：

```text
{f(U): U subseteq T}
```

即 `T` 外所有词均删除的 near-empty 输入。它同样只需 `2^|T|` 查询，但分布外风险更高，因此作为 sensitivity calibration。

如果对 presence hyperedge 使用 near-full 删除组合，则得到：

```text
Delta_T f(N \ T)
  = (-1)^|T| m_del(T)
  = sum_{V superset T} m_pre(V)
```

它只能验证固定完整背景下的聚合非加性，不能当作 `m_pre(T)` 真值。

### 6.3 验证协议

- 每个方法验证 top-5 非空 hyperedges；
- 使用相同阶数、相同数量的随机 hyperedges 作为 control；
- targeted queries 不进入 attribution budget；
- 进入独立 `interaction_verification_budget`；
- 通过全局 ValueOracle cache 去重；
- 先完成所有方法归因，再执行 targeted queries，避免信息泄漏。

## 7. 两种比较协议

### 7.1 Protocol A：Controlled-query comparison

所有方法共享完全相同的：

- players；
- attribution masks 和 values；
- CV folds；
- evaluation masks；
- mask operator；
- value function；
- 逻辑查询预算。

比较：

```text
additive_lasso
presence_mobius_lasso_d2
deletion_mobius_lasso_d2
fourier_lasso_d2
sklearn_gbt
proxyspex_fixed_observations  # 诊断项，不用于纯基底结论
```

该协议回答：同一批观测下，坐标系与估计器分别带来什么差异。

由于 degree-2 Mobius/Fourier 张成相同空间，需要额外加入无正则或弱 ridge 的全候选对照，验证二者最优函数空间拟合一致。LASSO 结果差异才可解释为稀疏坐标和正则化几何差异。

### 7.2 Protocol B：Native-method comparison

比较：

```text
sparse_mobius_attribution
official_proxyspex
additive baseline
```

允许每种方法使用自己的 sampler 和训练过程，但固定：

- 总 logical attribution budget；
- word players；
- predicted-class margin；
- mask operator；
- 共享且未见过的 evaluation masks；
- dataset/sample/seed。

该协议回答完整方法在实际使用时谁更好。ProxySPEX 必须执行树代理、Fourier extraction、refinement 和 interaction conversion 的完整链路，不能用 GBT prediction 冒充 ProxySPEX。

## 8. ProxySPEX 改造边界

ProxySPEX 改造限定在：

```text
baselines/shapiq-copy/src/shapiq/approximator/proxy/proxyspex.py
```

计划进行保持原算法语义的重构：

1. 将“拟合 proxy -> Fourier extraction -> refine -> Mobius/index conversion”抽成共享内部流程；
2. 现有 `approximate(budget, game)` 采样后调用该流程，保持行为不变；
3. 新增 `approximate_from_observations(X, y)`，支持 controlled masks；
4. 拟合后公开只读 diagnostics：

```text
coalitions_matrix_
coalition_values_
unrefined_fourier_
refined_fourier_
moebius_transform_
final_proxy_model_
```

5. 新增 refined-Fourier surrogate prediction，用于 held-out reconstruction；
6. 不改变 sampler、tree-to-Fourier、refinement、Fourier-to-Mobius 和 index converter 的数学逻辑；
7. 添加行为等价测试，确保相同 seed/game 下重构前后结果一致。

`mobius_verify/src/methods/proxyspex_adapter.py` 负责：

- 从 `baselines/shapiq-copy/src` 加载固定版本；
- 构造适合实际样本数的 proxy/HPO；
- 将 game 接到共享 ValueOracle 和 method ledger；
- 输出 refined Fourier surrogate、raw Mobius 和最终 FBII；
- 区分 controlled 与 native 模式。

不在本轮改造旧的 LIMA ProxySPEX runner 输出协议；新 benchmark 使用共享实验 harness，避免不同 value function 和 evaluator 混在一起。

## 9. 评价指标

### 9.1 Surrogate faithfulness

在共享 held-out masks 上按分布分别报告：

- uniform R2 / normalized RMSE / MAE；
- near-full R2 / normalized RMSE / MAE；
- fixed-cardinality R2 / normalized RMSE / MAE；
- query-to-R2 curve；
- AUC over normalized query budget；
- 达到 R2 0.8/0.9 所需逻辑查询数。

若某测试分布的 value variance 过低，标记为 degenerate，不伪造 R2。

### 9.2 Attribution faithfulness

使用同一 predicted-class margin evaluator：

- comprehensiveness；
- sufficiency；
- MoRF/LeRF 完整曲线；
- 多个删除比例；
- random-ranking control；
- additive-ranking control；
- 有人工 rationale 时报告 token/word span F1，但只作为次要指标。

现有 LIMA evaluator 以类别概率为主，不能直接作为主评价；需要在 `mobius_verify` 中实现共享 raw-score/margin evaluator。标准 LIMA probability 指标可作为兼容性附录。

### 9.3 Interaction quality

- targeted exact coefficient error；
- top-hyperedge sign accuracy；
- 相对随机 hyperedge 的 magnitude enrichment；
- 多 seed top-k Jaccard；
- selection frequency；
- coefficient sign consistency；
- 短文本 exact support precision/recall。

### 9.4 运行成本

- logical attribution queries；
- physical model forwards；
- cache hit rate；
- GPU scoring time；
- CPU fit time；
- peak memory；
- end-to-end wall-clock。

## 10. 最小 exact 与旧实验处理

保留但压缩 exact：

1. Synthetic exact：验证变换、符号、support recovery、遗漏高阶偏差；
2. 真实短文本：每任务 5-10 个 `n <= 10` 样本；
3. 正常文本：只做 top-5 targeted exact verification。

不再把数百个 conditional probes 的完整谱审计作为方法开发前置条件。

现有结果处理：

- 不删除 `results/exact_default`；
- metadata/报告中注明其 value 实际是 `raw_target_score`；
- 修复 `d90` 汇总只统计非 null 导致的误导；
- 作为 sensitivity 和失败案例库复用；
- margin 主实验使用新 ValueOracle 重新收集最小必要数据。

## 11. 实现策略：复用、改造与新增

### 11.1 直接复用

```text
mobius_verify/src/featureization.py
mobius_verify/src/masking.py
mobius_verify/src/schema.py
mobius_verify/src/datasets/
mobius_verify/src/models/
mobius_verify/src/transforms.py
mobius_verify/src/reconstruction_metrics.py
mobius_verify/src/subset_enumeration.py
mobius_verify/src/utils.py
```

这些模块继续承担 lexical words、文本 mask、数据/model adapter 和基础数值计算。

### 11.2 改造已有模块

```text
mobius_verify/src/value_functions.py
  - 真正实现 predicted-class margin
  - 同时暴露 raw-target sensitivity
  - 保存 all-label raw scores 和 competitor metadata

mobius_verify/src/fit_mobius.py
  - 改为经验标准化
  - presence/deletion 两种 orientation
  - support selection + refit
  - 返回原始尺度 coefficients 和 diagnostics

mobius_verify/src/fit_fourier.py
  - 使用同一标准化、CV、refit 和输出 schema

mobius_verify/src/subset_enumeration.py
  - 增加严格预算 sampler、evaluation exclusion 和分布化 test masks

mobius_verify/scripts/aggregate_results.py
  - 修正 null/censored d90
  - 以 sample 为统计单位
  - 区分 main margin 与 sensitivity raw score
```

### 11.3 新增模块

```text
mobius_verify/src/value_oracle.py
mobius_verify/src/query_ledger.py
mobius_verify/src/designs.py
mobius_verify/src/methods/__init__.py
mobius_verify/src/methods/sparse_mobius.py
mobius_verify/src/methods/controlled_baselines.py
mobius_verify/src/methods/proxyspex_adapter.py
mobius_verify/src/attribution_metrics.py
mobius_verify/src/interaction_verification.py
mobius_verify/src/benchmark.py

mobius_verify/scripts/run_attribution_benchmark.py
mobius_verify/scripts/aggregate_attribution.py

mobius_verify/configs/attribution_smoke.yaml
mobius_verify/configs/attribution_mvp.yaml
```

### 11.4 ProxySPEX copy 改造

```text
baselines/shapiq-copy/src/shapiq/approximator/proxy/proxyspex.py
baselines/shapiq-copy/tests/approximator/proxy/test_proxyspex.py
```

只增加 fixed-observation 路径、fitted diagnostics 和 surrogate prediction，并用测试保护现有行为。

## 12. 结果目录与 schema

```text
mobius_verify/results/<experiment>/
├─ run_config.json
├─ environment.json
├─ manifest.json
├─ cache/
│  └─ value_oracle.sqlite3
├─ features/<task>/<sample_id>.json
├─ protocols/
│  ├─ controlled/<method>/<task>/<sample>/<budget>/<seed>/
│  └─ native/<method>/<task>/<sample>/<budget>/<seed>/
├─ evaluation/<protocol>/<method>/...
├─ interaction_verification/<protocol>/<method>/...
├─ ledgers/<protocol>/<method>/...
├─ failures/
└─ aggregate/
   ├─ sample_metrics.csv
   ├─ query_metrics.csv
   ├─ interaction_metrics.csv
   ├─ hypothesis_results.json
   └─ hypothesis_report.md
```

每个 method result 至少保存：

```text
sample_id / task / seed / budget / protocol
n_features / mask_operator / value_function
train_masks_digest / evaluation_masks_digest
basis / orientation / max_degree
intercept / hyperedges / node_scores / rankings
fit_diagnostics / query_ledger / timing
surrogate_predictions_digest
status / failure_reason
```

所有长任务支持 resume、atomic write、skip completed 和显式 failure records。

## 13. 实现步骤与门控

### Step 0：契约测试与旧结果标注

- 为 presence/deletion/OR 符号写公式测试；
- 固定 predicted-class margin 公式；
- 标注旧 exact 输出实际为 raw target score；
- 修复旧 aggregate 的 null `d90` 统计。

完成条件：公式、配置名称和保存 metadata 一致。

### Step 1：ValueOracle 与 query ledger

- 实现 SQLite global cache；
- 实现 batch cache lookup/write；
- 实现每方法 logical ledger；
- 实现 margin/raw-target 从同一 score vector 派生；
- 测试运行顺序不改变任何方法的逻辑预算。

完成条件：两个方法请求同一 mask 时只发生一次物理评分，但两边各计一次逻辑查询。

### Step 2：统一稀疏字典估计器

- 实现 presence/deletion/Fourier designs；
- 经验标准化和不可识别列处理；
- LASSO/ElasticNet support selection；
- ridge/OLS refit；
- 原始尺度 coefficient 恢复；
- float32 与内存预检。

完成条件：无噪声 synthetic 上 full observations 可恢复已知 degree-2 函数；Mobius/Fourier degree-2 无正则拟合预测一致。

### Step 3：Sparse Mobius explanation output

- 生成 hyperedges；
- 生成 presence/deletion equal-share node scores；
- 输出正向/负向/绝对值 rankings；
- 实现统一 sample JSON schema。

完成条件：node 分摊满足对应 surrogate 的 full-empty efficiency identity。

### Step 4：Controlled protocol

- 实现共享 attribution/evaluation masks；
- 接入 additive、Mobius、Fourier、GBT；
- 运行 mock smoke；
- 在现有 exact tables 上做离线回归校准。

完成条件：所有方法使用完全相同的 observation digest，评价无数据泄漏。

### Step 5：ProxySPEX copy 扩展

- 重构共享 fit/extract 流程；
- 新增 `approximate_from_observations`；
- 保存 refined Fourier/Mobius diagnostics；
- 实现 held-out surrogate prediction；
- 添加原路径行为等价测试。

完成条件：原 `approximate()` 的固定 seed 输出和预算行为不变；controlled observations 可绕过内部 sampler。

### Step 6：Native protocol

- Sparse Mobius 和 ProxySPEX 分别使用 native sampler；
- ValueOracle 接入两个 method ledger；
- 生成排除 attribution-mask 并集的共享 evaluation masks；
- 保存 logical/physical query 与 wall-clock。

完成条件：同 budget 下逻辑查询公平，ProxySPEX 执行完整链路。

### Step 7：Targeted verification 与归因评价

- deletion top-edge exact coefficients；
- presence near-empty exact sensitivity；
- random matched-edge controls；
- margin 版 comprehensiveness/sufficiency/MoRF/LeRF；
- 多 seed stability。

完成条件：targeted queries 与 attribution budget 分账，且不会反向进入模型拟合。

### Step 8：真实 MVP

先运行一个模型、一个任务和有限样本：

```text
dataset: SST-2
model: Qwen2.5-7B-Instruct
players: lexical words
degree: 2
protocols: controlled + native
value: predicted-class margin
```

预算使用 `alpha * n log2(n)`，同时报告实际整数预算和固定预算对照。至少三个 query seeds。

### Step 9：Go/No-Go

只有同时出现以下信号，才进入 degree 3：

1. Degree-2 Mobius 相对 additive 在 primary held-out 分布上有稳定增益；
2. top deletion hyperedges 的 exact targeted calibration 优于随机 pair；
3. support/sign 在多 seed 下不是完全不稳定；
4. Native Sparse Mobius 对 ProxySPEX 至少非劣，或在 near-full/归因指标上存在明确优势。

若 degree 2 无增益，先判断是：

- value function 接近加性；
- Mobius design 不可识别；
- 查询预算不足；
- 结构高阶或稠密；
- ProxySPEX/GBT 确实更合适。

不能直接用 candidate screening 或 adaptive sampling 掩盖负结果。

### Step 10：条件扩展

通过门控后依次加入：

1. degree-3 all candidates，仅限短/中等文本；
2. residual screening，不要求父项入选；
3. all-triples 短文本对照，测 candidate recall；
4. static mixture sampler；
5. stability selection；
6. adaptive/group-testing sampler；
7. 更多任务、模型和 player granularity。

## 14. 预注册式判断标准

主指标：

```text
near-full held-out R2 AUC over normalized logical query budget
```

关键辅助指标：

- uniform held-out R2 AUC；
- matched-faithfulness support size；
- deletion targeted coefficient normalized error；
- top-edge sign accuracy；
- attribution deletion-curve AUC；
- multi-seed top-k Jaccard。

统计规则：

- 先在 sample 内聚合 masks 和 seeds；
- bootstrap 单位是原始 sample，不是 mask 或 probe；
- controlled 和 native 结论分开；
- margin 主实验与 raw-target sensitivity 分开；
- 报告 paired effect、置信区间和失败样本，不只报告平均数。

建议的 MVP 门槛不是论文最终结论，而是开发门控：

- 相对 additive 的 paired improvement 置信区间下界大于 0；
- targeted top-edge 指标优于 matched random control；
- 与 ProxySPEX 的 primary AUC 差异不低于预设 non-inferiority margin `-0.02`，或至少一个预注册 near-full/attribution 指标显著更好。

最终研究结论仍限定为：

```text
Supported
Partially supported
Not supported
Inconclusive
```

## 15. 明确不做的事情

MVP 不做：

- 把 deletion-Mobius/OR interaction 宣称为新指标；
- 声称回归系数就是真实 LLM Mobius interaction；
- 用 GBT prediction 代替完整 ProxySPEX；
- 强制 hierarchy 或 heredity support；
- degree-3/4 大规模候选筛选；
- adaptive querying；
- 大规模 exact enumeration；
- 用不同 value function 或 mask operator 比较方法；
- 把物理 cache hit 当作免费方法预算；
- 把 conditional probe 当作完整文本全局谱。

这条实现路线首先检验一个窄而可证伪的命题：

> 在相同低阶函数空间、相同 LLM 信息访问量和相同评价分布下，直接 presence/deletion-Mobius 稀疏估计能否产生比 Fourier/ProxySPEX 更紧凑、稳定、可校准的文本交互归因。
