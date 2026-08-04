# 论文实验执行计划 v2.3

## 1. 目标与范围

本计划以 `docs/paper-outline-v2.3.md` 为论文叙事基础，落实以下四个研究问题：

1. Sparse Deletion-Mobius 是否改善词级归因忠实度；
2. 有限删除查询能否恢复未见 mask 上的模型行为；
3. 恢复出的 feature pairs 是否对应 exact deletion interactions；
4. 恢复出的 interactions 是否真正改善最终词级 ranking。

正式实验继续使用完整数据集评估，不构建额外的 evaluation sample pool，也不根据分类准确率筛除 dataset-model cell。

## 2. 已冻结的实验协议

### 2.1 数据、模型与目标

- 数据集：SST-2 `validation`、Rotten Tomatoes `validation`、AG News `test`；
- 模型：Qwen3-8B、Llama-3.1-8B-Instruct；
- E1 在两个模型上运行，E2-E4 主要使用 Qwen3-8B；
- explanation chunk 与 evaluation granularity 均为 `word`；
- target 为完整输入的 predicted class，并在所有扰动上保持不变；
- value function 为候选 verbalizers 间归一化的 predicted-class probability；
- 主方法使用 degree 2、`uniform_size` sampler、无 hierarchy 和 signed Shapley projection；
- attribution seeds 为 42、43、44，主预算为 512；
- E2 的预算曲线使用 64、128、256、512。

不设置分类有效性门槛，但仍运行分类诊断并报告 accuracy、balanced accuracy、per-class recall 和 confusion matrix。分类结果仅用于说明实验对象的任务能力，不决定样本或实验 cell 是否进入论文结果。

### 2.2 Faithfulness 指标

主结果保留：

- **AUPC（越低越好）**：完整逐词删除 probability curve 的归一化梯形面积；
- **AOPC-Sufficiency（越低越好）**：在 `q={5,10,20,50}` 上计算
  `f_y(x)-f_y(keep_top_q(x))`，并沿用当前包含隐式零点的 AML 聚合口径。

不实现完整 `AUSufficiency`。当前 evaluator 只查询完整 deletion trajectory 和四个 retention 比例。完整 retention trajectory 需要额外查询最多 `n+1` 个前缀保留文本，除端点外不能从 deletion trajectory 推导，因此会增加真实 LLM evaluation queries。AOPC-Sufficiency 可以直接复用现有四个 retention 比例，不增加查询。

现有 comprehensiveness、sufficiency、AOPC 和 AOPC-Comprehensiveness 继续写入结果，作为补充诊断，不进入主表。

### 2.3 Function reconstruction 指标

正文报告：

- held-out `R2`（越高越好）；
- held-out `NRMSE_range`（越低越好）。

对样本 `r` 的 held-out values，定义：

```text
RMSE_r = sqrt(mean_m((y_rm - yhat_rm)^2))
range_r = max_m(y_rm) - min_m(y_rm)
NRMSE_range,r = RMSE_r / range_r
```

先逐样本计算，再跨样本做 macro mean 和 sample standard deviation。若 `range_r <= 1e-12`，该样本对 NRMSE 记为 degenerate，不进入均值，并单独报告 degenerate count。

不改用 `NRMSE_std`，理由如下：

- `NRMSE_std = RMSE/std(y)`；
- 在相同逐样本 held-out values 上，它严格满足 `NRMSE_std=sqrt(1-R2)`；
- 因而它与已经报告的逐样本 `R2` 不提供独立信息；
- `NRMSE_range` 则补充描述相对于该局部函数实际变化范围的预测误差。

## 3. ProxySPEX 的 E3 对齐方式

### 3.1 不修改 ProxySPEX 原生方法

ProxySPEX 的 attribution 仍完整保留其原生流程：

```text
native coalition sampling
-> GBT proxy fitting
-> tree Fourier extraction
-> coefficient refinement
-> FBII interaction conversion
-> native feature ranking
```

E1 中使用的仍是 ProxySPEX 原生 FBII interactions 和原生 ranking，不用 deletion-Mobius pair 替换其输出。

### 3.2 仅在 E3 做坐标对齐后的离线评价

FBII interactions 与 deletion-Mobius coefficients 不是同一对象，不能直接拿 FBII 系数和 exact deletion pair 比较。E3 改为把 ProxySPEX 已学习的 `refined_fourier` surrogate 视为函数 `ghat`，对每个 pair 离线计算：

```text
mhat^-_ij = ghat({i,j}) - ghat({i}) - ghat({j}) + ghat(empty)
```

序列化 predictor 接收 keep masks，因此实际批量预测：

```text
N, N\{i}, N\{j}, N\{i,j}
```

该操作只评价 ProxySPEX 已经拟合出的函数，不重新采样、不重新拟合、不修改 support、不修改 FBII 输出，也不调用 LLM。因此它不会改变 ProxySPEX 的正确性或原生方法，只是把两个方法的 surrogate 投影到同一个 deletion-interaction 评价坐标。

实现时必须增加一致性断言：序列化 `refined_fourier` predictor 在训练 masks 和一组确定性检查 masks 上的预测应与 ProxySPEX 原生 `predict_refined_fourier()` 一致；最大绝对误差超过预设浮点容差时，该样本的 E3 结果无效并显式报错。

### 3.3 E3 指标与样本

- 每个数据集从满足 exact-pair audit 长度条件的样本中随机抽取 100 条；
- 保持当前成本边界 `10 <= n_features <= 32`；
- 不做长度分层，使用固定 audit seed 随机抽取；
- exact oracle 的 LLM values 跨方法、跨 attribution seed 共用全局缓存；
- 每个 attribution seed 分别评价恢复结果，最终先按样本跨 seed 求平均。

正文指标改为：

- NDCG@10；
- Recall@10；
- Sign Agreement@10。

对 pair 数不足 10 的样本使用 `K=min(10,C(n,2))`。ProxySPEX 和 Sparse Mobius 都通过各自序列化 surrogate 的四点差分生成 estimated deletion-pair scores，确保 interaction semantics 完全一致。

## 4. E1-E4 实验矩阵

### E1：Overall Attribution Faithfulness

目的：验证完整方法在全集样本上的词级 ranking 是否更忠实。

比较方法：

- Sparse Deletion-Mobius；
- Additive Deletion；
- ProxySPEX；
- word-level Occlusion；
- word-level LIME。

Occlusion 和 LIME 使用与主方法相同的 task-aware prompt、word players、固定 predicted target 和 predicted-class probability。原生 Inseq 结果可以保留为附录，但不作为“相同 value function”的主表基线。

主表报告 AUPC 和 AOPC-Sufficiency。所有正式 split 样本都进入运行；方法失败数、共同完成样本数和逐样本配对覆盖率必须同时报告，配对统计只在共同成功样本上计算。

### E2：Finite-Query Function Recovery

目的：比较 interaction modeling、表示坐标和原生 ProxySPEX 在有限查询下的函数恢复能力。

端到端比较：

- budgets：64、128、256、512；
- 方法：Sparse Mobius、Additive Deletion、ProxySPEX；
- 每个方法保留自己的训练 masks；
- shared held-out 使用 Bernoulli-0.5 masks，并排除参与比较的全部 training masks；
- 主图横轴使用 realized logical unique queries，而不是只使用配置预算。

受控 basis 比较：

- 从 Sparse Mobius observation sidecar 读取完全相同的 masks 和 values；
- **Actual-estimator comparison**：Mobius 与 Fourier 使用相同 degree、列标准化、Lasso support selection 和 RidgeCV refit，与主方法的实际 estimator 协议一致；
- **Fixed-k diagnostic**：两种 basis 使用相同的确定性 standardized OMP 规则构建嵌套 support path，再对每个固定 support 使用 OLS refit；
- fixed-k support grid：1、2、4、8、16、32；
- 比较 held-out R2，并以 NRMSE-range 作为辅助指标。

OMP + OLS 只是 fixed-k 坐标压缩性诊断，不是 Sparse Deletion-Mobius 的默认估计器。这个诊断用完全相同的 support size 比较两种 basis，避免 Lasso 在不同 basis 上选出不同 support 数而混淆“表示是否更易压缩”的判断。OLS 则在 support 确定后去掉稀疏正则的系数收缩，使诊断主要反映 support 的表示能力。实现需同时记录每次 OLS 的设计矩阵秩、rank-deficient 标记和 condition number；论文汇总应报告病态拟合数量，避免把数值不稳定误判为 basis 差异。

### E3：Exact Pair Interaction Recovery

目的：直接检验 estimated top pairs 是否对应 exact deletion interactions。

查询真实模型的 full、所有 singleton deletion 和所有 pair deletion；这些查询只属于 audit，不计入 attribution budget。Sparse Mobius 与 ProxySPEX 均按第 3 节的 surrogate 四点差分得到同语义 pair scores。

### E4：Interaction-Aware Projection

所有变体复用 C 的 observations、support、coefficients 和 surrogate，仅改变 projection：

- Singleton-only Projection；
- Signed Shapley Projection；
- Sign-agnostic Projection。

Sign-agnostic 定义为先对每条 hyperedge coefficient 取绝对值，再 equal-share 到成员词。派生过程不调用 LLM；不同 ranking 的 faithfulness evaluation queries 仍单独记入 evaluation cost。

## 5. 代码改造任务树

### P0：协议与数据支持

1. 为 `mobius/data/loader.py`、ProxySPEX runner 和 first-order runner 增加 AG News；
2. AG News 固定使用 `wangrongsheng/ag_news`、`test` split 和四类 verbalizers；
3. 新增 Qwen3-8B 与 Llama-3.1-8B-Instruct 的三数据集正式配置；
4. 所有正式配置显式写 `predicted_probability`、word/word 和 `uniform_size`；
5. 保留分类诊断脚本，但移除 threshold gate 的正式流程要求。

### P1：指标协议

1. 不增加完整 retention trajectory；
2. 明确固化 AOPC-Sufficiency 的 q-grid、分母和方向；
3. 保留并规范命名 `nrmse_range`；
4. aggregation 使用逐样本 macro 统计并报告 degenerate count；
5. 更新 paper collector，使主 faithfulness 表只展示 AUPC 与 AOPC-Sufficiency。

### P2：First-order baselines

1. 实现共享 value-function 的 word Occlusion；
2. 实现共享 value-function 的 word LIME；
3. 保持 prompt 固定，只扰动原始文本 words；
4. 写入 schema-v2 sample、curve、metrics 和 query-cost artifacts；
5. 原生 Inseq runner 保留，不伪装成与主方法相同的 masked-function surrogate。

### P3：E2 与 E3 audits

1. shared held-out 支持一次登记多个 budgets 和 seeds 的 runs；
2. 结果中记录 requested budget、realized unique queries 和被排除 masks；
3. E3 改为从 serialized predictors 批量计算所有 pair 四点差分；
4. 增加 ProxySPEX predictor serialization invariance check；
5. 实现 NDCG@10、Recall@10、Sign Agreement@10；
6. 将 exact-pair 默认样本数改为 100，并取消现有 length-stratified round robin。

### P4：E4 与结果收集

1. 将 Sign-agnostic 加回 projection ablation 主表；
2. paper cell 路径加入 `budget-<B>`，避免多预算覆盖；
3. collector 支持 Qwen 完整 E1-E4 cell 和 Llama E1-only cell；
4. 三个 seeds 先在同一样本内求平均，再做样本级 mean/std、bootstrap CI 和 Wilcoxon；
5. 输出 E1-E4 专用表图数据，不要求人工从通用 CSV 二次筛选。

## 6. 论文结果产物

建议使用以下 canonical outputs：

```text
results/paper/paper-v2.3/aggregate/
├── classifier-diagnostics.csv
├── table-e1-faithfulness.csv
├── figure-e1-paired-effects.csv
├── figure-e2a-budget-recovery.csv
├── figure-e2b-fixed-support.csv
├── table-e3-exact-pairs.csv
├── table-e4-projection.csv
├── attribution-costs.csv
├── audit-costs.csv
└── manifest.json
```

cell 路径使用：

```text
cells/<dataset-split>/<model-id>/budget-<B>/seed-<seed>/
```

## 7. 验收与运行顺序

### 阶段 1：无 LLM 单元测试

- AOPC-Sufficiency 当前公式与历史结果兼容；
- NRMSE-range 的逐样本和 degenerate 处理正确；
- ProxySPEX 四点差分不读取 FBII coefficient；
- serialized/native refined-Fourier prediction 一致；
- @10 metrics 在短 pair 列表上使用动态 K；
- budget 路径互不覆盖；
- 跨 seed 聚合顺序符合论文定义。

### 阶段 2：真实模型 pilot

在 SST-2/Qwen3-8B 上使用少量样本和 budget 64，跑通：

```text
classifier diagnostics
-> C / A / ProxySPEX / Occlusion / LIME
-> shared held-out
-> representation audit
-> exact pair audit
-> singleton / signed / sign-agnostic projections
-> paper collector
```

pilot 只检查协议与 artifacts，不据此选择方法或超参数。

### 阶段 3：正式实验

1. Qwen3-8B 三数据集 E1 主预算；
2. Qwen3-8B 三数据集 E2 多预算；
3. Qwen3-8B 三数据集 E3 exact pair；
4. Qwen3-8B 三数据集 E4 projection；
5. Llama-3.1-8B-Instruct 三数据集 E1；
6. 汇总三个 seeds，生成 canonical paper tables；
7. 最后再运行 sampler、hierarchy、refit 和长度分析等附录实验。

## 8. 文档同步要求

代码完成后需要同步修改：

- `docs/paper-outline-v2.3.md`：AUSufficiency 改为 AOPC-Sufficiency，NRMSE-std 改为 NRMSE-range，E4 加入 Sign-agnostic；
- `docs/paper_experiment_usage.md`：更新 AG News、Llama、多预算、E3 和 collector 命令；
- `README.md`：给出最短可复现流程；
- baseline README：区分原生 Inseq 和共享 value-function first-order baselines；
- 结果 schema 文档：解释 attribution cost、evaluation cost 和 audit cost 的边界。

## 9. 实施状态

本计划已实现。关键入口如下：

- 正式配置：`configs/paper-v2.3/`；
- 主方法：`python -m mobius.cli.run`；
- Occlusion/LIME：`scripts/run_first_order_llm.py`；
- ProxySPEX：`baselines/shapiq-copy/run_proxyspex_llm_baseline.py`；
- E2：`scripts/build_surrogate_holdout.py`、`scripts/evaluate_surrogates.py`、`scripts/audit_representation.py`；
- E3：`scripts/audit_exact_pairs.py`；
- E4：`scripts/derive_projection_run.py`；
- 论文汇总：`scripts/collect_paper_results.py`；
- 中文运行说明：`docs/paper_experiment_usage.md`。

paper collector 会先写逐样本 cell artifacts，再在 summarize 阶段对 E1、E2、fixed-k basis 与 E3 执行“同一样本跨 seed 平均，随后样本级统计”的统一协议。cell 路径包含 budget，audit contract 同时校验 dataset、model、prompt、chunk、value function 与 target semantics。
