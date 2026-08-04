# 论文实验执行计划 v1

## 1. 目标

本计划不再设置独立的 development gate。每次计算都必须直接对应论文中的表、图、正文结论或附录诊断。

论文回答五个问题：

1. Sparse deletion-Möbius 的词级归因是否优于 additive 和 ProxySPEX；
2. 在相同 observations 下，Möbius 与 Fourier 坐标的有限查询恢复有何差异；
3. 恢复出的二阶边是否对应真实、可排序的 deletion interactions；
4. interaction modeling 与 signed projection 分别贡献多少；
5. 结果是否对 seed、任务、模型、文本长度和 held-out 分布稳定。

不预设 Möbius、非层级选择或某个 sampler 必然更优，最终表述由实验决定。

## 2. 固定科学协议

### 2.1 Value function

主实验使用完整输入预测类别的概率：

```text
value_function = predicted_probability
target_mode = predicted
```

`target_probability + target_mode=predicted` 与之视为等价配置。

### 2.2 文本粒度

```text
chunker = word
eval_granularity = word
```

解释 players 和 faithfulness 扰动均采用 word，避免 token/word 投影引入额外差异。

### 2.3 Deletion-Möbius 与归因方向

\[
g(D)=f(N\setminus D).
\]

节点归因为：

\[
a_i=-\sum_{T\ni i}\frac{\theta_T}{|T|}.
\]

它满足 surrogate completeness：

\[
\sum_i a_i=\widehat g(\varnothing)-\widehat g(N).
\]

论文必须区分 deletion coefficient 的符号与特征对完整输入预测的贡献方向。

### 2.4 Fourier 与 Möbius

固定最大阶数时，低阶 Fourier 与低阶 Möbius 张成相同函数空间。因此只能比较：

- 同 observations 下的有限查询恢复；
- 坐标 support size 与 top-k compression；
- held-out reconstruction；
- exact interaction ranking；
- 最终 feature projection。

不得声称 Möbius 具有 Fourier 不具备的低阶表达能力。

### 2.5 Faithfulness

主指标：

- AOPC，越高越好；
- AUPC，越低越好。

次指标：

- comprehensiveness，越高越好；
- sufficiency，越低越好；
- AOPC-comprehensiveness，越高越好；
- AOPC-sufficiency，越低越好。

默认 q-grid：

```yaml
eval_q_values: [5, 10, 20, 50]
```

主 AOPC/AUPC 来自完整删除轨迹。对旧 curves，只离线重算两个依赖 q-grid 的 AOPC 指标。

## 3. 结果生命周期

### 3.1 原始 runs

原始 attribution 结果不可变：

```text
results/<method-root>/<dataset>/<model>/<method>/<config>/
```

必须保留：

```text
run.json
status.json
metrics.json
curves-predicted.jsonl
samples/
observations/
surrogates/
```

### 3.2 Canonical audits

可复用的科学审计保存为：

```text
results/audits/<dataset>/<audit-kind>/<audit-id>/
```

主要 audit kinds：

```text
surrogate-heldout
representation
interactions
hierarchy
```

audit ID 由输入 runs 与科学配置生成，manifest 记录 dataset、model、seed、run paths 和查询成本。

### 3.3 Paper outputs

论文结果直接输出到：

```text
results/paper/<paper-version>/
├── cells/
│   └── <dataset>-<split>/
│       └── <model>-<hash8>/
│           └── seed-<seed>/
└── aggregate/
```

不同 dataset、model 和 seed 不会覆盖。这里不再生成自动论文路线推荐，只输出统计证据。

## 4. 论文实验

### E1 Overall Attribution Performance

比较：

- C：degree-2 sparse deletion-Möbius + signed projection；
- A：degree-1 additive deletion model；
- ProxySPEX：原生 sampler、tree proxy 与 Fourier refinement；
- 至少一个兼容 word deletion protocol 的常规一阶方法。

输出：

- 所有 faithfulness mean/std；
- C 对每个方法的逐样本 improvement；
- bootstrap 95% CI；
- Wilcoxon p-value；
- paired effect size；
- attribution model forward calls、逻辑查询数和每样本成本。

### E2 Representation and Finite-Query Recovery

使用 C 保存的同一批 training masks 和 values，离线拟合：

- degree-2 deletion-Möbius；
- degree-2 Fourier；
- 相同候选阶数、selector、CV、refit 和 hierarchy policy。

输出：

- train 与 shared held-out R²、NRMSE、MAE；
- nonzero support size；
- top-k OLS refit compression curve；
- Möbius-Fourier 的逐样本配对统计。

该实验不增加 LLM 查询。

### E3 Exact Interaction Quality

对 `realized_budget=2^n` 且 `n<=9` 的样本使用完整 value table，计算全部真实二阶 deletion-Möbius coefficients。

比较：

- Sparse Möbius ranking；
- ProxySPEX final interaction ranking；
- deterministic random ranking；
- oracle ranking。

输出：

- Precision@1/3/5；
- NDCG@1/3/5；
- Spearman；
- Möbius 全 pair 与 selected pair 的 coefficient MAE；
- sign agreement；
- exact sample count、pair count 和长度分布。

该实验与 E2 由 `scripts/audit_representation.py` 一次完成，不增加 LLM 查询。

### E4 Interaction and Projection Ablations

比较：

- A：degree 1 + singleton；
- B：复用 C surrogate，只使用 singleton；
- C：degree 2 + signed equal share；
- ABSOLUTE：复用 C surrogate，使用 absolute equal share。

解释：

- C vs A：interaction model 的总体收益；
- C vs B：已拟合 interactions 是否需要进入最终排名；
- C vs ABSOLUTE：符号方向是否有价值；
- degree-2 held-out vs degree-1 held-out：交互是否改善函数重建。

B 和 ABSOLUTE 的拟合不调用 LLM，但 ranking faithfulness evaluation 仍计入 evaluation cost。

### E5 Robustness and Scope

按成本从低到高执行：

1. 现有 SST-2、rotten_tomatoes、emotion 的任务差异；
2. 固定 200 样本的 seeds 42/43/44；
3. 文本长度分组；
4. Bernoulli 与 near-full held-out；
5. 第二个 8B 模型上的 SST-2；
6. 必要时再测试替代删除算子。

输出跨 seed 均值/方差、ranking Spearman、top-edge Jaccard 与 sign agreement。

### Appendix: Hierarchy Diagnostics

比较 none、parent screening 和 strict：

- support 被删除数量；
- orphan edge 比例；
- held-out 与 faithfulness 变化；
- 被删除边的真实 interaction；
- 个案分析。

Hierarchy 是结构诊断，不承担论文主结论。

## 5. 论文表格与图

主表：

1. Overall attribution faithfulness and cost；
2. Same-observation representation and recovery；
3. Exact interaction quality；
4. Interaction/projection ablation。

主图：

1. 方法流程图；
2. Möbius/Fourier top-k compression curve；
3. C vs A/ProxySPEX 的 paired improvement 分布；
4. exact interaction ranking 或 coefficient scatter。

附录：

- 全部 per-q 指标；
- hierarchy；
- sampler/refit/budget；
- 长度分组；
- qualitative examples；
- 查询与运行时间明细。

## 6. 执行顺序

1. 确认 A/B/C/ABSOLUTE/ProxySPEX runs 的 sidecars 完整；
2. 构建同时排除 C 与 ProxySPEX training masks 的 shared held-out；
3. 运行 `audit_representation.py`；
4. 对每个 dataset/model/seed 运行 `collect_paper_results.py`；
5. 运行 `collect_paper_results.py --summarize`；
6. 先根据现有任务结果形成论文主表；
7. 再补最小 seed stability；
8. 最后决定是否增加第二模型和替代删除算子。

## 7. 成本控制

零查询步骤优先：

- same-observation Fourier refit；
- exact small-n；
- paired attribution statistics；
- B/ABSOLUTE surrogate 派生；
- paper table collection。

新增查询仅用于：

- 正确 shared held-out 中尚未缓存的 masks；
- B/ABSOLUTE ranking evaluation；
- seed stability；
- 第二模型；
- near-full 或替代删除算子。

任何新增实验必须对应第 5 节中的一个表、图或附录。

## 8. 实现映射

```text
scripts/audit_representation.py
    -> E2 + E3 canonical audit

scripts/build_surrogate_holdout.py
scripts/evaluate_surrogates.py
    -> shared held-out

scripts/derive_projection_run.py
    -> B / ABSOLUTE

scripts/verify_interactions.py
scripts/analyze_hierarchy.py
    -> supplementary interaction/hierarchy diagnostics

scripts/collect_paper_results.py
    -> per-cell paper tables + cross-cell aggregate
```

详细命令见 `docs/paper_experiment_usage.md`。
