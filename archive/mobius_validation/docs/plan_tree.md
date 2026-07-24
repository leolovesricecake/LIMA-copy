# Sparse Möbius 与 ProxySPEX 独立比较计划树

## 0. 研究目标

验证低阶稀疏 deletion-Möbius 超图是否能作为一种 LLM 交互归因方法，并与 ProxySPEX 的 Fourier hierarchy 路线进行端到端比较。

比较对象是完整方法，而不是在同一观测表上替换回归基底：

```text
方法 = players + value function + mask sampler + estimator + interaction recovery + ranking
```

## 1. 数学对象

```text
f(S) = model_value(x_S)
g(D) = f(N \ D)
```

对 `g` 做 Möbius transform：

```text
g(D) = sum_{T subseteq D} theta[T]
```

主模型限制 `|T| <= d`，默认 `d=2`，使用稀疏选择恢复重要超边。

Deletion-Möbius 与 OR interaction 只差符号约定，不作为新指标。

## 2. 方法边界

### 2.1 ProxySPEX

由 `baselines/shapiq-copy/run_proxyspex_llm_baseline.py` 独立负责：

- ProxySPEX 原始 coalition sampler；
- tree/LightGBM proxy；
- Fourier coefficient extraction；
- Ridge refinement；
- Möbius 和 interaction index conversion；
- interaction 到节点 ranking 的 signed equal share。

`mobius_verify` 不包装、不重采样、不注入固定 observations。

### 2.2 Sparse deletion-Möbius

由 `mobius_verify/scripts/run_sparse_mobius_llm.py` 独立负责：

- deletion-aware mixture sampler；
- ValueOracle；
- degree-1/2 Möbius design；
- LASSO/ElasticNet support selection；
- selected support 上 Ridge refit；
- deletion coefficient 到节点分数的有符号均分；
- 可选 targeted exact coefficient verification。

## 3. 采样计划

每个 Sparse 样本在严格预算 `B` 内生成唯一 masks：

```text
anchors + global Bernoulli + near-full deletion + fixed-cardinality
```

默认比例：

```text
global       0.5
near-full    0.3
fixed-card   0.2
```

该设计同时支持：

- 全局 surrogate coverage；
- 完整输入附近的 deletion coefficient 识别；
- 多种集合基数覆盖；
- 不依赖“高阶项的重要子集也重要”的 hierarchy 假设。

采样比例属于方法超参数，后续可以做独立 sampler ablation，但不能把 ProxySPEX 强制改成同一 masks。

## 4. Value Function 计划

公共 raw score：verbalizer token 的平均 conditional log probability `z_c(S)`。

支持：

- `target_probability`；
- `predicted_probability`；
- `predicted_class_margin`；
- Sparse sensitivity 使用 `raw_target_score`。

ProxySPEX 默认 `predicted_probability`。旧实现复现使用 `target_probability + target_mode=gold`。

主 margin 比较要求两边都使用 `predicted_class_margin`；概率比较要求两边都使用 `predicted_probability`。比较脚本禁止混合 value function。

## 5. 公平契约

两边必须一致：

1. dataset/split/sample ID；
2. model path、max length 和 prompt；
3. verbalizers；
4. explanation chunks；
5. delete operator 和空文本；
6. target class 来源；
7. value function；
8. eval granularity 和 q values。

两边允许不同：

1. attribution masks；
2. masks 的查询顺序；
3. estimator；
4. interaction recovery；
5. CPU/GPU 实现细节。

## 6. 执行树

### A. Sparse 方法运行

每个叶节点是一条独立命令：

```text
dataset × budget × seed × value_function
```

叶节点内部只遍历 samples。每个样本结束后立即写入：

- explanation；
- observation table；
- query ledger；
- manifest 进度。

### B. ProxySPEX 运行

使用原始 copy runner，对相同实验契约分别运行 budget/seed/value。

### C. 公共评估

两个 runner 分别调用 `evaluate_saved_explanations()`，生成：

- aggregate `eval_report.json`；
- per-sample `eval_sample_metrics.jsonl`；
- deletion/retention curves。

### D. 离线比较

`compare_method_runs.py`：

1. 检查 comparison contract；
2. 核对共同样本的原文和 chunks；
3. 取 sample ID 交集；
4. 计算 faithfulness 配对差值；
5. bootstrap 95% CI；
6. 比较逻辑查询、物理评分、forward 和耗时；
7. 报告覆盖率及不兼容样本。

比较阶段不加载模型、不调用 LLM。

## 7. 指标树

### 公共主要指标

- AOPC；
- comprehensiveness；
- sufficiency gap；
- AOPC-comprehensiveness；
- AOPC-sufficiency；
- deletion/retention curve。

### 成本指标

- logical attribution queries；
- logical unique masks；
- unique masked texts；
- physical values scored；
- model forward calls；
- batch calls/rows；
- explanation elapsed time。

Evaluator 调用单独统计，不计入 attribution budget。

### Sparse 方法诊断

- train R2/NRMSE/MAE；
- support size；
- design coherence/effective rank；
- hyperedge coefficients；
- targeted exact deletion coefficient error；
- 多 seed support stability。

这些诊断不能替代公共 faithfulness 指标。

## 8. 实验矩阵

第一阶段：工程和语义验证。

```text
dataset = SST-2
samples = 5
budget = 32/64
seed = 0
value = predicted_probability, predicted_class_margin
```

第二阶段：主比较。

```text
dataset = SST-2, Rotten Tomatoes
budget = 128, 256, 512
seed = 0, 1, 2
chunker = word
eval = token
```

第三阶段：敏感性与扩展。

```text
chunker = adaptive
degree = 1/2/3
sampler mixture ablation
raw target score sensitivity
```

## 9. 判定原则

支持 Sparse Möbius 路线需要同时看到：

1. faithfulness 不劣于或优于 ProxySPEX；
2. 相同或更低 logical query budget；
3. 多 seed 下 ranking/超边具有可接受稳定性；
4. targeted deletion coefficients 具有方向或数值校准；
5. 结果不只出现在单一数据集或单一 value function。

若 faithfulness 接近但超边不稳定，只能说明它是有效 ranking surrogate，不能宣称恢复了可靠交互机制。

## 10. 已移除分支

- controlled protocol；
- native protocol；
- ProxySPEX fixed observations；
- `mobius_verify` 内的 ProxySPEX adapter；
- shared attribution masks/values；
- methods × budgets × seeds × values 的单进程笛卡尔积。

相同 observation table 下的 basis comparison 如有需要，只能作为独立离线 ablation，不进入主方法比较。
