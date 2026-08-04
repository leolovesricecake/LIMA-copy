# 论文实验执行计划 v2

## 1. 变更目的

v2 在 v1 的四个位置补强证据：

1. 增加 exact structural audit，区分真实结构与恢复算法；
2. 增加 fixed-k/matched-error 协议，严格比较 Möbius 与 Fourier；
3. 将极短样本 full-table oracle 与普通长度 exact-pair oracle 分开；
4. 将 signed equal share 明确为 negated deletion-Shapley allocation。

主归因方法、默认 sampler、value function、word 粒度和预算保持不变。

## 2. 固定主协议

```text
value_function = predicted_probability
target_mode = predicted
chunker = word
eval_granularity = word
sampler = uniform_size
max_degree = 2
projector = signed_equal_share
eval_q_values = [5, 10, 20, 50]
```

`target_probability + target_mode=predicted` 与
`predicted_probability` 视为等价。

## 3. 实验与实现

### E1 Attribution

输入：

- A/C/ProxySPEX；
- 可选 first-order baseline。

产出：

- `overall-attribution.csv`；
- `paired-effects.csv`；
- `costs.csv`。

### E2a Exact Structure

由 `scripts/audit_representation.py` 自动处理
`n<=max_exact_features` 且 observations 完整的样本。

计算：

- exact all-order deletion-Möbius transform；
- exact all-order Fourier transform；
- degree truncation；
- exact top-k OLS refit；
- basis-internal coefficient profile。

不新增模型查询。

### E2b Finite-Query Recovery

同一脚本使用 C observations：

- 复现实际 Möbius estimator；
- 用相同 estimator 离线拟合 Fourier；
- 用同一 standardized OMP + OLS 做 fixed-k 比较；
- 从 fixed-k curve 计算 matched-error support；
- 在最多 128 个确定性抽样候选列上报告 design geometry。

抽样列数必须随结果记录，不能将近似 coherence 描述为完整设计矩阵值。

### E3a Full-Table Exact Interactions

`audit_representation.py` 在完整 value table 上比较：

- C；
- ProxySPEX；
- random；
- oracle。

论文表中 `oracle_scope=full_table`。

### E3b Ordinary-Length Exact Pair Oracle

由 `scripts/audit_exact_pairs.py` 执行。

默认：

```text
min_features = 10
max_features = 32
max_samples = 50
```

每样本查询 \(1+n+\binom n2\) 个唯一 masks。样本按长度 strata
round-robin 选择，stratum 内使用固定 seed 哈希排序。

论文表中 `oracle_scope=pair_oracle`。

### E4 Projection Ablations

- A：重新拟合 degree 1；
- B：复用 C predictor，singleton only；
- C：negated deletion-Shapley allocation；
- ABSOLUTE：复用 C predictor，absolute equal share；
- STRICT：只作 hierarchy 诊断。

### E5 Robustness

执行顺序：

1. 三数据集主 seed；
2. 固定子集 seeds 42/43/44；
3. budget 与 length；
4. 第二模型；
5. 其余 sensitivity 放附录。

## 4. Paper Result Protocol

每个 dataset/model/seed cell 输出：

```text
overall-attribution.csv
paired-effects.csv
costs.csv
audit-costs.csv
surrogate-heldout.csv
representation-recovery.csv
exact-structure.csv
exact-interactions.csv
projection-ablation.csv
figure-data/attribution-paired.csv
```

`exact-interactions.csv` 必须通过 `oracle_scope` 区分两种 oracle。

## 5. 执行顺序

1. 完成 A/C/ProxySPEX 原始 runs；
2. 构建 shared held-out；
3. 运行 `evaluate_surrogates.py`；
4. 运行 `audit_representation.py`，得到 E2a/E2b/E3a；
5. 运行 `audit_exact_pairs.py`，得到 E3b；
6. 派生 B/ABSOLUTE 并执行 ranking evaluation；
7. 生成每个 paper cell；
8. 汇总 paper version；
9. 根据主结果决定 seeds、budget 和第二模型范围。

## 6. 成本

零查询：

- exact structural transform；
- Fourier refit；
- fixed-k OMP；
- matched-error support；
- design diagnostics；
- full-table exact interactions；
- paper statistics。

新增查询：

- shared held-out；
- ordinary-length exact pair oracle；
- B/ABSOLUTE faithfulness evaluation；
- seed/budget/second-model robustness。

默认 pair oracle 最坏每数据集：

\[
50\times\left(1+32+\binom{32}{2}\right)=26{,}450
\]

个逻辑 values，物理调用可因全局 cache 更少。

## 7. 解释规则

- deletion interaction 不写成 causal interaction；
- “真实交互”统一改成 exact deletion coefficient；
- raw Möbius/Fourier coefficient energy 不跨 basis 比较；
- fixed-k 结果用于坐标公平性，不替代主 estimator 结果；
- Shapley allocation 是规范选择，不写成经验最优 projection；
- exact full-table 的结论明确限制于短文本。

详细命令见 `docs/paper_experiment_usage.md`，论文叙事见
`docs/paper-outline-v2.md`。
