# Sparse Deletion-Möbius LLM Attribution：论文大纲 v2

## 核心问题

本文不把“使用 Möbius 坐标”本身视为贡献，也不预设 Fourier hierarchy
不适合语言模型。研究问题是：

> 能否从有限黑盒删除查询中直接、稳定地恢复 deletion interactions，
> 并通过有原则的分配得到有用的词级归因？

证据链被拆为四个不可相互替代的命题：

1. **结构前提**：真实 masked function 是否具有可利用的低阶、稀疏结构；
2. **有限查询恢复**：给定有限 observations，算法能否恢复该结构；
3. **交互恢复准确性**：估计边是否接近精确 deletion-Möbius coefficients；
4. **归因效用**：将边分配给词后，ranking 是否具有更好 faithfulness。

论文只根据四类证据共同支持的范围陈述结论。

---

## 1. Introduction

### 1.1 问题背景

词级归因通常为每个词分配一个独立分数，但否定、修饰、协同和冗余等
模型行为天然涉及词组合。仅拟合 singleton effects 会把非加性行为错误地
压回单词。

### 1.2 技术问题

SPEX 和 ProxySPEX 表明稀疏 Fourier interactions 可以用于归因，但留下
三个问题【TODO：这三个问题不太自然，尤其是第一个，可能需要重新考虑】：

1. 对删除扰动而言，直接恢复 deletion-Möbius coefficients 是否更自然；
2. 低阶、稀疏结构是真实函数性质，还是有限样本拟合产生的假象；
3. 恢复出的 interaction 是否真的改善最终词级 ranking。

### 1.3 方法概述

```text
word players
    → method-owned deletion masks
    → fixed-target model values
    → sparse low-degree deletion polynomial
    → signed deletion hypergraph
    → negated deletion-Shapley allocation
    → word ranking
```

### 1.4 贡献边界
【TODO：过于保守了，而且像是技术技术报告，而非研究论文；没有人会说自己的贡献是“分开评估”、“逐样本比较”，这是操作，而不是贡献，就算很有意义也得换个表述】

1. 提供直接恢复 sparse deletion hypergraph 的黑盒归因流程；
2. 将结构前提、有限查询恢复、精确交互恢复和归因效用分开评估；
3. 给出 signed projection 的 Shapley/Harsanyi 解释，而不把等分描述为经验最优规则；
4. 在匹配 value function、文本粒度和预算的条件下，与原生 ProxySPEX 及受控消融进行逐样本比较。

---

## 2. Related Work

### 2.1 LLM Feature Attribution

- gradient-based attribution；
- perturbation/deletion attribution；
- LIME、SHAP、Integrated Gradients；
- token、word 与 phrase players。

### 2.2 Interaction Attribution

- Harsanyi dividends 与 Möbius interactions；
- Shapley interaction indices；
- SPEX 与 ProxySPEX；
- interaction-to-feature allocation。

### 2.3 Sparse Boolean Function Recovery

- low-degree set functions；
- Fourier 与 monomial coordinates；
- sparse support selection；
- query design 与 finite-sample recovery。

本节明确：degree-\(d\) Fourier 与 degree-\(d\) monomial bases 张成相同
函数空间。坐标差异只可能体现在有限查询恢复、稀疏性、数值几何和删除
语义，而不是理论表达能力。

---

## 3. Problem Formulation

### 3.1 Masked Prediction Function

【TODO：怎么一开始就是 players？哪来的？输入不是自然语言吗？】给定词级 players：

\[
N=\{1,\ldots,n\}.
\]

令 \(D\subseteq N\) 为删除集合：

\[
g(D)=f(N\setminus D).
\]

同时明确相关定义：

- 完整输入的 predicted class；
- predicted-class probability；
- word-level deletion；
- 所有 coalition 均保持同一个 target class。

### 3.2 Deletion-Möbius Coefficients

\[
m^-(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}g(U).
\]

二阶系数为：

\[
m^-(\{i,j\})
=
g(\{i,j\})-g(\{i\})-g(\{j\})+g(\varnothing).
\]

【TODO：为什么又固定为二阶了？这相当于在表示我们的方法就是二阶方法，不能用于高阶！】

它衡量相对于给定删除算子的精确非加性，而不自动意味着人类语义合理、因果真实或跨删除算子稳定。【TODO：这个论述不对，我们只需要表述它衡量了什么，不必特地强调不衡量什么】

### 3.3 三种误差

【TODO：不应该讲三个误差，而应该是三个目标，比如：有限查询恢复、精确交互恢复和归因效用】

论文始终区分：

1. masked-function reconstruction error；
2. deletion-interaction coefficient/ranking error；
3. projected feature-ranking faithfulness。

任意一项更优都不能逻辑推出另外两项更优。

---

## 4. Method

【TODO：开头必须有流程图】

### 4.1 Query Sampling

预算为 \(B\)，训练 observations 为：

\[
\mathcal D_B=\{(D_b,g(D_b))\}_{b=1}^{B}.
\]

主方法使用 `uniform_size`，先均匀抽 deletion/keep cardinality，再在该
cardinality 内均匀抽 coalition，同时包含 empty/full anchors。

### 4.2 Sparse Low-Degree Recovery

\[
\widehat g(D)
=
\theta_\varnothing+
\sum_{\substack{T\subseteq N\\1\le |T|\le d}}
\theta_T\mathbf 1[T\subseteq D].
\]

主实验使用 \(d=2\)：

1. 标准化候选设计列；
2. Lasso/ElasticNet 选择 support；
3. 对保留 support 做稳定 refit；
4. 保存 masks、values、support、coefficients 和 fitting diagnostics。【TODO：保存内容完全是工程实现，不必写在论文里】

【TODO：怎么这一节的具体内容都被省略了？v0 中还会介绍方法呢，是移动到其他地方了吗？】

### 4.3 Signed Hypergraph

每个非零 \(\theta_T\) 是一条带符号超边。符号描述删除方向中的 interaction，
不能直接作为完整输入中特征贡献方向。

【TODO：怎么这一节的具体内容都被省略了？v0 中还会介绍方法呢，是移动到其他地方了吗？】

### 4.4 Negated Deletion-Shapley Allocation

【TODO：这一节为什么存在？为什么放在这个位置？和上下文有什么关系？】

删除 game 的 Shapley allocation 为：

\[
\phi_i(g)=\sum_{T\ni i}\frac{\theta_T}{|T|}.
\]

完整输入中的 feature-presence attribution 取相反方向：

\[
a_i=-\phi_i(g)
=-\sum_{T\ni i}\frac{\theta_T}{|T|}.
\]

它满足：

- efficiency；
- symmetry；
- linearity；
- dummy-player property。

并具有 surrogate completeness：

\[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
\]

该推导说明 equal share 是 Harsanyi dividends 的规范分配，而不是声称它
在所有 faithfulness 指标上经验最优。

### 4.5 Complexity

degree-2 candidate count 为：

\[
n+\binom n2.
\]

报告：

- candidate/support size；
- logical unique queries；
- physical values scored；
- model forward calls；
- fitting 与总运行时间。

【TODO：这部分应该放附录吧？】

---

## 5. Experimental Protocol

### 5.1 数据与模型

主数据集：

- SST-2；
- Rotten Tomatoes；
- Emotion。

模型： Qwen3-8B、LLaMa-3.1-8B instruct 只用于范围验证。

### 5.2 对比方法

- C：degree-2 sparse deletion-Möbius + signed allocation；
- A：degree-1 additive deletion model；
- ProxySPEX：保留原生 sampler、tree proxy、Fourier refinement；
- 其他 first-order baseline；
- B、ABSOLUTE、STRICT 作为受控消融。

### 5.3 公平性协议

- 相同 dataset、model、word chunks、target 和 value function；
- C 与 ProxySPEX 各自保留 method-owned training masks；
- basis comparison 复用 C 的同一 observations；
- shared held-out 排除所有纳入方法的 training masks；
- 每个 dataset/model/attribution seed 独立形成 paper cell；
- 逐样本 paired difference、bootstrap CI、Wilcoxon 和 effect size。

### 5.4 指标

Attribution：

- AOPC，越高越好；
- AUPC，越低越好；
- aopc-comprehensiveness，越高越好；
- aopc-sufficiency，越低越好。

Reconstruction：

- R²；
- range-normalized RMSE；
- MAE。

Interaction：

- Precision@1/3/5；
- NDCG@1/3/5；
- Spearman；
- coefficient MAE；
- sign agreement。

---

## 6. Experiments

### E1 Overall Attribution Faithfulness and Cost

**为什么设计**

回答词级 ranking 是否忠实。

**比较**

- C vs A：完整二阶方法相对加性模型；
- C vs ProxySPEX：相同预算下与原生 interaction baseline 比较；
- C vs protocol-compatible first-order baseline：确认收益不是删除评估协议
  本身带来的。

**控制变量**

- word chunker；
- predicted-class probability；
- predicted target；
- attribution budget；
- evaluation q-grid；
- 相同样本交集。

**要验证什么**

1. C 是否在 AOPC/AUPC 上具有稳定逐样本改善；
2. 优势是否同时出现在多个任务，而非 aggregate mean 偶然提高；
3. 改善是否以更多逻辑查询、forward calls 为代价。

**输出**

- Overall faithfulness and cost 主表；
- C 对各 baseline 的 paired-improvement 分布图；【TODO：这是什么？】

【TODO：是否需要实验 3 个 seed，报告 mean + CI 或 mean + std？】

---

### E2a Exact Structural Audit

**为什么设计**

有限 observations 下拟合失败可能来自真实函数不稀疏、query design
不足或 solver 失败。必须先用完整 value table 隔离“真实结构是否存在”。【TODO：我看不懂。到底在证明方法中的哪一部分？】

**样本与成本**

- 使用 C run 中 `n<=9` 且 `realized_budget=2^n` 的样本；
- 复用已保存 values；
- 不新增 LLM 查询。

**做法**

1. 对完整删除函数计算全阶精确 deletion-Möbius transform；
2. 对同一函数计算全阶精确 Fourier transform；
3. 比较 degree-\(d\) truncation reconstruction；
4. 在各基内部按标准化贡献排序，做 exact top-k OLS refit；
5. 报告按 degree 的 nonzero count 和 L1 mass，但不跨非正交基比较“系数能量”。

**要验证什么**

1. degree 1/2 是否已经解释完整函数的大部分变化；
2. Möbius 是否在小 k 下比 Fourier 更易压缩，或二者实际上相当；
3. 高阶残差是否足以否定“低阶结构”前提。

**输出**

- degree-truncation curves；
- exact top-k reconstruction curves；
- coefficient count/mass profile；
- exact sample count 和长度范围。

---

### E2b Finite-Query Basis Recovery

**为什么设计**

即使完整函数在某个坐标中可压缩，有限查询和正则化也可能无法恢复。
同时，分别调参后的 Lasso support size 不能单独证明某个坐标更稀疏。【TODO：我看不懂，这又是在证明什么？】

**做法**

使用完全相同的 C training masks 和 values，比较：

1. **实际 estimator**：两种 basis 各自使用相同 alpha grid、CV、standardization、hierarchy policy 和 refit；
2. **fixed-k diagnostic**：两种 basis 使用同一 standardized OMP 路径和 OLS refit，在相同 k 下比较 held-out error；
3. **matched-error diagnostic**：达到相同 R²/NRMSE threshold 时比较所需 k；
4. **design geometry**：报告抽样候选列数、coherence、effective rank 和 非零奇异值 condition number。

**要验证什么**

1. 在相同 observations 下，哪种坐标具有更好的有限查询恢复；
2. 差异是否仍存在于 fixed-k 和 matched-error 协议；
3. 差异是否可由设计矩阵尺度、相关性或 rank deficiency 解释；
4. 实际方法收益是否只是独立 CV 选择出不同 support size。

**输出**

- train/Bernoulli held-out R²、NRMSE、MAE；
- fixed-k OMP curves；
- support-at-matched-error；
- design geometry diagnostics；
- Möbius-Fourier paired statistics。

---

### E3a Full-Table Exact Interaction Fidelity

**为什么设计**

E2a 的完整 value table 可同时提供全部精确二阶 coefficients，用来验证
估计边是否符合 deletion-Möbius 定义。

**样本与成本**

- 与 E2a 相同的 `n<=9` 完整枚举样本；
- 零新增查询。

**比较**

- C recovered pair ranking；
- ProxySPEX final interaction ranking；
- deterministic random；
- exact oracle。

**要验证什么**

1. C 是否恢复正确的 top pair；
2. C 的 coefficient magnitude 与 sign 是否接近 exact values；
3. C 相对 ProxySPEX 的优势是否存在于同一 exact target 下。

**输出**

- Precision/NDCG/Spearman；
- all-pair 与 selected-pair coefficient MAE；
- sign agreement；
- scope 标记为 `full_table`。

---

### E3b Ordinary-Length Exact Pair Oracle

**为什么设计**

验证二阶 deletion coefficient 不需要完整 \(2^n\) 枚举。若仍只使用极短文本，审稿人无法判断方法能否恢复普通文本中的 pair interactions。

**样本与成本**

- 每数据集固定、长度分层抽取最多 50 个样本；
- 默认 \(10\le n\le32\)；
- 固定 sample IDs 和 analysis seed。

每个样本只查询：

\[
1+n+\binom n2
\]

个唯一 masks，即 full、所有 singleton deletion 和所有 pair deletion。
相同 mask 只查询一次，并通过全局 ValueOracle cache 复用物理结果。

**比较**

- C；
- ProxySPEX；
- deterministic random；
- exact pair oracle。

**要验证什么**

1. E3a 的 interaction ranking 结论是否延伸到普通长度；
2. 恢复误差是否随 n 增长；
3. C 与 ProxySPEX 的差异是否来自只在极短样本上完整枚举。

**输出**

- 与 E3a 相同的 ranking/coefficient metrics；
- length-stratified results；
- requested queries、cache hits 和 physical values；
- scope 标记为 `pair_oracle`。

【TODO：我看不懂，这又是在证明什么？为什么需要这个实验？】

---

### E4 Interaction Modeling and Projection Ablation

**为什么设计**

C 相对 A 的提升同时改变了函数阶数和最终 ranking。必须区分“拟合了 interactions”与“把 interactions 分配进节点分数”。

**比较**

- A：degree 1 + singleton projection；
- B：复用 C surrogate，只读取 singleton terms；
- C：degree 2 + signed Shapley allocation；
- ABSOLUTE：复用 C surrogate，忽略 coefficient sign；
- STRICT：与 C 相同但施加 strict hierarchy，仅作结构诊断。

**要验证什么**

1. C vs A：二阶建模的总体收益；
2. C vs B：已拟合 interactions 是否必须进入最终 ranking；
3. C vs ABSOLUTE：删除方向的 sign 是否有价值；
4. degree 2 vs degree 1 held-out：二阶项是否改善函数重建。

**控制**

B 与 ABSOLUTE 复用 C 的 predictor，因此 training masks、values、support
和 coefficients 完全一致，只改变 projection。

**输出**

- attribution paired effects；
- held-out reconstruction；
- projection ablation table；
- completeness residual。

---

### E5 Robustness and Scope

**为什么设计**

稀疏恢复可能对预算、文本长度高度敏感。没有范围实验，
主结果可能只是一个 dataset/model/seed 的偶然现象。

【TODO：这个设计正确吗？】

**正文优先级**

1. budget sensitivity；
2. text-length bins；

**附录优先级**

- Bernoulli vs near-full held-out；
- sampler；
- refit；
- hierarchy；

**要验证什么**

1. paired improvement 的方向是否跨 seed 稳定；
2. budget 增加时恢复与归因是否按预期改善；
3. 长文本的 candidate growth 是否导致明显退化；

---

## 7. Qualitative Analysis

只从预先定义的类别中选择案例：

- 否定；
- 程度修饰；
- 协同；
- 冗余；
- orphan interaction；
- 失败案例。

每个案例展示：

- singleton coefficients；
- signed pair coefficients；
- exact pair coefficients；
- Shapley-allocated word scores；
- deletion curve。

案例用于解释机制，不作为语义正确性的统计证明。

---

## 8. Main Tables and Figures

主表：

1. attribution faithfulness and cost；
2. exact structure and finite-query recovery；
3. exact interaction fidelity，分 full-table/pair-oracle；
4. interaction/projection ablation。

主图：

1. 方法与四段证据链；
2. exact degree/top-k 与 finite-query fixed-k curves；
3. C vs A/ProxySPEX paired improvement；
4. exact pair coefficient/ranking comparison。

附录：

- 所有 per-q 指标；
- design geometry；
- matched-error support；
- hierarchy/sampler/refit；
- length/budget/seed；
- query ledger 与失败样本。

---

## 9. Discussion

讨论：

1. 优势来自真实结构、有限查询恢复还是 projection；
2. deletion semantics 的解释价值与局限；
3. 与 ProxySPEX 的互补关系；
4. sparse regression 作为标准组件带来的新颖性边界；
5. 训练与 faithfulness 均使用 deletion operator 的共同偏置。

---

## 10. Limitations

- 删除文本可能不自然；
- exact interaction 只相对于删除算子定义；
- 主实验以二阶为主；
- degree-2 candidates 随长度二次增长；
- exact full-table audit 只覆盖极短文本；
- pair oracle 默认限制到 32 个 players；
- sparse support recovery 依赖预算与正则化；
- 主要任务仍是文本分类；
- equal allocation 是规范公理选择，不保证对每个指标经验最优。

---

## 11. Conclusion

结论根据证据分支：

- 四段证据均成立：强调有限查询 sparse deletion interaction attribution；
- exact structure 成立但恢复不稳定：强调 query/solver 仍是瓶颈；
- recovery 相当但 attribution 更优：强调 deletion semantics 与 allocation；
- attribution 相当但 interaction 更准确：收缩为 deletion hypergraph recovery；
- exact structure 不成立：否定低阶稀疏前提，不宣称方法普遍有效。
