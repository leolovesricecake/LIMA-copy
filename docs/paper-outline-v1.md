# Sparse Deletion-Möbius LLM Attribution：论文大纲 v1

## 核心主张

在匹配的黑盒查询预算下，直接拟合低阶稀疏 deletion-Möbius 表示，可以恢复具有明确删除语义、可被精确验证的交互；这些交互经过满足 surrogate completeness 的 signed projection 后，可以形成有竞争力的词级归因。

该主张分为三个必须独立验证的命题：

1. **Representation and recovery**：有限 observations 下能否稳定恢复 masked function；
2. **Interaction validity**：恢复的超边能否对应真实 deletion non-additivity；
3. **Attribution utility**：这些 interactions 是否改善最终词级 ranking。

若任一命题证据不足，论文结论必须相应缩小。

---

## 1. Introduction

### 1.1 问题

多数 LLM 归因方法最终为每个词给出独立分数，但预测经常依赖组合语义，例如否定、修饰、冗余与协同。

仅有 singleton importance 无法回答：

> 哪些词必须共同出现，模型才形成当前预测？

### 1.2 现有方法的不足

SPEX 与 ProxySPEX 展示了稀疏 Fourier interactions 的价值，但仍存在两个开放问题：

1. 文本删除归因是否更适合直接使用具有删除语义的坐标；
2. 间接 proxy/Fourier recovery 得到的 interactions 是否对应可验证的 deletion non-additivity。

本文不预设 hierarchy 错误，也不声称 Möbius 拥有 Fourier 不具备的低阶表达能力。

### 1.3 方法概述

1. 将输入划分为 word players；
2. 采样 deletion masks 并查询固定目标类别的预测值；
3. 在低阶 deletion-Möbius 字典中进行 sparse support selection 与 refit；
4. 保留 signed hyperedges；
5. 通过方向正确、满足 completeness 的投影得到词级归因。

### 1.4 贡献

1. 提出直接恢复 sparse signed deletion-Möbius hypergraph 的黑盒归因方法；
2. 将表示恢复、交互真实性和词级投影分开验证；
3. 在匹配预算与文本粒度下，对比 ProxySPEX、Fourier、additive 与投影消融；
4. 提供可复现的逐样本配对统计和查询成本分账。

---

## 2. Related Work

### 2.1 Feature Attribution for Language Models

- gradient-based attribution；
- perturbation/deletion attribution；
- LIME、SHAP、Integrated Gradients；
- word-level 与 token-level explanation。

### 2.2 Interaction Attribution

- Harsanyi/Möbius interactions；
- Shapley interaction indices；
- SPEX；
- ProxySPEX。

### 2.3 Sparse Boolean Function Recovery

- low-degree set functions；
- Fourier 与 monomial coordinates；
- sparse regression 与 support recovery；
- finite-query reconstruction。

本节明确：固定最大阶数时，两种坐标张成相同函数空间，论文比较的是有限查询恢复和下游语义，而非理论表达能力。

---

## 3. Problem Formulation

### 3.1 Masked Prediction Function

给定 word players：

\[
N=\{1,\ldots,n\}.
\]

令 \(D\subseteq N\) 为删除集合：

\[
g(D)=f(N\setminus D).
\]

主实验固定：

- predicted class of the full input；
- predicted-class probability；
- word-level deletion；
- 删除后不改变 target class。

### 3.2 Deletion-Möbius Interactions

\[
m^-(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}g(U).
\]

二阶：

\[
m^-(\{i,j\})
=
g(\{i,j\})-g(\{i\})-g(\{j\})+g(\varnothing).
\]

它表示联合删除中无法由 singleton deletion effects 解释的部分。

### 3.3 Three Evaluation Targets

论文区分：

1. **Function reconstruction error**；
2. **Interaction recovery error**；
3. **Feature attribution/projection error**。

更好的 held-out R² 不自动意味着更好的词级归因；更好的 AOPC 也不自动证明 interaction coefficients 更准确。

---

## 4. Method

### 4.1 Overview

```text
Word players
    → deletion masks
    → black-box values
    → sparse low-order recovery
    → signed deletion hypergraph
    → feature projection
```

### 4.2 Query Sampling

定义预算 \(B\) 和训练 observations：

\[
\mathcal D_B=\{(D_b,g(D_b))\}_{b=1}^{B}.
\]

主配置采用 `uniform_size`，空集与全集显式包含。模型调用、唯一文本和逻辑查询分别记账。

### 4.3 Low-Order Deletion Dictionary

\[
\widehat g(D)
=
\theta_\varnothing+
\sum_{\substack{T\subseteq N\\1\le |T|\le d}}
\theta_T\mathbf 1[T\subseteq D].
\]

主实验 \(d=2\)。方法形式支持更高阶，但不宣称在长文本上任意高阶仍高效。

### 4.4 Sparse Support Selection and Refitting

1. 标准化候选设计矩阵；
2. 稀疏回归选择 support；
3. 在选中 support 上 refit；
4. 输出系数、support 和训练 diagnostics。

不强制 hierarchy 只是 support policy，不单独宣称为算法贡献。

### 4.5 Signed Hypergraph-to-Feature Projection

\[
a_i
=
-\sum_{T\ni i}\frac{\widehat\theta_T}{|T|}.
\]

负号将 deletion effect 转换为特征对完整输入预测的贡献方向。

\[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
\]

该等式是 surrogate completeness。原模型与 surrogate 的残差单独报告。

### 4.6 Complexity and Outputs

- candidate count；
- query budget；
- support size；
- fitting time；
- model forward calls；
- node ranking 与 signed hyperedges。

---

## 5. Experimental Setup

### 5.1 Research Questions

- **RQ1**：方法是否具有竞争力的归因 faithfulness 与查询成本？
- **RQ2**：相同 observations 下 Möbius 与 Fourier 的恢复和稀疏性有何差异？
- **RQ3**：恢复的二阶边是否对应真实、可排序的 deletion interactions？
- **RQ4**：interaction modeling 与 signed projection 分别带来多少收益？
- **RQ5**：结论对 seed、任务、模型、长度和 held-out 分布是否稳定？

### 5.2 Datasets and Models

主任务：

- SST-2；
- Rotten Tomatoes；
- Emotion。

模型：

- Qwen3-8B；
- 第二个 8B instruct model，用于范围验证。

### 5.3 Compared Methods

- Sparse deletion-Möbius C；
- additive deletion A；
- ProxySPEX；
- 一个 protocol-compatible first-order baseline；
- B、ABSOLUTE、STRICT 作为消融。

### 5.4 Metrics

Faithfulness：

- AOPC；
- AUPC；
- comprehensiveness；
- sufficiency。

Surrogate：

- held-out R²；
- NRMSE；
- MAE；
- support size；
- top-k compression。

Interaction：

- Precision@k；
- NDCG@k；
- Spearman；
- coefficient MAE；
- sign agreement。

Cost：

- logical unique queries；
- physical values；
- attribution model forward calls；
- runtime；
- per-sample normalized cost。

### 5.5 Statistical Protocol

- 逐样本 paired difference；
- direction-normalized improvement；
- bootstrap 95% CI；
- Wilcoxon signed-rank test；
- paired effect size；
- common 与 unmatched sample counts。

---

## 6. Results

### 6.1 Overall Attribution Performance

**对应 RQ1。**

主表比较 C、A、ProxySPEX 和一阶 baseline：

- AOPC/AUPC；
- comprehensiveness/sufficiency；
- attribution forward calls；
- 查询预算与运行时间。

同时报告逐任务 paired improvement，避免只比较 aggregate means。

### 6.2 Representation and Finite-Query Recovery

**对应 RQ2。**

使用完全相同的 C training observations，比较 deletion-Möbius 与 Fourier：

- train/held-out R²、NRMSE、MAE；
- nonzero support size；
- top-k refit compression curve；
- Bernoulli 与 near-full held-out。

结论只讨论有限查询恢复，不讨论两种基的理论表达能力高低。

### 6.3 Exact Interaction Quality

**对应 RQ3。**

在完整枚举的 short-text samples 上比较：

- Möbius；
- ProxySPEX；
- random；
- oracle。

报告 Precision@k、NDCG@k、Spearman、coefficient MAE 和 sign agreement。

该实验验证 deletion interaction。

### 6.4 Interaction Modeling and Projection

**对应 RQ4。**

- C vs A：interaction model 的总体作用；
- C vs B：interactions 是否需要进入 ranking；
- C vs ABSOLUTE：signed direction 是否有价值；
- degree 2 vs degree 1 held-out：interaction 对函数重建的作用。

这一节明确区分 model improvement 与 projection improvement。

### 6.5 Robustness and Scope

**对应 RQ5。**

- seeds 42/43/44；
- dataset；
- text length；
- Bernoulli/near-full；
- second model；
- budget sensitivity。

Hierarchy、sampler 和 refit 放在本节末尾或附录。

---

## 7. Qualitative Analysis

展示：

- 否定；
- 程度修饰；
- 协同；
- 冗余；
- orphan interaction；
- 失败案例。

每个案例同时给出：

- singleton coefficients；
- signed pair edges；
- projected word scores；
- exact pair coefficients；
- deletion curve。

---

## 8. Discussion

讨论：

1. 效果来自坐标、恢复、采样还是 projection；
2. deletion semantics 的解释价值；
3. 与 ProxySPEX 的互补与差异；
4. surrogate fidelity 与 attribution faithfulness 的关系；
5. train/evaluation 均使用 deletion operator 的潜在偏置。

---

## 9. Limitations

- 删除后文本可能不自然；
- interaction 依赖 value function 与 target；
- 主实验以二阶为主；
- 长文本候选数量增长；
- sparse recovery 依赖预算和 regularization；
- 主要验证文本分类；
- exact audit 只覆盖 short-text samples。

---

## 10. Conclusion

结论只保留实验支持的强度：

- 若恢复与归因均优：强调完整方法；
- 若恢复相当但投影更优：强调 deletion semantics 与 signed projection；
- 若归因相当但 exact interaction 更好：收缩为 interaction hypergraph recovery。
