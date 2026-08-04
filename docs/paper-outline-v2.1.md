# Sparse Deletion-Möbius Attribution for Language Models

## 核心思想

语言模型的预测不仅取决于单个词，还取决于词之间的协同、冗余、否定和修饰关系。我们将模型在词删除扰动下的行为表示为一个集合函数，并从有限黑盒查询中直接恢复稀疏、带符号的 deletion-Möbius interactions。恢复的交互构成一个 signed hypergraph，并通过 Shapley allocation 转化为满足 completeness 的词级归因。

## 核心主张

> Directly recovering sparse deletion-Möbius interactions from limited black-box queries provides an effective way to model non-additive word effects and construct faithful feature attributions.

论文围绕三个问题展开：

1. 能否从有限删除查询中准确重建模型的 masked behavior？
2. 恢复的超边是否对应精确的 deletion interactions？
3. 将这些 interactions 分配到词后，能否改善词级 attribution faithfulness？

---

# 1. Introduction

## 1.1 Background and Motivation

现有语言模型归因方法通常为每个词分配一个独立的重要性分数。然而，许多模型行为依赖词组合，例如否定、程度修饰、语义协同和信息冗余。此时，模型对词组合的响应不能由各词独立效应的简单相加完整表示。

因此，忠实的词级归因不仅需要识别重要词，还需要刻画这些词如何共同作用。

## 1.2 Limitations of Existing Interaction Attribution

SPEX 和 ProxySPEX 表明，稀疏谱表示可以用于发现语言模型中的高阶交互。它们主要通过 Fourier representation 和 proxy model 发现 interaction structure。

对于基于删除的局部归因，一个自然的问题是：能否直接在具有明确删除语义的 Möbius 坐标中恢复交互，从而同时得到可验证的非加性表示和词级 attribution？

## 1.3 Key Insight

对于删除集合 (D)，模型输出可以写成 deletion-Möbius expansion。每个系数直接表示联合删除一个词集合时，相对于其低阶子集效应产生的额外变化。

如果该表示在局部具有低阶稀疏结构，就可以从有限黑盒查询中恢复少量重要超边，而无需枚举全部 (2^n) 个删除集合。

## 1.4 Method Overview

给定输入文本，我们：

1. 将文本划分为 word-level players；
2. 采样删除集合并查询固定目标类别的模型分数；
3. 在低阶 deletion-Möbius 字典中进行 sparse support recovery 和 coefficient refitting；
4. 将非零系数表示为 signed interaction hypergraph；
5. 通过 deletion-Shapley allocation 将超边贡献分配给成员词；
6. 输出词级 attribution ranking 和显式 interaction explanation。

## 1.5 Contributions

本文的主要贡献包括：

1. 提出一种从有限黑盒查询中直接恢复 sparse signed deletion-Möbius hypergraph 的归因方法；
2. 提出基于 Shapley/Harsanyi allocation 的 interaction-aware feature attribution，将删除交互转换为满足 surrogate completeness 的词级贡献；
3. 在匹配 value function、文本粒度和查询预算的条件下，系统比较直接 Möbius recovery、Fourier recovery、ProxySPEX 和 additive attribution；
4. 通过 exact interaction oracle、逐样本配对统计和多任务实验，验证恢复交互的准确性及其对 attribution faithfulness 的作用。

---

# 2. Related Work

## 2.1 Feature Attribution for Language Models

介绍：

* gradient-based attribution；
* perturbation and deletion attribution；
* LIME、SHAP、Integrated Gradients；
* token-level、word-level 和 phrase-level explanations。

指出多数方法最终将预测解释为独立特征分数，对显式组合效应的表示有限。

## 2.2 Interaction Attribution

介绍：

* Harsanyi dividends；
* Möbius interactions；
* Shapley interaction indices；
* SPEX；
* ProxySPEX；
* interaction-to-feature allocation。

## 2.3 Sparse Set-Function Recovery

介绍：

* low-degree Boolean and set functions；
* Fourier and monomial coordinates；
* sparse support recovery；
* finite-query approximation；
* query sampling。

说明固定最大阶数时，Fourier 和 monomial bases 可以表示相同的低阶函数空间，而不同坐标在稀疏结构、有限样本恢复和交互语义上可能表现不同。

---

# 3. Problem Formulation

## 3.1 Input and Explanation Target

给定输入文本：

[
x=[w_1,w_2,\ldots,w_n],
]

其中 (w_i) 是 word-level player，定义：

[
N={1,\ldots,n}.
]

令完整输入的预测类别为：

[
y=\arg\max_c f_c(x).
]

所有删除 coalition 均使用固定类别 (y) 的预测分数。

## 3.2 Masked Prediction Function

对于删除集合 (D\subseteq N)，定义：

[
g(D)=f_y(x_{\setminus D}).
]

其中 (x_{\setminus D}) 表示删除 (D) 中词后的文本。

## 3.3 Deletion-Möbius Interactions

任意集合 (T\subseteq N) 的 deletion-Möbius coefficient 为：

[
m^-(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}g(U).
]

它表示联合删除 (T) 时，无法由其低阶子集删除效应解释的部分。

例如，二阶交互为：

[
m^-({i,j})
=
g({i,j})
-g({i})
-g({j})
+g(\varnothing).
]

## 3.4 Learning Objective

在给定查询预算 (B) 和最大阶数 (d) 的条件下，我们希望恢复一个稀疏超边集合：

[
\widehat{\mathcal H}
=
{T:\widehat\theta_T\neq0},
]

使得：

1. (\widehat g) 能准确重建未查询删除集合上的模型行为；
2. (\widehat\theta_T) 能准确表示重要 deletion interactions；
3. 恢复的超边能够生成忠实的词级 attribution。

---

# 4. Sparse Deletion-Möbius Attribution

## 4.1 Overview

本节首先给出方法流程图：

```text
Input text
    ↓
Word-level players
    ↓
Deletion query sampling
    ↓
Sparse Möbius support recovery
    ↓
Coefficient refitting
    ↓
Signed interaction hypergraph
    ↓
Shapley interaction allocation
    ↓
Word-level attribution
```

## 4.2 Deletion Query Sampling

给定预算 (B)，构造：

[
\mathcal D_B
=
{(D_b,g(D_b))}_{b=1}^{B}.
]

主方法采用 cardinality-balanced deletion sampling：

1. 在不同删除规模之间均匀采样；
2. 在固定规模内均匀采样 coalition；
3. 显式加入 empty 和 full anchors；
4. 对重复 masked inputs 进行缓存。

## 4.3 Sparse Low-Degree Möbius Recovery

使用最高阶数为 (d) 的模型：

[
\widehat g(D)
=
\theta_\varnothing
+
\sum_{\substack{T\subseteq N\1\le |T|\le d}}
\theta_T\mathbf 1[T\subseteq D].
]

设计矩阵为：

[
\Phi_{b,T}
=
\mathbf 1[T\subseteq D_b].
]

方法包括：

1. 对候选设计列进行标准化；
2. 使用稀疏回归选择重要 interaction support；
3. 在选中 support 上重新估计系数；
4. 得到稀疏的低阶 deletion-Möbius representation。

主实验采用 (d=2)，并在短文本子集上分析更高阶设置。因为二阶已经能表示否定、修饰、协同和冗余等重要现象，并且适合在大量普通长度文本上进行精确验证。

## 4.4 Signed Interaction Hypergraph

每个非零系数 (\widehat\theta_T) 表示一条 signed hyperedge：

[
T\longleftrightarrow\widehat\theta_T.
]

超边的大小表示交互阶数，系数绝对值表示交互强度，符号表示联合删除对目标分数产生的非加性方向。

方法同时保留 singleton effects 和 interaction effects，从而生成完整的局部超图解释。

## 4.5 Interaction-Aware Feature Attribution

为了从超图得到词级 attribution，对每条 interaction contribution 进行 Shapley allocation：

[
\phi_i(\widehat g)
=
\sum_{T\ni i}
\frac{\widehat\theta_T}{|T|}.
]

由于 (\phi_i) 描述删除方向，完整输入中的 feature-presence attribution 定义为：

[
a_i
=
-\phi_i(\widehat g)

-\sum_{T\ni i}
\frac{\widehat\theta_T}{|T|}.
]

该分配满足 efficiency、symmetry、linearity 和 dummy-player property，并具有：

[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
]

最终根据 (a_i) 构造词级 ranking，同时保留原始 signed hyperedges 作为组合解释。

## 4.6 Algorithm and Complexity

给出完整算法伪代码，并分析：

* 查询复杂度：(B)；
* degree-2 candidate count：

[
n+\binom n2;
]

* 稀疏求解和 refit 的时间复杂度；
* 输出 support size 和实际模型调用成本。

正文只保留主要复杂度，详细运行成本和实现细节放入附录。

---

# 5. Experimental Setup

## 5.1 Research Questions

* **RQ1:** Does the proposed method improve word-level attribution faithfulness under a matched query budget?
* **RQ2:** How accurately can deletion-Möbius representations reconstruct masked model behavior from limited queries?
* **RQ3:** Do the recovered hyperedges correspond to exact deletion interactions?
* **RQ4:** How do interaction modeling, signed coefficients, and Shapley allocation contribute to the final attribution?
* **RQ5:** How robust is the method across budgets, text lengths?

## 5.2 Datasets and Models

Datasets:

* SST-2；
* Rotten Tomatoes；
* Emotion。

Models:

* Qwen3-8B；
* Llama-3.1-8B-Instruct。

说明数据集选择、verbalizer、样本筛选和文本长度分布。

## 5.3 Compared Methods

主要方法：

* Sparse Deletion-Möbius Attribution；
* Additive Deletion Model；
* ProxySPEX；
* protocol-compatible first-order attribution baseline。

受控消融：

* Singleton-only Projection；
* Absolute-Interaction Projection；
* Hierarchy-Constrained Recovery。

## 5.4 Evaluation Metrics

Attribution faithfulness:

* AOPC/AUPC，先选 AUPC 吧；
* aopc-comprehensiveness；
* aopc-sufficiency。

Function reconstruction:

* held-out (R^2)；
* NRMSE；
* MAE。

Interaction recovery:

* Precision@(k)；
* NDCG@(k)；
* Spearman correlation；
* coefficient MAE；
* sign agreement。

Efficiency:

* unique black-box queries；
* model forward calls；
* support size。

## 5.5 Evaluation Protocol

* 使用一致的 word chunking、target class 和 value function；
* 对整体归因方法匹配黑盒查询预算；
* basis comparison 使用完全相同的 training observations；
* held-out masks 与所有训练 masks 分离；
* 核心实验运行 3 个 attribution seeds；
* 报告 mean ± standard deviation；
* 对逐样本差异使用 bootstrap confidence interval、Wilcoxon signed-rank test 和 paired effect size。

---

# 6. Experimental Results

## 6.1 Overall Attribution Faithfulness and Efficiency

回答 RQ1。

比较：

* Full method vs Additive Deletion Model；
* Full method vs ProxySPEX；
* Full method vs first-order baseline。

报告：

* AUPC、aopc-comprehensiveness、aopc-sufficiency 表格（mean ± standard）；
* unique queries、forward calls（可选）；
* 逐样本 paired improvement 分布图。

这一节首先证明完整方法的实际归因价值。

## 6.2 Compactness and Limited-Query Reconstruction

回答 RQ2。

### 6.2.1 Exact Compactness on Short Texts

在可完整枚举的短文本上：

1. 计算完整 masked prediction table；
2. 得到精确 Möbius 和 Fourier representations；
3. 比较不同 degree truncation curve 图；
4. 比较 exact top-(k) reconstruction curves 图。

该实验分析 masked functions 是否可以由少量低阶项紧凑表示。
它回答“为什么用一个低阶稀疏模型去表示删除函数是合理的？”，即证明方法采用低阶稀疏模型的建模合理性：
- degree 1、2、3 分别可以解释多少函数变化；
- 保留 top-k 系数后能够达到什么重建精度；
- Möbius 和 Fourier 在真实完整函数上谁更紧凑。

### 6.2.2 Limited-Query Reconstruction

在相同 training masks 和 values 下，比较 Möbius 与 Fourier recovery：

* held-out (R^2)、NRMSE、MAE 表格（mean ± standard）；
* fixed-(k) reconstruction curves 图；
* 达到相同误差所需的 support size；
* 不同查询预算下的恢复曲线图。
【TODO：结果是不是有点太多了？】

该实验验证直接 deletion-Möbius recovery 在有限 observations 下的有效性。
即使上一小节中证明了真实函数可以被少量 Möbius 项表示，也不代表方法从少量查询中一定能找到这些项。因此，这里固定同一批训练 masks 和 values，测试：“在相同黑盒信息下，Möbius 和 Fourier 哪个更容易被稀疏估计器恢复？”

设计矩阵 coherence、rank 和 condition number 等诊断结果放入附录。

## 6.3 Exact Interaction Recovery

回答 RQ3。

使用两类 exact oracle：

1. **Short-text full-table oracle**：利用完整 masked table 获得精确 coefficients；
2. **Ordinary-length pair oracle**：查询 full、singleton deletion 和 pair deletion，获得普通长度样本的全部精确 pair coefficients。

比较：

* Sparse Deletion-Möbius；
* ProxySPEX；
* exact oracle。

报告：

* Precision@(k)；
* NDCG@(k)；
* Spearman；
* coefficient MAE；
* sign agreement；
* 不同文本长度下的结果。

该实验验证恢复出的 top hyperedges 是否对应模型的真实 deletion non-additivity。

## 6.4 Interaction and Projection Ablations

回答 RQ4。

比较：

1. **Full vs Additive**：interaction-aware modeling 的总体贡献；
2. **Full vs Singleton-only Projection**：恢复的 interactions 是否需要进入词级 ranking；
3. **Full vs Absolute-Interaction Projection**：interaction sign 是否有价值；
4. **Degree 2 vs Degree 1 reconstruction**：二阶项是否改善 masked-function approximation；
5. **Refit vs No Refit**：coefficient refitting 的作用。

报告 attribution metrics、held-out reconstruction 和 completeness residual 表格（mean ± standard）。

Hierarchy-constrained recovery 作为附加结构分析放在本节末尾或附录。

## 6.5 Robustness and Scaling Behavior

回答 RQ5。【TODO：上面结果已经很多了，这里视情况放附录】

正文分析：

* query-budget sensitivity；
* text-length bins；
* three-seed stability；
* second-model validation。

附录分析：

* alternative samplers；
* Bernoulli vs near-full held-out masks；
* hierarchy policy；
* regularization and refit sensitivity。

---

# 7. Qualitative Analysis

选择具有代表性的案例：

* negation；
* degree modification；
* semantic synergy；
* redundancy；
* isolated interaction；
* failure case。

每个案例展示：

* 输入文本和预测；
* singleton coefficients；
* recovered signed interactions；
* exact pair coefficients；
* allocated word attributions；
* deletion curve。

案例用于直观说明方法如何利用 interaction 改变词级解释。

---

# 8. Discussion

讨论：

1. masked-function reconstruction 与 attribution faithfulness 的关系；
2. deletion-Möbius coordinates 的语义和恢复特性；
3. 直接 recovery 与 ProxySPEX 的差异；
4. interaction hypergraph 相比单一词排序提供的额外信息；
5. 查询预算、文本长度和高阶扩展的实际影响；
6. deletion-based evaluation 的适用范围。

---

# 9. Limitations

简洁说明：

* 删除输入可能偏离自然语言分布；
* interaction 取决于 value function 和目标类别；
* 大规模实验主要使用二阶交互；
* 候选数量随文本长度和最大阶数增长；
* exact full-table analysis 仅适用于短文本；
* 当前主要验证文本分类任务。

不要在此重新否定论文贡献，只说明未来可扩展到更自然的替换扰动、候选筛选和生成任务。

---

# 10. Conclusion

本文提出一种从有限黑盒删除查询中直接恢复 sparse signed deletion-Möbius interactions 的归因方法。该方法将语言模型的非加性删除行为表示为稀疏超图，并通过 Shapley allocation 生成满足 completeness 的词级归因。实验从 masked-function reconstruction、exact interaction recovery 和 attribution faithfulness 三个层面验证了方法，并分析了 interaction modeling、符号和投影机制的作用。
