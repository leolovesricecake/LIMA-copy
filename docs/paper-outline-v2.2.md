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
=
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

* Sparse Deletion-Möbius Attribution：完整方法；
* Additive Deletion Model：使用相同采样与求解流程，但只保留一阶项；
* ProxySPEX；
* First-order Attribution Baseline：外部一阶方法，在相同 value function 和词粒度下运行。

受控消融：

* Singleton-only Projection：拟合完整二阶 surrogate，但最终排名只使用 singleton coefficients；
* Absolute-Interaction Projection：使用交互绝对值，忽略方向；
* No-refit：支持选择后不重新估计系数；
* Hierarchy-Constrained Recovery：附录中的结构分析。

## 5.4 Evaluation Metrics

Attribution faithfulness:

* AUPC，越低越好；

Function reconstruction:

* held-out (R^2)，越高越好；
* NRMSE，越低越好；
* MAE，越低越好；
* support size，用于衡量表示紧凑性。

Interaction recovery:

* Precision@(k)，正文 k 定 5；
* NDCG@(k)，正文 k 定 5；
* Spearman correlation；
* coefficient MAE；
* sign agreement。

Efficiency:

* unique black-box queries；
* model forward calls；

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

回答 RQ1，证明完整方法的实际归因价值。

### 结果
#### 表格：Overall Attribution Faithfulness（AUPC）。
行：不同方法
列：不同模型、不同数据集。

#### 图（可以不做或放附录）：Per-sample Paired AUPC Improvement

使用三组子图分别展示三个数据集。纵轴或横轴定义统一方向的改善值：
\[
\operatorname{AUPC}_i(\text{Baseline})
-
\operatorname{AUPC}_i(\text{Ours}),
]

因此 (\Delta_i>0) 表示我们的方法更好。

使用 violin plot 加内部 box plot，并绘制零线。分别展示：

* Ours vs Additive；
* Ours vs ProxySPEX；
* Ours vs First-order Baseline。

图中标出中位数、改善样本比例和 bootstrap confidence interval。

## 6.2 Compactness and Limited-Query Reconstruction

**本节全部在 Qwen3-8B 上实验。**

### 6.2.1 Exact Compactness on Short Texts

#### 目的
通过完整枚举，直接观察真实删除函数需要多少阶、多少个系数才能准确重建。验证使用低阶稀疏表示建模 masked prediction function 的合理性。

#### 操作
在可完整枚举的短文本上：
1. 查询全部 (2^n) 个删除集合；
2. 计算精确 deletion-Möbius transform 和 Fourier transform；
3. 分别进行 degree-(d) truncation；
4. 按系数贡献选择 top-(k) 项，并在完整 value table 上进行 OLS refit；
5. 计算精确 reconstruction NRMSE。

#### 结果
##### 图 ：Exact Compactness of Masked Prediction Functions

使用一个双子图。

(a): Degree-truncation curve

横轴：maximum degree (d)；
纵轴：exact reconstruction NRMSE；
曲线：Möbius 与 Fourier；
阴影：跨样本 bootstrap 95% CI。

用于展示 degree 1、2、3 分别可以解释多少 masked-function variation。

(b): Exact top-(k) reconstruction curve

横轴：保留的系数数量 (k)，可采用对数坐标；
纵轴：OLS-refitted reconstruction NRMSE；
曲线：Möbius 与 Fourier。

用于展示达到相同重建误差时，两种坐标各需要多少项。

#### 分析

低阶性：degree 2 相比 degree 1 是否显著降低 reconstruction error；
紧凑性：少量 top coefficients 是否能够恢复大部分 masked behavior；
坐标差异：在相同 (d) 或相同 (k) 下，Möbius 和 Fourier 哪个更紧凑，或者二者是否总体相当。

### 6.2.2 Limited-Query Reconstruction

#### 目的

验证在无法完整枚举的普通文本上，方法能否从有限 observations 中恢复未查询删除集合上的模型行为。

同时，在完全相同的 training masks 和 values 下比较 Möbius 与 Fourier，判断坐标选择对有限查询恢复的影响。

#### 操作

对于每个样本：

1. 使用相同的 training deletion masks 和 model values；
2. 分别构建 Möbius 和 Fourier design matrices；
3. 使用相同的标准化方式、alpha grid、稀疏求解器和 refit 规则；
4. 在与训练 masks 不重叠的 held-out masks 上评价；
5. 在多个查询预算 (B) 下重复实验。

#### 结果

##### 表格：Limited-query Reconstruction at the Default Budget

列：
Möbius Recovery；
Fourier Recovery；
Additive Degree-1 Model。

行按数据集分组，每个数据集报告：

* NRMSE (\downarrow)；
* (R^2) (\uparrow)；
* support size (\downarrow)。

##### 图：Recovery under Controlled Budgets and Support Sizes

双子图：

(a): Query-budget curve
* 横轴：query budget (B)；
* 纵轴：held-out NRMSE；
* 曲线：Möbius、Fourier、Additive；
* 阴影：跨 seed 的 uncertainty。

(b): Fixed-support reconstruction
* 横轴：固定 support size (k)；
* 纵轴：held-out NRMSE；
* 曲线：Möbius 与 Fourier。

#### 分析
1. 默认预算性能：根据 Table 2 比较相同 observations 下的 held-out reconstruction；
2. 样本效率：根据 Figure 4(a) 说明较少查询时哪种表示恢复更快，以及增加预算后是否趋于饱和；
3. 稀疏效率：根据 Figure 4(b) 判断相同 support size 下哪种坐标具有更低误差；
4. 与 Exact Compactness 的联系：结合 6.2.1 解释，紧凑表示是否真的能在有限查询下被恢复。

## 6.3 Exact Interaction Recovery

**本节全部在 Qwen3-8B 上实验。**

### 目的
验证恢复出的 top hyperedges 是否对应精确的 deletion-Möbius interactions。回答 RQ3。

### 操作

使用两类 exact oracle：

1. **Short-text full-table oracle**：利用完整 masked table 获得精确 coefficients；
2. **Ordinary-length pair oracle**：查询 full、singleton deletion 和 pair deletion，获得普通长度样本的全部精确 pair coefficients。

比较：
* Sparse Deletion-Möbius；
* ProxySPEX；

### 结果

#### 表格：Exact Interaction Recovery

采用上下两个 panel。
* Panel A: Short-text full-table oracle
* Panel B: Ordinary-length pair oracle

每个 panel 的行：
* Sparse Deletion-Möbius；
* ProxySPEX；
* Random。

列：
* Precision@5 (\uparrow)；
* NDCG@5 (\uparrow)；
* Spearman (\uparrow)；
* top-5 coefficient MAE (\downarrow)；
* sign agreement (\uparrow)。

Appendix：
- Precision@1/3/5、NDCG@1/3/5 结果。
- estimated coefficient 与 exact coefficient 的散点或密度图，并绘制 (y=x) 参考线。

需要注意：只有在 ProxySPEX 的输出被严格转换到 deletion-Möbius 坐标后，才能与 exact Möbius coefficients 计算 coefficient MAE 和 sign agreement。如果不等价转换，则 ProxySPEX 只比较 pair ranking metrics，系数指标标记为 N/A。（要不删除这两个指标吧？）

## 6.4 Interaction and Projection Ablations

**本节全部在 Qwen3-8B 上实验。**

### 目的

区分最终收益分别来自：

1. 使用二阶模型拟合 masked behavior；
2. 将恢复出的 interactions 纳入词级 ranking；
3. 保留 interaction sign；
4. 对稀疏选择后的系数进行 refit。

### 操作

比较以下变体：

- Full: degree-2 recovery + signed Shapley allocation；
- Additive: degree-1 model；
- Singleton-only Projection: 使用 Full surrogate，但排名只使用 singleton coefficients；
- Absolute Projection: 使用 Full surrogate，但忽略 interaction sign；
- No Refit: support selection 后不进行 coefficient refitting。

Singleton-only 和 Absolute Projection 完全复用 Full 的 training masks、support 和 coefficients，只改变最终 projection，确保能够隔离 projection 的作用。

### 结果

#### 表：Interaction Modeling

列：

* Additive；
* No Refit；
* Full。

行为不同数据集的：

* held-out NRMSE (\downarrow)；
* (R^2) (\uparrow)；
* AUPC (\downarrow)；

#### 表：Feature Projection - AUPC

行：

* Singleton-only Projection；
* Absolute Projection；
* Full Signed Projection。

列：数据集

### 分析

1. Full vs Additive：判断 interaction-aware framework 的总体收益；
2. degree 2 vs degree 1 reconstruction：说明 interaction terms 是否确实改善 masked-function fitting；
3. Full vs Singleton-only Projection：说明已经恢复的 interaction 是否必须进入最终 ranking；
4. Full vs Absolute Projection：说明保留 sign 是否有价值；
5. Full vs No Refit：说明 refitting 是否改善 coefficient estimation 和 attribution。

## 6.5 Robustness and Scaling Behavior（放附录）


* 不同 budget 下的 AUPC 图，对比 Full、Additive、ProxySPEX。
* alternative samplers 消融；
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
