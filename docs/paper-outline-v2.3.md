# Sparse Deletion-Möbius Attribution for Language Models

## 核心思想

语言模型的预测不仅取决于单个词，还取决于词之间的协同、冗余、否定和修饰关系。本文将黑盒词级归因建模为有限删除查询下的低阶稀疏 deletion-Möbius game recovery 问题，并利用恢复出的带符号交互生成词级归因。

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

1. We formulate black-box word attribution as sparse deletion-Möbius recovery and develop a query-efficient procedure to recover signed deletion interactions from model evaluations；
2. We derive an interaction-aware attribution scheme that converts recovered deletion interactions into word-level rankings while preserving both singleton and combinational effects；
3. We provide a controlled study of Möbius and Fourier representations under matched observations, estimators, and sparsity constraints；
4. We validate recovered interactions with exact deletion-interaction oracles and demonstrate their contribution to attribution faithfulness。

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

最终根据 (a_i) 构造词级 ranking，同时保留原始 signed hyperedges 作为组合解释。具体地，主方法按照 **signed attribution (a_i) 从大到小** 排序：

[
i \prec j
\quad\Longleftrightarrow\quad
a_i>a_j,
]

分数相同时按词在原文中的顺序打破平局。这里不对 (a_i) 取绝对值：正的 (a_i) 表示该词的存在支持完整输入的固定预测类别，因而删除它倾向于降低目标概率；负的 (a_i) 表示该词抑制该类别。按 (|a_i|) 排序会将强反向证据也提前放入 deletion/retention 的 top-ranked words，从而改变主方法的方向性语义。**Sign-agnostic Projection** 仅作为受控变体；其代码实现也不是简单计算 (|a_i|)，而是在分配前对每条超边系数取绝对值：

[
a_i^{\mathrm{abs}}
=
\sum_{T\ni i}\frac{|\widehat\theta_T|}{|T|},
]

再按 (a_i^{\mathrm{abs}}) 降序排列。主方法始终使用 signed (a_i)。

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

实验围绕以下四个研究问题展开：

* **RQ1:** Does the proposed method improve word-level attribution faithfulness under a matched black-box query budget?
* **RQ2:** Can the proposed method faithfully reconstruct masked model behavior from limited deletion queries?
* **RQ3:** Do the recovered feature pairs correspond to exact deletion interactions?
* **RQ4:** Do recovered interactions improve the final feature ranking?

四个问题分别对应完整方法效果、函数恢复、交互恢复和词级投影，彼此不相互替代。

## 5.2 Datasets and Models

Datasets:

* SST-2；
* Rotten Tomatoes；
* AG News（wangrongsheng/ag_news）。

Models:

* Qwen3-8B；
* Llama-3.1-8B-Instruct。

E1 在两个模型上进行，以验证整体 attribution 结果的跨模型一致性。E2–E4 主要在 Qwen3-8B 上进行受控分析。

所有实验使用一致的：

* word-level player construction；
* predicted class of the full input；
* predicted-class probability value function（严格定义如下）；
* deletion operator；
* 对每个正式 split 的全集样本进行评估，不另建 evaluation sample pool。

主实验使用的 value function **不是 target log probability**。对于任意掩码后的文本 (z) 和类别 (c)，令该类别 verbalizer 的 tokenizer 序列为 (v_c=(v_{c,1},\ldots,v_{c,L_c}))，固定分类 prompt 为 (P(z))。代码首先计算长度归一化的 verbalizer score：

[
s_c(z)
=
\frac{1}{L_c}
\sum_{t=1}^{L_c}
\log p_\theta
\left(
v_{c,t}
\mid
P(z),v_{c,<t}
\right),
]

再只在数据集的类别 verbalizers 之间进行 softmax：

[
f_c(z)
=
\frac{\exp(s_c(z))}
{\sum_{c'\in\mathcal C}\exp(s_{c'}(z))}.
]

完整输入的解释目标固定为：

[
y=\arg\max_{c\in\mathcal C}s_c(x)
=\arg\max_{c\in\mathcal C}f_c(x),
]

对该样本的所有删除集合均使用同一个 (y)，因此：

[
g(D)=f_y(x_{\setminus D}).
]

实现中该协议对应 `value_function=predicted_probability`；等价配置是 `value_function=target_probability` 与 `target_mode=predicted` 的组合。不会为每个删除文本重新选择目标类别。

## 5.3 Compared Methods

### Main methods

* **Sparse Deletion-Möbius:** degree-2 sparse deletion-Möbius recovery followed by signed Shapley allocation；
* **Additive Deletion:** 使用相同 deletion sampling 和 recovery pipeline，但仅包含 singleton terms；
* **ProxySPEX:** 使用其原生 sampling、GBT proxy 和 interaction extraction pipeline；
* **First-order Baseline:** Occlusion、LIME。

### Controlled variants

* **Sparse Fourier:** 与 Sparse Deletion-Möbius 使用相同 observations、最大阶数和稀疏估计协议，但使用 Fourier basis；
* **Singleton-only Projection:** 复用完整 Möbius surrogate，但最终 ranking 只使用 singleton coefficients；
* **Sign-agnostic Projection:** 复用完整 Möbius surrogate，但忽略 interaction coefficients 的方向。

## 5.4 Evaluation Metrics

### Attribution faithfulness

* **AUPC (\downarrow):** 按 attribution ranking 逐步删除词后，目标预测曲线的归一化梯形面积；
* **AOPC-Sufficiency (\downarrow):** 在 (Q={5,10,20,50}) 上令 (k_q=\lfloor qn/100\rfloor)，计算 (s_q=f_y(x)-f_y(\operatorname{keep\_top}_{k_q}(x)))，再按 AML 口径取 (\sum_{q\in Q}s_q/(|Q|+1))。分母中的额外一项对应未扰动时隐式加入的零损失点。

AUPC 衡量删除重要词后预测是否快速下降，AOPC-Sufficiency 衡量在固定 retention 比例下选出的词是否足以维持预测。完整 retention trajectory 会增加 LLM 查询，因此本研究不报告 AUSufficiency。

### Function reconstruction

* **Held-out (R^2) (\uparrow):** surrogate 在未参与训练的 deletion masks 上对真实模型输出的解释程度。
* **Held-out NRMSE_range (\downarrow):** 对每个样本独立计算，再在样本间做宏平均，不汇总所有样本和 masks 后统一计算。具体地，对样本 (r) 的 (M_r) 个 held-out masks，记真实 value 和 surrogate 预测分别为 (y_{rm}) 与 (\widehat y_{rm})，定义：

[
R_r
=
\max_m y_{rm}-\min_m y_{rm},
]

[
\operatorname{NRMSE}_{r}
=
\frac{
\sqrt{
\frac{1}{M_r}
\sum_{m=1}^{M_r}(y_{rm}-\widehat y_{rm})^2
}
}{R_r}.
]

若同时评估多种 held-out mask 分布，则对每种分布分别计算；主结果使用 Bernoulli-0.5 held-out masks。若 (R_r\le 10^{-12})，该样本记为 degenerate，不参与 NRMSE 的均值和标准差，并单独报告 degenerate count。多随机种子实验先计算每个 sample-seed 的 NRMSE，再对同一样本跨种子取平均，最后报告样本间的 macro mean 和 sample standard deviation（`ddof=1`，仅一个样本时记为 0）。

选择逐样本标准化是因为每个解释对应一个独立的局部集合函数；将所有样本和 masks 池化会让 held-out mask 数较多的样本获得更大权重，并把样本间基础置信度差异混入分母。NRMSE-std 在同一组 values 上与 (R^2) 满足确定关系，信息重复；NRMSE-range 则补充刻画误差相对该局部函数实际变化范围的大小。

### Interaction recovery

* **NDCG@10 (\uparrow):** recovered pair ranking 与 exact interaction magnitude ranking 的一致性；
* **Recall@10 (\uparrow):** exact top-10 interactions 被恢复的比例。
* **Sign Agreement@10 (\uparrow):** 在 exact top-10 中排除精确系数为零的 pairs 后，估计系数与精确系数符号一致的比例。pair 总数少于 10 时，三项指标均使用 (K=\min(10,{n\choose 2}))。

## 5.5 Evaluation Protocol

* 所有方法使用相同的 word chunking、target class 和 value function；
* E1 和 E2 的端到端方法比较匹配 unique black-box query budget；
* attribution evaluation queries 不计入方法预算，并对所有方法保持一致；
* E2 的 Möbius–Fourier 受控比较复用完全相同的 training masks 和 values；
* held-out masks 与所有方法的 training masks 分离；
* 核心实验使用三个 attribution seeds；
* 表格报告 mean (\pm) standard deviation；
* 显著性分析以样本为配对单位，对三个 seeds 先求样本均值，再使用 bootstrap confidence interval 和 Wilcoxon signed-rank test。

---

# 6. Experimental Results

## 6.1 Overall Attribution Faithfulness

### 目的

回答 RQ1，验证完整方法产生的词级 ranking 是否比现有 interaction 和 first-order attribution 方法更加忠实。

### 操作

在每个 dataset–model combination 上比较：

* Sparse Deletion-Möbius；
* Additive Deletion；
* ProxySPEX；
* First-order Baseline。

Sparse Deletion-Möbius、Additive Deletion、ProxySPEX 与 LIME 使用相同的 requested unique-query budget；Occlusion 使用其确定性的 full 加全部 singleton-deletion 查询，并报告 realized query cost。所有方法使用相同的 evaluation deletion/retention ratios。

### 结果

报告一张 overall attribution table。

行：

* Sparse Deletion-Möbius；
* Additive Deletion；
* ProxySPEX；
* First-order Baseline。

列按 dataset 和 model 分组，报告：

* AUPC (\downarrow)；
* AOPC-Sufficiency (\downarrow)。

### 分析

正文依次分析：

1. Sparse Deletion-Möbius 是否在多个数据集和模型上取得更低的 AUPC；
2. AOPC-Sufficiency 是否得到一致结论，说明选出的词不仅删除后影响较大，而且在固定保留比例下足以维持预测；
3. 相对 Additive Deletion 的提升是否说明显式 interaction modeling 具有整体价值；
4. 相对 ProxySPEX 的结果是否说明直接 deletion-interaction recovery 在匹配预算下具有竞争力；
5. 结论是否在两个模型上保持相同方向。

---

## 6.2 Finite-Query Function Recovery

### 目的

回答 RQ2，验证方法能否从有限 deletion queries 中恢复模型在未见 masks 上的行为。

本实验同时考察：

1. interaction modeling 是否改善 masked-function reconstruction；
2. 在相同 observations 和模型复杂度下，Möbius 与 Fourier 坐标的有限查询恢复差异；
3. 与 ProxySPEX 相比，完整方法的 query–faithfulness trade-off。

### 操作

为每个样本独立生成 training masks 和 shared held-out masks。

#### End-to-end recovery comparison

在多个 query budgets 下比较：

* Sparse Deletion-Möbius；
* Additive Deletion；
* ProxySPEX。

每个方法保留自己的原生训练流程，但使用相同的 unique LLM query budget，并在同一 held-out mask set 上计算 (R^2)。

#### Controlled basis comparison

为隔离表示坐标的影响，首先进行 actual-estimator comparison：Sparse Deletion-Möbius 和 Sparse Fourier 复用完全相同的 training masks 与 values，并使用相同的 maximum degree、列标准化、Lasso support selection 和 RidgeCV refit。这一设置回答在主方法真实估计协议下，坐标选择是否改变恢复质量。

随后进行 fixed-k compression diagnostic。两种 basis 使用：

* 完全相同的 training masks；
* 完全相同的 model values；
* 相同 maximum degree (d=2)；
* 相同 column standardization；
* 相同 fixed support size (k)；
* 相同的确定性 standardized OMP support path；
* support 固定后的 OLS refit。

OMP 逐步选择与当前残差标准化相关性最大的列，产生嵌套 support path；OLS 在固定 support 上无正则重估系数，以避免 Lasso 收缩影响 fixed-k 表示能力比较。OMP+OLS 不是主方法估计器。每个 (k) 同时记录设计矩阵秩、rank-deficient 标记与 condition number，并比较 held-out (R^2) 和 NRMSE-range。

### 结果

报告一个双子图。

#### (a) Query-budget faithfulness

* 横轴：unique black-box query budget；
* 纵轴：held-out (R^2)；
* 曲线：Möbius、Additive Deletion、ProxySPEX。

#### (b) Fixed-support basis comparison

* 横轴：support size (k)；
* 纵轴：held-out (R^2)；
* 曲线：Möbius 与 Fourier。

### 分析

1. interaction terms 是否改善 recovery；
2. Möbius 是否更适合有限恢复。

---

## 6.3 Exact Pair Interaction Recovery

### 目的

回答 RQ3，验证方法识别出的 top feature pairs 是否对应原模型的精确 deletion interactions。

该实验与 E2 的区别在于：E2 评价整个 masked function，而本实验直接评价 interaction ranking。

### 操作

在每个数据集的正式 split 中，使用预先固定的随机种子从满足长度条件的共同成功样本随机抽取 (M=100) 条。

不进行额外长度分层，候选样本固定满足 (10\le n\le 32)。对于每个样本，查询：

[
g(\varnothing),\qquad
g({i}),\qquad
g({i,j}),
]

并计算全部精确二阶 deletion interactions：

[
m^-_{ij}
=
g({i,j})
-g({i})
-g({j})
+g(\varnothing).
]

这些 oracle queries 只用于评价，不参与方法训练和超参数选择。

比较：

* Sparse Deletion-Möbius；
* ProxySPEX。

为了统一 interaction semantics，对每个方法学习到的 surrogate (\widehat g)，均计算：

[
\widehat m^-_{ij}
=
\widehat g({i,j})
-\widehat g({i})
-\widehat g({j})
+\widehat g(\varnothing).
]

根据 (|\widehat m^-*{ij}|) 对 pairs 排序，并与根据 (|m^-*{ij}|) 得到的 exact ranking 比较。

### 结果

报告 Exact Pair Interaction Recovery 表格。

行：

* Sparse Deletion-Möbius；
* ProxySPEX。

列按数据集分组：

* NDCG@10 (\uparrow)；
* Recall@10 (\uparrow)；
* Sign Agreement@10 (\uparrow)。

对于 pair 数量少于 10 的样本，使用：

[
K_i=\min\left(10,\binom{n_i}{2}\right).
]

### 分析

1. 是否找到真实重要 pair；
2. 是否比 ProxySPEX 更接近 deletion interaction。

coefficient calibration 作为补充分析，可以放附录。

---

## 6.4 Interaction-Aware Projection

### 目的

回答 RQ4，验证恢复出的 interactions 是否需要被纳入词级 ranking，以及 interaction direction 是否对归因有实际价值。

### 操作

所有变体共享同一个完整 degree-2 Möbius surrogate，包括相同的：

* training masks；
* model values；
* support；
* coefficients。

仅改变 hypergraph-to-feature projection。

#### Signed Shapley Projection

[
a_i^{\mathrm{signed}}
=
-\sum_{T\ni i}\frac{\widehat\theta_T}{|T|}.
]

#### Singleton-only Projection

[
a_i^{\mathrm{single}}
=
-\widehat\theta_{{i}}.
]

#### Sign-agnostic Projection

[
a_i^{\mathrm{abs}}
=
\sum_{T\ni i}\frac{|\widehat\theta_T|}{|T|}.
]

比较三种 projection 产生的词级 ranking。

### 结果

报告 Interaction-Aware Projection 表格。

行：

* Singleton-only Projection；
* Signed Shapley Projection；
* Sign-agnostic Projection。

列按数据集分组：

* AUPC (\downarrow)；
* AOPC-Sufficiency (\downarrow)。

### 分析

正文依次分析：

1. interaction 是否提供 singleton 之外的信息；
2. recovered interactions 是否真正贡献最终 attribution；
3. interaction 的符号方向是否比只看 magnitude 更有价值。

---

# 7. Additional Analyses

以下内容不单独形成 research question，作为补充稳健性分析：

* attribution faithfulness under different query budgets；
* performance across text-length bins；
* alternative deletion samplers；
* hierarchy-constrained support；
* refit and regularization sensitivity；
* exact sign agreement；
* additional qualitative examples。

这些实验用于说明方法行为和适用范围，不承担新的核心论文主张。

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

---

# 附录

除了 7 中提到的分析之外，还应该放：

* 具体超参数；
* prompt 模版与 verbalizer 打分。主方法和 ProxySPEX 共用 `mobius/models/hf.py` 中的 `HFVerbalizerScorer`（ProxySPEX 通过别名 `HFBackbone` 使用同一实现）。prompt 协议版本为 `task_classification_v1`，实际模版为：

```text
Task: {task_description}
Candidate labels: label_1 | label_2 | ...
Return exactly one candidate label without quotation marks or explanation.
Text:
{masked_text}
Label: {class_verbalizer}
```

其中 `{task_description}` 由数据集确定，候选标签按 dataset verbalizers 的 scoring order 完整列出。对于具有 `chat_template` 的 instruction model，逻辑内容被放入 user message，并由 tokenizer 构造 assistant generation prefix；Qwen3 设置 `enable_thinking=False`，随后直接拼接 class verbalizer。无 chat template 时才使用图示的 `Label:` plain-text fallback 和 verbalizer 前导空格。两种路径都设置 `add_special_tokens=False`，`_score_label()` 只对 verbalizer tokens 的条件对数概率取平均。若总长度超过 `max_length`，固定保留任务说明、候选标签、assistant/`Label:` prefix 和完整 verbalizer，只从 `{masked_text}` 左侧截断；对应的可解释词范围由 `mobius/text/coalitions.py` 中的 `visible_text_span()` 对齐。数据集 verbalizers 由 `mobius/data/loader.py` 提供。
