# 一、核心思想

## 1. 一句话版本

> 我们直接从有限的删除查询中恢复稀疏、带符号的 deletion-Möbius 超图，以显式建模特征组合对模型预测的非加性影响，并通过满足方向完整性的投影将这些交互用于特征归因。

更简短的英文思想可以是：

> **Directly recover sparse signed deletion interactions for faithful feature attribution.**

这里有四个关键词：

* **Direct**：直接在 deletion-Möbius 坐标中恢复，而不是先学习另一种代理表示再转换；
* **Sparse**：只保留少量有贡献的单词和特征组合；
* **Signed**：区分交互是在促进还是抑制目标预测；
* **Deletion interaction**：解释完整输入附近，删除特征组合所产生的额外非加性影响。

---

## 2. 一般阶数的数学表达

给定输入的特征集合：

\[
N=\{1,\ldots,n\},
\]

令 (D\subseteq N) 表示被删除的特征，定义删除集合函数：

\[
g(D)=f(N\setminus D).
\]

其 deletion-Möbius 系数为：

\[
m^-(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}g(U),
\qquad T\subseteq N.
\]

其中：

* (|T|=1)：单个特征的删除效应；
* (|T|=2)：词对交互；
* (|T|=3)：三个词共同作用的交互；
* 更大的 (|T|)：更高阶组合效应。

在最大阶数 (d) 下，我们学习：

\[
\widehat g(D)
=
\sum_{\substack{T\subseteq N\\|T|\le d}}
\theta_T\mathbf 1[T\subseteq D].
\]

也可以写成：

\[
\widehat g(z)
=
\sum_{\substack{T\subseteq N\\|T|\le d}}
\theta_T\prod_{i\in T}z_i,
\]

其中 (z_i=1) 表示删除特征 (i)。

这个公式明确说明：方法可以建模任意不超过 (d) 阶的交互。

---

## 3. 为什么当前实验以二阶为主？

论文中应明确区分：

### 方法能力

\[
d\ge1
\]

是可配置的，原则上可以加入三阶及以上候选。

### 本文主要实例

\[
d=2.
\]

选择二阶有三个实际原因：

1. 二阶已经能表示否定、修饰、短语组合等重要文本现象；
2. 候选数量和拟合成本可控；
3. 一个二阶交互只需四个删除组合即可精确验证。

对于 (r) 阶交互，精确验证需要：

\[
2^r
\]

个相关组合，候选数量也会迅速增加。因此本文以二阶为主要配置，是计算与验证上的取舍，不是方法定义的限制。

论文中可以写：

> We instantiate (d=2) in the main experiments to balance expressiveness, recovery difficulty, and exact verifiability.

需要避免声称：

> 方法可以高效扩展到任意高阶。

数学形式可以扩展，不代表在长文本上高阶候选仍然高效。

---

# 二、如何引入这项工作

引言不应从 Möbius 公式开始，也不应从“ProxySPEX 有什么缺点”开始。更好的逻辑是：

```text
文本预测依赖组合语义
→ 单特征归因不足
→ 已有交互方法仍存在间接恢复和结构偏置
→ 删除归因需要完整输入附近的带符号非加性解释
→ 直接恢复稀疏 deletion-Möbius 超图
```

建议采用五段式引入。

---

## 第一段：单特征归因不足

先从用户容易认同的问题开始：

> 现有归因通常给每个词分配一个重要性分数，但模型预测经常取决于词的组合，而不是独立词效应的简单相加。

可以使用：

* 否定：`not good`；
* 程度修饰：`hardly convincing`；
* 多词短语；
* 上下文依赖。

要表达的观点是：

\[
\text{组合的作用}
\neq
\text{各特征独立作用之和}.
\]

因此，忠实解释需要显式表示 interaction。

---

## 第二段：已有交互归因的进展与不足

先肯定已有工作：

* SPEX 利用稀疏谱表示发现高阶交互；
* ProxySPEX 观察到 Fourier 交互常呈现层级结构，并利用 GBT 提高查询效率。

然后指出尚未解决的问题：

1. 交互通过树代理和 Fourier support 间接发现；
2. 层级性是有用的经验规律，但不是所有有效交互都必须满足的硬条件；
3. 文本删除归因最终关心的是完整输入附近的删除效应；

措辞要克制：

> Hierarchical structure is often useful, but relying on it as a discovery bias may underrepresent interactions whose lower-order effects are weak.

---

## 第三段：提出核心洞察

核心洞察可以表述为：

> 删除型局部归因可以自然地表示为 deletion-Möbius 超图。每条超边刻画一个特征集合的联合删除效应中，无法由其低阶子集解释的部分。

这带来三个好处：

* 交互语义直接对应删除干预；
* 系数带有明确正负方向；
* 高阶项不必依赖其父项被选中。

由此自然提出：能否直接从有限查询中恢复稀疏 deletion-Möbius 表示，并在相同 observations 下获得不弱于 Fourier 的恢复与归因效果。这是本文需要验证的问题，而不是预设结论。

---

## 第四段：方法概述

引言中只需讲四步：

1. 采样特征删除集合，并查询原模型；
2. 构造最高到 (d) 阶的 deletion-Möbius 候选；
3. 使用稀疏估计选择少量重要超边，并重新估计其系数；
4. 保留超边的符号，将交互贡献聚合到成员特征，得到特征排名，同时保留原始超图解释。

---

## 第五段：贡献与证据

建议控制在三项。

### 贡献一：方法

提出一种直接的稀疏 deletion-Möbius 归因框架，从有限黑盒查询中恢复带符号的低阶特征交互。

### 贡献二：分层证据

系统区分表示、有限查询恢复和超图到特征投影，并通过 same-observation 与 exact small-n 审计分别验证。

### 贡献三：实验

在多个数据集和模型上验证方法的归因 faithfulness，并通过消融说明坐标、交互建模和 signed projection 各自的作用。只有 development gate 支持时，才将紧凑性、恢复优势或 hierarchy 写成论文结论。

---

# 三、全文大纲

## 1. Introduction

这一节完成五件事：

1. 说明单特征归因为什么不足；
2. 说明 interaction attribution 的必要性；
3. 肯定 SPEX、ProxySPEX 的进展；
4. 指出间接支持发现和层级偏置留下的空间；
5. 提出直接恢复 signed deletion-Möbius 超图。

引言最后给出方法概览、贡献和主要结果。

---

## 2. Related Work

### 2.1 Feature Attribution for Language Models

介绍：

* 梯度归因；
* 遮蔽/删除归因；
* SHAP、LIME 等.

最后指出：多数方法最终将预测分解为独立的特征分数，对组合效应表示有限。

---

### 2.2 Interaction Attribution

介绍：

* Shapley interaction；
* Harsanyi/Möbius interaction；
* interaction indices；
* SPEX；
* ProxySPEX。

并在末尾言简意赅地说明我们方法的定位。

---

### 2.3 Sparse Set-Function Approximation

这一节用于提供必要背景：

* 掩码函数；
* 稀疏低阶表示；
* 有限查询下的支持恢复。

---

## 3. Problem Formulation

### 3.1 Masked Prediction Function

给定输入：

\[
x=[w_1,\ldots,w_n],
\]

定义特征集合 (N)，删除集合 (D)，以及：

\[
g(D)=f(N\setminus D).
\]

明确：

* player 的定义；
* 删除算子；
* value function；
* 目标类别如何确定。

---

### 3.2 Deletion Interactions

定义一般阶数的 deletion-Möbius 系数：

\[
m^-(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}g(U).
\]

解释：

> (m^-(T)) 是联合删除 (T) 的效果中，无法由其所有真子集的删除效果解释的部分。

这里可以用二阶作为例子：

\[
m^-(\{i,j\})
=
g(\{i,j\})-g(\{i\})-g(\{j\})+g(\varnothing).
\]

---

### 3.3 Learning Objective

目标为：

> 在给定查询预算 (B) 和最大阶数 (d) 的条件下，恢复一个稀疏超边集合及其系数。

可以写成：

\[
\widehat g(D)
=
\sum_{\substack{T\in\widehat{\mathcal H}\\|T|\le d}}
\theta_T\mathbf 1[T\subseteq D],
\]

其中：

\[
|\widehat{\mathcal H}|
\ll
\sum_{r=0}^{d}\binom nr.
\]

我们希望该表示：

* 能近似原模型的掩码行为；
* 只保留少量超边；
* 保留交互符号；
* 不强制父项存在。

---

## 4. Direct Sparse Deletion-Möbius Attribution

方法章节标题应突出“直接恢复”，而不是某个具体优化器。

### 4.1 Overview

放一张流程图：

```text
Input features
    ↓
Deletion queries
    ↓
Sparse hyperedge recovery
    ↓
Coefficient refitting（不方便可视化的话，可以省略）
    ↓
Signed interaction hypergraph
    ↓
Feature attribution
```

简要说明每一步的输入输出。

---

### 4.2 Deletion Query Construction

介绍：

* 如何生成 deletion masks；
* 如何形成训练数据：

\[
\mathcal D_B={(D_b,g(D_b))}_{b=1}^{B};
\]

* 如何控制预算；
* 如何批量查询和缓存。

---

### 4.3 Low-Order Hyperedge Dictionary

对给定最大阶数 (d)，定义候选集合：

\[
\mathcal C_d
=
\{T\subseteq N:1\le |T|\le d\}.
\]

设计矩阵为：

\[
\Phi_{b,T}
=
\mathbf 1[T\subseteq D_b].
\]

这里保持一般 (d)，然后补充：默认采用 (d=2)，三阶及以上可通过扩大候选字典加入，但会增加候选数量和恢复难度。

固定最大阶数时，低阶 Möbius 字典与低阶 Fourier 字典张成相同的函数空间。因此，本方法不主张 Möbius 具有 Fourier 不具备的低阶表达能力。两者的差异需要在相同 observations 下通过坐标稀疏性、有限查询恢复、held-out reconstruction、interaction ranking 和最终 feature projection 进行比较。

---

### 4.4 Sparse Support Recovery and Refitting

介绍：

1. 稀疏估计选择候选超边；
2. 在选中支持上重新估计系数；
3. 输出：

\[
\widehat{\mathcal H}
=
\{T:\widehat\theta_T\neq0\}.
\]

讲解两个作用：

* support selection：找哪些特征集合重要；
* refitting：减小稀疏正则造成的系数偏差。

---

### 4.5 Signed Hyperedge Attribution

从恢复的超图产生特征级得分：

\[
a_i
=
-\sum_{\substack{T\in\widehat{\mathcal H}\\i\in T}}
\frac{\widehat\theta_T}{|T|}.
\]

这里：

* (|T|=1) 为单特征贡献；
* (|T|=2) 为词对贡献；
* 还可以有更高阶超边贡献。

解释：

* 负号把“删除集合对 \(g\) 的影响”转换为“特征对完整输入预测的贡献方向”；
* 系数符号被保留；
* 每条超边贡献平均分配到其成员；
* 最终同时输出节点排名和超边解释。

该投影满足 surrogate completeness：

\[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
\]

这里的 completeness 是相对于拟合后的 surrogate，而不是原始 LLM。原模型端仍需单独报告 reconstruction residual。平均分配也不是唯一可能的投影，而是满足对称性、线性和每条超边守恒的简单选择，必须通过 signed、absolute 和 singleton 投影消融验证其归因价值。

---

### 4.6 Hierarchy-Free Interaction Selection

这一节说明设计原则，而不是专门批评 ProxySPEX。

定义：方法允许一个高阶超边被选中，即使其所有低阶子集没有被选中。

例如：

\[
\theta_{{i,j}}\neq0,
\qquad
\theta_{{i}}=0.
\]

说明原因：

* 独立效应较弱不代表组合效应不存在；
* 强制层级可能排除孤立但有效的非加性。

但这是待实验验证的结构选择，而不是预先宣称层级一定错误。如果该设计在最终实验中贡献很弱，可以将本节合并进 4.4，避免把次要贡献写得过重。

---

## 5. Experimental Setup

### 5.1 Research Questions

先只列四个问题：

* **RQ1：** 方法是否具有竞争力的归因 faithfulness？
* **RQ2：** 在相同 observations 和模型阶数下，Möbius 与 Fourier 坐标的有限查询恢复、稀疏性和 held-out reconstruction 有何差异？
* **RQ3：** 恢复出的二阶 interactions 能否对应精确计算、可排序的 deletion non-additivity？
* **RQ4：** interaction modeling 与 signed projection 分别为最终归因带来多少收益？

强制 hierarchy 是否安全只作为结构诊断或附录问题，不预设它是主要效果来源。

---

### 5.2 Datasets and Models

说明：

* 模型：Qwen3-8B、LLaMa-3.1-8B-Instruct；
* 数据集（sst2、rtn、emotion）以及选择原因；
* 任务和 verbalizer；
* 样本选择；

---

### 5.3 Baselines

核心基线：

* ProxySPEX；
* additive deletion model；
* 相同 observations、相同最大阶数的直接 Fourier surrogate；
* 至少一个与 word-level deletion protocol 兼容的一阶归因方法。

---

### 5.4 Evaluation Metrics

以下只是按用途分组介绍所有可能指标，不一定都要在这里介绍；如果某个指标只在一个 RQ 中使用，可以只在对应 RQ 中介绍。

#### 词级归因

* AOPC；
* AUPC；
* comprehensiveness；
* sufficiency。

#### Surrogate 拟合

仅在 E2 需要时：

* held-out (R^2)；
* MAE；
* NRMSE。

#### 交互恢复

* 精确系数 MAE；
* 符号一致率；
* 排序相关性；
* selected edges 与随机词对的真实强度比较。

#### 成本

* 逻辑查询数；
* 唯一查询数；
* 模型 forward；

---

### 5.5 Implementation Details

说明：

* 最大阶数；
* 预算；
* 稀疏求解器；
* refit；
* 随机种子；
* 运行环境。

---

## 6. Results

### 6.1 Overall Attribution Performance

回答 RQ1。多数据集、多模型比较，讨论：

* deletion faithfulness；
* sufficiency；
* 查询成本；
* 不同任务上的差异。

---

### 6.2 Representation and Finite-Query Recovery

回答 RQ2。在相同训练 observations、相同最大阶数和相同 held-out masks 下比较：

* deletion-Möbius surrogate；
* direct Fourier surrogate。

报告：

* train 与 held-out \(R^2\)、NRMSE、MAE；
* nonzero support size；
* top-\(k\) refit compression curve；
* 不同 held-out mask distribution 下的稳定性。

这一节判断优势是否来自有限查询下的坐标和恢复差异，不把共同的低阶函数空间误写成表达能力差异。

---

### 6.3 Exact Interaction Quality

回答 RQ3。对主实验中的二阶边进行精确验证，因为二阶最适合大规模验证：

\[
m^-_{ij}
=
g(\{i,j\})-g(\{i\})-g(\{j\})+g(\varnothing).
\]

比较：

* 估计值和精确值；
* 方向；
* Möbius、ProxySPEX、随机排序与 oracle 排序的 Precision@k、NDCG@k 和 Spearman；
* top edges 与随机词对。

这里可以说明：方法定义支持更高阶，但大规模精确验证聚焦二阶，因为 (r) 阶验证需要 (2^r) 个组合。

---

### 6.4 Interaction Modeling and Projection

回答 RQ4，分开检验模型和投影：

* \(d=1\) additive 与 \(d=2\) interaction model；
* 固定 interaction model 后比较 signed、absolute 与 singleton projection。

分析：

* held-out reconstruction 变化；
* AOPC、AUPC 和 comprehensiveness/sufficiency 变化；
* signed interactions 是否确实进入并改善最终词级排名。

---

### 6.5 Structural Diagnostics and Robustness

作为次级分析报告：

* unrestricted、parent screening 与 strict hierarchy；
* sampler、refit、budget 和随机种子；
* 不同文本长度与 near-full held-out masks；
* 被 hierarchy 删除的超边数量、真实交互强度与案例。

结论应保持条件化：只有 gate 支持时，才讨论 hierarchy 对性能的影响，不将其预设为方法有效性的主要来源。

---

### 6.6 Additional Sensitivity

只保留必要内容：

* 最大阶数的小规模敏感性。

阶数敏感性可测试：

\[
d\in{1,2,3},
\]

但三阶只需在短文本子集上运行。

---

## 7. Qualitative Analysis

展示少量有解释价值的案例：

* 否定组合；
* 修饰关系；
* 冗余或协同词；
* 非层级边；
* 估计错误的失败案例。

每个案例最好同时展示：

* 一阶贡献；
* 超边及符号；
* 最终词分数；
* 精确删除验证。

---

## 8. Discussion

### Conclusion
回到一个简单观点：单独的特征贡献不足以描述模型预测。直接恢复稀疏、带符号的 deletion-Möbius 超图，可以显式表示并验证特征组合的非加性作用，并在不强制层级结构的情况下将这些作用用于特征归因。

### Limitations

承认一些限制，但同时要点出解决方法。可以从下面的限制中选择：

1. 交互取决于删除算子和 value function；
2. 主实验主要采用二阶；
3. 高阶候选数量和精确验证成本快速增长；
4. 删除后文本可能不自然；
5. 稀疏恢复依赖查询预算；
6. 方法只在文本分类上进行了验证。

---

# 四、论文主线

整篇文章只需要围绕下面的链条展开：

```text
模型预测依赖特征组合
        ↓
独立特征分数无法完整表示组合效应
        ↓
删除干预下的非加性可由 deletion-Möbius 超边直接刻画
        ↓
从有限模型查询中稀疏恢复低阶 signed hyperedges
        ↓
将超边贡献聚合到成员特征，形成词级归因
        ↓
验证交互是否改善归因、是否能被精确删除实验确认
        ↓
检验强制层级结构是否会排除有价值的交互
```

更适合作为论文中心句的版本是：

> **Faithful feature attribution requires not only identifying important features, but also recovering how features act together. We therefore directly learn a sparse signed deletion-Möbius hypergraph from black-box queries and use its interactions to construct feature-level explanations.**

核心主角应当是：

> **可恢复、可验证并能用于归因的 signed deletion interactions。**

各部分的角色是：

* deletion-Möbius：描述交互的数学语言；
* 稀疏估计：从有限查询中恢复交互的技术；
* signed projection：把交互用于词级解释的方法；
* hierarchy-free selection：允许孤立组合效应存在的结构设计；
