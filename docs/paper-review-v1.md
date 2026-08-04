# 总体判断

新版不是对原大纲的推翻，而是一次**围绕审稿风险进行的收缩、纠错和重组**。

两版的核心技术路线没有变化：

> 删除查询 → 稀疏恢复 deletion-Möbius 超边 → 保留交互符号 → 聚合为词级归因。

原版更像是在提出一个完整的“interaction-aware attribution framework”，重点突出直接恢复、signed interactions 和 hierarchy-free selection。
新版则将其改造成三个需要分别证明的技术命题：函数恢复、交互恢复、归因效用，并修正了投影方向、Fourier 对比、层级主张和统计协议等问题。

**结论上，新版明显更合理、更严谨，也更容易被审稿人接受。**
但它主要修复了“论证方式”和“主张边界”，尚未完全解决最根本的**方法新颖性**问题。因此，我会把评价从原来的 **Weak Reject** 提升到：

> **Borderline / Weak Reject，取决于最终实验结果和算法增量。**

如果有限查询恢复结果足够强，并能证明优势不是普通稀疏回归、正则化或投影规则造成的，则有机会达到 Weak Accept。

---

# 一、新版与原版的关系

## 1. 核心思想保持不变

两版都认为单个词的重要性不足以完整描述模型行为，需要显式恢复组合效应；都采用 deletion-Möbius 表示，并从有限黑盒查询中稀疏恢复低阶交互。

因此，新版并不是另一篇论文，而是对原论文的**证据链重构**。

可以将两版关系概括为：

[
\text{原版：提出一个方法框架}
]

[
\Downarrow
]

[
\text{新版：把方法框架拆成三个可证伪命题}
]

原版主要讲：

> 我们直接恢复 signed deletion interactions，并将其用于归因。

新版进一步追问：

1. 能否恢复 masked function？
2. 恢复的边是否对应真实 deletion non-additivity？
3. 这些边是否真的改善词级 ranking？

这是新版最重要的进步。

---

## 2. 新版保留了原版最有价值的部分

以下内容基本继承了原版：

* word-level players；
* deletion-based value function；
* 一般阶数定义、主实验采用 (d=2)；
* sparse support selection 与 refit；
* signed hypergraph；
* interaction-to-feature projection；
* 与 ProxySPEX 的比较；
* exact pair interaction verification；
* 多数据集、多模型实验；
* 查询成本统计。

因此，新版没有因为收缩主张而丢掉方法主体。

---

## 3. 新版弱化了原版风险较高的内容

原版将 hierarchy-free interaction selection 单独设为方法小节，并将“强 hierarchy 是否安全”作为核心 RQ。

新版明确写道：

> 不强制 hierarchy 只是 support policy，不单独宣称为算法贡献。

同时把 hierarchy 放到鲁棒性分析或附录。

这是正确调整。因为“不施加层级约束”本身通常不足以构成算法创新，而且简单的 hierarchy pruning 也未必能公平代表 ProxySPEX 的发现机制。

---

# 二、新版解决了哪些原版的关键问题

## 1. 修正了 projection 的符号方向

原版定义：

[
a_i
=
\sum_{T\ni i}
\frac{\widehat\theta_T}{|T|}.
]

但 (\theta_T) 描述的是删除后的变化。对于支持原预测的词，删除通常降低目标分数，因此 deletion coefficient 为负，而该词对完整预测的贡献应为正。

新版改成：

[
a_i
=
-\sum_{T\ni i}
\frac{\widehat\theta_T}{|T|},
]

并得到：

[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
]

这不仅修正了符号，还给出了 surrogate completeness。

这是实质性修复。原版如果不修改该问题，审稿人完全可能直接质疑全部 signed attribution 结果。

不过还需要进一步说明：

> completeness 证明了总贡献守恒，但并不能单独证明“平均分配给成员词”是最优 ranking 规则。

建议将等分规则与 Harsanyi dividend/Shapley allocation 联系起来：每个交互 dividend 平均分配给参与成员，是满足 symmetry、efficiency 和 linearity 的自然分配。这样它就不再只是经验规则。

---

## 2. 承认 Möbius 与 Fourier 的低阶表达空间相同

原版虽然没有直接宣称 Möbius 表达能力更强，但“直接恢复”与“另一种表示再转换”的叙事仍可能让人误解为二者在表示能力上有本质差异。

新版明确声明：

> 固定最大阶数时，两种坐标张成相同函数空间，论文比较的是有限查询恢复和下游语义，而非理论表达能力。

这是非常必要的。

由此，论文真正需要证明的优势被正确限定为：

* 稀疏性差异；
* 有限样本下的恢复差异；
* 条件数或稳定性差异；
* deletion semantics；
* 下游 attribution utility。

这比泛泛强调“直接”更可信。

---

## 3. 将三种误差明确分开

原版已经尝试将 surrogate 拟合和 interaction verification 分开，但整体实验仍容易让人产生以下错误推理：

[
R^2 \text{ 更高}
\Rightarrow
\text{交互更真实}
\Rightarrow
\text{词级解释更好}.
]

新版直接指出：

* 更好的 held-out (R^2) 不自动意味着更好的归因；
* 更好的 AOPC 不自动证明 interaction coefficients 更准确。

并将目标拆成：

1. Function reconstruction error；
2. Interaction recovery error；
3. Feature attribution/projection error。

这使论文的逻辑严谨度提高了一个层级。

---

## 4. 对 hierarchy 的态度更中立

原版仍有较强的潜在叙事：

> 层级偏置可能漏掉有价值的交互，因此我们采用 hierarchy-free selection。

虽然原版已经使用了较克制的措辞，但该问题仍占据一个完整 RQ 和方法小节。

新版不再预设 hierarchy 错误，只把它视为 support policy。这使文章避免了两个风险：

* 必须证明大量稳定且重要的 orphan interactions；
* 必须公平模拟 ProxySPEX 的层级发现机制。

如果后续结果确实发现 hierarchy 有害，再作为附加发现提升即可。

---

## 5. 加强了公平性和可复现性

新版新增了：

* 相同 observations 下比较 Möbius 与 Fourier；
* 匹配查询预算；
* 逐样本 paired differences；
* bootstrap 95% CI；
* Wilcoxon signed-rank test；
* effect size；
* common/unmatched sample counts；
* logical queries、physical values 和 forward calls 分账。

这些设计可以有效防止平均值被样本筛选、缓存或运行协议影响。

特别是逐样本配对统计，比只报告跨样本 mean/std 更有说服力。

---

## 6. 结论根据证据动态收缩

新版 Conclusion 预先规定：

* 恢复和归因均优：强调完整方法；
* 恢复相当但投影更优：强调 deletion semantics 和 projection；
* 归因相当但 interaction 更准确：收缩为 hypergraph recovery。

这是很成熟的研究规划。它表明论文主张服从实验，而不是让实验服务于预设故事。

---

# 三、新版仍然没有完全解决的问题

## 1. 最大问题仍是方法创新性

尽管新版的论证更严谨，但方法部分仍可能被审稿人概括为：

> Construct a low-degree monomial dictionary, run sparse regression, refit the selected coefficients, and equally distribute the recovered terms to their members.

也就是：

> 标准稀疏多项式回归 + deletion-Möbius 坐标 + Harsanyi-style projection。

当前大纲没有说明：

* 稀疏求解器是否是新的；
* 查询采样是否是新的；
* 候选筛选是否是新的；
* 是否有恢复保证；
* 是否利用了 deletion dictionary 的特殊结构；
* 为什么 `uniform_size` 优于通用采样；
* 为什么在严重欠定条件下 support 可恢复。

因此，新版提高的是**可信度**，不是**新颖性**。

最理想的补强方向是至少做出一个真正专门化设计，例如：

### 方向一：Möbius-aware query design

根据候选阶数和设计矩阵相关性，设计比 uniform sampling 更适合恢复低阶 deletion coefficients 的采样分布。

### 方向二：分阶段但非层级依赖的候选发现

在不要求父项显著的情况下，从全部 (O(n^2)) pair 中高效筛选候选。

### 方向三：有限查询恢复分析

给出在稀疏度、预算和设计条件下的 recovery error 或 support recovery 条件。

### 方向四：有原则的 projection

证明当前 signed projection 满足一组公理，或与 Shapley/Harsanyi allocation 建立严格对应关系。

至少需要其中一个，否则实验必须非常强，才能弥补方法本身的简单性。

---

## 2. 新版弱化了“结构前提审计”，这是一个退步

上一轮建议的一个关键实验是：

> 在小规模样本上完整枚举删除函数，检查真实函数在 Möbius 坐标中是否低阶、稀疏或可压缩。

新版 6.2 比较 Möbius 与 Fourier 的 held-out recovery 和 compression curve，但这仍然是在有限 observations 下拟合。它不能完全区分：

* 表示本身是否可压缩；
* 恢复算法是否有效；
* 查询样本是否足够。

也就是说，如果 Möbius 表现不好，我们不知道是：

1. 真实函数并不 Möbius-sparse；
2. 稀疏求解器没恢复出来；
3. 查询设计不好；
4. 最大阶数太低。

建议在 6.2 前增加一个小规模 exact structural audit：

* 对很短的样本完整枚举 (2^n)；
* 计算精确 Möbius 和 Fourier coefficients；
* 比较 top-(k) reconstruction curve；
* 比较 degree-(d) truncation error；
* 比较能量或绝对系数质量随阶数的分布。

这会直接回答：

> 采用低阶稀疏 Möbius 表示的前提是否真实存在？

这是论文方法成立的基础，不应只依赖有限样本拟合结果。

---

## 3. 6.3 中“完整枚举”的含义需要澄清

新版写：

> 在完整枚举的 short-text samples 上比较 Möbius、ProxySPEX、random、oracle。

这里至少可能有两种含义：

### 完整枚举全部删除集合

成本为：

[
2^n.
]

它可以获得完整 set function 和所有阶 Möbius coefficients。

### 精确计算全部二阶 pair

若只验证二阶边，仅需：

[
g(\varnothing),\quad
g({i}),\quad
g({i,j}),
]

总成本约为：

[
1+n+\binom n2,
]

不必枚举 (2^n)。

这两种实验回答不同问题，建议明确分开：

* **Exact full-table audit**：极短文本，检查结构和高阶截断；
* **Exact pair oracle**：较多普通长度样本，检查二阶恢复排名。

否则审稿人会不清楚 oracle 的来源，也会质疑可扩展性。

---

## 4. RQ2 的 Fourier 公平比较需要更严格的协议

“使用完全相同的 observations”只是必要条件，不是充分条件。

由于两种基之间存在可逆线性变换，有限样本下的差异可能来自：

* 列尺度不同；
* 设计矩阵标准化不同；
* 正则化参数不同；
* 稀疏惩罚不是坐标不变的；
* support size 不同；
* Fourier 到 Möbius 的转换引入稠密性；
* 选择的低阶定义是否严格对应。

建议明确采用至少两种公平约束：

### 固定稀疏度比较

让两种方法保留相同数量的非零系数，再比较 held-out error。

### 固定预测误差比较

达到相同 held-out error 时，比较 support size。

同时报告：

* 标准化前后的结果；
* 相同 solver；
* 相同正则化选择规则；
* condition number 或 feature coherence；
* 转换后 support 的稠密程度。

否则“谁更稀疏”很可能只是坐标依赖的正则化结果，而不是结构性发现。

---

## 5. projection 的 completeness 还不足以证明 ranking 合理

新版已经证明：

[
\sum_i a_i
=
\widehat g(\varnothing)-\widehat g(N).
]

但许多完全不同的分配规则都可以满足总和守恒。

例如对于二阶边 (\theta_{{i,j}})，可以分配为：

[
\frac12,\frac12,
]

也可以分配为：

[
0.8,0.2.
]

二者都可以满足 completeness。

因此需要进一步回答：

> 为什么交互应当等分给成员，而不是根据 singleton、边际贡献或上下文权重分配？

有两种处理方式。

第一种是理论化：

> 等分对应于 Harsanyi dividend 的 Shapley allocation，在 symmetry、efficiency 和 linearity 条件下具有自然性。

第二种是实验化：

比较：

* equal split；
* singleton-weighted split；
* leave-one-out marginal split；
* interaction only as edge，不投影。

然后说明等分不是任意选择，或者承认它只是一个简单、稳定的默认方案。

---

## 6. “interaction validity”一词仍稍强

精确计算：

[
m^-_{ij}
=
g({i,j})-g({i})-g({j})+g(\varnothing)
]

可以证明估计值是否对应 deletion-Möbius 定义，但它不能证明这个交互具有：

* 人类语义合理性；
* 因果意义；
* 对自然输入分布的真实性；
* 跨删除算子的稳定性。

因此建议将：

> Interaction validity

改成更精确的：

> Interaction recovery accuracy

或者：

> Exact deletion-interaction fidelity

定性案例和跨算子稳定性再负责讨论解释价值。

---

# 五、现在的研究问题是否更合理

整体上更合理，但建议进一步调整。

## 当前 RQ1

> 方法是否具有竞争力的归因 faithfulness 与查询成本？

合理，回答最终效果。

## 当前 RQ2

> 相同 observations 下 Möbius 与 Fourier 的恢复和稀疏性有何差异？

合理，而且避免了错误的表达能力主张。但应增加 exact structural comparison，否则混合了 representation 与 recovery。

## 当前 RQ3

> 恢复的二阶边是否对应真实、可排序的 deletion interactions？

基本合理，建议将“真实”改成“精确计算的”，避免因果或语义过度解释。

## 当前 RQ4

> interaction modeling 与 signed projection 分别带来多少收益？

非常必要。这正面修复了“拟合二阶”和“将二阶用于 ranking”被混淆的问题。

其中：

* (C) vs (A)：二阶是否改善整个 surrogate；
* (C) vs (B)：拟合出的二阶是否需要进入 ranking；
* (C) vs ABSOLUTE：符号是否必要。

这一组消融逻辑已经比较完整。

## 当前 RQ5

> 结论对 seed、任务、模型、长度和 held-out 分布是否稳定？

合理，但维度较多。建议将核心稳定性放正文：

* seed；
* budget；
* length；
* second model。

其余内容可进入附录，避免正文最后变成大量零散敏感性实验。

---

# 六、作为审稿人，我现在会如何评价

新版已经做到：

* 主张收缩；
* 数学方向修正；
* 三种目标分离；
* Fourier 关系澄清；
* hierarchy 降级；
* 比较协议增强；
* 统计与成本更完整；
* 结论强度服从实验。

因此，新版的主要问题不再是“论文逻辑错误”，而是：

> 这个正确、严谨的方案是否具有足够的新方法贡献，以及结果是否足够显著。

这是一种更健康的审稿状态。

我的倾向会变成：

> **Borderline / Weak Reject，具有明确提升空间。**

若最终出现以下证据，我会考虑 Weak Accept：

1. 在相同 observations、相同 support size 和相同 solver 下，Möbius 显著优于 Fourier 或 ProxySPEX；
2. 优势在 paired statistics 上稳定，而非只体现在平均值；
3. exact coefficient recovery 和 feature attribution 同时改善；
4. projection 消融证明 signed interactions 确实进入最终收益；
5. 方法包含至少一个非平凡的采样、恢复或投影设计；
6. 长度和预算实验表明方法并非只能用于极短文本。

---

# 七、建议的最终主线

新版主线已经接近可用，但可以再收紧为：

```text
文本预测包含非加性组合行为
        ↓
deletion-Möbius 为这种行为提供直接的干预语义
        ↓
问题不在表达能力，而在有限查询下能否稀疏、稳定地恢复
        ↓
比较 Möbius 与 Fourier 的真实可压缩性和有限查询恢复
        ↓
用精确 deletion coefficients 验证恢复结果
        ↓
通过满足 completeness 的 signed allocation 转换为词级归因
        ↓
分别验证函数拟合、交互准确性和最终 ranking
```

中心句建议进一步改成：

> **We study whether deletion interactions can be directly and reliably recovered from limited black-box queries, and whether their principled allocation improves feature-level attribution.**

这比直接宣称“我们提出了更忠实的归因方法”更稳健，因为它把论文的研究性问题表达得更清楚。

---

# 最终结论

新版比原版**实质性更好**，并非只是增加章节或实验，而是修复了以下核心问题：

* attribution 符号方向；
* completeness；
* Fourier 与 Möbius 的关系；
* representation、recovery 和 projection 的混淆；
* hierarchy 主张过重；
* 比较协议和统计不足；
* 结论强度预设过高。

因此，它已经从一个容易被指出概念漏洞的大纲，变成了一个**论证结构基本成立、主张相对克制的研究计划**。 

但距离“足够可接受”仍差两项：

1. **证明真实删除函数确实具有适合该方法的低阶可压缩结构；**
2. **明确说明除了标准稀疏回归之外，方法上的不可替代增量是什么。**

前者决定方法是否有基础，后者决定它是否值得成为一篇新论文。
