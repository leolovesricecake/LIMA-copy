## 总体评价

这份大纲已经具备一篇论文的基本骨架：问题动机清晰，数学定义完整，方法、研究问题和实验章节基本对应，而且对“方法支持一般阶数、实验主要采用二阶”的表述较为克制。核心叙事也比较集中，即从有限删除查询中直接恢复稀疏、带符号的 deletion-Möbius 超图，再将交互用于特征归因。

但以严格审稿标准来看，我目前会给出 **Weak Reject**。主要原因不是结构不好，而是以下三个根本问题尚未解决：

1. **“直接恢复 Möbius 系数”本身是否构成足够的方法创新，尚不清楚；**
2. **从 deletion-Möbius 系数到特征归因的数学语义存在潜在错误或至少缺少严格论证；**
3. **当前实验混合了函数拟合、交互恢复和词级归因三个不同目标，难以准确证明方法为何有效。**

这项工作有潜力，但需要从“一个组织良好的想法”进一步收敛为“一个可被验证的明确技术主张”。

---

# 一、值得保留的优点

## 1. 论文主线相对集中

当前大纲没有把 Möbius、稀疏恢复、层级结构、查询效率和词级归因都并列成独立贡献，而是将主角放在：

> 可恢复、可验证并能用于归因的 signed deletion interactions。

这个中心是合理的，也比单纯声称“我们发现了高阶交互”更有针对性。

## 2. 对一般阶数和二阶实例进行了区分

大纲明确说明方法形式支持最大阶数 (d)，但主实验采用 (d=2)，并承认高阶候选数量和验证成本快速增长。这种表述比泛泛宣称“支持任意高阶交互”更可信。

## 3. 研究问题与实验章节基本对齐

当前四个 RQ 分别对应：

* 整体归因效果；
* 是否需要交互；
* 交互是否真实；
* 层级约束是否安全。

至少从结构上看，实验不是简单堆表格，而是试图回答方法中的具体设计问题。

## 4. 有意识地区分 support selection 和 refitting

将稀疏选择与无偏重估分开是合理的。若实验证明 refitting 对系数准确性和稳定性有明显帮助，这可以成为方法实现中的有效设计。

---

# 二、最严重的问题

## 1. 当前特征归因公式的符号语义可能是反的

这是大纲中最需要优先修正的问题。

你们定义：

[
g(D)=f(N\setminus D),
]

其中 (D=\varnothing) 表示完整输入，(D=N) 表示所有特征均被删除。

Möbius 展开满足：

[
g(D)=\sum_{T\subseteq D}m^-(T).
]

因此：

[
g(N)-g(\varnothing)
=
\sum_{T\neq\varnothing}m^-(T).
]

当前大纲定义词级得分为：

[
a_i
=
\sum_{T\ni i}\frac{\widehat\theta_T}{|T|}.
]

于是：

[
\sum_i a_i
=
\sum_{T\neq\varnothing}\widehat\theta_T
\approx
g(N)-g(\varnothing).
]

但通常特征归因希望解释的是：

[
f(\text{full input})-f(\text{empty baseline})
=
g(\varnothing)-g(N).
]

因此，如果正归因表示某个词支持原始预测，那么理论上更自然的形式是：

[
a_i
=
-\sum_{T\ni i}\frac{\widehat\theta_T}{|T|}.
]

例如，删除一个支持目标类别的词通常会降低目标得分：

[
g({i})-g(\varnothing)<0.
]

其 deletion-Möbius singleton 是负数，但这个词对原预测的贡献应该被解释为正数。

这不只是符号记号问题，而会影响：

* signed interaction 的含义；
* 正负词归因的解释；
* 删除排序；
* 与其他归因方法的比较；
* completeness 性质。

论文必须明确区分：

> deletion effect 的方向

和

> feature contribution to the original prediction 的方向。

此外，平均分配 (\theta_T/|T|) 本身也需要论证。它是一种对称投影规则，但不是唯一规则。至少需要说明它满足哪些性质，例如：

* efficiency/completeness；
* symmetry；
* linearity；
* interaction contribution conservation。

否则审稿人会认为你们恢复了一个超图，却在最后用一个任意规则把它压回词分数。

---

## 2. “直接 Möbius 恢复”的方法创新仍然过弱

当前方法的核心步骤看起来是：

1. 构造最高 (d) 阶 Möbius 单项式字典；
2. 对删除查询建立设计矩阵；
3. 使用稀疏回归选择支持；
4. 在支持上 refit。

如果实现主要是 Lasso、Elastic Net、OMP 或其他标准稀疏回归，那么审稿人很可能概括为：

> The proposed method is essentially sparse polynomial regression in the deletion-Möbius basis.

此时仅仅强调“我们直接在 Möbius 坐标恢复，而不是先拟合 Fourier/GBT 再转换”，未必足以形成一篇强论文。

你们至少需要在以下方向中建立一个真正清晰的技术增量：

* 针对 deletion-Möbius 字典设计新的查询采样；
* 设计适合长文本的候选筛选或分阶段恢复算法；
* 给出有限查询下的支持恢复或误差界；
* 证明某类文本删除函数在 Möbius 坐标中更容易稀疏恢复；
* 在严格匹配查询预算的条件下，证明直接恢复显著优于间接恢复；
* 给出比简单平均分配更有原则的 interaction-to-feature 投影。

若这些都没有，当前贡献更像“一个合理的坐标选择和实现方案”，而不是充分独立的方法贡献。

---

## 3. 必须正面处理 Fourier 与 Möbius 的函数空间关系

在布尔掩码空间中，固定最高阶数时，低阶 Fourier 基与低阶 Möbius 单项式通常张成相同的函数空间。也就是说，你们不能将优势表述为：

> Möbius 能表达 Fourier 无法表达的交互。

真正需要比较的是：

* 哪种坐标中的系数更稀疏；
* 哪种表示在有限查询下更容易恢复；
* 哪种系数语义更贴近删除干预；
* 哪种表示对采样噪声更稳定；
* 哪种表示产生更好的最终归因。

因此，“直接恢复”不能只作为流程上的区别，而必须通过**可压缩性、恢复难度和归因效果**来证明价值。

当前大纲把可压缩性实验放在“完成后再决定是否加入”的位置，我认为不合适。稀疏低阶结构是整个方法成立的前提，不能作为可选实验。

---

## 4. 当前实验混淆了三个不同层次的问题

论文实际上包含三个彼此独立的映射：

### 第一层：表示能力

低阶稀疏 Möbius 模型能否逼近真实删除函数？

这是关于：

[
g(D)
\approx
\sum_{|T|\le d}\theta_T\mathbf 1[T\subseteq D].
]

需要考察的是截断误差和稀疏性。

### 第二层：有限查询恢复

在只有 (B) 个查询时，能否正确估计这些系数和支持？

这是关于估计误差、采样策略和稀疏求解器。

### 第三层：词级投影

即使超图恢复准确，将超边平均分配到成员之后，是否得到更好的词级排序？

这是关于：

[
{\theta_T}
\longrightarrow
{a_i}.
]

当前的 6.2、6.3 和 6.5 将这些问题交叉在一起。例如，(d=2) 比 (d=1) 的 surrogate 拟合更好，并不能自动证明 interaction-aware feature ranking 更好；而词级删除指标更好，也不能证明恢复的交互系数更准确。

建议全文显式区分三类误差：

[
\text{total error}
=
\text{representation error}
+
\text{recovery error}
+
\text{projection error}.
]

这会使论文的论证明显更严谨。

---

## 5. “精确有限差分验证”只能验证估计精度，不能证明交互有解释价值

对于二阶边：

[
m^-_{ij}
=
g({i,j})-g({i})-g({j})+g(\varnothing),
]

使用四次查询计算精确值，可以验证估计系数是否准确。这是必要的，但它实际上只是检查：

> 稀疏回归估计的数值是否接近 Möbius 系数的定义值。

它不能单独证明：

* 这个交互是语义上有意义的；
* 这个交互改善了特征归因；
* 这个交互比其他交互定义更忠实；
* 这个交互不是掩码造成的伪影。

因此，不建议把“我们通过精确有限差分验证交互”写成独立贡献。它更适合作为估计正确性的验证协议。

更强的证据应该包括：

1. 恢复系数与未参与训练的精确系数一致；
2. top edges 的精确强度显著高于随机边；
3. 使用这些边确实改善 unseen-mask prediction；
4. 使用这些边改善下游词级 faithfulness；
5. 在不同删除算子或扰动方式下具有一定稳定性。

---

## 6. hierarchy-free 目前被写得过重

允许：

[
\theta_{{i,j}}\neq0,
\qquad
\theta_{{i}}=0
]

本身是无层级稀疏选择的自然结果，但它是否足以成为论文的第四个核心 RQ，需要谨慎。

有两个风险。

第一，将恢复后的支持做 strong-hierarchy pruning，并不等价于一个从一开始就使用层级结构进行候选发现的方法。若用这种后处理消融攻击 ProxySPEX，比较可能不公平。

第二，即使发现 orphan interactions，仍需回答：

* 它们在所有恢复边中占多少比例？
* 它们是否稳定跨随机种子出现？
* 它们的精确系数是否显著？
* 删除它们是否真的损害归因？
* 它们是否只是 singleton 因正则化而未被选中？

建议把 hierarchy-free 降级为：

> 一项结构分析或附加设计研究。

只有在实验发现大量稳定、强且有实际作用的 orphan edges 时，再将其升级为核心贡献。

---

## 7. 删除算子既用于建模又用于评估，可能导致度量偏置

方法根据 deletion queries 拟合，主要指标又是 AOPC、comprehensiveness 等删除指标。审稿人可能质疑：

> The method is optimized on deletion behavior and evaluated using essentially the same deletion mechanism.

这不一定构成错误，但会使结果天然偏向你们的方法。

建议至少增加以下控制：

* 训练查询与测试查询使用不同 mask 分布；
* 在未见过的删除规模上评估；
* 同时报告 insertion 和 deletion；
* 使用另一种掩码替换方式进行迁移评估；
* 比较不同 mask token、空字符串、UNK 或语言模型填充；
* 报告解释在不同扰动算子下的稳定性。

否则你们可能只证明了：

> 一个 deletion-Möbius surrogate 很适合预测 deletion-based evaluation metric。

这比“更忠实的特征归因”弱得多。

---

## 8. 长文本上的可扩展性缺少真实方案

候选字典大小为：

[
|\mathcal C_d|
=

\sum_{r=1}^{d}\binom nr.
]

即使 (d=2)，也是：

[
n+\frac{n(n-1)}2.
]

当 (n=128) 时已经超过八千个候选；当 (n=512) 时约为十三万个候选。若查询预算只有几百或几千，设计矩阵会严重欠定，并且 Möbius 特征之间可能高度相关。

大纲目前只说“使用稀疏估计”，不足以回答：

* 如何控制候选数量？
* 是否先筛 singleton 再构造 pair？
* 如果这样做，是否又引入父项依赖？
* 查询分布如何保证 pair 特征被充分激活？
* 不同删除比例下设计矩阵条件数如何？
* 稀疏恢复是否稳定？
* 时间和内存开销如何随 (n) 增长？

如果没有新的候选筛选机制或恢复理论，方法可能只能适用于短文本。

---

# 三、实验设计中需要补强的部分

## 1. 增加“结构前提审计”

建议将它放在所有主实验之前。

在短文本或截断后的小规模样本上完整枚举 (2^n) 个删除集合，获得精确 value table，比较：

* Möbius 系数按绝对值排序后的累计能量；
* Fourier 系数的累计能量；
* 不同最大阶数 (d) 的精确重建误差；
* 保留相同数量系数时的压缩误差；
* singleton、pair、triple 的能量分布；
* orphan interactions 的真实比例。

它回答的是最基础的问题：

> 文本模型的局部删除函数是否确实具有低阶、稀疏或可压缩的 Möbius 结构？

若答案是否定的，后面的稀疏恢复缺乏基础。

---

## 2. 增加可控的合成交互实验

真实模型没有完整、无争议的“交互 ground truth”。建议构造具有已知支持和系数的 set functions，控制：

* 稀疏度；
* 阶数；
* 系数正负；
* 是否满足 hierarchy；
* 查询噪声；
* 查询预算。

比较支持 Precision、Recall、F1 和 coefficient error。

这能独立验证恢复算法，而不会受到语言模型、掩码不自然和解释评价指标的干扰。

---

## 3. 查询预算必须严格匹配

对所有黑盒方法，应固定：

[
\text{number of unique original-model evaluations}.
]

不要仅比较“逻辑查询数”，也不要把 surrogate 内部调用与原模型 forward 混为一谈。

至少报告：

* 唯一原模型输入数；
* 总 forward 数；
* 是否批处理；
* surrogate 训练时间；
* 总运行时间；
* 峰值内存。

若 ProxySPEX 使用原模型生成训练数据、再在代理模型上进行大量计算，应将这两部分分开报告。

---

## 4. baseline 需要更加完整

“ProxySPEX + additive deletion model + 一阶归因方法”可能不足以支持强结论。

基线至少应覆盖四类：

1. **同一 Möbius 字典、不同恢复方法**
   用于证明贡献不只是标准 Lasso。

2. **相同查询预算的一阶局部代理方法**
   用于证明 interaction 的价值。

3. **其他 interaction attribution 方法**
   用于证明不是只有 ProxySPEX 一个对手。

4. **oracle 或 exact upper bound**
   在小规模枚举数据上，展示完美系数或最佳低阶近似能达到什么程度。

尤其需要设置：

* same samples + (d=1)；
* same samples + (d=2)；
* same (d=2) dictionary + alternative sparse solver；
* exact low-order projection；
* exact full Möbius attribution，若小 (n) 可行。

---

## 5. 增加稳定性实验

稀疏支持恢复通常对采样和正则化敏感。建议报告：

* 不同随机种子的 top-edge Jaccard；
* 系数符号一致率；
* 特征排名相关性；
* 不同预算下支持稳定性；
* 不同正则化参数下的结果；
* refit 前后的稳定性。

单次运行得到漂亮超边不足以说明解释可靠。

---

# 四、建议重新组织研究问题

当前 RQ1–RQ4 的逻辑尚未直接覆盖方法成立的前提。建议改为：

### RQ1：删除函数是否具有适合直接恢复的低阶稀疏 Möbius 结构？

回答表示是否合理，而不是直接进入最终归因效果。

### RQ2：在相同查询预算下，直接恢复能否准确重建未见删除行为和交互系数？

回答有限查询恢复问题。

### RQ3：恢复的 signed interactions 是否能改善特征级归因？

回答从超图到词级解释是否真的有用。

### RQ4：方法在预算、文本长度、采样方式和删除算子变化下是否稳定？

回答实用性和鲁棒性。

将 hierarchy 分析放在 RQ2 或 RQ4 的子实验中：

> What types of interactions are missed by hierarchical support constraints?

这样即使 hierarchy 实验结果不强，论文主线也不会受损。

---

# 五、建议的结果章节

## 6.1 Exact Structural Audit

* Möbius 可压缩性；
* 低阶截断误差；
* 与 Fourier 表示的公平比较；
* 高阶能量和 orphan edge 分布。

## 6.2 Controlled Recovery Accuracy

* 合成函数支持恢复；
* 精确枚举样本上的系数误差；
* 查询预算曲线；
* 不同求解器和采样器。

## 6.3 Held-out Mask Prediction

* 未见 mask 上的 (R^2)、MAE、NRMSE；
* 不同删除规模；
* 不同文本长度；
* 与 ProxySPEX 和 additive surrogate 比较。

## 6.4 Interaction-aware Feature Attribution

* singleton-only；
* signed projection；
* absolute projection；
* 不同交互分配规则；
* deletion、insertion、sufficiency、comprehensiveness。

## 6.5 Structural and Robustness Analysis

* hierarchy/orphan edges；
* sampler；
* refit；
* budget；
* deletion operator；
* 随机种子稳定性。

这样每一节分别对应：

[
\text{表示}
\rightarrow
\text{恢复}
\rightarrow
\text{函数预测}
\rightarrow
\text{词级归因}
\rightarrow
\text{鲁棒性}.
]

---

# 六、对各章节的具体修改建议

## Introduction

当前五段式引入基本合理，但不建议过早突出 hierarchy。更强的矛盾应是：

> 现有方法要么只产生独立特征分数，要么通过间接代理恢复交互；然而我们尚不清楚能否在有限黑盒查询下直接、稳定地恢复具有明确删除语义的交互，并将其可靠地转化为特征归因。

贡献建议改成：

1. 提出直接稀疏恢复 deletion-Möbius interactions 的框架；
2. 系统区分并验证表示误差、恢复误差与归因投影误差；
3. 在匹配查询预算下验证交互恢复和 interaction-aware attribution。

“精确有限差分验证”不宜单独列为贡献。

## Related Work

“Sparse Set-Function Approximation”可以保留，但应服务于两个问题：

* 现有稀疏恢复方法为何不能直接解决你们的场景；
* 你们在算法、采样或解释投影上增加了什么。

如果只是介绍稀疏回归背景，可能反而让审稿人更容易判断方法是标准技术组合。

## Problem Formulation

必须明确：

* (f) 是概率、logit、log-probability 还是 margin；
* 解释固定 gold class、原预测 class，还是每次删除后的预测 class；
* 删除后目标类别是否保持不变；
* baseline 输入是什么；
* 空输入如何构造；
* mask token 是否被模型训练过；
* sign 的语义；
* feature contribution 与 deletion effect 的方向差异；
* 归因是否满足 completeness，以及是精确还是近似 completeness。

## Method

4.6 不建议单独成节，除非你们设计了真正的 hierarchy-free 优化方法。若只是没有施加层级约束，可在 4.4 中一句说明。

4.5 应扩展为“Hypergraph-to-Feature Projection”，系统讨论：

[
a_i=-\sum_{T\ni i}w_{i,T}\theta_T,
\qquad
\sum_{i\in T}w_{i,T}=1.
]

等分只是：

[
w_{i,T}=\frac1{|T|}.
]

最好解释为什么选择它，以及其他分配方式是否影响结果。

## Discussion 与 Conclusion

当前在 Discussion 内放 Conclusion 不够规范。建议拆分为：

* 8 Discussion；
* 9 Limitations；
* 10 Conclusion。

---

# 七、我作为审稿人最可能提出的问题

1. **除了将标准稀疏回归应用于 Möbius 字典，新的算法贡献是什么？**
2. **为什么 Möbius 坐标比 Fourier、Harsanyi 或 Shapley interaction 更容易恢复？**
3. **稀疏低阶假设在真实语言模型上是否成立？**
4. **为什么交互系数平均分配给成员词是正确的？**
5. **当前归因公式为何没有负号？正系数究竟表示支持还是抑制原预测？**
6. **interaction recovery accuracy 与 feature attribution faithfulness 有什么理论或经验联系？**
7. **使用相同删除机制训练和评估是否导致偏置？**
8. **在 (B\ll |\mathcal C_d|) 时，支持为何可识别？**
9. **对长文本如何避免二阶候选爆炸？**
10. **所谓 orphan interaction 是否可能只是 singleton 被正则化压成零？**
11. **对 hierarchy 的后处理消融是否真实代表 ProxySPEX 的层级发现机制？**
12. **目标类别、输出尺度和删除算子改变后，interaction sign 是否稳定？**
13. **为何最终必须输出词级排名，而不是直接保留更丰富的超图解释？**
14. **若二阶模型提高 surrogate (R^2)，为什么这意味着用户得到的解释更忠实？**

这些问题如果不能在论文中提前回答，很可能成为拒稿理由。

---

# 八、建议收敛后的核心主张

当前最稳妥、也最容易被实验验证的中心主张可以改为：

> 在匹配的黑盒查询预算下，直接拟合低阶稀疏 deletion-Möbius 表示，能够比间接代理方法更准确地恢复可验证的删除交互；在采用具有完整性约束的 signed projection 后，这些交互进一步改善特征级归因。

这句话包含三个必须分别证明的命题：

[
\text{direct recovery}
\Rightarrow
\text{better interaction estimation},
]

[
\text{better interaction estimation}
\Rightarrow
\text{better unseen-mask approximation},
]

[
\text{interaction-aware projection}
\Rightarrow
\text{better feature attribution}.
]

只要其中任意一步没有证据，论文主张都应相应缩小。

---

## 最终判断

这份大纲的**行文组织已经不错，但科学论证链条仍不够严密**。当前最优先的修改不是继续润色 Introduction 或增加更多指标，而是完成以下三件事：

1. 修正并严格论证 deletion interaction 到 feature attribution 的符号与完整性；
2. 用 exact audit 证明低阶稀疏 Möbius 假设，并与 Fourier 在同等条件下比较；
3. 将表示误差、有限查询恢复误差和词级投影误差拆开实验。

完成这三点后，论文才可能从“用 Möbius 稀疏回归做交互归因”提升为一项论证完整、贡献边界清楚的工作。
