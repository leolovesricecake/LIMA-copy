整体上，我赞成把主线改成：

> **方法优先实现，直接验证查询效率和归因效果；少量 exact enumeration 用于校验和诊断。**

这比先建设一个大规模 structural audit 工程更高效，也更符合“先验证方法有没有研究价值”的节奏。但目前方案还不能直接推出“低阶稀疏 Möbius 假设优于 Fourier hierarchy”，因为其中混合了表示基底、候选生成、采样分布和回归器等多个变量。

我的评价是：

* 作为 **MVP 和研究可行性实验**：合理，建议推进；
* 作为 **最终论文方法和实验设计**：还需修正几个关键问题；
* 不应完全删除 exact audit，而应压缩成一个最小、针对性强的诊断模块。

---

# 一、这个调整为什么合理

原来的 exact audit 需要完整实现：

* 词级特征化与掩码；
* 完整 value table；
* Fourier/Möbius 精确变换；
* oracle 压缩曲线；
* hierarchy 指标；
* 有限查询恢复；
* 统计检验。

而最终方法本身也需要其中大部分基础设施。若先完成大规模审计，可能投入很多工程成本后才发现：

* Möbius 结构虽然存在，但有限查询下难以恢复；
* Möbius design 高度共线，LASSO 表现很差；
* 候选生成漏掉关键交互；
* ProxySPEX 在实际 faithfulness 上仍然更强。

所以先回答：

> “一个简单、诚实的 Sparse Möbius 方法，在相同查询预算下有没有竞争力？”

是更有效率的推进顺序。

ProxySPEX 本身也是通过随机掩码、GBT 代理和 Fourier 提取来获得解释，其核心实证指标仍然是 held-out faithfulness、特征识别和查询效率。直接进行同预算方法比较是合理的第一步。

---

# 二、当前方案最关键的理论问题

## 1. 回归得到的 (\theta_T) 不一定是真正的 Möbius 系数

精确 Möbius 系数定义为：

[
m(T)=\sum_{U\subseteq T}(-1)^{|T|-|U|}f(U).
]

如果完整使用所有 (2^n) 个基函数并精确拟合，那么模型中的系数就是 Möbius 系数。

但当前方法只使用：

* 最高阶数不超过 (d) 的候选；
* candidate cap 后的一部分候选；
* LASSO/ElasticNet 正则化；
* 不完整随机查询。

此时拟合得到的是：

[
\hat\theta
==========

\arg\min_\theta
\sum_b
\left[
f(S_b)-\beta_0-\sum_{T\in\mathcal C}
\theta_T\mathbf1{T\subseteq S_b}
\right]^2
+\lambda|\theta|_1.
]

它是一个**稀疏 Möbius 字典代理模型的回归系数**，不一定等于真实 Möbius transform：

* 被遗漏的高阶项会产生 omitted-variable bias；
* Möbius 列之间高度相关；
* LASSO 会收缩系数；
* 采样分布会影响最优投影；
* candidate cap 会改变每个已选系数的含义。

因此输出不能直接声称：

> “(\theta_T) 就是真实 Möbius interaction。”

更准确的命名是：

* `sparse Möbius surrogate coefficient`；
* `Möbius-basis interaction attribution`；
* 或 `estimated Harsanyi interaction under a truncated sparse model`。

在少量 exact samples 上，需要额外比较：

[
\hat\theta_T
\quad\text{vs.}\quad
m(T)
]

以确认这些代理系数是否至少近似真实 Möbius 系数。

---

## 2. Möbius design 的共线性很严重

对于

[
\phi_T(S)=\mathbf1{T\subseteq S},
]

如果 (T\subset T')，那么：

[
\phi_{T'}(S)=1\Longrightarrow\phi_T(S)=1.
]

所以低阶项和高阶项天然嵌套，设计矩阵可能高度共线。这会导致：

* LASSO 在相关候选之间任意选择；
* support 随随机种子变化；
* 系数符号和大小不稳定；
* prediction (R^2) 很高，但 hyperedge attribution 不可靠。

因此不能只报告 held-out (R^2)，还必须报告：

* support stability；
* coefficient sign stability；
* bootstrap selection frequency；
* condition number 或 Gram matrix coherence；
* 相同预算下不同 seed 的 top-(k) Jaccard。

建议加入：

1. LASSO 做 support selection；
2. 在所选 support 上无正则重新拟合或 ridge refit；
3. 可选使用 stability selection；
4. 对最终系数给 bootstrap 置信区间。

---

## 3. 列标准化不应简单固定为 (2^{-|T|/2})

在均匀 Bernoulli-(0.5) 掩码下：

[
\mathbb E[\phi_T^2]=2^{-|T|},
]

所以 (2^{-|T|/2}) 是未中心化列的理论 (L_2) 范数。

但模型包含 intercept，实际回归通常会中心化设计列。此时方差为：

[
\operatorname{Var}[\phi_T]
==========================

2^{-|T|}
\left(1-2^{-|T|}\right).
]

更重要的是，你的采样是混合分布，而不是纯 Bernoulli-(0.5)，因此理论尺度不再适用。

建议：

> 对实际训练 masks 构成的每一列做经验中心化和标准化。

即：

[
\tilde\phi_T
============

\frac{\phi_T-\bar\phi_T}
{\operatorname{std}(\phi_T)}.
]

若某列在当前采样中恒为零或恒为一，应删除并记录为不可识别候选。

---

# 三、候选交互生成是当前方案最大的算法风险

你希望摆脱 hierarchy，但当前候选机制可能引入其他强先验：

* adjacent pairs 假设局部性；
* phrase-local pairs 假设句法局部性；
* random candidates 召回率不可控；
* marginal correlation screening 可能漏掉纯交互。

## 1. 简单相关性筛选可能漏掉关键项

如果直接计算：

[
\operatorname{corr}
\left(
\mathbf1{T\subseteq S},y
\right),
]

相关性可能主要来自：

* (T) 的低阶子项；
* 与 (T) 高度共现的其他候选；
* mask cardinality；
* 高阶项之间的抵消。

尤其是先使用所有候选与原始 (y) 做一次单变量相关性排序，不能可靠区分纯交互和低阶混杂。

更稳妥的候选生成是**逐阶残差筛选**：

1. 先拟合一阶模型；
2. 用一阶模型残差筛选二阶项；
3. 加入二阶项重新拟合；
4. 用新的残差筛选三阶项。

但要注意，这仍然只是计算策略，不应强制高阶项必须有显著父项。即使某个三阶项的所有二阶子项未被选中，也允许它因残差相关性而进入候选。

---

## 2. 第一版尽量避免候选筛选

为了判断 Möbius 基底本身是否有效，第一阶段应尽量减少 candidate generator 的影响。

建议首版限制在：

### 二阶模型

[
d=2
]

并包含所有：

* (n) 个 singleton；
* (\binom n2) 个 pair。

例如：

* (n=64)：总候选约 2,080；
* (n=128)：总候选约 8,256。

这在稀疏矩阵和分块实现下仍然可行。

只有三阶模型再引入 candidate cap。否则若结果不好，无法判断是：

* Möbius 假设不成立；
* 还是候选生成漏掉了正确交互。

推荐开发顺序：

1. `degree=1`；
2. `degree=2, all candidates`；
3. `degree=3, all candidates`，只在较短文本上；
4. `degree=3, screened candidates`；
5. 后续再研究 adaptive/group-testing candidate discovery。

---

# 四、与 ProxySPEX 的比较需要区分两种协议

“所有方法使用相同 masks”看似最公平，但还不够。

## 协议 A：Controlled-query comparison

所有方法使用完全相同的：

* 训练 masks；
* validation masks；
* test masks；
* mask operator；
* value function；
* 查询预算。

比较：

* Sparse Möbius regression；
* low-degree Fourier regression；
* GBT surrogate；
* additive LASSO。

这个协议用于回答：

> 在相同观测数据上，哪一种表示和估计器更有效？

它最适合分析 Möbius 基底本身的价值。

## 协议 B：Native-method comparison

允许每种方法使用自身推荐的采样和训练机制，但限制：

* 总模型查询数相同；
* 目标 value function 相同；
* 最终 evaluation masks 相同。

比较：

* 完整 ProxySPEX；
* SparseMobiusAttribution；
* 可选 SPEX；
* additive baseline。

这个协议用于回答：

> 作为一个完整归因方法，谁在相同前向预算下更好？

两种协议都需要。否则：

* 强制 ProxySPEX 使用 Möbius 方法设计的混合 masks，可能不利于它；
* 只允许各自使用 native sampler，又无法判断提升来自基底还是采样。

---

# 五、必须确保比较的真的是 ProxySPEX

ProxySPEX 不只是“训练一个 GBT”。它还包括：

1. 随机掩码查询；
2. GBT 拟合；
3. 从树中提取 Fourier representation；
4. 保留 top-(k) Fourier coefficients；
5. 可选系数重新回归。

如果只比较：

[
\text{Sparse Möbius surrogate}
\quad\text{vs.}\quad
\text{GBT prediction},
]

那比较的是代理模型，不是最终可解释的 ProxySPEX 表示。

因此：

* `GBT surrogate` 可以作为 controlled baseline；
* `ProxySPEX` 应使用论文官方实现，或实现等价的 Fourier extraction、sparsification 和 refinement；
* 二者在结果表中必须分开。

若暂时无法复现完整 ProxySPEX，可以在第一阶段明确写：

> 当前比较为 Sparse Möbius 与 ProxySPEX 的 GBT surrogate component，不构成完整 ProxySPEX 对比。

---

# 六、查询预算要如何计算

所有会调用被解释模型的 masks 都必须计入 attribution budget，包括：

* pilot masks；
* candidate-screening masks；
* training masks；
* validation masks；
* adaptive rounds 中新增的 masks；
* full/empty masks。

不计入方法预算的只有：

> 所有方法共享、仅用于论文离线评价的固定 test masks。

但必须单独报告 test evaluation 的前向数量。

推荐记录：

```text
training_queries
candidate_discovery_queries
validation_queries
adaptive_queries
total_attribution_queries
evaluation_only_queries
unique_model_queries
cache_hits
```

如果同一个 mask 被重复请求，只计一次真实前向，但要记录 cache hit。

超参数选择如果只在已查询训练数据上做交叉验证，不增加前向；如果另外查询 validation masks，则必须计入预算。

---

# 七、采样混合方案合理，但不能一次做得太复杂

你提出的：

* empty/full；
* Bernoulli (p=0.5)；
* near-full；
* fixed-cardinality；
* adaptive masks；

方向合理。但如果第一版同时引入全部机制，结果很难解释。

建议分三步：

## Sampling-1：Uniform

* empty/full 强制加入；
* 其余为 Bernoulli-(0.5)。

用于和 ProxySPEX 的基础设置对齐。

## Sampling-2：Static mixture

例如：

* 50% Bernoulli-(0.5)；
* 30% near-full；
* 20% fixed-cardinality；
* empty/full 强制加入。

用于测试更贴近实际归因目标的采样。

## Sampling-3：Adaptive

在 pilot 后根据：

* 当前预测方差；
* residual；
* leverage；
* 候选可识别性；

添加 masks。

Adaptive 采样应是后续贡献，不要阻塞首个可运行版本。

---

# 八、指标设计需要调整

你列出的指标总体合理，但要分清楚它们回答什么问题。

## 1. 函数逼近指标

必须保留：

* uniform held-out (R^2)；
* near-full (R^2)；
* normalized RMSE；
* query-to-(R^2) curve；
* AUC over normalized query budget。

这是最直接、最可信的比较。

## 2. 删除类指标

包括：

* comprehensiveness；
* sufficiency；
* MoRF；
* LeRF。

这些指标可以保留，但它们也依赖 mask operator，并可能受分布外输入影响。不能用它们单独证明 attribution 正确。

建议至少加入：

* random-ranking control；
* additive-attribution control；
* 相同删除数量；
* 多个删除比例；
* 结果按删除步数画完整曲线，而非只报告一个点。

## 3. 交互特有指标

只报告 top interaction stability 还不够。建议增加：

### Hyperedge ablation effect

对一个估计交互 (T)，比较同时移除 (T) 与分别移除其元素的非加性差异：

[
\Delta_T(A)
===========

\sum_{U\subseteq T}
(-1)^{|T|-|U|}
f(A\cup U),
]

其中 (A) 是固定背景集合。

这能验证 top hyperedge 是否真的对应局部非加性，而不是回归共线性造成的伪交互。

由于完整计算一个 (d) 阶交互需要 (2^d) 次查询，建议只验证 top-5 或 top-10 hyperedges，并将这些额外调用作为单独的 `interaction_verification_budget` 报告。

### Stability

至少报告：

* top-(k) Jaccard；
* rank correlation；
* sign consistency；
* selection frequency；
* 多 seed coefficient variation。

---

# 九、单点 ranking 的分摊公式需要谨慎

你提出：

[
\operatorname{score}_i
======================

\theta_i+
\sum_{T\ni i,\ |T|>1}
\frac{\theta_T}{|T|}.
]

这是把每个 hyperedge 均分给其成员，形式上接近对 Harsanyi dividend 的平均分配。

但如果 (\theta_T) 只是截断回归系数，而不是真实 Möbius dividend，这个分数不一定等价于严格的 Shapley value。

建议命名为：

```text
equal-share node score
```

而不是直接叫 Shapley value。

可同时提供：

* `main_effect = θ_i`；
* `interaction_share`；
* `total_equal_share_score`；
* 原始 top hyperedges。

交互解释应该是主输出，node ranking 只是辅助输出。

---

# 十、最小 exact sanity check 不能完全取消

大规模 audit 可以取消，但至少应保留三类 exact 检查。

## 1. Synthetic exact

必须覆盖：

* additive；
* sparse hierarchical Möbius；
* sparse non-hierarchical Möbius；
* sparse Fourier peak；
* dense low-degree；
* higher-order omitted-variable case。

用于验证：

* design matrix；
* coefficient recovery；
* support recovery；
* LASSO bias；
* candidate screening是否漏项。

## 2. 真实短文本 exact

建议只做：

* 每个任务 5–10 个；
* (n\leq10) 或 (12)；
* 全部 (2^n) 枚举。

用途不是做主要统计结论，而是回答：

* Sparse Möbius 的系数是否接近 exact Möbius；
* 高 (R^2) 是否伴随正确 support；
* 失败是表示错误还是估计错误；
* ProxySPEX 和 Möbius 分别漏掉什么。

## 3. Top-hyperedge targeted verification

在正常长文本中，不完整枚举全局谱，只对输出的少数 top hyperedges 做 (2^{|T|}) 局部验证。

这样以很低成本确认输出交互是否真实存在。

---

# 十一、我建议的修订版实施路径

## Phase 0：公共基础设施

实现：

* lexical word players；
* mask operator；
* deterministic value function；
* cache；
* query accounting；
* shared evaluation masks。

## Phase 1：无候选偏差的二阶 Sparse Möbius

只实现：

* degree 1；
* degree 2 全候选；
* LASSO / ElasticNet；
* selection 后 ridge/OLS refit；
* empirical column standardization。

比较：

* additive；
* Fourier degree-2 LASSO；
* GBT surrogate。

这一阶段先不做 adaptive candidate generation。

### 成功标准

在至少一个真实任务上，Sparse Möbius 在相同预算下满足：

* uniform 或 near-full (R^2) 显著优于 additive；
* 与 GBT 相当或更好；
* support stability 可接受；
* top hyperedges 的 targeted verification 显示真实非加性。

如果二阶都没有增益，不应立即开发复杂三阶方法。

## Phase 2：完整 ProxySPEX 对比

加入：

* 官方或等价 ProxySPEX；
* 相同 masks controlled comparison；
* native sampling comparison；
* wall-clock 与实际 query count。

## Phase 3：三阶候选生成

只在 Phase 1/2 有正面结果后加入：

* residual screening；
* candidate cap；
* streaming triple correlation；
* stability selection；
* targeted verification。

同时保留短文本 all-triples 对照，测量 candidate recall。

## Phase 4：主动查询

最后才研究：

* leverage-score masks；
* uncertainty sampling；
* adaptive group testing；
* batch-aware adaptive selection。

---

# 十二、最终评价

你的新工作流比“大规模 exact audit 先行”更高效，我建议采用，但要把研究主张分为两个阶段。

第一阶段只主张：

> **低阶稀疏 Möbius 字典是否能构成一个查询高效、faithful、可输出非层级 hyperedges 的代理归因方法。**

在这一阶段，不要过早声称：

* 恢复了真实 Möbius coefficients；
* 证明了 Möbius 结构比 Fourier hierarchy 更普遍；
* top hyperedges 就是真实模型机制。

第二阶段在少量 exact 和 targeted verification 支持后，才可以进一步主张：

> **Möbius 表示确实解释了 ProxySPEX 难以捕获的非层级交互。**

因此，建议把方案改成：

[
\boxed{
\text{方法优先}
+
\text{同数据基底对照}
+
\text{少量 exact 校准}
+
\text{top-hyperedge 定向验证}
}
]

而不是简单地：

[
\text{实现 Möbius LASSO}
\rightarrow
\text{看 held-out }R^2
\rightarrow
\text{宣布低阶稀疏 Möbius 假设成立}.
]

作为研究推进策略，你的调整是正确的；作为最终实验逻辑，需要上述约束才能避免把“一个回归器表现更好”误解释成“底层交互结构假设更正确”。
