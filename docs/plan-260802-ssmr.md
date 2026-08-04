可以把 support selection 完全模块化。当前工作流中，OMP 只是“从 Möbius 字典中选 support”的一个组件，后续 coefficient refit、hypergraph 构造和 Shapley allocation 并不依赖 support 一定由 plain OMP 产生。

不过，进一步研究后，我建议把原来比较模糊的 SSMR 收紧为一个更明确的方案：

> **SSMR = complementary-pairs stability selection for support + statistically certified sign aggregation + sign-constrained full-data refit。**

它不是简单地“重复 OMP 然后投票”，而是一套有阈值原则、误选控制和适用条件的 support recovery 模块。

---

# 一、先明确模块化工作流

## 1. 统一接口

整个恢复过程拆成三层：

[
\text{Query data}
\rightarrow
\boxed{\text{Support selector}}
\rightarrow
\text{Coefficient refitter}
\rightarrow
\text{Attribution projector}.
]

其中 support selector 可以选择：

### Plain OMP

[
(\Phi,\mathbf g)
\overset{\mathrm{OMP}}{\longrightarrow}
\widehat{\mathcal H}_{\mathrm{OMP}}.
]

### SSMR

[
(\Phi,\mathbf g)
\overset{\mathrm{repeated\ OMP}}{\longrightarrow}
{\widehat{\mathcal H}^{(r)},\widehat\theta^{(r)}}*{r=1}^{2R}
\overset{\mathrm{stability}}{\longrightarrow}
\widehat{\mathcal H}*{\mathrm{SSMR}}.
]

二者最后都进入相同的 refit 和 attribution：

[
\widehat{\mathcal H}
\rightarrow
\widehat{\boldsymbol\theta}
\rightarrow
a_i=-\sum_{T\ni i}\frac{\widehat\theta_T}{|T|}.
]

这样可以保证：

* 查询 masks 完全相同；
* LLM 输出完全相同；
* Möbius 字典完全相同；
* 唯一变化是 support selection；
* 可以公平判断 SSMR 是否真的优于 plain OMP。

---

# 二、为什么不能只做普通重复抽样

最初设想是：

> 随机取 70% observations，重复运行 OMP，按出现频率筛选。

这个版本能运行，但存在两个问题：

1. 70% 是任意比例；
2. 选择频率阈值 0.6、0.7 也是任意的。

更合理的是使用 **Complementary Pairs Stability Selection，CPSS**：

1. 把 observations 随机分成两个不相交的半样本；
2. 分别运行 OMP；
3. 重复生成 (R) 对互补半样本；
4. 共得到 (2R) 次 OMP 结果。

CPSS 相比普通随机子采样的优势是：它对有限数量的互补分割提供误选控制，并且其基本界不要求对 base selector 作特殊假设；原论文建议将 50 对互补样本作为合理默认值。([统计实验室][1])

因此建议默认：

[
R=50
]

即每个样本运行 100 次半样本 OMP。这里不增加任何 LLM 查询，只增加 CPU 上的稀疏回归计算。

---

# 三、正式定义 SSMR

假设已有 (B) 条模型查询：

[
\mathcal D_B={(D_b,g(D_b))}_{b=1}^{B},
]

以及 Möbius 设计矩阵：

[
\Phi_{b,T}=\mathbf 1[T\subseteq D_b].
]

候选 atom 数量为：

[
p=\sum_{r=1}^{d}\binom nr.
]

主实验 (d=2) 时：

[
p=n+\binom n2.
]

## 1. 互补半样本恢复

对于第 (r) 个随机划分：

[
I_r^{(1)}\cap I_r^{(2)}=\varnothing,
\qquad
|I_r^{(1)}|=|I_r^{(2)}|\approx B/2.
]

分别运行相同的 base OMP：

[
\widehat{\mathcal H}_r^{(1)},
\widehat{\mathcal H}_r^{(2)}.
]

每次 OMP 选择固定数量 (q) 个 atoms。为了隔离 aggregation 的作用，plain OMP 和 SSMR 的 base OMP 应使用相同的 (q)。

empty/full anchors 建议作为固定约束加入每次拟合，不参加随机划分。

## 2. 带符号的选择变量

对每个候选 interaction (T)，定义：

[
Z_T^{(r,j)}
=
\begin{cases}
+1,&T\in\widehat{\mathcal H}_r^{(j)}
\text{ 且 }\widehat\theta_T^{(r,j)}>0,\
-1,&T\in\widehat{\mathcal H}_r^{(j)}
\text{ 且 }\widehat\theta_T^{(r,j)}<0,\
0,&T\notin\widehat{\mathcal H}_r^{(j)}.
\end{cases}
]

由此得到三个总体概率：

[
\pi_T^+
=
P(Z_T=+1),
]

[
\pi_T^-
=
P(Z_T=-1),
]

[
\pi_T
=
\pi_T^++\pi_T^-.
]

其中：

* (\pi_T)：interaction 被选择的概率；
* (\pi_T^+)：以正系数被选择的概率；
* (\pi_T^-)：以负系数被选择的概率。

---

# 四、符号稳定性应该怎样正式定义

不能只说“正负号大部分一致”。建议同时定义两个量。

## 1. 条件符号一致率

在 interaction 已被选择的条件下：

[
c_T
=
\frac{
\max(\pi_T^+,\pi_T^-)
}{
\pi_T^++\pi_T^-
}.
]

其取值为：

[
c_T\in[1/2,1].
]

解释：

* (c_T=1/2)：正负符号各占一半，没有稳定方向；
* (c_T=1)：每次出现时符号完全一致。

## 2. Signed stability margin

定义：

[
m_T
=
|\pi_T^+-\pi_T^-|.
]

两者关系为：

[
m_T
=
\pi_T(2c_T-1).
]

这两个量反映不同内容：

* (\pi_T) 高：经常被选择；
* (c_T) 高：一旦被选择，方向一致；
* (m_T) 高：同时兼具高选择概率和高方向一致性。

例如：

| (\pi_T) | (c_T) | 含义            |
| ------: | ----: | ------------- |
|     0.9 |  0.55 | 经常出现，但符号不稳定   |
|     0.2 |   1.0 | 偶尔出现，但出现时方向一致 |
|     0.9 |  0.95 | 经常出现且符号稳定     |
|     0.1 |   0.5 | 基本是噪声         |

因此不能只用 (c_T)，也不能只用 (\pi_T)。

---

# 五、support 阈值不应手工选择

## 1. CPSS 的控制界

设 base OMP 在一个半样本上平均选择 (q) 个 atoms。稳定性选择阈值为：

[
\tau_{\mathrm{sel}}>1/2.
]

经典 stability selection 在额外交换性假设下，对错误选择数量给出：

[
\mathbb E[V]
\le
\frac{q^2}{p(2\tau_{\mathrm{sel}}-1)}.
]

CPSS 可以在不要求 base selector 满足这些强假设的情况下，控制低选择概率变量进入最终集合的数量；但此时严格来说，控制对象是“base procedure 选择概率较低的变量”，不一定直接等于真实 null variables。

因此，论文中应区分：

* **无额外假设：** 控制 low-stability atoms；
* **若所有假 interaction 都属于 low-stability 集合：** 可以解释为 false-interaction control。

## 2. 用目标误选数量反推阈值

预先指定允许保留的低稳定 atom 数量：

[
\nu>0.
]

要求：

[
\frac{q^2}
{p(2\tau_{\mathrm{sel}}-1)}
\leq \nu.
]

解得：

[
\boxed{
\tau_{\mathrm{sel}}
\geq
\frac12
\left(
1+\frac{q^2}{p\nu}
\right)
}
]

这就把原来任意的 0.6 或 0.7，变成一个可解释的阈值。

例如：

[
p=465,\qquad q=20,\qquad \nu=5,
]

则：

[
\tau_{\mathrm{sel}}
\geq
\frac12
\left(
1+\frac{400}{2325}
\right)
\approx0.586.
]

可以取最近的可表示频率：

[
\tau_{\mathrm{sel}}=0.59.
]

## 3. 不可行情形也有明确含义

如果：

[
\frac{q^2}{p\nu}>1,
]

那么没有任何 (\tau_{\mathrm{sel}}\leq1) 能满足要求。

这意味着：

* base OMP 每次选得太多；
* 候选空间太小；
* 或误选目标 (\nu) 设得过严。

这时应该减小 (q)，而不是把阈值强行设成 0.9。

## 4. 是否按阶数分别控制

singleton 与 pair 的数量差异很大：

[
p_1=n,
\qquad
p_2=\binom n2.
]

建议实现两种模式：

### 全局控制

所有 atoms 共享一个 (\tau_{\mathrm{sel}})。

优点是简单，适合作为第一版。

### Degree-wise 控制

分别统计：

[
q_1,q_2,
]

并指定：

[
\nu_1,\nu_2.
]

然后：

[
\tau_r
=
\frac12
\left(
1+\frac{q_r^2}{p_r\nu_r}
\right).
]

degree-wise 版本更符合 Möbius 字典的结构，但增加了一组错误预算分配。建议先实现全局版本，把 degree-wise 作为消融或后续增强。

---

# 六、sign 阈值不要再设一个任意常数

直接设：

[
c_T\geq0.8
]

仍然缺乏依据。

更合理的是把符号稳定性转化为统计检验。

## 1. 条件二项检验

假设 interaction (T) 在全部重复拟合中共被选择 (M_T) 次，其中：

[
N_T^+
]

次为正，

[
N_T^-
]

次为负。

定义：

[
K_T=\max(N_T^+,N_T^-).
]

检验：

[
H_0:c_T=\frac12
]

即正负方向没有稳定偏好。

在 (H_0) 下：

[
K_T
\sim
\operatorname{Binomial}
\left(M_T,\frac12\right)
]

的上尾分布。

双侧精确 (p)-value 可定义为：

[
p_T^{\mathrm{sign}}
=
2P
\left[
\operatorname{Bin}
\left(M_T,\frac12\right)
\geq K_T
\right],
]

并截断到 1。

## 2. 多重检验

只对已经通过 support threshold 的 atoms 做 sign test。

因为这些 tests 共享相同的 subsamples，彼此存在相关性，建议使用 **Holm correction**，而不是依赖独立性的普通 BH。

设：

[
\alpha_{\mathrm{sign}}=0.05.
]

只有 Holm-adjusted (p)-value 不超过 0.05 的 interaction 才被认为具有稳定方向。

这样就不需要人为设置：

[
\tau_{\mathrm{sign}}=0.8.
]

其原则变为：

> 只有当观察到的正负比例足以拒绝“方向随机”假设时，才输出带方向的 interaction。

## 3. 为什么 support 和 sign 必须分开判断

一个 interaction 可能：

* 经常被选择，但符号反复翻转；
* 不经常被选择，但偶尔出现时总是同一符号。

第一种不应该输出为 signed explanation；第二种也不应因为符号一致就进入 support。

因此最终 signed support 为：

[
\widehat{\mathcal H}_{\mathrm{SSMR}}
=
\left{
T:
\widehat\pi_T\geq\tau_{\mathrm{sel}},
\quad
p_{T,\mathrm{adj}}^{\mathrm{sign}}
\leq\alpha_{\mathrm{sign}}
\right}.
]

其稳定符号为：

[
\widehat s_T
=
\operatorname{sign}
\left(
N_T^+-N_T^-
\right).
]

---

# 七、最终 refit 不能忽略稳定符号

假设重复抽样认为某 interaction 稳定为正，但在全 observations 上做普通 OLS 后，它又变成负数，那么 sign stability 机制就失去了意义。

建议最终使用 **sign-constrained refit**：

[
\widehat{\boldsymbol\theta}
=
\arg\min_{\boldsymbol\theta}
\left|
\mathbf g-
\Phi_{\widehat{\mathcal H}_{\mathrm{SSMR}}}
\boldsymbol\theta
\right|_2^2
+
\lambda|\boldsymbol\theta|_2^2
]

满足：

[
\widehat s_T\theta_T\geq0,
\qquad
T\in\widehat{\mathcal H}_{\mathrm{SSMR}}.
]

其中 (\lambda) 可以很小，主要用于相关设计下的数值稳定性。

该问题是一个简单的带边界凸最小二乘问题。把列变换为：

[
\widetilde\Phi_T
=
\widehat s_T\Phi_T,
]

再要求新系数非负即可。

建议实现两个 refit 选项：

* `ols_refit`：不使用符号约束；
* `signed_refit`：使用稳定符号约束。

这样可以验证收益到底来自：

* support aggregation；
* 还是 sign constraint。

---

# 十、什么条件下 SSMR 明确优于 plain OMP

这是整个方案最关键的理论条件。

定义：

* 对任意真实 interaction：

[
\pi_T\geq\eta;
]

* 对任意假 interaction：

[
\pi_T\leq\theta;
]

并存在稳定性间隔：

[
\boxed{
\theta<\eta.
}
]

选择阈值满足：

[
\theta<\tau_{\mathrm{sel}}<\eta.
]

这就是 **stability gap condition**。

## 1. 假 interaction 被保留的概率

对一个假 interaction：

[
\pi_T\leq\theta.
]

使用 (R) 个独立互补对，经验选择频率满足：

[
P
\left(
\widehat\pi_T
\geq\tau_{\mathrm{sel}}
\right)
\leq
\exp
\left[
-2R
(\tau_{\mathrm{sel}}-\theta)^2
\right].
]

## 2. 真实 interaction 被漏掉的概率

对一个真实 interaction：

[
\pi_T\geq\eta.
]

则：

[
P
\left(
\widehat\pi_T
<
\tau_{\mathrm{sel}}
\right)
\leq
\exp
\left[
-2R
(\eta-\tau_{\mathrm{sel}})^2
\right].
]

因此，当存在稳定性间隔时，随着 (R) 增加：

* 假 interaction 保留概率指数下降；
* 真实 interaction 漏检概率指数下降。

## 3. 与单次 OMP 的明确比较

单次 OMP 对假 interaction 的选择概率是：

[
\pi_T\leq\theta.
]

SSMR 的上界低于 plain OMP 的 worst-case 上界 (\theta)，只需：

[
\exp
\left[
-2R(\tau_{\mathrm{sel}}-\theta)^2
\right]
<
\theta.
]

即：

[
R>
\frac{
\log(1/\theta)
}{
2(\tau_{\mathrm{sel}}-\theta)^2
}.
]

同理，SSMR 的真实 interaction 漏检上界低于 plain OMP 的 worst-case 漏检率 (1-\eta)，只需：

[
R>
\frac{
\log(1/(1-\eta))
}{
2(\eta-\tau_{\mathrm{sel}})^2
}.
]

所以可以正式写成：

> SSMR is provably preferable to a single OMP fit when true and null interactions are separated in their half-sample selection probabilities, and the number of complementary pairs is sufficiently large.

## 4. 它不保证普遍优于 OMP

若：

[
\theta\approx\eta,
]

不存在 stability gap，SSMR 不能区分真假 interaction。

尤其当多个高度相关的 atoms 可以相互替代时：

* 每次 OMP 选其中一个；
* 每个单独 atom 的频率都不高；
* stability selection 可能把它们全部删除。

这也是稳定性选择在高度相关代理变量下的已知失败模式。([arXiv][2])

因此不能声称：

> SSMR 解决了 Möbius dictionary 的 coherence。

更准确的是：

> SSMR filters interactions whose selection and direction are not reproducible under observation perturbations; it does not eliminate fundamental non-identifiability among near-equivalent atoms.

---

# 十一、Möbius dictionary 为什么特别容易不稳定

近期 sparse Möbius transform 工作也明确指出，AND/Möbius basis 的 basis vectors 是 coherent 的，因此标准 compressed-sensing 恢复条件难以直接适用。([arXiv][3])

对于你们的 sampling 方式，可以得到更具体的分析。

## 1. Möbius atom

定义：

[
\phi_T(D)=\mathbf1[T\subseteq D].
]

令删除集合大小：

[
K=|D|.
]

在固定 (K=k) 时：

[
P(T\subseteq D\mid K=k)
=
\frac{\binom{k}{|T|}}{\binom n{|T|}}.
]

如果 cardinality-balanced sampling 在：

[
K\in{0,\ldots,n}
]

之间均匀采样，则：

[
P(T\subseteq D)
=
\frac1{|T|+1}.
]

这是由组合恒等式：

[
\sum_{k=r}^{n}\binom kr
=
\binom{n+1}{r+1}
]

直接得到的。

## 2. 两个 atoms 的总体相关性

设：

[
a=|T|,
\qquad
b=|U|,
\qquad
c=|T\cap U|.
]

则：

[
|T\cup U|=a+b-c.
]

标准化后的总体相关系数为：

[
\boxed{
\rho_{a,b,c}
=
\frac{
\frac{(a+1)(b+1)}
{a+b-c+1}
-1
}{
\sqrt{ab}
}
}
]

这是 cardinality-balanced deletion distribution 下的闭式表达。

## 3. degree-2 情况

由该公式可得：

| atom 类型               |                       相关性 |
| --------------------- | ------------------------: |
| 两个不同 singleton        |                     (1/3) |
| singleton 与包含它的 pair  |    (1/\sqrt2\approx0.707) |
| singleton 与不包含它的 pair | (1/(2\sqrt2)\approx0.354) |
| 两个共享一个词的 pair         |               (5/8=0.625) |
| 两个不相交 pair            |                     (0.4) |

因此，degree-2 Möbius dictionary 的总体 mutual coherence 至少达到：

[
\mu=\frac1{\sqrt2}\approx0.707.
]

经典 OMP 的统一充分恢复条件为：

[
\mu<\frac1{2s-1},
]

其中 (s) 是真实 support size。([arXiv][4])

代入：

[
\mu\approx0.707
]

后，最坏情况保证基本只覆盖：

[
s=1.
]

这并不意味着 support 大于 1 时 OMP 必然失败，而是说明：

> 标准 mutual-coherence worst-case guarantee 对你们的设计几乎没有约束力。

这正是进行稳定性诊断的理论理由。

## 4. cardinality-balanced sampling 的副作用

cardinality-balanced sampling 能保证不同删除规模都被覆盖，但它也引入了一个全局相关来源：

> 一个 mask 较大时，几乎所有 singleton 和 pair atoms 都更容易激活。

因此，即使两个 pairs 不共享词，它们的列也可能显著正相关。

这说明 SSMR 解决的是一个真实结构问题，而不是纯粹为了增加算法复杂度。

---

# 十二、针对 coherence 的分析应该怎样进入论文

建议不尝试证明 SSMR “消除了 coherence”，而是建立三步论证：

## 第一步：字典具有结构性 coherence

给出上面的闭式相关公式以及 degree-2 表格。

## 第二步：coherence 导致单次 support 不可靠

在实际查询设计矩阵上报告：

* empirical mutual coherence；
* selected-support condition number；
* 不同子样本的 support Jaccard；
* sign-flip rate。

OMP 的恢复保证本来就依赖字典 coherence、信号强度、噪声和 sparsity。

## 第三步：SSMR 只保留可重复证据

SSMR 不改变字典，而是将解释标准从：

> 在一次相关残差竞争中获胜

改为：

> 在多个 observation perturbations 下持续获胜，并保持同一方向。

这是更准确、更可信的创新定位。

---

# 十三、SSMR 的新颖性到底够不够

进一步检索后，需要降低之前对“符号稳定性本身”的新颖性预期。

## 1. 不能声称的内容

以下都不是新的：

* stability selection；
* stability selection 与 OMP 结合；
* 通过重复拟合观察 coefficient sign；
* 利用 sign flips 过滤 false positives。

经典 stability selection 工作已经讨论了 randomized OMP，并指出稳定性选择可以在弱于普通 OMP exact-recovery 条件的假设下获得一致性。

2026 年也已有 GLM variable-selection 工作利用重复子采样中的 coefficient sign stability 来过滤 false positives，并明确指出 sign-stability 原则本身并非该工作的原创。([Springer Link][5])

## 2. 可以主张的创新组合

更合理的贡献是：

> We adapt complementary-pairs stability selection to coherent deletion-Möbius dictionaries and augment it with statistically certified direction recovery and sign-constrained coefficient refitting.

具体创新点是组合后的问题特定设计：

1. 面向 deletion-Möbius hyperedges，而不是普通回归变量；
2. support recurrence 与 directional stability 分开定义；
3. support threshold 由目标 low-stability error budget 确定；
4. sign 由 exact binomial test + multiple-testing correction确定；
5. 最终 refit 强制遵循稳定方向；
6. 推导 cardinality-balanced Möbius dictionary 的闭式 coherence；
7. 给出 stability gap 下相对 plain OMP 的支持和符号恢复界；
8. 使用 exact deletion-interaction oracle 验证。

这可以构成**中等强度的方法创新**，但不是新的基础统计范式。

---

# 十四、推荐的最终算法

## SSMR：Signed Stable Möbius Recovery

输入：

* 设计矩阵 (\Phi)；
* 查询值 (\mathbf g)；
* base OMP support size (q)；
* complementary pairs 数量 (R)；
* low-stability error budget (\nu)；
* sign test level (\alpha_{\mathrm{sign}})。

步骤：

1. 使用完整训练设计计算 column centering/scaling 参数；
2. 对非 anchor observations 生成 (R) 个 complementary splits；
3. 在每个 half-sample 上运行相同的 (q)-step OMP；
4. 在每个 support 上做局部 OLS refit，记录 coefficient sign；
5. 计算每个 atom 的选择频率 (\widehat\pi_T)；
6. 根据
   [
   \tau_{\mathrm{sel}}
   =
   \frac12
   \left(
   1+\frac{q^2}{p\nu}
   \right)
   ]
   选择 stable support；
7. 对 stable atoms 运行 exact sign test；
8. 使用 Holm correction 保留方向显著的 atoms；
9. 根据多数票确定 (\widehat s_T)；
10. 在全部 observations 上执行 sign-constrained ridge/least-squares refit；
11. 输出 support、coefficients、selection frequencies、sign counts 和 diagnostics。

建议默认：

[
R=50,
\qquad
\alpha_{\mathrm{sign}}=0.05.
]

(\nu) 不建议直接固定为一个绝对值跨所有样本使用。可以设成候选数量的比例：

[
\nu=\max(1,\rho_{\mathrm{false}}p),
]

例如：

[
\rho_{\mathrm{false}}\in
{0.005,0.01,0.02},
]

再在 validation set 上选择一个全局值。

---

# 十五、必须设计的对比实验

## 1. 模块对比

相同 masks、相同 values、相同 base support size：

* Plain OMP；
* SSMR-support：只使用稳定 support，普通 OLS refit；
* SSMR-signed：稳定 support + sign test + sign-constrained refit。

这三个版本分别回答：

* 重复 support selection 是否有用；
* sign mechanism 是否额外有用。

## 2. 固定最终 support size 对照

SSMR 通常会产生比 plain OMP 更小的 support。为避免“只是更稀疏所以更好”的质疑，再加入：

* Stability-top-(q)：按 (\widehat\pi_T) 排序，保留前 (q) 个；
* Plain OMP：固定 (q) 个。

这样可以判断收益是否来自更好的排序，而非只来自更强裁剪。

## 3. 指标

### 函数恢复

* held-out (R^2)；
* NRMSE。

### interaction recovery

* NDCG@10；
* Recall@10；
* Signed Recall@10；
* coefficient sign accuracy。

### 稳定性

* support Jaccard；
* sign-flip rate；
* attribution rank correlation；
* selected-support condition number。

### 最终 attribution

* AUPC；
* AUSufficiency。

---

# 最终建议

## 工程上

**现在就模块化并实现是合理的。**

建议接口至少支持：

```text
support_selector:
    plain_omp
    ssmr_support
    ssmr_signed
```

所有模式共享 query data、dictionary、base (q) 和后续 attribution pipeline。

## 理论上

正文可以包含三项结果：

1. cardinality-balanced Möbius dictionary 的闭式 coherence；
2. CPSS 的 low-stability interaction 控制；
3. stability gap 和 sign margin 条件下的支持、漏检和 sign-error 界。

## 新颖性上

该方案不是一个全新的 stability-selection 理论，但可以构成：

> **面向 coherent deletion-Möbius interaction recovery 的 signed、error-controlled stability-selection 方法。**

是否最终成为主方法，应由以下 gate 决定：

[
\boxed{
\text{SSMR 是否显著提高 exact interaction recovery 和 sign reliability，且不明显损害 function reconstruction}
}
]


[1]: https://www.statslab.cam.ac.uk/~rds37/papers/Shah%20Samworth%202013%20Variable%20selection%20with%20error%20control%20-%20another%20look%20at%20stability%20selection.pdf "Variable selection with error control: another look at stability selection"
[2]: https://arxiv.org/abs/2201.00494 "Cluster Stability Selection"
[3]: https://arxiv.org/abs/2602.06246?utm_source=chatgpt.com "Adaptive Sparse Möbius Transforms for Learning Polynomials"
[4]: https://arxiv.org/abs/1105.4408?utm_source=chatgpt.com "A Simple Proof of the Mutual Incoherence Condition for Orthogonal Matching Pursuit"
[5]: https://link.springer.com/article/10.1007/s11634-026-00675-8 "Randomized smart subset selection for high-dimensional generalized linear models with false positive control | Advances in Data Analysis and Classification | Springer Nature Link"
