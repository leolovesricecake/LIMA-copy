你需要实现一个完整、可复现的实验项目，用于验证以下研究假设：

“对于 LLM 或神经模型在局部输入上的掩码 value function，低阶稀疏 Möbius 超图结构，是否比 Fourier 层级结构更普遍、更稳定，并且能否用比 ProxySPEX 风格方法更少的黑盒前向查询恢复。”

实验必须允许得出“成立、部分成立或不成立”的结论。

# 一、背景与实验目的

## 1. 背景

给定一个输入 x，将其拆分为 n 个特征。特征可以是：

- 文本中的词组、句子或上下文块；
- 问答任务中的证据句；

对任意特征子集 S ⊆ [n]，保留 S 中的特征，并移除或替换其余特征，得到干预输入 x_S。定义黑盒模型的 value function：

    f(S) = model_value(x_S)

ProxySPEX 将 f 看作定义在 2^[n] 上的 set function，使用随机掩码查询模型，通过 Gradient Boosted Trees 拟合 f，再从 GBT 中提取 Fourier 交互。其主要结构假设是：

- Fourier 谱稀疏；
- 重要交互阶数较低；
- 高阶重要交互通常伴随重要低阶子交互，即 hierarchy / staircase structure。

我们要检验一个替代假设：

    f(S) ≈ Σ_{T ⊆ S, |T| ≤ d} θ_T

其中只有少数 θ_T 显著，且 d 较小。

这里 θ_T 是 Möbius 系数，可理解为一个低阶稀疏超图：

- 每个输入特征是节点；
- 非零 θ_T 是一个超边；
- |T| 是交互阶数；
- 不要求 T 的低阶子集也具有显著系数。

因此，该假设允许：

- 孤立的二阶、三阶或四阶交互；
- XOR 或组合条件类非层级交互；
- 高阶交互存在但低阶父项不显著。

## 2. 核心研究问题

实验需要回答以下问题：

RQ1. 在精确可枚举的小规模输入上，Möbius 表示是否具有明显的低阶性？

RQ2. 在达到相同 reconstruction faithfulness 时，Möbius 表示是否比 Fourier 表示需要更少的交互项？

RQ3. 当 Fourier hierarchy 较弱时，Möbius 表示是否仍然保持低阶和稀疏？

RQ4. 在相同黑盒查询预算下，低阶稀疏 Möbius 回归是否比 GBT / ProxySPEX 风格代理模型更容易恢复真实 value function？

RQ5. 该结论是否在不同任务、不同特征粒度、不同样本上稳定？

## 3. 实验原则

必须遵守以下原则：

- 不允许只使用 GBT 恢复出的谱来验证 hierarchy，因为 GBT 本身偏好层级结构。
- 小规模实验必须通过枚举所有 2^n 个子集得到精确 value function。
- Möbius 和 Fourier 的结构比较必须基于共同的 reconstruction error，不能直接比较两种基底的原始系数平方和。
- Fourier 基底是正交的，Möbius 基底不是正交的。必须在报告中明确这一点。
- 所有方法必须使用相同的训练 masks、验证 masks、测试 masks和查询预算。
- 所有结果都要保留 sample-level 数据，不能只报告跨样本平均数。
- 实验失败、数值不稳定和结构假设不成立都必须被记录，不得静默过滤。

# 二、实验总体框架

整个实验分为四个阶段。

阶段 A：Synthetic Sanity Check

使用具有已知结构的合成 set functions 验证：

- Fourier 和 Möbius 变换实现正确；
- hierarchy 指标实现正确；
- 稀疏恢复和 support recovery 指标实现正确；
- 不同结构下各基底应该呈现预期行为。

阶段 B：Exact Structural Audit

将真实输入压缩为 n=10～14 个特征，枚举全部 2^n 个掩码，获得精确 f(S)，计算：

- 精确 Fourier 系数；
- 精确 Möbius 系数；
- 低阶截断曲线；
- 稀疏近似曲线；
- Fourier hierarchy 指标；
- Möbius 与 Fourier 的可压缩性比较。

该阶段用于验证结构假设本身。

阶段 C：Limited-Query Recovery

从精确 value table 中仅抽取少量训练 masks，模拟真实黑盒查询预算，对比：

- Additive LASSO；
- 低阶 Möbius LASSO；
- 低阶 Fourier LASSO；
- Gradient Boosted Trees；
- 可选的 Random Forest。

在剩余未查询 masks 上评价。

该阶段用于验证：

“即使 Möbius 结构存在，能否用较少查询恢复出来。”

阶段 D：Medium-n Scalability Check

在 n=32～128 的输入上不再完整枚举，只使用有限查询：

- 比较 Möbius、Fourier 和 GBT 在 held-out masks 上的 faithfulness；
- 记录达到目标 R² 所需查询次数；
- 验证小规模精确实验的结论是否能扩展。

阶段 D 是补充实验。优先完整实现阶段 A、B、C。

# 三、数据集和模型

代码必须采用 adapter 结构，不要把数据和模型逻辑写死。

## 1. Sentiment Classification

默认配置：

- 数据集：支持 SST-2、Emotion、Eraser、IMDB；
- 模型：可配置；
- value function：
      f(S) = 原始完整输入预测类别对应的 logit
- 不使用 softmax probability，避免饱和。

特征构造：

- 按词进行切分；
- 若文本太短，跳过并记录原因。

## 2. Multi-hop QA / Context Attribution

默认配置：

- 数据集：HotpotQA；
- 模型：可配置；
- question 和固定 instruction 永远保留；
- 仅把 context 拆成特征；
- 优先以句子为特征；
- 句子数量过多时合并相邻句子；
- 句子数量不足时将长句拆为连续子块；
- 最终得到恰好 n 个 context features。

value function：

使用 teacher forcing，不进行随机生成：

    f(S) = reference answer token 的平均 log probability

即：

    f(S) = (1/L) Σ_t log p(y_t | masked_context_S, question, y_<t)

这样可以保证 value function 确定、连续且前向成本可控。

## 3. 样本数量

默认 exact 配置：

- 每个任务 20 个样本；
- n_exact = 12；
- 每个样本需要 4096 个子集查询；
- 先支持 n=10 的快速调试；
- 可选增加 n ∈ {8, 10, 12, 14} 的规模敏感性实验。

默认 medium-n 配置：

- 每个任务 30 个样本；
- n ∈ {32, 64, 128}；
- 不完整枚举。

# 四、掩码和数据处理

## 1. 子集编码

使用整数 bitmask 编码 S：

- bit i = 1：保留第 i 个特征；
- bit i = 0：移除第 i 个特征；
- 0 表示所有可归因特征均移除；
- 2^n - 1 表示完整输入。

所有数组必须以 bitmask 升序存储。

## 2. Mask operator

实现统一接口：

    masked_input = apply_mask(sample, feature_spec, bitmask, operator)

至少支持：

- delete：
  删除未保留的词块或上下文句；
- replace：
  使用模型 tokenizer 的 mask_token；
  若不存在 mask_token，则使用 unk_token。

默认使用 delete；

固定提示、问题和任务说明不能被 mask。

## 3. 推理要求

- model.eval()；
- torch.no_grad()；
- 固定随机种子；
- 不采样生成；
- 支持 batch inference；
- 所有 f(S) 必须缓存到磁盘；
- 重启后能够从缓存继续；
- 每个样本记录实际 model forward 次数；
- 检查 NaN、Inf 和恒定输出；
- 若 f(S) 的标准差小于 1e-8，将样本标记为 degenerate，不进入主要统计，但保留原始结果。

## 4. Value normalization

保存原始 f(S)。

用于回归和跨样本统计时，计算：

    y_norm(S) = [f(S) - mean(f)] / std(f)

R² 在原始值和标准化值上理论一致，但统一使用标准化值可以改善数值稳定性。

# 五、阶段 A：合成实验

至少生成以下五类 synthetic set functions。

1. Additive

    f(S) = Σ_{i∈S} a_i

2. Hierarchical staircase Fourier function

构造若干链式 interaction，例如：

    {1}
    {1,2}
    {1,2,3}
    {4}
    {4,5}

在 Fourier 基底中指定非零系数。

3. Sparse non-hierarchical Fourier peak

只设置若干孤立高阶 Fourier 系数，例如：

    {1,4,7}
    {2,5,9,10}

对应父集合系数为零。

4. Sparse low-degree Möbius hypergraph

随机采样 s 个阶数为 2～4 的超边，设置 Möbius 系数，其子集系数不强制非零。

5. Dense low-degree function

二阶或三阶候选中大量系数非零，用于验证“低阶但不稀疏”的失败情形。

每类函数至少：

- n=12；
- 50 个随机实例；
- 不同噪声水平；
- 可选 Gaussian observation noise。

必须验证：

- Möbius transform 后再 inverse transform 能恢复原函数；
- Fourier transform 后再 inverse transform 能恢复原函数；
- max absolute reconstruction error < 1e-8；
- 已知 support recovery 指标正确；
- hierarchy 指标在不同 synthetic 类型上表现符合预期。

# 六、阶段 B：精确结构审计

对每个真实样本枚举所有 2^n 个 masks，得到：

    values[mask] = f(S_mask)

## 1. 精确 Möbius 变换

定义：

    m(T) = Σ_{U⊆T} (-1)^(|T|-|U|) f(U)

使用 fast Möbius transform：

    m = f.copy()
    for i in range(n):
        for mask in range(2^n):
            if mask has bit i:
                m[mask] -= m[mask without bit i]

逆变换使用 fast zeta transform。

## 2. 精确 Fourier / Walsh-Hadamard 变换

定义：

    F(T) = 2^(-n) Σ_S (-1)^(|S∩T|) f(S)

使用 Fast Walsh-Hadamard Transform。

逆变换必须严格匹配当前符号约定。

## 3. 截距处理

所有结构指标默认排除空集系数 T=∅。

空集系数仍用于重建，但不能：

- 计入 sparsity；
- 计入 degree distribution；
- 计入 hierarchy numerator；
- 计入 top interaction 数量。

## 4. 低阶性指标

分别对 Möbius 和 Fourier 计算 degree truncation。

Möbius：

    f_M,≤d(S) = Σ_{T⊆S, |T|≤d} m(T)

Fourier：

    f_F,≤d(S) = Σ_{|T|≤d} F(T)(-1)^(|S∩T|)

对 d=1,...,d_max 计算：

    R²_degree(d)
    normalized RMSE(d)

定义：

    d90 = 达到 R² ≥ 0.90 的最小 d
    d95 = 达到 R² ≥ 0.95 的最小 d

若未达到，记录为 null，不要设为最大阶数。

## 5. 稀疏性与可压缩性指标

不能直接比较 Möbius 和 Fourier 的 coefficient energy，因为：

- Fourier 基底正交；
- Möbius 基底非正交。

主比较必须使用相同输出空间中的 reconstruction R²。

实现以下方法：

A. Fourier exact top-k

由于 Fourier 基底正交，按 |F(T)| 从大到小保留前 k 个非空系数。

B. Möbius weighted-magnitude top-k

计算单项在 uniform subset distribution 下的近似贡献分数：

    score(T) = |m(T)| * 2^(-|T|/2)

按 score 排序并重建。

该结果只作为辅助指标。

C. Möbius OMP oracle

构造最高阶数不超过 d_max 的 Möbius design：

    Φ_M[S,T] = 1{T⊆S}

在完整 2^n 个 masks 上运行 Orthogonal Matching Pursuit，逐步选择最能降低 reconstruction error 的交互项。

D. Fourier OMP oracle

构造：

    Φ_F[S,T] = (-1)^(|S∩T|)

同样运行 OMP，用于和 Möbius OMP 进行完全对称的比较。

默认：

    d_max = 4

若资源允许，可增加 d_max=5。

记录 k：

    k ∈ {1,2,4,8,16,32,64,128,256,...}

直到达到候选交互总数。

计算：

    R²_k
    k80
    k90
    k95
    AUC_R²_logk

AUC 使用 log2(k) 为横轴，先对固定公共网格插值后计算。

## 6. Fourier hierarchy 指标

从精确 Fourier 系数计算，不允许使用 GBT 恢复谱。

对：

    k ∈ {8,16,32,64,128,256}

计算：

Direct Subset Rate：

    DSR(k) =
      average over T in top-k:
      [number of direct parents T\{i} also in top-k] / |T|

Strong Hierarchy Rate：

    SHR(k) =
      fraction of T in top-k whose every subset is also in top-k

必须同时计算分阶指标：

    DSR_d(k)
    SHR_d(k)

其中只统计 |T|=d 的项。

空集和一阶项不能人为抬高高阶 hierarchy 指标。

额外计算：

    orphan_ratio(k)

定义为 top-k 中没有任何显著直接父项的二阶及以上交互所占比例。

额外计算 orphan spectral energy ratio：

    Σ orphan_T F(T)^2 / Σ_{|T|≥2, T in top-k} F(T)^2

## 7. 关键关联分析

对每个样本计算并保存：

- Möbius d90、k90；
- Fourier d90、k90；
- Fourier DSR、SHR；
- Möbius 与 Fourier 的 R²-k AUC；
- 输出方差；
- 输入长度；
- 任务类别。

分析：

A. Möbius k90 与 Fourier k90 的配对差值；

B. Möbius d90 与 Fourier d90 的配对差值；

C. Fourier DSR 与 Möbius k90 的相关性；

D. 在 Fourier DSR 低的样本中，Möbius 是否仍可用低阶少量项达到 R²≥0.9；

E. hierarchy 弱是否只出现在 value function 本身难以压缩的样本中。

# 七、阶段 C：有限查询恢复

该阶段仍使用 exact value table，但每个实验只允许访问一部分 masks，从而模拟有限黑盒调用。

1. 数据划分

对每个 sample 和 random seed：

- 先固定 20% masks 作为最终 test set；
- 再固定 10% masks 作为 validation set；
- 剩余 masks 作为可查询 pool；
- test 和 validation masks 永远不能进入训练。

必须保留：

- empty mask；
- full mask；

但不要强制将其放入训练集，单独记录是否被查询。

2. Query budgets

默认预算：

    m ∈ {2n, 4n, 8n, 16n, 32n}

同时确保：

    m < available_pool_size

对于 n=12：

    m ∈ {24,48,96,192,384}

每个 budget 使用 5 个随机种子。

所有模型在同一 sample、budget、seed 下使用完全相同的训练 masks。

3. Baseline 1：Additive LASSO

设计矩阵：

    Φ_add[S,i] = 1{i∈S}

包含不惩罚的 intercept。

通过 validation set 选择 regularization strength。

4. Baseline 2：Low-degree Möbius LASSO

候选交互：

    C_M(d) = {T: 1≤|T|≤d}

设计矩阵：

    Φ_M[S,T] = 1{T⊆S}

d ∈ {2,3,4}。

设计列必须标准化：

    column_norm(T) ≈ sqrt(E[Φ_M,T²]) = 2^(-|T|/2)

训练后将系数转换回原始尺度。

使用：

- Lasso；
- 可选 ElasticNet；
- validation 选择 d 和 λ；
- intercept 不惩罚。

必须报告 design matrix condition number 或数值稳定性诊断。

5. Baseline 3：Low-degree Fourier LASSO

候选交互：

    C_F(d) = {T: 1≤|T|≤d}

设计矩阵：

    Φ_F[S,T] = (-1)^(|S∩T|)

使用与 Möbius LASSO 完全相同的：

- d 候选；
- λ 网格；
- validation protocol；
- 最大迭代次数；
- convergence tolerance。

6. Baseline 4：GBT / ProxySPEX-style surrogate

输入：

    mask vector z ∈ {0,1}^n

输出：

    f(S)

超参数搜索：

- max_depth ∈ {2,3,5,None}
- n_estimators ∈ {100,300,1000}
- learning_rate ∈ {0.03,0.1}
- subsample ∈ {0.8,1.0}

使用 validation R² 选择模型。

由于 exact 阶段 n 较小，可将训练后的 GBT 在所有 2^n 个 masks 上预测，再对 GBT prediction table 做精确 Fourier transform，从而分析其恢复出的 interaction support。

不要求实现树结构的解析 Fourier extraction。

7. 评价指标

预测指标：

    test R²
    test normalized RMSE
    test MAE

查询效率指标：

    达到 test R² ≥ 0.80 所需最小 m
    达到 test R² ≥ 0.90 所需最小 m
    达到 test R² ≥ 0.95 所需最小 m
    AUC_R²_log_query

若从未达到阈值，记录为 null。

结构恢复指标：

定义每个样本的 Möbius oracle support：

- 使用完整数据上的 Möbius OMP；
- 取达到 R²≥0.90 的最小 support；
- 记为 G_M,90。

对估计模型取相同数量的 top interactions，计算：

    precision
    recall
    F1
    support Jaccard

Fourier 同理，使用 exact Fourier top-k 或 Fourier OMP oracle support。

系数指标：

- oracle support 上的 coefficient RMSE；
- Spearman rank correlation；
- top-10 interaction recall。

GBT 的 Fourier support 与 exact Fourier support 比较。

8. 统计方法

所有方法比较都必须是 sample-level paired comparison。

输出：

- 每个任务的均值、中位数、标准差；
- 95% bootstrap confidence interval；
- 配对 bootstrap 的差值区间；
- Wilcoxon signed-rank test，样本数足够时使用；
- 同时报告 effect size，不只报告 p-value。

Bootstrap 以 sample 为重采样单位，不以 masks 为单位。

# 八、阶段 D：中等维度实验

仅在阶段 A～C 通过后实现。

1. 特征数

    n ∈ {32,64,128}

2. 不能完整枚举。

3. 对每个样本预先生成：

- 一个固定训练查询 pool；
- 1000 个独立 test masks；
- 额外的 near-full masks，即随机删除 1～10 个特征；
- 可选 fixed-cardinality masks。

4. 比较：

- Additive LASSO；
- Möbius LASSO，最大 d=2 或 3；
- Fourier LASSO，最大 d=2 或 3；
- GBT。

5. 查询预算：

    α n log2(n), α ∈ {0.25,0.5,1,2,4}

同时报告实际查询数。

6. 评价：

- uniform random test R²；
- near-full test R²；
- 达到目标 R² 的查询次数；
- wall-clock；
- peak memory；
- 拟合时间；
- 推理时间；
- interaction support stability。

# 九、结果判定标准

不要只输出一句“假设成立”。

分别判断以下假设。

H1：低阶性

若多数任务和多数样本满足：

    Möbius d90 ≤ 4

则支持“低阶 Möbius”假设。

必须报告真实比例和置信区间，不要只给二元结论。

H2：稀疏性

比较：

    k90_Möbius_OMP
    k90_Fourier_OMP

若 Möbius 在多数样本上需要更少项，且配对差值置信区间显著偏向 Möbius，则支持 Möbius 更可压缩。

H3：对 hierarchy 失败的鲁棒性

筛选 Fourier DSR 较低的样本，例如：

    DSR_3/4 < 0.6

检查其中 Möbius 是否仍满足：

    d90 ≤ 4
    k90 较小

阈值必须做敏感性分析，不得只使用 0.6。

H4：有限查询可恢复性

在相同查询预算下比较：

    Möbius LASSO vs GBT
    Möbius LASSO vs Fourier LASSO

若 Möbius 仅在 oracle 曲线上更稀疏，但有限样本下恢复效果差，则结论应是：

    “结构可能存在，但当前采样和估计方法无法有效利用。”

H5：跨任务稳定性

分别报告各任务结论。

如果只在 sentiment 上成立，而 QA 不成立，必须判定为 task-dependent，而非总体成立。

最终结论只能使用以下类别：

- Supported；
- Partially supported；
- Not supported；
- Inconclusive due to insufficient evidence。

# 十、结果文件结构

项目根目录建议如下：

mobius_verify/
├── README.md
├── requirements.txt
├── configs/
│   ├── exact_default.yaml
│   ├── recovery_default.yaml
│   ├── medium_default.yaml
│   └── synthetic_default.yaml
├── src/
│   ├── datasets/
│   │   ├── base.py
│   │   ├── sentiment.py
│   │   └── hotpotqa.py
│   ├── models/
│   │   ├── base.py
│   │   ├── classifier.py
│   │   └── generative_qa.py
│   ├── featureization.py
│   ├── masking.py
│   ├── value_functions.py
│   ├── subset_enumeration.py
│   ├── transforms.py
│   ├── hierarchy_metrics.py
│   ├── reconstruction_metrics.py
│   ├── oracle_approximation.py
│   ├── fit_additive.py
│   ├── fit_mobius.py
│   ├── fit_fourier.py
│   ├── fit_gbt.py
│   ├── statistics.py
│   ├── plotting.py
│   └── utils.py
├── scripts/
│   ├── run_synthetic.py
│   ├── collect_exact_values.py
│   ├── compute_exact_spectra.py
│   ├── run_oracle_analysis.py
│   ├── run_limited_query_recovery.py
│   ├── run_medium_n.py
│   ├── aggregate_results.py
│   └── make_figures.py
├── tests/
│   ├── test_mobius_transform.py
│   ├── test_fourier_transform.py
│   ├── test_inverse_transform.py
│   ├── test_hierarchy_metrics.py
│   ├── test_synthetic_support.py
│   └── test_result_schema.py
└── results/
    ├── run_config.yaml
    ├── environment.json
    ├── manifests/
    │   ├── samples.csv
    │   └── exclusions.csv
    ├── features/
    │   └── <task>/<sample_id>.json
    ├── values/
    │   └── <task>/<sample_id>/<mask_operator>/
    │       ├── values.npy
    │       ├── masks.npy
    │       └── metadata.json
    ├── spectra/
    │   └── <task>/<sample_id>/<mask_operator>/
    │       ├── mobius.npy
    │       ├── fourier.npy
    │       ├── degree_metrics.json
    │       └── hierarchy_metrics.json
    ├── oracle/
    │   └── <task>/<sample_id>/<mask_operator>/
    │       ├── curves.parquet
    │       ├── mobius_selected_terms.jsonl
    │       ├── fourier_selected_terms.jsonl
    │       └── summary.json
    ├── recovery/
    │   └── <method>/<task>/<sample_id>/
    │       └── budget_<m>/seed_<seed>/
    │           ├── metrics.json
    │           ├── predictions.parquet
    │           ├── coefficients.npz
    │           ├── selected_terms.jsonl
    │           └── fit_metadata.json
    ├── aggregate/
    │   ├── sample_level_metrics.parquet
    │   ├── task_level_summary.csv
    │   ├── paired_comparisons.csv
    │   ├── bootstrap_intervals.csv
    │   ├── hypothesis_results.json
    │   └── hypothesis_report.md
    ├── figures/
    │   ├── degree_curves/
    │   ├── sparsity_curves/
    │   ├── hierarchy/
    │   ├── query_efficiency/
    │   ├── support_recovery/
    │   └── task_comparisons/
    └── logs/

# 十一、必须生成的图表

1. 每个任务的 degree-R² 曲线

横轴：最大交互阶数 d
纵轴：精确 reconstruction R²
曲线：Möbius、Fourier

2. 每个任务的 k-R² 曲线

横轴：保留交互数 k，使用 log scale
纵轴：R²
曲线：

- Möbius OMP
- Möbius weighted top-k
- Fourier exact top-k
- Fourier OMP

3. d90 和 k90 分布图

必须显示 sample-level 点，不只显示均值。

4. Fourier hierarchy 与 Möbius compressibility 散点图

例如：

横轴：分阶 DSR
纵轴：Möbius k90 或 d90

5. Query efficiency 曲线

横轴：模型查询数
纵轴：test R²
方法：

- Additive LASSO
- Möbius LASSO
- Fourier LASSO
- GBT

6. Support recovery 曲线

横轴：查询数
纵轴：support recall / F1

7. 按任务分组的 paired difference 图

例如：

    k90_Möbius - k90_Fourier
    R²_Möbius - R²_GBT

必须包含 bootstrap confidence interval。

# 十二、实现和工程要求

1. 先检查当前代码仓库、Python 环境、GPU、缓存模型和已有数据，不要直接重写已有模块。

2. 所有实验必须通过 CLI 运行，不能依赖人工执行 notebook。

示例：

    python scripts/run_synthetic.py --config configs/synthetic_default.yaml

    python scripts/collect_exact_values.py \
        --config configs/exact_default.yaml

    python scripts/compute_exact_spectra.py \
        --results_dir results/run_xxx

    python scripts/run_limited_query_recovery.py \
        --config configs/recovery_default.yaml

    python scripts/aggregate_results.py \
        --results_dir results/run_xxx

3. 所有长任务必须：

- 可恢复；
- 可跳过已完成样本；
- 原子写入结果文件；
- 日志中记录进度；
- 出错时保留失败样本信息。

4. 不要默认下载大型或 gated 模型。

模型名称、数据路径和设备全部通过 YAML 配置。

5. 提供 fast smoke test：

- synthetic；
- 2 个真实样本；
- n=8；
- 小模型；
- 几分钟内完成。

6. 提供单元测试。

特别验证：

- Möbius transform；
- inverse Möbius transform；
- FWHT；
- inverse FWHT；
- degree truncation；
- DSR / SHR；
- OMP curve 单调不下降；
- result schema。

7. 数值和内存要求

- 优先使用 float64 计算精确变换；
- 模型推理缓存可使用 float32；
- 不要一次性为过大的 n 构建完整 2^n × p design matrix；
- exact 阶段逐样本处理；
- 大矩阵使用 memmap、稀疏矩阵或分块；
- 输出 peak memory。

8. README 必须包括：

- 研究问题；
- 安装方式；
- 数据和模型配置；
- 每个阶段的运行命令；
- 结果目录说明；
- 指标定义；
- 如何复现实验；
- 已知限制。

# 十三、最终交付

完成后需要提供：

1. 可运行代码；
2. 全部配置文件；
3. 单元测试；
4. smoke test 结果；
5. 一个 hypothesis_report.md。

hypothesis_report.md 必须按照以下结构：

# Executive Summary

用三到五句话说明：

- Möbius 是否更低阶；
- Möbius 是否更稀疏；
- hierarchy 弱时是否仍有效；
- 有限查询下是否更容易恢复。

# Exact Structural Results

按任务报告：

- d90；
- k90；
- R²-k AUC；
- DSR / SHR；
- sample-level 异质性。

# Limited-query Results

报告：

- query-to-R²；
- support recovery；
- GBT、Fourier 和 Möbius 的配对比较；
- 失败率。

# Evidence For the Hypothesis

只列真实支持证据。

# Evidence Against the Hypothesis

必须列出反例、失败任务和不稳定结果。

# Confounders and Limitations

至少讨论：

- feature granularity；
- mask operator；
- Möbius 非正交性；
- value function 选择；
- 小 n 枚举和大 n 场景的差距；
- LASSO 求解和条件数问题。

# Final Verdict

只能从以下选一个：

- Supported
- Partially supported
- Not supported
- Inconclusive

并说明后续是否值得开发 adaptive sparse Möbius / hypergraph attribution 算法。

# 十四、实施顺序

建议按照以下顺序执行：

1. 实现和测试精确 Möbius / Fourier 变换；
2. 完成 synthetic sanity check；
3. 实现一个 sentiment sample 的 exact enumeration；
4. 验证缓存、变换和全部指标；
5. 扩展到完整 sentiment exact experiment；
6. 增加 QA adapter；
7. 完成 exact structural audit；
8. 实现 limited-query baselines；
9. 运行 paired query-efficiency comparison；
10. 汇总结果并生成 hypothesis report；
11. 只有前述结果支持假设时，再实现 medium-n 实验。
