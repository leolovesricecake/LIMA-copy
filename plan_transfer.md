# 背景
现代 LLM 拥有数千亿参数，其推理过程涉及复杂的注意力机制与专家混合架构（MoE），这使得传统基于梯度或单一注意力的解释方法在面对长文本输入与自回归输出时，往往表现出显著的噪声干扰、计算瓶颈以及保真度不足等缺陷。现有的文本归因方法主要可以归纳为梯度基方法、扰动基方法和博弈论基方法三类，但它们在迁移到如 DeepSeek 或 Qwen 等模型时均面临特定局限。

为了构建可信赖的 AI 系统，归因方法必须能够精确识别输入中对决策影响最大的核心区域。LIMA 框架通过将归因问题重新表述为子模函数优化下的子集选择问题，在图像模态中展现了卓越的效率与保真度。子模性本质上反映了“边际收益递减”的特性，这与文本逻辑中关键语义点的分布规律高度契合：一旦识别出最核心的因果片段，增加冗余的背景文本对模型输出概率的提升将迅速饱和。通过将归因转化为一个受限基数下的子模最大化问题，可以在保证保真度的前提下，寻找能够激活模型响应的最精简输入子集。

在判别式图像任务中，归因通常表现为像素级别的显著性地图。而在生成式文本大模型中，归因的目标是解释输入提示词（Prompt）中的哪些片段导致了特定输出序列（Response）的生成。为了将 LIMA 的核心思想——子区域划分、子模目标函数、以及双向贪心搜索——迁移至文本大模型，需重点解决文本序列的模态适配、超大规模参数量下的时空开销平衡以及特征提取层级的选择等核心挑战。


# 文本子区域划分

### 句子级/段落级划分

句子/段落作为表达独立逻辑的最小单元，在被独立提取或组合时，能较好地保持模型内在推理路径的稳定性。
- 句子级划分：NLTK/SpaCy 识别标点边界，通用。
- 递归划分：按段落、句子、词汇分层切分，适用于长文档总结与检索。

### 基于语义或不确定性的动态划分

1. 语义：利用嵌入模型的相似度度量，在文本语义发生剧烈偏移的位置设置断点，从而确保每个子区域包含一个完整的话题或论据。（类似于 RAG？）
2. 不确定性：Meta-Chunking 框架引入了基于模型感知不确定性的边界检测算法。通过计算模型对序列中每个位置的困惑度（Perplexity, PPL）或边缘采样（Margin Sampling, MSP）指标，可以识别文本中的逻辑转折点。具体而言，当 PPL 在某个位置突然升高时，往往预示着新语义信息的引入，该位置即为理想的子区域边界。

> Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception

# 文本化子模目标函数

### 置信度分数：从 EDL 到 LogU 转换
原论文利用证据深度学习（EDL）产生证据向量，并计算 $1 - K / \sum(e+1)$ 作为置信度。在 LLM 中，输出概率分布并非简单的分类层，而是经过 Softmax 归一化后的词表概率。
由于归一化过程会掩盖证据强度（Evidence Strength），建议采用“对数诱导 Token 不确定性”（LogTokU）框架。该框架将 LLM 词表原始 Logits 视为狄利克雷分布（Dirichlet Distribution）的先验证据参数 $\alpha$。此时，置信度分数 $s_{conf}$ 可以通过计算预测 Token 的认识不确定性（Epistemic Uncertainty, EU）来获得： $s_{conf}(S) = 1 - \frac{\alpha_{target} + 1}{\sum_{j=1}^{V} (\alpha_j + 1)}$
这种方法能够区分“模型知道有多个答案”和“模型完全不知道答案”两种情况，为子集搜索提供更鲁棒的奖励信号。

> Estimating LLM Uncertainty with Logits

### 一致性分数：潜在语义轨迹对齐
一致性分数衡量候选子集产生的特征向量与全量输入产生的目标语义特征 $f_s$ 的对齐程度。对于 LLM，特征向量不再是分类层之前的单一向量，而是分布在 Transformer 各层中的隐藏状态（Hidden States）轨迹。
在实现文本迁移时，应将 $f_s$ 定义为模型处理原始全量 Prompt 时，在特定深度层（如总层数的 70% 处，此时语义最丰富且未过度特化）生成的隐藏状态经过均值池化（Mean Pooling）后的向量。
$s_{cons}(S, f_s) = \cos(Emb(Subset\_Output), f_s)$
通过利用 BERTScore 或余弦相似度，该分数能确保被选中的文本片段在潜在空间内重现了原始输入的核心语义流。

> Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models
> Semantic Routing: Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers
> Layer by Layer: Uncovering Hidden Representations in Language Models

### 协作性分数：衡量片段的必要性与不可替代性
协作性分数识别那些单独看贡献微弱但群体效应显著的片段。在文本场景中，这与“必要性分数”（Necessity Score）概念一致，即评估移除某个文本块后，目标输出序列 log 概率的下降程度。
$s_{colla}(S) = \text{softmax\_entropy}(\text{Full\_Input}) - \text{softmax\_entropy}(\text{Full\_Input} \setminus S)$
这种方法能挖掘出那些作为推理前提（Premise）或关键约束（Constraint）的文本区域，即使它们不包含显式的关键词，但在逻辑构建中不可或缺。

> Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
> Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
> Aligning What LLMs Do and Say: Towards Self-Consistent Explanations

### 有效性分数：语义多样性与冗余消除
有效性分数通过最大化子区域间的特征距离来限制冗余信息的累积。文本输入中常包含大量的修辞、语气词或同义重复。迁移后的有效性分数 $s_{eff}$ 计算逻辑如下：
$s_{eff}(S) = \sum_{T_i \in S} \min_{T_j \in S, T_j \neq T_i} dist(Emb(T_i), Emb(T_j))$
其中 $dist$ 为余弦距离。该项能确保选出的子区域在语义上尽可能互补，从而用最少的词汇量覆盖最广的语义维度。

# 特征提取器与 EDL 模型
将 ResNet101、ViT 替换为千亿参数的 LLM 后，特征提取与置信度估计面临计算效率与特征定位的双重挑战。DeepSeek 和 Qwen 这类解码器架构（Decoder-only）逐 Token 输出的特性要求我们必须重新思考特征向量的获取方式。

### 隐藏状态的层级选择理论
研究 LLM 内部表示的质量发现，最后一层的隐藏状态往往过度倾向于预测下一个 Token，而在捕捉全局语义信息方面，中间层（Mid-depth layers）展现出更强的鲁棒性。
- 浅层（1-10层）： 主要捕捉词法、句法等低级特征。
- 中深层（约 70% 深度）： 发生“语义压缩”，隐藏状态包含最丰富的抽象概念表示，是提取特征向量的最佳位置。
- 末层： 隐藏状态被拉向词表预测分布，多样性降低。

因此，建议在迁移 LIMA 时，配置动态特征路由（RouteSAE）或在特定中间层注入 Hook 来获取特征。

> Layer by Layer: Uncovering Hidden Representations in Language Models

### 均值池化与时空开销平衡
由于大模型输入序列长度不一，获取固定维度向量必须经过池化。实验证明，在自回归模型中，均值池化（Mean Pooling）显著优于末尾 Token 池化，因为它能缓解“近因偏差”（Recency Bias），更公平地反映输入子区域中所有 Token 的贡献。
为了平衡空间开销，可以使用轻量级投影头（Projection Head）将高维隐藏状态（如 4096 维）映射到低维嵌入空间，从而加速子模函数的距离计算与对齐操作。

> Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models
> Pooling and Attention: What are Effective Designs for LLM-based Embedding Models?
> One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States
> Probing Large Language Model Hidden States for Adverse Drug Reaction Knowledge

### 其他
- 置信度模型 (EDL)使用 Logit-Dirichlet 模型，提取 Top-K Logits 作为证据强度。
- 语义锚点 ($f_s$)使用原文本生成的中间层均值，作为归一化一致性参考。

# 基于机理的高效搜索优化：KV Cache 复用与 UHeads
在 LIMA 原流程中，贪心搜索需要频繁调用模型以计算边际增益。对于千亿参数 LLM，每次前向传播的预填充（Prefill）阶段开销巨大，直接迁移会导致归因速度极慢。

### 跨段 KV Cache 复用技术（KVLink/SemShare）
传统 LLM 推理依赖严格的前缀匹配进行 KV Cache 复用。但在子集选择过程中，我们生成的文本序列是由不连续的子块 $S$ 拼接而成的，这会导致前缀失效，触发昂贵的重新计算。
推荐引入分段 KV Cache 共享框架，如 KVLink。该技术允许对每个文本子区域（Chunk）独立进行预编码并持久化其 KV 状态。在搜索阶段，只需将选中子集的 KV 块动态拼接，并通过旋转位置编码（RoPE）修正与特殊的“链路 Token”恢复跨段注意力依赖。

> KVLINK: Accelerating Large Language Models via Efficient KV Cache Reuse

### 不确定性头（UHeads）作为快速代理
为了进一步减少主模型的调用次数，可以训练参数量极小（<10M）的 UHeads。UHeads 直接读取冻结主模型的中间层激活，并预测该步推理的正确性或置信度。在贪心搜索初期，可以先利用 UHeads 快速筛选出具有潜力的候选子集，仅在最后阶段由主模型进行精确验证，实现类似“启发式搜索”的效果。

> Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads
> A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs

# 改进的双向贪心搜索算法

针对 LLM 巨大的调用开销，LIMA 提出的双向贪心搜索（Bidirectional Greedy Search）必须进行工程化适配。该算法的核心优势在于同时识别“最有影响力”和“最无关紧要”的片段，从而加速子集收敛。

### 算法逻辑的深度优化
在文本 LLM 迁移场景下，双向搜索过程建议如下：
1. 正向贪心 ($S_{forward}$): 每次迭代从剩余候选块中选出一个使子模目标函数 $F(S \cup \{\alpha\})$ 增益最大的块。
2. 反向排除 ($S_{reverse}$): 同时维护一个包含所有块的全集，每次识别 $n_p$ 个增益最小的样本。
3. 负采样优化: 通过 $n_p$ 参数（推荐为 8-12）控制每轮排除的粒度。增加 $n_p$ 虽然单次计算量略增，但能显著减少总搜索轮数，尤其适用于长篇 Prompt。

| 搜索策略 | 图像版 LIMA | 文本 LLM 迁移版 |
| 搜索方向 | 纯双向像素块选择 | 双向 Chunk 选择 + KV 注入 |
| 边际增益计算 | 完整模型前向 | 仅 Decode 段计算 (KV Reuse) |
| 候选集过滤 | 排序所有 Patch | 基于 UHeads/Attention 先验初筛 |
| 收敛判断 | 达到 $k$ 个 Patch | 语义信息增益低于阈值 |

> Double Greedy Algorithm for Submodular Maximization

### 随机梯度子模优化
对于极长文本归因，可以结合随机贪心算法（Stochastic Greedy）。不再遍历所有候选块，而是从剩余块中随机采样一个子集（大小为 $V/k$），在该子集中寻找局部最优增益块。这种方法能够以极小的精度损失换取线性的时间复杂度提升，使归因算法能够处理百万级 Token 的上下文。

>  Framework for Interpretability in Machine Learning for Medical Imaging
