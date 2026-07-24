你提出的三个问题中，第 1 点和第 3 点按此前确认执行，但第 2 点需要修改。

不要将所有文本强制 balanced merge 成固定的 n 个 block，作为主实验的特征定义。我们的最终归因目标是词级归因，因此主实验必须以真实词作为特征，并允许不同样本具有不同的特征数量 n_i。

固定 n 只能用于控制某些精确枚举实验的计算规模，不能通过合并单词来实现，否则会改变待验证的交互结构。

# 一、为什么不再固定特征数量

原设计将全文 balanced merge 为恰好 n=12 个连续 block，虽然方便枚举 2^n 个子集，但会改变实际研究对象。

例如原本的词级交互：

    {not, at, all, convincing}

可能因为若干词被放入同一个 block，而被压缩成一阶或二阶 block 效应。

这会系统性影响：

- 交互阶数；
- Möbius 稀疏度；
- Fourier hierarchy；
- support 大小；
- 不同方法的有限查询恢复难度。

因此，使用固定 block 后得到的结论不能直接外推到真实词级归因。

新的原则是：

1. 特征粒度与最终使用场景一致；
2. sentiment 任务中，每个自然词是一个特征；
3. 每个样本具有自己的 n_i；
4. exact enumeration 只在能够自然完整枚举的样本或局部探针上进行；
5. query budget、候选交互数和结果统计根据 n_i 归一化。

# 二、词级特征定义

1. 主特征单位

对 sentiment 任务，使用 lexical word 作为 feature。

优先通过 tokenizer 的 offset mapping 将 subword token 映射回原始文本中的词：

- 同一个原始词产生的多个 subword token 归为一个 feature；
- 特殊 token 不作为 feature；
- 不按模型 subword token 单独归因；
- 不将多个自然词合并为一个 feature；
- 不截取前 n 个词；
- 不根据梯度、注意力或其他 attribution 方法选择词。

2. 标点处理

标点不单独作为主特征，默认附着到相邻词：

- 前置标点附着到后一个词；
- 后置标点附着到前一个词；
- 缩写和带撇号形式尽量保持为一个 lexical unit；
- 必须能够通过所有 word spans 重建送入模型的文本。

若现有 tokenizer 或数据预处理无法严格恢复原始空白，至少保证 feature spans 与实际送入模型的规范化文本一致。

3. 每个样本保存

    n_features
    word_texts
    word_char_spans
    word_token_spans
    original_text
    normalized_model_text
    tokenizer_name
    token_count
    punctuation_attachment_policy

4. Masking

对一个词进行 mask 时，必须同时处理该词的全部 subword token 或对应字符区间。

分类器可使用 replace 或 delete 两种 operator：

- replace：使用 mask_token；若不存在则使用 unk_token；
- delete：从输入中删除整个词及其附着标点。

主结果使用预先指定的 operator，另一种作为敏感性实验。

# 三、修订后的实验总体框架

实验不再只有“固定 n 的 exact experiment”，而是分为三个互补层次。

--------------------------------------------------
阶段 A：自然短文本的全局精确枚举
--------------------------------------------------

目的：

在不改变词级特征定义的情况下，得到真实完整 value table 和真实 Möbius/Fourier 谱。

样本选择：

- 选择自然词数 n_i 不超过 exact_max_n 的文本；
- 不合并词；
- 不截断；
- 不补齐；
- 默认 exact_max_n=14；
- smoke test 使用 exact_max_n=8 或 10；
- 资源允许时可增加到 16。

建议按长度分层采样：

    n_i ∈ [6,8]
    n_i ∈ [9,11]
    n_i ∈ [12,14]

每个长度区间尽量保留相近数量样本。

对每个样本完整枚举：

    2^(n_i)

个词子集，得到全局词级 value function：

    f_x(S), S ⊆ [n_i]

然后计算精确 Möbius 和 Fourier 谱。

注意：

- 不要求所有样本具有相同 n；
- 所有指标逐样本计算；
- 汇总时按长度区间分别报告；
- 不能将短文本结论直接宣称适用于长文本。

这一阶段主要回答：

- 对自然短句，词级 Möbius 是否低阶、稀疏；
- Fourier hierarchy 是否稳定；
- 两种表示的精确可压缩性如何。

--------------------------------------------------
阶段 B：长文本上的条件局部精确审计
--------------------------------------------------

仅使用短文本会引入 selection bias，因此需要在长文本中构造词级 probe set。

给定一个具有 n_i 个词的完整输入 x，选择一个大小为 k 的词集合 A：

    A ⊂ [n_i], |A|=k

只对 A 中的词做组合干预，其余词保持原始状态，定义条件 value function：

    f_{x,A}(S)
      = model_value(
          A 中只保留 S，
          A 外所有词保持原样
        )

其中：

    S ⊆ A

这样可以完整枚举 2^k 个组合，同时仍然以单词为真实特征，并保留长文本的大部分上下文。

默认：

    k ∈ {8,10,12}
    每个长文本构造多个 probe sets

至少实现三种模型无关的 probe 选择方式：

1. contiguous_random

随机选择一个连续 k 词窗口。

2. dispersed_random

从全文均匀随机选择 k 个分散位置的词。

3. stratified_position

从文本开头、中间和结尾分层随机选词，避免全部集中在局部。

所有 probe 选择都必须在模型推理和 attribution 之前完成，不能利用梯度、注意力或模型输出选择。

每个 probe 保存：

    sample_id
    probe_id
    probe_strategy
    probe_word_indices
    probe_word_texts
    k
    conditioning_mode = rest_present
    random_seed

该阶段得到的是：

    条件于其他词存在时，所选词之间的局部交互结构

不能将它误写成完整文本的全局交互谱。

这一阶段主要回答：

- 长文本局部词交互是否仍呈现低阶稀疏 Möbius 结构；
- 该结论是否只出现在短句；
- 连续局部词和远距离词是否表现不同；
- Fourier hierarchy 在长上下文条件下是否更容易失效。

可选增加第二种 conditioning mode：

    rest_absent

即 A 外的词全部删除，只改变 A 中的词。

但它只能作为敏感性实验，主实验使用：

    rest_present

因为它更接近解释完整输入附近的模型行为。

--------------------------------------------------
阶段 C：完整词级输入上的有限查询恢复
--------------------------------------------------

这是最接近实际归因流程的实验。

对每个样本：

- 使用全部自然词；
- n_i 等于真实词数；
- 不合并；
- 不固定；
- 不完整枚举；
- 使用有限查询训练归因模型；
- 在独立 held-out masks 上评价 faithfulness。

查询预算按样本长度缩放：

    m = α n_i log2(n_i)

默认：

    α ∈ {0.25,0.5,1,2,4,8}

同时记录：

    m
    m / n_i
    m / [n_i log2(n_i)]

所有方法在相同 sample、budget、seed 下使用完全相同的训练 masks、validation masks 和 test masks。

对 medium / long text，需要至少使用三类测试分布：

1. uniform masks

每个词独立以 0.5 概率保留。

2. near-full masks

随机删除少量词，例如：

    r ∈ {1,2,3,5,10}

3. fixed-cardinality masks

固定保留比例或固定删除比例。

原因是实际归因通常关心完整输入附近的干预，而不仅是保留约一半词的均匀掩码。

这一阶段比较：

- Additive LASSO；
- Low-degree Möbius regression；
- Low-degree Fourier regression；
- sklearn GradientBoostingRegressor。

由于完整候选交互数随 n_i 快速增长：

    Σ_{j=1}^d C(n_i,j)

实现时需要限制：

- n_i 较小时可使用 d≤4；
- 中等 n_i 时使用 d≤3；
- 大 n_i 时主实验先使用 d≤2；
- 所有 d 的选择必须通过 validation 完成；
- 不得因为某方法内存不足而静默删除样本。

若候选矩阵过大，应记录：

    infeasible_due_to_candidate_size

而不是自动切成 word blocks。

# 四、不同长度样本的结果如何比较

由于 n_i 不同，不能只比较原始 k90 或原始查询数。

精确结构指标同时报告：

    d90
    d95
    k90
    k95
    k90 / n_i
    k90 / Σ_{j=1}^{d_max} C(n_i,j)

其中：

- d90 是达到 R²≥0.90 所需的最低阶数；
- k90 是达到 R²≥0.90 所需的交互项数；
- 相对 k90 用于跨长度比较。

有限查询指标同时报告：

    test R²
    normalized RMSE
    minimum m for R²≥0.80/0.90/0.95
    m / n_i
    m / [n_i log2(n_i)]
    AUC over normalized query budget

统计时：

- 先保留每个 sample-level 结果；
- 按任务和 n_i 长度区间分组；
- 对同一样本做 paired comparison；
- bootstrap 以 sample 为单位；
- 不能把每个 mask 当成独立样本。

# 五、数据集安排调整

第一阶段优先使用同时包含短句和中长文本的数据。建议：SST2、RTN、Emotion、Eraser

# 六、结果目录需要增加的结构

features 目录中每个样本允许不同 n：

    results/features/<task>/<sample_id>.json

exact 全局枚举：

    results/values_exact_global/
      <task>/<sample_id>/
        values.npy
        masks.npy
        metadata.json

长文本 probe 枚举：

    results/values_exact_probe/
      <task>/<sample_id>/<probe_id>/
        values.npy
        masks.npy
        metadata.json

metadata.json 至少包含：

    n_total_words
    probe_size
    probe_indices
    probe_strategy
    conditioning_mode
    mask_operator
    value_function

完整词级有限查询：

    results/recovery/
      <method>/<task>/<sample_id>/
        n_<n_i>/
          budget_<m>/
            seed_<seed>/

aggregate 表必须包含字段：

    n_features
    experiment_scope
      = exact_global
      | exact_probe
      | limited_query_full
    probe_id
    probe_strategy
    conditioning_mode
    normalized_query_budget
    normalized_support_size

# 七、配置修改

删除以下主配置：

    featureization:
      type: balanced_contiguous_word_blocks
      fixed_n: 12

改为：

    featureization:
      type: lexical_words
      merge_subwords: true
      preserve_all_words: true
      punctuation_policy: attach_to_neighbor
      fixed_n: null

    exact_global:
      enabled: true
      max_n: 14
      min_n: 6
      length_bins:
        - [6, 8]
        - [9, 11]
        - [12, 14]

    exact_probe:
      enabled: true
      probe_sizes: [8, 10, 12]
      probes_per_sample: 5
      strategies:
        - contiguous_random
        - dispersed_random
        - stratified_position
      conditioning_mode: rest_present

    limited_query:
      use_full_word_features: true
      query_budget_type: n_log_n
      alphas: [0.25, 0.5, 1, 2, 4, 8]

此前确认的 value function 不变：

    value_function:
      type: predicted_class_margin
      verbalizer_length_normalization: mean
      target_class_source: full_input_prediction

此前确认的 GBT backend 不变：

    gbt:
      backend: sklearn

# 八、实施顺序调整

请按以下顺序修改任务计划：

1. 实现真实 lexical-word featureization；
2. 验证 word-to-subword mapping 和文本重建；
3. 在自然短句上完成 variable-n exact enumeration；
4. 完成 exact Möbius/Fourier 结构审计；
5. 在长文本上实现 conditional probe enumeration；
6. 比较短文本 global spectrum 与长文本 conditional local spectrum；
7. 在全部自然词上运行 variable-n limited-query recovery；
8. 按 n_i 和归一化查询预算汇总结果；
9. 只有结果支持假设后，再开发 adaptive Möbius recovery。

不要再通过合并单词将任意文本压缩到固定 n。

固定 n 或固定 k 只允许出现在：

- exact enumeration 的自然短文本筛选；
- 长文本的 probe set 大小；
- 计算资源受控的敏感性实验。

它不能作为主词级归因的 featureization 方法。