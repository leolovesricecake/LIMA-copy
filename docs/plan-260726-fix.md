# 一、删除 target class 相关修改

不再考虑为每个样本建立共享 target manifest，也不需要增加跨实验 target-label 检查。

原因：

1. 当前各方法使用相同模型、prompt、verbalizer 和完整输入预测规则；
2. 之前出现的 target 不一致来自旧历史结果，不属于当前实验协议问题；
3. 每个实验独立运行、独立保存和独立汇总，没有必要为了极少发生且当前已排除的问题增加额外工程；
4. 后续实验默认 target 计算逻辑一致。

因此从计划中删除：

- shared target manifest；
- target consistency assertion；
- 不一致样本重新计算；
- 因 target 不一致而排除样本的流程。

不要让后续实验或汇总脚本依赖跨实验 target 检查。

# 二、明确三种 masks，不要混淆

实验中存在三种不同用途的 masks。

## 1. Attribution training masks

用于向原模型查询并拟合归因方法。这部分属于方法机制：

- Sparse Möbius 使用自己的 sampler；
- ProxySPEX 使用自己的 sampler；
- 两者不共享训练 masks；
- 两者只需要使用相同的 attribution budget。

## 2. Ranking-based deletion evaluation masks

由每种方法自己的词排名产生。也不共享具体 masks，否则无法评价两种排名的差异。

它们只共享：

- 删除比例；
- mask operator；
- value function；
- AOPC、AUPC、comprehensiveness、sufficiency 等计算协议。

## 3. Surrogate held-out test masks

这是一批独立于方法训练和词排名的测试 masks，用于检验：学到的 surrogate 是否能预测未查询 coalition 上的原模型输出。

理论上来说，这部分 masks 需要共享，才能保证 R^2、NRMSE 和 MAE 可比。但是实际实现难度大，因为：
- 需要先排除所有训练 masks 才能计算 held-out masks；
- 而每个方法的实验独立进行，独立产生训练 masks；
- 我们要对比好几种方法，比如 degree-1 vs degree-2，sparse vs proxySPEX。

因此，我们有两种方案：

### 不严格 held-out
每个方法独立产生 held-out masks 并计算相关指标，不共享。
当 masks 数量足够时，理论上效果差不多。

### 严格 held-out
1. samples 需要保存训练 masks；
2. 实现一个脚本，加载多个方法的结果目录，逐样本匹配，整合所有训练 masks 后，生成 held-out masks，并调用模型得到输出，作为测试数据。
3. 再实现一个脚本，加载测试数据与指定方法的 surrogate，测试相关指标。

# 三、调整后的实验计划

## E1：总体归因有效性

保持：

- 多数据集；
- 多模型；
- 多预算；
- AOPC、AUPC、aopc-comprehensiveness、aopc-sufficiency；
- 查询量、运行时间和失败率。

## E2：交互必要性

分为：

A. degree-1 vs degree-2

比较 held-out surrogate 指标，回答交互是否改善 masked function 拟合。

B. degree-2 singleton-only vs signed projector

比较词级 faithfulness，回答交互是否应该用于最终词排名。

（这里我有疑问：为什么两项的比较指标不同？是不是操作也不同？）

## E3：交互真实性

对 top 二阶边计算精确 deletion coefficient，并与匹配随机词对比较。

保存：

    estimated_coefficient
    exact_coefficient
    sign_correct
    absolute_error
    four queried values

## E4：Hierarchy

实现真正的：

    hierarchy = none
    hierarchy = strict

当前已有的父项筛选方案改名为，不能继续称为 strict hierarchy。

第一版不做 matched-support hierarchy，而报告：

    total_support_size
    singleton_support_size
    pair_support_size

## E5：可压缩性和恢复效率

只有 held-out protocol 完成后再做：

- R²-support size 曲线；
- R²-budget 曲线；
- faithfulness-budget 曲线；
- 达到指定 R² 所需项数和查询数。

## E6：消融和敏感性

保留：

- basis；
- projector；
- hierarchy；
- refit；
- sampler；
- degree；
- budget；
- regularization。

# 四、执行顺序
1. 修改实验协议
2. 在 SST-2/Qwen3-8B 上完成 E2–E5
3. 拓展到其他数据集和模型
4. 完成 E6