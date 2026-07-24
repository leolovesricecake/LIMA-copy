# Sparse deletion-Möbius 与 ProxySPEX 实验报告

## 1. 研究问题与本次比较

本实验考察两个问题：

1. 不施加 hierarchy 约束的低阶稀疏 deletion-Möbius 超图，能否在有限模型查询下形成可用的 LLM 局部归因？
2. 在相同模型、players、归因预算和评估协议下，它是否比带 Fourier hierarchy 结构直觉的 ProxySPEX 更适合这些文本分类任务？

主比较使用 Qwen2.5-7B-Instruct，word players、token evaluation、二阶交互、budget=512、k=8、seed=42，并解释完整输入的 predicted class。数据集与有效样本数为 SST-2 872、Emotion 2000、Rotten Tomatoes 1066。

Sparse 的 `target_probability + target=predicted` 与 ProxySPEX 的 `predicted_probability` 名称不同，但都计算

```text
softmax(z(S))[argmax_c z_c(N)]
```

因此它们是本报告的公平主比较组。Sparse 的 `predicted_class_margin` 结果只作为 value function 敏感性分析。

## 2. 指标读取修正

当前 evaluator 的顶层字段：

```text
eval_report.metrics_primary
```

是兼容字段，代码中固定由 `metrics_by_target.gold.metrics_primary` 展开。即使 `metric_settings.perturbation_target` 为 `predicted`，顶层 faithfulness 仍然是 gold-target 结果。

本报告统一读取：

```text
eval_report.metrics_by_target.predicted.metrics_primary
```

仅 `accuracy_full` 从顶层读取，因为它与 faithfulness target 无关。`scripts/collect_eval_reports.py` 已按这一规则修正，默认 `--target predicted`，并生成：

```text
mobius_verify/results/sparse_mobius/eval_summary_predicted.csv
results/baselines/proxyspex-copy/eval_summary_predicted.csv
```

## 3. 指标解释

- `AOPC`、`comprehensiveness`、`AOPC-comprehensiveness`：越高越好。删除高排名单元后，目标概率应下降更多。
- `sufficiency`、`AOPC-sufficiency`：这里是 `p(full)-p(keep-top)`，越低越好。负值表示只保留 top 单元时概率反而更高。
- `log_odds`：替换 top 20% 单元后的 `log p(perturbed)-log p(full)`，越低越好。
- `comprehensiveness` 和 `sufficiency` 是 top 20% 结果；AOPC 采用完整逐单元删除轨迹。

由于短文本使用 `floor(q% * unit_count)`，三个数据集的 q=1% 基本都没有选中单元，对应指标为 0，不能用于区分方法。

## 4. Faithfulness 主结果

### 4.1 汇总值

| Dataset | Method | AOPC ↑ | AOPC-C ↑ | Comp@20 ↑ | AOPC-S ↓ | Suff@20 ↓ | Log-odds ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SST-2 | Sparse Möbius | 0.4181 | 0.1851 | 0.3333 | 0.0096 | -0.0297 | -0.6997 |
| SST-2 | ProxySPEX | **0.4219** | 0.1834 | 0.3248 | 0.0137 | -0.0181 | -0.6460 |
| Emotion | Sparse Möbius | **0.3689** | **0.1502** | **0.2816** | **0.0240** | **0.0499** | -0.8432 |
| Emotion | ProxySPEX | 0.3577 | 0.1428 | 0.2613 | 0.0283 | 0.0651 | -0.8103 |
| Rotten Tomatoes | Sparse Möbius | 0.4268 | 0.1846 | 0.3144 | 0.0328 | 0.0278 | -0.8238 |
| Rotten Tomatoes | ProxySPEX | 0.4248 | 0.1829 | 0.3075 | 0.0322 | **0.0205** | -0.8035 |

三个任务的 macro average 全部略偏向 Sparse：AOPC +0.0031、AOPC-C +0.0036、Comp@20 +0.0119、AOPC-S -0.0026、Suff@20 -0.0065、Log-odds -0.0356。不过 macro average 主要受 Emotion 增益推动，不能代替逐数据集结论。

### 4.2 逐样本配对 bootstrap

下表为 `Sparse - ProxySPEX`，括号内是 95% paired bootstrap CI。AOPC/Comp 为正数有利，Suff/Log-odds 为负数有利。

| Dataset | AOPC Δ | Comp@20 Δ | Suff@20 Δ | Log-odds Δ |
|---|---:|---:|---:|---:|
| SST-2 | -0.0039 [-0.0083, 0.0009] | +0.0085 [-0.0010, 0.0189] | **-0.0116 [-0.0212, -0.0017]** | **-0.0537 [-0.0911, -0.0181]** |
| Emotion | **+0.0112 [0.0073, 0.0150]** | **+0.0202 [0.0128, 0.0276]** | **-0.0152 [-0.0237, -0.0065]** | -0.0329 [-0.0653, 0.0004] |
| Rotten Tomatoes | +0.0020 [-0.0028, 0.0070] | +0.0069 [-0.0051, 0.0188] | +0.0073 [-0.0050, 0.0204] | -0.0202 [-0.0705, 0.0291] |

补充指标中，SST-2 的 AOPC-S 对 Sparse 有显著优势 -0.0041 [-0.0079, -0.0004]；Emotion 的 AOPC-C 和 AOPC-S 也都显著有利于 Sparse。Rotten Tomatoes 所有主要差异的 CI 都跨 0。

据此应作分任务判断：

- **Emotion：Sparse 明确更好。** 删除、保留和完整轨迹指标方向一致，且多数达到配对显著。
- **SST-2：Sparse 更擅长找紧凑 top evidence。** Sufficiency 和 log-odds 显著更好，但完整 deletion AOPC 与 ProxySPEX 基本相当。
- **Rotten Tomatoes：总体持平。** Sparse 的删除指标均值略好，ProxySPEX 的保留指标略好，没有可靠统计差异。

### 4.3 不同删除比例

在 q=5%、10%、20% 时，Sparse 的 comprehensiveness 在三个数据集上都高于 ProxySPEX；说明它通常更早把影响预测的单元排到前面。到 q=50% 时，ProxySPEX 在 SST-2 和 Rotten Tomatoes 上反超，分别高 0.0140 和 0.0084。Emotion 在 q=50% 仍由 Sparse 高 0.0104。

Sufficiency 在 q=5% 和 10% 时也全部偏向 Sparse；Rotten Tomatoes 在 q=20% 和 50% 反转为 ProxySPEX 更好。这与主结果一致：Sparse 的优势主要集中在排名头部，ProxySPEX 在部分二分类任务上对排名尾部或大集合的组织更好。

## 5. 查询与运行成本

两种方法自行选择 attribution masks，但每个样本的逻辑归因查询数完全相同，因为预算和可枚举 coalition 上限相同。

| Dataset | Logical queries，二者相同 | Physical texts，Sparse / Proxy | Model forwards，Sparse / Proxy | 秒/样本，Sparse / Proxy |
|---|---:|---:|---:|---:|
| SST-2 | 472.5 | 467.7 / 471.7 | 234.8 / 235.9 | 12.12 / 12.22 |
| Emotion | 443.5 | 435.6 / 442.6 | 656.1 / 664.1 | 40.39 / 37.49 |
| Rotten Tomatoes | 469.2 | 464.6 / 468.1 | 233.3 / 234.1 | 12.89 / 11.71 |

Sparse 因文本/cache 去重，实际评分文本少 0.7%–1.6%，底层 forward 少 0.4%–1.2%。但稀疏设计矩阵、交叉验证和 refit 带来 CPU 拟合开销：Emotion 慢 7.7%，Rotten Tomatoes 慢 10.1%，SST-2 快 0.8%。三组归因总耗时约为 Sparse 29.19 小时、ProxySPEX 27.25 小时，Sparse 总体慢 7.1%。

Sparse 每个样本还执行约 9.4–10.3 个 targeted verification 逻辑查询。这些查询不参与归因拟合，也不计入 512 attribution budget；由于多数命中已有文本，包含验证后的 physical texts 仍与 ProxySPEX 接近。部署归因方法时可将 `targeted_top_k=0` 关闭这部分诊断成本。

`batch_calls` 不适合直接比较：Sparse 的 ValueOracle 每 16 行调用一次 batch API，ProxySPEX 把全部行交给一次外层 batch API，二者内部仍由 HFBackbone 拆成多个真实 forward。公平成本应看 logical queries、physical texts 和 `model_forward_calls`。

## 6. 稀疏恢复与交互结构

### 6.1 拟合诊断

| Dataset | Features 中位数 | Candidate 中位数 | Support 中位数 | Support/Candidate 中位数 | Train R² 中位数 | NRMSE 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 19 | 190 | 43 | 30.1% | 0.659 | 0.584 |
| Emotion | 17 | 153 | 39 | 24.0% | 0.681 | 0.565 |
| Rotten Tomatoes | 21 | 231 | 44 | 17.1% | 0.559 | 0.664 |

选择器收敛率为 99.0%、99.8%、99.2%，全部 Sparse run 均无失败或跳过样本。约 83%–90% 的样本真正使用满 512 queries；其余短样本因 `2^n < 512` 而精确枚举全部 coalition，因此不是被忽略。

这些结果说明有限查询通常能拟合出稳定、稀疏于完整二阶字典的模型，但 surrogate fidelity 只是中等水平，尤其 Rotten Tomatoes 较弱。Train R² 还是训练内指标，不能作为结构真实存在的充分证据。

### 6.2 精确 coefficient 验证

每个样本对估计绝对值最大的若干项，在完整输入附近额外查询 `2^|T|` 个删除组合，精确计算真实 deletion-Möbius coefficient。

| Dataset | Top coefficient 符号正确率 | 一阶符号正确率 | 二阶符号正确率 | Mean absolute error |
|---|---:|---:|---:|---:|
| SST-2 | 81.0% | 93.9% | 71.3% | 0.106 |
| Emotion | 81.0% | 93.1% | 72.4% | 0.125 |
| Rotten Tomatoes | 76.6% | 93.3% | 67.5% | 0.126 |

一阶项恢复可靠，二阶项明显更难但仍高于随机符号水平。有限查询已足以形成有用 ranking，却还不足以把所有估计超边当作可靠的机制事实。

### 6.3 Hierarchy 现象

Sparse 方法没有施加 hierarchy 约束。对其非零二阶边检查对应两个一阶父项：

| Dataset | 违反 strong hierarchy | 两个一阶父项均缺失 | “无父项”top 二阶边精确符号正确率 |
|---|---:|---:|---:|
| SST-2 | 46.6% | 18.2% | 65.6% |
| Emotion | 52.0% | 22.2% | 66.2% |
| Rotten Tomatoes | 71.0% | 38.3% | 61.7% |

这表明非层级二阶项并非少量边缘现象，而且其中一部分能通过真实 deletion-Möbius coefficient 的符号验证。它支持“强制 hierarchy 可能漏掉有用交互”的研究动机。

但这不是对 ProxySPEX Fourier hierarchy 的直接反证：这里检查的是 deletion-Möbius support hierarchy，两种 basis 的 support 不能逐项等同；另外，非层级项也可能包含有限样本拟合噪声。更准确的结论是：**允许非层级 Möbius 超边在归因任务上有实际价值，值得作为独立结构假设继续研究。**

## 7. 两种方法输出的一致性

两种方法虽然结构假设和采样不同，最终 word ranking 仍高度相关：

| Dataset | Top-8 平均重合率 | 全排名 Spearman 均值 | 完全相同排名 |
|---|---:|---:|---:|
| SST-2 | 82.6% | 0.823 | 6.2% |
| Emotion | 83.5% | 0.763 | 8.0% |
| Rotten Tomatoes | 79.5% | 0.791 | 5.4% |

因此 Sparse 的提升不是来自完全不同的解释，而是在相似主信号上重新排序部分重要单元。当前实验支持“可竞争或局部更优”，尚不支持“发现了完全不同且更真实的交互结构”。

## 8. Margin 敏感性分析

SST-2 和 Emotion 还运行了 Sparse `predicted_class_margin`：

- SST-2：相对 probability，AOPC 无显著变化；AOPC-C 下降 0.0027，AOPC-S 改善 0.0036，Suff@20 改善 0.0100，呈删除/保留权衡。
- Emotion：AOPC 显著下降 0.0068，但 log-odds 改善 0.0367；其他主要差异不稳定。
- Rotten Tomatoes 没有 margin run，不能形成完整三任务结论。

Margin 没有一致优于 probability。现阶段应继续用 predicted probability 作为与 ProxySPEX 的主比较 value，margin 保留为敏感性分析。

## 9. 评估一致性风险

SST-2 和 Rotten Tomatoes 两边逐样本 predicted label 完全一致。Emotion 有 32/2000 个样本的 predicted label 不一致，尽管模型路径、prompt、dtype 和 verbalizers 相同；两个 run 使用不同 GPU，且 `deterministic=false`。这意味着这 32 对样本实际解释的 target class 不同。

排除这 32 个样本后，Emotion 保留 1968 对：Sparse 的 AOPC +0.0099 [0.0068, 0.0130]、Comp@20 +0.0166 [0.0100, 0.0229]、Suff@20 -0.0134 [-0.0209, -0.0056]、Log-odds -0.0357 [-0.0676, -0.0067]，结论不变且 log-odds 也达到显著。

后续应让所有保存解释共享一次固定的 full-input prediction 表，或者在同一 GPU、确定性配置下统一重跑 evaluator。无需重跑 attribution。

## 10. 结论

1. **方法可行。** 在与 ProxySPEX 相同的主归因查询预算下，Sparse deletion-Möbius 在三个数据集上至少达到竞争水平，Emotion 明显更好，SST-2 在紧凑 top evidence 上更好，Rotten Tomatoes 持平。
2. **非层级交互有价值。** 大量入选二阶边违反 strong hierarchy，且一部分经精确 deletion coefficient 验证仍有正确方向；强 hierarchy 不是这些任务上显然安全的先验。
3. **有限查询恢复尚不完美。** 一阶项可靠，但二阶符号正确率只有 67%–72%，训练重构也只是中等。当前更适合把它称为有效 attribution surrogate，而不是已验证的真实机制超图。
4. **成本没有数量级劣势。** LLM 查询和 forward 略少，但 CPU 拟合使总墙钟时间约慢 7%。这属于可优化的求解器开销，不是模型查询瓶颈。
5. **尚不能宣布结构假设胜出。** 目前只有一个模型、一个 seed、一个预算，且方法排名高度相关。结果支持继续推进 Sparse Möbius，而非证明它普遍优于 Fourier hierarchy。

## 11. 建议的下一步

1. 固定一次 per-sample predicted target，重新评估现有 explanations，消除 Emotion 的 target 漂移。
2. 先在每个数据集固定子集上运行 seeds={0,1,2}、budgets={128,256,512}，画 faithfulness-query 曲线；确认趋势后再扩展全量，控制成本。
3. 增加 hierarchy-constrained Möbius ablation：保持 deletion basis、sampler、budget 和 solver 不变，只改变 hierarchy 约束，从而直接检验 hierarchy 本身。
4. 分开报告节点归因 faithfulness 与超边恢复质量；不要用前者替代后者。
5. 优化稀疏拟合：缓存 degree-2 design、减少重复 CV，或使用 warm-start 路径，重点降低 Emotion 的 13.7 秒/样本拟合开销。
