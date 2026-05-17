# LIMA-LLM P1 Drift RCA (2026-05-17)

## 1) 现象
- 对比基线 `lima_llm_results-0516-io`，P1 结果在主指标出现系统性漂移：
  - `log_odds/comprehensiveness/sufficiency` 绝对差约 `0.003~0.004`。
  - `aopc` 绝对差约 `4.12e-4`。
- 速度改善有限：
  - deterministic 运行约 `+1.76%`。
  - non-deterministic 运行约 `+3.93%`。
- 解释语义发生大范围变化：
  - `selected_chunk_ids` 变化样本占比 `79%`（158/200）。
  - `chunk_ranking` 变化样本占比 `96%`（192/200）。

## 2) 关键证据
- `p1-deter` 与 `p1`（同 commit）指标与样本输出完全一致，说明漂移不是随机噪声，而是计算路径改变导致的系统性偏移。
- `prefetch/cache` 统计与口径保持一致，漂移不来自评估定义变化。
- P1 commit 对核心推理路径的变更：
  - 单次 `predict_label_probs_batch` 级别一次性 tokenize。
  - label 融合单次前向（将 `(text,label)` 笛卡尔展开后一次 forward）。

## 3) 根因判断
- 根因不是 deterministic 开关，也不是数据口径变化。
- 根因是 backbone 推理路径数值组织变化（batch 形状与计算组织改变），导致目标概率产生小幅但系统性的变化；该变化在贪心搜索中被放大，最终导致解释集合显著变化并反馈到评估指标。

## 4) 处置决策
- 回退到 `best-0516 (b5eabe0)` 语义与实现。
- 不保留本轮 P1 的融合前向优化与附加 profiling 脚本。
- 后续优化遵循“解释级严格等价优先”，再谈速度收益。

## 5) 后续约束（重启优化）
- 每个优化点必须先过 20 样本解释等价：
  - `selected_chunk_ids` 全量一致。
  - `chunk_ranking` 全量一致。
  - `trace.total_score` 在 `1e-8` 容差内一致。
- 再过评估等价：`log_odds/comprehensiveness/sufficiency/aopc` 绝对差均 `<= 5e-4`。
- 仅当以上两项通过，才进入 200 样本速度对比。
