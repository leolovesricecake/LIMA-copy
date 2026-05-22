# LIMA-LLM Adaptive Chunking RCA 与调优记录（2026-05-22）

## 1. 背景与结论摘要

本记录针对 `chunker=adaptive` 在 `ERASER movie_reviews` 上的表现做机制级分析，并给出已落地改造与后续计划。

当前对比（你给定的 0518 结果锚点）：

- `sentence`：`log_odds=-0.29462, comp=0.09455, suff=-0.12039, a-c=0.06930, a-s=-0.03457`
- `sentence_v2`：`log_odds=-0.26433, comp=0.09733, suff=-0.12312, a-c=0.06972, a-s=-0.04134`
- `adaptive`：`log_odds=-0.31668, comp=0.10465, suff=-0.11422, a-c=0.07374, a-s=-0.03891`

观察：

1. `adaptive` 对 `LO/comp/a-c` 有优势。  
2. `adaptive` 在 `suff` 上有退化（相对 `sentence` 变大，绝对值变小）。  
3. `adaptive` 结构错误明显下降（`sentence` 的错误簇远高于 `adaptive`），说明其“切分质量修复”方向有效。

## 2. 指标受影响机理（本轮归因）

### 2.1 影响路径

主要传导链路：

`chunk 边界/粒度 -> chunk_count -> q=20% 的 top_count=floor(0.2*m) -> 扰动规模与保留规模 -> p_remove/p_keep -> faithfulness 指标`

在当前评估定义下：

- `comp = p_full - p_remove`（越大通常越好）
- `suff = p_full - p_keep`（越小通常越好）
- `log_odds = log(p_perturbed) - log(p_full)`（本实现下更负通常对应更强扰动效果）

因此当 `chunk_count` 被系统性抬升时，`top_count` 会随之上升，容易机械性推高某些扰动强度指标（特别是 `comp`），同时可能损伤 `suff` 稳定性。

### 2.2 本轮关键机制问题

`adaptive` 的核心异常集中在 `very_long` 桶：

- 存在“单段超长文本（raw≈1）在后处理中被大量拆碎”的情况。  
- 实测中可见 `raw -> final` 膨胀显著，导致：
  1. 评估侧 `top_count` 偏移；
  2. 搜索侧候选形态改变，`selected/ranking` 漂移放大；
  3. 最终出现 `LO/comp` 变好但 `suff` 变差的组合。

### 2.3 正向机制（为什么 adaptive 仍值得继续）

`adaptive` 同时具备明确收益：

- `orphan punctuation`、`leading close punct` 等错误簇显著下降；
- 中长文本的切分更灵活，避免 `sentence` 在边界噪声与长度适配上的弱点。

结论：问题不是“adaptive 无效”，而是“very_long 路由与后处理顺序需要收敛”。

## 3. 本轮已落地改造

### 3.1 机制分析工具补强

1. 新增 `scripts/adaptive_mechanism_report.py`：统一产出三组 run 的机制对比（指标 raw+方向化、chunk/top20 分布、very_long 膨胀、漂移、错误簇）。  
2. 扩展 `scripts/phase_a_chunking_compare.py`：新增
   - `top20_count_mean/p90`
   - `top20_count_delta_mean/p90`
   - `very_long_fragmentation_ratio_mean`、`very_long_fragmentation_delta_mean`
   - 新后处理统计字段聚合。

### 3.2 Adaptive 算法定向修复

在 `lima_llm/chunking/adaptive.py` 落地以下保守改造（不改评估定义）：

1. `very_long` 路由修正：当段落切分 `raw<=1` 时，改走“句子种子 + 词预算打包”路径，避免直接进入超长单块拆分风暴。  
2. 后处理顺序升级：`long_split` 后追加一轮 `invalid_merge + short_merge`，清理拆分后新增碎片。  
3. `very_long` 相邻块打包：按词预算贪心合并相邻小块，限制过碎化。  
4. 新增可观测字段：
   - `adaptive_very_long_single_paragraph_fallback_used`
   - `adaptive_postprocess.post_long_invalid_merge_count`
   - `adaptive_postprocess.post_long_short_merge_count`
   - `adaptive_postprocess.adjacent_pack_merge_count`
   - `adaptive_stage_chunk_counts.after_post_long_merge`

## 4. 当前 adaptive 机制评价（优缺点）

### 优点

1. 低成本特征驱动，推理开销可控。  
2. 结构性切分错误显著下降。  
3. 对中长文本的分层策略比固定 `sentence` 更具扩展性。

### 缺点/风险

1. `very_long` 路由对单段超长文本敏感，容易触发分块形态骤变。  
2. `chunk_count` 与 `top_count` 耦合导致评估指标可能出现“机械性上升/下降”。  
3. 分块形态变化会被贪心过程放大成较大的选择漂移。

## 5. 后续计划与验收口径

## 5.1 短期（本轮后）

1. 先跑 20 样本 deterministic 验证机制字段：
   - `very_long` 膨胀是否显著收敛；
   - 新增后处理计数是否触发且可解释。  
2. 通过后跑 200 样本：对照 0518 锚点看
   - 质量：保住 `LO/comp` 优势，同时改善 `suff/a-s`；
   - 稳定性：`failed_samples==0`；
   - 速度：解释阶段不明显变慢。

## 5.2 中期（如需继续调参）

1. 在不改评估定义前提下，优先调 `very_long` 打包预算与二次 merge 阈值。  
2. 对 `top20_count` 漂移设置软约束观测（先监控，不做硬闸门）。  
3. 若仍有 `suff` 顽固退化，再考虑分桶差异化阈值（仅 `very_long`）。

## 6. 结果解读边界

当前文档基于 `0518` 已有结果做归因与迭代锚定。该锚点中不同 run 的 commit 并非完全一致，已在结论中显式标注；后续若做严格科研归因，需要同 commit 同配置复核一轮。
