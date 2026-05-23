# LIMA-LLM 路线图（解释优先到生成扩展）

日期：2026-05-17  
适用范围：`lima_llm/` 主链路（判别式解释 -> 多数据集 -> 生成式迁移）

## 1. 当前状态与目标

当前主线已经具备：
- 判别式端到端链路（`chunking/objective/search/eval`）可运行。
- `ours/random/gradient` 三种方法可对比。
- 解释与评估均有基础可观测性，且近期优化已证明“等价优先 + 提速可行”。

当前主要短板：
- 子区域划分（`sentence`）在边界质量与长度适应性上不足。
- 打分分量信息只在 `trace` 的已选步上完整，缺少“全候选 chunk 分量级画像”。
- 贪心搜索仍有性能空间，但必须保持解释输出正确性。
- 数据集与任务形态仍集中在判别式（`sst2/eraser_movie_reviews`）。

目标：按依赖关系循序推进，先把“切分质量 + 分量诊断 + 正确提速”做稳，再扩到更多数据集与生成式任务。

## 2. 任务关系（依赖图）

任务关系不是并列，而是强依赖链：

1. **子区域划分优化**（任务1）是上游输入质量，直接决定后续打分与搜索的可解释性上限。  
2. **打分函数分析/优化**（任务2）依赖稳定切分，否则分量结论会被切分噪声污染。  
3. **贪心搜索优化**（任务3）依赖 1+2 的接口与语义冻结，否则速度收益无法归因。  
4. **拓展到其他数据集**（任务4）用于验证 1~3 的泛化稳定性。  
5. **拓展到生成式任务**（任务5）依赖 1~4 的工程与评估经验沉淀，不应提前并线替代主线。  

结论：建议采用 **1 -> 2 -> 3 -> 4 -> 5** 的主线顺序，4 可在 3 后并行小规模启动，5 单独立项。

## 3. 分阶段路线图

## Phase A：子区域划分方法优化（最高优先级）

目标：解决 `sentence` 误切分与长度不适配，建立可诊断、可比较的 chunking 体系。

实施项：
- 新增/重构切分策略层：
  - `sentence_v2`：使用 `PySBD` 后端（`pysbd==0.3.4`）提升引号、括号、换行、多标点边界质量；缺依赖时 fail-fast。
  - `adaptive`：基于文本长度动态切分，避免“短文单块/长文过碎”。
  - 保留现有 `sentence` 作为兼容基线，不直接替换默认行为。
- 增加切分诊断输出：
  - `chunk_count`、长度分布（min/mean/p90/max）、覆盖率、跨句断裂率、可疑边界样本。
  - 输出到样本 `metadata` 与 run 级汇总脚本。
- 增加切分可控参数：
  - `min_chunks`、`max_chunks`、`target_chunk_chars`（或 token 目标）等。

验收标准：
- 20样本人工抽检：你指出的错切分模式显著减少。
- 不破坏覆盖率与无重叠约束。
- 在 `sentence + greedy + lambda=1,1,1,1` 基线上，faithfulness 不出现系统性退化。

## Phase B：打分函数分量分析与优化

目标：回答“四项分数到底在起什么作用”，先诊断再改参数。

实施项：
- 增强分量记录（若尚未完整）：
  - 为每个 chunk 记录 `confidence/effectiveness/consistency/collaboration` 的 singleton 分量值。
  - 保留现有 `trace`，新增“全候选分量画像”字段。
- 系统化消融与相关性分析：
  - 完整法：`[1,1,1,1]`。
  - 逐项移除：`[0,1,1,1]`、`[1,0,1,1]`、`[1,1,0,1]`、`[1,1,1,0]`。
  - 结合 `lambda_sweep_report.py` 与 `trace_component_profile.py` 输出：
    - 五指标方向变化（`log_odds↑ / comp↑ / suff↓ / a-c↑ / a-s↓`）。
    - 与 `selected_chunk_ids/chunk_ranking` 稳定性关系。
- 形成一版“推荐权重区间”而非单点结论。

验收标准：
- 至少能给出每个分量的正/负贡献证据（相关性 + 消融）。
- 形成下一阶段搜索优化可复用的固定配置集（1~3组）。

## Phase C：贪心搜索优化（正确性前提下提速）

目标：在切分与打分配置稳定后，优化搜索吞吐，不牺牲解释一致性。

实施项：
- 保持语义不变的工程优化优先：
  - `evaluate_gains` 批处理与缓存复用深化。
  - 数据结构热路径优化（membership、key 构造、文本拼接复用）。
  - 观测增强：搜索阶段耗时与调用计数拆分。
- 明确禁区（当前阶段不做）：
  - 不引入近似贪心、随机剪枝、会改变候选组织顺序的重排策略。

验收标准：
- 解释级一致（同配置）：`selected_chunk_ids/chunk_ranking` 全一致，`trace.total_score` 在容差内。
- 在 200 样本上解释阶段时长有可复现实质改善（目标由阶段计划单列）。

### Phase C-Next 失败记录（2026-05-22-R）

失败现象：
- `adaptive` 机制修正后，20样本与后续复核均出现主指标整体退化，未满足“质量优先”目标。
- 同时未拿到可接受的稳定收益，属于“质量与效率双不达标”。

处置：
- 已执行非破坏式止损：`git revert aeff7a8`，恢复到上一步稳定语义基线（等价 `1f7f98b` 语义）。
- 当前主线继续使用 `sentence + greedy + lambdas=1,1,0,1`；`adaptive` 保留研发支线，不参与主结论。

经验与禁区：
- 任何 chunking 规则改动，必须先过 20 样本 deterministic 机制检查，再进入 200 样本验证；20 未过则禁止推进。
- 禁止在跨 commit、跨口径、跨配置结果上做因果归因结论；环境变化必须单列披露，不与算法结论混写。
- 若出现“整体退化 + 收益不显著”，优先回退，不做叠加修补。

## Phase D：拓展到其他数据集

目标：验证方法泛化，避免只在 `eraser_movie_reviews` 有效。

实施项：
- 优先级建议：`IMDB`（长文本）-> `Rotten Tomatoes`（中长）-> `Emotion`（多分类）-> 更多 ERASER 子任务。
- 数据适配统一：
  - 标签/口径映射、输入清洗、rationale 可用性检查。
  - 保持同一评估报告结构，方便跨数据集比较。
- 已落地（2026-05-22）：
  - `--dataset` 新增 `imdb / rotten_tomatoes / emotion`，并统一走 HF 加载。
  - `imdb` 在请求 `validation` 时自动映射到 `test`，其余数据集使用原生 split。
  - 样本 `metadata` 新增 `chunk_features_by_id` 与 `chunk_feature_coverage`，用于后续 chunking 机理分析。

验收标准：
- 每个新增数据集至少完成一次稳定复现实验与基线对比。
- 输出“数据集画像 -> 最优切分/权重建议”表。

## Phase E：拓展到生成式任务（独立立项）

目标：从“分类标签解释”迁移到“目标输出 token/span 解释”。

实施项：
- 抽象目标定义：`target_label` -> `target_output`（token 序列/span）。
- 扰动与评分迁移：
  - 以目标序列 logprob/稳定性为核心定义 faithfulness。
  - 重写或扩展评估协议，保留可对账 provenance。
- 数据与任务建议：先从短输出任务（分类式生成/短问答）启动，再到长输出。

验收标准：
- 完成一个生成式最小闭环（解释产物 + 评估 + 可复现）。
- 明确与判别式口径的可比边界，不混写结论。

## 4. 最近两轮执行清单（建议）

### Sprint-1（先做）
- Phase A：`sentence_v2` + `adaptive` + 切分诊断。
- Phase B：补全“每个chunk分量记录”与逐项移除实验脚本串联。

### Sprint-2（再做）
- Phase C：贪心热路径优化（仅等价优化）。
- Phase D：接入首个长文本数据集（建议 IMDB）做稳态复核。

## 5. 原则与边界

- 原则1：先观测后优化；每个优化点都要有可解释指标变化。  
- 原则2：先小样本验证语义，再大样本验证速度。  
- 原则3：环境变化（GPU/deterministic/commit）必须显式披露，不能与算法结论混写。  
- 原则4：当前主线不为分钟级提速牺牲解释正确性与指标稳定性。  

## 6. Adaptive 0523 结论与 V2.1 分支（2026-05-24）

最新跨数据集观测（`lima_llm_results-0523`）：
- `emotion / rotten_tomatoes / sst2`：`adaptive` 相比旧划分方法整体更优。
- `eraser_movie_reviews`：`adaptive` 指标整体退化。
- `imdb`：`adaptive` 大部分指标偏弱，方向不稳定。

机理结论（已固化为本轮研发假设）：
- 短文本收益主要来自 `effective_floor` 缓解 `top20_count=0`。
- 长文本退化主要与 `very_long` 的 guard 压缩策略相关，存在边界粗化风险。

V2.1 分支策略（`balanced_v2`，不替换主线）：
- `short`：floor 增加结构信号门控，降低误触发。
- `very_long`：guard 由 `hard_cap` 改为 `soft_band`，并加入边界保真约束。
- 新增机制字段：`adaptive_profile_version / adaptive_guard_mode / adaptive_fragmentation_target_min/max / adaptive_fragmentation_merge_ops`。

当前禁区（继续生效）：
- 禁止在未同口径（同配置、同 commit）结果上做强因果归因。
- 禁止以轻微耗时收益换取系统性 faithfulness 退化。
- `adaptive` 仍为实验支线；`sentence` 仍为稳定主线，待 V2.1 通过后再讨论替换策略。
