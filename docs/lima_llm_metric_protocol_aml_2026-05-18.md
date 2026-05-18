# LIMA-LLM 指标协议与口径（AML）

日期：2026-05-18  
适用范围：`lima_llm/` 当前主线实现（`sentence + greedy`）

当前 Phase C 质量优先主线配置：`lambdas=1,1,0,1`。  
`lambdas=1,0,1,0` 仅作为速度优先备选，不替换主线结论。

## 1. 评估指标（Faithfulness）

记号：
- `p_full(y)`: 原文输入下目标类别 `y` 的概率。
- `p_remove_q(y)`: 删除 top-`q%` 重要单元后目标概率。
- `p_keep_q(y)`: 仅保留 top-`q%` 重要单元后目标概率。
- 默认 `q=20` 用于主指标；AOPC 采用 `q in {1,5,10,20,50}`。

### 1.1 `log_odds`（LO@20）
- 计算：`log_odds = log(p_perturbed(y)) - log(p_full(y))`
- 其中 `p_perturbed` 来自“将 top-20% 单元替换为 reference token”后的文本。
- 含义：重要单元被遮蔽后，目标对数概率下降幅度。
- 方向：**越小越好**（更负表示遮蔽后下降更明显，faithfulness 更强）。

### 1.2 `comprehensiveness`（Comp@20）
- 计算：`comp = p_full(y) - p_remove_20(y)`
- 含义：删掉最重要单元后性能下降多少。
- 方向：**越大越好**。

### 1.3 `sufficiency`（Suff@20）
- 计算：`suff = p_full(y) - p_keep_20(y)`
- 含义：只保留最重要单元时与原文的差距。
- 方向：**越小越好**。

### 1.4 `aopc_comprehensiveness`（A-C）
- 计算：对 `q in {1,5,10,20,50}` 的 `comp_q` 做 AML 平均：
  `A-C = (sum_q comp_q) / (|Q| + 1)`
- 含义：多删除比例下的综合 comprehensiveness。
- 方向：**越大越好**。

### 1.5 `aopc_sufficiency`（A-S）
- 计算：对 `q in {1,5,10,20,50}` 的 `suff_q` 做 AML 平均：
  `A-S = (sum_q suff_q) / (|Q| + 1)`
- 含义：多保留比例下的综合 sufficiency。
- 方向：**越小越好**。

### 1.6 `aopc`（删除曲线平均）
- 计算：按 ranking 逐步删除 top-k（从 0 到 m）后，取
  `aopc = mean_k (p_full(y) - p_delete_k(y))`。
- 含义：整体删除曲线下，目标概率平均下降幅度。
- 方向：**越大越好**。

## 2. 其他评估指标

- `accuracy_full`：原文分类准确率，越大越好。
- `plausibility_f1 / plausibility_iou`：与人工 rationale 的字符级重叠，越大越好。
- `sparsity`：选中解释文本字符占比，越小越稀疏（是否“更好”取决于任务偏好，不单独作为 faithfulness 优劣判据）。

## 3. 目标函数分量（解释搜索阶段）

当前目标函数：`F(S) = λ1*conf + λ2*eff + λ3*cons + λ4*col`，搜索时最大化 `F(S)`。

### 3.1 `confidence`
- 当前实现使用 `target_probability`（目标标签概率）作为 `conf`。
- 含义：子集 `S` 对目标标签的直接支持强度。
- 方向（在 `F(S)` 内）：**越大越好**。
- 备注：仓库中存在 `entropy` 版 `confidence_score` 辅助函数，但主线 objective 目前使用的是 `target_probability`。

### 3.2 `effectiveness`
- 计算：对子集内每个块，取其到其他块的最小语义距离并求和。
- 含义：鼓励子集语义互补，抑制冗余。
- 方向（在 `F(S)` 内）：**越大越好**。
- 备注：`|S|<=1` 时按定义为 `0`。

### 3.3 `consistency`
- 计算：`cos(embed(S), anchor_embed(full_text))`
- 含义：子集语义与原文语义锚点一致性。
- 方向（在 `F(S)` 内）：**越大越好**。

### 3.4 `collaboration`
- 计算：`1 - cos(embed(complement(S)), anchor_embed(full_text))`
- 含义：补集与原文锚点越不一致，说明 `S` 对保留关键信号越“不可替代”。
- 方向（在 `F(S)` 内）：**越大越好**。

### 3.5 `lambda=0` 行为
- 当某分量 `lambda_i=0` 时，对应分量应被**真实跳过计算**，并记录 `*_skipped_due_to_zero_lambda`。
- 该机制用于同时做质量消融与开销归因。

## 4. 稳定性与性能判读（实验汇总口径）

- 解释一致性：
  - `selected_chunk_ids` 一致性（是否逐样本完全一致）
  - `chunk_ranking` 一致性
  - `trace.total_score` 最大绝对差
- 性能：
  - 解释阶段总时长（样本 `metadata.elapsed_seconds` 聚合）
  - `explain_timing_breakdown` / `objective_compute_stats` / `search_profile` 分段耗时与调用计数
- 方向化 gain 定义：
  - 对“越大越好”指标：`gain = current - baseline`
  - 对“越小越好”指标：`gain = baseline - current`
  - 因此统一解释为：`gain > 0` 表示变好。
