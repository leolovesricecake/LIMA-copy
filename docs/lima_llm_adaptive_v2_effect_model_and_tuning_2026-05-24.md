# LIMA-LLM Adaptive V2.1 机理建模与定向改进（2026-05-24）

## 1. 问题复盘（基于 `phase-d` 与 `0523`）

已观测到的稳定现象：
- `emotion / rotten_tomatoes / sst2`：`adaptive` 新版整体优于旧划分。
- `eraser_movie_reviews`：新版 `adaptive` 整体退化。
- `imdb`：新版 `adaptive` 大部分指标偏弱，方向不一致。

核心触发链（本轮固定口径）：
1. `chunk` 形态变化（块数、边界、长块压缩策略）  
2. 影响 `q=20%` 的 `top20_count=floor(0.2*chunk_count)`  
3. 改变搜索路径（`selected_chunk_ids` / `chunk_ranking`）  
4. 在 faithfulness 指标上体现为数据集差异化收益或退化

`log_odds` 方向说明（按实现）：
- `log_odds = log(p_perturbed) - log(p_full)`，因此**越小越好**。

## 2. 不同文本类型下的 adaptive 行为画像

短文本、弱结构（emotion/sst2/rotten 常见）：
- 旧版常出现 `top20_count=0` 或单块退化，评估信号不足。
- 新版通过 `effective_floor` 增加可选块，faithfulness 有明显改善。
- 风险：过度触发会拉高解释耗时。

长文本、强结构（eraser/imdb 常见）：
- `very_long + raw<=1` 时，旧 guard 的“硬上限压缩”容易把块数压得过低。
- 过度压缩会粗化语义边界，导致 sufficiency / A-S 等指标反向波动。
- 风险主要来自 `fragmentation_guard` 的压缩策略，而非随机噪声。

## 3. 本轮改造（Adaptive V2.1）

实现目标：
- 保留短文本收益。
- 优先修复长文本退化（eraser/imdb）。

改造点：
- 新增 `--adaptive-profile balanced_v2`（`balanced` 保持不变）。
- `short` 桶：最小有效分块（5块）改为“结构信号门控触发”：
  - 条件：`word_count>=15` 且满足以下其一  
    - `sentence_end_count_est>=1`
    - `punct_count>=2`
    - `newline_count>=1`
- `very_long` 桶：guard 从 `hard_cap` 改为 `soft_band`（仅在过碎时收敛到目标区间）。
- `raw<=1` 时引入“句子种子优先”路径，再做受控相邻合并。
- 相邻合并引入边界保真约束：优先合并弱边界，尽量避免跨强句界/括号引号断点。

新增诊断字段（兼容旧字段）：
- `adaptive_profile_version`
- `adaptive_effective_floor_condition_met`
- `adaptive_fragmentation_target_min/max`
- `adaptive_fragmentation_merge_ops`
- `adaptive_guard_mode`（`off/soft_band/hard_cap`）

## 4. 机制建模脚本与结论模板

新增脚本：
- `scripts/adaptive_effect_model.py`  
  输入：`baseline_root` 与 `candidate_root`。  
  输出：
  - 指标差异（raw + directional gain）
  - 分桶机理（`short/medium/long/very_long`）差异
  - 干预分组机理（`floor_only/guard_only/both/none`）漂移与形态统计
  - `adaptive_effect_model.json/csv`

扩展脚本：
- `scripts/adaptive_cross_dataset_report.py`
  - 支持跨根目录对比：`--baseline-root` + `--candidate-root`
- `scripts/adaptive_case_miner.py`
  - 新增按触发机制的回退样本榜单（`trigger_kind`）

建议解读顺序：
1. 看 `adaptive_effect_model.csv` 的跨数据集方向结果；
2. 对退化数据集看 `trigger_groups` 与 `bucket_groups`；
3. 用 `adaptive_case_miner` 定位样本级反例并回看 chunk 边界。

## 5. 实验命令（可直接运行）

环境：
```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
```

建议把 `balanced` 与 `balanced_v2` 输出到**不同**目录，避免同路径覆盖：

20样本 deterministic（机制检查，5数据集）：
```bash
# eraser
python -m lima_llm --dataset eraser_movie_reviews --split validation --chunker adaptive --adaptive-profile balanced --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced
python -m lima_llm --dataset eraser_movie_reviews --split validation --chunker adaptive --adaptive-profile balanced_v2 --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced_v2

# imdb
python -m lima_llm --dataset imdb --split test --chunker adaptive --adaptive-profile balanced --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced
python -m lima_llm --dataset imdb --split test --chunker adaptive --adaptive-profile balanced_v2 --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced_v2

# rotten_tomatoes
python -m lima_llm --dataset rotten_tomatoes --split validation --chunker adaptive --adaptive-profile balanced --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced
python -m lima_llm --dataset rotten_tomatoes --split validation --chunker adaptive --adaptive-profile balanced_v2 --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced_v2

# emotion
python -m lima_llm --dataset emotion --split validation --chunker adaptive --adaptive-profile balanced --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced
python -m lima_llm --dataset emotion --split validation --chunker adaptive --adaptive-profile balanced_v2 --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced_v2

# sst2
python -m lima_llm --dataset sst2 --split validation --chunker adaptive --adaptive-profile balanced --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced
python -m lima_llm --dataset sst2 --split validation --chunker adaptive --adaptive-profile balanced_v2 --search greedy --k 8 --lambdas 1,1,0,1 --run-eval --max-samples 20 --deterministic --output-dir lima_llm_results-0524-adaptive-balanced_v2
```

200样本 production：将 `--max-samples 20 --deterministic` 改为 `--max-samples 200`。

汇总与建模：
```bash
python scripts/adaptive_cross_dataset_report.py --baseline-root lima_llm_results-0524-adaptive-balanced --candidate-root lima_llm_results-0524-adaptive-balanced_v2
python scripts/adaptive_effect_model.py --baseline-root lima_llm_results-0524-adaptive-balanced --candidate-root lima_llm_results-0524-adaptive-balanced_v2
```

样本级回退定位（按机制）：
```bash
python scripts/adaptive_case_miner.py \
  --baseline-run-dir lima_llm_results-0524-adaptive-balanced/eraser_movie_reviews/model-Qwen2_5-7B-Instruct/chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours \
  --candidate-run-dir lima_llm_results-0524-adaptive-balanced_v2/eraser_movie_reviews/model-Qwen2_5-7B-Instruct/chunk-adaptive_search-greedy_k-8_lam-1-1-0-1_seed-42_method-ours
```

## 6. 验收重点（本轮）

- 长文本优先：`eraser/imdb` 5项指标至少4项方向改善。
- 短文本守底：不接受系统性退化（单项反向显著退化需单列解释）。
- 机制一致性：
  - `short`：`top20_count_zero_ratio` 不反弹；
  - `very_long`：guard 触发后不再出现“过度压缩”。
