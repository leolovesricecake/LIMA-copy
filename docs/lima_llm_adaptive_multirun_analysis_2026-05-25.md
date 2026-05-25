# Adaptive 多版本复盘与下一轮优化计划（2026-05-25）

## 1. 数据与口径

本次复盘基于四组结果：
- `lima_llm_results-phase-d`
- `lima_llm_results-0523`
- `lima_llm_results-0524-balanced`
- `lima_llm_results-0524-balanced_v2`

统一口径：
- `log_odds = log(p_perturbed) - log(p_full)`，越小越好。
- 方向化目标：`log_odds↓ / comprehensiveness↑ / sufficiency↓ / aopc_c↑ / aopc_s↓`。
- 本文比较按你的约定，不将 GPU/commit 差异作为主干扰项，但仍保留 provenance 追溯。

## 2. 已确认事实

### 2.1 `0524-balanced` 与 `0523` 解释级等价

对 5 个数据集运行 `scripts/check_explain_equivalence.py`（`tolerance=1e-8`）结论：
- `selected_chunk_ids` 全一致
- `chunk_ranking` 全一致
- `trace.total_score` 全一致

覆盖样本数：
- `emotion=2000`
- `eraser_movie_reviews=200`
- `imdb=1000`
- `rotten_tomatoes=1066`
- `sst2=872`

结论：`0524-balanced` 与 `0523` 可视为同一行为版本。

### 2.2 `0524-balanced_v2` 相对 `0523` 的影响

核心趋势（见 `lima_llm_results-0524-balanced_v2/adaptive_cross_vs_0523.csv`）：
- `emotion`：5 项全部反向（显著退化），且运行时间变慢。
- `eraser_movie_reviews`：`LO/comp/aopc_c` 改善，但 `suff/aopc_s` 变差，整体未形成稳态优势。
- `imdb`：变化幅度很小但偏负面（`LO/suff` 方向不利）。
- `rotten_tomatoes`：多数指标小幅改善。
- `sst2`：变化极小，接近噪声级。

结论：`balanced_v2` 不具备跨数据集稳健性，不能作为主线默认。

## 3. 机理归因（从现有诊断字段反推）

### 3.1 短文本收益链

当 `effective_floor` 生效时：
- `chunk_count` 增加
- `top20_count=0` 比例下降
- 评估信号不再失敏

这在 `rotten/sst2/emotion` 类型短文本中通常带来 `LO/comp` 改善。

### 3.2 长文本风险链

在 `very_long` 桶，guard 的压缩/重打包策略一旦过强：
- chunk 边界粗化
- 搜索路径（`selected/ranking`）漂移上升
- `suff/aopc_s` 更易退化

`eraser/imdb` 的不稳定主要来自这条链路。

## 4. 版本决策

- 主线 adaptive 行为锁定为：`balanced`（等价 `0523/0524-balanced`）。
- `balanced_v2` 已在 Phase D-2 中硬移除（历史结果仅用于复盘，不再作为可运行配置）。
- `sentence` 主线不受影响，继续保持稳定路径。

## 5. 下一轮优化策略（稳健优先）

采用“单一全局参数空间 + 每数据集独立最优参数”的低预算搜索：

1. 先构造每数据集 `train/dev` 子集（默认 `80/40`）。
2. Stage-1：12 组 grid（覆盖 short-floor 与 very-long-guard 主轴）。
3. Stage-2：围绕 Stage-1 前 3 名做 8 组 random 局部扰动。
4. Dev 复选 top-k，按稳健规则排序：
   - 优先质量增益
   - 对 `eraser/imdb` 的明显反向退化施加惩罚
   - runtime 仅作 tie-break。

## 6. 主流程与工具链

主流程（推荐）：
- 使用 `python -m lima_llm` 单数据集内联搜索（`--hparam-search-split`）。
- 一个命令只处理一个数据集，数据集之间互不影响。
- 搜索候选始终包含 baseline；总 trial 数受 `--hparam-max-trials` 统一预算约束（超预算时确定性下采样）。
- `lambda` 搜索默认关闭；仅在显式传入 `--hparam-enable-lambda-search` 时启用。

辅助工具（可选）：
- `scripts/build_adaptive_tune_split.py`
- `scripts/search_adaptive_hparams.py`
- `scripts/adaptive_hparam_report.py`

说明：辅助脚本可用于离线批处理与复盘，但不再是主推荐流程。

## 7. 启动命令（Phase D-2）

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
```

20 样本 deterministic（机制检查）：

```bash
python -m lima_llm \
  --dataset eraser_movie_reviews \
  --split validation \
  --chunker adaptive \
  --adaptive-profile balanced \
  --hparam-search-split validation \
  --hparam-train-size 60 \
  --hparam-dev-size 40 \
  --hparam-search-method grid+random \
  --hparam-random-trials 8 \
  --hparam-max-trials 16 \
  --search greedy --k 8 --lambdas 1,1,0,1 \
  --run-eval --deterministic --max-samples 20 \
  --output-dir lima_llm_results-adaptive-tune
```

200 样本 production（稳态结论）：

```bash
python -m lima_llm \
  --dataset eraser_movie_reviews \
  --split validation \
  --chunker adaptive \
  --adaptive-profile balanced \
  --hparam-search-split validation \
  --hparam-train-size 60 \
  --hparam-dev-size 40 \
  --hparam-search-method grid+random \
  --hparam-random-trials 8 \
  --hparam-max-trials 16 \
  --search greedy --k 8 --lambdas 1,1,0,1 \
  --run-eval --max-samples 200 \
  --output-dir lima_llm_results-adaptive-tune
```

其余数据集替换两项即可：
- `imdb` 使用 `--split test --hparam-search-split train`
- `rotten_tomatoes / emotion / sst2` 使用 `--split validation --hparam-search-split train`

## 8. 禁区与闸门

- 不在跨口径结果上直接做因果归因。
- 新配置必须先过 20 样本 deterministic，再进入 200 样本。
- 不以小幅时长收益换取 `eraser/imdb` 的质量波动。
