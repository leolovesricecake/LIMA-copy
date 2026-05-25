# LIMA-LLM Adaptive V2.1 机理复盘归档与 Phase D-2 当前流程（2026-05-24，更新于 2026-05-25）

## 1. 状态说明（先读）

- `balanced_v2` 已在 Phase D-2 中硬移除，不再是可运行配置。
- 本文中的 V2.1 内容只保留“历史复盘价值”，用于解释过去为什么出现跨数据集不稳。
- 当前可执行流程以 `--adaptive-profile balanced` + `--hparam-search-split`（内联搜索）为准。

## 2. V2.1 历史复盘结论（归档）

历史现象（基于 `phase-d` 与 `0523`）：
- 短文本数据集（`emotion/sst2/rotten_tomatoes`）常因 `top20_count=0` 失敏，`floor` 机制可缓解。
- 长文本数据集（`eraser/imdb`）在 `very_long` 桶更容易被 guard 策略影响边界，导致 `suff/a-s` 不稳。

历史口径：
- `log_odds = log(p_perturbed) - log(p_full)`，越小越好。

归档原因：
- `balanced_v2` 在跨数据集对比中不够稳健（尤其 `emotion` 退化明显），因此被硬移除，不进入主线。

## 3. 当前推荐流程（Phase D-2 可执行）

### 3.1 不搜索（基线运行）

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python -m lima_llm \
  --dataset imdb \
  --split test \
  --chunker adaptive \
  --adaptive-profile balanced \
  --search greedy \
  --k 8 \
  --lambdas 1,1,0,1 \
  --run-eval \
  --max-samples 200 \
  --output-dir lima_llm_results-adaptive-baseline
```

### 3.2 内联超参搜索 + 最终评估（单数据集单命令）

说明：
- `--split` 是最终评估集。
- `--hparam-search-split` 是搜索集；为空表示不搜索。
- 默认搜索子集大小是 `train=60 / dev=40`，可通过参数覆盖。
- 若搜索集与评估集相同，系统会先抽走搜索样本，再从剩余样本做评估，最后才应用 `--max-samples`。
- 搜索候选永远包含 baseline（`adaptive_overrides={}` + 当前 `--lambdas`）。
- 默认总 trial 预算由 `--hparam-max-trials` 控制（默认 `16`）；超预算时做固定 seed 的确定性下采样。
- `lambda` 搜索能力已支持，但默认关闭；启用时需加 `--hparam-enable-lambda-search`。

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python -m lima_llm \
  --dataset imdb \
  --split test \
  --chunker adaptive \
  --adaptive-profile balanced \
  --adaptive-overrides-json "" \
  --hparam-search-split train \
  --hparam-train-size 60 \
  --hparam-dev-size 40 \
  --hparam-search-method grid+random \
  --hparam-random-trials 8 \
  --hparam-max-trials 16 \
  --search greedy \
  --k 8 \
  --lambdas 1,1,0,1 \
  --run-eval \
  --max-samples 200 \
  --output-dir lima_llm_results-adaptive-inline-search
```

可选参数空间文件（JSON）格式：

```json
{
  "parameters": {
    "short_max_words": [96, 120],
    "medium_max_words": [384, 448],
    "long_max_words": [960, 1152],
    "min_effective_chunks": [4, 5],
    "short_floor_min_words": [12, 15],
    "short_floor_signal_mode": ["always", "structural"],
    "guard_mode": ["hard_cap", "soft_band"],
    "ratio_threshold": [20, 24, 28],
    "fragmentation_target_words": [28, 32]
  }
}
```

使用时增加：

```bash
--hparam-space-file path/to/adaptive_space.json
```

当需要搜索 lambda 时，在 space 文件加入 `lambda1~lambda4`，并显式开启：

```bash
--hparam-enable-lambda-search
```

### 3.3 搜索结果产物位置

- 运行目录下会生成 `hparam_search/` 子目录。
- 关键文件：
  - `hparam_search/search_summary.json`
  - `hparam_search/train_ids.json`
  - `hparam_search/dev_ids.json`
  - `hparam_search/eval_ids.json`

最终 `eval_report.json` 会附带：
- `hparam_search_enabled`
- `hparam_search_split`
- `best_adaptive_overrides`
- `best_lambdas`
- `lambda_search_enabled`
- `hparam_search_summary_path`

## 4. 分析脚本（仍可用）

- `scripts/adaptive_cross_dataset_report.py`
- `scripts/adaptive_effect_model.py`
- `scripts/adaptive_case_miner.py`

用途：
- 对比旧 run 与新 run 的质量与机制字段差异（bucket/floor/guard/top20/chunk_count）。
- 从样本级定位退化来源，避免“整体指标好看但局部机制恶化”。

## 5. 执行禁区（继续生效）

- 不再使用 `--adaptive-profile balanced_v2`（会直接参数报错）。
- 不在跨口径结果上做强因果归因。
- 不以小幅耗时收益交换跨数据集质量退化。
