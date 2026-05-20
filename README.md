# LIMA-main (LLM Transfer v1)

当前分支以文本 LLM 迁移实现为主，核心代码在 `lima_llm/`。

## 目录结构

- `lima_llm/`: 文本版 LIMA v1（数据适配、chunking、backbone、子模目标、搜索、评估、pipeline）
- `scripts/run_lima_llm_v1.sh`: 文本版统一启动脚本
- `tests/`: 文本版单测与集成测试
- `lima_origin/`: 原始图像版 LIMA 代码与历史脚本（归档）

## 快速开始

```bash
# 1) Dry-run（无需大模型）
python -m lima_llm --dataset sst2 --split validation --mock-backbone --dry-run 10

# 2) 端到端（mock，含评估）
bash scripts/run_lima_llm_v1.sh \
  --dataset sst2 \
  --split validation \
  --mock-backbone \
  --deterministic \
  --k 8 \
  --chunker sentence \
  --search greedy \
  --explain-method ours \
  --max-samples 100 \
  --output-dir lima_llm_results \
  --run-eval
```

`--explain-method` 支持 `ours/random/gradient`。每次运行只处理一个方法，不同方法使用独立输出目录，可在不同 GPU 并行运行。
`--chunker` 当前支持 `sentence/sentence_v2/fixed_token`，其中 `sentence_v2` 使用 `PySBD`（`pysbd==0.3.4`）作为分句后端。若未安装 `pysbd`，`sentence_v2` 会 fail-fast 并提示安装命令。
安装示例：`pip install pysbd==0.3.4`

`--deterministic` 可开启更强确定性设置（用于回归对账/复现实验）。运行产物 `run_config.json`、`eval_config.json`、`eval_report.json` 会包含 `provenance` 字段（git/命令/环境/时间信息）。

可用 `python scripts/analysis_snapshot.py --results-root lima_llm_results --primary-method ours --reference-method gradient` 生成单组对账快照（JSON/CSV）。

可用 `python scripts/lambda_sweep_report.py --baseline-run-dir <baseline_run_dir> --candidate-run-dirs <run_dir_1> <run_dir_2>` 生成 `lambda` 小网格的 Faithfulness 五指标方向判定与速度/稳定性副作用汇总。

可用 `python scripts/trace_component_profile.py --run-dir <run_dir>` 聚合每步 `confidence/effectiveness/consistency/collaboration` 分量轨迹，辅助定位 `comp/suff` 的主要牵引项。

可用 `python scripts/phase_a_chunking_compare.py --baseline-run-dir <sentence_run_dir> --candidate-run-dir <sentence_v2_run_dir>` 生成 Phase A 的切分质量/解释漂移/性能对比报告。

可用 `python scripts/chunk_error_audit.py --run-dir <run_dir>` 对单个 run 的 `samples/*.json` 做 chunk 错误全量审计，输出 `chunk_error_audit.json` 与 `chunk_error_manifest.csv`。


## 20 - deterministic
```
python -m lima_llm \
  --dataset eraser_movie_reviews \
  --split validation \
  --eraser-root hf://eraser-benchmark/movie_rationales \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --k 8 \
  --chunker sentence \
  --search greedy \
  --lambdas 1,1,1,1 \
  --dataset-cache-dir /mnt/huawei/nsq/LIMA-copy/datasets \
  --resume-check strict \
  --run-eval \
  --explain-method ours \
  --deterministic --max-samples 20 \
  --output-dir lima_llm_results-20-0515-deter --device cuda:1
```


## 20 - prefetch sort
```
LIMA_EVAL_PREFETCH_LENGTH_SORT=1 python -m lima_llm \
  --dataset eraser_movie_reviews \
  --split validation \
  --eraser-root hf://eraser-benchmark/movie_rationales \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --k 8 \
  --chunker sentence \
  --search greedy \
  --lambdas 1,1,1,1 \
  --dataset-cache-dir /mnt/huawei/nsq/LIMA-copy/datasets \
  --resume-check strict \
  --run-eval \
  --explain-method ours \
  --deterministic --max-samples 20 \
  --output-dir lima_llm_results-20-0515-deter-sort --device cuda:1
```
