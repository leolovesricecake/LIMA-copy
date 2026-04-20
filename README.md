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
  --k 8 \
  --chunker sentence \
  --search greedy \
  --max-samples 100 \
  --output-dir lima_llm_results \
  --run-eval
```

## 原始 LIMA 说明

原始代码已迁入 `lima_origin/`。若需使用历史图像版流程，请参考：

- `lima_origin/README_origin.md`
- `lima_origin/说明.md`
