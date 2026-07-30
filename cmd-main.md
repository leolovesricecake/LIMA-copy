# rtn

## 主方法

运行 SST-2 主配置：

```bash
python -m mobius.cli.run \
  --config configs/qwen3-8b/rotten_tomatoes.yaml \
  --device cuda:0
```

## P1

### 1. 运行 A 和 strict

A 是 degree-1 additive；
C 是主方法（degree-2 signed interaction），不必重新训练；B 也不重新训练，而是从 C 离线派生。

```bash
for seed in 42 43 44; do
  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-rtn/e2_a_additive.yaml \
    --seed "$seed" \
    --device cuda:0

  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-rtn/e4_strict.yaml \
    --seed "$seed" \
    --device cuda:0
done
```

### 2. 从 C 派生 B 并评估（训练完主方法，也就是后缀 std 那个就可以跑）

```bash
python scripts/derive_projection_run.py \
  --input-run results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --projector singleton_only \
  --output-root results/mobius \
  --run-suffix e2-b-fit-only
```

派生步骤复用 C 的 observations 和 surrogate，不产生模型调用，并保留 C 的 attribution cost。B 的 ranking faithfulness 需要单独评估，调用计入 evaluation cost：

```bash
python -m mobius.cli.evaluate \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --target predicted \
  --device cuda:0
```

### 3. 构建 shared held-out

建议每个 attribution seed 分别将 A/B/C/strict 放入同一 audit：

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o1-s42-e2-a-additive \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-e4-strict \
  --run-dir results/baselines/proxyspex/rotten_tomatoes/Qwen3-8B/proxyspex/b512-o2-s42-4cf0a836 \
  --output-dir results/audits/rotten_tomatoes/surrogate-heldout/s42 \
  --count-per-distribution 64 \
  --min-count 16 \
  --seed 42 \
  --device cuda:0

python scripts/evaluate_surrogates.py \
  --audit-dir results/audits/rotten_tomatoes/surrogate-heldout/s42
```

### 4. E3 精确验证（训练完主方法，也就是后缀 std 那个就可以跑）

```bash
python scripts/verify_interactions.py \
  --run-dir results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --top-k 5 \
  --top-k-per-parent-group 3 \
  --seed 42 \
  --device cuda:0
```

该 audit 对主方法 selected pair 和距离匹配的 random pair 查询四个完整输入附近的删除组合，精确计算 deletion-Möbius 二阶系数。查询不计入 attribution cost。

### 5. E4 hierarchy 分析

对每个 seed 分别运行：

```bash
python scripts/analyze_hierarchy.py \
  --none-run results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --strict-run results/mobius/rotten_tomatoes/Qwen3-8B/sparse_mobius/b512-o2-s42-e4-strict \
  --verification-dir results/audits/rotten_tomatoes/interactions/3c63dad0d84c \
  --heldout-audit results/audits/rotten_tomatoes/surrogate-heldout/s42 \
  --seed 42 \
  --device cuda:0

# --none-run：主方法，也就是 hierarchy=none 的完整 run。
# --strict-run：仅 hierarchy 改为 strict 的完整 run。
# --verification-dir：E3 audit 根目录，脚本实际需要其中的 rows.jsonl。
# --heldout-audit：shared held-out 根目录，需要 evaluation-index.json 和其引用的 metrics reports。
```

# emotion

## 主方法

```bash
python -m mobius.cli.run \
  --config configs/qwen3-8b/emotion.yaml \
  --device cuda:0
```

## P1

### 1. 运行 A 和 strict

A 是 degree-1 additive；
C 是主方法（degree-2 signed interaction），不必重新训练；B 也不重新训练，而是从 C 离线派生。

```bash
for seed in 42 43 44; do
  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-emotion/e2_a_additive.yaml \
    --seed "$seed" \
    --device cuda:0

  python -m mobius.cli.run \
    --config configs/qwen3-8b/mechanisms-emotion/e4_strict.yaml \
    --seed "$seed" \
    --device cuda:0
done
```

### 2. 从 C 派生 B 并评估（训练完主方法，也就是后缀 std 那个就可以跑）

```bash
python scripts/derive_projection_run.py \
  --input-run results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --projector singleton_only \
  --output-root results/mobius \
  --run-suffix e2-b-fit-only
```

派生步骤复用 C 的 observations 和 surrogate，不产生模型调用，并保留 C 的 attribution cost。B 的 ranking faithfulness 需要单独评估，调用计入 evaluation cost：

```bash
python -m mobius.cli.evaluate \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --target predicted \
  --device cuda:0
```

### 3. 构建 shared held-out

建议每个 attribution seed 分别将 A/B/C/strict 放入同一 audit：

```bash
python scripts/build_surrogate_holdout.py \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o1-s42-e2-a-additive \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-e2-b-fit-only \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-e4-strict \
  --run-dir results/baselines/proxyspex/emotion/Qwen3-8B/proxyspex/b512-o2-s42-41f8bf56 \
  --output-dir results/audits/emotion/surrogate-heldout/s42 \
  --count-per-distribution 64 \
  --min-count 16 \
  --seed 42 \
  --device cuda:0

python scripts/evaluate_surrogates.py \
  --audit-dir results/audits/emotion/surrogate-heldout/s42
```

### 4. E3 精确验证（训练完主方法，也就是后缀 std 那个就可以跑）

```bash
python scripts/verify_interactions.py \
  --run-dir results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --top-k 5 \
  --top-k-per-parent-group 3 \
  --seed 42 \
  --device cuda:0
```

该 audit 对主方法 selected pair 和距离匹配的 random pair 查询四个完整输入附近的删除组合，精确计算 deletion-Möbius 二阶系数。查询不计入 attribution cost。

### 5. E4 hierarchy 分析

对每个 seed 分别运行：

```bash
python scripts/analyze_hierarchy.py \
  --none-run results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-std \
  --strict-run results/mobius/emotion/Qwen3-8B/sparse_mobius/b512-o2-s42-e4-strict \
  --verification-dir results/audits/emotion/interactions/3c63dad0d84c \
  --heldout-audit results/audits/emotion/surrogate-heldout/s42 \
  --seed 42 \
  --device cuda:0

# --none-run：主方法，也就是 hierarchy=none 的完整 run。
# --strict-run：仅 hierarchy 改为 strict 的完整 run。
# --verification-dir：E3 audit 根目录，脚本实际需要其中的 rows.jsonl。
# --heldout-audit：shared held-out 根目录，需要 evaluation-index.json 和其引用的 metrics reports。
```
