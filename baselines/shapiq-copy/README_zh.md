# ProxySPEX LLM 基线

本目录保留 ProxySPEX 的原生实现。Runner 只接入 `mobius` 的数据、chunk、value function、schema-v2 结果和 evaluator，不改变 ProxySPEX 自己的：

- coalition sampler；
- LightGBM/XGBoost/tree proxy；
- Fourier 系数提取；
- Ridge refinement；
- Fourier 到 FBII 等 interaction index 的转换。

## 方法口径

默认配置：

```text
chunker=word
eval_granularity=token
value_function=predicted_probability
target_mode=predicted
index=FBII
max_order=2
budget=512
proxy_model=lightgbm
hpo=true
projector=signed_equal_share
```

ProxySPEX 输出超边 interaction。为了执行 token/word deletion 和 retention 评估，runner 将每条 interaction 有符号均分给其成员：

```text
score_i += interaction(T) / |T|, if i in T
```

解释 players 与评估 units 相互独立：`--chunker` 控制 ProxySPEX players，`--eval-granularity` 控制 faithfulness 扰动单元。

## 安装

```bash
pip install torch transformers datasets
pip install -e "baselines/shapiq-copy[proxy]"
```

## 运行

### sst2
```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --chunker word \
  --eval-granularity word \
  --value-function target_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/proxyspex-copy \
  --device cuda:0
```

### Rotten Tomatoes
```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset rotten_tomatoes \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --chunker word \
  --eval-granularity word \
  --value-function target_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/proxyspex-copy \
  --device cuda:0
```

### emotion
```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset emotion \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --chunker word \
  --eval-granularity word \
  --value-function target_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/proxyspex-copy \
  --device cuda:0
```

### eraser_movie_reviews
```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset eraser_movie_reviews \
  --split validation \
  --dataset-cache-dir /mnt/huawei/nsq/temp/hf \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --chunker word \
  --eval-granularity word \
  --value-function target_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/proxyspex-copy \
  --device cuda:0
```


数据 loader 会优先直接读取 `--dataset-cache-dir` 下的 Arrow split；找到完整缓存时不会先向 Hugging Face Hub 发起 metadata 请求。

短样本 smoke：

```bash
python baselines/shapiq-copy/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --model-path <model-path> \
  --max-samples 2 \
  --budget 16 \
  --proxy-model tree \
  --no-hpo \
  --device cuda:0
```

## Value Function

- `predicted_probability`：完整输入预测类别的概率，默认；
- `predicted_class_margin`：预测类别 raw verbalizer score 减最强竞争类别；
- `target_probability`：按 `--target-mode predicted|gold` 选择类别后取概率。

前两项始终强制 `target=predicted`。

## HPO 短样本修复

Proxy 拟合样本数为：

```text
min(budget, 2^n_players)
```

GridSearchCV 的折数根据真实拟合样本数缩小，并保证验证 fold 至少含两个样本；少于四个拟合样本时，该样本禁用 HPO，避免 `n_splits=5 > n_samples`。

## 输出

```text
results/baselines/proxyspex-copy/<dataset>/<model>/proxyspex/<run-id>/
├── run.json
├── status.json
├── metrics.json
├── curves-<target>.jsonl
├── samples/
└── diagnostics/       # output-level=debug
```

`samples/*.json` 中：

- `node_scores/ranking` 是投影后的节点解释；
- `method_summary.interaction_summary` 是高阶交互摘要；
- `method_summary.player_to_chunk_id` 是 ProxySPEX player 到文本 chunk 的映射；
- `attribution_cost` 记录原生 mask 查询、唯一扰动文本、model forwards 和耗时。

`metrics.json` 的 `target` 必须与待比较的 Möbius run 一致。
