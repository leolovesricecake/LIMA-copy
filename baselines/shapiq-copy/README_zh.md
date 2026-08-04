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
eval_granularity=word
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

Runner 与主方法共享 `task_classification_v1` prompt：明确数据集任务、列出全部候选 verbalizers，并要求只输出一个标签。ProxySPEX 仍然只负责选择 coalition masks 和拟合 proxy；每个 coalition text 由共享 scorer 自动放入固定任务 prompt，因此这项修改不改变 ProxySPEX 的采样、树模型或 Fourier/FBII 核心流程。prompt 版本写入 run 配置，旧无任务 prompt 的结果不会被断点续跑。

正式运行前先用对应主配置执行分类预检，例如：

```bash
python scripts/check_classifier.py \
  --config configs/paper-v2.3/qwen3-8b/rotten_tomatoes.yaml \
  --device cuda:0
```

## 安装

```bash
pip install torch transformers datasets
pip install -e "baselines/shapiq-copy[proxy]"
```

本地 `shapiq-copy` 要求 Python 3.12 或更高版本。

LightGBM 树转换优先使用 shapiq 的 C++ extension。若当前源码 checkout
没有构建该 extension，则自动使用 LightGBM `dump_model()` 的结构化输出构造相同的
`TreeModel` 拓扑、split feature 和 leaf value；这只替换反序列化后端，不改变
ProxySPEX 的 proxy、Fourier 提取或 refinement。每个样本会在
`proxyspex_tree_conversion_backends` 中记录实际使用的
`lightgbm_cext` 或 `lightgbm_python_dump`。转换完成后，runner 会在该样本的全部
训练 coalition 上比较 LightGBM proxy 与未精炼 Fourier 的预测；若两者不等价则
立即终止，最大绝对误差记录在
`proxyspex_tree_fourier_validation_max_abs_error`。

> 注意：runner 会固定加载本目录的 `src/shapiq`，因为论文结果协议依赖该版本导出的
> coalition observations 和 refinement 前后的 Fourier 系数。它不会回退到环境中
> 另一个已安装的 `shapiq` 版本。

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
  --value-function predicted_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 5,10,20,50 \
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
  --value-function predicted_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 5,10,20,50 \
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
  --value-function predicted_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 5,10,20,50 \
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
  --value-function predicted_probability \
  --target-mode predicted \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 5,10,20,50 \
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
├── observations/
├── surrogates/
└── diagnostics/       # output-level=debug
```

`samples/*.json` 中：

- `node_scores/ranking` 是投影后的节点解释；
- `method_summary.interaction_summary` 是高阶交互摘要；
- `method_summary.player_to_chunk_id` 是 ProxySPEX player 到文本 chunk 的映射；
- `attribution_cost` 记录原生 mask 查询、唯一扰动文本、model forwards 和耗时。

`surrogates/*.json` 保存 refinement 后的 Fourier predictor，供 held-out 与 E3
离线评价。runner 会在训练 masks、empty/full 与 singleton-deletion masks 上比较
该序列化 predictor 和原生 `predict_refined_fourier()`；误差超过 `1e-9` 会终止，
因此 E3 的四点差分不会静默改变 ProxySPEX 学到的函数。

`metrics.json` 的 `target` 必须与待比较的 Möbius run 一致。
