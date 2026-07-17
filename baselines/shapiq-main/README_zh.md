# ProxySPEX 基线说明

本目录是下载的 `shapiq` 源码。本轮额外增加了一个面向本项目 HF LLM 分类评测的入口：

- `run_proxyspex_llm_baseline.py`

它把 ProxySPEX 的高阶交互解释转成主线评测需要的 token/word ranking，并复用主线的数据加载、模型调用、指标计算和结果落盘格式。

## 1. ProxySPEX 是什么

ProxySPEX 全称是 Proxy SParse EXplainer。它解决的问题是：直接在原模型上枚举或大量采样所有 Shapley 高阶交互非常贵，尤其是 LLM。ProxySPEX 的核心思路是先用有限预算查询原模型，再用一个树模型 proxy 近似原 value function，最后从 proxy 的树结构中高效读出稀疏交互。

本地实现位于：

```text
baselines/shapiq-main/src/shapiq/approximator/proxy/proxyspex.py
```

它的大致流程是：

1. 对输入特征采样 coalition，也就是哪些 token/word 被保留。
2. 调用原模型得到每个 coalition 的 value。
3. 用这些 `(coalition, value)` 样本拟合一个树 proxy，默认是 LightGBM，缺少后端时 shapiq 会尝试回退。
4. 从树结构提取 Fourier 系数。
5. 用 RidgeCV 在已查询样本上 refine 重要 Fourier 系数。
6. 将 Fourier 表示转成 Moebius 表示，再转成指定 Shapley interaction index，例如 `FBII`。
7. 输出 `InteractionValues`，其中 key 是一个交互子集，value 是该子集的交互贡献。

## 2. 为什么需要把交互值投影回 token/word

ProxySPEX 原生输出的是高阶交互，例如：

```text
(token_3, token_8) -> 0.12
(word_1, word_4, word_7) -> -0.05
```

但本项目当前的 `LO / Sufficiency / Comprehensiveness / AOPC / A-S / A-C` 评测协议需要的是单个 token 或 word 的全序 ranking。比如删除 top-q 个单元、只保留 top-q 个单元、逐步删除等操作，都要求每个评估单元有一个分数。

因此 runner 使用一个明确的适配步骤：

```text
score_i += interaction(S) / |S|, if i in S
```

这个策略叫 `signed_equal_share`。它的含义是：如果没有额外理由区分一个交互里的成员，就把该交互的有符号贡献平均分给参与成员。

这个做法合理的地方是：

- 对称：同一个交互内的 token/word 不会被任意偏置。
- 保守：不取绝对值，保留正负方向，和本项目其他 attribution ranking 的 raw-score 口径一致。
- 可加：所有成员分到的总贡献等于原交互值，不凭空放大高阶贡献。
- 可评测：最终得到单个 token/word 的分数，可以和 Captum、梯度法、主线方法进入同一套指标。

它的限制也需要记住：均分不是 ProxySPEX 的原生解释形式，而是为了适配单元排序评测做的投影。如果你要分析“哪些 token 组合产生协同作用”，应查看 sample JSON 的 `metadata.interaction_summary`，而不是只看最终 `chunk_scores`。

## 3. 本项目实现约定

Runner 的关键约定如下：

- `--eval-granularity token` 时，players 就是 tokenizer offset mapping 得到的 token 单元。
- `--eval-granularity word` 时，players 就是 whitespace-preserving word 单元。
- 解释目标沿用主线分类 prompt：`Text:\n{text}\nLabel:`
- 默认 value function 是目标 verbalizer 的 probability。
- 默认 target 是 gold verbalizer，可用 `--target-mode predicted` 改成模型预测标签。
- 超长样本按 HFBackbone 的左截断逻辑处理：被截断掉的左侧 token/word 仍保留在输出 chunks 中，但分数为 `0`。
- ProxySPEX 的高阶 interaction 使用 `signed_equal_share` 投影回单个 token/word。
- 排序规则是原始分数降序，分数相同按 chunk id 升序，不取绝对值。
- `selected_chunk_ids` 是 top-`k` 前缀，`chunk_ranking` 覆盖全部 token/word 单元。

## 4. 依赖

本轮不修改根目录 `requirements.txt`。建议在项目环境里安装：

```bash
pip install torch transformers datasets
pip install -e "baselines/shapiq-main[proxy]"
```

如果本地下载的 `shapiq-main` 缺少 `lazy_dispatch`，请安装与该 checkout 匹配的 shapiq 发布版，或补齐它依赖的 `lazy_dispatch` 包。当前 runner 会在缺依赖时 fail-fast，并提示安装方式。

快速 smoke 可以不用 LightGBM：

```bash
--proxy-model tree --no-hpo --budget 16
```

正式复现实验建议从默认的 `lightgbm + hpo` 开始，但会更慢。

## 5. 运行方式

默认推荐命令：

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --eval-granularity token \
  --target-mode gold \
  --index FBII \
  --max-order 2 \
  --budget 512 \
  --proxy-model lightgbm \
  --proxy-n-jobs 1 \
  --quiet-proxy \
  --sampling-weight-mode uniform_coalition \
  --k 8 \
  --eval-q-values 1,5,10,20,50 \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

五个数据集模板如下。

`sst2`

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

`eraser_movie_reviews`

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset eraser_movie_reviews \
  --split validation \
  --eraser-root hf://eraser-benchmark/movie_rationales \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

`imdb`

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset imdb \
  --split validation \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

`rotten_tomatoes`

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset rotten_tomatoes \
  --split validation \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --base-save-dir results \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

`emotion`

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py \
  --dataset emotion \
  --split validation \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --save-dir baselines/proxyspex \
  --device cuda:0
```

## 6. 输出目录

默认输出目录为：

```text
results/baselines/proxyspex/<dataset>/model-<model>/index-<index>_order-<order>_budget-<budget>_proxy-<proxy>_hpo-<0|1>_target-<mode>_k-<k>_seed-<seed>/
```

每个 run 包含：

- `samples/*.json`
- `samples/*.txt`
- `summary.csv`
- `run_config.json`
- `eval_config.json`
- `eval_report.json`
- `trajectory_points.csv`
- `trajectory_points.jsonl`
- `trajectory_summary.csv`

`samples/*.json` 继续使用主线 `ExplanationResult` 结构。ProxySPEX 相关信息位于 `metadata`，包括：

- `proxyspex_index`
- `proxyspex_budget`
- `proxyspex_requested_max_order`
- `proxyspex_effective_max_order`
- `proxyspex_proxy_model`
- `proxyspex_effective_proxy_model`
- `proxyspex_proxy_n_jobs`
- `proxyspex_quiet_proxy`
- `proxyspex_sampling_weight_mode`
- `projection_strategy`
- `interaction_summary`
- `player_to_chunk_id`
- `truncation`
- `game_stats`
- `forward_counters_delta`

## 7. 常见坑

### 1) 为什么 token 模式要求 offset mapping

因为 runner 要保证 ProxySPEX players 和最终评测单元完全一致。`--eval-granularity token` 会要求 tokenizer 支持 `return_offsets_mapping=True`，否则脚本会报错。若模型 tokenizer 不支持 offset mapping，可以先用：

```bash
--eval-granularity word
```

### 2) 为什么默认 `FBII`

`shapiq` README 的 ProxySPEX quickstart 使用 `FBII` 和二阶交互。它适合作为本轮默认复现口径。你也可以通过 `--index` 改成 shapiq `MoebiusConverter` 支持的其他 index。

### 3) 为什么长文本会慢

ProxySPEX 的预算是每个样本查询原 LLM 的 coalition 数量。`--budget 512` 意味着每个样本最多会构造数百个保留文本并调用模型。建议先用：

```bash
--max-samples 5 --proxy-model tree --no-hpo --budget 16
```

冒烟后再扩大预算和样本数。

### 4) HPO 与路径

输出路径包含 `hpo-0` 或 `hpo-1`，避免同样 proxy/budget/order 但 HPO 开关不同的实验互相覆盖。

### 5) 如何看原始高阶交互

主线指标使用投影后的 `chunk_scores`。如果你要看 ProxySPEX 原生高阶交互，请查看：

```text
samples/<sample_id>.json -> metadata.interaction_summary.top_by_abs_value
```

其中 `players` 是 active player id，可通过 `metadata.player_to_chunk_id` 映射回输出 chunks。

### 6) `OverflowError: int too large to convert to float`

原版 ProxySPEX 在未显式传 `sampling_weights` 时，会用 `math.comb(n, k)` 构造 coalition-size 权重。长文本下 active token/word 数 `n` 可能接近模型上下文长度，`comb(n, k)` 在中间阶数会极大，转成 float 时可能溢出。

本 runner 默认使用：

```bash
--sampling-weight-mode uniform_coalition
```

它用 log-comb 方式稳定构造与 `comb(n, k)` 同语义的大小分布，避免这个溢出。另一个可选模式是：

```bash
--sampling-weight-mode uniform_size
```

它让每个 coalition size 近似等概率被采样，适合做消融，但和 ProxySPEX 原始默认分布不同。

### 7) `--base-save-dir: command not found`

这是 shell 命令续行问题，不是 Python 脚本问题。每一行如果还要继续传参数，行尾都要有 `\`。例如 `--model-path /path/to/model` 这一行后面也要加反斜杠。

### 8) NVIDIA driver too old

如果报错类似：

```text
RuntimeError: The NVIDIA driver on your system is too old
```

说明当前安装的 PyTorch CUDA build 比服务器 NVIDIA driver 新。解决方式有三种：

- 更新服务器 NVIDIA driver。
- 安装和服务器 driver 匹配的 PyTorch CUDA 版本。
- 先用 `--device cpu` 跑小样本 smoke，确认脚本逻辑和数据路径没问题。

### 9) LightGBM 一直打印 `No further splits with positive gain`

这个 warning 通常表示当前 coalition 样本上的 proxy tree 已经找不到能继续提升目标的分裂。对 ProxySPEX 这种每个样本都重新拟合 proxy 的流程，尤其是 HPO 会多次训练 LightGBM，因此日志会被刷很多行。

新版 runner 不再把字符串 `"lightgbm"` 直接交给 shapiq 默认构造，而是自己构造安静版 proxy：

- LightGBM 使用 `verbosity=-1` 和 `verbose=-1`
- XGBoost 使用 `verbosity=0`
- boosting proxy 默认 `--proxy-n-jobs 1`
- HPO 外层 `GridSearchCV` 固定 `n_jobs=1`
- 默认开启 `--quiet-proxy`，在每个样本拟合 proxy 时临时屏蔽 native stdout/stderr，避免 LightGBM C++ 层绕过 Python logging 继续刷屏

如果你仍然希望最快速、最容易 Ctrl+C 的 smoke，建议先用：

```bash
--proxy-model tree --no-hpo --budget 16
```

如果要用 LightGBM 复现实验，建议保留：

```bash
--proxy-model lightgbm --proxy-n-jobs 1 --quiet-proxy
```

`Ctrl+C` 在 LightGBM/XGBoost 的 C++ 训练阶段有时不会立刻响应，这是底层库的信号处理限制。单进程 proxy 会显著改善这个问题；如果只是做脚本 smoke，`--proxy-model tree --no-hpo --budget 16` 最容易中断。
