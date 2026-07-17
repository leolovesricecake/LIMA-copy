# ProxySPEX LLM 归因流程梳理

本文沿 `baselines/shapiq-main/run_proxyspex_llm_baseline.py` 的代码链路，说明本项目中 ProxySPEX 如何把 LLM 文本分类问题转成可评测的 token/word attribution，并记录本次 `rotten_tomatoes` 短样本触发的 HPO/CV 边界修复。

## 1. 入口与运行上下文

启动入口是：

```bash
python baselines/shapiq-main/run_proxyspex_llm_baseline.py ...
```

主函数链路为：

```text
main()
  -> build_parser()
  -> run(args, raw_argv)
      -> _require_runtime_dependencies()
      -> _validate_args()
      -> load_dataset_bundle()
      -> HFBackbone(...)
      -> _scan_resume()
      -> for sample in pending_samples: _explain_sample(...)
      -> rebuild_summary_csv()
      -> evaluate_saved_explanations()
```

关键配置包括：

- `--dataset / --split`：选择数据集和 split，支持 `sst2 / eraser_movie_reviews / imdb / rotten_tomatoes / emotion`。
- `--model-path / --device / --dtype / --max-length`：加载 HF causal LM，并控制推理设备和上下文长度。
- `--eval-granularity`：决定归因单元是 tokenizer token 还是 word。
- `--budget`：每个样本可查询原 LLM value function 的 coalition 数量上限。
- `--max-order / --index`：控制 ProxySPEX 输出的最大交互阶数和目标 Shapley interaction index，默认 `FBII`。
- `--proxy-model / --hpo`：选择树 proxy 后端及是否做超参搜索。
- `--target-mode`：解释 gold label probability，或解释模型 predicted label probability。

`_require_runtime_dependencies()` 先检查 `torch / transformers`，再导入 `shapiq.approximator.proxy.proxyspex.ProxySPEX`。如果已安装包不可用，会尝试把本目录的 `src` 加入 `sys.path` 作为 fallback。

输出目录由 `_output_root()` 生成，路径中包含 index、order、budget、proxy、HPO、target、k、seed，避免不同实验互相覆盖。`_scan_resume()` 会检查 `samples/<sample_id>.json` 和 `.txt`，已完成样本不会重复计算。

## 2. 数据、标签与 LLM value function

`load_dataset_bundle()` 返回 `DatasetBundle`：

```text
dataset_name
split
samples: List[TextSample]
label_names
verbalizers
```

对 `rotten_tomatoes` 这类 HF 文本分类数据集，加载函数会读取 `text` 和 `label`，从 HF feature 中取得 label names，并把 `label_names` 直接作为 `verbalizers`。因此二分类样本通常以 `negative / positive` 作为候选 verbalizer。

`HFBackbone` 使用固定分类 prompt：

```text
Text:
{text}
Label:
```

对每个 verbalizer，模型实际评分的是 `" " + label_text` 的条件平均 log probability。`predict_label_probs_batch()` 会对所有 label 的 log probability 做 softmax，得到分类概率。ProxySPEX 使用的 value function 是：

```text
v(S) = P(target_label | Text: coalition_text, Label:)
```

其中 `target_label` 来自：

- `target-mode=gold`：样本真实标签；
- `target-mode=predicted`：完整原文上的模型预测标签。

`HFBackbone` 会保留 label token，并在总长度超过 `--max-length` 时从 prompt 左侧截断。

## 3. 单样本解释流程

单样本入口是 `_explain_sample()`。

### 3.1 确定目标标签

首先调用：

```text
_sample_target_label(sample, bundle, backbone, args.target_mode)
```

得到完整文本的 label probability 向量 `full_probs`，并确定要解释的 `target_label` 和 `target_label_text`。

### 3.2 构造评测单元

然后调用：

```text
build_eval_units(text=sample.text, eval_granularity=args.eval_granularity, tokenizer=backbone.tokenizer)
```

如果 `eval-granularity=token`，优先用 fast tokenizer 的 `return_offsets_mapping=True` 得到 token 到字符区间的映射；如果 tokenizer 不支持 offset mapping，runner 会报错，避免 ProxySPEX players 和最终评测 units 不一致。

如果 `eval-granularity=word`，使用 whitespace-preserving word span。每个单元都是 `TextChunk`：

```text
chunk_id, start_char, end_char, text, token_start, token_end
```

### 3.3 对齐 HF 左截断后的可见单元

由于 `HFBackbone` 会从 prompt 左侧截断，runner 需要保证 ProxySPEX 不把已经不可见的左侧 token/word 当作 player。对齐函数是：

```text
_prompt_visible_text_span_after_left_truncation(...)
```

它用同一个 prompt 和同一个 target label token 长度，计算在 `max_length` 下被保留的 text 字符区间。随后：

```text
_active_unit_ids_from_visible_span(units, visible_start, visible_end)
```

只保留与可见区间有内容重叠的 units 作为 ProxySPEX players。被截断掉的 units 仍会出现在输出 chunks 中，但最终分数为 `0`，metadata 里记录为 `inactive_chunk_ids`。

此时有两个索引空间：

```text
player index: 0..n_active_players-1
chunk_id: 原始输出单元 id
```

二者通过 `player_to_chunk_id` 映射。

### 3.4 CoalitionGame：把 coalition 转成 LLM 查询

`ProxySPEXCoalitionGame.__call__()` 接收 shapiq 采样出来的 boolean matrix：

```text
shape = (n_coalitions, n_active_players)
```

每一行表示一个 coalition，即哪些 active players 被保留。runner 调用：

```text
_compose_coalition_text(units, player_to_chunk_id, coalition_row)
```

把被选中的 chunk 按原始顺序拼回文本。如果 coalition 为空，使用 `EMPTY_PERTURBATION_TEXT`，避免空字符串进入评测协议。

`ProxySPEXCoalitionGame` 内部有文本级 cache：

- 同一个 coalition text 只调用一次 LLM；
- 新文本会批量调用 `backbone.predict_label_probs_batch(missing, verbalizers)`；
- 返回值是目标标签概率 `probs[idx, target_label]`。

metadata 中的 `game_stats` 记录 `call_count / row_count / cache_hits / cache_misses / unique_text_count`。

## 4. ProxySPEX 内部算法

runner 构造 shapiq 的：

```text
ProxySPEX(
    n=len(player_to_chunk_id),
    max_order=effective_max_order,
    index=args.index,
    proxy_model=proxy_model,
    hpo=False,
    sampling_weights=sampling_weights,
    pairing_trick=args.pairing_trick,
    top_order=args.top_order,
    random_state=args.seed,
)
```

这里 `hpo=False` 是有意的：runner 已经在 `_build_proxy_model()` 中构造好了裸 estimator 或 `GridSearchCV`，因此不再让 shapiq 用字符串重新构造默认 proxy。

### 4.1 Coalition 采样

ProxySPEX 的 `approximate()` 首先执行：

```text
self._sampler.sample(budget)
coalitions_matrix = self._sampler.coalitions_matrix
coalition_values = game(coalitions_matrix)
```

采样器会永远优先包含 empty coalition 和 grand coalition。若 `budget > 2**n_players`，会用 border trick 把预算裁到全枚举 coalition 数量。因此实际训练 proxy 的行数是：

```text
n_fit_samples = min(budget, 2**n_active_players)
```

本 runner 默认 `--sampling-weight-mode uniform_coalition`，用 log-comb 稳定构造与 `comb(n, k)` 同语义的 coalition-size 权重，避免长文本下组合数转 float 溢出。`uniform_size` 可作为消融，让每个 coalition size 近似等概率被采样。

### 4.2 拟合树 proxy

`_build_proxy_model()` 支持：

- `tree`：`DecisionTreeRegressor`；
- `lightgbm`：`LGBMRegressor`，可包一层 `GridSearchCV`；
- `xgboost`：`XGBRegressor`，可包一层 `GridSearchCV`；
- 若 LightGBM 缺失，尝试 XGBoost；若 boosting 后端都缺失，回退到决策树。

ProxySPEX 需要树模型，因为后续要从树结构中读交互。boosting proxy 的 HPO 搜索空间是：

```text
LightGBM: max_depth in [3, 5], max_iter in [500, 1000], learning_rate in [0.01, 0.1]
XGBoost:  max_depth in [3, 5], n_estimators in [500, 1000], learning_rate in [0.01, 0.1]
```

本次修复后，HPO 的 CV 折数会按实际 proxy 训练行数自适应：

```text
cv = min(5, n_fit_samples // 2)
```

这样每个 R2 test fold 至少有 2 个样本。若 `n_fit_samples < 4`，无法形成 2 折且每折至少 2 个测试样本，runner 会对该短样本禁用 HPO，直接拟合裸 boosting proxy。

### 4.3 从树中提取 Fourier 系数

shapiq 的 `ProxySPEX.approximate()` 拟合 proxy 后：

```text
tree_models = convert_tree_model(final_model)
unrefined_fourier = self._sklearn_to_fourier(tree_models)
```

如果 proxy 是 ensemble，`convert_tree_model()` 返回多棵树。`_sklearn_tree_to_fourier()` 深度遍历每棵树：

- 叶节点返回常数项 `()`；
- 分裂节点把左右子树系数合并；
- 对被分裂 feature 增加一个 Fourier interaction 维度；
- ensemble 的系数逐项求和。

这一步得到的是 proxy 函数的 Fourier 表示，interaction key 是 player index tuple。

### 4.4 RidgeCV refine

随后：

```text
refined_fourier = self._refine(unrefined_fourier, coalitions_matrix, coalition_values)
```

`_refine()` 会按 Fourier 系数平方能量选择重要项，保留可解释大部分能量的 sparse support，然后在已查询的真实 LLM coalition values 上构造 Fourier basis，用 `RidgeCV` 重新估计这些系数。它的作用是用真实查询值校正 proxy 树读出的初始 Fourier 系数。

如果 Fourier 项很少、能量为零，或不值得回归，`_refine()` 会直接返回原系数。

### 4.5 Fourier -> Moebius -> 指定 interaction index

ProxySPEX 接着执行：

```text
moebius_transform = self.fourier_to_moebius(refined_fourier)
result = self._process_moebius(moebius_transform)
```

`fourier_to_moebius()` 对每个 Fourier interaction 的所有子集做变换累加。`_process_moebius()` 构造 `InteractionValues(index="Moebius")`，再用 `MoebiusConverter` 转成用户指定的 index，例如默认的 `FBII`。

最终返回：

```text
InteractionValues(
    values=result,
    index=approximation_index,
    min_order=...,
    max_order=...,
    n_players=n,
    estimated=True,
    estimation_budget=budget,
    target_index=args.index,
)
```

如果 `--top-order` 开启，则只保留恰好 `max_order` 阶的交互。

## 5. 从高阶交互投影到 token/word 分数

ProxySPEX 原生输出是高阶 interaction：

```text
(player_3,) -> value
(player_1, player_7) -> value
...
```

但主线评测需要每个 token/word 一个 scalar score，以便排序、删除 top-q、保留 top-q 和计算 AOPC 等指标。因此 runner 做 `signed_equal_share` 投影：

```text
score_i += interaction(S) / |S|, if i in S
```

特点：

- 保留正负号，不取绝对值；
- 同一交互内成员均分，不人为偏置某个 token；
- 分给成员的总和等于原 interaction value；
- 可直接生成 `chunk_scores` 和 `chunk_ranking`。

投影后：

```text
_rank_desc_scores(chunk_scores, k)
```

按 raw score 降序排序，分数相同按 chunk id 升序。`selected_chunk_ids` 是 top-k 前缀，`selected_text` 是这些 chunk 拼接后的文本。

如果某个原始 chunk 因左截断不在 active players 中，它不会参与 ProxySPEX，分数保持 0。

## 6. 输出与评测

每个样本保存为：

```text
samples/<sample_id>.json
samples/<sample_id>.txt
```

JSON 使用主线 `ExplanationResult` 结构，关键字段包括：

- `chunks`
- `chunk_scores`
- `chunk_ranking`
- `selected_chunk_ids`
- `selected_text`
- `scores`
- `trace`
- `metadata`

ProxySPEX 相关 metadata 包括：

- `proxyspex_budget`
- `proxyspex_requested_max_order`
- `proxyspex_effective_max_order`
- `proxyspex_proxy_model`
- `proxyspex_effective_proxy_model`
- `proxyspex_proxy_fit_sample_count`
- `proxyspex_hpo`
- `proxyspex_effective_hpo`
- `proxyspex_proxy_hpo_cv_splits`
- `proxyspex_sampling_weight_mode`
- `player_to_chunk_id`
- `truncation`
- `interaction_summary`
- `game_stats`
- `forward_counters_delta`

所有 pending samples 完成后，runner 重建 `summary.csv`，然后调用 `evaluate_saved_explanations()` 复用主线评测协议，输出 `eval_report.json`、trajectory 文件和曲线统计。

## 7. 本次报错根因与修复

报错为：

```text
ValueError: Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=4.
```

触发链路是：

```text
_explain_sample()
  -> _build_proxy_model(args)  # 原先固定 GridSearchCV(cv=5)
  -> ProxySPEX.approximate()
      -> sampler.sample(budget)
      -> coalitions_matrix.shape[0] == 4
      -> proxy_model.fit(coalitions_matrix, coalition_values)
      -> GridSearchCV(cv=5).fit(...)
      -> sklearn 报错
```

虽然命令行里 `--budget=512`，但对极短样本而言 active players 很少。若 `n_active_players=2`，全部可能 coalition 只有：

```text
2**2 = 4
```

shapiq sampler 会把预算裁到全枚举的 4 行。固定 `GridSearchCV(cv=5)` 因此出现 `n_splits=5 > n_samples=4`。`sst2` 不触发该报错，是因为当前运行中待解释样本没有落到这个短样本边界，或 active players 足够多使实际 coalition 行数不少于 5。

修复内容在 runner 中完成：

1. 新增 `_expected_proxy_fit_sample_count(n_players, budget)`，按 `min(budget, 2**n_players)` 估计 ProxySPEX 实际训练 proxy 的 distinct coalition 行数。
2. 新增 `_proxy_hpo_cv_splits(n_fit_samples)`，用 `min(5, n_fit_samples // 2)` 自适应 CV 折数，保证 R2 scoring 的每个 test fold 至少 2 个样本。
3. `_build_proxy_model(args, n_fit_samples=...)` 根据该折数构造 `GridSearchCV`；若样本数少到无法形成有效 2 折 CV，则对该样本禁用 HPO，直接使用裸 LightGBM/XGBoost proxy。
4. metadata 记录 `proxyspex_proxy_fit_sample_count / proxyspex_effective_hpo / proxyspex_proxy_hpo_cv_splits`，后续可以看出某个短样本是否走了 HPO fallback。
5. 参数校验改成 ProxySPEX 本身需要的 `--budget >= 2`，不再用过时的固定 `budget >= 5` 约束阻断运行。

修复后，`rotten_tomatoes` 这类含极短文本的数据集遇到 `n_active_players=2` 时，会使用 2 折 CV；遇到 `n_active_players=1` 时，会禁用该样本的 HPO，但仍能完成归因并写出结果。
