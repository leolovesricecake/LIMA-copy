# ProxySPEX LLM 归因机制

本文说明 `baselines/shapiq-main/run_proxyspex_llm_baseline.py` 中 ProxySPEX 的实际数据流。重点不是入口函数怎么调，而是每一步拿到什么输入、生成什么输出、这些数据如何进入下一步。

## 1. 总览

对单条文本样本，runner 做的事情可以压缩成一条链：

```text
TextSample
  -> TextChunk units
  -> active players
  -> coalition matrix X
  -> LLM value vector y
  -> tree proxy f_hat(X)
  -> Fourier coefficients
  -> Moebius coefficients
  -> Shapley interaction values
  -> per-token/per-word chunk_scores
```

ProxySPEX 的核心思想是：不要直接枚举所有交互项去问 LLM，而是先用有限个 coalition 查询 LLM，拟合一个树模型 proxy，再从树结构中读取稀疏高阶交互。

本项目中 value function 固定为目标标签概率：

```text
v(S) = P(target_label | prompt(coalition_text(S)))
```

其中 `S` 是被保留的 token/word 集合。

## 2. 原始输入

每个样本来自 `DatasetBundle.samples`，形如：

```text
TextSample(
    sample_id="rotten_tomatoes-hf-validation-12",
    text="a funny , sharply observed comedy",
    label=1,
    label_text="positive",
)
```

数据集还提供：

```text
label_names = ["negative", "positive"]
verbalizers = ["negative", "positive"]
```

runner 先用完整原文调用 backbone：

```text
full_probs = backbone.predict_label_probs(sample.text, verbalizers)
```

输出是一个长度等于类别数的向量，例如：

```text
full_probs = [0.18, 0.82]
```

若 `--target-mode gold`，解释 `sample.label` 对应的概率；若 `--target-mode predicted`，解释 `argmax(full_probs)` 对应的概率。

## 3. LLM 如何给一个文本打分类分

`HFBackbone` 使用固定 prompt：

```text
Text:
{text}
Label:
```

对每个 verbalizer，实际计算的是 label token 的条件平均 log probability。例如二分类会分别构造：

```text
Text:
a funny , sharply observed comedy
Label: negative

Text:
a funny , sharply observed comedy
Label: positive
```

然后得到两个 log probability，做 softmax：

```text
score_mat = [logp(" negative"), logp(" positive")]
probs = softmax(score_mat)
```

ProxySPEX 后面所有 coalition 的 value 都是这个概率向量中的目标标签位置。

## 4. 文本切成解释单元

`build_eval_units()` 把原文切成 `TextChunk` 列表。若 `--eval-granularity token`，使用 tokenizer 的 offset mapping；若是 `word`，使用保留空白的 word span。

示例文本：

```text
"a funny , sharply observed comedy"
```

word 粒度可能得到：

```text
units = [
  TextChunk(chunk_id=0, start_char=0,  end_char=2,  text="a "),
  TextChunk(chunk_id=1, start_char=2,  end_char=8,  text="funny "),
  TextChunk(chunk_id=2, start_char=8,  end_char=10, text=", "),
  TextChunk(chunk_id=3, start_char=10, end_char=18, text="sharply "),
  TextChunk(chunk_id=4, start_char=18, end_char=27, text="observed "),
  TextChunk(chunk_id=5, start_char=27, end_char=33, text="comedy"),
]
```

后续所有归因分数最终都会写回这些 `chunk_id`。

## 5. 处理左截断：哪些单元真正参与博弈

`HFBackbone` 在超过 `--max-length` 时会保留 label token，并从 prompt 左侧截断。因此 runner 会复刻这个逻辑，计算原文中实际可见的字符区间：

```text
truncation = {
  "visible_start_char": 10,
  "visible_end_char": 33,
  "prompt_token_count": ...,
  "kept_prompt_token_count": ...,
  "target_token_count": ...,
}
```

然后筛出与可见区间重叠的 units：

```text
active_chunk_ids = [3, 4, 5]
player_to_chunk_id = [3, 4, 5]
```

这里出现两个索引空间：

```text
player 0 -> chunk_id 3 -> "sharply "
player 1 -> chunk_id 4 -> "observed "
player 2 -> chunk_id 5 -> "comedy"
```

被截断的 chunks 仍保存在输出 JSON 中，但不作为 ProxySPEX player，默认分数保持 0。

## 6. Coalition matrix 是什么

ProxySPEX 对 active players 采样 coalition。若有 3 个 players，一行 boolean 向量表示保留哪些 player：

```text
coalitions_matrix =
[
  [False, False, False],  # empty coalition
  [True,  True,  True ],  # grand coalition
  [True,  False, False],  # 只保留 player 0
  [False, True,  True ],  # 保留 player 1 和 2
]
```

矩阵形状是：

```text
(n_coalitions, n_active_players)
```

采样预算 `--budget` 是行数上限，但实际行数不会超过 `2**n_active_players`。例如 active players 只有 2 个时，全部 coalition 只有 4 行：

```text
[], [0], [1], [0, 1]
```

这也是本次 `rotten_tomatoes` 报错的根源之一：短文本的实际 proxy 训练样本可能远小于命令行预算。

## 7. Coalition 如何变成 LLM value

`ProxySPEXCoalitionGame.__call__()` 接收上面的矩阵，并逐行转成文本。

假设：

```text
player_to_chunk_id = [3, 4, 5]
units[3].text = "sharply "
units[4].text = "observed "
units[5].text = "comedy"
```

则：

```text
[False, False, False] -> EMPTY_PERTURBATION_TEXT
[True,  True,  True ] -> "sharply observed comedy"
[True,  False, False] -> "sharply "
[False, True,  True ] -> "observed comedy"
```

这些文本会批量送入：

```text
backbone.predict_label_probs_batch(texts, verbalizers)
```

得到形状为 `(n_texts, n_labels)` 的概率矩阵，例如：

```text
probs =
[
  [0.55, 0.45],
  [0.10, 0.90],
  [0.30, 0.70],
  [0.20, 0.80],
]
```

如果目标标签是 `positive`，即 `target_label=1`，则返回给 ProxySPEX 的 value 向量是：

```text
coalition_values = [0.45, 0.90, 0.70, 0.80]
```

因此 proxy 的训练数据就是：

```text
X = coalitions_matrix        # bool, shape=(n_coalitions, n_players)
y = coalition_values         # float, shape=(n_coalitions,)
```

## 8. 拟合 proxy 树模型

runner 构造的 proxy 可以是：

```text
DecisionTreeRegressor
GridSearchCV(LGBMRegressor)
GridSearchCV(XGBRegressor)
LGBMRegressor
XGBRegressor
```

ProxySPEX 调用：

```text
proxy_model.fit(X, y)
```

拟合出的 proxy 是一个近似函数：

```text
f_hat([keep_player_0, keep_player_1, ...]) ~= v(S)
```

如果启用 HPO，runner 会根据实际训练行数自适应 CV：

```text
n_fit_samples = min(budget, 2**n_active_players)
cv = min(5, n_fit_samples // 2)
```

这样 R2 scoring 的每个 test fold 至少有 2 个样本。若 `n_fit_samples < 4`，该样本不做 HPO，直接拟合裸 boosting proxy。

## 9. 从树结构读 Fourier 系数

树模型是 piecewise constant 函数。ProxySPEX 把每棵树转成 Fourier 表示：

```text
four_dict: dict[tuple[player_id, ...], float]
```

例如：

```text
four_dict = {
  (): 0.62,
  (0,): 0.11,
  (2,): -0.04,
  (0, 2): 0.07,
}
```

含义是：

- `()`：基线项；
- `(0,)`：player 0 的一阶项；
- `(0, 2)`：player 0 与 player 2 的二阶项。

代码中每个树节点按下面的递归规则合并：

```text
combined[k] = (left[k] + right[k]) / 2
combined[k union {split_feature}] = (left[k] - right[k]) / 2
```

直观理解：

- 左右子树平均值形成“不依赖当前 split feature”的部分；
- 左右子树差值形成“依赖当前 split feature”的 Fourier 项；
- 如果一条路径连续依赖多个 feature，就会产生高阶项。

如果 proxy 是 ensemble，所有树的 `four_dict` 会按 key 相加。

## 10. 用真实 LLM 查询值 refine Fourier 系数

树里读出的 Fourier 系数来自 proxy，可能有偏。`_refine()` 会用已经查询过的真实 LLM value 再校正一次。

输入：

```text
four_dict = {
  (): 0.62,
  (0,): 0.11,
  (2,): -0.04,
  (0, 2): 0.07,
}
train_X = coalitions_matrix
train_y = coalition_values
```

第一步，按非基线 Fourier 系数的平方能量选择重要 support。比如保留：

```text
support = [
  (),      # baseline
  (0,),
  (0, 2),
]
```

第二步，把 support 转成 0/1 mask，并为每个 coalition 构造 Fourier basis：

```text
basis = real(exp(train_X @ (i * pi * support.T)))
```

由于 `train_X` 是 0/1，这个 basis 实际是 `+1/-1`：

```text
basis[row, col] = (-1) ** count(selected_players_in_support)
```

第三步，用 RidgeCV 拟合：

```text
basis @ refined_coef ~= train_y
```

输出仍是：

```text
refined_fourier: dict[tuple[player_id, ...], float]
```

如果 Fourier 项太少或能量为 0，代码会直接跳过 refine，返回原 `four_dict`。

## 11. Fourier 转 Moebius

ProxySPEX 接着把 Fourier 系数转成 Moebius 系数：

```text
moebius_transform: dict[tuple[player_id, ...], float]
```

代码规则是：对每个 Fourier 项 `F`，把它分配给 `F` 的所有子集 `T`：

```text
moebius[T] += fourier[F] * (-2) ** len(T)
```

例如 Fourier 项 `(0, 2): 0.07` 会贡献给：

```text
T = ()      -> +0.07
T = (0,)   -> -0.14
T = (2,)   -> -0.14
T = (0, 2) -> +0.28
```

Moebius 系数可理解为集合函数的增量分解，是后续转换成不同 Shapley interaction index 的中间表示。

## 12. Moebius 转指定交互指数

runner 默认 `--index FBII`。shapiq 使用：

```text
MoebiusConverter(moebius_coefficients)(index=args.index, order=max_order)
```

输出是 `InteractionValues`：

```text
interaction_lookup = {
  (): 0,
  (0,): 1,
  (1,): 2,
  (2,): 3,
  (0, 1): 4,
  (0, 2): 5,
  (1, 2): 6,
}

values = np.array([
  baseline,
  value_of_(0,),
  value_of_(1,),
  value_of_(2,),
  value_of_(0,1),
  value_of_(0,2),
  value_of_(1,2),
])
```

runner 通过 `_interaction_items()` 转成更直接的列表：

```text
interaction_items = [
  ((), 0.60),
  ((0,), 0.08),
  ((1,), -0.02),
  ((0, 2), 0.05),
]
```

这里的 player id 仍然是 active player 空间，不是原始 `chunk_id`。

## 13. 高阶交互如何变成单元分数

主线评测需要每个 chunk 一个分数，而 ProxySPEX 输出可以是二阶或更高阶交互。因此 runner 使用 `signed_equal_share` 投影：

```text
for interaction S with value phi_S:
    for player i in S:
        score[player_i] += phi_S / len(S)
```

例子：

```text
interaction_items = [
  ((0,), 0.08),
  ((1,), -0.02),
  ((0, 2), 0.05),
]
```

投影到 active players：

```text
player_score[0] = 0.08 + 0.05 / 2 = 0.105
player_score[1] = -0.02
player_score[2] = 0.05 / 2 = 0.025
```

再通过 `player_to_chunk_id` 写回原始 chunks：

```text
player_to_chunk_id = [3, 4, 5]

chunk_scores[3] = 0.105
chunk_scores[4] = -0.02
chunk_scores[5] = 0.025
```

不在 active players 中的 chunk 分数保持 0。

注意：排序使用有符号 raw score，不取绝对值：

```text
chunk_ranking = sorted(chunk_id, key=(-chunk_scores[chunk_id], chunk_id))
selected_chunk_ids = chunk_ranking[:k]
```

因此负贡献强的 token 不会因为绝对值大而排到前面。

## 14. 最终输出

每个样本保存为：

```text
samples/<sample_id>.json
samples/<sample_id>.txt
```

JSON 中关键字段长这样：

```text
{
  "chunks": [
    {"chunk_id": 0, "text": "a ", ...},
    {"chunk_id": 1, "text": "funny ", ...}
  ],
  "chunk_scores": [0.0, 0.12, -0.03, ...],
  "chunk_ranking": [1, 4, 0, 2, ...],
  "selected_chunk_ids": [1, 4, 0, ...],
  "selected_text": "funny observed ...",
  "metadata": {
    "player_to_chunk_id": [...],
    "interaction_summary": {...},
    "truncation": {...},
    "game_stats": {...}
  }
}
```

其中最适合排查 ProxySPEX 原生结果的是：

```text
metadata.interaction_summary.top_by_abs_value
```

它保存了绝对值最大的高阶 interaction；`players` 字段需要通过 `metadata.player_to_chunk_id` 映射回原始 chunk。

## 15. 本次报错对应的数据边界

报错：

```text
ValueError: Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=4.
```

对应的数据形态是：

```text
n_active_players = 2
budget = 512
actual coalitions = min(512, 2**2) = 4

X.shape = (4, 2)
y.shape = (4,)
```

旧代码固定创建：

```text
GridSearchCV(..., cv=5)
```

于是 sklearn 看到只有 4 行训练样本，却要求 5 折交叉验证，直接报错。

现在 runner 在构造 proxy 前先计算：

```text
proxyspex_proxy_fit_sample_count = min(budget, 2**n_active_players)
```

再决定 HPO：

```text
if n_fit_samples >= 4:
    cv = min(5, n_fit_samples // 2)
else:
    disable HPO for this sample
```

所以：

```text
n_active_players = 2 -> n_fit_samples = 4 -> cv = 2
n_active_players = 1 -> n_fit_samples = 2 -> no HPO
```

这不会改变 ProxySPEX 的 value function 或 interaction 投影，只是在短文本样本上避免不合法的 CV 配置。
