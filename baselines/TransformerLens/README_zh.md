# TransformerLens 说明

本目录保留了官方 README 与两个 notebook：

- `README.md`
- `LLaMA.ipynb`
- `Qwen.ipynb`

本轮交付只补充说明文件，不新增评测脚本。原因很直接：`TransformerLens` 和 `Captum` 的定位不同。

## 1. TransformerLens 是什么

TransformerLens 是一个偏机制解释（mechanistic interpretability）的工具库。它的核心能力不是“给输入 token 产出一个现成的重要性 ranking”，而是：

- 用 `TransformerBridge` 加载大量 HF 模型，并保持 logits 与 HF 对齐
- 用 `run_with_cache` 抓中间激活
- 用 hook 改写某层、某头、某条残差流
- 做 head ablation、activation patching、direct logit attribution 一类分析

换句话说，它更像：

- 一个“看模型内部电路”的实验平台

而不是：

- 一个现成的文本归因 baseline 套件

## 2. 为什么这轮不把它接进主线 baseline 评测

我们当前主线评测需要的是：

- 对输入 token/span 给出稳定全序 ranking
- 能直接落到 `LO / Suff / Comp / A-S / A-C`
- 能统一五个数据集、统一 verbalizer 分类 prompt

而 TransformerLens 官方材料当前展示的重点是：

- HF logits 对齐
- attention / activation cache 可视化
- 某个 head 或 layer 被 ablate 后 loss / logits 如何变化

这些能力当然很强，但它们离“直接产出一个可与 Captum / AML / gradient baseline 横向对比的 token ranking 文件”还有一层不小的设计工作。

因此，本轮不把 TransformerLens 直接接入主线评测，主要有三点原因：

### 1) 它不是现成的 span/token attribution API

官方 notebook 更多是在演示：

- `run_with_cache`
- `to_tokens / to_str_tokens`
- forward hooks
- head ablation

而不是像 Captum 那样直接给你一个：

- `attribute(...) -> seq_attr / token_attr`

### 2) 机制解释信号不等于 faithfulness ranking

例如：

- 某个 attention head 的 ablation 让 loss 变大

这说明它“重要”，但还不能直接等价成：

- 输入第几个 token 对分类 verbalizer 最重要

要把这些信号转成当前评测协议需要的 token ranking，还要补额外定义，例如：

- 用哪种 activation 作为 attribution carrier
- 如何把 head/layer 级信号投影回输入 token
- 如何跨层、跨头聚合

### 3) 可比性风险更高

如果本轮强行做一个“TL 派生 baseline”，它大概率会是：

- 一个强定制版本
- 解释对象和 Captum 不完全同构
- 结果是否公平可比需要额外论证

这不适合作为当前“先复现稳定 baseline”的第一优先级。

## 3. 官方 notebook 能告诉我们什么

### `LLaMA.ipynb`

主要展示：

- 如何用 `TransformerBridge.boot_transformers(...)` 加载 LLaMA
- 如何和 Hugging Face logits 对齐
- 如何抓 layer/head attention pattern
- 如何写 hook，对某个 attention head 做 ablation

### `Qwen.ipynb`

主要展示：

- 如何加载 Qwen / Qwen2
- 如何验证 TransformerLens 与 HF 前向结果接近

### `README.md`

核心信息是：

- 推荐使用 `TransformerBridge`
- 支持大量模型家族
- 保留 raw Hugging Face numerics
- 适合 activation cache、hook、patching、ablation

## 4. 它在你们项目里更适合扮演什么角色

这轮不做 baseline，不代表它没用。更合适的定位是：

### 1) 机制分析工具

当你们已经有一套 token/span ranking 后，可以用 TL 去回答：

- 某些高分 token 实际激活了哪些 heads / layers
- 某些 counter-example 为什么会失效
- 模型的 label verbalizer 是被哪些内部路径推高的

### 2) 失败案例诊断工具

比如：

- faithfulness 指标正常，但解释看起来不合理
- 或者某类数据集上 baseline 方向不稳定

这时 TL 很适合做层级定位。

### 3) 后续派生 baseline 的研究底座

如果以后要做 TL 派生 baseline，更合理的候选方向是：

- `direct logit attribution -> token projection`
- `activation patching` 导出的 token importance
- `head/layer ablation` 在输入 token 维度上的累计影响

但这已经是“新方法设计”了，不再是“直接调用官方库里的现成 baseline”。

## 5. 如果后续要继续做 TL baseline，建议路线

可以按这个顺序推进：

1. 先固定当前主线 verbalizer 分类 prompt
2. 明确解释目标是：
   - gold verbalizer logprob
   - 还是 predicted verbalizer logprob
3. 选择一个内部信号：
   - residual stream
   - attention head output
   - direct logit attribution
4. 明确如何把内部信号投影回输入 token
5. 再输出和主线相同格式的：
   - `samples/*.json`
   - `summary.csv`
   - `eval_report.json`

只有做到这一步，TransformerLens 才真正进入“可与其他 baseline 横向对比”的状态。

## 6. 结论

本轮对 TransformerLens 的结论是：

- 它适合 HF 模型加载、激活缓存、hook、ablation、机制解释
- 它不属于现成的文本 span/token 归因 baseline 库
- 因此本轮不直接接入 `LO / Suff / Comp / A-S / A-C` 评测脚本

如果后续你们要做：

- 机制解释支撑的自定义归因方法

TransformerLens 反而会非常重要；但那是下一阶段的事，不是本轮“稳定复现 baseline”的最短路径。
