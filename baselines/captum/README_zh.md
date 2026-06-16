# Captum 基线说明

本目录当前包含两类内容：

- 官方原始材料：`README.md`、`Llama2_LLM_Attribution.ipynb`
- 本仓库接入实现：`run_captum_llm_baselines.py`

本轮目标不是把 Captum 全家桶无差别搬进来，而是把其中适合 `HF AutoModelForCausalLM`、且能和本项目现有 faithfulness 评测协议对齐的方法，落成可复现 baseline。

## 1. Captum 是什么

Captum 是 PyTorch 解释库，方法大致分三类：

- 梯度类：`Integrated Gradients`、`LayerIntegratedGradients`、`Saliency`、`InputXGradient`、`GradientShap`
- 扰动类：`FeatureAblation`、`Lime`、`KernelShap`、`ShapleyValueSampling`
- 其他：概念、influence、neuron/layer 级解释等

对于 LLM，Captum 额外提供了两层封装：

- `LLMAttribution`
  - 面向扰动法
  - 适配文本输入与自回归 target token 打分
- `LLMGradientAttribution`
  - 面向 layer 级梯度法
  - 官方现在可包住 `LayerIntegratedGradients`、`LayerGradientShap`、`LayerGradientXActivation`
  - 但本轮只实现 `LayerIntegratedGradients`

## 2. 本项目里哪些方法适合 HF LLM

| 方法 | 适用于 HF LLM | 适用于本项目当前评测 | 本轮实现 |
| --- | --- | --- | --- |
| `FeatureAblation` | 是 | 是 | 是 |
| `LayerIntegratedGradients` | 是 | 是 | 是 |
| `KernelShap` | 是 | 是 | 是 |
| `Lime` | 是 | 是 | 是 |
| `ShapleyValueSampling` | 是，但很慢 | 是，但更慢 | 可选 |
| `ShapleyValues` | 理论上可接 | 长文本几乎不可用 | 否 |
| `LayerGradientShap` / `LayerGradientXActivation` | 官方可接 | 当前不纳入本轮基线集合 | 否 |
| `DeepLift` | 不适合作为当前 LLM 接入主线 | 否 | 否 |
| `Saliency` / `InputXGradient` / `GradientShap` | 需要额外自定义 LLM 接法 | 暂不稳 | 否 |

### 为什么 IG 在这里落成 `LayerIntegratedGradients`

对 decoder-only LLM，直接对离散 token id 做普通 IG 不合适。Captum 官方 LLM 梯度入口也是把解释目标挂到 embedding layer，因此本实现使用：

- `LayerIntegratedGradients(model_adapter, model_adapter.get_input_embeddings())`
- 再通过 `LLMGradientAttribution` 产出 token 级归因

### 为什么 exact `ShapleyValues` 不做

本项目输入是整段文本 token。exact Shapley 的特征子集枚举对长文本成本过高，即使在分类 verbalizer target 下也不现实，因此只保留采样版：

- `ShapleyValueSampling`

### 为什么 `Lime` / `KernelShap` 只用 `seq_attr`

Captum 的 `LLMAttribution` 对 `Lime` / `KernelShap` 不返回逐输出 token 的 `token_attr`，只稳定返回整段 target sequence 的 `seq_attr`。因此本项目统一用：

- `seq_attr` 作为最终 token ranking 分数

这也和我们下游 `LO / Comp / Suff / A-S / A-C` 的输入需求一致，因为评测只需要输入 token/span 的全序 ranking。

## 3. 实现约定

本仓库的 Captum runner 做了这些固定约定：

- 解释 prompt 统一为：
  - `Text:\n{text}\nLabel:`
- target 统一是 verbalizer 文本：
  - gold 模式：数据集真实标签
  - predicted 模式：模型原始全输入预测标签
- target 文本统一按 `" " + label_text` 编码，和主线 `HFBackbone` 一致
- 解释特征只覆盖原始 `text` token，不把 `Text:` / `Label:` / BOS / EOS 暴露给 attribution
- 超长样本采用左截断；被截断掉的前缀 token 在输出 chunk 中保留，但 attribution 分数强制写 `0`
- 对扰动法，`--forward-in-tokens` 会真实传给 Captum，但固定关闭 `use_cached_outputs`，因为我们这里的 prompt adapter 不是原生 generation wrapper，不能安全复用 KV cache
- 每个方法单独输出到自己的 run 目录，并复用主线的：
  - `samples/*.json`
  - `samples/*.txt`
  - `summary.csv`
  - `run_config.json`
  - `eval_config.json`
  - `eval_report.json`

## 4. 依赖

本轮没有修改根目录 `requirements.txt`。请自行安装至少这些依赖：

```bash
pip install torch transformers captum
```

如果你要跑本地 tiny smoke 或自建 fast tokenizer，通常还需要：

```bash
pip install tokenizers
```

如果模型仓库依赖 remote code：

```bash
python baselines/captum/run_captum_llm_baselines.py ... --trust-remote-code
```

## 5. 运行方式

### 默认推荐

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --model-path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --methods feature_ablation,layer_integrated_gradients,kernel_shap,lime \
  --k 8 \
  --target-mode gold \
  --eval-q-values 1,5,10,20,50 \
  --eval-granularity token \
  --base-save-dir results \
  --save-dir baselines/captum \
  --device cuda:0 \
  --deterministic
```

### 五个数据集模板

`sst2`

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --model-path /path/to/model
```

`eraser_movie_reviews`

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset eraser_movie_reviews \
  --split validation \
  --eraser-root hf://eraser-benchmark/movie_rationales \
  --model-path /path/to/model
```

`imdb`

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset imdb \
  --split validation \
  --model-path /path/to/model
```

`rotten_tomatoes`

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset rotten_tomatoes \
  --split validation \
  --model-path /path/to/model
```

`emotion`

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset emotion \
  --split validation \
  --model-path /path/to/model
```

### 只开慢方法

```bash
python baselines/captum/run_captum_llm_baselines.py \
  --dataset sst2 \
  --split validation \
  --sst2-source hf://nyu-mll/glue \
  --model-path /path/to/model \
  --methods shapley_value_sampling \
  --n-samples 64 \
  --num-trials 4
```

## 6. 输出目录

默认根目录是：

```text
results/baselines/captum/<dataset>/model-<model>/method-<method>_target-<mode>_k-<k>_seed-<seed>/
```

例如：

```text
results/baselines/captum/sst2/model-Qwen2_5-7B-Instruct/method-feature_ablation_target-gold_k-8_seed-42/
```

## 7. 常见坑

### 1) tokenizer 没有 `offset_mapping`

当前实现为了保证和主线评测 token/span 对齐，要求 fast tokenizer 支持 `return_offsets_mapping=True`。如果不支持，脚本会 fail-fast。

### 2) `LayerIntegratedGradients` 只支持 `--attr-target log_prob`

这是当前 Captum LLM 梯度封装的自然限制。若你传：

```bash
--methods layer_integrated_gradients --attr-target prob
```

会直接报错。

### 3) `forward-in-tokens`

这个参数在扰动法里是生效的：

- `--forward-in-tokens 1`
  - 按 target token 逐步前向
- `--forward-in-tokens 0`
  - 整段 target sequence 一次性前向

但当前实现固定关闭 `use_cached_outputs`。原因不是 Captum 不支持 cache，而是我们这里为了把 prompt 前后缀排除在可解释特征空间之外，额外包了一层 adapter；这层 adapter 不适合直接复用生成缓存。

### 4) `predicted` 模式不是“评测只看 predicted”

`--target-mode predicted` 只改变 attribution 时选用哪个 verbalizer 作为 target。最终 `eval_report.json` 仍然会像主线一样同时输出：

- `metrics_by_target.gold`
- `metrics_by_target.predicted`

### 5) `ShapleyValueSampling` 很容易慢

如果你直接在 `imdb` 或较长 `eraser` 样本上开较大 `n_samples`，耗时会明显上升。建议先在：

- `sst2`
- `rotten_tomatoes`

上冒烟，再扩到长文本数据。
