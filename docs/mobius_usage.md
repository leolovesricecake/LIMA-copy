# Möbius 使用与消融说明

## 1. 主方法工作流

对文本切分得到 `n` 个 players 后，主方法执行：

1. sampler 产生有限个 keep masks；
2. LLM 对每个 mask 对应的扰动文本输出全部 verbalizer raw scores；
3. value function 将 raw scores 化为目标类别的标量 `f(S)`；
4. basis 把观测编码为低阶 deletion-Möbius 或 Fourier 设计矩阵；
5. hierarchy 决定允许进入最终恢复的候选超边；
6. estimator 执行标准化、Lasso/ElasticNet CV 选 support、Ridge 重拟合；
7. projector 把超边 coefficient 分配到节点，得到完整 ranking；
8. evaluator 将 chunk ranking 按字符重叠投影到独立的 token/word eval units，计算 faithfulness。

Deletion-Möbius 的定义固定为：

```text
g(D) = f(N \ D)
theta(T) = sum_{U subseteq T} (-1)^(|T|-|U|) g(U)
```

它与 OR interaction 只差符号约定，不作为额外新指标。

## 2. 主配置

`configs/mobius/*.yaml` 的主实验 cell 是：

```text
deletion_mobius
× hierarchy=none
× sampler=deletion_mixture
× projector=signed_equal_share
```

其他固定项为：`max_degree=2`、`budget=512`、`seed=42`、`chunker=word`、`eval_granularity=token`、`value_function=target_probability`、`target_mode=predicted`、`k=8`。

此前已经完成的 SST-2、Emotion、Rotten Tomatoes sparse probability 结果就是这个 cell。SST-2 和 Emotion 的 margin sensitivity run 仅把 value function 改为 `predicted_class_margin`。旧结果使用旧目录与旧 schema，本次不继续 resume。

## 3. 消融含义

### 3.1 Basis

`basis: deletion_mobius`：

- mask 中的 1 表示保留 player；
- 设计列是 `1{T subseteq D}`，其中 `D` 是删除集合；
- coefficient 直接描述一组特征同时被删除时的离散高阶效应。

`basis: fourier`：

- 使用同一批 masks、values、候选阶数和 estimator；
- 只把设计列换为 parity basis；
- 用于判断优势来自 Möbius 坐标，还是任意低阶稀疏表示都足够。

配置：`configs/ablations/basis_fourier.yaml`。

### 3.2 Hierarchy

`hierarchy: none`：

- 所有一阶和二阶候选一起进入稀疏选择；
- 不要求交互的 singleton 父项也重要。

`hierarchy: strong`：

- 第一阶段只拟合 singleton；
- 第二阶段只保留已选 singleton，以及所有父 singleton 均已选的交互；
- 使用相同 observations 和 estimator 重新拟合最终模型。

它检验 ProxySPEX 式 hierarchy 直觉是否有益。这里不是拟合完整模型后再删边，因为那无法隔离 hierarchy 对恢复过程的影响。

配置：`configs/ablations/hierarchy_strong.yaml`。

### 3.3 Sampler

`deletion_mixture`：

- full/empty anchors；
- 50% 全局 Bernoulli；
- 30% near-full，即少量删除；
- 20% 固定保留比例。

`bernoulli`：

- anchors 之外，每个 player 独立以 0.5 概率保留；
- 检验 deletion-aware 采样是否必要。

`uniform_size`：

- 先均匀采 coalition size；
- 再在该 size 内均匀采 coalition；
- 避免 Bernoulli 采样过度集中在中等 cardinality。

三者都保证 masks 唯一、逻辑 budget 精确；当 `2^n <= budget` 时直接完整枚举。

配置：`sampler_bernoulli.yaml`、`sampler_uniform_size.yaml`。

### 3.4 Projector

`signed_equal_share`：

- 把 `theta(T)/|T|` 均分给超边成员；
- deletion 坐标带方向负号；
- 是当前主方法。

`absolute_equal_share`：

- 把 `|theta(T)|/|T|` 均分；
- 检查最终 ranking 是否主要受 coefficient 符号影响。

`singleton_only`：

- 拟合过程与超边完全不变；
- 最终节点分数只使用一阶 coefficient；
- 检查 faithfulness 增益是否真正来自交互项。

配置：`projector_absolute.yaml`、`projector_singleton.yaml`。

## 4. 运行命令

主方法：

```bash
python -m mobius.cli.run \
  --config configs/mobius/sst2.yaml \
  --device cuda:0
```

单轴消融：

```bash
python -m mobius.cli.run \
  --config configs/ablations/hierarchy_strong.yaml \
  --device cuda:0
```

CPU smoke：

```bash
python -m mobius.cli.run \
  --config configs/mobius/smoke.yaml \
  --overwrite
```

单独评估 predicted target：

```bash
python -m mobius.cli.evaluate \
  --run-dir <run-dir> \
  --target predicted \
  --device cuda:0
```

比较两个 run：

```bash
python -m mobius.cli.compare \
  --left <baseline-run-dir> \
  --right <mobius-run-dir> \
  --output docs/comparison.json
```

## 5. 指标方向

- `comprehensiveness`：越高越好，删除 top 特征后目标概率应明显下降；
- `sufficiency`：越低越好，只保留 top 特征时应尽量维持原概率；
- `aopc`：越高越好，按 ranking 逐步删除时概率应更快下降；
- `aupc`：越低越好；它是完整逐步删除目标概率曲线在归一化删除进度 `[0,1]` 上的梯形面积；
- `aopc_comprehensiveness`：越高越好；
- `aopc_sufficiency`：越低越好；
- attribution query、physical values、model forwards 和耗时：越低越好，但必须结合方法自己的原生查询协议解释。

评估 target 必须一致后才能比较。主表统一使用 `target: predicted`。

`aupc` 直接复用计算 `aopc` 时已经得到的 `deletion_probabilities`，不会生成新扰动文本，也不会增加 LLM 调用。`metrics.json` 始终只保存一个平铺的 `target` 及对应指标；默认是 `predicted`，再次指定其他 target 评估会覆盖该文件，而不是追加 target block。

## 6. 结果与恢复

run ID 默认使用配置哈希：

```text
b<budget>-o<order>-s<seed>-<scientific-config-hash-8>
```

也可以在配置文件中提供：

```yaml
run_suffix: paper-main
```

此时 run ID 为 `b<budget>-o<order>-s<seed>-paper-main`。后缀允许 1 到 64 个字母、数字、点、下划线或连字符，且必须以字母或数字开头。字段缺失或空白时才回退到 8 位配置哈希。

device 和输出路径不进入 hash；basis、hierarchy、sampler、projector 等科学配置进入 hash。自定义后缀只改变目录可读性，`run.json` 仍保存完整配置指纹；若同一后缀目录已属于不同科学配置，程序会拒绝混写。重新执行同一命令时，`samples/*.json` 中通过 schema 校验的样本会被跳过。
