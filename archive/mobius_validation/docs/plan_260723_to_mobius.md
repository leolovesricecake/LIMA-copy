## 背景
我们正在研究 LLM 归因。
之前我们尝试把视觉模型的归因方法 LIMA 直接迁移到 LLM 中，形成了 `lima_llm/`。但是实验效果比较差，因此决定重新设计。
现在我们探讨了 Sparse deletion-Möbius 方法，实现于 `mobius_verify/`，其效果与当前最强基线 ProxySPEX 可竞争，但需要进一步验证。

## 需求与问题
我们需要重构项目，删除 LIMA 相关代码，更改为以 Möbius 方法为主线。
问题包括：
- 许多公共逻辑都实现于 `lima_llm/` 中，`mobius_verify/` 与 `baselines` 都是直接引用。
- `mobius_verify/` 可以重命名，比如就叫 `mobius/`。
- Sparse deletion-Möbius 还需要通过消融试验判断优势究竟来自 Möbius 坐标、非层级建模，还是采样与排序差异，因此需要一定程度的模块化，并把一些模块设计成可配置的。
- 当前结果文件逻辑太复杂，包含太多信息，其中有大量无用信息，如（1）只是曾经 LIMA 方法诊断使用的，对现在没有意义；（2）重复出现的冗余信息。并且命名也特别长，不便于筛选与理解。比如：results/baselines/proxyspex-copy/sst2/model-Qwen2_5-7B-Instruct/chunk-word_eval-token_index-FBII_order-2_budget-512_proxy-lightgbm_hpo-1_weights-uniform_coalition_pair-0_top-0_value-predicted_probability_target-predicted_k-8_seed-42/eval_report.json

## 任务
新建一个分支 `mobius`，在其中重构代码，解决以上问题。
其中，结果文件需要你重新设计，不仅要移除无用信息，还要考虑我们在对比时、写论文时需要展示的信息有哪些。
请你先理解背景、需求、任务，思考还有哪里不清楚的、需要确认的地方，进一步和我沟通。直到任务完全清晰时，将完整计划落成文档在开始重构。