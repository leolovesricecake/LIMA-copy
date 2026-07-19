Mobius Verify
├─ 0. 项目骨架与环境预检
│  ├─ 建立 mobius_verify/{configs,src,scripts,tests,results}
│  ├─ README / requirements / 运行命令 / 结果 schema
│  ├─ 记录 Python、依赖、GPU、模型路径、数据缓存
│  └─ 所有长任务支持 resume、skip completed、atomic write、failure log
│
├─ 1. Lexical Word Featureization
│  ├─ 实现真实词级 feature：subword 合并回原始词
│  ├─ 标点 attach_to_neighbor，不单独作为主 feature
│  ├─ 保存 word_texts、char spans、token spans、tokenizer metadata
│  ├─ 校验所有 word spans 可重建 normalized_model_text
│  └─ 测试：空白、标点、撇号、多 subword、重建一致性
│
├─ 2. Masking 与 Probe 构造
│  ├─ apply_mask 支持 delete / replace
│  ├─ 对一个词同时处理其全部 subword/字符区间
│  ├─ exact_global：只筛选自然词数在 [min_n, max_n] 的短文本
│  ├─ exact_probe：contiguous_random / dispersed_random / stratified_position
│  └─ probe 主模式 rest_present，rest_absent 作为敏感性实验
│
├─ 3. Model 与 Value Function Adapter
│  ├─ 复用/扩展 LIMA sentiment 数据加载：SST2、RTN、Emotion、Eraser
│  ├─ classifier / causal LM 统一暴露 raw class scores
│  ├─ value_function = full-input predicted class raw margin
│  ├─ causal LM 使用 mean verbalizer conditional logprob
│  ├─ 保存 raw f(S)、all label scores、target class、forward counters
│  └─ 检查 NaN/Inf/constant output，degenerate 样本保留但主统计排除
│
├─ 4. 数值核心
│  ├─ bitmask 升序编码与 mask 枚举
│  ├─ fast Möbius transform / inverse zeta transform
│  ├─ FWHT Fourier transform / inverse transform
│  ├─ degree truncation、R²、normalized RMSE
│  ├─ hierarchy metrics：DSR、SHR、orphan ratio、orphan energy
│  └─ OMP oracle：Möbius / Fourier 对称重建曲线
│
├─ 5. Synthetic Sanity Check
│  ├─ additive、hierarchical Fourier、non-hierarchical Fourier peak
│  ├─ sparse low-degree Möbius、dense low-degree function
│  ├─ 验证 inverse error < 1e-8
│  ├─ 验证 support recovery 与 hierarchy 指标符合预期
│  └─ 输出 synthetic summary 与测试覆盖
│
├─ 6. 自然短文本 Exact Global
│  ├─ 按 n_i 长度区间 [6,8]、[9,11]、[12,14] 分层采样
│  ├─ 每个样本完整枚举 2^n_i 个 masks
│  ├─ 计算精确 Möbius/Fourier 谱和重建曲线
│  ├─ 报告 d90/d95、k90/k95、k90/n_i、normalized support size
│  └─ 明确结论仅对应自然短文本全局词级 value function
│
├─ 7. 长文本 Conditional Probe Exact
│  ├─ 对长文本构造多个 k∈{8,10,12} probe sets
│  ├─ A 外词默认 rest_present，只枚举 A 内 2^k 组合
│  ├─ 保存 probe_id、strategy、indices、conditioning_mode
│  ├─ 计算条件局部 Möbius/Fourier 谱
│  └─ 比较短文本 global 与长文本 conditional local 结构差异
│
├─ 8. 完整词级 Limited-Query Recovery
│  ├─ 使用全部自然词，variable n_i，不完整枚举
│  ├─ 预算 m = α n_i log2(n_i)，α∈{0.25,0.5,1,2,4,8}
│  ├─ 统一 train/validation/test masks，按 seed 固定
│  ├─ 测试分布：uniform、near-full、fixed-cardinality
│  ├─ 方法：Additive LASSO、Möbius regression、Fourier regression、sklearn GBT
│  └─ 候选过大记录 infeasible_due_to_candidate_size，不自动合并词
│
├─ 9. 聚合统计与图表
│  ├─ sample-level 保留全部结果
│  ├─ 按 task、n_i bin、experiment_scope 分组
│  ├─ paired bootstrap、Wilcoxon、effect size
│  ├─ 图：degree-R²、k-R²、d90/k90 分布、hierarchy-vs-compressibility
│  └─ 图：query efficiency、support recovery、paired difference
│
└─ 10. Hypothesis Report 与后续判断
   ├─ 生成 hypothesis_results.json 和 hypothesis_report.md
   ├─ 分别判断 H1-H5：低阶、稀疏、hierarchy 失败鲁棒性、可恢复性、跨任务稳定性
   ├─ 结论只能是 Supported / Partially supported / Not supported / Inconclusive
   └─ 只有证据支持后，再进入 adaptive sparse Möbius / hypergraph attribution 算法开发