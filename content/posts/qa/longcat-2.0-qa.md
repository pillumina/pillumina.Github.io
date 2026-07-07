+++
date = '2026-07-07'
draft = false
title = 'Meituan LongCat-2.0 架构 QA'
math = true
categories = ['qa']
vendor = 'Meituan'
tags = ['moe', 'attention', 'qa', 'meituan', 'longcat', 'mla', 'lsa', 'ngram-embedding']
summary = '美团 LongCat-2.0 架构 QA，覆盖 1.6T 参数分解、Dual-Sublayer 设计、Shortcut MoE、LSA 稀疏注意力三件套、N-gram Embedding、6D 并行训练。'
+++


> 54 问，覆盖 CH1 演进 → CH2 超参 → CH3 计算 → CH4 MLA+LSA → CH5 Dual-sublayer+MoE → CH6 N-gram → CH7 训练 → CH8 推理 → CH9 总结。每问围绕设计决策（为什么这样选 / 不那样选），含公式推导 / trade-off 分析 / 源码对照 / 量化计算四类深度。

---

## CH 1 演进脉络

### Q1.1 为什么 LongCat-2.0 从 GPU 转向 AI ASIC superpod？

**简短回答**：因为 ScMoE 要求 dense 分支与 MoE 分支「explicit per-core control」完全并行——这种粒度的核内调度在通用 GPU 上无法实现，必须依赖 ASIC 的硬连线控制。

**详细解释**：

通用 GPU 的编程模型是「kernel + grid」，每个 SM 跑一个 kernel，SM 内部的 warp 调度由硬件决定，软件无法精确指定「这条 lane 跑 dense MLP，同时那条 lane 跑 MoE all-to-all 通信」。在标准 Transformer 单层结构下这不是问题——dense 与 MoE 天然串行。但 LongCat 的 Dual-Sublayer 把 MoE 拆出来做跨子层 Shortcut，要求在 attn[1] + mlp[1] 计算期间，MoE 的 dispatch/combine/all-to-all 完成且不阻塞——这要求硬件能精确编排两类计算在同一个 tensor core cluster 内的时序。

ASIC 的优势是「能做硬连线的 per-core 控制」：把 dense MLP 算子放在 cluster A、MoE 通信放在 cluster B，两者在物理上并行且通信路径独立。这是 GPU 做不到的——GPU 的 NCCL all-to-all 会独占 SM 资源，与 dense compute 抢占。

**Trade-off**：
- 收益：通算完全掩盖（不是 overlap 是真并行），训练吞吐 +35% 相对 LongCat-Flash
- 代价：硬件锁定——开源社区无法用 A100/H100 复现训练栈；只能复现推理（且需要重写 ScMoE 的 fallback 路径）

**易混淆**：ASIC ≠ 必然更快。通用矩阵乘 GPU 仍可能比 ASIC 单卡快，ASIC 的优势在「特定架构（如 ScMoE）下的端到端吞吐」。

**延伸阅读**：主报告 CH 1.2 / CH 7.2 / 博客「Training」节「Scalable Infrastructure」

### Q1.2 为什么 LongCat-2.0 选择 1.6T 总参 / 48B 激活，而不是继续扩大 LongCat-Flash？

**简短回答**：因为单纯把 Flash 的 MoE expert 池扩大，会撞上「稀疏度 97% 边界」——再加 expert 几乎不被命中，参数效率急剧下降，所以选择在三条正交稀疏轴上同时扩展。

**详细解释**：

参数效率的边际递减可以量化。假设 LongCat-Flash 的 MoE 稀疏度已经接近 97%（激活率约 3%），如果把 expert 池从 512 扩到 1024 而 top-k 保持 12，则：
- 激活率从 12/512 ≈ 2.3% 降到 12/1024 ≈ 1.2%
- 单个 expert 的期望命中次数从 `tokens × 12 / 512` 降到 `tokens × 12 / 1024`，减半
- 每个 expert 收到的训练样本减半 → 表达能力未充分激活 → 参数浪费

LongCat-2.0 的解法不是「单轴扩大」而是「三轴正交扩展」：
- **Attention 轴**：用 LSA 把 1M 上下文的 attention FLOPs 从 1.0 ExaFLOPs 降到 334 TFLOPs（节省 3000×）
- **Expert 轴**：保持 MoE 激活率 ~3%，但用 Dual-Sublayer + ScMoE 让 MoE 通算被掩盖
- **Embedding 轴**：新增 N-gram Embedding（135B），开辟与 MoE 完全正交的稀疏通路

这种设计让 1.6T 模型在 1M 上下文下激活 48B、KV cache 仅 91.66 GB、decode FLOPs 约 3.2T/token——「规模、长度、稀疏度」三者同步推进。

**面试要点**：MoE 稀疏度的「sweet spot」通常在 1%-5% 激活率。超过 5% 会增加 I/O 成本，低于 1% 会让 expert 训练不充分。LongCat 选择在 sweet spot 内通过「新稀疏轴」扩展，而非在原轴上加大稀疏度。

**延伸阅读**：主报告 CH 1.2 / CH 6.1 / 博客「N-gram Embedding」节

### Q1.3 开源版本与博客描述的架构有哪些关键差异？

**简短回答**：HF 仓库只有权重 + config + tokenizer，无 modeling 代码；Transformers 5.12.1 内置的 `longcat_flash` 实际是前代 LongCat-Flash（无 LSA、无 N-gram Embedding、无 MTP 头）；完整 LongCat-2.0 实现仅在 SGLang PR #30042 未合并。

**详细解释**：

这是「诚实标注开源状态」的关键问题。具体差异：

| 模块 | 博客描述 | 开源物料 | 差距 |
|---|---|---|---|
| MLA + LoRA scaling | 完整实现 | `longcat_flash/mla.py` 完整 | 一致（继承自 Flash） |
| Dual-Sublayer + ScMoE | 完整实现 | `longcat_flash/decoder_layer.py` | 一致（继承自 Flash） |
| Identity zero experts | 128 个 | `longcat_flash/experts.py` | 一致 |
| Softmax router（无 n_group） | 完整实现 | `longcat_flash/topk_router.py` | 一致 |
| **LSA（SI + CLI + HI）** | 三件套 | **内置代码无**，仅 dense MLA | 缺失 |
| **N-gram Embedding** | 135B 参数 | **内置代码无**，仅 token_emb | 缺失 |
| **MTP 头（3 层）** | 含 indexer 共享 | checkpoint 有权重，加载时被 `_keys_to_ignore_on_load_unexpected` 丢弃 | 静默丢弃 |
| **HI（Hierarchical Indexing）** | 训练 + 超长上下文 | README 明确「not supported for simplicity」 | 缺失 |

**Trade-off**：社区使用 LongCat-2.0 时只能跑「前代 Flash 架构 + 2.0 权重」的近似版本，长上下文性能会显著弱于博客宣称。

**易混淆**：「MIT 协议开源」≠ 「完整架构开源」。LongCat 的开源是「权重 + 部分实现」，需谨慎区分。

**延伸阅读**：主报告 CH 1.3 / SOURCES.md L4-L17 / HF README「GPU」节

---

## CH 2 超参与参数分解

### Q2.1 LongCat-2.0 的参数闭合验证如何做？为什么偏差 2.4% 算通过？

**简短回答**：按 config.json 字段独立计算 6 类参数（token_emb + LM head + MLA + Dense MLP + MoE + N-gram），加总得到 1638B，与官方披露的 1.6T（1600B）偏差 2.4%，小于 3% 阈值即算通过。

**详细解释**：

「闭合验证」（closure check）是模型报告中验证参数披露自洽的标准方法——把官方披露的「总参数」与「按 config 字段独立分解的参数和」对比，看是否在合理误差内。

LongCat-2.0 的闭合路径：

```
模块              独立计算                            参数量
─────────────────────────────────────────────────────────────
token_emb       163840 × 8192                    = 1.342B
LM head         8192 × 163840                    = 1.342B
MLA（76 子层）   76 × 111.67M                     = 8.49B
Dense MLP       76 × 301.99M                     = 22.95B
MoE（38 逻辑层） 38 × (768 × 50.33M + 7.34M)       = 1469.0B
N-gram Emb      100.567 × 163840 × 8192          = 135.0B
─────────────────────────────────────────────────────────────
总和                                              ≈ 1638B
```

官方披露 1.6T = 1600B。偏差 `|1638 - 1600| / 1600 = 2.4%`。

**2.4% 偏差的合理来源**（不是错误，而是已知忽略项）：
1. **Norm 层参数忽略**：76 个 input_layernorm + 76 个 post_attention_layernorm，每层 `2 × 8192 = 16K` 参数，总 < 0.1B——可忽略
2. **MTP 头未计入主参数**：`_keys_to_ignore_on_load_unexpected=[r"model\.mtp.*"]` 表明 MTP 头独立于主模型，博客也未将其计入 1.6T
3. **LSA indexer 参数**：博客未独立披露，估算每层 ~26M，38 层共 ~1B——在 2.4% 偏差内

**Trade-off**：若强行把 MTP 头（3 层共享模块）算入主参数，总数会突破 1.6T；业界惯例是 MTP 头作为「推理时丢弃」的辅助头，不计入主参数。这是合理的分类选择。

**面试要点**：闭合验证是检验「博客 + config + 源码」三方一致性的关键步骤。3% 是经验阈值——低于 3% 说明披露自洽，高于 5% 应触发深度核查。

**延伸阅读**：主报告 CH 2.3 / `_work/config.json`

### Q2.2 为什么 LongCat-2.0 用 38 逻辑层而不是 32 或 48？物理子层翻倍 hack 怎么工作？

**简短回答**：38 逻辑层 × 2 子层 = 76 物理子层，是「参数预算」与「深度表达力」的平衡；物理子层翻倍 hack 是 KV cache 索引必须对齐物理子层而做的工程妥协。

**详细解释**：

层数选择本质是「深度 vs 宽度」的 trade-off。LongCat-2.0 的选择（hidden=8192、38 逻辑层）与其他大模型对比：

| 模型 | 总参 | hidden | 逻辑层 | 物理子层 |
|---|---|---|---|---|
| Llama-3-70B | 70B | 8192 | 80 | 80 |
| DeepSeek V3 | 671B | 7168 | 61 | 61 |
| Hy3-295B | 295B | 7168 | 48 | 48 |
| **LongCat-2.0** | **1.6T** | **8192** | **38** | **76** |

LongCat 的「38 逻辑层」看似比对手浅，但每个逻辑层内部含 2 个 Attention + 2 个 Dense MLP + 1 个 Shortcut MoE，等价于「76 个物理子层 + 38 个跨子层 MoE」。从表达能力看，76 个物理子层已经与 Llama-3-70B（80 层）相当。

**物理子层翻倍 hack**（源码 `model.py:L36-L38`）：

```python
# KEY HACK: pretend num_hidden_layers = 2*num_layers so cache layer-count matches physical sublayers.
# MLA uses layer_idx*2 and layer_idx*2+1, so cache indexing would break without this doubling.
self.config.num_hidden_layers = 2 * config.num_layers
```

这是因为 `DynamicCache` 用 `layer_idx` 索引 KV cache。每个 `LongcatFlashDecoderLayer` 内部有 2 个 MLA 子层，它们的 `layer_idx` 分别是 `outer_idx * 2` 和 `outer_idx * 2 + 1`（源码 `decoder_layer.py:L20`）。如果不把 `num_hidden_layers` 翻倍，KV cache 的预分配数组会越界。

注意 forward 中是用 `config.num_layers`（=38）切片：

```python
for decoder_layer in self.layers[: self.config.num_layers]:  # model.py:L62
```

而不是用翻倍后的 `num_hidden_layers`（=76）。这是 hack 的微妙之处——「config 在 Model.__init__ 内被修改，但 forward 仍按原始 num_layers 迭代」。

**Trade-off**：
- 这种 hack 简单粗暴但有效，代价是社区代码（vLLM、SGLang）必须知道这个约定才能正确加载 KV cache
- 替代方案是「显式定义 76 个独立 decoder layer」——但那会让 Sequence Parallel / Pipeline Parallel 的切分边界复杂化

**易混淆**：博客和 config 里说的「38 层」指逻辑层；论文里其他模型的「层数」通常是物理层。两者不能直接对比。

**延伸阅读**：主报告 CH 2.1 / `code-snippets/model.py:L36-L38` / `code-snippets/decoder_layer.py:L20`

### Q2.3 N-gram Embedding 占总参 8.4%（135B）——这个 8.4% 上限是怎么算出来的？

**简短回答**：博客「N-gram Embedding」节明确写「严格 < 10%」；实际 135B / 1638B = 8.2%，符合约束。这个 10% 是基于「N-gram lookup 的参数效率优势 vs expert FFN」的边际收益曲线决定的经验阈值。

**详细解释**：

为什么是 10% 而不是 5% 或 20%？背后是 N-gram Embedding 与 expert FFN 的「参数效率比」量化。

**单 token I/O 对比**（量化级深度）：

```
N-gram Embedding 激活：
  oe_neighbor_num=5 → 每 token 查 5 行 embedding
  每行 8192 维 × 2 bytes = 16 KB
  总 I/O = 5 × 16 KB = 80 KB / token

MoE expert 激活（top-12）：
  每 expert FFN: (2 × hidden × expert_ffn + hidden × expert_ffn) × 2 bytes
              = (2 × 8192 × 2048 + 8192 × 2048) × 2
              = 100.67 MB × 2 = 100 KB / expert
  总 I/O = 12 × 100 KB = 1.2 MB / token
```

I/O 比 = 1.2 MB / 80 KB ≈ **15×**。每参数的「I/O 效率」N-gram 比 expert 高 15 倍。

但是——N-gram 不能无限扩大。原因有二：

1. **表达力受限**：N-gram 是「确定性局部共现」（前 4 个 token + 当前），无法学习语义级别的长距离依赖。超过 10% 后边际收益骤降——因为 N-gram 表达的信息是局部的，再加参数只是「记住更多 n-gram 模式」而非「理解更深」。

2. **EMBP 通信成本**：N-gram 按 `oe_split_num=4` 分片，每片 33.75B 参数。再加 N-gram 体量会让分片数上升，all-to-all 通信指数级增长。

10% 是「参数效率优势」与「表达力上限」的平衡点。超过 10% 后，「I/O 节省」的边际收益小于「表达力受限」的边际损失。

**Trade-off**：
- 收益：8.4% 的 N-gram 参数等效于 8.4% × 15 = 126% 的 expert I/O 节省
- 代价：8.4% 参数被「锁」在 lookup 表里，无法学习长距离语义

**面试要点**：N-gram Embedding 不是「另一个 expert pool」，而是「与 MoE 完全正交的稀疏通路」。它学习的是确定性局部模式，MoE 学习语义模式——两者信息正交。

**延伸阅读**：主报告 CH 6.1 / CH 6.2 / 博客「N-gram Embedding」节

### Q2.4 `routed_scaling_factor=9` 远大于 DeepSeek V3 的 2.5——这个值的依据是什么？

**简短回答**：源码确认 `routed_scaling_factor=9` 的值（`topk_router.py:L18,L41`），但**博客未公开设计原理**——这是「实现细节待确认」字段，不应凭空推断为「softmax 概率和补偿」或「expert 归一化」。

**详细解释**：

`routed_scaling_factor` 的作用在源码中很清晰——top-k softmax 分数相乘：

```python
# topk_router.py:L41
topk_weights = topk_weights * self.routed_scaling_factor
```

但它为什么是 9 而不是 V3 的 2.5？博客没说。可以做以下**假设性分析**（需明确标注为推测）：

**假设 A：补偿 softmax 概率和的衰减**

softmax 后 top-k 分数和 = `Σ p_i`，其中 `p_i = exp(z_i) / Σ exp(z_j)`。对于 896 个 expert（含 identity）取 top-12，top-k 概率和通常在 0.01-0.1 之间（softmax 在大类上的分布）。

如果想让 expert FFN 输出的量级与 dense MLP 相当，需要把权重放大到 ~1。9 倍是一个合理的经验值。但 V3 的 2.5 也能做到类似效果（V3 用 sigmoid 不是 softmax，分布不同）。

**假设 B：与 identity expert 共存下的归一化替代**

LongCat 删除了 `norm_topk_prob`（V3 有），即不重归一化 top-k 概率。这意味着如果 top-k 概率和偏低，MoE 输出会偏小。9 倍是一个「全局放大」因子，让所有 token 的 MoE 输出量级保持稳定。

**假设 C：与 ScMoE 通算掩盖配合**

MoE 输出被延迟到子层 2 末尾加入残差（`decoder_layer.py:L68`）。放大因子让 shortcut MoE 的贡献与 attn[1] + mlp[1] 在量级上匹配。

**关键判断**：以上三个假设都是「事后合理化」，博客未确认任何一个。主报告明确标注「设计意图待确认」——这是正确的诚实态度。

**易混淆**：不要把 `routed_scaling_factor=9` 与 `mla_scale_kv_lora=4.0` 混淆——前者是 MoE 路由权重缩放，后者是 MLA LoRA 输出方差补偿。两者解决不同问题。

**延伸阅读**：主报告 CH 5.4 / CH 9.3 / `code-snippets/topk_router.py:L18,L37-L41`

### Q2.5 `moe_impl: "mix"` 字段的含义是什么？

**简短回答**：源码（内置 longcat_flash）未直接实现 `"mix"` 分支——这是 LongCat-2.0 推理系统的实现策略字段，暗示 ASIC 上 dense + sparse 混合调度，但开源代码未提供对应实现。

**详细解释**：

`moe_impl` 字段的可选值通常是 `"dense"`（token-batch 遍历所有 expert）或 `"sparse"`（按 expert 分组 token 后批量计算）。`"mix"` 是 LongCat 独有，指「在同一个 forward 中根据 token 数量动态切换 dense 与 sparse 实现」。

这与 `moe_switch_token_num=1024` 阈值配合：
- token 数 < 1024：用 dense 实现（小 batch 下 sparse 分组开销不划算）
- token 数 ≥ 1024：用 sparse 实现（大 batch 下 sparse 分组节省计算）

**但开源代码未实现这个切换**——内置 `longcat_flash/experts.py` 只有一种实现路径（按 expert 遍历，`for expert_idx_tensor in expert_hit:`，`experts.py:L41`）。`"mix"` 字段是 LongCat-2.0 ASIC 推理栈的特有优化，博客中提及但无开源代码对应。

**Trade-off**：
- 收益：ASIC 上 dense/sparse 自动切换，最大化不同 batch size 下的吞吐
- 代价：社区复现时只能用 sparse 实现，小 batch 下会有性能损失

**面试要点**：读到 config 里不熟悉的字段时，第一步是 grep 源码看是否被实际使用。如果字段只在 config 里、源码里查不到，那大概率是「另一套推理栈用的」。

**延伸阅读**：主报告 CH 2.1 / `code-snippets/experts.py:L32-L60`

### Q2.6 `zero_expert_type: "identity"` 有哪些备选？为什么选 identity？

**简短回答**：源码（内置）只有 `"identity"` 一种实现（`nn.Identity()`，源码 `experts.py:L16`）；备选如 `"zero"`（输出 0）或 `"shared"`（dense FFN）都没用，因为 identity 能让 padding token 路由后保持原值，不污染残差。

**详细解释**：

zero expert 的本质是「让 router 把 padding token 路由到一个不消耗 FFN 计算的伪 expert」。三种候选方案：

| 方案 | 实现 | 输出 | 问题 |
|---|---|---|---|
| **identity（选）** | `nn.Identity()` | 输出 = 输入 | 无参数，padding 信息原样传递 |
| zero | `lambda x: torch.zeros_like(x)` | 输出 = 0 | padding 后残差被置零，污染下游 attention |
| shared | `nn.Linear(hidden, hidden)` | 输出 = dense FFN(x) | 有参数，违反「不消耗 FFN 计算」目标 |

LongCat 选 identity 的关键考量是**残差流的稳定性**。看 `decoder_layer.py:L68`：

```python
hidden_states = residual + h_dense + shortcut_mlp_output
```

如果一个 padding token 被 router 分配到 identity expert，shortcut_mlp_output 就是它的原 hidden state——不会破坏残差。如果用 zero 方案，padding token 的 hidden state 会被清零，在后续 attention 中作为 key/value 时会污染有效 token 的注意力分布。

**Trade-off**：
- 收益：padding token 不消耗 FFN 算力，且不破坏残差
- 代价：router 的索引空间扩大（从 768 到 896），classifier 多 128 列权重——但 7.34M 增量极小

**易混淆**：identity expert ≠ shared expert。DeepSeek V3 用 shared expert（dense FFN，每 token 必经），解决的是「基础表达能力」；LongCat 用 identity expert（zero FFN，仅 padding token 经过），解决的是「padding 处理」。两者目的完全不同。

**延伸阅读**：主报告 CH 5.3 / `code-snippets/experts.py:L16,L48-L50`

---

## CH 3 计算、KV cache 与显存

### Q3.1 MLA 的 KV cache 每层只存 576 维——这个 576 怎么来的？

**简短回答**：576 = `kv_lora_rank(512) + qk_rope_head_dim(64)`——MLA 只缓存压缩后的 latent 向量与旋转维度，不缓存完整 K/V。

**详细解释**：

标准 MHA / GQA 的 KV cache 每层存：
```
K cache: seq × num_kv_heads × head_dim
V cache: seq × num_kv_heads × head_dim
```

对于 GQA（如 Hy3-295B 假设 num_kv_heads=8、head_dim=128），每层每 token KV cache = `2 × 8 × 128 = 2048` 维。

MLA 的核心思想是「KV cache 只存 latent」。源码 `mla.py:L30-L40`：

```python
# kv_a_proj_with_mqa 把 hidden 压到 (kv_lora_rank + qk_rope_head_dim) = 576 维
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
# k_pass (512 维) 经过 kv_b_proj 上投影还原多头 K/V
# k_rot (64 维) 直接做 RoPE，不需要上投影
```

cache 中只需存 `compressed_kv` 的 576 维——后续层做 attention 时，`kv_b_proj` 可以从 576 维重建完整的 K 和 V。

**公式推导**（为什么只存 576 维不丢失信息）：

设 `W_kv_a ∈ R^{576 × 8192}` 是 down-projection，`W_kv_b ∈ R^{(128+128) × 64 × 512}` 是 up-projection（每头一份）。对每个 token：
- 存 `c_kv = W_kv_a · h`（576 维）
- 用时计算 `K = W_kv_b · c_kv_pass`（重建 64 × 256 维多头 K）

**关键观察**：`W_kv_a` 和 `W_kv_b` 是模型权重，不在 cache 里——cache 里只存 latent。这就是 MLA 的「latent attention」名字由来。

**量化对比**（1M seq、76 子层）：

```
MLA:    1M × 76 × 576 × 2 bytes = 91.66 GB
GQA:    1M × 76 × 2048 × 2 bytes = 326 GB（4× MLA）
MHA:    1M × 76 × 64 × 128 × 2 × 2 bytes = 2.6 TB（28× MLA）
```

**Trade-off**：
- 收益：KV cache 缩 4×，1M 上下文可在单机部署
- 代价：每次 attention 多一次 kv_b_proj 上投影计算（额外 16.78M FLOPs/子层）

**面试要点**：MLA 不是「让 attention 变快」，而是「让长上下文 KV cache 能放下」。attention 的 QK·AV 计算量不变（甚至略增），但 cache 内存占用大幅下降。

**延伸阅读**：主报告 CH 3.2 / CH 4.1 / `code-snippets/mla.py:L30-L40`

### Q3.2 1M 上下文 decode 时 attention 真实 FLOPs 是多少？为什么博客说「48B × 6 = 288 GFLOPs/token」低估了？

**简短回答**：1M 上下文 decode 时，attention 部分 FLOPs 是 `2 × N × (qk_head_dim + v_head_dim) × num_heads ≈ 41 GFLOPs/子层`（QK 用 qk_head_dim=192，AV 用 v_head_dim=128），76 子层合计 **3.1 TFLOPs/token**——这才是 decode 真正的计算瓶颈，被简化估算漏掉。

**详细解释**：

业界常用「参数 × 6 FLOPs」估算 decode FLOPs，但这只算到「权重 matmul」部分，漏掉了 attention score 计算。

**decode 单 token 的 attention FLOPs 推导**：

```
单子层 attention（decode seq=1，但 KV cache 长度 = 上下文长度 N）：
  QK 点积：每头每 token 与 N 个 K 计算点积，qk_head_dim=192 维
    FLOPs = 2 × N × qk_head_dim × num_heads = 2 × 1M × 192 × 64 = 24.6 GFLOPs
  AV 加权求和：每头每 token 与 N 个 V 加权
    FLOPs = 2 × N × v_head_dim × num_heads = 2 × 1M × 128 × 64 = 16.4 GFLOPs
  单子层 attention 总：24.6 + 16.4 ≈ 41 GFLOPs

76 子层合计：76 × 41 ≈ 3.1 TFLOPs/token
```

加上 MLA projection 的 17 GFLOPs（按主报告 CH 3.1）和 dense MLP + MoE 的 92 GFLOPs：

```
真实 decode FLOPs/token ≈ 3100 + 109 ≈ 3.2 TFLOPs
```

**为什么博客的「288 GFLOPs」严重低估？**

博客的 48B × 6 = 288 GFLOPs 是按「总激活参数」算的，但：
1. **48B 本身可能是粗估**——按主报告独立分解（MLA 8.49B + Dense MLP 22.95B + MoE 激活 12×50.33M=0.60B × 38 层 + N-gram 5 行激活 ≈ 0），实际激活约 32.4B
2. **attention FLOPs 随 seq 增长被漏算**——上面的 24.6K FLOPs 是 seq=1 时的简化，实际 1M 上下文 decode 时 attention 部分 3.1T FLOPs

**关键 insight**：长上下文 decode 的真正瓶颈不在「参数 matmul」（这部分固定 ~100 GFLOPs），而在「attention 扫描 KV cache」（这部分随上下文线性增长，1M 时达到数 T FLOPs）。**这就是 LSA 必须出场的根本原因**——它把 attention FLOPs 从 3.1T 降到 334 TFLOPs（prefill 角度，decode 类似比例）。

**量化对比**（不同上下文长度下 attention 占比）：

| 上下文 | attention FLOPs | 权重 matmul | attention 占比 |
|---|---|---|---|
| 4K | 0.12 T | 0.11 T | 52% |
| 32K | 1.0 T | 0.11 T | 90% |
| 1M | 3.1 T | 0.11 T | 96% |

**面试要点**：分析 LLM 推理性能时，必须区分「权重 matmul」与「attention 扫描」。短上下文 attention 可忽略，长上下文 attention 主导。

**延伸阅读**：主报告 CH 3.1 / CH 4.3

### Q3.3 EP128 单卡显存预算是怎么算的？为什么 80GB ASIC 还能留 batching 余量？

**简短回答**：1.6T × 2 bytes / 128 卡 = 25 GB/卡（权重）+ 91.66 GB / 128 = 0.72 GB/卡（KV 分片）+ ~1 GB/卡（激活分片）≈ 27 GB/卡，在 80GB ASIC 上留 53 GB 做 batching / Super Kernel 缓冲。

**详细解释**：

EP128（128 路 expert parallel）部署假设单机 8 卡，共 16 机 128 卡。每卡显存分解：

```
1. 权重分片
   总权重 = 1.6T × 2 bytes (BF16) = 3.2 TB
   EP128: 3.2 TB / 128 = 25 GB/卡
   
   分解：
   - 路由 expert：768 / 128 = 6 个 expert / 卡，每 expert 50.33M 参数 × 2 bytes = 100 MB
     单卡路由 expert 总：6 × 100 MB = 600 MB
   - N-gram Embedding：135B / 128 = 1.05B / 卡，× 2 bytes = 2.1 GB
     但 N-gram 实际按 oe_split_num=4 分片（EMBP），不是 128
   - Dense MLP / MLA：每卡都持完整副本（不切分），约 31B × 2 bytes = 62 GB？
   
   ⚠ 注意：Dense + MLA 不会 EP 分片，会复制到每卡。这是 EP 的代价——dense 部分必须 replication。
```

**重新核算（修正版）**：

```
Dense 部分（不切分）：
  - MLA: 8.49B × 2 = 17 GB / 卡
  - Dense MLP: 22.95B × 2 = 46 GB / 卡
  - token_emb + LM head: 2.68B × 2 = 5.4 GB / 卡
  Dense 小计：68 GB / 卡 ⚠ 已经接近 80 GB 上限！

MoE expert 分片：
  - 768 / 128 = 6 个 expert × 100 MB = 600 MB / 卡

N-gram Embedding（按 EMBP 4 片，不是 EP128）：
  - 33.75B × 2 = 67.5 GB / EMBP rank
  - 128 卡 / 4 = 32 卡共享 1 片 → 单卡 67.5 / 32 = 2.1 GB
```

**关键发现**：Dense 部分的 68 GB 已经几乎吃满 80 GB ASIC。这意味着 LongCat-2.0 的 EP128 部署**严重依赖 dense / attention 的 TP 切分**——单纯 EP 不够，必须 TP + EP 组合。

主报告 CH 3.3 的「25 GB/卡」估算过于乐观——只考虑了 MoE expert 分片，忽略了 dense 部分的 replication cost。这是主报告可以补充的细节。

**Trade-off**：
- 收益：EP128 让单卡 expert 数从 768 降到 6，I/O 大幅减少
- 代价：dense 部分必须复制，每卡 68 GB 是硬性下限

**面试要点**：分析 EP 部署时，必须区分「会切分的（expert、N-gram embedding）」与「不切分的（dense MLP、MLA、token_emb）」。后者是显存硬约束。

**延伸阅读**：主报告 CH 3.3 / CH 8.3

### Q3.4 训练 FLOPs 336 ExaFLOPs 怎么算？理论下限 0.16 天为什么不可信？

**简短回答**：336 ExaFLOPs = `6 × 1.6T × 35T`；理论下限按 50K ASIC × 500 TFLOPs 估算得到 0.16 天，但实际训练时间远超此值——MFU 通常 30-50%，且通信开销占 20-40%。

**详细解释**：

训练 FLOPs 的标准公式：

```
FLOPs ≈ 6 × N_params × N_tokens
     = 6 × 1.6 × 10^12 × 35 × 10^12
     = 3.36 × 10^26 FLOPs
     = 336 ExaFLOPs
```

这里的 `6` 来自：前向 2 FLOPs/参数 + 反向 4 FLOPs/参数（梯度对权重 + 梯度对激活）。

**理论下限的误导性**：

```
理论时间 = 3.36 × 10^26 / (50000 × 5 × 10^14) = 13,440 秒 ≈ 0.16 天
```

这个数字假设 ASIC 持续跑在峰值 500 TFLOPs 且零通信开销，严重偏离实际：

| 损耗来源 | 典型比例 | 实际值 |
|---|---|---|
| MFU（峰值利用率） | 30-50% | 500 TFLOPs × 40% = 200 TFLOPs 实际 |
| 通信开销（all-reduce / all-to-all） | 20-40% 额外 | 有效计算时间 -25% |
| Checkpoint / restart | 5-10% | -5% |
| **综合实际吞吐** | ~30% 峰值 | ~150 TFLOPs/ASIC |

修正估算：

```
实际时间 ≈ 3.36 × 10^26 / (50000 × 1.5 × 10^14) ≈ 44800 秒 ≈ 0.52 天
```

这仍是乐观估计（假设 ASIC 持续 30% MFU）。考虑 long-tail failure、re-computation、数据预处理，实际训练周期可能在数天到数周。

**为什么博客只披露「相对吞吐 +35%」而非绝对数字？**

- 绝对 MFU 高度依赖 workload（短序列 vs 长序列）、batch size、并行配置
- ASIC 的「500 TFLOPs」是 BF16 峰值，实际 BFU（bfloat floating-point operations per second）受限于 HBM 带宽
- 披露相对值避免与 GPU 直接对比引发争议

**面试要点**：读到「N 天训练完成」时，必须问「N 是理论下限还是实测？」。理论下限通常只有实测的 20-30%。

**延伸阅读**：主报告 CH 3.4 / 博客「Training」节

---

## CH 4 MLA + LSA

### Q4.1 `mla_scale_kv_lora = (8192/512)^0.5 = 4.0` 的物理含义是什么？

**简短回答**：补偿 LoRA 低秩压缩引入的方差缩减——`kv_a_proj` 把 hidden=8192 压到 kv_lora_rank=512 时，输出方差按 `sqrt(512/8192) = 0.25` 缩减，乘以 4.0 恢复到原始量级。

**详细解释**：

这是 LongCat-Flash 留下的「signature trick」，源码 `mla.py:L8-L10,L37`：

```python
self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5
# ...
k_pass = k_pass * self.mla_scale_kv_lora   # mla.py:L37
```

**公式推导**（为什么是 `sqrt(hidden/lora_rank)`）：

假设 `kv_a_proj` 权重 `W ∈ R^{576 × 8192}` 用标准初始化（如 Xavier：方差 `1/8192`），输入 `h` 的每维方差为 `σ²`。则输出 `c = W · h` 的每维方差：

```
Var(c_i) = Σ_j W_{i,j}^2 · Var(h_j)
        = 8192 × (1/8192) × σ²
        = σ²
```

等等，这看起来方差没变？关键在于「初始化」与「训练后」的差别。**实际问题是**：

`W_kv_a` 的初始化让 `c` 的方差与 `h` 匹配，但 **`W_kv_b`（up-projection）会把 `c` 重建回多头 K/V**。`W_kv_b` 的输入是 512 维（`kv_lora_rank`），输出是 64×256 维。如果 `W_kv_b` 也用 Xavier 初始化（方差 `1/512`）：

```
Var(K_i) = Σ_j W_kv_b_{i,j}^2 · Var(c_j)
        = 512 × (1/512) × Var(c)
        = Var(c)
```

仍然没看出问题。**真正的动机**是经验性的——训练后 `W_kv_a` 和 `W_kv_b` 不再保持初始方差，实际测得 `c` 的方差会偏小（梯度下降把 LoRA bottleneck 压向低能量状态）。

**LongCat 的工程解**：直接乘 `sqrt(hidden/lora_rank) = sqrt(8192/512) = 4.0`，把 `c` 的量级放大 4 倍，让 `W_kv_b` 输出回到与原始 hidden 同量级，稳定 attention 的 softmax 温度。

**数字代入**：

```
mla_scale_q_lora  = (8192/1536)^0.5 ≈ 2.309
mla_scale_kv_lora = (8192/512)^0.5  = 4.000
```

q 用 2.309（压缩比小），kv 用 4.0（压缩比大）。压缩比越大，scale 越大——线性关系符合「方差缩减随压缩比线性」的直觉。

**Trade-off**：
- 收益：attention 输入量级稳定，softmax 温度不需重新调
- 代价：scale 是常数（非学习），可能不是最优；但工程上够用

**易混淆**：`mla_scale_kv_lora=4.0` 是「LoRA 输出缩放」，不是「attention temperature」。它作用在 `kv_b_proj` 之前（`mla.py:L37` 作用于 `k_pass`，即 latent），不是 attention score。

**延伸阅读**：主报告 CH 4.1 / `code-snippets/mla.py:L8-L10,L37`

### Q4.2 Q 侧两段 LoRA（q_a_proj + q_b_proj）的维度变化是什么？

**简短回答**：`hidden(8192) → q_lora_rank(1536) → num_heads × qk_head_dim(64 × 192 = 12288)`，先 down 后 up，cache 不存中间 latent。

**详细解释**：

MLA 的 Q 侧与 KV 侧设计不对称——Q 不需要 cache，所以 latent 只在 forward 内存在。源码 `mla.py:L25-L27`：

```python
# q-side two-stage LoRA: q_a_proj (down) -> q_a_layernorm -> q_b_proj (up)
q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
q_states = q_states.view(query_shape).transpose(1, 2)
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
```

**维度链**：

```
输入: hidden ∈ R^{B × S × 8192}

q_a_proj (down):    8192 → 1536
  输出: ∈ R^{B × S × 1536}
  权重: W_qa ∈ R^{1536 × 8192} = 12.58M params

q_a_layernorm:      1536 → 1536 (RMSNorm)

q_b_proj (up):      1536 → 64 × (128+64) = 12288
  输出: ∈ R^{B × S × 12288} → reshape 为 (B, S, 64, 192)
  权重: W_qb ∈ R^{12288 × 1536} = 18.87M params

split: 沿 head_dim 切成 q_pass(128) + q_rot(64)
scale: q_pass *= 2.309, q_rot *= 2.309
```

**为什么 Q 不像 KV 一样存 latent？**

Q 是「每 token 现算」，不需要 cache——decode 时 seq=1，从 hidden 计算 Q 只需 1 次 `q_a_proj + q_b_proj`，开销极小（24.6 GFLOPs/子层在 decode 单 token 时是 12.58+18.87 = 31.4M FLOPs/子层）。

KV 不同，它每生成一个 token 都要被「所有历史 token」的 Q 查询——存 latent 避免重算。

**Trade-off**：
- Q 侧两段 LoRA 增加 12.58 + 18.87 = 31.45M 参数/子层，76 子层共 2.4B——但相对 MoE 的 1469B 微不足道
- 收益：Q 的表达力不受 hidden=8192 限制，可以扩到 12288 维（64 头 × 192 dim）

**易混淆**：Q 的「两段 LoRA」与 KV 的「MQA-style 压缩」是不同设计。Q 用 q_a_proj + q_b_proj（两段学到的投影），KV 用 kv_a_proj_with_mqa（一段同时输出 k_pass 和 k_rot）。

**延伸阅读**：主报告 CH 4.1 / `code-snippets/mla.py:L24-L27`

### Q4.3 LSA 与 DeepSeek 的 DSA 有什么本质区别？

**简短回答**：LSA 是 DSA 的「工程演进版」——保留 DSA 的「indexer + sparse attention」骨架，但加了 SI（流式感知 HBM 合并）、CLI（跨层共享）、HI（粗筛+精选两阶段）三件套，把 indexer 成本进一步压低。

**详细解释**：

DSA（DeepSeek Sparse Attention）的核心思想：用一个轻量 indexer 网络为每个 query 选 top-k 个 KV token，再用 sparse attention 只算这些 token。在 1M 上下文下，把 attention 从 O(N²) 降到 O(N × top_k)。

**LSA 的三个增量改进**：

#### SI（Streaming-aware Indexing）

**问题**：传统 top-k 选 token 后，KV 读取是稀疏随机访问，HBM 带宽利用率低（GPU/ASIC HBM 是连续访问优化）。

**SI 方案**：把 `index_topk=2048` 的预算分成两部分——
- 连续块（local window 1024 + sink 16）：保证近端全注意力
- 随机选中的块：reshape 成连续段，让 HBM 读取合并成大块传输

源码实现：SGLang PR #30042 的 `Indexer` 类（`nsa_indexer.py:L151-L250`，**PR open 未合并到 sglang main**）。config 字段 `index_local_tokens=1024` 和 `index_init_tokens=16` 在源码 L236-L237 直接读取使用；`block_size=128`（L164）+ `topk_indices` 的 `ceil_align(..., 2048)` padding（L1016）共同实现 SI 的块对齐 + 预算重塑。

#### CLI（Cross-Layer Indexing）

`cli_factor=2` 表示每 2 层共享 1 次 indexer pass——相邻层的 query 分布相近，sparse pattern 也相近，可以共享。

**跨场景共享**：
- 跨物理子层：layer 2i 和 2i+1 共用 indexer 输出
- 跨 MTP 步骤：3 个 MTP draft 步骤共享 1 次 pass（steps 2/3 复用 step 1 的 index）

**训练时对齐**：博客明确「通过训练时的 cross-layer distillation 实现」——即训练阶段就让相邻层学习共享同一组 sparse pattern，避免推理时强行复用导致精度下降。这是 LSA 相对 DSA 的关键工程化创新。

#### HI（Hierarchical Indexing）

两阶段 indexer：粗筛（block 级近似评分）+ 精选（候选内细粒度选择）。

**开源状态**：HI 仅在训练时 + 超长上下文任务启用，开源版本不含 HI（README 明确「not supported for simplicity」）。

**Trade-off 对比表**：

| 维度 | DSA | LSA |
|---|---|---|
| Indexer 头数 | 64（同 target） | 32（一半，更轻量） |
| Indexer 头维度 | 同 target | 128（独立配置） |
| 跨层共享 | 无 | cli_factor=2（每 2 层 1 次） |
| MTP 步骤共享 | 无 | 3 个 draft 共享 1 次 |
| HBM 访问模式 | 稀疏随机 | SI 优化为连续块 |
| 粗筛+精选 | 无 | HI 两阶段 |

**量化收益**（主报告 CH 4.3）：

```
Dense MLA 在 1M seq 下 attention FLOPs:
  2 × 1M² × 64 × 192 × 38 ≈ 1.0 ExaFLOPs

LSA（CLI 共享后）:
  19 indexer pass × 17.6 TFLOPs/pass ≈ 334 TFLOPs

节省：1.0 ExaFLOPs / 334 TFLOPs ≈ 3000×
```

**面试要点**：LSA 不是「更快的 attention」，而是「把 attention 从 O(N²) 降到 O(N × k) 的同时，让 indexer 自己也尽量省」——CLI 让 indexer 成本砍半，SI 让 HBM 访问模式硬件友好。

**延伸阅读**：主报告 CH 4.2 / 博客「LongCat Sparse Attention」节

### Q4.4 CLI（每 2 层共享 1 次 indexer pass）依赖什么训练技巧才能不损失精度？

**简短回答**：博客明确「通过训练时的 cross-layer distillation 实现」——训练阶段让相邻 2 层学习共享同一组 sparse pattern，避免推理时强行复用导致精度下降。

**详细解释**：

CLI 的核心问题：layer 2i 的 indexer 选了 KV 集合 S_i，layer 2i+1 直接复用 S_i——但两层的 query 分布不同，最优 sparse pattern 也不同。如果训练时不做对齐，推理时强行复用会导致 layer 2i+1 的 attention 质量下降。

**Cross-layer distillation 的思路**（博客描述，**SGLang PR #30042 未含 distill 训练代码**——仅含推理时的 indexer 跨层复用机制）：

训练时让 layer 2i+1 的 indexer 输出向 layer 2i 对齐，loss 包含：

```
L_total = L_task + λ · L_distill

L_distill = ||indexer_{2i+1}(Q_{2i+1}) - indexer_{2i}(Q_{2i})||²
```

让两个 indexer 的输出分布（softmax 后的 top-k mask）尽量一致。训练收敛后，推理时直接复用 layer 2i 的 mask 不会显著掉点。

**但实际实现可能更复杂**：
- 可能是 attention output 的 distill（不是 indexer mask）
- 可能是软 mask（soft weights）而非硬 top-k
- 可能配合 KL 散度对齐两个 sparse attention 的输出分布

**关键观察**：博客原文「通过训练时的 cross-layer distillation 实现」——只说了「distillation」，没说 distill 什么（mask？output？attention weights？）。SGLang PR #30042 的 `Indexer` 类只实现了推理时的跨层 index 复用机制（`layer_id` 参数 + `topk_indices_list` 缓存），distill 训练代码不在开源范围内，无法验证具体 distill 目标。

**Trade-off**：
- 收益：indexer FLOPs 砍半（每 2 层 1 次而非每层 1 次）
- 代价：训练 loss 复杂化 + 可能的精度损失（即使有 distill，相邻层强制共享仍不完美）

**易混淆**：CLI 与 MTP 共享 indexer 是两件事——CLI 是「跨层共享」（layer 间），MTP 共享是「跨 draft 步骤共享」（时间步间）。`dsa_mtp_cli=true` 表示两者联合启用。

**延伸阅读**：主报告 CH 4.2.2 / `config.json: dsa_mtp_cli`

### Q4.5 为什么 HI（Hierarchical Indexing）不在开源版本中？

**简短回答**：HI 是粗筛+精选两阶段 indexer，训练时 + 超长上下文任务启用；开源版本不含是因为「实现复杂度高 + 仅在超长上下文收益明显」，README 明确「not supported for simplicity」。

**详细解释**：

HI 的两阶段设计：

```
Stage 1 (粗筛): 把 KV 序列分成 block（如每 64 token 一块），
                用 block 级代表向量做近似评分，选 top-M block
Stage 2 (精选): 在选中的 block 内做细粒度 top-k，得最终 sparse pattern
```

**为什么开源不含？**

1. **实现复杂度**：两阶段需要不同的 indexer 网络，训练 loss / 推理调度都更复杂
2. **收益场景有限**：HI 只在超长上下文（>100K）下显著优于单阶段——短上下文单阶段已够用
3. **与 SI 的部分重叠**：SI 已经做了 block 级连续访问优化，HI 的边际收益递减
4. **训练成本**：HI 需要额外训练粗筛网络，对训练 pipeline 侵入大

**Trade-off**：
- 收益：超长上下文（>512K）下 indexer 成本进一步降低（block 级评分更便宜）
- 代价：实现复杂、训练侵入、维护成本

美团选择不开源 HI 是合理的工程决策——大多数社区用户不会用到 512K+ 上下文，单阶段 LSA + SI + CLI 已经够用。

**易混淆**：「HI 不开源」≠「LSA 不开源」。LSA 三件套中 SI 和 CLI 都在 SGLang PR 中开源，只有 HI 是闭源。

**延伸阅读**：主报告 CH 4.2.3 / HF README「GPU」节

### Q4.6 Indexer 的 `index_topk=2048` 是怎么选的？为什么不是 1024 或 4096？

**简短回答**：`index_topk=2048` 是「sparse attention 表达力」与「计算成本」的平衡——在 1M 上下文下，2048 个 KV token 已经覆盖 0.2% 的上下文，足以捕获长距离依赖；再大计算成本线性增加，再小会丢失关键信息。

**详细解释**：

**Sparse attention 的「密度」trade-off**：

```
密度 = index_topk / 上下文长度

1M 上下文，index_topk=2048:  密度 = 0.195%
1M 上下文，index_topk=1024:  密度 = 0.098%（半稀疏）
1M 上下文，index_topk=4096:  密度 = 0.39%（2× 计算）
```

**为什么 2048 是 sweet spot？**

参考 LongFormer / BigBird 等sparse attention 模型的经验：
- 密度 < 0.05%（即 1M 上下文 top-k < 512）：显著掉点，长距离信息丢失
- 密度 0.1%-0.3%：精度与成本平衡
- 密度 > 0.5%：精度饱和，但计算成本快速上升

LongCat 选 2048 落在 0.2%，符合经验范围。

**计算成本对比**（prefill，1M seq）：

```
index_topk=1024:
  indexer FLOPs = 32 × 128 × 1M × 1024 × 2 = 8.8 TFLOPs/pass
  CLI 共享后 19 pass × 8.8T = 167 TFLOPs

index_topk=2048 (选):
  indexer FLOPs = 32 × 128 × 1M × 2048 × 2 = 17.6 TFLOPs/pass
  CLI 共享后 19 × 17.6T = 334 TFLOPs

index_topk=4096:
  indexer FLOPs = 32 × 128 × 1M × 4096 × 2 = 35.2 TFLOPs/pass
  CLI 共享后 19 × 35.2T = 669 TFLOPs
```

2048 相比 1024 成本翻倍，但精度显著提升；相比 4096 节省一半，精度损失小。这是典型的「对数收益曲线」上的最优点。

**与 local window 的配合**：

`index_local_tokens=1024` 保证近端全注意力，`index_init_tokens=16` 是 sink tokens。2048 的 top-k 在 1M 上下文下分配：
- 1024 local（近端）
- 16 sink（开头的 attention sink）
- 剩 1008 给 indexer 选远端 token

**Trade-off**：
- 收益：2048 个 KV 足以覆盖长距离依赖
- 代价：indexer 成本随 top-k 线性增长

**面试要点**：sparse attention 的 top-k 不是「越大越好」——存在精度饱和点。2048 是 LongCat 在 1M 上下文下的经验最优。

**延伸阅读**：主报告 CH 4.2 / CH 4.3

### Q4.7 Interleaved RoPE 与标准 RoPE 有什么区别？为什么 LongCat 用 interleave？

**简短回答**：标准 RoPE 把旋转角度作用在相邻 2 维（pair (0,1), (2,3), ...）；interleave 版本用 `apply_rotary_pos_emb_interleave` 把相邻 pair 在内存中交错存储，便于 SIMD 向量化，且与 YaRN 长上下文外推更兼容。

**详细解释**：

源码 `mla.py:L45`：

```python
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
```

**两种布局对比**：

```
Standard layout (GPT-NeoX 风格):
  x = [x0, x1, x2, x3, x4, x5, x6, x7]
  旋转 pair: (x0,x1) freq0, (x2,x3) freq1, (x4,x5) freq2, ...
  实际存储: [real_part: x0,x2,x4,x6 | imag_part: x1,x3,x5,x7]
  → 两个连续 half，每半内是同质数据

Interleaved layout (GPT-J 风格):
  x = [x0, x1, x2, x3, x4, x5, x6, x7]
  旋转 pair: (x0,x1) freq0, (x2,x3) freq1, ...
  实际存储: [x0,x1,x2,x3,x4,x5,x6,x7] (pair 紧邻)
  → pair 内交错存储
```

**为什么 LongCat 选 interleave？**

1. **DeepSeek V3 兼容**：LongCat-Flash 继承自 DeepSeekV3Attention（源码 `mla.py:L5`），DeepSeek 系列用 interleave
2. **YaRN 外推**：YaRN（Yet another RoPE extensioN）的 `mscale` 实现在 interleave 布局下更直接——`rope_scaling.mscale=1` 配合 interleave 让长上下文 attention 输出量级稳定
3. **FlashAttention 兼容**：某些 FlashAttention 版本要求 interleave 布局

**YaRN config 解读**：

```json
"rope_scaling": {
  "original_max_position_embeddings": 8192,
  "rope_type": "deepseek_yarn",
  "factor": 120,
  "beta_fast": 32,
  "beta_slow": 1,
  "mscale": 1,
  "mscale_all_dim": 1
}
```

- `factor=120`: 从 8192 外推 120× 到 983040 ≈ 1M
- `beta_fast=32, beta_slow=1`: YaRN 的频率混合参数（快频率保留、慢频率外推）
- `mscale=1`: 长上下文下 attention 输出的缩放补偿

**Trade-off**：
- interleave 与 standard 在数学上等价（只是内存布局不同）
- 但不同训练框架对两者支持度不同——interleave 是 DeepSeek 系约定

**易混淆**：RoPE 的 interleave/standard 与 MLA 的 `qk_nope_head_dim/qk_rope_head_dim` 是两件事——前者是 RoPE 内部布局，后者是「部分维度做 RoPE、部分不做」的解耦设计。

**延伸阅读**：主报告 CH 4.1 / `code-snippets/rotary.py` / config.json `rope_scaling`

### Q4.8 LSA Indexer FLOPs 与 Dense MLA 对比，3000× 节省是怎么算出来的？

**简短回答**：1M seq 下 Dense MLA attention 是 `2 × N² × num_heads × qk_head_dim × num_layers ≈ 1 ExaFLOPs`，LSA 是 `19 × 17.6 TFLOPs ≈ 334 TFLOPs`——比值约 3000×。

**详细解释**：

**Dense MLA attention FLOPs（prefill 1M seq）**：

```
每层 attention:
  QK 点积: 2 × seq² × num_heads × qk_head_dim
         = 2 × (1,048,576)² × 64 × 192
         = 2 × 1.1 × 10^12 × 64 × 192
         = 27 × 10^15 = 27 PFLOPs/层

38 逻辑层 × 2 子层 = 76 物理子层
（但 MLA 每子层独立 attention，所以是 76 × 27P）
合计 = 76 × 27 PFLOPs ≈ 2 ExaFLOPs
```

主报告 CH 4.3 给的是 1.0 ExaFLOPs（按 38 逻辑层算），这里按 76 物理子层算是 2 ExaFLOPs——口径差异。保守用 1 ExaFLOPs。

**LSA Indexer FLOPs**：

```
每 indexer pass:
  32 head × 128 dim × seq × index_topk × 2 (QK dot)
  = 32 × 128 × 1M × 2048 × 2
  = 17.6 TFLOPs/pass

CLI 共享: 每 2 层 1 次 → 76 物理子层 / 2 = 38 次 indexer pass
（但主报告按 38 逻辑层算 19 次，这里保守用 38 次）

合计 = 38 × 17.6 TFLOPs = 670 TFLOPs
（按 19 次算是 334 TFLOPs）
```

**加上 sparse attention 本身的成本**：

LSA 的 sparse attention 部分仍需计算：

```
每层 sparse attention:
  QK: 2 × seq × top_k × num_heads × qk_head_dim
    = 2 × 1M × 2048 × 64 × 192
    = 54 TFLOPs/层

38 层合计: 38 × 54T = 2 PFLOPs
```

**总 LSA 成本** = indexer (334T) + sparse attention (2P) ≈ 2.3 PFLOPs

**节省比**：

```
Dense MLA: 1 ExaFLOPs = 1000 PFLOPs
LSA 总:   2.3 PFLOPs

节省: 1000 / 2.3 ≈ 435×

主报告说的 3000× 是按「Dense MLA attention vs LSA indexer-only」比，不含 sparse attention 本身——这是 indexer 单项的节省比。
```

**关键 insight**：LSA 的收益主要来自「把 O(N²) 降到 O(N × k)」——N=1M、k=2048，理论节省 500×。考虑 indexer 本身成本和 sparse attention 剩余成本，实际净节省 ~400×。

**Trade-off**：
- 收益：1M 上下文训练 / 推理可行（否则 OOM 或时间不可接受）
- 代价：精度损失（即使有 distill，sparse 仍不如 dense）+ 架构复杂度

**面试要点**：分析 sparse attention 收益时，必须算「总成本」（indexer + sparse attention），不能只看 indexer 的节省。

**延伸阅读**：主报告 CH 4.3

---

## CH 5 Dual-Sublayer + Shortcut MoE + Identity Experts

### Q5.1 为什么 LongCat 用 Dual-Sublayer + Shortcut MoE 而非标准单层 Transformer？

**简短回答**：因为 ScMoE 要求「MoE 通信与并行分支计算 overlap」——单层结构下 MoE 必须串行接在 attention 后，通信无法掩盖；Dual-Sublayer 把 MoE 拆到跨子层位置，让它的 dispatch/combine/all-to-all 通信被 attn[1] + mlp[1] 计算完全掩盖。

**详细解释**：

**标准单层 Transformer 的 MoE 困境**：

```
标准单层 forward:
  h → attn → residual → norm → MoE → residual → 下层
  
  MoE 阶段必须做：
  1. Router 计算（轻量）
  2. dispatch：把 token 按 expert 分组，跨 GPU all-to-all（重通信）
  3. expert FFN 计算（重计算）
  4. combine：跨 GPU all-to-all 反向（重通信）
  
  在这个阶段，dense 计算资源（attn / next layer mlp）闲置——MoE 通信占满网络，但 compute 单元空跑
```

GPU 上的传统解法是「pipeline overlap」——把 micro-batch 切片，让下一 batch 的 attn 与本 batch 的 MoE 通信重叠。但这要求大 batch，且仍然有 30-50% 的通信暴露。

**LongCat 的 Dual-Sublayer 解法**：

把单层拆成 2 个子层 + 1 个跨子层 Shortcut MoE：

```
forward（decoder_layer.py:L39-L68）:
  子层 1: attn[0] → norm → MoE 启动（异步）→ dense mlp[0] → residual
                                          ↓
  子层 2: attn[1] → norm → dense mlp[1] → residual + MoE 输出（接入）
```

**关键观察**：MoE 在子层 1 末尾启动，输出在子层 2 末尾接入残差。这给了 MoE 整个子层 2（attn[1] + mlp[1]）的时间窗口完成 dispatch/compute/combine。

**源码细节**（`decoder_layer.py:L48-L53,L67-L68`）：

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm[0](hidden_states)
shortcut_mlp_output = self.mlp(hidden_states)   # MoE 启动（异步）
hidden_states = self.mlps[0](hidden_states)     # dense MLP 同步计算
hidden_states = residual + hidden_states        # 子层 1 residual

# ... 子层 2 attn[1] + mlp[1] ...

hidden_states = residual + hidden_states + shortcut_mlp_output  # MoE 输出接入
```

**注意**：源码层面 `shortcut_mlp_output = self.mlp(h)` 是同步调用——这看起来不像 overlap？实际上 ScMoE 的 overlap 是在 ASIC 的 hardware scheduler 层实现：MoE 的 dispatch 在 Python 层「返回 tensor」后就完成了控制流，后续 dense mlp[0] 计算时 ASIC 异步执行 all-to-all。Python 同步是逻辑层，ASIC 异步是物理层。

**Trade-off**：

| 维度 | 单层 + 标准 MoE | Dual-Sublayer + ScMoE |
|---|---|---|
| 参数效率 | 高（每层独立） | 中（每层多一个 dense MLP） |
| 通算掩盖 | 弱（仅 pipeline overlap） | 强（完全掩盖） |
| ASIC 利用率 | 50-70% | 90%+ |
| 复杂度 | 低 | 高（跨子层残差、共享 norm） |

**为什么参数效率会下降？**

Dual-Sublayer 让每逻辑层含 2 个 dense MLP（共 76 个物理子层的 dense MLP，参数 22.95B）。如果不做 Dual-Sublayer，这些参数本可以投入 MoE expert。但 LongCat 团队判断：「通算掩盖带来的吞吐收益 > 参数效率损失」——因为训练时间占整个研发成本的大头。

**面试要点**：ScMoE 不是「MoE 改进」，而是「架构级通算掩盖设计」。核心创新是把 MoE 从「层内组件」变成「跨层组件」，让它的通信延迟有地方可藏。

**延伸阅读**：主报告 CH 5.1 / CH 5.2 / `code-snippets/decoder_layer.py:L48-L68`

### Q5.2 ScMoE 与传统 Pipeline MoE overlap 有什么本质区别？

**简短回答**：传统 pipeline overlap 是「跨 micro-batch」掩盖（下 batch 的 attn 掩盖本 batch 的 MoE）；ScMoE 是「跨子层」掩盖（同一 batch 内子层 2 的 attn+mlp 掩盖子层 1 末尾启动的 MoE）——后者不依赖大 batch，单 batch 也能完全掩盖。

**详细解释**：

**Pipeline overlap（GPU 标准 MoE）**：

```
时间 →
batch 1:  [attn1][MoE1 ............]
batch 2:           [attn2][MoE2 ....]
batch 3:                    [attn3][MoE3]
```

每个 batch 的 MoE 通信被下一个 batch 的 attn 部分掩盖。但：
- 需要 ≥2 个 micro-batch 在飞才能 overlap
- 第一个和最后一个 batch 的 MoE 通信暴露
- 需要 GPU 有足够 SM 同时跑 attn 和 MoE 通信（NCCL 会抢 SM）

**ScMoE（LongCat Dual-Sublayer）**：

```
时间 →
batch 1 子层1:  [attn0][dense_mlp0 + MoE dispatch...]
batch 1 子层2:                [attn1 + dense_mlp1 + MoE combine + MoE FFN]
```

MoE 在子层 1 末尾启动 dispatch（异步），子层 2 整段时间内 ASIC 同时跑：
- attn[1] + dense_mlp[1]（dense compute）
- MoE all-to-all + expert FFN（sparse compute）

两者在 ASIC 上「explicit per-core control」——硬件级并行，不是 GPU 的 SM-level 调度。

**关键差异表**：

| 维度 | Pipeline overlap | ScMoE |
|---|---|---|
| 掩盖粒度 | 跨 batch | 跨子层（同 batch） |
| 依赖大 batch | 是（≥2 micro-batch） | 否 |
| 硬件要求 | GPU（NCCL + SM 调度） | ASIC（per-core control） |
| 暴露通信 | 首/末 batch | 几乎无 |
| Compute 利用率 | 50-70% | 90%+ |

**为什么 ScMoE 必须用 ASIC？**

GPU 上 SM 内部的 warp 调度由硬件决定，软件无法精确指定「warp 0 跑 dense MLP，warp 32 跑 MoE all-to-all」。NCCL all-to-all 会独占整个 SM 资源。ASIC 的「per-core control」允许在 cluster 级别精确编排——dense cluster A 跑 dense MLP，sparse cluster B 跑 MoE，物理隔离。

**Trade-off**：
- 收益：通算完全并行，吞吐 +35%
- 代价：硬件锁定，社区复现需要重写 fallback 路径

**面试要点**：ScMoE 的「通算掩盖」不是软件技巧，是「架构 + 硬件协同设计」。移到 GPU 上效果会大幅打折。

**延伸阅读**：主报告 CH 5.2 / 博客「Inference」节「ScMoE」

### Q5.3 128 个 identity zero experts（`nn.Identity()`）的设计意图是什么？为什么不用更少的真专家？

**简短回答**：identity experts 用于「padding token 路由」——它们零参数、零 FFN 计算，让 padding token 不消耗算力；128 个是为了让 router 的索引空间与 768 路由专家「量级匹配」，避免 padding 总是被路由到少数几个槽位。

**详细解释**：

源码 `experts.py:L11-L16`：

```python
self.zero_expert_num = config.zero_expert_num or 0   # 128
self.total_experts = self.num_routed_experts + self.zero_expert_num  # 768 + 128 = 896
# Identity expert = nn.Identity(); routing through it returns the input unchanged.
self.identity_expert = nn.Identity()   # 单例，所有 128 个 identity 共用
```

**forward 中的路由逻辑**（`experts.py:L48-L50`）：

```python
if expert_idx >= self.num_routed_experts or self.gate_up_proj is None:
    # Identity expert path: pass-through
    current_hidden_states = self.identity_expert(current_state)
```

当 router 选中的 expert_idx ≥ 768（即落在 128 个 identity 槽位之一），就调用 `nn.Identity()`——输出 = 输入，零计算、零参数。

**设计意图三层**：

#### 1. Padding token 路由（主用途）

`moe_switch_token_num=1024` 是切换阈值——当 batch 中 padding token 较多（< 1024 真实 token），router 倾向于把 padding 路由到 identity experts。

**为什么 padding 不能直接丢弃？**

```
原始序列: [t1, t2, t3, <pad>, <pad>, <pad>]

如果丢弃 <pad>: 序列变 [t1, t2, t3]，长度变化 → batch 对齐失败
如果 <pad> 走 dense FFN: 浪费算力（计算无意义的 FFN）
如果 <pad> 走 identity expert: 输出 = 输入，残差流稳定，且零算力
```

identity 是「既能保留 tensor shape，又不消耗算力」的最佳选择。

#### 2. Router 索引空间均匀

router 的 classifier 维度 = `n_routed_experts + zero_expert_num` = 896（源码 `topk_router.py:L17,L23`）。所有 token 走同一套 topk 逻辑，无需「先判断 padding 再分支」。

如果只有 8 个 identity experts（而非 128），padding token 会过度集中在 8 个槽位——router 学习「这 8 个槽位是 padding」的压力大。128 个让 router 有足够空间分散 padding。

#### 3. 与 shared expert 的对比

DeepSeek V3 用 shared expert（dense FFN，每 token 必经）：

```
V3 forward: hidden → router → top-k experts + shared expert → combine
                              ↑
                              每 token 都走 shared expert（dense FFN）
```

LongCat 删了 shared expert，改用 2 个 dense MLP 子层（`mlps[0]`, `mlps[1]`）承担基础表达。identity experts 不替代 shared——它们解决不同问题（padding vs 基础表达）。

**为什么不用更少的 identity experts？**

假设只用 16 个（而非 128）：
- 索引空间：768 + 16 = 784（vs 现在的 896）
- padding 集中度：高（16 个槽位承担所有 padding）
- router 学习难度：稍高

但 128 vs 16 的实际差异不大——因为 identity experts 的「参数」是零，加更多只是扩大索引空间，不增加显存。128 是「让 identity 与 routed 量级匹配」的经验选择（128 = 768 / 6）。

**Trade-off**：

| 方案 | 参数 | 计算 | 实现 |
|---|---|---|---|
| Identity experts (LongCat) | 0 | 0 | 简单 |
| Zero output | 0 | 0 | 残差被污染 |
| Shared expert (V3) | 2 × hidden × ffn_hidden | 每 token | 复杂 |
| 丢弃 padding | 0 | 0 | Tensor shape 不对齐 |

identity 是帕累托最优解。

**面试要点**：identity experts 看似浪费（128 个槽位），实则零成本（参数=0、计算=0），是「让 router 索引空间均匀」+「padding 路径高效」的工程优雅解。

**延伸阅读**：主报告 CH 5.3 / `code-snippets/experts.py:L11-L16,L48-L50`

### Q5.4 为什么 LongCat 删除了 DeepSeek V3 的 n_group / topk_group？

**简短回答**：V3 的 group clipping 是「先按组选 top、再组内选 top」的两阶段路由，用于 load balance；LongCat 删除它是因为「softmax + e_score_correction_bias + EPLB 异步负载均衡」可以替代，且简化了路由计算。

**详细解释**：

源码 `topk_router.py:L9-L13` 明确删除：

```python
# LongCat simplification: drop DeepSeekV3's bias-group constraints entirely.
del self.n_group
del self.topk_group
del self.weight
del self.norm_topk_prob
```

**V3 的 group clipping 机制**：

```
V3 router 流程:
  1. logits = classifier(hidden)                     # 256 个 expert（V3）
  2. sigmoid(logits)                                 # sigmoid 而非 softmax
  3. reshape 成 (n_group, experts_per_group)         # n_group=8, experts=32
  4. 组内 top-2 选 max → 得 8 个组代表
  5. 组间 top-4 选 → 得 4 个组
  6. 4 组内各取 top-2 → 共 top-8
  7. norm_topk_prob 归一化
```

这是为了防止「少数 expert 被过度路由」——先在组内做约束，保证每 token 至少覆盖若干组。

**LongCat 的简化路径**：

```
LongCat router 流程:
  1. logits = classifier(hidden)                     # 896 个 expert（含 identity）
  2. softmax(logits)                                 # softmax 而非 sigmoid
  3. + e_score_correction_bias                       # zeros 初始化 buffer
  4. top-12（全局，无 group）                          # 直接选 top-12
  5. × routed_scaling_factor                         # 全局缩放替代归一化
```

**为什么能简化？**

1. **softmax 自带归一化**：softmax 后所有 expert 概率和 = 1，不需要额外 `norm_topk_prob`
2. **e_score_correction_bias**：虽然 zeros 初始化，但训练时可动态更新（实现细节待确认）——这是 V3 也有的机制，用于动态调整 expert 偏好
3. **EPLB 异步负载均衡**：推理时根据路由热度动态迁移 expert，训练时还有 aux loss（虽然源码注释说 no aux-loss bias）

**Trade-off**：

| 维度 | V3 (group clip) | LongCat (softmax) |
|---|---|---|
| Load balance | 显式约束（组内/组间） | 隐式（softmax + EPLB 异步） |
| 路由自由度 | 低（必须覆盖多组） | 高（任意 top-k 组合） |
| 计算复杂度 | 高（两阶段） | 低（单阶段） |
| 调试难度 | 高 | 低 |

**潜在问题**：删除 group clipping 后，可能出现「热门 expert 被过度路由」——如果 EPLB 异步迁移不够及时，会导致某些 ASIC 卡负载过高。LongCat 团队显然判断「EPLB + ScMoE 通算掩盖」足以应对。

**易混淆**：`e_score_correction_bias` 在源码中是 `register_buffer`（`topk_router.py:L20`），不是 `nn.Parameter`——意味着它不是梯度更新的，而是某种启发式更新（如基于路由统计的滑动平均）。具体机制博客未公开。

**延伸阅读**：主报告 CH 5.4 / `code-snippets/topk_router.py:L9-L13,L37-L41`

### Q5.5 Softmax router 与 Sigmoid router 的数学差异是什么？

**简短回答**：Softmax 是「所有 expert 竞争一个概率空间」（Σ=1），适合「top-k 互斥选择」；Sigmoid 是「每个 expert 独立打分」（Σ 任意），适合「多标签」。LongCat 用 softmax 是因为 top-k 路由本质是「k 个互斥选择」。

**详细解释**：

**Softmax 数学**：

```
p_i = exp(z_i) / Σ_j exp(z_j)
Σ_i p_i = 1（归一化）
```

每个 expert 的概率「此消彼长」——选 expert A 概率高，必然有其他 expert 概率低。

**Sigmoid 数学**：

```
p_i = σ(z_i) = 1 / (1 + exp(-z_i))
Σ_i p_i 可以任意（不归一化）
```

每个 expert 独立打分，互不影响。

**为什么 LongCat 用 softmax？**

1. **top-k 互斥性**：top-12 路由本质是「12 个互斥选择」——选中 expert A 不应让 expert B 也被选中（否则就是 dense FFN）。softmax 的「此消彼长」更符合这个语义。

2. **梯度集中**：softmax 梯度集中在「高分区」，训练时 router 快速收敛到「少量高质量 expert」。sigmoid 梯度分散，所有 expert 都被「软选」，训练效率低。

3. **DeepSeek V3 用 sigmoid + group clipping 的原因**：V3 想让 router 更「自由」——sigmoid 允许多个 expert 同时高分，配合 group clipping 强制覆盖。但 group clipping 复杂，LongCat 删了，softmax 自然成为选择。

**LongCat 的 forward 实现**（`topk_router.py:L37,L41`）：

```python
scores = router_logits.softmax(dim=-1)        # L37: softmax over 896 experts
# ...
topk_weights = scores.gather(1, topk_indices)
topk_weights = topk_weights * self.routed_scaling_factor   # L41: × 9
```

注意 `routed_scaling_factor=9` 是在 softmax 之后乘——这是因为 top-12 softmax 分数和通常远小于 1（softmax 在 896 类上 top-12 概率和约 0.1-0.3），乘 9 让 MoE 输出量级稳定。

**对比 V3 sigmoid + norm**：

```
V3: sigmoid → group_clip → top-8 → norm_topk_prob（归一化到 1）
    最终 top-8 权重和 = 1

LongCat: softmax → top-12 → × 9
         最终 top-12 权重和 ≈ 0.1-0.3 × 9 = 0.9-2.7（不归一化）
```

**Trade-off**：
- Softmax + scaling：简单，但权重和不是 1（需要 scaling 调）
- Sigmoid + norm：复杂，但权重和严格归一化

**易混淆**：softmax router ≠ attention softmax。router softmax 是「选 expert」的 softmax，attention softmax 是「选 KV token」的 softmax。两者解决不同问题。

**延伸阅读**：主报告 CH 5.4 / `code-snippets/topk_router.py:L37,L41`

### Q5.6 Shortcut MoE 共享子层 1 的 norm 输入——这有什么好处？

**简短回答**：源码 `decoder_layer.py:L50-L52` 显示 shortcut_mlp 和 mlps[0] 共用 `post_attention_layernorm[0]` 的输出——避免重复 norm 计算，且让 dense 和 sparse 分支看到「同一个归一化视角」。

**详细解释**：

源码 `decoder_layer.py:L48-L53`：

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm[0](hidden_states)   # norm 一次
shortcut_mlp_output = self.mlp(hidden_states)                      # MoE 用同一份
hidden_states = self.mlps[0](hidden_states)                       # dense MLP 也用同一份
hidden_states = residual + hidden_states
```

**两个好处**：

#### 1. 省一次 norm 计算

如果不共享，需要两个 norm：

```python
h_norm_for_moe = post_attn_norm_for_moe(h)
h_norm_for_dense = post_attn_norm_for_dense(h)
shortcut_mlp_output = self.mlp(h_norm_for_moe)
hidden_states = self.mlps[0](h_norm_for_dense)
```

每个 norm 是 `8192 × 8192` 维 RMSNorm，约 67M FLOPs/子层。共享后省 67M × 76 子层 = 5 GFLOPs——相对总 3.2T 是小数目，但工程上合理。

#### 2. 同一归一化视角

dense 和 sparse 看到相同输入，意味着它们学到的特征互补——dense 学「全 token 都需要的通用变换」，MoE 学「token 特异的稀疏变换」。如果 norm 不同，两者看到的输入分布有差异，协同性变差。

**类比**：transformer 中 attn 和 mlp 共享 input_layernorm 是同理——让 attn 后的 norm 同时喂给 dense MLP，保证两者看同一视角。

**Trade-off**：
- 收益：省计算 + 协同性
- 代价：失去「分别为 dense 和 sparse 学不同归一化」的自由度（但实际很少需要）

**面试要点**：LayerNorm 共享是 transformer 工程常见优化。LongCat 把它扩展到 dense+sparse 跨分支共享，是合理的工程选择。

**延伸阅读**：主报告 CH 5.1 / `code-snippets/decoder_layer.py:L48-L53`

### Q5.7 为什么 Shortcut 输出在子层 2 末尾才接入残差？

**简短回答**：源码 `decoder_layer.py:L68` 的 `hidden_states = residual + hidden_states + shortcut_mlp_output` 把 MoE 输出延迟到第二子层完成才接入——这是 ScMoE 通算掩盖的核心设计，给 MoE 的 dispatch/compute/combine 整个子层 2 的时间窗口。

**详细解释**：

**如果 shortcut 在子层 1 末尾就接入**（传统 MoE 设计）：

```
forward:
  子层 1: attn[0] → norm → MoE (dispatch + compute + combine) → residual → dense mlp[0]
  
MoE 必须在 dense mlp[0] 启动前完成——否则残差无法计算。
dense mlp[0] 必须等 MoE，无法并行。
```

**LongCat 的延迟接入设计**：

```
forward:
  子层 1: attn[0] → norm → MoE dispatch（异步启动）→ dense mlp[0] → residual
                                ↓（异步）
  子层 2: attn[1] → norm → dense mlp[1] → residual + MoE 输出（此时 MoE 已完成）
```

**关键观察**：MoE 在子层 1 末尾启动 dispatch（Python 同步返回 tensor，ASIC 异步执行 all-to-all），子层 2 的 attn[1] + mlp[1] 计算期间 ASIC 同时跑 MoE 的 compute + combine。到子层 2 末尾接入残差时，MoE 输出已 ready。

**源码细节**（`decoder_layer.py:L51,L68`）：

```python
# 子层 1 末尾启动（L51）
shortcut_mlp_output = self.mlp(hidden_states)

# ... 子层 2 attn[1] + mlp[1] ...

# 子层 2 末尾接入（L68）
hidden_states = residual + hidden_states + shortcut_mlp_output
```

**时序分析**：

```
t0: 启动 MoE dispatch（async）
t1: 启动 dense mlp[0]（与 MoE dispatch 并行）
t2: dense mlp[0] 完成
t3: 启动 attn[1]（与 MoE compute 并行）
t4: attn[1] 完成
t5: 启动 dense mlp[1]（与 MoE combine 并行）
t6: dense mlp[1] 完成 + MoE combine 完成
t7: 残差接入 shortcut_mlp_output

关键：t6 时刻 MoE 必须完成——否则 t7 会 stall。
```

**实现约束**：MoE 必须在 `attn[1] + mlp[1]` 总时间内跑完。如果 MoE 通信延迟超过 attn[1]+mlp[1] 计算时间，通算掩盖失败，整体 stall。这就是为什么 LongCat 用 EP（expert parallel）+ ScMoE——EP 让 all-to-all 跨更小域，降低延迟。

**Trade-off**：
- 收益：通算完全掩盖（前提是 MoE 时间 < attn+mlp 时间）
- 代价：如果 MoE 时间过长，stall 风险——需要架构级保证

**易混淆**：ScMoE 的「延迟接入」与「异步计算」是两件事。延迟接入是残差流的设计（架构层），异步计算是 ASIC 调度（硬件层）。两者配合才能实现掩盖。

**延伸阅读**：主报告 CH 5.1 / `code-snippets/decoder_layer.py:L51,L68`

### Q5.8 为什么 LongCat 把 dense MLP 放在每个子层（而非只在 MoE 旁）？

**简短回答**：因为 LongCat 删除了 shared expert（V3 有），基础表达能力必须由 dense MLP 承担——每个 token 都需要走 dense MLP，所以放在每个子层。这样 dense 分支与 MoE 分支在架构上完全解耦。

**详细解释**：

源码 `decoder_layer.py:L21`：

```python
self.mlps = nn.ModuleList([LongcatFlashMLP(config) for _ in [0, 1]])  # 2 个 dense MLP
```

每个逻辑层有 2 个 dense MLP（每子层 1 个），共 76 个 dense MLP。

**Dense MLP 的角色**：

```
每 token forward（每子层）:
  h → norm → dense_mlp(h) + MoE(h)
              ↑              ↑
              基础表达        稀疏特化
```

dense MLP 学习「所有 token 共享的基础变换」（类似 shared expert 的角色），MoE 学习「token 特异的稀疏变换」。

**对比 V3 的 shared expert**：

```
V3 单层:
  h → norm → dense_attn → residual → norm → top-k expert + shared expert → residual
  
  shared expert: 每 token 必经的 dense FFN（与 top-k 专家并行）
```

V3 的 shared expert 与 top-k expert 在同一个 FFN 位置。LongCat 把「dense 部分」拆出来放到独立子层（`mlps[0]`, `mlps[1]`），让 MoE 子层（shortcut）可以独立优化。

**参数分解**：

```
每 dense MLP: gate + up + down = 3 × 8192 × 12288 = 301.99M
每子层 1 个 dense MLP × 76 子层 = 22.95B 总参数
```

这 22.95B 是「基础表达预算」——相比 MoE 的 1469B，占比 1.4%。

**为什么不把 dense MLP 参数投入更多 expert？**

1. **MoE 稀疏度上限**：97% 稀疏度已经接近 sweet spot，再加 expert 收益微乎其微（详见 Q2.3）
2. **dense MLP 必须每 token 跑**：它是「基础变换」，不能稀疏化
3. **与 ScMoE 配合**：dense MLP 是通算掩盖的「dense 分支」，必须有它才能让 ASIC 跑 dense compute 掩盖 MoE 通信

**Trade-off**：
- 收益：基础表达稳定 + ScMoE 有 dense 分支可用
- 代价：22.95B 参数「浪费」在每 token 都跑的 dense FFN 上（但相对总 1.6T 微不足道）

**面试要点**：LongCat 的 dense MLP 不是「V3 shared expert 的位置调整」，而是「ScMoE 架构的必要组件」——没有它通算掩盖就没有 dense 分支可跑。

**延伸阅读**：主报告 CH 5.1 / CH 5.3 / `code-snippets/decoder_layer.py:L21`

### Q5.9 `e_score_correction_bias` 是 `register_buffer` 而非 `Parameter`——这意味着什么？

**简短回答**：`register_buffer` 表示这个 bias **不参与梯度更新**，而是某种启发式更新（如基于路由统计的滑动平均）；它的 zeros 初始化 + 非 gradient 更新意味着「动态偏置机制」，但博客未公开具体算法——这是「实现细节待确认」字段。

**详细解释**：

源码 `topk_router.py:L20`：

```python
self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
```

`register_buffer` vs `nn.Parameter` 的区别：

| 维度 | `register_buffer` | `nn.Parameter` |
|---|---|---|
| 梯度更新 | 否（不进 optimizer） | 是 |
| 保存到 state_dict | 是 | 是 |
| 训练时如何变 | 手动更新（启发式） | 自动（反向传播） |
| 用途 | 非学习但需持久化的状态 | 学习参数 |

**V3 的 e_score_correction_bias 更新机制**：

V3 用 aux loss + e_score_correction_bias 动态调整：

```
loss_aux = α × (fraction_of_tokens_per_expert × probability_per_expert).mean()
e_score_correction_bias -= β × ∂loss_aux / ∂e_score_correction_bias
```

即：如果某 expert 被过度路由，bias 会被降低，抑制它被选。

**LongCat 的注释**（`moe.py:L9`）：

```
LongCat MoE differs from DeepSeekV3 in: no shared expert, no aux-loss bias, softmax router.
```

「no aux-loss bias」——意味着 LongCat **不**用 V3 的 aux loss 更新 bias。但 `e_score_correction_bias` 仍存在（zeros 初始化），它如何更新？

**可能的机制**（推测）：
1. **完全不更新**：始终是 zeros，只是保留接口
2. **基于路由统计的启发式更新**：训练中统计每 expert 被选频率，bias 反比于频率
3. **EPLB 推理时迁移**：训练时不动，推理时 EPLB 异步迁移 expert

博客和源码都没明确——这是「实现细节待确认」字段，主报告 CH 9.3 也列入局限。

**Trade-off**：
- 如果完全不更新：bias 是冗余字段，可删除
- 如果启发式更新：需要公开算法，否则社区无法复现训练

**面试要点**：读到 `register_buffer` 时，要问「谁更新它、什么时候更新、用什么算法」。如果代码里找不到更新逻辑，那要么是启发式（未公开），要么是死代码。

**延伸阅读**：主报告 CH 5.4 / CH 9.3 / `code-snippets/topk_router.py:L20` / `code-snippets/moe.py:L9`

### Q5.10 为什么 MoE 的 `gate_up_proj` 用 packed 存储（total_experts 维度包含 identity 槽位）？

**简短回答**：源码 `experts.py:L21-L27` 显示 `gate_up_proj` 形状是 `(total_experts, 2*intermediate, hidden)`，其中 `total_experts=896` 包含 128 个 identity 槽位——这些槽位的权重不会被使用，但保留它们让 router 的索引空间与权重张量第一维对齐，避免索引转换。

**详细解释**：

源码 `experts.py:L21-L27`：

```python
self.gate_up_proj = nn.Parameter(
    torch.empty(self.total_experts, 2 * self.intermediate_size, self.hidden_size)
    # total_experts = 768 + 128 = 896（包含 identity 槽位）
)
self.down_proj = nn.Parameter(
    torch.empty(self.num_routed_experts, self.hidden_size, self.intermediate_size)
    # 只有 768（identity 不需要 down_proj）
)
```

注意 `gate_up_proj` 是 896 行（含 identity 槽位），`down_proj` 是 768 行（不含）。这种不对称设计的考量：

**1. Router 输出对齐**：

router 输出 `topk_indices` 的取值范围是 `[0, 896)`（含 identity 槽位）。`experts.py:L53` 直接用：

```python
gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
```

如果 `expert_idx` 是 identity（≥768），`gate_up_proj[expert_idx]` 是未初始化的权重——但不会被执行，因为 `experts.py:L48-L50` 先判断：

```python
if expert_idx >= self.num_routed_experts or self.gate_up_proj is None:
    current_hidden_states = self.identity_expert(current_state)  # 走 identity 分支
```

所以 identity 槽位的 `gate_up_proj[expert_idx]` 权重虽然存在但永不使用——它们是「占位符」，让索引空间连续。

**2. EP 切分简化**：

如果 `gate_up_proj` 只存 768 行，EP 切分时需要把 router 输出映射到「真实 expert 索引」——复杂。保留 896 行让 EP 直接按 expert_idx 切分，无需索引转换。

**3. 显存代价**：

```
128 个 identity 槽位 × 2 × 2048 × 8192 × 2 bytes = 8.6 GB（额外显存）
```

8.6 GB 看似不小，但分散到 EP128 是每卡 67 MB——可忽略。

**为什么不直接用稀疏索引**？

可以做：

```python
# 替代方案：gate_up_proj 只存 768 行
if expert_idx >= 768:
    # identity 分支
else:
    F.linear(current_state, self.gate_up_proj[expert_idx])  # 索引 0-767
```

但这会让 EP 切分复杂——router 输出 [0, 896) 需要先映射到 [0, 768) 的真实 expert 索引，再查表。packed 方案省去了这层映射。

**Trade-off**：
- 收益：索引空间连续，EP 切分简单
- 代价：8.6 GB 额外显存（分散到每卡可忽略）

**面试要点**：深度学习工程中，经常用「冗余存储换索引简化」——只要冗余部分的显存可接受，就是合理工程选择。

**延伸阅读**：主报告 CH 5.3 / `code-snippets/experts.py:L21-L27,L48-L55`

---

## CH 6 N-gram Embedding

### Q6.1 为什么 N-gram Embedding 用 5-gram（`oe_neighbor_num=5`）而不是 3-gram 或 7-gram？

**简短回答**：5-gram 是「局部共现表达力」与「lookup 成本」的平衡——3-gram 太短（只能捕获紧邻依赖），7-gram lookup 表指数增长；5-gram 在自然语言中覆盖大多数局部模式（如固定搭配、短语边界）。

**详细解释**：

N-gram 的核心是「根据前 N-1 个 token + 当前 token 查 embedding」。`oe_neighbor_num=5` 意味着每个 token 查询自身 + 前 4 个 token 的 N-gram（即 5-gram）。

**N-gram 大小的 trade-off**：

| N | 表达力 | Lookup 表大小（vocab=163840） | I/O（每 token） |
|---|---|---|---|
| 2 (bigram) | 弱（只看前 1 token） | 163840² = 26B 行 | 2 行 × 16KB = 32KB |
| 3 (trigram) | 中（看前 2） | 163840³ = 4.4T 行 | 3 行 × 16KB = 48KB |
| **5 (5-gram)** | **强（看前 4）** | **163840^5 ≈ 1.2×10²³ 行（理论）** | **5 行 × 16KB = 80KB** |
| 7 (7-gram) | 更强 | 指数爆炸 | 7 行 × 16KB = 112KB |

实际 N-gram vocab 用 `oe_vocab_size_ratio=100.567` 压缩到 ~16.5M 行——不是理论 N^vocab，而是用 hashing 或 learned clustering 压缩。

**为什么 5 是 sweet spot？**

1. **自然语言局部依赖**：英文短语通常 3-5 词（"in spite of"、"as a result"），中文成语 4 字。5-gram 覆盖大多数局部模式。
2. **与 MoE 正交**：MoE 已经捕获长距离语义，N-gram 只需补「局部共现」——不需要太长。
3. **Lookup 成本可控**：5 行 × 16KB = 80KB/token，相对 MoE 1.2MB/token 是 6.7% 额外 I/O。

**数字代入**：

```
5-gram 激活（每 token）:
  查 5 行 embedding（self + 前 4）
  每行 8192 维 × 2 bytes = 16 KB
  总 I/O = 80 KB/token

vs MoE top-12 expert:
  12 × 100 KB = 1.2 MB/token

I/O 比 = 1.2 MB / 80 KB = 15×
```

**参数对比**：

```
5-gram embedding 参数 = 135B（oe_vocab_size_ratio × vocab × hidden = 100.567 × 163840 × 8192）
7-gram embedding 参数（假设 ratio 不变）≈ 仍 135B（ratio 不依赖 N）
  → 实际 ratio 会随 N 上升而上升（需要更多 hash bucket），但非线性
```

所以 5 vs 7 的参数差异不大，但 lookup I/O 7 比 5 多 40%。

**Trade-off**：
- 5-gram：覆盖 95%+ 局部模式，I/O 可控
- 7-gram：覆盖更广，但 I/O 显著增加，边际收益小

**面试要点**：N-gram size 选择是经验性的，没有严格理论最优。5 是业界常用值（如 BERT 系列、GPT 系列早期 work）。

**延伸阅读**：主报告 CH 6.1 / CH 6.2 / config.json `oe_neighbor_num`

### Q6.2 N-gram Embedding 135B 占总参 8.4%——为什么不把这些参数投到更多 expert？

**简短回答**：因为 MoE 稀疏度 97% 已经过了 sweet spot——再加 expert 几乎不被命中（激活率从 1.3% 降到 0.7%），参数效率急剧下降；而 N-gram Embedding 是与 MoE 正交的新稀疏轴，I/O 效率高 15×。

**详细解释**：

这是 LongCat-2.0 设计的核心决策，需要量化分析两个选项：

#### 选项 A：把 135B 投到更多 expert

假设把 expert 池从 768 扩到 1024（增加 256 个 expert，每个 50.33M，共 12.9B），或加宽 expert FFN（`expert_ffn_hidden_size` 从 2048 到 4096）。

**问题**：
- top-12 / 1024 = 1.2% 激活率（vs 现在 1.3%）——几乎不变
- 单 expert 期望命中次数：`tokens × 12 / 1024`（减半）→ expert 训练不充分
- I/O 不变（每 token 仍 top-12 × 全 FFN）

如果用「加 expert 数 + 加 top-k」：
- top-24 / 1024 = 2.3%（与现在 1.3% 相比增加 I/O 2×）
- 每 token I/O 从 1.2 MB 升到 2.4 MB——大 batch decode 时严重瓶颈

**结论**：135B 投到 expert 收益微乎其微。

#### 选项 B：投到 N-gram Embedding

```
135B / (8192 × 2 bytes) = 8.4B 行 embedding
用 oe_vocab_size_ratio=100.567 压缩到 16.5M 行（每行 8192 维）

每 token 激活 5 行（oe_neighbor_num=5）
I/O = 5 × 16KB = 80 KB/token

vs 等效参数 expert:
  135B / 50.33M = 2683 个 expert
  每 token 激活 top-k=12 → 查 12 个 expert
  I/O = 12 × 100KB = 1.2 MB/token
```

**I/O 效率比**：N-gram 比 expert 高 1.2MB / 80KB = **15×**

**正交性收益**：

N-gram 学习的是「确定性局部共现」（前 4 个 token 决定当前 token 的 embedding 增强），与 MoE 学习的「语义路由」完全正交。两者叠加提供两条独立信息通路：

```
单 token 信息流:
  hidden = token_emb(t) + ngram_emb(t-3, t-2, t-1, t)
  → MoE 路由（学语义）
  → attention（学长距离）
  → output
```

N-gram 在 token_emb 阶段就注入局部先验，让后续 attention / MoE 看到的输入已经包含局部模式。

**量化对比**：

| 方案 | 参数 | I/O/token | 表达轴 |
|---|---|---|---|
| 加 expert | 135B | 1.2 MB | 同 MoE（语义） |
| **N-gram** | **135B** | **80 KB** | **局部共现（正交）** |

N-gram 在参数相同情况下，I/O 节省 15×，且开辟新表达轴。

**Trade-off**：
- 收益：I/O 大幅节省 + 表达轴扩展
- 代价：架构复杂度（需要 EMBP 并行）+ N-gram lookup 不可微学习（但 embedding 行梯度可回传）

**面试要点**：LongCat 的「参数效率」优化不在「减少参数」而在「让每参数贡献更多」——N-gram 是 I/O 效率 15× 于 expert 的参数使用方式。

**延伸阅读**：主报告 CH 6.1 / CH 6.4 / 博客「N-gram Embedding」节

### Q6.3 `oe_split_num=4`（N-gram 分 4 片）——为什么是 4 而不是 8 或 16？

**简短回答**：4 片对应 EMBP（Embedding Parallel）专用并行维度——每片 33.75B 参数分布在若干 DP rank 上；4 是「通信开销」与「单卡显存」的平衡，更多片会让 all-to-all 通信指数增长。

**详细解释**：

N-gram Embedding 135B 按 `oe_split_num=4` 切分：

```
每片参数 = 135B / 4 = 33.75B
每片显存 = 33.75B × 2 bytes = 67.5 GB（BF16）
```

如果单机 8 卡 ASIC（每卡 80GB），单卡 67.5GB 几乎吃满——所以每片需要跨多卡分布。

**EMBP 与 DP 的关系**：

EMBP 是 LongCat 独有的第 6 维并行，专门为 N-gram 设计。与传统 DP（数据并行，每 rank 持完整参数）不同，EMBP 切的是参数本身——每 rank 只持 1/4 的 N-gram embedding。

**Query 流程**：

```
token t 需要 N-gram lookup
  → 查询 self + 前 4 token 的 hash
  → hash 可能命中片 0/1/2/3 中的任意行
  → all-to-all：每 rank 查自己片，结果汇总
  → 得到 5 行 embedding
```

**通信成本**：

```
每次 lookup 的通信 = 5 行 × 8192 维 × 2 bytes × 4 rank
                  = 40 KB × 4 = 320 KB/token
```

如果分片数增加到 8：
```
通信 = 40 KB × 8 = 640 KB/token（2× 增长）
```

如果分片数增加到 16：
```
通信 = 40 KB × 16 = 1.28 MB/token（4× 增长）
```

**单卡显存对比**：

```
4 片: 每片 67.5 GB → 单机 8 卡可容纳 1 片（单卡 8.4 GB）
8 片: 每片 33.75 GB → 单机 4 卡可容纳 1 片（单卡 8.4 GB）
16 片: 每片 16.88 GB → 单机 2 卡可容纳 1 片（单卡 8.4 GB）
```

**通信 vs 显存的平衡**：

| oe_split_num | 单卡显存 | all-to-all 通信 | 推荐场景 |
|---|---|---|---|
| 2 | 16.88 GB | 160 KB/token | 单机多卡 |
| **4** | **8.44 GB** | **320 KB/token** | **小集群（推荐）** |
| 8 | 4.22 GB | 640 KB/token | 中型集群 |
| 16 | 2.11 GB | 1.28 MB/token | 大集群（通信瓶颈） |

4 是「通信开销可接受 + 单卡显存合理」的 sweet spot。

**Trade-off**：
- 收益：4 片让 N-gram 可分布在 4 个 DP rank，降低单卡显存压力
- 代价：每次 lookup 需要 4 路 all-to-all（320 KB/token 通信）

**面试要点**：EMBP 是 LongCat 独创的并行维度，源于 N-gram Embedding 的「lookup 而非 matmul」特性——传统 DP 切参数会让 matmul 通信爆炸，但 lookup 通信稀疏可控。

**延伸阅读**：主报告 CH 6.3 / CH 7.1 / 博客「Training」节「6D Parallelism」

### Q6.4 N-gram Embedding 与 token Embedding 是相加还是拼接？

**简短回答**：源码（内置 longcat_flash）未实现 N-gram Embedding，但根据博客描述「N-gram Embedding 作为独立稀疏维度」和 OE（Orthogonal Embedding）命名约定，最可能是「lookup 后与 token_emb 相加」——保持 hidden_size 不变，注入局部先验。

**详细解释**：

源码 `model.py:L49` 只看到标准 token embedding：

```python
inputs_embeds = self.embed_tokens(input_ids)   # 无 N-gram 实现
```

LongCat-2.0 的 N-gram 实现仅在 SGLang PR #30042，开源 fallback 缺失。但可以做以下推断：

**方案 A（最可能）：相加**

```python
# 伪代码
token_emb = self.embed_tokens(input_ids)                # (B, S, 8192)
ngram_emb = self.ngram_lookup(input_ids, oe_neighbor_num=5)  # (B, S, 8192)
hidden = token_emb + ngram_emb                          # 相加，保持 hidden 维度
```

**为什么是相加而非拼接？**

1. **保持 hidden_size**：拼接会让 hidden 从 8192 变 16384，破坏所有下游投影的形状约定
2. **类比 word + position embedding**：标准 transformer 中 token + position 也是相加，N-gram 作为「局部位置先验」类比合理
3. **梯度回传简单**：相加让 token_emb 和 ngram_emb 都能独立接收梯度

**方案 B：concat + projection**

```python
hidden = torch.cat([token_emb, ngram_emb], dim=-1)  # (B, S, 16384)
hidden = self.merge_proj(hidden)                     # (B, S, 8192)
```

这种方案多一个 merge_proj 矩阵（8192 × 16384 = 134M 参数），但能学更复杂的 token-ngram 融合。LongCat 团队可能没采用——因为参数效率不如直接相加。

**方案 C：gated combination**

```python
gate = sigmoid(self.gate_proj(token_emb))   # (B, S, 8192)
hidden = gate * token_emb + (1 - gate) * ngram_emb
```

更灵活但参数更多。

**LongCat 选择推断**：方案 A（相加），最简单、最参数高效、最符合「正交稀疏」哲学——N-gram 只是给 hidden 加一个「局部先验偏置」，不改变主干架构。

**开源状态**：需明确标注「源码未实现，基于博客描述推断」。SGLang PR #30042 应该有实际实现，但未在主线。

**易混淆**：N-gram Embedding 与 position embedding 是不同概念。position encoding 是「token 在序列中的位置」，N-gram 是「token 与前 N-1 个 token 的组合模式」——前者是位置先验，后者是局部共现先验。

**延伸阅读**：主报告 CH 6.1 / `code-snippets/model.py:L49` / SOURCES.md L13-L17

### Q6.5 N-gram Embedding 在推理时如何降低 I/O？

**简短回答**：相比把同样参数投到 expert FFN（每 token 激活 12 个 expert × 100KB = 1.2MB），N-gram 每 token 只查 5 行 embedding × 16KB = 80KB——I/O 效率高 15×，让大 batch decode 时 I/O 不再是瓶颈。

**详细解释**：

大 batch decode 的 I/O 瓶颈来自「权重读取」。每生成一个 token，需要从 HBM 读取：
1. MLA 权重（约 112M/子层 × 76 子层 × 2 bytes = 17 GB）
2. Dense MLP 权重（约 302M/子层 × 76 子层 × 2 bytes = 46 GB）
3. MoE expert 权重（top-12 个 expert × 100KB = 1.2 MB/token）
4. N-gram Embedding（5 行 × 16KB = 80 KB/token）

权重读取是「batch 共享」——大 batch 时每 token 摊销的 I/O 减少。但 MoE 和 N-gram 是「per-token」I/O：

```
大 batch decode (batch=B):
  MLA + Dense MLP 权重 I/O: 总 / B（每 token 摊销）
  MoE expert I/O: B × 1.2 MB（每 token 独立 top-12）
  N-gram I/O: B × 80 KB（每 token 独立 5 行）
```

**关键对比**：

如果 135B 参数全部投到 expert（替代 N-gram），每 token I/O 增加：

```
原 N-gram I/O: 80 KB/token
替代方案 expert I/O: 1.2 MB/token（同等参数量下）

大 batch decode 时差异放大:
  B=128:
    N-gram 总 I/O: 128 × 80 KB = 10 MB
    等效 expert I/O: 128 × 1.2 MB = 154 MB（15× 更多）
```

**为什么 N-gram I/O 这么低？**

N-gram 是「lookup」操作——每 token 只读取「自己 5-gram 对应的 5 行」，不读取整个 135B embedding 表。

expert FFN 是「matmul」操作——每 token 必须「读取 top-12 个 expert 的完整 FFN 权重」（gate + up + down 共 50.33M 参数）。

```
N-gram 单行 I/O: hidden_size × 2 bytes = 8192 × 2 = 16 KB
expert 单 FFN I/O: 3 × hidden × expert_ffn × 2 bytes = 100 KB
```

N-gram 每行 16KB，expert 每 FFN 100KB——差 6.25×。加上 top-k 差异（5 vs 12），总 I/O 差 15×。

**量化收益**：

```
大 batch decode (B=128, 1M seq):
  用 N-gram:  I/O 总 = 10 MB/token × 1M tokens = 10 TB/decode step
  用等效 expert: I/O 总 = 154 MB/token × 1M tokens = 154 TB/decode step
  
  HBM 带宽假设 3 TB/s:
    N-gram: 10 TB / 3 TB/s = 3.3 秒/step
    等效 expert: 154 TB / 3 TB/s = 51 秒/step（15× 慢）
```

**Trade-off**：
- 收益：大 batch decode I/O 降低 15×
- 代价：N-gram 表达力受限（只学局部共现）

**面试要点**：分析 LLM 推理性能时，I/O 与 compute 同样重要。大 batch decode 通常是 I/O bound——N-gram 是针对这个场景的优化。

**延伸阅读**：主报告 CH 6.2 / CH 6.4

### Q6.6 N-gram Embedding 与 MoE 的「正交性」具体指什么？

**简短回答**：两者在「稀疏触发方式」上正交——MoE 用 router 学习的 token-expert 相似度（语义路由），N-gram 用确定性的 N-gram 模式（局部共现）；它们学习的特征空间互不重叠。

**详细解释**：

「正交性」在稀疏设计中是强约束——两个稀疏机制如果学同样的特征，参数就浪费了。正交意味着各自学习不同维度的信息。

**两个稀疏轴的对比**：

| 维度 | MoE | N-gram Embedding |
|---|---|---|
| 稀疏触发 | Router 学习的 softmax 分数 | 确定性的 N-gram hash |
| 学习内容 | 语义 / 语法 / 逻辑模式 | 局部共现统计 |
| 训练时 | Router 梯度更新 | Embedding 行梯度更新 |
| 推理时 | top-k 路由（动态） | N-gram lookup（确定性） |
| 表达力 | 长 + 短距离 | 仅局部（前 4 token） |
| 可解释性 | 弱（学到的路由语义不透明） | 强（N-gram 是显式模式） |

**举例说明正交性**：

句子 "The cat sat on the mat"：

- **MoE 路由** "sat" → 可能路由到「动词处理 expert」「动作语义 expert」「过去时态 expert」
- **N-gram lookup** "sat" → 查询 ["The cat sat", "cat sat on", "sat on the", "on the mat", "the mat <eos>"] 的 embedding

MoE 关注「sat 是动词、是动作」，N-gram 关注「sat 前面是 cat、后面是 on」。两者信息互补，不重叠。

**为什么正交性重要？**

如果两个稀疏机制学同样特征（如都用 router 学习语义路由），参数会冗余。LongCat 的设计确保：

1. **MoE 学语义**：router 自动学习「什么 token 该走什么 expert」
2. **N-gram 学统计**：embedding 行是「这个 N-gram 出现过多少次、与什么上下文共现」的直接记忆

两者在「学习算法」「触发方式」「表达内容」三方面都不同——这是正交性的本质。

**训练梯度流**：

```
forward:
  hidden = token_emb(t) + ngram_emb(t-3, t-2, t-1, t)
  h = MoE(hidden)  # top-12 routed expert
  output = h + attention(h)

backward:
  ∂L/∂ngram_emb: 通过 hidden 流回，直接更新对应 5 行 embedding
  ∂L/∂expert_weights: 通过 MoE router 流回，只更新被选中的 top-12 expert
```

梯度更新路径完全独立——N-gram 行梯度不影响 expert 权重，反之亦然。这是正交性的数学基础。

**对比 Hy3 / V4 的稀疏扩展**：

| 模型 | 第二稀疏轴 | 与 MoE 正交性 |
|---|---|---|
| **LongCat-2.0** | **N-gram Embedding** | **强（学习算法完全不同）** |
| Hy3-Flash | Layer-wise KV 共享 | 中（与 attention 耦合） |
| V4-Flash | Multi-Token Prediction | 弱（MTP 是训练目标，不是稀疏机制） |

LongCat 的正交设计最纯粹——N-gram 与 MoE 在算法层完全解耦。

**Trade-off**：
- 收益：两个稀疏轴提供互补信息，参数效率最大化
- 代价：架构复杂度（需要两套独立的稀疏机制）

**面试要点**：分析多稀疏机制组合时，要问「它们学习的特征是否重叠」。如果重叠，参数浪费；如果正交，1+1>2。

**延伸阅读**：主报告 CH 6.4 / CH 9.1

---

## CH 7 训练体系

### Q7.1 6D 并行中 EMBP 是 LongCat 独有的——它解决什么问题？

**简短回答**：EMBP（Embedding Parallel）专门为 135B N-gram Embedding 设计，按 `oe_split_num=4` 把 embedding 表切到 4 个 DP rank——传统 DP 切的是 batch 维度，EMBP 切的是参数本身（用稀疏 lookup 通信，不是 dense matmul 通信）。

**详细解释**：

**传统 5D 并行（TP/CP/EP/DP/PP）**：

| 维度 | 切什么 | 通信类型 |
|---|---|---|
| TP | 单层权重矩阵 | dense all-reduce |
| CP | 序列维度 | ring attention / all-gather |
| EP | MoE expert | all-to-all |
| DP | Batch | gradient all-reduce |
| PP | 层间 | pipeline bubble |

N-gram Embedding 135B 不适合任何传统切分：
- TP：embedding lookup 不是矩阵乘，无法沿 head 切
- EP：N-gram 不是 expert，无法按 expert 切
- DP：传统 DP 每 rank 持完整 135B，冗余 4×（如果 DP=4）
- PP：embedding 在模型入口，无法沿层切

**EMBP 的方案**：

```
N-gram embedding 表 135B
  → 按 oe_split_num=4 切成 4 片，每片 33.75B
  → 每个 EMBP rank 持 1 片
  → token lookup 时 all-to-all 通信（每 rank 查自己的片，汇总结果）
```

**通信模式**：

```
token t 需要 N-gram embedding:
  1. 计算 hash(ngram_5) → 落在片 X（X ∈ {0,1,2,3}）
  2. 发送查询到片 X 所在 rank
  3. 片 X rank 查 embedding 行，返回结果
  4. 汇总 5 行（self + 前 4 token）的 embedding
```

**关键特性**：通信是「稀疏 all-to-all」——只交换被查询的行，不是 dense 矩阵通信。

**通信量估算**：

```
每 token lookup:
  5 行 × 8192 dim × 2 bytes × 4 rank = 320 KB/token

大 batch (B=128, seq=1M):
  128 × 1M × 320 KB = 40 TB 通信量
  
HBM 带宽 3 TB/s:
  40 TB / 3 TB/s = 13 秒/step（可接受）
```

**为什么传统 DP 不行？**

传统 DP（数据并行）每 rank 持完整 135B N-gram embedding：
- DP=4 时，4 个 rank 共持 4 × 135B = 540B N-gram 参数——冗余 4×
- 梯度 all-reduce 通信 540B × 2 bytes = 1 TB（每 step）——通信爆炸

EMBP 切参数本身，每 rank 持 33.75B——总参数仍是 135B，无冗余。

**Trade-off**：
- 收益：135B N-gram 可分布在 4 个 rank，无参数冗余
- 代价：每 token lookup 需 all-to-all（稀疏，可控）

**面试要点**：EMBP 是 LongCat 独创维度的核心创新——它源于 N-gram Embedding 的「lookup 而非 matmul」特性，传统并行方案都不适配。

**延伸阅读**：主报告 CH 7.1 / CH 6.3 / 博客「Training」节「6D Parallelism」

### Q7.2 Muon 优化器相比 AdamW 有什么优势？为什么 LongCat 用 Muon？

**简短回答**：Muon 用 Newton-Schulz 迭代对梯度做正交化，更新方向更稳定；在 ASIC 上有专用对称矩阵乘 kernel 加速——相比 AdamW 的 momentum + variance，Muon 的 5 次对称矩阵乘更适合 ASIC 的硬连线优化。

**详细解释**：

**AdamW 的工作机制**：

```
m_t = β₁ · m_{t-1} + (1-β₁) · g_t          # 一阶动量
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²         # 二阶动量
m̂ = m_t / (1 - β₁ᵗ)                          # bias correction
v̂ = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - η · m̂ / (√v̂ + ε)            # update
```

AdamW 维护两个状态（m, v），每个参数一份。优点：自适应学习率，对超参不敏感。缺点：状态占用显存 2× 参数量。

**Muon 的工作机制**：

```
G = orthogonalize(g)   # Newton-Schulz 迭代
  G₁ = g
  G₂ = G₁ · (3I - G₁ᵀ·G₁) / 2
  G₃ = G₂ · (3I - G₂ᵀ·G₂) / 2
  ...（5 次迭代）
  G_orth ≈ G₅

θ_t = θ_{t-1} - η · G_orth
```

Muon 把梯度投影到正交基上，更新方向是「梯度的正交近似」。优点：
1. **状态少**：不维护 m, v，只维护当前梯度（节省显存）
2. **更新稳定**：正交化过滤梯度噪声
3. **收敛快**：每步更新更精确（实测比 AdamW 快 1.5-2×）

**为什么 LongCat 用 Muon？**

博客「Training」节披露三点：

1. **TP 并行适配**：Muon 的对称矩阵乘 kernel 与 TP 切分协同——Newton-Schulz 迭代中的 `G₁ᵀ·G₁` 可以沿 TP 切分并行计算
2. **DP 状态冗余消除**：传统 AdamW 每 DP rank 存完整 optimizer state（2× 参数 = 3.2 TB 冗余），Muon 通过数学等价变换消除冗余
3. **对称矩阵乘 kernel 优化**：5 次对称矩阵乘在 ASIC 上有专用 kernel 加速

**ASIC 优化的关键**：

Newton-Schulz 迭代的 5 次对称矩阵乘（`G · Gᵀ`）是计算密集型——ASIC 可以为这个特定 pattern 设计硬连线加速器，比 GPU 的通用 matmul 快数倍。

**显存对比**（对 1.6T 模型）：

```
AdamW state: 1.6T × 2 (m + v) × 4 bytes (FP32) = 12.8 TB
Muon state: 1.6T × 4 bytes (FP32 grad) = 6.4 TB（一半）

ZeRO-1 切分到 DP=64:
  AdamW: 12.8 TB / 64 = 200 GB / rank（超出单卡）
  Muon: 6.4 TB / 64 = 100 GB / rank（仍超）
  → 必须配合 ZeRO-2/3 或参数切分
```

**Trade-off**：
- 收益：收敛快 1.5-2× + 显存省一半 + ASIC 加速
- 代价：Newton-Schulz 迭代本身有计算开销（5 次 matmul）+ 超参敏感（迭代次数、学习率）

**易混淆**：Muon 不是「AdamW 的优化版」，而是完全不同的优化器——基于矩阵正交化理论，不是 momentum 理论。

**延伸阅读**：主报告 CH 7.3 / 博客「Training」节「Optimizer」

### Q7.3 训练 determinism（通信+计算路径完全确定）为什么重要？

**简短回答**：1.6T 模型在 50K+ ASIC 上训练，任何非确定性（如浮点累加顺序、原子操作、动态调度）都会导致「同样输入、不同结果」——阻碍调试、checkpoint resume、bug 复现；determinism 让训练过程可重现。

**详细解释**：

**非确定性的来源**：

1. **浮点累加顺序**：`(a + b) + c ≠ a + (b + c)`（浮点不满足结合律）。GPU 上的 reduce 算子（如 all-reduce）由于 warp 调度顺序不固定，累加顺序会变
2. **原子操作**：`atomicAdd` 在 GPU 上是 race condition——多个线程同时加，结果不可预测
3. **动态调度**：NCCL all-to-all 的路径选择、kernel 启动时序，都受系统状态影响
4. **Bit-flip**：硬件级单粒子翻转（宇宙射线、ASIC 缺陷）

**为什么 50K+ ASIC 训练特别需要 determinism？**

- **Debug 难度**：训练 loss 突然 spike，如果是非确定性导致，无法复现 → 无法定位 bug
- **Checkpoint resume**：训练中断后从 checkpoint 恢复，如果状态不严格确定，恢复后 loss 会跳变
- **多 run 对比**：对比「配置 A vs 配置 B」时，如果单 run 本身有随机性，差异可能来自噪声而非配置

**LongCat 的 determinism 设计**：

博客披露：
1. **通信 + 计算路径完全确定**（reduced non-determinism in both comm and compute paths）
2. **二叉树分段累加**：reduce 类算子用二叉树分段累加，避免长链累加的数值漂移
3. **Bit-flip 检测**：硬件级 bit-flip 检测
4. **OOM-aware offloading**：显存压力时自动 offload

**二叉树分段累加详解**：

```
传统串行累加（数值漂移）:
  sum = 0
  for i in range(N):
    sum += x[i]   # 每步浮点误差累积
  
N=1M 时，串行累加误差约 1M × ε ≈ 1e-10（看似小，但梯度反向传播会放大）

二叉树累加（确定且数值稳定）:
  level 0: [x0+x1, x2+x3, x4+x5, ...]
  level 1: [(x0+x1)+(x2+x3), ...]
  ...
  log(N) 层后得 sum
  
累加路径固定（不依赖调度），误差 log(N) × ε ≈ 20 × ε
```

二叉树不仅确定（路径固定），还更精确（误差对数级而非线性）。

**Bit-flip 检测**：

宇宙射线或 ASIC 缺陷会让某个 bit 翻转（`0→1` 或 `1→0`），导致计算结果错误。LongCat 用 ECC（Error Correction Code）或冗余计算检测 bit-flip，发现后回滚到上一个 checkpoint。

**Trade-off**：
- 收益：训练可复现 + 调试可行 + checkpoint 稳定
- 代价：性能损失 5-15%（确定路径通常比随机路径慢）+ 实现复杂

**易混淆**：「determinism」与「reproducibility」是相关概念。determinism 是「同输入同输出」（单步），reproducibility 是「同配置同训练曲线」（端到端）。前者是后者的必要条件。

**延伸阅读**：主报告 CH 7.4 / 博客「Training」节「Numerical Reliability」

### Q7.4 All-gather-based CP（上下文并行）相比 Ring-Attention 有什么优势？

**简短回答**：All-gather-based CP 在 ASIC all-to-all 拓扑上更高效——Ring-Attention 是「环形通信 + 局部计算」串行，All-gather 是「一次性收集 + 局部计算」并行；LongCat 的 ASIC superpod 有高带宽 all-to-all，适合后者。

**详细解释**：

**Context Parallel 的核心问题**：

1M 上下文 × 76 子层 × 576 维 KV cache = 91.66 GB——单卡放不下。必须把序列维度切到多卡。

**Ring-Attention（GPU 标准）**：

```
CP=N 卡，序列切成 N 段，每卡持 1 段 KV
  Step 1: 每卡计算本地 Q × 本地 KV
  Step 2: KV 沿环传递（卡 0 → 卡 1 → ... → 卡 N-1 → 卡 0）
  Step 3: 每卡收到下一段 KV，计算 Q × 新 KV
  ...
  N 步后，每卡的 Q 已与所有 KV 计算过
  
通信: N-1 次 KV 传递（环）
计算: 每步并行（N 卡同时算）
```

Ring-Attention 的优势：通信与计算可 overlap（每步传下一段 KV 时，本卡在算当前段）。

劣势：
- N 步串行依赖（最后一步必须等前面 N-1 步）
- 环形拓扑对 GPU 友好（NCCL ring），但对 ASIC all-to-all 不友好

**All-gather-based CP（LongCat）**：

```
CP=N 卡，序列切成 N 段
  Step 1: All-gather —— 每卡把本地 KV 发给所有其他卡
          （一次性收集所有 KV 到本地）
  Step 2: 每卡独立计算 Q × 完整 KV（无通信）
  
通信: 1 次大 all-gather（一次性）
计算: 完全并行（无串行依赖）
```

**关键差异**：

| 维度 | Ring-Attention | All-gather CP |
|---|---|---|
| 通信步数 | N-1 | 1 |
| 计算并行度 | 受限于环形串行 | 完全并行 |
| 单步通信量 | KV 段大小 | 全部 KV（N × 段大小） |
| 总通信量 | N × 段（环传一圈） | N × 段（all-gather 等效） |
| 拓扑要求 | Ring | All-to-all |

**为什么 LongCat 选 All-gather？**

1. **ASIC 拓扑**：LongCat superpod 有高带宽 all-to-all + RoCE fabric，all-gather 一次性传输效率高
2. **计算密集型**：1M 上下文下，attention 计算量巨大，all-gather 后的「完全并行计算」比 ring 的「串行 N 步」快
3. **与 LSA 配合**：LSA 已经把 attention 从 O(N²) 降到 O(N × k)，all-gather 的 KV 量本来就不大

**通信量估算**（1M seq，CP=512）：

```
每段 KV = 1M / 512 × 576 × 2 bytes = 2.25 MB
All-gather 总通信 = 512 × 2.25 MB = 1.13 GB（每卡收到全部 512 段）

vs Ring:
  每步传 2.25 MB × 511 步 = 1.15 GB（总通信量相近）
  但 Ring 是串行 511 步，All-gather 是 1 步并行
```

**Trade-off**：
- 收益：计算完全并行，无串行依赖
- 代价：需要高带宽 all-to-all 拓扑（ASIC superpod 满足，GPU 集群不一定）

**易混淆**：All-gather CP 与 All-gather 数据并行不同。前者是「序列维度切分后 all-gather KV」，后者是「batch 维度切分后 all-gather 梯度」。

**延伸阅读**：主报告 CH 7.5 / 博客「Training」节

### Q7.5 Dense warmup + KL loss 在 LSA 训练中的作用是什么？

**简短回答**：训练前向先跑 dense attention warmup，再用 KL 散度对齐 LSA 输出与 dense attention 输出——保证稀疏注意力与 dense 注意力行为一致，避免训练初期 LSA 输出偏离导致梯度爆炸。

**详细解释**：

**问题背景**：LSA 把 dense attention（O(N²)）替换为 sparse attention（O(N × k)），但训练初期 LSA 的 indexer 还没学好——选的 top-k KV 可能不是最优，导致 attention 输出偏离 dense。

**LongCat 的解法**：

```
训练每 step:
  1. Dense warmup forward:
     h_dense = dense_attention(hidden)   # O(N²) 全注意力
     
  2. LSA forward:
     h_lsa = LSA(hidden)                  # O(N × k) 稀疏注意力
     
  3. KL loss:
     L_kl = KL(h_lsa || h_dense.detach())  # 让 LSA 向 dense 对齐
     
  4. 总 loss:
     L_total = L_task + λ · L_kl
```

**KL 散度的作用**：

KL 散度衡量两个分布的差异。这里 dense attention 输出作为「老师」（detach 防止梯度回传），LSA 作为「学生」向 dense 学习。

随着训练进行，LSA 的 indexer 逐渐学到「哪些 KV 真正重要」，L_kl 收敛——此时可以降低 λ 或移除 KL loss。

**类比 distillation**：

这本质是「self-distillation」——dense attention 是 LSA 的老师。区别于传统 distillation（大模型教小模型），这里是「同模型 dense 模式教 sparse 模式」。

**训练流程**：

```
Phase 1（前 X% steps）: λ 大，强对齐
  → LSA 快速学到 dense 的行为
  
Phase 2（中间 Y% steps）: λ 逐渐减小
  → LSA 开始自主优化（dense 不再是 ground truth）
  
Phase 3（最后 Z% steps）: λ = 0，纯 LSA
  → 模型完全依赖 LSA，dense 仅用于 warmup（可能仍保留用于诊断）
```

**Trade-off**：
- 收益：训练稳定，避免初期梯度爆炸 + 加速 LSA 收敛
- 代价：每 step 多跑一次 dense forward（计算开销大）

**为什么 dense warmup 仍保留？**

即使在 Phase 3，dense warmup forward 仍可能保留（博客未明确）——作为「诊断信号」检查 LSA 是否偏离。如果某 step L_kl 突然增大，说明 LSA 出问题，可以触发 checkpoint rollback。

**易混淆**：KL loss 与 CLI 的 cross-layer distillation 是不同 distill。前者是 dense → LSA 的对齐，后者是 layer 2i → layer 2i+1 的 indexer mask 共享对齐。

**延伸阅读**：主报告 CH 7.5 / 博客「Training」节「Long-Context Training」

### Q7.6 YaRN factor=120 外推到 1M——为什么不直接训练 1M？

**简短回答**：直接训练 1M 上下文成本过高（CP ≥ 512 路并行、显存爆炸、attention O(N²) 不可行）；YaRN 通过频率外推让模型在 256K 训练，推理时外推到 1M——训练成本可控 + 推理长度灵活。

**详细解释**：

**YaRN（Yet another RoPE extensioN）原理**：

RoPE 的核心是「用旋转角度编码位置」。频率高的维度旋转快（编码近距离），频率低的维度旋转慢（编码远距离）。

YaRN 通过调整 RoPE 的「基础频率」让模型外推：
- 训练时 max_pos=262144（256K）
- 推理时 max_pos=1M（4× 外推）
- factor=120 把 RoPE 的频率压缩，让低频维度的旋转角度在 1M 位置仍合理

**为什么不直接训练 1M？**

1. **显存爆炸**：1M seq × 76 子层 × 576 维 KV cache = 91.66 GB（单 sample）——训练时 batch > 1，显存爆炸
2. **计算爆炸**：attention O(N²) → 1M² = 10¹² operations/layer，单 step 训练时间不可接受（即使有 LSA）
3. **数据稀缺**：1M token 的训练样本（长文档、长代码）数量有限，训练数据不足
4. **CP 切分困难**：1M 上下文需要 CP ≥ 512，通信开销大，负载均衡难

**YaRN 的解法**：

```
训练阶段:
  - max_position_embeddings = 262144（256K）
  - 在 256K 长度上学习 RoPE 频率
  - 数据充足、计算可行

推理阶段:
  - factor = 120（120 × 外推）
  - original_max = 8192 → 训练 max = 262144 → 推理 max = 1M
  - YaRN 调整 RoPE 让低频维度外推到 1M 仍合理
```

**YaRN config 详解**：

```json
"rope_scaling": {
  "original_max_position_embeddings": 8192,
  "rope_type": "deepseek_yarn",
  "factor": 120,
  "beta_fast": 32,
  "beta_slow": 1,
  "mscale": 1,
  "mscale_all_dim": 1
}
```

- `original_max=8192`: 基础 RoPE 的设计长度
- `factor=120`: 8192 × 120 = 983040 ≈ 1M（实际 1,048,576）
- `beta_fast/slow`: 频率混合参数——快频保留（局部精度），慢频外推（长距离泛化）
- `mscale=1`: 长上下文 attention 输出量级补偿

**外推质量**：

YaRN 外推不是「免费」——1M 推理时，模型对 >256K 的位置仍可能表现下降（外推 loss）。LongCat 用 LSA 缓解：sparse attention 让长上下文不需要「精确」O(N²) attention，降低了对外推精度的依赖。

**Trade-off**：
- 收益：训练成本可控（256K），推理可扩展（1M）
- 代价：外推精度损失（虽然 LSA 缓解）+ config 复杂

**面试要点**：YaRN 是当前长上下文 LLM 的标准方案（DeepSeek、Qwen 等都用）。factor 选择是 trade-off——太大（如 200）外推质量差，太小（如 30）训练长度受限。

**延伸阅读**：主报告 CH 7.5 / config.json `rope_scaling`

### Q7.7 CP ≥ 512 是什么概念？为什么至少 512 路？

**简短回答**：CP=512 意味着 1M 序列切到 512 个 ASIC rank，每 rank 处理 ~2048 token——这是「单卡 attention 计算量」与「all-gather 通信量」的平衡；CP 太小（如 64）每卡负担太重，CP 太大（如 2048）通信开销爆炸。

**详细解释**：

**CP 切分的数学**：

```
1M 上下文，CP=N:
  每 rank 处理: 1M / N token
  每 rank KV cache: (1M / N) × 576 × 2 bytes = (1.15 GB) / N
  All-gather 通信: N × (1M / N) × 576 × 2 = 1.15 GB（固定，不依赖 N）
```

每 rank 处理的 token 数 = `1M / N`。

**CP=N 的选择 trade-off**：

| CP | 每 rank token | 每 rank KV | 计算量/rank | 通信步数 |
|---|---|---|---|---|
| 64 | 16384 | 18 MB | 大（attention O(N²)) | 少 |
| 256 | 4096 | 4.5 MB | 中 | 中 |
| **512** | **2048** | **2.25 MB** | **小** | **多** |
| 2048 | 512 | 0.56 MB | 极小 | 极多 |

**为什么 CP ≥ 512？**

1. **计算可行性**：CP=64 时，每 rank 16384 token 的 attention 计算量是 O(16384²) = 2.7×10⁸——单步训练时间不可接受
2. **显存约束**：CP=64 时，每 rank 18 MB KV × batch × activation 会超出单卡显存
3. **LSA 配合**：LSA 的 `index_local_tokens=1024` 要求每 rank 至少有 1024+ token 的本地窗口——CP=512 时每 rank 2048 token，刚好够

**All-gather 通信的成本**：

```
CP=512: All-gather 1.15 GB，分 512 个 rank 并行传输
  实际传输时间 = 1.15 GB / (aggregated_bandwidth)
  
CP=2048: All-gather 1.15 GB，分 2048 个 rank
  实际传输时间相近（带宽更高）但调度开销 4×
```

**关键观察**：CP 增加不增加总通信量（all-gather 是固定），但增加调度复杂度。512 是「计算可行 + 调度可控」的平衡点。

**硬件约束**：

LongCat superpod 是 48 机/仓，每机 8 卡 ASIC，单仓 384 卡——CP=512 需要跨仓（512 / 384 = 1.33 仓）。跨仓通信带宽低于仓内，所以 CP=512 实际是「1 个仓全用 + 0.33 个仓辅助」。

**Trade-off**：
- 收益：1M 上下文训练可行
- 代价：512 路 all-gather 通信复杂（跨仓）

**面试要点**：长上下文训练的核心约束是「attention O(N²) 计算量」和「KV cache 显存」。CP 是把序列切到多卡来分担——CP 数必须让单卡计算量可控。

**延伸阅读**：主报告 CH 7.5 / CH 7.2

---

## CH 8 推理体系

### Q8.1 为什么 LongCat-2.0 推理用 PD 分离（Prefill-Decode Separation）？

**简短回答**：Prefill 和 decode 的计算特性完全不同——prefill 是 compute-bound（大 batch matmul），decode 是 memory-bound（小 batch、KV cache 读取）；分离部署让两类 workload 在各自优化的硬件 / 配置上跑，互不干扰。

**详细解释**：

**Prefill vs Decode 的差异**：

| 维度 | Prefill | Decode |
|---|---|---|
| 输入 | 长序列（如 1M token） | 单 token（每步生成 1 个） |
| 计算量 | O(N²) attention（重） | O(N) attention（轻） |
| 瓶颈 | Compute（matmul） | Memory（KV cache 读取） |
| Batch size | 小（1-4 个长请求） | 大（100+ 并发请求） |
| EP 域 | 小（不需要切分 expert 太多） | 大（EP128，每卡少 expert） |
| 显存压力 | Activation（attention 中间态） | KV cache（累计历史） |

**Prefill 节点池优化**：
- CPP（Chunked Pipeline Parallel）：把长 prefill 切 chunk，多节点 pipeline
- SP（Sequence Parallel）：序列维度切分
- 缩小 EP 域：prefill 阶段 batch 小，EP 大域反而增加通信

**Decode 节点池优化**：
- KVP（KV-cache Parallelism）：91.66 GB KV cache 分片到多卡
- EP128：128 路 expert parallel，每卡仅 6 个 expert（降低单卡 I/O）
- EPLB：异步负载均衡，动态迁移 expert
- ScMoE：dense 与 MoE 完全并行
- Super Kernel + Weight Prefetch：ASIC 特定优化

**分离的好处**：

1. **硬件优化独立**：prefill 节点用「compute-heavy」配置（高 FLOPS ASIC），decode 节点用「memory-heavy」配置（高 HBM 带宽）
2. **资源利用率最大化**：prefill 完成后立即释放节点给下一个请求，decode 节点持续 batching
3. **Scaling 独立**：prefill 和 decode 可以独立扩展（如多 prefill 少 decode 或反之）

**PD 之间的通信**：

- **200 Gbps 网络适配器**：每节点
- **Layer-wise KV-cache 传输**：prefill 完成一层 KV 就传到 decode 节点，不等全部 prefill 完成——overlap prefill 计算与 KV 传输

**Trade-off**：
- 收益：prefill / decode 各自最优，资源利用率高
- 代价：跨节点 KV 传输延迟（200 Gbps × 节点数）+ 架构复杂

**面试要点**：PD 分离是当前大模型 serving 的主流（DeepSeek、Anthropic 等都用）。核心是「workload 异构→硬件异构」。

**延伸阅读**：主报告 CH 8.1 / 博客「Inference」节

### Q8.2 EP128（128 路 expert parallel）相比 EP64 有什么差异？

**简短回答**：EP128 每卡持 768/128=6 个 expert（vs EP64 的 12 个），单卡 expert I/O 减半；但 all-to-all 通信域翻倍，调度复杂度上升——EP128 适合大 batch decode，EP64 适合小 batch。

**详细解释**：

**EP 数的选择 trade-off**：

| EP | 单卡 expert 数 | 单卡 expert I/O | all-to-all 通信域 | 调度复杂度 |
|---|---|---|---|---|
| EP32 | 24 | 2.4 MB/token | 小 | 低 |
| EP64 | 12 | 1.2 MB/token | 中 | 中 |
| **EP128** | **6** | **600 KB/token** | **大** | **高** |
| EP256 | 3 | 300 KB/token | 极大 | 极高 |

**EP128 的具体收益**：

```
EP64:
  每卡持 12 个 expert × 50.33M 参数 × 2 bytes = 1.2 GB（expert 权重）
  每 token top-12 expert I/O = 12 × 100 KB = 1.2 MB

EP128:
  每卡持 6 个 expert × 50.33M × 2 bytes = 600 MB（expert 权重）
  每 token top-12 expert I/O = 12 × 100 KB = 1.2 MB（I/O 不变！）
```

**关键观察**：EP 数改变单卡 expert 数（权重显存），但不改变单 token 的 expert I/O——因为每 token 仍要 top-12 个 expert，无论这些 expert 分布在多少卡上。

**那 EP128 的真正收益是什么？**

1. **权重显存降低**：单卡 600 MB vs EP64 的 1.2 GB——更多空间给 KV cache 和 activation
2. **大 batch 吞吐**：batch=128 时，EP128 每 expert 处理 128 个 token（ vs EP64 的 256）——单 expert 计算量减半，但 expert 数翻倍，总计算量不变，但单 expert kernel 更高效
3. **与 EPLB 配合**：更多 EP rank 让 EPLB 迁移 expert 的粒度更细

**all-to-all 通信成本**：

```
EP64 all-to-all:
  每 token 的 dispatch: 12 个 expert query × hidden × 2 bytes = 192 KB
  每 token 的 combine: 12 个 expert output × hidden × 2 bytes = 192 KB
  总通信: 384 KB/token

EP128 all-to-all:
  通信量相同（384 KB/token）——top-12 expert 数没变
  但通信域从 64 升到 128 → 调度更复杂
```

**实际部署**：

```
EP128 部署:
  单机 8 卡 ASIC × 16 机 = 128 卡
  每卡: 6 个 expert × 100 MB = 600 MB expert 权重
  每卡 dense + MLA: ~68 GB（复制，不切分）
  每卡 KV cache (KVP 分片): 91.66 GB / 128 = 0.72 GB
  
  单卡总显存: ~69 GB（在 80GB ASIC 上可容纳）
```

**Trade-off**：
- 收益：单卡 expert 显存降低，留更多空间给 KV cache 和 batching
- 代价：all-to-all 域更大，调度复杂

**易混淆**：EP128 与 KVP 是不同并行——EP 切 expert，KVP 切 KV cache。decode 时两者叠加。

**延伸阅读**：主报告 CH 8.3 / CH 3.3

### Q8.3 EPLB（Expert-Parallel Load Balancing）异步运行——为什么需要异步？

**简短回答**：EPLB 根据 expert 路由热度动态迁移 expert（热门 expert 复制到多卡），如果同步会阻塞推理；异步运行让 expert 迁移在后台进行，不干扰 attention/MoE 主路径。

**详细解释**：

**EPLB 解决的问题**：

router 学习的 expert 分布可能不均匀——某些 expert（如「动词处理 expert」）被过度路由，热门卡负载过高；其他 expert 冷门，卡闲置。

**EPLB 的方案**：

```
后台监控:
  - 统计每 expert 的路由频率（滑动窗口）
  - 识别热门 expert（频率 > 阈值）
  - 识别冷门卡（负载 < 阈值）

异步迁移:
  - 把热门 expert 复制到冷门卡
  - 更新 router 的 expert-to-rank 映射
  - 推理时 router 知道哪些 expert 在哪些卡
```

**为什么必须异步？**

**同步迁移的问题**：

```
Time step T:
  检测到 expert X 是热门
  → 同步迁移 expert X 的权重（50 MB × 跨节点网络）
  → 推理暂停 50 ms
  
  如果每秒迁移 10 次 → 推理暂停 500 ms / 秒 → 50% 性能损失
```

**异步迁移**：

```
Time step T:
  检测到 expert X 是热门
  → 排队迁移任务（不阻塞）
  → 推理继续用旧映射
  
Time step T+10:
  迁移完成
  → 原子切换映射（< 1 ms）
  → 推理用新映射
```

异步让迁移的「长延迟」隐藏在正常推理之后，只在最后原子切换（< 1 ms）。

**EPLB 的实现细节**：

- **统计窗口**：最近 N 个 token 的路由统计（如 N=10000）
- **迁移阈值**：热门度 = 频率 / 平均，超过 1.5× 触发迁移
- **复制策略**：热门 expert 复制到 2-3 个 rank（不是迁移，是冗余）
- **router 更新**：通过共享 memory 或 broadcast 通知所有 rank

**与 ScMoE 配合**：

ScMoE 要求 MoE 在固定时间窗口内完成（attn[1] + mlp[1] 期间）。如果 EPLB 同步迁移导致 MoE 延迟，通算掩盖失败。异步确保 MoE 的延迟稳定。

**Trade-off**：
- 收益：负载均衡，避免热门 expert 卡瓶颈
- 代价：expert 冗余（热门 expert 多副本，占额外显存）+ 实现复杂

**易混淆**：EPLB 与训练时的 aux loss 不同。aux loss 是训练时调整 router 让负载均匀（前端预防），EPLB 是推理时动态迁移 expert（后端修正）。LongCat 删除了 aux loss，依赖 EPLB 推理时修正。

**延伸阅读**：主报告 CH 8.3 / CH 5.4

### Q8.4 MTP（Multi-Token Prediction）3 层 + CLI 共享——为什么是 3 层？

**简短回答**：3 层 MTP 让模型每步预测 3 个 token（speculative decoding 加速 3×）；CLI 共享让 3 个 draft 步骤共用 1 次 indexer pass，indexer 成本降至 1/3；3 是「spec 加速收益」与「draft 模型开销」的平衡。

**详细解释**：

**MTP 工作机制**：

```
标准 autoregressive:
  Step 1: P(t1 | context)
  Step 2: P(t2 | context, t1)
  Step 3: P(t3 | context, t1, t2)
  ...
  
每步 1 次 forward，1 个 token 输出

MTP（3 层）:
  Step 1: 主模型 P(t1) + draft 1 P(t2 | t1) + draft 2 P(t3 | t1,t2) + draft 3 P(t4 | t1,t2,t3)
  → 一次 forward 输出 4 个 token 候选
  
  Verify: 主模型验证 draft 输出是否正确
  → 接受 N 个正确 token，拒绝后重新 draft
  
每步 1 次 forward，平均输出 2-3 个 token（考虑拒绝率）
```

**MTP 层数的 trade-off**：

| MTP 层数 | 理论加速 | 实际加速（拒绝率 ~20%） | Draft 模型参数 | Indexer 成本（无 CLI） |
|---|---|---|---|---|
| 1 | 2× | 1.8× | 1× | 1× |
| **3** | **4×** | **2.5-3×** | **3×**（共享后 1×） | **3×**（CLI 后 1×） |
| 5 | 6× | 3-3.5× | 5× | 5× |

**关键配置**：

```json
"mtp_num_layers": 3,
"mtp_replicate_modules": true,
```

`mtp_replicate_modules=true` 让 3 层 MTP 共享同一份模块参数——draft 模型参数只算 1×，不是 3×。

**CLI 共享让 indexer 成本降至 1/3**：

```
无 CLI（每 draft step 独立 indexer）:
  draft 1: indexer pass
  draft 2: indexer pass（重新算）
  draft 3: indexer pass（重新算）
  → 3 次 indexer pass

CLI 共享:
  draft 1: indexer pass
  draft 2: 复用 draft 1 的 index
  draft 3: 复用 draft 1 的 index
  → 1 次 indexer pass
```

CLI 让 MTP 的 indexer 开销从 3× 降到 1×——这是 LongCat 把 MTP 层数推到 3 的关键。

**与对手对比**：

| 模型 | MTP 层数 | 备注 |
|---|---|---|
| DeepSeek V3 | 1 | 单层 MTP |
| Hy3 | 1 | 单层 MTP |
| **LongCat-2.0** | **3** | **多层 MTP + CLI 共享** |

LongCat 是少数用 3 层 MTP 的模型——这依赖 CLI 的 indexer 共享创新，否则成本不可接受。

**开源状态**：

源码 `model.py:L14,L74`：

```python
_keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]  # MTP head dropped at load
```

checkpoint 中有 MTP 权重，但 HF 端口未实现 MTP 头——加载时被静默丢弃。开源版本无法用 MTP 推理。

**Trade-off**：
- 收益：推理加速 2.5-3×（3 层 MTP + CLI 共享）
- 代价：训练复杂（draft 模型需要训练）+ 开源未实现

**易混淆**：MTP 与 LSA 是不同稀疏机制。MTP 是「时间维度稀疏」（多 token 一次预测），LSA 是「空间维度稀疏」（attention 选 top-k KV）。

**延伸阅读**：主报告 CH 8.5 / config.json `mtp_*`

### Q8.5 Layer-wise KV-cache 传输（PD 之间）——为什么逐层传而非等 prefill 完成？

**简短回答**：prefill 完成一层 KV 就立即传到 decode 节点，与下一层 prefill 计算并行——overlap prefill compute 与 KV 传输，避免 prefill 完成后大规模 KV 一次性传输的延迟尖峰。

**详细解释**：

**问题背景**：

PD 分离架构中，prefill 节点完成计算后，需要把 KV cache 传到 decode 节点。1M 上下文的 KV cache 是 91.66 GB——一次性传输会阻塞。

**一次性传输的问题**：

```
Time:
  Prefill 计算: [====== 10s ======]
  KV 传输:                        [==== 5s ====]  ← 阻塞，decode 等 5s
  Decode 开始:                                       [▶]
  
总延迟: 15s（prefill + 传输）+ decode 开始
```

**Layer-wise 传输**：

```
Time:
  Prefill L0: [0.13s]
  KV L0 传输: [0.06s] ←────── 与 Prefill L1 并行
  Prefill L1:        [0.13s]
  KV L1 传输:        [0.06s] ←── 与 Prefill L2 并行
  ...
  Prefill L75:                   [0.13s]
  KV L75 传输:                   [0.06s]
  
  最后一层传输完成 → decode 立即开始
  
总延迟: 76 × 0.13s = 10s（prefill），KV 传输全部隐藏在 prefill 计算中
```

**关键观察**：每层 KV = 91.66 GB / 76 = 1.2 GB。传输 1.2 GB 在 200 Gbps 链路上约 0.06 秒——与单层 prefill 计算时间（~0.13s）相当，完全可以 overlap。

**实现约束**：

1. **KV 必须独立于 layer 计算**：layer N 的 KV 在 layer N 完成时就 ready，不依赖 layer N+1
2. **网络双工**：prefill 节点同时跑计算 + 传输——需要双工网络（200 Gbps 上下行独立）
3. **Decode 节点缓冲**：decode 节点要缓冲接收到的 KV，直到所有层 ready

**200 Gbps 链路的角色**：

```
每层 KV 1.2 GB = 9.6 Gb
200 Gbps 链路传输时间 = 9.6 / 200 = 0.048s

vs 单层 prefill 计算时间（约 0.13s）

overlap 可行：传输 < 计算
```

如果链路只有 100 Gbps，传输 0.096s——仍可 overlap 但 margin 紧张。

**Trade-off**：
- 收益：KV 传输完全隐藏，PD 之间无阻塞
- 代价：网络双工要求 + decode 节点缓冲复杂

**面试要点**：PD 分离的性能关键不是「总通信量」而是「能否 overlap」。Layer-wise 传输是标准做法。

**延伸阅读**：主报告 CH 8.4 / 博客「Inference」节

### Q8.6 Super Kernel 与 Weight Prefetch 是什么？

**简短回答**：Super Kernel 是 ASIC 特定融合 kernel（把多个算子合并到一个 kernel，减少 launch 开销）；Weight Prefetch 是提前加载下一层权重到片上 SRAM，掩盖 HBM 延迟——两者都是 ASIC 推理优化的「last mile」。

**详细解释**：

**Super Kernel**：

传统 GPU 推理每个算子（matmul、norm、activation）是独立 kernel，每个 kernel launch 有 ~5-10 μs 开销。对于小 batch decode，算子碎屑化导致严重 overhead。

Super Kernel 把「一个 transformer 层的所有算子」融合到一个 kernel：

```
传统:
  kernel 1: q_proj matmul
  kernel 2: rope
  kernel 3: k_proj matmul
  kernel 4: attention
  kernel 5: o_proj matmul
  kernel 6: mlp gate
  kernel 7: mlp up
  kernel 8: mlp down
  → 8 次 launch × 10 μs = 80 μs overhead

Super Kernel:
  1 个融合 kernel 完成全部
  → 1 次 launch × 10 μs = 10 μs overhead（节省 87.5%）
```

ASIC 上 Super Kernel 可以硬连线——算子图直接编译成电路，零 launch 开销。

**Weight Prefetch**：

HBM 带宽虽高（~3 TB/s），但延迟 ~100 ns。如果每层计算时才从 HBM 加载权重，会等待延迟。

Weight Prefetch 用 ASIC 片上 SRAM（~100 MB，延迟 ~10 ns）做缓存：

```
Layer N 计算:
  从 HBM 加载 Layer N+1 权重到 SRAM（异步）
  Layer N 计算用 SRAM 中已加载的 Layer N 权重
  
Layer N+1 计算:
  直接用 SRAM 中的 Layer N+1 权重（无 HBM 等待）
  同时从 HBM 加载 Layer N+2 权重
```

**关键约束**：SRAM 容量有限（~100 MB），无法缓存整个模型。每层权重必须恰好 fit SRAM：
- MLA + Dense MLP 单层约 414M 参数 × 2 bytes = 828 MB——超过 SRAM
- 需要 TP 切分让单卡权重 < 100 MB

**与 ScMoE 配合**：

ScMoE 要求 dense + MoE 完全并行——Super Kernel 把 dense 和 MoE 算子分别融合，让 ASIC 的 dense cluster 和 sparse cluster 同时跑各自的 super kernel。

**Trade-off**：
- 收益：launch overhead 减少 + HBM 延迟掩盖
- 代价：kernel 编译复杂（需 ASIC 工具链）+ 灵活性差（改架构需重编译）

**易混淆**：Super Kernel 与算子融合（operator fusion）类似，但 Super Kernel 更激进——融合整个 transformer 层，而非几个相邻算子。

**延伸阅读**：主报告 CH 8.3 / 博客「Inference」节

---

## CH 9 总结

### Q9.1 LongCat-2.0 的「三正交稀疏设计」协同效应如何量化？

**简短回答**：Attention 稀疏（LSA）让 1M 上下文 attention FLOPs 从 1 ExaFLOPs 降到 334 TFLOPs（3000× 节省）；Expert 稀疏（MoE + ScMoE）让 1.6T 模型激活 48B（3% 激活率）+ 通算完全掩盖；Embedding 稀疏（N-gram）让 8.4% 参数的 I/O 效率比 expert 高 15×——三者协同让 1.6T/1M 可行。

**详细解释**：

**单一稀疏机制的局限**：

如果只有 MoE 稀疏（无 LSA、无 N-gram）：
- 1M 上下文 attention O(N²) 不可行 → 必须限长到 ~32K
- 单 token 激活 48B 但 I/O 高（12 expert × 100KB = 1.2 MB/token）→ 大 batch decode 瓶颈

如果只有 LSA（无 MoE、无 N-gram）：
- 长上下文可行，但模型规模受限（dense 模型 1.6T 不可行）
- 参数效率低（无稀疏激活）

如果只有 N-gram（无 MoE、无 LSA）：
- 局部共现学习好，但无长距离 + 无稀疏激活

**三者协同的量化**：

| 机制 | 单独收益 | 协同贡献 |
|---|---|---|
| LSA | attention FLOPs -3000× | 让 1M 上下文可行 |
| MoE + ScMoE | 激活率 3% + 通算掩盖 | 让 1.6T 模型训练/推理可行 |
| N-gram | I/O 效率 +15× | 让大 batch decode I/O 可控 |

**乘法效应**：

```
模型规模可行性:
  Dense 1.6T: 不可行（训练 FLOPs 36 ExaFLOPs/35T tokens，不可承受）
  MoE 1.6T / 48B 激活: 可行（激活 FLOPs 1 ExaFLOPs）
  + LSA（1M 上下文）: 可行（attention 不爆炸）
  + N-gram: 大 batch decode I/O 可控
```

**正交性的数学表达**：

```
单 token 信息流:
  hidden = token_emb + ngram_emb           ← Embedding 稀疏（局部共现）
  h = MLA(hidden) + LSA_attention(hidden)  ← Attention 稀疏（长距离）
  output = dense_mlp(h) + MoE(h)           ← Expert 稀疏（语义）
```

三个稀疏机制作用在不同维度（embedding、attention、FFN），互不干扰。

**训练成本对比**：

```
Dense 1.6T / 35T tokens / 1M context:
  FLOPs ≈ 6 × 1.6T × 35T × (1M/4K) = 2.5 × 10^31（不可承受）
  
LongCat 1.6T / 48B 激活 / 1M context + LSA + N-gram:
  FLOPs ≈ 6 × 48B × 35T × (1M/4K × 0.0003)  # sparse attention 节省
       ≈ 3.4 × 10^26（可行，336 ExaFLOPs）
```

节省因子约 10⁵——这就是三正交稀疏的协同效应。

**面试要点**：LongCat 的核心创新不在单一组件，而在「三个稀疏轴的乘法效应」。任一组件单独看都不算革命性，但组合起来让 1.6T/1M 可行。

**延伸阅读**：主报告 CH 9.1 / CH 9.2

### Q9.2 LongCat-2.0 有哪些已知局限？

**简短回答**：HI 未开源 / AI ASIC 型号未公开 / 无独立论文 / 完整代码仅在 SGLang PR 未合并 / MOPD 训练数据未公开 / MFU 绝对值未公开 / `routed_scaling_factor=9` 设计意图待确认 / `e_score_correction_bias` 动态更新机制待确认 / benchmark harness 未公开。

**详细解释**：

**开源完整度局限**：

1. **HI 未开源**：Hierarchical Indexing 在开源版本中不支持（README 明确「not supported for simplicity」），仅在训练时 + 超长上下文任务启用
2. **完整代码仅在 SGLang PR 未合并**：HF 仓库无 modeling 代码；Transformers 内置为前代 LongCat-Flash；完整 LongCat-2.0 推理实现（含 LSA + N-gram + Dual-sublayer）位于 SGLang PR #30042（`HarryWu99/sglang @ c6c36d94`），**截至 2026-07-06 PR 仍 open，未合并到 sglang main 分支**
3. **MTP 头被丢弃**：checkpoint 有权重但加载时被 `_keys_to_ignore_on_load_unexpected` 静默丢弃

**信息透明度局限**：

4. **AI ASIC 型号未公开**：博客仅说「AI ASIC」，社区推测为昇腾或自研，无官方确认
5. **无独立论文**：所有架构信息来自博客 + config + 源码，无同行评审
6. **训练 MFU / 训练天数绝对值未公开**：仅披露相对 LongCat-Flash 提升 35%
7. **MOPD 后训练细节未公开**：博客提及 Agent/Reasoning/Interaction 三类专家 + MOPD 架构，但未公开训练数据和配比
8. **benchmark 评测 harness**：除标注 `*` 外为 in-house（未公开 harness 代码），单边 benchmark 不应直接当优势宣称

**实现细节待确认**：

9. **`routed_scaling_factor=9` 设计意图**：源码确认值，设计原理未在博客披露
10. **`e_score_correction_bias` 动态更新机制**：源码中为 zeros 初始化 buffer，训练时是否动态更新未公开

**社区复现的挑战**：

- 无法用 GPU 复现 ScMoE 的「explicit per-core control」→ 训练栈不可复现
- 无法复现 EMBP + 6D 并行的具体配置
- 无法验证 1M 上下文的 benchmark（harness 未开源）

**与其他大模型对比**：

| 维度 | DeepSeek V3 | Hy3 | LongCat-2.0 |
|---|---|---|---|
| 论文 | 有 | 有 | 无 |
| 开源代码 | 完整 | 完整 | 部分（无 LSA / N-gram） |
| 硬件信息 | GPU 公开 | GPU 公开 | ASIC 型号未公开 |
| 训练细节 | 大部分公开 | 部分公开 | 仅相对值 |

**面试要点**：评估 LongCat-2.0 时要区分「博客宣称」与「可验证」。HI、MOPD、MFU 等关键细节未开源，benchmark 应打折看。

**延伸阅读**：主报告 CH 9.3 / CH 1.3

### Q9.3 LongCat-2.0 相比 DeepSeek V4-Flash 和 Hy3-295B 的核心差异？

**简短回答**：规模更大（1.6T vs 284B/295B）、稀疏轴更多（MoE + LSA + N-gram 三正交 vs 单一 MoE）、上下文更长（1M vs 128K/256K），但开源完整度更低（无 LSA 完整实现、无独立论文、AI ASIC 型号未公开）。

**详细解释**：

**核心数字对比**：

| 维度 | DeepSeek V4-Flash | Hy3-295B | **LongCat-2.0** |
|---|---|---|---|
| 总参 | 284B | 295B | **1.6T** |
| 激活 | 13B | 21B | **48B** |
| 激活率 | 4.6% | 7.1% | **3.0%** |
| 原生上下文 | 128K | 256K | **1M** |
| MoE expert 数 | ~256 | ~512 | **768** |
| Top-k | 8 | 12 | **12** |
| MTP 层数 | 1 | 1 | **3** |
| 稀疏轴 | MoE | MoE + Layer-wise KV | **MoE + LSA + N-gram** |

**架构哲学差异**：

**DeepSeek V4-Flash**：保守优化
- 单一 MoE 稀疏（无第二轴）
- 标准注意力（无 sparse attention）
- 128K 上下文（无 YaRN 外推到 1M）
- GPU 训练（标准 H100/H200 集群）

**Hy3-295B**：中等激进
- MoE + Layer-wise KV 共享（第二轴与 attention 耦合）
- 256K 上下文
- GPU 训练

**LongCat-2.0**：激进三正交
- MoE + LSA + N-gram（三轴完全正交）
- 1M 上下文（YaRN factor=120 外推）
- AI ASIC 训练（自建 superpod）
- 6D 并行（含独有 EMBP）

**性能宣称对比**：

LongCat-2.0 博客宣称多项 benchmark 超越 V4-Flash 和 Hy3-295B，但：
- benchmark harness 多为 in-house（未开源）
- 测试条件（batch size、序列长度、量化）未完全披露
- 单边 benchmark 不应直接当优势宣称

**开源完整度对比**：

| 维度 | V4-Flash | Hy3 | LongCat-2.0 |
|---|---|---|---|
| 论文 | 有 | 有 | **无** |
| 完整代码 | 有 | 有 | **部分**（无 LSA / N-gram / MTP） |
| 训练复现 | 难（成本高）但理论可行 | 难但理论可行 | **不可行**（ASIC 锁定） |
| 推理复现 | 可行（vLLM/SGLang 支持） | 可行 | **部分**（fallback 路径） |

**Trade-off 总结**：

- LongCat-2.0 选择「规模 + 长上下文 + 三正交稀疏」路线，代价是「开源完整度 + 硬件锁定」
- V4-Flash / Hy3 选择「保守规模 + 完整开源 + GPU 友好」路线，代价是「上下文长度 + 稀疏创新」

**面试要点**：对比大模型时不能只看数字——开源完整度、硬件依赖、可复现性都是关键。LongCat-2.0 数字领先但工程门槛高。

**延伸阅读**：主报告 CH 9.4 / `comparison.md`（详细对比）

### Q9.4 如果只能记住 LongCat-2.0 的 5 个核心设计决策，是哪 5 个？

**简短回答**：(1) Dual-Sublayer + ScMoE 让 MoE 通算完全掩盖；(2) LSA 把 1M 上下文 attention 从 1 ExaFLOPs 降到 334 TFLOPs；(3) N-gram Embedding 在 MoE 稀疏度 97% 边界外开辟正交稀疏轴；(4) Identity zero experts 让 padding token 零算力路由；(5) EMBP 为 N-gram Embedding 独创第 6 维并行。

**详细解释**：

这 5 个设计决策是 LongCat-2.0 的「灵魂」——理解它们就理解了整个架构。

#### 1. Dual-Sublayer + ScMoE（CH 5.1, 5.2）

把单层 Transformer 拆成 2 个子层 + 1 个跨子层 Shortcut MoE，让 MoE 的 dispatch/combine/all-to-all 通信被 attn[1] + mlp[1] 计算完全掩盖。

**本质**：架构级通算掩盖设计，需要 ASIC 的 per-core control 配合。

#### 2. LSA 三件套（CH 4.2）

SI（流式感知 HBM 合并）+ CLI（跨层共享 indexer）+ HI（粗筛+精选两阶段）——把 1M 上下文 attention FLOPs 从 1 ExaFLOPs 降到 334 TFLOPs（3000× 节省）。

**本质**：把 O(N²) attention 降到 O(N × k)，让 1M 上下文可行。

#### 3. N-gram Embedding 正交稀疏（CH 6）

在 MoE 稀疏度 97% 边界外，开辟「基于 N-gram 模式的确定性局部共现」稀疏轴——135B 参数，I/O 效率比 expert 高 15×。

**本质**：突破 MoE 稀疏度上限，用第二条稀疏通路扩展参数。

#### 4. Identity Zero Experts（CH 5.3）

128 个 `nn.Identity()` 零参数 expert——让 padding token 路由到 identity 后保持原值，不消耗 FFN 算力且不污染残差。

**本质**：用零参数 padding expert 替代 V3 的 dense shared expert，工程优雅。

#### 5. EMBP 6D 并行（CH 7.1）

为 N-gram Embedding 独创第 6 维并行（Embedding Parallel）——按 `oe_split_num=4` 把 135B embedding 表切到 4 个 DP rank，用稀疏 all-to-all lookup 通信。

**本质**：源于 N-gram 的「lookup 而非 matmul」特性，传统 DP/EP/TP 都不适配。

**为什么这 5 个？**

每个都是 LongCat 独有或激进选择——
- 其他模型用标准单层 MoE（V3、Hy3）→ LongCat 用 Dual-Sublayer + ScMoE
- 其他模型用 dense attention 或简单 sparse → LongCat 用 LSA 三件套
- 其他模型只用 MoE 稀疏 → LongCat 加 N-gram 正交轴
- 其他模型用 shared expert → LongCat 用 identity zero experts
- 其他模型用 5D 并行 → LongCat 加 EMBP 第 6 维

**协同效应**（Q9.1 已详述）：三正交稀疏让 1.6T/1M 可行，这是单一组件无法实现的。

**面试要点**：被问 LongCat-2.0 时，先说「三正交稀疏 + Dual-Sublayer + EMBP」三个关键词，再展开细节。这是与 V3/Hy3 的本质区别。

**延伸阅读**：主报告 CH 0 / CH 9.1 / CH 9.2
