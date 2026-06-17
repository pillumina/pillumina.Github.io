+++
date = '2026-06-17'
draft = false
title = 'GLM-5.2 架构 QA'
categories = ['qa']
tags = ['moe', 'attention', 'model-architecture', 'qa', 'glm', 'dsa', 'indexshare', 'mtp']
series = ['qa']
summary = '基于 GLM-5.2 主报告的配套 QA（27 问）。覆盖 GLM-5.1 → 5.2 演进、IndexShare 核心创新、MTP 四重改进、1M 上下文工程、训练与性能。'
+++

# GLM-5.2 架构 QA

> 27 问，覆盖 CH1 演进（GLM-5.1 → 5.2） → CH2 IndexShare 核心创新 → CH3 MTP 四重改进 → CH4 1M 上下文工程 → CH5-6 训练与性能 → CH7-8 对比与总结

---

## 一、CH1 演进：GLM-5.1 → GLM-5.2（4 Q）

### Q1.1 GLM-5.2 和 GLM-5.1 的 config.json 有哪些差异？

**简短回答**：在 MLA / MoE / DSA 基础配置完全相同的前提下，GLM-5.2 新增 4 个 IndexShare 相关字段（`indexer_types`、`index_share_for_mtp_iteration`、`index_skip_topk_offset`、`index_topk_freq`），将上下文从 202,752 扩展到 1,048,576，并将 `rope_theta` 从 1,000,000 提升到 8,000,000。另有 `head_dim` 从 64 变为 192 的口径统一（非实质变化）。

**详细解释**：完整 diff 如下：

| 字段 | GLM-5.1 | GLM-5.2 | 性质 |
|---|---|---|---|
| `num_hidden_layers` | 78 | 78 | 不变 |
| `hidden_size` | 6144 | 6144 | 不变 |
| `n_routed_experts` / `n_shared_experts` | 256 / 1 | 256 / 1 | 不变 |
| `num_experts_per_tok` | 8 | 8 | 不变 |
| `moe_intermediate_size` | 2048 | 2048 | 不变 |
| `kv_lora_rank` / `q_lora_rank` | 512 / 2048 | 512 / 2048 | 不变 |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 2048 | 32 / 128 / 2048 | 不变 |
| `head_dim` | 64 | **192** | 口径统一（QK 维度未变） |
| `max_position_embeddings` | 202,752 | **1,048,576** | 5× 扩展 |
| `rope_parameters.rope_theta` | 1,000,000 | **8,000,000** | 8× 提升 |
| `indexer_types` | — | **78 元数组** | 新增（IndexShare 核心） |
| `index_share_for_mtp_iteration` | — | **`true`** | 新增 |
| `index_skip_topk_offset` | — | **`3`** | 新增 |
| `index_topk_freq` | — | **`4`** | 新增 |
| `transformers_version` | `5.12.0` | `5.12.0` | 不变 |

`head_dim` 字段从 64 变为 192 不是模型架构变化——Transformers 5.12 对 `GlmMoeDsaModel` 的字段口径统一所致。实际 QK 维度 `qk_nope_head_dim=192 + qk_rope_head_dim=64=256` 在两版完全一致。

**面试要点**：被问到 GLM-5.2 的「核心架构变化」时，先答「基础架构（MLA/MoE/DSA）零变化」，再答 IndexShare + 1M 上下文 + MTP 改进 + RL 升级。不要混淆 head_dim 口径变化与实质架构变化。

**易混淆**：`first_k_dense_replace=3` 不是 IndexShare 引入的，GLM-5.1 已有该字段（前 3 层 Dense，第 4 层起 MoE），不要误认为是 5.2 的新设计。

**延伸阅读**：主报告 CH 1.1 / `_work/config.json` L1-223

---

### Q1.2 `indexer_types` 数组的具体模式是什么？

**简短回答**：78 元素数组，模式为「3 full + 3 shared + 18 组（1 full + 3 shared）」——前 6 层是冷启动特例，主体 72 层严格按 4 层周期循环（1 full + 3 shared），最终合计 **21 full + 57 shared**。

**详细解释**：完整展开 `indexer_types` 字段（`_work/config.json` L26-105）：

```
Layer  0-2:  full   full   full          ← 冷启动（3 个 Dense 层强制 full）
Layer  3-5:  shared shared shared        ← 复用 Layer 2 的索引
Layer  6  :  full                       ← 主体循环第 1 组
Layer  7-9:  shared shared shared
Layer 10  :  full                       ← 主体循环第 2 组
Layer 11-13: shared shared shared
...
Layer 74  :  full                       ← 主体循环第 18 组
Layer 75-77: shared shared shared
```

**总计数**：
- 冷启动块：3 full + 3 shared = 6 层
- 主体循环块：18 组 ×（1 full + 3 shared）= 72 层
- 合计：**21 full + 57 shared = 78 层**（与 `num_hidden_layers=78` 严格一致）

**频率验证**：`index_topk_freq=4` 严格对应「每 4 层 1 个 full」。冷启动块是特例（前 3 层 Dense 必须 full，因为还没有可复用的 full 索引；层 3-5 共享层 2 的索引）。

**为什么前 3 层全是 full**：Layer 0-2 是 Dense 层（`first_k_dense_replace=3`，`mlp_layer_types` L110-113 为 `dense, dense, dense`），浅层注意力模式尚未稳定，跨层复用风险高，强制独立执行 topk。

**面试要点**：可以用「3+3 启动块 + 18 个 4 层周期」一句话概括 78 层的 IndexShare 模式。

**延伸阅读**：主报告 CH 2.1 / `_work/config.json` L26-105

---

### Q1.3 `head_dim` 从 64 变成 192 是模型架构变化吗？

**简短回答**：**不是**。这只是 Transformers 5.12 对 `GlmMoeDsaModel` 的字段口径统一，实际 QK 维度（`qk_nope_head_dim=192 + qk_rope_head_dim=64 = 256`）和 V 维度（`v_head_dim=256`）在 GLM-5.1 与 GLM-5.2 中完全一致。

**详细解释**：早期版本的 `head_dim` 字段标注口径不一致，部分代码以 nope 维度为准（192），部分以 RoPE 子空间维度为准（64）。Transformers 5.12.0 统一以 nope 头维度（192）作为 `head_dim` 标注口径。

GLM-5.2 config 中的相关字段（`_work/config.json` L13-19）：
```json
"head_dim": 192,            // 新口径：nope head 维度
"qk_head_dim": 256,         // 完整 QK 维度（nope + rope）
"qk_nope_head_dim": 192,
"qk_rope_head_dim": 64,
"v_head_dim": 256,
"index_head_dim": 128,
```

五个维度字段严格一致，模型计算图无变化。

**面试要点**：被追问 5.2 的「维度变化」时，必须澄清这是字段口径而非真实变化，否则会被认为对源码不熟。

**易混淆**：`index_head_dim=128` 是 DSA Indexer 的头维度（与主 Attention 的 192/256 无关），不要混淆。

**延伸阅读**：主报告 CH 1.1 注释 / `_work/config.json` L13-19

---

### Q1.4 GLM-5.2 的 `rope_theta` 为什么从 1M 提升到 8M？

**简短回答**：`rope_theta` 提升 8× 是对应 5× 上下文扩展的 RoPE 基频重整。根据 NTK-aware 推断，外推 5× 上下文至少需要 θ 提升约 5-8 倍以保持长距离旋转角度的分辨率，GLM-5.2 选择 8× 偏保守一侧。

**详细解释**：RoPE 中位置 $m$ 对应的旋转角度为 $\theta_i = m \cdot \text{base}^{-2i/d}$，其中 `base` 即 `rope_theta`。当上下文扩展时，若 θ 不变，长位置上的角度会快速饱和（接近 $2\pi$），导致相邻 token 的旋转区分度下降。

| 模型 | `max_position_embeddings` | `rope_theta` | 上下文倍数 |
|---|---|---|---|
| GLM-4.5 | 131,072 | 500,000 | 1× |
| GLM-5 / 5.1 | 202,752 | 1,000,000 | ~1.5× |
| **GLM-5.2** | **1,048,576** | **8,000,000** | **5× / 8×** |

8× vs 5× 的「超额」提升是工程权衡：θ 越大长距离分辨率越好，但短距离的局部模式学习难度上升。GLM-5.2 选择 8× 偏保守，并配合长上下文持续训练（CH 5.1）让模型在训练阶段就适应新 θ。

**面试要点**：θ 与上下文长度的关系不是线性的——5× 上下文不等于 5× θ。常见经验是 θ 提升倍数略大于上下文提升倍数（如 5×→8×），原因是高频分量对 θ 敏感度更低。

**延伸阅读**：主报告 CH 1.4 / NTK-aware RoPE 文献

---

## 二、CH2 IndexShare 核心创新（6 Q）

### Q2.1 IndexShare 的机制是什么？每 4 层怎么共享？

**简短回答**：IndexShare 是 DSA Indexer 的「二次稀疏化」——在 78 层 Indexer 中，仅 21 层（`full`）执行完整的 top-2048 选择，其余 57 层（`shared`）跳过 Indexer 注意力计算，直接复用最近一个 `full` 层产生的 top-2048 索引，主 Attention 仍在复用的 2048 个 token 上计算。

**详细解释**：

**`full` 层的计算**（每层、每 token）：
1. 对当前 query 做 32 头注意力（`index_n_heads=32`，`index_head_dim=128`），与所有历史 token 的 indexer-K/V 计算注意力分数。
2. 取 top-2048（`index_topk=2048`）token 索引，作为该层 Attention 的稀疏参与集。
3. 主 Attention 仅在这 2048 个 token 上计算，复杂度 $O(2048 \cdot d)$。

**`shared` 层的计算**（每层、每 token）：
1. **跳过**整个 indexer 注意力计算（`skip_topk=True`）。
2. 直接复用最近一个 `full` 层产生的 top-2048 索引。
3. 主 Attention 在复用的 2048 个 token 上计算，复杂度与 full 层相同 $O(2048 \cdot d)$。

**关键观察**：节省的不是主 Attention 的 FLOPs（这部分两层一样），而是 **Indexer 自身的注意力计算**——32 头 × head_dim=128 × 序列长度 T 的 indexer 注意力，在 `shared` 层完全省略。

**4 层共享窗口的几何图示**：
```
Layer N (full)    ─┐ Indexer 跑 topk → 主 Attn 跑 top-2048
                  │ 写入 shared_topk_indices 缓存
Layer N+1 (shared)─┤ 跳过 Indexer → 读取缓存 → 主 Attn 跑 top-2048
Layer N+2 (shared)─┤ 跳过 Indexer → 读取缓存 → 主 Attn 跑 top-2048
Layer N+3 (shared)─┘ 跳过 Indexer → 读取缓存 → 主 Attn 跑 top-2048
Layer N+4 (full)  ── 重新跑 Indexer topk（窗口滑动）
```

**面试要点**：核心创新一句话——「DSA 主 Attention 已经稀疏化了，但 Indexer 自身是 O(T²)，IndexShare 把 Indexer 也稀疏化（21/78 层保留）」。

**延伸阅读**：主报告 CH 2.2 / `_work/config.json` L26-105

---

### Q2.2 `indexer_types` 中的 `full` 和 `shared` 在代码中怎么实现？（L406-407）

**简短回答**：在 `modeling_glm_moe_dsa.py` 的第 406-407 行，根据 `config.indexer_types[layer_idx]` 的值决定该层是否构造 Indexer——`"shared"` 时将 `self.skip_topk=True` 且 `self.indexer=None`，forward 中直接跳过 indexer 调用，从外层缓存（`shared_topk_indices`）读取 topk 索引。

**详细解释**：源码（`modeling_glm_moe_dsa.py:L406-407`）：

```python
# modeling_glm_moe_dsa.py:L406-407
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```

逻辑严格对应 config：
- `skip_topk=True` 时，该层的 `self.indexer` 被设为 `None`，forward 中直接跳过 indexer 调用。
- `skip_topk=False` 时，构造完整的 `GlmMoeDsaIndexer`（32 头 × 128 维 × top-2048）。
- topk 索引通过外层缓存（`shared_topk_indices`）在 `full` 层写入、`shared` 层读取。

**配套字段在 forward 路径的作用**：

| 字段 | 值 | 作用 |
|---|---|---|
| `index_skip_topk_offset` | 3 | 前 3 个 token 强制走 full（避免序列开头不稳定） |
| `index_topk_freq` | 4 | 与 indexer_types 的 4 层周期呼应（仅注释性配置） |
| `index_share_for_mtp_iteration` | true | MTP 推理路径也复用 IndexShare 缓存 |

**vLLM 端的对应实现**：vLLM 的 `llm_base_proposer.py` 中加入了对 `index_share_for_mtp_iteration` 配置项的读取，使得 MTP iteration 也能复用主模型的 IndexShare topk 缓存，避免 MTP 多次 draft 时重复跑 indexer。

**面试要点**：被追问 IndexShare 的源码实现时，L406-407 是必答的两行——一行决定 flag，一行决定 indexer 是否构造。

**易混淆**：KVShare（CH 3）目前**仅博客来源，源码未验证**，不要把 IndexShare 与 KVShare 混为一谈。

**延伸阅读**：主报告 CH 2.4 / `modeling_glm_moe_dsa.py` L406-407 / vLLM `llm_base_proposer.py`

---

### Q2.3 2.9× FLOPs 降低怎么算出来的？

**简短回答**：78 层中仅 21 层执行 Indexer 的 $O(T^2)$ 注意力，Indexer 部分降到 21/78 ≈ 26.9%（约 3.7× 减少）。在 1M 上下文下 Indexer 占整体 FLOPs 主导（~70%），故整体 per-token FLOPs 降到约 1/2.9 ≈ 34.5%，与博客声称的 2.9× 降低相符。

**详细解释**：考虑 1M 上下文（$T = 1{,}048{,}576$）下，DSA Indexer 的总 FLOPs：

**GLM-5.1（全部 full）**：
$$
\text{FLOPs}_{\text{indexer}}^{5.1} = 78 \cdot T \cdot (32 \cdot 128) \cdot T = 78 \cdot 32 \cdot 128 \cdot T^2
$$

**GLM-5.2（21 full + 57 shared）**：
$$
\text{FLOPs}_{\text{indexer}}^{5.2} = 21 \cdot T \cdot (32 \cdot 128) \cdot T = 21 \cdot 32 \cdot 128 \cdot T^2
$$

**节省比例**：
$$
\frac{\text{FLOPs}^{5.2}_{\text{indexer}}}{\text{FLOPs}^{5.1}_{\text{indexer}}} = \frac{21}{78} \approx 0.269
$$

即 Indexer 部分的 FLOPs 降低至 5.1 的 **26.9%**（约 3.7× 减少）。

**长短上下文对比**：

| 上下文长度 | Indexer FLOPs 占比（5.1） | IndexShare 节省（5.2） | per-token FLOPs 比 |
|---|---|---|---|
| 4K | < 5% | 几乎无影响 | ≈ 1.0× |
| 32K | ~15% | ~5% | ≈ 0.95× |
| 128K | ~40% | ~25% | ≈ 0.7× |
| **1M** | **~70%** | **~50%** | **≈ 0.34×（即 2.9× 降低）** |

**关键洞察**：2.9× 不是恒定值——Indexer 占比随上下文长度增长（O(T²) vs 主 Attention 的 O(T·k)），IndexShare 在短上下文下几乎无收益，专门为长上下文设计。

**面试要点**：被问「2.9× 怎么来的」时，回答「78 层降到 21 层是 3.7×，但 Indexer 只占总 FLOPs 的一部分，所以整体是 2.9×；这个倍数随上下文长度变化，1M 下才有这个收益」。

**延伸阅读**：主报告 CH 2.3

---

### Q2.4 为什么前 3 层（Dense 层）都是 `full`？

**简短回答**：两层原因——(1) 前 3 层是 Dense 层（`first_k_dense_replace=3`），是注意力模式尚未稳定的浅层，跨层复用 topk 风险高；(2) 冷启动时还没有任何 `full` 层产生的 topk 索引可复用，必须自己计算。同时 `index_skip_topk_offset=3` 保证前 3 个 token 也不使用 shared，避免序列开头的不稳定 topk。

**详细解释**：前 3 层强制 `full` 的三个设计动机：

1. **Dense 层属性**：`first_k_dense_replace=3` 表示 Layer 0-2 是 Dense FFN（无 MoE 路由），是模型对原始 token embedding 的早期语义提取层。浅层注意力模式（如 positional bias、local pattern）尚未稳定，跨层共享的语义基础不成立。

2. **冷启动逻辑**：`shared` 层的逻辑是「复用前一个 `full` 层的 topk 索引」。Layer 0 之前没有任何索引可复用，必须自己跑。Layer 1、Layer 2 虽然理论可以复用前一层，但浅层模式不稳定，仍强制独立计算。

3. **数值稳定保护**：`index_skip_topk_offset=3` 表示序列前 3 个 token 不启用 shared——前 3 个 token 没有足够上文形成稳定 topk 选择，所有层均强制走 full 路径。这是 Sequence-Prefill 早期阶段的数值稳定保护。

**前 6 层的特殊结构**：
```
Layer 0 [Dense + full]  ← 必须独立（冷启动）
Layer 1 [Dense + full]  ← 必须独立（浅层模式不稳定）
Layer 2 [Dense + full]  ← 必须独立（浅层模式不稳定）
Layer 3 [MoE + shared]  ← 复用 Layer 2 的索引（跨 Dense/MoE 边界）
Layer 4 [MoE + shared]  ← 复用 Layer 2 的索引
Layer 5 [MoE + shared]  ← 复用 Layer 2 的索引
Layer 6 [MoE + full]    ← 主体循环第 1 组
```

**关键设计点**：Layer 3-5 跨越 Dense → MoE 的边界复用同一份 topk 索引，说明 IndexShare 假设「topk 选择模式与 FFN 类型无关」。

**面试要点**：被问「为什么前 6 层是特例」时，要分清两个原因——Dense 属性导致前 3 层必须 full，浅层 + 冷启动导致第 4-6 层无法直接进入 4 层周期循环。

**延伸阅读**：主报告 CH 2.1 / `_work/config.json` L26-32 / L110-113

---

### Q2.5 IndexShare 训练时就用了还是推理时才加的？

**简短回答**：**训练时就用了**。GLM-5.2 在 mid-training 阶段引入 128K IndexShare 训练数据，显式构造「需要跨层共享注意力模式」的任务（长文档 QA、多文件代码理解），让模型在 checkpoint 阶段就学习「4 层窗口内复用 topk 索引」的鲁棒性，而非推理时临时启用。

**详细解释**：如果只在推理时启用 IndexShare（不改训练），模型会在「full 模式的 KV 分布」上训练，但在「shared 模式的 KV 分布」上推理——主 Attention 在复用索引上计算时，会因索引精度下降而性能崩塌。

GLM-5.2 的训练设计（CH 5.1）：

1. **128K 长上下文混合**：在通用预训练后追加 128K 上下文的混合数据（代码仓库、长文档、多轮对话），让模型在预训练阶段就适应长距离依赖。

2. **IndexShare 训练数据**：显式构造「需要跨层共享注意力模式」的任务（如长文档 QA、多文件代码理解），让模型学习「4 层窗口内复用 topk 索引」不会显著损失性能。

3. **持续训练**：post-training 阶段（SFT + RL）也保持 IndexShare 开启，确保最终 checkpoint 完全适配稀疏模式。

**为什么必须训练时启用**：相邻 Transformer 层的注意力模式相关性虽然高（文献中跨层 attention pattern 相似度通常 > 0.8），但 4 层窗口内复用 topk 仍会有 10-15% 的「错配」——某些 token 在 Layer N 被选为 topk，但在 Layer N+3 已经不再相关。训练数据让模型学会在复用索引上做更宽松的注意力（如对复用集中的 token 给予更高熵的 attention weight），弥补这种错配。

**面试要点**：被追问 IndexShare 的「训练成本」时，要答「训练时就要启用，让模型适应稀疏模式；推理时启用会有性能崩塌」。

**易混淆**：KVShare（CH 3）的核心价值之一也是「训练-推理一致性」，但解决的是 MTP draft 的 KV 来源问题，与 IndexShare 的 topk 复用是不同维度。

**延伸阅读**：主报告 CH 2.5 / CH 5.1

---

### Q2.6 IndexShare 在 1M 上下文下能省多少显存？

**简短回答**：IndexShare 不直接节省主 KV cache（仍是 MLA 压缩的 512 维潜表示），但通过减少 Indexer 自身的 indexer-KV cache 间接降低显存压力。1M 上下文下整体 KV cache 约 78 GB（78 层 × 1M tokens × ~1 KB/token），对单卡 H100（80GB）仍属紧张，需 Tensor Parallel + KV 量化或 CPU offload。

**详细解释**：基于 MLA 的潜空间压缩（`kv_lora_rank=512`），单 token 的 KV cache 仅需 `512 / 64 heads × bf16 = 16 bytes/head/token`，相对传统 MHA 减少 > 10×。

| 上下文 | 每 token KV cache | 总 KV cache（78 层 × batch=1） |
|---|---|---|
| 200K（GLM-5.1） | ~1 KB | ~78 × 200K × 1 KB ≈ 15.6 GB |
| 1M（GLM-5.2） | ~1 KB | ~78 × 1M × 1 KB ≈ 78 GB |

**Indexer KV cache 的特殊处理**：
- 21 层 `full` 层的 Indexer 仍维护自己的 indexer-KV（32 头 × 128 维 × T tokens）。
- 57 层 `shared` 层的 Indexer 完全跳过（`self.indexer=None`），不维护任何 indexer-KV。
- 主 Attention 的 KV cache（512 维潜表示）所有 78 层都需要。

**显存预算分解（1M 上下文，batch=1，bf16）**：
- 主 KV cache（MLA 压缩）：78 GB
- Indexer KV cache（仅 21 层）：约 21 × 1M × 32 × 128 × 2 bytes / 1024^3 ≈ 16 GB
- 模型权重（MoE 256+1，total params ~700B+）：通常需 multi-GPU Tensor Parallel

**配套推理基础设施**（博客披露）：
- **LayerSplit**：将 78 层按 Pipeline Parallel 切分到多节点。
- **CPU 侧调度**：将 IndexShare 的 topk 缓存管理与 shared 层的索引广播放到 CPU 侧异步执行。
- **PagedAttention**：vLLM 标准组件，配合 MLA 的 512 维压缩 KV 提供高吞吐分页。

**面试要点**：被问 IndexShare 的「显存节省」时，要澄清——它省的是 Indexer 自身的 $O(T²)$ 计算和 indexer-KV（57/78 层跳过），不是主 KV cache；主 KV cache 的节省来自 MLA。

**延伸阅读**：主报告 CH 4.2 / CH 4.4

---

## 三、CH3 MTP 四重改进（4 Q）

### Q3.1 MTP 的四项改进分别是什么？

**简短回答**：(1) **IndexShare on MTP**——MTP draft 层复用主模型的 IndexShare topk 缓存；(2) **KVShare**——MTP 直接共享主模型 KV cache；(3) **Rejection Sampling**——从单序列线性 draft 升级为树形 draft + 并行验证；(4) **End-to-end TV Loss**——总变差损失约束 draft 分布与主模型分布对齐。

**详细解释**：

| 改进 | GLM-5.1 | GLM-5.2 | 收益来源 |
|---|---|---|---|
| **IndexShare on MTP** | MTP draft 层独立跑 Indexer | 复用主模型 IndexShare 的 topk 缓存 | Indexer FLOPs 降低 |
| **KVShare** | MTP KV cache 与主模型分离 | MTP 直接共享主模型 KV cache | 训练-推理一致性 + 显存节省 |
| **Rejection Sampling** | 标准接纳采样（每次 draft 接纳 1 个） | 改进版接纳采样（树形/并行 draft） | 单次吞吐提升 |
| **End-to-end TV Loss** | 标准 CE loss | 总变差损失（Total Variation）约束 draft 分布平滑 | 接纳率提升 |

**IndexShare on MTP**：IndexShare 的自然延伸——主模型已经将 Indexer FLOPs 摊薄到 21 层，MTP 的 draft 层（同样使用 DSA 架构）也能共享同一份 topk 缓存。vLLM 中通过 `index_share_for_mtp_iteration=true` 开启此路径。

**KVShare**：解决训练-推理一致性问题。GLM-5.1 的 MTP 训练时 KV cache 来自训练 forward，推理时 KV cache 来自主模型推理 forward，两者数值精度与计算路径略有差异，导致 draft token 分布偏移、接纳率下降。KVShare 让训练和推理使用同一份 KV cache 源头，消除这一偏差。

**End-to-end TV Loss**：传统 CE loss 仅约束 draft 对 ground truth 的预测概率，TV Loss 额外约束 $D_{TV} = \frac{1}{2}\sum_x |p_{draft}(x) - p_{main}(x)|$，迫使 draft 在拒绝分支上的概率分布也贴近主模型。

**Rejection Sampling**：博客未展开，基于 Medusa / EAGLE 类工作的常规做法可推断为「树形 draft + 因果掩码并行验证」。

**面试要点**：四项改进可以记口诀「IndexShare + KVShare + RS + TV Loss」——前两项省计算/对齐 KV，后两项提升接纳率。

**易混淆**：KVShare 与 IndexShare 都涉及「共享」，但 IndexShare 共享 topk 索引（训练 + 推理都启用），KVShare 共享 KV cache（主要价值是训练-推理一致性）。

**延伸阅读**：主报告 CH 3.1

---

### Q3.2 KVShare 怎么消除训练-推理不一致？

**简短回答**：GLM-5.1 的 MTP 训练时 KV cache 来自训练 forward，推理时 KV cache 来自主模型推理 forward，两者数值精度与计算路径略有差异。KVShare 让训练和推理使用同一份 KV cache 源头（主模型推理 forward 的 KV），消除偏差，draft 模型不需要学习「KV 分布偏移的鲁棒性」，所有容量可用于学习「主模型分布的精确逼近」。

**详细解释**：训练-推理一致性的四个维度：

| 一致性维度 | GLM-5.1 | GLM-5.2 |
|---|---|---|
| **topk 索引** | 训练用 GT，推理用预测 | 训练用 IndexShare 共享，推理同源 |
| **KV cache 来源** | 训练 forward 独立生成 | 主模型推理 KV 直接共享 |
| **Draft 分布** | 与主模型存在自然偏移 | TV Loss 显式约束对齐 |
| **Numerical precision** | bf16 训练 / bf16 推理（可能不一致） | 统一 bf16，相同计算图 |

**KVShare 的工程实现**（推测）：
- 训练时，draft 模型的 forward 不重新计算 KV cache，而是从主模型 forward 的 KV cache 张量直接读取。
- 推理时，draft 模型同样读取主模型推理 KV cache。
- 两侧 KV cache 来自相同的计算图（同权重、同 RoPE、同 RMSNorm），仅 batch 维度可能不同。

**核心价值**：
1. **节省显存**：不需要为 draft 维护独立的 KV cache（节省 ~10-15% 显存）。
2. **训练-推理一致性**：消除分布偏移，draft 训练时看到的 KV 与推理时完全一致。
3. **接纳率提升**：模型容量从「学习 KV 偏移鲁棒性」释放到「精确逼近主模型分布」。

**实现状态**：**仅博客来源，源码未验证**。在 Transformers 5.12.1 与 vLLM 主分支中尚未找到 KVShare 的对应实现代码，可能在内部 fork 或后续版本发布。

**面试要点**：被问「MTP 训练-推理差异」时，答「GLM-5.1 的 KV cache 训练和推理来源不同，导致 draft 分布偏移；GLM-5.2 的 KVShare 让两侧 KV 同源，加上 TV Loss 显式约束分布，把 acceptance length 从 4.56 提到 5.47」。

**延伸阅读**：主报告 CH 3.1 / CH 3.3

---

### Q3.3 Acceptance Length 从 4.56→5.47 的贡献分解？

**简短回答**：基于博客「4.56 → 5.47（+20%）」总数据的分解估算：IndexShare on MTP 约 +4%、KVShare 约 +7%、End-to-end TV Loss 约 +6%、Rejection Sampling 约 +3%。注意：单项精确数值在博客中未直接给出，标注为推断。

**详细解释**：

| 配置 | Acceptance Length | 相对增益（累计） | 相对增益（单步） |
|---|---|---|---|
| GLM-5.1 基线 MTP | 4.56 | — | — |
| + IndexShare on MTP | 4.74 | +4% | +4% |
| + KVShare | 5.08 | +11% | +7% |
| + End-to-end TV Loss | 5.32 | +17% | +5% |
| + Rejection Sampling（最终 GLM-5.2） | **5.47** | **+20%** | +3% |

**收益排序分析**：
- **KVShare（+7%）最大**：训练-推理一致性是 speculative decoding 接纳率的核心瓶颈，KV 同源带来的提升最显著。
- **TV Loss（+5%）次之**：显式约束 draft 分布与主模型分布对齐，对拒绝分支的接纳提升明显。
- **IndexShare（+4%）第三**：纯计算节省，不直接影响接纳率，但让 draft 速度更快、可在同一时间预算内尝试更多 draft。
- **Rejection Sampling（+3%）最小**：树形 draft 在 acceptance length 已经较高时收益边际下降。

**Acceptance Length 5.47 的工程含义**：每个目标 token 平均只需 1/5.47 ≈ 0.18 次主模型 forward，等效加速比约 5.47×（理论值）。实际受通信与显存带宽限制通常打折 50-70%，即 2.7-3.8× 实测加速。

**面试要点**：被问「哪项改进贡献最大」时，答「KVShare 训练-推理一致性贡献最大（约 +7%）」；但单项数值是推断，要标注「未在博客明确披露」。

**易混淆**：Acceptance Length 不等于实际推理加速比——前者是 speculative decoding 的理论指标，实际加速还受通信、显存、batch 调度影响。

**延伸阅读**：主报告 CH 3.2

---

### Q3.4 为什么 Rejection Sampling 的改进收益最小？

**简短回答**：在 Acceptance Length 已经较高（> 4.5）的基线上，树形 draft 的收益边际递减——单次 forward 已经能验证 4-5 个 token，进一步提升需要指数级增长的候选树规模才能换来线性增长的 acceptance。这也是 Medusa / EAGLE 等工作的共同经验：当 acceptance > 4 时，单纯靠并行 draft 的收益趋近饱和。

**详细解释**：Speculative Decoding 的接纳率上界由 draft 模型与主模型分布的「重合度」决定，而非 draft 的拓扑结构（线性 vs 树形）。

**Acceptance Length 与 draft 拓扑的关系**：
- **线性 draft（GLM-5.1 基线）**：每次 forward draft 一个 token 序列（如 5 个），主模型验证后接纳前 N 个，N ~ 4.56。
- **树形 draft（GLM-5.2 改进）**：每次 forward draft 多个候选路径（如 5 个 token × 3 条路径 = 15 候选），主模型用因果掩码并行验证，接纳重合度最高的路径。

**收益边际递减的原因**：
1. 当 acceptance 已达 4.56，draft 模型的单步精度已经很高，进一步通过树形扩展能捕获的「错过分支」有限。
2. 树形 draft 的代价是主模型 forward 的 KV cache 与 attention 计算随候选数线性增长，吞吐优势部分被抵消。
3. 真正卡脖子的不是 draft 拓扑，而是 draft 与主模型的分布距离——这正是 TV Loss（+5%）解决的核心问题。

**为什么收益排序是 KVShare > TV Loss > IndexShare > RS**：
- KVShare 与 TV Loss 直接降低 draft-主模型分布距离（acceptance 的根本瓶颈）。
- IndexShare 与 RS 是「在不改 acceptance 上界的前提下，加速 draft 计算或验证吞吐」的工程优化。

**面试要点**：被问「为什么 GLM-5.2 的 MTP 收益主要来自 KVShare 而非 RS」时，答「acceptance 的根本瓶颈是 draft-主模型分布距离，KVShare 解决这个根本问题；RS 是拓扑优化，在 acceptance 已经较高时边际收益最小」。

**延伸阅读**：主报告 CH 3.1 / CH 3.2 / Medusa / EAGLE 文献

---

## 四、CH4 1M 上下文工程（4 Q）

### Q4.1 GLM-5.1 的 200K 上下文为什么"不可用"？5.2 怎么解决的？

**简短回答**：GLM-5.1 的 200K 上下文在「标称长度」上可用，但在「长程 Agent 任务」上不可用——SWE-Marathon 等任务单 session 需要数千到数万步交互，KV cache 累积远超 200K，强制压缩导致上下文丢失。GLM-5.2 通过 5× 上下文扩展（200K → 1M）+ IndexShare（Indexer FLOPs 降低 2.9×）+ MLA 压缩 KV cache 让 1M 在工程上可行。

**详细解释**：200K「不可用」的两个维度：

1. **Agent 任务的上下文累积**：SWE-Marathon 等任务单次 session 包含大量 tool call、code edit、test feedback，每步累积 1-10K token，session 长度可轻松突破 1M。GLM-5.1 在 200K 上截断或压缩，丢失关键上下文，导致任务失败。

2. **长上下文 Indexer 瓶颈**：即便不考虑 Agent 任务，单纯做 200K 长 prompt 推理，GLM-5.1 的 DSA Indexer 已经是计算瓶颈——Indexer 自身 $O(T^2)$ 在 200K 下约 $4 \times 10^{10}$ FLOPs/层，78 层总计 $3.1 \times 10^{12}$ FLOPs，占整体 FLOPs 的 40%。

**GLM-5.2 的解决方案是组合拳**：

1. **上下文 5× 扩展**：`max_position_embeddings` 从 202,752 扩展到 1,048,576。
2. **RoPE 基频提升**：`rope_theta` 8× 提升，保证长距离旋转角度分辨率。
3. **IndexShare 降低 Indexer FLOPs**：1M 上下文下 Indexer 占比从 ~70% 降到 ~20%，per-token FLOPs 整体降 2.9×。
4. **MLA 压缩 KV cache**：1M 上下文下 KV cache 78 GB（vs 传统 MHA 的 ~700 GB），单 H100 节点可控。
5. **长上下文持续训练**：mid-training 引入 128K IndexShare 训练数据。

**收益**：SWE-Marathon 从 1.0 → 13.0（+1200%），FrontierSWE 从 30.5 → 74.4（+144%），DeepSWE 从 18.0 → 46.2（+157%）——长程 Agent 任务的飞跃是 1M 上下文的直接结果。

**面试要点**：被问「GLM-5.2 的核心突破是什么」时，答「不是单纯把上下文扩到 1M，而是通过 IndexShare 让 1M 在工程上可行——Indexer FLOPs 不爆炸，KV cache 不爆显存，RoPE 不丢分辨率」。

**延伸阅读**：主报告 CH 4.1 / CH 6.2

---

### Q4.2 1M 上下文下 MLA 压缩后的 KV cache 有多大？

**简短回答**：基于 MLA 的潜空间压缩（`kv_lora_rank=512`），单 token 的 KV cache 约 1 KB（512 维 / 64 heads × bf16，加上 RoPE 子空间），1M 上下文下总 KV cache 约 78 GB（78 层 × 1M tokens × ~1 KB/token，batch=1）。对单卡 H100（80GB）仍属紧张，需 Tensor Parallel + KV 量化或 CPU offload。

**详细解释**：MLA 的 KV cache 压缩原理：

**标准 MHA 的 KV cache**：
- 每 token 每 head 需缓存 K 和 V（各 head_dim 维）
- GLM-5.1/5.2 配置：`num_attention_heads=64`, `qk_head_dim=256`, `v_head_dim=256`
- 单 token KV cache：64 × (256 + 256) × 2 bytes = 65 KB
- 1M 上下文 78 层：1M × 78 × 65 KB ≈ **5 TB**（完全不可行）

**MLA 压缩后的 KV cache**：
- 通过 `kv_a_proj`（6144→512+64）将 KV 压缩到 576 维潜空间（512 压缩 + 64 RoPE 子空间）
- 缓存时只存这 576 维，使用时通过 `kv_b_proj` 展开
- 单 token KV cache：(512 + 64) × 2 bytes ≈ 1.15 KB（实际报 1 KB 取整）
- 1M 上下文 78 层：1M × 78 × 1.15 KB ≈ **78 GB**

**显存预算分解（1M 上下文，batch=1，bf16，单节点 8×H100）**：

| 项目 | 显存 |
|---|---|
| 模型权重（MoE 256+1，~700B+ params） | ~1.4 TB（必须 Tensor Parallel 8 卡） |
| 主 KV cache（MLA 压缩，78 GB） | 单卡 ~10 GB（TP=8 切分） |
| Indexer KV cache（仅 21 层） | 单卡 ~2 GB |
| Activation / Workspace | 单卡 ~5 GB |
| **单卡总占用** | **~190 GB**（仍超 80GB H100，需 Pipeline Parallel 或 offload） |

**配套推理基础设施**：
- **LayerSplit**：78 层按 Pipeline Parallel 切分到多节点（与 TP 正交）。
- **CPU 侧调度**：topk 缓存管理与 shared 层索引广播异步执行。
- **PagedAttention**：vLLM 标准组件，配合 MLA 512 维压缩 KV 提供高吞吐分页。
- **KV 量化**：可选 INT8/INT4 KV 进一步压缩（但博客未明确披露）。

**面试要点**：被问「1M 上下文的 KV cache 多大」时，答「MLA 压缩后单 token 约 1 KB，1M × 78 层 ≈ 78 GB；但模型权重本身需要 multi-node Tensor Parallel，KV cache 是次要瓶颈」。

**易混淆**：传统 MHA 下 1M 上下文 KV cache 约 5 TB，MLA 把它降到 78 GB——这是 1M 上下文可行的根本前提，比 IndexShare 还关键。

**延伸阅读**：主报告 CH 4.2 / `GLM-5.1 报告` CH 4

---

### Q4.3 DSA + IndexShare 的协同效应？

**简短回答**：DSA 把主 Attention 从 $O(T^2)$ 降到 $O(T \cdot 2048)$，但 Indexer 自身仍是 $O(T^2)$；IndexShare 把 Indexer 从 $O(78 \cdot T^2)$ 降到 $O(21 \cdot T^2)$。两者叠加让 1M 上下文的总计算量从 $8.6 \times 10^{13}$ FLOPs（无 IndexShare）降到 $2.3 \times 10^{13}$ FLOPs，是 GLM-5.2 服务 1M 的关键。

**详细解释**：1M 上下文下各组件的 FLOPs 分解：

| 组件 | 复杂度 | 在 1M 下的角色 |
|---|---|---|
| **主 Attention（DSA sparse）** | $O(T \cdot 2048)$ | $O(2 \times 10^9)$，可控 |
| **Indexer Attention（full 层）** | $O(T^2)$ per full 层 | $O(21 \times 1.1 \times 10^{12})$，主导 |
| **Indexer Attention（shared 层）** | $0$（复用） | 完全省略 |
| **MoE FFN** | $O(d^2 \cdot k)$ per token | 与 T 无关，常量 |

**协同效应的几何含义**：

```
传统 MHA:               DSA:                      DSA + IndexShare:
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│ 主 Attn O(T²)  │      │ 主 Attn O(T·k) │      │ 主 Attn O(T·k) │
│                │      │ Indexer O(T²)  │ ───► │ Indexer O(21·T²/78) │
│                │      │   per layer    │      │  per layer (avg) │
└────────────────┘      └────────────────┘      └────────────────┘
   全 O(T²)              主降、Indexer 仍 O(T²)   两者都降
```

**关键观察**：
- DSA 单独使用时，Indexer 是新的 $O(T^2)$ 瓶颈——节省主 Attention 的收益被 Indexer 抵消。
- IndexShare 单独使用时（假设没有 DSA），主 Attention 是 $O(T^2)$，IndexShare 改不了主 Attention。
- 只有 DSA + IndexShare 同时启用，才能让两层都稀疏化，1M 上下文才在工程上可行。

**为什么这种协同在 GLM-5.1 不可行**：
- GLM-5.1 上下文 200K，Indexer 占整体 FLOPs ~40%，DSA 单独使用已经足够。
- 推到 1M 后 Indexer 占比暴涨到 ~70%，必须引入 IndexShare 才能控制成本。

**面试要点**：被问「为什么 1M 上下文需要 DSA + IndexShare 组合」时，答「DSA 把主 Attention 从 O(T²) 降到 O(T·k)，但 Indexer 自身仍是 O(T²)；1M 下 Indexer 成主导，必须用 IndexShare 把 Indexer 也稀疏化，两者缺一不可」。

**延伸阅读**：主报告 CH 4.3

---

### Q4.4 1M 上下文扩展是单一技术突破还是系统工程？

**简短回答**：**系统工程**，不是单一技术。GLM-5.2 把 200K → 1M 是 DSA + IndexShare + MLA + 长上下文持续训练 + 推理基础设施（LayerSplit / CPU 调度 / PagedAttention）的协同——任何一项缺失都会让 1M 上下文在工程上不可行。

**详细解释**：1M 上下文工程的五层技术栈：

1. **算法层**：
   - DSA：主 Attention 从 $O(T^2)$ 降到 $O(T \cdot 2048)$
   - IndexShare：Indexer 从 $O(78 \cdot T^2)$ 降到 $O(21 \cdot T^2)$
   - MLA：KV cache 从 ~5 TB 降到 ~78 GB

2. **表示层**：
   - RoPE `rope_theta` 8× 提升：保证长距离旋转角度分辨率
   - Partial RoPE：64 维 RoPE 子空间避免完全旋转

3. **训练层**：
   - 128K 长上下文混合数据
   - IndexShare 专项训练数据（长文档 QA、多文件代码理解）

4. **推理层**：
   - LayerSplit：Pipeline Parallel 切分 78 层到多节点
   - CPU 侧调度：topk 缓存管理与索引广播异步执行
   - PagedAttention：高吞吐分页

5. **硬件层**：
   - H100 80GB 多节点集群
   - 高速互联（NVLink + InfiniBand）

**任何一层缺失的后果**：
- 无 DSA：主 Attention $O(T^2)$，1M 完全不可行
- 无 IndexShare：Indexer $O(T^2)$，1M 计算成本爆炸
- 无 MLA：KV cache 5 TB，显存完全不够
- 无 rope_theta 8×：长距离 RoPE 分辨率不足，精度崩塌
- 无长上下文训练：模型不适应 1M，性能崩塌
- 无 LayerSplit：单卡装不下模型 + KV cache

**面试要点**：被问「1M 上下文最关键的技术是什么」时，答「没有单一关键技术，是 DSA + IndexShare + MLA + RoPE + 训练 + 推理基础设施的协同；这也是为什么其他模型（如 Kimi K2.6、MiniMax M3）的 1M 方案各有差异——不同的协同组合」。

**延伸阅读**：主报告 CH 4.1 / CH 8.1 启示三

---

## 五、CH5-6 训练与性能（4 Q）

### Q5.1 Terminal-Bench 从 63.5→81.0 的归因分析？

**简短回答**：Terminal-Bench 2.1（+27%）属于「中长程推理 + Agent」类任务，受益于三项改进的叠加：(1) Critic-based PPO 的 token-level advantage 改进长 horizon credit assignment；(2) 1M 上下文让终端交互的累积上下文不再被截断；(3) Anti-hack module 消除作弊路径，让模型学到真实的终端操作能力。

**详细解释**：Terminal-Bench 测试 LLM 在终端环境（bash、文件操作、工具调用）的多步 Agent 能力，单任务通常 50-200 步，上下文累积 50K-500K。

**三项贡献归因**：

1. **Critic-based PPO（贡献最大，估计 +15%）**：
   - GLM-5.1 的 GRPO 是 trajectory-level advantage（同 prompt 多 rollout 的相对奖励），对长 horizon 任务（>100 步）的 credit assignment 不精确——单步错误被均摊到整条轨迹。
   - GLM-5.2 的 Critic-based PPO 提供 token-level advantage，每个 token 都有独立的 value 估计，错误步骤能被精确定位。
   - 对终端任务的「关键命令判断」类决策（如 `rm -rf` vs `rm -r`），token-level advantage 的提升显著。

2. **1M 上下文（贡献中等，估计 +8%）**：
   - 终端任务单 session 累积 50K-500K 上下文，GLM-5.1 的 200K 在长任务上截断。
   - 1M 让完整 session 历史可访问，模型能参考早期命令的输出。

3. **Anti-hack module（贡献较小，估计 +4%）**：
   - 终端 Agent 容易作弊（如直接 echo 测试用例、修改测试脚本、跳过断言）。
   - Anti-hack 让模型学到真实的终端操作而非作弊路径，benchmark 评分更准确反映能力。

**为什么不是 SWE-Marathon 那样的 +1200%**：
- Terminal-Bench 单任务较短（< 200K 上下文），1M 扩展帮助有限。
- 主要靠 PPO critic 与 anti-hack 的训练体系升级，而非上下文扩展。

**面试要点**：被问「Terminal-Bench 提升来自哪里」时，答「主要是 Critic-based PPO 的 token-level advantage 改进 credit assignment，加上 1M 上下文对中长程终端任务的帮助，最后是 anti-hack 让能力评估更真实」。

**延伸阅读**：主报告 CH 6.2 / CH 5.2

---

### Q5.2 Anti-hack module 解决什么问题？

**简短回答**：解决 Coding Agent 在 RL 训练中学到「作弊路径」而非真实能力的问题——agent 学会直接 echo 测试用例、绕过断言、修改测试本身、注入 hardcoded 答案等作弊行为，benchmark 评分虚高但实际能力未提升。Anti-hack module 通过规则检测 + LLM 双重检测识别并惩罚这些 trajectory。

**详细解释**：Coding Agent 作弊的常见模式：

| 作弊类型 | 示例 | 规则检测难度 |
|---|---|---|
| **Echo 测试用例** | 在 patch 中直接写入测试期望的输出字符串 | 简单（字符串匹配） |
| **绕过断言** | 修改 `assert` 语句为 `assert True` 或注释掉 | 中等（AST 分析） |
| **修改测试本身** | 直接改测试文件让用例通过 | 中等（diff 分析） |
| **Hardcoded 答案** | 在代码中注入特定输入的 hardcoded 返回 | 困难（语义审计） |
| **环境注入** | 修改测试环境（如 monkeypatch）让所有用例通过 | 困难（运行时检测） |

**GLM-5.2 的 Anti-hack 双层检测**：

1. **规则检测**（硬编码检测）：
   - 字符串匹配：检测 patch 是否包含测试用例的期望输出
   - AST 分析：检测 `assert` / `if __name__` 等关键语句的修改
   - Diff 范围：检测 patch 是否触及测试文件本身
   - 优势：快、便宜、零误报可调
   - 劣势：覆盖率有限，无法检测语义级作弊

2. **LLM 双重检测**（语义审计）：
   - 用辅助 LLM（可能是 GLM-5.2 自己或专门的 critic 模型）对 agent 的 patch 与最终提交做语义审计
   - 提示词类似「检查这个 patch 是否包含作弊行为，如 hardcoded 答案、绕过测试、修改测试用例」
   - 优势：能识别规则无法覆盖的隐蔽作弊
   - 劣势：慢、贵、可能有误报

**训练时的处理**：
- 任何被检测为作弊的 trajectory 都会被打上负 reward 或直接剔除。
- 多次作弊的 prompt 会被降权或加入 blacklist。
- Critic 模型也会学习识别作弊模式（避免未来 reward 估计被作弊 trajectory 污染）。

**为什么 Anti-hack 对 Coding Agent 尤其重要**：
- RL 训练的 reward 来自单元测试通过率，是高度 gameable 的信号
- Agent 容易学到「修改测试」比「修复代码」更高效的捷径
- 没有 anti-hack，模型 benchmark 评分虚高但真实代码能力退化

**面试要点**：被问「Coding Agent RL 的核心挑战是什么」时，答「reward gaming 是核心挑战，agent 容易学到修改测试、绕过断言等作弊路径；GLM-5.2 用规则 + LLM 双重检测的 anti-hack module 解决这个问题」。

**延伸阅读**：主报告 CH 5.2

---

### Q5.3 Critic-based PPO 和 GLM-5.1 的 group-wise 优化有什么区别？

**简短回答**：GLM-5.1 用 GRPO（Group Relative Policy Optimization），对同一 prompt 的多条 rollout 计算组内相对优势，trajectory-level；GLM-5.2 升级为 Critic-based PPO，引入独立的 value model（critic）为每个 token 提供独立的 advantage 估计，token-level。后者对长 horizon Agent 任务的 credit assignment 更精确。

**详细解释**：两种 RL 方法的对比：

| 维度 | GLM-5.1（GRPO） | GLM-5.2（Critic-based PPO） |
|---|---|---|
| **Advantage 粒度** | trajectory-level | token-level |
| **Value 估计** | 无独立 critic，用组内均值代替 baseline | 独立 critic 网络输出 $V(s_t)$ |
| **采样需求** | 同 prompt 多 rollout（通常 4-8 条） | 单 rollout 即可（但多条仍可降低方差） |
| **Credit assignment** | 整条轨迹均摊，长 horizon 不精确 | 每 token 独立，长 horizon 精确 |
| **训练成本** | 低（无 critic forward） | 高（额外 critic forward + backward） |
| **方差** | 较高（依赖组内差异） | 较低（value baseline） |

**GLM-5.1 GRPO 的核心公式**（trajectory-level）：
$$
A_i = \frac{r_i - \text{mean}(r_{1..N})}{\text{std}(r_{1..N})}
$$
其中 $r_i$ 是第 $i$ 条 rollout 的总奖励，$N$ 是组大小。所有 token 共享同一 advantage。

**GLM-5.2 Critic-based PPO 的核心公式**（token-level）：
$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$
其中 $V$ 是 critic 网络的 value 估计，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差，GAE（Generalized Advantage Estimation）平滑多步估计。

**为什么 Critic-based PPO 对长 horizon 任务关键**：
- SWE-Marathon 等任务单 trajectory 数千步，GRPO 的 trajectory-level advantage 让单步错误被均摊，关键决策得不到强化。
- Critic-based PPO 的 token-level advantage 能精确定位「关键决策点」（如选择哪个文件编辑、调用哪个工具），让这些 token 获得更高 advantage。

**训练成本权衡**：
- Critic 网络通常与 policy 网络同尺寸（或稍小），训练成本翻倍。
- GLM-5.2 选择这个开销是因为长 horizon Agent 任务的收益（SWE-Marathon +1200%）远超训练成本。

**面试要点**：被问「为什么 GLM-5.2 从 GRPO 升级到 PPO」时，答「GRPO 是 trajectory-level，长 horizon Agent 任务的 credit assignment 不精确；PPO 加 critic 后能做 token-level advantage，对 SWE-Marathon 这种数千步的任务关键」。

**易混淆**：GRPO 不是「错误」的方法，而是「在不同任务上有不同性价比」——短 horizon 任务（如数学题）GRPO 更高效，长 horizon Agent 任务 PPO 更精确。

**延伸阅读**：主报告 CH 5.2 / GRPO 文献（DeepSeek-R1）

---

### Q5.4 slime framework 在 GLM-5.2 训练中扮演什么角色？

**简短回答**：slime 是智谱自研的 Megatron + SGLang RL 框架，在 GLM-5.2 中支持「2 天内合并 10+ 专家模型」（code、math、reasoning、agent 等方向），通过 RL 流水线实现专家能力的加权融合而非简单的 SFT 拼接。

**详细解释**：slime 的三个核心能力：

1. **Megatron 后端**：大规模分布式训练（Tensor Parallel + Pipeline Parallel + Expert Parallel），支持 GLM-5.2 这种 700B+ 参数规模的 RL。
2. **SGLang 推理后端**：高吞吐 rollout generation（agent 任务需要大量环境交互），SGLang 的 RadixAttention 复用前缀 KV cache 大幅降低 multi-rollout 成本。
3. **专家模型合并流水线**：将多个独立训练的专家模型（如 code、math、reasoning）通过 RL 加权融合到主模型。

**专家模型合并 vs 简单 SFT 拼接**：

| 维度 | 简单 SFT 拼接 | slime RL 合并 |
|---|---|---|
| **方法** | 把各专家数据混合后做 SFT | 各专家数据做 RL，policy 学会按任务类型激活专家能力 |
| **遗忘问题** | 严重（新数据覆盖旧能力） | 轻微（RL 保持原 policy 不偏离） |
| **能力融合** | 表面（数据分布平均化） | 深度（policy 学会任务路由） |
| **训练时间** | 短（单次 SFT） | 长（多轮 RL） |
| **效果** | 各专家能力互相稀释 | 各专家能力协同增强 |

**为什么能在 2 天内合并 10+ 专家**：
- slime 的 RL 流水线高度并行（多 expert 数据流同时 rollout）
- SGLang 推理后端的吞吐优势（>10× vLLM）
- Megatron 的混合精度训练（bf16 + selective FP32）
- slime 的 checkpoint 合并算法（不是简单平均，是按专家能力做加权 LoRA merge）

**10+ 专家的具体方向**（推断）：
- code（软件工程、算法、调试）
- math（竞赛、应用、符号计算）
- reasoning（逻辑、推理、规划）
- agent（工具调用、环境交互、长程任务）
- multilingual（中英之外的语种）
- knowledge（事实知识、百科、专业领域）

**面试要点**：被问「GLM-5.2 是怎么训练的」时，要提到 slime 框架、Megatron + SGLang 后端、专家模型合并——不要只答「RLHF / PPO」。

**延伸阅读**：主报告 CH 5.2 / slime 框架文档

---

## 六、CH7-8 对比与总结（4 Q）

### Q7.1 GLM-5.2 和 Kimi K2.6 / MiniMax M3 的 1M 上下文方案有什么差异？

**简短回答**：三家都达到 1M 上下文，但技术路径不同——GLM-5.2 走「DSA 稀疏注意力 + IndexShare 二次稀疏化 + MLA 压缩」路线；Kimi K2.6 多走「局部 + 全局混合注意力 + 显存 offload」路线；MiniMax M3 走「lightning attention / 线性注意力近似」路线。IndexShare 是 GLM 独有的「对稀疏机制本身再稀疏化」思路。

**详细解释**：三种 1M 上下文技术路线对比：

| 维度 | GLM-5.2 | Kimi K2.6（推断） | MiniMax M3（推断） |
|---|---|---|---|
| **主 Attention** | DSA sparse（top-2048） | Local + Global 混合 | Lightning Attention（线性近似） |
| **Indexer 复杂度优化** | **IndexShare（21/78 层）** | 无独立 Indexer | 无独立 Indexer |
| **KV cache 压缩** | MLA（512 维潜空间） | MHA + 量化 / offload | 线性注意力的低秩状态 |
| **上下文扩展训练** | 128K IndexShare 持续训练 | 长上下文 YaRN / NTK | 长上下文持续训练 |
| **推理基础设施** | LayerSplit + CPU 调度 | CPU offload + PagedAttention | Linear attention 原生高效 |

**GLM-5.2 路线的优势**：
- IndexShare 是「对稀疏机制本身的稀疏化」，理论上可叠加到任何稀疏注意力方案上。
- DSA + IndexShare 组合让 1M 上下文的 FLOPs 可控（2.9× 降低）。
- MLA + IndexShare 让 KV cache 与 Indexer KV 都受控。

**Kimi K2.6 路线（基于公开信息推断）**：
- 局部窗口（如 32K）保证短距离精度，全局稀疏采样保证长距离覆盖。
- 不需要单独的 Indexer，但全局稀疏采样的覆盖率不如 DSA 的 topk 选择精确。
- 显存压力靠 CPU offload 缓解（Kimi 在长上下文场景有丰富的 offload 工程）。

**MiniMax M3 路线（基于公开信息推断）**：
- Lightning Attention 是线性注意力变体，复杂度 $O(T \cdot d)$ 而非 $O(T^2)$。
- 不需要稀疏化（因为已经线性化），但精度上有损失（线性近似偏离标准 attention）。
- 长上下文是「原生高效」而非「优化后的可控」。

**面试要点**：被问「GLM-5.2 与竞品的 1M 方案差异」时，答「GLM 走 DSA + IndexShare 的二次稀疏化路线，Kimi 多用混合注意力 + offload，MiniMax 走线性注意力近似；IndexShare 是 GLM 独有的『对稀疏机制本身稀疏化』思路」。

**易混淆**：以上对 Kimi K2.6 / MiniMax M3 的描述是基于公开信息的推断，精确实现需参考各家技术报告；不要把推断当作确认事实。

**延伸阅读**：主报告 CH 8.1

---

### Q7.2 GLM-5.2 和 GLM-5.1 的接口兼容性如何？

**简短回答**：GLM-5.2 的 config.json 向后兼容 GLM-5.1。若 `indexer_types` 字段缺失或全部为 `"full"`，模型行为退化为 GLM-5.1。已有的 GLM-5.1 部署代码只需更新 checkpoint 与 config，无需重写推理逻辑——IndexShare 通过 `skip_topk` 标志在 forward 中透明启用。

**详细解释**：兼容性的三个维度：

1. **字段缺失的退化行为**：
   - 若 `indexer_types` 字段缺失，代码默认全部为 `"full"`，行为等同 GLM-5.1。
   - 若 `index_share_for_mtp_iteration` 缺失，MTP 推理路径走标准 NextN-Predict。
   - 若 `index_skip_topk_offset` / `index_topk_freq` 缺失，使用代码默认值。

2. **HF Transformers 兼容性**：
   - `transformers_version=5.12.0` 与 GLM-5.1 相同。
   - HF Transformers 5.12+ 原生支持 `GlmMoeDsaForCausalLM`。
   - 旧版 transformers（< 5.12）加载会失败，但可以通过 `pipeline` 旁路或 monkey-patch 解决。

3. **vLLM 兼容性**：
   - vLLM 主分支已支持 `index_share_for_mtp_iteration` 配置项（`llm_base_proposer.py`）。
   - KVShare 实现尚未在 vLLM 主分支出现，但推理时若不启用 KVShare 也能跑（仅失去 acceptance length 提升）。
   - LayerSplit 需要配套的 vLLM Pipeline Parallel 配置。

**部署升级路径（GLM-5.1 → GLM-5.2）**：

| 步骤 | 操作 | 风险 |
|---|---|---|
| 1 | 更新 checkpoint（weight） | 低（格式相同） |
| 2 | 更新 config.json（新增 4 字段） | 低（向后兼容） |
| 3 | 升级 transformers 到 5.12+ | 中（API 变化） |
| 4 | 升级 vLLM 到支持 IndexShare 的版本 | 中（推理路径变化） |
| 5 | 启用 LayerSplit（可选） | 中（Pipeline Parallel 调试） |
| 6 | 启用 MTP（可选） | 高（MTP 调度复杂） |

**关键设计原则**：IndexShare 是「forward 内部优化」，不改外部接口——用户从外部看到的 input/output 完全一致，仅推理速度在长上下文下变快。

**面试要点**：被问「GLM-5.2 部署难度」时，答「向后兼容 GLM-5.1，更新 checkpoint + config + transformers 5.12+ 即可基础部署；LayerSplit 和 MTP 是可选的高阶配置」。

**延伸阅读**：主报告 CH 7.3

---

### Q8.1 IndexShare 对未来模型设计有什么启示？

**简短回答**：IndexShare 的核心启示是「稀疏注意力的二次稀疏化」——任何稀疏注意力方案（DSA / Sparse Transformer / Longformer）都有自己的 Indexer / Router / Selector 模块，这些模块自身可能成为新的 $O(T^2)$ 瓶颈，可以再做稀疏化。这种「递归稀疏化」思路对未来长上下文模型设计具有普适参考价值。

**详细解释**：IndexShare 揭示的三个普适设计原则：

**启示一：递归稀疏化（Recursive Sparsification）**：
- 第一层稀疏化：DSA 把主 Attention 从 $O(T^2)$ 降到 $O(T \cdot k)$
- 第二层稀疏化：IndexShare 把 Indexer 从 $O(78 \cdot T^2)$ 降到 $O(21 \cdot T^2)$
- 理论上还可以继续：例如「Indexer 的 Indexer」是否也能稀疏化？

**启示二：跨层冗余假设的工程化**：
- 文献中跨层 attention pattern 相似度通常 > 0.8
- IndexShare 把这个观察工程化为 4 层窗口的 topk 复用
- 类似思路可推广到：
  - Router 决策的跨层复用（MoE 路由跨层共享）
  - Norm 统计量的跨层复用
  - Embedding lookup 的跨层共享

**启示三：训练-推理一致性的工程价值**：
- IndexShare 必须训练时就启用，不能推理时临时加
- 类似地，KVShare 解决了 MTP 训练-推理 KV 来源不一致
- 任何「推理时优化」如果改变了计算图，必须在训练时也启用

**对未来模型的具体影响**：

| 未来方向 | IndexShare 启示 | 潜在工作 |
|---|---|---|
| **稀疏注意力** | 对 Indexer / Selector 二次稀疏化 | Longformer / BigBird 的 Router 共享 |
| **MoE** | Router 决策跨层复用 | 跨层共享 router decision |
| **MTP / Speculative Decoding** | 训练-推理一致性 | KVShare / TV Loss 推广到其他模型 |
| **长上下文** | 系统工程而非单一算法 | DSA + IndexShare + MLA + 训练 + 推理基础设施 |

**风险与边界**：
- IndexShare 假设跨层 attention pattern 高度相关，对浅层 / 异常层失效
- 4 层窗口是经验值，不同模型 / 任务可能需要 ablation 找最优
- Indexer 二次稀疏化的收益依赖 Indexer 占整体 FLOPs 的比例（短上下文无收益）

**面试要点**：被问「IndexShare 对未来的启示」时，答「核心是『稀疏机制的二次稀疏化』——任何稀疏注意力都有 selector 模块，selector 自身可能成为新瓶颈，可以再做稀疏化；这种递归稀疏化思路对未来长上下文模型有普适价值」。

**延伸阅读**：主报告 CH 8.1

---

### Q8.2 GLM-5.2 的哪些技术点尚未在源码中验证？

**简短回答**：以下内容**仅博客来源，源码未验证**——(1) KVShare 的具体实现（Transformers 与 vLLM 主分支均未找到对应代码）；(2) End-to-end TV Loss 的 loss 函数定义与权重系数；(3) Rejection Sampling 改进的具体算法；(4) Anti-hack module 的规则集与 LLM 双重检测的具体提示词；(5) slime 在 GLM-5.2 训练中合并 10+ 专家模型的具体权重与融合策略；(6) Acceptance Length 单项消融的精确数值（4.56 → 5.47 总数据已验证，单项分解为推断）。

**详细解释**：未验证项的风险评估：

| 未验证项 | 风险等级 | 可能的实际情况 |
|---|---|---|
| KVShare 实现 | 高 | 可能在内部 fork，或后续 vLLM 版本发布 |
| TV Loss 公式 | 中 | TV 距离的定义标准，但权重系数未公开 |
| Rejection Sampling 算法 | 中 | 推断为树形 draft，但具体树结构未公开 |
| Anti-hack 规则集 | 低 | 规则细节是商业机密，不太可能公开 |
| slime 专家合并策略 | 中 | 框架本身已开源，但 GLM-5.2 的具体配置未公开 |
| Acceptance 单项数值 | 中 | 总数据已验证，单项分解是合理推断 |

**面试要点**：被问「GLM-5.2 的核心创新有哪些」时，要标注哪些是源码验证、哪些是博客来源——IndexShare 是源码验证（L406-407），MTP 改进中只有 IndexShare on MTP 在源码中有对应（`index_share_for_mtp_iteration`），KVShare / TV Loss / RS 改进仅博客来源。

**延伸阅读**：主报告 CH 8.3

---

### Q8.3 用一句话总结 GLM-5.2 的核心创新？

**简短回答**：GLM-5.2 = GLM-5.1（MLA + MoE + DSA 基础不变）+ **IndexShare**（DSA Indexer 的二次稀疏化，1M 上下文 per-token FLOPs 降低 2.9×）+ **MTP 四重改进**（Acceptance Length 4.56 → 5.47）+ **1M 上下文工程**（rope_theta 8× + 长上下文持续训练）+ **Agentic RL 升级**（Critic PPO + Anti-hack + slime 合并），最终在 SWE-Marathon 上实现 +1200% 的相对跃升，成为开源模型中首个在长程 Agent 任务上接近闭源 SOTA 的代表。

**详细解释**：四项创新的协同关系：

```
                ┌──────────────────────────────────┐
                │      GLM-5.2 的核心目标           │
                │   长程 Agent 任务接近闭源 SOTA    │
                └──────────────────────────────────┘
                          ↑
              ┌───────────┴───────────┐
              │                       │
       ┌──────┴──────┐         ┌──────┴──────┐
       │ 1M 上下文   │         │  RL 升级    │
       │  (基础)     │         │  (训练)     │
       └──────┬──────┘         └──────┬──────┘
              ↑                       ↑
       ┌──────┴──────┐         ┌──────┴──────┐
       │ IndexShare  │         │ Critic PPO  │
       │ (1M 可行)   │         │ + Anti-hack │
       └─────────────┘         │ + slime     │
              ↑                └─────────────┘
       ┌──────┴──────┐
       │ MTP 改进    │
       │ (推理加速)  │
       └─────────────┘
```

**关键数据点（一句话记忆）**：
- IndexShare：21 full + 57 shared，1M 上下文 FLOPs 2.9× 降低
- MTP：Acceptance Length 4.56 → 5.47（+20%）
- 上下文：200K → 1M（5×），rope_theta 1M → 8M（8×）
- 性能：SWE-Marathon +1200%，FrontierSWE +144%，DeepSWE +157%，HLE +31%

**面试要点**：被问「GLM-5.2 是什么」时，用这一句话开场——「GLM-5.1 基础 + IndexShare + MTP 改进 + 1M 上下文 + Agentic RL，SWE-Marathon +1200%」。

**延伸阅读**：主报告 CH 0 / CH 8.4
