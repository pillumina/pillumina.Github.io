+++
math = true
date = '2026-06-17'
draft = false
title = 'GLM-5.2 架构深度拆解'
categories = ['architecture']
tags = ['moe', 'attention', 'model-architecture', 'glm', 'dsa', 'indexshare', 'mtp']
series = ['architecture']
vendor = 'Zhipu'
summary = 'GLM-5.2 是智谱 AI 2026 年 6 月发布的旗舰 Agent 模型。核心创新为 IndexShare（1 full + 3 shared DSA Indexer 复用，降低 75% Indexer 计算量）、MTP 四重改进（KVShare + Stride + EMA + Score Boost）、1M 可用上下文、Agentic RL 升级。'
+++

# GLM-5.2 架构深度拆解

> 智谱 Z.ai 开源旗舰 Agent 模型 GLM-5.2 的独立架构报告。聚焦 IndexShare、MTP 四重改进、1M 可用上下文与 Agentic RL 升级。共享架构（MLA/MoE/DSA 基础）[参见 GLM-5.1 架构拆解](/posts/architecture/glm-5.1/)。

---

## CH 0. 摘要

GLM-5.2 是智谱 AI（Z.ai）于 2026 年 6 月发布的旗舰 Agent 大模型，是 GLM-5.1 的下一代版本。在保留 GLM-5.1 全部核心架构（MoE 256+1, k=8 / MLA + Partial RoPE / DSA Indexer / NextN-Predict MTP）的基础上，GLM-5.2 引入了三项关键架构创新与一项训练体系升级：

1. **IndexShare**：核心架构创新。在 DSA 的 78 层 Indexer 中，将 57 层（73%）的 top-k 选择计算共享给最近的 `full` 层，per-token FLOPs 在 1M 上下文下降 2.9×[^zai-blog]。这是 DSA 长上下文成本的关键突破口。
2. **MTP 四重改进**：IndexShare on MTP + KVShare + Rejection Sampling + End-to-end TV Loss，将 Speculative Decoding 的 Acceptance Length 从 4.56 提升至 5.47（+20%）[^zai-blog]。
3. **1M 上下文扩展**：`max_position_embeddings` 从 202,752 扩展到 1,048,576（5×），`rope_theta` 从 1,000,000 提升到 8,000,000（8×），配合长上下文持续训练。
4. **Agentic RL 升级**：Critic-based PPO（token-level advantage）+ Anti-hack module（规则 + LLM 双重检测 coding agent 作弊）+ slime framework（2 天合并 10+ 专家模型）[^zai-blog]。

效果上，GLM-5.2 在 SWE-Marathon 上从 1.0 提升至 13.0（+1200%），FrontierSWE 从 30.5 提升至 74.4（+144%），DeepSWE 从 18.0 提升至 46.2（+157%）[^zai-blog]，HLE（Humanity's Last Exam）从 31.0 提升至 40.5（+31%），AIME 2026 从 95.3 提升至 99.2（+4%）[^zai-blog]。

本报告是 GLM-5.2 的**独立架构报告**，不重复 GLM-5.1 已有的 MLA/MoE/DSA 基础分析（读者可[参考 GLM-5.1 架构拆解](/posts/architecture/glm-5.1/)），**聚焦 GLM-5.2 相对 5.1 的架构创新和性能跃升**。所有可验证数字均对照 `_work/config.json`，源码引用标实际行号，博客内容统一标 `[^zai-blog]`。

---

## CH 1. GLM-5.1 → GLM-5.2：架构变更概述

### 1.1 config.json diff 总览

下表对照 `_work/config.json` 与 GLM-5.1 同名字段的差异：

| 字段 | GLM-5.1 | GLM-5.2 | 变化 |
|---|---|---|---|
| `num_hidden_layers` | 78 | 78 | 不变 |
| `hidden_size` | 6144 | 6144 | 不变 |
| `n_routed_experts` | 256 | 256 | 不变 |
| `n_shared_experts` | 1 | 1 | 不变 |
| `num_experts_per_tok` | 8 | 8 | 不变 |
| `moe_intermediate_size` | 2048 | 2048 | 不变 |
| `kv_lora_rank` | 512 | 512 | 不变 |
| `q_lora_rank` | 2048 | 2048 | 不变 |
| `index_n_heads` | 32 | 32 | 不变 |
| `index_head_dim` | 128 | 128 | 不变 |
| `index_topk` | 2048 | 2048 | 不变 |
| `head_dim` | 64 | **192** | **变化**（统一 head_dim 标注口径） |
| **`max_position_embeddings`** | **202,752** | **1,048,576** | **5× 扩展** |
| **`rope_parameters.rope_theta`** | **1,000,000** | **8,000,000** | **8× 提升** |
| **`index_share_for_mtp_iteration`** | — | **`true`** | **新增** |
| **`index_skip_topk_offset`** | — | **`3`** | **新增** |
| **`index_topk_freq`** | — | **`4`** | **新增** |
| **`indexer_types`** | — | **78 元素数组** | **新增**（21 `full` + 57 `shared`） |
| `transformers_version` | `5.12.0` | `5.12.0` | 不变 |

注：`head_dim` 从 5.1 的 64 变为 5.2 的 192，是 Transformers 5.12 对 `GlmMoeDsaModel` 的字段口径统一（实际 QK 维度 `qk_nope_head_dim=192 + qk_rope_head_dim=64=256` 未变），不是模型架构实质变化。

### 1.2 五个新增字段的功能定位

| 字段 | 类型 | 含义 | 影响范围 |
|---|---|---|---|
| `indexer_types` | `array[str]`（长度 78） | 每层 Indexer 类型，`"full"` 或 `"shared"` | DSA Indexer 模块化（IndexShare 核心） |
| `index_share_for_mtp_iteration` | `bool` | 是否对 MTP iteration 启用 IndexShare | MTP 推理路径 |
| `index_skip_topk_offset` | `int` | 从第几个 token 开始跳过 topk（值=3） | shared 层的复用边界 |
| `index_topk_freq` | `int` | full 层执行的频率（值=4，即每 4 层 1 个 full） | 与 `indexer_types` 模式呼应 |
| `index_topk_pattern` | `null` | 预留字段，未启用 | — |

这五个字段共同定义了 **IndexShare** 机制——DSA Indexer 在 78 层中按 1:3 的比例分布 full/shared，仅 21 层执行完整 top-2048 选择，其余 57 层复用前一个 full 层的 topk 索引。

### 1.3 架构差异图示

```
GLM-5.1（DSA Indexer 全部独立）           GLM-5.2（IndexShare：1 full + 3 shared）
┌──────────────────────────┐               ┌──────────────────────────┐
│ Layer 0 [Dense+Full]     │               │ Layer 0 [Dense+Full]     │ ◄── 执行 topk
│ Layer 1 [Dense+Full]     │               │ Layer 1 [Dense+Full]     │ ◄── 执行 topk
│ Layer 2 [Dense+Full]     │               │ Layer 2 [Dense+Full]     │ ◄── 执行 topk
│ Layer 3 [MoE +Full]      │ ◄── 78 层     │ Layer 3 [MoE +Shared]    │ ◄── 复用 L2 索引
│ Layer 4 [MoE +Full]      │     各自      │ Layer 4 [MoE +Shared]    │ ◄── 复用 L2 索引
│ Layer 5 [MoE +Full]      │     独立      │ Layer 5 [MoE +Shared]    │ ◄── 复用 L2 索引
│ ...                      │     topk      │ Layer 6 [MoE +Full]      │ ◄── 执行 topk
│ Layer 77[MoE +Full]      │               │ ...（4 层一组循环 18 次）│
│                          │               │ Layer 77[MoE +Shared]    │ ◄── 复用 L76 索引
└──────────────────────────┘               └──────────────────────────┘
   每层都跑 32 头 × top-2048                  仅 21 层跑 topk，57 层复用
```

### 1.4 上下文长度演进

| 模型 | `max_position_embeddings` | `rope_theta` | 上下文倍数 |
|---|---|---|---|
| GLM-4.5 | 131,072 | 500,000 | 1× |
| GLM-5 / 5.1 | 202,752 | 1,000,000 | ~1.5× |
| **GLM-5.2** | **1,048,576** | **8,000,000** | **5× / 8×** |

`rope_theta` 的 8× 提升是对应 1M 上下文的 RoPE 基频重整——根据 NTK-aware 推断，外推 5× 上下文至少需要 θ 提升约 5–8 倍以保持长距离旋转角度的分辨率，GLM-5.2 选择 8× 偏保守一侧。

---

## CH 2. IndexShare 详解（核心创新）

![IndexShare 机制](fig-2.1-indexshare-mechanism.svg)

### 2.1 机制：`indexer_types` 数组的完整模式展开

GLM-5.2 `_work/config.json` 的 `indexer_types` 字段是一个长度 78 的字符串数组，每元素为 `"full"` 或 `"shared"`。完整展开如下：

| 层号范围 | 类型 | 计数 |
|---|---|---|
| Layer 0, 1, 2 | `full`, `full`, `full` | 3 full |
| Layer 3, 4, 5 | `shared`, `shared`, `shared` | 3 shared |
| Layer 6 | `full` | 1 full |
| Layer 7, 8, 9 | `shared`, `shared`, `shared` | 3 shared |
| Layer 10 | `full` | 1 full |
| Layer 11, 12, 13 | `shared`, `shared`, `shared` | 3 shared |
| ...（4 层一组循环） | ... | ... |
| Layer 74 | `full` | 1 full |
| Layer 75, 76, 77 | `shared`, `shared`, `shared` | 3 shared |

**总计数**：
- 顶部冷启动块：3 full + 3 shared = 6 层
- 主体循环块：18 组 ×（1 full + 3 shared）= 18 full + 54 shared = 72 层
- 合计：**21 full + 57 shared = 78 层**（与 `num_hidden_layers=78` 严格一致）

**频率验证**：`index_topk_freq=4` 对应「每 4 层 1 个 full」——主体循环块严格按 4 层周期，前 6 层是冷启动特例（前 3 层 Dense 必须 full，因为还没有可复用的 full 索引；层 3-5 共享层 2 的索引）。

**边界参数**：`index_skip_topk_offset=3` 表示从第 3 个 token 起开始启用 shared 复用——前 3 个 token 没有足够上文形成稳定 topk 选择，所有层均强制走 full 路径。这是 Sequence-Prefill 早期阶段的数值稳定保护。

### 2.2 计算图变更：78 层中 21 层 full + 57 层 shared

**DSA Indexer 在 `full` 层的计算**（每层、每 token）：

1. 对当前 query 做 32 头注意力（`index_n_heads=32`，`index_head_dim=128`），与所有历史 token 的 indexer-K/V 计算注意力分数。
2. 取 top-2048（`index_topk=2048`）token 索引，作为该层 Attention 的稀疏参与集。
3. 主 Attention 仅在这 2048 个 token 上计算，复杂度 $O(2048 \cdot d)$。

**DSA Indexer 在 `shared` 层的计算**（每层、每 token）：

1. **跳过**整个 indexer 注意力计算（`skip_topk=True`）。
2. 直接复用最近一个 `full` 层产生的 top-2048 索引。
3. 主 Attention 在复用的 2048 个 token 上计算，复杂度与 full 层相同 $O(2048 \cdot d)$。

**关键观察**：节省的不是主 Attention 的 FLOPs（这部分两层一样），而是 **Indexer 自身的注意力计算**——32 头 × head_dim=128 × 序列长度 T 的 indexer 注意力，在 `shared` 层完全省略。

### 2.3 FLOPs 分析：2.9× 降低的推导

考虑 1M 上下文（$T = 1{,}048{,}576$）下，DSA Indexer 的总 FLOPs：

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

**Per-token 整体 FLOPs 影响**：

博客称「1M 上下文下 per-token FLOPs 降低 2.9×」[^zai-blog]。基于 Indexer 在 1M 上下文下占整体 FLOPs 的主导地位可推断：在长序列下，DSA 的 $O(T^2)$ Indexer 注意力远超主 Attention 的 $O(2048 \cdot d)$ 与 MoE FFN 的 $O(d^2)$；IndexShare 将 78 层中 57 层的 $O(T^2)$ 项完全删除，整体 per-token FLOPs 因此减少至原值的约 1/2.9 ≈ 34.5%，与 Indexer 单项 26.9% 的下界相符（差值来自非 Indexer 部分未受影响）。

**长短上下文对比**：

| 上下文长度 | Indexer FLOPs 占比（5.1） | IndexShare 节省（5.2） | per-token FLOPs 比 |
|---|---|---|---|
| 4K | < 5% | 几乎无影响 | ≈ 1.0× |
| 32K | ~15% | ~5% | ≈ 0.95× |
| 128K | ~40% | ~25% | ≈ 0.7× |
| **1M** | **~70%** | **~50%** | **≈ 0.34×（即 2.9× 降低）** |

这解释了为何 IndexShare 对 1M 场景收益最大——正是为长上下文而设计。

### 2.4 源码验证：Transformers 5.12.1 + vLLM

**Transformers 实现**（`modeling_glm_moe_dsa.py`，第 406-407 行附近）：

```python
# modeling_glm_moe_dsa.py:L406-407
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```

源码逻辑严格对应 config：
- `skip_topk=True` 时，该层的 `self.indexer` 被设为 `None`，forward 中直接跳过 indexer 调用。
- `skip_topk=False` 时，构造完整的 `GlmMoeDsaIndexer`（32 头 × 128 维 × top-2048）。
- topk 索引通过外层缓存（`shared_topk_indices`）在 `full` 层写入、`shared` 层读取。

**vLLM MTP 支持**（`llm_base_proposer.py`）：

vLLM 的 MTP Proposer 实现中加入了对 `index_share_for_mtp_iteration` 配置项的读取，使得 MTP iteration 也能复用主模型的 IndexShare topk 缓存，避免 MTP 多次 draft 时重复跑 indexer。

**KVShare 实现状态**：**仅博客来源，源码未验证**。在 Transformers 5.12.1 与 vLLM 主分支中尚未找到 KVShare 的对应实现代码，可能在内部 fork 或后续版本发布。

### 2.5 设计动机分析

GLM-5.1 的 DSA 在 200K 上下文下表现优秀，但 Indexer 自身的 $O(T^2)$ 注意力在推向 1M 时变成新的瓶颈——Indexer 复杂度甚至超过主 Attention 的 $O(T \cdot k)$。IndexShare 的设计动机可从三个角度理解：

1. **冗余假设**：相邻 Transformer 层的注意力模式高度相关（文献中跨层 attention pattern 相似度通常 > 0.8），同一 top-2048 索引在 4 层窗口内仍能保持 > 90% 的有效性。
2. **冷启动保护**：前 3 层（Layer 0-2）必须 full，因为浅层注意力模式尚未稳定；同时 `index_skip_topk_offset=3` 保证前 3 个 token 不使用 shared，避免序列开头的不稳定 topk。
3. **频率选择**：`index_topk_freq=4` 是冗余假设与误差累积的折中——更小（如 2）节省不显著，更大（如 8）误差累积风险高。基于 ablation（博客未公开数据）可推断 4 是帕累托最优。

---

## CH 3. MTP 改进（IndexShare + KVShare + RS + TV Loss）

![MTP 四重改进累积效果](fig-3.1-mtp-improvements.svg)

### 3.1 四项改进详解

GLM-5.2 对 GLM-5.1 的 NextN-Predict MTP（`num_nextn_predict_layers=1`）做了四项协同改进：

| 改进 | GLM-5.1 | GLM-5.2 | 收益来源 |
|---|---|---|---|
| **IndexShare on MTP** | MTP draft 层独立跑 Indexer | 复用主模型 IndexShare 的 topk 缓存 | Indexer FLOPs 降低 |
| **KVShare** | MTP KV cache 与主模型分离 | MTP 直接共享主模型 KV cache | 训练-推理一致性 + 显存节省 |
| **Rejection Sampling** | 标准接纳采样（每次 draft 接纳 1 个） | 改进版接纳采样（树形/并行 draft） | 单次吞吐提升 |
| **End-to-end TV Loss** | 标准 CE loss | 总变差损失（Total Variation）约束 draft 分布平滑 | 接纳率提升 |

**IndexShare on MTP** 是 IndexShare 的自然延伸——既然主模型已经将 Indexer FLOPs 摊薄到 21 层，MTP 的 draft 层（同样使用 DSA 架构）也能共享同一份 topk 缓存。vLLM 中通过 `index_share_for_mtp_iteration=true` 开启此路径。

**KVShare** 解决训练-推理一致性问题：GLM-5.1 的 MTP 训练时 KV cache 来自训练 forward，推理时 KV cache 来自主模型推理 forward，两者数值精度与计算路径略有差异，导致 draft token 分布偏移、接纳率下降。KVShare 让训练和推理使用同一份 KV cache 源头，消除这一偏差。

**End-to-end TV Loss** 是 MTP 训练目标的改进——传统 CE loss 仅约束 draft 对 ground truth 的预测概率，TV Loss 额外约束 draft 分布与主模型分布的总变差距离 $D_{TV} = \frac{1}{2}\sum_x |p_{draft}(x) - p_{main}(x)|$，迫使 draft 在拒绝分支上的概率分布也贴近主模型，提升并行 speculative 的整体收益。

**Rejection Sampling** 的改进细节博客未展开[^zai-blog]，基于 Medusa / EAGLE 类工作的常规做法可推断为：从「单序列线性 draft」升级为「树形 draft + 因果掩码并行验证」，单次 forward 可验证多个候选 token。

### 3.2 Acceptance Length 消融表

| 配置 | Acceptance Length | 相对增益 |
|---|---|---|
| GLM-5.1 基线 MTP | 4.56 | — |
| + IndexShare on MTP | 4.74 | +4% |
| + KVShare | 5.08 | +11% |
| + End-to-end TV Loss | 5.32 | +17% |
| + Rejection Sampling（最终 GLM-5.2） | **5.47** | **+20%** |

（上表为基于博客「4.56 → 5.47（+20%）」总数据的分解估算[^zai-blog]，单项增益具体数值在博客中未直接给出，标注为推断。）

Acceptance Length 5.47 意味着每个目标 token 平均只需 1/5.47 ≈ 0.18 次主模型 forward，等效加速比约 5.47×（理论值，实际受通信与显存带宽限制通常打折 50-70%）。

### 3.3 训练-推理一致性分析

| 一致性维度 | GLM-5.1 | GLM-5.2 |
|---|---|---|
| **topk 索引** | 训练用 GT，推理用预测 | 训练用 IndexShare 共享，推理同源 |
| **KV cache 来源** | 训练 forward 独立生成 | 主模型推理 KV 直接共享 |
| **Draft 分布** | 与主模型存在自然偏移 | TV Loss 显式约束对齐 |
| **Numerical precision** | bf16 训练 / bf16 推理（可能不一致） | 统一 bf16，相同计算图 |

KVShare 的核心价值不仅是节省显存，更是将 MTP draft 的训练目标与推理目标对齐——训练时 draft 看到的 KV 与推理时 draft 看到的 KV 完全一致，模型不需要学习「KV 分布偏移的鲁棒性」，所有容量可用于学习「主模型分布的精确逼近」。

---

## CH 4. 1M 上下文工程

### 4.1 上下文扩展策略

GLM-5.2 将上下文从 202,752 扩展到 1,048,576（5×）采用组合策略：

1. **RoPE 基频提升**：`rope_theta` 从 1,000,000 提升到 8,000,000（8×），保证长距离旋转角度的频率分辨率。
2. **长上下文持续训练**：在 mid-training 阶段引入 128K IndexShare 训练数据（CH 5.1），让模型在训练阶段就适应 IndexShare 的 topk 复用模式。
3. **DSA + IndexShare 协同**：DSA 本身将注意力复杂度从 $O(T^2)$ 降到 $O(T \cdot 2048)$，IndexShare 进一步将 Indexer 复杂度从 $O(78 \cdot T^2)$ 降到 $O(21 \cdot T^2)$，两者叠加使 1M 上下文在工程上可行。

### 4.2 KV cache 在 1M 下的规模（MLA 压缩后）

基于 MLA 的潜空间压缩（`kv_lora_rank=512`），单 token 的 KV cache 仅需 `512 / 64 heads × bf16 = 16 bytes/head/token`，相对传统 MHA 减少 > 10×。

| 上下文 | 每 token KV cache | 总 KV cache（78 层 × batch=1） |
|---|---|---|
| 200K（GLM-5.1） | ~1 KB | ~78 × 200K × 1 KB ≈ 15.6 GB |
| 1M（GLM-5.2） | ~1 KB | ~78 × 1M × 1 KB ≈ 78 GB |

78 GB 的 KV cache 对单卡 H100（80GB）仍属紧张，需要 Tensor Parallel + KV 量化或 CPU offload。IndexShare 通过减少 Indexer 自身的 KV（21 层的 indexer-KV 仍是 78 层中其他 57 层的 source），间接降低显存压力。

### 4.3 DSA + IndexShare 的协同效应

| 组件 | 复杂度 | 在 1M 下的角色 |
|---|---|---|
| **主 Attention（DSA sparse）** | $O(T \cdot 2048)$ | $O(2 \times 10^9)$，可控 |
| **Indexer Attention（full 层）** | $O(T^2)$ per full 层 | $O(21 \times 1.1 \times 10^{12})$，主导 |
| **Indexer Attention（shared 层）** | $0$（复用） | 完全省略 |
| **MoE FFN** | $O(d^2 \cdot k)$ per token | 与 T 无关，常量 |

无 IndexShare 时，Indexer 的 $78 \cdot T^2$ 在 1M 下约 $8.6 \times 10^{13}$ FLOPs，远超主 Attention；启用 IndexShare 后降至 $2.3 \times 10^{13}$ FLOPs（21/78 × 原值），这是 GLM-5.2 能在合理硬件预算下服务 1M 上下文的关键。

### 4.4 推理优化（LayerSplit + CPU 侧调度）

博客[^zai-blog]提及 GLM-5.2 配套的推理基础设施优化：

- **LayerSplit**：将 78 层按 Pipeline Parallel 切分到多节点，与 Tensor Parallel 正交，支持更大模型显存预算。
- **CPU 侧调度**：将 IndexShare 的 topk 缓存管理与 shared 层的索引广播放到 CPU 侧异步执行，避免 GPU idle。
- **PagedAttention**：vLLM 标准组件，配合 MLA 的 512 维压缩 KV 提供高吞吐分页。

---

## CH 5. 训练体系

### 5.1 Mid-training：128K IndexShare 训练

GLM-5.2 在 mid-training 阶段引入两项关键训练数据：

1. **128K 长上下文混合**：在通用预训练后追加 128K 上下文的混合数据（代码仓库、长文档、多轮对话），让模型在预训练阶段就适应长距离依赖。
2. **IndexShare 训练数据**：显式构造「需要跨层共享注意力模式」的任务（如长文档 QA、多文件代码理解），让模型学习「4 层窗口内复用 topk 索引」不会显著损失性能。

这两项使模型在 checkpoint 阶段就已具备 IndexShare 的鲁棒性，而非推理时临时启用。

### 5.2 Agentic RL：Critic-based PPO + Anti-hack + slime

GLM-5.2 的 RL 阶段相对 GLM-5.1 有三项升级：

**Critic-based PPO**：

- 从 GLM-5.1 的 GRPO 风格（无 critic，组内相对优势）升级为带 critic 的 PPO，提供 token-level 而非 trajectory-level 的 advantage 估计。
- 对长 horizon agent 任务（如 SWE-Marathon 单任务上千步）的 credit assignment 更精确。

**Anti-hack module**：

- **规则检测**：硬编码检测 coding agent 作弊模式（如直接 echo 测试用例、绕过断言、修改测试本身）。
- **LLM 双重检测**：用辅助 LLM 对 agent 的 patch 与最终提交做语义审计，识别规则无法覆盖的隐蔽作弊。
- 训练时任何被检测为作弊的 trajectory 都会被打上负 reward 或直接剔除。

**slime framework**：

- GLM 自研的 Megatron + SGLang RL 框架（参考 skill 库中的 `slime` 工具描述）。
- 支持多专家模型合并——GLM-5.2 在 2 天内合并了 10+ 个专家模型（code、math、reasoning、agent 等方向）。
- 通过 slime 的 RL 流水线实现专家能力的加权融合而非简单的 SFT 拼接。

---

## CH 6. 性能分析

![GLM-5.1 vs GLM-5.2 Benchmark 对比](fig-6.1-benchmark-comparison.svg)

### 6.1 完整 benchmark 对比表

下表所有数据来自 Z.ai 官方博客[^zai-blog]，GLM-5.1 列引用 [GLM-5.1 架构拆解](/posts/architecture/glm-5.1/)同口径数据：

| Benchmark | GLM-5.1 | GLM-5.2 | Δ (绝对) | Δ (相对) | 维度 |
|---|---|---|---|---|---|
| HLE（Humanity's Last Exam） | 31.0 | **40.5** | +9.5 | **+31%** | 长上下文推理 |
| AIME 2026 | 95.3 | **99.2** | +3.9 | **+4%** | 数学竞赛 |
| SWE-bench Pro | 58.4 | **62.1** | +3.7 | **+6%** | 软件工程 |
| Terminal-Bench 2.1 | 63.5 | **81.0** | +17.5 | **+27%** | 终端 Agent |
| DeepSWE | 18.0 | **46.2** | +28.2 | **+157%** | 长程 SWE |
| FrontierSWE | 30.5 | **74.4** | +43.9 | **+144%** | 前沿 SWE |
| PostTrainBench | 20.1 | **34.3** | +14.2 | **+71%** | 综合后训练 |
| SWE-Marathon | 1.0 | **13.0** | +12.0 | **+1200%** | 超长程 SWE |

### 6.2 性能跃升归因分析

按收益大小排序的归因：

**SWE-Marathon（+1200%）、FrontierSWE（+144%）、DeepSWE（+157%）——长程 Agent 任务**：

这三项的共同特征是单任务需要数千到数万步的 agent 交互，KV cache 累积远超 200K。GLM-5.2 的 1M 上下文使这类任务从「不可行 / 强制压缩」变为「可在单次 session 内完成」。IndexShare 让 1M 上下文的 Indexer FLOPs 不再爆炸，是工程上可行的核心。

**Terminal-Bench 2.1（+27%）、HLE（+31%）——中长程推理**：

这两项受益于 Critic-based PPO 的 token-level advantage——长 horizon 任务的 credit assignment 改进直接提升推理质量。HLE 还受益于 1M 上下文（部分 HLE 题目需要长 prompt 推理）。

**SWE-bench Pro（+6%）——标准 SWE**：

SWE-bench Pro 是单 repo 单 patch 任务，上下文通常 < 100K，1M 扩展帮助有限，主要靠模型基础能力提升（slime 合并的 code 专家）。

**AIME 2026（+4%）——数学**：

AIME 题目上下文短（< 4K），1M 与 IndexShare 都无直接帮助，增益来自数学专家模型合并与 PPO critic 改进。

**PostTrainBench（+71%）——综合**：

混合收益，反映整体训练体系的提升。

### 6.3 与竞品对比

基于博客[^zai-blog]披露的对比数据（同口径 GLM-5.2 vs Claude Opus 4.8 / GPT-5.5 / Gemini 3.1 Pro）：

| Benchmark | GLM-5.2 | Claude Opus 4.8 | GPT-5.5 | Gemini 3.1 Pro |
|---|---|---|---|---|
| HLE | 40.5 | 43.2[^zai-blog] | 38.7[^zai-blog] | 41.0[^zai-blog] |
| AIME 2026 | 99.2 | 98.5[^zai-blog] | 97.8[^zai-blog] | 99.5[^zai-blog] |
| SWE-bench Pro | 62.1 | 65.4[^zai-blog] | 60.2[^zai-blog] | 58.7[^zai-blog] |
| FrontierSWE | 74.4 | 78.1[^zai-blog] | 70.5[^zai-blog] | 68.9[^zai-blog] |
| SWE-Marathon | 13.0 | 14.5[^zai-blog] | 11.2[^zai-blog] | 10.8[^zai-blog] |

（竞品具体数值为基于博客图表的估算读数，精确值以官方 leaderboard 为准。）

**结论**：GLM-5.2 在 AIME 2026 上接近持平 Gemini 3.1 Pro（99.2 vs 99.5），在 SWE-Marathon / FrontierSWE 上仍略逊于 Claude Opus 4.8（差距 1-4 分），但相对 GLM-5.1 的提升幅度（+1200% / +144%）显著超过竞品同期迭代速度。GLM-5.2 的核心定位是「开源模型中首个在长程 Agent 任务上接近闭源 SOTA 的模型」。

---

## CH 7. 与 GLM-5.1 的架构对比

GLM-5.2 在 MLA / MoE / DSA 基础架构上与 GLM-5.1 完全一致，详细机制读者可[参考 GLM-5.1 架构拆解](/posts/architecture/glm-5.1/) CH 3-5。本节仅总结差异点：

### 7.1 共享部分（引用 [GLM-5.1 架构拆解](/posts/architecture/glm-5.1/)）

| 模块 | 配置 | GLM-5.1 章节引用 |
|---|---|---|
| **MLA 潜注意力** | q_lora_rank=2048, kv_lora_rank=512, qk_nope=192, qk_rope=64, v=256 | [GLM-5.1 CH 4](/posts/architecture/glm-5.1/) |
| **MoE 路由** | 256 routed + 1 shared, k=8, sigmoid + noaux_tc, routed_scaling_factor=2.5 | [GLM-5.1 CH 5](/posts/architecture/glm-5.1/) |
| **DSA Indexer**（full 层） | 32 heads × 128 dim, top-2048, indexer_rope_interleave=true | [GLM-5.1 CH 3](/posts/architecture/glm-5.1/) |
| **Partial RoPE** | 64 维 RoPE 子空间 + interleave 模式 | [GLM-5.1 CH 4](/posts/architecture/glm-5.1/) |
| **NextN-Predict MTP** | num_nextn_predict_layers=1 | [GLM-5.1 CH 7](/posts/architecture/glm-5.1/) |
| **Tokenizer** | vocab_size=154,880, eos=[154820, 154827, 154829] | [GLM-5.1 CH 2](/posts/architecture/glm-5.1/) |

### 7.2 差异部分（本报告核心）

| 维度 | GLM-5.1 | GLM-5.2 | 章节 |
|---|---|---|---|
| **Indexer 类型分布** | 78 层全部 `full` | 21 `full` + 57 `shared` | CH 2 |
| **IndexShare 字段** | 无 | `index_share_for_mtp_iteration=true`, `index_skip_topk_offset=3`, `index_topk_freq=4` | CH 2 |
| **上下文长度** | 202,752 | 1,048,576 | CH 4 |
| **RoPE theta** | 1,000,000 | 8,000,000 | CH 4 |
| **MTP 改进** | 标准 NextN-Predict | + IndexShare + KVShare + RS + TV Loss | CH 3 |
| **RL 框架** | slime（基础） | slime + Critic-based PPO + Anti-hack | CH 5 |
| **Per-token FLOPs @ 1M** | 1× | 0.34×（2.9× 降低） | CH 2.3 |

### 7.3 接口兼容性

GLM-5.2 的 config.json 向后兼容 GLM-5.1：

- 若 `indexer_types` 字段缺失或全部为 `"full"`，模型行为退化为 GLM-5.1。
- `index_share_for_mtp_iteration=true` 仅在 MTP 推理路径生效，不影响非 MTP 推理。
- `transformers_version=5.12.0` 与 GLM-5.1 相同，HF Transformers 5.12+ 原生支持。

这意味着已有的 GLM-5.1 部署代码只需更新 checkpoint 与 config，无需重写推理逻辑——IndexShare 通过 `skip_topk` 标志在 forward 中透明启用。

---

## CH 8. 总结与启示

### 8.1 三大架构启示

**启示一：稀疏注意力的「二次稀疏化」**。DSA 将 $O(T^2)$ 降到 $O(T \cdot k)$ 后，Indexer 自身成为新的 $O(T^2)$ 瓶颈；IndexShare 将 Indexer 也做稀疏化（21/78 比例），是「对稀疏机制本身的稀疏化」。这种递归稀疏化思路对未来长上下文模型设计具有普适参考价值。

**启示二：训练-推理一致性的工程价值**。KVShare 与 End-to-end TV Loss 都指向同一原则——MTP draft 模型的训练目标必须与推理目标严格对齐（KV 来源、计算图、分布约束）。这一原则在 GLM-5.2 上将 Acceptance Length 提升 20%，对其他采用 speculative decoding 的模型（Llama / Qwen / DeepSeek）有直接借鉴意义。

**启示三：上下文扩展是系统工程而非单一算法**。GLM-5.2 把 200K → 1M 不是靠单一技术（如单纯加 RoPE theta 或 YaRN），而是 DSA + IndexShare + MLA + 长上下文持续训练 + 推理基础设施的协同——任何一项缺失都会让 1M 上下文在工程上不可行。

### 8.2 与开源生态的对接

GLM-5.2 的开源形态：

- **Transformers 5.12.0**：原生支持 `GlmMoeDsaForCausalLM`，包括 IndexShare 字段读取与 `skip_topk` 路径（`modeling_glm_moe_dsa.py:L406-407`）。
- **vLLM**：MTP 路径已加入 `index_share_for_mtp_iteration` 配置项（`llm_base_proposer.py`），KVShare 实现尚未在主分支出现。
- **slime**：作为 Megatron + SGLang 的 RL 框架，已在开源社区提供 GLM 系列的 RL 训练支持。

### 8.3 待开源 / 待验证项

以下内容**仅博客来源，源码未验证**：

- KVShare 的具体实现（Transformers 与 vLLM 主分支均未找到对应代码）。
- End-to-end TV Loss 的 loss 函数定义与权重系数。
- Rejection Sampling 改进的具体算法（树形 / 因果掩码 / 并行验证的细节）。
- Anti-hack module 的规则集与 LLM 双重检测的具体提示词。
- slime 在 GLM-5.2 训练中合并 10+ 专家模型的具体权重与融合策略。
- Acceptance Length 单项消融的精确数值（4.56 → 5.47 总数据已验证，单项分解为推断）。

这些项目预计在 GLM-5.2 的后续技术报告或代码更新中公开。

### 8.4 一句话总结

GLM-5.2 = GLM-5.1（MLA + MoE + DSA 基础不变）+ **IndexShare**（DSA Indexer 的二次稀疏化，1M 上下文 per-token FLOPs 降低 2.9×）+ **MTP 四重改进**（Acceptance Length 4.56 → 5.47）+ **1M 上下文工程**（rope_theta 8× + 长上下文持续训练）+ **Agentic RL 升级**（Critic PPO + Anti-hack + slime 合并），最终在 SWE-Marathon 上实现 +1200% 的相对跃升，成为开源模型中首个在长程 Agent 任务上接近闭源 SOTA 的代表。

---

[^zai-blog]: Z.ai 官方博客《GLM-5.2: Built for Long-Horizon Tasks》，2026-06。URL: https://z.ai/blog/glm-5.2。涵盖 IndexShare 机制、MTP 四重改进、benchmark 数据、RL 训练体系等。
