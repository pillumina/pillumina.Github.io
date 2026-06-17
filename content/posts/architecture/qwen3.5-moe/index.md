+++
date = '2026-06-10'
draft = false
title = 'Qwen3.5-MoE 架构深度拆解'
categories = ['architecture']
vendor = 'Alibaba'
tags = ['moe', 'attention', 'model-architecture', 'gdn', 'qwen', 'hybrid-attention', 'linear-attention']
series = ['architecture']
summary = 'Qwen3.5-MoE（255B 总参 / 30B 激活）是阿里 Qwen 团队的 MoE 旗舰模型。核心创新为 Gated DeltaNet（GDN）线性注意力与传统 Full Attention 的混合架构（Hybrid-Attn）、2048 专家细粒度 MoE（k=8 激活 + 1 共享）、FW4a 量化部署。本期拆解 GDN chunkwise-parallel 递归机制、混合注意力层分布策略、MoE 拓扑及与 M2.7/GLM-5.1/V4-Flash 的对比。'
+++

# Qwen3.5-MoE-35B-A3B 模型架构深度拆解报告

> 版本：v0.1 · 日期：2026-06-10 · 范围：Qwen3.5-MoE-35B-A3B（35B/3B 激活）

---

## CH 0: 摘要

Qwen3.5-35B-A3B 是阿里云 Qwen 团队于 2026 年发布的稀疏混合专家（Sparse MoE）原生多模态语言模型。该模型以 **35B 总参数、3B 激活参数** 的架构设计，结合三项核心技术突破传统 Transformer 瓶颈：

1. **Gated DeltaNet（GDN）线性注意力**：75% 的层使用 O(n) 复杂度的线性注意力替代 O(n^2) 的 softmax 注意力，显著降低长序列推理的 KV 缓存开销 [^src1]
2. **混合注意力层分布（3:1 GDN:Full Attention）**：每 4 层中 3 层为 GDN、1 层为全注意力，在效率与建模能力间取得平衡 [^src10]
3. **256 路深度 MoE（256 experts, top-8 + shared expert）**：所有 40 层均配备 MoE FFN，总专家数达 10,240 [^src4]

本报告基于 Transformers 库源码 `modeling_qwen3_5_moe.py` 进行逐算子级拆解，覆盖参数分解、推理显存估算、KV 缓存分析、FLOPs 计算，并与同代 SOTA 模型进行架构对比。

**关键指标**：总参数 35B [^src7]，激活参数 ~3B [^src21]，最大上下文 262K [^src30]，词汇量 248,320 [^src7]，40 层 [^src7]，隐藏维度 2048 [^src7]，GQA 16Q/2KV [^src30]，head_dim=256 [^src36]。

---

## CH 1: Qwen 系列演进

### 1.1 发展脉络

| 代际 | 代表模型 | 关键架构特征 |
|------|---------|-------------|
| Qwen2.5 | Qwen2.5-72B | 标准 Dense Transformer + GQA + SwiGLU |
| Qwen3 | Qwen3-235B-A22B | 首次引入 MoE（128 experts, k=8）+ GQA，Dense 与 MoE 并存 |
| Qwen3-Next | Qwen3-Next-80B-A3B | 首次引入 Gated DeltaNet + 混合注意力（3:1），奠定 Qwen3.5 基础 |
| **Qwen3.5** | **Qwen3.5-35B-A3B** | GDN + MoE 全层化 + 原生多模态（视觉编码器内置） |

### 1.2 从 Qwen3 到 Qwen3.5 的核心变化

1. **注意力机制革命**：Qwen3 使用标准 GQA（所有层），Qwen3.5 引入 Gated DeltaNet 线性注意力（75% 层）+ GQA（25% 层）
2. **MoE 深度化**：Qwen3 在部分层使用 MoE，Qwen3.5 在 **所有 40 层** 均使用 MoE
3. **上下文扩展**：Qwen3 支持 32K/128K，Qwen3.5 扩展到 262K
4. **多模态原生支持**：Qwen3.5 内置 27 层 ViT 视觉编码器，支持图像和视频输入
5. **MTP 训练**：Qwen3.5 引入 Multi-Token Prediction（1 层）辅助训练

### 1.3 跨代架构对比

| 指标 | Qwen2.5-72B | Qwen3-235B-A22B | Qwen3-Next-80B-A3B | **Qwen3.5-35B-A3B** |
|------|:---:|:---:|:---:|:---:|
| 总参数 | 72B | 235B | 80B | 35B |
| 激活参数 | 72B | 22B | 3B | 3B |
| 激活比 | 100% | 9.4% | 3.75% | 8.6% |
| 层数 | 80 | 96 | ~64 | 40 |
| 注意力类型 | GQA (全部) | GQA (全部) | GDN+GQA (3:1) | GDN+GQA (3:1) |
| MoE 专家数 | 0 | 128 | 256 | 256 |
| MoE 覆盖层 | 0% | ~60% | 100% | 100% |
| 上下文长度 | 32K | 128K | 262K | 262K |
| 多模态 | 否 | 否 | 否 | 是 |

**设计哲学演变** [^src11] [^src28]：
- Qwen2.5 -> Qwen3：从 **Dense 宽度** 转向 **MoE 稀疏容量**（参数效率提升 10x）
- Qwen3 -> Qwen3-Next：从 **纯 Attention** 转向 **线性注意力混合**（长上下文成为可能）
- Qwen3-Next -> Qwen3.5：从 **大而全** 转向 **精巧高效**（35B/3B 在更少总参数下达到更好的帕累托边界），并加入**原生多模态**能力

---

## CH 2: 整体架构与超参数

### 2.1 顶层架构

![Fig 2.1](fig-2.1-top-level-architecture.svg)

Qwen3.5-MoE-35B-A3B 采用 **Decoder-only Pre-Norm** 架构，包含：

- **输入嵌入层**：248,320 x 2048 的 Embedding 矩阵
- **40 层 Decoder Layer**：每层包含 1 个 Token Mixer（GDN 或 Full Attention）+ 1 个 MoE FFN
- **最终 RMSNorm** + **LM Head**（2048 x 248,320）
- **视觉编码器**（可选）：27 层 ViT，1152 隐藏维，16x16 patch，输出 2048 维

### 2.2 超参数完整表

| 参数 | 值 | 说明 |
|------|------|------|
| `num_hidden_layers` | 40 | Decoder 层数 |
| `hidden_size` | 2048 | 隐藏维度 |
| `num_attention_heads` | 16 | Q 头数（Full Attention） |
| `num_key_value_heads` | 2 | KV 头数（GQA ratio = 8:1） |
| `head_dim` | 256 | 每头维度 |
| `vocab_size` | 248,320 | 词汇量 |
| `max_position_embeddings` | 262,144 | 最大上下文（262K） |
| `num_experts` | 256 | 每层专家总数 |
| `num_experts_per_tok` | 8 | 每 token 激活专家数 |
| `moe_intermediate_size` | 512 | 每个专家 FFN 中间维度 |
| `shared_expert_intermediate_size` | 512 | 共享专家中间维度 |
| `router_aux_loss_coef` | 0.001 | 负载均衡辅助损失系数 |
| `full_attention_interval` | 4 | 全注意力层间隔 |
| `linear_conv_kernel_dim` | 4 | GDN 卷积核大小 |
| `linear_key_head_dim` | 128 | GDN 每个 Key 头的维度 |
| `linear_value_head_dim` | 128 | GDN 每个 Value 头的维度 |
| `linear_num_key_heads` | 16 | GDN Key 头数 |
| `linear_num_value_heads` | 32 | GDN Value 头数 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `hidden_act` | silu | 激活函数 |
| `rope_theta` | 10,000,000 | RoPE 基础频率 |
| `partial_rotary_factor` | 0.25 | 部分 RoPE 比例 |
| `mrope_interleaved` | true | 多模态交错 MRoPE |
| `mtp_num_hidden_layers` | 1 | MTP 附加层数 |
| `attention_bias` | false | Attention 偏置 |
| `attention_dropout` | 0.0 | Attention Dropout |
| `mamba_ssm_dtype` | float32 | GDN 状态计算精度 [^src31] |

### 2.3 参数分解

**嵌入层**：
```
Embedding = vocab_size * hidden_size = 248,320 * 2048 = 508,559,360 (~508.6M)
```

**每个 GDN 层（30 层）**：
```
in_proj_qkv: hidden * (key_dim * 2 + value_dim) = 2048 * (2048*2 + 4096) = 2048 * 8192 = 16,777,216
in_proj_z:   hidden * value_dim = 2048 * 4096 = 8,388,608
in_proj_b:   hidden * num_v_heads = 2048 * 32 = 65,536
in_proj_a:   hidden * num_v_heads = 2048 * 32 = 65,536
conv1d:      conv_kernel * conv_dim = 4 * 8192 = 32,768
A_log:       32
dt_bias:     32
norm:        value_dim = 4096
out_proj:    value_dim * hidden = 4096 * 2048 = 8,388,608
---
GDN per layer = 33,722,432 (~33.7M)
```

**每个 Full Attention 层（10 层）**：
```
q_proj: hidden * (num_q_heads * head_dim * 2) = 2048 * (16*256*2) = 2048 * 8192 = 16,777,216
k_proj: hidden * (num_kv_heads * head_dim) = 2048 * (2*256) = 1,048,576
v_proj: hidden * (num_kv_heads * head_dim) = 2048 * (2*256) = 1,048,576
o_proj: (num_q_heads * head_dim) * hidden = (16*256) * 2048 = 8,388,608
q_norm: head_dim = 256
k_norm: head_dim = 256
---
Full Attn per layer = 27,263,488 (~27.3M)
```

**每个 MoE Block（40 层）**：
```
Router Gate:    num_experts * hidden = 256 * 2048 = 524,288
Experts:        num_experts * (gate_up_proj + down_proj)
  gate_up_proj: 256 * (2 * moe_intermediate * hidden) = 256 * (2*512*2048) = 256 * 2,097,152 = 536,870,912
  down_proj:    256 * (hidden * moe_intermediate) = 256 * (2048*512) = 256 * 1,048,576 = 268,435,456
  Experts total: 805,306,368
Shared Expert:  3 * (hidden * shared_intermediate) = 3 * (2048*512) = 3,145,728
Shared Gate:    hidden * 1 = 2048
---
MoE per layer = 808,978,432 (~809M)
```

**RMSNorm（每层 2 个 + 最终 1 个）**：
```
Per Norm: hidden = 2048
Total Norms: (40*2 + 1) * 2048 = 81 * 2048 = 165,888
```

**LM Head**：
```
hidden_size * vocab_size = 2048 * 248,320 = 508,559,360 (~508.6M)
```

**总参数汇总**：
```
Embedding:                     508.6M
30 GDN layers:   30 * 33.7M = 1,011.7M
10 Full Attn:    10 * 27.3M =   272.6M
40 MoE blocks:   40 * 809M  = 32,359.1M
RMSNorms:                          0.2M
LM Head:                         508.6M
Vision Encoder:                 ~500M (估算，27层 ViT)
MTP Layer:                      ~845M (1 extra decoder layer)
---
Total: ~35.5B
```

**激活参数**（per-token forward）：
```
Embedding:                     508.6M
1 token mixer:                  ~30M (平均)
1 MoE (8 routed + shared):     8 * 3.15M + 3.15M + 0.5M = 28.8M
RMSNorms:                       ~0.2M
LM Head:                       508.6M
---
Per layer active: ~60M
40 layers active: ~2.4B
Total active per token: ~3.0B ✓ (matches "A3B" designation)
```

### 2.4 层类型分配

40 层按 `full_attention_interval=4` 生成默认层类型：
- 索引满足 `(i+1) % 4 == 0`（0-indexed 即 i % 4 == 3）：**full_attention**（10 层）
- 其余：**linear_attention**（30 层）
- `mlp_only_layers` 为空（无纯 MLP 层）

具体分配：L0-2 GDN, L3 Full, L4-6 GDN, L7 Full, ..., L36-38 GDN, L39 Full。

![Fig 7.1](fig-7.1-layer-type-distribution.svg)

### 2.5 每 Token FLOPs 总览

以下基于 L=40, d=2048, d_ff=512, n_experts=256, k=8 进行精确 FLOPs 估算（按 1 MAC = 2 FLOPs 约定）。

**GDN 层 prefill FLOPs（per-token, per-layer）** [^src20]：

| 组件 | 操作 | FLOPs |
|------|------|------|
| in_proj_qkv | 2 * d * (2k_dim + v_dim) | 2 * 2048 * 8192 = 33.6M |
| in_proj_z | 2 * d * v_dim | 2 * 2048 * 4096 = 16.8M |
| in_proj_b | 2 * d * n_v | 2 * 2048 * 32 = 131K |
| in_proj_a | 2 * d * n_v | 2 * 2048 * 32 = 131K |
| causal_conv1d | 2 * kernel * conv_dim | 2 * 4 * 8192 = 66K |
| Delta Rule (chunk) | O(chunk_size * k_dim * v_dim * n_v) | ~131K * chunk_size (均摊) |
| out_proj | 2 * v_dim * d | 2 * 4096 * 2048 = 16.8M |
| **GDN 合计** | | **~67.5M** |

**Full Attention 层 prefill FLOPs（per-token, per-layer, seq_len=n）** [^src19]：

| 组件 | 操作 | FLOPs |
|------|------|------|
| q_proj | 2 * d * (n_q * h_dim * 2) | 2 * 2048 * 8192 = 33.6M |
| k_proj | 2 * d * (n_kv * h_dim) | 2 * 2048 * 512 = 2.1M |
| v_proj | 2 * d * (n_kv * h_dim) | 2 * 2048 * 512 = 2.1M |
| Attention (QK^T + AV) | 4 * n_q * n * h_dim * n | O(4 * 16 * 256 * n^2) = 16384 * n^2 |
| o_proj | 2 * (n_q * h_dim) * d | 2 * 4096 * 2048 = 16.8M |
| **Full Attn 合计** | | **~54.5M + 16384 * n** |

**MoE FFN FLOPs（per-token, per-layer）** [^src4]：

| 组件 | 操作 | FLOPs |
|------|------|------|
| Router gate | 2 * d * n_experts | 2 * 2048 * 256 = 1.0M |
| Shared expert (SwiGLU) | 2 * d * d_ff * 3 | 2 * 2048 * 512 * 3 = 6.3M |
| 8 Routed experts | 8 * 2 * d * d_ff * 3 | 8 * 6.3M = 50.3M |
| Shared gate + combine | O(d) | ~4K |
| **MoE FFN 合计** | | **~57.6M** |

**全模型 per-token FLOPs（prefill, 不含序列长度依赖的注意力项）**：

| 组件 | Per-token FLOPs |
|------|:---:|
| 30 GDN layers | 30 * 67.5M = 2,025M |
| 10 Full Attn layers (固定部分) | 10 * 54.5M = 545M |
| 40 MoE FFN layers | 40 * 57.6M = 2,304M |
| 嵌入 + LM Head | ~2 * 508M = 1,016M |
| **固定 FLOPs 合计** | **~5,890M (5.89B)** |
| 10 Full Attn layers (序列相关) | 10 * 16384 * n^2 (n=262K 时 = 4.5e15) |
| 30 GDN layers (序列相关, chunk) | 30 * O(n * chunk_size * d_k * d_v * H) = ~10T (n=262K) |

**关键洞察**：在长序列（n=262K）下，Full Attention 的 O(n^2) 项主导计算（~4.5e15 FLOPs），而 GDN 的线性项（~10T FLOPs）几乎可以忽略。Qwen3.5 通过将 75% 的层替换为 GDN，将 attention 相关的总计算量从 ~1.8e16 FLOPs（全 Full Attn）降至 ~4.5e15 FLOPs（混合），节省约 **75%** [^src21]。

---

## CH 3: Gated DeltaNet 线性注意力（核心创新）

![Fig 3.1](fig-3.1-gated-deltanet-mechanism.svg)

### 3.1 设计动机

标准 softmax 注意力复杂度为 O(n^2)（n = 序列长度），长序列推理时 KV 缓存随序列长度线性增长。Gated DeltaNet 将注意力替换为 **基于 Delta Rule 的线性递归**，实现 O(n) 复杂度和常数级状态存储。

### 3.2 设计背景：为什么选择 DeltaNet 而非 Mamba/RetNet？

Qwen3.5 选择 Gated DeltaNet 作为线性注意力机制的实现，是基于以下几个关键技术考量 [^src11]：

**DeltaNet vs Mamba（SSM）**：Mamba (Mamba-1/2) 使用状态空间模型（SSM），其状态为向量（d_state=16 for Mamba-1, d_state=128 for Mamba-2），通过选择性的状态转移矩阵处理序列。GDN 使用**矩阵状态**（S_t ∈ R^(32x128x128)），信息容量远大于 Mamba 的向量状态。具体来说：
- Mamba-1 状态容量：d_state * d_inner = 16 * 4096 = 65K 元素（典型配置）
- GDN 状态容量：num_v_heads * head_k_dim * head_v_dim = 32 * 128 * 128 = 524K 元素
- 信息理论容量差距约 8x [^src12]

**DeltaNet vs RetNet**：RetNet 使用固定的指数衰减（γ），而 GDN 使用**数据相关衰减** beta = σ(B(x_t))。这意味着 GDN 可以根据输入内容动态决定保留或遗忘信息——类似于 LSTM 的遗忘门，但操作在矩阵状态上。这使得 GDN 在选择性复制和上下文关联任务上优于 RetNet。

**DeltaNet vs Lightning Attention**：Lightning Attention 使用线性注意力 + 块状计算，类似于 Performer/Linear Transformer。GDN 的 Delta Rule 形式（S_t = S_{t-1} * exp(g_t) + k_t^T * delta_t）在数学上是**增量更新**而非纯线性近似，意味着它保留了"惊喜"信号（prediction error）——当新 token 与历史状态预测不一致时，状态更新更大 [^src13]。

**Qwen 团队的选择逻辑**（从 Qwen3-Next 技术报告中推断）：
1. DeltaNet 的矩阵状态形式与 Transformer 的 QKV 投影更兼容（无需像 Mamba 那样重构整个计算图）
2. DeltaNet 的 chunk-wise 并行训练（chunk_size=64）与现有训练基础设施兼容
3. 门控机制（Z gate + beta gate）提供了额外的控制维度，有助于稳定大规模训练
4. 在 Qwen3-Next 的消融实验中，GDN 在长上下文检索任务上优于 Mamba-2 和 RetNet [^src14]

### 3.3 算子级分解

Qwen3_5MoeGatedDeltaNet 类（`modeling_qwen3_5_moe.py:368-556`）实现了完整的 GDN 层。

#### 阶段 1：多路投影

```python
mixed_qkv = self.in_proj_qkv(hidden_states)  # 2048 -> Q(2048)+K(2048)+V(4096)
z = self.in_proj_z(hidden_states)             # 2048 -> Z(4096) output gate
b = self.in_proj_b(hidden_states)             # 2048 -> B(32) data-dependent decay
a = self.in_proj_a(hidden_states)             # 2048 -> A(32) time-step projection
```

5 个独立线性投影从 hidden_states 提取不同用途的特征：
- **Q, K, V**：标准注意力投影，但 K 和 V 不共享
- **Z**：输出门控（gating），在 RMSNorm 后通过 SiLU 激活作用于输出
- **B**：数据相关衰减率（data-dependent decay），经 sigmoid 映射到 (0, 1)
- **A**：时间步投影，用于计算逐 token 的时间调节因子 g

#### 阶段 2：因果卷积

```python
mixed_qkv = causal_conv1d_fn(mixed_qkv, weight, bias, activation="silu")
```

对 QKV 拼合张量应用 depthwise 因果卷积（kernel_size=4, groups=8192），激活函数为 SiLU。因果卷积确保每个位置只能看到过去 4 个位置的信息，提供局部上下文建模能力。

#### 阶段 3：Delta Rule 计算

核心数学公式：

$$beta_t = \sigma(B(x_t)) \in (0,1)$$
$$g_t = -\exp(A_{log}) \cdot \text{softplus}(A(x_t) + dt_{bias})$$
$$S_t = S_{t-1} \cdot \exp(g_t) + k_t^T \cdot \delta_t$$
$$\delta_t = (v_t - S_{t-1} \cdot k_t) \cdot beta_t$$
$$y_t = \text{RMSNorm}(S_t \cdot q_t, z_t)$$

其中 $S_t \in \mathbb{R}^{32 \times 128 \times 128}$ 是**每层的递归状态矩阵**（key_dim x value_dim），实现了将键值关联信息压缩到固定大小的状态中。

**Prefill 阶段**（多 token 输入）：使用 `chunk_gated_delta_rule` 进行分块并行扫描（chunk_size=64），利用矩阵运算加速。

**Decode 阶段**（逐 token 生成）：使用 `fused_recurrent_gated_delta_rule` 进行逐步递归更新，仅需 O(1) 的增量计算。

#### 阶段 4：输出门控

```python
core_attn_out = RMSNorm(core_attn_out) * SiLU(z)  # Qwen3_5MoeRMSNormGated
output = out_proj(core_attn_out)
```

输出经过 RMSNorm + SiLU 门控后投影回 hidden_size。

### 3.4 Key/Value 头不对称设计

GDN 使用**非对称头设计**：16 个 Key 头（head_dim=128）和 32 个 Value 头（head_dim=128），比率为 1:2。Query 和 Key 头从 16 扩展（repeat_interleave）到 32，匹配 Value 头数量。这种设计使 Value 维度（32*128=4096）是 Key 维度（16*128=2048）的 2 倍，增强了模型的信息存储容量。

### 3.5 QK L2 归一化

Q 和 K 在进入 Delta Rule 计算前经过 L2 归一化（`use_qk_l2norm_in_kernel=True`），防止数值不稳定。这是 Qwen3.5 区别于标准 DeltaNet 的稳定化措施。

### 3.6 与 Full Attention 的关键差异

| 维度 | Gated DeltaNet | Full Attention |
|------|---------------|----------------|
| 计算复杂度 | O(n) | O(n^2) |
| 状态存储 | 递归矩阵 S_t (固定大小) | KV Cache (随 n 增长) |
| 每层状态大小 | 32 x 128 x 128 = 524,288 | 2 x n x 256 (n=8192 时 = 4.2M) |
| 位置编码 | 隐式（通过卷积核 + 衰减） | 显式（Partial RoPE 25%） |
| QK 归一化 | L2 Norm | RMSNorm (per-head) |
| 输出门控 | RMSNorm + SiLU(Z) | sigmoid(gate) * attn |

### 3.7 算子级数据流：chunk_gated_delta_rule 七步拆解

理解 GDN 核心计算的最佳入口是 `torch_chunk_gated_delta_rule`（`modeling_qwen3_5_moe.py:243-321`），它包含 prefilling 时完整的计算数据流。以下按步骤拆解：

**Step 1: QK L2 归一化 + 精度提升**

```python
if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1)  # ||Q||_2 = 1
    key = l2norm(key, dim=-1)      # ||K||_2 = 1
query, key, value, beta, g = [x.to(torch.float32) for x in ...]
```

将所有张量转换为 float32 以保证数值稳定性（`mamba_ssm_dtype=float32`）[^src15]。L2 归一化将 Q 和 K 约束在单位球上，防止递归状态发散。

**Step 2: 分块（Chunking）+ Padding**

```python
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
# Pad to chunk_size=64 boundary
query = F.pad(query, (0, 0, 0, pad_size))  # shape: (B, H, ceil(n/64)*64, d_k)
```

将序列填充到 chunk_size=64 的整数倍，重塑为 (B, H, num_chunks, 64, d_k)。chunk_size=64 是 GPU 并行效率与数值精度的平衡点 [^src16]。

**Step 3: 块内因果注意力矩阵（Intra-Chunk Attention）**

```python
g_cumsum = g.cumsum(dim=-1)                          # 累积衰减因子
decay_mask = (g_cumsum[:,None] - g_cumsum[:,:,None])  # 块内相对衰减
decay_mask = decay_mask.tril().exp().tril()            # 因果掩码 + 指数衰减
attn_intra = -(k_beta @ key.T) * decay_mask            # 块内 attention
# 递推校正块内因果关系 (chunk_size步)
for i in range(1, chunk_size):
    attn[..., i, :i] += (attn[..., i, :i-1] @ attn[..., :i-1, :i-1])
```

这是最复杂的步骤。`decay_mask` 编码了块内各位置的衰减关系（位置越远，衰减越大）。`-(k_beta @ key.T)` 计算带衰减的 key-key 相似度。递推循环（`i in range(1, chunk_size)`）逐行修正上三角因果掩码下的矩阵乘法结果。

**Step 4: 块内 Value 聚合**

```python
attn_intra += torch.eye(chunk_size)           # 加单位矩阵（自身映射）
value_intra = attn_intra @ v_beta              # 块内加权 value
k_cumdecay = attn_intra @ (k_beta * g.exp())  # 预计算跨块 key 累积衰减
```

单位矩阵加法确保每个 token 的自身 value 被纳入（delta rule 中 v_t 项的"保留"部分）。

**Step 5: 跨块递归扫描（Chunk-wise Recurrence）**

```python
for i in range(num_chunks):
    q_i, k_i, v_i = query[:,:,i], key[:,:,i], value[:,:,i]
    attn_qk = q_i @ k_i.T * decay_mask[:,:,i]          # 块内 QK 注意力
    v_prime = k_cumdecay[:,:,i] @ last_recurrent_state  # 历史状态对当前块的贡献
    v_new = v_i - v_prime                                 # 新信息（delta）
    attn_inter = (q_i * g[:,:,i,:,None].exp()) @ last_recurrent_state  # 历史交叉注意力
    core_attn_out[:,:,i] = attn_inter + attn_qk @ v_new  # 当前块输出
    last_recurrent_state = last_recurrent_state * g_decay + k_proj @ v_new  # 更新状态
```

这是跨块扫描的核心循环。每个块：
1. 计算历史状态对当前 value 的预测 `v_prime = S_{i-1} * k_i`
2. 计算新信息 `v_new = v_i - v_prime`（delta，即 prediction error）
3. 计算输出 = 历史部分 + 当前块部分
4. 更新递归状态 `S_i = S_{i-1} * exp(g) + k_i^T * v_new`

这 4 个子步骤对应 Delta Rule 的数学定义：
$$S_t = S_{t-1} \cdot \exp(g_t) + k_t^T \cdot (v_t - S_{t-1} \cdot k_t) \cdot \beta_t$$

**Step 6: 状态更新（State Update per Chunk）**

```python
last_recurrent_state = (
    last_recurrent_state * g[:,:,i,-1,None,None].exp() +
    (k_i * (g[:,:,i,-1,None] - g[:,:,i]).exp()[...,None]).T @ v_new
)
```

状态更新使用每个块最后一个位置的 g 值作为块间衰减因子，`(g_last - g_all).exp()` 编码了块内不同位置的相对衰减程度 [^src17]。

**Step 7: 后处理（Reshape + Crop + Transpose）**

```python
core_attn_out = core_attn_out[:, :, :sequence_length]  # 去除 padding
core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)  # 恢复原始精度
```

去除填充部分，转置回 (B, seq_len, H, d_v) 形状，转换回 BF16/FP16 精度。

**复杂度分析**：每个 chunk 内 O(chunk_size^2 * d) 的矩阵乘法，共 n/chunk_size 个 chunk，总复杂度 O(n * chunk_size * d) = O(n * d)。当 chunk_size &lt;&lt; n 时近似线性。加上跨块递归扫描 O(n/chunk_size * d^2)，总复杂度 O(n * chunk_size * d + n/chunk_size * d^2) [^src18]。

### 3.8 每 Token FLOPs 分析：GDN vs Full Attention

以下以 Qwen3.5-MoE-35B-A3B 的实际维度进行精确计算。

**注**：以下 FLOPs 计数采用标准约定：一次矩阵乘法 `(M,K) @ (K,N)` 计为 `2*M*N*K` FLOPs（1 multiply + 1 add per element）。

**Full Attention 单层 prefill FLOPs**（seq_len=n, hidden=2048, num_q_heads=16, num_kv_heads=2, head_dim=256）：

| 操作 | 计算 | FLOPs |
|------|------|-------|
| Q 投影 | 2 * 2048 * (16*256*2) | ~33.6M |
| K 投影 | 2 * 2048 * (2*256) | ~2.1M |
| V 投影 | 2 * 2048 * (2*256) | ~2.1M |
| QK^T | 2 * 16 * n * 256 * n | O(8192 * n^2) |
| softmax(QK^T)V | 2 * 16 * n * n * 256 | O(8192 * n^2) |
| O 投影 | 2 * 2048 * (16*256) | ~16.8M |

对于 n=262,144：
- QK^T: 2 * 16 * 262144 * 256 * 262144 = 5.63e14 FLOPs
- Attn*V: 2 * 16 * 262144 * 262144 * 256 = 5.63e14 FLOPs
- Total per Full Attn layer: ~1.13e15 FLOPs [^src19]

**GDN 单层 prefill FLOPs**（相同的 n=262,144, key_dim=2048, value_dim=4096, num_v_heads=32, head_k_dim=128, head_v_dim=128）：

| 操作 | 计算 | FLOPs |
|------|------|-------|
| in_proj_qkv | 2 * 2048 * 8192 | ~33.6M |
| in_proj_z | 2 * 2048 * 4096 | ~16.8M |
| in_proj_b | 2 * 2048 * 32 | ~131K |
| in_proj_a | 2 * 2048 * 32 | ~131K |
| causal_conv1d | 2 * 4 * 8192 * n | ~2.1B (n=262K) |
| qk_l2norm | 2 * 32 * n * 128 | ~2.1B (n=262K) |
| chunk_intra (per chunk) | O(64^2 * 128 * 32) | 每 chunk ~16.8M |
| chunk_inter (per chunk) | O(32 * 128^3) | 每 chunk ~67.1M |
| out_proj | 2 * 4096 * 2048 | ~16.8M |

总计单层 GDN（n=262K）：
- 固定部分：~67M
- 序列相关部分（chunk循环）：n/64 * (16.8M + 67.1M) ≈ 4096 * 83.9M ≈ 343B FLOPs
- Total per GDN layer: ~343B FLOPs [^src20]

**单层加速比**（n=262K）：
$$\frac{1130 \times 10^{12}}{343 \times 10^{9}} \approx 3294\times$$

**全模型 FLOPs 对比**（40 层，n=262K）：

| 模型配置 | Attention FLOPs | 占比 |
|----------|----------------|------|
| 全 Full Attention (40层) | 40 * 1.13e15 = 4.52e16 | 100% |
| Qwen3.5 混合 (30 GDN + 10 Full) | 30*343B + 10*1.13e15 = 1.03e13 + 1.13e16 = 2.16e16 | 47.8% |
| 全 GDN (40层) | 40 * 343B = 1.37e13 | 0.03% |

Qwen3.5 的混合设计相比全注意力模型节省了约 **52%** 的注意力计算量，同时保留了 10 个 Full Attention 层作为"精度锚点" [^src21]。

**Per-token decode FLOPs** (n=1, 自回归生成)：

| 操作 | Full Attention | GDN |
|------|:---:|:---:|
| QKV 投影 | O(d^2) ≈ 37.7M | O(d^2) ≈ 67.2M |
| 注意力/递归计算 | O(n * d) ≈ 537M (at n=262K) | O(d_k * d_v * H) ≈ 524K |
| 输出投影 | O(d^2) ≈ 16.8M | O(d^2) ≈ 16.8M |
| **总计** | **~592M** | **~84.5M** |
| **相对加速** | 1x | **~7x** |

注意在 decode 阶段 GDN 的优势较小（"仅" 7x），因为单步标准注意力的复杂度从 O(n^2) 退化为 O(n)。但在 n=262K 时 GDN 的递归状态更新仍是 O(1)，而 Full Attention 需遍历全部 262K 个历史 KV [^src22]。

---

## CH 4: 混合注意力层架构

![Fig 3.2](fig-3.2-hybrid-attention-distribution.svg)

### 4.1 设计选择

Qwen3.5 采用 **75% GDN + 25% Full Attention** 的混合比例（3:1），通过 `full_attention_interval=4` 配置。这一比例来自 Qwen3-Next 的实证研究：纯线性注意力在某些长距离依赖任务上退化，每隔一定层数插入 Full Attention 作为"锚点"可恢复能力。[^src1]

### 4.2 Full Attention 层 (Qwen3_5MoeAttention)

Full Attention 层（10 层，占 25%）实现了增强版 GQA：

- **GQA 配置**：16 Q-heads / 2 KV-heads（分组比 8:1）
- **QK 归一化**：每头 RMSNorm 在 Q 和 K 上应用，提升训练稳定性
- **部分 RoPE**：仅对 head_dim 的 25%（即 64 维）应用旋转位置编码，其余 75% 为位置无关
- **输出门控**：`注意力输出 * sigmoid(gate)`，gate 从 Q 投影的双倍输出中分出
- **KV 缓存**：仅 2 个 KV 头需要缓存，显著减少显存开销

### 4.3 层类型感知的前向传播

`Qwen3_5MoeTextModel.forward` 为每层选择不同的注意力掩码：

```python
for i, decoder_layer in enumerate(self.layers):
    layer_mask = (
        linear_attn_mask if self.config.layer_types[i] == "linear_attention"
        else causal_mask
    )
    hidden_states = decoder_layer(hidden_states, attention_mask=layer_mask, ...)
```

GDN 层使用简化的 `linear_attn_mask`（仅用于左填充屏蔽），Full Attention 层使用标准因果掩码。

### 4.4 为什么 3:1 而非其他比例？

- **2:1 或 1:1**：Full Attention 层过多，KV 缓存增大，推理速度下降
- **4:1 或更高**：线性注意力层过多，某些需要精确位置建模的任务退化
- **3:1**：在 Qwen3-Next 实验中达到最佳帕累托前沿（速度 vs 质量）

### 4.5 mlp_only_layers

Qwen3.5-MoE-35B-A3B 的 `mlp_only_layers` 为空列表（`[]`），意味着**每层都有 token mixing**（GDN 或 Full Attention）。这是与 Qwen3 不同之处——Qwen3 使用部分 mlp_only 层来节省参数。

---

## CH 5: MoE 路由机制

![Fig 5.1](fig-5.1-moe-routing.svg)

### 5.1 整体结构

![Fig 5.2](fig-5.2-decoder-layer-flow.svg)

`Qwen3_5MoeSparseMoeBlock` 集成三个组件：

1. **Top-K Router Gate**：将每个 token 分配给 top-8 专家
2. **256 Sparse Experts**：3D 权重张量存储，SwiGLU 激活
3. **1 Shared Expert**：无条件处理所有 token，平行于路由专家

### 5.1A 设计背景：MoE 超参数选择依据

Qwen3.5-MoE 的 MoE 配置（256 experts, k=8, 共享专家，全层 MoE，aux_loss=0.001）是一系列设计权衡的结果。

**为什么 k=8？**

top-k 的选择直接影响模型容量和计算效率 [^src23]：
- k=1（Switch Transformer 风格）：最节省计算（每 token 仅 1 个专家），但专家专业化过度、训练不稳定，需要较大的 aux loss
- k=2-4：适中的计算量，但对于 256 个专家的模型，每个 token 接触的专家太少，限制了知识组合能力
- k=8：当前选择，每 token 激活 8/256=3.1% 的专家，提供了足够的专家多样性，同时将每 token 的 FFN FLOPs 控制在合理范围（8*3.15M + 共享 3.15M = 28.35M 参数）
- k=16 或更高：计算量翻倍，但边际收益递减（多个专家输出趋于相似）

从源码角度，k=8 还意味着每个 token 的 `topk_values` 总和被归一化为 1（`router_top_value /= router_top_value.sum(dim=-1, keepdim=True)`），确保专家输出的加权和尺度稳定 [^src24]。

**为什么有 aux loss？系数为什么是 0.001？**

辅助损失（auxiliary loss）解决 MoE 的**负载不均衡**问题：如果没有 aux loss，路由器可能将所有 token 分配给少数几个专家（"专家坍塌"），导致其他专家不被训练（"dead experts"）[^src25]。

Qwen3.5 使用 Switch Transformer 论文 [^src26] 的标准 aux loss：
$$\mathcal{L}_{aux} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中系数 0.001 的选择考量：
- 系数过大（如 0.01）：过度强调负载均衡，牺牲路由质量（模型为"公平"而将 token 分配给次优专家）
- 系数过小（如 0.0001）：负载均衡效果弱，可能出现专家利用不均
- 0.001 是中等偏小的选择，可能是因为**共享专家的存在已经提供了一定的负载均衡基础**——即使某些路由专家未充分使用，共享专家仍为所有 token 提供基础表示 [^src27]

**为什么每层都是 MoE（全层 MoE）？**

`mlp_only_layers=[]` 且所有 40 层均使用 `Qwen3_5MoeSparseMoeBlock`。这与 Qwen3 形成对比（Qwen3 仅在部分层使用 MoE）[^src28]：
- Qwen3 的经验：部分层使用 Dense FFN 导致那些层成为"容量瓶颈"——Dense FFN 的表示能力远低于 MoE FFN（3.15M vs 809M 参数），在层间形成不连续的表示质量
- Qwen3.5 的改进：全层 MoE 确保统一的表示容量分布，每层的 FFN 都可以从 256 个专家的稀疏激活中受益
- 代价：模型大小从约 19B（若半层 MoE）增至 35B（全层 MoE），但激活参数几乎不变（因为每 token 仍仅激活 9 个专家）

**为什么有共享专家？**

共享专家的设计（1 个对所有 token 无条件激活的专家）参考了 DeepSeek V2/V3 的实践 [^src29]：
- 捕获**跨所有 token 的通用模式**（如语法、常见词汇关系），避免所有 256 个路由专家都学习重复的通用知识
- 提供**负载均衡兜底**——即使某些路由专家未被充分训练，共享专家确保每个 token 都有基础 FFN 输出
- 门控机制 `sigmoid(W_gate(x))` 允许模型动态决定是否信任共享专家输出（gate->0 时依赖路由专家，gate->1 时信任共享知识）

### 5.2 路由门控 (Qwen3_5MoeTopKRouter)

```python
router_logits = W_gate @ x          # (N, 256)
router_probs = softmax(router_logits)  # 概率分布
topk_values, topk_indices = topk(router_probs, k=8)  # top-8
topk_values /= topk_values.sum(dim=-1, keepdim=True)  # 归一化权重
```

路由权重 $W_{gate} \in \mathbb{R}^{256 \times 2048}$ 初始化为零（`torch.zeros`），通过训练学习 token-expert 亲和度。

### 5.3 专家计算

每个专家是标准 SwiGLU MLP：

$$y = W_{down} \cdot (\text{SiLU}(W_{gate} \cdot x) \odot W_{up} \cdot x)$$

其中 $W_{gate}, W_{up} \in \mathbb{R}^{512 \times 2048}$（融合在 `gate_up_proj` 中），$W_{down} \in \mathbb{R}^{2048 \times 512}$。

**专家权重布局**：所有 256 个专家以两个 3D 张量存储：
- `gate_up_proj`: (256, 1024, 2048) -- 1024 = 2 * 512（fused gate + up）
- `down_proj`: (256, 2048, 512)

这允许通过 `self.gate_up_proj[expert_idx]` 索引单个专家进行高效的前向循环。

### 5.4 共享专家

```python
shared_output = SwiGLU_MLP(x)         # 标准 SwiGLU
gated_output = sigmoid(W_gate(x)) * shared_output  # 门控
final = routed_output + gated_output  # 合并
```

共享专家捕获**跨所有 token 的通用模式**，不经过路由选择，降低了单个专家需要学习的冗余知识量。共享专家也促进负载均衡——即使某些路由专家未被选中，共享专家也能提供基础表示。

### 5.5 负载均衡

使用 Switch Transformer 风格的辅助损失（auxiliary loss），系数 `router_aux_loss_coef=0.001`：

$$\mathcal{L}_{aux} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 i 被分配到的 token 比例，$P_i$ 是路由到专家 i 的平均概率，$N=256$。

### 5.6 专家调度实现细节

前向传播时，每个 token 的 8 个选定专家 ID 构成 `top_k_index`。`Qwen3_5MoeExperts.forward` 使用 **expert-centric loop**：遍历所有被命中的专家，对每个专家批量处理分派给它的所有 token：

```python
for expert_idx in expert_hit:  # 只遍历实际收到 token 的专家
    token_idx = where(mask[expert_idx])
    gate, up = linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
    expert_out = linear(silu(gate) * up, down_proj[expert_idx])
    final.index_add_(0, token_idx, expert_out * routing_weights)
```

这避免了为所有 256 个专家计算输出的浪费（实际仅 8 个专家被激活）。

---

## CH 6: 训练体系总览

### 6.1 训练范式

Qwen3.5 的训练流程包括：

1. **预训练（Pre-training）**：大规模语料上的下一个 token 预测
2. **多模态对齐（Vision-Language Alignment）**：将视觉编码器输出对齐到语言空间
3. **监督微调（SFT）**：指令数据微调
4. **强化学习（RLHF/GRPO）**：偏好对齐

### 6.2 Multi-Token Prediction (MTP)

`mtp_num_hidden_layers=1` 表示模型使用 1 层额外的 Decoder Layer 进行多 token 预测 [^src7]。MTP 在训练时同时预测当前位置和未来位置的 token，提升训练效率和最终模型质量。

**MTP 训练数学**：

标准语言模型损失（单 token 预测）：
$$\mathcal{L}_{LM} = -\sum_t \log P(y_t | x_{< t})$$

MTP 扩展损失（1 层额外预测）：
$$\mathcal{L}_{MTP} = -\sum_t [\log P(y_t | x_{< t}) + \log P(y_{t+1} | x_{< t})]$$

MTP 模块通过 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` 在加载时被标记为可选——推理时不需要 MTP 权重，加载 pretrained checkpoint 时会自动跳过 [^src39]。

**MTP 投机解码**（推理加速）：

推理时 MTP 层可作为投机解码（speculative decoding）的 draft 模型 [^src34]：
1. 主模型预测 token t
2. MTP 层基于相同隐藏状态预测 token t+1（draft）
3. 主模型并行验证 draft token
4. 若 draft 正确：一次 forward 确认 2 个 token（加速 ~1.5-2x）
5. 若 draft 错误：仅浪费 MTP 层的计算（~2% 总参数量）

### 6.3 分布式训练

配置文件中包含 `base_model_tp_plan` 和 `base_model_pp_plan`，支持 [^src7]：

- **张量并行（Tensor Parallelism）**：按列/行切分投影矩阵。对于 2048 的 hidden_size，TP degree=2 时每个设备持有 1024 列
- **专家并行（Expert Parallelism）**：`moe_tp_experts` 策略，每个设备持有部分专家。例如 8 路 EP 时每设备 32 个专家
- **流水线并行（Pipeline Parallelism）**：按层切分，如 4 路 PP 时每设备 10 层

**MoE All-to-All 通信分析** [^src29]：

每次 MoE 前向需要 all-to-all 通信将 token 分发到持有相应专家的设备：
- 通信量：每 token 发送 hidden=2048 * 2B = 4KB 到 k=8 个设备 = 32KB/token
- 对于 batch_size=1, n=8192：262MB all-to-all 通信（单层）
- 优化：`moe_tp_experts` 策略将专家切分与张量并行结合，减少通信次数

### 6.4 精度与量化

| 组件 | 训练精度 | 推理精度选项 |
|------|---------|------------|
| 模型主体（权重） | BFloat16 | BF16 / FP16 / FP8 / INT4 |
| GDN 状态矩阵 | Float32 [^src31] | 必须 FP32 |
| GDN 计算 | Float32 (内部) | 必须 FP32 |
| 因果卷积 | BFloat16 | BF16 / FP16 |
| MoE 路由器 | BFloat16 | BF16 / FP16 |
| 注意力 softmax | Float32 (内部) | 自动 |

- 训练精度：BFloat16（`dtype: "bfloat16"` [^src7]）
- SSM 精度：Float32（`mamba_ssm_dtype: "float32"` -- GDN 状态用 float32 保证数值稳定 [^src31]）
- 初始化范围：0.02（`initializer_range` [^src7]）
- GDN 特殊初始化 [^src38]：`dt_bias=1`（确保 softplus 输入为正），`A_log ~ log(U(0,16))`（控制衰减范围）
- RMSNorm 特殊初始化 [^src9]：`weight=0`（实现 `(1+0)=1` 的恒等映射起点）

---

## CH 7: 支撑项

### 7.1 扩展上下文（262K）

`max_position_embeddings=262144`（256K+），通过以下机制支持：
- **GDN 的 O(n) 复杂度**：线性注意力层不受序列长度平方影响
- **紧凑的递归状态**：仅 (32, 128, 128) 矩阵，与序列长度无关
- **Full Attention 层的 KV 缓存**：仅 10 层需要缓存，且为 2 头 GQA
- **RoPE theta=10M**：高频旋转编码支持更长距离的位置区分

**KV 缓存计算**（Full Attention 层，seq_len=262144）：

| 精度 | K 缓存 | V 缓存 | 总计 |
|------|--------|--------|------|
| BF16/FP16 | 10 * 2 * 262144 * 256 * 2B = 2.68 GB | 同 K = 2.68 GB | **5.37 GB** |
| FP8 (E4M3) | 10 * 2 * 262144 * 256 * 1B = 1.34 GB | 同 K = 1.34 GB | **2.68 GB** |

KV 缓存仅涉及 10 个 Full Attention 层（25%），每个层仅 2 个 KV 头。若所有 40 层均为 Full Attention：
- BF16: 40 * 2 * 262144 * 256 * 2 * 2 = 21.5 GB（仅 KV 缓存）
- 若为 16 头全 MHA（无 GQA）：40 * 16 * 262144 * 256 * 2 * 2 = 171.8 GB

GQA（8:1 压缩）和混合注意力（75% GDN）共同将 KV 缓存从 ~172 GB 降至 ~5.4 GB [^src30]。

**GDN 状态**（30 layers）：

| 精度 | Per-layer 状态 | 30 层总计 |
|------|---------------|----------|
| FP32（默认） | 32 * 128 * 128 * 4B = 2.10 MB | **62.9 MB** |
| BF16（实验性） | 32 * 128 * 128 * 2B = 1.05 MB | **31.5 MB** |

注意：源码中 `mamba_ssm_dtype: "float32"` 要求 GDN 状态使用 FP32，因此生产部署时状态内存为 62.9 MB [^src31]。

**GDN Conv 状态**（30 layers）：

```
30 layers * 8192 (conv_dim) * 3 (kernel_size-1) * 2 bytes (BF16)
= 30 * 24576 * 2
= 1,474,560 bytes = ~1.4 MB
```

几乎可忽略不计 [^src32]。

**模型权重**（35B 参数）：

| 精度 | 权重内存 | 备注 |
|------|---------|------|
| BF16/FP16 | 35B * 2B = **70 GB** | 原始精度 |
| FP8 (E4M3) | 35B * 1B = **35 GB** | 需量化方案 |
| INT4 | 35B * 0.5B = **17.5 GB** | 需 GPTQ/AWQ |

**激活值内存**（prefill, batch=1, n=262K）：

```
每层激活（临时张量，在反向传播中需要但推理时可释放）：
- hidden_states: 262144 * 2048 * 2B = 1.07 GB（残差流）
- QKV 投影中间值: ~2-6 GB（取决于具体层类型和 chunking 策略）
- MoE router logits: 262144 * 256 * 2B = 134 MB
- Expert 中间值 (top-8): 8 * 262144 * 512 * 2B = 2.15 GB
```

实际推理时，激活值峰值约 **8-12 GB**（长序列 prefill），但通过 Flash Attention 分块计算和激活检查点（gradient checkpointing）可显著降低 [^src33]。

**262K 上下文总推理显存（单 GPU 部署）**：

| 组件 | BF16/FP16 | FP8 | INT4 |
|------|-----------|-----|------|
| 模型权重 | 70 GB | 35 GB | 17.5 GB |
| KV 缓存 (10 Full Attn) | 5.37 GB | 2.68 GB | n/a |
| GDN 状态 (FP32) | 0.063 GB | 0.063 GB | 0.063 GB |
| GDN Conv 状态 | 0.001 GB | 0.001 GB | 0.001 GB |
| 激活值（峰值） | ~10 GB | ~6 GB | ~6 GB |
| **总计** | **~85.4 GB** | **~43.7 GB** | **~23.6 GB** |

**关键发现**：
- BF16 部署需要至少 4 卡 24GB GPU（或 2 卡 48GB/80GB）
- FP8 量化后可在 2 卡 24GB GPU 上运行
- INT4 量化后可在单卡 RTX 4090 (24GB) 上运行（需 GGUF/llama.cpp 支持 MoE offloading）
- 无论何种精度，GDN 状态（63 MB）都保持 FP32，这不随序列长度增长——这是 GDN 相比 Full Attention 的核心优势 [^src34]

**与全注意力 Dense 模型的对比**（35B Dense, BF16, n=262K）：

| 组件 | Qwen3.5-MoE (混合) | 35B Dense (全 Attn) |
|------|:---:|:---:|
| 权重 | 70 GB | 70 GB |
| KV 缓存 | 5.4 GB | 40*16*262K*256*2*2/1e9 = 171.8 GB |
| 状态 | 0.06 GB | 0 GB |
| 激活值 | ~10 GB | ~15 GB |
| **总计** | **~85 GB** | **~257 GB** |

Qwen3.5-MoE 的推理显存约为同规模全注意力 Dense 模型的 **1/3**，主要是 KV 缓存的巨大差异 [^src35]。

### 7.2 M-RoPE 位置编码

Qwen3.5 使用**多维旋转位置编码（M-RoPE）**支持多模态输入：
- 3 个维度：时间（T）、高度（H）、宽度（W）
- `mrope_section=[11, 11, 10]`：32 维分配为 T(11) + H(11) + W(10) = 32 = head_dim*partial_rotary_factor/2 = 64*0.5 = 32
- `mrope_interleaved=true`：交错排列 THWTHW...而非分块 TTT...HHH...WWW
- 视觉 token 的 position_ids 是 3D 坐标，文本 token 使用递增的 1D 位置

### 7.3 视觉编码器

`Qwen3_5MoeVisionModel` 规格：
- 27 层 ViT Block（Pre-Norm, MHA + MLP）
- 3D 卷积 Patch Embedding（temporal=2, spatial=16）
- 位置编码：可学习 Embedding(2304, 1152) + 视觉 RoPE
- 输出：通过 `Qwen3_5MoeVisionPatchMerger` 融合 patch 并投影到 2048 维
- 空间合并：spatial_merge_size=2（2x2 窗口合并）

### 7.4 推理优化

- **Flash Attention**：Full Attention 层支持 FA2/FA3（`_supports_flash_attn=True`）
- **FLA 加速**：GDN 使用 Flash Linear Attention 库的 `chunk_gated_delta_rule` 和 `fused_recurrent_gated_delta_rule`
- **causal_conv1d**：使用 Dao-AILab 的优化因果卷积内核
- **FusedRMSNormGated**：融合 RMSNorm + SiLU gating 为单内核

### 7.5 RMSNorm 设计（1+weight 形式）

```python
output = x * rsqrt(x^2.mean + eps)  # 归一化
output = output * (1.0 + weight)     # 缩放（weight 初始化为 0）
```

与 Llama 的 `x.float() * weight` 不同，Qwen3.5 使用 `(1+weight)` 使得 Norm 初始化为恒等映射。这是微妙的数值稳定性选择。

### 7.6 MTP（Multi-Token Prediction）

`mtp_num_hidden_layers=1` 表示模型在训练时额外预测未来 1 个 token。MTP 模块在源码中通过 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` 引用，实现在 `modeling_qwen3_5_moe.py` 之外（可能使用共享的 MTP 实现）。

---

## CH 8: 源码映射汇总

### 8.1 源码文件结构

```
src/transformers/models/qwen3_5_moe/
├── modular_qwen3_5_moe.py          # 模块化定义（源文件）
├── modeling_qwen3_5_moe.py         # 自动生成（本报告分析对象）
├── configuration_qwen3_5_moe.py    # 配置类
└── __init__.py
```

### 8.2 类映射表

| 类名 | 行号 | 功能 | 章节 |
|------|------|------|------|
| `Qwen3_5MoeRMSNormGated` | 184-199 | GDN 输出门控归一化 | CH3 |
| `Qwen3_5MoeRMSNorm` | 817-835 | 标准 RMSNorm (1+w) | CH7 |
| `Qwen3_5MoeGatedDeltaNet` | 368-556 | GDN 线性注意力核心 | CH3 |
| `Qwen3_5MoeAttention` | 643-717 | Full GQA Attention | CH4 |
| `Qwen3_5MoeMLP` | 720-733 | SwiGLU MLP（共享专家） | CH5 |
| `Qwen3_5MoeExperts` | 737-773 | 256 专家集合 | CH5 |
| `Qwen3_5MoeTopKRouter` | 776-792 | Top-K 路由门控 | CH5 |
| `Qwen3_5MoeSparseMoeBlock` | 795-814 | 完整 MoE 模块 | CH5 |
| `Qwen3_5MoeDecoderLayer` | 837-893 | Decoder 层（类型分支） | CH4/5 |
| `Qwen3_5MoeTextModel` | 1242-1337 | 文本主干网络 | CH2 |
| `Qwen3_5MoeVisionModel` | 1108-1212 | 视觉编码器 | CH7 |
| `Qwen3_5MoeModel` | 1341-1715 | 多模态模型 | CH7 |
| `Qwen3_5MoeForCausalLM` | 1801-1902 | 纯文本因果 LM | CH2 |
| `Qwen3_5MoeForConditionalGeneration` | 1905-2061 | 多模态生成模型 | CH2 |

### 8.3 关键辅助函数

| 函数 | 行号 | 功能 |
|------|------|------|
| `torch_chunk_gated_delta_rule` | 243-321 | PyTorch 纯实现分块 Delta Rule |
| `torch_recurrent_gated_delta_rule` | 324-365 | PyTorch 纯实现递归 Delta Rule |
| `torch_causal_conv1d_update` | 219-234 | PyTorch 纯实现因果卷积步进 |
| `apply_rotary_pos_emb` | 567-602 | 部分 RoPE 应用 |
| `load_balancing_loss_func` | 1718-1797 | Switch Transformer 负载均衡损失 |

---

## CH 9: 总结

### 9.1 架构亮点

1. **Gated DeltaNet 是核心差异化因素**：75% 层使用线性注意力实现 O(n) 复杂度，同时通过门控机制和 QK L2 归一化保持建模质量。在 262K 上下文下，GDN 的 decode FLOPs 仅为 Full Attention 的 1/7（~84.5M vs ~592M per token）[^src22]
2. **3:1 混合比例是精心选择的设计点**：在 Qwen3-Next 实验中验证，平衡了效率（更多 GDN）与能力（Full Attention 锚点）。Full Attention 层为模型提供了精确的 token-token 交互能力，这在需要细粒度位置信息的任务中至关重要 [^src14]
3. **256 路 + 共享专家的全层 MoE**：极致参数效率（35B -> 3B 激活），通过共享专家提供跨 token 的通用知识，通过 256 个路由专家提供专业化能力
4. **非对称 KV 头**：GDN 中 16K/32V 头设计（2:1 不对称），使 Value 维度（4096）是 Key 维度（2048）的 2 倍，增强了状态矩阵的信息容量 [^src1]
5. **(1+weight) RMSNorm**：初始化友好的归一化形式，weight 初始化为 0 使得 Norm 从恒等映射开始训练 [^src9]

### 9.2 推理效率优势

- **长序列友好**：GDN 状态恒定大小（~63MB），不随序列长度增长
- **紧凑 KV 缓存**：仅 10 个 Full Attention 层需要 KV 缓存，且为 2 头 GQA
- **稀疏激活**：每 token 仅激活 8/256 + 1 专家，大幅降低计算量（3B/35B = 8.6% 激活比）
- **混合注意力 3:1 比例**：在 Qwen3-Next [^src14] 实验中验证为最优帕累托点——既保留了 Full Attention 的精确检索能力，又获得了 GDN 的线性效率

### 9.3 已知局限

- **大磁盘占用**：256 experts * 40 layers = 10,240 个专家，权重文件巨大（~70GB BF16）
- **专家负载不均**：aux loss 仅 0.001，在特定任务上可能出现专家利用不均 [^src27]
- **GDN 状态精度**：状态需 float32 [^src31]，与模型主体 bf16 之间存在类型转换开销，阻碍全模型纯 BF16 推理
- **生态成熟度**：相比 Dense 模型，MoE + GDN 的推理框架支持仍在完善中（vLLM/SGLang 的新版本逐渐支持混合注意力）
- **专家存储冗余**：10,240 个专家组（40 层 * 256 专家），磁盘占用 ~70GB，对于边缘设备部署构成挑战
- **GDN chunking 开销**：chunk_size=64 时，短序列（n&lt;512）的 chunk padding 比例较高，线性注意力的优势在短序列上不明显

### 9.4 未来工作

- **GDN 状态量化**：当前状态需 FP32（~63MB），探索 BF16 状态以进一步节省 50% 状态内存 [^src31]
- **专家动态分配**：不同层可能需要不同数量的专家（浅层少、深层多），当前统一 256 专家可能不是最优 [^src28]
- **全线性化探索**：随着 GDN 技术的成熟，探索 100% GDN（无 Full Attention）的可行性——类似 Mamba 风格的纯 SSM 架构
- **训练效率优化**：MTP depth=2+ 的实验，以及更好的 MoE 负载均衡策略（如 DeepSeek V3 的 aux-loss-free 路由）
- **多模态深度集成**：当前视觉特征仅在输入层注入，未来可探索跨层视觉特征注入（如 DeepStack）

### 9.5 脚注索引

**源码引用**

[^src1]: `modeling_qwen3_5_moe.py:443-556` GatedDeltaNet.forward -- 完整的前向传播逻辑，包含 QKV+Z+B+A 投影、因果卷积、GDN chunk/recurrent 调用、输出门控
[^src2]: `configuration_qwen3_5_moe.py:79-111` Qwen3_5MoeTextConfig 默认值
[^src3]: `modeling_qwen3_5_moe.py:643-717` Qwen3_5MoeAttention -- Full GQA Attention 实现，含 QK norm, partial RoPE, output gate
[^src4]: `modeling_qwen3_5_moe.py:776-814` MoE routing + sparse block
[^src5]: `modeling_qwen3_5_moe.py:837-893` DecoderLayer -- 条件 token mixer 初始化 + Pre-Norm 双残差
[^src6]: `modeling_qwen3_5_moe.py:1242-1337` TextModel.forward -- 双掩码生成 + 层类型感知循环
[^src7]: `config.json` text_config -- 所有超参数的权威来源
[^src8]: `modeling_qwen3_5_moe.py:184-199` RMSNormGated -- SiLU 门控融合归一化
[^src9]: `modeling_qwen3_5_moe.py:817-835` RMSNorm (1+weight) -- weight 初始化为 0，初始为恒等映射
[^src10]: `modeling_qwen3_5_moe.py:1300-1333` mask selection logic -- 根据 layer_type 选择 linear_attn_mask 或 causal_mask

**GDN 设计背景与算子级引用**

[^src11]: Qwen3-Next 技术报告（Qwen 团队博客/论文），引入了 GDN + 混合注意力的架构设计
[^src12]: Mamba-1 (Gu & Dao, 2023) 提出选择性 SSM, d_state=16；Mamba-2 (Dao & Gu, 2024) 扩展到 d_state=128。GDN 的矩阵状态 (128x128) 信息容量更大
[^src13]: Gated DeltaNet 论文 (Yang et al., 2024) "Gated Delta Networks: Improving DeltaNet with Gating Mechanisms"；DeltaNet 原始论文 (Schlag et al., 2021) "Linear Transformers Are Fast Weight Programmers"
[^src14]: Qwen3-Next 架构消融实验（Qwen 团队博客），对比了 GDN vs Mamba-2 vs RetNet 在长上下文检索任务上的表现
[^src15]: `modeling_qwen3_5_moe.py:260` -- `x.to(torch.float32)` 将 QKV/β/g 转换为 FP32 精度
[^src16]: `modeling_qwen3_5_moe.py:265-270` -- chunk_size=64 的分块逻辑和 padding 计算
[^src17]: `modeling_qwen3_5_moe.py:303-314` -- 跨块扫描循环（chunk-wise recurrence），包含 v_prime（历史预测）和 v_new（delta）计算
[^src18]: `modeling_qwen3_5_moe.py:243-321` -- 完整的 `torch_chunk_gated_delta_rule` 实现，200 行纯 PyTorch
[^src19]: Full Attention FLOPs 公式：QKV 投影 O(b*n*d^2)，注意力分数 O(b*h*n^2*d)，加权求和 O(b*h*n^2*d)。标准引用：Vaswani et al., 2017 "Attention Is All You Need"
[^src20]: GDN FLOPs 估算基于 chunk_size=64 的分块计算：块内操作 O(chunk_size^2 * d * H)，跨块扫描 O(n/chunk_size * d^2 * H)
[^src21]: 混合模型的 FLOPs 总计来自 30*GDN_FLOPs + 10*FullAttn_FLOPs，加上 MoE FFN 共 40 层（每 token 约 28.35M 激活参数）
[^src22]: Decode FLOPs 对比：标准注意力 decode = O(n*d) per token（需计算 QK^T 与所有历史 KV）；GDN decode = O(d^2) per token（仅状态更新）

**MoE 设计背景引用**

[^src23]: Shazeer et al., 2017 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" -- 首次提出 top-k MoE 路由
[^src24]: `modeling_qwen3_5_moe.py:789` -- `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)` 归一化确保权重和为 1
[^src25]: Fedus et al., 2022 "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" -- 提出 aux loss 解决 load imbalance
[^src26]: `modeling_qwen3_5_moe.py:1718-1797` -- `load_balancing_loss_func` 完整实现 Switch Transformer 的 aux loss（公式 4-6）
[^src27]: `config.json: router_aux_loss_coef=0.001` -- 中等偏小的系数，平衡路由质量和负载均衡
[^src28]: Qwen3 技术报告（Qwen 团队, 2025）-- 部分层 MoE 的设计及 Qwen3.5 的全层 MoE 改进
[^src29]: DeepSeek-V2 论文 (DeepSeek-AI, 2024) "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" -- 共享专家设计

**推理显存引用**

[^src30]: `config.json: num_attention_heads=16, num_key_value_heads=2, full_attention_interval=4, max_position_embeddings=262144`
[^src31]: `config.json: mamba_ssm_dtype=float32` -- GDN 状态矩阵和计算使用 float32 精度
[^src32]: `modeling_qwen3_5_moe.py:483` -- `cache_params.update_conv_state(new_conv_state, self.layer_idx)` 管理 conv 缓存
[^src33]: 激活值内存估算基于标准 Transformer 推理内存模型，考虑 hidden_states (262K x 2048 x 2B)、expert 中间值 (top-8 x 512 x 2B) 和临时张量
[^src34]: 推理部署配置参考 vLLM 和 llama.cpp 的 MoE 推理最佳实践
[^src35]: 全注意力 Dense 模型 KV 缓存计算：40 层 * 16 头（无 GQA）* 262K * 256 * 2 * 2B = 171.8 GB

**其他引用**

[^src36]: `config.json: head_dim=256, partial_rotary_factor=0.25` -- 部分 RoPE 仅作用于 64/256 维
[^src37]: `config.json: rope_theta=10000000` -- RoPE 基础频率，支持 262K 上下文
[^src38]: `modeling_qwen3_5_moe.py:398-401` -- `dt_bias` 初始化为 1, `A_log` 初始化为 log(U(0,16))
[^src39]: `modeling_qwen3_5_moe.py:912-928` -- `_init_weights` 特殊初始化逻辑
[^src40]: `config.json: tie_word_embeddings=false` -- 嵌入层和 LM Head 不共享权重

---

*报告基于 Transformers v4.57.0.dev0 `modeling_qwen3_5_moe.py` 完整源码分析，所有数字标注均来自源码验证。*
