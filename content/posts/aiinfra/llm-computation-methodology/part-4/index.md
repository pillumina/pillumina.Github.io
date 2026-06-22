+++
date = '2026-06-15'
draft = false
title = 'LLM 计算方法论（四）：M3 实战推演与 Roofline 模型'
categories = ['aiinfra']
tags = ['computation', 'flops', 'kv-cache', 'roofline', 'inference', 'methodology', 'm3']
series = 'llm-computation-methodology'
series_order = 4
math = true
summary = 'MiniMax M3 完整推演：从 config.json 到参数量、FLOPs、KV Cache、推理显存的全链路计算。Roofline 模型分析推理延迟，理解 FP8/INT4 量化的性能收益。'
+++

## CH 6 | 实战——MiniMax M3 完整推演

以 MiniMax M3 为目标，从 config.json 出发，完整推演参数分解 → FLOPs 估算 → KV Cache → 推理显存 → 部署方案，覆盖 GQA + MSA + MoE + Vision + MTP 五种架构变体的计算。M3 是目前覆盖计算变体最多的开源模型——一个模型练完基本上所有架构你都会算了。

> **系列导航**：[（一）预备知识与参数分解](../part-1/) ← [（二）FLOPs 估算](../part-2/) ← [（三）KV Cache 与推理显存](../part-3/) ← 当前

> **本章使用的前置知识**（如果你是跳读的，这些概念在这里能找到定义）：
> - `FLOPs = 2×m×n×k` 及 MAC 概念 → [CH 1.2](#12-矩阵乘法-flops-是怎么算的)
> - GQA 中 K/V 投影「变窄」→ [CH 2.3.2](#232-gqagrouped-query-attention)（⚠️ 注意 `H_kv × D_h ≠ d`）
> - SwiGLU 的 3 矩阵结构（gate/up/down）→ [CH 2.4.2](#242-swiglu标准-moe-专家)
> - 激活参 vs 总参（MoE 中每个 token 只激活 top-k 专家）→ [CH 2.9](#29-激活参-vs-总参)
> - MSA 的 Index Branch 机制 → [CH 3.3](#33-msa-稀疏-attention-flops)
> - MLA/标准 KV cache 公式 → [CH 4.2](#42-标准-mhagqa-的-kv-cache) 或 [CH 4.3](#43-mla-的-kv-cache)

### 6.1 从 config.json 出发

打开 `MiniMax-M3` 的 `config.json`，提取以下核心字段（`text_config` 为主，`vision_config` 为辅）：

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 6144 | 残差流维度 $d$ |
| `num_hidden_layers` | 60 | 总层数 $L$ |
| `num_attention_heads` | 64 | Q 头数 $H_q$ |
| `num_key_value_heads` | 4 | KV 头数 $H_{kv}$ |
| `head_dim` | 128 | 每头维度 $D_h$ |
| `vocab_size` | 200,064 | 词表大小 $V$ |
| `rope_theta` | 5,000,000 | RoPE 基频 |
| `partial_rotary_factor` | 0.5 | rotary_dim = 0.5 × 128 = 64 |
| `num_local_experts` | 128 | 路由专家数 $E$ |
| `num_experts_per_tok` | 4 | 每 token 激活专家 $k$ |
| `n_shared_experts` | 1 | 共享专家 |
| `intermediate_size` | 3072 | MoE 专家中间维 $d_{moe\_ff}$ |
| `dense_intermediate_size` | 12288 | Dense FFN 中间维（前 3 层） |
| `shared_intermediate_size` | 3072 | 共享专家中间维 |
| `scoring_func` | sigmoid | 路由评分函数 |
| `sparse_block_size` | 128 | MSA block 大小 |
| `sparse_topk_blocks` | 16 | 每 query 选择 top-k blocks |
| `sparse_num_index_heads` | 4 | Index heads 数 |
| `sparse_index_dim` | 128 | Index head_dim |
| `sparse_disable_index_value` | [0,0,0,1,...1] | 层 0-2: Full Attn, 层 3-59: MSA |
| `moe_layer_freq` | [0,0,0,1,...1] | 层 0-2: Dense FFN, 层 3-59: MoE |
| `vision_config.hidden_size` | 1280 | ViT 隐藏维度 |
| `vision_config.num_hidden_layers` | 32 | ViT 层数 |
| `vision_config.num_attention_heads` | 16 | ViT 头数 |
| `vision_config.patch_size` | 14 | Patch 大小 |
| `vision_config.image_size` | 2016 | 输入图像尺寸 |
| `num_mtp_modules` | 7 | MTP 模块数 |
| `max_position_embeddings` | 1,048,576 | 最大上下文 1M |

**层类型分配**：

| 层范围 | Attention 类型 | FFN 类型 | 层数 |
|---|---|---|---|
| 0-2 | Full Attention (GQA 16:1) | Dense FFN (SwiGLU-OAI, $d_{ff}=12288$) | 3 |
| 3-59 | MSA Sparse Attention | MoE (128E, top-4, sigmoid) | 57 |

### 6.2 参数分解

以下按模块逐一计算，所有数值均从 6.1 节的 config.json 字段推导。

#### Embedding 层

$$N_{\text{embed}} = V \times d = 200{,}064 \times 6144 = 1{,}229{,}193{,}216 \approx \mathbf{1.229\text{B}}$$

`tie_word_embeddings=false` → 输入 Embedding + 输出 LM Head 各一份：

$$N_{\text{embed+head}} = 2 \times 1.229\text{B} = \mathbf{2.458\text{B}}$$

#### Attention 模块（per layer, Full Attn / MSA 共享）

Q 投影：$d \times H_q \times D_h = 6144 \times 64 \times 128 = 50{,}331{,}648 \approx 50.3\text{M}$
K 投影：$d \times H_{kv} \times D_h = 6144 \times 4 \times 128 = 3{,}145{,}728 \approx 3.1\text{M}$
V 投影：$d \times H_{kv} \times D_h = 3{,}145{,}728 \approx 3.1\text{M}$
O 投影：$H_q \times D_h \times d = 64 \times 128 \times 6144 = 50{,}331{,}648 \approx 50.3\text{M}$

**Per-layer Q/K/V/O 合计**：$\approx \mathbf{107.0\text{M}}$

#### Indexer（仅 MSA 层 3-59，57 层）

Index Q 投影：$d \times H_{\text{idx}} \times D_{\text{idx}} = 6144 \times 4 \times 128 = 3{,}145{,}728 \approx 3.1\text{M}$
Index K 投影：$d \times 1 \times D_{\text{idx}} = 6144 \times 128 = 786{,}432 \approx 0.79\text{M}$
Index QK Norm：$2 \times (4 \times 128) + 2 \times 128 = 1{,}280$（可忽略）

**Per-layer Indexer 合计**：$\approx 3.93\text{M}$

**Attention 总参**：

$$\begin{aligned}
N_{\text{attn}} &= 3 \times 107.0\text{M} \quad \text{(层 0-2: Full Attn)} \\
&+ 57 \times (107.0\text{M} + 3.93\text{M}) \quad \text{(层 3-59: MSA + Indexer)} \\
&= 321.0\text{M} + 6{,}323.0\text{M} = \mathbf{6.644\text{B}}
\end{aligned}$$

#### Dense FFN（层 0-2，SwiGLU-OAI，$d_{ff}=12288$）

Per layer（non-gated SwiGLU：gate_up 合并为 $6144 \to 2 \times 12288$）：

$$N_{\text{gate\_up}} = 6144 \times 2 \times 12288 = 150{,}994{,}944$$
$$N_{\text{down}} = 12288 \times 6144 = 75{,}497{,}472$$

Per-layer 合计：$\approx 226.5\text{M}$。3 层汇总：$\mathbf{0.679\text{B}}$。

#### MoE 模块（层 3-59，57 层）

**每个路由专家**（SwiGLU-OAI，$d_{ff}=3072$）：

$$N_{\text{expert}} = 6144 \times 2 \times 3072 + 3072 \times 6144 = 37{,}748{,}736 + 18{,}874{,}368 = 56{,}623{,}104 \approx 56.62\text{M}$$

**每层 128 个路由专家**：

$$N_{\text{experts\_per\_layer}} = 128 \times 56.62\text{M} = 7{,}247{,}757{,}312 \approx 7.25\text{B}$$

**共享专家**（per layer, 1 个）：

$$N_{\text{shared}} = 56.62\text{M} \quad (\text{维度与路由专家相同})$$

**路由器**（per layer）：

$$N_{\text{router}} = d \times E = 6144 \times 128 = 786{,}432 \approx 0.79\text{M}$$

**每层 MoE 合计**：$7.25\text{B} + 0.057\text{B} + 0.0008\text{B} \approx 7.31\text{B}$

**57 层 MoE 汇总**：$57 \times 7.31\text{B} = \mathbf{416.6\text{B}}$

#### Vision（ViT + Projector）

ViT 32 层（$d_{vit}=1280$, $H_{vit}=16$, $D_{vit}=80$, $d_{ff}^{vit}=5120$）：

Per-layer Attention：$4 \times (1280 \times 16 \times 80) = 6.55\text{M}$
Per-layer MLP：$2 \times 1280 \times 5120 = 13.11\text{M}$
32 层合计：$32 \times 19.66\text{M} \approx 0.63\text{B}$
加 patch embedding + Pre-LN + 3D RoPE：$\approx \mathbf{0.65\text{B}}$

Projector（双阶段 MLP）：

Stage 1：$1280 \times 6144 + 6144 \times 6144 \approx 45.6\text{M}$
Stage 2（spatial merge）：$(4 \times 6144) \times 6144 + 6144 \times 6144 \approx 188.7\text{M}$
合计：$\approx \mathbf{0.23\text{B}}$

**Vision 总计**：$\mathbf{0.88\text{B}}$

#### 汇总与自洽性验证

| 组件 | 参数量 (B) | 占比 |
|---|---|---|
| Embedding + LM Head | 2.458 | 0.58% |
| Attention (Q/K/V/O × 60) | 6.420 | 1.50% |
| Indexer (57 层 MSA) | 0.224 | 0.05% |
| Dense FFN (3 层) | 0.679 | 0.16% |
| MoE 路由专家 (128 × 57) | 413.25 | 96.7% |
| MoE 共享专家 | 3.227 | 0.76% |
| MoE 路由器 | 0.045 | 0.01% |
| Vision (ViT + Projector) | 0.880 | 0.21% |
| Norm 等 | ~0.001 | ~0% |
| **直接求和** | **~427.2** | 100% |
| **官方标称** | **~428B** | — |

偏差 < 0.2%，自洽性验证通过。

一个 428B 参数的模型，96.7% 的参数在 MoE 专家里。Attention 只有 6.4B（1.5%）——所以"优化 Attention"（GQA、MSA、MLA）主要是优化计算量和 KV Cache，而不是参数量。参数量的主战场永远是 FFN/MoE。

#### 激活参数

$$\begin{aligned}
N_{\text{active}} &= N_{\text{embed}} + N_{\text{attn}} + N_{\text{dense\_ffn}} + N_{\text{shared}} + k \times N_{\text{expert}} \times 57 + N_{\text{router}} + N_{\text{head}} \\
&= 1.23 + 6.64 + 0.68 + 3.23 + (4/128) \times 413.25 + 0.045 + 1.23 \\
&= 1.23 + 6.64 + 0.68 + 3.23 + 12.91 + 0.045 + 1.23 \\
&\approx \mathbf{26.0\text{B}}
\end{aligned}$$

加上 Vision 编码器（图像输入时激活 $\approx 0.88\text{B}$）：$\approx 26.9\text{B}$。

官方标称 $\sim 23\text{B}$。差异可能来源：(1) Vision 编码器在纯文本推理时不激活；(2) 部分参数共享（如 non-gated SwiGLU 中 gate/up 共享投影可视为半激活）。

$$\text{激活率} = \frac{26}{428} \approx \mathbf{6.1\%}$$

---

### 6.3 FLOPs 估算（Decode, T=1M）

计算 M3 在 1M 上下文下 decode 单个 token 的 FLOPs，并对 MSA 和 Full Attention 做定量对比。理解 MSA 到底省了多少计算——不是省了几个百分点，而是省了几个数量级（在 Attention 计算部分）。

以 **decode 阶段**（$T_{\text{new}} = 1$，$T_{\text{cached}} = 1{,}048{,}576$）为例，BF16 精度，统计 multiply-add 为 2 FLOPs。

#### 6.3.1 Full Attention 层（3 层，层 0-2）

**QK 点积**（decode 时 Q 只有 1 token，K 有 T cached）：

$$\begin{aligned}
\text{FLOPs}_{\text{QK}} &= 2 \times H_q \times D_h \times T \\
&= 2 \times 64 \times 128 \times 1{,}048{,}576 \\
&= 16{,}384 \times 1{,}048{,}576 = 1.718 \times 10^{10} \approx \mathbf{17.2 \text{ GFLOPs}}
\end{aligned}$$

**Attention-V 加权**：

$$\begin{aligned}
\text{FLOPs}_{\text{AttnV}} &= 2 \times H_q \times T \times D_h \\
&= 2 \times 64 \times 1{,}048{,}576 \times 128 = 17.2 \text{ GFLOPs}
\end{aligned}$$

**Per Full Attn layer decode FLOPs**：$17.2 + 17.2 = \mathbf{34.4 \text{ GFLOPs}}$

3 层合计：$3 \times 34.4 = \mathbf{103.1 \text{ GFLOPs}}$

在 1M 上下文中，即使 Q 只有 1 个新 token，QK 点积也要算 1M 次内积（每个 cached K 对新 Q 算一次相似度）。64 个 Q 头 × 128 维 × 1M tokens × 2 = 16.4B 次运算。这就是 Full Attention 在长上下文 decode 中的致命弱点——每生成一个新 token，要跟之前所有 token 做一次全量比较。

#### 6.3.2 MSA 层（57 层，层 3-59）

MSA 分为 Index Branch + Main Attention。

**Index Branch**：

$$\begin{aligned}
\text{FLOPs}_{\text{idx QK}} &= 2 \times H_{\text{idx}} \times D_{\text{idx}} \times T \\
&= 2 \times 4 \times 128 \times 1{,}048{,}576 = 1{,}024 \times 1{,}048{,}576 \\
&\approx \mathbf{1.074 \text{ GFLOPs}}
\end{aligned}$$

**Main Attention**（仅在 $K = \text{topk\_blocks} \times \text{block\_size} = 16 \times 128 = 2{,}048$ 个 token 上做精确 attention）：

$$\begin{aligned}
\text{FLOPs}_{\text{main QK}} &= 2 \times H_q \times D_h \times K \\
&= 2 \times 64 \times 128 \times 2048 = 33{,}554{,}432 \approx \mathbf{33.6 \text{ MFLOPs}} \\
\text{FLOPs}_{\text{main AttnV}} &= 2 \times H_q \times K \times D_h = 33.6 \text{ MFLOPs}
\end{aligned}$$

**Per MSA layer decode FLOPs**：$1{,}074 + 33.6 + 33.6 \approx \mathbf{1.14 \text{ GFLOPs}}$

57 层合计：$57 \times 1.14 = \mathbf{65.0 \text{ GFLOPs}}$

#### 6.3.3 QK 加速比：Full Attn vs MSA

**Attention QK 计算部分的加速比**（只比较 QK 点积，不含线性投影）：

$$\frac{\text{FLOPs}_{\text{QK}}^{\text{Full}}}{\text{FLOPs}_{\text{QK}}^{\text{MSA}}} = \frac{2 \times 64 \times 128 \times 1{,}048{,}576}{2 \times 64 \times 128 \times 2048} = \frac{1{,}048{,}576}{2{,}048} = \mathbf{512\times}$$

**单层总 Attention FLOPs 加速比**（含 Index Branch + Main Attention 的所有 attention 操作）：

$$\frac{34.4 \text{ G}}{1.14 \text{ G}} \approx \mathbf{30.2\times}$$

为什么 512× 变成了 30×？因为 MSA 的 Index Branch 自身也有 FLOPs（1.07 GFLOPs），而且这 1.07G 在层总 FLOPs 中占比不小（1.07/1.14 ≈ 94%）。Index Branch 仍然需要 O(T) 的 QK 计算——它的目的是**筛选** top-k blocks，而非跳过 QK 计算。

MSA 的 512× QK 加速是在 **Main Branch** 上实现的（2,048 vs 1M tokens），但 Index Branch 自身仍做 O(T) 扫描（只不过用了更少的 head：4 vs 64，所以也便宜了 16×）。总体效果约 30×，这意味着同样 1M 上下文，MSA 的 decode 比 Full Attention 快 30 倍——但仍然比短上下文（如 4K）的 Full Attention 要慢（因为 Index Branch 的 O(T) 扫描无法避免）。

#### 6.3.4 线性投影 FLOPs（60 层共享）

Q、K、V、O 四个线性投影（per layer）：

$$\begin{aligned}
\text{Q proj} &= 2 \times 1 \times 6144 \times (64 \times 128) = 2 \times 6144 \times 8192 = 100.7 \text{ MFLOPs} \\
\text{K proj} &= 2 \times 1 \times 6144 \times (4 \times 128) = 2 \times 6144 \times 512 = 6.3 \text{ MFLOPs} \\
\text{V proj} &= 2 \times 1 \times 6144 \times 512 = 6.3 \text{ MFLOPs} \\
\text{O proj} &= 2 \times 1 \times (64 \times 128) \times 6144 = 100.7 \text{ MFLOPs}
\end{aligned}$$

Per-layer 投影合计：$\approx 213.9 \text{ MFLOPs}$。60 层：$\mathbf{12.8 \text{ GFLOPs}}$。

#### 6.3.5 MoE FFN FLOPs（57 层，per token）

**共享专家**（intermediate=3072，SwiGLU-OAI）：

$$\begin{aligned}
\text{gate\_up} &= 2 \times 1 \times 6144 \times (2 \times 3072) = 2 \times 6144 \times 6144 = 75.5 \text{ MFLOPs} \\
\text{down} &= 2 \times 1 \times 3072 \times 6144 = 37.7 \text{ MFLOPs} \\
\text{shared total} &= 75.5 + 37.7 = \mathbf{113.2 \text{ MFLOPs}}
\end{aligned}$$

**4 个路由专家**：

$$\text{routed total} = 4 \times 113.2 = \mathbf{452.8 \text{ MFLOPs}}$$

**路由器**：$2 \times 1 \times 6144 \times 128 = \mathbf{1.6 \text{ MFLOPs}}$

**Per MoE layer decode FLOPs**：$113.2 + 452.8 + 1.6 = \mathbf{567.6 \text{ MFLOPs}}$

57 层 MoE：$57 \times 0.5676 = \mathbf{32.4 \text{ GFLOPs}}$

#### 6.3.6 Dense FFN FLOPs（3 层，per token）

Per layer（intermediate=12288）：

$$\begin{aligned}
\text{gate\_up} &= 2 \times 1 \times 6144 \times (2 \times 12288) = 2 \times 6144 \times 24576 = 302.0 \text{ MFLOPs} \\
\text{down} &= 2 \times 1 \times 12288 \times 6144 = 151.0 \text{ MFLOPs} \\
\text{per layer total} &= \mathbf{453.0 \text{ MFLOPs}}
\end{aligned}$$

3 层合计：$\mathbf{1.36 \text{ GFLOPs}}$

#### 6.3.7 MSA Indexer 投影 FLOPs（57 层）

Index Q 投影：$2 \times 1 \times 6144 \times (4 \times 128) = 6.3 \text{ MFLOPs}$
Index K 投影：$2 \times 1 \times 6144 \times 128 = 1.6 \text{ MFLOPs}$
Per-layer：$\approx 7.9 \text{ MFLOPs}$。57 层：$\mathbf{0.45 \text{ GFLOPs}}$

#### 6.3.8 全模型 Decode FLOPs 汇总

| 组件 | 层数 | Per-layer (GFLOPs) | 合计 (GFLOPs) |
|---|---|---|---|
| Full Attention (QK + AttnV) | 3 | 34.4 | 103.1 |
| MSA Attention (Idx + Main) | 57 | 1.14 | 65.0 |
| 线性投影 (Q/K/V/O) | 60 | 0.214 | 12.8 |
| Dense FFN | 3 | 0.453 | 1.36 |
| MoE FFN (shared + 4 routed) | 57 | 0.568 | 32.4 |
| Indexer 投影 | 57 | 0.008 | 0.45 |
| Embedding + LM Head | 1 | — | ~0.02 |
| **Total per decode token @1M** | | | **~215 GFLOPs** |

#### 6.3.9 与"全 Full Attention M3"对比

如果 M3 的 57 个 MSA 层全部替换为 Full Attention（保持所有其它参数不变）：

$$\begin{aligned}
\text{FLOPs}_{\text{Full-only QK+AttnV}} &= 103.1 + 57 \times 34.4 = 103.1 + 1{,}960.8 = \mathbf{2{,}064 \text{ GFLOPs}} \\
\text{FLOPs}_{\text{MSA (actual)}} &= 103.1 + 65.0 = \mathbf{168.1 \text{ GFLOPs}}
\end{aligned}$$

$$\text{Attention 计算加速比} = \frac{2{,}064}{168.1} \approx \mathbf{12.3\times}$$

若计算全模型 FLOPs（含投影 + FFN）：

$$\text{FLOPs}_{\text{Full-only total}} = 2{,}064 + 12.8 + 1.36 + 32.4 + 0.02 = 2{,}111 \text{ GFLOPs}$$

$$\text{Overall speedup} = \frac{2{,}111}{215} \approx \mathbf{9.8\times}$$

Attention 计算加速 12.3×，但因线性投影和 FFN 不变，总体加速约 10×。M3 花了 57 层 Indexer 的代价（+0.224B 参数，占总参 0.05%），换来了约 10 倍的 decode 速度提升。这是 MSA 被称为 "architectural free lunch" 的原因。

---

### 6.4 KV Cache（T=1M）

#### Main KV Cache（60 层）

$$\begin{aligned}
M_{\text{kv}}^{\text{main}} &= 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 128{,}849{,}018{,}880 \text{ bytes} \\
&\approx \mathbf{120.0 \text{ GiB}}
\end{aligned}$$

#### Index K Cache（57 层 MSA）

$$\begin{aligned}
M_{\text{kv}}^{\text{index}} &= 57 \times 1 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 15{,}300{,}329{,}472 \text{ bytes} \\
&\approx \mathbf{14.3 \text{ GiB}}
\end{aligned}$$

#### 总 KV Cache（per sample, BF16）

$$M_{\text{kv}}^{(1)} = 120.0 + 14.3 = \mathbf{134.3 \text{ GiB}}$$

**分项占比**：

```
KV Cache per sample @1M = 134.3 GiB
┌──────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████          │ Main KV (120.0 GiB, 89.4%)
│██████                                                      │ Index K (14.3 GiB, 10.6%)
└──────────────────────────────────────────────────────────┘
```

MSA 的 KV Cache 与 Full Attention 完全相同——稀疏性只体现在**计算**（哪些 KV 被访问），不体现在**存储**（所有 KV 仍需缓存，因为不同 query 可能选择不同 blocks）。Index K Cache 额外增加了约 10.6% 的 KV Cache 开销。这是 MSA 与 sliding window attention 的本质区别——后者可以裁剪 KV Cache，但 MSA 不能（理论上可以 evict 从未被任何 query 选中的 block，但这需要额外的 bookkeeping）。

#### Batch Scaling

| Batch Size | Main KV (GB) | Index KV (GB) | Total KV (GB) |
|---|---|---|---|
| 1 | 120.0 | 14.3 | 134.3 |
| 2 | 257.7 | 30.6 | 288.3 |
| 4 | 515.4 | 61.2 | 576.6 |
| 8 | 1,030.7 | 122.4 | 1,153.1 |

Batch=4 时 KV Cache 已超过 500 GiB——仅 KV Cache 就够塞满 4 张 H200。这是长上下文推理的核心瓶颈。

---

### 6.5 推理显存

#### BF16 精度，单样本，1M 上下文

$$\begin{aligned}
M_{\text{weights}} &= 428 \times 10^9 \times 2 = \mathbf{856 \text{ GB}} \\
M_{\text{kv}} &= \mathbf{134.3 \text{ GiB}} \\
M_{\text{act+overhead}} &\approx \mathbf{5 \text{ GiB}} \\
M_{\text{total}}^{(1)} &= 856 + 134.3 + 5 \approx \mathbf{995 \text{ GB/GiB}} \quad (\text{权重 GB + 其余 GiB，单位混合，近似值})
\end{aligned}$$

#### 硬件匹配

以 8 × H200（141 GiB/card，合计 1,128 GiB）为目标平台：

**Step 1：权重装得下吗？**

$$M_{\text{weights}} = 856 \text{ GB} < 1{,}128 \text{ GiB} \quad \checkmark$$

**Step 2：可用显存**

$$M_{\text{available}} = 1{,}128 - 856 = \mathbf{272 \text{ GiB}} \quad (\text{8 卡合计})$$

**Step 3：最大并发 batch**

$$B_{\text{max}} = \left\lfloor \frac{272}{134.3 + 5} \right\rfloor = \left\lfloor \frac{272}{139.3} \right\rfloor = \lfloor 1.82 \rfloor = \mathbf{1 \text{ sample}}$$

**结论**：BF16 下 8×H200 可以跑 M3 的 1M 上下文 BF16 推理，但只能支持最多 1 个并发请求。batch=2 理论上可能（$272 / 139.3 \approx 1.8$），但接近显存上限，实际部署中不建议。

#### FP8 权重 + BF16 KV Cache

$$\begin{aligned}
M_{\text{weights}} &= 428 \times 10^9 \times 1 = \mathbf{428 \text{ GB}} \\
M_{\text{available}} &= 1{,}128 - 428 = \mathbf{700 \text{ GiB}} \\
B_{\text{max}} &= \left\lfloor \frac{700}{139.3} \right\rfloor \approx \mathbf{4 \text{ samples}}
\end{aligned}$$

#### FP8 权重 + FP8 KV Cache

$$\begin{aligned}
M_{\text{kv}}^{(1)\text{ FP8}} &= 134.3 / 2 = \mathbf{67.15 \text{ GiB}} \\
M_{\text{available}} &= 1{,}128 - 428 = 700 \text{ GiB} \\
B_{\text{max}} &= \left\lfloor \frac{700}{67.15 + 5} \right\rfloor \approx \mathbf{9 \text{ samples}}
\end{aligned}$$

#### 汇总表

| 精度方案 | 权重 (GB) | KV/样本 (GB) | 可用 (GB) | Max Batch @1M |
|---|---|---|---|---|
| BF16 W + BF16 KV | 856 | 134.3 | 272 | 1 |
| FP8 W + BF16 KV | 428 | 134.3 | 700 | 4 |
| FP8 W + FP8 KV | 428 | 67.15 | 700 | 9 |
| INT4 W + FP8 KV | 214 | 67.15 | 914 | 12 |

对比 Nemotron（5.7 节）：Nemotron 从 BF16→FP8 后 batch 从 1→38，M3 只从 1→4。原因：M3 的 KV Cache 占比高（144 GiB/样本），量化权重解放的显存很快被 KV Cache 吃掉。M3 的显存瓶颈是**双重的**——权重和 KV Cache 都制约并发。

---

### 6.6 验算与交叉对比

#### 与 M3 官方博客声明对照

M3 官方博客声称 MSA 在 1M 上下文下实现 ~30× decode 加速。本节 6.3.3 的直接计算给出：

$$\text{Per-layer Attention FLOPs ratio} = \frac{34.4 \text{ G}}{1.14 \text{ G}} \approx 30.2\times$$

全模型（含投影+FFN）：$\approx 9.8\times$。

**差异解释**：官方的 30× 是指 Attention 计算部分（QK + AttnV），不含线性投影（Q/K/V/O）和 FFN。两者都是正确的——只是口径不同：
- 30×：Attention 算子层面（孤立地看 MSA 替代 Full Attention 的效果）
- 10×：端到端 decode 速度（含所有矩阵乘法和 FFN）

当别人说"MSA 让 M3 快了 30 倍"，他说的是注意力计算。当你说"为什么我实测只快了 10 倍"，因为你还算上了 FFN 和线性投影。两者都对，但需要明确口径。

#### 与纯 Full Attention M3 的显存对比

如果 M3 不使用 MSA（即全部 60 层 Full Attention），KV Cache 变化：

$$\begin{aligned}
M_{\text{kv}}^{\text{Full-only}} &= 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2 = 120.0 \text{ GiB} \\
M_{\text{kv}}^{\text{MSA (actual)}} &= 134.3 \text{ GiB}
\end{aligned}$$

KV Cache 不降反增（+14.3 GiB Index K）——MSA 在显存上不是优化，是略微增加了开销。MSA 的价值在**计算**（FLOPs），不在**存储**（Memory）。

#### 与 Nemotron 550B 的横向对比

| 维度 | Nemotron 550B | MiniMax M3 | 比值 |
|---|---|---|---|
| 总参 | 550B | 428B | 1.29× |
| BF16 权重 | 1,100 GB | 856 GB | 1.29× |
| KV Cache (1M, BF16) | 13 GiB | 144 GiB | **0.09×** |
| KV/Weights 比 | 1.2% | 16.8% | **14× 差异** |
| Decode FLOPs/T | ~300G (estimate) | ~215G | ~1.4× |
| 显存瓶颈类型 | **纯权重** | **权重 + KV Cache 双瓶颈** | — |
| FP8 后 Batch (8×H200) | 38 | 4 | **9.5× 差异** |

核心洞见：Nemotron 用 Mamba-2 置换 Attention 的策略，在 1M 上下文下产生了约 10× 的 KV Cache 优势。这个优势在短上下文（< 32K）下不明显（因为 KV Cache 本来就小），但随着上下文增长到 1M 时成为决定性的架构差异。MSA 解决了 Attention 的**计算**瓶颈，但没有解决**存储**瓶颈——在极端长上下文下，Mamba-2/MLA 的 KV Cache 优势会越来越明显。

---

## CH 7 | Roofline 模型与推理延迟 —— 「算力够，为什么还是慢」

CH 3 给出了每 token 的 FLOPs，CH 6 算出了需要多少张卡。现在回答第三个问题：**在这些卡上，每 token 实际要跑多少毫秒？**

一个最直观的错误做法：把 FLOPs 除以 GPU 峰值算力。

以 M3 BF16 decode（$T=1\text{M}$，8×H200，FP8 权重 + BF16 KV）为例：

$$\text{Naive latency} = \frac{215\text{G FLOPs/token}}{8 \times 989\text{ TFLOPS}} \approx 0.027\text{ ms}$$

如果真是 0.027 ms/token，那就是 37,000 tokens/s——没有任何 LLM 推理系统达到过这个速度。实际 M3 的 decode 延迟大约在 10-30 ms/token 量级。**差了 200-1000 倍（取决于精度和配置）。**

不是 FLOPs 算错了（前几章已经反复验证），而是**前提错了**：GPU 的峰值算力只有在**算术强度足够高**的 workload 上才能跑满。Decode 根本不具备这个条件。

### 7.1 算术强度：每次读一个字节的数据，能做多少 FLOPs

#### 定义

$$\text{Arithmetic Intensity (AI)} = \frac{\text{Total FLOPs}}{\text{Total Bytes Read from HBM}}$$

单位是 FLOPs/byte。AI 高 → 算力瓶颈（compute-bound），AI 低 → 带宽瓶颈（memory-bound）。

GPU 可以同时视为两个设备：
- **算力设备**（Tensor Core）：上限 = 峰值 FLOPs（如 H200 BF16 = 989 TFLOPS）
- **带宽设备**（HBM）：上限 = 内存带宽（如 H200 = 4.8 TB/s）

实际能达到的算力由两者共同决定：

$$\text{Achievable TFLOPS} = \min(\text{Peak TFLOPS}, \text{AI} \times \text{Bandwidth})$$

这就是 **Roofline 模型**的核心公式。画成图是一条折线：低 AI 时是斜率为带宽的斜线，高 AI 时是峰值算力的天花板。拐点叫 **Roofline Ridge**：

$$\text{AI}_{\text{ridge}} = \frac{\text{Peak TFLOPS}}{\text{Bandwidth}}$$

对 H200 BF16：$\text{AI}_{\text{ridge}} = \frac{989 \times 10^{12}}{4.8 \times 10^{12}} \approx 206 \text{ FLOPs/byte}$。

这意味着：任何 AI < 206 FLOPs/byte 的 workload 都跑不满 H200 的算力——它被带宽卡住了。AI 越低，利用率越惨。

#### 关键 GPU 的 Roofline 参数

| GPU | BF16 Peak | HBM BW | AI_ridge (BF16) | FP8 Peak | AI_ridge (FP8) |
|---|---|---|---|---|---|
| H100 SXM | 989 TFLOPS | 3.35 TB/s | **295** | 1,979 TFLOPS | **591** |
| H200 SXM | 989 TFLOPS | 4.8 TB/s | **206** | 1,979 TFLOPS | **412** |
| B200 | ~2,250 TFLOPS | 8 TB/s | **281** | ~4,500 TFLOPS | **562** |
| A100 SXM | 312 TFLOPS | 2.0 TB/s | **156** | — | — |

注意：H200 的算力与 H100 相同，但带宽高 43%。Ridge 从 295 降到 206——更容易进入 memory-bound 了。这是因为 H200 的"带宽提升"跑赢了"算力不变"。

### 7.2 Decode 为什么一定 memory-bound

#### 7.2.1 关键观察：每生成一个 token，要读一遍所有权重

Decode 是逐 token 生成的。每生成一个新 token，整个模型的所有层都要算一遍 forward pass。每次 forward pass 必须从 HBM 读取权重（假设不在 cache 中）。

对**密集模型**（如一个 70B Dense 模型，BF16）：

$$\text{AI}_{\text{decode}} \approx \frac{2N}{N \times \text{bytes\_per\_param}} = \frac{2}{\text{bytes\_per\_param}}$$

其中 $2N$ 是因为每个参数约对应 2 次 FLOPs（1 multiply + 1 add per MAC），$N \times \text{bytes\_per\_param}$ 是把所有权重读一遍的字节数。

代入 BF16：

$$\text{AI}_{\text{decode}} \approx \frac{2}{2} = 1 \text{ FLOPs/byte}$$

**1 FLOPs/byte。** 对比 H200 的 ridge = 206。差了 200 倍。Decoder-only 模型的 decode 阶段**天生 memory-bound**，这不是实现问题，是架构决定的。

更精确地算，需要考虑 Attention 部分（QK 点积要读 KV cache），但对大模型来说 FFN 权重占绝对主导，AI ≈ 1 是很好的近似。

#### 7.2.2 MoE 稀疏化的帮助

MoE 改变了这个情况。每 token 只激活 $k$ 个专家，所以需要读的权重少了：

$$\text{AI}_{\text{decode}}^{\text{MoE}} \approx \frac{2N_{\text{active}}}{N_{\text{active}} \times \text{bytes\_per\_param}} = \frac{2}{\text{bytes\_per\_param}}$$

等等——公式和密集模型一样？是的，因为活跃 FLOPs 和活跃字节数同比缩放。**MoE 不改变每活跃参数的 Arithmetic Intensity。**

但 MoE 改变了什么？**改变了总计算量和延迟的绝对值。**因为活跃参是总参的 ~5-10%，所以虽然 AI 还是 ~1，但 workload 变小了——读 23B 参的权重比读 428B 参的权重快 18.6×。

换一个角度：**MoE 的密集等价模型**（Dense model with same FLOPs）的 AI 相同，延迟也相同。MoE 用更少的活跃参达到同等的模型能力，从而用更小的 HBM 读取量实现了更低的延迟。

#### 7.2.3 Prefill 为什么不同

Prefill（首次处理整个 prompt）的 FLOPs 包含 Attention 的 $O(T^2)$ 项：

$$\text{AI}_{\text{prefill}} \approx \frac{2N + O(T^2 \cdot H \cdot D)}{N \times \text{bytes\_per\_param}}$$

当 $T$ 较大时（对 70B 模型约 $T > 100\text{K}$，具体 crossover 点取决于 $N/(H_q \cdot D_h \cdot L)$），Attention QK 点积的 FLOPs（$O(T^2)$）超过线性投影的 FLOPs（$O(T)$）。此时 FLOPs 急剧增长但权重读取不变，AI 飙升，**prefill 会从 memory-bound 转为 compute-bound**。

这就是为什么 prefill 的 MFU 可以到 40-50%，而 decode 的 MFU 通常只有 1-5%——不是 decode 写得差，是 decode 的算术强度决定了它只能在带宽限制内跑。

### 7.3 从 AI 到延迟——三步推导法

给定一个模型和一个 GPU，推导 decode 延迟只需三步。

#### 第一步：算出一个 token 要读多少字节

$$\text{Bytes}_{\text{HBM}} = \underbrace{N_{\text{active}} \times \text{bytes\_per\_param}}_{\text{权重}} + \underbrace{\text{KV\_bytes\_per\_token}}_{\text{KV Cache 读取}}$$

对密集模型，$N_{\text{active}} = N$。对 MoE，$N_{\text{active}}$ = 非 MoE 参数 + $k \times$ 每专家参数。

KV Cache 读取是 attention 计算的一部分——需要把缓存的所有 K 和 V 读出来做点积。对 GQA，全序列 attention 的 KV 读取量 = $H_{kv} \times D \times T \times \text{bytes\_per\_elem} \times 2$（K+V）。当 $T$ 很大时，这一项不可忽略。

#### 第二步：用 Roofline 算 achievable TFLOPS

$$\text{Achievable TFLOPS} = \min\left(\text{Peak TFLOPS}, \frac{\text{FLOPs}}{\text{Bytes}_{\text{HBM}}} \times \text{Bandwidth}\right)$$

对 decode，由于 AI ≈ 1–5 << 206，一定取右边（memory-bound）：

$$\text{Achievable TFLOPS} = \text{AI} \times \text{Bandwidth}$$

#### 第三步：算延迟下限

$$T_{\text{per\_token}} \geq \frac{\text{FLOPs\_per\_token}}{\text{Achievable TFLOPS}} = \frac{\text{Bytes}_{\text{HBM}}}{\text{Bandwidth}}$$

最后一个等号在 memory-bound 时成立。**直觉**：memory-bound 时延迟不由 FLOPs 决定，而由"读完所有权重要多久"决定。

### 7.4 案例：70B Dense 模型 on 1×H200（BF16）

这是最简单的情况，用来建立直觉。

**模型**：Llama-3 70B 级别，$d = 8192$，$L = 80$，GQA 8:1。

**每 token 权重读取**：$70 \times 10^9 \times 2 = 140 \text{ GB}$

**每 token FLOPs**：$\approx 2 \times 70 \times 10^9 = 140 \text{ GFLOPs}$

**KV Cache 读取**（per layer，$T = 128\text{K} = 131,072$）：

$$\text{KV\_read} = H_{kv} \times D \times T \times 2 \times 2 \text{ (K + V, BF16)} = 8 \times 128 \times 131{,}072 \times 2 \times 2 \approx 5.4 \times 10^8 = 0.54 \text{ GB}$$

对比：每层权重读取 $\approx 70\text{B} / 80 \times 2 \approx 1.75 \text{ GB}$。每层 KV 读取约为权重读取的 30%——不算完全可忽略，但为简化推导我们先用权重主导近似（$\text{AI} \approx 1$），更精确计算（$\text{AI} \approx 0.76$）不改变 memory-bound 结论。

**算术强度**：

$$\text{AI} = \frac{140 \times 10^9}{140 \times 10^9} = 1.0 \text{ FLOPs/byte}$$

**H200 Roofline**：AI = 1 ≪ 206 → memory-bound。

$$\text{Achievable TFLOPS} = 1.0 \times 4.8 \times 10^{12} = 4.8 \text{ TFLOPS}$$

**实际 MFU**：$4.8 / 989 = 0.49\%$。不到 1%。

**延迟**：

$$T_{\text{per\_token}} \approx \frac{140 \times 10^9}{4.8 \times 10^{12}} \approx 29 \text{ ms}$$

对应 34 tokens/s。如果 naive 地用 FLOPs / peak：$140\text{G} / 989\text{T} ≈ 0.14\text{ ms}$——**差了 200×。**

#### 为什么实际可能比 29ms 略慢

- 权重可能读不到完整的 4.8 TB/s 带宽（实际 ~70-80%）
- 存在 kernel launch overhead
- Attention 计算本身有额外的 HBM 访问模式不连续（GQA gather）
- 实际项目中还会有采样、KV cache 管理等开销

实测 Llama-3 70B 在单卡 H200 上的 decode 延迟约为 35-45 ms/token，与我们的理论下限 29 ms 在同一量级。

### 7.5 案例：M3 MoE BF16 Decode on 8×H200

现在用 Roofline 分析 CH 6 推演过的 M3。

**活跃参数**（CH 2）：~23B/ token。BF16 下 = 46 GB 权重读取。

**FLOPs**（CH 6）：~215 GFLOPs/token。

**KV Cache 读取**（$T=1\text{M}$）：MSA 层只选 2048 个 token 的 KV（$57 \times 4 \times 128 \times 2048 \times 2 \times 2 \approx 0.24$ GB），全 attention 层（3 层）需读全部 $T$（$3 \times 4 \times 128 \times 1\text{M} \times 2 \times 2 \approx 6.4$ GB）。合计约 6.7 GB/token，占活跃权重读取（46 GB）的 ~15%。属于次要项，为简化先忽略。

**算术强度**：

$$\text{AI} = \frac{215 \times 10^9}{46 \times 10^9} \approx 4.7 \text{ FLOPs/byte}$$

4.7 ≪ 206，仍然 memory-bound。

**单卡 achievable TFLOPS**：$4.7 \times 4.8 \times 10^{12} ≈ 22.4 \text{ TFLOPS}$。

**但这里是多卡。** 8×H200 共享权重的读取——专家分布在 8 张卡上（EP=8），每张卡只持有部分专家。对于单个 token，该 token 激活的 4 个专家可能分布在 4 张不同的卡上，每张卡读取的权重约为：

$$\text{Bytes}_{\text{per\_card}} \approx \frac{46}{4} \approx 11.5 \text{ GB}$$

$$\text{单卡延迟} \approx \frac{11.5 \times 10^9}{4.8 \times 10^{12}} \approx 2.4 \text{ ms}$$

**但**，EP 引入了 all-to-all 通信——卡之间需要 shuffle token 到持有对应专家的卡。这个通信开销（Gap 3 详述）可能使实际延迟翻倍。初步估计 decode 延迟在 **5-15 ms** 量级，与社区报告的 MoE 推理延迟一致。

关键是：**如果 naive 地用 215G / (8 × 989T) 算，得到 0.027 ms，偏差 > 200×。** FLOPs 算对了，但没算"这些 FLOPs 在什么条件下执行"。

### 7.6 FP8/INT4 量化对延迟的影响

量化（FP8、INT4）减少的是**每个参数的字节数**，直接提升 AI。

| 精度 | bytes/param | Dense AI | M3 MoE AI | Bandwidth-bound time |
|---|---|---|---|---|
| BF16 | 2 | 1.0 | 4.7 | baseline |
| FP8 | 1 | 2.0 | 9.3 | **2× 提升** |
| INT4 | 0.5 | 4.0 | 18.7 | **4× 提升** |

但要注意：AI 提升只在 memory-bound 阶段有效。如果 AI 达到 ridge（对 H200 BF16 是 206），再降精度也不会提速——此时已进入 compute-bound。对 decode，即使 INT4 的 AI = 18.7 也远未到 206，所以量化对 decode **全程有效**。

这就是为什么 FP8 推理能让 decode 延迟接近减半：不是算得快了（算力没变），而是**读得少了**（每参数 1 byte vs 2 bytes），带宽负担减半。

### 7.7 Batch Size 对 Decode 延迟的影响

当 batch size $B > 1$ 时，权重读取被 $B$ 个样本共享（权重读完一次，可以给 $B$ 个 token 分别算）：

$$\text{AI}(B) \approx \frac{B \times 2N_{\text{active}} + B \times \text{Attn\_FLOPs}}{N_{\text{active}} \times \text{bytes\_per\_param} + B \times \text{KV\_read\_per\_token}}$$

- **分子**：$B$ 个 token 的总 FLOPs（线性投影随 $B$ 线性增长）
- **分母**：权重只读一次，KV cache 每个 token 都要读

关键特性：**权重读取不随 batch 增长（同一个权重矩阵用 $B$ 次），KV cache 读取随 batch 增长**。

当 $B$ 较小时（$B=1-4$，推理常见场景），权重读取占主导，$B$ 增大带来显著的 AI 提升：

$$\text{B=1: AI ≈ 4.7, B=4: AI ≈ 4 × 4.7 = 18.8}$$

当 $B$ 足够大时，KV cache 读取成为新的瓶颈（因为它随 $B$ 线性增长）。但即使 $B=128$，对 M3 来说 AI ≈ 50-80，仍低于 H200 ridge = 206——仍在 memory-bound。

**工程启示**：在显存允许范围内，增大 batch size 能提升吞吐（tokens/s），但对单个 token 的首 token 延迟（TTFT）几乎无帮助——那是 prefill 的问题，而 prefill 通常已是 compute-bound。

### 7.8 三个模型 Decode 延迟对比

用 Roofline 方法估算几个代表性模型在 $T=1\text{M}$、BF16 下的 decode 延迟（不考虑通信开销的理论下界）：

| 模型 | 活跃参 | 活跃权重(BF16) | FLOPs/T | AI | 理论延迟/卡 | 主要限制 |
|---|---|---|---|---|---|---|
| **Llama-3 70B (Dense, 1×H200)** | 70B | 140 GB | 140G | 1.0 | ~29 ms | 纯带宽 |
| **Nemotron 550B (MoE+Mamba)** | 55B | 110 GB | ~300G | ~2.7 | ~23 ms | 带宽+EP通信 |
| **M3 (MoE+MSA)** | 23B | 46 GB | 215G | ~4.7 | ~5-15 ms | 带宽+EP通信 |

注意 Nemotron 活跃参（55B）比 M3（23B）大 2.4×，即使 Mamba-2 层的计算效率更高，其 decode 延迟仍受制于更大的权重读取量。这就是为什么**活跃参数**是推理延迟的第一性指标——它是 AI 公式分子的底层影响因素。

### 7.9 Roofline 视角下的架构设计启示

从 Roofline 分析出发，理解当前架构设计的选择逻辑：

**MLA（K2.5 / DeepSeek V4）**：MLA 压缩 KV cache 的主要收益在**存储**（CH 4 详述），对 decode latency 的影响是间接的——KV cache 更小意味着同张卡可以装更大的 batch，间接提升吞吐。MLA 不直接改变每 token 的权重读取量，所以**不直接降低单 token 延迟**。

**MSA（M3）**：MSA 减少 attention 计算 FLOPs，但 decode 阶段 attention 计算本身不是瓶颈（它是 memory-bound，减 FLOPs 不改变延迟）。MSA 对 decode 延迟的收益是**减少 KV cache 读取量**——只读 2048 个 token 的 KV 而非全序列。在 $T=1\text{M}$ 时，这个优化把 KV 读取从 ~50 GB 降到 ~2 GB。

**MoE**：MoE 降低每 token 的活跃参数量 → 减少权重读取 → 降低延迟。这是 decode 延迟最直接的架构杠杆。

**Mamba-2（Nemotron）**：用状态空间替代 Attention + KV cache。根本不需要 KV cache 读取 → 消除了 decode 中一个重要的带宽消耗源。结合 MoE 的稀疏化，进一步降低每 token 的有效工作量。

### 7.10 速查：Roofline 分析四步法

给定任意模型 + GPU，判断推理延迟：

1. **算 AI**：$AI = \frac{\text{FLOPs\_per\_token}}{\text{active\_weight\_bytes} + \text{KV\_read\_bytes}}$
2. **判瓶颈**：AI < AI_ridge → memory-bound；AI ≥ AI_ridge → compute-bound
3. **求 achievable TFLOPS**：$\min(\text{Peak}, \text{AI} \times \text{Bandwidth})$
4. **得延迟**：$T_{\text{token}} = \frac{\text{FLOPs\_per\_token}}{\text{achievable\_TFLOPS}}$

对绝大多数 LLM decode 场景，第 2 步的答案是 **memory-bound**。牢记这一点，就不会再犯 "FLOPs ÷ Peak" 的错误。

---

> **本章定位**：Roofline 模型给出了单卡无通信场景下的延迟下限。实际多卡推理中，TP/PP/EP 的通信开销会在此基础上叠加——这是 `communication-methodology.md` 要解决的主题。本章的 "B=1, no comm" 延迟应理解为 **理论下界**，实际系统在此基础上 ×1.5–3×。

---

## 本章公式速查

| 计算目标 | 公式 | 说明 |
|---|---|---|
| 权重显存 | $M_w = N \times \text{bytes}$ | $N$ 为总参数量 |
| GQA KV Cache (per layer) | $2 \times H_{kv} \times D_h \times T \times \text{bytes}$ | K 和 V 两份 |
| MLA KV Cache (per layer) | $(d_{kv} + D_{rope}) \times T \times \text{bytes}$ | 只存压缩向量 |
| MSA Index KV (per layer) | $1 \times D_{\text{idx}} \times T \times \text{bytes}$ | 只有 K |
| 总显存 | $M_w + B \times M_{kv}^{(1)} + M_{act}$ | Batch 乘 KV |
| MoE 激活参 | $\text{Non-MoE} + k \times \text{Params}_{\text{expert}}$ | $k$ 为 top-k |
| Full Attn decode QK FLOPs | $2 \times H_q \times D_h \times T$ | $T$ = cached length |
| MSA Main QK FLOPs | $2 \times H_q \times D_h \times K$ | $K$ = 2048 (16 blocks × 128) |
| 最低卡数 | $\lceil M_w / \text{per\_card} \rceil$ | 仅考虑权重 |
| 最大 Batch | $\lfloor (M_{pool} - M_w) / (M_{kv}^{(1)} + M_{act}) \rfloor$ | 考虑 KV Cache |
| 算术强度 (AI) | $\text{AI} = \frac{\text{Total FLOPs}}{\text{Bytes read from HBM}}$ | FLOPs/byte |
| Roofline Ridge | $\text{AI}_{\text{ridge}} = \frac{\text{Peak TFLOPS}}{\text{Bandwidth}}$ | AI < ridge → memory-bound |
| Achievable TFLOPS | $\min(\text{Peak}, \text{AI} \times \text{Bandwidth})$ | memory-bound 时取右边 |
| Dense decode AI（近似） | $\text{AI} \approx 2 / \text{bytes\_per\_param}$ | BF16 → 1, FP8 → 2, INT4 → 4 |
| Decode 延迟下限 | $T_{\text{token}} \geq \frac{\text{Bytes}_{\text{HBM}}}{\text{Bandwidth}}$ | memory-bound 时成立 |
| Prefill AI（大 T 近似） | $\text{AI}_{\text{prefill}} \gg \text{AI}_{\text{decode}}$ | O(T²) FLOPs 提升 AI → compute-bound |

---

## 本章常见计算错误

| # | 错误 | 正确做法 |
|---|---|---|
| 1 | 用 $H_q$ 代替 $H_{kv}$ 算 KV Cache | KV Cache 宽度由 KV 头数决定（GQA 下 $H_{kv} \ll H_q$），与 Q 头数无关 |
| 2 | MSA 的 KV Cache 忘记 Index K | MSA 额外存储 1 个 Index K（每层 1 头 × $D_{idx}$），约占总 KV Cache 的 10% |
| 3 | 认为 MSA 减少了 KV Cache | MSA 减少的是**计算**（FLOPs），不是**存储**（KV Cache）——两者解耦 |
| 4 | EP 只看总显存够不够 | EP 要求每张卡装得下其分配的专家子集 + 非 MoE 参数副本，不能只看平均数 |
| 5 | Batch 乘 KV Cache 时忘记 batch 效应 | 权重跨 batch 共享，KV Cache 不共享——$B=100$ 就是 100× KV |
| 6 | 混淆 Attention 加速比和端到端加速比 | 30× 是 per-layer Attention 算子加速；10× 是全模型 end-to-end 加速（含不变的线性投影和 FFN） |
| 7 | 激活值完全忽略 | 虽然通常 < 权重的 5%，但在 tight memory budget 下 5% = 50 GiB（8 卡场景），可能就是 OOM 的原因 |
| 8 | 用 FLOPs ÷ Peak TFLOPS 算 decode 延迟 | Decode 的 AI ≈ 1-5 ≪ Ridge(~200)，是 memory-bound。延迟 = Bytes / BW，不是 FLOPs / Peak。偏差 50-500× |
| 9 | 认为 FP8 使 decode 算得更快 | FP8 加速 decode 不是算力翻倍（memory-bound 下用不到 Peak），而是权重字节数减半 → 读取时间减半 |
| 10 | 认为 MoE 的 decode AI 比 Dense 高很多 | AI 公式中活跃 FLOPs 和活跃 Bytes 同比缩放（≈ 2 / bytes_per_param），MoE 降低的是绝对值而非 AI。例外：MoE router + dispatch 有额外开销 |
| 11 | 忽略 KV cache 读取对带宽的消耗 | 短上下文可忽略，但 $T=1\text{M}$ 时 KV 读取可达 GB/层量级。MSA/稀疏注意力通过减少读 KV 量来缓解 |
| 12 | 认为增大 batch 能无限提升吞吐 | 提升 AI 的边际递减——KV cache 读取随 B 增长，权重读取固定。超大 B 时 KV 读取成为新瓶颈 |

---

## 各模型 BF16 推理显存横评

| 模型 | 总参 | 权重 (GB) | KV/样本 (GB) | 可用 (GB) | Max Batch |
|---|---|---|---|---|---|
| Nemotron 3 Ultra | 550B | 1,100 | 13 | 28 | 1 |
| MiniMax M3 | 428B | 856 | 144 | 272 | 1 |
| DeepSeek V4 Flash | ~300B | ~600 | ~72 (MLA) | ~528 | ~7 |
| Kimi K2.5 | ~1T | ~2,000 | ~72 (MLA) | < 0 (OOM!) | 需 16 卡+ |

> 注：K2.5 BF16 推理即使 16 张 H200 (2,256 GiB) 也只能负载 ~2,000 GiB 权重 + 少量 KV。实际部署需要 FP8 或 INT4 量化。

---

> **系列导航**：CH 1-2（预备知识 + 参数分解）→ CH 3（FLOPs 估算）→ CH 4（KV Cache）→ CH 5（推理显存）→ CH 6（M3 实战推演）→ **CH 7（Roofline 与推理延迟）**

---

## 附录

## 附录 A: 常见 config.json 字段速查表

哪些字段影响哪些计算：

| config 字段 | 影响的计算 | 示例值 |
|---|---|---|
| `hidden_size` | 所有投影矩阵参数 + QKV/O 的 FLOPs | 6144 (M3), 7168 (K2.5), 8192 (Nemotron) |
| `num_hidden_layers` | 总层数 → 乘到每层参数/FLOPs/KV cache | 60 (M3), 61 (K2.5), 108 blocks (Nemotron) |
| `num_attention_heads` | Q 投影大小 + QK 点积 FLOPs | 64 (大多数 7B+ 模型) |
| `num_key_value_heads` | K/V 投影大小 + KV cache 大小 | 4 (M3 GQA), 2 (Nemotron), 64 (K2.5 MHA) |
| `head_dim` | QK 点积维度 + KV cache 的 D | 128 (大多数) |
| `intermediate_size` | FFN 参数（up/gate/down gate） | 12288 (M3 dense), 18432 (K2.5) |
| `moe_intermediate_size` | MoE expert 参数 | 2048 (M3), 5120 (Nemotron) |
| `n_routed_experts` | MoE 总专家数 → 总 MoE 参数 | 128 (M3), 256 (GLM-5.1), 384 (K2.5), 512 (Nemotron) |
| `num_experts_per_tok` | 激活参数计算 | 4 (M3), 8 (K2.5), 22 (Nemotron) |
| `kv_lora_rank` | MLA KV 压缩维度 → KV cache 大小 | 512 (K2.5, DeepSeek V3/V4) |
| `q_lora_rank` | MLA Q 压缩维度 → Attention 参数 | 1536 (K2.5) |
| `qk_rope_head_dim` | MLA k_rope 维度 → KV cache 的 rope 分量 | 64 (K2.5) |
| `ssm_state_size` | Mamba-2 state 维度 → 替代 KV cache 的状态大小 | 128 (Nemotron) |
| `max_position_embeddings` | 最大上下文 → KV cache 最大 T + FLOPs 最大 T | 262144 (K2.5), 1048576 (M3/Nemotron) |
| `vocab_size` | Embedding 参数 + LM head 参数 | 131072 (Nemotron), 200064 (M3) |
| `dense_intermediate_size` | MoE 模型的 dense FFN 层参数 | 12288 (M3) |
| `shared_intermediate_size` | 共享 expert 的 FFN 参数 | 3072 (M3) |
| `sparse_block_size` | MSA 的 block 大小 → FLOPs 计算 | 128 (M3) |
| `sparse_topk_blocks` | MSA 的 top-k blocks → FLOPs 计算 | 16 (M3) |
| `vision_config.hidden_size` | ViT 参数 + FLOPs | 1280 (M3), 1152 (K2.5) |
| `vision_config.num_hidden_layers` | ViT 层数 | 32 (M3), 27 (K2.5) |
| `patch_size` | 图像 token 数 → Vision encoder FLOPs | 14 (大多数) |
| `rope_theta` | 位置编码 theta → 上下文扩展策略判断 | 50000 (K2.5), 5000000 (M3), 10000 (Nemotron) |

## 附录 B: 符号与缩写表

| 符号 | 含义 | 常用值示例 |
|---|---|---|
| $d$ / $d_{model}$ | 隐藏维度 (`hidden_size`) | 6144, 7168, 8192 |
| $H$ | Q（Query）头数 (`num_attention_heads`) | 64 |
| $H_{kv}$ | K/V 头数 (`num_key_value_heads`) | 4 (GQA), 2 (GQA), 64 (MHA) |
| $D$ | 每个 head 的维度 (`head_dim`) | 128 |
| $d_{ff}$ | FFN 中间维度 (`intermediate_size` / `moe_intermediate_size`) | 2048-18432 |
| $L$ | 总层数 | 60-108 |
| $L_{attn}$ | 使用 Attention 的层数（Mamba hybrid 中仅部分层） | 12 (Nemotron) |
| $T$ | 序列长度（当前总 token 数） | 4K-1M |
| $T_{new}$ | 新生成 token 数（decode 时为 1） | 1 |
| $N_E$ | MoE 专家总数 (`n_routed_experts`) | 128-512 |
| $k$ | 每个 token 激活的专家数 (`num_experts_per_tok`) | 4-22 |
| $B$ | Batch size | 1 (单样本推理) |
| $d_{kv}$ | MLA KV 压缩维度 (`kv_lora_rank`) | 512 |
| $d_{rope}$ | MLA RoPE 维度 (`qk_rope_head_dim`) | 64 |
| $D_{nope}$ | MLA 每头无位置编码维度 (`qk_nope_head_dim`) | 128 (K2.5), 192 (GLM-5) |
| $D_v$ | MLA 每头 V 维度 (`v_head_dim`) | 128 (K2.5), 256 (GLM-5) |
| $D_{qk}$ | MLA 每头 QK 有效维度（$D_{nope} + D_{rope}$） | 192 (K2.5), 256 (GLM-5) |
| $d_{inner}$ | Mamba-2 内部维度（$H_{mamba} \times D_{mamba}$） | 16384 (Nemotron) |
| $d_{conv}$ | Mamba-2 conv1d 通道数 | 18432 (Nemotron) |
| $H_{mamba}$ | Mamba-2 SSD head 数 | 256 |
| $d_{state}$ | Mamba-2 状态空间维度 | 128 |
| $d_{latent}$ | LatentMoE 低秩维度 | 2048 (Nemotron) |
| $C$ | Mamba-2 chunk 大小 | 128 |
| $B_{msa}$ | MSA block 大小 | 128 |
| $K_{msa}$ | MSA top-k blocks | 16 |
| $N_{img}$ | 每图像 token 数 | 576 |
| $V$ | 词表大小 (`vocab_size`) | 131072, 200064 |
| **字节精度** | BF16=2, FP8=1, FP4=0.5, FP32=4 | — |

## 附录 C: 8 个已拆解模型的计算结果速览

| 模型 | 总参 | 激活参 | FLOPs (decode, T=1M) | KV Cache (1M) | 推理显存 (BF16, 1 sample) |
|---|---|---|---|---|---|
| **Nemotron 3 Ultra** | 550B | 55B | ~1.2×10¹⁵ | 12.0 GiB (仅12 Attn层) | ~1.13 TiB |
| **MiniMax M3** | 428B | 23B | ~2.2×10¹¹ | 144 GiB | ~1,000 GiB |
| **Kimi K2.5** | 1T | 32B | ~(未在1M下) | ~21.5 GiB (256K) | — (256K context) |
| **DeepSeek V4-Flash** | ~300B | 37B | — | ~131 GiB (1M, MLA) | — |
| **MiniMax M2.7** | ~275B | ~17B | — | — (Full Attn O(T²)) | — |
| **GLM-5.1** | 744B | 32B | — | — | — |
| **Qwen3.5-MoE** | ~35B | ~3B | — | — | — |
| **MiMo-V2-Flash** | ~140B | ~7B | — | — | — |

("—" = 该模型未在该上下文长度下做详细估算，或报告未公开该维度数据)

---

> **关于本文**：本文档从 8 个开源 LLM 的深度架构拆解中提炼而成。每个公式、每个数字都在对应模型上验证通过。如果你发现错误或有改进建议，欢迎反馈。

> **系列导航**：[（一）预备知识与参数分解](../part-1/) ← [（二）FLOPs 估算](../part-2/) ← [（三）KV Cache 与推理显存](../part-3/) ← 当前
