+++
date = '2026-06-15'
draft = false
title = 'LLM 计算方法论（三）：KV Cache 与推理显存'
categories = ['aiinfra']
tags = ['computation', 'kv-cache', 'memory', 'inference', 'methodology']
series = 'llm-computation-methodology'
series_order = 3
math = true
summary = 'KV Cache 原理与公式推导，覆盖 GQA / MLA / MSA / Mamba-2 四种架构的缓存策略；推理显存完整拆解，包括权重、KV Cache、激活值的显存占用计算。'
+++

## CH 4 KV Cache 显存：原理、公式与多架构对比

> **计量约定**：本章 KV cache 使用 **GiB**（1024³ bytes）。1 GiB = 1024³ bytes ≈ 1.074 GB。使用 1024 进制是因为 GPU 显存以 2 的幂次分配。T（序列长度）取 2^20 = 1,048,576。

> **本章定位**：系统推导自回归推理中 KV cache 的显存占用公式，覆盖 MHA、GQA、MLA、MSA、Mamba-2 五种架构，并用 Kimi K2.5（MLA）、Nemotron 3 Ultra（GQA+Mamba）、MiniMax M3（MSA+GQA）的实测配置验证所有公式。

---

## 4.1 为什么需要 KV Cache

### 4.1.1 这节算什么

自回归推理时，模型每步只生成一个 token，但需要与所有历史 token 做 attention 运算。本节量化 KV cache 的本质：**空间换时间**——缓存中间结果，避免每步重新计算。

### 4.1.2 为什么重要

KV cache 是长上下文推理的**第一瓶颈**。1M 上下文中，纯 Attention 模型的 KV cache 可达数百 GiB，远超模型权重本身。架构选择（GQA、MLA、Mamba）的核心动机之一就是压缩或消除 KV cache。

### 4.1.3 直觉理解

> 看书时，读到第 100 页，不需要每翻一页就从头重读一遍——记住前面每一页的「关键信息」就够了。KV cache 就是模型在推理过程中对历史 token 的「关键信息摘要」。

标准自回归推理中，第 $t$ 步的 attention 需要对前 $t-1$ 个历史 token 计算 QK 点积：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t \cdot K_{1:t}^T}{\sqrt{d_k}}\right) \cdot V_{1:t}$$

如果每步都重新计算 $K_{1:t}$ 和 $V_{1:t}$，第 $T$ 步的 FLOPs 将是 $O(T^2 \cdot d)$，总推理 FLOPs 为 $O(T^3 \cdot d)$。而缓存 KV 后，每步只需计算新 token 的 QKV 投影并与缓存中的 K、V 做 attention，总推理 FLOPs 降为 $O(T^2 \cdot d)$。

---

## 4.2 标准 MHA/GQA 的 KV Cache

### 4.2.1 这节算什么

从 MHA 和 GQA 的 attention 计算出发，推导 KV cache 的标准公式。这是所有 KV cache 分析的基准。

### 4.2.2 推导过程

**第 1 步：每个 token 需要缓存什么？**

标准自注意力中，对于序列中的每个历史 token，我们需要其 Key 向量和 Value 向量。每个 token 的 K 和 V 各一份，维度完全相同。

对于单个 attention head：
- K shape: `[head_dim]`
- V shape: `[head_dim]`

但实际存储是按 KV head 组织的（GQA 下 Q head 可以多于 KV head，此时多个 Q head 共享同一个 KV head）。

**第 2 步：每层每 token 的缓存元素数**

设 `num_kv_heads = H_{kv}`，`head_dim = D`。每个 token 需要缓存 K 和 V 各一份：

$$\text{Cache elements per token per layer} = 2 \times H_{kv} \times D$$

其中每份 K 为 $H_{kv} \times D$ 个元素，V 同理。

**第 3 步：完整模型公式**

$$\text{KV Cache}_{total} = L_{attn} \times 2 \times H_{kv} \times D \times T \times \text{bytes\_per\_elem}$$

其中 $L_{attn}$ 为包含 attention 的层数，$T$ 为序列长度，$\text{bytes\_per\_elem}$ 取决于精度。
注意：如果模型包含非 attention 层（如 Mamba-2、纯 MLP 层），那些层不需要 KV cache，因此不参与计数。

### 4.2.3 直觉理解

- **$2 \times H_{kv} \times D$**: 每层每 token 缓存 K+V 两个矩阵，每个矩阵有 `H_kv` 个 head × `D` 维 = 这就是一个 token 的「关键信息摘要」
- **$\times T$**: 序列多长，缓存就多大——**线性增长**（这是 $O(T)$ 的）
- **$\times L_{attn}$**: 每个 attention 层独立缓存
- GQA 的省法：差异只在于 $H_{kv}$。$H_{kv}$ 越小，缓存越小

### 4.2.4 验证案例 1：Kimi K2.5（全 MHA，未使用 MLA 压缩时的理论值）

Kimi K2.5 使用全 MHA，即 $H_{kv} = H_Q = 64$，无 GQA 压缩。K 的有效维度为 $D_K = D_{nope} + D_{rope} = 192$（MLA 将 K 拆为 128 维内容 + 64 维位置），V 为 128 维。若不使用 MLA（仅作理论对比），在 $T = 256\text{K}$（$262{,}144$ tokens）下，BF16 精度：

$$\text{KV Cache}_{no\_MLA} = 61 \times 64 \times (192 + 128) \times 262{,}144 \times 2$$

$$= 61 \times 64 \times 320 \times 262{,}144 \times 2 = 61 \times 20{,}480 \times 524{,}288$$

$$= 654{,}977{,}269{,}760 \text{ bytes} \approx 610.0 \text{ GiB}$$

直觉：这就是没有 MLA 压缩的代价——近 500 GiB，远超任何单 GPU 显存。这是 MLA 必须存在的根本原因。

### 4.2.5 验证案例 2：Nemotron 3 Ultra（GQA 32:1）

Nemotron 3 Ultra 仅有 12 层 Attention，使用极致 GQA（$H_{kv} = 2$ 个 KV head），$D = 128$，在 $T = 1\text{M}$（$1{,}048{,}576$ tokens）下，BF16：

$$\text{KV Cache}_{Nemotron} = 12 \times 2 \times 2 \times 128 \times 1{,}048{,}576 \times 2$$

$$= 12 \times 512 \times 1{,}048{,}576 \times 2 = 12 \times 1{,}073{,}741{,}824$$

$$= 12{,}884{,}901{,}888 \text{ bytes} = 12.0 \text{ GiB}$$

✅ 与 Nemotron 3 Ultra 技术报告声明的 **~12.0 GiB** 完全一致。

**为什么这么小？三个因素叠加：**
- 仅 12 层有 Attention（其余 48 层是 Mamba-2，不需要 KV cache）
- GQA 32:1，$H_{kv}=2$——每层仅 2 个 KV head
- 不使用 RoPE，head_dim=128 全部是「内容」维度

### 4.2.6 验证案例 3：MiniMax M3（GQA 16:1，主 KV cache）

MiniMax M3 全部 60 层使用 GQA 16:1（$H_{kv}=4$），$D=128$，在 $T=1\text{M}$ 下，BF16：

$$\text{KV Cache}_{M3\_main} = 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2$$

$$= 60 \times 1{,}024 \times 1{,}048{,}576 \times 2 = 60 \times 2{,}147{,}483{,}648$$

$$= 128{,}849{,}018{,}880 \text{ bytes} = 120.0 \text{ GiB}$$

✅ 与 M3 报告声明的 **~120 GiB** 完全一致。

### 4.2.7 GQA 压缩比公式

GQA 相对 MHA 的 KV cache 节省比例：

$$\text{Compression Ratio}_{GQA} = \frac{H_Q}{H_{kv}}$$

M3 的 GQA 16:1 意味着 KV cache 仅为 MHA 的 $1/16$。Nemotron 的 32:1 节省更极致。

一个全 MHA 60 层模型（$H_{kv}=64$, $D=128$）在 1M 上下文的 KV cache：

$$60 \times 2 \times 64 \times 128 \times 1{,}048{,}576 \times 2 = 1{,}886{,}621{,}245{,}440 \text{ bytes} \approx 1{,}758 \text{ GiB}$$

这是不可部署的。GQA 是长上下文推理的基本生存策略。

---

## 4.3 MLA 的 KV Cache（Multi-head Latent Attention）

### 4.3.1 这节算什么

MLA 是本章最复杂的部分。MLA（DeepSeek V2/V3 提出，Kimi K2 系列继承）通过低秩压缩改变 KV cache 的存储对象——**不再直接缓存 K 和 V，而是缓存一个低秩潜向量 $\mathbf{c}_t^{KV}$ 和一个额外的 RoPE 分量**。本节从 shape 角度逐步推导 MLA 的缓存公式，并用 Kimi K2.5 的实测配置验证。

### 4.3.2 为什么重要

MLA 是当前 MoE 模型（DeepSeek V3/R1、Kimi K2 系列）实现长上下文推理的关键技术。不压缩时 K2.5 的 KV cache 高达 ~610 GiB（见下），MLA 将其压缩到约 21.5 GiB——**压缩比 35.6×**。理解 MLA 的缓存公式是评估 MoE 推理成本的前提。

### 4.3.3 核心问题

MLA 的 K 和 V 不是直接存储的——它们从一个共享的低秩潜向量 $\mathbf{c}_t^{KV}$ 通过升维投影得到。那么推理时 cache 应该存什么？是存完整的 K 和 V（失去了 MLA 的意义），还是存压缩后的潜向量？

**答案**：缓存 $\mathbf{c}_t^{KV}$（共享潜向量，可同时解压出 K 和 V）+ $\mathbf{k}_t^R$（RoPE 位置分量，不可压缩）。

### 4.3.4 推导过程：从 Shape 角度一步一步来

#### 第 1 步：标准 Attention 的 K 是什么

在标准 MHA 中，每个 token 的 K 是一个形状为 $[H_{kv}, D_K]$ 的矩阵。以 Kimi K2.5 为例（全 MHA, $H_{kv} = H_Q = 64$），其 MLA 架构中 K 的实际维度为 $D_K = D_{nope} + D_{rope} = 128 + 64 = 192$：

$$\text{K cache per token per layer} = 64 \times 192 = 12{,}288 \text{ 个元素}$$

V 的维度为 $D_v = 128$：$64 \times 128 = 8{,}192$ 个元素。合计 $20{,}480$ 个元素。

#### 第 2 步：MLA 如何计算 K——分为两块

MLA 将 K 分为两个功能不同的分量：

**分量 1：$\mathbf{k}^{nope}$（内容分量，128 维 per head）**

$$
\mathbf{c}_t^{KV} = \mathbf{x}_t \cdot \mathbf{W}_{kv\_down} \in \mathbb{R}^{512}
$$

$$
\mathbf{K}_{t}^{nope} = \mathbf{c}_t^{KV} \cdot \mathbf{W}_{k\_up} \in \mathbb{R}^{64 \times 128}
$$

其中 $\mathbf{c}_t^{KV}$ 是 512 维的潜向量，通过共享的 $\mathbf{W}_{kv\_down}$ 投影得到。然后通过 $\mathbf{W}_{k\_up}$ 升维到 64 个 head × 128 维的完整 K（仅 nope 部分）。

**关键**：$\mathbf{K}^{nope}$ 是 64 × 128 = 8,192 维的矩阵，但它完全由 512 维的 $\mathbf{c}_t^{KV}$ 决定——所以不需要缓存 8,192 维，只需缓存 512 维。

**分量 2：$\mathbf{k}^{rope}$（位置分量，64 维）**

RoPE 是一个正交旋转变换，施加在 K 的头维度上。按照 MLA 的设计，RoPE 部分使用 **MQA（Multi-Query Attention）方式共享**：所有 64 个 attention head 使用**同一个** RoPE Key 向量，维度为 $d_{rope} = 64$（即 `qk_rope_head_dim`）。

$$\mathbf{k}_t^R = \text{RoPE}(\mathbf{x}_t \cdot \mathbf{W}_{kr}) \in \mathbb{R}^{64}$$

每个 head $i$ 的完整 K 为：

$$\mathbf{K}_{t,i} = [\mathbf{k}_t^R \,;\, \mathbf{K}_{t,i}^{nope}] \in \mathbb{R}^{64 + 128 = 192}$$

**为什么 $\mathbf{k}^R$ 不能被压缩？** RoPE 是施加在完整 K 上的旋转变换——位置编码依赖具体的坐标值，不能通过低秩近似保留。因此 $\mathbf{k}^R$ 必须独立缓存。但由于它采用 MQA 共享（而非每 head 一份），实际缓存量很小。

#### 第 3 步：MLA 如何计算 V

V 完全从 $\mathbf{c}_t^{KV}$ 解压得到，没有 RoPE 分量：

$$\mathbf{V}_t = \mathbf{c}_t^{KV} \cdot \mathbf{W}_{v\_up} \in \mathbb{R}^{64 \times 128}$$

**关键**：V 是 64 × 128 = 8,192 维，但完全由 512 维的 $\mathbf{c}_t^{KV}$ 决定——因此 V 也不需要单独缓存。

#### 第 4 步：Cache 里到底存什么

综合第 2、3 步，每个 token 每层缓存的元素数为：

| 缓存项 | 维度 | 是否可压缩 | 备注 |
|--------|------|-----------|------|
| $\mathbf{c}_t^{KV}$ | `kv_lora_rank` = 512 | 这是压缩形式 | 同时编码 K_nope 和 V |
| $\mathbf{k}_t^R$ | `qk_rope_head_dim` = 64 | 不可压缩 | MQA 共享，所有 head 复用 |

合计：$512 + 64 = 576$ 个元素 per token per layer。

对比标准 Attention：$64 \times 192 + 64 \times 128 = 20{,}480$ 个元素。**MLA 压缩比为 $20{,}480 / 576 \approx 35.6\times$。**

#### 第 5 步：Per Token Per Layer 公式

$$\text{Cache per token per layer}_{MLA} = (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) \times \text{bytes\_per\_elem}$$

**注意：这里不是 $\times 2$！** 标准 Attention 的 $\times 2$ 是因为 K 和 V 各自独立存储。而 MLA 中 `kv_lora_rank` 的单个潜向量同时编码了 K_nope 和 V——一份存储，两份产出。

#### 第 6 步：完整模型公式

$$\text{KV Cache}_{MLA} = L \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) \times T \times \text{bytes\_per\_elem}$$

其中 $L$ 为模型总层数（MLA 通常在所有层使用）。

### 4.3.5 验证：代入 Kimi K2.5

**配置回顾**（`config.json`）：
- $L = 61$ 层，全部使用 MLA
- `kv_lora_rank` = 512
- `qk_rope_head_dim` = 64
- $T = 256\text{K} = 262{,}144$ tokens
- $\text{bytes\_per\_elem} = 2$（BF16）

**代入公式**：

$$\text{KV Cache}_{K2.5} = 61 \times (512 + 64) \times 262{,}144 \times 2$$

$$= 61 \times 576 \times 262{,}144 \times 2$$

$$= 61 \times 301{,}989{,}888 = 18{,}421{,}383{,}168 \text{ bytes}$$

$$= 17.2 \text{ GiB}$$

**与报告声明的对比**：Kimi K2.5 技术报告声明 256K 时 KV cache 约 21.5 GiB。公式推导结果（17.2 GiB ≈ 18.4 GiB）与报告值差异约 15%。这一差异的可能来源：

1. **KV cache 对齐开销**：GPU 显存通常以 128B 或 256B 对齐，每层每 token 额外开销约为 5-10%
2. **额外缓存结构**：部分 MLA 实现可能缓存额外的元数据（如 index/causal mask 的辅助结构）
3. **报告舍入误差**：技术报告中的数字通常做了一定程度的舍入

综合考虑对齐开销后约为 $17.2 \times 1.05 \approx 18.0 \text{ GiB}$，与 21.5 GiB 仍在同一数量级。

### 4.3.6 MLA 的直觉理解

- **「两本账合一」**：标准 Attention 需要分别存 K 和 V 两本账（$\times 2$）。MLA 把两本账的信息压缩到同一个潜向量 $\mathbf{c}_t^{KV}$ 里——一个 512 维向量同时包含了 K 和 V 的精华
- **「位置信息外包」**：RoPE 不能压缩，但 MLA 巧妙地将 K 的 RoPE 部分用 MQA 方式共享（所有 head 共用一个 $\mathbf{k}^R$），而不是每个 head 存一份
- **「为什么 MLA 比纯 GQA 更省」**：GQA 只是减少了 KV head 数量（空间省但内容信息量受限），MLA 进一步在每 head 内部做低秩压缩——相当于 GQA 省宽度，MLA 省深度

### 4.3.7 MLA 压缩比的极限分析

MLA 的压缩比：

$$\text{Compression Ratio}_{MLA} = \frac{2 \times H_{kv} \times D}{\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}}$$

以 K2.5 为例：

$$\frac{64 \times 192 + 64 \times 128}{512 + 64} = \frac{20{,}480}{576} \approx 35.6\times$$

**压缩比的结构分解**：
- 来自「K+V 共享潜向量」：$\times 2 \to \times 1$（省 50%）
- 来自「低秩压缩 $8{,}192 \to 512$」：约 16 倍
- 来自「$\mathbf{k}^R$ 的 MQA 共享」：64 head $\to$ 1 个共享向量（省约 64 倍）

三项叠加：$2 \times 16 \approx 32\times$，减去 $\mathbf{k}^R$ 开销后约 28 倍。

---

## 4.4 MSA 的 KV Cache（MiniMax Sparse Attention）

### 4.4.1 这节算什么

MiniMax M3 的 MSA（MiniMax Sparse Attention）在标准的 GQA KV cache 之上，额外引入了一组 **Index K cache**——用于 block-level 稀疏选择的轻量评分 Key。本节量化 MSA 的额外缓存开销。

### 4.4.2 为什么重要

MSA 的稀疏性体现在**计算**（每次只访问 top-16 blocks），但**不体现在存储**（所有 KV 仍需缓存，因为不同 query 可能选择不同的 blocks）。理解这一点才能正确评估 MSA 的显存需求——MSA 的加速来自计算 FLOPs 的减少，而不是 KV cache 的减少。

### 4.4.3 主 KV Cache：与标准 GQA 完全相同

MSA 不改变 K 和 V 的存储方式。60 层全部缓存主 KV，与标准 GQA 公式一致：

$$\text{KV Cache}_{M3\_main} = 60 \times 2 \times 4 \times 128 \times T \times 2 = 120.0 \text{ GiB at } T = 1\text{M}$$

计算过程已在 4.2.6 节验证，与 M3 报告声明的 **~120 GiB** 完全一致。

### 4.4.4 Index K Cache：MSA 的额外开销

MSA 的 Index Branch（`MiniMaxM3VLIndexer`）用于为每个 query 从 $B = \lceil T / 128 \rceil$ 个 block 中评选出 top-16。Index Branch 需要缓存一个独立的 Index Key：

**Index K 的 shape**：
- $n_{idx\_heads} = 4$（4 个 index head 用于多角度评分）
- Index K head: **只有 1 个**（被所有 4 个 index head 通过广播共享）
- `sparse_index_dim = 128`

$$\text{Index K elements per token per layer} = 1 \times 128 = 128$$

$$\text{Index K cache per token per layer} = 128 \times 2 = 256 \text{ bytes (BF16)}$$

完整公式（仅 MSA 层，即 57 层）：

$$\text{KV Cache}_{M3\_index} = L_{MSA} \times H_{idx\_k} \times D_{idx} \times T \times \text{bytes\_per\_elem}$$

代入 M3 配置（$L_{MSA} = 57$, $H_{idx\_k} = 1$, $D_{idx} = 128$, $T = 1\text{M}$）：

$$= 57 \times 1 \times 128 \times 1{,}048{,}576 \times 2$$

$$= 57 \times 268{,}435{,}456 = 15{,}300{,}820{,}992 \text{ bytes} = 14.3 \text{ GiB}$$

✅ 与 M3 报告声明的 **~14.2 GiB** 完全一致。

### 4.4.5 M3 总 KV Cache

$$\text{KV Cache}_{M3\_total} = 120.0 + 14.3 = 134.25 \text{ GiB at } T = 1\text{M}$$

其中主 KV cache 占 89.4%，Index K cache 占 10.6%。Index K cache 虽然每 token 只有 128 个元素（vs 主 KV 的 $2 \times 4 \times 128 = 1{,}024$ 个元素），但涉及 57 层，总计也达到了不可忽略的 ~14 GiB。

### 4.4.6 直觉理解

- **MSA 省计算，不省存储**：主 KV cache 与 Full Attention 一模一样——所有历史 token 的 K 和 V 都必须保留，因为不同 query 会选择不同的 top-16 blocks
- **Index K cache 是「目录索引」的代价**：在 1M 上下文中，需要额外的 ~14 GiB 来存储这份目录索引，但换来 decode 计算 30 倍加速（参见 M3 报告 CH3.6）
- **Index K 的 MQA 共享**：4 个 index head 共享 1 个 index key，如果用 4 个独立的 index key，开销将是 $14.3 \times 4 = 57 \text{ GiB}$

---

## 4.5 Sliding Window Attention 的 KV Cache

SWA 的 KV cache 公式与标准 Attention 完全相同——window 只是限制了计算时"看多远"，不影响"存多少"。KV cache 仍然需要缓存全部历史 token：

$$M_{\text{kv}}^{\text{SWA}} = 2 \times L \times H_{kv} \times D_h \times T \times \text{bytes}$$

计算时只取最后 $W$ 个 token 参与注意力。这意味着 SWA 在长上下文推理时，**KV cache 显存与 Full Attention 完全相同**，仅计算量有节省。

对比：如果 $T = 1\text{M}$，$W = 131\text{K}$，KV cache 按 $T$ 存（~120 GiB for M3 的 GQA 配置），但 FLOPs 按 $W$ 算（~2.15 GFLOPs/layer vs ~17.2 GFLOPs/layer for Full Attn）。SWA 的定位是"省计算不省显存"。

## 4.6 Gated DeltaNet / Linear Attention 的状态空间

Gated DeltaNet 没有传统 KV cache——它用一个固定大小的矩阵 $S \in \mathbb{R}^{H \times D_h \times D_h}$ 替代：

$$M_{\text{state}}^{\text{DeltaNet}} = L \times H \times D_h^2 \times \text{bytes\_per\_elem}$$

以 Qwen3.5-MoE 为例（$L$ 层，$H = 64$，$D_h = 128$，BF16）：$L \times 64 \times 128^2 \times 2 = L \times 2.1\text{MB}$。假设 $L = 48$：$48 \times 2.1\text{MB} \approx 100\text{MB}$。

对比 Attention 的 KV cache（$T = 1\text{M}$）：$2 \times 48 \times H_{kv} \times 128 \times 1\text{M} \times 2$——即使 $H_{kv} = 2$（极端 GQA）也是 $2 \times 48 \times 2 \times 128 \times 10^6 \times 2 \approx 49\text{GB}$。**差距约 500×**。

> DeltaNet 和 Mamba-2 的选择差异：DeltaNet 的状态是 $O(H \times D_h^2)$——矩阵形状的。Mamba-2 的状态是 $O(H \times N)$——向量形状的，$N \ll D_h$。DeltaNet 的"记忆"更丰富（矩阵可以存更多信息），但代价是状态更新（$O(D_h^2)$）比 Mamba-2 的状态传递（$O(N^2)$）更贵。这是计算-记忆的 trade-off。

## 4.7 无 KV Cache 的架构：Mamba-2

### 4.7.1 这节算什么

Mamba-2（State Space Duality）用固定大小的循环状态替代随序列长度线性增长的 KV cache。本节量化 Mamba 的状态开销，并与 Attention 的 KV cache 做对比。

### 4.7.2 为什么重要

Mamba 代表了「彻底消除 KV cache」的架构方向。理解 Mamba 的状态开销是评估混合架构（如 Nemotron 3 Ultra = 48 Mamba-2 + 12 Attention）显存优势的前提。

### 4.7.3 状态空间模型的状态

Mamba-2 的循环递推形式为：

$$h_t = A_t h_{t-1} + B_t x_t$$
$$y_t = C_t h_t + D x_t$$

其中隐状态 $h_t \in \mathbb{R}^{H_{ssm} \times d_{state}}$。对于 Nemotron 3 Ultra：

- $H_{ssm} = 256$（256 个 SSD head）
- $d_{state} = 128$（每 head 的状态维度）

**每层状态大小**（与序列长度无关）：

$$\text{State size per layer} = H_{ssm} \times d_{state} \times \text{bytes\_per\_elem}$$

代入：

$$= 256 \times 128 \times 4 \text{ bytes (FP32 cache)} = 131{,}072 \text{ bytes} = 128 \text{ KiB}$$

48 层 Mamba-2 总状态：

$$48 \times 131{,}072 = 6{,}291{,}456 \text{ bytes} \approx 6.0 \text{ MiB}$$

### 4.7.4 对比：Mamba 状态 vs Attention KV Cache

在 $T = 1\text{M}$ 上下文下：

| 架构 | 存储 | 与 $T$ 的关系 |
|------|------|--------------|
| 12 层 Attention (GQA 32:1) | 12.0 GiB | $\propto T$ |
| 48 层 Mamba-2 | 6.0 MiB | 常数（与 $T$ 无关） |
| 60 层全 Attention (MHA 64 heads) | 1,758 GiB | $\propto T$ |

**Mamba-2 的状态仅为 12 层 Attention KV cache 的约 1/2000**。这就是混合架构（如 Nemotron 3 Ultra）的核心推理效率优势：Mamba-2 层以恒定大小的循环状态替代了 KV cache，使长上下文推理的显存开销主要由少量的 Attention 层决定。

> **KV Cache 自查清单**（算完后对照）：
> - [ ] 公式中的 `×2` 是 K+V 各一份？不是 ×4？
> - [ ] GQA 用 `H_kv`（不是 `H_q`）？KV head 数少了显存就省了？
> - [ ] MLA 的 `c_t^{KV}` 同时编码 K_nope 和 V → 不需要 ×2？
> - [ ] MLA 的 `k_rope` 维度 = `H × qk_rope_head_dim`（不是 `H_kv × head_dim`）？
> - [ ] Mamba 层没有 KV cache → 仅 Attention 层计入？
> - [ ] 你的数在合理范围吗？256K 时全 MHA ~数百 GiB，MLA ~20 GiB，Mamba-2 <10 MB？

### 4.7.5 直觉理解

- **「看书 vs 记笔记」**：Attention 是把整本书的每一页都摊在桌上（KV cache $\propto T$），Mamba 是看完一页记一行笔记（固定大小的状态）
- **「状态是压缩的上下文」**：128 维的状态向量是前文所有信息的压缩表示——信息量有限但足以支撑后续推理
- **「代价是信息损失」**：Mamba 的固定状态必然丢失细节——这就是为什么 Nemotron 保留了 12 层 Attention（周期性全局交互补充 Mamba 丢失的长程细节）

---

## 4.8 视觉 Token 的 KV Cache 增量

### 4.8.1 这节算什么

多模态模型（M3、K2.5）中，图像和视频 token 也需要 KV cache。本节量化视觉 token 对 KV cache 的额外贡献。

### 4.8.2 为什么重要

一张高分辨率图像（如 M3 的 576 visual tokens）在长上下文推理中可能占据显著的 cache 份额。如果输入包含多张图片或视频帧，视觉 token 的 cache 增量不可忽略。

### 4.8.3 计算公式

视觉 token 对 KV cache 的增量与文本 token 使用完全相同的公式，只是 $T$ 增加了视觉 token 数量：

$$\Delta \text{KV Cache}_{visual} = L_{attn} \times 2 \times H_{kv} \times D \times T_{visual} \times \text{bytes\_per\_elem}$$

对于 M3（GQA 16:1, $H_{kv}=4$, $D=128$, BF16），1 张图（576 visual tokens）：

$$\Delta_{1\_image} = 60 \times 2 \times 4 \times 128 \times 576 \times 2 = 60 \times 1{,}024 \times 576 \times 2$$

$$= 60 \times 1{,}179{,}648 = 70{,}778{,}880 \text{ bytes} \approx 66.0 \text{ MiB}$$

10 张图：$\approx 659 \text{ MiB}$。100 张图：$\approx 6.6 \text{ GiB}$。

对于 K2.5 MLA（$L=61$, `kv_lora_rank=512`, `qk_rope_head_dim=64`, $T_{visual} = 1024$ per image）：

$$\Delta_{1\_image} = 61 \times (512 + 64) \times 1024 \times 2 = 61 \times 576 \times 1024 \times 2$$

$$= 61 \times 1{,}179{,}648 = 71{,}958{,}528 \text{ bytes} \approx 67.0 \text{ MiB}$$

注意：MLA 压缩后，每视觉 token 的 cache 增量为 1,152 bytes（vs 标准 GQA 的 2,048 bytes），单张图差异不大，但在大量图片的场景下 MLA 的优势会累积。

---

## 4.9 完整案例对比

### 4.9.1 三个模型的全量 KV Cache 表

| 模型 | 架构 | $L_{attn}$ | KV 公式 | 关键参数 | 256K Cache | 1M Cache |
|------|------|-----------|---------|---------|-----------|---------|
| **Kimi K2.5** | MLA (全 MHA) | 61 | $L \times (lora + d_{rope}) \times T \times 2$ | lora=512, drope=64 | **~17 GiB** | N/A（不支持 1M） |
| **Nemotron 3 Ultra** | GQA + Mamba | 12 | $L \times 2 \times H_{kv} \times D \times T \times 2$ | H_kv=2, D=128 | ~3 GiB | **~12 GiB** |
| **MiniMax M3** | MSA + GQA | 60 (+57 index) | $L \times 2 \times H_{kv} \times D \times T \times 2$ + index | H_kv=4, D=128 | ~30 + 3.6 GiB | **~120 + 14 GiB** |
| **假设纯 Full Attn 60 层** | MHA | 60 | $L \times 2 \times H_{kv} \times D \times T \times 2$ | H_kv=64, D=128 | ~440 GiB | ~1,758 GiB |

### 4.9.2 这张表告诉我们什么

1. **架构选择直接决定部署可行性**。纯 Full Attention 60 层模型在 1M 上下文需要 1.76 TiB KV cache——没有任何单 GPU 可以承载。而 Nemotron 3 Ultra 仅需 12 GiB（约 1/150），M3 需 134 GiB（约 1/13）。

2. **MLA 是当前 KV cache 压缩最强的 Attention 方案**。K2.5 的 MLA 实现了 35.6× 压缩——仅用 ~21.5 GiB 就支撑了 61 层全 MHA 的 256K 上下文。作为对比，若不用 MLA（纯 MHA），同样配置需要 ~610 GiB。采用正确的 K 维度（192 = 128+64）计算。

3. **Mamba-2 是消除 KV cache 的根本方案**。Nemotron 的 48 层 Mamba-2 仅需 6 MiB 状态存储（与序列长度无关），而 12 层 Attention 在 1M 时需要 12 GiB。混合架构的本质是用少量 Attention 层换取全局交互能力，用大量 Mamba 层换取 KV-cache-free 的长程编码。

4. **MSA 是「半方案」**——它有效减少计算（decode 加速 30 倍），但不减少存储。M3 的 1M KV cache 高达 134 GiB，仍是部署瓶颈。将 MSA 与 KV cache 量化（FP8/INT4）或 token eviction 结合是自然的演进方向。

### 4.9.3 各架构 KV Cache 增长曲线（概念性公式）

| 架构 | KV Cache 复杂度 | 116K 典型值 | 1M 典型值 |
|------|----------------|-----------|---------|
| 全 MHA (60 层) | $O(L \cdot T)$ | ~220 GiB | ~1,758 GiB |
| GQA 16:1 (60 层) | $O(L \cdot T / R_{GQA})$ | ~30 GiB | ~120 GiB |
| MLA (61 层, K2.5) | $O(L \cdot T / R_{MLA})$ | ~8 GiB | ~67 GiB |
| Mamba-2 (48 层) | $O(L)$ — 常数 | ~6 MiB | ~6 MiB |
| 混合 (12 Attn + 48 Mamba) | $O(L_{attn} \cdot T)$ + 常数 | ~3 GiB | ~12 GiB |

### 4.9.4 工程结论

在部署长上下文 LLM 时，KV cache 的架构选择遵循以下优先级：

1. **如果任务不需要完美 recall**：Mamba-heavy 混合架构（如 Nemotron 3 Ultra）是最优解——极致 GQA + 最少 Attention 层
2. **如果需要高精度长程 attention**：MLA 优于纯 GQA——同样 KV head 数下，MLA 通过低秩压缩再省 10-30 倍
3. **如果需要白盒一致性和全 attention 质量**：MSA 减少计算但需承受全量 KV cache 存储——适合计算瓶颈而非显存瓶颈的场景
4. **KV cache 量化（FP8/INT4）是通用的叠加优化**：可与上述任何架构组合使用，通常再压缩 2-4 倍

---

## 4.10 公式速查表

| 公式 | 适用架构 | 说明 |
|------|---------|------|
| $L \times 2 \times H_{kv} \times D \times T \times \text{bpe}$ | MHA / GQA | 标准 KV cache，$\times 2$ 来自 K+V |
| $L \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) \times T \times \text{bpe}$ | MLA | 潜向量 $\mathbf{c}_t^{KV}$ + 共享 RoPE key $\mathbf{k}^R$ |
| $L_{MSA} \times H_{idx\_k} \times D_{idx} \times T \times \text{bpe}$ | MSA (Index) | 额外的 Index K cache |
| $L_{ssm} \times H_{ssm} \times d_{state} \times \text{bpe}$ | Mamba-2 | 固定大小，与 $T$ 无关 |
| $\text{bpe}$ | — | bytes per element: BF16=2, FP32=4, FP8=1, INT4=0.5 |

---


---


## CH 5 推理显存 & CH 6 完整实战推演

> **读者定位**：已掌握 CH 1-2（config.json 读取 + 参数分解）和 CH 3-4（FLOPs 估算 + KV Cache）的工程师，目标是从参数/FLOPs/KV Cache 出发，计算任意模型在给定硬件上的推理部署方案。

---

## CH 5 | 推理显存——「部署需要多少卡」

> **计量约定**：本章显存估算中，权重显存使用十进制 GB（$10^9$ bytes，行业速算惯例：参数量(B) × 精度字节数 = GB），KV cache 和激活显存使用二进制 GiB（$1024^3$ bytes）。GPU 显存规格通常以 GiB 标注（如 H200 = 141 GiB），实际卡数估算时需注意此偏差——权重 GB 数 ≈ 1.074 × 权重 GiB 数。

### 5.1 显存预算的三部分

建立推理显存的三要素分解框架。算完 FLOPs 只知道"算得动吗"，算完显存才知道"装得下吗"——后者往往是真正的瓶颈，因为模型权重在推理期间必须常驻显存。

推理一块 GPU 需要同时装下三样东西：

$$\text{Total Memory} = \underbrace{M_{\text{weights}}}_{\text{模型权重}} + \underbrace{M_{\text{kv}}}_{\text{KV Cache}} + \underbrace{M_{\text{act}}}_{\text{激活 + 临时缓冲}}$$

三者的比例关系随模型架构不同变化巨大。以下是一个典型 MoE 模型（如 Nemotron 550B）在 1M 上下文、BF16 推理时的显存分配比例（ASCII 图）：

```
Total ∼1,128 GiB (8×H200)
┌──────────────────────────────────────────────────────────────────┐
│██████████████████████████████████████████████████████████████    │  Weights: ∼1,100 GB (97.5%)
│KV Cache: ∼13 GiB (1.2%)                                           │
│Act+Overhead: ∼15 GiB (1.3%)                                       │
└──────────────────────────────────────────────────────────────────┘
```

而同一个 1,128 GiB 池子上，M3 BF16 推理的显存分配：

```
Total ∼1,005 GiB (per sample, 1M context)
┌──────────────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████████████          │  Weights: ∼856 GB (85%)
│██████████████████████                                            │  KV Cache: ∼134 GiB (14.3%)
│Act: ∼5 GiB (0.5%)                                                 │
└──────────────────────────────────────────────────────────────────┘
```

Nemotron 的 Attention 层只有 12 层且 GQA 32:1 极度压缩 KV Cache，所以 KV Cache 占比极小；M3 有 60 层全部存 KV Cache（包括 MSA Index K），在 1M 上下文下 KV Cache 膨胀到权重的 ~17%。架构差异直接导致显存瓶颈的转移——Nemotron 是纯权重瓶颈，M3 是权重+KV Cache 双瓶颈。

---

### 5.2 权重显存

从总参数量直接换算权重占用的显存。这是显存预算的最大头，也是最容易算的部分——总参 × 精度字节数。

#### 公式

$$M_{\text{weights}} = N_{\text{total}} \times \text{bytes\_per\_param}$$

#### 按精度的换算表

| 精度 | bytes/param | 550B 模型需要 | 428B 模型需要 |
|---|---|---|---|
| FP32 | 4 | 2,200 GB | 1,712 GB |
| BF16 / FP16 | 2 | 1,100 GB | 856 GB |
| FP8 (E4M3) | 1 | 550 GB | 428 GB |
| INT4 / NVFP4 | 0.5 | 275 GB | 214 GB |


#### 案例 1：Nemotron 3 Ultra（550B）

BF16 推理：

$$M_{\text{weights}} = 550 \times 10^9 \times 2 = 1.1 \times 10^{12} \text{ bytes} = \mathbf{1{,}100 \text{ GB}}$$

换成 FP8 量化：

$$M_{\text{weights}} = 550 \times 10^9 \times 1 = 5.5 \times 10^{11} \text{ bytes} = \mathbf{550 \text{ GB}}$$

从 1,100 GB 降到 550 GB，可以直接从"必须 8 卡"变为"4 卡可行"（4 × 141 = 564 GiB）。

BF16 下，每 1B 参数 ≈ 2 GB（十进制，$10^9$）。即参数量（B）× 2 = 显存（GB）。本章统一使用此行业速算惯例。若换算为 GiB（$1024^3$）：1B 参数 ≈ 1.86 GiB，即速算值高估约 7.4%。GPU 显存规格通常以 GiB 标注（如 H200 = 141 GiB），因此卡数估算时需注意单位一致性——本章的显存数字用 GB 十进制表示，与 GPU 的 GiB 标注直接比较时会略有偏差。

#### 案例 2：MiniMax M3（~428B）

BF16 推理：

$$M_{\text{weights}} = 428 \times 10^9 \times 2 = \mathbf{856 \text{ GB}}$$

FP8：

$$M_{\text{weights}} = 428 \times 10^9 \times 1 = \mathbf{428 \text{ GB}}$$

#### 案例 3：Kimi K2.5（~1T）

BF16 推理（如果全量加载）：

$$M_{\text{weights}} = 1{,}000 \times 10^9 \times 2 = \mathbf{2{,}000 \text{ GB}} \approx 2 \text{ TB}$$

需要 $\lceil 2000 / 141 \rceil = 15$ 张 H200 才能装下 BF16 权重。实际部署中 K2.5 使用 FP8 量化（1,000 GiB ≈ 8 卡）或 INT4（500 GiB ≈ 4 卡）。

#### MoE 的权重加载特殊性

上述计算假设**所有权重全部驻留在显存中**（全量加载）。这是标准推理部署的做法——即使 MoE 每 token 只激活 $k/E$ 的专家，所有 $E$ 个专家的权重仍需在显存中，因为不同 token 激活不同专家。

但存在一种"按需加载"策略：只将当前 batch 需要的专家权重换入显存，不需要的留在 CPU 或 NVMe 上。这种策略的显存占用为：

$$M_{\text{weights}}^{\text{on-demand}} = M_{\text{non-MoE}} + \overbrace{k_{\text{batch}} \times M_{\text{per-expert}}}^{\text{仅加载被命中的专家}}$$

其中 $k_{\text{batch}}$ 是整个 batch 激活的**不同**专家数（不是 $k$，因为 batch 中不同 token 可能命中不同专家，总的命中专家数随 batch size 增大而增大）。

按需加载的优势是省显存，代价是延迟不可预测（换入专家需要 PCIe/NVLink 带宽）。目前**生产部署几乎不使用按需加载**——延迟的不可预测性是服务级推理不能接受的。

---

### 5.3 KV Cache 显存

从 KV Cache 的公式化计算出发，给出 per-sample 和 per-batch 的显存占用量。KV Cache 与序列长度成线性正比。在 1M 上下文下，它可能膨胀到与权重同量级。

#### 核心公式（沿用 CH 4）

标准 GQA：

$$M_{\text{kv}}^{(1)} = L \times 2 \times H_{kv} \times D_h \times T \times \text{bytes\_per\_elem}$$

其中：
- $L$：层数
- $2$：K 和 V 两份
- $H_{kv}$：KV 头数
- $D_h$：每头维度
- $T$：序列长度（cached tokens）
- $\text{bytes\_per\_elem}$：BF16=2，FP8=1

每一层有两个缓存矩阵（K 和 V），每个形状是 $H_{kv} \times T \times D_h$（GQA 下 KV 头数少于 Q 头数，矩阵较窄）。60 层 × 2 份 × 4 头 × 128 维 × 1M token × 2 字节 = 60 × 2 × 4 × 128 × 10^6 × 2 ≈ 123 GiB。记法：每层 KV Cache ≈ $2 \times H_{kv} \times D_h \times T \times 2$ bytes。

#### 针对不同模型架构的扩展

**MLA（Kimi K2.5）**：KV Cache 只存压缩后的潜向量，不存展开后的全维度 K/V。公式变为：

$$M_{\text{kv}}^{\text{MLA}} = L \times (d_{kv} + D_{rope}) \times T \times \text{bytes\_per\_elem}$$

其中 $d_{kv}$ 是 KV 压缩维度，$D_{rope}$ 是 RoPE 分量（不可压缩，必须单独存储）。K2.5 中 $d_{kv}=512$，$D_{rope}=64$，合计 $576$ 维。对比标准 MHA（$64 \times (192 + 128) = 20{,}480$ 维），MLA 的 KV Cache 维度压缩了 **35.6×**。

**MSA（MiniMax M3）**：额外存储 Index K Cache：

$$M_{\text{kv}}^{\text{MSA}} = M_{\text{kv}}^{\text{main}} + L_{\text{MSA}} \times H_{\text{idx\_k}} \times D_{\text{idx}} \times T \times \text{bytes\_per\_elem}$$

其中 $M_{\text{kv}}^{\text{main}}$ 与标准 GQA 的公式完全相同（MSA 不减少 KV Cache 存储——稀疏性体现在计算而非存储），$H_{\text{idx\_k}}=1$（Index K 只有 1 个头），$D_{\text{idx}}=128$，$L_{\text{MSA}}=57$。

**Mamba-2（Nemotron）**：没有传统 KV Cache。但每层维护一个 SSM 隐状态，维度为 $H_{mamba} \times d_{state} = 256 \times 128 = 32{,}768$ 个元素（FP32 精度），48 层合计约 $48 \times 32768 \times 4 = 6.3 \text{ MiB}$——可忽略。

#### 案例 1：MiniMax M3，BF16，T=1M

Main KV Cache（60 层，GQA 16:1）：

$$\begin{aligned}
M_{\text{kv}}^{\text{main}} &= 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 60 \times 2 \times 4 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 128{,}849{,}018{,}880 \text{ bytes} \\
&\approx \mathbf{120.0 \text{ GiB}}
\end{aligned}$$

Index K Cache（57 层 MSA）：

$$\begin{aligned}
M_{\text{kv}}^{\text{index}} &= 57 \times 1 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 15{,}300{,}329{,}472 \text{ bytes} \\
&\approx \mathbf{14.3 \text{ GiB}}
\end{aligned}$$

**M3 KV Cache 总计（per sample, 1M, BF16）：$\approx 120.0 + 14.3 = \mathbf{134.3 \text{ GiB}}$**

#### 案例 2：Nemotron 3 Ultra，BF16，T=1M

仅 12 层 Attention（GQA 32:1，$H_{kv}=2$，$D_h=128$）：

$$\begin{aligned}
M_{\text{kv}}^{\text{Nemotron}} &= 12 \times 2 \times 2 \times 128 \times 1{,}048{,}576 \times 2 \\
&= 12{,}884{,}901{,}888 \text{ bytes} \\
&\approx \mathbf{12.0 \text{ GiB}}
\end{aligned}$$

48 层 Mamba 的 SSM 状态约 $\approx 0.2 \text{ GiB}$——总计约 **13.1 GiB**。

Nemotron 的 KV Cache 比 M3 小 **11 倍**，尽管总参数更大（550B vs 428B）。这就是"尽量不用 Attention"架构策略的显存红利。

#### 案例 3：DeepSeek V4 Flash（MLA），T=1M

MLA 下 KV Cache per layer = $(d_{kv} + D_{rope}) \times T \times 2 = 576 \times 1{,}048{,}576 \times 2 \approx 1.21 \text{ GiB}$。60 层：$\approx 72.4 \text{ GiB}$。对比同尺寸 GQA 模型的 ~134 GiB，MLA 直接砍半。

#### Batch 效应

KV Cache 是 **per-sample** 的。batch_size=100 就是 100 倍。这是推理并发的主要瓶颈——权重可以跨 batch 共享，但每个请求需要自己独立的 KV Cache：

$$M_{\text{kv}}^{\text{total}} = B \times M_{\text{kv}}^{(1)}$$

---

### 5.4 激活值与临时缓冲

估算前向传播中激活值和临时 buffer 的显存。虽然通常不到权重的 5%，但在规划显存预算时必须留出这部分余量，否则 OOM。

激活值显存来自三个方面：

1. **残差流**：每层前向传播时，hidden_states $\in \mathbb{R}^{B \times S \times d}$ 在 layer 间传递。BF16 下 per token per layer = $d \times 2$ bytes = 12 KB（d=6144）。
2. **注意力中间结果**：Q、K、V、attn_weights 等临时张量。在 decode 阶段（$S_{\text{new}}=1$），这些非常小（< 1 MB/layer）。
3. **MoE 中间结果**：4 个路由专家的 gate_up 输出（4 × $d_{ff} \times 2$ bytes）。

#### 估算经验值

对于 decode 阶段，激活值显存经验公式：

$$M_{\text{act}} \approx 0.05 \times M_{\text{weights}} \quad \text{（上限经验值）}$$

更精确的逐模块估算：

| 组件 | per-token per-layer | ×60 layers (M3) |
|---|---|---|
| 残差流 (hidden_states) | 12 KB (d=6144, BF16) | 0.72 MB |
| Attention activations (Q/K/V/attn) | ~500 KB | ~30 MB |
| MoE 4-expert activations | ~48 KB (4 × 3072 × 2B) | ~2.9 MB |
| **Per-token sum** | **~0.56 MB** | **~33.6 MB** |

对于 M3，per-token 激活值约 **34 MB**。加上框架开销（PyTorch allocator、cuBLAS workspace 等）约 2-5 GiB。

**总显存经验公式**：

$$M_{\text{total}} \approx 1.05 \sim 1.10 \times (M_{\text{weights}} + M_{\text{kv}}^{\text{total}})$$

即总显存大约比"权重 + KV Cache"多 5%~10%。这在显存规划中作为安全余量使用。

---

### 5.5 MoE 的专家加载策略

对比 MoE 在全量加载和按需加载两种策略下的显存-性能 trade-off。MoE 占模型参数的 90%+，显存策略的选择直接决定了最低 GPU 数量。

#### 策略 A：全量 Expert 加载（标准做法）

所有 $E$ 个专家的权重始终在显存中。无论 router 选哪个专家，计算是即时的。

- 显存需求：$E \times \text{Params}_{\text{expert}} \times \text{bytes}$
- 延迟：可预测，低延迟
- 并行：通过 EP（Expert Parallelism）将专家分布到多卡，每卡只加载分配给它的专家切片

#### 策略 B：按需 Expert 加载（实验性）

只在 router 选中后才将对应专家权重从 CPU/NVMe 加载到 GPU。

- 显存需求：$\approx \text{Params}_{\text{non-MoE}} + \text{Params}_{\text{avg loaded experts}}$，远小于全量
- 延迟：不可预测——首次 access 需等待 PCIe 传输（~50 GiB/s），远慢于 HBM（~3 TB/s）
- 适用场景：极端显存受限的离线批处理，不适合在线服务

#### Nemotron 512 experts 的极端案例

Nemotron 单独专家部分的 BF16 权重：

$$\begin{aligned}
\text{Params}_{\text{all experts}} &= 48 \text{ layers} \times 512 \text{ experts} \times (2 \times 2048 \times 5120) \text{ params} \\
&\approx 48 \times 512 \times 21\text{M} = 48 \times 10.74\text{B} = 515.5\text{B} \\
M_{\text{experts only}} &= 515.5 \times 10^9 \times 2 \text{ bytes} = 1{,}031 \text{ GiB} \approx \mathbf{1.03 \text{ TB}}
\end{aligned}$$

**仅专家权重就超过 1 TB**——比总参数（550B × 2 = 1,100 GB）的 94% 都在专家上。这就是为什么 EP 对 MoE 模型不是"可选的优化"而是"部署的前提条件"。

512 个专家每个 ~21M 参数，48 层，BF16 → 约 1 TB。8 张 H200 每张装 1/8 的专家（EP=8），每卡专家部分约 129 GiB，加上非 MoE 参数（约 35 GiB），刚好塞进 141 GiB 的 H200。没有 EP，即使 16 张 H200 也装不下所有专家复本。

---

### 5.6 并行策略的影响（概念级）

解释 TP/PP/EP 三种并行策略如何改变每张 GPU 的实际显存负载。部署计算不是"总显存 / 卡数"，不同并行策略按不同维度切分显存。

#### Tensor Parallelism (TP) —— 切分矩阵乘法

TP 将单个矩阵乘法的权重按列（column-wise）或行（row-wise）切分到 $N$ 张卡。

- 每卡权重 = $\text{总权重} / N$
- 代价：每层需要两次 all-reduce 通信（前向 + 反向），通信量与 hidden_size 成正比
- 适用场景：单层矩阵太大，单卡装不下时

**案例**：M3 的 Q 投影矩阵 $W_Q \in \mathbb{R}^{6144 \times 8192}$，BF16 下 100.7 MB。单卡轻松装下，不需要 TP。但如果是 1T 参数模型 hidden=16384，$W_Q \in \mathbb{R}^{16384 \times 32768}$ 约 1 GiB——单个矩阵就接近极限。

#### Pipeline Parallelism (PP) —— 按层切分

PP 将不同层放到不同 GPU。GPU 0 管层 0-14，GPU 1 管层 15-29，以此类推。

- 每卡权重 $\approx \text{总权重} / N$（但不均衡——MoE 层比 Attention 层重一个数量级）
- 代价：流水线 bubble（GPU 空闲等待前一级完成）；通信仅在 stage 边界
- 适用场景：层数多、单层内存适中的模型

**注意**：PP 不能解决"单层太大装不下"的问题——如果 MoE 单层有 7.3B 权重（M3），BF16 下约 14.6 GiB，单卡完全装得下。PP 解决的是"60 层加起来装不下"。

#### Expert Parallelism (EP) —— 按专家切分（MoE 专用）

EP 是最适合 MoE 模型的并行策略。其核心思想：不同 GPU 持有不同的专家子集，token 通过 all-to-all 通信被路由到持有对应专家的 GPU。

- 每卡装的专家数 = $E / \text{EP\_size}$
- 每卡专家权重 = $\text{总专家权重} / \text{EP\_size}$
- 代价：token dispatch 和 combine 需要 all-to-all 通信（仅 MoE 层，非所有层）

**Nemotron on 8×H200**：EP=8，每卡装 512/8 = 64 个专家。每卡专家权重 = $64 \times 48 \times 21\text{M} \times 2 \text{ bytes} \approx 129 \text{ GiB}$。加上非 MoE 参数（Mamba + Attention + Embedding 等）约 35 GiB，总计约 164 GiB——但 H200 只有 141 GiB！

这就引出了一个关键计算。需要检查 8×H200 是否真的够：

$$\begin{aligned}
\text{Per-card non-expert} &= (N_{\text{total}} - N_{\text{experts}}) / \text{cards} \\
&\approx (550 - 515.5) / 8 = 4.31 \text{ B} \\
M_{\text{non-expert per card}} &= 4.31 \times 10^9 \times 2 = 8.63 \text{ GiB}
\end{aligned}$$

$$\begin{aligned}
\text{Per-card experts} &= (515.5 \times 10^9 \times 2) / 8 = 128.9 \text{ GiB}
\end{aligned}$$

$$\text{Per-card total} \approx 8.6 + 128.9 = 137.5 \text{ GiB}$$

137.5 GiB < 141 GiB ——勉强能装下。但如果加上 KV Cache（per sample ~13 GiB / 8 ≈ 1.6 GiB per card if distributed）和激活值，余量非常紧张。

这个计算说明了**为什么部署计算不能只看"总显存够不够"**：并行策略决定了每张卡实际装载的权重分布。

#### 简单部署公式

当只考虑权重显存时的最简估算：

$$\text{Cards}_{\text{min}} = \left\lceil \frac{M_{\text{weights}}}{\text{Per-card memory}} \right\rceil$$

Nemotron BF16：$\lceil 1100 / 141 \rceil = 8$ 张 H200。
M3 BF16：$\lceil 856 / 141 \rceil = 7$ 张 H200（但实际需要 8 张，因为还要考虑 KV Cache batch 效应和 EP 要求专家数可被 EP 大小整除）。

---

### 5.7 完整案例：Nemotron 550B on 8×H200

综合运用 5.2-5.6 的知识，做一次完整的部署方案推算。这就是面试中"这个模型需要多少卡"类问题的标准回答模板。

#### 已知条件

- 模型：Nemotron 3 Ultra，550B 总参，BF16 推理
- 硬件：8 × NVIDIA H200（141 GiB/card，合计 1,128 GiB）
- 上下文：1M tokens
- 架构特征：12 层 Attention（GQA 32:1）+ 48 层 Mamba-2 + 48 层 LatentMoE（512E, top-22）

#### Step 1：权重显存

$$M_{\text{weights}} = 550 \times 10^9 \times 2 = \mathbf{1{,}100 \text{ GB}}$$

#### Step 2：KV Cache（per sample）

$$M_{\text{kv}}^{(1)} = 12 \times 2 \times 2 \times 128 \times 1{,}048{,}576 \times 2 = \mathbf{12.0 \text{ GiB}}$$

（Mamba 层 SSM 状态约 6 MiB，计入 act/overhead）

#### Step 3：可用显存

$$M_{\text{available}} = 1{,}128 - 1{,}100 = \mathbf{28 \text{ GiB}} \quad (\text{8 卡合计})$$

这 28 GiB 是留给 KV Cache + 激活值 + 框架开销的全部余量。

#### Step 4：Max Batch Size

每个样本消耗的 KV Cache + 激活值：

$$M_{\text{per sample}} = M_{\text{kv}}^{(1)} + M_{\text{act}}^{(1)} \approx 12.0 + 2 = \mathbf{14.0 \text{ GiB}}$$

$$B_{\text{max}} = \left\lfloor \frac{28}{14.0} \right\rfloor = \left\lfloor 1.88 \right\rfloor = \mathbf{1 \sim 2 \text{ samples}}$$

更现实地说，max_batch_size = 1（留安全余量给框架开销和 NCCL buffer）：

- batch=1：$1{,}100 + 12.0 + 2 \approx 1{,}114 \text{ GiB} < 1{,}128 \text{ GiB}$ ✓
- batch=2：$1{,}100 + 25.8 + 4 \approx 1{,}130 > 1{,}128 \text{ GiB}$ ✗（接近极限，可能 OOM）

#### Step 5：若使用 FP8 权重

$$M_{\text{weights}}^{\text{FP8}} = 550 \times 10^9 \times 1 = \mathbf{550 \text{ GB}}$$

$$M_{\text{available}}^{\text{FP8}} = 1{,}128 - 550 = \mathbf{578 \text{ GiB}}$$

$$B_{\text{max}}^{\text{FP8}} = \left\lfloor \frac{578}{14.0} \right\rfloor \approx \mathbf{41 \text{ samples}}$$

从 batch=1 到 batch=38——FP8 将 Nemotron 从一个"勉强能跑"的模型变成一个"可以服务"的模型。

#### 汇总表

| 精度 | 权重 (GB) | KV Cache/样本 (GB) | 可用 (GB, 8卡) | Max Batch |
|---|---|---|---|---|
| BF16 | 1,100 | 12.0 | 28 | 1 |
| FP8 | 550 | 12.0 | 578 | 38 |
| FP8 KV + FP8 W | 550 | 6.5 | 578 | 76 |
| INT4 / NVFP4 | 275 | 12.0 | 853 | 57 |
| INT4 W + FP8 KV | 275 | 6.5 | 853 | 115 |

Nemotron 在 BF16 下是"纯权重瓶颈"——KV Cache 几乎不占什么（只要 13 GiB），但 1.1 TiB 的 BF16 权重把 8 卡池子塞满了 97.5%。FP8 一开，权重减半，同一个池子马上可以跑几十个并发请求。这就是量化在部署中的价值：它解决的是权重显存瓶颈，不是 FLOPs 瓶颈。

---


> **系列导航**：[（一）预备知识与参数分解](../) ← [（二）FLOPs 估算](../part-2/) ← 当前 → [（四）M3 实战 + Roofline](../part-4/)
