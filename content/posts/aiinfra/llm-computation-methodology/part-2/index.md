+++
date = '2026-06-22T09:05:00+08:00'
draft = false
title = 'LLM 系统分析方法论（二）：FLOPs 估算'
categories = ['aiinfra']
tags = ['computation', 'flops', 'parameters', 'methodology']
series = 'llm-computation-methodology'
series_order = 2
math = true
summary = 'FLOPs 完整估算：从矩阵乘法到 Attention 到 FFN，覆盖 Full Attention / MSA / MLA / Mamba-2 / GDN 六种注意力架构。'
+++

## CH 3 FLOPs 估算

> **读者定位**：已完成 CH1-2 的参数计算，目标是推导 prefilling / decoding 的单 token 计算量，并理解不同架构（Full Attn / MSA / MLA / Mamba-2）的 FLOPs 差异根源。

> **系列导航**：[（一）预备知识与参数分解](../part-1/) ← 当前 → [（三）KV Cache 与推理显存](../part-3/) → [（四）M3 实战 + Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)

---

## 3.1 通用原理

> **本文各节描述的是前向（推理）FLOPs。训练 FLOPs 需要乘以训练系数（线性投影 ×6，QK ×4，AV ×3，Indexer ×1）——系数推导见 [§3.12 从推理到训练：系数体系](#312-从推理到训练系数体系)。**

建立”前向 FLOPs = 所有权重矩阵乘法之和”的底层逻辑。参数量是”模型存了多少数”，FLOPs 是”每次前向要算多少下”——两者直接决定推理延迟和硬件成本。

### 核心公式

单层 FLOPs = 该层内所有矩阵乘法的 $2 \times m \times n \times k$ 之和（见 1.2 节）。

$$\text{FLOPs}_{\text{total}} = \sum_{l=1}^{L} \text{FLOPs}_{\text{attn}}^{(l)} + \text{FLOPs}_{\text{ffn}}^{(l)} + \text{FLOPs}_{\text{norm}}^{(l)}$$

其中 norm（RMSNorm / LayerNorm）的 FLOPs 为 $4 \times d$（乘 $\gamma$ + 加 $\beta$），在大模型中可忽略（$d=8192$ 时 $\approx 32\text{K FLOPs}$，而 Q 投影是 $\approx 134\text{M FLOPs}$）。

### Prefill vs Decode

- **Prefill**：输入 $T_{in}$ 个 token，所有层对所有 token 完整计算一次。总 FLOPs 正比于 $T_{in}$（线性部分）或 $T_{in}^2$（注意力部分）。
- **Decode**：每次只产生 1 个新 token，但需要 attend 到所有历史 token（$T_{total}$）。**只有新 token 的 QKV 需要投影**，但 QK 点积和 V 加权要覆盖全部历史。

$$\text{FLOPs}_{\text{decode\_per\_token}} = \sum_{l=1}^{L} \text{FLOPs}_{\text{new\_token}}^{(l)}$$

Prefill 是“一口气读完整本书再回答问题”，Decode 是“每次多读一个字就要把所有笔记翻一遍”。前者吞吐高但延迟长，后者每步轻量但被历史长度拖累。Attention 的 O(T²) 项只在 Prefill 是全量爆炸，Decode 时变成 O(T)（因为只有 1 个 query）。

### 单 Token FLOPs 计算范式

对每个矩阵乘法，固定范式为：

$$\text{FLOPs} = 2 \times (\text{输出第一维}) \times (\text{输出第二维}) \times (\text{被缩并的公共维度})$$

**案例**：Attention 层 Q 投影，输入 hidden $[1, d]$，权重 $W_Q [d, H_q \times D_h]$：

$$\text{FLOPs}_Q = 2 \times 1 \times (H_q \times D_h) \times d$$

Nemotron 12 个 Attention 层之一（$d=8192$，$H_q=64$，$D_h=128$）：

$$\text{FLOPs}_Q = 2 \times 1 \times (64 \times 128) \times 8192 = 2 \times 8192 \times 8192 = 134{,}217{,}728 \approx 134.2\text{M FLOPs/token}$$

每产生一个 token，Q 投影就要把 8192 维向量乘上 $8192 \times 8192$ 的矩阵——相当于做 8192 次 8192 维的内积。这就是一个 token 经过一层 Attention 的“起步价”。

---

## 3.2 Full Attention FLOPs

逐项拆解标准 Attention（含 GQA）的四部分 FLOPs，区分线性项和平方项。不理解 O(T²) 项从哪里来，就无法理解为什么长上下文推理会变慢——以及为什么 MSA、Mamba 等替代架构有意义。

### 3.2.1 QKV 投影（线性项，O(T)）

投影部分在 Prefill 时随 T 线性增长，在 Decode 时是**常数**（只投影新 token）。

$$\text{FLOPs}_{Q} = 2 \times d \times (H_q \times D_h) \times T_{\text{new}}$$

$$\text{FLOPs}_{K} = 2 \times d \times (H_{kv} \times D_h) \times T_{\text{new}}$$

$$\text{FLOPs}_{V} = 2 \times d \times (H_{kv} \times D_h) \times T_{\text{new}}$$

**GQA 的精髓**：K 和 V 投影的输出维度是 $H_{kv} \times D_h$ 而非 $H_q \times D_h$——这是 GQA 相比于 MHA 在计算量（而不仅是参数量）上的直接节省。

**案例代入**：Nemotron Attention 层（GQA 32:1，$d=8192$，$H_q=64$，$H_{kv}=2$，$D_h=128$）。

**Prefill（$T=4096$）**：

$$\text{FLOPs}_{Q} = 2 \times 8192 \times (64 \times 128) \times 4096 = 2 \times 8192 \times 8192 \times 4096$$

$$= 2 \times 67{,}108{,}864 \times 4096 = 549{,}755{,}813{,}888 \approx 550 \text{ GFLOPs}$$

$$\text{FLOPs}_{K} = 2 \times 8192 \times (2 \times 128) \times 4096 = 2 \times 8192 \times 256 \times 4096$$

$$= 2 \times 2{,}097{,}152 \times 4096 = 17{,}179{,}869{,}184 \approx 17.2 \text{ GFLOPs}$$

$$\text{FLOPs}_{V} = 17.2 \text{ GFLOPs} \quad (\text{与 K 相同})$$

Prefill 一次性投影所有 4096 个 token 的 Q、K、V。注意 K 投影（17 GFLOPs）只占 Q 投影（550 GFLOPs）的约 3%——因为 $H_{kv} = 2$ 只有 $H_q = 64$ 的 1/32。

**Decode（$T_{\text{new}}=1$，$T_{\text{total}}=1\text{M}$）**：

$$\text{FLOPs}_{Q} = 2 \times 8192 \times (64 \times 128) \times 1 = 134{,}217{,}728 \approx 134.2\text{M FLOPs}$$

$$\text{FLOPs}_{K} = 2 \times 8192 \times (2 \times 128) \times 1 = 4{,}194{,}304 \approx 4.2\text{M FLOPs}$$

$$\text{FLOPs}_{V} = 4.2\text{M FLOPs}$$

QKV 投影在 decode 时总共 $\approx 142.6\text{M FLOPs}$——与上下文长度**无关**。

QKV 投影就像“打字”——每个新 token 只需要把自己的向量投影一次。历史 token 的 K 和 V 投影结果被缓存在 KV cache 里，不用重算。

### 3.2.2 QK 点积（平方项，O(T²) 的根源）

$$\text{FLOPs}_{\text{QK}} = 2 \times H_q \times T_{\text{new}} \times T_{\text{total}} \times D_h$$

**Prefill（$T=T_{\text{new}}=T_{\text{total}}=4096$，causal mask 下约计算一半）**：

$$\text{FLOPs}_{\text{QK}} = 2 \times 64 \times 4096 \times \frac{4096}{2} \times 128 = 2 \times 64 \times 4096 \times 2048 \times 128$$

$$= 2 \times 64 \times 8{,}388{,}608 \times 128 = 137{,}438{,}953{,}472 \approx 137 \text{ GFLOPs}$$

（精确无 causal 时为 275 GFLOPs，causal mask 下约折半。）

**Decode（$T_{\text{new}}=1$，$T_{\text{total}}=1\text{M}$）——这就是长上下文问题的核心**：

$$\text{FLOPs}_{\text{QK}} = 2 \times 64 \times 1 \times 1{,}000{,}000 \times 128$$

$$= 2 \times 64 \times 128 \times 10^6 = 16{,}384 \times 10^6 = 1.6384 \times 10^{10} \approx 16.4 \text{ GFLOPs}$$

当上下文达到 1M tokens 时，仅一个 Attention 层的 QK 点积就需要 **164 亿次浮点运算**。对于有 12 个 Attention 层的 Nemotron：$12 \times 16.4 \approx 197 \text{ GFLOPs}$，仅此一项就超过了 QKV 投影（12 × 142.6M ≈ 1.7 GFLOPs）两个数量级。

QK 点积是把新 token 的一个 query 与缓存中所有 1M 个 key 逐一算相似度。1M 个 key，每个 128 维，每个维度一次乘法+一次加法=$2 \times 128 = 256$ FLOPs，64 个 head 各做一次，总计就是 $64 \times 1\text{M} \times 256 = 16.4\text{GFLOPs}$。这就是 Attention 在长上下文下“喘不过气”的根本原因。

### 3.2.3 V 加权（同样是 O(T) 项，decode 中体量等于 QK）

$$\text{FLOPs}_{\text{V}} = 2 \times H_q \times T_{\text{new}} \times T_{\text{total}} \times D_h$$

**Decode（$T_{\text{new}}=1$，$T_{\text{total}}=1\text{M}$）**：

$$\text{FLOPs}_{\text{V}} = 2 \times 64 \times 1 \times 1{,}000{,}000 \times 128 = 16.4 \text{ GFLOPs}$$

与 QK 点积**等量级**！原因：注意力权重要乘上 V 矩阵——1M 个 value 向量，每个 128 维，64 个 head。计算量路径：$[1, 64, 1, 1\text{M}] \times [1, 64, 1\text{M}, 128] \to [1, 64, 1, 128]$，缩并维度是 1M。

算完“每个历史 token 有多相关”（QK 点积）后，还要把 1M 个 value 向量按相关性加权平均。这个“加权平均”的运算量跟“计算相似度”一样大——都是 $2 \times H \times T \times D_h$。所以 Attention 的 decode 成本 = QK + V ≈ $4 \times H \times T \times D_h$。

### 3.2.4 输出投影（线性项，O(T)）

$$\text{FLOPs}_{\text{O}} = 2 \times d \times (H_q \times D_h) \times T_{\text{new}}$$

**decode 时为常数**（Nemotron）：$\text{FLOPs}_O = 2 \times 8192 \times 8192 \times 1 = 134.2\text{M FLOPs}$

与 Q 投影相同——因为输入和输出的维度都是 $d \times d$。

### 3.2.5 单层 Full Attention Decode FLOPs 汇总

以 Nemotron Attention 层（GQA 32:1，T=1M）为例：

| 组件 | 公式 | $T=1\text{M}$ 时 FLOPs | 占比 |
|---|---|---|---|
| Q 投影 | $2 \times d \times (H_q \times D_h)$ | 134.2M | 0.4% |
| K 投影 | $2 \times d \times (H_{kv} \times D_h)$ | 4.2M | 0.01% |
| V 投影 | $2 \times d \times (H_{kv} \times D_h)$ | 4.2M | 0.01% |
| QK 点积 | $2 \times H_q \times T \times D_h$ | **16.4G** | **49.7%** |
| V 加权 | $2 \times H_q \times T \times D_h$ | **16.4G** | **49.7%** |
| O 投影 | $2 \times d \times (H_q \times D_h)$ | 134.2M | 0.4% |
| **单层合计** | — | **~33.1G** | 100% |

**关键观察**：在 1M 上下文下，Attention 层 99.4% 的计算量花在 QK 点积和 V 加权上——这两个 O(T) 项（decode 时）。投影部分是常数，可以忽略。**任何想加速长上下文推理的架构，都是从这两个 O(T) 项下手。**

### 3.2.6 GQA 对 FLOPs 的影响

GQA 降低了 K 和 V 投影的 FLOPs（$H_{kv}$ 替代 $H_q$），**但不降低 QK 点积和 V 加权的 FLOPs**。原因是 K 和 V 在注意力计算前会被 `repeat_kv` 扩展到与 Q 相同的头数：

```python
# 标准 GQA 实现（transformers 源码）
K = K.repeat_interleave(H_q // H_kv, dim=1)  # [B, H_kv, T, D] -> [B, H_q, T, D]
```

所以 QK 点积的规模仍然是 $2 \times H_q \times T \times D_h$——**与 MHA 完全相同**。

GQA 节省的是：
- K、V 投影的 FLOPs（节省比例 $\frac{H_q}{H_{kv}}$ 倍，如 64/2=32×）
- KV cache（同样 32×）

GQA 节省的 **不是**：
- QK 点积的 FLOPs
- V 加权的 FLOPs

GQA 就像“出版社印了 64 份杂志（Q head），但只审了 2 份稿子（KV head），审稿费省了 32×，但印杂志的成本（读者阅读 = QK 点积）没省——因为每份杂志都要发给所有读者看。”

---

## 3.3 MSA 稀疏 Attention FLOPs（MiniMax M3）

推导 M3 的 Multi-stage Sparse Attention 计算量，理解“用廉价 Index Branch 筛选 + 昂贵 Main Branch 只在筛选区域计算”的 FLOPs 逻辑。M3 在 1M 上下文时实现约 30× 的 decode 加速——这是稀疏 Attention 的标杆案例。

### 3.3.1 MSA 架构概述

M3 的 MSA 将 Attention 分为两个分支：

- **Index Branch**（廉价筛选器）：用少量 head（$H_{\text{idx}} = 4$）在全部 T 个 token 上做 QK 评分 + max-pool + top-k，选出 16 个 block（每 block 128 token，共 $16 \times 128 = 2048$ 个候选 token）。
- **Main Branch**（精准计算器）：用全部 head（$H_q = 64$）**只在 2048 个入选 token 上**做完整 Attention。

M3 有 60 层：3 层 Full Attention（Layer 0-2）+ 57 层 MSA（Layer 3-59）。

### 3.3.2 Index Branch FLOPs

> **训练提示**：MSA Indexer 在源码中被 `@torch.no_grad()` 包裹（`modeling_minimax_m3_vl.py:L695`），训练时不计算梯度——所有 Indexer 操作的系数为 **1（仅前向）**，不是 6 或 7。详见 [§3.13](#313-indexer-与-router-的-no_grad-特性)。

维度回顾：$d = 6144$，$H_{\text{idx}} = 4$，$D_{\text{idx}} = 128$，$H_q = 64$，$D_h = 128$。

**(1) Index Q 投影**

$$\text{FLOPs}_{\text{idx\_Q}} = 2 \times d \times (H_{\text{idx}} \times D_{\text{idx}}) \times T_{\text{new}}$$

Decode（$T_{\text{new}}=1$）：

$$\text{FLOPs}_{\text{idx\_Q}} = 2 \times 6144 \times (4 \times 128) \times 1 = 2 \times 6144 \times 512 = 6{,}291{,}456 \approx 6.3\text{M FLOPs}$$

**(2) Index K 投影**

Index K 只有一个 head 的维度（128），4 个 index head 共享同一个 K：

$$\text{FLOPs}_{\text{idx\_K}} = 2 \times d \times D_{\text{idx}} \times T_{\text{new}} = 2 \times 6144 \times 128 \times 1 = 1{,}572{,}864 \approx 1.6\text{M FLOPs}$$

**(3) Index QK 评分（O(T²) in prefill，O(T) in decode）**

这是 Index Branch 的计算主体。Index Branch 用 4 个 head 在全序列上做 QK 点积。

**Decode（$T_{\text{new}}=1$，$T_{\text{total}}=1\text{M}$）**：

$$\text{FLOPs}_{\text{idx\_QK}} = 2 \times H_{\text{idx}} \times 1 \times T \times D_{\text{idx}} = 2 \times 4 \times 1 \times 10^6 \times 128$$

$$= 2 \times 512 \times 10^6 = 1{,}024{,}000{,}000 \approx 1.02\text{ GFLOPs}$$

**对比 Full Attention 的 QK 点积**（如果用全部 64 个 head 做全序列评分）：

$$\text{FLOPs}_{\text{full\_QK}} = 2 \times 64 \times 1 \times 10^6 \times 128 = 16{,}384 \times 10^6 \approx 16.4\text{ GFLOPs}$$

Index Branch 的 QK 评分仅需要 1.02 GFLOPs，而 Full Attention 需要 16.4 GFLOPs——**减少了 16×**。原因直截了当：4 个 head vs 64 个 head，$64/4 = 16$。

**这就是 Index Branch 设计的精妙之处**：用 16× 更便宜的计算，筛选出哪些 token 值得做完整的 64-head Attention。

**(4) Max-pool + Top-k**

Max-pool 将分数按 block 聚合（每 128 token 一个 block，共 $T/128$ 个 block），再选出 top-16 个 block。这部分本质是遍历和排序，FLOPs $\approx T/128 \times \log(16)$，约 $10^4$ 级别，完全可忽略。

### 3.3.3 Main Branch FLOPs

Main Branch 的核心：只在入选的 2048 个 token 上做完整 Attention。

$$\text{访问 token 数} = \text{block\_size} \times \text{top\_k\_blocks} = 128 \times 16 = 2048$$

**(1) Main QK 点积**

$$\text{FLOPs}_{\text{main\_QK}} = 2 \times H_q \times T_{\text{new}} \times T_{\text{selected}} \times D_h$$

$$\text{Decode} = 2 \times 64 \times 1 \times 2048 \times 128 = 2 \times 64 \times 262{,}144$$

$$= 33{,}554{,}432 \approx 33.6\text{M FLOPs}$$

**关键对比**：Full Attention 的 QK = $16.4\text{G FLOPs}$，MSA Main QK = $33.6\text{M FLOPs}$。**加速比** = $16.4\text{G} / 33.6\text{M} \approx 488\times$（T=1M 时，仅 QK 部分）。

**(2) Main V 加权**

$$\text{FLOPs}_{\text{main\_V}} = 2 \times H_q \times T_{\text{new}} \times T_{\text{selected}} \times D_h = 33.6\text{M FLOPs}$$

与 Main QK 对称。

### 3.3.4 MSA 单层 Decode FLOPs 汇总（T=1M）

| 组件 | FLOPs | 类别 |
|---|---|---|
| Index Q 投影 | 6.3M | 常数 |
| Index K 投影 | 1.6M | 常数 |
| Index QK 评分 | 1.02G | O(T)，但 16× 小 |
| Index max-pool + top-k | ~0 | 可忽略 |
| Main Q 投影 | $2 \times 6144 \times (64 \times 128) = 100.7\text{M}$ | 常数 |
| Main K 投影 | $2 \times 6144 \times (4 \times 128) = 6.3\text{M}$ | 常数（GQA 16:1） |
| Main V 投影 | 6.3M | 常数 |
| Main QK 点积 | 33.6M | 常数（仅 2048 个 token） |
| Main V 加权 | 33.6M | 常数 |
| Main O 投影 | $2 \times 6144 \times (64 \times 128) = 100.7\text{M}$ | 常数 |
| **总计** | **~1.31G** | — |

对比 Full Attention 层的 $\approx 33.1\text{G FLOPs}$（相同 $d$, $H_q$ 配置在 T=1M 下），MSA 单层仅需 $\approx 1.31\text{G FLOPs}$——**加速约 25×**。

MSA 单层最大的开销是 Index QK 评分（1.02G，占 78%），这一项仍然随 T 线性增长——但它是用 4 个 head 而非 64 个，系数差距是 16×。

### 3.3.5 总体加速比

**Decode 场景（T=1M）**：

对于 M3 的 57 层 MSA + 3 层 Full Attention：
- 3 层 Full Attention：$3 \times 33.1\text{G} \approx 99.3\text{G FLOPs}$（$d=6144$, $H_q=64$, $H_{kv}=4$）
- 57 层 MSA：$57 \times 1.31\text{G} \approx 74.7\text{G FLOPs}$
- 总计：$\approx 174\text{G FLOPs}$ 用于 Attention 部分

假如同样的 60 层全部是 Full Attention：
- $60 \times 33.1\text{G} \approx 1986\text{G FLOPs} \approx 1.99\text{T FLOPs}$
- 加速比 $\approx 1986 / 174 \approx 11.4\times$（仅 Attention 部分）

**Prefill 场景（T=1M，causal）**，加速更显著：
- Index QK 的 O(T²) 部分：$2 \times 4 \times (10^6)^2/2 \times 128 \approx 5.12 \times 10^{14}$ FLOPs/层
- Full Attention QK 的 O(T²) 部分：$2 \times 64 \times (10^6)^2/2 \times 128 \approx 8.19 \times 10^{15}$ FLOPs/层
- Main Branch QK：$2 \times 64 \times 10^6 \times 2048 \times 128 \approx 3.36 \times 10^{13}$ FLOPs/层（常数，不随 T² 增长）
- 加速比 $\approx 8.19 \times 10^{15} / (5.12 \times 10^{14} + 3.36 \times 10^{13}) \approx 15\times$（仅 QK 部分）

综合其他恒定开销，实际整体 decode 加速约 **2-5×**，Prefill 加速约 **10-20×**（取决于序列长度和 overhead 比例）。论文声称的 30× 是 decode 场景下 Attention 部分 QK+V 的加速。

MSA 的哲学是“先粗筛再精算”。花 1 GFLOPs（Index Branch）扫一眼全场，发现最有戏的 2048 个 token，然后花 67 MFLOPs（Main QK+V）在这 2048 个 token 上精算。而 Full Attention 要花 33 GFLOPs 在所有 1M 个 token 上精算。前者总花费 $\approx 1.1\text{G}$，后者 $\approx 33\text{G}$，高下立判。

---

## 3.4 MLA FLOPs（Kimi K2.5 / DeepSeek V4）

推导 Multi-head Latent Attention 的 FLOPs，区分低秩投影的线性节省和 QK 点积的不变性。MLA 的卖点是“省 KV cache”而非“省 FLOPs”——但低秩投影确实也节省了一部分线性 FLOPs。

### 3.4.1 MLA 计算流程回顾

以 Kimi K2.5 为例（$d=7168$，$d_{kv}=512$，$d_q=1536$，$H=64$，$D_{\text{nope}}=128$，$D_{\text{rope}}=64$，$D_v=128$）：

**MLA 的两阶段计算**：
1. **压缩阶段**：hidden $\to$ latent（$W_{kv\_a}$, $W_{q\_a}$）
2. **解压阶段**：latent $\to$ per-head K, V, Q（$W_{kv\_b}$, $W_{q\_b}$）
3. **RoPE 直接投影**：hidden $\to$ per-head Q/K rope（$W_{q\_rope}$，不经过 latent）

### 3.4.2 KV 侧 FLOPs（线性项节省的来源）

**(1) KV 压缩投影 $W_{kv\_a}$**

$$W_{kv\_a}: [d] \to [d_{kv} + D_{\text{rope}}] = 7168 \to 512 + 64 = 576$$

$$\text{FLOPs}_{kv\_a} = 2 \times d \times (d_{kv} + D_{\text{rope}}) \times T_{\text{new}}$$

Decode：$= 2 \times 7168 \times 576 \times 1 = 8{,}257{,}536 \approx 8.3\text{M FLOPs}$

这个投影产生两部分输出：
- 前 512 维：压缩的 KV latent，进入 $W_{kv\_b}$ 解压
- 后 64 维：K 的 RoPE 分量（不压缩），直接用于注意力计算

**(2) KV 解压投影 $W_{kv\_b}$**

$$W_{kv\_b}: [d_{kv}] \to [H \times (D_{\text{nope}} + D_v)] = 512 \to 64 \times (128 + 128) = 64 \times 256 = 16384$$

$$\text{FLOPs}_{kv\_b} = 2 \times d_{kv} \times H \times (D_{\text{nope}} + D_v) \times T_{\text{new}}$$

Decode：$= 2 \times 512 \times 64 \times 256 \times 1 = 16{,}777{,}216 \approx 16.8\text{M FLOPs}$

这个投影从 512 维 latent 中“解压”出 64 个 head，每个 head 有 128 维 nope K 和 128 维 V。等效于用一个 $512 \times 16384$ 的矩阵做投影——但比直接从 $7168 \to 16384$（MHA 方式）的 $7168 \times 16384 = 117.4\text{M}$ 矩阵 **小了 7×**。

### 3.4.3 Q 侧 FLOPs

**(3) Q 压缩投影 $W_{q\_a}$**

$$W_{q\_a}: [d] \to [d_q] = 7168 \to 1536$$

$$\text{FLOPs}_{q\_a} = 2 \times d \times d_q \times T_{\text{new}}$$

Decode：$= 2 \times 7168 \times 1536 \times 1 = 22{,}020{,}096 \approx 22.0\text{M FLOPs}$

**(4) Q nope 解压投影 $W_{q\_b}$**

$$W_{q\_b}: [d_q] \to [H \times D_{\text{nope}}] = 1536 \to 64 \times 128 = 8192$$

$$\text{FLOPs}_{q\_b} = 2 \times d_q \times H \times D_{\text{nope}} \times T_{\text{new}}$$

Decode：$= 2 \times 1536 \times 64 \times 128 \times 1 = 25{,}165{,}824 \approx 25.2\text{M FLOPs}$

**(5) Q RoPE 直投投影 $W_{q\_rope}$**

RoPE 分量必须直接从 hidden 维度投影，不能经过压缩——因为 RoPE 的旋转操作施加在维度对上，压缩会破坏这个结构。

$$W_{q\_rope}: [d] \to [H \times D_{\text{rope}}] = 7168 \to 64 \times 64 = 4096$$

$$\text{FLOPs}_{q\_rope} = 2 \times d \times H \times D_{\text{rope}} \times T_{\text{new}}$$

Decode：$= 2 \times 7168 \times 64 \times 64 \times 1 = 58{,}720{,}256 \approx 58.7\text{M FLOPs}$

**注意**：$W_{q\_rope}$ 是 MLA 中第二大的单项 FLOPs（仅次于输出投影），因为 RoPE 部分不能享受低秩压缩的红利。

### 3.4.4 QK 点积与 V 加权（O(T²) 项——与 MHA 完全等同）

> **训练提示**：QK 用 $D_{qk} = D_{\text{nope}} + D_{\text{rope}} = 192$ 维，V 用 $D_v = 128$ 维——两者维度不同，训练时系数也不同（QK: 4 passes，V: 3 passes），不能用统一的 $7 \times (D_{qk} + D_v)$，必须分开 $(4 \times D_{qk} + 3 \times D_v)$。详见 [§3.12](#312-从推理到训练系数体系)。

MLA 的 QK 点积分为两部分：

**(6a) nope 分量的 QK 点积**

$$\text{FLOPs}_{QK_{\text{nope}}} = 2 \times H \times T_{\text{new}} \times T_{\text{total}} \times D_{\text{nope}}$$

Decode（T=1M）：$= 2 \times 64 \times 1 \times 10^6 \times 128 = 16.4\text{G FLOPs}$

**(6b) rope 分量的 QK 点积**

$$\text{FLOPs}_{QK_{\text{rope}}} = 2 \times H \times T_{\text{new}} \times T_{\text{total}} \times D_{\text{rope}}$$

Decode（T=1M）：$= 2 \times 64 \times 1 \times 10^6 \times 64 = 8.2\text{G FLOPs}$

**(6c) 合计 QK 点积**

$$\text{FLOPs}_{QK} = 2 \times H \times T \times (D_{\text{nope}} + D_{\text{rope}}) = 2 \times H \times T \times D_h$$

$$= 2 \times 64 \times 10^6 \times 192 = 24.6\text{G FLOPs}$$

其中 $D_h = 128 + 64 = 192$。**这与标准 MHA（$D_h=192$）的 QK 点积 FLOPs 完全相等。**

**(7) V 加权**

$$\text{FLOPs}_{V} = 2 \times H \times T_{\text{new}} \times T_{\text{total}} \times D_v$$

Decode（T=1M）：$= 2 \times 64 \times 1 \times 10^6 \times 128 = 16.4\text{G FLOPs}$

### 3.4.5 输出投影

**(8) 输出投影 $W_o$**

$$W_o: [H \times D_v] \to [d] = (64 \times 128) = 8192 \to 7168$$

$$\text{FLOPs}_o = 2 \times H \times D_v \times d \times T_{\text{new}}$$

Decode：$= 2 \times 64 \times 128 \times 7168 \times 1 = 117{,}440{,}512 \approx 117.4\text{M FLOPs}$

### 3.4.6 MLA 单层 Decode FLOPs 汇总（T=1M）

| 组件 | FLOPs | 类型 | vs MHA 同配置 |
|---|---|---|---|
| $W_{kv\_a}$（KV 压缩） | 8.3M | 常数 | —（MLA 新增） |
| $W_{kv\_b}$（KV 解压） | 16.8M | 常数 | —（MLA 新增） |
| $W_{q\_a}$（Q 压缩） | 22.0M | 常数 | —（MLA 新增） |
| $W_{q\_b}$（Q nope 解压） | 25.2M | 常数 | —（MLA 新增） |
| $W_{q\_rope}$（Q RoPE 直投） | 58.7M | 常数 | MHA Q proj 176.2M → **节省 3×** |
| QK 点积（nope + rope） | **24.6G** | O(T) | **相同** |
| V 加权 | 16.4G | O(T) | **相同** |
| $W_o$（输出投影） | 117.4M | 常数 | 相同 |
| **单层合计** | **~41.2G** | — | — |  （MLA 通过低秩投影节省了线性项，QK+V 与 MHA 相同维度）

MLA 单层节省的 FLOPs 主要来自于：用多个小矩阵（低秩）替代 Q、K、V 的直投大矩阵。$W_{kv\_a}$ + $W_{kv\_b}$ + $W_{q\_a}$ + $W_{q\_b}$ + $W_{q\_rope}$ 合计 $\approx 131\text{M FLOPs}$，而标准 MHA 的 Q+K+V 三个直投矩阵合计 $\approx 2 \times 7168 \times 64 \times 192 \times 3 \approx 528\text{M FLOPs}$。**线性项节省约 4×**。

但 QK 点积（24.6G）+ V 加权（16.4G）= 41G——这部分在 T=1M 时占比超过 99%，且**与标准 MHA 完全相同**。

### 3.4.7 关键结论

**MLA 省的是 KV cache，不是 FLOPs 的主体。**

- **线性项（投影）**：MLA 将 QKV 投影从 $\approx 528\text{M}$ 降到 $\approx 131\text{M FLOPs/token}$，但这项在长上下文下只占总 FLOPs 的 $\sim 0.3\%$。
- **平方项/长上下文项（QK + V）**：MLA 的 FLOPs 与 MHA **完全相同**——$2 \times H \times T \times D_h$——因为最终 attention 计算的维度规模没有变。
- **KV Cache**：MLA 将每个 token 的 KV cache 从 $H \times D_{qk} + H \times D_v = 64 \times 192 + 64 \times 128 = 20{,}480$ 个元素压到 $d_{kv} + D_{\text{rope}} = 512 + 64 = 576$ 个元素——**压缩 35.6×**。这才是 MLA 的主要价值。

MLA 就像“快递打包”——包裹运输时压缩（KV cache 小），但到了收件人手里必须拆开原样呈现（注意力计算时的 K、V 维度与 MHA 完全相同）。运费省了（显存），但收件人验货的工作量没少（FLOPs）。

---

## 3.5 Mamba-2 SSD FLOPs（Nemotron）

逐项拆解 Mamba-2 Structured State Space Duality 层的 FLOPs，展示为什么它是 O(T) 而非 O(T²)。Mamba-2 是 Nemotron 的核心非 Attention 序列建模层——48 个 Mamba 层的 FLOPs 特征决定了整个模型的长上下文行为。

### 3.5.1 Mamba-2 计算流程回顾

维度回顾（Nemotron）：$d=8192$，$\text{expand}=2 \Rightarrow d_{\text{inner}}=16384$，$H_{\text{mamba}}=256$，$D_{\text{mamba}}=64$，$N=128$（`ssm_state_size`），$n_{\text{groups}}=8$，$C=128$（chunk size）。

验证自洽性：$d_{\text{inner}} = H_{\text{mamba}} \times D_{\text{mamba}} = 256 \times 64 = 16384$。$\checkmark$

Mamba-2 的 SSD 将序列分成大小为 C 的 chunk，每个 chunk 内部做因果 matmul（对角块），chunk 之间通过状态传递（非对角块）。总计算量分为四部分：

### 3.5.2 (a) `in_proj` 输入投影（线性项主力）

`in_proj` 一次性产生所有需要的分量：$\mathbf{x}$、$\mathbf{z}$、$\mathbf{B}$、$\mathbf{C}$、$\boldsymbol{\Delta}$。

投影维度：$d \to 2 \times d_{\text{inner}} + 2 \times n_{\text{groups}} \times N + H_{\text{mamba}}$
$= 8192 \to 2 \times 16384 + 2 \times 8 \times 128 + 256$
$= 8192 \to 32768 + 2048 + 256 = 35072$

$$\text{FLOPs}_{\text{in\_proj}} = 2 \times d \times 35072 \times T_{\text{new}}$$

Decode：$= 2 \times 8192 \times 35072 \times 1 = 574{,}619{,}648 \approx 574.6\text{M FLOPs}$

这是 Mamba-2 层单 token 计算中**最大的一项**。对比 Attention 的 Q 投影（134M），Mamba 的 `in_proj` 约大 4.3×——因为它是一次性投影出 5 个分量（x, z, B, C, Δ），相当于把 Attention 的 Q、K、V、外加两个额外的分量合并到一个矩阵里。

### 3.5.3 (b) `conv1d` 深度卷积（可忽略）

一维深度卷积，核大小 = 4，输入通道数 = $d_{\text{conv}} = d_{\text{inner}} + 2 \times n_{\text{groups}} \times N = 16384 + 2048 = 18432$。

$$\text{FLOPs}_{\text{conv1d}} = 2 \times d_{\text{conv}} \times \text{kernel} \times T_{\text{new}}$$

Decode：$= 2 \times 18432 \times 4 \times 1 = 147{,}456 \approx 0.15\text{M FLOPs}$

卷积核只有 4 个元素宽，而且是深度卷积（每个通道独立的 1D 卷积），所以计算量跟 `in_proj` 比可以忽略不计——就像“顺丰快递的包装费相对于货品价值”。

### 3.5.4 (c) SSD 对角块（chunk 内因果 matmul）

这是 Mamba-2 "Attention 等价" 的部分。在每个 chunk 内，SSD 做类似因果 Attention 的计算：

$$\text{FLOPs}_{\text{diag}} = 2 \times \frac{T}{C} \times \frac{C^2}{2} \times H_{\text{mamba}} \times D_{\text{mamba}} = T \times C \times H_{\text{mamba}} \times D_{\text{mamba}}$$

代入：$= T \times 128 \times 256 \times 64 = T \times 2{,}097{,}152$

**Prefill（T=4096）**：$4096 \times 2{,}097{,}152 \approx 8.59 \times 10^9 \approx 8.6\text{G FLOPs}$

**Decode（$T_{\text{new}}=1$，但 chunk 内的因果 matmul 在 decode 时仅涉及当前 chunk 的累积状态）**：约 4.2M FLOPs（与 T 无关）。

这里需要澄清：在 decode 阶段，Mamba-2 不需要对每个新 token 重做所有 chunk 的内部计算——SSD 的递归特性意味着新 token 只需要更新当前 chunk 的对角块和状态传递。因此 decode 时这部分是常数。

### 3.5.5 (d) SSD 非对角块：chunk 间的状态传递

前面的对角块是每个 chunk "内部消化"——chunk 里的每个 token 看到前面 token 的计算。但 chunk 1 的最后一个 token 怎么看到 chunk 0 的第一个 token？这需要**状态传递**。

Mamba-2 的 SSM 在每个 chunk 边界维护一个隐藏状态 $h \in \mathbb{R}^{H_{\text{mamba}} \times N}$（$N = d_{state} = 128$）。这个状态向量"记住"了之前所有 chunk 的摘要。

当一个 chunk 结束时，它的状态 $h_{i}$ 需要"传递"给下一个 chunk。传递的数学操作是：下一个 chunk 的每个位置，将传入状态与当前 chunk 的 $C$（输出投影）相乘，得到对当前 chunk 内每个 token 的修正量。这个操作为每个 chunk 边界做一次 $h_i \times C_{i+1}$。

$h_i$ 的形状是 $[H_{\text{mamba}}, N]$，$C_{i+1}$（经过 decay 加权后）的形状也是 $[H_{\text{mamba}}, N]$。这不是简单的向量点积——Mamba-2 需要在 $N$ 维空间内做"状态混合"，让 $N$ 维的每个分量都能影响当前 chunk 的输出。因此，实际的状态传递矩阵是一个 $[N, N]$ 的变换：

$$\text{FLOPs}_{\text{off-diag}} = 2 \times \underbrace{\frac{T}{C}}_{\text{chunk 数}} \times \underbrace{H_{\text{mamba}}}_{\text{heads}} \times \underbrace{N^2}_{\text{状态传递矩阵}}$$

代入 Nemotron 的数值：chunk 数 $= T/128$，$H_{\text{mamba}} = 256$，$N = 128$：

$$= 2 \times \frac{T}{128} \times 256 \times 128^2 = 2 \times \frac{T}{128} \times 256 \times 16{,}384$$

$$= 2 \times \frac{T}{128} \times 4{,}194{,}304 = T \times 65{,}536 \approx 6.55 \times 10^4 \times T$$

**Prefill（T=4096）**：$4096 \times 65{,}536 \approx 0.27\text{G FLOPs}$

**Decode**：约 $6.55 \times 10^4$ FLOPs（常数级别）。

> 对角块和非对角块加起来，就是 SSD 的完整 FLOPs。对角块做"块内注意"（$O(C^2)$），非对角块做"块间传递"（$O(N^2)$）。$C = 128$、$N = 128$ 时，$C^2 = N^2$——这是设计上的巧合，不是必然。如果 chunk_size 变了，对角块和非对角块的比例就会偏移。

### 3.5.6 (e) `out_proj` 输出投影

$$\text{FLOPs}_{\text{out\_proj}} = 2 \times d_{\text{inner}} \times d \times T_{\text{new}}$$

Decode：$= 2 \times 16384 \times 8192 \times 1 = 268{,}435{,}456 \approx 268.4\text{M FLOPs}$

### 3.5.7 Mamba-2 单层 FLOPs 汇总

**Prefill（T=4096）**：

| 组件 | FLOPs | 占比 | 复杂度 |
|---|---|---|---|
| in_proj | $574.6\text{M} \times 4096 = 2.35\text{T}$ | 92.3% | O(T) |
| conv1d | $0.15\text{M} \times 4096 = 0.61\text{G}$ | ~0% | O(T) |
| SSD 对角块 | 8.6G | 0.3% | O(T×C) |
| SSD 非对角块 | 0.27G | ~0% | O(T) |
| out_proj | $268.4\text{M} \times 4096 = 1.10\text{T}$ | 7.4% | O(T) |
| **单层合计** | **~3.46T FLOPs** | 100% | **O(T)** |

48 层合计：$\approx 166\text{T FLOPs}$（prefill 4096 token）。全部是 O(T)——没有任何 O(T²) 项。

**Decode（$T_{\text{new}}=1$，$T=1\text{M}$）**：

| 组件 | FLOPs | 复杂度 |
|---|---|---|
| in_proj | 574.6M | O(1) |
| conv1d | 0.15M | O(1) |
| SSD 对角块 (decode) | ~4.2M | O(1) |
| SSD 非对角块 (decode) | ~0.07M | O(1) |
| out_proj | 268.4M | O(1) |
| **单层合计** | **~847M** | **O(1)** |

**48 层 Mamba-2 合计**：$\approx 40.7\text{G FLOPs/token}$（与 T 无关！）

这是最关键的数字：**Mamba-2 层的 decode FLOPs 与上下文长度完全无关**——每 token 固定 $\approx 847\text{M FLOPs}$。而 Attention 层在 T=1M 时需要 $\approx 33.1\text{G FLOPs/token}$。

### 3.5.8 与 Attention 的对比：O(T) vs O(T²)

以 1M 上下文为例，**单层对比**：

| 指标 | Full Attention (GQA) | Mamba-2 SSD | 比率 |
|---|---|---|---|
| 线性项 (proj) | 277M | 843M | 0.33×（Mamba 更贵） |
| 长上下文项 (QK/sSD) | 32.8G | ~4.3M | **7600×**（Attention 更贵） |
| **单层总计** | **33.1G** | **847M** | **39×**（Mamba 更快） |

48 层 Mamba-2（$\approx 40.7\text{G FLOPs}$） vs 48 层 Full Attention（$\approx 48 \times 33.1\text{G} \approx 1.59\text{T FLOPs}$）——Mamba 快 **39×**。

Mamba-2 的 SSD 是“聪明地算”——把 O(T²) 的 Attention 变成了 chunk 内 O(C²) 的因果 matmul（C=128，常数）。1M 个 token 被切成 ~7812 个 chunk，每个 chunk 内部做的计算量恒定。新 token 到来时，只更新当前 chunk 并传播状态。而 Attention 每来一个新 token，都要跟全部 1M 个历史 token 逐一“打招呼”。这就是 O(T) vs O(T²) 的本质区别。

---

## 3.6 Sliding Window Attention（SWA）FLOPs

Sliding Window Attention 是 MiMo-V2-Flash、Mistral 等模型使用的稀疏 Attention 方案。每个 token 只关注它前面固定窗口 $W$ 内的 token，而非全部 $T$ 个 token。

QK 点积的复杂度从 $O(T^2)$ 降到 $O(T \times W)$：

$$\text{FLOPs}_{\text{QK, SWA}} = 2 \times H_q \times T_{\text{new}} \times \min(T, W) \times D_h$$

- Prefill（每个 token 看到前面 $W$ 个）：$2 \times H_q \times T \times W \times D_h$
- Decode（新 token 只往前看 $W$ 步）：$2 \times H_q \times 1 \times W \times D_h$

以 MiMo-V2-Flash 为例：$H_q = 64$，$W = 131072$，$D_h = 128$。Prefill 时 $T=W=131\text{K}$：$2 \times 64 \times 131072 \times 131072 \times 128 \approx 2.8 \times 10^{14}$ FLOPs，是 Full Attention（$8.4 \times 10^{14}$）的约 $1/3$。但 decode 时：$2 \times 64 \times 1 \times 131072 \times 128 = 2.15 \times 10^9$ FLOPs——**与 Full Attention decode 完全相同**（因为 decode 时 $T_{new}=1$，Full Attn 也只看全部 $T$ 个历史 token）。

SWA 省的是 prefill 而非 decode。它适合吞吐优先的短上下文场景，但在长上下文 decode 上没有优势。

> SWA 的 $W$ 不是凭空取的——通常等于 `max_position_embeddings` 或 `sliding_window` 字段。如果 config 中找不到 `sliding_window` 但模型声称是 SWA，查看 `max_position_embeddings` 是否与上下文窗口匹配。

## 3.7 Gated DeltaNet（Linear Attention）FLOPs

Gated DeltaNet 是 Qwen3.5-MoE 等模型使用的线性注意力方案。与 Mamba-2 共享核心思想——用固定大小的隐藏状态 $S_t \in \mathbb{R}^{H \times D_h \times D_h}$ 取代 Attention 的 $O(T^2)$ 点积。

DeltaNet 的更新规则（简化）：

$$S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot (k_t \otimes v_t)$$

其中 $k_t \otimes v_t$ 是 key 和 value 的外积，形状为 $[H, D_h, D_h]$。$\alpha_t$ 是遗忘门（decay），$\beta_t$ 是输入门（input gate），两者都是通过投影从当前输入得到的标量。

输出：$y_t = S_t \cdot q_t$，其中 $S_t \cdot q_t$ 将一个 $[H, D_h, D_h]$ 矩阵与 $[H, D_h]$ 向量相乘，得到 $[H, D_h]$ 的注意力输出。

**每 token FLOPs 分解**：

$$\text{FLOPs}_{\text{DeltaNet}} = \underbrace{2 \times H \times D_h^2}_{\text{外积 } k_t \otimes v_t} + \underbrace{2 \times H \times D_h^2}_{\text{状态乘 } S_t \cdot q_t} + \underbrace{2 \times H \times D_h^2}_{\text{状态更新 } S_t = \alpha S_{t-1} + \beta(k \otimes v)}$$

三项各 $2 \times H \times D_h^2$，合计 $6 \times H \times D_h^2$。全与 $T$ 无关——**DeltaNet 的 decode FLOPs 是常数**。

以 Qwen3.5-MoE 为例（$H = 64$，$D_h = 128$）：$6 \times 64 \times 128^2 = 6 \times 64 \times 16384 \approx 6.3 \times 10^6$ FLOPs/token/layer。对比 Full Attention 的 decode（$2 \times 64 \times 10^6 \times 128 \approx 1.6 \times 10^{10}$），DeltaNet 节省了约 **2500×**。

与 Mamba-2 的核心差异：Mamba-2 通过 `in_proj` 一次性产生所有 SSM 参数（$\Delta, B, C$），其输入投影的 FLOPs 远大于 SSM 核心计算。DeltaNet 的投影更简单（类似标准 Attention 的 QKV 投影），所以整体 FLOPs 更小。但 Mamba-2 的状态维度 $H \times N$（$256 \times 128$）比 DeltaNet 的 $H \times D_h^2$（$64 \times 128^2$）小得多——状态大小是 $O(H \times N)$ vs $O(H \times D_h^2)$，差了 $D_h$ 倍。

## 3.8 MoE Gating FLOPs

计算路由器（Router / Gate）的 FLOPs，证明它在总计算量中占比 <1%。很多人担心 MoE 的路由开销会抵消稀疏化的收益——这一页数值直接打消这个顾虑。

### Router FLOPs

标准 sigmoid/softmax 路由器的核心计算是一个矩阵乘法：

$$\text{FLOPs}_{\text{router}} = 2 \times d \times E \times T_{\text{new}}$$

**Nemotron**（$d=8192$，$E=512$，decode）：

$$\text{FLOPs}_{\text{router}} = 2 \times 8192 \times 512 \times 1 = 8{,}388{,}608 \approx 8.4\text{M FLOPs}$$

**M3**（$d=6144$，$E=128$，decode）：

$$\text{FLOPs}_{\text{router}} = 2 \times 6144 \times 128 \times 1 = 1{,}572{,}864 \approx 1.6\text{M FLOPs}$$

对比单层 MoE 的专家计算量（激活 4-22 个专家，每个专家做 $2 \times d \times d_{ff}$ 或 $3 \times d \times d_{ff}$ 的 FFN）：

- Nemotron 单专家（ReLU$^2$，latent 空间）：$2 \times 2048 \times 5120 \approx 21\text{M FLOPs}$
- 激活 22 个专家：$\approx 462\text{M FLOPs}$

Router 的 8.4M FLOPs 占 462M 的 **1.8%**。在 M3（128 专家，激活 4 个）中占比更低。

**DeepSeek V4 Flash** 的 hash routing 稍复杂，但本质仍是查表+少量矩阵乘法，FLOPs 在百万量级，可忽略。

Router 就是给 512 扇门各配一把锁（一个 8192 维向量），新 token 来了用自己的 8192 维特征跟 512 把锁各算一次相似度。这个开销相当于一扇门打开后干活（一个专家 FFN）的几十分之一。**Router 的 FLOPs 约等于半个 Attention 的 K 投影——在总计算量的大海里是一滴水。**

---

## 3.9 Vision Encoder FLOPs

计算 ViT 编码器的 FLOPs，理解为什么视觉编码在总推理成本中的占比。多模态模型输入一张图时，ViT 要处理 576-2916 个 patch token——这部分计算量是“固定入场券”，与文本长度无关。

### 3.9.1 MiniMax M3 ViT FLOPs

M3 ViT：32 层，$d_{\text{vit}}=1280$，$H_{\text{vit}}=16$，$D_{\text{vit}}=80$，$d_{ff}^{\text{vit}}=5120$。

图像 token 数：$\left(\frac{2016}{14}\right)^2 = 144^2 = 20736$ patches，经过 pixel unshuffle（$\times 4$ 压缩）后：$20736 / 4 = 5184$，再经 spatial merge：$5184 / 9 = 576$ tokens。本文取 576。

**单层 Attention**（标准 MHA）：

$$\text{FLOPs}_{\text{ViT QKV}} = 4 \times 2 \times d_{\text{vit}} \times H_{\text{vit}} \times D_{\text{vit}} \times T_{\text{img}}$$

$$= 8 \times 1280 \times 16 \times 80 \times 576 = 8 \times 1{,}638{,}400 \times 576$$

$$= 8 \times 943{,}718{,}400 = 7{,}549{,}747{,}200 \approx 7.55\text{G FLOPs}$$

（$4 \times 2 = 8$ 来自 Q、K、V、O 四个投影各 $2 \times m \times n \times k$）

**QK 点积**（causal 不适用，ViT 对图像做双向 Attention）：

$$\text{FLOPs}_{\text{ViT QK}} = 2 \times H_{\text{vit}} \times T_{\text{img}}^2 \times D_{\text{vit}} = 2 \times 16 \times 576^2 \times 80$$

$$= 2 \times 16 \times 331{,}776 \times 80 = 849{,}346{,}560 \approx 0.85\text{G FLOPs}$$

**V 加权**：

$$\text{FLOPs}_{\text{ViT V}} = 2 \times H_{\text{vit}} \times T_{\text{img}}^2 \times D_{\text{vit}} = 0.85\text{G FLOPs}$$

**单层 MLP**（GELU，2 个矩阵）：

$$\text{FLOPs}_{\text{ViT MLP}} = 2 \times 2 \times d_{\text{vit}} \times d_{ff}^{\text{vit}} \times T_{\text{img}}$$

$$= 4 \times 1280 \times 5120 \times 576 = 4 \times 6{,}553{,}600 \times 576$$

$$= 4 \times 3{,}774{,}873{,}600 = 15{,}099{,}494{,}400 \approx 15.1\text{G FLOPs}$$

**单层合计**：$7.55 + 0.85 + 0.85 + 15.1 \approx 24.35\text{G FLOPs}$

**32 层合计**：$32 \times 24.35\text{G} \approx 779\text{G FLOPs}$

加上 patch embedding、projector 等：$\approx 800\text{G FLOPs} = 0.8\text{T FLOPs}$（per image）。

对比文本骨干（60 层，prefill 4096 token，$\approx 100\text{T+ FLOPs}$），ViT 的 0.8T FLOPs 占比 <1%。

ViT 虽深（32 层），但维度小（1280 vs 6144）且 token 数固定（576 vs 4096+）。相当于“一辆 Smart 虽也能开到 120 迈，但跟重卡（文本骨干）不是一个吨位的”。

### 3.9.2 Kimi K2.5 ViT FLOPs（速算）

K2.5 ViT：27 层，$d_{\text{vit}}=1152$，$H_{\text{vit}}=16$，$D_{\text{vit}}=72$，$d_{ff}^{\text{vit}}=4304$。图像 token 数约 576-2916（取决于分辨率模式）。

用 576 token 近似：

$$\text{单层 Attn + MLP} \approx 8 \times 1152 \times 16 \times 72 \times 576 + 4 \times 1152 \times 4304 \times 576$$

$$\approx 6.1\text{G} + 11.4\text{G} \approx 17.5\text{G FLOPs}$$

27 层：$\approx 0.47\text{T FLOPs}$。加上 PatchMerger 和投影器：$\approx 0.5-0.7\text{T FLOPs}$。

---

## 3.10 完整案例对比：1M 上下文下三种架构的 FLOPs

在同一张表中呈现纯 Full Attention、Nemotron Hybrid（Mamba + Attn）、M3 MSA 三种方案的 FLOPs 分解。这张表是 CH3 的终极输出——一行看懂 Mamba 和 MSA 为什么殊途同归地解决了 O(T²) 问题。

### 3.10.1 场景设定

- 上下文长度：T = 1M tokens
- 解码阶段：$T_{\text{new}} = 1$（单 token decode）
- 对比模型：
  - **纯 Full Attn (hypothetical)**：60 层 Full Attention，$d=8192$，$H_q=64$，$H_{kv}=64$（MHA，无 GQA），$D_h=128$，SwiGLU FFN $d_{ff}=8192 \times 4 \approx 32768$（无 MoE 时 FFN 占比较小，此处简化用大维度）
  - **Nemotron 3 Ultra (hybrid)**：48 层 Mamba-2 + 12 层 Attention（GQA 32:1，2 KV heads）+ 48 层 MoE（22/512 激活）。$d=8192$，$H_q=64$，$H_{kv}=2$，$D_h=128$。MoE 专家在 latent 空间计算。
  - **M3 (MSA)**：57 层 MSA（GQA 16:1，4 KV heads）+ 3 层 Full Attention（GQA 16:1）+ 57 层 MoE（4/128 激活）。$d=6144$，$H_q=64$，$H_{kv}=4$，$D_h=128$。

### 3.10.2 逐项 FLOPs 分解（decode per token, T=1M）

**Attention 部分（QK + V 加权）**：

| 模型 | Attention 层数 | 单层 QK+V FLOPs | Attn 部分合计 |
|---|---|---|---|
| 纯 Full Attn | 60 | $4 \times 64 \times 1\text{M} \times 128 = 32.8\text{G}$ | $60 \times 32.8\text{G} = 1.97\text{T}$ |
| Nemotron Hybrid | 12 | 32.8G (GQA 下 QK+V 仍为 $4 \times 64 \times T \times 128$) | $12 \times 32.8\text{G} = 393.6\text{G}$ |
| M3 MSA | 3 Full + 57 MSA | Full: 32.8G（改用 $d=6144$，$H_q=64$，$H_{kv}=4$ 后实际 ~32.8G）；MSA: Index QK 1.02G + Main QK+V 67.2M ≈ 1.09G | $3 \times 32.8\text{G} + 57 \times 1.09\text{G} \approx 160.5\text{G}$ |

**Mamba/SSD 部分**：

| 模型 | Mamba/SSD 层数 | 单层 FLOPs | Mamba 部分合计 |
|---|---|---|---|
| 纯 Full Attn | 0 | 0 | 0 |
| Nemotron Hybrid | 48 | ~847M | $48 \times 847\text{M} = 40.7\text{G}$ |
| M3 MSA | 0 | 0 | 0 |

**线性投影部分**（QKV proj + O proj + in_proj + out_proj + FFN）：

| 模型 | 单层投影估算 | 投影部分合计 |
|---|---|---|
| 纯 Full Attn | Q(134M) + K(134M) + V(134M) + O(134M) + FFN(~1.6G) ≈ 2.14G | $60 \times 2.14\text{G} \approx 128\text{G}$ |
| Nemotron Hybrid | Attn 投影(~277M) × 12 + Mamba 投影(~843M) × 48 + MoE FFN(~462M) × 48 | $\approx 3.3\text{G} + 40.5\text{G} + 22.2\text{G} \approx 66\text{G}$ |
| M3 MSA | MSA 投影(~220M) × 57 + Full Attn 投影(~220M) × 3 + MoE FFN(~220M) × 57 | $\approx 12.5\text{G} + 0.7\text{G} + 12.5\text{G} \approx 26\text{G}$ |

> 注：以上为近似量级估算。投影部分具体数值取决于 $d_{ff}$、MoE 专家数等配置细节，精确计算需代入各模型 `config.json` 的实际值。本表的重点是横比数量级差异。

### 3.10.3 总表

| 模型 | Attn QK+V 部分 | Mamba/SSD 部分 | 线性投影 | 总 FLOPs/token | 相对纯 Full Attn |
|---|---|---|---|---|---|
| 纯 Full Attn (hypothetical) | **~1.97T** | 0 | ~128G | **~2.10T** | 1×（基线） |
| Nemotron 3 Ultra (hybrid) | **~394G** | ~41G | ~66G | **~501G** | ~1/4 |
| M3 (MSA) | **~161G** | 0 | ~26G | **~187G** | ~1/11 |

核心发现：

1. **纯 Full Attn 在 1M 上下文下几乎不可用**：每产生一个 token 需要 2.1T FLOPs，单看 Attention QK+V 部分的 1.97T 占 94%。即使最强大的推理硬件也难以达到可接受的吞吐（2.1T / 989 TFLOPS（H100 FP16）$\approx 2.1$ 秒/ token）。

2. **Nemotron Hybrid 将 QK+V 开销砍到原来的 1/5**（394G vs 1970G），因为 80% 的层（48/60）用 Mamba-2 完全避开了 O(T) Attention。但 12 个 Attention 层仍贡献了总 FLOPs 的 78%——**12 个 Attention 层的成本超过了 48 个 Mamba 层的总和**。

3. **M3 MSA 更进一步**：3 个 Full Attention 层占 98G 的 QK+V，57 个 MSA 层才占 62G（Index QK $57 \times 1.02\text{G} = 58.1\text{G}$ + Main QK+V $57 \times 0.067\text{G} = 3.8\text{G}$）。MSA 的 Index Branch 虽然仍是 O(T)，但以 16× 的廉价系数执行。

4. **殊途同归**：Nemotron 用 Mamba-2（状态空间，O(1) decode），M3 用稀疏 Attention（O(T) 但系数极小）——两条不同的技术路线，但都在 1M 上下文上将 Attention 部分从 TFLOPs 量级压到了 GFLOPs 量级。**原理不同，效果趋同。**

### 3.10.4 不同上下文长度下的横比

为直观展示 O(T) vs O(1) 的差别，固定模型配置，变化 T。仅计算 Attention 相关的 QK+V 部分（不含投影和 FFN）：

| T | 纯 Full Attn QK+V (60层) | Nemotron Hybrid Attn QK+V (12层) | M3 QK+V (3 Full + 57 MSA) |
|---|---|---|---|
| 4K | $60 \times 4 \times 64 \times 4096 \times 128 = 8.05\text{G}$ | $12 \times 4 \times 64 \times 4096 \times 128 = 1.61\text{G}$ | 3 Full: $3 \times 4 \times 64 \times 4096 \times 128 = 0.40\text{G}$<br>57 MSA Index: $57 \times 2 \times 4 \times 4096 \times 128 = 0.24\text{G}$<br>57 MSA Main: $57 \times 4 \times 64 \times 2048 \times 128 = 3.82\text{G}$<br>**合计: ~4.46G** |
| 32K | $60 \times 4 \times 64 \times 32768 \times 128 = 64.4\text{G}$ | $12 \times 4 \times 64 \times 32768 \times 128 = 12.9\text{G}$ | 3 Full: $3 \times 4 \times 64 \times 32768 \times 128 = 3.22\text{G}$<br>57 MSA Index: $57 \times 2 \times 4 \times 32768 \times 128 = 1.91\text{G}$<br>57 MSA Main: $57 \times 4 \times 64 \times 2048 \times 128 = 3.82\text{G}$<br>**合计: ~8.95G** |
| 128K | $60 \times 4 \times 64 \times 131072 \times 128 = 258\text{G}$ | $12 \times 4 \times 64 \times 131072 \times 128 = 51.5\text{G}$ | 3 Full: $3 \times 4 \times 64 \times 131072 \times 128 = 12.9\text{G}$<br>57 MSA Index: $57 \times 2 \times 4 \times 131072 \times 128 = 7.65\text{G}$<br>57 MSA Main: $57 \times 4 \times 64 \times 2048 \times 128 = 3.82\text{G}$<br>**合计: ~24.4G** |
| 1M | $60 \times 4 \times 64 \times 1\text{M} \times 128 = 1.97\text{T}$ | $12 \times 4 \times 64 \times 1\text{M} \times 128 = 394\text{G}$ | 3 Full: $3 \times 4 \times 64 \times 1\text{M} \times 128 = 98.3\text{G}$<br>57 MSA Index: $57 \times 2 \times 4 \times 1\text{M} \times 128 = 58.4\text{G}$<br>57 MSA Main: $57 \times 4 \times 64 \times 2048 \times 128 = 3.82\text{G}$<br>**合计: ~160.5G** |

> 注：M3 MSA 的 Main Branch 始终只在 2048 个入选 token 上做 Attention——**与 T 无关，常数 3.82G**。Index Branch 的 QK 评分随 T 线性增长但只有 4 个 head。Full Attention 的 3 层和 Index Branch 的 O(T) 项共同主导 M3 的长上下文成本。

**观察**：
- 在 **4K** 短上下文：三种方案差距较小（8.0G vs 1.6G vs 4.5G）。MSA 反而比纯 Full Attn（12 层）慢，因为 Index Branch 的额外开销 + Main Branch 选了 2048/4096=50% 的 token——稀疏化的好处在短序列上不明显。
- 在 **128K** 中上下文：差距拉开（258G vs 52G vs 24G）。MSA Main Branch 仅访问 2048/131072 = 1.6% 的 token，而 Index Branch O(T) 项（7.7G）仍远小于 Full Attn O(T) 项（258G）。
- 在 **1M** 长上下文：差距成为鸿沟（1970G vs 394G vs 161G）。MSA Main Branch 仅访问 2048/1M = 0.2% 的 token——近乎常数。M3 比纯 Full Attn 的 QK+V 部分快 ~12×，Nemotron Hybrid 快 ~5×。
- **关键洞察**：MSA 在超长上下文时 Main Branch 趋近于常数，Index Branch 成为唯一 O(T) 项。但因为 Index 只有 4 head，实际斜率仅为 Full Attn 的 1/16。**MSA 本质是用 O(T) 斜率 1/16 的廉价计算替代全量 O(T)。**

如果说短上下文（4K）是“在大厅里找人”，那长上下文（1M）就是“在鸟巢体育场里找人”。Full Attention 的做法是跟每一个观众对视一眼（O(T)），Mamba 的做法是先把体育场分片区，只跟片区组长沟通（chunk + state），MSA 的做法是先派几个侦察兵扫一眼观众席（Index Branch），找到目标区域后再派大队人马过去（Main Branch）。

---

## 3.11 速查表：FLOPs 公式汇总

给一张“查表即算”的公式大全。不需要重读整章，从这里抄公式代入 `config.json` 的数值即可。

| 组件 | 公式 | 适用场景 |
|---|---|---|
| Q/K/V 投影 | $2 \times d \times (H_{\text{type}} \times D_h) \times T_{\text{new}}$ | Q 用 $H_q$，K/V 用 $H_{kv}$ |
| QK 点积 | $2 \times H_q \times T_{\text{new}} \times T_{\text{total}} \times D_h$ | Prefill 时 $T_{\text{new}}=T_{\text{total}}$（causal 约 /2） |
| V 加权 | $2 \times H_q \times T_{\text{new}} \times T_{\text{total}} \times D_h$ | 与 QK 等量级 |
| O 投影 | $2 \times d \times (H_q \times D_h) \times T_{\text{new}}$ | 与 Q 投影等量级 |
| MLA $W_{kv\_a}$ | $2 \times d \times (d_{kv} + D_{\text{rope}}) \times T_{\text{new}}$ | MLA 模型 |
| MLA $W_{kv\_b}$ | $2 \times d_{kv} \times H \times (D_{\text{nope}} + D_v) \times T_{\text{new}}$ | MLA 模型 |
| MLA $W_{q\_a}$ | $2 \times d \times d_q \times T_{\text{new}}$ | MLA 模型 |
| MLA $W_{q\_b}$ | $2 \times d_q \times H \times D_{\text{nope}} \times T_{\text{new}}$ | MLA 模型 |
| MLA $W_{q\_rope}$ | $2 \times d \times H \times D_{\text{rope}} \times T_{\text{new}}$ | MLA 模型 |
| MSA Index QK | $2 \times H_{\text{idx}} \times T_{\text{new}} \times T_{\text{total}} \times D_{\text{idx}}$ | M3 式 MSA |
| MSA Main QK/V | $2 \times H_q \times T_{\text{new}} \times T_{\text{selected}} \times D_h$ | $T_{\text{selected}} = \text{block\_size} \times \text{top\_k}$ |
| Mamba-2 in_proj | $2 \times d \times (2d_{\text{inner}} + 2n_{\text{groups}}N + H_{\text{mamba}}) \times T_{\text{new}}$ | Nemotron 式 Mamba-2 |
| Mamba-2 SSD diag | $T \times C \times H_{\text{mamba}} \times D_{\text{mamba}}$ | Prefill; decode 时为常数 |
| Mamba-2 SSD off-diag | $T / C \times H_{\text{mamba}} \times N^2 \times 2$ | Prefill; decode 时常数可忽略 |
| Mamba-2 out_proj | $2 \times d_{\text{inner}} \times d \times T_{\text{new}}$ | 总是 |
| Router | $2 \times d \times E \times T_{\text{new}}$ | 所有 MoE 模型 |
| FFN (ReLU$^2$) | $2 \times 2 \times d \times d_{ff} \times T_{\text{new}}$ | Nemotron |
| FFN (SwiGLU) | $2 \times 3 \times d \times d_{ff} \times T_{\text{new}}$ | M3, K2.5 |
| ViT Attn | $4 \times 2 \times d_{\text{vit}} \times H_{\text{vit}} \times D_{\text{vit}} \times T_{\text{img}}$ | VL 模型视觉编码器 |
| ViT MLP (GELU) | $2 \times 2 \times d_{\text{vit}} \times d_{ff}^{\text{vit}} \times T_{\text{img}}$ | VL 模型视觉编码器 |

**实战口诀**：
1. 先确定场景：prefill 还是 decode？
2. 线性项（投影 + FFN）：直接代入 $T_{\text{new}}$（prefill = 输入长度，decode = 1）
3. 平方项（QK + V）：将 $T_{\text{new}}$ 和 $T_{\text{total}}$ 分开——prefill 时两者相等，decode 时 $T_{\text{new}}=1$ 但 $T_{\text{total}}$ 是全部历史
4. 稀疏/MSA 项：把 $T_{\text{total}}$ 换成 $T_{\text{selected}}$（入选 token 数）
5. Mamba 项：decode 时全部为常数，prefill 时乘以 $T$
6. 把每层加起来，乘以层数，得到单 token FLOPs
7. 乘以 bytes 和 batch size 得到总计算吞吐需求

---

## CH3 常见计算错误

## 3.12 从推理到训练：系数体系

CH 3.1-3.11 描述的是**前向（推理）FLOPs**。训练时需要前向 + 反向，总 FLOPs 是前向的倍数。这个倍数不是笼统的 ×3——不同操作的系数不同，且受梯度重计算（gradient checkpointing）影响。

### 3.12.1 线性投影：系数 6

每个 `nn.Linear`（$Y = X \cdot W$）在训练中执行 3 次 matmul：

| Pass | 计算 | FLOPs |
|---|---|---|
| 前向 | $Y = X \cdot W$ | $2 \times m \times n \times k$ |
| 反向（权重梯度） | $\partial L/\partial W = (\partial L/\partial Y)^T \cdot X$ | $2 \times m \times n \times k$ |
| 反向（输入梯度） | $\partial L/\partial X = \partial L/\partial Y \cdot W^T$ | $2 \times m \times n \times k$ |
| **合计** | | **$6 \times m \times n \times k$** |

所以训练 FLOPs = $6 \times \text{params} \times \text{tokens}$（训练 FLOPs = 6 × params × tokens 即由此而来）。

### 3.12.2 Attention QK 与 V：系数不同（4 vs 3）

Attention 的 Q@K^T 和 A@V 在训练中的 pass 数**不同**，原因是梯度重计算（Flash Attention 的核心机制）。

Flash Attention 前向时**不存储**注意力矩阵 $A = Q \cdot K^T$（$S \times S$ 矩阵太大），反向时重算 Q@K^T 恢复 $A$。但 A@V 不需要重算——它直接用重算出的 $A$。

**Q@K^T 的训练 pass 数推导**：

| Pass | 计算 | 维度 | FLOPs |
|---|---|---|---|
| 前向 | $A = Q \cdot K^T$ | $[H,S,D_{qk}] \times [H,D_{qk},S]$ | $2 \times H \times S^2 \times D_{qk}$ |
| 反向（$\partial L/\partial Q$） | $\partial L/\partial A \cdot K$ | $[H,S,S] \times [H,S,D_{qk}]$ | $2 \times H \times S^2 \times D_{qk}$ |
| 反向（$\partial L/\partial K$） | $\partial L/\partial A^T \cdot Q$ | 同上 | $2 \times H \times S^2 \times D_{qk}$ |
| 重计算前向 | $A = Q \cdot K^T$（恢复 $A$） | 同前向 | $2 \times H \times S^2 \times D_{qk}$ |
| **合计** | | | **$4 \times H \times S^2 \times D_{qk}$** |

**A@V 的训练 pass 数推导**：

| Pass | 计算 | 维度 | FLOPs |
|---|---|---|---|
| 前向 | $O = A \cdot V$ | $[H,S,S] \times [H,S,D_v]$ | $2 \times H \times S^2 \times D_v$ |
| 反向（$\partial L/\partial A$） | $\partial L/\partial O \cdot V^T$ | $[H,S,D_v] \times [H,D_v,S]$ | $2 \times H \times S^2 \times D_v$ |
| 反向（$\partial L/\partial V$） | $A^T \cdot \partial L/\partial O$ | 同上 | $2 \times H \times S^2 \times D_v$ |
| **合计** | | **无重计算** | **$3 \times H \times S^2 \times D_v$** |

**关键公式**：

$$\text{Attention FLOPs}_{\text{train}} = (4 \times D_{qk} + 3 \times D_v) \times H \times S^2 \times L$$

- 标准 Attention（$D_{qk} = D_v = D$）：$(4+3) \times D = 7 \times D$ → 简记为系数 7
- MLA（$D_{qk} \neq D_v$）：**不能**用 $7 \times (D_{qk} + D_v)$，必须分开算

**如果不使用梯度重计算**（关闭 gradient checkpointing）：Q@K^T 的重计算 pass 消失，系数从 4 降到 3，总系数变为 $3 + 3 = 6$。

### 3.13 Indexer 与 Router 的 no_grad 特性

稀疏注意力模型（DSA/MSA/CSA-HCA）的 Indexer 和 MoE 的 Router 都包含一个 `torch.topk()` 操作——**离散选择，数学上不可导**。你无法对「选择第 42 号 token」这个动作求梯度。

这一步**必然**不在 autograd 图中。但 Indexer/Router 内部的可学习参数（线性投影）是否训练，是工程选择：

| 方案 | 做法 | 梯度来源 |
|---|---|---|
| GLM-5/M3/V4 的选择 | `@torch.no_grad()` 包裹整个 indexer | 无——参数冻结 |
| 理论替代方案 | straight-through estimator | 梯度通过 topk 近似传播 |
| 理论替代方案 | REINFORCE / policy gradient | 梯度通过奖励信号传播 |

GLM-5 选择完全冻结 indexer（源码 `modeling_glm_moe_dsa.py:L197`），可能出于训练稳定性和计算成本考虑——indexer 的 O(S²) 如果要反向传播，计算量翻 3 倍。

#### MoE Router 的情况不同

MoE 的 router 和 DSA Indexer 有本质区别：

- **DSA/MSA Indexer**：整个 forward 被 `@torch.no_grad()` 包裹 → 参数完全冻结
- **MoE Router**：router 的 `nn.Linear` 在 autograd 图中 → 权重通过专家输出反向传播正常训练
- MoE 的 `no_grad` 只包裹 dispatch 逻辑（token 到专家的 gather/scatter），不包裹 router 本身

**从原理可推断的部分**：MoE router 权重需要学习「哪个 token 给哪个专家」，这必须通过下游 loss 的梯度训练，所以 router Linear 不应该 no_grad。

**从原理不可推断的部分**：DSA Indexer 选择完全冻结（而非用 straight-through estimator），这是工程决策，只能从源码 `@torch.no_grad()` 确认。

#### 对 FLOPs 计算的影响

| 操作 | 训练系数 | 原因 |
|---|---|---|
| Linear 投影（标准） | 6 | 前向 + 反向×2 |
| Attention Q@K^T（标准） | 4 | 前向 + 反向×2 + 重计算 |
| Attention A@V（标准） | 3 | 前向 + 反向×2 |
| **Indexer 全部操作**（`no_grad`） | **1** | 仅前向，无反向 |
| **MoE Router Linear** | **6** | 正常前向+反向 |
| **MoE dispatch（topk/gather）** | **1** | `no_grad` 内，仅前向 |
| TopK 比较 | 1 | 非 matmul，仅前向 |
| Conv1d（depthwise） | 3 | 前向 + 反向 |

### 3.14 IndexShare 对训练 FLOPs 的影响

GLM-5.2 的 IndexShare 机制（每 4 层共享 1 个 indexer）在推理时节省 indexer FLOPs。训练时的影响取决于实现：
- GLM-5.2 的 indexer 在 `@torch.no_grad()` 下，系数 1 × 21 层（full）= 仅 21/78 的 indexer 前向 FLOPs

$$\text{Indexer FLOPs}_{\text{GLM-5.2}} = \frac{21}{78} \times \text{Indexer FLOPs}_{\text{GLM-5.1}}$$

这对总 FLOPs 的影响 <1%（indexer 本身占比小），但对推理延迟的改善显著（博客声称 2.9× per-token FLOPs 降低，因为推理时 indexer 的 O(S²) 在长上下文下占比大）。

| # | 常见错误 | 正确做法 |
|---|---|---|
| 1 | decode 时把 QKV 投影乘以 $T_{\text{total}}$ | decode 只投影 1 个新 token，投影 FLOPs 是常数 |
| 2 | GQA 下 QK 点积用 $H_{kv}$ 算 | QK 点积前 K 已经被 `repeat_kv` 扩展到 $H_q$，用 $H_q$ 算 |
| 3 | MLA 的 QK 点积以为能省 FLOPs | MLA 省的是 KV cache（显存），不是 QK 点积 FLOPs——最终 attention 的 $D_h = D_{\text{nope}} + D_{\text{rope}}$ 与 MHA 相同 |
| 4 | 把 prefill 的 causal /2 也用在 decode | decode 的 query 只有 1 个，不存在 causal mask 的对称简化，公式是 $T_{\text{new}} \times T_{\text{total}}$ 而非 $T^2/2$ |
| 5 | MSA 的 Index QK 以为不用算 O(T²) | Index QK 仍然是 O(T²)（prefill）或 O(T)（decode），只是 head 数少（4 vs 64），系数省 16× |
| 6 | Mamba-2 decode 时把 SSD 对角块按 O(T) 算 | Mamba-2 decode 是 O(1)——只需更新当前 chunk 的状态，不重算全部 chunk |
| 7 | 忘记乘 2（MAC 系数） | 深度学习框架中 1 MAC = 2 FLOPs，所有矩阵乘法公式必须有因子 2 |
| 8 | 把参数数量当 FLOPs | 参数量是“存了多少数”，FLOPs 是“每次前向算多少下”，两者中间隔着序列长度 T（对 O(T) 项）或 T²（对 O(T²) 项） |

---

> **下一章预告**：CH 4 内存分析——KV Cache 大小推导、MLA/GQA 的缓存压缩比、显存带宽瓶颈（Roofline 模型）、batch size 与延迟的权衡。


---



> **系列导航**：[（一）预备知识与参数分解](../part-1/) ← 当前 → [（三）KV Cache 与推理显存](../part-3/) → [（四）M3 实战 + Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)
