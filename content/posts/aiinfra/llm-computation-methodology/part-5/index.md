+++
date = '2026-06-22T09:04:00+08:00'
draft = false
title = 'LLM 系统分析方法论（五）：训练显存估算'
categories = ['aiinfra']
tags = ['training', 'memory', 'gpu', 'parallelism', 'methodology', 'm3']
series = 'llm-computation-methodology'
series_order = 5
math = true
summary = '训练显存完整估算：从单卡四笔账（权重/优化器/梯度/激活）出发，叠加 TP/PP/DP/CP/EP 并行折扣，结合 ZeRO/FSDP、Gradient Checkpointing、Offload 建立训练显存体系。含 M3 完整案例和多模态/LoRA 微调场景。'
+++


> 从单卡极限出发，叠加并行策略、重计算、Offload 三层折扣，建立训练显存的完整估算体系。

> **系列导航**：[（一）预备知识](../part-1/) → [（二）FLOPs](../part-2/) → [（三）KV Cache](../part-3/) → [（四）M3+Roofline](../part-4/) ← 当前 → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)

## 本文定位

[系列（三）（四）篇](../part-3/) 解决了推理显存（权重 + KV Cache + 激活缓冲），本文解决训练显存。两者的根本差异：

| | 推理 | 训练 |
|---|---|---|
| 权重 | $N \times \text{bytes}$ | $N \times \text{bytes}$（同推理） |
| 优化器状态 | **无** | Adam为 $N \times 12$ bytes，这是训练独有的最大头 |
| 梯度 | **无** | $N \times 2$ bytes（BF16）或 $N \times 4$（FP32） |
| 激活 | 仅 1 token 的前向 | 整条序列（B × T tokens）的前向中间结果，需保留到反向传播 |
| **简化记忆** | $\approx 2N$ bytes | $\approx (2 + 12 + 2)N = 16N$ bytes + 激活 |

16N vs 2N——训练的基础显存（不含激活）是推理权重显存的 8 倍。

---

## CH 1 | 单卡极限：四笔账的 base 方程

在引入任何并行策略之前，先建立单卡训练的显存模型。这个方程是后续所有"折扣"的基础——并行策略本质上是把 base 方程的各项除以不同的并行度因子。

### 1.1 第一笔：模型权重

与推理相同，BF16 训练时每参数 2 bytes：

$$M_{\text{weight}} = N \times 2$$

混合精度训练中，这是前向/反向计算用的"工作副本"。此外还有一个 FP32 的 master 副本——它归入优化器状态，不在这里。

### 1.2 第二笔：优化器状态

AdamW 维护三个 FP32 张量 per parameter：

$$M_{\text{optimizer}} = N \times 4 \times 3 = 12N \text{ bytes}$$

三项分别是：

| 分量 | 符号 | 精度 | bytes/param |
|---|---|---|---|
| Master weight | $\theta$ | FP32 | 4 |
| 一阶动量 | $m$ | FP32 | 4 |
| 二阶动量 | $v$ | FP32 | 4 |
| **合计** | | | **12** |

Adafactor 将 $m$ 和 $v$ 用行/列外积近似，降至 4 bytes/param。FP8 optimizer（如 DeepSpeed 的 FP8 Adam）将 $m$ 和 $v$ 存为 FP8，master weight 仍为 FP32，降至 $4 + 1 + 1 = 6$ bytes/param，采用分块更新时等效约 4 bytes/param。

**Muon（MomentUm Orthogonal by Nesterov）**：近年 Nemotron 等 SOTA 模型采用的优化器。与 Adam 的核心差异是**无二阶矩**（$v$），只维护一阶动量 $m$。混合精度下 optimizer states 仅 $4 + 4 = 8$ bytes/param（FP32 master + FP32 动量），对比 Adam 的 $12$ bytes/param — **省 1/3**。Muon 的 Newton-Schulz 正交化迭代是 in-place 操作，不额外占用显存。实践中 embedding 和 norm 等非矩阵参数通常仍用 AdamW，但线性层权重占模型参数的绝大多数（> 95%），整体优化器开销约为全 AdamW 的 $8/12 \approx 67\%$。

各优化器对比：

| 优化器 | 状态构成 | bytes/param | 适用 |
|---|---|---|---|
| AdamW | master(FP32) + $m$(FP32) + $v$(FP32) | 12 | 通用基线 |
| Muon | master(FP32) + $m$(FP32) | 8（仅 2D 参数） | 权重矩阵（线性层） |
| Adafactor | master(FP32) + 外积 $m,v$ | 4 | 显存极端受限 |
| FP8 Adam | master(FP32) + $m$(FP8) + $v$(FP8) | 6 | DeepSpeed/H100+ |
| SGD (momentum) | master(FP32) + $m$(FP32) | 8 | 小模型/微调 |

本文后续推导以 **AdamW（12 bytes/param）** 为主基线，在 CH 7 案例中额外标注 Muon 的显存节省。

### 1.3 第三笔：梯度

反向传播产生的梯度，精度通常与计算精度一致：

$$M_{\text{grad}} = N \times 2 \quad (\text{BF16})$$

注意 FP32 梯度（某些配置为数值稳定选用）会使此项翻倍：$N \times 4$。

### 1.4 第四笔：激活

激活是训练显存中最复杂的部分。前向传播时，每个算子产生的中间结果都需要保留到反向传播——因为反向需要它们来计算梯度。这些中间结果就是"激活"。

#### 1.4.1 为什么激活会爆

推理 decode 时，序列长度 T = 1（只生成一个 token）。训练时，T 是完整序列长度（通常 2048-8192，长上下文训练可达 128K+），且乘以 batch size。激活随 T 和 B 线性增长。

#### 1.4.2 Per-layer 激活估算

基于 Korthikanti et al. (2022) 的经典公式，一个标准 Transformer decoder layer（无 FlashAttention）前向传播需要保留的激活量：

$$A_{\text{layer}} = B \times T \times d \times \left(34 + \frac{5 \times H_q \times T}{d}\right) \text{ bytes}$$

各形状分量的含义：

| 来源 | 字节数（单位 $B \times T \times d$）| 说明 |
|---|---|---|
| Input & output LayerNorm | 4 | 前后各一，每处存输入和归一化结果 |
| QKV 投影输入 | 3 | pre-norm 后的 hidden state |
| Attention score $QK^T$ | $5 H_q T / d$ | 每头一个 $[T, T]$ 矩阵，经 softmax + dropout + PV |
| Attention output 投影 | 1 | 投影前的 multi-head 拼接结果 |
| Residual + dropout masks | 4 | 两次残差连接各存 mask + 输入 |
| FFN 输入 & gate | 2 + 4 = 6 | up/gate 投影输入 + 四个中间激活（up, gate, SiLU 输出, 逐元素乘） |
| FFN output 投影 | 1 | 投影前的 intermediate hidden state |
| Residual + dropout | 2 | 第二次残差 |
| **线性项小计** | **34** | |
| **Attention 平方项** | **$5H_qT/d$** | 长序列时此项主导 |

该公式使用 FP16/BF16（2 bytes/element），已包含在系数中。若用 FP32 需额外 ×2。

#### 1.4.3 FlashAttention 的影响

FlashAttention 的核心优化之一是**不显式构造完整的 $[T, T]$ attention score 矩阵**。因此公式中的 $5H_qT/d$ 项消失：

$$A_{\text{layer}}^{\text{FA}} \approx 34 \times B \times T \times d \text{ bytes}$$

这是巨大的节省——在 $T = 8192$ 时，原始公式中的平方项约为 $5 \times 64 \times 8192 / 8192 = 320$ 单位，比线性项（34）大近 10 倍。FlashAttention 将激活从 ~350 单位压到 34 单位——约 1/10。

**本文后续激活分析默认假定使用 FlashAttention**，$C_{\text{act}} \approx 34$。

> **FP8 训练的额外节省**：NVIDIA Transformer Engine 将激活存储为 FP8（1 byte/elem），$C_{\text{act}}$ 进一步折半至 ~17（BF16 当量）。FP8 训练现已成为 H100/B200 的标准配置。本文保持 BF16 基线以保证公式保守性——FP8 场景下实际激活可再省约 50%。

#### 1.4.4 全模型激活

$$M_{\text{activation}} = L \times A_{\text{layer}} = L \times 34 \times B \times T \times d \text{ bytes}$$

对 M3 模型（$d = 6144$, $L = 60$, $B_{\text{micro}} = 1$, $T = 8192$）：
$$M_{\text{activation}} = 60 \times 34 \times 1 \times 8192 \times 6144 = 1.03 \times 10^{11} \approx \mathbf{103 \text{ GB}}$$

仅激活就 103 GB，还没有算子图额外引用、PyTorch 分配器开销等。实际通常是公式值的 1.2-1.5×。

### 1.5 单卡 base 方程

$$M_{\text{total}} = \underbrace{2N}_{\text{weight}} + \underbrace{12N}_{\text{optimizer}} + \underbrace{2N}_{\text{grad}} + \underbrace{34 \times L \times B \times T \times d}_{\text{activation}}$$

代入 M3（$N = 428\text{B}$, $d = 6144$, $L = 60$），$B_{\text{micro}} = 1$, $T = 8192$：

| 组分 | 公式 | 值 |
|---|---|---|
| 权重 | $428 \times 10^9 \times 2$ | 856 GB |
| 优化器 | $428 \times 10^9 \times 12$ | 5,136 GB |
| 梯度 | $428 \times 10^9 \times 2$ | 856 GB |
| 激活 | $60 \times 34 \times 1 \times 8192 \times 6144$ | ~103 GB |
| **合计** | | **~6,951 GB ≈ 7.0 TB** |

对比：单张 H100 只有 80 GB HBM。~7.0 TB 需要约 87 张 H100——而且这只是装下，不算实际训练。这就是为什么训练大模型必须用并行策略切分。

---

## CH 2 | Megatron 折扣体系：TP + PP + DP

Megatron-LM 的训练并行是三维叠加：TP（切矩阵乘法维度）、PP（切层）、DP（切 batch）。三个维度的显存节省是**乘性叠加**的——但注意不同组分对三个维度的响应不同：权重/优化器/梯度享受 $tp \times pp$ 折扣（DP 不省），激活享受 $pp$ 折扣（TP 不省，DP 通过 $B_{\text{micro}}$ 间接稀释）。

### 2.1 TP：切隐藏维度

TP 将每个线性层的权重矩阵沿列（column parallelism）或行（row parallelism）切分到 $tp$ 张卡。

列并行（如 QKV 投影）：每卡持有 $1/tp$ 的权重列。前向时各自算各自的列，输出通过 all-reduce 聚合。
行并行（如 attention output、FFN output）：每卡持有 $1/tp$ 的权重行。输入已经 all-reduce 过，各自算各自的行，输出分片。

**显存折扣**：

$$M_{\text{weight}}(tp) = \frac{2N}{tp}, \quad M_{\text{optimizer}}(tp) = \frac{12N}{tp}, \quad M_{\text{grad}}(tp) = \frac{2N}{tp}$$

但 TP **不减少激活**——每张卡虽然只算 `d/tp` 维度的 matmul，但需要完整的 `B × T × d` 输入。激活在 TP 组内**全量复制**。

$$M_{\text{activation}}(tp) = 34 \times L \times B \times T \times d \quad (\text{不随 tp 变化})$$

**TP 的附属优化：Sequence Parallelism (SP)**。Megatron 的 TP 实现通常附带 SP——将 LayerNorm 和 Dropout 的激活也沿序列维度切分到 $tp$ 张卡上。SP 不改变大矩阵乘法的激活（它们仍是全量复制），但将 LayerNorm 输入的 $B \times T \times d$（激活中约 10-15% 的体量）打 $1/tp$ 折。TP 组内的整体激活节省约 10-15%。SP 与 CP 不同——SP 沿 $tp$ 组切分且仅覆盖 norm/dropout，CP 沿独立 $cp$ 组切分且覆盖全序列。

### 2.2 PP：切层

PP 将 L 层切分为 $pp$ 个 stage，每张卡只持有 $L/pp$ 层。

$$M_{\text{weight}}(pp) = \frac{2N}{pp}, \quad M_{\text{optimizer}}(pp) = \frac{12N}{pp}, \quad M_{\text{grad}}(pp) = \frac{2N}{pp}$$

$$M_{\text{activation}}(pp) = 34 \times \frac{L}{pp} \times B \times T \times d$$

激活同样打 $1/pp$ 折——每张卡只需要存自己管的那些层的激活。

### 2.3 DP：切 batch

DP 将 global batch 均分到 $dp$ 张卡。每卡每步处理 $B_{\text{micro}}$ 个样本（$B_{\text{global}} = dp \times B_{\text{micro}} \times \text{grad\_accum}$）。每张卡持有**全部模型副本**（权重、优化器、梯度全量），只有激活随 micro batch 缩水：

$$M_{\text{activation}}(dp) = 34 \times L \times B_{\text{micro}} \times T \times d$$

$$M_{\text{weight}}(dp) = 2N, \quad M_{\text{optimizer}}(dp) = 12N, \quad M_{\text{grad}}(dp) = 2N$$

DP 本身不省权重/优化器/梯度——这是它和 FSDP 的根本区别。

注意：$B_{\text{micro}}$ 本身已隐含了 $dp$ 的稀释效应（$B_{\text{micro}} = B_{\text{global}} / (dp \times \text{grad\_accum})$），激活公式中不再额外除以 $dp$。

### 2.4 三维叠加公式

Megatron 三个维度叠加后，每张 GPU 的显存：

$$M_{\text{total}}^{\text{Megatron}} = \underbrace{\frac{2N}{tp \times pp}}_{\text{weight}} + \underbrace{\frac{12N}{tp \times pp}}_{\text{optimizer}} + \underbrace{\frac{2N}{tp \times pp}}_{\text{grad}} + \underbrace{34 \times \frac{L}{pp} \times B_{\text{micro}} \times T \times d}_{\text{activation}}$$

激活**不受 TP 影响**（原因见 2.1），**受 PP 影响**（切层）。DP 的影响已隐含在 $B_{\text{micro}}$ 中（$B_{\text{micro}} = B_{\text{global}} / (dp \times \text{grad\_accum})$）。

### 2.5 案例：M3 on Megatron TP=4, PP=4, DP=8

参数：$N = 428\text{B}$, $d = 6144$, $L = 60$, $B_{\text{global}} = 32$, $B_{\text{micro}} = 1$, $T = 8192$

| 组分 | 计算 | 值 |
|---|---|---|
| 权重 | $428 \times 10^9 \times 2 / (4 \times 4)$ | 53.5 GB |
| 优化器 | $428 \times 10^9 \times 12 / (4 \times 4)$ | 321.0 GB |
| 梯度 | $428 \times 10^9 \times 2 / (4 \times 4)$ | 53.5 GB |
| 激活 | $34 \times (60/4) \times 1 \times 8192 \times 6144$ | ~25.7 GB |
| **合计** | | **~453.7 GB** |

453.7 GB ≫ 80 GB（H100）。即使加上 gradient checkpointing（CH 5），优化器仍占 321 GB——**Megatron 纯 DP 不切优化器，这在 MoE 大模型上完全不可行**。这就是为什么 ZeRO（CH 3）和 EP（Expert Parallelism）是预训练的前提条件。

注意：本例 $B_{\text{micro}}=1$, $dp=8$。$B_{\text{global}} = dp \times B_{\text{micro}} \times \text{grad\_accum} = 8 \times 1 \times 4 = 32$。激活公式中用 $B_{\text{micro}}$ 而非 $B_{\text{micro}}/dp$——因为 $B_{\text{micro}}$ 已经是 per-GPU 值。

---

## CH 3 | FSDP / ZeRO 折扣体系

FSDP（PyTorch）和 ZeRO（DeepSpeed）的核心思想相同：**把 DP 维度的"全量复制"改成"分片持有，用时收集"**。ZeRO 分三级，从浅到深逐级把更多东西从 DP 维度切走。

### 3.1 ZeRO 三级的显存公式

记 $dp$ 为数据并行组大小。

| 组分 | Baseline (DP) | ZeRO-1 | ZeRO-2 | ZeRO-3 / FSDP1 |
|---|---|---|---|---|
| 权重 | $2N$ | $2N$ | $2N$ | $\mathbf{2N/dp}$ |
| 优化器 | $12N$ | $\mathbf{12N/dp}$ | $\mathbf{12N/dp}$ | $\mathbf{12N/dp}$ |
| 梯度 | $2N$ | $2N$ | $\mathbf{2N/dp}$ | $\mathbf{2N/dp}$ |
| 激活 | $34LBTd$ | $34LBTd$ | $34LBTd$ | $34LBTd$ |

注意激活从未被任何 ZeRO stage 减少——FSDP/ZeRO 只切参数，不切激活。激活始终是 `34 L B_micro T d`。

### 3.2 FSDP 的权重峰值 vs 驻留

FSDP/ZeRO-3 的一个重要细节：**前向传播时，all-gather 将权重临时展开为全量**。

```
时间线（FSDP1，单层前向）：
  [all-gather: 收集权重分片 → 2N bytes]  ← 此时峰值 = 2N
  [前向计算: 用全量权重]                  ← 峰值仍含全量  
  [丢弃全量: 只保留 2N/dp 分片]          ← 驻留 = 2N/dp
```

因此：
- **平均驻留显存**： $2N/dp$（分片后即释放）
- **瞬时峰值显存**： $2N$（all-gather 展开时）

这与 Megatron TP 不同——TP 的权重切分是**永久**的（每次 matmul 只用 `d/tp` 维度的权重组，不需要展开全量）。FSDP 的"省"是以 DP 组内的通信代价换的。

### 3.3 FSDP2 / HSDP：两级分片

FSDP2 的核心创新是 HSDP（Hybrid Sharding）：将 DP 世界分为两层。

$$dp_{\text{total}} = dp_{\text{replicate}} \times dp_{\text{shard}}$$

- `dp_shard`：节点内 GPU（通过 NVLink 互联），做 FSDP 分片
- `dp_replicate`：跨节点复制（通过 IB 互联），做纯 DP

分层后的显存和通信：

| 组分 | 驻留 | 峰值 |
|---|---|---|
| 权重 | $2N / dp_{\text{shard}}$ | $2N / dp_{\text{replicate}}$ |
| 优化器 | $12N / dp_{\text{shard}}$ | $12N / dp_{\text{shard}}$ |
| 梯度 | $2N / dp_{\text{shard}}$ | $2N / dp_{\text{shard}}$ |
| 激活 | $34LBTd$ | $34LBTd$（不变） |

关键改进在**权重峰值**：FSDP1 的 all-gather 展开到全量（$2N$），HSDP 只展开到节点内全量（$2N/dp_{\text{replicate}}$）。当 $dp_{\text{replicate}} = 8$（8 节点）时，权重峰值从 $2N$ 降到 $2N/8$。

通信差异（细节见[系列第六篇](../part-6/)）：
- 节点内 reduce-scatter + all-gather（NVLink 高带宽，几乎无感）
- 节点间 all-reduce（IB，仅触达梯度，不触达参数）

HSDP 的本质是"把昂贵的跨节点通信配置在梯度上（ZeRO-2 级），把便宜的节点内通信配置在参数上（ZeRO-3 级）"。

### 3.4 Megatron + ZeRO 的联合折扣

实践中经常 Hybrid：TP + PP 用 Megatron 切权重，ZeRO-1/2 额外切优化器/梯度：

$$M_{\text{total}}^{\text{Hybrid}} = \frac{2N}{tp \times pp} + \frac{12N}{tp \times pp \times dp} + \frac{2N}{tp \times pp \times dp} + 34 \times \frac{L}{pp} \times B_{\text{micro}} \times T \times d$$

优化器和梯度同时享受 TP/PP 折扣（矩阵切分 + 层切分）和 DP/ZeRO 折扣（数据并行切分）。激活仅受 PP 影响，$B_{\text{micro}}$ 自身已吸收 DP 稀释。

### 3.5 案例：M3 on Megatron TP4+PP4 + ZeRO-2 DP8

$N = 428\text{B}$, $B_{\text{micro}} = 1$, $T = 8192$。$B_{\text{global}} = 8 \times 1 \times 4 = 32$。

| 组分 | 计算 | 值 |
|---|---|---|
| 权重 | $428B \times 2 / (4 \times 4)$ | 53.5 GB |
| 优化器 | $428B \times 12 / (4 \times 4 \times 8)$ | 40.1 GB |
| 梯度 | $428B \times 2 / (4 \times 4 \times 8)$ | 6.7 GB |
| 激活 | $34 \times 15 \times 1 \times 8192 \times 6144$ | ~25.7 GB |
| **合计** | | **~126.0 GB** |

126.0 GB > 80 GB——ZeRO-2 将优化器和梯度分别从 321 GB 和 53.5 GB 压缩到 40.1 GB 和 6.7 GB，极大幅度省出了空间，但**仍然放不进 H100**。权重（53.5 GB）+ 激活（25.7 GB）已经 79.2 GB，优化器和梯度哪怕只占 1 GB 也会超。必须继续叠加优化。

---

## CH 4 | Context Parallelism (CP)

当训练序列长度超过约 128K 时，即使有 TP/PP/ZeRO，激活显存仍然爆——因为激活随 $T$ 线性增长。CP 通过切分序列维度来解决这个瓶颈。

**CP 为什么独立成章**：CP 切的是序列维度（$T \rightarrow T/cp$），与 TP（切 hidden dim）、PP（切层）、FSDP/ZeRO（切 DP 维度）完全正交。Megatron 可以用 CP，FSDP 也可以用 CP——它不是某个训练框架的附属功能，而是一个独立的并行维度。因此 CP 的显存折扣是**乘性叠加**的：激活 = $34 \times (L/pp) \times B_{\text{micro}} \times (T/cp) \times d$，所有框架通用。

### 4.1 CP 的显存折扣

CP 将序列长度 $T$ 均分为 $cp$ 份，每张卡只处理 $T/cp$ 个 token 的序列段。

$$M_{\text{activation}}(cp) = 34 \times \frac{L}{pp} \times B_{\text{micro}} \times \frac{T}{cp} \times d$$

激活与 $cp$ 成反比——这就是 CP 的核心价值。

KV cache 同样切分（虽然训练时通常不显式缓存 KV，但反向传播需要保留 key 和 value 供 attention backward 使用）：

$$M_{\text{KV\_for\_bwd}} = 4 \times L \times H_{kv} \times D \times \frac{T}{cp} \text{ bytes} \quad (\text{K + V, BF16})$$

CP **不切权重/优化器/梯度**——每张卡仍持有完整的模型副本（或 Megatron/ZeRO 折扣后的副本）。CP 与 TP/PP/ZeRO 正交，可以叠加。

### 4.2 CP 叠加后的激活公式

将 CP 加入 Megatron + ZeRO 混合公式：

$$M_{\text{activation}}^{\text{full}} = 34 \times \frac{L}{pp} \times B_{\text{micro}} \times \frac{T}{cp} \times d$$

TP 仍不影响激活（激活在 TP 组内全量复制）。DP 的影响已隐含在 $B_{\text{micro}}$ 中。

### 4.3 什么时候必须用 CP

给定 GPU 显存上限 $M_{\text{avail}}$（除权重/优化器/梯度后剩余的显存），CP 需求为：

$$cp \geq \frac{34 \times (L/pp) \times B_{\text{micro}} \times T \times d}{M_{\text{avail}}}$$

代入 $T = 128\text{K} = 131,072$，M3 参数，$pp=4$, $B_{\text{micro}}=1$。以 CH 7 的 EP=8 配置为例（权重+优化器+梯度合计 ~27 GB），$M_{\text{avail}} = 80 - 27 \approx 53\text{GB}$（H100）：

$$\text{无 ckpt (C=34): } cp \geq \frac{34 \times 15 \times 1 \times 131,072 \times 6144}{53 \times 10^9} \approx 7.7 \rightarrow cp \geq 8$$

$$\text{Full ckpt (C=2, 详见 CH 5): } cp \geq \frac{2 \times 15 \times 1 \times 131,072 \times 6144}{53 \times 10^9} \approx 0.45 \rightarrow cp = 1 \text{（不需要）}$$

对于 $T = 1\text{M} = 1,048,576$：

$$\text{Full ckpt (C=2): } cp \geq \frac{2 \times 15 \times 1 \times 1,048,576 \times 6144}{53 \times 10^9} \approx 3.6 \rightarrow cp \geq 4$$

关键结论：**无 ckpt 时即使 128K 也需要大量 CP；full ckpt 把 CP 需求推迟到 1M+；但 1M 训练 CP 是必须的（cp ≥ 4）。**

### 4.4 CP 的 trade-off

- **收益**：激活和 KV 打 $1/cp$ 折，长序列训练的显存瓶颈被打破
- **代价**：通信变为 p2p ring pass（$cp-1$ 跳串行），延迟敏感。$cp$ 大到一定程度后通信成为瓶颈——$cp$ 的边际收益递减
- **与其他并行叠加**：CP 与 TP/PP 正交，但 CP 组最优拓扑通常与 DP 组同置（节点内 CP 通信走 NVLink）

---

## CH 5 | Gradient Checkpointing：用计算换显存

Gradient checkpointing（activation checkpointing，AC）是最直接也最常用的训练显存优化——不改变任何并行配置，只放弃存储部分激活，在反向传播时重新计算它们。

### 5.1 原理

标准训练：前向存所有激活 → 反向逐层回传梯度。
Checkpointing：只存"checkpoint"层（通常每 $\sqrt{L}$ 层或每 2 层存一次），中间层的激活在反向时从前一个 checkpoint 重新前向计算。

```
标准训练（L=12，每层都存激活）：
F0---F1---F2---F3---F4---F5---F6---F7---F8---F9---F10---F11
                     [反向]

Selective ckpt（L=12，每 2 层存一次）：
F0=====F1---F2=====F3---F4=====F5---F6=====F7---F8=====F9---F10=====F11
[存]        [存]        [存]        [存]        [存]        [存]
  [重算F1]     [重算F3]     [重算F5]     [重算F7]     [重算F9]     [重算F11]
```

### 5.2 节省公式

**Full checkpointing（每层都重算）**：
$$M_{\text{activation}}^{\text{full\_ckpt}} \approx 2 \times \frac{L}{pp} \times B_{\text{micro}} \times \frac{T}{cp} \times d$$

$C_{\text{act}}$ 从 34 降到约 2——只存每层的输入 hidden state（1 份）和 residual（1 份）。节省约 **17×**（34 → 2）。

额外计算开销：前向 pass 在反向传播时被完整重算一次，forward FLOPs 翻倍。但 forward 通常占总 FLOPs 的 ~1/3（backward 约为 forward 的 2×），所以 **total FLOPs 增加约 30-40%**。

**Selective checkpointing（每 $K$ 层存一次）**：
$$M_{\text{activation}}^{\text{selective}} \approx 34 \times \frac{L}{pp \times K} \times B_{\text{micro}} \times \frac{T}{cp} \times d + \cancel{\text{attention scores}}$$

每 $K$ 层只存一个 checkpoint 的完整激活集（34 × BTd），其余 $K-1$ 层的激活按 full ckpt 处理（只存输入，$C \approx 2$）。上述公式为近似（严格形式为 $(34 + 2 \times (K-1)) \times L/(pp \times K)$，误差 < 10%）。注意 FlashAttention 已经消除了 attention score 存储——这是 selective ckpt 与 FlashAttention 的天然协同。

额外计算开销：约 15-25%（取决于 $K$）。

### 5.3 与并行策略的协同

Checkpointing 与 PP 天然互补：PP 已经将激活限制在 $L/pp$ 层内，checkpointing 进一步压缩。

但 checkpointing 与 FSDP 有交互：重算时 FSDP 需要重新 all-gather 权重——这意味着 checkpointing 不仅增加了计算，还可能增加通信。当 FSDP + full ckpt 叠加时，每层的权重被 all-gather 了两次（一次前向建 checkpoint，一次重算）。这是实践中 FSDP 训练比 Megatron 更慢的原因之一。

### 5.4 案例：M3 on Megatron TP4+PP4+ZeRO2 DP8, full ckpt

$L/pp = 60/4 = 15$ 层 per stage，$B_{\text{micro}} = 1$。

不加 ckpt 时激活 25.7 GB（见 CH 3.5）。加 full ckpt：

$$M_{\text{activation}} = 2 \times 15 \times 1 \times 8192 \times 6144 \approx 1.51 \text{ GB}$$

激活从 25.7 GB 降到 1.51 GB——节省约 17×。总显存从 126.0 GB 降到 $53.5 + 40.1 + 6.7 + 1.51 = 101.8$ GB。

101.8 GB > 80 GB——**即使 ZeRO-2 + full ckpt 叠加，仍然放不进 H100。** 权重（53.5 GB）和优化器（40.1 GB）两个大头合起来已占 93.6 GB，远超 80 GB 上限。这说明对 M3 这种 MoE 大模型，TP=4、PP=4 是不够的——需要更激进地切分权重和优化器。EP（Expert Parallelism）是唯一能把 expert 权重再打折的杠杆。

如果增大 $B_{\text{micro}}$（更大的 microbatch 有利于流水线填充和收敛稳定性）：

| $B_{\text{micro}}$ | 激活 (无 ckpt) | 激活 (full ckpt) | 总显存 (ckpt) |
|---|---|---|---|
| 1 | 25.7 GB | 1.51 GB | 101.8 GB |
| 2 | 51.3 GB | 3.0 GB | 103.3 GB |
| 4 | 102.6 GB | 6.0 GB | 106.3 GB |
| 8 | 205.3 GB | 12.1 GB | 112.4 GB |

所有配置均超 H100 80 GB——**在省略 EP 的前提下，此配置不可行**。CH 7 将引入 EP 展示完整解决方案。

---

## CH 6 | Offload：超出 GPU 的去处

当并行策略和 checkpointing 仍然不够时（或用户愿意用吞吐换更大的模型），Offload 把部分状态搬到 CPU 甚至 NVMe。

### 6.1 Optimizer Offload（ZeRO-Offload）

**做法**：优化器状态（FP32 master + momentum + variance = 12 bytes/param）搬到 CPU 内存。GPU 只保留 BF16 工作参数和梯度。反向完成后，梯度传 CPU → CPU 跑 Adam → 更新后的 FP32 参数传回 GPU。

**显存节省**：GPU 释放 $12N_{\text{per\_gpu}}$ bytes 优化器状态。$N_{\text{per\_gpu}}$ 取决于已叠加的并行折扣——含 EP 时 $N_{\text{per\_gpu}}$ 显著更小（约 5B vs 26.75B for M3），offload 的绝对数据量和吞吐代价也相应降低。

**吞吐代价**：每 iteration 的 PCIe 传输量取决于 per-GPU 参数量，而非总参数量。对于 M3, tp=4, pp=4（无 ZeRO optimizer sharding 时），每 GPU 持有 $428\text{B} / 16 \approx 26.75\text{B}$ 参数。单 GPU 每步传输 $4 \times 26.75 \times 10^9 = 107$ GB（$2N_{\text{per\_gpu}}$ 梯度 GPU→CPU + $2N_{\text{per\_gpu}}$ 更新参数 CPU→GPU）。PCIe 4.0 ×16 有效带宽约 25-32 GB/s 单向，传输需约 **3-5 秒**。加上 CPU 侧 Adam 计算（~8B params/s，约 3.3 秒），CPU+PCIe 阶段约 6-8 秒。GPU 前向+反向约 2-3 秒（B_micro=2, T=8192）。总体单步时间从 2-3 秒变为 8-11 秒——**吞吐降 3-5×**。若配合 ZeRO-2（dp=8）额外将 per-GPU 优化器分片缩至 $26.75\text{B}/8 \approx 3.3\text{B}$ params，传输量仅 13.4 GB，PCIe 时间 < 0.5 秒，吞吐代价可忽略——但此时 ZeRO-2 已经把优化器放进了 GPU，offload 不再必要。

关键结论：**optimizer offload 对大模型（tp×pp 已将 per-GPU 参数压缩）的吞吐代价是 3-5×，不是 50-100×**。50-100× 的经典数字适用于单卡小模型场景（如 ZeRO-Offload 论文中的 13B 单卡），那里 per-GPU N 就是全模型 N，且 GPU 计算步极快（< 1 秒），PCIe 时间占绝对主导。在大规模分布式训练中，per-GPU 参数量已被 TP/PP 压缩，offload 的相对代价显著降低。

**实践中 optimizer offload 更多用于 fine-tuning 而非 pre-training**：fine-tuning 的步数少、对吞吐不敏感。

### 6.2 Param Offload（ZeRO-3 + CPU Offload）

将 FSDP 的 all-gather 来源从 GPU HBM 改为 CPU 内存：参数分片常驻 CPU，前向时通过 PCIe 拉到 GPU，用后释放。

**显存节省**：GPU 释放 $2N/dp$ 的参数分片。极限情况下 GPU 上只有激活 + 临时计算缓冲。

**代价更大**：每层前向都要 PCIe 拉一次参数，后向要拉一次参数 + 传一次梯度。总 PCIe 传输量 $6N$ bytes/iteration（前向 all-gather $2N$ + 后向 all-gather $2N$ + 梯度 reduce-scatter $2N$）。

### 6.3 Activation Offload

只在最极端的场景使用（如单卡 fine-tune 超大模型 + 长序列）。将 checkpoint 之间的激活异步写到 CPU，反向时异步预取回来。

**吞吐代价极大**：激活量级通常远超参数（`34 × L × B × T × d` vs `N`），PCIe 成为严重瓶颈。实践中几乎只有学术研究场景使用。

### 6.4 Offload 选择决策树

```
显存放不下？
├── 权重放不下 → FSDP/ZeRO-3（GPU 分片）
│   └── 还是放不下 → Param Offload（CPU 分片）【吞吐降 5-20×】
├── 优化器放不下 → ZeRO-1/2（GPU 分片）
│   └── 还是放不下 → Optimizer Offload（CPU）【吞吐降 3-5×】
├── 激活放不下 → Gradient Checkpointing
│   └── 还是放不下 → 加 PP 或 CP
└── 激活还是放不下 → Activation Offload（CPU）【吞吐降 20-100×】
```

核心原则：优先级从高到低——GPU 内分片 > GPU 内重计算 > CPU offload > NVMe offload。每降一级，吞吐少一个数量级。

---

## CH 7 | 完整案例：M3 训练显存建模

从 config.json 出发，一步步建出 M3 训练的显存模型。本案例同时覆盖 Megatron TP+PP、ZeRO-2、gradient checkpointing 和 CP 的多层叠加。

### 7.1 模型基准数据

| 参数 | 符号 | 值 | 来源 |
|---|---|---|---|
| 总参 | $N$ | 428B | CH 2 参数分解 |
| 隐藏维 | $d$ | 6144 | config.json |
| 层数 | $L$ | 60 | config.json |
| Q 头数 | $H_q$ | 64 | config.json |
| KV 头数 | $H_{kv}$ | 4 | config.json, GQA 16:1 |
| head_dim | $D$ | 128 | config.json |
| FFN 中间维 (dense) | $d_{ff}^{\text{dense}}$ | 12288 | config.json |
| FFN 中间维 (MoE expert) | $d_{ff}^{\text{moe}}$ | 2048 | config.json |
| 专家数 | $N_E$ | 128 | config.json |
| top-k | $k$ | 4 | config.json |
| 词表大小 | $V$ | 200064 | config.json |

### 7.2 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 全局 batch size | $B_{\text{global}} = 32$ | 8192 tokens/样本 |
| Micro batch size | $B_{\text{micro}} = 2$ | 每 GPU 每步前向的样本数 |
| 序列长度 | $T = 8192$ | 短序列训练（非长上下文） |
| 精度 | BF16 + FP32 optimizer | 混合精度训练 |
| Optimizer | AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$) | — |

**并行配置**（CH 7 完整版，含 EP）：

| 维度 | 度 | 说明 |
|---|---|---|
| TP | 4 | 机内 NVLink：切 QKV/O/FFN 矩阵列和行 |
| PP | 4 | 60 层切 4 段，每段 15 层 |
| EP | 8 | 128 experts / 8 = 16 experts/GPU |
| DP (ZeRO-2) | 4 | 数据并行 + 优化器和梯度分片 |
| CP | 1 | $T = 8192$ 不需要 CP |

总 GPU 数 = $tp \times pp \times ep \times dp = 4 \times 4 \times 8 \times 4 = \mathbf{512}$ 张 H100。

**为什么 CH 3-5 的简化案例（无 EP）算出来放不下，而加上 EP 就能放下**：CH 3-5 的权重公式使用 $428/(tp \times pp) = 26.75\text{B}$ params/GPU，BF16 权重 53.5 GB，加上优化器 40.1 GB（ZeRO-2）已超 80 GB。这不是设计错误——它证明了 **EP 对 MoE 大模型不是可选项，是硬需求**。以下 EP=8 将 expert 权重（约占 93%）额外打 $1/8$ 折，per-GPU params 从 26.75B 降到约 5B（expert 部分 3.1B + 非 expert 部分 1.9B），权重从 53.5 GB 降到 ~10 GB。

**Gradient accumulation**：

$$B_{\text{global}} = dp \times B_{\text{micro}} \times \text{grad\_accum\_steps} \quad \Rightarrow \quad 32 = 4 \times 2 \times 4$$

### 7.3 逐项显存分解（含 EP=8）

#### 权重

非 expert 参数（Attention、Dense 层、Embedding、LM Head，约占 7% ≈ 30B）受 $tp \times pp$ 折扣。Expert 参数（128 experts × MoE FFN，约占 93% ≈ 398B）受 $tp \times pp \times ep$ 折扣：

$$M_{\text{weight}} \approx \frac{2 \times 30 \times 10^9}{4 \times 4} + \frac{2 \times 398 \times 10^9}{4 \times 4 \times 8}$$

$$= 3.75 + 6.22 \approx \mathbf{10.0 \text{ GB}}$$

（精确值取决于 expert 和非 expert 的实际比例，此处取近似。10 GB 在合理的精度范围内。）

#### 优化器 (ZeRO-2)

EP 对优化器状态的折扣：非 expert 部分 $(tp \times pp \times dp)$，expert 部分 $(tp \times pp \times ep \times dp)$：

$$M_{\text{optimizer}} \approx \frac{12 \times 30 \times 10^9}{4 \times 4 \times 4} + \frac{12 \times 398 \times 10^9}{4 \times 4 \times 8 \times 4}$$

$$= 5.63 + 9.33 \approx \mathbf{14.9 \text{ GB}}$$

#### 梯度 (ZeRO-2)

同理：$$M_{\text{grad}} \approx \frac{2 \times 30 \times 10^9}{4 \times 4 \times 4} + \frac{2 \times 398 \times 10^9}{4 \times 4 \times 8 \times 4} \approx 0.94 + 1.55 = \mathbf{2.5 \text{ GB}}$$

#### 激活（full ckpt）

激活不受 EP 影响（只受 PP、B_micro、T 影响），与前述计算一致：

$$M_{\text{activation}}^{\text{ckpt}} = 2 \times 15 \times 2 \times 8192 \times 6144 \approx \mathbf{3.0 \text{ GB}}$$

### 7.4 汇总表

| 组分 | 值 | 占比 |
|---|---|---|
| 权重 | 10.0 GB | 32.8% |
| 优化器 (ZeRO-2) | 14.9 GB | 48.9% |
| 梯度 (ZeRO-2) | 2.5 GB | 8.2% |
| 激活 (full ckpt) | 3.0 GB | 9.8% |
| **合计** | **~30.4 GB** | — |

30.4 GB $\ll$ 80 GB（H100）——**加上 EP 后终于安全了**。对比不加 EP 的 101.8 GB（CH 5.4），EP=8 将权重和优化器合起来从 93.6 GB 压到 24.9 GB——节省 3.75×。

**Muon 优化器的节省**：若用 Muon 替代 AdamW（仅 expert 权重矩阵部分），优化器从 14.9 GB 降到约 10.0 GB，总显存从 30.4 GB 降到 25.5 GB。

### 7.5 敏感度分析

在 EP=8 的前提下，$B_{\text{micro}}$ 和 $T$ 的变动只影响激活（激活不受 EP 影响），权重/优化器/梯度固定。

#### $B_{\text{micro}}$ 的影响

| $B_{\text{micro}}$ | 激活 (ckpt) | 总显存 | H100 余量 |
|---|---|---|---|
| 1 | 1.5 GB | 28.9 GB | 51.1 GB |
| 2 | 3.0 GB | 30.4 GB | 49.6 GB |
| 4 | 6.0 GB | 33.4 GB | 46.6 GB |
| 8 | 12.1 GB | 39.5 GB | 40.5 GB |

$B_{\text{micro}}=1$ 时 $B_{\text{global}} = 4 \times 1 \times 8 = 32$，$B_{\text{micro}}=2$ 时 $B_{\text{global}} = 4 \times 2 \times 4 = 32$。$B_{\text{micro}} \geq 4$ 需调整 global batch 或 grad_accum。

#### 长序列训练：$T = 128\text{K}$

$$M_{\text{activation}}^{\text{ckpt}} = 2 \times 15 \times 2 \times 131{,}072 \times 6144 \approx 48.3 \text{ GB}$$

总显存 = $10.0 + 14.9 + 2.5 + 48.3 = 75.7$ GB。紧贴 H100 上限。CP=2 可压到 $10.0 + 14.9 + 2.5 + 24.2 = 51.6$ GB。

#### 极长序列训练：$T = 1\text{M}$

$$M_{\text{activation}}^{\text{ckpt}} = 60 \times 1{,}048{,}576 \times 6144 \approx 386 \text{ GB}$$

必须 CP ≥ 8。CP=8 时 $386/8 = 48.3$ GB，总显存 = 75.7 GB。

### 7.6 关键结论

1. **EP 是 MoE 训练的硬需求，不是可选优化**——不加 EP 时权重+优化器即占 93.6 GB（> 80 GB），所有其他优化（ZeRO、ckpt、CP）都救不了。EP 将 expert 部分打 $1/ep$ 折，使权重和优化器降到 GPU 可容纳的范围。
2. **激活是训练显存的第二大变量**——EP 解决权重和优化器后，激活（full ckpt 下 3.0 GB）占比较小，但随 $T$ 线性增长。$T=128\text{K}$ 时激活跃升为主要瓶颈（48.3 GB）。
3. **TP×PP×EP×DP 四维叠加才能装下 M3**——缺一不可。TP+PP 切权重（$16\times$），EP 切 experts（$8\times$），ZeRO-2 切优化器和梯度（$4\times$），full ckpt 压激活（$17\times$）。总节省倍率 $\approx 16 \times 8 \times 4 \times 17 \approx 8700\times$，从单卡 6.8 TB 压到 30 GB。

---

## CH 8 | 多模态训练：Vision Encoder 的显存叠加

M3、K2.5 等多模态模型在 LLM 主干之外还有一个 Vision Encoder（通常是 ViT），以及连接两者的 Projector。本章在 CH 1-7 的 LLM 显存框架上叠加视觉部分的建模。

### 8.1 多模态训练的显存全景

多模态训练的数据包含图文对——每张图像经 ViT 编码为 $N_{\text{img}}$ 个视觉 token，与文本 token 拼接后送入 LLM 主干。显存结构变为：

$$M_{\text{total}} = M_{\text{LLM}} + M_{\text{ViT}} + M_{\text{Projector}}$$

其中 $M_{\text{LLM}}$ 按 CH 1-7 完整建模（含 TP/PP/EP/ZeRO/ckpt/CP）。$M_{\text{ViT}}$ 和 $M_{\text{Projector}}$ 是本章的新增部分。

### 8.2 视觉 Token 对 LLM 主干的影响（核心变量）

ViT 本身的显存不大（~1%），但视觉 token **进入 LLM 主干后会膨胀激活和注意力计算**。这是多模态训练显存最容易被忽视的变量。

视觉 token 经 Projector 投影后与文本 token 拼接，送入 LLM 主干。LLM 的序列长度变为：

$$T = T_{\text{text}} + N_{\text{img}} \times N_{\text{images}}$$

LLM 主干的激活按 CH 1.4 的公式随 $T$ **线性增长**，attention QK 点积随 $T$ **平方增长**。以 M3（$d=6144$, $L=60$, $B_{\text{micro}}=2$, full ckpt, $T_{\text{text}}=8192$）为例：

| 图像配置 | $N_{\text{img}}$ | 总 $T$ | LLM 激活 (ckpt) | vs 纯文本 |
|---|---|---|---|---|
| 纯文本 | 0 | 8,192 | 3.0 GB | baseline |
| 1 张图（标准分辨率） | 576 | 8,768 | 3.2 GB | +7% |
| 4 张图（标准分辨率） | 2,304 | 10,496 | 3.9 GB | +28% |
| 1 张图（高分） | 2,916 | 11,108 | 4.1 GB | +36% |
| 4 张图（高分, 如 M3 动态分辨率） | 11,664 | 19,856 | 7.3 GB | **+142%** |

**4 张高分图 + full ckpt 下，LLM 激活从 3.0 GB 涨到 7.3 GB——这才是多模态训练真正的显存代价，而非 ViT 本身。** 在无 ckpt 场景下，激活从 51.3 GB 涨到 124.5 GB——直接超过单卡容量。

Prefill 阶段（训练的前向 pass 一次性处理所有 token），attention QK 点积随 $(T_{\text{text}} + N_{\text{img}})^2$ 增长。当 $T=19{,}856$ 时，attention score 矩阵的 FLOPs 是 $T=8{,}192$ 时的 $(19856/8192)^2 \approx 5.9\times$——prefill 的计算时间和临时显存都大幅增加。

**KV cache for visual tokens**：视觉 token 的 KV 同样需要缓存（供反向传播的 attention backward 使用）。按 CH 4.1 的公式，额外 KV = $4 \times L \times H_{kv} \times D \times N_{\text{img}}$ bytes。对 M3（$H_{kv}=4$, $D=128$）：每张 576-token 图像增加约 $4 \times 60 \times 4 \times 128 \times 576 \approx 71$ MB。4 张高分图约 1.4 GB——在极紧的显存预算下需要计入。

### 8.3 ViT 的参数和激活

ViT 是一个标准 Dense Transformer（无 MoE），通常只有 ~20-40 层，隐藏维远小于 LLM 主干。其显存方程与 CH 1 的 Dense 模型完全相同：

$$M_{\text{weight}}^{\text{ViT}} = N_{\text{ViT}} \times 2$$

$$M_{\text{optimizer}}^{\text{ViT}} = N_{\text{ViT}} \times 12$$

$$M_{\text{grad}}^{\text{ViT}} = N_{\text{ViT}} \times 2$$

$$M_{\text{activation}}^{\text{ViT}} = 34 \times L_{\text{ViT}} \times B \times N_{\text{img}} \times d_{\text{ViT}} \text{ bytes}$$

以 M3 的 ViT 为例（$L_{\text{ViT}} = 32$, $d_{\text{ViT}} = 1280$, $H_{\text{ViT}} = 16$, $D_{\text{ViT}} = 80$, $N_{\text{img}} = 576$ per image）：

| 组分 | 计算 | 值 |
|---|---|---|
| ViT 权重 | $N_{\text{ViT}} \times 2$（$N_{\text{ViT}} \approx 0.4\text{B}$） | ~0.8 GB |
| ViT 优化器 | $N_{\text{ViT}} \times 12$ | ~4.8 GB |
| ViT 梯度 | $N_{\text{ViT}} \times 2$ | ~0.8 GB |
| ViT 激活（B=2, N_img=576） | $34 \times 32 \times 2 \times 576 \times 1280$ | ~1.6 GB |

与 LLM 主干（428B 参数 + 103 GB 激活）相比，ViT 的显存开销几乎可忽略——**权重仅 0.8 GB vs 53.5 GB（1.5%），激活 1.6 GB vs 25.7 GB（6%）**。这是因为 ViT 的规模比 LLM 主干小两个数量级。

但有一个细节：**Projector 的权重和优化器**。Projector 通常是一个 2 层 MLP（如 `d_ViT → d → d`）或更复杂的 QFormer 结构。以 2 层 MLP 为例，参数量约为 $2 \times d_{\text{ViT}} \times d + d \times d \approx 2 \times 1280 \times 6144 + 6144 \times 6144 \approx 53\text{M}$，权重约 0.1 GB——可忽略。Projector 的激活也极小（每层约 $B \times N_{\text{img}} \times d \times 2 \approx 14$ MB per layer for B=2），因为视觉 token 数远小于文本 token 数。

### 8.4 ViT 的并行策略：与 LLM 主干的差异

ViT 是 Dense 模型（无 MoE），这意味着：

**不需要 EP**。ViT 的 FFN 层没有 experts，无法使用 Expert Parallelism。好消息是 ViT 本身参数少，不需要 EP。

**TP 对 ViT 有效但通常不必要**。ViT 权重仅 0.8 GB，TP 的额外通信得不偿失。实践中 ViT 通常在每张 GPU 上全量复制（数据并行），或跟随 LLM 主干的 PP 放置在第一 stage。

**PP 下的放置**。ViT 和 Projector 通常放在 PP 的第一个 stage（与 embedding 层同 stage）。这意味着第一个 PP stage 的 GPU 额外承担了 ViT + Projector 的显存——取决于 ZeRO 配置，约 3-7 GB per GPU。其他 PP stage 的 GPU 完全不受影响。在 tight budget 下（如 $T=1\text{M}$），这可能成为第一个 stage OOM 的最后一根稻草——此时需要减小 $B_{\text{micro}}$ 或增大 CP。

**图像 token 数的影响**。$N_{\text{img}}$ 决定 ViT 的激活量，也影响 Projector 的 I/O。高分辨率图像或多图像输入会显著增加 $N_{\text{img}}$——如 M3 支持的动态分辨率最高可输出 2916 tokens/image。在 $B=2$, 4 张图的场景下，$N_{\text{img}} = 11,664$，ViT 激活上升到约 8 GB——不再是可忽略的项。

### 8.5 核心结论

1. **ViT 本身的显存开销通常可忽略**（< LLM 主干的 2%）
2. **视觉 token 对 LLM 激活的膨胀才是真正代价**——$T = T_{\text{text}} + N_{\text{img}}$，激活随 $T$ 线性增长，attention QK 随 $T^2$ 增长。4 张高分图可让 LLM 激活 +142%
3. **PP 第一 stage 是瓶颈**——同时承载 embedding + ViT + Projector + 首段 LLM 层 + 视觉 token 导致的激活膨胀
4. **ViT 不用 EP**（无 MoE），通常全量复制或放 PP 第一 stage。ViT 参数少，不需要特殊的并行优化

---

## CH 9 | LoRA / QLoRA 微调：另一种显存方程

预训练需要数百到数千张 GPU，但绝大多数从业者做的是**微调**（fine-tuning）——在预训练模型上用自己的数据继续训练。LoRA 将微调的显存需求从"必须多卡并行"压到"单卡可做"，是当前最主流的微调范式。

LoRA 的显存方程与预训练有**结构性差异**——不再是"所有权重都有优化器状态"，只有一小部分可训练参数需要优化器。

### 9.1 LoRA 的显存结构

LoRA 冻结预训练模型的所有原始权重，只在选定的权重矩阵旁插入低秩适配器：

$$W_{\text{updated}} = W_{\text{frozen}} + \frac{\alpha}{r} \cdot A \cdot B$$

其中 $A \in \mathbb{R}^{d_{\text{in}} \times r}$, $B \in \mathbb{R}^{r \times d_{\text{out}}}$，$r \ll d$（通常 $r = 8\text{-}64$）。

**只有 $A$ 和 $B$ 需要训练**——这意味着优化器和梯度只对 LoRA 参数生效。

显存结构变为：

$$M_{\text{total}} = \underbrace{M_{\text{weight}}^{\text{frozen}}}_{\text{base model}} + \underbrace{M_{\text{weight}}^{\text{LoRA}}}_{\text{adapters}} + \underbrace{M_{\text{optimizer}}^{\text{LoRA}}}_{\text{only LoRA params}} + \underbrace{M_{\text{grad}}^{\text{LoRA}}}_{\text{only LoRA params}} + \underbrace{M_{\text{activation}}}_{\text{unchanged}}$$

与预训练的核心差异：

| 组分 | 预训练 | LoRA 微调 |
|---|---|---|
| 权重 | $2N$（全部可训） | $2N$（frozen） + $2N_{\text{LoRA}}$ |
| 优化器 | $12N$ | $\mathbf{12N_{\text{LoRA}}}$（只对 adapter） |
| 梯度 | $2N$ | $\mathbf{2N_{\text{LoRA}}}$（只对 adapter） |
| 激活 | $34LBTd$ | $34LBTd$（**完全相同**） |

关键洞察：**激活不变**。前向传播仍然经过所有原始层和 LoRA adapter 拼合后的完整计算图，反向传播需要同样的中间结果。因此 LoRA 不省激活——只省优化器和梯度。

### 9.2 LoRA 参数量公式

设 LoRA 应用到所有 Attention 层的 Q、K、V、O 四个投影矩阵（最常见配置），每层 LoRA 参数量：

$$N_{\text{LoRA}}^{\text{per\_layer}} = 4 \times 2 \times d \times r = 8dr$$

全模型 LoRA 参数量：$N_{\text{LoRA}} = L \times 8dr$。

以 Llama-3 70B（$d = 8192$, $L = 80$, $r = 16$）为例：

$$N_{\text{LoRA}} = 80 \times 8 \times 8192 \times 16 = 83{,}886{,}080 \approx \mathbf{84\text{M}}$$

对比总参（70B）：**84M / 70B ≈ 0.12%**。优化器状态和梯度的显存因此降低三个数量级。

### 9.3 单卡 LoRA 微调 Llama-70B（BF16）

| 组分 | 计算 | 值 |
|---|---|---|
| Base model 权重（frozen, BF16） | $70 \times 10^9 \times 2$ | 140 GB |
| LoRA 权重（BF16） | $84 \times 10^6 \times 2$ | ~0.17 GB |
| LoRA 优化器（Adam） | $84 \times 10^6 \times 12$ | ~1.0 GB |
| LoRA 梯度（BF16） | $84 \times 10^6 \times 2$ | ~0.17 GB |
| 激活（B=1, T=8192, 无 ckpt） | $34 \times 80 \times 1 \times 8192 \times 8192$ | ~183 GB |
| 激活（full ckpt） | $2 \times 80 \times 1 \times 8192 \times 8192$ | ~10.7 GB |
| **合计（无 ckpt）** | | **~324 GB** |
| **合计（full ckpt）** | | **~152 GB** |

152 GB > 80 GB——**即使 LoRA 将优化器和梯度压到几乎为零，base model 权重（140 GB）仍然远超单卡 H100 的 80 GB**。加上激活后完全不可行。

这就是 QLoRA 存在的理由。

### 9.4 QLoRA：量化 base model，让权重装得下

QLoRA 将 base model 权重量化为 4-bit（NF4 格式），训练时动态反量化到 BF16 做前向/反向传播：

$$M_{\text{weight}}^{\text{frozen}} = N \times 0.5 \text{ bytes} \quad (\text{INT4/NF4})$$

| 组分 | 计算 | 值 |
|---|---|---|
| Base model（NF4） | $70 \times 10^9 \times 0.5$ | **35 GB** |
| LoRA 权重 + 优化器 + 梯度 | 同上 | ~1.3 GB |
| 激活（full ckpt） | 同上 | ~10.7 GB |
| **合计（full ckpt）** | | **~47 GB** |

47 GB < 80 GB（H100）——**QLoRA 将 70B 模型的微调塞进了单张 H100**。甚至 48 GB 的 A6000 也能勉强跑（需减小 batch 或序列长度）。

**反量化的临时显存**：QLoRA 的前向传播中，每个被计算的层需要将 NF4 权重临时反量化为 BF16，瞬时产生 $2d^2$ 的额外显存（约 $2 \times 8192^2 \approx 134$ MB per layer）。由于反量化是逐层进行的，只有当前层的反量化结果驻留，前一层的即时释放——这一开销通常 < 500 MB，不是瓶颈。

### 9.5 LoRA rank 和模块选择的工程 trade-off

**Rank 的影响**：$N_{\text{LoRA}} \propto r$。从 $r=8$ 到 $r=64$，LoRA 参数量 8×，但优化器和梯度仍然远小于 base model 权重，显存影响微乎其微。Rank 的选择更多取决于**模型质量**（更高 rank 通常更好，但有边际递减），而非显存约束。

**模块选择的影响**。扩展 LoRA 到更多模块会增加可训参数：

| LoRA 目标 | $N_{\text{LoRA}}$ 倍率 vs QKVO only | 典型用途 |
|---|---|---|
| QKVO only | 1× | 最常见，性价比最高 |
| + FFN gate/up/down | ~2.5× | 需要更强的领域适配 |
| + 所有线性层 | ~3.5× | 接近全参微调效果 |

即使适配所有线性层（$N_{\text{LoRA}} \approx 300\text{M}$ for 70B），优化器状态也仅约 3.6 GB——仍远小于 base model 权重。**LoRA 的显存约束几乎永远不会来自 LoRA 参数量本身，而是来自 base model 权重和激活。**

### 9.6 Batch size 和序列长度对 QLoRA 的约束

QLoRA 场景下，显存预算的三部分比重为：

$$\text{base model (35 GB)} \gg \text{activation (ckpt, ~10 GB)} \gg \text{LoRA states (~1 GB)}$$

base model 权重固定占 35 GB（以 70B 模型为例），剩余约 45 GB（H100 80 GB 下）可供激活使用。

| $T$ | 激活 (B=1, ckpt) | 激活 (B=2, ckpt) | 激活 (B=4, ckpt) | 最大可行 $B$ |
|---|---|---|---|---|
| 2048 | 2.7 GB | 5.4 GB | 10.7 GB | ~14 |
| 8192 | 10.7 GB | 21.5 GB | 43 GB | 4 |
| 32768 | 43 GB | 86 GB（OOM） | — | 1 |

结论：QLoRA 的微调在长上下文（>32K）下仍需减小 batch 或使用 gradient accumulation。这跟预训练长序列要加 CP 是同一个根本原因——**激活随 $T$ 线性增长，ckpt 能压 17× 但不能消除它。**

### 9.7 核心结论

1. **LoRA 不省激活**——省的是优化器和梯度（从全参缩到 LoRA 参数）。激活量与全参微调完全相同
2. **QLoRA 的核心价值是将 base model 权重从 140 GB 压到 35 GB**，配合 LoRA 使 70B 微调在单卡可行
3. **LoRA rank 和模块选择对显存几乎无影响**——瓶颈始终在 base model 权重和激活
4. **长上下文 QLoRA 微调仍需 ckpt + CP**——因为激活方程与预训练完全相同

---

## CH 10 | 从 Engineering 视角总结

### 10.1 显存优化的帕累托前沿

每项显存优化都有代价。按"性价比"排序（每 GB 节省的代价）：

| 优化 | 显存节省 | 代价 | 性价比 |
|---|---|---|---|
| FlashAttention | 激活 $C_{\text{act}}$: 354→34（~10×） | 无 | ★★★★★ |
| Gradient ckpt (full) | 激活 ~17× | 30% 重算 FLOPs | ★★★★★ |
| ZeRO-1 (optimizer shard) | 优化器 $12N \rightarrow 12N/dp$ | 轻度梯度通信 | ★★★★☆ |
| Megatron PP | 全项 $1/pp$ | Bubble + 跨 stage 通信 | ★★★★☆ |
| Megatron TP | 权重 $1/tp$ | 重度 all-reduce（机内 NVLink） | ★★★☆☆ |
| Megatron EP (MoE 专用) | Expert 权重 $1/ep$ | all-to-all（跨节点昂贵） | ★★★★☆ |
| ZeRO-3 (param shard) | 权重 $2N \rightarrow 2N/dp$ | 重度 all-gather + 通信→显存换吞吐 | ★★★☆☆ |
| HSDP (FSDP2) | 权重峰值 $2N \rightarrow 2N/dp_{\text{replicate}}$ | 分层通信，配置复杂 | ★★★☆☆ |
| CP | 激活 $1/cp$ | P2P ring 通信（串行延迟） | ★★☆☆☆ |
| Optimizer Offload | 优化器 → 0（GPU）| PCIe 传输 3-5× 吞吐降（大规模分布式下） | ★★☆☆☆ |
| Param Offload | 权重 → 0（GPU）| PCIe + CPU 延迟 5-20× | ★☆☆☆☆ |

### 10.2 训练配置决策树

```
给定模型 + GPU 预算
│
├─ Step 1: 算 base 方程（单卡装得下吗？）
│   └─ 428B × (2+12+2) = 6.9 TB → 不可能
│
├─ Step 2: 选 TP（切矩阵，机内 NVLink 支撑）
│   └─ TP 通常固定在机内 GPU 数（4 or 8）
│
├─ Step 3: 选 PP（切层，减轻激活压力）
│   └─ pp 越大 bubble 越大，取 pp ≤ 8
│
├─ Step 4: (MoE 专用) 选 EP（切 experts）
│   └─ 使权重+优化器落入 GPU 预算。EP ≥ 8 对百 B 级 MoE 几乎必须
│
├─ Step 5: 选 ZeRO stage（切优化器/梯度）
│   └─ ZeRO-2 是 sweet spot，ZeRO-3 仅在权重仍超预算时启用
│
├─ Step 6: 开 FlashAttention + full ckpt
│   └─ FA 消 attention score + ckpt 压激活 ~17×
│
├─ Step 7: 判断是否需要 CP
│   └─ 检查 $T$ 下的激活是否超预算 → 加 cp
│
└─ Step 8: 如果还不够 → Offload 或增加 GPU 预算
```

### 10.3 读者自检清单

读完本文后，你应该能回答：

- [ ] 为什么训练显存是推理的约 8 倍？8 倍的来源是什么？
- [ ] Adam 优化器的 12 bytes/param 分别对应哪些 FP32 张量？Muon、Adafactor 和 FP8 optimizer 如何压缩？
- [ ] FlashAttention 减少激活存储的机制是什么？它消除了公式中哪一项？
- [ ] TP 为什么不能减少激活？PP 为什么能？EP 为什么对 MoE 训练是硬需求？
- [ ] FSDP/ZeRO-3 的"权重峰值"和"权重驻留"有什么区别？为什么会有这个差异？
- [ ] HSDP 如何改进 FSDP1 的权重峰值问题？
- [ ] 给定 M 模型 + G GPU 配置，如何快速判断是否需要 CP？
- [ ] Gradient checkpointing 的 17× 激活节省从何而来？代价是什么？
- [ ] Optimizer offload 和 param offload 分别更适合什么场景？
- [ ] LoRA 优化器和梯度为什么能省三个数量级？LoRA 省不省激活？为什么？
- [ ] QLoRA 的核心价值是什么？base model 权重从 140 GB 降到 35 GB 靠的是什么？
- [ ] 多模态训练中，ViT 的显存开销在什么场景下从"可忽略"变为"不可忽略"？
- [ ] Megatron TP+PP vs FSDP 的选择，激活压力如何影响决策？

---

> **系列导航**：[（一）预备知识](../part-1/) → [（二）FLOPs](../part-2/) → [（三）KV Cache](../part-3/) → [（四）M3+Roofline](../part-4/) ← 当前 → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)

> **符号约定**：与[系列第一篇](../part-1/)完全一致。新增符号：$B_{\text{global}}$ = 全局 batch size，$B_{\text{micro}}$ = 每 GPU 每次前向的 micro batch size，$tp/pp/dp/ep/cp$ = 各并行维度度数。

> **关于本文**：所有公式均在 MiniMax M3 (428B) 和 Llama-3 70B 的已知公开配置上验证通过。激活公式基于 Korthikanti et al. (2022) 并经 FlashAttention 修正。FSDP2 HSDP 信息基于 PyTorch 2.6 文档。CP 公式基于 Ring Attention 原始论文和 LoongTrain 的工程验证。

> **计量约定**：训练显存估算中，权重/优化器/梯度使用十进制 GB（$10^9$ bytes，行业速算惯例），激活存储使用二进制 GiB（$1024^3$ bytes，贴近 GPU 硬件规格）。两者差异约 7%，本章的估算精度（±15%）内可混用。GPU 显存规格以 GiB 标注（如 H100 = 80 GiB）。

> **系列导航**：[（四）M3 实战 + Roofline](../part-4/) ← 当前 → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)
