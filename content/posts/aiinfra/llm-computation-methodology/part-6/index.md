+++
date = '2026-06-22'
draft = false
title = 'LLM 系统分析方法论（六）：训练通信与掩盖分析'
categories = ['aiinfra']
tags = ['training', 'communication', 'nccl', 'parallelism', 'methodology', 'm3', 'ascend']
series = 'llm-computation-methodology'
series_order = 6
math = true
summary = '训练通信完整分析：从物理原理到框架实现，覆盖 TP/PP/DP/EP/CP/FSDP2 六种并行维度的通信模式、时间线建模和掩盖策略。含 M3 完整 step time 推演和 Dense 70B/M3 MoE 多场景实战。跨 NVIDIA + Ascend 双平台。'
+++


> 从物理原理出发，按框架行为拆解，建立跨平台（NVIDIA + Ascend）的通信分析能力。

## 本文定位

[系列第五篇](../part-5/) 解决了"装得下吗"（显存），本文解决"跑多快"（通信）。两者的衔接点：显存文档的 CH 7 给出了 M3 的可行并行配置（tp=4,pp=4,ep=8,dp=4），本文在这个配置上分析每步迭代的通信时间、掩盖率和并行效率 $\eta$。

核心问题：**GPU 之间必须传数据，传多久，能不能藏到计算背后。**

本文覆盖 NVIDIA（NCCL + Megatron-LM/FSDP）和 Ascend（HCCL + MindSpeed-LLM）两个平台。物理原理跨平台通用，实现细节分别展开。Ascend 侧以 **910C** 为主要参考硬件，以 **MindSpeed-LLM**为框架参考。

---

## CH 1 | GPU 为什么会通信

在引入任何通信原语之前，先建立通信需求的来源。

### 1.1 并行维度的数据分布

每种并行策略创造了一种"数据不在一张卡上"的局面，通信是为了修正这个局面：

| 并行维度 | 数据分布 | 通信需求 | 为什么必须通信 |
|---|---|---|---|
| DP | 每张卡有完整模型，但梯度来自不同数据 | 梯度求平均 | 否则每张卡的参数更新方向不同 |
| TP | 每张卡有矩阵的一部分（列/行） | 部分结果求和 | matmul 的数学定义要求完整结果 |
| PP | 每张卡有模型的一部分（层） | 激活/梯度传递 | 数据流经层序列 |
| EP | 每张卡有部分 experts | token 路由 | 每个 token 需要找到它激活的 expert |
| CP | 每张卡有序列的一部分 | KV chunk 交换 | attention 需要跨序列段的信息 |
| ZeRO | 每张卡有参数/优化器的分片 | 收集/归约 | 计算时需要完整参数，更新后需要分片 |

这张表是全文的索引。后续每个章节展开其中一个维度，回答三个问题：传什么、传多少、能不能藏。

### 1.2 训练的通信全景

以 M3 on Megatron (tp=4, pp=4, ep=8, dp=4) 为例，单步训练的通信总览：

```
一个 iteration 中的所有通信操作（按发生顺序）:

Forward:
  MoE 层 ×57: [gate] → [EP all-to-all (dispatch)] → [expert FFN (含内部 TP all-reduce)] → [EP all-to-all (combine)]
  Dense 层 ×3: [attention (含内部 TP all-reduce)] → [FFN (含内部 TP all-reduce)]
  PP boundary: [p2p send/recv 激活]

Backward:
  MoE 层 ×57: [expert FFN bwd (含内部 TP all-reduce)] → [EP all-to-all (combine bwd)] → [EP all-to-all (dispatch bwd)]
  Dense 层 ×3: [attention bwd (含内部 TP all-reduce)] → [FFN bwd (含内部 TP all-reduce)]
  PP boundary: [p2p send/recv 梯度]
  DP: [reduce-scatter 梯度] (与 backward 重叠, ZeRO-2 分桶机制)
```

TP all-reduce（RowParallelLinear 中的 `reduce_from_tensor_model_parallel_region`）发生在 FFN 和 attention output 的 matmul 内部，不是独立的通信步骤。ColumnParallelLinear 的梯度 all-reduce 发生在 backward 中，由 autograd 的 backward hook 触发。

---

## CH 2 | 通信掩盖的物理原理

这是全文最重要的基础章节。回答一个核心问题：**计算和通信为什么能同时进行？**

### 2.1 硬件分离：计算单元和通信单元是两套独立的物理硬件

以 NVIDIA H100 为例的芯片架构：

```
┌─────────────────────────────────────────┐
│                H100 GPU                 │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  SM (132个)  │  │  Copy Engine    │  │ ← 物理上独立
│  │  + Tensor    │  │  (DMA 引擎)     │  │
│  │  Core (528个)│  │                 │  │
│  │  [计算单元]  │  │  [通信搬运单元] │  │
│  └──────┬───────┘  └───────┬─────────┘  │
│         │                  │            │
│  ┌──────┴──────────────────┴─────────┐  │
│  │          HBM3 (80 GB)             │  │ ← 共享显存
│  │          3.35 TB/s                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │                    │
    NVLink (900 GB/s)   InfiniBand (50 GB/s)
    [机内, 全互联]       [跨机, 每GPU一个HCA]
```

关键事实：**SM 做矩阵乘法时，Copy Engine 可以同时在 NVLink 上搬运另一块数据。** 两者访问同一块 HBM（共享显存），但执行单元独立。

Ascend 910C 的架构（双 Die MCM 封装，统一 128 GB HBM2e 地址空间，每 Die 64 GB）：

```
┌──────────────────────────────────────────────────┐
│                  Ascend 910C                     │
│  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Die 0          │  │  Die 1          │        │
│  │  ┌───────────┐  │  │  ┌───────────┐  │        │
│  │  │ AICore    │  │  │  │ AICore    │  │        │
│  │  │ (Da Vinci)│  │  │  │ (Da Vinci)│  │        │
│  │  └───────────┘  │  │  └───────────┘  │        │
│  │  TS + RoCE 引擎 │  │  TS + RoCE 引擎 │        │
│  │  ┌───────────┐  │  │  ┌───────────┐  │        │
│  │  │ HBM2e     │  │  │  │ HBM2e     │  │        │
│  │  │ 64 GB     │  │  │  │ 64 GB     │  │        │
│  │  └───────────┘  │  │  └───────────┘  │        │
│  └────────┬────────┘  └────────┬────────┘        │
│           │       Die间       │                  │
│           └──── 540 GB/s ─────┘                  │
│           (270 GB/s 单向, D2D)                    │
└──────────────────────────────────────────────────┘
           │                          │
    HCCS (392 GB/s bidir)    UB 总线 (Spine-Leaf Clos, 全光 LPO)
    [机内, 8卡全互联]         [跨节点, 实测 164 GB/s 读/135 GB/s 写]
                             [<1μs 额外延迟, <3% 带宽退化]
```

关键差异：910C 的 **UB 总线跨节点带宽与机内 HCCS 带宽接近**（实测 164 GB/s 读 vs 理论 392 GB/s），延迟几乎相同（<1μs 退化），这使得 Ascend 侧的跨节点通信分析**不能简单套用 NVIDIA 的"跨节点 >10× 慢于机内"假设**。注意 UB 的 392 GB/s 是链路层理论聚合值（7×224 Gbps ×2 Die），应用层实测约 42% 利用率。文档后续使用实测值。

### 2.2 软件接口：CUDA Stream 和 ACL Stream

**Stream 是 GPU 上操作（kernel launch、memcpy、collective）的排队机制。** 核心法则：

1. **同一 stream 上的操作严格串行**——必须先完成 A 才能开始 B
2. **不同 stream 上的操作可以并发**——前提是没有显式同步（`cudaStreamSynchronize` / `aclrtStreamSynchronize`）
3. **数据依赖打破并发**——如果操作 B 需要操作 A 的输出，即使它们在不同 stream 上，B 也必须等 A 完成

第 3 点是掩盖分析的核心约束：**不是所有通信都能被掩盖，只有那些与后续计算没有数据依赖的通信才有可能。**

### 2.3 掩盖成功的判定条件

```
时间线（掩盖成功）:
Stream 0 (compute):  [===== 计算 A =====][===== 计算 B =====]
Stream 1 (comm):          [== 通信 ==]                      ← 通信完成时，计算 B 还在进行
                          ↑ 通信启动          ↑ 通信完成，结果已就绪

时间线（掩盖失败）:
Stream 0 (compute):  [== 计算 A ==][······ 空闲 ······][== 计算 B ==]  
Stream 1 (comm):          [====== 通信 ======]   ← 通信比计算 A 长，计算 B 被迫等待
                                           ↑ 通信完成时，计算 B 才能开始
```

掩盖成功的充分条件：

$$T_{\text{compute\_after\_comm}} \geq T_{\text{comm}}$$

即**通信操作之后、需要通信结果的那个计算之前，中间插入的独立计算必须足够长，长到覆盖通信时间。** 如果中间没有独立计算——或者说中间的计算需要通信结果（有数据依赖）——则掩盖不可能。

这个公式是全文所有掩盖分析的共同基础。

### 2.4 跨平台差异汇总

| | NVIDIA H100 | Ascend 910C |
|---|---|---|
| 计算单元 | SM + Tensor Core (989 TFLOPS BF16) | AICore ×2 (Da Vinci, ~800 TFLOPS BF16) |
| 通信库 | NCCL | HCCL |
| 机内互联 | NVLink 4.0 (900 GB/s bidir, 18 links × 50 GB/s) | HCCS (392 GB/s bidir, 14 links × 28 GB/s) |
| 跨机互联 | InfiniBand NDR (50 GB/s per HCA) | **UB 总线** (Spine-Leaf Clos, 实测 164 GB/s 读/135 GB/s 写, <1μs 延迟) |
| 流抽象 | CUDA stream | ACL stream |
| All-reduce 实现 | 机内 Ring, 跨机 Tree | 机内 Ring (≤8卡), 跨机 Tree (≥16卡) |
| All-to-all 优化 | NVSwitch SHARP | HCCL 细粒度分层流水线 + UB 无阻塞 Clos |
| 实际 all-reduce (8卡) | ~360 GB/s (NVLink) | ~48 GB/s per Die, ~96 GB/s 双 Die (HCCS) |
| 跨节点 p2p (实测) | ~12.5 GB/s (IB) | **~164 GB/s 读 / 135 GB/s 写** (UB) |
| 跨节点 all-to-all (EP) | ~20 GB/s (IB, ~40% util) | **~103–131 GB/s** (UB, CloudMatrix384 EP320 实测) |
| 训练框架 | Megatron-LM / FSDP2 | **MindSpeed-LLM** (继承 Megatron TP/PP + DualPipe + moe_alltoall_overlap_comm) |

**910C 超节点边界**：CloudMatrix384 容纳 ≤384 NPU 通过 UB 全互联。超出此规模需跨超节点走 RoCE（400 Gbps ≈ 50 GB/s per NPU），带宽从 164 GB/s 降到 ~50 GB/s——约 3× 降级，但仍优于 H100 的 IB（50 GB/s vs 12.5 GB/s p2p，且 all-to-all 在 RoCE 上效率低于 UB 的 103-131 GB/s）。

**带宽数字可信度说明**：UB 的 392 GB/s 是链路层理论聚合值（7×224 Gbps ×2 Die），非应用层实测。本文后续 Ascend 侧计算统一使用 CloudMatrix384 论文（arXiv:2506.12708）的实测值：p2p 164 GB/s（读）、all-to-all 103-131 GB/s。HCCL all-to-all 的逐消息大小基准测试未公开，103-131 GB/s 是端到端 MoE 分发带宽。

### 2.5 读者自检：掩盖分析的通用方法

给定任何一个通信操作，判断它能否被掩盖：

1. **找到依赖链**：这个通信的结果被哪个后续计算消费？在源码中找到消费通信结果的第一条指令
2. **找到掩盖窗口**：通信操作和它的消费计算之间，是否有**不依赖通信结果**的计算？
3. **比较时长**：掩盖窗口中的计算总时间是否 ≥ 通信操作的时间？
4. **检查流隔离**：通信和掩盖窗口中的计算是否在不同的 stream 上？如果在同一 stream，则串行、无法掩盖

这个四步法贯穿全文。

---

## CH 3 | 时间线坐标系

在进入各并行维度之前，建立阅读和绘制通信时间线的能力。

### 3.1 计算时间估算

一个 Transformer decoder layer 的 forward 计算时间由两部分组成：

$$T_{\text{layer\_fwd}} = T_{\text{linear}} + T_{\text{attention}} + T_{\text{FFN}}$$

**线性投影（matmul）**是计算的主要部分。一个 matmul $[M, K] \times [K, N]$ 的计算时间为：

$$T_{\text{matmul}} = \frac{2 \cdot M \cdot K \cdot N}{\text{GPU\_TFLOPS} \cdot \text{matmul\_efficiency}}$$

其中 `matmul_efficiency` 通常为 0.7-0.85（取决于矩阵形状和框架优化）。大矩阵（$M=B \cdot T$ 足够大）效率更高。

**Attention**（FlashAttention）：$T_{\text{attn}} \approx \frac{4 \cdot B \cdot T^2 \cdot H_q \cdot D}{\text{TFLOPS} \cdot \eta}$

**FFN (SwiGLU)**：$T_{\text{FFN}} \approx \frac{6 \cdot B \cdot T \cdot d \cdot d_{ff}}{\text{TFLOPS} \cdot \eta}$

以 M3 为例（$d=6144$, $d_{ff}=2048$, $B_{\text{micro}}=2$, $T=8192$，H100 BF16 989 TFLOPS, $\eta=0.75$）：

$$T_{\text{FFN}} \approx \frac{6 \cdot 2 \cdot 8192 \cdot 6144 \cdot 2048}{989 \cdot 10^{12} \cdot 0.75} \approx \frac{1.24 \cdot 10^{12}}{742 \cdot 10^{12}} \approx 1.7 \text{ ms}$$

一个 15 层 PP stage 的 forward 约为 $15 \cdot (1.7 + 0.3 + 0.2) \approx 33$ ms（含 attention 和 overhead）。Backward 约为 forward 的 2× ≈ 66 ms。一个 microbatch 的完整 forward+backward 约 100 ms。

### 3.2 通信时间估算

通信时间 = 数据量 / 有效带宽。有效带宽取决于原语类型和物理链路：

| 原语 | 数据量（per GPU） | H100 NVLink 有效带宽 | IB NDR 有效带宽 |
|---|---|---|---|
| all-reduce | $2 \cdot \frac{P-1}{P} \cdot D$ | ~360 GB/s (Ring, 8 GPU) | ~45 GB/s (Tree, 跨节点) |
| all-gather | $\frac{P-1}{P} \cdot D$ | ~360 GB/s | ~40 GB/s |
| reduce-scatter | $\frac{P-1}{P} \cdot D$ | ~360 GB/s | ~40 GB/s |
| all-to-all | $2 \cdot \frac{P-1}{P} \cdot D$ | ~150 GB/s (显著低于 all-reduce) | ~15-25 GB/s |
| p2p | $D$ | ~45 GB/s (单链路单向) | ~12.5 GB/s (单 HCA) |

**all-to-all 带宽为什么低？** all-to-all 中每个 GPU 向 $P-1$ 个目标发送不同数据，产生大量小消息。NCCL 的 all-to-all 实现不如 all-reduce 成熟——无法利用 SHARP in-network reduction（SHARP 只能做 reduction，不能做 routing），跨节点带宽利用率通常只有 20-40%。

### 3.3 时间线画法（极简示例）

以 2 层 MLP + TP=2 的 forward 为例：

```
时间 →
Stream 0:  [L0 matmul] [L0 all-reduce] [L0 act] [L1 matmul] [L1 all-reduce] [L1 act]
Stream 1:               [=== L0 all-reduce ===]          [=== L1 all-reduce ===]
```

这里 L0 all-reduce 在 Stream 1 上，与 Stream 0 上的 L0 act（激活函数）**存在数据依赖**——L0 act 需要 all-reduce 的结果作为输入。所以即使跨 stream，也必须串行。这就是第 2.2 节第 3 法则的体现。

真正可以重叠的场景：

```
时间 →
Stream 0:  [L0 matmul] [L1 matmul] [L2 matmul]
Stream 1:  [==== L0 all-reduce ====]
```

关键洞察：**跨层重叠在理论上是可能的**——层 i 的 all-reduce 聚合的是层 i 的输出，而层 i+2 的 matmul 输入来自层 i+1（不依赖层 i 的 all-reduce）。但层 i+1 的 matmul **必须等**层 i 的 all-reduce 完成（因为需要层 i 的完整输出作为输入）。

在实践中，Megatron 默认不利用跨层重叠——TP 通信和计算在同一 stream 上，串行执行。CH 4 会详细展开这个约束。

---

## CH 4 | TP 通信

### 4.1 Megatron TP 的通信位置（源码验证）

**ColumnParallelLinear**（`megatron/core/tensor_parallel/layers.py:L986`）：

```python
# forward():
output = self._forward_impl(input, ...)  # L1054: 本地 matmul
if self.gather_output:
    output = gather_from_tensor_model_parallel_region(output)  # L1087: all-gather
```

**ColumnParallelLinear 的 forward 不做 all-reduce。** all-reduce 发生在**反向传播**中——这是由 `copy_to_tensor_model_parallel_region` 在 forward 中是 identity、backward 中是 all-reduce 的机制保证的。

**RowParallelLinear**（`megatron/core/tensor_parallel/layers.py:L1299`）：

```python
# forward():
output_parallel = self._forward_impl(input_parallel, ...)  # L1328: 本地 matmul
output_ = reduce_from_tensor_model_parallel_region(output_parallel)  # L1348: all-reduce
```

**RowParallelLinear 的 forward 在本地 matmul 之后立即做 all-reduce。** 没有中间计算可以掩盖它。

### 4.2 TP 的通信量

每个 TP 维度操作的数据量：

$$D_{\text{TP}} = B \cdot T \cdot d \cdot \text{bytes\_per\_elem}$$

以 M3 配置（$B_{\text{micro}}=2$, $T=8192$, $d=6144$, BF16）：$D_{\text{TP}} = 2 \cdot 8192 \cdot 6144 \cdot 2 \approx 201$ MB。

每次 all-reduce 传输 $2 \cdot \frac{tp-1}{tp} \cdot 201 \approx 302$ MB（tp=4）。在 H100 NVLink ~360 GB/s 下，耗时约 **0.84 ms**。

### 4.3 TP 掩盖窗口分析

**RowParallelLinear 的 all-reduce**：发生在本地 matmul 之后，紧接着就是依赖 all-reduce 结果的后续计算。**无掩盖窗口。**

**ColumnParallelLinear 的 all-reduce（在 backward 中）**：all-reduce 的是输入梯度，后续是对上一层参数的梯度计算。如果上一层是大 matmul（如 FFN），可以形成掩盖窗口。但这不是框架保证的——取决于网络结构。

**实际情况**：TP all-reduce 在 Megatron 的默认实现中**基本无法被计算掩盖**。这不是设计缺陷——而是 **TP all-reduce 与后续计算有数据依赖，掩盖的前提条件不成立**。

但 TP 的通信量相对计算量是小的（$O(BTd)$ vs $O(BTd^2)$ 的 matmul），所以即使不能掩盖，通信时间占比也不高。以 M3 为例：

$$T_{\text{matmul}} \approx \frac{2 \cdot B \cdot T \cdot d \cdot (d/tp)}{\text{TFLOPS} \cdot \eta} \approx \frac{2 \cdot 2 \cdot 8192 \cdot 6144 \cdot 1536}{742 \cdot 10^{12}} \approx 0.4 \text{ ms}$$

matmul 0.4 ms vs all-reduce 0.84 ms——通信是计算的 2×。但因为 matmul 和 all-reduce **必须串行**（all-reduce 结果被下一个操作消费），最终层时间为 $0.4 + 0.84 = 1.24$ ms，其中通信占 68%。

**这就是为什么 TP 必须放机内（NVLink/HCCS）**——如果跨节点走 IB，通信时间会从 0.84 ms 涨到 6.7 ms（IB 50 GB/s），通信占比从 68% 涨到 94%，训练几乎停滞。

### 4.4 TP 的背靠背效应

一个 Transformer 层内有两次 RowParallelLinear 的 all-reduce（attention output 投影和 FFN output 投影）。两者之间隔了残差连接、LayerNorm 和整个 FFN 的计算——其中残差和 LayerNorm 极短（element-wise），但 FFN matmul 足够长（~1.7 ms），足以让 attention output 的 all-reduce 在新一次的 all-reduce 启动前完成。**同一层内，两次 all-reduce 自然间隔，通信链路利用率尚可。**

跨层则不同：层 i 的 FFN output all-reduce 和层 i+1 的 attention output all-reduce 之间只有残差 + LayerNorm（极短）——间隙不足以掩盖任何一方的通信。但由于 all-reduce 本身是成批触发的（不是每层独立管理），在实践中这一间隙不造成额外 bubble。

核心事实不变：**Megatron 的 TP 通信通常在主流上串行执行，跨层重叠不被利用。TP 掩盖主要靠"大计算块之间的自然间隙"而非显式的 prefetch 机制。**

### 4.5 Ascend 侧（MindSpeed-LLM, 910C）

MindSpeed-LLM 的 TP 实现**直接继承 Megatron-core** 的 `ColumnParallelLinear` 和 `RowParallelLinear`（`mindspeed_llm/core/tensor_parallel/layers.py`），通信模式与 CH 4.1 完全相同。在此基础上增加了 2D TP（`tp_2d/`）和 TP-extended-EP（TP 维度扩展覆盖 EP 组），用于 MoE 场景下联合优化 TP 和 EP 的通信。

HCCL all-reduce 在 910C 双 Die 架构上需考虑 NUMA 效应——Die 间带宽（540 GB/s）远高于 HCCS 单链路（28 GB/s ×14），HCCL Ring 算法会自动利用 Die 间带宽加速。跨节点时（虽然不推荐 TP 跨节点），UB 总线实测 164 GB/s 读带宽使 TP 跨节点成为**技术上可行**的选择——这与 H100 的 IB 50 GB/s 受限场景有本质区别。

---

## CH 5 | PP 通信

### 5.1 PP 的通信特点

PP 的通信量极小——每 microbatch 只需传递 $B_{\text{micro}} \cdot T \cdot d \cdot 2$ bytes（BF16 的隐藏状态）。

以 M3（$B_{\text{micro}}=2$, $T=8192$, $d=6144$）：不到 201 MB。在 IB 50 GB/s 下只需 4 ms，在 NVLink 450 GB/s 下只需 0.45 ms。

对比一整层 forward 的计算时间约 33 ms（15 层 per stage），**PP 的 p2p 通信通常能被掩盖——PP 的真正代价不是通信，是 bubble。** 极端小 batch（$B_{\text{micro}}=1$, $T=2048$）时一层仅 ~8 ms，掩盖窗口收窄，但 p2p 量也对应缩小，通常仍可掩盖。

### 5.2 Bubble：等待的本质

1F1B 调度下，pipeline 的 warmup 和 cooldown 阶段有空闲。稳态阶段每个 device 交替执行 forward 和 backward，没有空闲——但整体迭代时间中包含了 warmup/cooldown 的 bubble。

$$t_{\text{bubble}} = \frac{pp - 1}{\mu} \cdot (t_f + t_b)$$

其中 $\mu$ 是 microbatch 数。$\mu$ 越大，bubble 相对越小——但 $\mu$ 受限于 $B_{\text{global}} = dp \cdot B_{\text{micro}} \cdot \mu$。增大 $\mu$ 意味着减小 $B_{\text{micro}}$ 或增大 $B_{\text{global}}$（后者不一定被数据量支持）。

### 5.3 VPP：切碎来填缝

VPP（interleaved 1F1B）将每个 device 的 $L/pp$ 层再切为 $v$ 个 virtual stage。一个 device 上有 $v$ 个"虚拟层组"，调度器可以在它们之间交替 forward 和 backward，从而减少 warmup/cooldown 的空白：

$$t_{\text{bubble}}^{\text{VPP}} = \frac{pp - 1}{v \cdot \mu} \cdot (t_f + t_b)$$

bubble 与 $v$ 成反比。代价是实现复杂度——virtual stage 之间的切换需要额外的状态管理。VPP 的 p2p 通信量不变（每次传递的还是同样大小的隐藏状态），只是传递次数增多了——但传递本身不是瓶颈。

**DualPipe：让通信链路双向饱和**

VPP 只能减少 bubble，不能解决 EP 通信暴露。DualPipe（DeepSeek V3 提出）的核心思想不同：**让不同 microbatch 的 forward 和 backward 在时间线上交错，使 EP 通信链路持续有数据在传输**。

传统 1F1B/VPP：同一时刻只有一种通信方向（全 forward 或全 backward），EP 链路利用率 ~30%。
DualPipe：forward 的 dispatch all-to-all 和 backward 的 combine all-to-all 在时间线上重叠——方向相反、不冲突——链路利用率可提升到 ~60%。

从掩盖公式 $T_{\text{compute\_after\_comm}} \geq T_{\text{comm}}$ 的角度：DualPipe **不创造掩盖窗口**（EP 仍然无法被计算掩盖），而是**压缩等效 $T_{\text{comm}}$**——通过让链路双向工作，单位时间内完成更多的通信量。代价：调度复杂度极高，且前向和后向同时存在使峰值激活显存增加。

DualPipe 不在本文展开完整时间线分析——它依赖具体的 microbatch 数和层结构——但核心原理可概括为：**"掩盖不了通信，就让通信链路别闲着"**。

### 5.4 Ascend 侧（MindSpeed-LLM, 910C）——原理驱动的 PP 优化

MindSpeed-LLM 支持 1F1B、VPP、RiPipe、DualPipe 四种 PP 调度（`mindspeed_llm/core/pipeline_parallel/`）。

**DualPipe —— 对应 CH 2.3 掩盖条件（两个互补机制）**

DualPipe（`dualpipe/adaptor.py`）的效果可以从两个层面理解：

**计算层面（对应掩盖公式）**：不同 microbatch 的前向和反向在时间线上双向交错。$\mu$B$_i$ 的反向通信期间，$\mu$B$_j$ 的前向计算在并行执行——这些计算块来自不同 microbatch，与当前通信没有数据依赖。从 $T_{\text{compute\_after\_comm}} \geq T_{\text{comm}}$ 的角度：DualPipe 利用跨-μB 的独立性**创造了掩盖窗口**——这是传统 1F1B/VPP 无法做到的（它们严格按先全 forward 再全 backward 的顺序）。

**链路层面（CH 5.3 的视角）**：forward 和 backward 的通信方向相反（dispatch 发送 vs combine 接收），DualPipe 让它们在同一时刻占据同一物理链路的不同方向，使链路双向饱和。等效于在相同时间内完成更多通信量——不是 $T_{\text{comm}}$ 变短了，而是单位时间内有效通信量翻倍。

两个机制叠加：计算掩盖减少了通信暴露的**次数**（某些 all-to-all 落入其他 μB 的计算窗口），链路饱和减少了通信暴露的**绝对时长**（两次 all-to-all 在时间上重叠）。代价：调度复杂度极高，前向和反向同时存在使峰值激活显存增加。

**RiPipe —— 用重计算换取掩盖窗口**

RiPipe（recompute-in-advance）将 ckpt 重算提前到通信阶段执行，把原本的空闲时间（等通信）变为有效的重算时间。原理：通信期间 GPU SM 空闲（通信走 Copy Engine/TS）→ 利用空闲 SM 提前做 ckpt 重算 → 减少通信后的纯计算时间。这是 CH 2.1 硬件分离原理（SM 和 Copy Engine 独立）的充分利用。

UB 总线的低延迟使 PP p2p 几乎免费，但 DualPipe 和 RiPipe 的掩盖逻辑跨平台通用——它们解决的是"通信和计算的时间线排列"问题，与物理带宽无关。

---

## CH 6 | DP 与 ZeRO 通信

### 6.1 朴素 DP 的梯度 all-reduce

DP 的通信只有一个操作：backward 完成后，对所有 GPU 的梯度做 all-reduce。

通信量 = $2 \cdot \frac{dp-1}{dp} \cdot N_{\text{per\_gpu}} \cdot 2$ bytes（per-GPU 参数量 × BF16 × ring all-reduce 系数 2）。以 M3（$N_{\text{per\_gpu}}=26.75\text{B}$, dp=4）：$2 \cdot \frac{3}{4} \cdot 26.75 \cdot 10^9 \cdot 2 \approx 80$ GB。

在 IB 50 GB/s 下约需 1.6 秒——这不能接受。

### 6.2 分桶：不等全部算完就传

梯度不是一次性 all-reduce 的。DeepSpeed 将梯度分成 buckets（默认 `reduce_bucket_size=5e8` elements ≈ 1 GB per bucket for BF16）。每桶梯度算完即触发 reduce-scatter/all-reduce。

```
时间线:
Stream 0: [L0 bwd] [L1 bwd] [L2 bwd] ... [L59 bwd]
Stream 1:           [bucket1 reduce-scatter] [bucket2 reduce-scatter] ...
                         ↑ L0-L1 的梯度已算完, 启动通信
```

最后一层的梯度通信与倒数几层的 backward 计算重叠。**在 backward 执行了 70-80% 之后，梯度通信已全部完成**——掩盖率接近 100%。

前提条件：`contiguous_gradients=True`（DeepSpeed 默认）。如果不连续，需要在通信前做一次梯度拷贝——拷贝本身要时间，且与 backward 在同一 stream 上，破坏掩盖。

### 6.3 ZeRO-1/2：reduce-scatter 替代 all-reduce

ZeRO-1 和 ZeRO-2 用 reduce-scatter（而非 all-reduce）归约梯度。数据量相同，但 reduce-scatter 的结果是分布式的（每个 GPU 只有 $1/dp$ 的梯度），避免了后续的 all-gather 参数——因为 ZeRO-1 的优化器只更新本地的分片。

分桶机制同样适用。掩盖窗口同样在 backward 的尾段。

### 6.4 ZeRO-3：all-gather 前置于 forward

ZeRO-3 的通信模式不同于 ZeRO-1/2——它在 **forward 阶段就需要 all-gather 参数**（因为只有分片、没有全量）。这改变了掩盖分析：

```
ZeRO-3 forward (每层):
Stream 0:                                          [L0 fwd]
Stream 1: [AG L0 params] [AG L1 params]
            ↑ 提前触发 (prefetch)
```

如果 L1 的 all-gather 与 L0 的 forward 计算重叠——需要 L0 的计算足够长，覆盖 L1 的 all-gather 时间。这依赖于框架的 prefetch 机制（见 CH 9 FSDP2）。

### 6.5 Ascend 侧（MindSpeed-LLM, 910C）

HCCL 的 reduce-scatter 和 all-gather 在 HCCS 上的带宽与 all-reduce 水平相当（~48 GB/s per Die, ~96 GB/s 双 Die）。跨节点梯度通信走 UB 总线（实测 164 GB/s 读），远优于 H100 的 IB 50 GB/s——在 CloudMatrix384 超节点中，DP/ZeRO 的梯度通信几乎可以按机内带宽估算。HCCL Tree 算法在 ≥16 卡跨节点时比 Ring 快 2.6×。

---

## CH 7 | EP 通信

这是 MoE 训练中最难的通信问题。

### 7.1 EP 通信的发生位置（源码验证）

Megatron MoE 层的 forward 执行顺序（`megatron/core/transformer/moe/moe_layer.py`）：

```
1. gate(hidden_states)        → probs, routing_map    (极小的 matmul: d × N_E)
2. permute(hidden_states)      → 本地重排 token
3. all-to-all(dispatch)        → 发送 token 到对应 expert 的 EP rank
4. expert_FFN(dispatched)      → 每个 expert 算 FFN    (这是主要的计算)
5. all-to-all(combine)         → 回收 token 到原 rank
6. unpermute + shared_expert   → 还原 + 叠加共享 expert
```

### 7.2 dispatch all-to-all 为什么无法掩盖

看时间线：

```
gate:  [=== gate matmul ===]  ← d × N_E = 6144 × 128, ~0.035 ms
       [== permute ==]         ← 本地内存操作, ~0.01 ms
       [============ all-to-all (dispatch) ============]  ← ~17.6 ms (跨节点, 20 GB/s)
                                        [== expert FFN ==] ← ~1.7 ms
```

gate 的计算量是 $2 \cdot B \cdot T \cdot d \cdot N_E = 2 \cdot 2 \cdot 8192 \cdot 6144 \cdot 128 \approx 2.6 \cdot 10^{10}$ FLOPs。在 H100 989 TFLOPS 下仅需 **0.035 ms**。加上 permute 的 ~0.01 ms，dispatch 之前的总计算不到 0.05 ms。

**dispatch all-to-all 的掩盖窗口 = 0.05 ms。而通信需要 ~17.6 ms（跨节点）。掩盖比 > 350×——掩盖完全不可能。**

这不是框架设计缺陷，而是**结构性的**：MoE 的 gate 必须算完才能知道 token 去哪，所以 gate 不可能放在 all-to-all 之后；而 all-to-all 必须完成才能开始 expert FFN。gate 的计算量天生小（只是一个 router），无法形成掩盖窗口。

### 7.3 combine all-to-all 同样无法掩盖

expert FFN 之后的 combine all-to-all 同样没有掩盖窗口——expert FFN 完成后，下一操作就是 combine all-to-all（数据依赖）。combine 之后才能 unpermute 和叠加 shared expert，没有独立计算可以插入中间。

### 7.4 EP 通信量

每次 all-to-all 每 GPU 传输约 $2 \cdot \frac{ep-1}{ep} \cdot B \cdot T \cdot d \cdot 2$ bytes（dispatch + combine 各一次）。以 M3（$B_{\text{micro}}=2$, $T=8192$, $d=6144$, $ep=8$, BF16）：

$$D_{\text{EP}} \approx 2 \cdot \frac{7}{8} \cdot 2 \cdot 8192 \cdot 6144 \cdot 2 \approx 352 \text{ MB} \text{ (per all-to-all)}$$

57 个 MoE 层 × 2 次 all-to-all（dispatch + combine）× 352 MB = **约 40 GB per iteration per GPU** 的 EP 通信总量。

在跨节点 20 GB/s 有效 all-to-all 带宽下（IB NDR 50 GB/s 峰值 × ~40% all-to-all 利用率），纯通信时间约 **2.0 秒**——**EP 通信是 M3 训练的 #1 瓶颈**。

### 7.5 EP 通信为什么通常跨节点

在 512 GPU 的配置下（tp=4, pp=4, ep=8, dp=4），tp×pp 模型并行组已占 16 GPU（≥ 2 个 8-GPU 节点）。EP 组包含 8 GPU，这些 GPU 来自不同的 tp×pp 组——由于 tp×pp 组跨节点（2 节点），EP 组也必然跨节点。即使在小规模下 EP 可以放机内，在百 B 级 MoE 的训练规模下 EP 跨节点几乎不可避免。这就是为什么 EP 通信的带宽只能按 IB/RoCE 估算，不能按 NVLink 估算。

### 7.6 怎么缓解 EP 通信？（详见 CH 10.4）

- DualPipe：让不同 microbatch 的 dispatch 和 combine 在时间线上交错，提高链路利用率
- 通信压缩：dispatch 前对 token hidden states 做 top-k 稀疏化（只传 top-k 维度的值）
- 分层 all-to-all：DeepSpeed-MoE 的做法——将 all-to-all 拆为节点内和节点间两级，减少跨节点数据量

### 7.7 Ascend 侧（MindSpeed-LLM, 910C）——原理驱动的 EP 优化

MindSpeed-LLM 的 EP 通信优化展现了**如何将 CH 2 的掩盖原理转化为框架特性**。以下逐条分析。

**`moe_alltoall_overlap_comm` —— 对应 CH 2.2 法则 2（跨 stream 并发）**

源码（`mindspeed/core/transformer/moe/moe_layer_overlap_all2all.py`）使用独立的 `COMM_STREAM` 发起 all-to-all，同时默认 stream 上的 expert FFN 计算开始执行。这是 CH 2.2 法则 2 的直接应用——不同 stream 上无数据依赖的操作可以并发。

但 CH 7.2 的分析指出 dispatch all-to-all **无法掩盖**——因为 gate 计算太短（0.05 ms）、all-to-all 太长（17.6 ms），且存在数据依赖（expert FFN 必须等 all-to-all 完成才能开始）。`moe_alltoall_overlap_comm` 如何突破这个限制？

答案：它重叠的不是 dispatch all-to-all 与 expert FFN（那不可能），而是 **dispatch all-to-all 与共享专家（shared expert）的计算**，以及 **反向传播中的梯度 all-to-all 与 expert FFN 的梯度计算**。共享专家不依赖 dispatch 结果——这是**唯一的掩盖缝隙**，框架精准地利用了它。

**TP-extended-EP —— 对应 CH 2.3 掩盖条件（减少 T_comm）**

TP-extended-EP（`mindspeed_llm/features_manager/moe/tp_extend_ep.py`）将 TP 和 EP 通信域合并为单一 `alltoallv(tp*ep_group)`，替代原来分开的 `alltoall(tp_group) → alltoallv(ep_group)`。从掩盖公式 $T_{\text{compute\_after\_comm}} \geq T_{\text{comm}}$ 的角度：**减少通信步骤降低了总 $T_{\text{comm}}$，使掩盖条件更容易满足。** 代价是通信域扩大（tp×ep 个 rank 参与），单次 all-to-all 的数据量略增。

**UB 总线的无阻塞 All-to-All —— 对应 CH 2.3 掩盖条件（减少 T_comm 的物理时间）**

UB 的 Spine-Leaf Clos 拓扑使 all-to-all 实测带宽达到 103-131 GB/s——是 H100 IB 方案（~20 GB/s）的 5-6.5×。从掩盖公式的角度：**物理带宽直接压缩 $T_{\text{comm}}$，即使掩盖窗口为 0，通信的绝对时间也大幅缩短。**

**跨超节点的回退：EP 重归瓶颈**

超出 CloudMatrix384 的 384 NPU 后走 RoCE（400 Gbps per NPU）。RoCE all-to-all 带宽回退到 ~25 GB/s（~50% 利用率），$T_{\text{comm}}$ 增大 4-5×——EP 重新成为瓶颈。这就是为什么 MoE 训练倾向于将模型规模控制在单个超节点容量内。

---

## CH 8 | CP 通信

### 8.1 Ring Attention 的通信模式

CP 用 Ring Attention 在 $cp$ 张卡之间传递 KV chunk。每层 forward 需要一轮完整的 ring pass：

```
每步 ring step:
  recv(K_chunk_prev) → compute_attn(local_Q, received_KV) → send(K_chunk_local)
```

$cp-1$ 步完成全 ring。每步传 $2 \cdot H_{kv} \cdot D \cdot T/cp \cdot 2$ bytes（K+V, BF16）。

### 8.2 CP 掩盖窗口极窄

每一步 ring step 的计算量只是 local attention（$B \cdot T/cp$ 个 query 对 $T/cp$ 个 KV 的 attention），计算时间极短（通常在微秒到亚毫秒量级）。而 p2p 通信即使走 NVLink 也有 ~10 μs 的固定延迟。**CP 通信几乎完全暴露**——这是 Ring Attention 的结构性限制。

### 8.3 CP 拓扑约束

CP 的每一跳如果跨节点，延迟会直接累加。$cp=8$ 时，7 跳 × 10-20 μs 的 NVLink 延迟 ≈ 70-140 μs——尚可。但如果跨节点（IB ~1-2 μs 延迟 + 低带宽），每跳变成 ~0.1-0.5 ms——总计 0.7-3.5 ms per layer。**CP 组必须在节点内（NVLink/HCCS）。**

对 Ascend 910C：HCCS 延迟与 NVLink 可比，UB 总线延迟 <1μs——使 CP 跨节点也**技术上可行**。但 MindSpeed-LLM 支持双层 Ring（`cp_inner_ranks` / `cp_outer_ranks`），允许将 CP 组分拆为节点内 Ring（走 HCCS）和跨节点 Ring（走 UB），兼顾规模和延迟。MindSpeed-LLM 还支持 Hybrid CP（Ring + Ulysses）和 KV-AllGather CP 共 3 种算法，可按序列长度和 GPU 拓扑选择最优方案。

---

## CH 9 | FSDP2 通信体系

### 9.1 FSDP2 的通信全貌

FSDP2（`torch/distributed/fsdp/_fully_shard/`）将模型参数按 FSDP unit 分组（通常每层一个 unit）。每个 unit 在 forward 前 all-gather 参数，用后立即释放。

```
FSDP2 forward (每层, 有 prefetch):
Stream 0:                              [L0 fwd] [L1 fwd] [L2 fwd]
Stream 1 (all-gather): [AG L0 params] [AG L1 params] [AG L2 params]
                            ↑ L0 AG 提前触发    ↑ L1 AG 与 L0 fwd 重叠
```

### 9.2 Prefetch：FSDP2 掩盖的核心

FSDP2 的 prefetch 机制（`_fsdp_param_group.py:L775`）在 forward 阶段预取下一层的参数：

```python
# FSDPState._pre_forward (line ~680):
for target_param_group in fsdp_state._fsdp_param_groups:
    FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
```

all-gather 在独立的 `all_gather_stream` 上执行，与默认 stream 上的计算重叠。Backward 阶段同理，但顺序变为 reversed（匹配 backward 的逆序执行）。

**Prefetch 成功的条件**：当前层的 forward 计算时间 ≥ 下一层参数的 all-gather 通信时间。

以 M3 为例（一层 ~2 ms forward vs all-gather ~0.5 ms: 201 MB / 360 GB/s ≈ 0.56 ms）——条件成立，prefetch 可行。

### 9.3 FSDP2 vs Megatron：通信模式对比

| | Megatron TP+PP | FSDP2 |
|---|---|---|
| 通信频率 | TP all-reduce 每层多次 | all-gather 每层 1 次 |
| 通信粒度 | 脉冲式（all-reduce 量大集中） | 绵延式（all-gather 量小分散） |
| 掩盖机制 | 无（数据依赖） | Prefetch（无依赖，可提前触发） |
| PP bubble | 有（$pp \geq 2$ 时） | 无（无 PP） |
| 激活显存 | $1/pp$ | 全量 |
| 最适合 | VL 大模型（激活是瓶颈，需要 PP 省激活） | Dense 模型（激活压力小，通信简单） |

核心差异：Megatron 牺牲了通信掩盖（TP all-reduce 串行暴露），但通过 PP 大幅省激活（$1/pp$）；FSDP2 通信掩盖做得更好（prefetch），但没有 PP 的激活节省。选择取决于激活是不是瓶颈——回到[系列第五篇](../part-5/)的结论。

### 9.4 Ascend 侧（MindSpeed-LLM, 910C）

MindSpeed-LLM 提供独立的 FSDP2 训练后端，与 PyTorch FSDP2 的 prefetch + all-gather 机制等价。HCCL 的 all-gather 在 HCCS 上带宽 >90 GB/s（双 Die），跨节点走 UB 总线实测 164 GB/s——prefetch 的条件极易满足。910C 的 HBM（128 GB）也使全量激活存储压力小于 H100（80 GB），FSDP2 的显存约束在 Ascend 侧相对缓解。

---

## CH 10 | 完整推演：M3 训练 step time

基于[系列第五篇](../part-5/)的显存可行配置（tp=4, pp=4, ep=8, dp=4, vpp=2, 512×H100），估算单步迭代时间。

### 10.1 计算时间（无通信、无 bubble 的理想情况）

一个 microbatch（$B_{\text{micro}}=2$, $T=8192$, 15 layers per stage, full ckpt): forward ~33 ms, backward ~66 ms, 合计 ~99 ms。包含 ckpt 重算的额外 forward（~30% overhead）: ~132 ms per microbatch。

$\mu = B_{\text{global}} / (dp \cdot B_{\text{micro}}) = 32 / (4 \cdot 2) = 4$（考虑 grad_accum）。

### 10.2 通信时间

| 维度 | per-μB 数据量 | 有效带宽 | per-μB 耗时 | 注 |
|---|---|---|---|---|---|
| EP (14 MoE 层, 2 a2a/层) | 28 × 352 MB | 20 GB/s (IB) | **~493 ms** | PP=4 将 57 层分到 4 stage, 每 stage ~14 MoE 层 |
| TP (15 层, 2 ar/层) | 30 × 302 MB | 360 GB/s (NVLink) | ~25 ms | 机内 NVLink, 可掩盖有限 |
| PP (p2p) | 8 × 201 MB | 50 GB/s (IB) | ~32 ms | 量极小, 通常可掩盖 |
| DP (grad reduce-scatter) | ~80 GB | 50 GB/s (IB) | ~160 ms 暴露 | 分桶掩盖 ~90% |

**EP 通信 ~493 ms/μB，4 个 μB 合计 ~1,972 ms——是绝对主导项。** TP 通信 ~100 ms/4 μB，PP 通信 ~128 ms/4 μB。

### 10.3 Step time 和 MFU

加 VPP bubble（$v=2$, $pp=4$, $\mu=4$）：$t_{\text{bubble}} \approx \frac{3}{2 \cdot 4} \cdot 132 \approx 50$ ms。

总迭代时间 ≈ $4 \times (132 + 493 + 25 + 32) + 50 + 160 \approx 2728 + 50 + 160 \approx \mathbf{2,938 \text{ ms} \approx 2.9 \text{ 秒}}$。

有效吞吐: tokens/step = $B_{\text{global}} \cdot T = 32 \cdot 8192 = 262,144$。Throughput = $262,144 / 2.9 \approx 90,000$ tokens/s。

理论 peak throughput（纯计算、无通信）：$262,144 / (132 \cdot 4 / 1000) \approx 496,000$ tokens/s。

$$\eta = \frac{90,000}{496,000} \approx 18\%$$

**EP 通信吃掉约 67% 的计算能力。** 这就是 MoE 训练的通信瓶颈。

### 10.4 优化方向

- **ep=16**：EP 通信量减半（$352 \rightarrow 176$ MB），但 EP 组扩大 → 跨节点 all-to-all 的 $P$ 增大 → 有效带宽可能更低。需要实测验证
- **DualPipe**：利用不同 microbatch 的 dispatch/combine 时间线交错，将 EP 通信链路利用率从 ~30% 提升到 ~60%，等效通信时间减半
- **通信压缩**：dispatch 时只传 top-K 维度的 hidden state（而非完整 $d$ 维），通信量压缩 2-4×

### 10.5 Ascend 910C 上的等效估算（CloudMatrix384 超节点内）

将 H100 的有效带宽替换为 910C + UB 总线的**实测值**：

| 维度 | H100 per-μB | 910C per-μB | 注 |
|---|---|---|---|
| EP (14 MoE 层) | ~493 ms (20 GB/s) | **~95 ms (103 GB/s, UB 实测)** | UB 使 EP 降 5.2× |
| TP (15 层) | ~25 ms (360 GB/s) | ~94 ms (~96 GB/s, HCCS 双 Die) | HCCS 带宽 ~3.75× 低于 NVLink |
| PP p2p | ~32 ms (50 GB/s) | ~10 ms (164 GB/s, UB 实测读) | UB 使 p2p 降 3.2× |
| DP (ZeRO-2) | ~160 ms 暴露 | ~30 ms 暴露 | UB >100 GB/s, 几乎全掩盖 |

Ascend 整体步时间 ≈ $4 \times (132 + 95 + 94 + 10) + 50 + 30 \approx 1324 + 80 \approx \mathbf{1,404 \text{ ms} \approx 1.4 \text{ 秒}}$。

$$\eta_{\text{910C}} \approx \frac{528}{1404} \approx 38\%$$

**vs H100 的 ~18%——UB 总线使 910C 在 MoE 训练场景下具备约 2.1× 的 MFU 优势。** 

**跨超节点回退**：超出 CloudMatrix384 的 384 NPU 后走 RoCE（400 Gbps ≈ 50 GB/s per NPU）。此时 EP 带宽回退到 ~25 GB/s（RoCE all-to-all ~50% 利用率），per-μB EP 时间约 394 ms，总体步时间约 $4 \times (132+394+94+5)+80 \approx 2580$ ms, $\eta \approx 20\%$——回退到与 H100 IB 方案相当的水平。

---

## CH 11 | 实战演练：从零分析三个场景

CH 10 展示了 M3 的结论。本章展示**过程**——从零分析三个不同复杂度的场景。读完本章，你应该能独立分析任何新配置。

### 11.1 案例 1：Dense 70B + FSDP2 on 32×H100

**场景**：Llama-3 70B 级别 Dense 模型（$d=8192$, $L=80$, $H_q=64$, $H_{kv}=8$, $D=128$, $d_{ff}=28672$），FSDP2 (dp_shard=8, dp_replicate=4)，$B_{\text{micro}}=2$, $T=8192$, $B_{\text{global}}=8$。32×H100（4 节点 × 8 GPU）。无 PP、无 TP、无 EP。

**Step 1: 画数据分布图**

```
dp_shard=8 (节点内 FSDP), dp_replicate=4 (跨节点 DP)
32 = 8 × 4

每个节点 8 GPU: dp_shard 组 — 每 GPU 持有 1/8 参数分片
4 个节点: dp_replicate 组 — 互相复制

前向: all-gather 收集参数分片 → 还原全量 → 计算 → 丢弃
后向: all-gather 收集参数 → 计算梯度 → reduce-scatter 归约 → 1/8 分片
```

**Step 2: 标通信位置**

Dense 模型，无 MoE。每层仅 2 次通信：
- Forward: `all_gather(params_N)` — 层 N 启动前，层 N-1 计算期间触发（prefetch）
- Backward: `reduce_scatter(grads_N)` — 层 N 梯度算完后

80 层 × 2 = 160 次通信操作 per μB。

**Step 3: 算通信量**

每层参数 $\approx 70\text{B}/80 \approx 0.875\text{B}$。Per-layer all-gather 数据量（dp_shard=8）：
$$D_{\text{AG}} = \frac{dp-1}{dp} \cdot 0.875\text{B} \cdot 2 = \frac{7}{8} \cdot 1.75 \approx 1.53 \text{ GB}$$

Reduce-scatter 同理 ~1.53 GB。

**Step 4: 算通信时间**

节点内 NVLink（dp_shard=8, ~360 GB/s）：
$$T_{\text{AG}} = 1.53 / 360 \approx 4.2 \text{ ms} \qquad T_{\text{RS}} = 1.53 / 360 \approx 4.2 \text{ ms}$$

**Step 5: 找掩盖窗口（核心分析）**

FSDP2 的 prefetch（CH 9.2）：层 N 的 all-gather 在层 N-1 的 forward 期间触发。

一层 forward 计算时间（70B, $d_{ff}=28672$）：
$$T_{\text{FFN}} \approx \frac{6 \cdot 2 \cdot 8192 \cdot 8192 \cdot 28672}{989 \cdot 10^{12} \cdot 0.75} \approx 31 \text{ ms}$$

加上 attention ~4 ms、overhead ~2 ms → **~37 ms per layer forward**。

掩盖分析：计算 37 ms vs all-gather 4.2 ms。**37 ms ≫ 4.2 ms——prefetch 完全有效，all-gather 被充分掩盖。**

同理 backward reduce-scatter 也可被前一层 backward 计算掩盖（backward per-layer ≈ forward × 2 ≈ 74 ms ≫ 4.2 ms）。

**Step 6-7: 时间线 + bubble**

无 PP → 无 bubble。通信由于 prefetch 完全掩盖 → **暴露通信 ≈ 0**（仅首层 all-gather 无法 prefetch，1 次 ~4.2 ms，可忽略）。

DP 梯度 all-reduce (跨节点 IB): per-GPU 梯度 = 70B/32 × 2 ≈ 4.4 GB。All-reduce 数据 = $2 \cdot (3/4) \cdot 4.4 \approx 6.6$ GB。$T = 6.6/50 \approx 132$ ms，分桶掩盖 90% → 暴露 ~13 ms。

**Step 8: 算 η**

$B_{\text{global}}=8, B_{\text{micro}}=2$, dp_replicate=4 → $\mu = 1$。

Useful compute per μB（不含 ckpt 重算）: 80 层 × (37 ms fwd + 74 ms bwd) = 8,880 ms。
含 ckpt 重算的总 compute: 8,880 × 1.3 ≈ 11,544 ms。
总步时间: 11,544 + 4.2 (首层 AG) + 13 (DP 残留) ≈ 11,561 ms ≈ 11.6 秒。

$$\eta \approx \frac{8,880}{11,561} \approx 77\%$$

> **注**：77% 以 ckpt 重算为开销（MFU 标准定义）。若将 ckpt 重算视为必要计算，η ≈ 99.9%——说明**通信开销几乎为零**。本案例的核心结论：FSDP2 prefetch 在 Dense 大模型上可实现近乎完美的通信掩盖。

**Step 9: 诊断 + 优化**

FSDP2 的 prefetch 在 Dense 大模型上表现优异——因为每层计算足够重（37 ms），all-gather 的掩盖窗口非常充裕。主要残留开销来自跨节点 DP 梯度通信（13 ms），可忽略。

如果 $B_{\text{micro}}$ 缩小到 1（每层计算减半到 ~18 ms），掩盖窗口 18 ms 仍 > 4.2 ms——prefetch 依然有效。**FSDP2 对 batch size 的鲁棒性很强**，只要每层计算时间超过 all-gather 时间即可。

**Step 10: 换平台（32×910C HCCS，机内）**

HCCS all-gather ~96 GB/s, $T_{\text{AG}} = 1.53/96 \approx 16$ ms。掩盖窗口 37 ms > 16 ms——**prefetch 依然有效**。与 H100 的差异仅为 all-gather 绝对时间长 4×，但不影响掩盖率。

FSDP2 在 Ascend 上同样适用——只要 $\text{layer\_fwd\_time} > \text{AG\_time}$。对于 $d_{ff}=28672$ 的大模型，条件轻易满足。

---

### 11.2 案例 2：同一 Dense 70B + Megatron TP4+PP2 on 32×H100

**同样模型，换用 Megatron TP4+PP2+DP4 (ZeRO-2)**。对比 FSDP2 方案。

**Step 1: 数据分布**

TP=4: 每张卡权重 = $70\text{B}/(4 \times 2) = 8.75\text{B}$。PP=2: 每张卡 40 层。DP=4: ZeRO-2 切优化器+梯度。

**Step 2: 通信位置**

每层 2 次 TP all-reduce（RowParallel output ×2）。40 层 × 2 = 80 次 TP all-reduce per μB，全部暴露（CH 4.3：紧随 matmul，无掩盖窗口）。

PP=2: 1 个 stage 边界，每 μB forward 1 次 p2p send, backward 1 次 p2p recv。可掩盖（CH 5.1）。

**Step 3-4: 通信量 + 时间**

每层 TP all-reduce: $2 \cdot (3/4) \cdot 2 \cdot 8192 \cdot 8192 \cdot 2 \approx 403$ MB。NVLink 360 GB/s: **1.1 ms/layer, 完全暴露。**

PP p2p: $2 \cdot 8192 \cdot 8192 \cdot 2 \approx 268$ MB。IB 50 GB/s: 5.4 ms。**可掩盖。**

**Step 5-7: 掩盖 + 时间线**

每层 compute（TP=4 将 FLOPs 降为 1/tp）: FFN ~8 ms + attention ~5 ms ≈ **13 ms per layer**。40 层: forward ~520 ms, backward ~1,040 ms。Useful compute = 520 + 1,040 = 1,560 ms。含 ckpt 重算: 1,560 × 1.3 ≈ 2,028 ms。

TP 暴露: 80 × 1.1 ms = **88 ms per μB**。TP 与 compute 串行（同 stream，CH 4.4），暴露率 100%。

μ=1（$B_{\text{global}}=8, dp=4, B_{\text{micro}}=2$）。VPP=2 bubble: $(2-1)/(2 \cdot 1) \cdot (520+1040) = 780$ ms。

总步时间: 2,028 + 88 + 780 + 13（DP 残留）≈ **2,909 ms**。

**Step 8: η ≈ 1,560/2,909 ≈ 54%**

**Step 9: 对比 FSDP2**

| | FSDP2 (案例 1) | Megatron TP+PP |
|---|---|---|
| per-μB layers | 80 | **40**（PP=2 减半） |
| per-layer compute | 37 ms | **13 ms**（TP=4 减为 1/4） |
| 主要通信 | all-gather (完全掩盖) | TP all-reduce (暴露 1.1 ms/层) |
| 通信暴露占比 | ~0.15% | **3.0%**（88/2909） |
| Bubble | 0 | **780 ms（27%）** |
| η | **~77%** | **~54%** |
| **主要瓶颈** | **—（通信几乎免费）** | **PP bubble, 远超 TP 通信** |

**核心发现**：Megatron 的 TP 通信暴露（88 ms）反而很小——因为 TP=4 已经把每层的 compute 和通信量都除以 4。**真正的开销来自 PP bubble（780 ms, 27%）**，不是 TP 通信。

**选择法则**：
- Dense 模型 on H100：**FSDP2 明显优于 Megatron**（η 77% vs 54%，配置简单，无 bubble）
- 仅在**显存受限**（激活是瓶颈）时选 Megatron TP+PP——PP 省 50% 激活（PP=2），代价是 27% bubble + 较低的 η

**Ascend 侧**：TP all-reduce 在 HCCS 96 GB/s 下 ~4.1 ms/layer。40 层 × 2 × 4.1 = 328 ms 暴露。步时间: 2,028 + 328 + 780 + 13 = 3,149 ms。η ≈ 1,560/3,149 ≈ 50%。HCCS 带宽低使 TP 通信占比升至 10%，整体 η 进一步降低。

---

### 11.3 案例 3：M3 MoE + Megatron on 512 GPU — H100 vs 910C CloudMatrix

这是 CH 10 完整推演的方法论回顾。此处侧重**MoE 特有的掩盖分析步骤**和**跨平台对比**。具体通信量和计算数字沿用 CH 10.2-10.3 的推导。

**配置**: tp=4, pp=4, ep=8, dp=4, vpp=2, $B_{\text{micro}}=2$, $T=8192$, 512×H100。

**Step 1-2: 数据分布 + 通信全景**

6 个并行维度的通信在此交汇（详见 CH 1.2 全景图）。核心是 **EP all-to-all 是唯一无法掩盖的通信维度**（CH 7.2）：gate 仅 0.05 ms，dispatch all-to-all 17.6 ms——掩盖比 >350×。TP all-reduce 虽不能掩盖但通信量小（25 ms/μB）。DP 梯度通信被分桶完全掩盖。

**Step 3-4: 只算暴露的通信**

不是所有通信都要算——只算**无法掩盖的部分**。按 CH 10.2 的 per-μB 分解：

- EP（每 GPU 14 MoE 层, PP=4）: 28 次 all-to-all × 352 MB = 9,856 MB。$T_{\text{EP}} = 9{,}856 / 20 \approx 493$ ms（H100 IB）。**全部暴露。**
- TP（15 层）: 30 次 all-reduce × 302 MB = 9,060 MB。$T_{\text{TP}} = 9{,}060 / 360 \approx 25$ ms。**全部暴露**（CH 4.3）。
- Bubble（VPP=2）: ~50 ms。**全部暴露**（调度结构性开销）。
- DP: ~80 GB 梯度通信，分桶掩盖 90% → 暴露 ~160 ms。

**Step 5: 关键掩盖判断（只对 MoE）**

这里执行 CH 2.5 的四步法，只针对 EP：

1. **依赖链**：gate → all-to-all dispatch → expert FFN。expert FFN 消费 all-to-all 的结果。
2. **掩盖窗口**：gate (0.035 ms) + permute (0.01 ms) = **0.05 ms**。expert FFN 之前无独立计算。
3. **比较**：$T_{\text{comm}} = 17.6$ ms $\gg$ $T_{\text{window}} = 0.05$ ms。
4. **流隔离**：无关——即使跨 stream，数据依赖也阻止掩盖。

→ **EP dispatch 在所有平台、所有配置下都完全暴露。这是 MoE 训练的结构性约束。**

**Step 6-7: 时间线**

与 Case 1-2 不同，这里不需要画完整时间线——因为各维度的暴露时间是**可加的**（不同维度走不同物理链路，不互相掩盖）。总步时间 = compute + EP暴露 + TP暴露 + bubble + DP残留。

加 VPP bubble（$v=2$, $pp=4$, $\mu=4$）：$t_{\text{bubble}} \approx \frac{3}{2 \cdot 4} \cdot 132 \approx 50$ ms。

**Step 8: η（双平台）**

| | H100 IB (20 GB/s) | 910C UB (103 GB/s) | 差距 |
|---|---|---|---|
| per-μB EP 暴露 | ~493 ms | ~95 ms | 5.2× |
| per-μB TP 暴露 | ~25 ms | ~94 ms | 0.27× |
| Bubble | ~50 ms | ~50 ms | 1× |
| DP 残留 | ~160 ms | ~30 ms | 5.3× |
| **总步时间** | **~2.9 s** | **~1.4 s** | |
| **η** | **~18%** | **~38%** | **2.1×** |

**Step 9: 瓶颈诊断**

- **H100**：EP 占步时间的 $493 \times 4 / 2{,}938 \approx 67\%$——绝对主导。TP（3.4%）和 bubble（6.8%）相比之下微不足道。优化方向：把 EP=8 放机内 NVLink（如果拓扑允许）可降 $T_{\text{EP}}$ 到 ~66 ms（150 GB/s NVLink all-to-all），η 可升至 ~50%+。
- **910C**：EP 降为第二大开销（$95 \times 4 / 1{,}404 \approx 27\%$）。TP 反转为第一开销（$94 \times 4 / 1{,}404 \approx 27\%$）。Bubble 占 14%。优化方向：增大 TP（利用 HCCS 96 GB/s 多掩盖一些 EP）或减小 EP 规模。

**Step 10: 换平台分析 → 三步洞察**

1. **物理带宽 > 掩盖机制**：EP dispatch 双平台都无法掩盖，但 UB 的绝对带宽优势直接决定了 η 差 2.1×。在 MoE 场景下，**网络硬件比调度软件更重要。**
2. **Ascend 在 TP 上吃亏，但 EP 优势碾压**：HCCS 使 TP 暴露高 3.8×，但 EP 优势 5.2×——净效果 Ascend 胜出 2.1×。
3. **跨超节点回退**：超出 384 NPU 后 EP 走 RoCE（~25 GB/s），η 回退到 ~20%——与 H100 持平。

---

### 11.4 从三个案例提炼的配置选择法则

| | Dense + FSDP2 (H100) | Dense + Megatron (H100) | MoE + H100 | MoE + 910C UB |
|---|---|---|---|---|
| 主要通信 | all-gather (完全掩盖) | TP all-reduce (暴露, 但小) | EP all-to-all (暴露, 大) | EP all-to-all (暴露, 小) |
| 真正瓶颈 | —（通信几乎免费） | **PP bubble（27%）** | **EP 暴露（67%）** | TP 暴露 + bubble |
| η | ~77% | ~54% | ~18% | ~38% |
| 关键启示 | Prefetch 对 Dense 大模型极有效 | TP 不是瓶颈, bubble 才是 | EP 是 MoE 的 #1 瓶颈 | 物理带宽 > 掩盖机制 |

将这四列翻译为配置选择法则：

- **Dense 中型模型 + NVIDIA**：**FSDP2**（η 77%, 通信几乎完全掩盖, 无 bubble, 配置简单）
- **Dense VL 大模型（激活瓶颈）**：**Megatron TP+PP**（PP 省激活是硬需求, 牺牲 η — 54% vs 77%, 差距显著）
- **MoE 模型 + H100**：EP 瓶颈无法消除, 尽量把 EP 放机内 NVLink, 必要时加 DualPipe
- **MoE 模型 + 910C CloudMatrix**：UB 高带宽是核心优势, EP 规模控制在 384 NPU 内
- **Dense + Ascend**：FSDP2 仍适用（prefetch 条件满足）, TP+PP 的 bubble 是主要开销

---

## CH 12 | 跨框架、跨平台的分析能力

### 12.1 通用分析清单

- [ ] **画数据分布图**：每个并行维度，哪些数据在哪些 GPU 上？用一张草图标注权重的分片方式
- [ ] **标通信位置**：在源码中找到每个 `all_reduce` / `all_to_all` / `all_gather` 调用的精确位置（函数 + 行号），标注在计算图上的位置
- [ ] **算通信量**：$D = f(B, T, d, P, \text{bytes\_per\_elem})$，对每个通信操作
- [ ] **算通信时间**：$T_{\text{comm}} = D / \text{BW}_{\text{effective}}$，区分 NVLink/HCCS 和 IB/RoCE
- [ ] **找掩盖窗口**：对每个通信操作，找它和它的消费计算之间是否有独立计算，以及计算时长是否 ≥ 通信时长
- [ ] **画时间线**：将上述信息汇总为一张 microbatch 级时间线
- [ ] **加 bubble**：如果有 PP，加上 $(pp-1)/(v \cdot \mu) \cdot (t_f + t_b)$ 的 bubble
- [ ] **算 $\eta$**：$\eta = t_{\text{compute}} / t_{\text{total}}$
- [ ] **诊断瓶颈**：哪类通信占总通信时间的 50% 以上？对应并行维度是否可调整？
- [ ] **换平台重算**：替换有效带宽数字，其他不变

### 12.2 常见错误

| 错误 | 纠正 |
|---|---|
| 用理论带宽峰值算通信时间 | 用实测有效带宽（all-reduce ~77% NVLink peak, all-to-all ~20-40% IB peak） |
| 假设通信始终可被掩盖 | 先检查数据依赖——通信结果是否被下一步计算消费 |
| 忽略 all-to-all 与 all-reduce 的带宽差距 | all-to-all 有效带宽远低于 all-reduce（尤其在跨节点） |
| 认为 TP all-reduce 可被掩盖 | Megatron 默认实现中不可掩盖——RowParallelLinear 的 all-reduce 紧随 matmul |
| 用全模型参数量算 DP 梯度通信 | DP 通信量 = per-GPU 参数量 × $(dp-1)/dp$ × 2 |

---

> **系列导航**：[（五）训练显存估算](../part-5/) ← 当前（系列终篇）

> **符号约定**：与前两篇文档完全一致。$t_f$ = forward time, $t_b$ = backward time, $\mu$ = microbatch 数, $v$ = virtual pipeline stage 数, $\eta$ = 并行效率 or MFU（根据上下文）。

> **系列导航**：[（五）训练显存估算](../part-5/) ← 当前（系列终篇）
