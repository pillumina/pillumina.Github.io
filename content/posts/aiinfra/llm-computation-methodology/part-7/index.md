+++
date = '2026-06-22T09:06:00+08:00'
draft = false
title = 'LLM 系统分析方法论（七）：推理服务性能建模'
categories = ['aiinfra']
tags = ['inference', 'serving', 'vllm', 'performance', 'methodology', 'moe']
series = 'llm-computation-methodology'
series_order = 7
math = true
summary = '推理服务完整性能建模：从单 token 延迟到多请求并发，覆盖连续批处理、PagedAttention、Prefill-Decode 分离、推测解码、量化部署。含 Llama-70B 完整服务分析和 MoE 模型服务策略。跨 NVIDIA + Ascend 双平台。'
+++


> 从单 token 延迟出发，叠加连续批处理、KV cache 管理、分离式架构、推测解码，建立推理服务的完整性能模型。

> **系列导航**：[（一）预备知识](../part-1/) → [（二）FLOPs](../part-2/) → [（三）KV Cache](../part-3/) → [（四）M3+Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) ← 当前（系列终篇）

## 本文定位

本系列[第四篇 Roofline 分析](../part-4/)给出了**单个请求的单 token decode 延迟**。本文扩展到**多请求并发**场景——这是推理服务的实际运行模式。

两篇的衔接点：Roofline 告诉你"一个 token 在 GPU 上至少 42 ms（Llama 70B on H100）"。本文告诉你"50 个用户同时请求时，每个人感受到的 token 延迟是多少？系统总吞吐多少？延迟 SLA 约束下最大并发数是多少？"

核心问题：**一批请求同时到来，GPU 怎么调度才能让每个人都觉得"快"？**

本文覆盖 NVIDIA（vLLM on H100）和 Ascend（vllm-ascend on 910C）两个平台。vllm-ascend（`github.com/vllm-project/vllm-ascend`）是 vLLM 的 OOT 硬件插件，实现了与上游**完全相同的调度机制**（连续批处理、PagedAttention、chunked prefill、前缀缓存、推测解码、PD 分离）。系统级分析方法跨平台通用，差异仅在硬件常数。关键差异：

| | H100 | 910C |
|---|---|---|
| HBM 带宽 | 3.35 TB/s (HBM3) | **~3.2 TB/s** (HBM2e) |
| HBM 容量 | 80 GB | **64 GB**（每 die） |
| BF16 算力 | 989 TFLOPS | ~800 TFLOPS |
| **FP8 支持** | **原生**（E4M3, ~1979 TFLOPS） | **不支持** |
| INT8 支持 | 原生 | 原生 |
| 机内 TP 带宽 | ~360 GB/s (NVLink) | **~48 GB/s** (HCCS all-reduce) |
| 推理框架 | vLLM | vllm-ascend（API 兼容） |

> 910C 尚无公开的 Llama-70B 单卡服务基准测试。Ascend 侧 TPOT/TTFT 采用公式推导 + 带宽常数估算。下文在每个相关章节末尾标注 Ascend 侧差异。

---

## CH 1 | 为什么推理服务不同于训练

### 1.1 训练 vs 推理的根本差异

| | 训练 | 推理服务 |
|---|---|---|
| 请求模式 | 固定 batch，持续迭代 | **动态到达**，任意时刻来任意数量 |
| 序列长度 | 固定（8192 tokens） | **变长**（1 token 的 "Hi" 到 100K token 的文档） |
| 输出长度 | 不适用（只算 loss） | **变长**（1-4096 tokens），事先不知道 |
| 延迟要求 | 无（吞吐优先） | **硬 SLA**（TTFT < 200ms, TPOT < 50ms） |
| 显存占用 | 优化器 + 梯度 + 激活 | **KV cache 是最大的变量** |
| 批处理 | 静态 batch，同步 | **连续批处理**，请求可随时加入/离开 |

### 1.2 推理服务的核心指标

$$\text{端到端延迟} = \underbrace{\text{TTFT}}_{\text{首个 token 时间}} + \underbrace{\text{TPOT} \times (N_{\text{out}} - 1)}_{\text{后续 token 时间}}$$

- **TTFT（Time-To-First-Token）**：从请求到达到第一个 token 输出的时间。受 prompt 长度 $S$ 影响最大（prefill 需一次性处理整个 prompt）
- **TPOT（Time-Per-Output-Token）**：每个 decode token 的平均时间。受并发数 $B$ 和总 KV cache 大小影响
- **吞吐（Throughput）**：tokens/s，服务端视角的总产出
- **Goodput**：满足 SLA 约束下的有效吞吐（超出 SLA 的请求不算）

**典型 SLA 基准**：

| 指标 | 目标 | 体验 |
|---|---|---|
| TTFT | < 150 ms | "即时响应" |
| TPOT | < 50 ms | 流畅流式（> 20 tok/s） |
| 总延迟（100 token 输出） | < 2 s | 人类阅读速度以内 |

### 1.3 推理服务的显存全景

与训练不同，推理的显存结构为：

$$M_{\text{total}} = \underbrace{M_{\text{weight}}}_{\text{模型权重}} + \underbrace{M_{\text{KV}}}_{\text{KV Cache}} + \underbrace{M_{\text{act}}}_{\text{激活 + 临时缓冲}}$$

权重固定（$N \times \text{bytes\_per\_param}$，详见 本系列[第三篇](../part-3/)）。KV cache **随并发请求数和序列长度线性增长**——这是推理服务显存的最大变量。

本章后续各节的核心问题是：**在有限的 GPU 显存中，如何在 KV cache 约束下最大化并发请求数，同时满足延迟 SLA。**

---

## CH 2 | 连续批处理：请求可以随时加入和离开

### 2.1 为什么静态批处理不够

静态批处理（训练中使用的模式）要求一个 batch 内的所有请求一起开始、一起结束。在服务场景下：

- 短输出（10 tokens）的请求必须等长输出（1000 tokens）的请求完成才能整体结束——短请求被"拖累"
- 新请求必须等当前 batch 全部完成才能加入——GPU 在 batch 末尾出现空闲

连续批处理（Continuous Batching, 也叫 iteration-level scheduling）解决这个问题：**每个 iteration 独立决定 batch 的组成。**

### 2.2 vLLM 的调度机制

vLLM（当前最主流的开源推理引擎）维护三个请求队列：

| 队列 | 含义 | 转移条件 |
|---|---|---|
| **waiting** | 尚未开始 prefill | prefill 被调度时 → running |
| **running** | 正在 prefill 或 decode | decode 完成 → 移除；KV cache 不足 → swapped |
| **swapped** | KV cache 被换出到 CPU | 显存释放后 → running |

每步迭代的调度逻辑（`vllm/core/scheduler.py`）：

```
1. 调度所有 running 请求的 1 个 decode token
2. 如果 token 预算有剩余，调度 waiting 请求的 prefill
3. 如果显存不足，将部分 running 请求的 KV cache swap 到 CPU
4. 新 prefill 可按 chunk 分批（chunked prefill），与 decode 混合在同一 iteration
```

### 2.3 分块 Prefill

长 prompt（如 100K tokens 的文档）的单次 prefill 会阻塞所有其他请求。vLLM 的 chunked prefill（`--enable-chunked-prefill`）将长 prefill 切分为多个 chunk，每个 iteration 处理一个 chunk，中间穿插 decode 请求。

```
传统 prefill（阻塞）:
  [==== 100K token prefill ====] [decode] [decode] ...
  
Chunked prefill（不阻塞）:
  [prefill 8K] [decode] [prefill 8K] [decode] [prefill 8K] ...
```

每次 iteration 的 token 预算由 `--max-num-batched-tokens` 控制。chunked prefill 使长 prompt 的 TTFT 略有增加（分多次处理），但大幅改善了其他请求的 TPOT。

### 2.4 多步调度

为了减少调度器开销，vLLM 支持一次调度决策执行多个 forward step（`--num-scheduler-steps N`）。实测在 Llama 8B on H100 上，多步调度（N=8）将吞吐从 25.96 req/s 提升到 44.44 req/s（**+71%**）。

---

## CH 3 | PagedAttention：让 KV Cache 像虚拟内存一样管理

### 3.1 传统方案的问题

推理系统需要为每个请求的每一层缓存 K 和 V 张量。传统方案（如 FasterTransformer）为每个请求预分配**连续的** `max_seq_len × 2 × L × H_kv × D × bytes` 空间。

问题：
- **内部碎片**：短请求（20 tokens）占用了 4096 tokens 的空间——**利用率仅 20-40%**
- **外部碎片**：变长分配的连续空洞无法合并使用

### 3.2 PagedAttention 的解决方案

PagedAttention（vLLM, SOSP 2023）将 KV cache 划分为**固定大小的块**（默认 16 tokens per block），通过块表（类似页表）映射逻辑位置到物理块：

```
请求 A（30 tokens）的 KV cache:
  块 0: tokens 0-15  → 物理块 #7
  块 1: tokens 16-29 → 物理块 #3

请求 B（8 tokens）的 KV cache:
  块 0: tokens 0-7   → 物理块 #12
```

**惰性分配**：块在需要时才分配（decode 每生成 16 tokens 分配一个新块），请求结束时立即释放。**块共享**：多个请求的前缀相同时（如共享 system prompt），它们的 KV cache 块可以共用（引用计数），节省显存并避免重复计算。

### 3.3 显存利用率

PagedAttention 将 KV cache 利用率从传统方案的 **20-40% 提升到 ~96%**（SOSP 论文数据），直接使最大并发数提升 2-4×。

### 3.4 块大小选择

| 块大小 | 优点 | 缺点 |
|---|---|---|
| 4 tokens | 碎片最少 | 块表开销大 |
| **16 tokens（默认）** | **碎片与开销的最佳平衡** | — |
| 32 tokens | 块表更小 | 每个序列平均浪费 16 tokens 空间 |
| 256 tokens | 管理简单 | 内部碎片严重（>50%） |

### 3.5 前缀缓存的命中率

在系统级 prompt 相同（所有请求共享同一个 system prompt）或多轮对话（共享历史）场景下，KV cache 的块级共享带来显著收益：

| 场景 | KV cache 命中率 | 吞吐提升（vs 无共享） |
|---|---|---|
| 多轮对话（4 轮历史） | ~81% | ~30% |
| RAG（共享 2K token 文档） | ~72% | ~25% |
| Agent（共享工具定义 + 记忆） | ~88% | ~40% |
| 独立 prompt | 0% | 无 |

vLLM 通过 `--enable-prefix-caching` 启用自动前缀缓存。

---

## CH 4 | Prefill-Decode 分离：让不同 GPU 做不同的事

### 4.1 问题的本质

Prefill（处理 prompt）和 decode（逐 token 生成）的计算特性完全不同：

| | Prefill | Decode |
|---|---|---|
| 计算模式 | **计算密集型**（O(S²) attention + 大 GEMM） | **内存带宽密集型**（权重 I/O 主导） |
| 延迟敏感度 | TTFT（首个 token） | TPOT（流式体验） |
| 资源需求 | 高 TFLOPS GPU | 高 HBM 带宽即可（甚至可用更便宜的 GPU） |

放在同一 GPU 上会互相干扰——长 prefill 会阻塞 decode，导致 TPOT 飙升。**分离部署**（disaggregation）将 prefill 和 decode 放到不同的 GPU 上。

### 4.2 DistServe 的分离架构

DistServe（OSDI 2024）的架构：

```
请求 → [Prefill 节点组] → [KV cache 传输] → [Decode 节点组] → 输出
         ↑ H100                  ↑                ↑ 带宽充足即可
         计算密集型              通信开销          内存带宽密集型
```

**核心收益**：Prefill 和 Decode 节点可以独立扩缩容。当流量中的 prefill 请求激增（长文档查询），只扩 Prefill 节点；当并发对话增多（decode 为主），只扩 Decode 节点。

实测数据（OPT-66B, 4×A100, DistServe vs 协同部署）：

| 指标 | 值 |
|---|---|
| per-GPU goodput | **+2.0-4.6×** |
| SLO 满足率（<200ms TTFT, <50ms TPOT） | **>90%** |
| KV cache 传输开销 | ~25% 请求时间（可优化至 <1% 通过 FlowKV 等技术） |

### 4.3 分离的代价：KV Cache 传输

Prefill 节点生成的 KV cache 需要传给 Decode 节点。传输量 = $2 × L × H_{kv} × D × S × 2$ bytes（K+V, BF16, S=prompt length）。

对于 Llama-70B（$L=80$, $H_{kv}=8$, $D=128$, $S=32768$）：约 1.07 GB per 请求。在 IB 50 GB/s 下约 21 ms——对长 prompt 可接受，对短 prompt 是显著开销。

优化方向：FlowKV 将不连续的 PagedAttention 块合并为连续缓冲区传输，延迟降低 **96%**（0.94s → 0.05s per 13K prompt）。KVDirect 减少同步开销将传输带宽利用率从 1.8% 提升到 >80%。

### 4.4 什么时候值得分离

- **长 prompt + 高并发**：几乎总是值得（prefill 瓶颈与 decode 延迟叠加）
- **短 prompt（<512 tokens）+ 低并发**：不必要（传输开销 > 收益）
- **同构 GPU 池**：分离仍有效（隔离干扰），但收益不如异构池
- **绑定式部署**：prefill 和 decode 仍放一起，通过 chunked prefill 缓解干扰——大部分小规模部署的选择

---

## CH 5 | 推测解码：用小模型加速大模型

### 5.1 原理

推测解码（Speculative Decoding, Leviathan et al. ICML 2023）用一个小型"草稿模型"（draft model）快速生成 $\gamma$ 个候选 token，然后让大模型（target model）一次性验证所有候选：

```
1. Draft model 自回归生成 [t1, t2, t3, t4]（4 个候选 token）
2. Target model 一次 forward pass 验证全部 4 个候选
3. 接受前 3 个（t1, t2, t3 与 target 分布一致），重采样第 4 个
4. 结果：1 次 target forward pass → 4 个新 token
```

**数学保证**：推测解码的输出分布与直接使用目标模型采样完全一致——没有质量损失。

### 5.2 加速比

$$\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$

其中 $\alpha$ = 逐 token 接受率（draft 与 target 一致的概率），$\gamma$ = 每次猜测 token 数，$c$ = draft 推理时间 / target 推理时间。

典型值（7B draft → 70B target, $\alpha \approx 0.8$, $\gamma=4$, $c=0.1$）：

$$\text{Speedup} \approx \frac{1 - 0.8^5}{(1 - 0.8)(4 \times 0.1 + 1)} \approx \frac{0.67}{0.28} \approx 2.4\times$$

### 5.3 实测数据

| 方法 | Draft | Target | 加速比 | 平均接受长度 |
|---|---|---|---|---|
| 标准推测解码 | 7B | 70B | **~1.9-2.0×** | 3.4-3.8 |
| EAGLE-2 | 轻量 head | 70B | **2.6-5.2×** | 4.4-5.2 |
| 低温度（<0.5）时 | — | — | **可达 9×** | — |
| 高温度（>1.0）时 | — | — | 加速衰减 | 接受率降低 |

### 5.4 批大小的影响

推测解码在**小 batch**（B ≤ 4）时效果最好——因为 decode 在小 batch 下是极度 memory-bound 的（AI ≈ 1，见 Roofline 分析）。draft model 的额外开销（$c \approx 0.1$）相比 target model 的权重 I/O 节省微不足道。

**在大 batch（B > 8）时加速比显著缩小**：decode 逐渐变为 compute-bound（batch 增大使 AI 接近 Ridge），draft model 的相对收益下降。实测 B=32 时加速比通常 <1.3×。

**工程启示**：推测解码适合**低并发、交互式**场景（如聊天助手），不适合**高吞吐、批处理**场景（如批量评估）。

---

## CH 6 | 从 Roofline 到 TTFT 和 TPOT

本节将[系列第四篇](../part-4/) CH 7 的 Roofline 模型扩展到**单请求延迟**的精确估算，为后续章节的**并发场景**提供基础输入。

### 6.1 Prefill 延迟（TTFT）

Prefill 一次性处理 $S$ 个 prompt token。计算时间由两部分组成：

$$T_{\text{prefill}} = \max\left(\frac{\text{FLOPs}_{\text{prefill}}}{\text{TFLOPS}_{\text{eff}}}, \frac{\text{Bytes}_{\text{prefill}}}{\text{BW}_{\text{eff}}}\right)$$

$$\text{FLOPs}_{\text{prefill}} = 2NS + 4S^2 H_q D L$$

$$\text{Bytes}_{\text{prefill}} = 2N + 2L H_{kv} D S \quad (\text{权重读 + KV 写})$$

其中 $2N$ 是模型权重 I/O（BF16），$2L H_{kv} D S$ 是 KV cache 写入。

以 Llama-70B（$d=8192$, $L=80$, $H_q=64$, $H_{kv}=8$, $D=128$）on H100 BF16 为例：

$S=128$（短 prompt）: $\text{FLOPs}_{\text{prefill}} \approx 2×70e9×128 + 4×128^2×64×128×80 \approx 1.79×10^{13}$。$T_{\text{compute}} \approx 1.79e13/742e12 \approx 24$ ms。$T_{\text{mem}} \approx 140\text{GB}/3.35\text{TB/s} \approx 42$ ms。**Memory-bound** → TTFT ≈ max(24, 42) ≈ 42 ms。

$S=2048$（长 prompt）: $\text{FLOPs}_{\text{prefill}} \approx 2×70×10^9×2048 + 4×2048^2×64×128×80 \approx 2.87×10^{14} + 1.10×10^{13} \approx 2.98×10^{14}$。$T_{\text{compute}} \approx 2.98×10^{14}/742×10^{12} \approx 0.40$ s = **400 ms**。$T_{\text{mem}} \approx (140+0.08)/3.35e3 \approx 29$ ms。**Compute-bound** → TTFT ≈ 400 ms。

**分水岭**：约 $S \approx 400$ tokens 时，attention O(S²) 使 prefill 从 memory-bound 转为 compute-bound（$\text{AI} = \text{FLOPs}/\text{Bytes}$ 超过 H100 Ridge ≈ 206）。短 prompt 的 TTFT 由**权重 I/O**决定（~42 ms）；长 prompt 由 **attention 计算**决定（随 $S^2$ 增长）。

### 6.2 Decode 延迟（TPOT）

Decode 每步只产生 1 个新 token——与 Roofline 分析（本系列[第四篇 CH 7.4](../part-4/)）完全一致：

$$T_{\text{decode}} \approx \frac{2N}{\text{BW}_{\text{HBM}}} \quad (\text{memory-bound, AI ≈ 1})$$

对于 Llama-70B on H100：$T_{\text{decode}} \approx 140\text{GB}/3.35\text{TB/s} \approx 42$ ms。TPOT ≈ **42 ms per token**（单请求，无并发）。

**Ascend 910C 侧**：HBM 带宽 3.2 TB/s 仅为 H100 3.35 TB/s 的 96%，容量 64 GB 为 H100 80 GB 的 80%。Llama-70B BF16 需 TP=4（140/4=35 GB < 64 GB）。TPOT = 35/3.2 ≈ 11 ms（理想）或 ~14 ms（HBM2e 效率 ~80%）。与 H100 TP=4 的 10.4 ms 接近（1.05×）。**公式推导，待实测验证。**

### 6.3 并发对延迟的影响

当 $B$ 个请求同时 decode 时，权重 I/O 被 $B$ 个请求共享（权重读完一次，可同时为 $B$ 个请求计算注意力和 FFN），但 KV cache 读取随 $B$ 线性增长（每个请求需要自己的 KV）：

$$T_{\text{decode}}(B) \approx \max\left(\frac{2N}{\text{BW}_{\text{HBM}}}, \frac{2N \cdot B}{\text{TFLOPS}_{\text{eff}}}\right)$$

第一个分支（memory-bound）在 $B$ 较小时成立，第二个分支（compute-bound）在 $B$ 增大后成立。**TPOT 不随 $B$ 线性增长**——权重 I/O 是固定的，这让 decode 在 moderate batch 下非常高效。

以 Llama-70B on H100 为例，不同 $B$ 下的 TPOT：

| $B$ | TPOT | 总吞吐 | 注 |
|---|---|---|---|
| 1 | ~42 ms | 24 tok/s | 纯 memory-bound |
| 4 | ~45 ms | 89 tok/s | 仍 memory-bound |
| 16 | ~45 ms | 356 tok/s | 接近 compute-bound 临界 |
| 64 | ~80 ms | 800 tok/s | compute-bound，受 TFLOPS 限制 |

每 token 延迟随 $B$ 增大而上升，但**总吞吐**持续增长——这是推理服务的核心 trade-off：**延迟 vs 吞吐**。

### 6.4 端到端公式

$$\text{Latency} = T_{\text{prefill}}(S) + T_{\text{decode}}(B) \times (N_{\text{out}} - 1)$$

其中 $S$ 是 prompt 长度，$N_{\text{out}}$ 是输出 token 数，$B$ 是当前 batch 中同时 decode 的请求数。

---

## CH 7 | 多 GPU 推理：TP 和 PP 在服务中的角色

### 7.1 TP vs PP 的服务场景对比

训练中 TP 放机内（NVLink）、PP 跨节点是标准做法。推理中有所不同——因为**权重 I/O 是 decode 瓶颈**，TP 将权重分片到多卡上，可同时降低每卡的权重 I/O 量和计算量：

| | TP=4 | PP=4 |
|---|---|---|
| 每卡权重 I/O | $2N/4 = 35$ GB | $2N/4 = 35$ GB（但只有 L/4 层） |
| 单请求 TPOT | **~10 ms** | ~42 ms（各 stage 串行, 总延迟不变） |
| 吞吐（高并发） | 中等 | **高**（通信少） |

TP 降低单请求延迟（分片+all-reduce 并行），PP 提升吞吐（减少通信开销）。**推理服务通常选 TP**——因为延迟比吞吐更敏感。

**Ascend 910C 侧**：HCCS all-reduce 有效带宽 ~48 GB/s（vs NVLink ~360 GB/s, 7.5× 差距）。TP 通信开销在 910C 上更显著。对于 Llama-70B TP=4：每层 all-reduce ~302 MB（B=1 时 ~8 KB），在 48 GB/s 下 < 0.2 ms/层，仍可忽略。但 TP=8 时每卡权重减半（17.5 GB），TPOT 降到 ~4 ms 但通信开销翻倍——**910C 的 TP 最优值通常 ≤4，NVLink 上可以到 8。**

### 7.2 具体模型配置

| 模型 | 精度 | 单卡显存 | 推荐配置 |
|---|---|---|---|
| Llama 7B | BF16 | 14 GB | 1×H100 |
| Llama 70B | BF16 | 140 GB | **TP=2**（2×H100, 每卡 70 GB） |
| Llama 70B | FP8 | 70 GB | 1×H100 |
| Llama 405B | FP8 | 405 GB | **TP=8**（8×H100, 每卡 ~51 GB） |
| Mixtral 8×7B | BF16 | ~94 GB | **TP=2** 或 EP=2 |

### 7.3 上下文并行在服务中的角色

**上下文并行（CP）仅在 prefill 阶段使用**——将长 prompt 的 $S$ 切分为 $S/cp$ 份，并行计算 attention，然后用 all-reduce 聚合。CP 对 decode 没有帮助（decode 只需 1 token 的 attention）。

CP 的加速比接近 $cp$（实测在 $S=1M$ 时 128 GPU 的 93% 并行效率），但 CP 只降 TTFT，不降 TPOT。在长 prompt 服务中（RAG、文档分析），CP 是 TTFT SLA 的关键保障。

---

## CH 8 | 量化在服务中的角色

### 8.1 量化的两个收益

| 收益 | 机制 | 效果 |
|---|---|---|
| **显存节省** | 权重和 KV cache 变小 | 可以装更大模型、更大 batch、更长上下文 |
| **延迟降低** | decode 时读取的字节数减少（memory-bound 下 I/O 是瓶颈） | TPOT 近似与 bytes/param 成正比 |

在 memory-bound 的 decode 阶段（CH 6.2），$T_{\text{decode}} \approx 2N / \text{BW}$。量化将 $2N \times \text{bytes\_per\_param}$ 变小——是**读得少了**，不是算得快了（与 本系列[第四篇 CH 7.6](../part-4/) 的结论一致）。

### 8.2 实际效果

H100 FP8 vs BF16（Llama-70B, B=1）：

| 精度 | TPOT | 加速 | 显存节省 |
|---|---|---|---|
| BF16 | ~42 ms | 1× | baseline |
| FP8 | **~21 ms** | **2.0×** | ~50% |
| INT4 AWQ | **~11 ms** | **2.7×** | ~75% |

**重要警告**：INT4 在高 batch（B>32）下性能**退化**——H100 无原生 INT4 Tensor Core，解量化 kernel 在高并发时成为瓶颈。实测 Llama-70B at B=64: FP8 2.1× 加速 vs INT4 0.88×（比 BF16 还慢）。**高并发服务优先选 FP8，低并发交互优先选 INT4。**

### 8.3 精度损失

| 量化 | Llama-70B MMLU 损失 | 推荐场景 |
|---|---|---|
| FP8 E4M3 | **0.5%** | H100 通用首选，质量损失可忽略 |
| INT4 AWQ | **1.3%** | A100 或显存受限场景 |
| 两者差距 | <1% on 70B, 更大 on 小模型 | 大模型更耐量化 |

**Ascend 910C 侧**：**910C 不支持原生 FP8**（需等下一代 950）。量化路径为 INT8（W8A8）和 INT4（W4A16 AWQ/GPTQ）。INT8 权重将 70B 从 140 GB 降到 70 GB + 量化参数，TPOT 预期从 ~42 ms 降到 ~21 ms（×2 加速）。INT4 可进一步降到 ~11 ms。但 910C 同样存在高 batch 下 INT4 退化问题。**核心差异：H100 的"首选 FP8"在 910C 上不可用，INT8 是实际上的第一梯队量化方案。**

---

## CH 9 | 完整推演：Llama-70B 推理服务分析

### 9.1 配置

Llama-70B BF16 on 4×H100（TP=4），vLLM 部署。

### 9.2 单请求基线（CH 6 模型）

$S=1024$, $N_{\text{out}}=256$, TP=4 将每卡权重从 140 GB 降到 35 GB，每卡计算量降为 1/4：

$$T_{\text{prefill}} \approx \max\left(\frac{2×70×10^9×1024 + 4×1024^2×64×128×80}{4 × 742×10^{12}}, \frac{35\text{GB}}{3.35\text{TB/s}}\right) \approx \max(49, 10.4) \approx 49 \text{ ms}$$

$$T_{\text{decode}} \approx 35\text{GB}/3.35\text{TB/s} \approx 10.4 \text{ ms}$$

端到端：$49 + 10.4 \times 255 \approx 49 + 2652 \approx 2.70$ s。吞吐：$256/2.70 \approx 95$ tok/s。

### 9.3 KV Cache 约束下的最大并发

4×H100，每卡 80 GB = 320 GB 总显存。权重 = 140 GB（BF16 全量，TP=4 分片后每卡 35 GB，总计 140 GB）。剩余显存 = 320 - 140 = 180 GB 用于 KV cache。

每请求 KV cache（$S=1024$, $N_{\text{out}}=256$，总序列长 1280, $L=80$, $H_{kv}=8$, $D=128$）：
$$M_{\text{KV}}^{(1)} = 2 × 80 × 8 × 128 × 1280 × 2 \approx 419 \text{ MB}$$

PagedAttention 利用率 ~96%：实际约 436 MB/请求。最大并发 $B_{\text{max}} \approx 180\text{GB}/0.436\text{GB} \approx 413$ 个请求。

**但在到达 KV cache 上限之前，延迟 SLA 率先成为约束。**

### 9.4 SLA 约束下的可行并发

设 SLA: TTFT < 200ms, TPOT < 50ms。TTFT = 49 ms ——远小于 200ms（OK）。TPOT 随 $B$ 增长：

| $B$ | TPOT (ms) | 满足 <50ms？|
|---|---|---|
| 1 | 10.4 | ✓ |
| 4 | 11.5 | ✓ |
| 16 | 12 | ✓ |
| 64 | 28 | ✓ |
| 128 | 52 | **✗（超出 SLA）** |

**SLA 约束下的最大并发 $B_{\text{max}}^{\text{SLA}} \approx 100$**，远小于显存约束的 413。**SLA 是真正的瓶颈——不是显存，是延迟容忍。**

### 9.5 换 FP8 后的提升

FP8 下，70B 模型仅需 70 GB 显存（vs BF16 的 140 GB），可用更少的 GPU。三个配置对比：

**配置 A: FP8 + 1×H100（无 TP）**
- 权重 70 GB on 1 GPU。TPOT = 70/3.35 ≈ 20.9 ms——比 BF16 TP=4 的 10.4 ms 慢 2×，但只用 1/4 的 GPU。
- 剩余显存: 80 - 70 = 10 GB → $B_{\text{max}}^{\text{KV}} \approx 10/0.436 \approx 23$。

**配置 B: FP8 + TP=2（2×H100）**
- 每卡权重 35 GB。TPOT = 35/3.35 ≈ 10.4 ms——与 BF16 TP=4 延迟相同，但只需一半 GPU。
- 剩余显存: 2×80 - 70 = 90 GB → $B_{\text{max}}^{\text{KV}} \approx 206$。

**配置 C: FP8 + TP=4（4×H100，与基线相同 GPU 数）**
- 每卡权重 17.5 GB。TPOT = 17.5/3.35 ≈ 5.2 ms——延迟比 BF16 TP=4 快 2×。
- 剩余显存: 4×80 - 70 = 250 GB → $B_{\text{max}}^{\text{KV}} \approx 573$。若 KV cache 也量化到 FP8（每请求 210 MB）→ $B_{\text{max}} \approx 2352$。
- **TPOT 降低使 SLA 约束大幅放松**：B=128 时 TPOT 从 52 ms（超 SLA）降到 ~26 ms（满足 SLA）。实际最大并发从 ~100 提升到 ~200+。

启示：**FP8 的核心价值是"单位 GPU 的延迟-吞吐曲线整体上移"——相同延迟下支持更多并发，或者相同并发下延迟更低。** 具体选 A/B/C 取决于预算和 SLA 要求。

### 9.6 加推测解码后的提升

B=1 时推测解码加速 2.4×（CH 5.3）→ 等效 TPOT = 10.4/2.4 ≈ 4.3 ms。但推测解码在 B>8 时加速衰减：

| $B$ | 无推测 TPOT | 推测加速比 | 等效 TPOT | 满足 <50ms？|
|---|---|---|---|---|
| 1 | 10.4 | 2.4× | 4.3 ms | ✓ |
| 4 | 12   | 1.8× | 6.7 ms | ✓ |
| 16 | 12 | 1.3× | 9.2 ms | ✓ |
| 64 | 28 | 1.1× | 25 ms | ✓ |
| 128 | 52 | ~1× | 52 ms | ✗ |

**推测解码在低并发下效果极佳，但不能提升 SLA 约束下的最大并发——因为在那个并发级别加速比已经接近 1×。**

### 9.7 Ascend 910C 等效分析（简要）

系统分析方法完全一致，仅替换硬件常数。以 Llama-70B BF16, TP=4, 4×910C 为例（对标 CH 9.1-9.4）：

| 指标 | H100 TP=4 | 910C TP=4 | 差异 |
|---|---|---|---|
| 每卡权重 | 35 GB | 35 GB | 相同 |
| TPOT (B=1) | ~10.4 ms | **~14 ms** | 1.3×（HBM2e 效率 ~80%，理想值 1.05×） |
| KV cache (4卡合计) | 180 GB | **116 GB** | 0.80×（910C 64 GB vs H100 80 GB） |
| B_max (显存) | ~413 | **~266** | 0.64× |
| B_max (SLA, <50ms) | ~100 | **~75** | 0.75× |

910C 在单卡层面略弱于 H100（64 GB vs 80 GB, 3.2 vs 3.35 TB/s），差距在 30% 以内（TPOT）和 20%（容量）。其推理优势在集群层面（CloudMatrix UB 总线跨节点带宽远超 IB），不在单卡。

---

## CH 10 | MoE 模型服务

当前 SOTA 模型（DeepSeek-V3/R1、Mixtral、Qwen-MoE）几乎全是 MoE。MoE serving 的分析框架与 Dense 相同（CH 6-9），但有两个 Dense 不具备的杠杆和一个约束：

- **杠杆 1**：活跃参数 $\ll$ 总参数 → 权重 I/O 小 → TPOT 低
- **杠杆 2**：EP 将 experts 分布到多卡 → 每卡显存需求降低
- **约束**：MLA 架构下 TP 无法分片 KV cache → DP+EP 几乎是强制选择

### 10.1 为什么 MoE serving 需要 EP

**显存约束**：MoE 总参数量极大。DeepSeek-V3 671B，BF16 总权重 = 1342 GB——远超任何单卡（H100 80 GB, H200 141 GB, 910C 64 GB）。

不用 EP 的话，每个 GPU 必须加载**全部 expert**。即使 TP 把每层的权重矩阵切了（如 TP=8 时每卡 $1/8$ 的参数），每个 TP rank 仍要存所有 256 个 expert 的 $1/8$ 分片——总量仍是 1342/8 ≈ 168 GB > 80 GB。TP 只切矩阵维度，不减少 expert 数量。

**EP 解决这个问题**：将 256 个 expert 分布到 EP 个 GPU 上，每卡只持有 $256/EP$ 个 expert 的**完整权重**（无需分片）。EP=128 时每卡约 2 个 expert + 1 个 shared expert，权重约 5.2B（INT8 下 ~5.2 GB）。

**vLLM、TensorRT-LLM、SGLang、vllm-ascend 全部原生支持 EP for serving。** DeepSeek-V3 生产部署用 EP128 decode, CloudMatrix384 走 EP320。

### 10.2 两种 EP 模式：TP+EP vs DP+EP

EP 在 serving 中不是独立的并行维度——它总是与 TP 或 DP 组合使用，形成两种模式：

**模式 A: TP+EP（低延迟）**

`ep = tp × dp`，其中 `dp=1`。所有 GPU 属于同一个 DP 组。

每个 GPU 持有 $1/ep$ 的 expert 完整权重。前向传播时，token 路由到目标 expert 所在的 GPU。但关键差异在于——因为所有 GPU 在同一个 DP 组内，TP 的 AllReduce 聚合被用于"模拟" token 的路由收集，而**不是 AllToAll**。流程为：

```
每个 GPU: 本地 gate → 本地 expert FFN（只有路由到本地的 token）
        → TP AllReduce（聚合所有 GPU 的部分结果）
```

AllReduce 的延迟远低于 AllToAll（NVLink ~360 GB/s vs all-to-all ~150 GB/s）。TP+EP 在低并发下延迟更优。

代价：KV cache 在所有 TP rank 上复制，显存效率低。

**模式 B: DP+EP（高吞吐）**

`dp > 1`。EP 组分布在不同的 DP 组之间，通信使用真正的 AllToAll：

```
每个 DP 组: gate → AllToAll(dispatch) → expert FFN → AllToAll(combine)
```

AllToAll 的延迟高于 AllReduce，但 KV cache 按 DP 维度分区（如 DP=32 时每 rank 仅 $1/32$ 的 KV cache），显存效率极高。

**两种模式的对比**：

| | TP+EP (dp=1) | DP+EP (dp>1) |
|---|---|---|
| 通信原语 | AllReduce | AllToAll |
| 每卡 expert 数 | $N_E / EP$ | $N_E / EP$ |
| KV cache 分布 | TP rank 间复制 | **DP rank 间分区** |
| 适用场景 | 低并发、延迟敏感 | 高并发、吞吐优先 |
| 典型配置 | EP=8~32, TP=4~8 | EP=64~320, DP=16~64 |

### 10.3 MLA 为什么强制要求 DP+EP

DeepSeek-V3/R1 使用 MLA（Multi-head Latent Attention，见 本系列[第三篇 CH 4.3](../part-3/)）。MLA 的关键特性：所有 attention head 共享一个低秩潜向量 $\mathbf{c}_t^{KV}$，不存在独立的 per-head K/V。

这对 TP 的影响是致命的：**TP 无法沿 head 维度分片 KV cache**（因为只有 1 个"逻辑头"）。使用 TP+EP 时，KV cache 在所有 TP rank 上**完全复制**——8 卡 TP 下 KV cache 浪费 8× 显存。DP+EP 将 KV cache 沿 DP 维度按请求分区，每 rank 只存 $1/DP$ 的 KV，避免了复制。

**这也是为什么 DeepSeek-V3 的推理配置是 EP128 = DP32 × TP4，而非 TP=32 或其他配置。** DP+EP 是 MLA 架构下的唯一高效方案。

### 10.4 AllToAll 延迟如何控制

DP+EP 的致命弱点是 AllToAll 延迟。CH 7.2 详细分析了 EP AllToAll 在训练中无法掩盖——serving 中同样面临这个挑战。工程上通过三个手段缓解：

**（1）微批次流水线**：DeepSeek 将每层 MoE 计算拆为 5 个阶段（Gate → Dispatch → Expert FFN → Combine → Post-norm），两个微批次交替执行。dispatch AllToAll 与另一个微批次的 shared expert 计算重叠，实测降低延迟 29%。

**（2）专用通信库**：DeepEP（DeepSeek 开源）使用 NVLink 零拷贝 + RDMA，在 GB200 NVL72 上 dispatch 带宽可达 753 GB/s。HCCL/XCCL 在 CloudMatrix UB 总线上 AllToAll 带宽 103-131 GB/s。

**（3）大 EP 减小单次通信量**：EP 越大，每 GPU 持有的 expert 越少，每次 AllToAll 的数据量越小（$T_{\text{comm}} \propto 1/EP$）。EP=320 时单次 AllToAll 的数据量仅为 EP=8 时的 $1/40$。

### 10.5 案例对比

| | Dense 70B | Mixtral 8×7B | DeepSeek-V3 |
|---|---|---|---|
| 总参 / 活跃参 | 70B / 70B | 47B / **13B** | 671B / **37B** |
| 总权重 (BF16) | 140 GB | 94 GB | 1342 GB |
| 并行策略 | TP=2 | TP=2 | **EP128 = DP32 × TP4** |
| 每卡权重 I/O | 70 GB | 26 GB | **~7.2 GB** (INT8) |
| TPOT (B=1) | ~21 ms | ~7.8 ms | **~2.1 ms** |
| GPU 需求 | 2 卡 | 2 卡 | 128+ 卡 |
| KV cache 分布 | TP 复制 | TP 复制 | **DP 分区** |

### 10.6 MoE serving 选择决策树

```
MoE 模型 serving
├─ 总权重 < 单卡显存（如 Mixtral 47B on H100 80GB）
│  └─ 无需 EP，TP 即可。活跃参数小 → TPOT 已很低
│
├─ 总权重 >> 单卡显存（如 DeepSeek 671B）
│  ├─ 非 MLA 架构（如 Qwen-MoE, GQA attention）
│  │  ├─ 低并发 → TP+EP（AllReduce, 延迟优）
│  │  └─ 高并发 → DP+EP（KV cache 分区, 吞吐优）
│  │
│  └─ MLA 架构（DeepSeek-V2/V3/R1）
│     └─ **必须 DP+EP**（TP 无法分片 KV cache）
│        EP=DP×TP, 选择使 DP 足以分区 KV cache 的配置
│
└─ 超大规模（>384 卡）
   └─ CloudMatrix UB（910C）或 NVL72（B200）
      超大 EP（128-320），依赖专用通信库控制延迟
```

### 10.7 总结

1. **活跃参数决定 TPOT，EP 决定可部署性**：两者是独立的杠杆。活跃参数小让 decode 快；EP 让大模型装得进 GPU 池
2. **EP 在 serving 中不仅被使用，而且是必须的**——671B 模型没有 EP 无法部署
3. **MLA 架构强制 DP+EP**：TP 无法分片单头 KV cache，DP+EP 是唯一高效方案
4. **AllToAll 延迟通过微批次流水线 + 专用通信库控制在可接受范围**，与 本系列[第六篇 CH 7](../part-6/) 的训练 EP 分析形成呼应

---

## CH 11 | 推理服务分析方法论

### 11.1 分析清单

- [ ] **算单 token decode 延迟**（Roofline, 见[系列第四篇](../part-4/)）
- [ ] **建模 KV cache**：$M_{\text{KV}} = f(L, H_{kv}, D, S, B)$ —— 显存上限约束
- [ ] **算 TTFT**：$T_{\text{prefill}} = \max(T_{\text{compute}}, T_{\text{mem}})$ —— 分水岭在 $S \approx 400$
- [ ] **算 TPOT(B)**：随 $B$ 从 memory-bound 渐变到 compute-bound
- [ ] **设 SLA 约束**：找到最大 $B_{\text{SLA}}$（TPOT < 50ms）
- [ ] **算最大并发**：$\min(B_{\text{SLA}}, B_{\text{KV}})$ —— 通常 SLA 先触顶
- [ ] **评估优化手段**：量化（FP8/INT4）、推测解码、分离部署
- [ ] **多 GPU 扩展**：TP 降延迟、PP 提吞吐

### 11.2 常见错误

| 错误 | 纠正 |
|---|---|
| 用峰值 TFLOPS 算 decode 延迟 | Decode 是 memory-bound（AI≈1），用 HBM BW |
| 认为 batch 越大吞吐越高（无上限） | SLA 约束下存在最优 $B$，超过后 TPOT 违规 |
| 对所有场景用推测解码 | B>8 时加速趋近 1×，推测解码不是万能药 |
| 忽略 PagedAttention 的碎片 | 块大小 16 tokens, 每个序列浪费 ~8 tokens 的空间 |
| 用全模型参数算权重 I/O（多 GPU 时） | TP 后 per-GPU 权重 = 总参 / TP，不是总参 |

---

> **系列导航**：[（一）预备知识](../part-1/) → [（二）FLOPs](../part-2/) → [（三）KV Cache](../part-3/) → [（四）M3+Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) ← 当前（系列终篇）
