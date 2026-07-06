+++
math = true
date = '2026-07-06'
draft = false
title = 'Tencent Hy3-295B 架构深度拆解'
categories = ['architecture']
vendor = 'Tencent'
tags = ['moe', 'attention', 'model-architecture', 'tencent', 'hy3', 'gqa', 'mtp', 'sigmoid-router', 'qk-norm']
series = ['architecture']
summary = '腾讯混元 Hy3-295B 是腾讯 2026 年发布的开源 MoE 模型（Apache 2.0）。核心设计：隐藏维度仅 4096（业内最小）通过 80 层 + 192E Sigmoid 路由器补偿容量、QK-Norm 稳定深层训练、1 层 MTP 加速推理、原生 256K 上下文。本期完整拆解 295B 规模配置、参数分解、FLOPs/KV Cache/推理显存、Sigmoid 路由与 QK-Norm 机制、训练体系。'
+++


## CH 0: 摘要

Hy3-295B 是腾讯混元团队 2026 年 5 月发布的开源 MoE 大语言模型[^1]，采用 Apache 2.0 协议。模型总参数量 295B[^2]，每 token 仅激活 21B 参数（活跃率 7.1%）[^3]，支持原生 256K 上下文窗口[^4]。

Hy3 在架构上有四处显著设计选择：(1) **隐藏维度仅 4096**，远小于同类 295B 级 MoE 模型常见的 6144-8192，通过 80 层深度 + 192 个路由专家补偿容量；(2) **Sigmoid 路由器**替代 Softmax，允许每个专家独立打分，无需辅助负载均衡损失；(3) **QK-Norm** 在 RoPE 前对 Query 和 Key 分别施加 RMSNorm，稳定深层训练；(4) **1 层 MTP**（Multi-Token Prediction）加速推理。

阅读路径：CH 1 梳理 Hy 系列演进脉络，CH 2 展开完整超参配置与参数分解，CH 3 进行 FLOPs 与推理显存计算，CH 4 深入 Sigmoid 路由、QK-Norm、MTP 三大核心机制，CH 5 讨论训练体系，CH 6 总结。

[^1]: 来源：HF 仓库 `tencent/Hy3`，模型架构字段确认。
[^2]: 来源：`config.json` 超参推导 + 官方 README 验证。
[^3]: 来源：`num_experts_per_tok=8`，21B / 295B = 7.1%。官方 README 中"21B active parameters"交叉验证。
[^4]: 来源：`config.json` 中 `max_position_embeddings=262144`。

---

## CH 1: 演进脉络

### 1.1 Hy1 与 Hy2：奠定 MoE 路线

腾讯混元大模型（HunYuan, Hy）系列的公开演进始于 Hy1（2023 年），随后进入 Hy2（2024 年）。Hy2 是一个超过 4000 亿参数的超大规模 MoE 模型，在中文 NLP 任务和通用对话能力上对标 GPT-4 级水平。Hy2 的核心架构特征是大隐藏维度（据 Hy2 时期公开信息推断为 6144-8192 量级）、标准 Softmax 路由器、以及大量的训练数据投入。Hy2 时期混元团队在主模型对齐（SFT + RLHF）上积累了丰富经验，但也意识到：继续扩大模型总参数带来的推理成本与部署门槛已远超收益增速。

### 1.2 Hy3 Preview：转向"智能密度"

2026 年 4 月底，混元团队发布 Hy3 Preview（即 Hy3-Flash 的前身预览版），姚顺雨领衔重建了训练基础设施与 RL 框架。Preview 版本的核心信号是：**Hy3 不再追求参数总量，转而追求"智能密度"——即每单位推理 FLOPs 贡献的下游能力**。这一定位变化直接反映在最终 295B 超参上：隐藏维度从 Hy2 时代的大维度（据公开信息判断为 6144+）骤降至 4096，层数增至 80，专家数扩大到 192。

Preview 版本发布后，团队从 50+ 产品团队收集反馈，修复了任务执行与交互上的多个问题，并显著提升了后训练管线的质量与规模。这一快速迭代周期（约 1 个月）体现了混元团队在工程管线上的成熟度。

### 1.3 Hy3 正式版：架构定型与开源

2026 年 5 月正式发布的 Hy3 在架构上与 Preview 版本一致，主要提升来自后训练阶段的数据质量、多样性以及 RL 训练的规模扩展。模型以 Apache 2.0 协议开源，提供 BF16 和 FP8 两种精度权重，支持 vLLM 和 SGLang 两种主流推理框架。MTP 推测解码在部署中默认启用（num_speculative_tokens=2）。

Hy3 的定位明确：在相近规模的模型中显著领先，在 2-5 倍参数的旗舰开源模型中保持竞争力。这一定位决定了其架构设计的核心原则——**用更少的活跃参数、更窄的隐藏维度、更多的专家和更深的层数，换来更高的推理效率与部署灵活性**。

### 1.4 设计哲学：层数 vs 隐藏维的 trade-off

Hy3 最核心的架构哲学是一个显式的设计权衡：**窄隐藏维（4096）+ 深层数（80）+ 多专家（192）** vs **宽隐藏维（6144-8192）+ 浅层数（40-60）+ 少专家（64-128）**。

窄隐藏维的优势在于：(1) 单 token 的 Attention 计算量 O(S x d) 与 d 成线性正比——4096 vs 8192 即是 2 倍差距；(2) KV cache 大小同样与 d 成线性正比；(3) 权重加载带宽需求更低。劣势在于：d=4096 时单层 FFN 的表征容量受限（中间维 13312 对应 3.25x 扩张比，属于窄范围）。

Hy3 用三个策略补偿窄维度的容量损失：(1) 层数增至 80（是同等规模 MoE 中偏多的），让信息通过更多非线性变换逐步精炼；(2) 192 个路由专家（远多于常见的 64-128），让 MoE 层的总知识容量大幅提升；(3) Sigmoid 路由器允许 token 同时被多个高分专家处理，弱化单个专家容量瓶颈。

---

## CH 2: 超参与配置

### 2.1 超参表

![架构总览](figures/fig-2.1-architecture-overview.svg)

以下超参全部来自 `config.json`（HuggingFace 标准格式）与源码 `HyV3Config` 类交叉验证。

| 参数 | 值 | 来源 |
|---|---|---|
| `hidden_size` (d) | 4096 | config.json |
| `num_hidden_layers` (L) | 80 | config.json |
| `first_k_dense_replace` | 1（仅第 0 层用 Dense FFN） | config.json |
| `num_attention_heads` (h_q) | 64 | config.json |
| `num_key_value_heads` (h_kv) | 8 | config.json |
| `head_dim` (d_head) | 128 | config.json |
| `intermediate_size` (d_ff_dense) | 13312 | config.json |
| `moe_intermediate_size` (d_ff_expert) | 1536 | config.json |
| `num_experts` (E) | 192 | config.json |
| `num_experts_per_tok` (k) | 8 | config.json |
| `num_shared_experts` | 1 | config.json |
| `num_nextn_predict_layers` (MTP) | 1 | config.json |
| `vocab_size` (V) | 120832 | config.json |
| `max_position_embeddings` (T_max) | 262144 (256K) | config.json |
| `rope_theta` | 11,158,840 | config.json |
| `rope_type` | "default"（标准 RoPE） | config.json |
| `moe_router_use_sigmoid` | true | config.json |
| `moe_router_enable_expert_bias` | true | config.json |
| `router_scaling_factor` | 2.826 | config.json |
| `route_norm` | true | config.json |
| `qk_norm` | true | config.json |
| `rms_norm_eps` | 1e-5 | config.json |
| `hidden_act` | "silu"（SwiGLU） | config.json |
| `tie_word_embeddings` | false（不共享） | config.json |

### 2.2 GQA 配置

![GQA 头配置](figures/fig-3.2-gqa-config.svg)

Hy3 采用标准的 GQA（Grouped Query Attention），64 个 Query 头对应 8 个 KV 头，压缩比为 **8:1**。这是最常见的 GQA 配置——与 Llama 3（8 KV 头）一致，与 DeepSeek V4 的 MQA（1 KV 头）形成对比。GQA 8:1 在推理显存和质量之间取得了广泛认可的平衡：相比 MHA 节省 8 倍的 KV cache，但相比 MQA 保留了多个 KV 头以维持不同注意力模式的表达力。

### 2.3 参数分解与自洽验证

以下分解逐一验证各项之和是否收敛到官方宣称的 295B 总参。

**Embedding 层**（不共享）：
- 输入 Embedding：$V \times d = 120832 \times 4096 = 494.93 \times 10^6 \approx 495\text{M}$
- LM Head（不共享）：$d \times V = 4096 \times 120832 = 494.93 \times 10^6 \approx 495\text{M}$

**Attention（每层）**：
- Q 投影：$h_q \times d_{\text{head}} \times d = 64 \times 128 \times 4096 = 33.55\text{M}$
- K 投影：$h_{\text{kv}} \times d_{\text{head}} \times d = 8 \times 128 \times 4096 = 4.19\text{M}$
- V 投影：$h_{\text{kv}} \times d_{\text{head}} \times d = 8 \times 128 \times 4096 = 4.19\text{M}$
- O 投影：$h_q \times d_{\text{head}} \times d = 64 \times 128 \times 4096 = 33.55\text{M}$
- 单层 Attention 合计：$75.50\text{M}$
- Attention 总计（80 层）：$80 \times 75.50\text{M} = 6.04\text{B}$

**Dense FFN 层（第 0 层）**：
- Dense FFN SwiGLU 三层投影：$3 \times d \times d_{\text{ff\_dense}} = 3 \times 4096 \times 13312 = 163.58\text{M}$

**MoE 层（第 1-79 层，共 79 层）**：
- 路由 Gate：$E \times d = 192 \times 4096 = 0.79\text{M}$
- 192 个路由专家（每个为 SwiGLU）：
  - 每专家 gate_up_proj：$2 \times d_{\text{ff\_expert}} \times d = 2 \times 1536 \times 4096 = 12.58\text{M}$
  - 每专家 down_proj：$d \times d_{\text{ff\_expert}} = 4096 \times 1536 = 6.29\text{M}$
  - 每专家合计：$18.87\text{M}$
  - 192 个专家合计：$192 \times 18.87\text{M} = 3.62\text{B}$
- 1 个共享专家（SwiGLU，d_ff=1536）：$3 \times d \times d_{\text{ff\_expert}} = 3 \times 4096 \times 1536 = 18.87\text{M}$
- 单层 MoE 合计：$\approx 3.64\text{B}$
- MoE 总计（79 层）：$79 \times 3.64\text{B} = 287.84\text{B}$

**RMSNorm（80 层 x 2 个 Norm + 1 个 final Norm）**：参数极少，约 $3 \times 4096 = 12\text{K}$，可忽略。

**参数汇总**：

$$
\begin{aligned}
\text{Total} &= \underbrace{495\text{M}}_{\text{Embedding}} + \underbrace{495\text{M}}_{\text{LM Head}} + \underbrace{6.04\text{B}}_{\text{Attention}} + \underbrace{0.16\text{B}}_{\text{Dense FFN (L0)}} + \underbrace{287.84\text{B}}_{\text{MoE (L1-79)}} \\
&\approx 295.0\text{B}
\end{aligned}
$$

与官方宣称的 295B 一致。

**MTP 层参数**：官方 README 标注 MTP 层参数为 3.8B[^5]，由一个独立 Embedding + 1 层 Decoder（含 Attention + MoE）+ 共享 LM Head 组成。

**活跃参数（per token）**：
- Attention（1 层）：75.50M
- MoE 激活（1 层）：$k$ 个路由专家 + 1 个共享专家 = $(8+1) \times 18.87\text{M} = 169.83\text{M}$
- 加上 Gate + Norm：$169.83\text{M} + 0.79\text{M} \approx 170.6\text{M}$
- 总计（80 层 Attention + 79 层 MoE + 1 层 Dense）：$80 \times 75.50\text{M} + 79 \times 170.6\text{M} + 163.58\text{M} + 495\text{M} + 495\text{M} \approx 20.6\text{B}$
- 与官方 21B 活跃参数一致（MTP 推理时通常不激活，差额来自 Embedding/LM Head 等旁路）。

[^5]: 来源：HF README Model Introduction 表格，"MTP Layer Parameters: 3.8B"。

---

## CH 3: 计算与性能分析

### 3.1 Prefill FLOPs 分解

Prefill 阶段一次处理整个 prompt 序列，分为 Attention 和 FFN 两部分。

**Attention Prefill FLOPs**：

标准因果 Attention 的 FLOPs 公式为：

$$
F_{\text{attn}}(S, d) = 2 \cdot \frac{S(S+1)}{2} \cdot d \approx S^2 \cdot d
$$

其中因子 2 对应乘法+加法，$S(S+1)/2$ 是因果 mask 下非零注意力对数，d 是每对的点积维度。

代入 $S=262144$（256K）、$d=4096$：

$$
F_{\text{attn}}^{\text{single\_layer}} \approx (2.62 \times 10^5)^2 \times 4096 \approx 2.81 \times 10^{14} \text{ FLOPs}
$$

$$
F_{\text{attn}}^{\text{80\_layers}} = 80 \times 2.81 \times 10^{14} \approx 2.25 \times 10^{16} \text{ FLOPs}
$$

**MoE Prefill FLOPs**（79 层 MoE，每 token 激活 k=8 个路由专家 + 1 个共享专家）：

每 token 每个专家 SwiGLU FLOPs：

$$
F_{\text{expert}} = 3 \times 2 \times d \times d_{\text{ff\_expert}} = 6 \times 4096 \times 1536 \approx 3.77 \times 10^7 \text{ FLOPs}
$$

（其中因子 3 对应 gate/up/down 三个矩阵乘，因子 2 对应乘加）

$$
\begin{aligned}
F_{\text{MoE}}^{\text{single\_layer}} &= S \times (k+1) \times 6 \times d \times d_{\text{ff\_expert}} \\
&= 262144 \times 9 \times 6 \times 4096 \times 1536 \\
&\approx 8.90 \times 10^{13} \text{ FLOPs}
\end{aligned}
$$

$$
F_{\text{MoE}}^{\text{79\_layers}} = 79 \times 8.90 \times 10^{13} \approx 7.03 \times 10^{15} \text{ FLOPs}
$$

加上第 0 层 Dense FFN（$S \times 6 \times d \times d_{\text{ff\_dense}} = 262144 \times 6 \times 4096 \times 13312 \approx 8.58 \times 10^{13}$ FLOPs），以及 QKV/O 投影：

$$
F_{\text{proj}} = 80 \times S \times 2 \times d \times d_{\text{head}} \times (h_q + h_{kv} + h_{kv} + h_q) \approx 80 \times 262144 \times 2 \times 4096 \times 128 \times 144 \approx 3.17 \times 10^{15} \text{ FLOPs}
$$

（注：每层 4 个投影矩阵——Q($h_q$ 头)、K($h_{kv}$ 头)、V($h_{kv}$ 头)、O($h_q$ 头)——各需 $2 \times S \times d \times h_{\text{heads}} \times d_{\text{head}}$ FLOPs）

**Prefill 总 FLOPs（256K 序列）**：

$$
F_{\text{prefill}}^{\text{256K}} \approx 2.25 \times 10^{16} + 7.03 \times 10^{15} + 3.17 \times 10^{15} + 8.58 \times 10^{13} \approx 3.27 \times 10^{16} \text{ FLOPs}
$$

**Prefill 分段估算**（短 prompt 场景）：常见部署中 prompt 长度通常远小于 256K。以 S=4096 为例：

$$
F_{\text{attn}}^{4096} \approx 4096^2 \times 4096 \times 80 \approx 5.50 \times 10^{12} \text{ FLOPs}
$$

对 S=4096，Attention FLOPs 仅占 256K 场景的约 $1/4000$（因为 $S^2$ 二次项主导）。此时 MoE FLOPs（与 S 成线性正比）成为 prefill 的主要计算瓶颈。

### 3.2 Decode FLOPs（per-token）

Decode 阶段每生成一个新 token，需要计算该 token 对所有历史 KV 的 Attention + 路由到 k 个专家。

**Attention Decode FLOPs（per token）**：

$$
F_{\text{attn}}^{\text{decode}} = 2 \times S \times d = 2 \times 262144 \times 4096 \approx 2.15 \times 10^9 \text{ FLOPs/layer}
$$

80 层合计：$80 \times 2.15 \times 10^9 \approx 1.72 \times 10^{11}$ FLOPs/token。

**MoE Decode FLOPs（per token）**：

$$
F_{\text{MoE}}^{\text{decode}} = (k+1) \times 6 \times d \times d_{\text{ff\_expert}} = 9 \times 6 \times 4096 \times 1536 \approx 3.40 \times 10^8 \text{ FLOPs/layer}
$$

79 层 MoE 合计：$79 \times 3.40 \times 10^8 \approx 2.68 \times 10^{10}$ FLOPs/token。

加上第 0 层 Dense FFN（$6 \times 4096 \times 13312 \approx 3.27 \times 10^8$）和 QKV/O 投影（约 $2 \times 4 \times 4096 \times (64+16) \times 128 \times 80 \approx 2.68 \times 10^{10}$ FLOPs/token）。

**Decoder per-token 总计**：

$$
F_{\text{decode}} \approx 1.72 \times 10^{11} + 2.68 \times 10^{10} + 2.68 \times 10^{10} \approx 2.26 \times 10^{11} \text{ FLOPs/token}
$$

其中 Attention 占比约 76%，这意味着 **256K 场景下 decode 的绝对瓶颈在 Attention**——Hy3 没有采用稀疏注意力或 KV 压缩技术（如 CSA/HCA），标准 O(S) Attention 在长上下文下计算成本随序列长度线性增长。

### 3.3 Attention vs MoE 计算量对比

在长上下文（S=256K）decode 场景下，单 token Attention FLOPs vs MoE FLOPs 的对比直观说明了 Hy3 的计算特征：

| 阶段 | Attention（80 层） | MoE（79 层） | 比例 |
|---|---|---|---|
| Prefill (S=4096) | 5.50e12 | 1.11e14 | 1 : 20 |
| Prefill (S=256K) | 2.25e16 | 7.03e15 | 3.2 : 1 |
| Decode (per token) | 1.72e11 | 2.68e10 | 6.4 : 1 |

**关键观察**：
- **短 prompt prefill**：MoE 主导（S 小，Attention $S^2$ 项未起量）。
- **长 prompt prefill**：Attention 主导（$S^2$ 项指数增长）。
- **Decode**：Attention 始终主导（与 S 成正比而非 $S^2$，但 S=256K 时系数已很大）。

这一特征解释了 Hy3 为何需要 256K 上下文支持但未采用稀疏注意力——在典型生产部署中（prompt 长度 4K-32K），Attention 并非唯一瓶颈；而当 prompt 增长至 256K 时，预填充阶段的 Attention 计算已成为单次请求的最大开销。设计意图待确认：混元团队可能认为长上下文场景占比不高，且通过 MTP 推测解码可以有效摊薄 decode 阶段的 per-token 延迟。

### 3.4 KV Cache 估算

Hy3 采用标准 MHA/GQA 风格的 KV cache（无 MLA 潜空间压缩），每个 Decoder 层需缓存 $h_{kv} \times d_{\text{head}}$ 维的 K 和 V 各一份。GQA 8:1 将 KV 头数从 64 压缩至 8，cache 节省 8 倍。

$$
\text{KV\_Cache\_per\_layer} = 2 \times T \times h_{kv} \times d_{\text{head}} \times \text{bytes}
$$

代入 $T=262144$、$h_{kv}=8$、$d_{\text{head}}=128$、$\text{bytes}=2$（BF16）：

$$
\text{KV\_Cache\_per\_layer} = 2 \times 262144 \times 8 \times 128 \times 2 = 1.07 \times 10^9 \text{ bytes} = 1.07 \text{ GB}
$$

80 层总计：

$$
\text{KV\_Cache\_total} = 80 \times 1.07 \text{ GB} = 85.9 \text{ GB}
$$

**对比**：若使用 MHA（64 KV 头），总 cache 为 $85.9 \times 8 = 687 \text{ GB}$。GQA 8:1 将 256K 上下文的 KV cache 从不可部署（687 GB 单卡不可行）压至 86 GB——配合 8 卡张量并行，单卡约 10.7 GB。

### 3.5 推理显存预算

以 256K 上下文、单 batch、BF16 精度为例：

| 显存项目 | 大小 | 说明 |
|---|---|---|
| 模型权重 | 295B x 2 ≈ 590 GB | BF16 全精度加载 |
| KV Cache | 85.9 GB | 256K x 80 层 GQA |
| 激活值（峰值） | 约 8-16 GB | Prefill 阶段，与 batch size 有关 |
| 其他开销 | 约 5-10 GB | CUDA context, workspace 等 |
| **总计** | **约 700 GB** | 需 8 x H100-80GB 或更高容量 GPU |

**权重 I/O 分析（decode 阶段）**：

Decode 阶段每生成一个 token，需要从显存读取的权重为：活跃专家权重 + Attention 权重 + 共享组件。Hy3 每 token 激活 21B 参数 ≈ 42 GB（BF16）。作为对比，DeepSeek V4-Flash（284B 总参 / 13B 激活）每 token 读取约 26 GB。Hy3 的权重 I/O 更大，主要因为激活 8 个路由专家（vs V4-Flash 的 6 个），且隐藏维度虽然更窄但专家中间维也更大（1536 vs 2048 的对比需结合具体实现）。

---

## CH 4: 核心机制深析

### 4.1 Sigmoid 路由 vs Softmax：MoE 路由的范式差异

![MoE 路由拓扑](figures/fig-3.1-moe-routing.svg)

Hy3 路由器最显著的特征是使用 **Sigmoid** 而非 Softmax 作为专家评分函数。这是 MoE 路由设计的两个范式分水岭。

**Sigmoid 路由**（Hy3）的形式化：

对每个 token 的 hidden state $x \in \mathbb{R}^d$，路由权重矩阵 $W \in \mathbb{R}^{E \times d}$，每个专家的独立分数为：

$$
g_i(x) = \sigma(W_i \cdot x) = \frac{1}{1 + e^{-W_i \cdot x}}
$$

其中 $\sigma$ 是逐元素的 Sigmoid 函数，各专家的打分互相独立——专家 A 的高分不会压低专家 B 的分数。源码确认（`router.py` L31-L32）：

```python
router_logits = F.linear(hidden_states.float(), self.weight.float())
routing_weights = torch.sigmoid(router_logits)
```

**Softmax 路由**（传统方案，如 DeepSeek V3 早期版本）的形式化：

$$
g_i(x) = \frac{e^{W_i \cdot x}}{\sum_{j=1}^{E} e^{W_j \cdot x}}
$$

Softmax 将所有专家的分数归一化为概率分布，总分恒为 1——这意味着"提一个专家必然压另一个"。

**Sigmoid 路由的三个核心优势**：

1. **无需辅助负载均衡损失**：Softmax 路由天然倾向于将所有概率集中在少数专家上，需要额外的 aux loss（$\lambda \cdot \sum_i f_i \cdot p_i$）强制"拉平"专家负载。Sigmoid 独立打分 + `e_score_correction_bias` 的组合实现了**无 aux loss 的负载均衡**（§4.2 详述）。源码 `causal_lm.py` L43 明确返回 `"aux_loss": None`。

2. **允许多专家高分共存**：Sigmoid 打分下，一个 token 可以同时对多个专家打出 0.95 的高分——这在数学、编程等交叉领域 token 上至关重要。Softmax 下"总分为 1"的约束意味着一个 0.95 分的专家必然将其他 191 个专家的总分压到 0.05。

3. **训练动力学更稳定**：Sigmoid 的梯度 $\sigma(x)(1-\sigma(x))$ 在 logits 较大时平滑衰减，避免了 Softmax 的"赢者通吃"式梯度集中。

**Hy3 路由的完整流程**（源码 `router.py` L35-L43）：

1. 计算 `routing_weights = sigmoid(W @ x)`（每个专家独立打分）
2. 加入可学习偏置：`scores_for_choice = routing_weights + e_score_correction_bias`
3. Top-k 选择：`top_k_index = topk(scores_for_choice, k=8)`
4. 归一化：`top_k_weights = sigmoid_i / sum_j(sigmoid_j)`（仅对选中的 8 个归一化）
5. 缩放：`top_k_weights *= router_scaling_factor`（$2.826$）  
![Sigmoid 路由器](figures/fig-3.3-sigmoid-router.svg)

其中第 2 步的 `e_score_correction_bias` 是关键设计——它是一个可训练的 per-expert 偏置（shape `[192]`），仅用于 top-k 选择（不参与第 4 步的权重计算），在训练中根据专家命中率动态调整（命中率过高则减小偏置，过低则增大）。这实现了**无辅助损失的负载均衡**。

**`router_scaling_factor=2.826` 的设计意图**：归一化后的八选一权重平均为 0.125，乘以 2.826 后约为 0.353——这意味着 routed 加权和在未归一化意义上约为 2.826，与共享专家输出（幅度约 1）形成合理的量级关系。设计意图待确认：2.826 的具体取值可能来自训练初期对路由输出幅度的统计校准。

### 4.2 QK-Norm：稳定深层 GQA 训练

Hy3 在 Attention 计算中引入了 QK-Norm——在施加 RoPE 之前，对 Query 和 Key 分别进行 RMSNorm 归一化。这是源码 `attention.py` L35-L36 的明确实现：

```python
query_states = self.q_norm(query_states)   # RMSNorm per head
key_states   = self.k_norm(key_states)     # RMSNorm per head
```

**设计动机**：当模型深度达到 80 层且隐藏维度仅 4096 时，Attention logits（$QK^T / \sqrt{d_{\text{head}}}$）的数值分布容易随深度漂移。具体而言：
- 深层 Block 的 hidden state 范数可能逐渐增大（残差累积效应），导致 Q 和 K 的 L2 范数增长，进而使 $QK^T$ 的内积值超出 softmax 的敏感区间。
- 80 层串联残差使这一问题比 40-60 层的模型严重得多。

QK-Norm 在每层 Attention 计算前将 Q 和 K 的 per-head 范数重新校准到单位量级，确保 $QK^T / \sqrt{d_{\text{head}}}$ 始终落在 softmax 的有效梯度范围内。这与 DeepSeek V3/V4 系列中的 QK-Norm 设计属于同一思路——在极深 Transformer 中，归一化 Query 和 Key 是稳定训练的刚需而非可选项。

### 4.3 MTP：Multi-Token Prediction 加速推理

Hy3 配置了 1 层 MTP（Multi-Token Prediction），总参数 3.8B[^5]。MTP 的核心思想是在主模型输出 hidden state 的基础上，附加一个独立的 Transformer 层来预测"下一个 token 之后的一个 token"。

**推理时的 MTP 用法**（推测解码）：主模型每次前向产生一个 token，MTP 层同时产生 1-2 个候选 draft token（`num_speculative_tokens=2`，来自官方 vLLM 部署配置），由主模型在下一轮前向中对 draft token 做一次性验证。如果 draft token 正确，则本轮实际产生 2-3 个 token（1 个主模型 + 1-2 个验证通过），推理吞吐量可提升 1.5-2 倍。

**训练时的 MTP 角色**：MTP 层在训练时承担两个角色。第一，作为额外的预测头参与损失计算，通过"提前看 1 个 token"的监督信号提升中间层表征质量（类似 Auxiliary Loss 的思路）。第二，MTP 层的参数在推理时可直接用于推测解码，无需额外的 draft model。

Hy3 选择 1 层（而非 DeepSeek V3 的 1 层 + V4 的 1 层）是合理的平衡：每增加一层 MTP 增加约 3.8B 参数和对应的训练开销，但推测解码的收益受限于 draft token 的接受率——1 层 MTP 的接受率通常在 80-90%，追加更多层收益递减。

### 4.4 层数 vs 隐藏维的工程 trade-off：深层窄维的代价

CH1 已指出 Hy3 的核心设计哲学是"窄维深层"。本节补充这一 trade-off 的具体工程代价。

**优势侧**：
- KV cache 与 d 成正比：4096 的 KV cache 是 8192 的 50%（同样 GQA 配置下）
- Attention 计算量 $O(S \times d)$ 同样减半
- 权重矩阵参数量与 d 成正比：单层 Attention Q 投影从 $(64 \times 128 \times 8192) = 67.1\text{M}$ 降至 $(64 \times 128 \times 4096) = 33.6\text{M}$

**代价侧**：
- 层数 80 使 prefill 延迟（Attention $S^2$ 项）乘以 80 而非 40-60
- 80 层串联残差要求更精细的归一化（QK-Norm 是必要条件）
- 单层 FFN 容量受限：中间维 13312（仅 3.25x 扩张比），相比之下 Llama 3 405B 的 14336（也是约 3.5x），DeepSeek V3 的 18432（2.57x）。窄维下需要更多专家（192）来补偿知识容量

这是 Hy3 最具辨识度的架构决策——直接决定了所有下游计算特征。

---

## CH 5: 训练体系

**本章标注：官方未公开详细训练数据（数据配比、loss 曲线、消融实验、硬件消耗、优化器超参等），以下分析仅基于 README 公开信息与 config.json 推导。**

### 5.1 RL 框架重建

根据官方 README，姚顺雨领衔重建了训练与 RL 基础设施。这一重建的直接证据来自 REAMDE 中两处描述：(1) "Following the Hy3 Preview launch in late April, we gathered feedback from 50+ product teams"——表明 Preview 版已上线并接受广泛测试；(2) "scaling up RL training"——表明 RL 是后训练阶段的核心投入方向。

同时，Hy3 在 Agent 任务（SWE-Bench Verified, tool calling）上的显著提升暗示其 RL 框架支持了工具调用、代码执行等多模态反馈信号的训练。README 中提到"Tool-call success rates and error recovery improved"和"accuracy variance across scaffoldings...remains within 4%"，表明 RL 管线在跨框架泛化性上做了专项优化。

### 5.2 MoE 路由的稳定性特征

Hy3 使用 Sigmoid 路由 + `e_score_correction_bias` 实现无辅助损失的负载均衡（§4.1）。这一设计隐含了两个训练稳定性假设：

1. **Sigmoid 评分天然避免"赢者通吃"**：不需要 Softmax 那样在 192 个专家之间竞争，训练早期的路由分布比 Softmax 更均匀。
2. **偏置更新独立于主损失梯度**：`e_score_correction_bias` 的更新不通过反向传播，而是基于统计式的"命中率高 → bias 减小，命中率低 → bias 增大"规则（由训练框架实现，不在开源推理代码中体现）。

这两个特征共同使 Hy3 在 80 层深度 + 192 个专家的组合下能够稳定训练——无需调 $\lambda$（aux loss 系数）这一敏感超参。

### 5.3 上下文长度支持

Hy3 使用标准 RoPE（`rope_type="default"`），但将 $\theta$ 设为 11,158,840（远大于 Llama 3 的 500,000）以原生支持 256K 上下文。大 $\theta$ 的作用是减缓高频分量的旋转速度，使远距离位置之间的内积衰减得更慢——等价于扩展了 RoPE 的有效上下文窗口。

标准 RoPE 对第 $i$ 对维度施加的旋转频率为 $\theta^{-2i/d_{\text{head}}}$（$i=0,1,\dots,d_{\text{head}}/2-1$）。最低频分量（$i=d_{\text{head}}/2-1$）的波长约为 $2\pi \times \theta \approx 7.0\times 10^7$ tokens——远超 256K，意味着所有维度在 256K 范围内都有可区分的旋转编码。

相比使用 YaRN 等外推技术从较短上下文扩展到 1M（如 DeepSeek V4 从 128K → 1M），原生大 $\theta$ 的优势在于训练和推理的 RoPE 行为完全一致，无需在推理时切换频率。

### 5.4 未公开内容与推测

以下信息在官方公开资料中缺失：
- 训练数据总量与配比（README 仅提到"improved post-training data quality and diversity"）
- 预训练阶段的优化器选择与超参（学习率、batch size、warmup 等）
- 并行策略（张量并行度、流水线并行度、ZeRO 阶段等）
- 硬件消耗（GPU 型号与数量、训练耗时）
- 预训练阶段的损失曲线与消融实验

上述信息标注"官方未公开"，不进行猜测性填充。

---

## CH 6: 总结

### 6.1 核心洞察

Hy3-295B 是 2026 年上半年最值得关注的"效率优先"型大规模 MoE 模型，其核心洞察可概括为一句话：**通过窄隐藏维（4096）+ 深层数（80）+ 多专家（192）+ Sigmoid 独立路由的组合，在 21B 活跃参数下实现对接近规模的旗舰模型的竞争力**。

具体架构贡献包括：
- **Sigmoid 路由器**摆脱了 Softmax 的"零和博弈"约束，配合可训练偏置实现无辅助损失的负载均衡，使路由训练更加稳定。
- **QK-Norm** 使 80 层深层 Transformer 在仅 4096 隐藏维的条件下训练不崩。
- **1 层 MTP** 提供了低成本（3.8B 参数，约 1.3% 总参）的推测解码能力。
- **GQA 8:1** 将 256K 上下文的 KV cache 压至约 86 GB——相比 MHA 节省 8 倍。

### 6.2 关键 trade-off 总结

1. **窄维 vs 深层**：d=4096 使单层容量受限，但 80 层提供足够的非线性变换深度来补偿。代价是 prefill 阶段 Attention FLOPs 随层数线性累积。
2. **Sigmoid 路由 vs 专家利用率**：Sigmoid 独立打分避免了 aux loss 对主损失的污染，但需要 `e_score_correction_bias` 单独维护负载均衡。偏置收敛较慢（通常需要数千步才稳定），且缺乏理论保证其必然收敛到均匀分布。
3. **多专家 vs 参数效率**：192 个专家（每个仅 18.87M，d_ff=1536）使总参数膨胀至 295B，但活跃参数仅 21B（7.1%）。这一"大库小取"策略以权重存储为代价换取推理效率——295B 权重的加载和分发是部署工程的主要挑战。
4. **标准 Attention vs 长上下文效率**：Hy3 未采用稀疏注意力或 KV 压缩技术，256K decode 阶段 Attention 占 per-token FLOPs 的 76%。在超长上下文场景下，这成为比 MoE 更显著的瓶颈。

### 6.3 与 DeepSeek V4-Flash 的定性对比

Hy3-295B 追求参数效率（"用更少的活跃参数做更多事"），DeepSeek V4-Flash（284B/13B 激活）追求极致长上下文能力（CSA+HCA 稀疏注意力 + 1M 上下文 + FP4 量化专家）——两者同为 2026 年上半年开源的高效 MoE 模型，前者以 Sigmoid 独立路由和深层窄维设计见长，后者以稀疏注意力和多通道残差创新见长。
