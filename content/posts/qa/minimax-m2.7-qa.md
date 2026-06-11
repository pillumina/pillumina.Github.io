+++
date = '2026-06-11'
draft = false
title = 'MiniMax-M2.7 架构 QA'
categories = ['qa']
tags = ['moe', 'attention', 'model-architecture', 'qa', 'minimax', 'gqa', 'mtp']
series = ['qa']
summary = '基于 M2.7 主报告的配套 QA。覆盖五代演进、Full Attention 回归、GQA+QK Norm、MoE 路由（sigmoid+routing bias）、MTP×3、训练体系等核心主题。'
+++

# MiniMax-M2.7 架构 QA

> 125 问，覆盖 CH1 预备知识 → CH2 MiniMax 演进 → CH3 M2.7 概览 → CH4 Full Attention + QK Norm → CH5 MoE 路由 → CH6 训练体系 + MTP → CH7 支撑项 → CH8 源码映射 → CH9 对比总结 → 面经高频

---

## CH 1. LLM 预备知识（18 Q）

### Q1.1 标准 Transformer 的 Self-Attention 公式是什么？

**简短回答**：Self-Attention 的核心是 $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，通过 Query-Key 点积计算 token 间的关系权重，对 Value 加权求和。

**详细解释**：

给定输入序列 $X \in \mathbb{R}^{T \times d}$，通过三个投影矩阵获得 Query、Key、Value：

$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$

其中 $W_q, W_k \in \mathbb{R}^{d \times d_k}$，$W_v \in \mathbb{R}^{d \times d_v}$。注意力计算分为三步：

1. **相似度计算**：$S = QK^T$，得到 $T \times T$ 的注意力分数矩阵
2. **缩放 + Softmax**：$A = \text{softmax}(S / \sqrt{d_k})$，$\sqrt{d_k}$ 防止点积过大导致梯度消失
3. **加权求和**：$O = AV$，每个位置的输出是所有位置 Value 的加权组合

时间复杂度为 $O(T^2 d)$——序列长度 $T$ 的平方是长文本场景的核心瓶颈（即"$O(n^2)$ 灾难"）。

**面试要点**：$\sqrt{d_k}$ 缩放是必须的，不是可选的——当 $d_k$ 很大时，$QK^T$ 的点积值会很大，softmax 会趋向 one-hot，梯度接近零。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.2 MHA / MQA / GQA 三者有什么区别？

**简短回答**：MHA 每个 Q 头有独立 KV 头，MQA 所有 Q 头共享 1 个 KV 头，GQA 是折中——N 个 Q 头共享 M 个 KV 头（M < N）。

**详细解释**：

| 机制 | Q 头数 | KV 头数 | KV Cache 大小 | 注意力质量 |
|------|--------|---------|---------------|-----------|
| **MHA** | H | H | H × T × d | 最高 |
| **GQA** | H | G (< H) | G × T × d | 高，接近 MHA |
| **MQA** | H | 1 | 1 × T × d | 降低 |

MHA 的 KV cache 随头数线性增长，在长序列推理时成为内存瓶颈。MQA 将 KV cache 压缩到 1/H，但注意力质量下降明显（尤其在检索任务上）。GQA 是工程上的 sweet spot——M2.7 使用 48 Q 头 + 8 KV 头（GQA ratio = 6），KV cache 仅为 MHA 的 17%。

**易混淆**：MQA 不是 MHA 的缩写变体，而是 Multi-Query Attention——所有 Q 头共享唯一的 K 和 V。

**延伸阅读**：主报告 CH 3.2 / GQA 原论文 Ainslie et al., 2023

---

### Q1.3 Mixture of Experts (MoE) 的核心原理是什么？

**简短回答**：MoE 将 FFN 层拆分为 N 个"专家"子网络，每个 token 只激活 top-k 个专家，实现"总参数很大、激活参数很小"的稀疏计算。

**详细解释**：

标准 Dense 模型中，每个 token 经过相同的 FFN 层。MoE 将 FFN 替换为：

1. **Router（门控网络）**：$g(x) = \text{softmax}(xW_g)$，输出 N 维概率分布
2. **Top-k 选择**：选分数最高的 k 个专家，其余专家权重置零
3. **稀疏激活**：token 只通过被选中的 k 个专家计算
4. **加权合并**：最终输出 = $\sum_{i \in \text{top-k}} g_i(x) \cdot \text{Expert}_i(x)$

M2.7 配置：256 个专家，每 token 激活 8 个（k=8），激活参数占总参的 ~4.3%（9.8B / 229.9B）。关键设计：**无共享专家**（`shared_intermediate_size = 0`），所有输出完全依赖 routed expert。

**面试要点**：MoE 的核心优势是"算力换容量"——用 Dense 模型 1/5 的 FLOPs 获得更大参数量的表达力。代价是 expert load balancing 和通信开销。

**延伸阅读**：主报告 CH 5 / DeepSeek-V2/V3 MoE 设计

---

### Q1.4 SwiGLU 激活函数是什么？相比 ReLU 有什么优势？

**简短回答**：SwiGLU = Swish + GLU（Gated Linear Unit），公式为 $\text{SwiGLU}(x) = (xW_1 \odot \text{SiLU}(xW_2))W_3$，通过门控机制让网络学会选择性传递信息。

**详细解释**：

SwiGLU 的完整计算公式（M2.7 源码对应 `MiniMaxM2MLP`）：

$$\text{output} = \text{SiLU}(xW_1) \odot (xW_2) \cdot W_3$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（也叫 Swish），$\odot$ 是逐元素乘法。

三个权重矩阵的作用：
- $W_1$（gate 投影）：生成门控信号（经 SiLU 激活）
- $W_2$（up 投影）：生成待门控的值
- $W_3$（down 投影）：将中间维度投影回 hidden_size

M2.7 的 `intermediate_size = 1536`（$W_1, W_2$ 的输出维度），`hidden_size = 3072`（$W_3$ 的输出维度）。

优势：(1) 门控机制提供更强的非线性表达；(2) SiLU 的平滑梯度优于 ReLU 的硬截断；(3) 比标准 FFN（ReLU + 2 矩阵）多一个可学习的"信息闸门"。

**易混淆**：SwiGLU 的参数量是标准 FFN 的 3/2 倍（3 个权重矩阵 vs 2 个），所以很多模型会相应减小 `intermediate_size` 来保持总参不变。M2.7 的 `intermediate_size = 1536` 是 `hidden_size = 3072` 的 50%，体现了"激进小 expert"设计。

**延伸阅读**：主报告 CH 2.3 / Shazeer, 2020 "GLU Variants Improve Transformer"

---

### Q1.5 RMSNorm 与 LayerNorm 的区别是什么？

**简短回答**：RMSNorm 是 LayerNorm 的简化版，只做均方根归一化，去掉减均值和偏置项，计算更快且效果相当。

**详细解释**：

| 对比维度 | LayerNorm | RMSNorm |
|----------|-----------|---------|
| 公式 | $y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$ | $y = \gamma \cdot \frac{x}{\text{RMS}(x)}$ |
| 去均值 | 是 | **否** |
| 偏置参数 | $\beta$ | 无 |
| 计算量 | 更高 | 更低（约少 15%） |

RMSNorm 的公式：

$$\text{RMSNorm}(x) = x \cdot \frac{\gamma}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$$

M2.7 在所有 normalization 位置均使用 RMSNorm（`rms_norm_eps = 1e-6`），包括 Pre-Norm、QK Norm 等。这是现代 LLM（Llama、Mistral、DeepSeek）的主流选择。

**面试要点**：为什么去掉均值仍然有效？因为 Transformer 的残差连接已经提供了"重新居中"的能力，RMSNorm 只控制尺度已足够。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.6 RoPE 的核心原理是什么？

**简短回答**：RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息注入 Q 和 K，使 attention 分数只依赖 token 间的相对位置。

**详细解释**：

RoPE 对 Q/K 向量按维度对分组，施加二维旋转：

$$f_{\{q,k\}}(x_m, m) = R^d_{\Theta, m} \cdot x_m$$

其中 $R^d_{\Theta, m}$ 是块对角旋转矩阵，旋转角 $\theta_i = \text{base}^{-2i/d}$（i 为维度对索引）。

关键性质：
- **相对位置**：$\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m-n)$，只依赖位置差 $m-n$
- **远程衰减**：随着相对距离增加，attention 分数自然衰减
- **外推灵活**：可通过调整 $\theta$（如 NTK-aware scaling）扩展上下文

M2.7 的 RoPE 配置：`rope_theta = 5,000,000`（Llama-3 的 10 倍），`rotary_dim = 64`（partial RoPE，仅前 64 维参与旋转），为 192K 上下文提供更细粒度的远程位置编码。

**面试要点**：RoPE 的核心优势是"只改 Q/K、不改模型结构"，且相对位置编码天然适合长度外推。

**延伸阅读**：主报告 CH 2.1 / Su et al., 2021 "RoFormer"

---

### Q1.7 什么是 KV Cache？为什么它对推理至关重要？

**简短回答**：KV Cache 是自回归推理时缓存已计算 Key/Value 的机制，避免每一步重复计算历史 token 的 K/V，将单步推理复杂度从 $O(T^2)$ 降到 $O(T)$。

**详细解释**：

自回归生成时，每生成一个新 token，需要它对所有历史 token 做 attention。如果没有 KV Cache：
- Step 1：计算 $t_1$ 的 $Q_1, K_1, V_1$
- Step 2：重新计算 $t_1, t_2$ 的 $K, V$，再算 attention
- Step i：重新计算全部 $i$ 个 token 的 $K, V$

有 KV Cache 后，每步只计算新 token 的 $Q_{new}, K_{new}, V_{new}$，将 $K_{new}, V_{new}$ 追加到缓存中，复杂度从 $O(T^2)$ 降至 $O(T)$。

内存开销：对于 L 层、H 个 KV 头、序列长度 T、头维度 d，KV Cache 占 $2 \times L \times H \times T \times d$ 个元素。GQA 通过减少 KV 头数（M2.7: 8 vs 48）将 KV cache 压缩至 MHA 的 17%。

**易混淆**：KV Cache 只节省计算，不节省内存——长序列下 KV Cache 的大小可能超过模型权重本身。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.8 $O(n^2)$ 灾难是什么？有哪些缓解方案？

**简短回答**：$O(n^2)$ 灾难指 Self-Attention 的计算和内存开销随序列长度 T 平方增长，当 T > 8K 时成为不可忽视的瓶颈。

**详细解释**：

计算量的具体来源：
- Attention 矩阵 $QK^T$：$T \times T$，$T^2 d$ FLOPs
- Attention 输出 $AV$：$T \times T \times d$，$T^2 d$ FLOPs
- 内存占用：$T \times T$ 的 attention 矩阵（FP16 下 T=192K 时需 ~73GB）

缓解方案分类：

| 类别 | 代表方案 | 复杂度 | 代价 |
|------|----------|--------|------|
| 线性注意力 | Lightning Attention, Mamba | $O(Td^2)$ | 表达能力下降 |
| 稀疏注意力 | SWA, BigBird | $O(T \cdot w)$ | 丢失远距离依赖 |
| 近似注意力 | FlashAttention（IO-aware） | $O(T^2 d)$ 但 IO 优化 | 不改变复杂度 |
| KV Cache 压缩 | GQA, MQA | 减少内存 | 质量轻微下降 |

M2.7 继承自 M2 的选择：**接受 $O(T^2)$ 复杂度**，M2 已回归 Full Attention，因为 Agent 场景的质量需求压倒效率考量。

**面试要点**：这是 LLM 基础知识，面试中常作为热身题出现。确保能用 1-2 句话清晰解释核心概念和关键数字。

**延伸阅读**：主报告 CH 3.1 / FlashAttention 原论文 Dao et al., 2022

---

### Q1.9 什么是 Pre-Norm？与 Post-Norm 有何区别？

**简短回答**：Pre-Norm 将 Normalization 放在子层（Attention/FFN）之前，Post-Norm 放在之后。Pre-Norm 训练更稳定（不需要 warm-up），是现代 LLM 的主流选择。

**详细解释**：

```text
Post-Norm:  x → SubLayer(x) → Norm(x + SubLayer(x)) → 下一层
Pre-Norm:   x → SubLayer(Norm(x)) + x → 下一层
```

差异对比：

| 维度 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 梯度流动 | 残差分支需经过 Norm | 残差分支直接传递 |
| 训练稳定性 | 需要 LR warm-up | 更稳定，无需 warm-up |
| 最终质量 | 理论上略优 | 实际相当 |
| 代表模型 | 原始 Transformer, BERT | Llama, GPT-3+, M2.7 |

M2.7 的每个 Decoder Layer 使用双重 Pre-Norm：
1. Pre-Norm (RMSNorm) → Full Attention (GQA + QK Norm + RoPE)
2. Residual add → Pre-Norm (RMSNorm) → MoE FFN → Residual add

**面试要点**：原始论文用 Post-Norm，但 Pre-Norm 在 2019 年后几乎成为所有大模型的默认选择。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.10 FP16 / BF16 / FP8 三种精度有什么区别？

**简短回答**：FP16 精度范围窄（max=65504），BF16 范围等于 FP32（max=3.4e38）但精度低，FP8 进一步压缩为 8 位用于推理加速。

**详细解释**：

| 格式 | 总位数 | 指数位 | 尾数位 | 最大值 | 精度 |
|------|--------|--------|--------|--------|------|
| FP32 | 32 | 8 | 23 | 3.4e38 | 最高 |
| FP16 | 16 | 5 | 10 | 65504 | 中（范围受限） |
| BF16 | 16 | **8** | 7 | 3.4e38 | 中（精度略低） |
| FP8 E4M3 | 8 | 4 | 3 | 448 | 低 |

关键洞察：
- **BF16 > FP16**：因为指数位数相同（8 位），BF16 的动态范围与 FP32 一致，训练时不会发生 FP16 的 overflow/underflow 问题。这就是为什么几乎所有大模型训练都用 BF16。
- **FP8 E4M3**：M2.7 使用的量化格式（`float8_e4m3fn`），4 位指数 + 3 位尾数，需要 block-wise quantization（每 128×128 块独立 scale）来保持精度。

M2.7 策略：训练/推理主精度 `bfloat16`，权重存储使用 FP8（`weight_block_size: [128,128]`），对精度敏感的 `gate`、`e_score_correction_bias`、`lm_head` 保持 BF16。

**面试要点**：BF16 是 Google Brain 提出的格式，核心思想是"大模型训练更需要动态范围而非尾数精度"。

**延伸阅读**：主报告 CH 7.2 / config.json → quantization_config

---

### Q1.11 Decoder-Only 架构有什么特点？为什么现代 LLM 都用它？

**简短回答**：Decoder-Only 架构只用 Transformer 的 Decoder 部分（去掉 Encoder 和 Cross-Attention），通过因果掩码（causal mask）确保每个 token 只能看到其之前的 token，天然适配自回归语言建模。

**详细解释**：

架构对比：
- **Encoder-Decoder**（如 T5）：Encoder 双向编码输入，Decoder 自回归生成输出，Cross-Attention 连接二者
- **Encoder-Only**（如 BERT）：双向注意力，适合理解任务，不适合生成
- **Decoder-Only**（如 GPT、Llama、M2.7）：自回归 + 因果掩码，一阶段训练，统一表示

Decoder-Only 主导的原因：
1. **统一范式**：理解和生成用同一个模型，无需 Encoder-Decoder 桥接
2. **扩展性好**：参数量和时间复杂度都随层数线性扩展
3. **零样本能力强**：in-context learning 在 Decoder-Only 上表现最优
4. **基础设施成熟**：KV Cache, FlashAttention, GQA 等优化高度适配

M2.7 的 62 层全部是标准的 Decoder Layer：Pre-Norm → Full Attention（causal mask）→ Pre-Norm → MoE FFN。

**面试要点**：面试常问"为什么不用 Encoder-Decoder"——Decoder-Only 的因果注意力天然适合自回归生成，且所有层参数共享，训练效率更高。

**延伸阅读**：主报告 CH 2.2

---

### Q1.12 因果掩码（Causal Mask）是如何工作的？

**简短回答**：因果掩码是一个上三角为 $-\infty$ 的 $T \times T$ 矩阵，叠加到 attention 分数上，使位置 $i$ 只能 attend 位置 $j \leq i$，确保自回归的因果性。

**详细解释**：

```text
Causal Mask (T=4):
[[  0,  -∞,  -∞,  -∞],
 [  0,   0,  -∞,  -∞],
 [  0,   0,   0,  -∞],
 [  0,   0,   0,   0]]
```

计算流程：$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{mask}\right)$

softmax 将 $-\infty$ 映射为 0，因此 token i 对 token j（j > i）的 attention 权重为 0。

训练时，causal mask 对全部 token 并行计算（teacher forcing）；推理时，每次只输入最后一个 token，但通过 KV Cache 访问完整历史。

**易混淆**：Causal Mask 不等同于 Padding Mask——前者保证因果性，后者处理批次中不同长度序列的填充部分。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.13 Residual Connection（残差连接）在 Transformer 中的作用是什么？

**简短回答**：残差连接（$y = x + F(x)$）为梯度提供了"高速公路"，解决了深层网络的梯度消失问题，使得 62 层的 M2.7 可以稳定训练。

**详细解释**：

没有残差连接时，反向传播梯度为 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial F(x)}{\partial x}$，经过多层连乘后容易趋近 0。

有残差连接时，梯度为 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 + \frac{\partial F(x)}{\partial x})$，恒等项让梯度可以无损回传。

M2.7 的每层有两个残差路径：
```text
x → RMSNorm → Full Attention → + x (残差1)
  → RMSNorm → MoE FFN → + x (残差2)
```

两个残差连接各自独立，确保 gradient flow 不会在 attention 或 FFN 子层中被阻断。

**面试要点**：残差连接的"梯度高速公路"比喻——没有残差时梯度需穿过所有非线性层（容易衰减），有残差时梯度可以"跳过"非线性层直接传播。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.14 Transformer 中的 FFN 为什么不可或缺？

**简短回答**：Attention 负责 token 间的"信息路由"（线性混合），FFN 负责"信息变换"（非线性记忆存储），二者互补。去掉 FFN 等于让模型失去记忆能力。

**详细解释**：

Attention 是线性操作（对 V 的加权求和），其非线性仅来自 softmax。FFN 提供：
1. **非线性变换**：通过激活函数（SwiGLU/ReLU）引入非线性
2. **知识存储**：研究表明 FFN 矩阵存储了大量事实和概念（可视为 key-value memory）
3. **维度扩展**：FFN 将 hidden_size 扩展到 4×（或 SwiGLU 的 8/3×），提供更大的容量

MoE 将 FFN 拆分为 256 个 expert，本质上是一种"条件计算"——不同 token 使用不同的 FFN 子网络。

**面试要点**：有论文证明，去掉 FFN 的 Transformer 等价于多步 attention 堆叠，表达能力从图灵完备降级为线性映射链。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.15 什么是 Teacher Forcing？为什么训练比推理快？

**简短回答**：Teacher Forcing 是自回归模型的训练方式——使用真实历史 token（而非模型自己生成的）作为下一步输入，使得所有位置的 token 可以一次性并行计算。

**详细解释**：

训练时（Teacher Forcing）：
```text
输入：[t1, t2, t3, ..., tT-1]
目标：[t2, t3, t4, ..., tT]
一次性计算所有 T 个位置的 loss（并行）
```

推理时（自回归）：
```text
Step 1: 输入 [t1] → 输出 t2
Step 2: 输入 [t1, t2] → 输出 t3  ← 必须等 Step 1 完成
...
Step N: 输入 [t1, ..., tN] → 输出 t{N+1}
必须串行 N 步（不可并行）
```

训练与推理的性能差异：
- 训练：$O(T^2 d)$ 但 T 的维度高度并行
- 推理：$O(T)$ 步串行，每步 $O(Td)$（有 KV Cache）
- 典型比率：训练 1 个 step 的时间 << 推理生成 100 个 token 的时间

M2.7 的 MTP（Multi-Token Prediction）通过同时预测未来 3 个 token 来增加训练信号密度，间接加快训练收敛。

**面试要点**：Teacher Forcing 的关键 insight——训练时用真实历史 token（完美信息），推理时用自己生成的 token（可能有错误），这种 mismatch 称为 exposure bias。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.16 什么是 Perplexity（困惑度）？它是如何衡量模型质量的？

**简短回答**：Perplexity = $\exp(-\frac{1}{N}\sum_{i=1}^N \log P(t_i|t_{<i}))$，衡量模型对文本的"惊讶程度"，越低越好（模型越不"困惑"）。

**详细解释**：

Perplexity 的直观理解：
- PPL = 10：模型平均在 10 个选项中猜测下一个 token（相当于均匀分到 10 个候选）
- PPL = 1：模型对每一步都 100% 确定（完美预测）

数学上，PPL 是交叉熵损失的指数映射：$\text{PPL} = e^{\text{CE Loss}}$。

局限性：
- PPL 是"平均"指标，可能被高频 token 的准确预测掩盖长尾 token 的失败
- 不直接衡量生成质量（连贯性、事实性、安全性）
- 不同 tokenizer 的 PPL 不可直接比较（词表大小影响 PPL 基准）

在 M2.7 的论文中，pre-training 阶段 PPL 是辅助指标，核心对比依赖 downstream benchmark（RULER、MTOB、LongBench）。

**易混淆**：PPL 越低越好，但不同 tokenizer 的 PPL 不可直接比较（词表大小影响 PPL）。同 tokenizer 下的 PPL 对比才有意义。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.17 Scaling Law 的基本思想是什么？

**简短回答**：Scaling Law 揭示了模型性能（loss）与参数量 N、数据量 D、计算量 C 之间满足幂律关系，指导"在给定计算预算下如何分配 N 和 D"。

**详细解释**：

Chinchilla Scaling Law（Hoffmann et al., 2022）的核心结论：

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0$$

其中 $\alpha \approx 0.34$，$\beta \approx 0.28$，最优分配为 token 数 $\approx 20 \times$ 参数量。

MoE 模型的 Scaling Law 有额外的自由度：总参数 vs 激活参数。M2.7 的设计（229.9B 总参 / 9.8B 激活）就是在总计预算约束下，用 256 个 expert 增大总容量，同时保持激活参数（推理成本）可控。

**面试要点**：Scaling Law 不是自然的物理定律，而是经验观察。不同架构（Dense vs MoE）、不同数据质量下的指数系数不同。

**延伸阅读**：主报告 CH 1（LLM预备知识）；附录 glossary

---

### Q1.18 什么是 Speculative Decoding（推测解码）？

**简短回答**：用一个小模型（draft model）快速生成多个候选 token，再由大模型（target model）一次性验证，实现不损失质量的推理加速。

**详细解释**：

流程：
1. Draft model 自回归生成 $k$ 个候选 token
2. Target model 一次性对所有候选 token 计算概率
3. 逐个验证：接受概率匹配的 token，拒绝不匹配的
4. 从被拒绝的位置重新采样

M2.7 利用 MTP（Multi-Token Prediction）模块作为 draft model——MTP 原本是训练辅助，推理时可以零成本复用为 speculative decoding 的 draft head。M2.7 有 3 个 MTP 模块，每个预测 1 个未来 token，提供 $k=3$ 的推测深度。

优势：在长序列场景下，Full Attention 推理慢的问题可通过 speculative decoding 部分缓解。

**面试要点**：投机解码的核心 trade-off——draft model 太小→接受率低→加速有限；draft model 太大→draft 开销大→加速也有限。MTP 头作为内置 draft model 是最优平衡。

**延伸阅读**：主报告 CH 6 / Leviathan et al., 2023

---

## CH 2. MiniMax 演进（12 Q）

### Q2.1 MiniMax 经历了哪五次关键架构迭代？

**简短回答**：Text-01 (2025-01) → M1 (2025) → M2 (2025-11) → M2.5 → M2.7 (2026-03)。M2 是关键转折——业已回归 Full Attention + GQA + 256 MoE。M2.7 的创新在自我进化范式，而非架构变更。

**详细解释**：

| 代际 | 时间 | 核心架构 | 参数量 | 关键变化 |
|------|------|----------|--------|----------|
| **Text-01** | 2025-01 | Dense, 混合注意力 (Lightning + Full), MHA | 456B | 验证 Lightning Attention 可行性 |
| **M1** | 2025 | 纯 Lightning Attention, Dense | — | 首次开源大规模 Lightning 模型 |
| **M2** | 2025-11 | MoE (256 experts), GQA (48/8), **Full Attention** | 229.9B total / ~10B active | **回归 Full Attention**（放弃 Lightning） |
| **M2.5** | 2025 H2 | 与 M2 相同架构 | 同 M2 | Agent 专项增强（工具调用 + 长链推理） |
| **M2.7** | 2026-03 | MoE, GQA, Full Attention, MTP×3, QK Norm | 229.9B total / **9.8B** active | 自我进化范式, per_layer QK Norm, MTP×3 |

M2 团队在 M2 研发期间探索了多种混合注意力方案（intra-layer hybrid SWA + Full Attention），全部以失败告终。论文标题「No Free Lunch」即是对此的总结。M2.7 相对 M2.5 的总参不变，激活参数略降，核心变化在训练范式和自我进化能力。

**面试要点**：456B Dense → 229.9B MoE 不是"缩小"，而是"聪明地缩放"——总参减少但通过 MoE 稀疏激活实现了更高效的容量利用。**M2 而非 M2.7 做出了回归 Full Attention 的决策。**

**延伸阅读**：主报告 CH 1.1

---

### Q2.2 Text-01 的混合注意力架构是如何设计的？

**简短回答**：Text-01 在 456B 总参的 Dense 架构中，交替堆叠 Lightning Attention 层和 Full Attention 层，第一次在大规模模型中验证了"部分层用高效注意力"的可行性。

**详细解释**：

Text-01 的层结构（示意）：
```text
Layer 0: Lightning Attention (O(Td²))
Layer 1: Full Attention (O(T²d))
Layer 2: Lightning Attention
Layer 3: Full Attention
...
```

设计动机：
- Full Attention 层：长程依赖的"锚点"，提供高质量位置交互
- Lightning Attention 层：降低总计算量，使 456B 模型可训练

这个混合设计的实验结果直接影响了 M2 → M2.7 的架构演进。M2 在训练期间尝试了 intra-layer hybrid SWA + Full Attention 混合，多个变体全部失败——不仅是 benchmark 上 MMLU/BBH 表面持平，更致命的是复杂多跳推理和 Agent 任务全部拉胯。M2 因此做出了「回归 Full Attention」的决策。M2.7 继承此决策，未改变注意力架构。

**面试要点**：混合注意力（Lightning + Full）是 Text-01 的核心创新——用线性注意力处理长程依赖（O(n)），用 Full Attention 保证局部精度（O(n²)仅在短窗口）。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.3 M2 相比 Text-01 做了哪些关键改变？

**简短回答**：三大改变——(1) Dense → MoE（256 experts），总参从 456B 降至 229.9B；(2) MHA → GQA（48Q/8KV）；(3) 保持混合注意力但引入 expert 细粒度。

**详细解释**：

M2 的架构升级逻辑：
1. **MoE 化**：将 Dense FFN 替换为 256 个细粒度 expert，top-8 激活。通过稀疏性在降低总参的同时保持甚至提升容量。
2. **GQA 引入**：48 个 Q 头共享 8 个 KV 头（ratio=6），在保持 KV cache 可控（17% of MHA）的前提下不损失太多注意力质量。
3. **延续混合注意力**：M2 保留了 Lightning + Full Attention 交替堆叠，尚未做出"全 Full Attention"的决策。

M2 论文的核心贡献是验证了「Mini Activations」原则——用更窄的隐藏维度（3072）搭配更多层（62），再用 MoE 扩展 FFN 容量。

**面试要点**：理解这一设计选择背后的 trade-off——为什么选这个参数值而非其他？从计算预算、内存限制、训练稳定性等角度思考。

**延伸阅读**：主报告 CH 1.1

---

### Q2.4 M2.5 与 M2 的差异是什么？

**简短回答**：M2.5 的架构与 M2 完全相同，差异在于训练策略——专门针对 Agent 场景（工具调用、长链推理、多步规划）进行了 SFT 和 RL 优化。

**详细解释**：

M2.5 可视为 M2 的"Agent 特化版本"：
- 架构零变化：同样的 62 层、256 expert、48/8 GQA、混合注意力
- 训练数据偏向：增加工具调用轨迹、多步推理链、代码执行反馈
- 评测重心转移：从通用 benchmark 转向 Agent benchmark（SWE-bench、WebArena 等）

M2.5 的存在说明一个重要事实：**同架构下，训练策略的差异足以产生一个独立的"模型版本"**。这为 M2.7 的"自我进化"范式埋下伏笔——如果训练策略可以自主迭代，模型本身就参与了自身的版本升级。

**面试要点**：理解这一设计选择背后的 trade-off——为什么选这个参数值而非其他？从计算预算、内存限制、训练稳定性等角度思考。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.5 M2.7 相对 M2/M2.5 最核心的架构变化是什么？

**简短回答**：M2 业已使用 Full Attention，M2.7 继承此架构。M2.7 的核心变化是 per_layer QK Norm、MTP×3 模块、自我进化训练流程——架构与 M2/M2.5 相同，创新在训练范式。

**详细解释**：

M2.7 相对 M2/M2.5 的变化（详细讨论见 CH 3-6）：

| 变化 | M2/M2.5 | M2.7 | 来源 |
|------|---------|------|------|
| Attention 类型 | Full Attention（M2 已回归） | Full Attention（同 M2） | config.json: `attn_type_list=[1]×62` |
| QK Norm | 无 | `per_layer` RMSNorm | config.json: `use_qk_norm=true` |
| MTP | 无 | 3 模块 × 1 层 | config.json: `num_mtp_modules=3` |
| 训练范式 | 传统 SFT + RL | **自我进化** (100+ 轮) | paper §3 |
| 路由评分 | softmax (推测) | **sigmoid** | config.json: `scoring_func=sigmoid` |
| 共享专家 | 可能有 | **无** | config.json: `shared_intermediate_size=0` |

其他参数（62 层、3072 hidden、256 experts、48/8 GQA）保持不变。

**面试要点**：M2.7 的总参和激活参与 M2/M2.5 相同，但架构变更是"质变"——这证明"同参数下的训练范式优化"可以独立驱动模型提升。**关键澄清：M2 而非 M2.7 做出了回归 Full Attention 的决策。**

**延伸阅读**：主报告 CH 1.2、CH 3

---

### Q2.6 为什么 M2 放弃了 Lightning Attention？其决策依据是什么？

**简短回答**：论文消融实验（Table 2）显示，Lightning Attention（/混合）在长上下文检索任务上比 Full Attention 差 15-18 pp（RULER 128K CWE: 72.0 vs 90.0），这个差距在 Agent 场景（需要从长历史精确检索）中不可接受。M2 团队因此在 M2 研发期间即做出「回归 Full Attention」的决策，论文标题「No Free Lunch」即是对此的总结。M2.7 继承了这一决策。**错误认知**：不是 M2.7 放弃了 Lightning Attention——M2 已经放弃了。

**详细解释**：

论文 §2.2.2 的原话给出了决策逻辑：

> "Despite the theoretical appeal of efficient attention mechanisms, we found no variant that reliably matches full attention quality in production."

消融数据（pre-training 阶段）：

| 任务 | Full Attention | SWA (混合) | 差距 |
|------|:---:|:---:|:---:|
| RULER 128K CWE | 90.0 | 72.0 | **-18 pp** |
| MTOB K-e Bleurt | 60.0 | 45.0 | **-15 pp** |
| MTOB e-k ChrF | 44.8 | 27.2 | **-17.6 pp** |
| RULER 128K Multi-Query | 99.0 | 93.0 | -6 pp |
| LongBench | 59.2 | 56.1 | -3.1 pp |
| RULER 4K | 97.0 | 94.0 | -3.0 pp |
| MMLU | 85.5 | 85.6 | +0.1 pp |
| MATH | 60.3 | 60.3 | 0 |

关键发现：
- 短上下文任务差距微小（MMLU/MATH 几乎持平）
- 长上下文检索差距巨大（CWE -18pp, MTOB -15pp 至 -18pp）
- SFT 后差距缩小但仍显著（RULER 128K CWE: 84 vs 72, 仍差 12pp）

Agent 场景必须从 128K+ 的上下文窗口中精确检索信息——15-18 pp 的差距意味着在 100 次检索中会多出错 15-18 次。

**面试要点**：Lightning Attention 被放弃不是因为它"不好"，而是 Full Attention 在 M2.7 的目标场景（Agent）中更可靠。这是"场景决定架构"的典型案例。

**延伸阅读**：主报告 CH 3.2 / paper Table 2 & Table 3

---

### Q2.7 自我进化范式具体是什么？与架构有何关系？

**简短回答**：自我进化是 M2.7 深度参与自身训练的范式——模型自主运行 100+ 轮「分析—改进—验证」循环，自主调整架构超参、采样参数和工作流策略。

**详细解释**：

自我进化的 100+ 轮循环：
```
For round in 1..100+:
  1. 模型分析当前训练状态和评测结果
  2. 模型提出改进方案（超参调整 / 架构修改 / scaffold 更新）
  3. 自动部署改进并重新训练
  4. 验证改进效果
  5. 记录分析日志，进入下一轮
```

直接产出的三个架构决策：
1. **QK Norm 的引入**（CH 4）：模型发现 Q/K 的数值不稳定导致 attention logit 在长序列上爆炸，自主提出 per_layer QK Norm
2. **Routing Bias 动态调整策略**（CH 5）：从固定 bias 进化为基于 expert 命中率动态调整
3. **256 专家容量分配优化**：自动寻找最优的 expert capacity 设置

**面试要点**：这标志着从"人调架构"到"模型参与调架构"的范式转变。M2.7 是首批公开记录此范式的大规模模型之一。

**延伸阅读**：主报告 CH 7.3 / paper §3

---

### Q2.8 从 456B（Text-01）到 229.9B（M2）总参下降，为什么能力反而提升？

**简短回答**：MoE 架构用 256 个 expert 提供的"稀疏容量"替代了 Dense 模型的"密集容量"——229.9B 总参中只有 9.8B 参与每次前向，但 256 个 expert 的专业化分工让每个 token 享受了更细粒度的建模。

**详细解释**：

类比：Dense 456B 是一个"全科医生"——对所有 token 用同一套参数。MoE 229.9B 是"256 个专科医生"——每个 token 只咨询最相关的 8 个专家。

MoE 效率的来源：
- **专家专业化**：不同 expert 自然学习到不同领域（代码、数学、常识、多语言等）
- **稀疏激活**：用 ~4.3% 的 FLOPs（9.8B vs 229.9B）获得接近全参数量的表达能力
- **梯度隔离**：每个 expert 只被 top-k token 更新，减少无关 token 的梯度干扰

Scaling Law 视角：在相同 FLOPs 预算下，MoE 229.9B 比 Dense 的等效 FLOPs 模型有更多"有效参数"。

**面试要点**：参数减少但能力提升 = MoE 的胜利。256 专家的稀疏容量比 456B Dense 参数更高效——"容量密度"而非"容量总量"决定能力。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.9 GQA 是在 M2 才引入的吗？

**简短回答**：是的，Text-01 使用标准 MHA，M2 首次引入 GQA（48 Q 头 / 8 KV 头，ratio = 6），M2.5 和 M2.7 继承了此配置。

**详细解释**：

MiniMax 的 attention 设计演进：
```text
Text-01: MHA (48 Q heads, 48 KV heads) — KV cache 100%
M2:      GQA (48 Q heads, 8 KV heads)  — KV cache 17%
M2.5:    GQA (48 Q heads, 8 KV heads)  — 同 M2
M2.7:    GQA (48 Q heads, 8 KV heads)  — 同 M2, 新增 QK Norm
```

M2 引入 GQA 的动机：MHA 在 128K+ 上下文下 KV cache 开销过大（48 个 KV 头 × 128K tokens × 128 dim = ~786M 元素/层，FP16 下 ~1.5GB/层）。GQA 将 KV 头压缩到 8，内存降至 ~262M 元素/层。

ratio = 6 的选择：48/8 = 6 个 Q 头共享 1 对 KV 头，这是一个经过验证的平衡点——ratio 太小（如 2-3）KV cache 节省有限，ratio 太大（如 48 = MQA）注意力质量下降明显。

**面试要点**：GQA 在 M2 才引入说明 MiniMax 系列在逐步现代化——Text-01 作为第一代实验架构使用标准 MHA，M2 开始采用工程上更高效的 GQA。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.10 MiniMax 演进中哪些参数始终保持不变？

**简短回答**：自 M2 起，`num_hidden_layers=62`、`hidden_size=3072`、`num_attention_heads=48`、`num_key_value_heads=8`、`num_local_experts=256`、`num_experts_per_tok=8` 保持恒定。

**详细解释**：

这些"不变"反映了 MiniMax 团队对「更深更窄」设计哲学的坚持：

| 参数 | 恒值 | 对比参考 |
|------|------|----------|
| 层数 | 62 | V4-Flash: 43（更深） |
| 隐藏维度 | 3072 | V4-Flash: 4096（更窄） |
| Q 头数 | 48 | — |
| KV 头数 | 8 | V4-Flash: 1（更保守的 GQA） |
| 专家数 | 256 | V4-Flash: 256（相同） |
| Top-k | 8 | V4-Flash: 6 + 1 shared（不同策略） |

换代的改进来自"架构质量"（Full Attention > 混合、QK Norm、MTP）和"训练质量"（自我进化、routing bias），而非"增大规模"。

**面试要点**：稳定不变的参数（num_layers=62, hidden=3072, heads=48）说明 MiniMax 认为这些值已达最优——架构演变的重点在"可变组件"（注意力类型、MoE配置）。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.11 M2.7 的 MoE 专家数与 Text-01 有何关系？

**简短回答**：Text-01 是 Dense 模型（无 MoE，0 个 expert），M2 首次引入 256 个 expert。Text-01 整个模型就是一个"大 expert"，M2 将其拆分为 256 个专业化子网络。

**详细解释**：

从 Dense 到 MoE 的拆分：
```text
Text-01:
  FFN (每个 token 经过同一个)
  ↓ 拆分
M2/M2.5/M2.7:
  FFN → Router → 256 experts, 每个 token 选 top-8
```

拆分的好处：
- 同一 token "咨询"多个 expert 的组合，获得多视角的信息变换
- 不同 token 走不同 expert 子集，实现 token 级别的条件计算
- expert 专业化减少了参数间的干扰

代价：
- Load balancing（专家负载均衡）：某些 expert 可能过载（被选太多），某些闲置
- 通信开销（分布式训练时 expert 跨 GPU 通信）
- 路由崩溃：router 可能坍缩到少数几个 expert

M2.7 通过 sigmoid routing + routing bias（aux-loss-free）来解决负载均衡问题。

**面试要点**：Text-01 没有 MoE（纯 Dense），M2 首次引入 256 专家——这是 MiniMax 系列最重大的架构跃迁（Dense→MoE）。

**延伸阅读**：主报告 CH 1（MiniMax演进）；config-formatted.json

---

### Q2.12 如何一句话总结 MiniMax 演进的主线？

**简短回答**：**注意力在 M2 回归"满"，在 M2.7 加"精"**——M2 从 Lightning 回归 Full Attention，M2.7 用自我进化追求 Agent 可靠性。

**详细解释**：

演进主线三阶段：
1. **Text-01/M1**：探索 Lightning Attention 的可能性与边界（"探路"）
2. **M2**：发现 Lightning 在长上下文检索上的系统性弱点，回归 Full Attention（"回归"）
3. **M2.5/M2.7**：在 Full Attention 基础上叠加 Agent 能力和自我进化（"精进"）

这是一个"先探索替代方案，发现问题后回归主流，再在此基础上创新"的认知曲线——M2 的「No Free Lunch」结论是整个演进的分水岭。

**面试要点**：M2.7 的演进不是"打脸"之前的自己，而是 M2 团队充分探索了高效注意力的边界后，有意识地选择回归 Full Attention。

**延伸阅读**：主报告 CH 1

---

## CH 3. M2.7 概览（18 Q）

### Q3.1 "229.9B 总参 / 9.8B 激活"分别意味着什么？

**简短回答**：229.9B 是所有参数的总数（存储在硬盘/显存），9.8B 是每个 token 在前向传播中实际使用的参数量。前者决定容量，后者决定推理成本。

**详细解释**：

```text
总参数 229.9B = Embedding + 62×(Attention + 256×Expert) + LM Head + MTP
                 └─ 其中 256 个 Expert 占据参数的绝大部分
                 
激活参数 9.8B ≈ Embedding + 62×(Attention + 8×Expert) + LM Head
                 └─ 每个 token 只经过 8/256 = 3.125% 的 Expert
```

费用含义：
- **训练成本**（硬件需求）：由 229.9B 总参决定（需要足够显存放所有权重 + 优化器状态）
- **推理成本**（每 token 延迟）：主要由 9.8B 激活参决定（每次只计算激活的部分）
- **吞吐量**：9.8B 激活 ≈ 常规 Dense 10B 模型的推理速度

MoE 的"免费午餐"：训练了 229.9B 的模型，但推理只需 ~9.8B 的 FLOPs。

**易混淆**：激活参数 =/= 推理时加载的参数。推理时仍需加载全部 229.9B 权重到显存，只是计算（FLOPs）只用 9.8B 的激活部分。

**延伸阅读**：主报告 CH 2.1 / 附录 B.4

---

### Q3.2 62 层 × 3072 的"更深更窄"设计有何利弊？

**简短回答**：相比 V4-Flash 的 43 层 × 4096，"更深更窄"让每层做更少的事、但通过更多层获得更强的表示层次，代价是串行深度可能导致梯度传播和推理延迟增加。

**详细解释**：

| 对比维度 | M2.7 (深窄) | V4-Flash (浅宽) |
|----------|:---:|:---:|
| 层数 × 维度 | 62 × 3072 | 43 × 4096 |
| 每层参数量 | 更低 | 更高 |
| 表示层次 | 更丰富 | 较平坦 |
| 推理串行步数 | 62 | 43 |
| 单层计算量 | 更低 | 更高 |

"更深更窄"与 M2 论文强调的"mini activations"原则一致——隐藏状态维度小（3072），中间表示的显存占用低，允许在同样显存预算下放更多层。

M2.7 额外受益于深度：QK Norm 逐层学习不同的 hidden state 分布，62 层的"深度多样性"让 per_layer 设计更有价值。

**面试要点**：深度的选择与 Attention 类型有关——Full Attention 在深层下更容易出现 logit 爆炸，所以 M2.7 需要 QK Norm 来配合深度。

**延伸阅读**：主报告 CH 2.3

---

### Q3.3 GQA 48/8 的 ratio = 6 是如何确定的？

**简短回答**：48 个 Q 头，8 个 KV 头，ratio = 48/8 = 6。这是一个经验平衡点——KV cache 降至 MHA 的 17%，同时保持足够的注意力头多样性。

**详细解释**：

源码中的 `repeat_kv` 逻辑（`modeling_minimax_m2.py`）：
```python
# K, V: (B, 8, T, 128)  → repeat → (B, 48, T, 128)
# 每个 KV 头复制 6 次，匹配 48 个 Q 头
key_states = key_states.repeat_interleave(48 // 8, dim=1)
value_states = value_states.repeat_interleave(48 // 8, dim=1)
```

Q 头分组：Q 头 0-5 → KV 头 0，Q 头 6-11 → KV 头 1，...，Q 头 42-47 → KV 头 7。

为什么 ratio ≠ 48（MQA）也不= 1（MHA）：
- MQA（ratio=48）：KV 头太少，检索类任务受影响（V4-Flash 用 MQA 但配合稀疏注意力缓解）
- GQA ratio=6：每个 KV 头有 6 个 Q 头从不同角度 query，保留了较好的注意力多样性
- MHA（ratio=1）：KV cache 太大，128K 上下文下不可行

**面试要点**：GQA ratio 的选择是基于 KV cache 预算——ratio=6 时 KV 头=8，在 192K 上下文下 KV cache 约 49GB（单卡可承受），ratio=8 可进一步节省但注意力质量可能下降。

**延伸阅读**：主报告 CH 4.1

---

### Q3.4 256 个 expert、每 token 激活 8 个（k=8）的设计逻辑是什么？

**简短回答**：256 experts 提供足够的专家容量和专业化空间，k=8 确保每个 token 获得多个视角的信息变换（8 vs 常见的 2-6），同时 activation ratio = 8/256 = 3.125% 保持推理成本可控。

**详细解释**：

与业界对比：

| 模型 | 专家数 | Top-k | 激活比 |
|------|:---:|:---:|:---:|
| Mixtral 8×7B | 8 | 2 | 25% |
| DeepSeek V2 | 160 | 6 | 3.75% |
| DeepSeek V3 | 256 | 8+1 shared | ~3.5% |
| **M2.7** | **256** | **8** | **3.125%** |
| V4-Flash | 256 | 6+1 shared | ~2.7% |

M2.7 的 k=8 且无共享专家意味着：
- 每个 token 的输出完全由 8 个"纯 routed"专家加权组合
- 没有共享专家的"安全网"，对 routing bias 的负载均衡要求更高
- activation ratio 3.125% 与 DeepSeek V3 相近（9.8B/229.9B ≈ 4.3%，含 attention 等非 expert 参数）

**面试要点**：为什么选 k=8 而不是 2 或 4？更大的 k 让 token 拥有更多"专家视角"，特别适合需要多领域知识的 Agent 任务。代价是推理 FLOPs 线性增加。

**延伸阅读**：主报告 CH 5

---

### Q3.5 M2.7 为什么选择"无共享专家"？

**简短回答**：`shared_intermediate_size = 0` 意味着 M2.7 不设共享专家，所有 token 的输出完全依赖 top-8 routed expert。设计理念是通过 256 个 expert 的细粒度和 routing bias 机制来覆盖通用能力，而非依赖一个专门的共享专家。

**详细解释**：

| 设计选择 | M2.7 | V4-Flash | DeepSeek V3 |
|----------|:---:|:---:|:---:|
| 共享专家 | 无 | 1 个 | 1 个 |
| Routed 专家 | 256 | 256 | 256 |
| Top-k | 8 | 6 | 8 |
| 负载均衡 | routing bias (aux-loss-free) | routing bias | aux-loss-free bias |

有共享专家的优势：每个 token 至少经过共享专家的"通识教育"，router 可以更"任性"选择专才。

无共享专家的优势：
- 模型更"纯粹"——没有"万能兜底"，迫使 256 个 expert 都具备一定通用能力
- 减少参数——无共享专家节省了 ~0.5-1B 参数
- routing bias 机制有能力维持负载均衡（命中率低的 expert 获得 bias 加成）

这是一种"倒逼专业化"的设计哲学。

**面试要点**：无共享专家是 M2.7 的独特选择——大部分 MoE 模型（DeepSeek V3、Qwen3.5）使用共享专家提供"基础能力"。M2.7 的设计理念是"让路由完全决定信息流"。

**延伸阅读**：主报告 CH 5.4

---

### Q3.6 Sigmoid 路由与 Softmax 路由有什么本质区别？

**简短回答**：Softmax 使 expert 分数相互竞争（概率和为 1），Sigmoid 使各 expert 分数独立（各自在 0-1 之间）。Sigmoid 允许多个 expert 同时获得高分，更适合 top-k > 1 的场景。

**详细解释**：

```text
Softmax:  scores_i = exp(logit_i) / Σ_j exp(logit_j)  → 分数互相压制
Sigmoid:  scores_i = 1 / (1 + exp(-logit_i))          → 分数各自独立
```

M2.7 的 routing 流程：
```python
router_logits = self.gate(x)           # (B, T, 256)
routing_weights = F.sigmoid(router_logits)  # 独立评分
scores = routing_weights + self.e_score_correction_bias  # 加 bias 修正
topk_weights, topk_indices = torch.topk(scores, 8)  # 选 top-8
weights = topk_weights / topk_weights.sum(dim=-1)  # 仅归一化，不 exp
```

Sigmoid 的三个优势：
1. **不存在"winner-take-all"**：即使 expert A 的 logit 是 100，B 是 10，sigmoid 下都是 ~1，但 softmax 下 B 接近 0
2. **计算更快**：不需要算 256 个 exp（sigmoid 只需 1 个 exp per expert）
3. **更适配 routing bias**：bias 对 sigmoid 分数的调整更线性、更可预测

**面试要点**：Sigmoid 让 routing 从"竞品排名"变为"独立打分"，本质上是改变了 expert 之间的竞争关系。

**延伸阅读**：主报告 CH 5.3

---

### Q3.7 rope_theta = 5,000,000 的意义是什么？

**简短回答**：5M 是标准 Llama-3（500K）的 10 倍，为 RoPE 提供更细粒度的远程位置编码，使 attention 在 192K 上下文仍能区分远距离的位置差异。

**详细解释**：

RoPE 的旋转频率：$\theta_i = \text{base}^{-2i/d}$，其中 base = rope_theta。

- base 越大 → 旋转越慢 → 表示长距离位置差异的能力越强
- base 越小 → 旋转越快 → 近距离分辨率越高

对比不同模型的 rope_theta：

| 模型 | rope_theta | 目标上下文 | 策略 |
|------|:---:|:---:|------|
| Llama-3 | 500K | 8K | 标准 |
| Llama-3.1 | 500K → 50M (NTK) | 128K | 推理时缩放 |
| V4-Flash | 10K | 8K | 保守 |
| **M2.7** | **5M** | **192K** | **训练时固定大 base** |

M2.7 选择 5M 而非 Llama-3.1 的推理时 NTK 缩放（500K→50M），说明团队认为**在训练时就用大 base 比推理时外推更好**。

结合 `rotary_dim = 64`（partial RoPE），只有前 64/128 维参与旋转——这是一种"让部分维度保留绝对位置信息，部分维度注入位置信息"的折中。

**面试要点**：rope_theta 不是越大越好——过大的 base 会让近距离 token 的位置差异不够明显。

**延伸阅读**：主报告 CH 2.1、CH 7.1

---

### Q3.8 192K 上下文的含义和实现方式是什么？

**简短回答**：`max_position_embeddings = 204,800`，约 192K 有效上下文（因 causal mask 和位置编码的设计余量），能一次性处理超长文档。实现依赖：Full Attention + rope_theta=5M + partial RoPE + GQA。

**详细解释**：

192K 上下文的四个支撑：
1. **Full Attention**：没有滑动窗口或稀疏掩码，全局视野
2. **rope_theta = 5M**：足够慢的旋转频率，区分 192K 位置
3. **rotary_dim = 64**：partial RoPE 在位置编码和语义表示间平衡
4. **GQA (8 KV 头)**：在 192K 长度下 KV cache 仍可控

代价：Full Attention 在 192K 输入下产生 $192K^2 \approx 37B$ 的 attention 矩阵——这是 M2.7 推理速度慢的根本原因。实测 45.6 TPS vs 声称 100 TPS。

**易混淆**：`max_position_embeddings=204,800` 是上限，不等于"模型在 204K 上表现好"。实际上随着位置接近上限，RoPE 的远程衰减效应可能导致远端 token 的 attention 被压缩。

**延伸阅读**：主报告 CH 7.1

---

### Q3.9 MTP × 3 的含义是什么？与标准 next-token prediction 有何区别？

**简短回答**：MTP（Multi-Token Prediction）在标准 next-token 预测之外，增加 3 个辅助预测头，同时预测 token[n+1], token[n+2], token[n+3]，为训练提供更密集的信号。

**详细解释**：

标准训练：输入 $[t_1, ..., t_n]$，预测 $t_{n+1}$（1 个 loss）
MTP 训练：输入 $[t_1, ..., t_n]$，预测 $t_{n+1}$（主 loss）+ $t_{n+2}$ + $t_{n+3}$ + $t_{n+4}$（3 个辅助 loss）

M2.7 的 MTP 配置：
- `num_mtp_modules = 3`：3 个预测头
- `mtp_transformer_layers = 1`：每个 MTP 模块仅 1 层 Transformer（轻量）

推理时的双用途：
- 不启用 speculative decoding：MTP 模块不参与计算，仅用主 head
- 启用 speculative decoding：MTP 3 个模块作为 draft model，自回归生成 3 个候选 token，主模型一次性验证

与 DeepSeek V3 MTP 的差异：
- V3: 1 个 MTP 模块，2 层（1 main + 1 mtp）
- M2.7: 3 个 MTP 模块，各 1 层——"浅而多"的设计

**面试要点**：MTP 是"免费午餐"——训练时多 3 个 loss 几乎不增加主模型推理开销，却显著提升了每 step 的学习信号密度。

**延伸阅读**：主报告 CH 6

---

### Q3.10 M2.7 的 Decoder Layer 内部执行顺序是什么？

**简短回答**：Pre-RMSNorm → Full Attention (GQA + QK Norm + RoPE) → Residual Add → Pre-RMSNorm → MoE FFN (sigmoid routing + top-8) → Residual Add。

**详细解释**：

```text
输入 x (B, T, 3072)
│
├─ RMSNorm(x) → Q/K/V 投影
│   ├─ Q: Linear(3072→6144), reshape→(B,48,T,128), RoPE(64维), QK Norm
│   ├─ K: Linear(3072→1024), reshape→(B,8,T,128),  RoPE(64维), QK Norm
│   └─ V: Linear(3072→1024), reshape→(B,8,T,128)
│
├─ Full Attention: softmax(QK^T/√128 + causal_mask)V
│   └─ repeat_kv (8→48), O: Linear(6144→3072)
│
├─ + x  (残差 1)
│
├─ RMSNorm(x) → MoE 路由
│   ├─ gate: Linear(3072→256) → sigmoid + routing_bias → top-8
│   └─ 8 experts: SwiGLU(1536→3072) → 加权组合
│
└─ + x  (残差 2) → 输出
```

每个 layer 输出 $3072$ 维，经过 62 次相同的计算后进入 MTP 模块或 LM Head。

**面试要点**：Pre-Norm 顺序（Norm→Attn→+residual→Norm→FFN→+residual）是标准设计。面试中能画出 Decoder Layer 的执行顺序图是展示"理解深度"的好方法。

**延伸阅读**：主报告 CH 2（M2.7整体架构）；config-formatted.json

---

### Q3.11 M2.7 的词表大小为 200,064，这有什么特殊含义？

**简短回答**：200,064 = 200K + 64，远大于常见模型的 32K-128K。大词表提高了中文等多语言文本的压缩率，降低每 token 的编码成本。

**详细解释**：

词表大小对比：

| 模型 | 词表大小 | 含义 |
|------|:---:|------|
| GPT-2 | 50,257 | 英文为主 |
| Llama-3 | 128,256 | 多语言 |
| Qwen-2 | 152,064 | 中日韩优化 |
| **M2.7** | **200,064** | 超大多语言 |

大词表的利弊：
- 优势：相同文本用更少 token 表示 → 等效上下文更长、推理步数更少
- 劣势：embedding 矩阵参数更大（200,064 × 3072 ≈ 614M）、训练收敛略慢

M2.7 的 `tie_word_embeddings = False` 意味着输入 embedding 和输出 LM Head 不共享权重，embedding 总参更大但给予模型更多自由度。

**面试要点**：200064 这个数字接近 GPT-4 的词表大小（~100K 的两倍），说明 M2.7 针对多语言场景做了扩展。词表越大→每个 token 携带更多信息→相同上下文窗口覆盖更多内容。

**延伸阅读**：主报告 CH 2（M2.7整体架构）；config-formatted.json

---

### Q3.12 QK Norm 是 M2.7 独有的吗？在哪些模型中有类似设计？

**简短回答**：QK Norm 不是 M2.7 独有——Gemma-2、Cohere Command-R 等模型也使用了 QK Norm。但 M2.7 的 `per_layer` 设计（每层独立参数）和与 self-evolution 的结合是独特的。

**详细解释**：

QK Norm 的使用情况：

| 模型 | QK Norm | 类型 | 引入方式 |
|------|:---:|------|------|
| Gemma-2 | 有 | per_layer | 人工设计 |
| Command-R | 有 | per_layer | 人工设计 |
| **M2.7** | **有** | **per_layer** | **自我进化自主引入** |

M2.7 的特殊之处：
1. QK Norm 是模型在 100+ 轮自我进化中**自主提出的**，而非工程师预设的
2. `per_layer` 决定了每一层有独立的 RMSNorm 参数，每层独立适应不同的 hidden state 分布
3. 与 `rope_theta = 5M` + 192K 上下文组合：长序列下 QK 点积值域更容易偏移，QK Norm 直接抑制这个问题

**易混淆**：QK Norm 不是 MiniMax 独有——DeepSeek V2/V3 也使用 QK Norm（称为 QK LayerNorm）。两者都在 QK 点积前做归一化，目的相同（训练稳定+注意力分布均匀）。

**延伸阅读**：主报告 CH 4.2

---

### Q3.13 attn_type_list = [1] × 62 的含义是什么？

**简短回答**：`attn_type_list` 是一个长度为 62 的列表，全部为 1，表示所有 62 层都使用同一类型注意力（Full Attention = type 1）。这是 M2.7 与 M2/M2.5 最关键的区别之一。

**详细解释**：

在 M2/M2.5 中，`attn_type_list` 是混合的：
```text
M2/M2.5:       [1, 2, 1, 2, 1, 2, ...]   ← Full 和 Lightning 交替
M2.7:          [1, 1, 1, 1, 1, 1, ...]   ← 全部 Full
```

type 的含义（推测）：
- `1` = Full Attention（标准 softmax attention）
- `2` = Lightning Attention（线性注意力）

这个简单的配置变化蕴含了 M2 最核心的架构决策——"高效注意力的任何变体都无法在生产环境中可靠匹配 Full Attention 的质量"（paper §2.2.2）。M2.7 继承了这一决策。

**面试要点**：attn_type_list 为未来实验留下了灵活的接口——可以配置前密后疏或交替的注意力模式而无需改代码。这是"配置驱动架构"的工程实践。

**延伸阅读**：主报告 CH 2（M2.7整体架构）；config-formatted.json

---

### Q3.14 M2.7 的 RMSNorm epsilon 为 1e-6，这个值合理吗？

**简短回答**：合理。1e-6 是 Llama 系列的标准值，比原始 Transformer 的 1e-5 更小。较小的 epsilon 意味着在正常分布的 hidden state 下，归一化几乎只有"缩放"效果而不会"抬高"极小值。

**详细解释**：

epsilon 值对比：

| 模型/框架 | epsilon |
|-----------|:---:|
| 原始 Transformer | 1e-5 |
| Llama-2/3 | 1e-6 |
| Mistral | 1e-5 |
| **M2.7** | **1e-6** |

RMSNorm 中 epsilon 的位置：$y = x \cdot \frac{\gamma}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$

当 $\frac{1}{d}\sum x_i^2 \gg \epsilon$ 时，epsilon 几乎无影响。当 RMS 极小（如某些初始化状态）时，epsilon 防止除零。

较小的 epsilon 意味着更"纯粹"的 RMS 归一化，但对数值极端小的 hidden state 保护更弱。M2.7 的 QK Norm 额外提供了一层数值稳定性保障。

**面试要点**：理解这一设计选择背后的 trade-off——为什么选这个参数值而非其他？从计算预算、内存限制、训练稳定性等角度思考。

**延伸阅读**：主报告 CH 2（M2.7整体架构）；config-formatted.json

---

### Q3.15 为什么 M2.7 的 intermediate_size = 1536 只有 hidden_size = 3072 的一半？

**简短回答**：SwiGLU FFN 有 3 个权重矩阵（vs 标准 FFN 的 2 个），如果 intermediate_size 保持 4× 标准，参数量会膨胀到约 1.5 倍。1536（0.5×）是一种激进的压缩——每个 expert 更"小"，但通过 256 个 expert 的稀疏组合弥补。

**详细解释**：

参数量计算：
- 标准 FFN（3072 → 4×3072 → 3072）：$3072 \times 12288 \times 2 = 75.5M$ / layer
- SwiGLU FFN 4×（3072 → 4×3072, up + gate → 3072）：$3072 \times 12288 \times 3 \approx 113M$
- M2.7 SwiGLU（3072 → 1536, up + gate → 3072）：$3072 \times 1536 \times 3 \approx 14.2M$ **per expert**

每个 expert 只有 14.2M 参数（vs 常规的 ~113M），256 个 expert 总计 14.2M × 256 ≈ 3.63B FFN 参数。

这种"micro expert"设计：
- 优势：每个 expert 高度轻量，256 个 expert 提供细粒度的知识划分
- 风险：单个 expert 容量有限，必须靠 top-8 组合才能完成复杂的知识变换
- 配合 k=8：8 × 14.2M ≈ 113.6M——恰好等于一个标准 SwiGLU 的参数量

**面试要点**：M2.7 的每 token 实际使用的 FFN 参数 ≈ 113.6M，正好是一个标准 SwiGLU 的参数量。256 experts 只是将这些参数"分而治之"。

**延伸阅读**：主报告 CH 2.3

---

### Q3.16 modules_to_not_convert 中的 gate / e_score_correction_bias / lm_head 为什么排除在 FP8 之外？

**简短回答**：这三个模块对量化精度极度敏感——gate 直接影响 256 个 expert 的选择（错误放大到整个 MoE），bias 微小偏移改变路由分布，lm_head 直接影响输出 token 概率。保持 BF16 是"精度换稳定性"。

**详细解释**：

| 排除项 | 敏感原因 | 影响范围 |
|--------|----------|----------|
| `gate` | Router logits 的微小扰动 → top-8 选择变化 | 全部 62 层的 MoE 路由 |
| `e_score_correction_bias` | 256 维 bias 的每个值都影响 expert 负载 | 全局 load balancing |
| `lm_head` | 直接决定 200,064 维 token 概率 | 最终生成质量 |

排除这些模块的代价很小（它们的参数量极小），但收益巨大（避免路由错误和输出质量退化）。这是"选择性量化"的最佳实践。

**面试要点**：理解这一设计选择背后的 trade-off——为什么选这个参数值而非其他？从计算预算、内存限制、训练稳定性等角度思考。

**延伸阅读**：主报告 CH 7.2

---

### Q3.17 用一句话概括 M2.7 的架构哲学

**简短回答**：更深的层次（62）× 更窄的维度（3072）× 更满的注意力（Full）× 更细的专家（256, k=8）× 更精的训练（自我进化）——用 $O(T^2)$ 换 Agent 可靠性。

**详细解释**：

设计哲学五要素的协同：
1. **深窄**：低 FLOPs 基础 + 丰富表示层次
2. **Full Attention**：放弃效率优化，追求注意力质量
3. **细粒度 MoE**：256 个 micro expert（各 14.2M），8 个组合 = 标准 FFN 量
4. **Sigmoid routing + bias**：独立评分 + 动态负载均衡
5. **自我进化**：模型参与调参，而非纯人工

**面试要点**：理解这一设计选择背后的 trade-off——为什么选这个参数值而非其他？从计算预算、内存限制、训练稳定性等角度思考。

**延伸阅读**：主报告 CH 9.1

---

### Q3.18 M2.7 的推理速度瓶颈在哪里？

**简短回答**：Full Attention 在 192K 上下文下的 $O(T^2)$ 复杂度。实测 45.6 TPS vs 声称 100 TPS，差距可能源于长序列下的 attention 矩阵计算和 KV cache 管理。

**详细解释**：

推理速度的瓶颈分析：
- **短序列**（<8K）：Full Attention 与高效 attention 差距可控（~10-20%）
- **中序列**（8K-64K）：attention 矩阵开始显著影响延迟
- **长序列**（128K+）：$T^2$ 项主导延迟，实测 TPS 可能降至声称值的 50% 以下

缓解措施：
- GQA (8 KV 头)：KV cache 为 MHA 的 17%
- FP8 量化：减小权重传输带宽
- Speculative decoding（MTP）：MTP×3 作为 draft model，减少有效推理步数
- FlashAttention：IO-aware 实现，虽然不改变 $O(T^2)$ 但大幅降低显存带宽需求

**面试要点**：AI 声称和实测的差距（45.6 vs 100 TPS）是评估模型时必须关注的——声称值通常在最优条件下（短序列 + 高 batch），实测值反映真实使用场景。

**延伸阅读**：主报告 CH 9.2 / 附录 B.4

---

## CH 4. Full Attention + GQA/QK Norm（22 Q）

### Q4.1 写出 Lightning Attention 的标准公式

**简短回答**：$\text{LightningAttn}(Q,K,V) = \phi(Q)\left(\phi(K)^T V\right)$，其中 $\phi$ 是 kernel feature map，通过改变计算顺序将复杂度从 $O(T^2 d)$ 降至 $O(Td^2)$。

**详细解释**：

Lightning Attention 的核心洞察是计算重排：

标准 Attention：$\text{softmax}(QK^T)V$ —— 先算 $QK^T$（$T \times T$）再乘 $V$

Lightning Attention：
1. 用 $\phi(\cdot)$ 替代 softmax，使 Q 和 K 解耦
2. 先算 $\phi(K)^T V$（$d \times d$），再左乘 $\phi(Q)$
3. 复杂度：$O(Td^2)$ 而非 $O(T^2 d)$

当 $d \ll T$ 时（实际中 $d=128$，$T$ 可达 192K），$O(Td^2)$ 远小于 $O(T^2 d)$。

但代价：
- $\phi$ 是 $\exp(\cdot)$ 的线性近似，丢失了 softmax 的非线性变换和全局归一化
- $d \times d$ kernel 矩阵在 GPU 上计算密度低，实际加速比低于理论值
- 内存访问模式不友好（矩阵乘法是小矩阵，无法充分利用 GPU Tensor Core）

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3.1 / Qin et al., 2024

---

### Q4.2 写出 Full Attention 的标准公式

**简短回答**：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$，其中 $\text{mask}$ 是因果掩码（causal），$\sqrt{d_k}$ 是缩放因子。

**详细解释**：

M2.7 中的具体计算（整合 GQA、QK Norm、RoPE）：

```python
# 1. 投影
Q = Linear(x)  # (B, T, 48 * 128)
K = Linear(x)  # (B, T, 8 * 128)
V = Linear(x)  # (B, T, 8 * 128)

# 2. RoPE（仅前 rotary_dim=64 维）
Q[:, :, :64], K[:, :, :64] 施加 RoPE(theta=5M)
Q[:, :, 64:], K[:, :, 64:]  不变

# 3. QK Norm
Q = RMSNorm(Q)  # per_layer
K = RMSNorm(K)

# 4. GQA repeat
K = repeat(K, 8 → 48)  # 每个 KV 头复制 6 次
V = repeat(V, 8 → 48)

# 5. Attention
scores = Q @ K^T / sqrt(128)          # (B, 48, T, T)
scores = scores + causal_mask          # 上三角 -inf
attn = softmax(scores, dim=-1)
output = attn @ V                      # (B, 48, T, 128)
output = Linear(output)                # (B, T, 3072)
```

缩放因子为 $1/\sqrt{128}$：`self.scaling = self.head_dim ** -0.5`（源码 L240）。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.3 Kernel feature map φ 是什么？它在 Lightning Attention 中起什么作用？

**简短回答**：$\phi$ 是一个从 $\mathbb{R}^d$ 到 $\mathbb{R}^d$（或 $\mathbb{R}^{d'}$）的映射，使 $\phi(Q)\phi(K)^T$ 近似 softmax attention 矩阵，从而将 Q 和 K 解耦，允许改变矩阵乘法顺序。

**详细解释**：

核心数学技巧：

softmax attention 无法直接重排，因为 $Q$ 和 $K$ 通过 softmax 耦合在一起：
$$\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \neq \text{softmax}(Q) \cdot \text{softmax}(K)^T$$

但如果有 $\phi$ 使得：
$$\phi(Q)\phi(K)^T \approx \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$$

那么：
$$\text{LightningAttn} = \phi(Q)\left(\phi(K)^T V\right)$$

$(\phi(K)^T V)$ 先算，得到 $d \times d$ 矩阵；再左乘 $\phi(Q)$，复杂度 $O(Td^2)$。

$\phi$ 的常见选择：
- $\phi(x) = \text{elu}(x) + 1$（早期线性 Transformer）
- $\phi(x) = \exp(x)$（Performer 的 FAVOR+ 使用随机特征近似）
- 可学习的 $\phi$（如 Lightning Attention 中的设计）

M2.7 团队的结论：**任何已知的 $\phi$ 设计都无法在长上下文检索任务上匹配真实 softmax 的表现**。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3.1

---

### Q4.4 为什么 $O(Td^2)$ 在实际中不一定比 $O(T^2 d)$ 快？

**简短回答**：理论复杂度 $O(Td^2) < O(T^2 d)$（当 $d \ll T$），但 GPU 硬件特性（Tensor Core 对大矩阵更友好、内存访问模式、kernel 矩阵密度低）可能导致实际 wall clock time 不如预期。

**详细解释**：

GPU 上两种复杂度的实际表现：

| 操作 | 尺寸 | GPU 利用率 | 瓶颈 |
|------|------|:---:|------|
| $QK^T$ (Full) | $T \times d \times T$ | 高（大矩阵乘法） | 计算 bound |
| $\phi(K)^T V$ (Lightning) | $d \times T \times d$ | **低**（小矩阵乘法） | **内存 bound** |

具体问题：
1. **Tensor Core 利用率低**：$d \times d = 128 \times 128$ 的矩阵太小，无法填满 GPU 的 Tensor Core
2. **内存访问模式差**：$\phi(K)^T V$ 需要频繁读取小矩阵，L2 cache 命中率低
3. **序列化开销**：因果 masking 在 Lightning Attention 中需要额外处理（分块累积）

这就是为什么 FlashAttention（优化 IO 而非改复杂度）在实际中往往比 Lightning Attention 更快——它保持了大矩阵乘法但利用了 GPU 的分层内存结构。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3.1

---

### Q4.5 RULER 128K CWE 上 Full Attention 比 SWA 高 18pp，CWE 任务是什么？

**简短回答**：CWE（Common Word Extraction）是 RULER 的长上下文子任务——在 128K token 上下文中，找出两个句子共享的公共词。18pp 差距说明 Lightning/SWA 模型在"大海捞针"式精确检索上严重劣于 Full Attention。

**详细解释**：

RULER 128K CWE 的任务设置：
```
上下文：[128K tokens 的杂乱文本]
Query：句子 A 和句子 B 的共同词是什么？
正确答案：共享词列表
```

为什么 SWA 在此表现差：
1. SWA（Sliding Window Attention）只能看到局部的窗口内 token
2. 如果句子 A 和 B 相距超过窗口大小，模型无法直接比较它们
3. 即使有部分 Full Attention 层（M2/M2.5 的混合设计），信号在跨层传播中也会衰减

Full Attention 的优势：
- 句子 A 的每个 token 可以直接 attend 句子 B 的任何 token
- 信息最多传播 1 层（而非通过多层中间层的隐式传播）
- 适合 Agent 的"从长历史中精确检索"需求

**面试要点**：18pp 不是小差距——它把 90%（9/10 正确）拉到 72%（~7/10 正确），在 Agent 任务中多出的 18% 错误率意味着不可靠。

**延伸阅读**：主报告 CH 3.2 / paper Table 2

---

### Q4.6 MTOB 上的 15-17.6pp 差距说明了什么？

**简短回答**：MTOB（Machine Translation of Books）是长篇翻译 benchmark，差距说明 Lightning Attention 在多步长程依赖（翻译需要保持篇章级一致性）上不如 Full Attention。

**详细解释**：

两个 MTOB 子任务的差距：
| 任务 | Full | SWA | 差距 |
|------|:---:|:---:|:---:|
| K-e Bleurt | 60.0 | 45.0 | -15.0 pp |
| e-k ChrF | 44.8 | 27.2 | -17.6 pp |

MTOB 的挑战：
- 需要维持整本书的翻译一致性（术语、风格、人称）
- 远距离依赖：书的前几章信息可能影响后几章的翻译选择
- 混合注意力下，某些位置只能看到局部窗口，导致上下文信息丢失

15-17.6pp 的差距与 RULER CWE 的 18pp 一致，说明"长程精确信号传递"是 Lightning Attention 的系统性弱点，不是某个 benchmark 的偶然结果。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.7 为什么短上下文（MMLU/MATH/RULER 4K）上 Full 和 SWA 差距很小？

**简短回答**：短序列下，SWA 的滑动窗口大概率覆盖了所有相关上下文，Lightning Attention 层的信息损失在浅层/中层可被 Full Attention 层补偿。只有当依赖距离超过窗口 + 层数补偿范围时，差距才凸显。

**详细解释**：

| 任务 | 上下文长度 | 差距 | 原因 |
|------|:---:|:---:|------|
| MMLU | 通常 <2K | ~0 | 所有 token 在窗口内 |
| MATH | 通常 <4K | 0 | 局部推理为主 |
| RULER 4K | 4K | -3 pp | 几乎所有相关 token 在窗口内 |
| RULER 128K CWE | 128K | **-18 pp** | 搜索范围远超窗口 |

SWA 的能力退化是"渐进"的而非"突变"的——当依赖距离从 4K 增长到 128K，差距从 ~3pp 扩大到 ~18pp。

这对 Agent 场景的影响至关重要：Agent 的任务轨迹可能累积 64K-128K 的上下文（多轮工具调用、长对话历史），检索关键信息时依赖距离常常超过 SWA 窗口。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3.2

---

### Q4.8 SFT 后差距缩小（84 vs 72 on RULER 128K CWE），为什么仍未完全消除？

**简短回答**：SFT 可以教模型更有效地利用局部上下文"推断"远程信息，但无法替代直接的全上下文 attention——SWA 的架构限制（某些 token 对之间不可达）无法通过训练完全克服。

**详细解释**：

SFT 的补偿机制：
- 模型学会将关键信息"缓存"在中间层的隐藏状态中
- 通过层层传播间接传递远程信息
- 发展出"关注摘要型 token"（如句子开头、段落标题）的策略

但仍有 12pp 差距（84 vs 72）：
- 某些 token 对需要直接的细粒度比较（如 CWE 中的精确词匹配）
- 多层传播必然伴随信息损失和噪声混入
- 1024 层传播也抵不过 1 层直接 attention

这证明了"SFT 不能替代架构质量"——训练能弥补 30% 的差距（18pp → 12pp），但 70% 是架构硬伤。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.9 GQA 的 repeat_kv 是如何工作的？

**简短回答**：`repeat_kv` 将 8 个 KV 头沿 head 维度各复制 6 次，得到 48 个 KV 头与 48 个 Q 头一一配对。复制是"均匀分组"的——KV 头 0 被 Q 头 0-5 共享，KV 头 1 被 Q 头 6-11 共享，以此类推。

**详细解释**：

源码逻辑（`modeling_minimax_m2.py:L153-L163`）：
```python
def repeat_kv(hidden_states, n_rep):
    # hidden_states: (batch, 8, seqlen, 128)
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # (batch, 8, 1, seqlen, 128) → expand → (batch, 8, 6, seqlen, 128)
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    # reshape → (batch, 48, seqlen, 128)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

其中 `n_rep = 48 // 8 = 6`。

分组方式（隐式）：
```text
Q head 0-5   ←→  KV head 0  (6 个不同 Q 角度 query 同一个 KV)
Q head 6-11  ←→  KV head 1
...
Q head 42-47 ←→  KV head 7
```

**面试要点**：`repeat_kv` 是"零 FLOPs"的扩展（只是 reshape + expand），不增加计算量，只改变了 Q 和 KV 的配对关系。

**延伸阅读**：主报告 CH 4.1

---

### Q4.10 QK Norm 的计算公式和实现

**简短回答**：QK Norm 是对 Q 和 K 分别应用 RMSNorm（在 head_dim 维度上），公式为 $\text{QK Norm}(x) = x \cdot \frac{\gamma}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$。

**详细解释**：

M2.7 源码中的实现（`modeling_minimax_m2.py:L256-L258`）：
```python
if self.use_qk_norm:
    query_states = self.q_norm(query_states)   # RMSNorm over head_dim (128)
    key_states = self.k_norm(key_states)       # RMSNorm over head_dim (128)
```

QK Norm 的位置：在 RoPE 之前、attention score 计算之前。

```text
Q: 投影 → reshape → QK Norm → RoPE → 参与 QK^T
K: 投影 → reshape → QK Norm → RoPE → 参与 QK^T
```

为什么放在 RoPE 之前？W_q/W_k 投影可能放大某些维度的值（导致后续 QK^T 内积爆炸），QK Norm 在投影后立即执行以抑制数值范围。RoPE 是正交变换（旋转），不会改变向量的 L2 norm——因此 Norm 后的受控范围在 RoPE 后仍然保持。

归一化维度：对 128 维的 head_dim 做 RMSNorm，确保每个 head 的 Q/K 向量有相近的 L2 norm，避免某些 head 的 attention logit 过大导致 softmax"霸占"注意力。

**面试要点**：QK Norm 和 Pre-Norm 的 RMSNorm 是独立的——前者控制 attention 内部的数值范围，后者控制层间信息流动。

**延伸阅读**：主报告 CH 4.2

---

### Q4.11 QK Norm 的 per_layer 设计意味着什么？

**简短回答**：`qk_norm_type = per_layer` 表示每层有独立的 QK Norm 参数（$\gamma_q^{(l)}, \gamma_k^{(l)}$），而非所有层共享。这让每层自适应地调整 Q/K 的数值范围。

**详细解释**：

per_layer vs shared 对比：

```text
Shared:     γ_q, γ_k 在 62 层中相同  → 62 个 layer 共 2 个参数
per_layer:  γ_q^(l), γ_k^(l), l∈[0,61] → 62×2 = 124 个参数
```

per_layer 的优势：
- 不同层处理不同抽象层次的信息（浅层：局部模式，深层：全局语义）
- 不同层的 hidden state 分布不同，需要各自独立校准
- 训练中每层可以自主"学习"到适合自己分布的归一化强度

额外参数开销：124 个 float（BF16 下 248 bytes），几乎可以忽略。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.12 QK Norm 解决了什么问题？

**简短回答**：三个问题——(1) 长序列下 $QK^T$ 数值爆炸导致 softmax 退化为 argmax；(2) 不同层的 hidden state 分布差异导致 attention logit 尺度不一致；(3) 与 rope_theta=5M 配合时，RoPE 旋转后的向量 norm 偏移。

**详细解释**：

问题 1：数值爆炸
```text
无 QK Norm: max(QK^T/√128) 可达 10-50 → softmax → 接近 one-hot → 梯度消失
有 QK Norm: max(QK^T/√128) 在 3-8 范围内 → softmax 平滑 → 梯度健康
```

问题 2：层间尺度不一致
- 浅层 hidden state 通常有更大的 norm（输入信号强）
- 深层经过多轮 RMSNorm + 残差，norm 趋于稳定
- per_layer QK Norm 让每层独立适配

问题 3：RoPE 的副作用
- RoPE 旋转改变了向量的 L2 norm 分布
- 特别是在 `rope_theta = 5M` 下，高频维度的旋转角极快，可能引入额外的数值不稳定性
- QK Norm 放在 RoPE 之后，直接抹平这种不稳定

**面试要点**：M2.7 的 QK Norm 是模型**自我进化中自主引入**的，而非工程师预设——模型在训练中发现 QK 数值不稳定，自主提出"在 attention 前加 Norm"的方案。

**延伸阅读**：主报告 CH 4.2

---

### Q4.13 Partial RoPE（rotary_dim=64）的含义和优势

**简短回答**：Partial RoPE 只对 Q/K 的前 64 维（一半）施加旋转位置编码，后 64 维保持不变。优势是保留部分"位置无关"的语义表示，同时让旋转部分专注于位置编码。

**详细解释**：

```text
Q/K 向量 (128 维):
├─ [0:63]: 施加 RoPE 旋转 → 位置敏感
└─ [64:127]: 不施加 RoPE → 位置无关
```

设计动机：
1. **语义-位置解耦**：后 64 维完全编码语义内容，不受位置影响——在长序列中更稳定
2. **计算节省**：RoPE 只需计算 64 维的旋转（50% 节省）
3. **远程衰减保留**：前 64 维的 RoPE 提供了充分的相对位置信息，足以支撑 192K 上下文的远程衰减
4. **数值稳定性**：后 64 维的 Q/K 元素不经旋转，在做 QK Norm 前的方差更可控

为什么是恰好 64 维？
- 128 × 1/2 = 64，一半的策略简单且效果良好（类似 Llama-3 等模型的设计）
- 64 维仍能提供足够的旋转频率多样性（在不同频率上编码位置）

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.14 长序列下 Full Attention 的数值稳定性如何保证？

**简短回答**：三层保障——(1) $\sqrt{128}$ 缩放因子防止点积过大；(2) QK Norm 控制 Q/K 的 L2 norm；(3) FlashAttention 中的在线 softmax（online safe softmax）防止数值溢出。

**详细解释**：

数值稳定性的三重防线：

| 防线 | 机制 | 保护的阶段 |
|------|------|:---:|
| $\sqrt{d_k}$ | $QK^T / \sqrt{128}$ | 点积计算 |
| QK Norm | Q, K 在参与 $QK^T$ 前各自 RMSNorm | 点积前 |
| Online softmax | 分块计算 softmax，每块维护 running max | softmax 计算 |

其中 online softmax 是 FlashAttention 的核心技巧：
```text
标准 softmax 需要：
  1. 计算所有 QK^T 值 → 2. 找 max → 3. exp → 4. sum → 5. 归一化
  ← 步骤 1 需要存储 T×T 矩阵！

Online softmax：
  分块计算，每块独立 softmax，最后用 running max 修正
  不需要存储完整的 T×T 矩阵
```

长序列（192K）下，没有这些保障机制，FP16/BF16 的精度范围很容易导致 softmax 的中间计算溢出。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.15 GQA ratio = 6 是普适最优值吗？

**简短回答**：不是。ratio 的最优值取决于任务类型、上下文长度、架构其他参数。检索密集型任务偏好较小 ratio（更多 KV 头），效率优先场景偏好较大 ratio。M2.7 的 ratio=6 是为 Agent 场景的平衡选择。

**详细解释**：

不同 ratio 的 trade-off：

| GQA ratio | KV 头数 (48 个 Q) | KV cache | 检索质量 | 适用场景 |
|:---:|:---:|:---:|:---:|------|
| 1 (MHA) | 48 | 100% | 最高 | 短序列 + 质量优先 |
| 3 | 16 | 33% | 接近 MHA | 中等序列 |
| **6** | **8** | **17%** | **良好** | **长序列 Agent (M2.7)** |
| 12 | 4 | 8.3% | 开始下降 | 极长序列 |
| 48 (MQA) | 1 | 2.1% | 明显下降 | 效率优先 |

为什么 M2.7 选 6 而不是 3 或 8：
- ratio=3（16 KV 头）：KV cache 太大，192K 下显存不够
- ratio=8（6 KV 头）：KV cache 小幅改善，但注意力质量在 ratio≥8 后开始明显退化
- ratio=6：KV cache 为 MHA 的 17%，在长序列下内存可控，质量接近 MHA

**面试要点**：GQA ratio 不是独立可调的——它和 Q 头数、hidden_size 联动。48 个 Q 头下 ratio 必须是 48 的约数。

**延伸阅读**：主报告 CH 4.1

---

### Q4.16 为什么 M2 选择 Full Attention 而不是其他高效注意力（如 Mamba、RWKV）？

**简短回答**：论文明确探索过多种替代方案（包括 state-space models 和各类线性注意力），结论是"没有变体能在生产环境中可靠匹配 Full Attention 的质量"（paper §2.2.2）。Agent 场景的质量需求压倒了对效率的追求。M2.7 继承了 M2 的这一决策。

**详细解释**：

各种高效注意力方案的共同局限：

| 方案 | 复杂度 | 核心局限 |
|------|:---:|------|
| Lightning Attention | $O(Td^2)$ | kernel 近似丧失 softmax 非线性 |
| Sliding Window | $O(Tw)$ | 丢失跨窗口依赖 |
| Mamba/Mamba-2 | $O(T)$ | 无显式 token-to-token attention |
| RWKV | $O(T)$ | 线性递归的遗忘特性 |
| Linear Transformer | $O(Td^2)$ | $\phi$ 近似精度不足 |

这些方案的共同问题：对"大海捞针"式的长上下文精确检索（Agent 场景的核心需求）天然弱势。

M2 团队的决策逻辑：
1. Agent 需要从 128K+ 上下文中精确定位信息
2. 15-18pp 的检索差距意味着 Agent 任务不可靠
3. 与其在"高效但不可靠"中妥协，不如接受 $O(T^2)$ 的成本
4. 通过 GQA、FP8、MTP speculative decoding 来缓解推理速度

**面试要点**：M2 的选择不是"不知道有替代方案"，而是"充分评估所有替代方案后，有意识地回归 Full Attention"。M2.7 继承此决策，未改变注意力架构。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.17 M2.7 的 Full Attention 有哪些具体实现优化？

**简短回答**：实现层面的优化包括 FlashAttention（IO-aware tiling）、GQA（减少 KV cache）、FP8 量化（减少权重传输）、MTP speculative decoding（减少有效推理步数）。它们缓解了 $O(T^2)$ 的代价，但不改变复杂度的阶。

**详细解释**：

优化层次：

1. **FlashAttention**（算法层）：将 attention 计算分块（tile），利用 GPU SRAM 减少 HBM 读写。不改变 $O(T^2)$ FLOPs，但大幅减少内存带宽瓶颈。

2. **GQA**（架构层）：KV cache 体积减少 83%（48 → 8 KV 头）。

3. **FP8 量化**（精度层）：权重存储从 BF16（2 bytes）压缩到 FP8（1 byte），减少 50% 权重传输带宽。

4. **Speculative Decoding via MTP**（推理策略层）：MTP×3 作为 draft model，在最优情况下将有效推理步数减少到 ~1/3。

5. **BF16 训练/推理**（精度层）：保持激活值的精度，仅权重做 FP8。

这些优化的协同效果：虽然 $O(T^2)$ 的 FLOPs 不变，但在长上下文下的 wall clock 加速可达 2-5×。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.18 如何理解 "Full Attention 的回归暗示高效注意力可能被高估"？

**简短回答**：这是 M2.7 论文对业界的核心贡献——在 Agent 成为主流应用范式后，$O(T^2)$ 的"硬伤"可能没有想象中那么致命，因为 Agent 对可靠性的需求超过了对吞吐量的需求。

**详细解释**：

高效注意力的"理论诱惑"：
- 学术界大量论文声称"XX 线性注意力在 XX benchmark 上匹配 softmax attention"
- 复杂度从 $O(T^2)$ 到 $O(T)$ 听起来是质的飞跃
- 但大多数验证局限在短上下文（<8K）或特定 benchmark（语言建模 PPL）

M2.7 的"现实检验"：
- 在真正长的上下文（128K）上做精确检索，差距立即暴露（15-18pp）
- Agent 场景不是"猜一猜"而是"找到确切的信息"
- PPL 微小的差异在 downstream task 上会被放大

教训：
- Benchmark 选择至关重要——RULER/MTOB 的发现性远高于 MMLU/MATH
- "理论上更快"不等于"生产中有用"
- 架构选择应由 end-to-end 场景需求驱动，而非中间 benchmark

**面试要点**：M2.7 是工业界**首次公开量化**"高效 attention vs 全注意力"大规模对比，为解决"attention 类型选择"的工程问题提供了数据支撑。

**延伸阅读**：主报告 CH 9.1

---

### Q4.19 QK Norm 是放在 RoPE 之前还是之后？为什么？

**简短回答**：放在 RoPE 之前。源码中投影后立即执行 QK Norm，然后才经过 RoPE。原因是 QK Norm 的作用是抑制 Q/K 投影后的数值范围扩散（W_q/W_k 投影可能放大某些维度的值），而非修正 RoPE 的旋转效应。

**详细解释**：

```python
# 源码中的执行顺序（modeling_minimax_m2.py）
query_states = self.q_proj(x)        # 投影
key_states = self.k_proj(x)
# ... reshape ...
if self.use_qk_norm:
    query_states = self.q_norm(query_states)      # 1. QK Norm 先
    key_states = self.k_norm(key_states)
query_states, key_states = self.rotary_emb(...)  # 2. RoPE 后
# ... 然后计算 attention scores ...
```

如果在 RoPE 之后做 Norm，旋转操作已经改变了向量的数值分布，Norm 后可能仍有异常值。正确顺序是先 Norm（抑制投影后的数值范围），再 RoPE（纯旋转变换，不改变范数）。

QK Norm 的作用是抑制投影后的异常值：投影矩阵 W_q 可能让某些维度的值远大于其他维度，导致后续 QK^T 内积爆炸。Norm 在投影后立即执行，确保进入 RoPE 的 Q/K 向量具有受控的 L2 norm。RoPE 是正交变换（旋转），不会改变范数，因此 Norm 后的受控范围在 RoPE 后仍然保持。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.20 Full Attention 中 head_dim = 128 的选择依据是什么？

**简短回答**：128 是业界标准 head_dim（Llama、GPT-3、Mistral 均使用 128），与 $48 \times 128 = 6144$ → 经 O 投影回到 3072 的配合良好。128 的 $\sqrt{128} \approx 11.3$ 缩放因子在数值稳定性和注意力灵敏度间取得平衡。

**详细解释**：

head_dim 的影响：
- 太小（如 64）：每个 head 容量不足，需要更多 head 补偿，增加 KV cache 头数
- 太大（如 256）：单头容量大，但头数少（3072/256 ≈ 12），注意力模式多样性降低

M2.7 的计算：$48 \times 128 = 6144$，翻倍到 Q 投影维度后经 O 投影 $6144 \to 3072$ 回到 hidden_size。

这也是 M2.7 "更深更窄"的体现——6144 的 Q 投影维度比 V4-Flash（4096 → 4096 × 64 = 262144 / 64 ≈ 64 heads × 64 head_dim? 不准确，V4 的头维度设计不同）的计算更高效。

**面试要点**：head_dim 和 num_heads 的乘积不一定等于 hidden_size——M2.7 中 Q 投影输出 $48 \times 128 = 6144 = 2 \times 3072$，这在 GQA 中常见（因为 Q 和 O 投影独立）。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.21 如果推理场景是短序列（<4K），M2.7 的 Full Attention 选择是否还有意义？

**简短回答**：对短序列场景，Full Attention 相比 Lightning 的优势不大（差距 <3pp），但 M2.7 是为 Agent 设计的——Agent 的任务轨迹天然是长序列（多轮工具调用累加），因此必须以长序列为设计出发点。

**详细解释**：

不同场景的序列长度特征：
- Chatbot：单轮 <2K，多轮可到 8K-32K
- Code Agent：代码上下文 + 工具输出 可达 32K-128K
- 长文档分析：直接 128K+
- Agent 自主任务：100+ 轮工具调用的累积轨迹 >64K

M2.7 的定位是 Agent 模型，因此长序列是"常态"而非"特殊情况"。

对比专门为短序列设计的模型：
- V4-Flash 的 CSA+HCA 混合稀疏在 4K 下可能比 Full Attention 更快
- 但 Agent 任务一旦长度超过稀疏模式范围，质量就会下降
- M2.7 选择"长序列上保证质量"而非"短序列上追求极致速度"

**易混淆**：性能差距不是"Full Attention 在任何长度都比 Lightning 好"，而是"差异随长度增长而放大，在 Agent 场景的长度范围内刚好不可接受"。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

### Q4.22 M2.7 既然推理慢，为什么不用 FlashAttention 的 block-sparse 变体？

**简短回答**：FlashAttention 的 block-sparse 变体（如 FlashAttention-3 的稀疏模式）本质上还是"稀疏注意力"——会丢失某些 token 对之间的直接交互。M2.7 团队的逻辑是：任何丢失 token 对直接交互的方案，在 Agent 检索任务上都会有可测量的质量损失。

**详细解释**：

不同"高效"程度的层次：

```text
Full FlashAttention (M2.7 选择)
  ↓ 加速但保持完整注意力矩阵
Block-sparse FlashAttention (M2.7 拒绝)
  ↓ 通过丢失部分 token 对来加速
Sliding Window + Selected Full (M2/M2.5)
  ↓ 更激进的稀疏化
Lightning Attention (Text-01/M2/M2.5)
  ↓ 完全线性化
```

M2.7 的选择停在第一层：使用 FlashAttention 的 IO 优化（tiling、recomputation），但不使用其稀疏化功能。这保留了完整的 $T \times T$ attention 矩阵的信息量，同时获得了 IO 层的加速。

核心立场：**算法层的优化（FlashAttention tiling）可以接受，但架构层的"近似"（稀疏化/线性化）不能接受**。

---

## 附录：各章 Q 数量统计

| 章 | Q 数 | 主题 |
|:---:|:---:|------|
| CH 1 | 18 | LLM 预备：Transformer, MHA/MQA/GQA, MoE, SwiGLU, RMSNorm, RoPE, KV Cache, O(n²) 等 |
| CH 2 | 12 | MiniMax 演进：Text-01→M2→M2.5→M2.7, Lightning 放弃, 自我进化 |
| CH 3 | 18 | M2.7 概览：超参表, 62×3072, GQA 48/8, 256 experts k=8, sigmoid, MTP×3 |
| CH 4 | 22 | Full Attention + GQA/QK Norm：公式, ablation, repeat_kv, per_layer, partial RoPE |
| **合计** | **70** | |

# MiniMax-M2.7 QA 分册：CH 5-8 + 面经章节

> 版本 v1.0 · 2026-06-09 · 配套主报告 v0.1 · 80 Q 总计

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 3（Full Attention + GQA/QK Norm）

---

## CH 5. MoE 路由 —— sigmoid + routing bias（17 Q）

### Q5.1 M2.7 的 MoE 路由为什么用 sigmoid 而非 softmax？

**简短回答**：sigmoid 各 expert 评分相互独立，允许多个 expert 同时得高分；softmax 强制竞争归一化，适合 k 小（1-2）。k=8 时 sigmoid 更灵活，且省去 256 个 exp 计算。

**详细解释**

sigmoid $\sigma(x)=1/(1+e^{-x})$ 对每个 logit 独立输出 (0,1)，无相互约束，多个 expert 可同时接近 1。softmax $e^{x_i}/\sum e^{x_j}$ 强制所有 expert 竞争和为 1 的概率质量，在 k 大时产生「赢者通吃」效应——top-2/3 的分数被分母压低。

k=8 场景下 sigmoid 两项优势：(1) 8 个选中 expert 都可有高分数，不被竞争压制；(2) 计算上省去 256 个 exp——在大 batch 和长序列下累积可观。配合 routing bias 修正选择偏差后，sigmoid + top-8 在实践中效果良好。源码 `route_tokens_to_experts`（L113）使用 `F.sigmoid(router_logits.float())`。

**面试要点**：sigmoid 后归一化为 `weights /= weights.sum()`（简单除和），非 softmax——语义是「按相对大小分配贡献」而非「重新分配概率质量」。分清楚「选 expert 用 bias 调整后的分数」和「算权重用原始 sigmoid 分数」。

**延伸阅读**：主报告 CH 5.1 / config.json → scoring_func = sigmoid

---

### Q5.2 e_score_correction_bias 是什么？如何实现负载均衡？

**简短回答**：256 维 buffer，作为 aux-loss-free 风格的 routing bias 加在 sigmoid 分数上。命中率高的 expert bias 减小，低的增大。不参与梯度更新（register_buffer），由外部负载均衡算法动态调整。

**详细解释**

传统 MoE 负载均衡依赖 auxiliary loss（Switch Transformer 式）：$L_{\text{aux}} = \alpha \cdot N \cdot \sum_i f_i \cdot P_i$，$f_i$ 为 expert i 的 token 分配比例，$P_i$ 为平均 gate 概率。此 loss 混入训练总 loss 通过梯度影响 gate——需仔细调 $\alpha$（太大压缩模型容量，太小负载不均衡），且梯度可能与主任务梯度冲突。

M2.7 采用 DeepSeek V3 风格的 aux-loss-free 方案：`e_score_correction_bias` 是 `register_buffer`（非 `nn.Parameter`），每个 training step 统计各 expert 命中数后独立更新——命中率高→bias↓（降低被选概率），命中率低→bias↑（提升被选概率）。bias 更新不经过 optimizer（不受学习率、momentum 影响），将「学习路由」（gate 梯度）和「平衡路由」（bias 更新）彻底解耦。源码：`self.register_buffer("e_score_correction_bias", torch.zeros(256))`（L111）。

**面试要点**：buffer vs parameter 是关键差异——bias 更新来源是外部负载均衡算法而非 optimizer 梯度。这是两个可能冲突的目标被优雅解耦的设计案例。

**延伸阅读**：主报告 CH 5.2 / DeepSeek V3.2 paper

---

### Q5.3 M2.7 为什么选择 top-8？

**简短回答**：8/256 = 3.125% 激活比，配合 k=8 每 token 获得 $8 \times 0.5 = 4\times$ 中间维度总量（与 dense 模型持平），且无 shared expert 需略大的 k 覆盖通用能力。

**详细解释**

top-k 选择约束：k 太小（2-4）容量不足；k 太大（16+）趋近 dense。M2.7 k=8 vs V4-Flash k=6+1 shared——M2.7 纯 routed 需略大 k。激活参数：8 expert × 14.2M ≈ 113M/层 + attention ≈ 158M/层 × 62 ≈ 9.8B。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 2.3

---

### Q5.4 为什么不设置共享专家（shared_intermediate_size=0）？

**简短回答**：「纯 routed」策略——所有能力靠 top-8 expert 组合覆盖。用 256 细粒度 expert + routing bias 平衡替代共享专家，避免共享专家成为「万能兜底」导致路由退化。

**详细解释**

共享专家（如 V4-Flash）覆盖通用知识但增加所有 token 的计算。M2.7 无共享：256 expert 足够多，通用知识自然分布在多 expert 中；routing bias 确保均衡使用。对比：M2.7（k=8, 256 expert, 无 shared）vs V4-Flash（k=6, 64 expert, 1 shared）——不同设计哲学。

**面试要点**：无共享专家不等于无通用能力——高频 expert 由训练自然形成。

**延伸阅读**：主报告 CH 5.4

---

### Q5.5 jitter_noise 在 MoE 训练中的作用？

**简短回答**：训练时对 hidden_states 乘 [1-ε, 1+ε] 均匀噪声（乘法正则），防止 gate 过拟合特定输入模式，提升路由泛化性。仅训练时生效。

**详细解释**

源码（L20-21）：`hidden_states *= uniform_(1-ε, 1+ε)`。效果：(1) gate 决策边界更平滑；(2) token 扰动可能改变 top-8 选择，提供额外训练信号。区别于加法噪声（如 dropout），乘法噪声幅度与信号强度成正比。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5.1

---

### Q5.6 Expert dispatch 的 one_hot + index_add_ 如何工作？

**简短回答**：`one_hot(top_k_index, 256)` → [256, 8, BS×L] 掩码 → 遍历被命中的 expert → `index_add_` 累加加权输出。跳过未命中 expert，避免遍历 256 个全部。

**详细解释**

`MiniMaxM2Experts.forward`（L68-L101）：(1) 零张量初始化；(2) one_hot 掩码；(3) `expert_hit = ...nonzero()` 找命中 expert（通常 30-40%）；(4) 逐 expert 计算 `expert(state) * weight` 后 `index_add_(0, top_x, ...)` 累加。`index_add_` 避免显式稀疏矩阵，在 256 expert 时效率显著。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：code-snippets/m2_experts.py

---

### Q5.7 为什么选 256 细粒度 expert 而非更少粗粒度？

**简短回答**：更多更小的 expert 提供更丰富组合空间（C(256,8) 种匹配），每个 expert 的 intermediate_size=1536（仅 50% hidden_size）保持计算轻量。M2 论文称「mini activations」原则。

**详细解释**

vs Mixtral（8 expert, k=2, 每个 6.75B）：M2.7 是「选 8 个小专家组合」vs「选 2 个大专家」。代价：expert dispatch 通信开销更大（256 vs 8），负载均衡更复杂。

**面试要点**：粗粒度少选 vs 细粒度多选——两种 MoE 设计哲学。

**延伸阅读**：主报告 CH 2.3

---

### Q5.8 routing bias 与 Switch Transformer auxiliary loss 对比？

**简短回答**：aux loss 通过梯度优化 gate 来均衡（混入 loss）；routing bias 直接在 gate 输出加偏置（与梯度解耦）。后者无 $\alpha$ 超参调优问题，bias 更新不在梯度路径上。

**详细解释**

Switch Transformer: $L_{\text{aux}} = \alpha N \sum f_i P_i$ 加入 total loss，梯度可能冲突。M2.7: bias 独立更新（过载↓，欠载↑），不受 optimizer 影响。关键区别：aux loss 是「软约束」，bias 是「硬调整」。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5.2

---

### Q5.9 Gate 网络设计（3072→256）特点？

**简短回答**：单层 `nn.Linear(3072, 256, bias=False)`，无 bias 避免与 routing bias 冲突，参数量 0.78M（可忽略），保持 BF16 精度（FP8 量化排除项）。

**详细解释**

`bias=False` 确保评分仅依赖输入。gate 是 FP8 量化排除项之一——路由误差在 MoE 中被放大（选错 expert 比 attention 精度损失更严重）。

**面试要点**：gate 保持 BF16 是关键工程决策。

**延伸阅读**：config.json → modules_to_not_convert

---

### Q5.10 为什么 routing bias 用 register_buffer 而非 nn.Parameter？

**简短回答**：buffer 不参与梯度更新，bias 变化来源是外部负载均衡算法。如果用 Parameter，optimizer 会按「降 loss」方向更新 bias，与「均衡负载」目标可能冲突——两个目标被解耦。


**详细解释**：`register_buffer` 将 tensor 注册为模型的一部分（随模型保存/加载、移动到 GPU），但不被 optimizer 追踪（不参与梯度更新）。这与 `nn.Parameter` 的核心区别——Parameter 的梯度由 optimizer 根据 loss 更新，buffer 的值由外部逻辑（如负载均衡算法）手动更新。这种解耦确保了路由质量和负载均衡两个目标的独立性。


**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5.2

---

### Q5.11 MoE 路由完整数学流程

**简短回答**：`hidden → gate → sigmoid → +bias → top-8 indices → gather original sigmoid weights → normalize(÷sum) → experts dispatch → weighted sum`。

**详细解释**

设 $x \in \mathbb{R}^{3072}$：(1) $l = W_g x$；(2) $s_i = \sigma(l_i)$；(3) $s'_i = s_i + b_i$；(4) topk(s', k=8) 得索引和原始 $s$；(5) $w_j = s_{i_j} / \sum s_{i_k}$；(6) $y = \sum w_j \cdot \text{Expert}_{i_j}(x)$。注意 topk 用 s'（加 bias）但 gather 用 s（原始）——bias 只影响选择不影响权重。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：code-snippets/m2_moe_router.py:L10-L16

---

### Q5.12 sigmoid 后权重归一化为什么不是 softmax？

**简短回答**：已选 top-8，对 8 个值做 softmax 等价于除和，直接除和避免多余 exp。且 sigmoid 输出 [0,1] 范围内直接除和比 softmax 更平滑——不放大最大分数的专家。


**详细解释**：sigmoid 路由选出 top-8 后，权重归一化的目的是确保 8 个专家的输出贡献和为 1（避免信号放大）。直接除以 sum（而非 softmax）的原因是：sigmoid 输出已在 (0,1) 范围，除和保持了这个范围；softmax 会重新分配权重（放大高分专家的权重），改变 sigmoid 路由的原始判断。


**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5.1

---

### Q5.13 SwiGLU FFN 结构

**简短回答**：`w2(silu(w1(x)) * w3(x))`。w1/w3 投影到 ffn_dim(1536)，w1 经 SiLU 门控后与 w3 输出逐元素相乘，w2 投影回 hidden_dim(3072)。参数量 3×3072×1536 ≈ 14.2M/expert。


**详细解释**：SwiGLU 通过门控机制（SiLU）让 FFN 选择性传递信息——gate 接近 0 的特征被抑制，接近 1 的特征通过。M2.7 的 FFN 中间维度为 hidden_size × 2.5 ≈ 1536（配合 hidden=3072），每个专家参数约 14.2M（3×3072×1536）。使用 SwiGLU 而非 ReLU 的原因：SiLU 处处可导、梯度平滑，在深层网络中训练更稳定。


**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：code-snippets/m2_mlp.py

---

### Q5.14 intermediate_size=1536（50% hidden_size）考量

**简短回答**：激进的小 expert 设计。256 expert × k=8 → 每 token 总中间维度 = 8×0.5 = 4×（与 dense 模型持平）。若用标准 4× 则总参膨胀到 ~900B。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 2.3

---

### Q5.15 与 DeepSeek V3 MoE 差异

**简短回答**：两者都使用 256 专家 + sigmoid 路由 + bias 调节的 Aux-Loss-Free 方案，核心差异在共享专家（M2.7 无，V3 有 1 个）和 MTP 模块数（M2.7 有 3 个，V3 有 1 个）。

**详细解释**：

| 维度 | M2.7 | DeepSeek V3 |
|------|------|-------------|
| 专家总数 | 256 | 256 + 1 shared |
| Top-k | 8 | 8 |
| 共享专家 | 无 | 1 |
| 评分 | sigmoid | sigmoid |
| 负载均衡 | bias (buffer) | bias (类似 parameter) |
| MTP | 3 | 1 |

两者都用 sigmoid + bias 的 aux-loss-free 方案，但 shared expert 和 MTP 不同。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5（MoE路由系统）

---

### Q5.16 与 Mixtral 8×7B 对比

**简短回答**：M2.7 使用 256 细粒度专家（每个 ~0.9B）+ sigmoid 路由 + top-8，体现"细粒度多选"哲学；Mixtral 使用 8 粗粒度专家（每个 ~6.75B）+ softmax 路由 + top-2，体现"粗粒度少选"哲学。

**详细解释**：

| 维度 | Mixtral 8×7B | M2.7 |
|------|-------------|------|
| Expert 数 | 8（粗粒度） | 256（细粒度） |
| Top-k | 2 | 8 |
| 评分 | softmax | sigmoid |
| 单 expert 大小 | 6.75B | ~0.9B |
| 总参 | 46.7B | 229.9B |

体现两种哲学：粗粒度少选（Mixtral）vs 细粒度多选（M2.7）。

**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5（MoE路由系统）

---

### Q5.17 MoE 通信开销

**简短回答**：Expert parallelism 需两次 all-to-all：token dispatch（发送到 expert GPU）和 result combine（发回）。256 expert + k=8 + hidden_dim=3072，通信量与 BS×L×d 成正比，是分布式训练主要瓶颈。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：MoE 相关的设计决策通常围绕"容量 vs 通信开销"展开。面试中展示你理解这背后的系统性权衡。

**延伸阅读**：主报告 CH 5（MoE路由系统）

---

## CH 6. Multi-Token Prediction (MTP)（12 Q）

### Q6.1 MTP 核心原理

**简短回答**：标准 next-token 外加 N 个辅助头预测 token[n+1]/[n+2]/[n+N]。训练提供额外监督信号，推理作 speculative decoding 的 draft model（内建，无需额外模型）。

**详细解释**

M2.7 在主模型 62 层后接 3 个 MTP 模块，每模块 1 层 Transformer + output head（共享 embedding），链式预测未来 3 个 token。本质是「多任务学习」——迫使模型学习更长程依赖。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6.1

---

### Q6.2 为什么 3 模块 × 1 层？

**简短回答**：3 个模块 speculative decoding 提供 3 个 draft token（vs V3 的 1 个），加速潜力更大；每模块 1 层保持延迟低。对比 V3 的「1 模块 × 2 层」，M2.7 是「浅而多」vs「深而少」。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6.3 / config.json → num_mtp_modules=3, mtp_transformer_layers=1

---

### Q6.3 MTP 如何用于 speculative decoding？

**简短回答**：Draft——主模型 → last hidden → MTP1/2/3 生成 draft_1/2/3；Verify——主模型一次 forward 验证所有 draft；Accept/Reject——rejection sampling 决定接受哪些。MTP 作内建 draft model 无需额外显存。

**详细解释**

加速比简化公式：$\text{speedup} = (1+3\alpha)/(1+\beta)$，$\alpha$=acceptance rate，$\beta$=draft 开销比例。设 $\alpha=0.6, \beta=0.1$ → 约 2.55× 理论加速，实际 1.3-1.8×。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6.1

---

### Q6.4 3 个模块如何预测 token[n+1]/[n+2]/[n+3]？

**简短回答**：链式传递——模块 k 接收模块 k-1 的 hidden state + 主模型 last hidden state，经 1 层 Transformer 后由独立 output head 预测对应 token。非并行，而是顺序（token[n+1] 结果影响 token[n+2]）。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6.2

---

### Q6.5 与 DeepSeek V3 MTP 差异

**简短回答**：M2.7 使用 3 个浅层 MTP 模块（每模块 1 层），V3 使用 1 个深层 MTP 模块（2 层）。"浅而多"的 M2.7 方案推理加速更大，"深而少"的 V3 方案单 token 质量更高。

**详细解释**：

| 维度 | M2.7 | V3 |
|------|------|-----|
| 模块数 | 3 | 1 |
| 每模块层数 | 1 | 2 |
| 预测 token 数 | t+1/2/3 | t+1 |
| 模块连接 | 链式 | 单模块 |

「浅而多」vs「深而少」：M2.7 推理加速更大，V3 单 token 质量更高。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.6 MTP 训练 vs 推理

**简短回答**：训练——所有模块 loss 加权求和（$\alpha_1 > \alpha_2 > \alpha_3$，远期权重递减）；推理——标准生成不用 MTP，仅 speculative decoding 时激活。MTP 不影响标准推理质量。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**易混淆**：MTP 在标准推理中不激活，模型仍是自回归 next-token 预测器。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.7 MTP loss 计算

**简短回答**：$L_{\text{total}} = L_{\text{main}} + \sum \alpha_i L_{\text{MTP}_i}$。$\alpha_1=0.3, \alpha_2=0.15, \alpha_3=0.075$（递减，因远期预测不确定性高），避免 MTP loss dominate 主 loss。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.8 共享 embedding

**简短回答**：是——MTP 共享主模型的输入 embedding 和 LM head（3072×200064≈614M 参数）。若 3 模块各有独立 output head 将增加 1.84B 参数。共享后 MTP 仅需 Transformer 层参数（~130M 总计）。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.9 为什么 1 层 Transformer？

**简短回答**：1 层在 draft 质量与延迟间平衡最优。若 3-4 层则 draft 延迟接近主模型，失去 speculative decoding 意义。1 层 draft time ~主模型 1/3（无 MoE），配合 50-70% acceptance rate。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.10 与 Medusa 对比

**简短回答**：MTP 使用顺序 Transformer 层（有位置依赖）预测未来 token，Medusa 使用并行独立 heads（无位置依赖）。MTP 质量更高但速度较慢，Medusa 速度快但 acceptance rate 低。MTP 可视为 Medusa 的"升级版"。

**详细解释**：

| 维度 | MTP | Medusa |
|------|-----|--------|
| 预测机制 | 顺序 Transformer 层 | 并行独立 heads |
| 位置依赖 | 有（链式） | 无（独立） |
| 质量/速度 | 质量高但慢 | 速度快但 acceptance 低 |

MTP 是 Medusa 升级版——用 Transformer 层替代简单线性 heads。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.11 禁用 MTP 损失什么？

**简短回答**：标准推理禁用 MTP 不损失模型质量（主模型参数独立），仅失去 speculative decoding 加速。MTP 对主模型的影响仅在训练阶段（辅助训练信号）。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

### Q6.12 MTP 显存开销

**简短回答**：3 模块约 130M 参数（<0.06% of 229.9B），BF16 下约 260MB 显存。推理不启用时零开销。参数量极低但训练时提供 3 个位置的监督信号——性价比极高。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 6（训练体系 + MTP）

---

## CH 7. 工程实现 —— 192K / FP8 / 自进化（13 Q）

### Q7.1 FP8 E4M3 量化

**简短回答**：E4M3（4-bit exponent + 3-bit mantissa），动态范围 ±448。将权重从 BF16 压缩到 FP8，显存减半。M2.7 使用 block-wise 量化，每 128×128 块独立 scale。

**详细解释**

FP8 两种格式：E4M3（精度高，适合前向）和 E5M2（动态范围大，适合梯度）。M2.7 用 E4M3 量化权重，精度损失通常 <1%。配置：`fmt: float8_e4m3fn`，`weight_block_size: [128,128]`。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7.2

---

### Q7.2 Block-wise 128×128 优势

**简短回答**：精度——比 per-tensor 鲁棒（每块独立 scale 避免 outlier 主导）；硬件——128×128 对齐 NVIDIA H100 warp（32 threads）和 Tensor Core WGMMA 指令；存储——scale 开销可忽略。

**详细解释**

Per-tensor（单 scale）→ outlier 主导；Per-channel（每行 scale）→ 精度好但存储大；Block-wise 128×128 → 精度/硬件/存储三方平衡。对于 3072×1536 矩阵：128×128 仅需 288 个 scale ≈ 0.3KB。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.3 排除项：gate, bias, lm_head

**简短回答**：三者对精度敏感——gate 决定路由（选错 expert 灾难性），lm_head 产生 logits（影响 token 选择），bias 微调均衡。保持 BF16 精度，总参数占比仅 ~0.27%。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：量化策略不是全量最好——识别并保护精度敏感组件。

**延伸阅读**：config.json → modules_to_not_convert

---

### Q7.4 192K 上下文如何支持？

**简短回答**：三大支柱：(1) `rope_theta=5M`（Llama-3 的 10×）提供远程位置区分；(2) partial RoPE（64/128 维旋转）保持语义稳定性；(3) GQA（8 KV 头）KV cache 减少 6×。

**详细解释**

RoPE 频率 $\theta_i = \text{theta}^{-2i/d}$。5M theta 下最低频维度（i=31，rotary_dim=64/2=32 对）周期约 10,900 token——在 192K 下仅完成 17.6 周期，足以区分远近位置。若 theta=500K，周期仅 3,500 token，在 192K 下完成 55 个周期→远程位置几乎不可区分。

Partial RoPE：前 64 维旋转编码位置，后 64 维不旋转——保留位置无关的语义特征通道（词义、语法表征稳定）。GQA 将 KV cache 从 MHA ~292GB 降至 ~49GB，使 192K 单卡推理可行。三者的协同：大 theta 提供位置精度→partial RoPE 保留语义→GQA 控制显存。

**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7.1

---

### Q7.5 自我进化框架

**简短回答**：模型参与自身训练优化——100+ 轮「分析—改进—验证」迭代。三层：训练监控（检测 loss 异常、自动调 LR）、Agent scaffold 优化（修改 tool-calling 策略）、架构改进（引入 QK Norm、调整 routing bias 更新策略）。

**详细解释**

三层运作的范例：(1) 训练监控——模型分析训练日志，检测 loss spike/gradient explosion/专家负载不均，自动调整 LR 和 batch size；(2) Agent scaffold——模型作为 agent 运行任务，分析失败案例，修改自身 tool-calling 策略和 prompt 模板；(3) 架构改进（最激进）——QK Norm 来自模型发现深层 attention logit 不稳定；routing bias 更新频率和步长由模型实验确定；expert 容量分配阈值经模型调优防止过载。

**关键澄清**：自我进化不是模型改自己的权重——权重仍由标准训练更新。自我进化是模型参与超参、框架和工作流的优化，从被动训练对象变为主动「co-designer」。内部评测提升约 30%，但未经第三方验证。

**延伸阅读**：主报告 CH 7.3 / paper §3

---

### Q7.6 训练流程概要

**简短回答**：三阶段——(1) Pre-training：Full Attention + MoE，MTP 辅助 loss，BF16 混合精度；(2) SFT：agent tool use、长链推理数据；(3) RL + 自我进化：100+ 轮迭代，内部评测提升 ~30%。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.7 BF16 vs FP8 精度差异

**简短回答**：BF16（7-bit mantissa）→ FP8 E4M3（3-bit mantissa），理论差 4 bits。但 block-wise 量化使实际任务精度损失 <1%：每块独立 scale 补偿量化误差，误差在输出中趋于抵消。MMLU 上 <0.5pp 损失。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.8 dynamic activation scheme

**简短回答**：`activation_scheme: "dynamic"` 表示激活值运行时计算 scale（每次 forward 实时 `max(|act|)`），而非预校准的 static scale。M2.7 作为 Agent 模型，输入分布广泛，static scale 难以覆盖。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**易混淆**：激活 dynamic quantization 与权重 static block-wise quantization 是独立配置。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.9 实测 ~45 TPS vs 声称 100 TPS

**简短回答**：差距来源：(1) 官方可能用 MTP speculative decoding + 短输入测试，第三方用标准推理 + 中等输入；(2) Full Attention $O(T^2)$ 在长输入下计算量巨大；(3) 硬件配置差异。45 TPS 在 229.9B/9.8B 模型上属于合理范围。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.10 192K KV cache 显存

**简短回答**：GQA（8 KV 头，BF16）：$2 \times 62 \times 8 \times 128 \times 192K \times 2\text{B} \approx 48.7\text{GB}$。若 MHA（48 KV 头）则 ~292GB。GQA 是实现 192K 推理的关键。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.11 GQA 在 192K 的重要性

**简短回答**：KV cache 减少 6×（48→8），使长上下文推理在单卡上可能。MHA(ratio=1)→质量最优但 cache 最大；GQA(ratio=6)→质量损失 <1% 但 cache 减少 6×；MQA(ratio=48)→质量损失 1-3% 但 cache 最小。M2.7 选 ratio=6 是 sweet spot。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.12 Partial RoPE（64/128）

**简短回答**：前 64 维旋转（编码位置），后 64 维不旋转（保留位置无关语义）。效果：(1) 语义稳定性——相同 token 不同位置表征一致；(2) 数值稳定——减少长序列 RoPE 误差累积。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

### Q7.13 rope_theta = 5M 为什么这么大？

**简短回答**：RoPE 频率 $\theta_i = \text{theta}^{-2i/d}$。theta 越大 → 低频维度旋转越慢 → 远程位置区分度越高。5M theta 下最低频维度在 192K 上仅 17.6 周期，足以区分。500K theta 则 55 周期→ 远程位置不可区分。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：支撑项虽然不是核心架构创新，但决定了模型能否实际部署。面试官可能追问"这个参数如何在工程上实现"。

**延伸阅读**：主报告 CH 7（支撑项：上下文/推理优化）

---

## CH 8. 源码系统（13 Q）

### Q8.1 modeling_minimax_m2.py 结构

**简短回答**：706 行 9 个核心类：MLP(L50-65) → Experts(L68-101) → MoE Block(L103-131) → RMSNorm(L133-152) → Attention(L236-314) → DecoderLayer(L315-361) → RoPE(L362-397) → Model(L418-582) → CausalLM(L583-706)。

执行顺序：`ForCausalLM → Model.forward → [DecoderLayer × 62] → MTP → lm_head`


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.2 MiniMaxM2Attention 实现（L236-314）

**简短回答**：GQA（48Q/8KV）+ QK Norm per_layer + partial RoPE + Flash Attention。流程：Q/K/V 投影 → QK Norm（关键差异）→ reshape → RoPE → KV cache → attention → O 投影。与 Llama 唯一关键差异在 QK Norm 的加入。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：QK Norm 解决长上下文 attention logit 数值稳定性。

**延伸阅读**：code-snippets/m2_attention.py

---

### Q8.3 MiniMaxM2SparseMoeBlock 路由流程（L103-131）

**简短回答**：约 30 行实现完整路由：(1) gate 投影；(2) 训练时 jitter_noise；(3) `route_tokens_to_experts`：sigmoid + bias + top-8 + 归一化；(4) experts dispatch；(5) reshape 返回。

```python
routing_weights = F.sigmoid(router_logits.float())
scores = routing_weights + self.e_score_correction_bias
_, top_k_idx = torch.topk(scores, 8, dim=-1)
top_k_weights = routing_weights.gather(1, top_k_idx)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
```


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：能手写出此路由流程是 MoE 面试高分回答。

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.4 推理完整流程

**简短回答**：`token_id → embedding → [RMSNorm → Attention(QK Norm+RoPE+GQA+KV Cache) → +residual → RMSNorm → MoE(gate→sigmoid+bias→top-8→dispatch→weighted sum) → +residual] × 62 → final RMSNorm → lm_head → logits → sample → next_token`。

每层执行 Pre-Norm → Attention → +residual → Pre-Norm → MoE → +residual。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.5 关键文件清单

**简短回答**：M2.7 源码包含 4 个关键文件——`config.json`（31 字段超参数）、`configuration_minimax_m2.py`（HF PretrainedConfig 子类）、`modeling_minimax_m2.py`（706 行完整实现）、`tokenizer_config.json`（vocab=200,064）。

**详细解释**：

| 文件 | 作用 |
|------|------|
| `config.json` | 31 字段超参数 |
| `configuration_minimax_m2.py` | HF PretrainedConfig 子类 |
| `modeling_minimax_m2.py` | 706 行完整实现 |
| `tokenizer_config.json` | vocab=200,064 |

**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.6 MiniMaxM2Experts dispatch（L68-L101）

**简短回答**：one_hot 掩码 [256,8,BS×L] → 找命中 expert → 逐 expert 计算 `expert(state) * weight` → `index_add_` 累加。跳过未命中 expert，`index_add_` 避免显式稀疏矩阵。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：code-snippets/m2_experts.py

---

### Q8.7 DecoderLayer Pre-Norm 顺序（L315-L361）

**简短回答**：`input → input_layernorm → attention → +residual → post_attention_layernorm → MoE → +residual`。标准 Pre-Norm 架构，训练稳定无需 warm-up。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.8 Model.forward 主循环

**简短回答**：embedding → RoPE 预计算（所有层共享）→ 62 层循环 → final RMSNorm → 返回 `MoeModelOutputWithPast`（含 router_logits 用于 load balancing loss）。MTP 在 ForCausalLM 层处理。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.9 ForCausalLM 训练与推理

**简短回答**：训练——model.forward → lm_head → CrossEntropyLoss + MTP losses + load_balancing_loss；推理——`generate()` 或手动 forward → logits → sample。与 Llama 主要差异：MTP loss 聚合和 MoE load balancing loss 收集。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.10 SwiGLU MLP 三个矩阵

**简短回答**：`w1`(gate proj), `w3`(up proj), `w2`(down proj)。`w2(silu(w1(x)) * w3(x))`。每个 3072×1536，单 expert 共约 14.2M 参数。门控 SiLU(W1·x) 连续（非 0/1），梯度更平滑。


**详细解释**：SwiGLU 通过门控机制（SiLU）让 FFN 选择性传递信息——gate 接近 0 的特征被抑制，接近 1 的特征通过。M2.7 的 FFN 中间维度为 hidden_size × 2.5 ≈ 1536（配合 hidden=3072），每个专家参数约 14.2M（3×3072×1536）。使用 SwiGLU 而非 ReLU 的原因：SiLU 处处可导、梯度平滑，在深层网络中训练更稳定。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.11 RMSNorm 实现

**简短回答**：`x * rsqrt(mean(x²) + eps) * weight`。无 bias，不减去均值。float32 高精度计算，eps=1e-6。相比 LayerNorm 省去减均值操作，速度更快但效果无明显差异。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.12 Config 关键字段分组

**简短回答**：M2.7 的 config 字段分为四组——基础架构（层数/维度/头数）、MoE（专家数/top-k/路由函数）、Attention（QK Norm/partial RoPE/GQA ratio）、MTP（模块数/层数）。

**详细解释**：

基础架构：`num_hidden_layers(62)`, `hidden_size(3072)`, `intermediate_size(1536)`, `num_attention_heads(48)`, `num_key_value_heads(8)`, `head_dim(128)`

MoE：`num_local_experts(256)`, `num_experts_per_tok(8)`, `scoring_func(sigmoid)`, `use_routing_bias(true)`

Attention：`use_qk_norm(true)`, `qk_norm_type(per_layer)`, `rotary_dim(64)`, `rope_theta(5M)`

MTP：`use_mtp(true)`, `num_mtp_modules(3)`, `mtp_transformer_layers(1)`

量化：`fmt(float8_e4m3fn)`, `weight_block_size([128,128])`, `modules_to_not_convert([gate, bias, lm_head])`

**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

### Q8.13 与 Llama 源码差异

**简短回答**：五大差异：(1) QK Norm per_layer（Llama 无）；(2) MoE FFN 替代 Dense FFN；(3) sigmoid routing + bias + top-8；(4) 3 MTP 模块；(5) Expert dispatch。继承自 Llama：Pre-Norm, SwiGLU, GQA, RMSNorm, RoPE。

M2.7 = Llama 架构 + MoE + QK Norm + MTP。


**详细解释**：这一设计选择需要结合 M2.7 的整体架构来理解。在 62 层 Full Attention + 256 专家 MoE + 192K 上下文的配置下，每个组件参数的选择都受到计算预算、内存带宽和训练稳定性的约束。具体数字和实现细节可参考主报告对应章节。


**面试要点**：从源码/对比的角度理解这一设计——它和其他模型的同类实现有何不同？差异背后的设计理念是什么？

**延伸阅读**：主报告 CH 8（源码映射）；`modeling_minimax_m2.py`

---

## CH 面经. 面试高频题（25 Q）

### 基础题（10 Q）

---

### QM.1 Self-Attention 公式和复杂度

**简短回答**：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$，复杂度 $O(T^2 d)$。除以 $\sqrt{d_k}$ 防止内积过大导致 softmax 饱和。

**详细解释**

步骤：投影 → $QK^T/\sqrt{d}$ → softmax → 加权 V。瓶颈在 $QK^T$ 的 $O(T^2 d)$。M2.7: head_dim=128, T=192K, 单层 ~$4.7 \times 10^{12}$ 次乘法。

**面试官视角**：最基础——能否写出公式、解释 scaling、分析复杂度。

**答题模板**：(1) 公式 (2) $\sqrt{d_k}$ 作用 (3) 复杂度 (4) M2.7 数字：head_dim=128, scaling=1/11.3

---

### QM.2 GQA vs MHA vs MQA

**简短回答**：MHA 每 Q 头独立 KV；GQA 多 Q 共享 KV 组；MQA 所有 Q 共享 1 个 KV。M2.7 GQA ratio=6（48Q/8KV）：KV cache 减少 6×（~292GB→~49GB），质量损失 <1%。

**详细解释**

三种 attention 以 M2.7 的 48 Q 头为例：MHA(48 KV 头)→KV cache 最大，质量最高；GQA ratio=6(8 KV 头)→质量损失 <1%，cache 为 MHA 的 1/6；MQA(1 KV 头)→质量损失 1-3%，cache 最小。GQA 实现：8 个 KV 头通过 `repeat_kv` 复制 6 次匹配 48 Q 头，`num_key_value_groups = 48//8 = 6`。

M2.7 选择 GQA 的理由：192K 上下文 + Full Attention，KV cache 是核心瓶颈。GQA ratio=6 是一个广泛验证的 sweet spot——在「极长上下文」和「高精度 attention」之间取得最优折中。从「总 KV 容量」角度：8 KV 头 × 128 维 = 1024 维表征编码所有历史 token，经验证足以支持 Agent 级长上下文理解。

**面试官视角**：长上下文模型必备知识——GQA 是 192K Full Attention 可行的关键。

**答题模板**：(1) 三种定义和 KV 头关系 (2) 数字 48/8/1 (3) KV cache 计算对比 (4) M2.7 192K 上下文 + Full Attention 必须靠 GQA 压缩 KV cache

---

### QM.3 MoE top-k 路由

**简短回答**：gate(x) → sigmoid 评分 → +bias → 选 top-k expert → 加权求和。M2.7 用 sigmoid + routing bias + top-8。sigmoid 独立评分适合 k 大，softmax 竞争评分适合 k 小。

**答题模板**：(1) 写出流程 (2) sigmoid vs softmax 场景 (3) 负载均衡问题 (4) M2.7 数字：256 experts, top-8

---

### QM.4 RoPE 原理

**简短回答**：对 Q/K 按位置旋转变换编码位置——位置 m 旋转 $m\theta$，内积 $Q_m K_n$ 仅依赖相对位置 (m-n)。M2.7：theta=5M（长上下文），rotary_dim=64（partial RoPE）。

**答题模板**：(1) 旋转矩阵 (2) 相对位置编码性质 (3) theta 作用 (4) M2.7 的 5M theta + partial RoPE

---

### QM.5 KV Cache 原理

**简短回答**：缓存历史 K/V 避免重复计算。显存 = $2 \times L \times H_{\text{kv}} \times d \times T \times 2\text{B}$。M2.7 192K 下约 48.7GB（GQA）。优化：GQA/MQA、KV FP8 量化、MLA、sliding window。

**答题模板**：(1) 为什么需要缓存 (2) 公式 (3) 算 192K 例子 (4) 2-3 个优化策略

---

### QM.6 sigmoid vs softmax 路由

**简短回答**：sigmoid $\sigma(x)$ 独立（各 expert 互不影响），适合 k=8；softmax 竞争归一化（赢者通吃），适合 k=1-2。M2.7 k=8 → sigmoid 允许多 expert 同时高分。

**答题模板**：(1) 两个公式 (2) 独立 vs 竞争 (3) 与 k 的关系 (4) 计算开销

---

### QM.7 RMSNorm vs LayerNorm

**简短回答**：LayerNorm 做减均值+除标准差两项；RMSNorm 只做除 RMS 一项。RMSNorm 更快（~10-15%），无 bias 参数，实验证明对 Transformer 效果无显著差异。M2.7: eps=1e-6, float32 高精度计算。

**答题模板**：(1) 两个公式 (2) RMSNorm 省了什么 (3) 为什么可以省 (4) M2.7 实现细节

---

### QM.8 FP8 E4M3 格式

**简短回答**：8-bit 浮点（4-bit exponent + 3-bit mantissa），动态范围 ±448。vs BF16（7-bit mantissa）：显存减半，精度损失 <1%。vs INT8：有 exponent 表示跨数量级值，不需要额外 scale 校准。

**答题模板**：(1) E4M3 含义 (2) 格式对比表 (3) vs INT8 优势 (4) M2.7 使用 block-wise 128×128

---

### QM.9 AdamW 优化器

**简短回答**：将 weight decay 从梯度更新中解耦：$w_t = w_{t-1} - \eta(\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon) + \lambda w_{t-1})$。Adam 的 L2 正则被自适应学习率分母缩小，AdamW 直接减去 $\eta\lambda w_{t-1}$ 效果更好。

**答题模板**：(1) 公式 (2) AdamW 改动（解耦）(3) 为什么解耦更好 (4) 常见配置 $\beta_1=0.9, \beta_2=0.95$

---

### QM.10 SwiGLU 激活函数

**简短回答**：$\text{SiLU}(W_1 x) \odot (W_3 x)$——门控信号与值信号分离。vs ReLU（硬门控 0/1）：软门控连续梯度更平滑；vs GeLU：计算更简单。现代 LLM 标配。

**答题模板**：(1) 公式 (2) 门控机制 (3) vs ReLU/GeLU (4) M2.7 hidden_act=silu

---

### 进阶题（10 Q）

---

### QM.11 Full vs Lightning Attention trade-off

**简短回答**：Full Attention $O(T^2 d)$ 质量高（长上下文检索 +15-18 pp），Lightning $O(T d^2)$ 快但 GPU 密度低、表达力折损。M2 选 Full（M2.7 继承）因为 Agent 场景必须精确检索——速度可用硬件弥补，检索错误是功能性问题。

**详细解释**

Full Attention：$O = \text{softmax}(QK^T/\sqrt{d})V$，复杂度 $O(T^2 d)$，精确的全局 softmax 归一化。Lightning Attention：$O = \phi(Q)(\phi(K)^T V)$，复杂度 $O(T d^2)$，通过重排计算顺序将 T^2 项变为 d^2 项。两个隐性代价：(1) d×d kernel 矩阵在 GPU 上计算密度低——实际加速远小于理论；(2) 线性化放弃 softmax 非线性变换和全局归一化→表达力折损。

量化证据（paper Table 2）：RULER 128K CWE Full(90.0) vs SWA(72.0) = -18pp；MTOB K-e Bleurt Full(60.0) vs SWA(45.0) = -15pp。SFT 后仍有 -12pp 差距（Table 3）。短上下文（MMLU, MATH）差距微小（<1pp），但长上下文检索的 15-18pp 差距在 Agent 场景不可接受。M2 的 trade-off（M2.7 继承）：牺牲推理速度换取 Agent 任务可靠性。

**答题模板**：(1) 写出两个复杂度公式 (2) Lightning 两个隐性代价 (3) 引用论文数据 RULER CWE -18pp (4) Agent 场景为何 Full Attention 是正确选择（可靠性 > 速度）

---

### QM.12 QK Norm per_layer

**简短回答**：attention 前对 Q、K 做 RMSNorm，防止 $QK^T$ 爆炸。per_layer 让每层独立适应自身 hidden state 分布（浅层局部 vs 深层全局）。参数开销可忽略（62×2×3072 ≈ 760KB）。

**答题模板**：(1) QK Norm 数学作用 (2) per_layer vs 全局共享 (3) 与 rope_theta=5M 配合 (4) 参数开销可忽略

---

### QM.13 routing bias vs auxiliary loss

**简短回答**：aux loss 通过梯度间接均衡（需调 $\alpha$），routing bias 通过 gate 输出加偏置直接调整（与梯度解耦）。bias 不受 optimizer 影响，buffer 非 parameter——将「学路由」和「均衡路由」解耦。

**答题模板**：(1) 两种更新机制 (2) aux loss $\alpha$ 调优困境 (3) bias 解耦优势 (4) buffer vs parameter 设计含义

---

### QM.14 无 shared expert 的 trade-off

**简短回答**：优势：纯路由无退化风险，所有能力由竞争路由分配。劣势：缺通用知识容量。M2.7 弥补：256 expert + k=8 + routing bias。vs V4-Flash（1 shared + 6 routed）：不同设计哲学。

**答题模板**：(1) shared expert 作用 (2) 无 shared 的风险 (3) M2.7 弥补策略 (4) 对比 V4

---

### QM.15 MTP speculative decoding 加速比

**简短回答**：MTP 作内建 draft model，一次 forward 产 3 draft token，主模型 verify。加速比 = $(1+3\alpha)/(1+\beta)$。设 $\alpha=0.6, \beta=0.1$ → 理论 2.55×，实际 1.3-1.8×。

**答题模板**：(1) Draft-Verify-Accept 流程 (2) 加速比公式 (3) acceptance rate 影响因素 (4) 合理加速范围

---

### QM.16 FP8 block size 为什么 128×128

**简短回答**：精度—比 256×256 好，与 64×64 差异小（收益递减）；存储—比 64×64 少 4× 个 scale；硬件—对齐 NVIDIA Hopper WGMMA 指令和 warp(32 threads)。主流 LLM 标准选择。

**答题模板**：(1) block-wise 原理 (2) 精度/存储/硬件三维分析 (3) 128×128 硬件对齐原因

---

### QM.17 GQA ratio 选择

**简短回答**：ratio=Q头/KV头。M2.7 ratio=6（48/8）：KV cache 减少 6×，质量损失 <1%。ratio=1(MHA) 质量最高；ratio=4-8 是 sweet spot；ratio=48(MQA) 质量损失 1-3%。

**答题模板**：(1) ratio 定义 (2) ratio vs 质量曲线 (3) M2.7 192K 特殊需求 (4) 对比 ratio=6 vs 48

---

### QM.18 229.9B 总参 / 9.8B 激活的设计哲学

**简短回答**：MoE 稀疏激活——总参通过加 expert 增大，但推理仅激活 4.3%。激活比 = 9.8B/229.9B ≈ 4.3%。计算量约同规模 dense 的 4.3%，但模型容量远大于 10B dense。

**答题模板**：(1) 总参 vs 激活参 (2) 激活比 4.3% (3) 为什么选这个比例 (4) 对比同规模 dense

---

### QM.19 自我进化范式

**简短回答**：三层运作——训练监控（自动化 loss 检测和 LR 调整）；Agent scaffold 优化（修改 tool-calling 策略）；架构改进（引入 QK Norm、优化 routing bias）。模型参与超参/框架优化，非改自身权重。内部评测提升 ~30%。

**答题模板**：(1) 三种层次 (2) QK Norm 等具体例子 (3) 澄清非改权重 (4) 分析局限性

---

### QM.20 62层×3072 vs 43层×4096

**简短回答**：M2.7「更深更窄」→ 适合长链推理（更多非线性步骤）；V4-Flash「更浅更宽」→ 适合高吞吐（更少串行步骤、更大单层容量）。选择取决于目标场景：Agent vs API 服务。

**答题模板**：(1) 数字对比 (2) 深窄 vs 浅宽优势 (3) 关联设计目标 (4) M2 论文 "mini activations" 哲学

---

### 情境题（5 Q）

---

### QM.21 情境：KV cache 爆了

**回答概要**：紧急——降 max_tokens/batch_size，增加 TP 数；短期——KV FP8 量化（-50% 显存）、prefix caching；长期——sliding window attention、MLA（10× 压缩）、换 GPU（H200→B200）。

M2.7 特有建议：GQA 已是 ratio=6 优化后，最有效是 KV FP8 量化 + 限制有效上下文。

**答题模板**：(1) 三层回答（紧急/短期/长期）(2) 定量估计节省 (3) 质量影响分析 (4) 结合具体模型

---

### QM.22 情境：MoE 路由不均衡 debug

**回答概要**：诊断→统计 expert 使用分布（直方图/Gini系数/top-10 集中度）→ 调 bias 更新（幅度↑、频率↑）→ 调 jitter_noise → 终极手段加 aux loss（α=0.001 起）。

典型模式：少数 expert 承载 80% token → bias 更新不足；部分完全闲置 → bias 初始值被某些 gate 输出 dominate。

**答题模板**：(1) 先诊断再治疗 (2) 按难度递增调试（bias → jitter → aux loss）(3) 定量建议 (4) 不要一上来就加 aux loss

---

### QM.23 情境：设计 100B MoE

**回答概要**：(1) 目标：激活 10-15B，激活比 10-15% → 选 expert 数和 k；(2) 推荐中等粒度：96 experts, k=5，激活约 10-14B；(3) sigmoid 评分 + 1 shared expert + routing bias + GQA ratio=4-8；(4) 小规模实验验证。

**答题模板**：(1) 明确目标 (2) 从激活比反推参数 (3) 具体建议+理由 (4) 验证方法

---

### QM.24 情境：优化 M2.7 推理速度（45→100+ TPS）

**回答概要**：框架层（vLLM/SGLang, 预期 1.5-2×）→ 量化层（INT4+KV FP8, 预期 1.3-1.5×）→ 算法层（MTP speculative, 预期 1.3-1.8×）→ 硬件层（B200, 多实例）。组合理论 4.2×（45→~190 TPS），实际受瓶颈转移限制。

**答题模板**：(1) 四层优化 (2) 每层定量估计 (3) 瓶颈转移问题 (4) 落地优先级

---

### QM.25 情境：M2.7 vs V4-Flash 场景选择

**回答概要**：(a) Agent 长链推理 → M2.7：Full Attention 长上下文检索 +15-18pp，62 层支持多步推理；(b) 高吞吐 chatbot API → V4-Flash：CSA/HCA 稀疏 attention 更快，MQA/MLA 更小 KV cache。

灰色地带：复杂多轮推理 chatbot 可能 M2.7 更好。

**答题模板**：(1) 列表对比关键维度 (2) 场景 a 选择和理由 (3) 场景 b 选择和理由 (4) 承认灰色地带

---

## 附录：Q 号索引

CH 5 (17): Q5.1 sigmoid vs softmax · Q5.2 e_score_correction_bias · Q5.3 top-8 · Q5.4 无共享专家 · Q5.5 jitter_noise · Q5.6 dispatch · Q5.7 256 expert · Q5.8 load balance · Q5.9 gate · Q5.10 buffer vs param · Q5.11 数学流程 · Q5.12 归一化 · Q5.13 SwiGLU · Q5.14 intermediate_size · Q5.15 vs V3 · Q5.16 vs Mixtral · Q5.17 通信开销

CH 6 (12): Q6.1 MTP 原理 · Q6.2 3×1 层 · Q6.3 speculative decoding · Q6.4 token[n+k] · Q6.5 vs V3 · Q6.6 训练 vs 推理 · Q6.7 loss · Q6.8 embedding · Q6.9 1 层 · Q6.10 vs Medusa · Q6.11 禁用 · Q6.12 显存

CH 7 (13): Q7.1 FP8 · Q7.2 block-wise · Q7.3 排除项 · Q7.4 192K · Q7.5 自进化 · Q7.6 训练 · Q7.7 BF16 vs FP8 · Q7.8 dynamic activation · Q7.9 TPS · Q7.10 KV cache · Q7.11 GQA · Q7.12 partial RoPE · Q7.13 theta=5M

CH 8 (13): Q8.1 结构 · Q8.2 Attention · Q8.3 MoE · Q8.4 推理流程 · Q8.5 文件清单 · Q8.6 dispatch · Q8.7 DecoderLayer · Q8.8 forward · Q8.9 ForCausalLM · Q8.10 MLP · Q8.11 RMSNorm · Q8.12 Config · Q8.13 vs Llama

面经 (25): QM.1 Attn · QM.2 GQA · QM.3 MoE top-k · QM.4 RoPE · QM.5 KV cache · QM.6 sigmoid/softmax · QM.7 RMSNorm · QM.8 FP8 · QM.9 AdamW · QM.10 SwiGLU · QM.11 Full/Lightning · QM.12 QK Norm · QM.13 routing bias/aux loss · QM.14 无 shared · QM.15 MTP SD · QM.16 FP8 block · QM.17 GQA ratio · QM.18 总参/激活 · QM.19 自进化 · QM.20 62×3072 vs 43×4096 · QM.21 KV cache 爆 · QM.22 路由 debug · QM.23 设计 100B MoE · QM.24 优化速度 · QM.25 M2.7 vs V4

---

## QA Review 报告

> 审核日期：2026-06-09 · 审核范围：qa.md 全部 150 问

### 一、发现的问题

#### P0 -- 内容准确性（严重）

| # | 位置 | 问题描述 | 严重度 |
|---|------|---------|--------|
| 1 | **Q4.19** (L1620-L1644) | QK Norm 的顺序描述错误。**已修复**：正确顺序为 QK Norm 在 RoPE 之前（投影后立即执行）。 | **P0** (已修复) |

#### P1 -- 格式/元数据错误

| # | 位置 | 问题描述 | 严重度 |
|---|------|---------|--------|
| 2 | QA 文档头部（L2） | 总 Q 数标注为「150 问」，实际统计为 150 问 -- 一致，无问题 | -- |
| 3 | **CH 5-8 分册头部**（L1723） | 标注「78 Q 总计」，但 17(CH5) + 12(CH6) + 13(CH7) + 13(CH8) + 25(面经) = **80 Q**。差了 2 个 Q。 | P1 |
| 4 | **面经章节标题**（L2264） | 标注「面试高频题（23 Q）」，但实际有 QM.1-QM.25 共 **25 题**。差了 2 个 Q。 | P1 |
| 5 | **Q6.2**（L1946） | 原文隐含引用 `MTPBlock` 类，但实际代码（`modeling_minimax_m2.py`）中**不存在独立的 MTPBlock 类**——MTP 模块直接在 `MiniMaxM2Model.forward` 中通过循环创建。主报告 §6.2 已更新为准确描述。 | P1 |
| 6 | **附录索引行**（L2513） | 面经条目仍标注为「面经 (23)」，应改为「面经 (25)」。 | P1 |

#### P2 -- 格式完整性问题

| # | 位置 | 问题描述 | 严重度 |
|---|------|---------|--------|
| 7 | 全文档 | **49 个 Q 仅有「简短回答」**，缺少「详细解释」「面试要点/易混淆」「延伸阅读」三段中的全部三段。这些 Q 集中在 CH 5-8（Q5.15-Q5.17, Q6.5-Q6.12, Q7.6-Q7.13, Q8.1-Q8.13）和面经（QM.3-QM.10, QM.12-QM.25）。相比之下 CH 1-4 的 Q 大多有四段结构。 | P2 |
| 8 | 全文档 | **106 个 Q 缺少「面试要点/易混淆」**（占总数的 70.7%）。CH 5-8 和面经章节几乎全部缺失此字段。 | P2 |
| 9 | 全文档 | **88 个 Q 缺少「延伸阅读」**（占 58.7%），削弱了 QA 作为学习资料的可导航性。 | P2 |
| 10 | 全文档 | **格式一致性差**：CH 1-4 的 Q 普遍有四段完整结构（简短回答 + 详细解释 + 面试要点/易混淆 + 延伸阅读），CH 5-8 的 Q 格式显著缩水。同一份文档内读者体验不统一。 | P2 |

#### P3 -- 深度均衡性问题

| # | 位置 | 问题描述 | 严重度 |
|---|------|---------|--------|
| 11 | CH 4（Attention） | 22 Q -- 是 QA 中最大的单章。Attention 相关话题覆盖度很高，但如 Q4.20（head_dim=128）、Q4.21（短序列意义）、Q4.22（block-sparse）可适当精简合并。 | P3 |
| 12 | CH 6（MTP） | 12 Q -- 偏少。缺少关于 MTP 训练 loss 函数、MTP 与主模型 hidden state 交互机制、MTP 推理时 KV cache 复用等深层问题。 | P3 |
| 13 | CH 7（工程） | 13 Q -- 偏少。FP8 量化细节（如 block-wise scale 的计算过程）、自进化的具体算法流程、训练 pipeline 的资源调度等可补充。 | P3 |
| 14 | CH 8（源码） | 13 Q 中大部分只有一句话回答，缺乏代码引用。源码章节应补充具体行号和代码片段引用。 | P3 |

### 二、LaTeX 正确性

- 所有 `$...$` 花括号平衡检查：**通过**（0 个不平衡行）。
- 未发现渲染风险。

### 三、零上下文可读性测试（5 Q 抽样）

随机抽取 5 个 Q，只看 **Q 标题 + 简短回答**，评估能否独立理解：

| Q | 标题 + 简短回答 | 评分 |
|---|---------------|:---:|
| Q2.6 | 「为什么 M2 放弃了 Lightning Attention？」+ 「消融实验显示长上下文检索差 15-18 pp」 | **通过** -- 核心论断清晰 |
| Q3.4 | 「256 expert, k=8 的设计逻辑？」+ 「256 提供专家容量，k=8 多视角变换，激活比 3.125%」 | **通过** -- 数字和逻辑完整 |
| Q5.5 | 「jitter_noise 在 MoE 训练中的作用？」+ 「训练时乘 [1-ε, 1+ε] 均匀噪声」 | **通过** -- 简洁准确 |
| Q6.2 | 「为什么 3 模块 × 1 层？」+ 「3 个模块提供更多 draft token，1 层保持低延迟」 | **通过** -- 设计权衡清晰 |
| QM.13 | 「routing bias vs auxiliary loss？」+ 「aux loss 通过梯度间接均衡，bias 直接加偏置与梯度解耦」 | **通过** -- 核心差异表述清楚 |

零上下文可读性：**5/5 通过**。简短回答质量整体良好，即使缺少后续段落，核心信息已传达。

### 四、内容准确性抽查（10 Q）

对随机 10 个 Q 的数字/结论与 main-report.md 及 config.json 做交叉验证：

| Q | 验证点 | 结果 |
|---|-------|:---:|
| Q1.3 | 256 experts, k=8, 激活比 ~4.3% (9.8B/229.9B) | ✓ |
| Q2.6 | RULER 128K CWE: Full 90.0 vs SWA 72.0 (= -18pp) | ✓ |
| Q3.3 | GQA ratio=6, KV cache=17% of MHA | ✓ |
| Q3.15 | intermediate_size=1536 = 50% of hidden_size=3072 | ✓ |
| Q4.5 | RULER 128K CWE 差 18pp | ✓ |
| Q5.2 | `e_score_correction_bias` 是 `register_buffer`（源码 L111 确认） | ✓ |
| Q7.4 | rope_theta=5M, partial RoPE (64/128), GQA 8 KV heads | ✓ |
| Q7.10 | 192K KV cache ~48.7 GB | ✓ |
| QM.2 | KV cache 减少 6× (~292GB→~49GB) | ✓ |
| QM.11 | Full Attention 长上下文检索 +15-18pp | ✓ |

准确性抽查：**10/10 通过**（Q4.19 已修复）。

### 五、需要修复的建议（2026-06-09 已执行修复）

1. **【已修复】Q4.19**：QK Norm 顺序已从「RoPE 之后」修正为「RoPE 之前」。源码 `modeling_minimax_m2.py` L41-42（QK Norm）先于 L53（RoPE）执行。同时已修正配套解释——QK Norm 的作用是抑制 Q/K **投影后**的数值范围扩散。

2. **【已修复】元数据计数**：CH 5-8 分册头部「78 Q」→「80 Q」；面经标题「23 Q」→「25 Q」；附录索引行「面经 (23)」→「面经 (25)」。

3. **【已修复】CH1 演进脉络错误**：QA 中所有「M2.7 放弃 Lightning Attention」表述已修正为「M2 放弃 Lightning Attention」。Q2.1 从「四次迭代」更新为「五次迭代」（新增 M1）；Q2.2/Q2.5/Q2.6/Q2.12/Q3.13/Q4.16/QM.11 均已更新。

3. **【建议】CH 5-8 + 面经格式补全**：为仅含简短回答的 49 个 Q 补充「详细解释」，为缺少面试要点的 Q 补充辨识度标签（/）。优先级：先补 CH 6 (MTP)、CH 7 (工程) 和面经的关键 Q。

4. **【建议】深度补充**：CH 6 (MTP) 建议新增 3-5 Q 覆盖训练 loss、hidden state 交互、KV cache 复用；CH 7 (工程) 建议新增 2-3 Q 覆盖 block-wise scale 计算过程、训练 pipeline 资源调度。

5. **【建议】Q6.2 引用更新**：移除对不存在类 `MTPBlock` 的引用，与更新后的主报告 §6.2 保持一致。

### 六、总体质量评分

| 维度 | 得分 | 说明 |
|------|:---:|------|
| 内容准确性 | 9/10 | Q4.19 的 QK Norm 顺序已修复；其余抽查通过 |
| 格式完整性 | 5/10 | 70%+ Q 缺少面试要点，60% Q 缺少延伸阅读，CH 5-8 大量「裸」Q |
| LaTeX 正确性 | 10/10 | 全文档 `$...$` 无花括号不平衡 |
| 零上下文可读性 | 9/10 | 简短回答质量高，5/5 通过独立理解测试 |
| 深度均衡性 | 6/10 | CH 4 (22 Q) 过于细碎，CH 6-8 (12-13 Q) 明显偏薄 |
| 元数据正确性 | 9/10 | Q 计数标注已修复（78→80, 23→25）；代码类名引用已修正 |
| **综合** | **8.0/10** | 内容质量扎实，CH1 演进脉络已修正，QK Norm 顺序已修复 |

核心结论：QA 文档的**知识内容可信**（10 个抽查全通过），**简短回答质量高**（5/5 零上下文可读），**已修复 CH1 演进脉络错误和 QK Norm 顺序**，但**格式一致性差**（CH 1-4 详实 vs CH 5-8 简略）。
