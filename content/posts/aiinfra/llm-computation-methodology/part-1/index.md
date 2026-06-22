+++
date = '2026-06-22'
draft = false
title = 'LLM 系统分析方法论（一）：预备知识与参数分解'
categories = ['aiinfra']
tags = ['computation', 'flops', 'kv-cache', 'parameters', 'methodology', 'inference']
series = 'llm-computation-methodology'
series_order = 1
math = true
summary = '从 config.json 到参数量、FLOPs、KV Cache、推理显存的完整计算推导。覆盖 Full Attention / MSA / MLA / Mamba-2 / SWA / GDN 六种注意力架构。第一篇：矩阵乘法基础与参数分解。'
pinned = true
+++

---

## 目录

- **CH 1** 预备知识：从 config.json 到矩阵乘法
- **CH 2** 参数分解：这个模型有多大
- **CH 1-2** 常见计算错误

> 本系列共 7 篇：[（二）FLOPs 估算](../part-2/) → [（三）KV Cache 与推理显存](../part-3/) → [（四）M3 实战 + Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)

---

## 阅读导航

| 你的目标 | 推荐阅读路径 | 预计时间 |
|---|---|---|
| **快速了解全貌** | Part 1 CH 1.2（FLOPs 基础）→ CH 2.3（Attention 参数）→ [Part 3](../part-3/) CH 4.2（KV cache 公式）→ [Part 4](../part-4/) 附录 C（8 模型速览） | 30 min |
| **学会算参数量** | Part 1 全篇（config 字段 + 符号表 + 4 个案例代入） | 60 min |
| **学会算 FLOPs** | [Part 2](../part-2/)（六种架构的 FLOPs 公式推导与跨架构对比） | 45 min |
| **学会算 KV cache** | [Part 3](../part-3/)（五种架构的 KV cache 公式与验证案例） | 40 min |
| **学会算推理显存** | [Part 3](../part-3/) + [Part 4](../part-4/)（推理显存拆解 + M3 完整推演 + Roofline 延迟分析） | 60 min |
| **学会算训练显存** | [Part 5](../part-5/)（单卡四笔账 → 并行折扣 → ZeRO/FSDP → Checkpointing → Offload → M3 案例） | 60 min |
| **理解通信分析** | [Part 6](../part-6/)（通信物理原理 → 时间线建模 → 六种并行维度通信模式 → M3 step time 推演） | 60 min |
| **分析推理服务性能** | [Part 7](../part-7/)（连续批处理 → PagedAttention → PD 分离 → 推测解码 → Llama-70B 服务分析） | 60 min |
| **查漏补缺** | [Part 4 附录](../part-4/)（config 字段速查 + 符号表 + 公式速查 + 常见错误 Top 12） | 5 min |

各篇依赖关系：Part 1（基础）→ Part 2（FLOPs）/ Part 3（KV Cache）可并行 → Part 4（M3 实战，依赖 1-3）→ Part 5（训练显存，依赖 1-4）→ Part 6（通信，依赖 1-5）→ Part 7（推理服务，依赖 Part 4 的 Roofline + Part 6 的通信分析）。

> **新读者建议**：从 Part 1 CH 1.2（5 分钟搞懂 FLOPs 怎么数）和 [Part 3 CH 4.2](../part-3/)（10 分钟搞懂 KV cache 怎么算）开始——这两节能让你最快建立「能算」的感觉。然后按兴趣选择 Part 2-7 任意章节深入。

---

## CH 1-2 预备知识与参数分解

> **读者定位**：有 Transformer 基础知识的工程师，目标是从 `config.json` 独立推导任意模型的参数量。

---

## CH 1 | 预备知识（续）

### 1.1 读 `config.json`

参数量不是猜出来的——`config.json` 是唯一真相来源。

下表列出影响计算的核心字段，每种架构类型给一个真实案例：

| 字段 | 含义 | Nemotron 3 Ultra | MiniMax M3 | Kimi K2.5 | DeepSeek V4 Flash |
|---|---|---|---|---|---|
| `hidden_size` | 残差流维度 $d$ | 8192 | 6144 | 7168 | 4096 |
| `num_attention_heads` | Q 头数 $H_q$ | 64 | 64 | 64 | 64 |
| `num_key_value_heads` | KV 头数 $H_{kv}$ | **2** | **4** | 64 | **1** |
| `head_dim` | 每 head 维度 $D_h$ | 128 | 128 | (见 MLA) | 512 |
| `intermediate_size` | Dense FFN 中间维 $d_{ff}$ | 5120 | 3072 | 18432 | — |
| `moe_intermediate_size` | MoE 专家中间维 | 5120 | 3072 | 2048 | 2048 |
| `n_routed_experts` | 路由专家数 $E$ | 512 | 128 | 384 | 256 |
| `num_experts_per_tok` | 每 token 激活专家数 $k$ | 22 | 4 | 8 | 6 |
| `vocab_size` | 词表大小 $V$ | 131072 | 200064 | 163840 | 129280 |

**MLA（Multi-head Latent Attention）特有字段**：

| 字段 | Kimi K2.5 | 含义 |
|---|---|---|
| `kv_lora_rank` | 512 | K 和 V 的压缩维度 $d_{kv}$ |
| `q_lora_rank` | 1536 | Q 的压缩维度 $d_q$ |
| `qk_nope_head_dim` | 128 | 每头无位置编码维度 $D_{nope}$ |
| `qk_rope_head_dim` | 64 | 每头 RoPE 维度 $D_{rope}$ |
| `v_head_dim` | 128 | 每头 V 维度 $D_v$ |

> 注：MLA 中 $D_h = D_{nope} + D_{rope}$，对 K2.5 而言 $D_h = 128 + 64 = 192$。GQA/MHA 模型通常直接给 `head_dim`，不需要这几个字段。

**Mamba-2 特有字段**（Nemotron）：

| 字段 | Nemotron | 含义 |
|---|---|---|
| `ssm_state_size` | 128 | SSM 隐状态维度 $N$ |
| `mamba_num_heads` | 256 | Mamba 头数 $H_{mamba}$ |
| `mamba_head_dim` | 64 | Mamba 每头维度 $D_{mamba}$ |
| `n_groups` | 8 | A 矩阵分组数（Mamba-2 的多头扩展） |
| `conv_kernel` | 4 | 1D 深度卷积核大小 |
| `expand` | 2 | 内部扩展因子（$d_{inner} = 2 \times d$） |

**Vision 相关字段**（M3）：

| 字段 | M3 值 | 含义 |
|---|---|---|
| `vision_config.hidden_size` | 1280 | ViT 隐藏维度 |
| `vision_config.num_attention_heads` | 16 | ViT 注意力头数 |
| `vision_config.num_hidden_layers` | 32 | ViT 层数 |
| `vision_config.intermediate_size` | 5120 | ViT MLP 中间维 |
| `vision_config.patch_size` | 14 | Patch 大小 |
| `vision_config.image_size` | 2016 | 输入图像尺寸 |

**MoE 相关补充字段**：

| 字段 | 含义 | 例子 |
|---|---|---|
| `moe_latent_size` | Nemotron 低秩投影维度 | 2048 |
| `moe_shared_expert_intermediate_size` | 共享专家中间维 | Nemotron: 10240 |
| `dense_intermediate_size` | Dense 层 FFN 中间维（M3 前 3 层） | M3: 12288 |
| `shared_intermediate_size` | 共享专家中间维（M3） | M3: 3072 |
| `n_shared_experts` | 共享专家数量 | 通常为 1 |
| `scoring_func` | 路由评分函数 | `sigmoid` / `softmax` |
| `tie_word_embeddings` | 输入/输出 Embedding 是否共享权重 | `false`（多数大模型不共享） |

**实战提示**：打开 `config.json` 后，先把上述字段圈出来列成一个小表。后续所有计算都不需要看源码——只看这个表就能推出 95% 以上的参数量。

---

### 1.2 矩阵乘法 FLOPs 是怎么算的

建立“矩阵乘法的计算量直觉”。参数量是“存了多少数”，FLOPs 是“每次前向要算多少步”——两者是同一个硬币的两面。

#### 1.2.1 基本定义

现代深度学习框架中，一次 Multiply-Accumulate（MAC，乘加）计为 **2 FLOPs**（1 次乘法 + 1 次加法）。

矩阵乘法 $C = A \cdot B$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$：

$$\text{FLOPs} = 2 \cdot m \cdot n \cdot k$$

#### 1.2.2 完整代入案例

假设我们在计算 Attention 层的 Q 投影：

$$\text{hidden\_states} \in \mathbb{R}^{1 \times 4096 \times 6144}$$

$$W_Q \in \mathbb{R}^{6144 \times 6144}$$

$$\text{FLOPs}_{Q} = 2 \times 4096 \times 6144 \times 6144 = 2 \times 4096 \times 37,748,736$$

$$= 309{,}237{,}645{,}312 \approx 309 \text{ GFLOPs}$$

$m \times n$ 是输出矩阵的大小（$4096 \times 6144$），每个输出元素需要做 $k=6144$ 次乘加。把这 2500 万个输出元素每个都做 6144 次运算，再 ×2，就是总的浮点运算次数。

#### 1.2.3 分解技巧

一个大矩阵乘法的 FLOPs 可以按“**输出形状 × 2 × 公共维度**”来记：

- `hidden [B, S, d] × W [d, d_out]` $\to$ FLOPs = $2 \cdot B \cdot S \cdot d \cdot d_{out}$
- `Q [B, H, S, D] × K^T [B, H, D, S]` $\to$ FLOPs = $2 \cdot B \cdot H \cdot S \cdot S \cdot D$
- `attn [B, H, S, S] × V [B, H, S, D]` $\to$ FLOPs = $2 \cdot B \cdot H \cdot S \cdot S \cdot D$

其中 $B$ 是 batch size，$S$ 是序列长度，$H$ 是头数，$D$ 是每头维度。

---

### 1.3 einsum 是什么

读懂 PyTorch/Flax 代码中 `einsum` 的维度缩并记法。绝大多数模型源码用 einsum 写注意力计算，看不懂 einsum 就看不懂代码。

#### 1.3.1 基本语法

```python
torch.einsum("bhqk,bhkv->bhqv", Q, K_T, V)
```

规则：
- `->` 左边是**输入**的维度标签，逗号分隔多个输入
- `->` 右边是**输出**的维度标签
- 出现在左边但不出现在右边的标签 = **被求和缩并掉的维度**
- 字母顺序任意，但同一字母在同一个输入中只能出现一次

#### 1.3.2 具体案例

```python
# Q: [Batch=2, Heads=16, Seq_q=4096, Dim=64]  -> 标签 bhqk
# K: [Batch=2, Heads=16, Seq_k=4096, Dim=64]  -> 标签 bhkv  
# (注意 K 的最后一维用了与 Q 不同的标签 v，Q 最后一维是 k)

# einsum("bhqk,bhkv->bhqv", Q, K)
# 缩并维度：k（Q 的第 4 维 和 K 的第 4 维做点积）
# 保留维度：b, h, q, v
# 输出形状: [2, 16, 4096, 4096]  -> 即注意力分数矩阵
```

einsum 就是“对着字母做操作”——同名字母在左边多个输入中出现就做点积（乘法+求和），只在一个输入中出现的字母保留到输出。你不需要想象循环嵌套，只需要追踪每个字母的维度大小。

#### 1.3.3 常见 Attention 计算模式

| einsum 模式 | 含义 | 输入形状 | 输出形状 |
|---|---|---|---|
| `bhqk,bhkv->bhqv` | QK 点积求 attention score | `[B,H,S_q,D] × [B,H,S_k,D]` | `[B,H,S_q,S_k]` |
| `bhqv,bhvd->bhqd` | Attention × V 加权 | `[B,H,S_q,S_k] × [B,H,S_k,D]` | `[B,H,S_q,D]` |
| `bnhd,hdo->bno` | Output 投影（合并 heads） | `[B,H,S,D] × [H,D,d]` | `[B,S,d]` |
| `bsi,io->bso` | 标准 Linear 层 | `[B,S,in] × [in,out]` | `[B,S,out]` |

**实战提示**：在 PyTorch 代码中遇到 `einsum` 时，第一步不是理解计算逻辑，而是写出每个字母代表的维度大小——然后你就可以心算 FLOPs 了。

---

### 1.4 符号约定

统一本文后续所有公式的符号，避免混淆。RoPE 的 head_dim 和 V 的 head_dim 在 MLA 中是两个不同的值，不约定清楚会算错。

| 符号 | 含义 | Nemotron | M3 | K2.5 |
|---|---|---|---|---|
| $B$ | Batch size | — | — | — |
| $S_q$ / $S_k$ / $S_v$ | Query / Key / Value 序列长度（prefill 时 $S_q = S_k$） | — | — | — |
| $L$ | 层数（不含 MTP/Vision） | 60 | 60 | 61 |
| $d$ | `hidden_size`，残差流维度 | 8192 | 6144 | 7168 |
| $V$ | `vocab_size`，词表大小 | 131072 | 200064 | 163840 |
| $H_q$ / $H_{kv}$ | Q 头数 / KV 头数 | 64 / 2 | 64 / 4 | 64 / 64 |
| $D_h$ | `head_dim`，每头维度（GQA/MHA 中） | 128 | 128 | — |
| $D_{nope}$ / $D_{rope}$ | MLA 中无位置/有位置编码维度 | — | — | 128/64 |
| $D_v$ | MLA 中每头 V 维度 | — | — | 128 |
| $d_{kv}$ / $d_q$ | MLA 压缩维度（`kv_lora_rank` / `q_lora_rank`） | — | — | 512/1536 |
| $d_{ff}$ | FFN 中间维度 | 5120 | 3072 (MoE) / 12288 (Dense) | 2048 (MoE) / 18432 (Dense) |
| $d_{moe\_ff}$ | `moe_intermediate_size`，MoE 专家中间维 | 5120 | 3072 | 2048 |
| $E$ | `n_routed_experts`，路由专家总数 | 512 | 128 | 384 |
| $k$ | `num_experts_per_tok`，每 token 激活专家数 | 22 | 4 | 8 |
| $H_{mamba}$ / $D_{mamba}$ | Mamba 头数/每头维度 | 256/64 | — | — |
| $N$ | `ssm_state_size`，SSM 状态维度 | 128 | — | — |
| $T$ | 序列长度（tokens） | — | — | — |
| bytes | 每个参数的字节数（BF16=2, FP8=1, FP4=0.5） | — | — | — |

**关键区分**：
- $D_h$（GQA/MHA）：一个值，Q、K、V 的 head_dim 相同
- $(D_{nope}, D_{rope}, D_v)$（MLA）：三个独立值，Q/K 的维度是 $D_{nope}+D_{rope}$，V 的维度是 $D_v$

---

### 1.5 Bytes 换算

参数个数 $\to$ 显存占用的转换。算完参数量不乘 bytes 等于白算——内存是按字节分配的，不是按“个数”。

| 精度 | 字节/参数 | 典型应用场景 |
|---|---|---|
| FP32 | 4 | 训练主权重（full precision） |
| BF16 | 2 | 推理权重、训练前向（主流默认） |
| FP16 | 2 | 部分训练框架 |
| FP8 (E4M3) | 1 | 推理量化、部分训练（如 DeepSeek V4 的 `quantization_config`） |
| INT8 | 1 | 推理量化 |
| INT4 / NVFP4 | 0.5 | 极端推理量化（如 Nemotron 的 NVFP4 训练） |
| FP4 (E2M1) | 0.5 | 权重级极限压缩 |

**换算公式**：

$$\text{Memory (GiB)} = \frac{\text{Params} \times \text{bytes}}{1024^3} = \frac{\text{Params} \times \text{bytes}}{1{,}073{,}741{,}824}$$

> 本文使用 GiB（2³⁰ bytes）而非 GB（10⁹ bytes）用于显存计算，因为 GPU 显存以 2 的幂次分配。1 GiB ≈ 1.074 GB。

**实战案例**：Nemotron 3 Ultra 的 550B 参数以 BF16 存储：

$$550 \times 10^9 \times 2 \text{ bytes} = 1.1 \times 10^{12} \text{ bytes} \approx 1024 \text{ GiB} \approx 1 \text{ TiB}$$

如果换成 NVFP4 推理（仅权重部分）：

$$550 \times 10^9 \times 0.5 \text{ bytes} = 275 \text{ GiB}$$

**这一节内容少，但每次计算都要用到——建议手写贴在显示器旁边。**

---

## CH 2 | 参数分解

### 2.1 通用原理

建立“参数就是矩阵元素数”的底层逻辑。所有花哨的架构（GQA、MLA、MoE、Mamba）最终都可以归结为“有多少个权重矩阵，每个矩阵的形状是什么”。

#### 核心公式

$$\text{Params} = \sum_{W \in \text{所有权重矩阵}} \text{size}(W)$$

其中 $\text{size}(W) = \text{in\_features} \times \text{out\_features}$（不含 bias，大模型中 bias 通常为 `False` 或可忽略）。

#### 一级近似

一个 Decoder-only Transformer 的总参数主要由以下模块构成：

$$\text{Params}_{\text{total}} = \text{Params}_{\text{embed}} + L \times \text{Params}_{\text{attn}} + L_{\text{dense}} \times \text{Params}_{\text{ffn\_dense}} + L_{\text{moe}} \times \text{Params}_{\text{moe}} + \text{Params}_{\text{norm}} + \text{Params}_{\text{head}} + \text{Params}_{\text{other}}$$

参数量就是“把所有权重矩阵的元素数加起来”。Embedding 是一个大矩阵，每层有一个 Attention（QKV+O）和一个 FFN（gate+up+down 或 up+down），MoE 把 FFN 复制了 $E$ 份，Mamba 把 Attention 换成了自己的 in/out/ssm 参数。

---

### 2.2 Embedding 层

计算输入/输出 Embedding 的参数量。对 100K+ 词表的模型，Embedding 就占了 ~1B 参数——不是小数目。

#### 公式

$$\text{Params}_{\text{embed\_in}} = V \times d$$

$$\text{Params}_{\text{embed\_out}} = \begin{cases} V \times d & \text{若 } \texttt{tie\_word\_embeddings} = \texttt{false} \\ 0 & \text{若 } \texttt{tie\_word\_embeddings} = \texttt{true} \text{（共享输入权重）} \end{cases}$$

#### 案例

**Nemotron 3 Ultra**：$V = 131072$（`vocab_size`），$d = 8192$（`hidden_size`），`tie_word_embeddings = false`：

$$\text{Params}_{\text{embed\_in}} = 131{,}072 \times 8192 = 1{,}073{,}741{,}824 \approx 1.07\text{B}$$

$$\text{Params}_{\text{embed\_out}} = 131{,}072 \times 8192 = 1.07\text{B}$$

$$\text{Params}_{\text{embed\_total}} \approx 2.15\text{B}$$

**MiniMax M3**：$V = 200{,}064$，$d = 6144$，`tie_word_embeddings = false`：

$$\text{Params}_{\text{embed\_in}} = 200{,}064 \times 6144 = 1{,}229{,}193{,}216 \approx 1.23\text{B}$$

Embedding 就是一个查表操作——$V$ 行，每行是一个 $d$ 维向量。输入和输出通常各有一个独立的表（因为 `tie_word_embeddings=false` 在大模型中很常见），所以 Embedding 部分大致是 $2Vd$ 个参数。131K 词表 × 8K 维度 ≈ 1B 参数一块，两块就是 2B。

---

### 2.3 Attention 参数

计算 Q、K、V、O 四个投影矩阵的参数。Attention 的计算量由序列长度主导，但**参数量只由维度决定**——理解这一点才能区分“参数”和“FLOPs”两个概念。

#### 2.3.1 标准 MHA（Multi-Head Attention）

每个头独立的 Q、K、V，无 GQA 压缩。

$$\text{Params}_{Q} = d \times H_q \times D_h$$

$$\text{Params}_{K} = d \times H_{kv} \times D_h$$

$$\text{Params}_{V} = d \times H_{kv} \times D_h$$

$$\text{Params}_{O} = H_q \times D_h \times d$$

$$\text{Params}_{\text{MHA}} = d \times H_q \times D_h + 2 \times d \times H_{kv} \times D_h + H_q \times D_h \times d$$

当 $H_{kv} = H_q$ 时（纯 MHA，无 GQA）：

$$\text{Params}_{\text{MHA}} = 4 \times d \times H_q \times D_h = 4 \times d^2 \quad (\text{若 } H_q \times D_h = d)$$

**Kimi K2.5 的 Layer 0**（全 MHA，$d=7168$，$H_q=H_{kv}=64$，$D_h = D_{nope} + D_{rope} = 192$）：

$$\text{Params}_{Q} = 7168 \times 64 \times 192 = 88{,}080{,}384 \approx 88.1\text{M}$$

$$\text{Params}_{K} = 7168 \times 64 \times 192 = 88.1\text{M}$$

$$\text{Params}_{V} = 7168 \times 64 \times 128 = 58{,}720{,}256 \approx 58.7\text{M}$$

$$\text{Params}_{O} = 64 \times 128 \times 7168 = 58.7\text{M}$$

$$\text{Params}_{\text{MHA, per layer}} \approx 293.6\text{M}$$

注意 V 的头维度是 128（`v_head_dim`），不是 192——这是 MLA 的设计，即使在不压缩的 Layer 0 也遵循同样的维度约定。

MHA 就是四个大矩阵——Q 把 d 维投影到 d 维（$H \times D = d$），K 也一样，V 也一样，O 把 d 维映射回 d 维。`4 × d²` 就是每层 Attention 的“起步价”。

#### 2.3.2 GQA（Grouped Query Attention）

GQA 的核心：Q 头数不变，K 和 V 头数减少。**K、V 矩阵“变窄”了**。

$$\text{Params}_{\text{GQA}} = d \times H_q \times D_h + 2 \times d \times H_{kv} \times D_h + H_q \times D_h \times d$$

**Nemotron 3 Ultra**（GQA 32:1，$d=8192$，$H_q=64$，$H_{kv}=2$，$D_h=128$）：

$$\text{Params}_{Q} = 8192 \times 64 \times 128 = 67{,}108{,}864 \approx 67.1\text{M}$$

$$\text{Params}_{K} = 8192 \times 2 \times 128 = 2{,}097{,}152 \approx 2.1\text{M}$$

$$\text{Params}_{V} = 8192 \times 2 \times 128 = 2.1\text{M}$$

$$\text{Params}_{O} = 64 \times 128 \times 8192 = 67.1\text{M}$$

$$\text{Params}_{\text{GQA, per layer}} \approx 138.4\text{M}$$

对比全 MHA（$H_{kv}=64$）的 $4 \times 8192^2 = 268.4\text{M}$，GQA 32:1 将 Attention 参数量压到了 **48%**。

**MiniMax M3**（GQA 16:1，$d=6144$，$H_q=64$，$H_{kv}=4$，$D_h=128$）：

$$\text{Params}_{Q} = 6144 \times 64 \times 128 = 50{,}331{,}648 \approx 50.3\text{M}$$

$$\text{Params}_{K} = 6144 \times 4 \times 128 = 3{,}145{,}728 \approx 3.1\text{M}$$

$$\text{Params}_{V} = 6144 \times 4 \times 128 = 3.1\text{M}$$

$$\text{Params}_{O} = 64 \times 128 \times 6144 = 50.3\text{M}$$

$$\text{Params}_{\text{GQA, per layer}} \approx 107.0\text{M}$$

GQA 公式速记：Q 矩阵 $d \times d$（因为 $H_q \times D_h = d$），K 矩阵 $d \times (H_{kv} \times D_h)$——“变窄”的矩阵，V 同理，O 矩阵 $d \times d$。GQA 比 MHA 省的就是 $K$、$V$ 投影省出来的 $2 \times d \times (H_q - H_{kv}) \times D_h$ 个参数。

> ⚠️ **最容易犯的错误**：GQA 中 $H_{kv} \times D_h \neq d$（因为 $H_{kv} < H_q$，而 $H_q \times D_h = d$）。K/V 投影的输出维度不是 `hidden_size`，而是 `num_kv_heads × head_dim`。很多人直接写 `d × d` 给 K/V——这是 MHA 才对的。你可以在心里验证：Nemotron GQA 32:1 → K 投影 = $8192 \times (2 \times 128) = 8192 \times 256 = 2.1\text{M}$，远小于 Q 投影的 $8192^2 = 67.1\text{M}$。

MHA 里 K 和 V 也是 $d \times d$ 的方阵，GQA 把它们变窄了——因为 KV 头数只有 Q 头数的几十分之一，所以 KV 投影矩阵的列数就是 Q 投影的几分之一。O 投影不受影响，因为输出维度（$H_q \times D_h = d$）不变。

#### 2.3.3 MLA（Multi-head Latent Attention）

**这节是最复杂的部分**。MLA 的核心思想：**不在高维空间存 K 和 V，而是先压缩到一个低维“潜空间”，再从潜空间解压恢复**。这就像一个 zip 压缩——存储/传输时用压缩格式，使用时解压。

MLA 将 K 分解为两部分：
- **nope 分量**（No Position Encoding）：128 维，可压缩（通过潜空间）
- **rope 分量**（RoPE Position Encoding）：64 维，**不可压缩**（RoPE 必须在全维度上旋转，不能压缩后旋转）

##### 矩阵清单

以 Kimi K2.5 为例：$d=7168$，$d_{kv}=512$，$d_q=1536$，$H=64$，$D_{nope}=128$，$D_{rope}=64$，$D_v=128$。

| 矩阵 | 形状 | 含义 |
|---|---|---|
| $W_{kv\_a}$ | $d \times (d_{kv} + D_{rope})$ | K 压缩：hidden $\to$ 压缩 KV + RoPE 分量 |
| $W_{kv\_b}$ | $d_{kv} \times H \times (D_{nope} + D_v)$ | K/V 解压：压缩空间 $\to$ 所有 head 的 nope K + V |
| $W_{q\_a}$ | $d \times d_q$ | Q 压缩：hidden $\to$ 压缩 Q |
| $W_{q\_b}$ | $d_q \times H \times D_{nope}$ | Q 解压：压缩 Q $\to$ 所有 head 的 nope Q |
| $W_{q\_rope}$ | $d \times H \times D_{rope}$ | Q RoPE 分量：hidden $\to$ 所有 head 的 rope Q（不压缩） |
| $W_o$ | $H \times D_v \times d$ | 输出投影 |

##### 逐项代入计算

**(1) KV 压缩投影 $W_{kv\_a}$**

$$\text{Params}_{kv\_a} = d \times (d_{kv} + D_{rope}) = 7168 \times (512 + 64) = 7168 \times 576 = 4{,}128{,}768 \approx 4.13\text{M}$$

这个投影将 hidden 映射为 512 维的压缩潜向量 + 64 维的 RoPE 分量。后 64 维不参与压缩，直接作为 K 的 rope 部分使用。

**(2) K/V 解压投影 $W_{kv\_b}$**

$$\text{Params}_{kv\_b} = d_{kv} \times H \times (D_{nope} + D_v) = 512 \times 64 \times (128 + 128) = 512 \times 64 \times 256 = 8{,}388{,}608 \approx 8.39\text{M}$$

从 512 维潜空间“解压”出 64 个 head、每个 head 的 128 维 nope K 和 128 维 V。

**(3) Q 压缩投影 $W_{q\_a}$**

$$\text{Params}_{q\_a} = d \times d_q = 7168 \times 1536 = 11{,}010{,}048 \approx 11.01\text{M}$$

**(4) Q nope 解压投影 $W_{q\_b}$**

$$\text{Params}_{q\_b} = d_q \times H \times D_{nope} = 1536 \times 64 \times 128 = 12{,}582{,}912 \approx 12.58\text{M}$$

**(5) Q RoPE 直投投影 $W_{q\_rope}$**

$$\text{Params}_{q\_rope} = d \times H \times D_{rope} = 7168 \times 64 \times 64 = 29{,}360{,}128 \approx 29.36\text{M}$$

为什么 Q 的 rope 部分不压缩？因为 RoPE 是按维度旋转的——如果先压缩再解压，旋转操作会被破坏。所以 rope 部分直接从 hidden 维度投影，不经过压缩/解压。

**(6) 输出投影 $W_o$**

$$\text{Params}_{o} = H \times D_v \times d = 64 \times 128 \times 7168 = 58{,}720{,}256 \approx 58.72\text{M}$$

**(7) 单层 MLA 总计**

$$\text{Params}_{\text{MLA, per layer}} \approx 4.13 + 8.39 + 11.01 + 12.58 + 29.36 + 58.72 = 124.19\text{M}$$

##### MLA vs MHA vs GQA 对比

假设同样 $d=7168$，$H_q=64$：

| 架构 | KV 头数 | Attention 参数/层 | 相对 MHA |
|---|---|---|---|
| MHA | 64 | ~293.6M | 100% |
| GQA 8:1 | 8 | ~165.2M | 56% |
| GQA 16:1 | 4 | ~143.7M | 49% |
| **MLA (K2.5)** | 64 (压缩后) | **~124.2M** | **42%** |

MLA 将 Attention 参数压缩到了全 MHA 的 42%，同时保持了 64 个独立 KV 头的能力（因为解压发生在注意力计算前）。

MLA 就像“快递打包”——把 64 个 KV 头的内容先折成一个小包裹（512 维潜向量）运输（存储/KV cache），到了目的地（注意力计算前）再拆开还原。包裹小所以运费低（KV cache 小），但内容还原后跟原来差不多（注意力质量高）。额外代价是打包（$W_{kv\_a}$）和拆包（$W_{kv\_b}$）的少量参数。

---

### 2.4 FFN 参数：SwiGLU vs ReLU$^2$

区分两种主流 FFN 的门控机制，正确计算参数量。SwiGLU 比 ReLU$^2$ 多 50% 参数——不知道这个区别会把 MoE 参数量算错三分之一。

#### 2.4.1 ReLU$^2$（Nemotron 风格的“无门控 FFN”）

只有 up 和 down 两个矩阵：

$$\text{FFN}(\mathbf{x}) = W_{down} \cdot \text{ReLU}(\mathbf{x} \cdot W_{up})^2$$

$$\text{Params}_{\text{ReLU}^2} = 2 \times d \times d_{ff}$$

Nemotron 共享专家（$d=8192$，$d_{ff}=10240$）：

$$\text{Params} = 2 \times 8192 \times 10240 = 167{,}772{,}160 \approx 167.8\text{M}$$

#### 2.4.2 SwiGLU（标准门控 FFN）

有三个矩阵：gate、up、down。

$$\text{FFN}(\mathbf{x}) = W_{down} \cdot (\text{SiLU}(\mathbf{x} \cdot W_{gate}) \odot \mathbf{x} \cdot W_{up})$$

$$\text{Params}_{\text{SwiGLU}} = 3 \times d \times d_{ff}$$

K2.5 路由专家（$d=7168$，$d_{ff}=2048$）：

$$\text{Params} = 3 \times 7168 \times 2048 = 44{,}040{,}192 \approx 44.0\text{M}$$

同维度下 ReLU$^2$ 只需 $2 \times 7168 \times 2048 \approx 29.4\text{M}$。

#### 2.4.3 对比表

| 激活函数 | 矩阵数 | 公式 | 相对参数 |
|---|---|---|---|
| ReLU$^2$ | 2 | $2 \times d \times d_{ff}$ | 100% |
| SwiGLU | 3 | $3 \times d \times d_{ff}$ | **150%** |
| Non-gated SwiGLU (M3) | 2 (合并) | $d \times 2d_{ff} + d_{ff} \times d$ | **100%** (等价于 $3 \times d \times d_{ff}$) |

> **注意**：M3 的 "non-gated SwiGLU" 将 gate 和 up 合并为 `gate_up_proj(d → 2×d_ff)` 一个矩阵，参数量 $d \times 2d_{ff} = 2 \times d \times d_{ff}$，与分离的 gate + up 的总参数相同（$d \times d_{ff} + d \times d_{ff} = 2 \times d \times d_{ff}$）。区别是**计算路径**而非参数数量。

ReLU$^2$ 是"一个门+一条路"，SwiGLU 是"两个独立门汇合到一条路"。多一个门就多 $d \times d_{ff}$ 个参数。检查 `config.json` 的 `hidden_act` 字段——如果是 `silu` 或 `swigluoai`，大概率是 SwiGLU（3 个矩阵）；如果是 `relu2`，就是 ReLU$^2$（2 个矩阵）。

---

### 2.5 MoE 参数

计算 MoE 层的完整参数量——路由器 + 所有专家 + 共享专家。MoE 占模型总参数的 90%+，算错一个专家的维度会导致总参估算差出几十 B。

#### 2.5.1 路由器

对于最简单的 sigmoid 路由（M3、Nemotron）：

$$\text{Params}_{\text{router}} = d \times E$$

M3（$d=6144$，$E=128$）：$\text{Params}_{\text{router}} = 6144 \times 128 = 786{,}432 \approx 0.79\text{M}$

Nemotron（$d=8192$，$E=512$）：$\text{Params}_{\text{router}} = 8192 \times 512 = 4{,}194{,}304 \approx 4.2\text{M}$

> 路由器还可以包含 `e_score_correction_bias`（$E$ 个标量，可忽略）。更复杂的路由（如 DeepSeek V4 的 hash routing）参数更大，但原理相同——最终是一个 $d \times \text{num\_experts}$ 的矩阵。

#### 2.5.2 总 MoE 参数量

$$\text{Params}_{\text{MoE, per layer}} = \underbrace{d \times E}_{\text{router}} + \underbrace{E \times \text{Params}_{\text{expert}}}_{\text{所有路由专家}} + \underbrace{\text{Params}_{\text{shared}}}_{\text{共享专家}}$$

#### 2.5.3 完整案例：MiniMax M3

M3 有 57 个 MoE 层（第 3-59 层），每层 128 个路由专家 + 1 个共享专家。

**路由专家**（SwiGLU，$d_{ff}=3072$）：

$$\text{Params}_{\text{expert}} = 3 \times 6144 \times 3072 = 56{,}623{,}104 \approx 56.62\text{M}$$

实际上 M3 用 non-gated SwiGLU（gate/up 合并）：$6144 \times (2 \times 3072) + 3072 \times 6144 = 56.62\text{M}$，结果相同。

**每层所有路由专家**：

$$\text{Params}_{\text{all\_experts\_per\_layer}} = 128 \times 56.62\text{M} = 7{,}247{,}757{,}312 \approx 7.25\text{B}$$

**共享专家**（每层 1 个）：

$$\text{Params}_{\text{shared}} = 56.62\text{M}$$

**每层 MoE 总计**：

$$\text{Params}_{\text{MoE, per layer}} = 0.79\text{M} + 7.25\text{B} + 56.62\text{M} = 7.31\text{B}$$

**57 层 MoE 总计**：

$$\text{Params}_{\text{MoE, 57 layers}} = 57 \times 7.31\text{B} \approx 416.4\text{B}$$

这占了 M3 总参数（~428B）的 **97%**。

MoE 的本质是“把 FFN 复制 E 份”。每份是一个完整的 FFN，参数量 = SwiGLU 的 $3 \times d \times d_{ff}$（或 ReLU$^2$ 的 $2 \times d \times d_{ff}$）。128 份 × 56M/份 × 57 层 ≈ 400B。路由器本身才 0.79M/层——跟专家的参数量比相当于“一根羽毛跟一头大象”。

#### 2.5.4 Nemotron 的 LatentMoE（低秩专家）

Nemotron 的 MoE 有个特殊设计：专家在**低秩空间** $d_{latent}=2048$ 中计算，而非全维度 8192。

**每层 MoE 结构**：

- 路由器：$8192 \times 512 = 4.2\text{M}$
- 低秩投影入：$8192 \times 2048 = 16.8\text{M}$
- 低秩投影出：$2048 \times 8192 = 16.8\text{M}$
- 路由专家（ReLU$^2$，在 latent 空间）：$2 \times 2048 \times 5120 = 20.97\text{M}$/专家
- 512 专家：$512 \times 20.97\text{M} = 10.74\text{B}$
- 共享专家（ReLU$^2$，在 full 空间）：$2 \times 8192 \times 10240 = 167.8\text{M}$

**每层 MoE 总计**：$\approx 10.94\text{B}$

48 层合计：$\approx 525.4\text{B}$

Nemotron 的 MoE 是“先降维再升维”的低秩设计——hidden 从 8192 压到 2048，在 2048 维空间里做 512 个专家计算，再升回 8192。这比直接在 8192 维做专家（每个专家 $2 \times 8192 \times 5120 = 83.9\text{M}$）节省了 **75%** 的参数量——代价是低秩压缩的信息损失。

---

### 2.6 Mamba-2 参数（Nemotron）

计算 Mamba-2 SSD 层的参数量。Nemotron 有 48 个 Mamba 层——它不是 Attention，不能套用 QKV 公式。

#### 2.6.1 维度推导

从 `config.json` 直接读到：
- $d = 8192$
- $H_{mamba} = 256$（`mamba_num_heads`）
- $D_{mamba} = 64$（`mamba_head_dim`）
- $N = 128$（`ssm_state_size`）
- $n_{groups} = 8$（`n_groups`）
- kernel = 4（`conv_kernel`）
- expand = 2（`expand`）

推导内部维度：
- $d_{inner} = \text{expand} \times d = 2 \times 8192 = 16384$
- 验证：$H_{mamba} \times D_{mamba} = 256 \times 64 = 16384$ ← 自洽
- $d_{conv} = d_{inner} + 2 \times n_{groups} \times N = 16384 + 2 \times 8 \times 128 = 16384 + 2048 = 18432$

#### 2.6.2 逐项参数

**(1) `in_proj`（输入投影，一投多产）**

Mamba 的 `in_proj` 一次性投影出所有需要的分量：$x$、$z$、$B$、$C$、$\Delta$ 的参数。

$$\text{Params}_{\text{in\_proj}} = d \times (d_{inner} + d_{conv} + H_{mamba}) = 8192 \times (16384 + 18432 + 256)$$

$$= 8192 \times 35072 = 287{,}309{,}824 \approx 287.3\text{M}$$

**分解**：35072 = 16384（$x$ 和 $z$ 各 $d_{inner}$，共 $2 \times d_{inner}$）+ 18432（$B$ 和 $C$ 的 $d_{conv}$）+ 256（$\Delta$ 的 $H_{mamba}$）。

等等，让我重新梳理。$2 \times d_{inner} = 32768$，$2 \times n_{groups} \times N = 2048$（B 和 C），$H_{mamba} = 256$（Δ）。合计 $32768 + 2048 + 256 = 35072$。自洽。

**(2) `conv1d`（深度卷积）**

$$\text{Params}_{\text{conv1d}} = d_{conv} \times \text{kernel} + d_{conv} = 18432 \times 4 + 18432 = 92{,}160 \approx 0.09\text{M}$$

深度卷积（每个通道独立卷积核），参数极少。

**(3) `A_log` 和 `D` 和 `dt_bias`（SSM 内部标量）**

$$\text{Params}_{A\_log} = H_{mamba} = 256$$

$$\text{Params}_{D} = H_{mamba} = 256$$

$$\text{Params}_{dt\_bias} = H_{mamba} = 256$$

三个加起来不到 1000 个参数——完全可以忽略。

**(4) `out_proj`（输出投影）**

$$\text{Params}_{\text{out\_proj}} = d_{inner} \times d = 16384 \times 8192 = 134{,}217{,}728 \approx 134.2\text{M}$$

#### 2.6.3 单层 Mamba-2 汇总

| 组件 | 参数量 |
|---|---|
| `in_proj` | 287.3M |
| `conv1d` | 0.09M |
| `A_log` + `D` + `dt_bias` | ~0.001M |
| `out_proj` | 134.2M |
| **单层合计** | **~421.6M** |

48 层 Mamba-2 合计：$48 \times 421.6\text{M} \approx 20.2\text{B}$

Mamba 的 `in_proj` 是“一拖多”——一个矩阵输出 5 件事（x, z, B, C, Δ），所以它特别胖（8192 × 35072 = 287M）。`out_proj` 再把它收回来。其余部分（卷积、状态标量）几乎不占参数。对比一下：Nemotron 的 Attention 层（138M）比 Mamba 层（422M）**便宜 3 倍**。

---

### 2.7 Vision Encoder 参数（M3 / K2.5）

计算 ViT 编码器和投影器的参数量。VL 模型的视觉编码器通常有 0.6-2B 参数，在算总参和激活参时都要考虑。

#### 2.7.1 MiniMax M3 视觉编码器

ViT 32 层（`vision_config`），$d_{vit}=1280$，$H_{vit}=16$，$D_{vit}=1280/16=80$，$d_{ff}^{vit}=5120$。

**每层 Attention**（标准 MHA，无 GQA）：

$$\text{Params}_{\text{ViT attn}} = 4 \times (d_{vit} \times H_{vit} \times D_{vit}) = 4 \times (1280 \times 16 \times 80)$$

$$= 4 \times 1{,}638{,}400 = 6{,}553{,}600 \approx 6.55\text{M}$$

**每层 MLP**（GELU，2 个矩阵）：

$$\text{Params}_{\text{ViT mlp}} = 2 \times d_{vit} \times d_{ff}^{vit} = 2 \times 1280 \times 5120 = 13{,}107{,}200 \approx 13.11\text{M}$$

**每层合计**：19.66M。32 层：$\approx 629\text{M} \approx 0.63\text{B}$。
加上 patch embedding（Conv3d）+ Pre-LN + 3D RoPE：$\approx 0.65\text{B}$。

**投影器**（双阶段 MLP，$d_{vit} \to d \to d$ + spatial merge）：

$$\text{Stage 1}: 1280 \times 6144 + 6144 \times 6144 \approx 7.86\text{M} + 37.75\text{M} = 45.6\text{M}$$

$$\text{Stage 2}: (4 \times 6144) \times 6144 + 6144 \times 6144 \approx 150.99\text{M} + 37.75\text{M} = 188.7\text{M}$$

$$\text{Params}_{\text{projector}} \approx 0.23\text{B}$$

**视觉总计**：$\approx 0.88\text{B}$。

#### 2.7.2 Kimi K2.5 视觉编码器

ViT 27 层（`vision_config.vt_num_hidden_layers`），$d_{vit}=1152$，$H_{vit}=16$，$d_{ff}^{vit}=4304$。

**每层 Attention**：

$$\text{Params}_{\text{ViT attn}} = 4 \times (1152 \times 16 \times 72) = 4 \times 1{,}327{,}104 \approx 5.31\text{M}$$

（$D_{vit} = 1152/16 = 72$，但 config 中 `mm_hidden_size=1152`，`vt_hidden_size=1152`，需验证 head_dim = 1152/16 = 72）

**每层 MLP**：

$$\text{Params}_{\text{ViT mlp}} = 2 \times 1152 \times 4304 \approx 9.92\text{M}$$

**每层合计**：~15.23M。27 层：$\approx 0.41\text{B}$。加 PatchMerger 和投影器共约 2B。

ViT 就是一个小号 Transformer。算它跟算文本骨干的方法完全一样——QKV+O + MLP up/down，只是维度小得多（1152/1280 vs 6144/7168）。但 27-32 层加起来也有 ~0.6-2B 参数，不容忽略。

---

### 2.8 MTP Predictor 参数

计算 Multi-Token Prediction 模块的参数。MTP 模块不算在激活参数里（推理时是独立的投机解码模块），但算总参时不能漏。

MTP 模块的结构与主干的单个 layer 相同：1 个 Attention + 1 个 MoE（或 Mamba）。

**Nemotron 3 Ultra**（1 个 MTP 层，类型 `["attention", "moe"]`）：

$$\text{Params}_{\text{MTP}} = \text{Params}_{\text{attn}} + \text{Params}_{\text{MoE, 1 layer}} \approx 138.4\text{M} + 10.94\text{B} \approx 11.1\text{B}$$

**MiniMax M3**（7 个 MTP 模块，`num_mtp_modules=7`，每个含 1 layer）：

M3 的 MTP 模块共享 Embedding 和 LM Head，每个模块的结构和主干层类似但维度可能不同。精确参数量需从源码确认，当前一级近似：

$$\text{Params}_{\text{MTP, per module}} \approx \text{Params}_{\text{attn}} + \text{Params}_{\text{MoE, 1 layer}} \approx 111\text{M} + 7.31\text{B} \approx 7.42\text{B}$$

7 个模块：$\approx 52\text{B}$。但官方标称 MTP 不显著增加推理显存（因为 MTP 权重可能与主干有共享或使用更小的维度），实际数值以官方技术报告为准。

> **设计意图待确认**：M3 的 `_keys_to_ignore_on_load_unexpected: [r"mtp.*"]` 表明 MTP 权重在独立命名空间下。参数可能比主干层小（使用更小的 intermediate 维度），或通过参数共享减少总量。

MTP 就是“多长了几层”——如果是 1 个 MTP 模块，等于多 1 个 Attention + 1 个 MoE 层。如果 7 个 MTP 模块就是多 7 层。区别在于 MTP 只用于预测 future tokens，不是 backbone 的一部分。

---

### 2.9 激活参 vs 总参

区分“模型存了多少参数”和“每次前向要用多少参数”。推理显存 = 激活参数 × bytes/param + KV cache + 其他。不懂激活参就算不了推理成本。

#### 核心概念

- **总参数量**（Total Params）：所有权重矩阵的元素总数。模型文件的大小。
- **激活参数量**（Active Params）：单次前向传播实际参与计算的参数。MoE 中只激活 top-k 专家。

$$\text{Params}_{\text{active}} = \text{Params}_{\text{non-MoE}} + \text{Params}_{\text{router}} + k \times \text{Params}_{\text{expert}} + \text{Params}_{\text{shared}}$$

$$\text{激活率} = \frac{\text{Params}_{\text{active}}}{\text{Params}_{\text{total}}} \times 100\%$$

#### 案例 1：Nemotron 3 Ultra

| 组件 | 总参 (B) | 激活参 (B) | 说明 |
|---|---|---|---|
| Embedding + LM Head | 2.15 | 2.15 | 全激活 |
| 48 Mamba-2 层 | 20.24 | 20.24 | 全激活（无 MoE） |
| 12 Attention 层 | 1.66 | 1.66 | 全激活 |
| 48 MoE 层 (512E, top-22) | 525.4 | **32.04** | 只激活 22/512 |
| MTP Predictor | 11.1 | 不计入 | 独立模块，推理时按需使用 |
| Norm 等 | ~0.001 | ~0.001 | — |
| **总计** | **~560B** | **~56B** | — |
| 官方标称 | 550B | 55B | 偏差 ~2% |

$$\text{激活率} = \frac{55}{550} = 10\%$$

Nemotron 虽然存了 550B 参数，但每次只用其中 55B——因为 48 个 MoE 层每层只在 512 个专家中激活 22 个（4.3%）。剩下 95.7% 的专家参数“休眠”。这就是 MoE 的核心价值：总容量大，推理成本低。

#### 案例 2：MiniMax M3

$$\text{Params}_{\text{active}} \approx 1.23\text{B} + 6.64\text{B} + 0.68\text{B} + 12.91\text{B} + 3.23\text{B} + 1.23\text{B} \approx 25.9\text{B}$$

（Embedding + Attention + Dense FFN + 4/128 专家激活 + 共享专家 + LM Head）

$$\text{激活率} = \frac{25.9}{428} \approx 6.0\%$$

加上 Vision 编码器（0.88B，图像输入时激活）约为 26.8B。官方标称 ~23B。

#### 各模型激活率对比

| 模型 | 总参 | 激活参 | 激活率 | 每 token 专家激活比例 |
|---|---|---|---|---|
| Nemotron 3 Ultra | 550B | ~55B | 10.0% | 22/512 = 4.3% |
| MiniMax M3 | ~428B | ~23-26B | 5.4-6.0% | 4/128 = 3.1% |
| Kimi K2.5 | ~1T | ~32B | 3.2% | 8/385 ≈ 2.1% |

激活率越高，同等总参下推理越贵。Nemotron 的 10% 激活率看起来高，但因为它有 48 个 Mamba 层（无稀疏化），这些层每个 token 都要全部跑一遍。M3 和 K2.5 的激活率更低是因为它们几乎所有层的 FFN 都是 MoE。

---

### 2.10 完整案例：Nemotron 3 Ultra 参数分解

从 `config.json` 出发，一步步列出每类模块的参数，求和验证 ≈ 550B。这是本章所有知识的综合运用——读完你应该能对任何模型做同样的事。

#### Step 0：读取 config.json

关键字段值（见 1.1 节表）。

#### Step 1：Embedding

$$131072 \times 8192 = 1.07\text{B (输入)} + 1.07\text{B (输出)} = \mathbf{2.15\text{B}}$$

#### Step 2：48 个 Mamba-2 层

$$48 \times (287.3\text{M} + 0.09\text{M} + 134.2\text{M}) = 48 \times 421.6\text{M} = \mathbf{20.24\text{B}}$$

#### Step 3：12 个 Attention 层（GQA 32:1）

$$12 \times (67.1\text{M} + 2.1\text{M} + 2.1\text{M} + 67.1\text{M}) = 12 \times 138.4\text{M} = \mathbf{1.66\text{B}}$$

#### Step 4：48 个 MoE 层

每层：
- Router: $8192 \times 512 = 4.2\text{M}$
- 低秩投影入+出: $8192 \times 2048 + 2048 \times 8192 = 33.6\text{M}$
- 512 专家 (ReLU$^2$, latent 空间): $512 \times 2 \times 2048 \times 5120 = 10,737.4\text{M} = 10.74\text{B}$
- 共享专家 (ReLU$^2$, full 空间): $2 \times 8192 \times 10240 = 167.8\text{M}$

单层：$4.2\text{M} + 33.6\text{M} + 10,737.4\text{M} + 167.8\text{M} = 10,943\text{M} \approx 10.94\text{B}$

48 层：$48 \times 10.94\text{B} = \mathbf{525.4\text{B}}$

#### Step 5：MTP Predictor（1 attention + 1 moe）

$$\mathbf{11.1\text{B}}$$

#### Step 6：求和

| 模块 | 参数 (B) | 占比 |
|---|---|---|
| Embedding + LM Head | 2.15 | 0.4% |
| 48 Mamba-2 层 | 20.24 | 3.6% |
| 12 Attention 层 | 1.66 | 0.3% |
| 48 LatentMoE 层 | 525.4 | 94.1% |
| MTP Predictor | 11.1 | 2.0% |
| Norm 等 | ~0.001 | ~0% |
| **直接求和** | **~560.5** | — |
| **官方标称** | **550** | — |

偏差 ~1.9%，可能来源：MTP 权重有部分与主干共享；部分维度在实现中与 config 有细微差异；NVFP4 训练下的有效参数量口径不同。

#### Step 7：激活参验证

$$\text{Active} = 2.15 + 20.24 + 1.66 + 48 \times (4.2\text{M} + 33.6\text{M} + 22 \times 21\text{M} + 167.8\text{M}) \div 10^3$$

$$= 2.15 + 20.24 + 1.66 + 48 \times 0.6675\text{B}$$

$$= 2.15 + 20.24 + 1.66 + 32.04 = \mathbf{56.1\text{B}}$$

与官方 55B 偏差 ~2%。扣除 MTP（11.1B）后 backbone 激活 ≈ 56B，与标称一致。

> **自查清单**（算完参数量后对照）：
> - [ ] Embedding = `vocab_size × hidden_size`？weight tying 只乘一次？
> - [ ] GQA 的 K/V 矩阵是 `d × (H_kv × D_h)` 不是 `d × d`？
> - [ ] SwiGLU 是 3 个矩阵（gate/up/down），ReLU² 是 2 个？
> - [ ] MoE = Router + N_experts × expert + shared_expert（别忘了 Router）？
> - [ ] 各项之和 ≈ 官方标称值（允许 1-2% 偏差）？
> - [ ] 激活参 ≠ 总参？激活率通常在 3-10%？

---

### 2.11 速查表：从 config.json 到参数量

给一张“抄作业”级别的公式汇总表。以后算任何模型，打开这张表逐行代入即可。

| 模块 | 公式 | 适用条件 |
|---|---|---|
| Embedding (in) | $V \times d$ | 总是 |
| Embedding (out) | $V \times d$ | `tie_word_embeddings=false` 时 |
| MHA Attention | $4 \times d^2$ | $H_{kv}=H_q$ 且 $H_q \times D_h = d$ |
| GQA Attention | $d \times (H_q \times D_h) + 2 \times d \times (H_{kv} \times D_h) + (H_q \times D_h) \times d$ | 通用 |
| MLA (Q 侧) | $d \times d_q + d_q \times H \times D_{nope} + d \times H \times D_{rope}$ | `kv_lora_rank` 和 `q_lora_rank` 存在时 |
| MLA (KV 侧) | $d \times (d_{kv} + D_{rope}) + d_{kv} \times H \times (D_{nope} + D_v)$ | 同上 |
| MLA (output) | $H \times D_v \times d$ | 同上 |
| SwiGLU FFN | $3 \times d \times d_{ff}$ | `hidden_act=silu` 且 gate/up/down 分离 |
| ReLU$^2$ FFN | $2 \times d \times d_{ff}$ | `hidden_act=relu2` |
| MoE Router | $d \times E$ | 总是 |
| MoE 总/层 | $d \times E + E \times \text{Params}_{expert} + \text{Params}_{shared}$ | 总是 |
| Mamba-2 in_proj | $d \times (2d_{inner} + 2n_{groups}N + H_{mamba})$ | `model_type` 含 mamba |
| Mamba-2 out_proj | $d_{inner} \times d$ | 同上 |
| Dense FFN | 同 SwiGLU/ReLU$^2$，见 `intermediate_size` / `dense_intermediate_size` | `moe_layer_freq[i]=0` 的层 |
| RMSNorm | $d$ | 每层 2 个，可忽略 |
| 激活参 | $\text{非MoE} + \text{Router} + k \times \text{Params}_{expert} + \text{Params}_{shared}$ | MoE 模型 |
| 总参 | 上述所有求和 | — |

**实战口诀**：
1. 打开 `config.json`，圈出 $d, V, H_q, H_{kv}, D_h, d_{ff}, d_{moe\_ff}, E, k$
2. Embedding: $2Vd$（如果 `tie_word_embeddings=false`）
3. Attention/层: 查 GQA/MHA/MLA 公式
4. FFN/层: 查 `hidden_act` 决定 ×2 还是 ×3
5. MoE: $E \times$ FFN/层 + router
6. 乘层数，加 MTP，加 Vision
7. 总参 = 以上求和；激活参 = 非 MoE + $k \times$ 单专家
8. 显存 = 激活参 × bytes/param（见 1.5 节）

---

## 术语中英对照

| 中文 | 英文 | config 字段 |
|---|---|---|
| 隐藏维度 | hidden size / model dimension | `hidden_size` |
| 注意力头数 | number of attention heads | `num_attention_heads` |
| KV 头数 | number of key-value heads | `num_key_value_heads` |
| 每头维度 | head dimension | `head_dim` |
| 中间维度 | intermediate size | `intermediate_size` |
| 词表大小 | vocabulary size | `vocab_size` |
| 路由专家 | routed experts | `n_routed_experts` |
| 共享专家 | shared experts | `n_shared_experts` |
| 每 token 专家数 | experts per token | `num_experts_per_tok` |
| 激活参数 | active / activated parameters | — |
| 总参数 | total parameters | — |
| 权重绑定 | weight tying | `tie_word_embeddings` |
| 分组查询注意力 | Grouped Query Attention (GQA) | $H_{kv} < H_q$ |
| 多头潜注意力 | Multi-head Latent Attention (MLA) | `kv_lora_rank` 存在 |
| 状态空间模型 | State Space Model (SSM) | `ssm_state_size` 存在 |
| 多 token 预测 | Multi-Token Prediction (MTP) | `num_nextn_predict_layers` |

---

## CH1-2 常见计算错误

| # | 常见错误 | 正确做法 |
|---|---|---|
| 1 | 用 $d$ 代替 $H_q \times D_h$ 算 K 投影 | GQA 中 K 的维度是 $H_{kv} \times D_h$，不是 $d$ |
| 2 | 忘记 MLA 的 rope 投影不能压缩 | rope 部分用 $d \times H \times D_{rope}$，不经过潜空间 |
| 3 | 混淆 `intermediate_size` 和 `moe_intermediate_size` | Dense 层和 MoE 专家层可能用不同维度 |
| 4 | 忘记乘 bytes | 参数量是“个数”，显存是“字节数”，中间差 2×（BF16） |
| 5 | Mamba 的 $d_{inner}$ 没验证自洽 | $d_{inner} = \text{expand} \times d = H_{mamba} \times D_{mamba}$ |
| 6 | 漏掉了 LM Head | `tie_word_embeddings=false` 时 LM Head 是独立矩阵 |
| 7 | Router 参数当成 0 | 虽然小（几 M），但要把所有层加起来 |
| 8 | 激活参计算时忘记共享专家 | 共享专家对每个 token 都激活，不算在 top-k 里 |

---

> **下一章预告**：CH 3 FLOPs 估算——从参数量到计算量，推导 prefill/decode 的单 token FLOPs 公式，并给出 Nemotron/M3/K2.5 的完整 FLOPs 分解表。


---



> **系列导航**：[（二）FLOPs 估算](../part-2/) → [（三）KV Cache 与推理显存](../part-3/) → [（四）M3 实战 + Roofline](../part-4/) → [（五）训练显存](../part-5/) → [（六）通信分析](../part-6/) → [（七）推理服务](../part-7/)
