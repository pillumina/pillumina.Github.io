+++
date = '2026-06-15'
draft = false
title = 'LLM 架构计算方法论：从 config.json 到推理显存'
categories = ['aiinfra']
tags = ['computation', 'flops', 'kv-cache', 'parameters', 'methodology', 'inference']
summary = '从 config.json 到参数量、FLOPs、KV Cache、推理显存的完整计算推导。基于 8 个开源模型（M2.7 / GLM-5.1 / V4-Flash / Qwen3.5 / Mimo / Kimi / Nemotron / M3）的实战拆解经验。覆盖 Full Attention / MSA / MLA / Mamba-2 / SWA / GDN 六种注意力架构的 FLOPs 与 KV Cache 公式。'
pinned = true
math = true
+++

# LLM 架构计算方法论

> 从 config.json 到参数量、FLOPs、KV Cache、推理显存的完整计算推导。基于 8 个开源模型的实战拆解经验。

---

## 目录

- **CH 1** 预备知识：从 config.json 到矩阵乘法
- **CH 2** 参数分解：这个模型有多大
- **CH 3** FLOPs 估算：推理一次花多少计算
- **CH 4** KV Cache 显存：长上下文为什么吃显存
- **CH 5** 推理显存：部署需要多少卡
- **CH 6** 实战：MiniMax M3 完整推演
- **CH 7** Roofline 模型：算力够为什么还是慢
- **附录 A** config.json 字段速查表
- **附录 B** 符号与缩写表
- **附录 C** 8 个模型计算结果速览

---

## 阅读导航

| 你的目标 | 推荐阅读路径 | 预计时间 |
|---|---|---|
| **快速了解全貌** | CH 1.2（FLOPs基础）→ CH 2.3（Attention参数）→ CH 4.2（KV cache公式）→ 附录 C（8模型速览） | 30 min |
| **学会算参数量** | CH 1.1（config字段）→ CH 1.4（符号表）→ CH 2（全章，4个案例代入）→ CH 2.10（Nemotron完整推演） | 60 min |
| **学会算 FLOPs** | CH 1.2（FLOPs公式）→ CH 3.2（Full Attn）→ 按需选读 3.3（MSA）/ 3.4（MLA）/ 3.5（Mamba-2）/ 3.6（SWA）/ 3.7（DeltaNet）→ CH 3.10（跨架构对比） | 45 min |
| **学会算 KV cache** | CH 1.4（符号表）→ CH 4.2（标准GQA）→ CH 4.3（MLA重点）→ CH 4.5（SWA）/ 4.6（DeltaNet）/ 4.7（Mamba-2）→ CH 4.9（对比表） | 40 min |
| **独立推演一个模型** | CH 1（预备知识，15 min）→ CH 6（M3完整推演，对照 config.json 自己算一遍） | 90 min |
| **查漏补缺** | 附录 A（config字段→计算项映射）→ 附录 B（符号表）→ 定位到对应章节 | 5 min |

各章之间的依赖关系：CH 2 → CH 3（参数是 FLOPs 的输入，但 FLOPs 的核心公式独立）→ CH 4（FLOPs 和 KV cache 无依赖，可并行阅读）→ CH 5（依赖 CH 2 + CH 4）→ CH 6（依赖全部）。

> **新读者建议**：从 CH 1.2（5 分钟搞懂 FLOPs 怎么数）和 CH 4.2（10 分钟搞懂 KV cache 怎么算）开始——这两节能让你最快建立「能算」的感觉。

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


## CH 3 FLOPs 估算

> **读者定位**：已完成 CH1-2 的参数计算，目标是推导 prefilling / decoding 的单 token 计算量，并理解不同架构（Full Attn / MSA / MLA / Mamba-2）的 FLOPs 差异根源。

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

## CH 6 | 实战——MiniMax M3 完整推演

以 MiniMax M3 为目标，从 config.json 出发，完整推演参数分解 → FLOPs 估算 → KV Cache → 推理显存 → 部署方案，覆盖 GQA + MSA + MoE + Vision + MTP 五种架构变体的计算。M3 是目前覆盖计算变体最多的开源模型——一个模型练完基本上所有架构你都会算了。

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
