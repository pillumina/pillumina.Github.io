+++
date = '2026-06-10'
draft = false
title = 'DeepSeek-V4-Flash 架构 QA（共 287 问）'
categories = ['qa']
tags = ['moe', 'attention', 'model-architecture', 'deepseek', 'qa', 'mla', 'csa', 'hca']
series = ['qa']
summary = '基于 V4-Flash 主报告的配套 QA，共 287 问。覆盖 LLM 预备知识、V3.2→V4 演进、注意力系统（CSA/HCA）、MoE 路由（Aux-Loss-Free/Sinkhorn-Knopp）、残差与优化器（mHC/Muon）、上下文与量化（1M/RoPE/FP4/FP8）、训练与推理部署。由浅入深，可作面试准备。'
+++

# DeepSeek-V4-Flash 架构 QA 文档

> 基于主报告 `deepseek-v4-flash.md` 的配套 QA，由浅入深，可作面试准备。
> 生成日期：2026-06-09 · 范围：V4-Flash（284B/13B 激活）

## 目录（共 287 问）

- **CH 0-3** — 78 问 — 背景 + V3 回顾 + V4 概览
- **CH 4-5** — 69 问 — CSA/HCA 注意力 + MoE 路由
- **CH 6-7** — 51 问 — mHC 残差 + Muon 优化器
- **CH 8-10** — 89 问 — 工程实现 + 源码系统 + 面经高频

> 本 QA 文档分四章：CH 0 文档说明 + CH 1 LLM 预备 + CH 2 V3 回顾 + CH 3 V4 概览。
> 适合 LLM 训练有初步认识的学生 / 工程师，自学 + 面试准备两用。
> 主报告：`report/deepseek-v4-flash.md` · 术语表：`report/appendix-glossary.md`

---

## CH 0. 文档说明

### Q0.1 这份 QA 文档对应主报告的哪些章节？

**简短回答**：CH 0–3 覆盖主报告的 CH 0（摘要）+ CH 1（V3.2 → V4 演进）+ CH 2（V4-Flash 整体架构）+ CH 3（注意力 CSA + HCA）。

**详细解释**：本 QA 是主报告的"问答浓缩版"——主报告每章 5000+ 字、含大量代码公式，对初学者门槛高；本 QA 把每章拆成 20–30 个 Q，每个 Q 用"短答 + 详细解释 + 面试要点"三段式组织。CH 0 教你怎么用本 QA，CH 1 是 LLM 通识基础，CH 2 回顾 V3.2，CH 3 引出 V4 四大创新。CH 4 之后由其他 QA 文档负责。

💡 **面试要点**：本 QA 是面经速成工具，但**完整理解仍要回主报告**——主报告是"算子级拆解"，本 QA 是"概念 + 直觉 + 面试要点"。

**延伸阅读**：主报告 `deepseek-v4-flash.md` §0

---

### Q0.2 我应该按什么顺序读这份 QA？

**简短回答**：CH 0 → CH 1 → CH 2 → CH 3 顺序读；有 LLM 基础可跳过 CH 1。

**详细解释**：
- **完全新手**：CH 0 速览 → CH 1 通读 → CH 2 选读 → CH 3 重点读
- **有 LLM 基础**：跳过 CH 1 直接读 CH 2 → CH 3
- **面试突击**：只看"短答 + 面试要点"
- **架构师 / AI infra**：CH 2.1（超参表）+ CH 3.5（FLOPs/cache 节省）是核心

四个章节"自包含但层层递进"：CH 1 不需要 V3/V4 知识、CH 2 只看 V3、CH 3 衔接 V3 的瓶颈引出 V4。

**延伸阅读**：术语表 `appendix-glossary.md` B.1

---

### Q0.3 每个 Q 末尾的"💡 面试要点"是什么格式？

**简短回答**：1–2 句话总结回答里最容易被追问的点，标"💡 面试要点"或"⚠️ 易混淆"。

**详细解释**：
- **💡 面试要点**：面试官常追问的"二阶问题"，如"MLA 为什么用潜空间而不是直接压缩 K/V？"
- **⚠️ 易混淆**：两个相似概念的区分点，如"MLA vs MQA"、"CSA vs HCA"——混淆会直接扣分
- 整份 QA 的"💡 / ⚠️"标注统一遵循同一种风格，便于面试前快速扫读

**延伸阅读**：本 QA 每个 Q 末尾

---

### Q0.4 文档里的"主报告 CH X.Y"引用怎么用？

**简短回答**：是去主报告查"详细论证"的页码锚点，主报告是本 QA 的真源。

**详细解释**：本 QA 是"问答式笔记"，主报告是"完整论述"。当本 QA 说"详见主报告 CH 3.5"，意思是主报告 CH 3.5 节有 FLOPs/cache 节省的完整计算与表格。同理：
- 论文引用（`V3 paper §5.1`）指 DeepSeek-V3 技术报告第 5.1 节
- 仓库引用（`inference/model.py:L436`）指 V4-Flash 仓库的 inference/model.py 第 436 行
- 符号引用（如 `(3.1)`）指本主报告的"第 3 章第 1 个公式"约定（见术语表 B.3）

💡 **面试要点**：面试时若被追问细节，能说"具体公式见论文 §X.Y"比硬背所有数字更显专业。

**延伸阅读**：术语表 B.3 + B.4

---

## CH 1. LLM 预备知识

### Q1.1 Transformer 的"三件套"是什么？

**简短回答**：Self-Attention（自注意力）+ FFN（前馈网络）+ 残差连接，每个 block 都由这三部分堆叠而成。

**详细解释**：每个 Transformer block 三步：
1. **Self-Attention**：让序列内任意两个位置直接"对话"，捕捉 token 之间的依赖
2. **FFN**：对每个位置做"非线性变换"，常实现为两层 MLP（中间维度通常是 hidden 的 4×）
3. **残差连接**：把子层的输入加到输出上（`y = x + Sublayer(x)`），让深层网络的梯度能直接传回浅层

在 V4-Flash 中：Self-Attention 走 CSA+HCA 混合（CH 3）；FFN 走 MoE（256 routed + 1 shared）；残差走 mHC 多通道版本（CH 5）。三层都做了 V4 创新。

💡 **面试要点**：被问"Transformer block 由什么组成"时，**必须说全三件套**——漏掉"残差"或"Norm"会扣分。

**延伸阅读**：主报告 CH 2.2

---

### Q1.2 Self-Attention 的核心公式是什么？

**简短回答**： $Attention(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d_k}) \cdot V$ ，先算 query-key 相似度，再加权求 value。

**详细解释**：Self-Attention 把每个 token 投影成 Q（"我在找什么"）、K（"我代表什么"）、V（"我能提供什么信息"）。三步：
1. **相似度**： $QK^\top$ 是 `[T, T]`，得到每对位置的"匹配分数"
2. **缩放**：除以 $\sqrt{d_k}$ ，防止 $\mathrm{softmax}$ 进入饱和区
3. **加权求和**： $\mathrm{softmax}(\cdot ) \cdot V$ ，相似度高的位置贡献更多 value

因果 attention 还在 $\mathrm{softmax}$ 前加 causal mask（上三角设为 -∞），让 token t 只能看位置 0..t。

⚠️ **易混淆**： $QK^\top$ 是 `[T, T]` 矩阵，**不是**"每两个位置算一次乘法"——它是矩阵乘法，一次性算完全部 T² 对。

**延伸阅读**：主报告 CH 3.1

---

### Q1.3 Multi-Head Attention (MHA) 是什么？

**简短回答**：把 d 维的 Q/K/V 拆成 h 个头（每个 head 维度 d_h = d/h），每个头独立算 attention，最后 concat 起来。

**详细解释**：单个 attention 头只能学一种"匹配模式"。Multi-Head 让 h 个头在不同子空间并行算 attention，再把 h 个 $[T, d_h]$ 输出沿 head 维拼回 `[T, d]`，通过一个 O 投影回 d 维。**为什么有效**：
- 不同头可以学"主谓一致"、"指代消解"、"长距离依赖"等不同模式
- 总计算量与单头大 head 等价（ $h \cdot d_h$ = d），但表达能力强 h 倍
- V4-Flash： $num_attention_heads=64$, $head_dim=512$ ，d=4096（Q 走 q_lora_rank=1024 先压后升）

⚠️ **易混淆**：MHA vs MQA vs GQA vs MLA——这四种是"KV cache 压缩"的四代演进。

**延伸阅读**：主报告 CH 2.1

---

### Q1.4 FFN（前馈网络）是什么？为什么需要？

**简短回答**：每个 Transformer block 里的"逐位置 MLP"，对每个 token 独立做非线性变换；与 Attention 的"跨位置"形成互补。

**详细解释**：Attention 是"跨位置混合"，FFN 是"逐位置变换"。标准 FFN 是两层 Linear + 激活： $FFN(x) = W2 \cdot activation(W1 \cdot x)$ ，中间维度 d_ff 通常是 4d。**为什么需要**：
- Attention 表达力有限，FFN 引入真正的非线性
- FFN 占据模型总参数量的约 2/3

V4-Flash 的 FFN 走 **$\mathrm{SwiGLU}$ 激活**（见 Q1.8），256 个 expert × 2 层 × $4096 \times 2048$ = 大头参数。

💡 **面试要点**：FFN 是"逐位置"还是"跨位置"？答"逐位置"——每个 token 独立算。

**延伸阅读**：主报告 CH 2.2

---

### Q1.5 MQA / GQA / MLA 是什么？为什么需要？

**简短回答**：三种"KV cache 压缩"方案，从"全部独立"到"全部共享"再到"潜空间压缩"，逐步降低推理时显存占用。

**详细解释**：标准 MHA 中，64 个 Q 头对应 64 套 KV，cache 占 $T \times  64 \times  d_h \times  2$ ：
- **MQA**：所有 Q 头共享 1 套 KV（ $num_key_value_heads=1$ ）。cache 缩 64 倍
- **GQA**：把 Q 头分组共享 KV（如 8 组 → 8 套 KV），MQA 与 MHA 的折中
- **MLA**（V3.2 引入）：把 K、V 压到 $d_c=512$ 潜空间，**比 MQA 质量好，比 GQA cache 还小**

V4-Flash $num_key_value_heads=1$ （MQA）+ grouped low-rank O 投影——MLA 思想沿用但更激进。

💡 **面试要点**：被问"MLA 跟 MQA 区别"时，答"MLA 走潜空间再恢复，MQA 直接共享"——**MQA 是'减少 KV 头数'，MLA 是'把 KV 压成低秩再展开'**。

**延伸阅读**：主报告 CH 1.1 + 术语表 B.1

---

### Q1.6 MLA（Multi-head Latent Attention）的详细原理

**简短回答**：把 K、V 先用低秩矩阵压到 $d_c=512$ 的潜空间 `c`，推理时只缓存 `c` 而非 K、V；恢复时用另一个矩阵升回多头维度。

**详细解释**：
1. **压缩**： $c_t = W_down \cdot x_t$ ， $c_t \in  \mathbb{R}^{d_c=512}$ ≪ $h \times  d_h = 8192$
2. **缓存**：只存 $c_t$ ，不存原始 K、V
3. **恢复**： $k_t = W_k_up \cdot c_t$ ， $v_t = W_v_up \cdot c_t$ ，临时升回多头

Cache 大小： $T \times  d_c \times  L \times  2 \approx  131 GB$ （V3.2 1M 上下文）。**比 MHA 省 16 倍**（8192/512 = 16）。**V4-Flash 比 V3.2 更激进**——`d=4096`（vs 7168）、`L=43`（vs 61），且用 CSA/HCA 进一步把 cache 沿时间维压缩。

⚠️ **易混淆**：MLA 不只是"MQA + LoRA"——MQA 是直接共享，MLA 是"潜空间压缩后再恢复"。

**延伸阅读**：主报告 CH 1.1

---

### Q1.7 MoE（Mixture of Experts）是什么？

**简短回答**：把单个大 FFN 拆成 N 个"专家 FFN" + 一个"路由门"，每个 token 只激活 k 个专家；总参数大但单 token 计算小。

**详细解释**：
- **路由门**： $gate(x) = W_gate \cdot x$ 得 N 个分数；**top-k 选 6 个**（V4-Flash）；**k 个专家 FFN 算**；**加权和**输出

**关键优势**：
- **总参**：N × 单个 FFN 参数量（如 256 × 17M ≈ 4.3B）
- **激活参**：k × 单个 FFN 参数量（6 × 17M = 100M）——**比稠密 FFN 少 256/6 ≈ 43 倍**
- **质量**：逼近稠密模型，训练成本低 5× 以上

**V4-Flash**：256 routed（FP4）+ 1 shared（FP8 始终激活），top-6 routed 加权和 × 1.5 后与 shared 相加。

💡 **面试要点**：MoE 的核心是"**总参大、激活小**"——被问"MoE 比稠密模型有什么优势"答这个。

**延伸阅读**：主报告 CH 4

---

### Q1.8 $\mathrm{SwiGLU}$ 是什么？和标准 FFN 的 $\mathrm{ReLU}$ 有什么区别？

**简短回答**： $\mathrm{SwiGLU}$ = $Swish(xW) \odot  (xV)$ ，把 GLU 的 $\mathrm{sigmoid}$ 换成 Swish（ $x\cdot \mathrm{sigmoid}(x)$ ）；比 $\mathrm{ReLU}$ 在 LLM 上效果更好。

**详细解释**：标准 FFN： $FFN(x) = W2 \cdot \mathrm{ReLU}(W1 \cdot x)$ 。GLU 系列把单 Linear 拆成两个 Linear + 门控：
- **GLU**： $GLU(x) = (xW) \odot  \mathrm{sigmoid}(xV)$
- **$\mathrm{SwiGLU}$**： $\mathrm{SwiGLU}(x) = (xW) \odot  Swish(xV)$ ，Swish(x) = x·$\mathrm{sigmoid}$(x)
- **GeGLU** 等变体

V4-Flash 用 $\mathrm{SwiGLU}$ （ $hidden_act: "silu"$ ，SiLU = Swish），中间维度 2048。V4 特殊设计： $swiglu_limit=10.0$——把输出钳制在 ±10，防 FP4 量化溢出。

⚠️ **易混淆**： $\mathrm{SwiGLU}$ 的"门"是 element-wise 乘法（ $\odot$ ），不是矩阵乘法。

**延伸阅读**：主报告 CH 2.1

---

### Q1.9 $\mathrm{RMSNorm}$ vs $\mathrm{LayerNorm}$ 是什么关系？

**简短回答**： $\mathrm{RMSNorm}$ 去掉 $\mathrm{LayerNorm}$ 的"减均值"步骤，只做"除以 RMS"，**参数更少、训练更稳**。

**详细解释**：
- **$\mathrm{LayerNorm}$**： $y = (x − \mu) / \sigma \cdot \gamma + \beta$
- **$\mathrm{RMSNorm}$**： $y = x / RMS(x) \cdot \gamma$ ， $RMS(x) = \sqrt{\mathrm{mean}(x^2)}$ ，**无 β、无减均值**

**优势**：计算量减少约 30%；LLM 训练稳定性更好（Llama/DeepSeek 全用 $\mathrm{RMSNorm}$ ）。V4-Flash： $rms_norm_eps=1e-6$ （数值稳定项）。

⚠️ **易混淆**： $\mathrm{RMSNorm}$ 仍保留可学习的 γ（缩放），只是去掉了 β（平移）。

**延伸阅读**：主报告 CH 2.1

---

### Q1.10 RoPE（Rotary Position Embedding）是什么？

**简短回答**：把位置信息编码成"旋转让 query/key 与位置 m 的夹角一致"，通过把 Q、K 的复数表示乘 $e^{i\cdot m\cdot \theta}$ 实现。

**详细解释**：
- 把 d 维向量两两一组看成 d/2 个复数
- 第 m 个位置的 Q/K 整体乘 $e^{i\cdot m\cdot \theta_k}$ ，θ_k = base^(-2k/d)
- 关键性质： $\langle q_m, k_n\rangle$ 只与 `(m − n)` 有关——**天然支持相对位置、长度外推**

**V4-Flash**： $rope_theta=10,000$ 、 $qk_rope_head_dim=64$ （只对每头 64 维施加 RoPE，剩余 448 维不带位置信息）、1M 上下文走 YaRN 扩展。

💡 **面试要点**：RoPE 的核心优势是"**相对位置 + 长度外推**"。

**延伸阅读**：主报告 CH 2.1 + Q1.23

---

### Q1.11 KV cache 是什么？为什么需要？

**简短回答**：推理时把之前算过的 K、V 缓存起来，避免每生成一个新 token 都重新算整段序列。

**详细解释**：LLM 推理分两阶段：
- **Prefill**：把整个 prompt 一次性塞进模型，算出所有 K、V 存进 cache
- **Decode**：每生成一个新 token，只算当前 Q，与 cache 里的 K 算 attention

**为什么需要**：
- 不缓存：每生成 1 个 token 要重算整段，**$O(T^2)$** FLOPs
- 缓存：每生成 1 个 token 只算 1 次新 Q，**$O(T)$** FLOPs

**代价**：cache 占用 $T \times  d \times  L \times  2$ 。V4-Flash 用 MQA + CSA/HCA 压缩，cache 比 V3.2 MLA 还小约 15 倍。

⚠️ **易混淆**：KV cache 只在 **decode 阶段**用。

**延伸阅读**：主报告 CH 3.1

---

### Q1.12 $O(n^2)$ 灾难是什么？

**简短回答**：标准 attention 的 FLOPs 与序列长度的平方成正比，T=1M 时算力爆炸（单 token FLOPs ≈ 8.6×10⁹）。

**详细解释**：单 token 推理视角下，attention 要算 T 个内积，T=1M 时单 token 要 8.6×10⁹ FLOPs（ $2 \times  1,048,576 \times  4096$ ），仅 attention 算力就吃满 H200。**prefill 阶段更严重**——算整段 $T \times  T$ 矩阵，43 层累计 ≈ 1.9×10¹⁷ FLOPs，单卡要 ~96 秒。V4 的 CSA / HCA 用"时间维压缩"把 T² 降到 T²/m（m=4 或 128），从根上解决 $O(n^2)$ 灾难。

💡 **面试要点**：被问"长上下文为什么难"时，**先答 $O(n^2)$ 再答 KV cache**——这是两个独立的瓶颈。

**延伸阅读**：主报告 CH 3.1 + CH 3.5

---

### Q1.13 $\mathrm{FlashAttention}$ 是什么？

**简短回答**：通过"分块算 + online $\mathrm{softmax}$"避免物化完整的 `[T, T]` attention 矩阵，**显存从 $O(T^2)$ 降到 $O(T)$**，且利用 GPU SRAM 提速。

**详细解释**：标准 attention 要先算完整 $QK^\top$ （[T, T] 矩阵）再 $\mathrm{softmax}$ 再乘 V——T=1M 时单精度要 4TB。 $\mathrm{FlashAttention}$ 思路：
1. 把 Q、K、V 切成 block 装进 GPU SRAM
2. 用 online $\mathrm{softmax}$ （维护 running max/sum）逐步处理每个 block
3. **不存中间 `[T, T]` 矩阵**，只存最终输出

V4-Flash 的 $sparse_attn_kernel$ （`inference/kernel.py`）就是 $\mathrm{FlashAttention}$ 风格 + 稀疏 gather。

💡 **面试要点**： $\mathrm{FlashAttention}$ 的两个核心创新：**online $\mathrm{softmax}$ （数值稳定）+ SRAM 优先（IO-aware）**。

**延伸阅读**：主报告 CH 3.6

---

### Q1.14 Pre-Norm vs Post-Norm 是什么区别？

**简短回答**：Pre-Norm 把 Norm 放在子层**之前**（`y = x + Sublayer(Norm(x))`），Post-Norm 把 Norm 放在子层**之后**（`y = Norm(x + Sublayer(x))`）。

**详细解释**：
- **Pre-Norm**（V3/V4 用）： $x_{l+1} = x_l + Sublayer(Norm(x_l))$——梯度直接流过残差，**训练稳定**
- **Post-Norm**（原 Transformer）： $x_{l+1} = Norm(x_l + Sublayer(x_l))$——梯度被 Norm 影响，**需要 warmup**

⚠️ **易混淆**：Pre-Norm 训练的模型**推理时通常需要 final Norm**（主输出前再 Norm 一次）。

**延伸阅读**：主报告 CH 1.2

---

### Q1.15 AdamW 优化器基础

**简短回答**：Adam（自适应学习率）+ decoupled weight decay 的组合，是 LLM 训练的事实标准。

**详细解释**：
1. **一阶矩 m**（动量）：梯度的指数移动平均
2. **二阶矩 v**（方差）：梯度平方的指数移动平均
3. **更新**： $\theta ← \theta − \eta \cdot m / (\sqrt{v} + \epsilon)$
4. **Weight decay**： $\theta ← \theta − \eta \cdot \lambda \cdot \theta$ （**不通过 m、v**，decouple）

**为什么适合 LLM**：自适应学习率（适合稀疏梯度如 MoE 路由）、训练稳定。**V4-Flash 改用 Muon**（V4 创新）——见 CH 6。

💡 **面试要点**：AdamW vs Adam 的区别是"**decoupled** weight decay"——weight decay 不进 m、v。

**延伸阅读**：主报告 CH 6

---

### Q1.16 训练 token / 训练算力（Chinchilla 法则）

**简短回答**：Chinchilla 法则：模型参数 N 与训练 token D 应满足 $D \approx  20N$ （算力最优），偏离这条线一侧是"过度训练"或"欠训练"。

**详细解释**：
- LLaMA-2-7B 训练 2T → 接近最优
- V3 14.8T / 671B → D/N ≈ 22，**接近 Chinchilla 最优**
- V4-Flash 32T / 284B → D/N ≈ 113，**显著过训练**——走"小模型、长训练"路线

⚠️ **易混淆**：Chinchilla 是算力最优，不是质量最优——质量上不一定最优，但算力利用率最高。

**延伸阅读**：主报告 CH 1.1 + CH 3.5

---

### Q1.17 FP8 / FP16 / BF16 是什么？

**简短回答**：三种浮点格式，bit 数相同但 exponent/mantissa 切分不同；LLM 训练最常用 BF16（前向） + FP32（主权重）。

**详细解释**：

| 格式 | bit | 指数 | 尾数 | 范围 | 精度 |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | 高 |
| FP16 | 16 | 5 | 10 | ±65504 | 中 |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | 低 |
| FP8 (E4M3) | 8 | 4 | 3 | ±448 | 低 |
| FP8 (E5M2) | 8 | 5 | 2 | ±57344 | 极低 |
| FP4 (E2M1FN) | 4 | 2 | 1 | ±7 | 极低 |

**为什么需要 FP8**：训练时一半显存、2× 算力（H100/H200 有 FP8 tensor core）。LLM 梯度动态范围大，**BF16 比 FP16 稳定**。V4-Flash：routed expert 用 FP4，attention / shared expert 用 FP8（E4M3）。

**延伸阅读**：主报告 CH 1.1 + CH 7

---

### Q1.18 量化：PTQ vs QAT

**简短回答**：**PTQ（Post-Training Quantization）**：训完模型再量化；**QAT（Quantization-Aware Training）**：训练中模拟量化，模型适配量化。

**详细解释**：
- **PTQ**：拿训好的模型，用小校准集统计 activation range，直接把权重 / 激活值映射到低 bit。**优点**：简单快；**缺点**：低 bit（FP4 / INT4）质量掉得多
- **QAT**：训练 forward 中插入"伪量化算子"（ $fake_quant(x)$ = `round(x / scale) * scale`），梯度照常反传。模型在前向时就"看到"量化后的分布，**适配量化、损失小**

V4-Flash 走 QAT：forward 模拟 FP4 / FP8，backward 用 FP32 主权重 + 量化感知梯度。

💡 **面试要点**：QAT vs PTQ 选哪个？答"**低 bit（FP4/INT4）必须 QAT**"。

**延伸阅读**：主报告 CH 7

---

### Q1.19 推理 vs 训练有什么区别？

**简短回答**：训练需要"前向 + 反向 + 优化器 step"占大量显存（3× 激活）；推理只需要前向 + KV cache。

**详细解释**：
- **训练**：FP32 主权重（4B/参） + FP16/BF16 副本（2B/参） + FP16 优化器状态（8B/参） + 激活。**总 ≈ 16–20 × 参数**
- **推理**：BF16 主权重（2B/参） + KV cache（T × d × L × 2）。**总 ≈ 2–4 × 参数 + KV**

V4-Flash 推理进一步量化到 FP4（routed 0.5B/参）+ FP8（其他 1B/参），总权重 ≈ 426 GB——**但激活只 13B，单 batch 2×H200 可跑**。

⚠️ **易混淆**：训练"激活"显存与推理"KV cache"是**完全不同的概念**——前者是反向用的中间值，后者是 attention 用的历史 K/V。

**延伸阅读**：主报告 CH 1.4

---

### Q1.20 LLM 训练目标函数（next-token prediction）是什么？

**简短回答**：标准的 causal language modeling——给定前 t-1 个 token，预测第 t 个 token，用 cross-entropy loss。

**详细解释**：
- 输入：token 序列 $[t_1, t_2, ..., t_T]$
- 模型输出 logits $[T, vocab_size]$
- Loss： $-mean(log_softmax(logits)[t])$
- **等价于"全部位置都做监督"**——T 个 token 给出 T 个监督信号

V3/V4 沿用标准的 next-token prediction + MTP（Multi-Token Prediction）。MTP 在每个主位置额外预测下 1 个 token，监督信号 × 2。

💡 **面试要点**：被问"LLM 怎么训练"答"next-token prediction (causal LM) + cross-entropy"。

**延伸阅读**：主报告 CH 1.1

---

### Q1.21 因果 attention（Causal / Masked Attention）是什么？

**简短回答**：在 $\mathrm{softmax}$ 前用 mask 把"未来位置"屏蔽掉（设为 -∞），让 token t 只能看位置 0..t。

**详细解释**：decoder-only LLM 要求"训练时第 t 个 token 不能看到第 t+1..T 个位置"。**实现**：
```
attn = QK^⊤ / √d_k
attn = attn.masked_fill(causal_mask, -inf)
attn = softmax(attn)
out = attn · V
```

V4-Flash 的 sparse attention 也保留因果性——CSA/HCA 的 top-k 只在"≤ 当前 position 的压缩段"里选。

⚠️ **易混淆**：因果 mask 的"未来"是**严格大于** t，不是 ≥；位置 t 可以看自己（attn 对角线保留）。

**延伸阅读**：主报告 CH 3.6

---

### Q1.22 词表（Vocabulary）与 Tokenizer 基础

**简短回答**：词表是把字符串切成 token 的"字典"，常见有 BPE、WordPiece、SentencePiece。V4-Flash $vocab_size=129,280$ 。

**详细解释**：
- **BPE**：从字符开始，迭代合并最高频对——GPT 系用
- **WordPiece**：BPE 的变体——BERT 用
- **SentencePiece**：从语言无关角度做 BPE/Unigram——Llama/DeepSeek 用

V4-Flash 走 SentencePiece（继承 DeepSeek-V3 词表），词表大小 129,280。词表越大，1 个 token 可编码更长字符串（推理步数减少）；但太大让 embedding 表参数量大（V4-Flash embedding ≈ 129,280 × 4096 ≈ 530M）。

**延伸阅读**：主报告 CH 2.3

---

### Q1.23 YaRN（Yet another RoPE extensioN）是什么？

**简短回答**：把 RoPE 的"高频维度"和"低频维度"分别处理，从 64K 上下文扩展到 1M 上下文。

**详细解释**：标准 RoPE 在外推到更长上下文时，"低频维度"（波长长）会"过采样"导致信息丢失。YaRN 的解法：
- **高频维度**（波长短）：直接外推即可
- **低频维度**（波长长）：插值（"拉伸"），让波长变长以匹配新长度
- **中间维度**：两者线性插值

V4-Flash： $rope_scaling.type="yarn"$, `factor=16`, $original_max_position_embeddings=65536$ （64K），目标 1,048,576（1M）= 64K × 16。

**延伸阅读**：主报告 CH 2.1

---

### Q1.24 推理时 KV cache 的"prefix caching"是什么？

**简短回答**：同一个 prompt 被多次请求时，cache 里的 K/V 可以复用，省掉第二次的 prefill 计算。

**详细解释**：两个用户问相同问题，第二次的 prompt 完全一样——KV cache 完全可以复用。**工程实现**：
- 用哈希标识 prompt prefix
- LRU 淘汰旧 cache
- vLLM、TensorRT-LLM 等推理引擎默认支持

V4-Flash 的 CSA/HCA 也支持 prefix caching（KV cache 本身就是可复用的）。

**延伸阅读**：术语表 B.1

---

### Q1.25 Grouped Low-Rank O 投影（grouped LoRA for O）

**简短回答**：把 O 投影（attention 输出）按 head 分组，组内共享一个低秩矩阵——**省 cache + 提速**，质量损失小。

**详细解释**：标准 O 投影： $O = attn_out \cdot W_O$ ，参数量 d² = 4096² = 16.8M。**Grouped 版本**：
- 64 个 head 分 8 组（ $o_groups=8$ ）
- 每组共享一个 $W_O_a: 4096 \to  1024$ （ $o_lora_rank=1024$ ）
- 再 $W_O_b: 1024 \to  4096$ 升回 d
- 参数量： $8 \times  (4096\times 1024 + 1024\times 4096) \approx  67M$——**比 8×16.8M = 134M 少一半**

V4-Flash 沿用 V3 的 grouped low-rank O 投影——Q 投影走 $q_lora_rank=1024$ （同样的"先压后升"），O 投影走 8 组分组。

💡 **面试要点**：grouped low-rank 是"**分组 + 低秩**"的组合。

**延伸阅读**：主报告 CH 3.4 + Q1.5

---

### Q1.26 $O(n^2)$ 灾难的另一个视角：prefill vs decode

**简短回答**：prefill 是"算完整 T×T attention 矩阵"，decode 是"每生成一个 token 算一次新 Q 的 attention"；前者 $O(T^2)$ 算力，后者 $O(T)$ 算力但 T 大时仍吃不消。

**详细解释**：
- **Prefill**（V4-Flash 1M prompt）：单层 4.5×10¹⁵ FLOPs，43 层 1.9×10¹⁷ FLOPs——单卡 H200 ~96 秒
- **Decode**（每 token）：单层 8.6×10⁹ FLOPs，43 层 3.7×10¹¹ FLOPs——单卡 ~0.2 秒（**但 cache 占用 1.4 TB**，根本装不下）

V4 的 CSA/HCA 同时解决**预填充算力**（T²/m 减 4–128 倍）和**解码 cache**（cache 减 4–128 倍）。

**延伸阅读**：主报告 CH 3.1

---

### Q1.27 推理引擎（vLLM / TensorRT-LLM / SGLang）的作用是什么？

**简短回答**：把"训练好的模型"包装成"高吞吐、低延迟的 HTTP 服务"，处理 batching、KV cache 管理、量化 kernel 等。

**详细解释**：原始 HF 模型**不能直接上线**——需要：
- **Continuous batching**：动态拼 batch，最大化 GPU 利用率
- **PagedAttention**（vLLM）：把 KV cache 分页管理
- **TensorRT-LLM kernel 优化**：针对特定 GPU 调优
- **量化加载**：FP4 / FP8 / INT4 权重加载

V4-Flash 官方仓库 `DeepSeek-AI/DeepSeek-V4-Flash` 提供**自己的推理代码**（`inference/` 目录），不走 vLLm（因 CSA/HCA 是新结构）。**生产部署通常需要等 vLLM 适配 CSA/HCA**。

**延伸阅读**：主报告 CH 2.3

---

### Q1.28 SFT / RLHF / DPO 是什么？

**简短回答**：模型"对齐"的三阶段：SFT（监督微调）→ RLHF（PPO 等）→ DPO（直接偏好优化）。V4 走"领域专家 + on-policy 蒸馏"。

**详细解释**：
- **SFT**：在高质量"指令-回答"对上做监督微调
- **RLHF**：训练奖励模型（RM），用 PPO 让 LLM 输出 RM 高分
- **DPO**：跳过 RM，直接在"好回答 vs 坏回答"对上训练

V4 走**领域专家 + on-policy 蒸馏**（V4 创新）：训多个领域专家 → 用专家生成 on-policy 输出 → 主模型蒸馏。

⚠️ **易混淆**：V4 的"on-policy distillation"与 RLHF 共享"用模型自己的输出做训练"思想，但**目标函数不同**——蒸馏是匹配专家分布，RLHF 是匹配奖励。

**延伸阅读**：主报告 CH 7

---

## CH 2. V3 回顾

### Q2.1 V3 的发布时间和参数量是多少？

**简短回答**：DeepSeek-V3 2024-12 发布，总参 671B / 激活 37B；2025-09 发布 V3.2-Exp 改进版。

**详细解释**：
- **DeepSeek-V3**（2024-12-26）：671B 总参 / 37B 激活，**首次大规模 FP8 训练** + MLA + DeepSeekMoE
- **DeepSeek-V3.2** 与 **V3.2-Exp**（2025-09-29）：V3 的后续改进版，引入**aux-loss-free 负载均衡**
- 训练量：V3 公开 14.8T tokens（V3.2-Exp 训练量未官方公开）

V4-Flash 的"前身"是 V3.2——V4 的所有设计都是"V3.2 的瓶颈 + V4 的解法"。

**延伸阅读**：主报告 CH 1.1

---

### Q2.2 V3 与 V2 的关键区别是什么？

**简短回答**：V2 用"细粒度专家 + 共享专家"的 DeepSeekMoE；V3 在 V2 基础上**规模从 236B 扩到 671B**，并首次大规模用 FP8 训练。

**详细解释**：
- **V2**（2024-05）：236B 总参 / 21B 激活，DeepSeekMoE 雏形
- **V3**（2024-12）：671B / 37B 激活，**参数 × 2.8 / 激活 × 1.8**
- **关键技术差异**：
  1. **MLA**：V2 没有，V3 引入
  2. **FP8 训练**：V2 走 BF16，V3 走 E4M3/E5M2 FP8
  3. **专家数**：V2 160 routed / 2 shared，V3 256 routed / 1 shared
  4. **共享专家策略**：V3 收敛到 1 个 shared（V2 的 2 个 shared 表现差）

💡 **面试要点**：V2 → V3 → V4 的演进主轴是"**稀疏化 + 低精度**"。

**延伸阅读**：主报告 CH 1.1

---

### Q2.3 V3 的 MLA 详细原理（与 Q1.6 的差异）

**简短回答**：MLA 把 K、V 压到 $d_c=512$ 潜空间；V3 实际是"V2 的多头 attention + 潜空间压缩"。

**详细解释**（补充 Q1.6）：
- **V3 MLA 输入**： $x \in  \mathbb{R}^{T\times d}$ ，`d=7168`（V3 维度，V4-Flash 是 4096）
- **压缩**： $c_t = W_down \cdot x_t$ ， $c_t \in  \mathbb{R}^{d_c=512}$
- **缓存**：cache 存 $c_t$ （一个 512 维向量）
- **Q 端**： $q_t = W_q_up \cdot x_t$ 直接投影，**不**压缩 Q
- **K, V 端恢复**： $k_t = W_k_up \cdot c_t$ 、 $v_t = W_v_up \cdot c_t$

V3 与 V4-Flash 的 MLA 区别：
- V3：61 层，`d=7168`，128 head
- V4-Flash：43 层，`d=4096`，64 head，**但 V4-Flash 走 MQA（KV 头=1）+ CSA/HCA**，cache 比 V3 MLA 还小

**延伸阅读**：主报告 CH 1.1 + Q1.6

---

### Q2.4 V3 的 MoE 设计（细粒度专家 + 共享专家）

**简短回答**：256 个 routed expert（细粒度）+ 1 个 shared expert（始终激活），top-8 routed 加权 + shared 输出。

**详细解释**：
- **细粒度专家**：单个 expert 中间维度小（如 2048），但 expert 数量多（256），**总参数大、激活小**
- **共享专家**：1 个 shared expert **始终激活**——不参与路由，保证"通用知识"在所有 token 上都有覆盖
- **top-k 路由**：每个 token 选 8 个 routed expert（V3.2 沿用 8，V4-Flash 降到 6）
- **routed scaling factor**：V3 用 1.0，V4-Flash 用 1.5

**为什么"细粒度"？**专家数 N 大 + 单个 expert 中间维度小，比"少量大专家"组合灵活性高。

💡 **面试要点**：细粒度的关键不是"专家小"而是"**专家多 + 单个不独大**"。

**延伸阅读**：主报告 CH 1.1 + Q1.7

---

### Q2.5 V3 的 FP8 训练

**简短回答**：V3 是**第一个用 FP8 训练 671B 模型**的开源工作；前向用 E4M3，反向用 E5M2（动态范围更大）。

**详细解释**：
- **FP8 E4M3**（4 exponent + 3 mantissa）：前向激活 + 权重，范围 ±448，精度较高
- **FP8 E5M2**（5 exponent + 2 mantissa）：反向梯度，范围 ±57344，精度低但范围大
- **Block-wise scaling**：每 128 个元素共享一个 scale factor（ $weight_block_size=[128, 128]$ ）
- **Fine-grained quantization**：按 token × block 动态算 scale

V4-Flash 进一步：attention / shared / 公共部分用 FP8，**routed expert 量化到 FP4**。

⚠️ **易混淆**：FP8 训练 ≠ FP8 推理——FP8 训练指训练时前向/反向用 FP8，FP8 推理指部署时权重用 FP8。

**延伸阅读**：主报告 CH 1.1 + Q1.17

---

### Q2.6 V3 的负载均衡（aux loss）

**简短回答**：训练时加一个"专家负载均衡辅助 loss"，鼓励各 expert 命中率均匀。

**详细解释**：
- **传统 aux loss**： $L_aux = \alpha \cdot \Sigma_e (f_e \cdot p_e)$ ，f_e 是命中率，p_e 是平均概率
- 最小化 L_aux 等价于"让所有 expert 命中率都接近 1/N"
- **问题**：aux loss 会"污染"主 loss
- V3 用 $\alpha=0.0001$ （很小的权重）

V3.2-Exp 改进：改用 **aux-loss-free bias**——给每个 expert 加一个标量 b_e，命中率高 → b 减小，命中率低 → b 增大。

**延伸阅读**：主报告 CH 1.1 + Q2.7

---

### Q2.7 V3.2-Exp 引入的"aux-loss-free bias"是什么？

**简短回答**：放弃 aux loss，改用 per-expert 标量偏置 b_e 调节 top-k 选择；b_e 不进梯度，不影响 routing weight 本身。

**详细解释**：
- 训练时：跟踪每个 expert 的命中率 f_e（用 EMA 平滑）
- 命中率 > 1/N：把 b_e 减小 ε → 该 expert 排名下降 → 命中率回落
- 命中率 < 1/N：把 b_e 增大 ε → 排名上升 → 命中率回升
- b_e 只在 `topk(score + b)` 时加，**不影响 $gate(x) = W_gate \cdot x$ 本身**

**为什么更好**：主 loss 不被 aux loss 污染，训练更稳定。V4-Flash 直接采用（ $topk_method: "noaux_tc"$ ）。

💡 **面试要点**：被问"MoE 怎么均衡负载"时，**先答 aux loss（传统），再答 aux-loss-free bias（V3.2+）**。

**延伸阅读**：主报告 CH 1.1

---

### Q2.8 V3.2 相对 V3 的改进点

**简短回答**：V3.2 主要是 V3 的稳定性 / 推理优化版本——FP8 推理 kernel、aux-loss-free bias、MTP 微调等。

**详细解释**：
- **V3（2024-12）**：671B / 37B 激活，14.8T tokens 训练
- **V3.2 / V3.2-Exp（2025-09）**：在 V3 基础上加入：
  - aux-loss-free bias（见 Q2.7）
  - 优化推理 kernel
  - 改进的 post-training 流程
  - 公开的"思考模式"（reasoning mode）支持

V3.2 是 V4 的"直接前身"——V4 设计文档明确说"V3.2 在 1M 长上下文与超大规模训练上还有四类具体瓶颈"。

**延伸阅读**：主报告 CH 1.1

---

### Q2.9 V3.2-Exp 是什么？

**简短回答**：V3.2 的"实验版"——DeepSeek 2025-09 发布，作为 V3.2 的实验性变体，公开了更多训练细节。

**详细解释**：
- V3.2：定位"生产可用"的稳定版本
- V3.2-Exp：定位"探索性"实验版，公开更多架构细节（特别是 aux-loss-free bias 的具体数值）

V4 设计文档多处引用 V3.2-Exp 报告。V3.2-Exp 的训练量**未官方公开**（v0.1 草稿阶段）。

**延伸阅读**：主报告 CH 1.1 + CH 1.2

---

### Q2.10 V3 训练 14.8T tokens 的含义

**简短回答**：V3 在 14.8T tokens 上训练完成（V3 论文 §5.1 公开）；按 Chinchilla 法则，14.8T/671B ≈ 22 tokens/param——**接近算力最优**。

**详细解释**：
- 14.8T = 14.8 万亿
- 比 LLaMA-2-7B（2T tokens）多 7×
- 14.8T 是"V3 论文公开训练量"

V4-Flash 公开 32T tokens，是 V3 公开 14.8T 的 2.2×——**V4 走"小模型、长训练"路线**。

**延伸阅读**：主报告 CH 1.1 + Q1.16

---

### Q2.11 V3 与 V4 的关系

**简短回答**：V4 是 V3.2 的"超大规模 + 1M 上下文 + 四大创新"版本；V3.2 的四个瓶颈直接催生 V4 的四处创新。

**详细解释**（V3.2 瓶颈 → V4 创新一一对应）：

| V3.2 瓶颈 | V4 创新 |
|---|---|
| 1. 长上下文 FLOPs 灾难（T²） | CSA + HCA 混合注意力（CH 3） |
| 2. KV cache 仍线性增长 | CSA m=4 + HCA m=128 + MQA + FP4 专家（CH 3 + 7） |
| 3. 训练稳定性问题 | mHC 多通道残差（CH 5） |
| 4. 优化器收敛速度 | Muon 正交化优化器（CH 6） |

V4-Flash（284B / 13B 激活）相比 V3.2（671B / 37B 激活）：
- **总参缩 2.4×**（671B → 284B）
- **激活缩 2.8×**（37B → 13B）
- **block 数缩 1.4×**（61 → 43）
- **top-k 缩 1.3×**（8 → 6）

💡 **面试要点**：V4 不是"V3 升级版"而是"V3 的瓶颈回答"——**四大瓶颈 → 四大创新** 一定要背。

**延伸阅读**：主报告 CH 1.2

---

### Q2.12 V3 的 Multi-Token Prediction (MTP) 是什么？

**简短回答**：每个主位置额外预测下一个 token（让监督信号 × 2），提升训练效率 + 数据效率。

**详细解释**：
- 标准 next-token prediction：每个位置预测下一个，监督信号 T 个
- MTP：每个位置预测下 1 + 下 2 个 token，监督信号 2T 个
- V3 用 $num_nextn_predict_layers=1$ （预测下 1 个），V4-Flash 沿用 `=1`

**优势**：数据效率高（同样 token 给出 2× 监督）；推理时可"投机解码"（speculative decoding）——MTP 输出先验证、命中则一次推进多步。

**延伸阅读**：主报告 CH 2.1

---

### Q2.13 V3 的训练时长

**简短回答**：V3 训练用约 2 个月（V3 论文 §5.1 公开：约 2.788M H800 GPU 小时）；V3.2-Exp 与 V4-Flash 的具体时长未公开。

**详细解释**：
- V3：2.788M H800 GPU 小时 = 2× 3584 GPU × 约 16.4 天 = 约 55 天（实际稍长）
- 训练 FLOPs ≈ 6 × 671B × 14.8T ≈ 5.95×10²⁵ FLOPs
- 实际 2.788M GPU 小时 ≈ 32 天（理论最低 8 倍，差距来自 IO、通信、kernel 未完全优化）

V4-Flash 训练 32T tokens，需要的 FLOPs 是 V3 的 32/14.8 × (13/37) × ...，**整体训练成本量级相近**。

**延伸阅读**：主报告 CH 1.1

---

### Q2.14 V3 论文关键章节速查

**简短回答**：V3 论文的关键章节：§1 引言、§3 架构（MLA + MoE）、§5 训练（FP8 + DualPipe + aux loss）、§6 评测。

**详细解释**：
- **§1 引言**：总参 671B / 激活 37B / 14.8T tokens
- **§3 架构**：MLA（d_c=512）、DeepSeekMoE（256 routed + 1 shared, k=8）
- **§4 训练数据**：14.8T tokens 数据集组成
- **§5 训练**：FP8 路径、DualPipe 流水线、aux loss 系数、Pre-Norm
- **§6 评测**：MMLU、HumanEval、GSM8K、Math 等基准

**延伸阅读**：主报告 CH 1.1

---

### Q2.15 V3 vs V4 训练数据的差异

**简短回答**：V3 训练 14.8T tokens（公开）；V4-Flash 训练 32T tokens（公开），**2.2× 量**；数据配比 V4 调整（更多推理 / 代码 / 多语言）。

**详细解释**：
- **V3 数据**：14.8T tokens，中英 8:2
- **V4-Flash 数据**：32T tokens，**更偏 reasoning 与 code**（"思考模式"需要）

V4-Flash 训练量翻倍但激活只 13B——这意味着**单位算力的训练密度大幅提升**（小模型 + 长训练 + Muon 优化器 + FP4 量化）。

**延伸阅读**：主报告 CH 1.1 + CH 3.5

---

### Q2.16 V3 的路由函数

**简短回答**：V3 用 $\mathrm{softmax}(W_gate \cdot x)$ 后取 top-k；V3.2-Exp 引入 "noaux_tc"（无 aux loss + $\mathrm{sigmoid}$ scoring）——V4-Flash 沿用 $\mathrm{sqrtsoftplus}$ 变体。

**详细解释**：
- **V3 标准路由**： $s = \mathrm{softmax}(W_gate \cdot x)$ ，`topk(s, k)` —— 用 $\mathrm{softmax}$ 归一化
- **V3.2-Exp 改进**：用 $\mathrm{sigmoid}(W_gate \cdot x)$ 取代 $\mathrm{softmax}$——$\mathrm{sigmoid}$ 不强制概率和为 1
- **V4-Flash**：用 $sqrt(\mathrm{softplus}(s \cdot W_gate))$ （ $\mathrm{sqrtsoftplus}$ ）——$\mathrm{softplus}$ 的平滑性 + sqrt 的数值稳定性

**延伸阅读**：主报告 CH 2.1

---

### Q2.17 V3 的"训练 14.8T tokens"为什么对 V4 设计重要？

**简短回答**：V3 的 14.8T 是开源 LLM 公开训练量最大的之一；V4-Flash 公开 32T（2.2×），**证明 V4 走"小模型 + 更长训练"路线**。

**详细解释**：
- 14.8T tokens 对 671B 来说 D/N ≈ 22（Chinchilla 最优）——V3 接近"算力最优"
- V4-Flash 284B / 32T tokens → D/N ≈ 113——**显著过训练**
- 这意味着 V4-Flash 的"单位参数训练量"是 V3 的 5×——**"小而精"路线**

**延伸阅读**：主报告 CH 1.1 + Q1.16

---

### Q2.18 V3.2 vs V3.2-Exp 的差异

**简短回答**：V3.2 是"生产版"（部署友好），V3.2-Exp 是"实验版"（公开更多训练细节）；架构差异不大。

**详细解释**：
- **V3.2**：定位稳定生产；FP8 推理 kernel 优化
- **V3.2-Exp**：定位实验探索；公开 aux-loss-free bias 的具体数值；未优化的 FP4 探索

V4 设计文档多处引用 V3.2-Exp 的"瓶颈描述"——V3.2-Exp 是 V4 的"问题清单"。

**延伸阅读**：主报告 CH 1.2

---

### Q2.19 V3 的 DualPipe 流水线

**简短回答**：DualPipe 是 V3 论文公开的"通信-计算 overlap 流水线"，把数据并行 + 流水线并行的通信隐藏在计算里。

**详细解释**：
- **数据并行（DP）**：每张卡持有一份完整参数，参数同步走 all-reduce
- **流水线并行（PP）**：把模型按层切到不同卡，stage 间走 P2P
- **DualPipe**：把"算下一个 stage"和"同步当前 stage 的梯度"重叠，减少 bubble

V3 训练用 2048 H800，DualPipe 让 MFU 达到 ~52%——**比单纯 PP 高很多**。

V4-Flash 训练是否沿用 DualPipe **未公开**——v0.1 草稿阶段。

**延伸阅读**：主报告 CH 1.2

---

### Q2.20 V3 的"负载均衡"为什么对 V4 重要？

**简短回答**：V3 走"aux loss"会污染主 loss，V3.2-Exp 改用 "aux-loss-free bias"——V4-Flash 沿用 V3.2-Exp 的设计（ $topk_method: "noaux_tc"$ ）。

**详细解释**：
- 传统 aux loss：让主 loss 加一项 L_aux 鼓励 expert 命中均匀——影响主 loss 收敛
- aux-loss-free bias：用 per-expert 标量 b_e 调节 top-k，**b_e 不进梯度**——主 loss 不被污染
- V4-Flash： $topk_method="noaux_tc"$——直接走 V3.2-Exp 的方案

💡 **面试要点**：被问"V3.2 与 V3 训练上最大的区别"答"aux-loss-free bias"。

**延伸阅读**：主报告 CH 1.1 + Q2.6/Q2.7

---

### Q2.21 V3 的"4D 并行"组合

**简短回答**：V3 训练用 TP（张量并行）+ PP（流水线并行）+ DP（数据并行）+ EP（专家并行）的 4D 并行。

**详细解释**：
- **TP（Tensor Parallel）**：把单个矩阵乘切到多卡，NCCL all-reduce 通信
- **PP（Pipeline Parallel）**：按层切到多卡，stage 间 P2P 通信
- **DP（Data Parallel）**：每卡持有一份完整参数，all-reduce 同步梯度
- **EP（Expert Parallel）**：把不同 expert 放到不同卡，all-to-all 路由 token

V3 用 2048 H800，4D 组合让 671B 模型训练变成可解问题。V4-Flash 训练量级与 V3 接近，**4D 并行沿用**（具体未公开）。

**延伸阅读**：主报告 CH 1.2

---

## CH 3. V4 概览

### Q3.1 V4 的发布时间和 4 个模型卡是什么？

**简短回答**：DeepSeek-V4 2026-04-24 发布；4 个模型卡 = V4-Pro-Base（1.6T / 49B 激活 / FP8）+ V4-Pro（1.6T / 49B / FP4+FP8）+ V4-Flash-Base（284B / 13B / FP8）+ V4-Flash（284B / 13B / FP4+FP8）。

**详细解释**：
| 模型 | 总参 | 激活 | 上下文 | 精度 | 定位 |
|---|---|---|---|---|---|
| V4-Pro-Base | 1.6T | 49B | 1M | FP8 | 旗舰基座 |
| V4-Pro | 1.6T | 49B | 1M | FP4+FP8 | 旗舰 Instruct |
| V4-Flash-Base | 284B | 13B | 1M | FP8 | 开源主力基座 |
| V4-Flash | 284B | 13B | 1M | FP4+FP8 | 开源主力 Instruct |

**切分逻辑**：
1. **Base vs Instruct**：post-training 流水线差异
2. **FP8 vs FP4+FP8**：Instruct 才有 FP4 变体（FP4 需要 QAT）
3. **Pro vs Flash**：参数规模切片

💡 **面试要点**：4 个模型卡的"两个维度"——**规模（Pro/Flash）× 训练形态（Base/Instruct）**。

**延伸阅读**：主报告 CH 1.3

---

### Q3.2 V4-Flash 的总参 / 激活参数

**简短回答**：本报告全程追踪的目标模型 V4-Flash 总参 284B、激活 13B；激活占总参 4.6%——**典型 MoE 稀疏比**。

**详细解释**：
- **总参 284B**：256 routed expert（FP4）+ 1 shared expert（FP8）+ attention 投影 + embedding
- **激活 13B**：单 token 实际计算的参数（routed 6 × 17M ≈ 100M + shared 17M + attention 100M）
- **稀疏比**：284/13 ≈ 22×——**比 V3.2 的 671/37 ≈ 18× 略高**

⚠️ **易混淆**："激活参数"指"单 token 实际经过的参数"——是**计算量**的代理，不是"运行时加载的参数量"（仍要加载全部 284B 权重）。

**延伸阅读**：主报告 CH 1.3 + CH 2.1

---

### Q3.3 V4-Flash 的核心超参

**简短回答**：43 层、 $hidden_size=4096$ 、64 个 attention head、1 个 KV head（MQA）、256 个 routed expert、k=6、 $hc_mult=4$ 。

**详细解释**：
- $num_hidden_layers=43$ ：Block 数
- $hidden_size=4096$ ：隐藏维度 d
- $num_attention_heads=64$ ：Q 头数
- $head_dim=512$ ：单头维度
- $num_key_value_heads=1$ ：MQA
- $n_routed_experts=256$ ：路由专家数
- $num_experts_per_tok=6$ ：top-k
- $n_shared_experts=1$ ：共享专家
- $moe_intermediate_size=2048$ ：单个 expert 中间维度
- $max_position_embeddings=1,048,576$ ：1M 上下文
- $hc_mult=4$ ：mHC 残差流通道数
- $compress_ratios$ ：44 项 list，前 43 项实际决定逐层压缩比
- $index_topk=512$ ：CSA / HCA 选位置数

💡 **面试要点**：被问"V4-Flash 关键超参"——**43 / 4096 / 64 / 1 / 256 / 6 / hc_mult=4** 这七个数字必背。

**延伸阅读**：主报告 CH 2.1

---

### Q3.4 V4 的"四大创新"是什么？

**简短回答**：CSA + HCA 混合注意力（解决 $O(n^2)$ + KV cache）、mHC 多通道残差（解决训练稳定性）、Muon 正交化优化器（解决收敛速度）、FP4 专家量化（解决 cache 占用）。

**详细解释**（V3.2 瓶颈 → V4 创新）：

| V3.2 瓶颈 | V4 创新 | 章节 |
|---|---|---|
| 1. 长上下文 FLOPs 灾难 | CSA + HCA 混合注意力 | CH 3 |
| 2. KV cache 仍线性增长 | CSA m=4 + HCA m=128 + MQA + FP4 | CH 3 + 7 |
| 3. 训练稳定性问题 | mHC 多通道残差 | CH 5 |
| 4. 优化器收敛速度 | Muon 正交化优化器 | CH 6 |

**核心思想**：
- **CSA + HCA**：把 K、V 沿时间维压缩（CSA m=4 / HCA m=128），从根上解决 $O(T^2)$ 和 cache 线性增长
- **mHC**：用 Sinkhorn-Knopp 投影约束残差流到 4 通道（ $hc_mult=4$ ）
- **Muon**：对梯度做 Newton-Schulz 正交化，让优化方向更"等向"
- **FP4 专家**：routed expert 量化到 FP4（E2M1FN + E8M0 scale）

💡 **面试要点**：四大创新 = **稀疏化（CSA/HCA + MoE）+ 稳定性（mHC）+ 收敛（Muon）+ 量化（FP4）**。

**延伸阅读**：主报告 CH 1.2

---

### Q3.5 V4-Pro 27% FLOPs / 10% KV cache，V4-Flash 10% / 7% 的来源

**简短回答**：V4-Pro 用 1.6T / 49B 激活 + 多数 CSA，V4-Flash 用 284B / 13B 激活 + 更多 HCA + FP4 专家——**Flash 更激进**。

**详细解释**（V4 技术报告 §1 实测数字）：

| 指标 | V3.2（MLA） | V4-Pro | V4-Flash |
|---|---|---|---|
| FLOPs / token | 100% | 27% | **~10%** |
| KV cache size | 100% | 10% | **~7%** |

**为什么 V4-Flash 比 V4-Pro 更省**：
1. **激活参数**：V4-Pro 49B vs V4-Flash 13B（3.8× 差距）
2. **CSA/HCA 配比**：V4-Flash HCA 层数更多（HCA m=128 比 CSA m=4 又省 32×）
3. **FP4 专家**：V4-Flash 256 个 routed expert 全部 FP4，cache 减半
4. **hc_mult=4 残差流复用**：4 份残差流共享 cache
5. **滑动窗口 cache 仅 128 项**

**延伸阅读**：主报告 CH 3.5

---

### Q3.6 1M 上下文是怎么支持的？

**简短回答**：RoPE 走 YaRN 扩展（从 64K 扩到 1M，factor=16）+ 每层 128-token 滑窗 + CSA/HCA 时间维压缩，三档配合。

**详细解释**：
1. **RoPE 扩展**： $rope_scaling.type="yarn"$ ，`factor=16`， $original_max_position_embeddings=65536$ ，目标 1,048,576（1M = 64K × 16）
2. **滑窗**： $sliding_window=128$——每层 attention 永远拼上前 128 个 raw token 的 K、V
3. **CSA/HCA 时间压缩**：CSA m=4 / HCA m=128 把 K、V 沿时间维压缩

**为什么三层都需要**：
- 没有 RoPE 扩展：Q、K 在 > 64K 位置上"未见"过，attention 输出全 0
- 没有滑窗：最近的 token 要走压缩（m=128 时平均信息损失 1/128）
- 没有 CSA/HCA：1M 上下文 cache 装不下

💡 **面试要点**：1M 上下文 = **RoPE + 滑窗 + 时间压缩** 三档配合，缺一不可。

**延伸阅读**：主报告 CH 1.4 + CH 3.4

---

### Q3.7 V4 与 V3 的核心差异（按章节映射）

**简短回答**：V3 走 MLA + 单一 attention + V3 MoE + 标准残差 + AdamW；V4 走 CSA+HCA + V4 MoE（FP4）+ mHC + Muon + FP4 量化。

**详细解释**：

| 维度 | V3.2 | V4-Flash |
|---|---|---|
| 注意力 | MLA（潜空间压缩 K/V）| CSA + HCA（时间维压缩 + Indexer） |
| MoE | 256 routed + 1 shared, k=8, FP8 | 256 routed + 1 shared, k=6, **FP4** |
| 残差 | Pre-Norm + 单残差 | Pre-Norm + **mHC 多通道**（hc_mult=4） |
| 优化器 | AdamW | **Muon**（Newton-Schulz 正交化） |
| 路由 | aux loss | **aux-loss-free bias** |
| 上下文 | 128K | **1M**（YaRN 扩展 + 滑窗 + 压缩） |
| 训练量 | 14.8T tokens | 32T tokens（2.2×）|

V4 在 V3.2 基础上**几乎每个维度都做了改进**——是"全面升级"而非局部优化。

**延伸阅读**：主报告 CH 1.1 + CH 1.2

---

### Q3.8 4 个模型卡分别适合什么场景？

**简短回答**：V4-Pro-Base（API 旗舰基座、需自己微调）/ V4-Pro（生产部署、最高质量）/ V4-Flash-Base（开源基座、需自己后训练）/ V4-Flash（开源主力、直接部署）。

**详细解释**：
- **V4-Pro-Base**：1.6T / 49B / FP8——给需要自己微调做垂直领域模型的企业用户
- **V4-Pro**：1.6T / 49B / FP4+FP8——DeepSeek 自家 API 用的版本，**质量最高**
- **V4-Flash-Base**：284B / 13B / FP8——开源生态用，可作下游研究基座
- **V4-Flash**：284B / 13B / FP4+FP8——**本报告全程追踪的目标模型**

**选型建议**：
- 单 batch 2×H200 / 8×H100：V4-Flash
- 多卡集群 + 质量优先：V4-Pro
- 自训练 / 学术研究：V4-Flash-Base
- API 服务 / 商业部署：V4-Pro（API 闭源）或 V4-Flash（自部署）

**延伸阅读**：主报告 CH 1.3

---

### Q3.9 V4-Flash 的 3 个推理模式

**简短回答**：Non-think（直接出答案）/ Think High（中等推理）/ Think Max（最大推理，推荐 context window ≥ 384K）。

**详细解释**：

| 模式 | $thinking_mode$ | $reasoning_effort$ | 典型场景 |
|---|---|---|---|
| Non-think | `"chat"` | `None` | 日常对话、低风险决策 |
| Think High | `"thinking"` | `"high"` | 复杂问题、规划 |
| Think Max | `"thinking"` | `"max"` | 探索推理极限 |

**切换方式**：API 调用时 $extra_body={"thinking_mode": "thinking", "reasoning_effort": "max"}$ 。

⚠️ **易混淆**：Think Max 不是"模型变聪明了"——是**给了更多"思考 token"预算**。模型本身权重不变。

**延伸阅读**：主报告 CH 2.4

---

### Q3.10 V4-Flash 的训练量

**简短回答**：V4-Flash 公开训练约 32T tokens，是 V3 公开 14.8T 的 2.2×。

**详细解释**：
- **V3 公开 14.8T tokens**（V3 论文 §5.1）
- **V4-Flash 公开 ~32T tokens**（V4 README / config 推断）
- 32T / 284B ≈ 113 tokens/param——**显著过训练**（Chinchilla 最优是 20）
- 训练量翻倍但激活只 13B——"**小模型 + 长训练**"路线

**为什么选过训练**：
- 推理成本低（13B 激活）
- 长训练让 MoE 路由、稀疏 attention 稳定收敛
- Muon 优化器 + FP4 量化让"长训练"不至于成本爆炸

**延伸阅读**：主报告 CH 1.4 + Q1.16

---

### Q3.11 V4-Flash 的 license / 仓库

**简短回答**：MIT License，权重 + 推理代码全部开源；仓库 `DeepSeek-AI/DeepSeek-V4-Flash`（HF 上）。

**详细解释**：
- **License**：MIT（最宽松）
- **仓库**：`DeepSeek-AI/DeepSeek-V4-Flash`（DeepSeek-AI 组织下）
- **可商用、可修改、可分发**——对开源生态友好

V4-Flash 是 V4 系列中**唯一开源 FP4+FP8 Instruct** 的模型卡（V4-Pro 仅 API 闭源）。

**延伸阅读**：主报告 CH 1.4

---

### Q3.12 V4 在 LLM 生态中的定位

**简短回答**：V4-Flash 是"开源 LLM 的 1M 上下文 + 高性价比 + 学术友好"代表——vs V4-Pro 走 API 闭源 / vs Llama-3 / Qwen-3 等走开源 + 128K 上下文。

**详细解释**：
- **vs V4-Pro**：V4-Pro 走 API 闭源，质量更高；V4-Flash 走开源 + 单 batch 2 卡可跑
- **vs Llama-3.1-405B**：Llama-3.1 是稠密 405B，V4-Flash 是 MoE 284B（13B 激活）——V4 推理成本显著低
- **vs Qwen-3-235B-A22B**：Qwen-3 也是 MoE，V4-Flash 走 1M 上下文 + FP4 + Muon
- **vs Claude / GPT**：闭源 API，V4-Flash 是开源替代

**核心差异点**：
1. **1M 上下文**：开源 LLM 极少有
2. **FP4 量化**：开源 LLM 极少有
3. **mHC + Muon**：V4 独有

**延伸阅读**：主报告 CH 1.4

---

### Q3.13 V4-Flash 的"43 层 / d=4096 / 64 head"代表什么设计选择？

**简短回答**：V4-Flash 是"小而精"模型——43 层（V3 是 61）、d=4096（V3 是 7168）、64 head（V3 是 128）——**单 batch 在 2×H200 上可跑**。

**详细解释**：

| 维度 | V3.2 | V4-Flash | 比例 |
|---|---|---|---|
| 层数 L | 61 | 43 | 0.70× |
| 隐藏维度 d | 7168 | 4096 | 0.57× |
| 头数 h | 128 | 64 | 0.50× |
| 专家数 N | 256 | 256 | 1.00× |
| top-k | 8 | 6 | 0.75× |

激活参数 ≈ 0.35× (13/37)，单 batch 显存需求大幅下降。

💡 **面试要点**：V4-Flash 走"**深度 + 宽度都缩，但专家数保留**"——专家数 N 不变是 MoE 设计的关键。

**延伸阅读**：主报告 CH 2.1

---

### Q3.14 V4-Flash 的 1 KV head（MQA）有什么意义？

**简短回答**：V4-Flash $num_key_value_heads=1$ （MQA）——所有 64 个 Q 头共享 1 组 KV，cache 缩 64×，**与 CSA/HCA 配合把 1M 上下文 cache 压到 ~15GB**。

**详细解释**：
- 标准 MHA：64 个 Q 头对应 64 套 KV
- V4-Flash MQA：64 个 Q 头共享 1 套 KV——cache 缩 64×
- 配合 CSA m=4：cache 缩 4×
- 配合 HCA m=128：cache 缩 128×

**为什么 MQA 在 V4 有效**：
- CSA/HCA 把 K、V 压缩后再算 attention——**多套 KV 与 1 套 KV 在压缩后差异小**
- Q 头之间本来就通过 O 投影的"分组"（ $o_groups=8$ ）间接共享——MQA 进一步省

⚠️ **易混淆**：V4-Flash 不是纯 MLA——V3 MLA 把 K、V 压到 $d_c=512$ 潜空间再恢复；V4-Flash 直接走 MQA（1 套 KV），CSA/HCA 压缩后再 attention。

**延伸阅读**：主报告 CH 1.4 + Q1.5/Q1.6

---

### Q3.15 V4-Flash 的 256 routed expert + 1 shared expert + k=6 怎么理解？

**简短回答**：每个 token 从 256 个 routed expert 中选 top-6 激活，加权输出 × 1.5 后与 1 个 shared expert（始终激活）相加。

**详细解释**：
- **256 个 routed expert**：通过 $gate(x) = W_gate \cdot x$ + `topk` 选 6 个
- **1 个 shared expert**：始终激活，**不参与路由**——保证"通用知识"在所有 token 上都有覆盖
- **routed scaling factor = 1.5**：routed 加权和 × 1.5 后与 shared 相加

```python
out = 1.5 * sum(routed_out_i * score_i) + shared_out
```

**为什么 1.5**：V3 用 1.0，V4 调高到 1.5——因为 shared expert 始终激活可能"喧宾夺主"，调高 routed 让稀疏专家的贡献相对突出。

**延伸阅读**：主报告 CH 2.1

---

### Q3.16 V4-Flash 的 mHC $hc_mult=4$ 是什么？

**简短回答**：V4-Flash 用 mHC（Manifold-Constrained Hyper-Connections）多通道残差， $hc_mult=4$ 表示残差流扩展为 4 通道。

**详细解释**：
- **标准残差**： $x_{l+1} = x_l + Sublayer(x_l)$——1 通道
- **mHC 残差**： $X_{l+1} = \alpha \cdot Sublayer(X_l) + \beta \cdot X_l$——X 是 $[T, hc_mult, d]$ ，α、β 走 Sinkhorn-Knopp 投影约束到流形

$hc_mult=4$ 意味着每个 block 维护 4 条并行残差流，**信号分 4 通道传播，深层 block 漂移被分散**。

**为什么 hc_mult=4 有效**：
- 深层 block 漂移（V3.2 稳定性问题）被 4 通道分摊
- 训练稳定，V4-Flash 不需要像 V3.2 那样仔细调 lr / init_std
- 4 通道共享 cache（ $hc_mult$ 不增加 cache 数量——$hc_mult=4$ 让 cache × 1/4）

**延伸阅读**：主报告 CH 1.2 + CH 5

---

### Q3.17 V4-Flash 的 1M 上下文具体怎么存 cache？

**简短回答**：每层 cache = $window_size=128$ （最近 128 token 原始 KV） + `T/4` 或 `T/128`（CSA/HCA 压缩段 KV），实际 cache ≈ 15GB（1M 上下文）。

**详细解释**：
- **滑窗段**：128 个 token 的 raw K、V——128 × d = 128 × 512 = 65K
- **CSA 段**（21 层）：T/4 = 262,144 个压缩 K、V
- **HCA 段**（20 层）：T/128 = 8,192 个压缩 K、V
- **每层 cache** ≈ 65K + 134M 或 4.2M
- **43 层累计** ≈ 15 GB（BF16）

**为什么 cache 不爆炸**：
- HCA 把 cache 缩 128×，是 1M 上下文工程化的关键
- mHC 让 4 通道共享 cache
- FP4 专家权重 cache 进一步压

**延伸阅读**：主报告 CH 3.5

---

### Q3.18 V4-Flash 与 Qwen-3 / Llama-3.1 的核心差异是什么？

**简短回答**：V4-Flash 1M 上下文 + FP4 量化 + mHC + Muon；Qwen-3 / Llama-3.1 走 128K 上下文 + 标准 attention + AdamW。

**详细解释**：
| 维度 | V4-Flash | Qwen-3-235B-A22B | Llama-3.1-405B |
|---|---|---|---|
| 总参 | 284B | 235B | 405B（稠密）|
| 激活 | 13B | 22B | 405B |
| 上下文 | 1M | 128K | 128K |
| 注意力 | CSA + HCA | GQA | GQA |
| 残差 | mHC（4 通道）| 标准 | 标准 |
| 优化器 | Muon | AdamW | AdamW |
| 量化 | FP4 routed + FP8 | FP8 | FP8 |

**V4 独有**：1M 上下文、CSA/HCA、mHC、Muon、FP4——五大创新。

**延伸阅读**：主报告 CH 1.3 + Q3.12

---

### Q3.19 V4-Flash 的"非思考模式"和"思考模式"在模型权重上有差异吗？

**简短回答**：没有——同一份权重。差异在 $thinking_mode$ 字段控制 prompt 模板是否插入 `<think>` 起始 token。

**详细解释**：
- **权重**：V4-Flash Instruct 权重**只有一份**——Non-think / Think High / Think Max 用同一份
- **Prompt 模板**：
  - $thinking_mode="chat"$ ：直接给答案
  - $thinking_mode="thinking"$ ：插入 `<think>` 起始 token
  - $reasoning_effort="max"$ ：在 system prompt 前插入 $REASONING_EFFORT_MAX$ 前缀

**核心**：模型本身不变，**推理时 prompt 模板 + context window** 控制模式。

⚠️ **易混淆**：3 个模式不是"3 个不同模型"——是"3 种 prompt + 3 种 token 预算"。

**延伸阅读**：主报告 CH 2.4

---

### Q3.20 V4-Flash 在 MRCR 等长上下文基准上的表现

**简短回答**：V4-Flash 在 MRCR（Multi-Round Co-Reference Resolution）上比 V3.2 显著提升（具体数字待 V4 评测公开）。

**详细解释**：
- **MRCR**：多轮指代消解——给一个长对话，问"第 N 轮提到的 X 实际指代什么"
- 是 1M 上下文的"试金石"——短上下文模型（< 128K）做不了
- V4 公开评测中，V4-Flash 1M 上下文版的 MRCR 分数比 V3.2 128K 版显著提升

**为什么 V4-Flash 适合 MRCR**：
- 1M 上下文：能装下整个多轮对话
- CSA + HCA：长上下文 attention 仍准
- Think Max 模式：可以"思考"指代链

**延伸阅读**：术语表 B.1

---

### Q3.21 V4-Flash 的"思考模式"最多能用多少 context window？

**简短回答**：Think Max 模式推荐 context window ≥ 384K tokens——因为 reasoning chain 可能非常长。

**详细解释**：
- **Non-think / Think High**：128K context 即可
- **Think Max**：384K context（V4 README 推荐）
- 1M context 是 max_position_embeddings 上限，但实际推理成本随 context 增长

**为什么 Think Max 要 384K**：`<think>` 内的 reasoning 可能非常长；简单任务用 Think Max 是浪费；实际部署时根据任务难度选择模式。

**延伸阅读**：主报告 CH 2.4 + Q3.9

---

### Q3.22 V4-Flash 的"训练 32T tokens"具体意味着什么？

**简短回答**：V4-Flash 训练用约 32 万亿 tokens，是 V3 公开 14.8T 的 2.2×——"小模型 + 长训练"路线的代表。

**详细解释**：
- 32T tokens × 13B 激活参数 = "**小模型、长训练**" 配比
- 实际训练 FLOPs ≈ 6 × 13B × 32T ≈ 2.5×10²⁵（单 token 视角）
- 总训练 FLOPs 还要乘 43（层）× 系数（forward + backward） ≈ 实际 ≈ 5×10²⁶ 量级
- 与 V3 训练 FLOPs ≈ 6×10²⁵ 量级相比，V4-Flash 训练量**显著更大**——但激活小所以单步便宜

**为什么公开 32T**：让社区知道"这是 2.2× V3 训练量"；证明 V4 走"过训练"路线；给开源 LLM 一个"训练量透明"的标杆。

**延伸阅读**：主报告 CH 1.4 + Q1.16/Q3.10

---

### Q3.23 V4-Flash 的 FP4 专家与 FP8 公共部分的分工

**简短回答**：256 个 routed expert 量化到 FP4（ $float4_e2m1fn_x2$ ，block=32，E8M0 scale）；attention / shared expert / 公共部分仍为 FP8（E4M3）。

**详细解释**：
- **FP4 路径**：仅 routed expert 权重；block-wise scale（ $weight_block_size=[32, 32]$ ，block=32）
- **FP8 路径**：attention Q/K/V/O 投影、shared expert、embedding、LM head——`e4m3` + `ue8m0` scale
- **激活值**：默认 BF16，attention 计算时模拟 FP8 量化（QAT）

**为什么 routed expert 走 FP4，shared 不走**：
- routed expert 占总权重绝大部分（256 × 17M = 4.3B，shared 1 个 = 17M）——FP4 省 cache 在 routed
- shared expert 始终激活，影响"通用知识"——保持 FP8 精度
- FP4 需要 QAT 训练——V4 Instruct 才能用，V4 Base 是 FP8

**延伸阅读**：主报告 CH 1.4 + Q1.18

---

### Q3.24 V4-Flash 的"MIT License"具体允许多少事？

**简短回答**：MIT 是最宽松的开源 License——可商用、可修改、可分发，唯一义务是保留版权声明。

**详细解释**：
- **可商用**：可以基于 V4-Flash 做商业产品
- **可修改**：可以 fine-tune、merge、quantize
- **可分发**：可以再发布（保留 License）
- **唯一限制**：必须保留原版权声明
- **无 copyleft**：修改后**不必**开源

**与其他 License 对比**：
- **Apache 2.0**：类似 MIT，但多了"专利授权"条款
- **Llama Community License**：商用月活 > 7 亿需单独授权
- **V4-Flash MIT**：无此限制

**延伸阅读**：主报告 CH 1.4

---

### Q3.25 V4-Flash 的 5 大关键定位总结

**简短回答**：（1）13B 激活 / 2 卡可跑；（2）FP4 专家；（3）1M 上下文；（4）MIT License；（5）3 个推理模式。

**详细解释**：
1. **13B 激活**：单 batch 在 2×H200 / 8×H100 可跑——开源 LLM 极少见
2. **FP4 专家**：256 routed expert 全部 FP4——比 FP8 进一步省 cache
3. **1M 上下文**：默认 $max_position_embeddings=1,048,576$
4. **MIT License**：可商用、可修改、可分发
5. **3 个推理模式**：Non-think / Think High / Think Max（适配不同任务难度）

**目标用户**：AI infra 工程师（2 卡可跑）、长上下文应用开发者（1M）、商业产品（MIT）、推理增强需求（3 个模式）。

**延伸阅读**：主报告 CH 1.4

---

## QA 文档结束

本 QA 涵盖 CH 0（4 个 Q）+ CH 1（28 个 Q）+ CH 2（21 个 Q）+ CH 3（25 个 Q）= **78 个 Q**。
完整的"算子级拆解"请回主报告 `deepseek-v4-flash.md`；CH 4 之后（MoE / mHC / Muon / 后训练）由其他 QA 文档负责。


> 本文件是主报告 `/Users/huangyuxiao/models-arch/report/deepseek-v4-flash.md` 的配套 QA。每道题先给"简短回答"，再展开"详细解释"，最后给"💡 面试要点 / ⚠️ 易混淆"和"延伸阅读"。**主报告里有完整论证的，本 QA 简述 + 给行号。**
>
> **读者画像**：对 LLM 训练有初步认识的学生 / 工程师 + 面试准备 + 自学。
>
> **范围**：CH 4（CSA / HCA 注意力，Q4.1–Q4.38） + CH 5（MoE 路由，Q5.1–Q5.31）= 共 69 道题。

---

## CH 4 — CSA / HCA 注意力

### Q4.1 1M 上下文下标准 Attention 为什么"灾难"？

**简短回答**：标准 attention 的计算量为 $O(T^2 \cdot d)$ 、KV cache 为 $O(T \cdot d \cdot L)$ ，1M 上下文时单 token FLOPs ≈ $8.6 \times  10⁹$ ，KV cache ≈ 1.4 TB / 样本，**既算不动又存不下**。

**详细解释**：

- **FLOPs 灾难**。对 V4-Flash（`d=4096`，`T=1,048,576`），单层单 token 解码 FLOPs ≈ $2 \times  T \times  d = 2 \times  10⁶ \times  4096 \approx  8.6 \times  10⁹$ 。prefill 视角更夸张：单层 $T^2 \times  d_h / 2 \approx  4.5 \times  10¹⁵$ FLOPs，43 层累计 ≈ $1.9 \times  10¹⁷$ FLOPs——单卡 H200 跑都要 96 秒。
- **KV cache 灾难**。 $T \times  d \times  2 \times  L = 1,048,576 \times  4096 \times  2 \times  43 \approx  3.7 \times  10¹¹$ float，BF16 下约 1.4 TB / 样本，100 并发 140 TB——V4-Flash 这种"8×H100 跑得动"的目标完全无解。
- **MLA 救了 cache 但救不了 compute**。MLA 把 K/V 压到 $d_c=512$ ，cache 减到 168 GB，但 attention 计算量仍 $O(T^2)$——T=1M 时 FLOPs 主导项没变。

💡 **面试要点**：要能说清"T² 灾难"和"cache 线性"两个独立瓶颈——CSA / HCA 是同时解决两件事，**不能只答"稀疏化"**。

**延伸阅读**：主报告 §3.1（CH 4 主入口）。

---

### Q4.2 稀疏注意力有哪几类？V4 走哪条路？

**简短回答**：4 大类——(1) fixed pattern（如 Sliding Window、Longformer 的 dilated pattern）、(2) learned（如 Reformer 的 LSH）、(3) top-k（如 Sparse Transformer）、(4) content-based / hash。V4 的 CSA 走 **learned top-k**（Indexer 评分 + top-k 选位置）。

**详细解释**：

| 类别 | 代表 | 优势 | 劣势 |
|---|---|---|---|
| Fixed pattern | Longformer, Mistral Sliding Window | 实现简单 | 位置固化，可能错过关键 token |
| Learned cluster / hash | Reformer LSH, Routing Transformer | 多 query 共享 | 哈希质量难调 |
| Top-k per query | Sparse Transformer (Child 2019), BigBird | 灵活 | top-k 选择本身 $O(T)$ |
| **Content-based top-k** | V4 CSA / HCA | 端到端学习"哪些位置重要" | Indexer 计算不可省 |

V4 的 CSA 严格属于"content-based top-k"：每个 query 通过一个独立 Indexer 算出对所有压缩位置的分数，取 top-512 算 attention。

💡 **面试要点**：能列出 4 类 + 各 1 个代表方法 = 加分。

**延伸阅读**：主报告 §3.2.1（CH 4 主入口）。

---

### Q4.3 V3.2 的 MLA 是怎么压 KV 的？为什么"压了 cache 压不了 compute"？

**简短回答**：MLA 把 K、V 各自先投影到 $d_c=512$ 的潜空间（ $c_t = W_KV \cdot x_t$ ），推理只缓存 $c_t \in  \mathbb{R}^{T\times d_c}$ ，恢复多头 K/V 时再乘 $W_K$ 、 $W_V$ 。 $d_c << h \times  head_dim = 32768$ ，所以 cache 大幅减小；但**实际算 attention 时仍展开成 64 头**，FLOPs $2 \times  T \times  d_h \times  h = 2 \times  T \times  32768$——与标准 attention 等价。

**详细解释**：

- **MLA 投影**： $c_t = W_KV \cdot x_t$ ， $W_KV \in  \mathbb{R}^{d \times  d_c}$ ， $d_c=512$ 。推理时 cache 存 $c_t$ 而不是 $k_t / v_t$ ， $d_c << h \times  d_h = 32768$ ，省一个量级以上。
- **compute 没省**：attention 算的是 $QK^\top$ ，MLA 恢复 $k_t = W_K \cdot c_t$ 代入后， $Q \cdot k_t^\top = Q \cdot (W_K \cdot c_t)^\top$——**这是两次矩阵乘（Q × W_K × c_t），不省 FLOPs**。
- **V3.2 KV cache 实测**：1M × 512 × 61 × 2 ≈ 168 GB / 样本（详见 Q4.1 表格），比纯标准 attention 的 1.4 TB 好，但工程上仍困难。

⚠️ **易混淆**：MLA ≠ "压了 attention 计算量"；它只压了 cache 形状。V4 的 CSA / HCA 才是"压 compute"。

**延伸阅读**：主报告 §3.1 末尾 + §1.1（V3.2 MLA 简介）。

---

### Q4.4 CSA 的"压缩"具体指什么？

**简短回答**：把 K、V 沿时间维做**有学习权重的 $\mathrm{softmax}$-pool**——每 `m=4` 个相邻 token 合成 1 个压缩向量 $ĉ_K / ĉ_V \in  \mathbb{R}^{d}$ ，cache 从 `[T, d]` 减到 `[T/4, d]`。

**详细解释**：

- **数学形式**（公式 3.1）：

  $$

  c_K = \mathrm{softmax}\text{-pool}(W_{kv} x) = \frac{\sum_{i=1}^{m} \exp(s_i) \cdot k_i}{\sum_{i=1}^{m} \exp(s_i)}

  $$

  其中 $s_i = (W_{gate} x)_i$ 是由 token 特征学习出的"权重"——不是简单的平均池。
- **m=4 的含义**：每 4 个相邻 token 池化为 1 个，T 长度 → T/4。
- **重叠模式（overlap=True）**：CSA 进一步维护 `2m=8` 个 token 的滑动窗口，每 4 token 输出 2 个压缩向量（前 4 / 后 4），让边界处不那么"硬切"。

💡 **面试要点**：要解释"为什么是 $\mathrm{softmax}$-pool 不是平均池"——见 Q4.6。

**延伸阅读**：主报告 §3.2 + §3.7.1。

---

### Q4.5 Compressor 这个模块具体做什么？前向怎么走？

**简短回答**：Compressor 是 CSA / HCA 的核心算子，把 `m` 个相邻 token 的 K/V 通过可学习门控池化为 1 个压缩向量。它有 3 个分支：**prefill**（整段批处理）、**decode**（单 token FIFO 累加）、**overlap**（CSA 专属的 2m 窗口）。

**详细解释**：

Compressor 维护两个 buffer（ $kv_state$ 、 $score_state$ ）做 decode 阶段的增量压缩：

- **prefill ($start_pos==0$)**：把整段 seqlen 切成 $n_group = seqlen // m$ 个 m-token 段，reshape 成 $[B, n_group, m, d]$ ，加 per-position bias `ape`，在 `dim=2` 上做 $\mathrm{softmax} + sum$ 。
- **decode ($start_pos>0$)**：每来一个 token 写进 `state[bsz, slot]`，其中 $slot = start_pos % m$ ；当 $(start_pos+1) % m == 0$ （凑齐 m 个），做一次 $\mathrm{softmax}$-pool 输出。
- **overlap (m=4 时)**：`state` 容量扩到 `2m=8`，每 4 token 一次性输出 2 个压缩向量（左半窗口 / 右半窗口），`kv = cat(state[:4,:d], state[4:,d:])`。

**overlap 的关键 trick**：把"前 4 token 给 group A / 后 4 token 给 group B"换成"每个 token 给出 2d 维表示"——前 d 维供前组，后 d 维供后组，避免边界处信息被切碎。

💡 **面试要点**：能讲清"prefill / decode 两条路径 + overlap 是 CSA 专属"即可。

**延伸阅读**：主报告 §3.7.1 + `code-snippets/compressor.py`（核心算子）。

---

### Q4.6 为什么 Compressor 用 $\mathrm{softmax}$-pool 而不是平均池？

**简短回答**：平均池对 4 个 token 一视同仁，但 attention 实际只关心少数"重要" token——$\mathrm{softmax}$-pool 通过可学习 `wgate` 让模型端到端学"哪些 token 该被池化时占更大权重"。

**详细解释**：

- **平均池的问题**：对 `k1..k4` 等权求和，相当于 `attention` 把这 4 个 token 的 `V` 都乘 `1/4`——忽略了"某些 token 信息密度更高"的事实。
- **$\mathrm{softmax}$-pool 的好处**： $s_i = (W_gate \cdot x)_i$ 是 token 维度的"重要性分数"， $exp(s_i)$ 让模型能"硬选"某些 token。训练时梯度会通过 $\mathrm{softmax}$ 自动调高有用 token 的 $s_i$ 。
- **per-position 偏置 `ape`**：Compressor 还学了 `[m, coff*d]` 的 `ape` 参数，加到 score 上——给不同"窗口内位置"额外先验（通常是中间位置权重大，边界权重小）。

💡 **面试要点**：能说清"平均池 vs $\mathrm{softmax}$-pool = 静态 vs 端到端学习"。

**延伸阅读**：主报告 §3.7.1（`+self.ape` 那段）。

---

### Q4.7 overlap 模式（m=4）的 2m=8 token 窗口怎么工作？

**简短回答**：CSA 把 Compressor 维护的 buffer 扩到 `2m=8`，每 4 token 一次性输出 2 个压缩向量——前 4 token 的"前 d 维"组成 group A，后 4 token 的"后 d 维"组成 group B。**两个 group 不重叠，但每个 token 都给两个 group 提供 1 维信息**。

**详细解释**：

源码 $overlap_transform$ （`compressor.py:L29-L36`）：
- 输入 `[b, s, m, 2d]`（已经切成 m=4 段、每段 2d 维）
- 输出 `[b, s, 2m, d] = [b, s, 8, d]`
- 第 0..3 行是"前 token 的前 d 维"（供 group A）
- 第 4..7 行是"后 token 的后 d 维"（供 group B）

**为什么这样设计**：m=4 时压缩比太大，边界处的 4 个 token 直接丢弃信息太重。overlap 模式让 2m=8 个 token 的**全部信息**通过 2 个压缩向量保留下来——单看 group A，4 个 token 全部参与 $\mathrm{softmax}$-pool；单看 group B，也 4 个 token 全部参与。

**HCA 为什么不开 overlap**： $compress_ratio=128$ 已经把 T 压到 8192，每组 128 个 token 的边界处"信息切碎"对总 cache（8192 个向量）影响极小；再开 overlap 会让 $kv_state$ 容量 ×2，得不偿失。V4 设计是"只在 ratio 极小（m=4）时才花这个开销"。

⚠️ **易混淆**：overlap 不是"8 token 互相重叠"——而是"4+4 token 各贡献 1 维给 2 个独立 group"。

**延伸阅读**：主报告 §3.7.1 公式 (3.8) 推导。

---

### Q4.8 CSA 的 prefill 和 decode 路径有什么不同？

**简短回答**：**prefill**（ $start_pos=0$ ）整段批处理——reshape 成 $[B, n_group, m, d]$ 后向量化 $\mathrm{softmax}$-pool；**decode**（ $start_pos>0$ ）单 token FIFO——维护 $kv_state / score_state$ 累加器，每 m 个 token 触发一次压缩。

**详细解释**：

- **prefill**：`seqlen = 8192` 时，整段切成 `8192/4 = 2048` 个 group，向量化做 2048 次 $\mathrm{softmax}$-pool。`cutoff = seqlen - seqlen % m` 处理不被 m 整除的尾部；`remainder` 部分（不足 1 个 group）暂存进 $kv_state$ 。
- **decode**：每来一个 token 写进 `state[bsz, slot]`，其中 $slot = start_pos % m$ 。当 $(start_pos+1) % m == 0$ 才触发一次 $\mathrm{softmax}$-pool；否则 $should_compress=False$ 直接 return。
- **decoder 与 prefill 的统一**：两者都从同一个 `self.compressor` 走，调用方 `Attention.forward` 在 prefill 时用 `kv` 包含 batch 内的所有压缩段；decode 时压缩后的向量写进 $kv_cache[bsz, start_pos//m]$ 等待后续 query 读。

💡 **面试要点**：要意识到"两个路径共享同一份代码"是 V4 工程的优雅之处。

**延伸阅读**：主报告 §3.7.1 + `code-snippets/compressor.py` 第 38-99 行。

---

### Q4.9 HCA 与 CSA 的本质区别是什么？

**简短回答**：HCA 是 CSA 的"超压缩"变体——把 m 从 4 拉到 128，**同时砍掉独立 Indexer**，改用"位置等距采样 top-512"。HCA 的设计前提是"重度压缩后所有位置都重要"，省 Indexer 反而是收益。

**详细解释**：

| 维度 | CSA | HCA |
|---|---|---|
| 压缩比 m | 4 | 128 |
| 压缩后序列长度 | T/4 ≈ 262,144 | T/128 ≈ 8,192 |
| overlap | True | False |
| Indexer | 有（独立 Compressor + Hadamard + FP4） | **无** |
| top-k 选择 | Indexer 评分 | 位置等距 |
| cache 相对量 | 1/4 | 1/128 |

**为什么 HCA 不需要 Indexer**：T/128 = 8192 个压缩位置，**全部参与 attention 也才 $8192 \times  d = 3.4 \times  10⁷$ FLOPs**——T=1M 时这已经远小于 $T\cdot k=1.5\times 10⁹$ 的 top-k 限制项，再做 Indexer 评分反而引入额外开销。V4 的取舍是：**让"压缩足够稀疏"成为前提，省掉 Indexer**。

⚠️ **易混淆**：HCA 不是"更激进的 CSA"——它**省略了 Indexer 整个模块**，这是结构上的差异。

**延伸阅读**：主报告 §3.3（含完整对比表）。

---

### Q4.10 HCA 为什么不需要 Indexer？位置等距采样的依据是什么？

**简短回答**：HCA 假设"重度压缩后 8192 个位置信息量均匀"，按 $(start_pos + 1) // 128$ 的物理位置均匀采样 top-512。**这个假设在 1M 上下文 + m=128 时经验成立**，论文 ablation 显示省 Indexer 不显著损失质量。

**详细解释**：

- **位置等距采样**：源码 $get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)$ 直接生成 `(seqlen // ratio)` 个等距位置索引。
- **核心论点**：当压缩后只剩 8192 个位置，每个位置都是"128 个原始 token 的加权综合"，信息密度比单 token 高。**top-512 / 8192 ≈ 6.25% 的采样率**——6% 的高质量位置已经覆盖了绝大部分长程依赖。
- **省了什么**：省掉一个独立 Compressor（hadamard + FP4 路径）+ Indexer 评分（`einsum + relu + topk`）。在长 prefill 时 Indexer 算量也不小，砍掉后 HCA 路径更轻量。

💡 **面试要点**：能讲清"m 大 → 压缩后信息密度均匀 → 位置采样够用"这个推理链。

**延伸阅读**：主报告 §3.3 公式 (3.5)。

---

### Q4.11 Indexer 是什么？怎么工作？

**简短回答**：Indexer 是 CSA 专属的"压缩位置评分器"——为每个 query 算出对 `T/m` 个压缩位置的分数，取 top-k=512。包含自己的 Compressor（Hadmard + FP4）、query 投影 $wq_b$ 、per-head 权重 $weights_proj$ ，最后 $einsum + \mathrm{ReLU} + topk$ 。

**详细解释**（对应 `code-snippets/indexer.py`）：

```python
# 1. q 端: wq_b + RoPE + Hadamard + FP4 模拟
q = self.wq_b(qr).unflatten(-1, (n_local_heads, head_dim))  # [B, S, 64, 128]
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)                  # Hadamard 旋转
fp4_act_quant(q, fp4_block_size, True)     # FP4 模拟
self.compressor(x, start_pos)             # Indexer 自带 Compressor

# 2. per-head 权重
weights = self.weights_proj(x) * (softmax_scale * n_heads ** -0.5)

# 3. index_score: q 与 T/4 个压缩 key 算内积
index_score = einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])

# 4. ReLU + 权重求和 + topk
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

四步：① 准备 q（投影 + RoPE + Hadamard + FP4）；② 算 weights（per-head 标量）；③ q 与压缩 k 算内积得到 `[B, S, 64, T/m]` 的分数；④ $\mathrm{ReLU}$ 截断 + per-head 加权求和 + topk。

💡 **面试要点**：能写出完整 4 步 = 加分。

**延伸阅读**：主报告 §3.7.2 + `code-snippets/indexer.py`（第 22-54 行）。

---

### Q4.12 Indexer 的 per-head 权重 $w_t$ 是什么？为什么不是 per-token？

**简短回答**： $w_t \in  \mathbb{R}^h$ （h=64）是 Indexer 给每个 query 位置算出的 64 个 per-head 标量权重，与 $index_score$ （shape `[B, S, h, T/m]`）逐 head 相乘。**必须是 per-head 而非 per-token**，否则 64 头共享一个权重会损失"头间特异化"。

**详细解释**：

- **形状**： $weights_proj: ColumnParallelLinear(dim, n_heads)$——把每个 token 的 d 维投影成 $n_heads=64$ 维。
- **per-head 的必要性**： $index_score$ 是 64 头对 T/m 个压缩位置各自打的分； $w_t$ 也要 per-head 让每头有独立的"query 重要性"调节。比如第 3 头擅长语法角色，第 27 头擅长实体识别——它们对"该把哪个压缩位置选进来"的判断权重应该不同。
- **如果用 per-token**（shape `[B, S, 1]`）：64 头被同一个权重拉伸，等于强制所有头共享重要度判断——这在 64 头的稀疏选择器中会损失大量灵活性。

⚠️ **易混淆**： $w_t$ 不是 attention 里的 $W_O$ （输出投影），而是"query 维度的 per-head 缩放因子"。

**延伸阅读**：主报告 §3.7.2 (1)。

---

### Q4.13 Indexer 的 $1/\sqrt{d} \cdot 1/\sqrt{h}$ 缩放因子从哪儿来？

**简短回答**： $1/\sqrt{d}$ 是 attention 的标准缩放（让 $q\cdot k^\top$ 数值稳定）， $1/\sqrt{h}$ 是按头数归一化（让 64 头 $w_t$ 的平均权重稳定在 1 附近）。两者合起来让 $relu(q\cdot k^\top) \cdot w_t$ 的输出均值稳定在 `$O(1)$`，topk 不被某一头主导。

**详细解释**：

源码（`indexer.py:L16, L39`）：
```python
self.softmax_scale = self.head_dim ** -0.5   # 1/√d, d=128
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
```

- **$1/\sqrt{d}$ 来源**：attention 公式 $\mathrm{softmax}(Q\cdot K^\top/\sqrt{d})$ ，分母 $\sqrt{d}$ 让 $q\cdot k^\top$ 的方差 = 1，避免 $\mathrm{softmax}$ 进入饱和区。Indexer 借用同一缩放。
- **$1/\sqrt{h}$ 来源**： $index_score$ 在 dim=2 上对 64 头做 `sum`，等价于把 64 个独立随机变量相加——其方差为 $64 \times  Var(单头)$ 。要保持 `sum(dim=2)` 后的均值稳定在 $O(1)$ ，需要把单头权重预先除以 $\sqrt{h}$ 。
- **综合效果**： $relu(q\cdot k^\top/\sqrt{d}) \cdot w_t$ 对 64 头求和后输出在 $O(1)$ 量级——topk 选位置时不会被某一头的偏大权重"垄断"。

💡 **面试要点**：能讲清"$\sqrt{h}$ 是因 sum-over-heads"即可。

**延伸阅读**：主报告 §3.7.2 (2)。

---

### Q4.14 公式 (3.2) 用 $\mathrm{ReLU}$ 不用 $\mathrm{softmax}$ 的原因？

**简短回答**： $\mathrm{ReLU}$ 是 $O(1)$ 的逐元素 `max(0, x)`， $\mathrm{softmax}$ 需要 O(topk) 个 `exp`；对 1M 上下文（T/m ≈ 262144 个压缩位置、64 头）， $\mathrm{softmax}$ 会让 Indexer 算量增加 1 个 `exp/位置`——**$\mathrm{ReLU}$ 完全省掉这笔开销**。质量影响： $\mathrm{ReLU}$ 把负分数归零，indexer 本来就只挑 top-k，负分数不会进 top-k，截断不损失信息。

**详细解释**：

- **$\mathrm{ReLU}$ 的"硬截断"语义**： $relu(q\cdot k^\top) \cdot w_t$ 把负分数直接归零——只有"正分数"才参与后续 sum-over-heads。
- **$\mathrm{softmax}$ 的"概率化"语义**： $\mathrm{softmax}(q\cdot k^\top) \cdot w_t$ 把分数归一化为概率分布，所有 262144 个位置都有非零贡献——这是给"全 attention"用的，对 top-k 选择器是浪费。
- **数量级差异**：T/m=262144、64 头、batch=1 时， $\mathrm{ReLU}$ 操作 262144 × 64 ≈ 1.7 × 10⁷ 次比较； $\mathrm{softmax}$ 要算 1.7 × 10⁷ 次 `exp`——后者在 GPU 上慢 5-10×。

⚠️ **易混淆**： $\mathrm{ReLU}$ 不是"没有归一化"——它靠 `sum(dim=2)` 后配合 `weights` 隐式归一化；topk 选择完全不依赖分数的绝对值。

**延伸阅读**：主报告 §3.7.2 (4)。

---

### Q4.15 Indexer 内的 Compressor 与 Attention 共用的 Compressor 有什么差异？

**简短回答**：两者**结构相同**（都继承 `Compressor`），但**量化路径不同**——Indexer 内的 `rotate=True` 触发 Hadamard 旋转 + FP4 模拟（ $fp4_act_quant$ ），Attention 共用的 `rotate=False` 走 FP8 量化（ $act_quant$ ）。

**详细解释**：

源码对比：
- `indexer.py:L19` → $Compressor(args, compress_ratio, head_dim, True)$ （rotate=True，FP4）
- `attention.py:L32` → $Compressor(args, compress_ratio, head_dim)$ （rotate=False，FP8）

**为什么需要两套**：

| 路径 | 用途 | 量化要求 |
|---|---|---|
| Indexer | 选 top-k 位置 | 激进（FP4）——只关心"哪些位置最相关"，单位置数值精度不重要 |
| Attention | 算 attention output | 保守（FP8）——压缩向量要进 $\mathrm{softmax}(Q\cdot K^\top)\cdot V$ 算最终输出，量化噪声会被 $\mathrm{softmax}$ 放大 |

**Hadamard 旋转的作用**：把 d 维向量与固定 Hadamard 矩阵相乘——等价于无参数的随机正交变换，**打散信息让量化更鲁棒**（不旋转的话，少数大值主导量化，分布不均；旋转后能量均匀分布到各维度，FP4 的 16 个 level 能更精细地表示）。

💡 **面试要点**：要意识到"**两套 Compressor 共享类、共享主结构、走不同量化**"是 V4 的工程精细之处。

**延伸阅读**：主报告 §3.7.2 (5)。

---

### Q4.16 FP4 vs FP8 在 Indexer / Attention 内的选择依据？

**简短回答**：Indexer 是"sparse top-k 选择器"——只关心哪些位置最相关，单位置精度不重要，可激进量化（FP4）；Attention 实际算 $\mathrm{softmax}(Q\cdot K^\top)\cdot V$——压缩向量是最终输出的原料，量化噪声会被 $\mathrm{softmax}$ 放大，必须保守（FP8）。

**详细解释**：

- **FP4 = $float4_e2m1fn_x2$**：1 sign + 2 exponent + 1 mantissa，4 bit 表示，约 16 个离散 level。
- **FP8 = E4M3**：1 sign + 4 exponent + 3 mantissa，8 bit 表示，256 个 level。
- **选择逻辑**：
  - top-k 选择是"相对排序"——只要 A 比 B 大，不需要 A 精确等于某个值 → FP4 够用
  - attention output 是"加权求和"——$\mathrm{softmax} \cdot V$ 对单 K 值的精度敏感 → FP8 必须
- **QAT（Quantization-Aware Training）**：FP4 路径在训练时就用 $fp4_act_quant$ 模拟量化（FP16 计算 + 量化/反量化），让前向分布贴近部署形态——这是 V4 训练时已经"按 FP4 形态优化"的关键。

💡 **面试要点**：要能说"QAT 是什么"+"为什么 Indexer 不在乎 FP4 的精度损失"。

**延伸阅读**：主报告 §3.2 + §3.7.2 (5)。

---

### Q4.17 top-k=512 的选择依据是什么？

**简短回答**：V4 论文 ablation 显示 top-512 在 1M 上下文下达到"质量 / FLOPs"最优折中——top-256 时 PPL 显著上升（漏掉太多长程依赖），top-1024 时 FLOPs 翻倍（sparse 优势不再），top-512 是甜点。

**详细解释**：

- **质量侧**：1M 上下文 = $2^2⁰ = 1,048,576$ 位置；top-512 = 0.05% 采样率。**经验上 0.05%–0.1% 足够覆盖长程依赖**（Mistral Sliding Window 选 4096 已经证明小窗口不显著损害长程建模）。
- **FLOPs 侧**：T=1M 时， $T\cdot k = 1,048,576 \times  512 \approx  5.4 \times  10⁸$ ，与 $T^2/m = 10¹^2/4 = 2.5 \times  10¹¹$ （CSA T² 项）相比，top-k 项是次要——**sparse 的主要收益来自压缩而不是 top-k 截断**。
- **与 V3 沿用**：V3 时代就选 512，V4 沿用同一选择（实验证明 256 / 1024 都没明显更优）。

⚠️ **易混淆**：top-k=512 不是"绝对最优"——它是 1M 上下文的 sweet spot；T=128K 时 top-128 就够，T=10M 时可能需要 top-2048。

**延伸阅读**：主报告 §3.2（公式 3.2）+ §3.4。

---

### Q4.18 $sparse_attn_kernel$ 的 TileLang 实现做了什么？

**简短回答**：把 CSA / HCA 选中 top-k 个位置后实际算 attention output 的过程用 TileLang 写成 GPU kernel。核心是 **online $\mathrm{softmax}$** 三件套（scores_max / scores_scale / scores_sum）+ $block=64 num_stages=2$ 的双缓冲 pipeline + 编译期常量化 `h=64, d=128`。

**详细解释**（对应 $code-snippets/sparse_attn_kernel.py$ ）：

```python
@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)  # 512/64 = 8
```

**关键设计**：
- `h, d` 编译期常量化（`h=64, d=128`），TileLang 会做特化（循环展开、register 分配）
- `b, m, n, topk` 符号维度（运行期可变）
- `block=64`：每 64 个 top-k 位置为 1 个 block
- $num_stages=2$ ：双缓冲 pipeline（边算当前 block 边加载下一 block）
- $num_blocks = topk / 64 = 8$ ：每 query 8 个 block

💡 **面试要点**：理解"online $\mathrm{softmax}$ 在 sparse attention 中的角色"是关键。

**延伸阅读**：主报告 §3.7.3（最详尽算子级拆解）。

---

### Q4.19 online $\mathrm{softmax}$ 的三件套是哪三个状态？怎么更新？

**简短回答**： $scores_max$ （历史最大值）、 $scores_scale$ （rescale 因子 $exp(m_{t-1} - m_t)$ ）、 $scores_sum$ （累加 `exp(s - m)`）。每进一个新 block，先更新 max，再算 rescale 因子，最后把旧累加乘上 rescale + 加上新 block 的贡献。

**详细解释**（公式 3.9）：

$$

\begin{aligned}
m_t &= \max(m_{t-1}, b_t^{\max}) \\
s_t &= \exp(m_{t-1} - m_t) \\
\Sigma_t &= \Sigma_{t-1} \cdot s_t + b_t^{\mathrm{sum}}
\end{aligned}

$$

- **$scores_max$**：`init = -inf`，每个 block 后 $reduce_max(acc_s, dim=1)$ 更新——跟踪历史最大值。
- **$scores_scale$**： $exp(scores_max_prev - scores_max)$——rescale 因子，让旧累加 $acc_o / sum_exp$ 乘上这个因子，相当于在新的 max 基准下重新归一化。
- **$scores_sum$**：累加 $exp(acc_s - scores_max)$ （减 max 防 exp 溢出），公式 $sum_exp * scores_scale + scores_sum$ 。
- **每 block 末尾**： $acc_o = acc_o * scores_scale + 新 block 贡献 $ ，最后 $o = acc_o / scores_sum$ 。

**为什么是 online**：512 个位置不必一次性算完，可以 8 个 block 渐进更新。**比"全 $\mathrm{softmax}$ 后再算 V"省内存**——后者要把 $T\times T$ 分数矩阵存在 shared memory。

💡 **面试要点**：能讲清"$m_t$ 更新 → $s_t$ rescale → 累加器同步更新"三步即 OK。

**延伸阅读**：主报告 §3.7.3 (3) 公式 (3.9)(3.10)。

---

### Q4.20 $block=64 / num_stages=2$ 的切分依据是什么？

**简短回答**：`block=64` 由 shared memory 容量决定（每 block 64 个 K/V × 128 dim × 2B = 16KB，恰好适配 H100 的 100+ KB shared memory）； $num_stages=2$ 是双缓冲 pipeline 的标准选择（3+ stages 在 H100 上收益递减）。

**详细解释**：

- **`block=64` 选多大**：
  - 太小（如 16）：8 个 block 内的 GEMM 算量不足，GPU SM 闲置
  - 太大（如 256）： $kv_shared$ 占 $256 \times  128 \times  2B = 64KB$ ，超过 SM 的 shared memory 上限
  - **64 平衡点**：8 个 block × 64 位置 = 512 top-k，shared memory 用 16KB/warp，余量充足
- **$num_stages=2$ 为何不是 3 或 4**：
  - num_stages=1：无 pipeline，load 与 compute 串行
  - num_stages=2：双缓冲，当前 block 算 GEMM 时下一 block 加载 kv_shared——延迟隐藏到位
  - num_stages=3+：寄存器压力增大，H100 上实测收益 < 5%——不值得
- **num_blocks = topk / block = 8**：8 个 pipeline stage × 双缓冲 = 16 步延迟隐藏。

💡 **面试要点**：能讲清"block 大小 = shared memory 容量 / 平衡点"即可。

**延伸阅读**：主报告 §3.7.3 (4)。

---

### Q4.21 $attn_sink$ 的语义是什么？为什么要加？

**简短回答**： $attn_sink$ 是 per-head 标量偏置（V3.2 沿用），在 online $\mathrm{softmax}$ 末尾加进 $sum_exp$——给"无 K/V 位置"贡献一个 fallback 概率。**防止 query 的 top-k 分数都很低时 output 变成 NaN/塌缩**。

**详细解释**：

源码：
```python
attn_sink: T.Tensor[(h,), FP32]    # per-head 标量
...
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
acc_o[i, j] /= sum_exp[i]
```

- **场景**：某个 query 被 indexer 选错了——top-512 个位置分数都很低， $\mathrm{softmax}$ 分布"无主"。如果没有 attn_sink， $sum_exp$ 极小， $acc_o / sum_exp$ 数值爆炸。
- **机制**：把 attn_sink 当作"虚拟 token"加进 $\mathrm{softmax}$ 分母——$\mathrm{softmax}([s1, s2, ..., s512, attn_sink])$ 保证总概率质量 ≥ $exp(attn_sink)$ ，output 有界。
- **per-head**：每个 head 学自己的 attn_sink 标量，让每头独立调"防御强度"。

**与 HCA 无关**：HCA 没有 indexer 也照样有 attn_sink——它是 V3 沿用的"通用防御"。

⚠️ **易混淆**：attn_sink 不是 attention sink（StreamingLLM 里的"前几个 token"）——它是 per-head 可学习标量。

**延伸阅读**：主报告 §3.7.3 (5) + §3.6 源码片段 3。

---

### Q4.22 $compress_ratios$ 逐层配置的依据是什么？前 2 层 / 21 层 / 20 层？

**简短回答**：V4-Flash 的 $compress_ratios = [0,0,4,128,4,128,...,4,0]$ 共 44 项——**前 2 层纯滑窗（ratio=0）**让浅层观察原始 token；**21 层 CSA（ratio=4）+ 20 层 HCA（ratio=128）逐层交替**保质省算力；末尾占位 0 不读。

**详细解释**：

```
compress_ratios = [
  0,    # Layer 0   纯滑窗 (window=128)
  0,    # Layer 1   纯滑窗 (window=128)
  4,    # Layer 2   CSA (m=4)
  128,  # Layer 3   HCA (m=128)
  4, 128, 4, 128, ..., 4, 128, 4,  # Layer 4-41 交替
  0,    # Layer 42  CSA
  0,    # 占位
]
```

**逐层模式**：
- **Layer 0, 1**： $compress_ratios=0$ ，纯 sliding window（`window=128`），完全不上压缩——浅层需要 raw token 信号建模
- **Layer 2-42**：41 层交替 CSA / HCA
  - CSA（m=4）：21 层
  - HCA（m=128）：20 层
  - **Layer 42 也是 ratio=4**（CSA），最后一层给"细节保护"——深层的输出质量敏感
- **末尾占位 0**：config 列表有 44 项但实际只读前 43 层（ $layer_id \in  [0, 42]$ ）

**为什么交替而非全部 CSA / 全部 HCA**：
- 全部 CSA：FLOPs 省 4 倍但 cache 占 25% of V3.2
- 全部 HCA：cache 省 128 倍但 m=128 损失太多细节
- **交替：CSA 隔层"补细节"，HCA 隔层"省算力"**——既保质又省 FLOPs/cache

💡 **面试要点**：能讲清"前 2 层纯滑窗 + 21 CSA + 20 HCA + 末尾占位"的数字结构。

**延伸阅读**：主报告 §3.4（公式 3.7）+ $_work/config.json$ 第 66 行。

---

### Q4.23 sliding_window=128 的作用是什么？

**简短回答**：每层 attention 都有一个 128-token 的"原始 K/V 窗口"——最近 128 个 token 永远走 raw attention，不上压缩。保证**最近语境的精度无损**。

**详细解释**：

- **物理位置**：在 kv_cache 中，前 $window_size=128$ 个位置存原始 K/V； $window_size:$ 之后是 CSA / HCA 压缩向量。
- **prefill 路径**：`attention.py:L83-L91` 把原始 K/V 放进 $self.kv_cache[:bsz, :128]$ ，再拼接 $kv_compress$ ； $topk_idxs$ 总是把 `[0:128]` 范围的索引拼在前面。
- **decode 路径**： $kv_cache[:bsz, start_pos % win]$ 循环写入最近 128 个 token 的 K/V——$start_pos % win$ 实现环形 buffer。
- **滑窗 + 压缩的语义**：**滑窗保"短期精确"，压缩保"长期稀疏"**。两者拼接后每个 query 的 attention 范围 = "最近 128 个 raw + 远端 512 个压缩"。

**为什么选 128**：Mistral / V3 时代用 4096 / 8192；V4 把窗口压到 128 是因为压缩已经覆盖长程依赖，滑窗只负责"局部精度"。128 是经验最优——再小（如 32）会损害局部句法，再大（如 1024）会与压缩 cache 重复。

💡 **面试要点**：要意识到"滑窗 + 压缩"是**互补**而非冗余。

**延伸阅读**：主报告 §3.4（window_size 始终拼接部分）。

---

### Q4.24 公式 (3.7) 逐层分支怎么写代码？

**简短回答**：源码在 $Attention.__init__$ 中按 $compress_ratios[layer_id]$ 选择性创建 `Compressor` 和 `Indexer`；`forward` 中根据 `ratio` 决定走 Indexer 路径（ratio=4）还是位置 topk 路径（ratio=128），并与 $window_topk_idxs$ 拼接。

**详细解释**（代码结构）：

```python
# __init__ 中按 ratio 分支
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)
    else:
        self.indexer = None
```

- `ratio=0` → 既无 Compressor 也无 Indexer，纯 $kv_cache[:bsz, :128]$
- `ratio=4` → Compressor（`overlap=True`）+ Indexer（Hadamard + FP4）
- `ratio=128` → Compressor（`overlap=False`），**不创建 Indexer**

```python
# forward 中拼接 topk
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)  # CSA 路径
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)  # HCA 路径
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

**核心**：无论 ratio=0/4/128，**$topk_idxs$ 最终都是一个 `[B, S, 128+512]` 的索引列表**，kernel 一视同仁地 gather + attention。

💡 **面试要点**：理解"同 kernel 不同分支"是 V4 的工程优雅。

**延伸阅读**：主报告 §3.6 源码片段 3 + `code-snippets/attention.py` 第 31-79 行。

---

### Q4.25 V4-Flash 实测 10% FLOPs / 7% KV cache 的来源？

**简短回答**：相对 V3.2（100% / 100%），V4-Flash 在 1M 上下文下把单 token FLOPs 压到 ~10%、KV cache 压到 ~7%。这来自 5 个工程优化的叠加：(1) CSA / HCA 压缩省 T² 项；(2) hc_mult=4 残差流复用省 cache；(3) FP4 专家省 cache；(4) 共享 MQA-KV；(5) 滑窗 cache 仅 128 项。

**详细解释**：

- **CSA 单独**：FLOPs 省 4 倍（理论 T²/4 vs T²）
- **HCA 单独**：FLOPs 省 128 倍（理论 T²/128 vs T²），cache 省 128 倍
- **50% CSA + 50% HCA 配合**：FLOPs 节省约 6.5%（粗估），但 V4 论文 10% 数字包含了：
  - ① 滑窗层的全 attention（不是 sparse）
  - ② Indexer 计算开销
  - ③ YaRN 频率内插
  - ④ grouped low-rank O 投影（ $o_groups=8$ ）
- **KV cache 7% 怎么来**：
  - CSA / HCA 压缩：1/4 + 1/128 配合 ≈ 1/8
  - 滑窗仅 128 项：忽略不计
  - hc_mult=4 残差流共享：cache × 1/4
  - FP4 专家权重：~37 GB / 277B = 7%（专家权重本身省一半）
  - **7% 是多个工程优化叠加的最终值，不是单一因素**

💡 **面试要点**：要意识到"10% / 7%"是端到端数字，不要把它直接对应到"sparse 4×"。

**延伸阅读**：主报告 §3.5（详细理论估算）。

---

### Q4.26 CSA + HCA + 滑窗 + MQA + mHC + FP4 的协同效应是什么？

**简短回答**：6 项工程优化在 4 个维度叠加——(1) **attention 计算稀疏化**（CSA/HCA）；(2) **attention cache 稀疏化**（CSA/HCA + 滑窗）；(3) **MoE cache 压缩**（FP4 专家）；(4) **残差流复用**（mHC hc_mult=4 + MQA-KV 共享）。

**详细解释**：

| 优化项 | FLOPs 省 | Cache 省 | 备注 |
|---|---|---|---|
| CSA / HCA sparse | 4-128× | 4-128× | 核心 |
| Sliding window 128 | — | T→128（局部） | 短程保真 |
| MQA-KV（ $num_key_value_heads=1$ ）| — | 64× | 64 个 Q 头共享 1 个 K/V |
| mHC hc_mult=4 | — | 4× | 4 通道残差共享同一份 cache |
| FP4 routed expert | — | 2× | 单 expert 25.2M → 6.3M |
| grouped low-rank O | — | 4× | 64 头分 8 组，O 投影 1024 维 |

**总 cache 节省比** = 4（CSA） × 64（MQA） × 4（mHC） × 2（FP4） ≈ **2000×**（理论下界）。实际 V4-Flash ~7% / V3.2 100% ≈ 14×——理论值与实测有差距，原因是工程实现不可能每个优化都跑到理论极限。

**FLOPs 节省** = 4-128× (attention) + 其它 ≈ **10×**（实测）。

💡 **面试要点**：能列出 4 个维度的优化项 + 数字 = 加分。

**延伸阅读**：主报告 §3.5 末尾（"为什么 V4-Flash 实测 ~7% 不只是 25%"部分）。

---

### Q4.27 训练 vs 推理的 attention 路径差异是什么？

**简短回答**：**训练时** prefill 长序列一次过、topk_idxs 在 prefill 阶段批量生成、attention 与 backward 联动；**推理时** 分 prefill（处理 prompt）与 decode（自回归生成）两阶段——decode 阶段单 token 流入，每 token 重算一次 top-k。

**详细解释**：

- **训练时**：
  - 一次前向处理 batch 内所有 token（如 8192）
  - prefill 路径：整段批处理压缩 + 整段算 attention
  - backward 时梯度要追溯到 Compressor / Indexer 的中间状态
  - **不区分 prefill / decode**——只是"全段批处理"
- **推理 prefill**（长 prompt）：
  - 一次性处理完整 prompt（可能是 1M tokens）
  - 走 prefill 路径（reshape + 整段 $\mathrm{softmax}$-pool）
  - 顶部 top-512 索引批量生成
- **推理 decode**（batch=1）：
  - 每次来一个新 token，单 token 流入
  - Compressor 维护 $kv_state / score_state$ 累加器
  - Indexer 重算 top-512（每 token 一次）——**这是 decode 阶段最大的额外开销**
  - sparse_attn_kernel 单 query 启动

**关键差异**：

| 维度 | 训练 | 推理 prefill | 推理 decode |
|---|---|---|---|
| Compressor 路径 | prefill | prefill | decode（FIFO） |
| Indexer 计算 | 整段批 | 整段批 | 单 token |
| 注意力 kernel | 全 batch | 全 batch | 单 query |
| Cache 写入 | 全段 | 全段 | 单 token |

**Decode 阶段的 Indexer 开销**是 V4 的工程瓶颈——每生成一个 token 都要重算 top-512。V4 论文没有公开具体数字，但 batch=1 decode 延迟中 Indexer 占比可能达 30%+。

💡 **面试要点**：要意识到"训练 vs 推理的 attention 不是同一条代码路径"。

**延伸阅读**：主报告 §3.6 + §3.7.1（prefill vs decode 路径对比）。

---

### Q4.28 "V3.2 vs V4-Flash" 注意力对比一览？

**简短回答**：V3.2 走纯 MLA（d_c=512 latent 投影）+ dense attention；V4-Flash 走 MLA + CSA/HCA 混合 + 滑窗 + Indexer。FLOPs 10×、cache 14× 节省。

**详细解释**：

| 维度 | V3.2 (DeepSeek-V3.2-Exp) | V4-Flash |
|---|---|---|
| 注意力类型 | MLA | MLA + CSA/HCA |
| 压缩 K/V | d_c=512 latent | 额外 m=4 / m=128 软压缩 |
| Cache shape | `[T, 512]` | `[128 + T/4, d]` 或 `[128 + T/128, d]` |
| FLOPs（1M 上下文） | 100% | ~10% |
| KV cache（1M 上下文）| 100% | ~7% |
| 滑窗 | 无 | 128 |
| 选位置机制 | 全 attention | Indexer top-512 (CSA) / 位置 top-512 (HCA) |
| attn_sink | 有 | 有（沿用） |
| 实现 | 标准 dense | sparse + TileLang kernel |

**关键差异**：V3.2 解决"压 cache"，V4 解决"压 cache + 压 compute"。两者是**演进关系**而非替代——V4 完整保留 MLA 的 $d_c=512$ 设计，再叠一层时间维软压缩。

💡 **面试要点**：能讲清"V3 解决 cache，V4 解决 compute"是核心对比。

**延伸阅读**：主报告 §1.2（V3.2 四大瓶颈）+ §3.5（V4 实测）。

---

### Q4.29 为什么 V4 不用 Linear Attention？

**简短回答**：Linear attention 把 $\mathrm{softmax}$ 替换为核函数 $φ(q) \cdot φ(k)^\top$ ，可 $O(T)$ 计算——但**核函数近似 $\mathrm{softmax}$ 损失精度**，V4 在 1M 上下文要求下质量不能妥协；CSA / HCA 是"在保持 $\mathrm{softmax}$ 精度的前提下做 sparse"，是 V4 的工程取舍。

**详细解释**：

- **Linear attention 的优势**： $O(T \cdot d^2)$ ，无 T² 项；推理 $O(1)$ per token。
- **Linear attention 的劣势**：
  - 核函数 `φ(x) = elu(x) + 1` 等不能精确表达 $\mathrm{softmax}$ 的"指数放大"行为
  - 长程任务（PPL、长上下文检索）质量明显下降
  - 训练-推理一致性差：很多实现要重写 kernel
- **V4 的取舍**：
  - 1M 上下文对长程质量敏感（needle-in-haystack、多跳 QA）
  - **CSA / HCA 在"保 $\mathrm{softmax}$ 精度的前提下做 sparse"**——top-512 的位置仍走完整 $\mathrm{softmax}$
  - 复杂度从 $O(T^2)$ 降到 $O(T\cdot k + T^2/m)$——sparse 但不近似
- **V4 不是不用 linear attention，而是"在它做不到的位置不用"**：短程走 attention + 滑窗；长程走 sparse attention（top-k 仍 $\mathrm{softmax}$ ）。

⚠️ **易混淆**：Linear attention ≠ 稀疏 attention——前者是**近似**（kernel 替换），后者是**截断**（只算部分位置）。

**延伸阅读**：主报告 §3.1 末尾（"CSA + HCA 的核心创新"）。

---

### Q4.30 为什么 V4 不用 State-Space Model（Mamba）？

**简短回答**：SSM（Mamba / S4）走 `$O(T)$` 递归推理，质量在语言建模上仍弱于 Transformer；V4-Flash 定位是"高质量 1M 上下文 + MoE"，**质量优先**——CSA / HCA 是"在 Transformer 框架内做 sparse"，比换 SSM 风险更小。

**详细解释**：

- **SSM 的优势**：
  - 训练 $O(T)$ 并行，推理 $O(1)$ per token
  - 长程依赖建模理论上无限长
- **SSM 的劣势**：
  - 在语言建模上 PPL 仍差于 Transformer（同参数量）
  - 检索类任务（"找到第 3 段提到的实体"）质量显著下降
  - 训练稳定性差（HiPPO 初始化敏感）
  - 主流 benchmark（MMLU、HumanEval）落后 Transformer 5-10 个百分点
- **V4 的取舍**：
  - V4-Flash 是"开源主力 Instruct"——质量门槛高，不能换 backbone
  - CSA / HCA 保留 Transformer 框架，质量与 V3 持平
  - 工程上只需加 Compressor / Indexer / sparse_kernel——风险可控
- **SSM 仍是开放方向**：DeepSeek / Mistral / Qwen 都在做 SSM-Transformer 混合架构，但目前没有"全 SSM 替代 Transformer"的强证据。

💡 **面试要点**：要意识到"Transformer + sparse" vs "SSM" 不是"哪个更快"的问题，而是"质量与速度的 trade-off"。

**延伸阅读**：主报告 §3.1 末尾（V4 设计动机）。

---

### Q4.31 Lightning Attention 与 V4 Indexer 的区别？

**简短回答**：Lightning Attention（RetNet / TransNormer 系）是"线性 attention + chunk-wise 并行"，**近似 $\mathrm{softmax}$**；V4 Indexer 是"压缩位置评分器"——**不替代 attention，只选 top-k 位置给真正的 $\mathrm{softmax}$ attention 算**。两者结构完全不同。

**详细解释**：

- **Lightning Attention**：
  - 把 $\mathrm{softmax}(Q\cdot K^\top)\cdot V$ 替换为 $φ(Q) \cdot (φ(K)^\top \cdot V)$ （核函数 + 矩阵结合律）
  - chunk 内并行列式不变，跨 chunk 递归
  - 训练 $O(T \cdot d^2)$ ，推理 $O(1)$ per token
  - **代价**：核函数 `φ` 不能精确表达 $\mathrm{softmax}$ ，质量有损
- **V4 Indexer**：
  - **不替代 attention**——只算"哪些位置重要"
  - 真正算 attention output 时仍是标准 $\mathrm{softmax}$ （ $\mathrm{softmax}(q\cdot k^\top/\sqrt{d})\cdot v$ ，见 sparse_attn_kernel）
  - Indexer 用 $\mathrm{ReLU}$ + topk 选 top-512
  - **代价**：每 token 算一次 Indexer，但 attention 只算 512 个位置
- **关键区别**：

| 维度 | Lightning Attention | V4 Indexer + sparse_attn |
|---|---|---|
| 计算量 | $O(T \cdot d^2)$ | $O(T \cdot k \cdot d)$ + $O(T \cdot d \cdot d_{\mathrm{idx}})$ |
| 替代 $\mathrm{softmax}$ ？ | 是 | **否**（仍 $\mathrm{softmax}$ ） |
| 质量 | 近似，损失 | sparse，保精度 |
| 推理每 token | $O(1)$ | O(k + d_idx) |

⚠️ **易混淆**：V4 Indexer 不属于 linear attention 家族——它只**选位置**，不**算 attention**。

**延伸阅读**：主报告 §3.7.2（Indexer 设计）。

---

### Q4.32 V4 sparse attention 与 NSA（Native Sparse Attention）的关系？

**简短回答**：NSA（DeepSeek 2025 论文）也走"压缩 + 选位置"路线——与 V4 CSA 思路同源，但**NSA 是 top-k of 3 种固定 pattern（compress / sliding / selected）**，V4 CSA 是 top-k by Indexer score。V4-Flash 是 NSA 思路的**生产化演化**。

**详细解释**：

- **NSA 思路**（Yuan et al. 2025）：
  - 把 K/V 分 3 个分支：压缩 branch（reduce）、滑窗 branch（local）、selected branch（top-k）
  - 3 个分支各自的 attention 拼接输出
- **V4 CSA / HCA 思路**：
  - CSA：1 个压缩 branch + 滑窗 + Indexer top-k（合并为 1 个 attention）
  - HCA：1 个压缩 branch + 滑窗（无 Indexer）
- **共同点**：
  - 都用"压缩 K/V"省 cache / compute
  - 都用"top-k / sliding window" 选位置
- **差异**：
  - NSA 的 3 个 branch 各自算 attention，输出拼接
  - V4 把压缩向量 + top-k 选位置合并为 1 个 attention kernel
  - V4 HCA 进一步省 Indexer
- **V4 是 NSA 的工程化演化**：V4-Flash 论文在 2026 年发布，吸收了 NSA（2025）思路并加入 Indexer 学分、滑窗、cascading 配合等生产级细节。

💡 **面试要点**：要意识到 V4 sparse attention 不是凭空设计，是站在 NSA / Mistral Sliding Window 等前人工作上的工程演化。

**延伸阅读**：主报告 §3.1 末尾（V4 创新定位）。

---

### Q4.33 MQA（Multi-Query Attention）怎么省 cache？V4 怎么用？

**简短回答**：MQA 让所有 Q 头共享 1 个 K/V 头——cache 从 $[T, h \times  d_h]$ 减到 $[T, d_h]$ 。V4-Flash 用 $num_key_value_heads=1$ （64 个 Q 头共享 1 个 K/V），cache 比 MHA 省 64×。

**详细解释**：

- **MHA（Multi-Head Attention）**：每个 Q 头有独立的 K/V。 $num_key_value_heads = num_attention_heads = 64$ 。
- **MQA（Multi-Query Attention）**：所有 Q 头共享 1 个 K/V。 $num_key_value_heads = 1$ 。
- **GQA（Grouped-Query Attention）**：分组共享。 $num_key_value_heads$ 在 [1, 64] 之间。

V4-Flash 配置 $num_key_value_heads=1$——严格 MQA。**cache 形状**： $[B, T, 1, d_h] = [B, T, 512]$ （d_h=512）；vs MHA 的 `[B, T, 64, 512]`，省 64×。

**为什么 MQA 不显著损害质量**：
- K/V 头少 → 参数量少 → 容量小
- 但**多 Q 头仍可学不同 attention pattern**（每个 Q 头有独立 $W_Q$ ）
- 经验上 $num_key_value_heads=1$ 在 7B+ 模型上质量损失 < 1%（参见 PaLM、Llama 2 的 GQA 实验）

**V4 的 cache 节省比** = MQA 64× + CSA/HCA 4-128× + mHC 4× + 滑窗 ≈ 1000-10000×（理论下界）。

💡 **面试要点**：能区分 MHA / MQA / GQA 三档 + 各自的 cache 形状。

**延伸阅读**：主报告 §3.4（MQA-KV 部分）。

---

### Q4.34 $q_lora_rank=1024$ 和 $o_lora_rank=1024$ 是什么？为什么需要？

**简短回答**： $q_lora_rank=1024$ 是 Q 投影的中间 bottleneck 维度（ $d \to  1024 \to  h \times  d_h$ ，d=4096, h×d_h=32768）； $o_lora_rank=1024$ 是 O 投影的中间维度。**两者都用 low-rank 投影省参数 + 提速**，组内头共享 O 投影（ $o_groups=8$ ）。

**详细解释**：

- **Q 投影**（`attention.py:L22-L24`）：
  - 原始： $W_Q \in  \mathbb{R}^{d \times  h \times  d_h} = \mathbb{R}^{4096 \times  32768}$ （134M 参数）
  - LoRA： $W_Q = W_Q_b \cdot W_Q_a$ ， $W_Q_a \in  \mathbb{R}^{4096 \times  1024}$ ， $W_Q_b \in  \mathbb{R}^{1024 \times  32768}$——**4.2M + 33.5M = 37.7M 参数**，省 72%
  - 等价变换： $q = x \cdot W_Q_a \cdot W_Q_b$ 替代 $x \cdot W_Q$
- **O 投影**（`attention.py:L27-L28`）：
  - grouped low-rank：`64 头分 8 组`，每组 8 头共享 $o_lora_rank=1024$ 的 O 投影
  - $wo_a \in  \mathbb{R}^{4096/8 \times  1024 \times  8} = \mathbb{R}^{512 \times  1024 \times  8}$ （4.2M）
  - $wo_b \in  \mathbb{R}^{8 \times  1024 \times  4096}$ （33.5M）
  - 原始： $W_O \in  \mathbb{R}^{32768 \times  4096}$ （134M）
  - **省 72%**
- **为什么需要 low-rank**：
  - 训练时省 72% 参数量
  - 推理时省 72% 权重加载带宽
  - **质量损失 < 1%**——Q/O 矩阵的低秩结构本身近似

💡 **面试要点**：能算清"4.2M + 33.5M vs 134M 的参数对比"即可。

**延伸阅读**：主报告 §3.4（grouped low-rank O 投影）。

---

### Q4.35 公式 (3.5) HCA 位置 top-k 的具体代码？

**简短回答**： $get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)$ 直接生成 `(seqlen // ratio)` 个等距位置索引，**不走 Indexer 评分**。

**详细解释**（`attention.py:L72` 上下文）：

```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)  # 前 128 个
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)  # CSA
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)  # HCA
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

**$get_compress_topk_idxs$ 的逻辑**：
- HCA 时 `ratio=128`， $end_pos // 128$ 是当前可用的压缩位置数
- 选 $[0, 1, 2, ..., end_pos//128 - 1]$ 的前 `topk=512` 个（或全部）
- 加 `offset` 让索引指向正确的 $kv_cache$ 位置（HCA 路径中 offset = 滑窗大小 128）

**核心假设**：HCA 的 8192 个压缩位置（1M / 128）信息量均匀，**前 512 个就够**——这是 V4 的工程经验。

⚠️ **易混淆**：HCA 的"位置 top-k"不是"均匀采样"——是直接取前 512 个位置（`positions[0:512]`）。

**延伸阅读**：主报告 §3.3 公式 (3.5) + `code-snippets/attention.py` 第 72-79 行。

---

### Q4.36 滑窗中的 128 个 raw K/V 在 cache 中怎么放？

**简短回答**：源码在 $Attention.kv_cache$ 中前 $window_size=128$ 个位置存原始 K/V； $window_size:$ 之后是 CSA / HCA 压缩向量。 $topk_idxs$ 总是把 `[0:128]` 范围拼在前面。

**详细解释**：

- **Cache 布局**：
  ```
  kv_cache: [B, window_size + max_compressed_len, d]
            └─ 前 128: 原始 K/V ─┘└──── 压缩 K/V (T/4 或 T/128) ────┘
  ```
- **prefill 路径**（`attention.py:L83-L88`）：
  ```python
  if seqlen <= win:
      self.kv_cache[:bsz, :seqlen] = kv
  else:
      cutoff = seqlen % win
      self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = \
          kv[:, -win:].split([win - cutoff, cutoff], dim=1)
  ```
  滑窗只保留**最近 128 个 token**——更早的 token 从滑窗 cache 中滚出。
- **decode 路径**（`attention.py:L95`）：
  ```python
  self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
  ```
  环形 buffer： $start_pos % 128$ 实现 128 个位置的循环覆盖。
- **拼接 topk**： $topk_idxs = cat([window_idxs, compress_idxs])$——window_idxs 是 `[0, 1, ..., 127]`，compress_idxs 偏移 `128`（`offset = win`）后指向压缩段。

💡 **面试要点**：要理解"环形 buffer"是滑窗的标准实现。

**延伸阅读**：主报告 §3.6 + `code-snippets/attention.py` 第 38-39, 83-98 行。

---

### Q4.37 $attn_sink$ 是什么时候加进 sparse_attn_kernel 的？

**简短回答**：在 online $\mathrm{softmax}$ 循环结束后、 $acc_o / sum_exp$ 之前——$sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])$ 。这是 V3.2 沿用的"虚拟 token"防御。

**详细解释**：

源码位置（ $code-snippets/sparse_attn_kernel.py$ 第 70-73 行）：

```python
# 5. attn_sink: V3.2 沿用的偏置 (per-head 标量)
for i in T.Parallel(h):
    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
for i, j in T.Parallel(h, d):
    acc_o[i, j] /= sum_exp[i]
```

**位置选择**：
- 在 8 个 block 循环**结束后**——此时 $scores_max / sum_exp / acc_o$ 已是 512 位置累加结果
- 在 $acc_o / sum_exp$ **之前**——attn_sink 是 $\mathrm{softmax}$ 分母的一部分，必须先加
- 减 $scores_max$ 是为了数值稳定（与 online $\mathrm{softmax}$ 主体一致）

**为什么 V3 引入**：StreamingLLM / V3 都观察到——attention 在没有 sink token 时， $\mathrm{softmax}$ 分布会"无主"，sum_exp 极小，output 数值爆炸。加 attn_sink 给所有 head 一个 fallback 概率，**保证 sum_exp 始终 > $exp(attn_sink - scores_max)$**。

⚠️ **易混淆**：attn_sink 不是"前 4 个 token"（那是 StreamingLLM 的 attention sink），是**per-head 学到的标量**。

**延伸阅读**：主报告 §3.7.3 (5) + $code-snippets/sparse_attn_kernel.py$ 第 70-73 行。

---

### Q4.38 CSA / HCA 训练时 Indexer 怎么参与反向传播？

**简短回答**：Indexer 的参数（ $wq_b$ 、 $weights_proj$ 、Compressor 的 `wkv/wgate/ape`）都通过 $index_score = relu(q\cdot k^\top) \cdot w$ 反向传播；Indexer 输出 $topk_idxs$ 是离散索引（无梯度），但 score 本身连续可导。Compressor 的 `wkv/wgate/ape` 通过"压缩向量的 k"反向传播。

**详细解释**：

- **topk_idxs 无梯度**：argmax / topk 操作不可微——梯度不能通过索引反传
- **但 score 连续可导**： $index_score = relu(q\cdot k^\top) \cdot w$ 是连续函数，反向传播通过 `q` / `k` / `w` 三个路径：
  - `q` 路径： $wq_b$ 的梯度
  - `k` 路径：Indexer-Compressor 的 `wkv/wgate/ape` 的梯度
  - `w` 路径： $weights_proj$ 的梯度
- **Compressor 梯度**：
  - **Indexer-Compressor**（`rotate=True`）的 `wkv/wgate/ape` 通过 `k = Compressor(x)` 反传
  - **Attention-Compressor**（`rotate=False`）的 `wkv/wgate/ape` 通过 `k = Compressor(x)` 在 sparse_attn_kernel 内反传
- **topk 截断的反向 trick**：被 topk 选中的位置梯度正常回传；**未被选中的位置梯度 = 0**——这相当于"硬注意力掩码"，实现简单。
- **Indexer vs Attention 反向差异**：
  - Indexer 反向：标准 dense matmul（ $wq_b$ 、 $weights_proj$ 、Indexer-Compressor）
  - Attention 反向：sparse_attn_kernel 内只有 top-512 位置反传，其余位置为 0

💡 **面试要点**：要理解"topk 是不可微操作，但 score 本身连续可导"——这是 CSA 能端到端训练的关键。

**延伸阅读**：主报告 §3.6 + `code-snippets/indexer.py`（ $index_score = ... .relu_() * weights.unsqueeze(-1)$ ）。

---

## CH 5 — MoE 路由

### Q5.1 MoE 与稠密 FFN 的对比？

**简短回答**：稠密 FFN 所有 token 走同一份参数；MoE 把 FFN 拆成 E 个 expert，每个 token 只激活 top-k 个——**总参可任意大、激活参数保持小**。V4-Flash 256 routed expert + 1 shared，每 token 激活 7 个（6 routed + 1 shared），激活 13B / 总参 284B。

**详细解释**：

- **稠密 FFN**（LLaMA / Mistral）： $FFN(x) = W_2 \cdot silu(W_1 \cdot x) \odot  W_3 \cdot x$ ，单层 3 × 4096 × 11008 = 135M 参数。所有 token 走同一份。
- **MoE FFN**（DeepSeekMoE）：拆成 E=256 个 expert，每个 expert 是独立 $\mathrm{SwiGLU}$ ；Gate 网络把每个 token 路由到 top-k=6 个 expert。
- **V4-Flash 数据**：
  - 256 routed expert × 3 × 4096 × 2048 = 6.45B / 层
  - 1 shared expert × 3 × 4096 × 2048 = 25.2M / 层
  - 每 token 激活 6 routed + 1 shared = 7 × 25.2M = 176M / 层
  - **总参 284B / 激活 13B = 22× 稀疏比**

**核心优势**：在不增加激活 FLOPs 的前提下，**总参可以拉大 22×**——专家容量大、可学更多知识，但单次推理只跑 13B 激活。

💡 **面试要点**：要算清"总参 vs 激活参数"的稀疏比。

**延伸阅读**：主报告 §4.1（含完整参数表）。

---

### Q5.2 Top-k 路由是什么？

**简短回答**：Gate 网络算出 token 对 E 个 expert 的亲和度分数，挑 top-k 个激活——**token x 算出 `[E]` 维分数，取 top-k 个最大值对应的 expert 索引**。

**详细解释**：

- **流程**：
  1. $W_g \cdot x \to  \mathbb{R}^E$ （E=256）
  2. 评分函数（ $\mathrm{sqrtsoftplus}$ / $\mathrm{softmax}$ / $\mathrm{sigmoid}$ ）
  3. top-k 选索引（k=6）
  4. 对 top-6 个 expert 算前向，加权求和
- **稀疏性**：256 个 expert 中只有 6 个被激活，250 个完全跳过——**节省 250/256 ≈ 97.7% 的 expert FFN 计算**
- **可微性**：topk 索引不可微，但通过"路由权重 = score.gather(topk_idx)"保持连续性
- **Top-1 vs Top-k**：
  - Top-1（Switch Transformer）：极度稀疏，但容量小
  - Top-2/4/6/8：平衡容量与稀疏度
  - V4-Flash 选 k=6（在 13B 激活约束下尽量大）

⚠️ **易混淆**：Top-k 路由 ≠ "k 个分数高的 expert 加权"——是"k 个分数高的 expert **按分数**加权"，权重来自原分数。

**延伸阅读**：主报告 §4.2 公式 (4.2) + §4.5 源码片段 1。

---

### Q5.3 Switch Transformer 的简化路由？

**简短回答**：Switch Transformer（Fedus et al. 2022）把 top-k 简化为 **top-1**——每个 token 只路由到 1 个 expert，**简化 routing 逻辑 + 减少通信量**。V4 仍用 top-6，但 Switch 思想（极简路由）是后续 MoE 工作的起点。

**详细解释**：

- **Switch Transformer 路由**：
  - 每个 token 路由到 score 最高的 1 个 expert
  - `indices = scores.argmax(dim=-1)`，单次 expert FFN
  - 通信量：每个 token 1 个 expert id
- **优势**：
  - 实现极简（argmax 比 topk 简单）
  - 通信量最小（专家并行时 all-to-all 负载轻）
- **劣势**：
  - 单 expert 容量小，质量下降
  - 负载不均更严重
- **V4 的取舍**：
  - V4-Flash 走 top-6（不是 top-1）——质量优先
  - 但**保留 Switch 的"极简 dispatch"思想**——`MoE.forward` 用 Python 循环 + `torch.where` 找 token，避免复杂 CUDA kernel

💡 **面试要点**：Switch Transformer 是 MoE 简化的里程碑，但要意识到"V4 不直接用 top-1"。

**延伸阅读**：主报告 §4.1（V3.2 vs V4 路由对比表）。

---

### Q5.4 细粒度专家（V2 风格）是什么？

**简短回答**：V2 风格把 expert 数量从 8 拉到 160+、单 expert 维度从 8192 砍到 256（细粒度）——**专家更多、更细、更专门化**。V4-Flash 沿用这一思想：256 个 expert、 $moe_intermediate_size=2048$ （单 expert 中间维度），比 V1 的"少而大"expert 更细。

**详细解释**：

- **V1 风格**（GShard, 2020）：8 个 expert、单 expert 维度 8192。每个 expert 容量大但数量少。
- **V2 风格**（DeepSeekMoE, 2024）：160-256 个 expert、单 expert 维度 256-2048。**专家多 + 细粒度**。
- **V4 风格**：
  - 256 routed expert
  - 单 expert 中间维度 $moe_intermediate_size=2048$ （远小于 V1 的 8192）
  - 单 expert 参数量 = 3 × 4096 × 2048 = 25.2M
- **细粒度的优势**：
  - 单 expert 容量小 → 学"专精"模式（语法、实体、数学、代码等）
  - 多 expert 组合 → 表达力强（256 个 25.2M expert 总容量 6.45B / 层）
  - 激活比例 = 6/256 = 2.3%——**比 V1 的 1/8 = 12.5% 更稀疏**

⚠️ **易混淆**："细粒度"是指 expert 数量多、单 expert 小——不是指 token 路由细。

**延伸阅读**：主报告 §4.1（DeepSeekMoE 引用）。

---

### Q5.5 共享专家（V3 风格）的作用？

**简短回答**：共享 expert 永远激活，承担"通用知识"——让 256 个 routed expert 可以更专门化（学"专项能力"），不浪费容量在"通识"上。**降低 routed expert 的等效容量压力**。

**详细解释**：

- **设计动机**：
  - 纯 routed MoE：每个 expert 都要学"通用 + 专项"——容量浪费
  - 纯 shared MoE：所有 expert 共享——失去细粒度优势
  - **共享 + routed 混合**：1 个 shared 学通用 + 256 个 routed 学专项
- **V4-Flash 数值**：
  - 1 shared expert = 25.2M 参数（永远激活）
  - 承担 1.08B / 13B ≈ 8% 的激活开销
  - 256 routed = 6.45B 总参，6 个激活
- **典型分工**：
  - shared 学：常见词、句法、标点、常用短语
  - routed 学：领域知识（代码 / 数学 / 多语言 / 长尾实体）
- **训练稳定性**：shared 永远激活，给所有 token 一个"基线信号"——routed 偏离时不会塌缩。

💡 **面试要点**：要意识到"shared expert 不只是为了省参数"——是为了"把 routed expert 解放出来做更细的专门化"。

**延伸阅读**：主报告 §4.1（共享 expert 解释）。

---

### Q5.6 "1 + 256 + 6" 拓扑的具体含义？

**简短回答**：1 个 shared expert（永远激活）+ 256 个 routed expert（top-6 选择激活）——**单层 MoE 形态**。V4-Flash 43 层 × 这个拓扑 = 43 个独立的 MoE 层（**层间不共享 expert**）。

**详细解释**：

- **1**：1 个 shared expert（永远激活， $\mathrm{SwiGLU}$ dim=2048）
- **256**：256 个 routed expert（ $\mathrm{SwiGLU}$ dim=2048，FP4 量化）
- **6**：每 token 激活 top-6 个 routed + 1 个 shared = 7 个 expert 实际工作
- **层间独立**：43 层各自有独立的 256 routed + 1 shared——共 43 × 257 = 11,051 个 expert
- **总参**：43 × 257 × 25.2M ≈ 278B（其中 routed 占 277B，shared 占 1.08B）
- **激活**：43 × 7 × 25.2M ≈ 7.6B（experts 部分）+ 4-5B（attn/norm/embed） ≈ 13B

**与 V3.2 的关键差异**（V3.2 是 1+256+8，V4-Flash 是 1+256+6）：

| 维度 | V3.2 | V4-Flash |
|---|---|---|
| 1 (shared) | 1 | 1 |
| 256 (routed) | 256 | 256 |
| 6 / 8 (top-k) | 8 | **6** |
| routed_scaling_factor | 1.0 | **1.5** |
| num_hash_layers | 0 | **3** |

V4 把 k 从 8 砍到 6，配合 $route_scale=1.5$——"用 6 个专家达到 8 个专家的等效表达力"。

💡 **面试要点**：要算清"1+256+6" 的 13B 激活 vs 256 routed 的 6.45B / 层参数。

**延伸阅读**：主报告 §4.1（含完整参数计算）。

---

### Q5.7 公式 (4.1) Gating 评分是什么？

**简短回答**： $g_i = \mathrm{sqrtsoftplus}((W_g \cdot x)_i)$——Gate 矩阵 $W_g \in  \mathbb{R}^{d \times  E}$ 乘 token $x \in  \mathbb{R}^d$ ，得到 256 维 logits，再过 $\mathrm{sqrtsoftplus}$ 评分函数。

**详细解释**（`code-snippets/gate.py` L20 + L26）：

```python
scores = linear(x.float(), self.weight.float())  # W_g · x → [B, 256]
scores = F.softplus(scores).sqrt()                 # sqrtsoftplus
```

- **Gate 矩阵**： $W_g \in  \mathbb{R}^{4096 \times  256}$ （1M 参数）—— V4-Flash 把 d=4096 投影到 E=256
- **$\mathrm{softplus}$**： $\mathrm{softplus}(z) = log(1 + exp(z))$——把任意实数映射到 `(0, +∞)`
- **开方**： $sqrt(\mathrm{softplus}(z))$——V4 的工程小创新

**为什么 $\mathrm{sqrtsoftplus}$ 不是 $\mathrm{sigmoid}$ / $\mathrm{softmax}$**：
- $\mathrm{sigmoid}$ 把分数压到 (0, 1)，对负 logits 不敏感
- $\mathrm{softmax}$ 让一个 expert 主导，破坏"细粒度"假设
- $\mathrm{sqrtsoftplus}$ 在 $z \to  -\infty$ 时趋近 0，在 $z \to  +\infty$ 时趋近 `sqrt(z)`——**数值范围更稳**

💡 **面试要点**：能写出"$sqrt(\mathrm{softplus}(z))$"和"它与 $\mathrm{sigmoid}$/$\mathrm{softmax}$ 的差异"即可。

**延伸阅读**：主报告 §4.2 公式 (4.1)。

---

### Q5.8 $\mathrm{sqrtsoftplus}$ 是什么？为什么不用 $\mathrm{softmax}$ / $\mathrm{sigmoid}$ ？

**简短回答**： $\mathrm{sqrtsoftplus}(z) = sqrt(log(1 + exp(z)))$——$\mathrm{softplus}$ 后开方。**不归一化、不压范围、数值稳定**——是 routing weight 与 topk 分数"同源"的最简选择。

**详细解释**：

- **$\mathrm{softplus}$ 性质**：
  - $z \to  -\infty$ ： $\mathrm{softplus}(z) \to  0$
  - $z \to  +\infty$ ： $\mathrm{softplus}(z) \to  z$
  - 处处可导，平滑
- **开方的作用**：
  - 把 $\mathrm{softplus}(z)$ 的 `(0, +∞)` 范围压到 `(0, sqrt(z))`
  - 让 256 个 expert 的分数**在 $O(1)$ 量级**（不是 O(z) 或 $O(1)$ ）
  - topk 选位置时不被某一极端高分主导
- **vs $\mathrm{sigmoid}$**：
  - $\mathrm{sigmoid}$ ：`(0, 1)`，对负 logits 截断为 0.5
  - $\mathrm{sqrtsoftplus}$ ：`(0, sqrt(z))`，对负 logits 截断为 0——更"硬"的稀疏性
- **vs $\mathrm{softmax}$**：
  - $\mathrm{softmax}$ ：归一化到概率分布，所有 expert 都有非零权重
  - $\mathrm{sqrtsoftplus}$ ：独立计算，每个 expert 权重独立
  - $\mathrm{softmax}$ 让"top expert 主导"，破坏细粒度
- **V4 论文 ablation**：在 MMLU / GSM8K 上 $\mathrm{sqrtsoftplus}$ 略优于 $\mathrm{sigmoid}$ ，显著优于 $\mathrm{softmax}$

⚠️ **易混淆**： $\mathrm{sqrtsoftplus}$ 不是"开方 + $\mathrm{softplus}$ 分别"——是"先 $\mathrm{softplus}$ 再开方"。

**延伸阅读**：主报告 §4.2 公式 (4.1) 解释。

---

### Q5.9 $noaux_tc$ 路由方法是什么？

**简短回答**： $noaux_tc$ 是 V4-Flash 配置中的 $topk_method$——指 V4 论文中的 **"no-aux top-k constraint"** 方法，本质就是**aux-loss-free bias**（无 aux loss、靠 per-expert 标量偏置平衡负载）。V4 论文完整名是 $noaux_tc$ （no-aux-loss top-k with transform constraint）。

**详细解释**：

- **$noaux_tc$ 含义**：
  - `no aux`：没有传统 aux loss（ $\lambda \cdot \Sigma f_i \cdot p_i$ ）
  - `tc`：transform constraint，指"用 bias 变换分数约束 topk"
- **核心机制**：
  1. 每个 expert 维护标量偏置 $b_e$ （fp32，可训练）
  2. topk 选择用 $score + b_e$
  3. 训练时按命中频率更新 $b_e$ （过载减小、欠载增大）
  4. 推理时 b 冻结
- **V3.2 沿用**：V3.2-Exp 论文就提出 aux-loss-free bias；V4 沿用并调优
- **config 表现**： $"topk_method": "noaux_tc"$ （`config.json` 第 60 行）——告诉训练框架"用这个方法"

💡 **面试要点**：要意识到 $noaux_tc$ 就是 aux-loss-free bias 的工程名。

**延伸阅读**：主报告 §4.2 + §4.3.2（aux-loss-free 详解）。

---

### Q5.10 Aux-loss-free bias 的核心思想？

**简短回答**：每个 expert 维护一个可训练标量偏置 $b_e$——topk 选择用 $score + b_e$ ，**$b_e$ 不进 routing weight 计算**。训练时按命中频率更新 $b_e$ （过载减、欠载增），推理时冻结。**核心创新是"不污染主损失梯度"——bias 只影响 topk 选择**。

**详细解释**：

**核心规则**：
1. 每个 routed expert `e` 维护偏置 $b_e \in  \mathbb{R}$ （fp32，形状 `[256]`）
2. 训练 step 结束时统计各 expert 命中频率 $f_e$
3. 目标频率 $p = 1/256 \approx  0.39%$
4. 更新规则：
   ```
   b_e ← b_e - η · (1/E - f_e)         (E=256, η=0.001, 论文指定)
   ```
   - 若 $f_e > p + \delta$ （过载）： $b_e -= \eta$ → 下次 topk 选该 expert 概率下降
   - 若 $f_e < p - \delta$ （欠载）： $b_e += \eta$ → 下次 topk 选该 expert 概率上升
5. **关键约束**： $b_e$ **仅影响 topk 选择**，**不参与 routing weight**

**为什么"aux-loss-free"比"aux loss"好**：

| 方案 | 优势 | 劣势 |
|---|---|---|
| 传统 aux loss $\lambda \cdot \Sigma f_i \cdot p_i$ | 实现简单 | 污染主损失梯度；λ 是超参需调 |
| Expert capacity | 强约束，硬件友好 | 硬截断会丢 token，性能差 |
| **Aux-loss-free bias** | 不污染 loss；无 λ 超参；硬件透明 | 偏置收敛慢，需几千 step 才稳定 |

V3 论文报告：200B token 训练后，aux-loss-free 与 aux loss 达到几乎相同的负载均衡（max=0.012 vs 0.011），但下游任务 +0.5% MMLU。

💡 **面试要点**：要算清"$b_e$ 只影响 topk，不影响 weight"是核心 trick。

**延伸阅读**：主报告 §4.3.2（含完整更新规则）。

---

### Q5.11 偏置 $b_e$ 的更新规则细节？

**简短回答**： $b_e ← b_e - \eta \cdot (1/E - f_e)$ ，其中 `E=256`， $\eta=0.001$ （论文指定）， $\delta=0.05$ （容忍带）。**每 step 末尾更新**——过载减、欠载增，**仅影响 topk 选择**。

**详细解释**：

- **更新公式**：
  ```
  b_e ← b_e - η · (1/E - f_e)
  ```
  - $f_e$ ：当前 step 中 expert `e` 被 top-6 选中的频率
  - $1/E = 1/256 \approx  0.39%$ ：目标频率
  - $\eta = 0.001$ ：学习率（论文指定）

- **等价描述**：
  - 若 $f_e > p + \delta$ （过载）： $b_e -= \eta$ → 下 step topk 选该 expert 概率下降
  - 若 $f_e < p - \delta$ （欠载）： $b_e += \eta$ → 下 step topk 选该 expert 概率上升
  - δ 是"容忍带"，论文给 δ ≈ 0.05
- **频率统计**：每 step 末尾 `counts = bincount(indices.flatten())`， $f_e = counts[e] / total_tokens$
- **关键约束**： $b_e$ **不进梯度**——它是 buffer-style 参数（fp32 单独存），更新由训练框架侧（不在本 inference 仓库内）
- **推理时冻结**：推理路径完全跳过 `bincount` 统计， $b_e$ 是常量偏移

💡 **面试要点**：要能写出"$b_e -= \eta \cdot (1/256 - f_e)$"公式。

**延伸阅读**：主报告 §4.3.2（含完整公式 + δ 容忍带）。

---

### Q5.12 公式 (4.2) Top-k 选择的具体代码？

**简短回答**（`code-snippets/gate.py` L29-L34）：

```python
# bias 加和（仅 score 模式）
scores = scores + self.bias

# hash 模式：一次查表
if self.hash:
    indices = self.tid2eid[input_ids]
# score 模式：top-6
else:
    indices = scores.topk(self.topk, dim=-1)[1]
```

**详细解释**：

- **score 模式**（后 40 层）：
  - $score_with_bias = score + self.bias$ —— bias 仅影响 topk
  - `indices = scores.topk(self.topk, dim=-1)[1]` —— 取 top-6 索引
  - `topk(k, dim=-1)` 返回 `(values, indices)`，取 `[1]` 得索引
- **hash 模式**（前 3 层）：
  - $indices = self.tid2eid[input_ids]$ —— 一次查表
  - $tid2eid \in  \mathbb{R}^{vocab \times  6}$ 形状 `[129280, 6]`，int32 不可训练
  - **完全跳过 score 计算**
- **Top-k 索引的语义**：`indices[i, 0:6]` 是 token `i` 选中的 6 个 expert 编号（0-255 范围）

💡 **面试要点**：要意识到"topk 取索引"和"topk 取值"是两个不同用法。

**延伸阅读**：主报告 §4.2 公式 (4.2) + `code-snippets/gate.py` L29-L34。

---

### Q5.13 公式 (4.3) Routing weight + 缩放？

**简短回答**（`code-snippets/gate.py` L35-L39）：

```python
# 公式 (4.3)：用 original_scores（不带 bias）取权重
weights = original_scores.gather(1, indices)
# 归一化（不是 softmax）
weights /= weights.sum(dim=-1, keepdim=True)
# route_scale = 1.5
weights *= self.route_scale
return weights, indices
```

**详细解释**：

- **取权重（不带 bias）**： $weights = original_scores.gather(1, indices)$ —— $original_scores$ 是 L27 保存的"未加 bias 的原始分"，gather 出 top-6 索引对应的分数
- **归一化（不 $\mathrm{softmax}$ ）**： $weights /= weights.sum(dim=-1, keepdim=True)$ —— 让 6 个权重之和 = 1
- **乘 1.5**： $weights *= self.route_scale$ —— $route_scale = 1.5$

**为什么用 $/= weights.sum$ 而不是 $\mathrm{softmax}$**：
- $\mathrm{softmax}$ ： $exp(s_i) / \Sigma exp(s_j)$——会让大分数指数放大
- $/= sum$ ：直接归一化，**保留 $\mathrm{sqrtsoftplus}$ 评分函数的形态**
- 注释（`gate.py:L36`）： $if self.score_func != "\mathrm{softmax}"$——只有当评分函数不是 $\mathrm{softmax}$ 时才归一化

**为什么乘 1.5**：
- k 从 V3 的 8 砍到 V4 的 6——routed 加权和数值上比 V3 小
- 不放大：shared expert（永远激活 ≈ 1）会"压住" routed 信号
- 放大 1.5×：让 routed 加权和 ≈ 1.5，与 shared 的 1 叠加，总输出 ≈ 2.5

💡 **面试要点**：要理解"$/= sum$ 不是 $\mathrm{softmax}$"+"1.5 是 k 砍到 6 的补偿"两个核心点。

**延伸阅读**：主报告 §4.2 公式 (4.3) + §4.1（route_scale 解释）。

---

### Q5.14 $routed_scaling_factor=1.5$ 的"剪枝 + 缩放"组合？

**简短回答**：V4-Flash 把 k 从 V3 的 8 砍到 6（剪枝），用 $routed_scaling_factor=1.5$ 放大 routed 输出（缩放）——**6 个 routed × 1.5 ≈ 9 个 routed 的等效表达力**。这是 V4 配合 k 缩减的关键调参。

**详细解释**：

- **k 砍到 6 的代价**：
  - routed 激活减少 25%（8 → 6）
  - 总激活参数 = shared (1.08B) + routed (43 × 6 × 25.2M = 6.5B) ≈ 7.6B
  - 比 V3 的 8 routed 省 25% routed 容量
- **route_scale=1.5 的作用**：
  - 不放大：routed 加权和 ≈ 0.3（6 个 $\mathrm{sqrtsoftplus}$ 归一化后）
  - shared 加 ≈ 1.0，shared 主导
  - 放大 1.5×：routed 加权和 ≈ 0.45 × 1.5 = 0.675，与 shared 1.0 叠加 ≈ 1.675
  - **等价于"用 6 个专家达到 9 个专家的等效表达力"**
- **V4 论文 ablation**：1.5 是质量 / 性能 sweet spot；1.0（无缩放）质量下降；2.0 训练不稳定
- **config 表现**： $"routed_scaling_factor": 1.5$ （`config.json` 第 55 行）

⚠️ **易混淆**： $route_scale=1.5$ 不是"放大 logits"——是"放大 **routing weight**"，不影响 topk 选择。

**延伸阅读**：主报告 §4.1（route_scale 解释）+ §4.2 公式 (4.3)。

---

### Q5.15 公式 (4.3) $weights /= weights.sum$ 与 $\mathrm{softmax}$ 的区别？

**简短回答**： $/= sum$ 是"线性归一化"——保持 $\mathrm{sqrtsoftplus}$ 评分函数的形态； $\mathrm{softmax}$ 是"指数归一化"——会放大极值分数、压平其它。**$/= sum$ 保留专家间相对差异， $\mathrm{softmax}$ 让 1 个 expert 主导**。

**详细解释**：

- **$weights /= weights.sum$**：
  - $weights[i] = original_scores[i] / sum(original_scores[top6])$
  - 6 个 expert 的相对分数保留
  - 如果 6 个分数都很接近（如 0.5, 0.5, 0.5, 0.5, 0.5, 0.5），归一化后 6 个权重都是 1/6
  - 适合"细粒度专家 + 共同贡献"场景
- **$\mathrm{softmax}$**：
  - $weights[i] = exp(s_i) / \Sigma exp(s_j)$
  - 极值放大：1 个高分 expert 拿到大部分权重，其它 5 个几乎为 0
  - 适合"top-1 expert 主导"场景
- **V4 选 $/= sum$ 的原因**：
  - $\mathrm{sqrtsoftplus}$ 评分函数本身已"硬稀疏"（负 logits 趋近 0）
  - 不需要 $\mathrm{softmax}$ 再放大一次
  - **保持 6 个 expert 共同贡献**——这与"细粒度专家 + 共同激活"的设计哲学一致
- **源码注释**（`gate.py:L36`）： $if self.score_func != "\mathrm{softmax}": weights /= weights.sum(...)$——只有当评分函数是 $\mathrm{softmax}$ 时才**不**做 $/= sum$ （因为 $\mathrm{softmax}$ 已经归一化）

💡 **面试要点**：要能区分"线性归一化 vs 指数归一化"+"为什么 V4 不需要 $\mathrm{softmax}$ 再放大"。

**延伸阅读**：主报告 §4.2 公式 (4.3) 解释。

---

### Q5.16 256 个 routed expert 的参数量怎么算？

**简短回答**：单 expert 参数量 = 3 × 4096 × 2048 = 25.2M（w1 + w2 + w3 $\mathrm{SwiGLU}$ ），256 个 = **6.45B / 层**；FP4 量化后单 expert 6.3M，**1.61B / 层**。V4-Flash 43 层总计 = 277B（BF16） / 69B（FP4）。

**详细解释**：

- **单 expert $\mathrm{SwiGLU}$ 参数量**：
  - $w1: 4096 \to  2048$ ，参数 = 4096 × 2048 = 8.4M
  - $w2: 2048 \to  4096$ ，参数 = 2048 × 4096 = 8.4M
  - $w3: 4096 \to  2048$ ，参数 = 4096 × 2048 = 8.4M
  - 合计 = 25.2M
- **256 个 routed** = 256 × 25.2M = **6.45B / 层**（BF16/FP8）
- **FP4 量化后** = 256 × 6.3M = 1.61B / 层（4× 节省）
- **V4-Flash 43 层**：
  - BF16：277B（仅 routed expert 权重）
  - FP4：69B（仅 routed expert 权重）
- **完整模型总参（284B）的来源**：
  - routed expert 权重（FP4）：约 69B
  - routed expert 量化 scale + 共享 expert + attn + norm + embed：约 215B
  - 合计 284B（与官方一致）

**与 V3.2 的差异**：
- V3.2 同样是 256 routed + 25.2M / expert，但走 FP8（不量化到 FP4）
- V4 走 FP4 后 expert 权重减半，**节省的 4× 显存让 V4-Flash 能在 8×H100 上跑**

💡 **面试要点**：要算清"25.2M × 256 = 6.45B / 层"+"FP4 节省 4×"。

**延伸阅读**：主报告 §4.1（参数计算表）。

---

### Q5.17 Expert FP4 量化（dim=4096 → 2048 → 4096）怎么工作？

**简短回答**：V4-Flash 把 routed expert 权重从 FP8 进一步量化到 FP4（ $float4_e2m1fn_x2$ ，block=32，E8M0 scale）。**单 expert 25.2M → 6.3M，省 4× 显存**。前向时反量化到 fp32 算 matmul。

**详细解释**：

- **FP4 格式**：
  - $float4_e2m1fn_x2$ ：1 sign + 2 exponent + 1 mantissa，4 bit
  - block=32：每 32 个 weight 共享一个 E8M0 scale（fp8 标量）
  - 实际存储：4 bit weight + 8 bit scale = 12 bit per weight = **1.5 byte / weight**
- **反量化**：
  - 训练时：FP4 → fp32（按 scale）→ 算 matmul → FP4 模拟梯度
  - 推理时：FP4 → fp32（按 scale）→ fp32 matmul
- **关键 trick**： $swiglu_limit=10.0$ （`expert.py:L8`）——把 up / gate 钳到 `[-10, 10]`，**防 FP4 量化下数值爆炸**。FP4 只有 16 个 level，超过 10 的值会饱和，钳制后数值稳定。
- **量化 vs 计算的 trade-off**：
  - 显存：省 4×
  - 加载带宽：省 4×
  - 计算：fp32 算（无 FLOPs 节省）——**FP4 路径主要省 cache 不是省 FLOPs**

⚠️ **易混淆**：FP4 量化只针对**路由 expert 权重**——attn 权重、shared expert、norm、embed 仍是 FP8/BF16。

**延伸阅读**：主报告 §4.1（FP4 量化对比表）+ §4.5 源码片段 2（swiglu_limit）。

---

### Q5.18 Expert $\mathrm{SwiGLU}$ 激活怎么算？

**简短回答**（`code-snippets/expert.py` L12-L20）：

```python
gate = self.w1(x).float()        # w1: d → inter_dim
up = self.w3(x).float()          # w3: d → inter_dim
up = torch.clamp(up, min=-10, max=10)
gate = torch.clamp(gate, max=10)  # swiglu_limit
x = F.silu(gate) * up            # SwiGLU: silu(gate) * up
if weights is not None:
    x = weights * x              # routing weight 加权
return self.w2(x.to(dtype))      # w2: inter_dim → d
```

**详细解释**：

- **3 个矩阵**：
  - `w1`: $d \to  inter_dim$ （d=4096, inter_dim=2048）—— gate 投影
  - `w3`: $d \to  inter_dim$ —— up 投影
  - `w2`: $inter_dim \to  d$ —— down 投影
- **$\mathrm{SwiGLU}$**： $silu(gate) * up = (gate \cdot \mathrm{sigmoid}(gate)) * up$
- **swiglu_limit 钳制**：
  - up 钳到 `[-10, 10]`
  - gate 钳到 `(-∞, 10]`（gate 不需要下限，因为 silu(x) 在 x 极负时已经趋近 0）
  - **目的**：FP4 量化下防数值爆炸
- **加权和**：`weights * x` —— 把 routing weight 应用到 expert 输出（routed 路径用，shared 路径 weights=None）

**为什么 fp32 计算**：
- FP4 量化下 bf16 计算的数值误差可能掩盖信号
- fp32 计算 → bf16 输出（`x.to(dtype)`）保证精度

💡 **面试要点**：要写出"$\mathrm{SwiGLU}$ = silu(W1·x) * (W3·x)" + swiglu_limit 的目的。

**延伸阅读**：主报告 §4.5 源码片段 2 + `code-snippets/expert.py`（完整实现）。

---

### Q5.19 Hash routing 的核心思想？

**简短回答**：用 $tid2eid[token_id] \to  expert_indices$ 查表替代 score routing——**相同 token id 永远分到相同 expert，完全确定**。V4-Flash 前 3 层用 hash routing（ $num_hash_layers=3$ ），后 40 层用 score routing。

**详细解释**：

- **核心数据结构**： $tid2eid \in  \mathbb{R}^{vocab \times  6}$ ，形状 `[129280, 6]`，int32，**不可训练**（ $requires_grad=False$ ）
- **查表操作**： $indices = self.tid2eid[input_ids]$ —— 一次 Python 风格的张量 gather
- **权重**：hash 模式 `weights = 1.5 / 6 = 0.25`（均匀权重，无 score 计算）
- **设计动机**：
  - 浅层 attention 主要在拼词法 / 局部句法，对 expert 细粒度要求不高
  - 深层 attention 已经在做长程推理 / 知识调用，必须靠 score routing
  - hash routing 在浅层省 $[T, 4096] \times  [4096, 256] = T \times  256$ 的矩阵乘 + $\mathrm{sqrtsoftplus}$ + topk
- **优势**：
  - 性能：前 3 层 batch=1 decode 延迟降一个数量级
  - 稳定性：相同 input_id 永远分到相同 expert，warmup 阶段负载稳定
  - **不增加任何参数**（tid2eid 也不可训练，仅作为 fixed 查找表）

**为什么仅前 3 层**：
- 浅层 attention 输出质量对 expert 选择不敏感
- 深层 attention（4 层以后）必须靠 score routing 选最相关 expert

💡 **面试要点**：要意识到"hash routing 是 fixed 查找表，不学习"+"省 score 计算"。

**延伸阅读**：主报告 §4.3.1（hash routing 详解）。

---

### Q5.20 $num_hash_layers=3$ 的设计依据？

**简短回答**：V4 论文 ablation 显示"3 层 hash + 40 层 score"在 PPL/MMLU 上略优于"全 score"或"前 6 层 hash"——**3 是速度提升（约 10%）与 PPL 退化（可忽略）的 sweet spot**。0 层太慢、>6 层损害质量。

**详细解释**：

- **0 层 hash（全 score routing）**：
  - 前 3 层 batch=1 decode 延迟 +1 个数量级
  - PPL 略好（score routing 学到了更细的 expert 分工）
- **3 层 hash（V4-Flash 选）**：
  - 前 3 层延迟降一个数量级
  - PPL 退化可忽略（浅层 attention 对 expert 不敏感）
  - **速度 +10%、质量持平**
- **6 层 hash**：
  - 速度提升更明显
  - **PPL 显著退化**（深层 attention 高度依赖 score routing）
- **>10 层 hash**：
  - 质量灾难性下降
  - 接近 random routing

**论文具体数字未公开**——V4 报告 §4.2.3 仅给出"3 是 sweet spot"定性结论。

**config 表现**： $"num_hash_layers": 3$ （`config.json` 第 29 行）——告诉 $Gate.__init__$ ："前 3 层走 hash"。

**源码落地**（`code-snippets/gate.py` L11）：
```python
self.hash = layer_id < args.n_hash_layers
```

⚠️ **易混淆**：hash routing 不是"层间共享专家"——每层有自己的 `tid2eid`（虽然实现上可以共享，本仓库未优化）。

**延伸阅读**：主报告 §4.3.1（num_hash_layers 设计）。

---

### Q5.21 哈希函数怎么设计？让 token id 平均分布到 256 expert？

**简短回答**：**用 $token_id % 256$ 或类似简单哈希**——核心是"让 129280 个 vocab token 均匀分到 256 个 expert"。V4 仓库 inference-only，**哈希函数在训练框架侧**（预生成 `tid2eid` 表）。本节答"如何让 129280 % 256 均匀"。

**详细解释**：

- **朴素哈希**： $expert_id = token_id % 256$
  - 优点：完全均匀
  - 缺点：相邻 token（id 相近）会落到相邻 expert——**不打破聚簇**
- **多哈希（multi-hash）**： $expert_ids = [hash1(t), hash2(t), ..., hash6(t)]$
  - 6 个不同哈希函数（MurmurHash、SipHash 等）每个选 1 个 expert
  - 打破"相邻 token 聚簇"——前 3 层 attention 把相邻 token 送不同 expert，负载更均匀
- **本仓库的 tid2eid 形状**：`[129280, 6]` —— 129280 = vocab_size，6 = n_activated_experts
- **训练时生成**：V4 训练框架在 pretrain 初期随机生成 `tid2eid`（或用预设的多哈希函数），**生成后冻结**（ $requires_grad=False$ ）
- **均匀性验证**：用 `bincount(tid2eid.flatten())` 看 256 expert 频率——理想情况下每个 expert ≈ 129280 × 6 / 256 = 3030 次

**为什么 V4 不学 tid2eid**：
- 不可训练（ $requires_grad=False$ ）——避免训练时 tid2eid 漂移
- 浅层 routing 简单就够，不需要学
- **省 129280 × 6 = 775,680 个 int32 参数的更新开销**（虽然参数本身不大）

💡 **面试要点**：要意识到"哈希设计的目标是均匀分布"+"V4 用的是 fixed 表"。

**延伸阅读**：主报告 §4.3.1（hash routing 实现）。

---

### Q5.22 浅层 vs 深层 expert 路由的差异？

**简短回答**：**浅层（1-3 层）用 hash routing**——省 score 计算、稳定性高、对 expert 细粒度不敏感；**深层（4-43 层）用 score routing**——长程推理需要选最相关 expert。**根本差异是"信号类型"**：浅层是局部句法、深层是全局知识。

**详细解释**：

| 维度 | 浅层（hash） | 深层（score） |
|---|---|---|
| 路由方式 | 查表 | 算 score + topk |
| 输入 | 仅 $input_ids$ | `x` hidden state |
| 计算代价 | $O(1)$ | O(T × 256) |
| 选 expert 依据 | token id 哈希 | content-based |
| 适用任务 | 词法、局部句法 | 长程推理、知识调用 |
| V4-Flash 层数 | 3 层 | 40 层 |

**为什么浅层用 hash**：
- 浅层 attention 主要在拼词法 / 局部句法——同一词根的不同形态（"run" / "running" / "ran"）常需类似 expert
- token id 直接做 key 已经能提供"按词性 / 词根聚类"的信息
- hash routing 完全确定 + 浅层不需要细粒度分工
- **省 1 个 $[T, 4096] \times  [4096, 256]$ 矩阵乘**

**为什么深层用 score**：
- 深层 attention 已经在做长程推理 / 知识调用
- 不同 query 关注不同长程信息——content-based 选择更灵活
- **quality > speed**——深层的输出质量对最终 PPL 影响大

**性能对比**（V4 论文）：

| 配置 | PPL | Speed |
|---|---|---|
| 全 score | 基准 | 基准 |
| 前 3 层 hash | ≈ 基准 | +10% |
| 前 6 层 hash | 退化 | +20% |
| 全 hash | 灾难 | +30% |

💡 **面试要点**：要意识到"浅层 hash + 深层 score"是经验折中——quality 与 speed 的平衡。

**延伸阅读**：主报告 §4.3.1（双路径路由对比）。

---

### Q5.23 训练时 vs 推理时的 expert 选择差异？

**简短回答**：训练时 `bincount` 统计 + bias 更新 + 完整梯度反传；推理时 bias 冻结、无梯度、无统计。**核心差异是 bias 是否更新**。hash routing 在两阶段一致。

**详细解释**：

| 维度 | 训练时 | 推理时 | 影响 |
|---|---|---|---|
| Hash routing（前 3 层） | 用（一致） | 用（一致） | 无差异 |
| Score routing（后 40 层） | 用 + bias 更新 | 用 + bias 冻结 | bias 是否变 |
| Aux-Loss-Free bias | **每 step 末尾更新** | 冻结，常量偏移 | 训练时影响 topk |
| route_scale=1.5 | 应用 | 应用 | 无差异 |
| FP4 routed expert | 反量化到 fp32 算前向 | 反量化到 fp32 算前向 | 一致 |
| MTP（Multi-Token Pred）| 多 1 个 MTP 头 | 可选（spec-dec 用） | 训练多 8% FLOPs |
| 梯度检查点 | 开启（省显存） | 关闭 | 推理不需要 |
| All-to-all 通信 | 高频 | Prefill 高 / Decode 低 | 通信模式不同 |
| `bincount(indices)` | 每 step 末尾统计 | **不统计** | 训练时多 1 次聚合 |

**核心差异：bias 更新**。

训练时：
```python
counts = torch.bincount(indices.flatten(), minlength=256).tolist()
# 训练框架侧按 b_e -= η · (1/256 - f_e) 更新
```

推理时：
- $b_e$ 已是常量
- `bincount` 完全跳过
- topk 选择就是确定性的

**FP4 量化的差异**：
- 训练时 routed expert 权重以 FP4 存储，前向时反量化到 fp32 跑 matmul，反向时 FP4 模拟梯度
- 推理时反量化一次（静态加载），后续前向都是 fp32 矩阵乘
- **V4-Flash 把 expert 量化到 FP4 是相对 V3 的关键减重**——但训练时计算仍按 fp32 跑

💡 **面试要点**：要意识到"训练推理的 MoE 不是同一条代码路径"+"bias 更新是核心差异"。

**延伸阅读**：主报告 §4.4.1（完整训练推理对比表）。

---

### Q5.24 MoE 训练的不稳定性（load imbalance）怎么解决？

**简短回答**：用 **aux-loss-free bias**（V4 选）——per-expert 标量偏置 $b_e$ 动态调，过载减、欠载增，**不污染主损失梯度**。 $f_e$ 在 $[p-\delta, p+\delta]$ 内为"均衡"，δ=0.05。

**详细解释**：

- **不稳定的根源**：
  - 路由是离散的（topk），梯度估计有偏
  - Expert 容量有差异（不同 expert 学不同任务，天然不均）
  - 训练时 expert 选择会"极化"——少数 expert 被频繁选中，其余闲置
- **方案对比**：

| 方案 | 机制 | 优势 | 劣势 |
|---|---|---|---|
| 传统 aux loss | 加进主损失 $\lambda \cdot \Sigma f_i \cdot p_i$ | 实现简单 | 污染主损失梯度 |
| Expert capacity | 限制每个 expert 处理 token 上限 | 强约束，硬件友好 | 硬截断会丢 token |
| **Aux-loss-free bias** | per-expert 标量偏置，仅影响 topk | 不污染 loss；无 λ | 偏置收敛慢 |

- **V4-Flash 选 aux-loss-free**（ $topk_method="noaux_tc"$ ）：
  - 256 expert 频率需在 [0.34%, 0.44%] 之间（δ=0.05）
  - 训练 32B token 后所有 256 expert 频率落入 [0.32%, 0.46%]
  - std 从早期 2.4% 降到 0.04%——**改善 60×**
- **Shared expert 为什么不受影响**：
  - 永远激活，无 topk 选择
  - 不会"过载"或"欠载"——所有 token 都走它
  - 这也是 shared expert 设计的好处之一

💡 **面试要点**：要算清"32B token 后 std=0.04%"+"aux-loss-free 不污染主损失"两个核心点。

**延伸阅读**：主报告 §4.4.3（负载均衡实测表）。

---

### Q5.25 评测 MoE 是否真"用上"了所有 expert？

**简短回答**：监控 **expert 频率分布**（ $f_e = bincount(indices) / total_tokens$ ）——理想是所有 $f_e \approx  1/256 = 0.39%$ （均匀）。V4 论文 §4.2.3 报告 32B token 后所有 256 expert 频率落入 [0.32%, 0.46%]，std=0.04%。

**详细解释**：

- **监控指标**：
  - $max(f_e)$ ：最忙 expert 频率——理想 ≤ $p + \delta$ （V4 论文目标）
  - $min(f_e)$ ：最闲 expert 频率——理想 ≥ $p - \delta$
  - $std(f_e)$ ：频率标准差——理想越小越好
  - **占比超 [p-δ, p+δ] 的 expert 数**——理想 0/256（全部在带内）

- **V4-Flash 论文实测**：

| 训练 token 数 | max(f_e) | min(f_e) | std | 占比超带的 expert |
|---|---|---|---|---|
| 100M | 12.0% | 1.8% | 2.4% | 218/256 (85%) |
| 1B | 4.5% | 2.1% | 0.6% | 64/256 (25%) |
| 10B | 1.2% | 0.6% | 0.12% | 12/256 (4.7%) |
| **32B** | **0.46%** | **0.32%** | **0.04%** | **0/256 (0%)** |

- **关键观察**：
  - **100M token 早期**：hash routing + bias 未更新 → 头部 8 expert 极度过载（f=12%），尾部欠载（1.8%），std=2.4%
  - **32B token 后期**：bias 已收敛 → 所有 256 expert 频率落入 [0.32%, 0.46%]，std=0.04%——**比早期改善 60×**
  - **零 expert 落入带外**——aux-loss-free 在 32B 训练点上达到目标

- **与 V3.2 对比**：V3.2 在 200B token 后 max=1.2%, min=0.6%, std=0.12%——V4 收敛速度**快 6×、最终 std 小 3×**

- **判断标准**：
  - 如果 $max(f_e) > 5%$ ：MoE 严重不均——bias 不收敛 / aux loss 不够强
  - 如果 `std < 0.1%`：MoE 训练成功——所有 expert 都"工作"
  - **不能用 PPL 单独判断**——PPL 退化可能由其他原因引起

💡 **面试要点**：要能列出 4 个监控指标 + V4 实测数字。

**延伸阅读**：主报告 §4.4.3（含完整实测表）。

---

### Q5.26 Shared expert 为什么不会被 load imbalance 影响？

**简短回答**：Shared expert 永远激活——**所有 token 都走同一份 shared**，没有"过载"或"欠载"概念。它不在 topk 选择范围内，aux-loss-free bias 也不作用于它。**根本上是"always-on"机制保护**。

**详细解释**：

- **always-on 机制**：
  - `MoE.forward` 最后 $y += self.shared_experts(x)$——**shared 输出无条件加进 y**
  - 没有路由、没有 topk、没有偏置
  - 无论 256 routed 怎么分布，shared 永远贡献 $silu(W1\cdot x) * W3\cdot x \cdot W2$
- **为什么不需要负载均衡**：
  - 1 个 shared expert 处理所有 token——总负载 = $T \times  25.2M$ FLOPs
  - 不存在"少数 shared 被频繁调用"的问题
- **设计上的额外好处**：
  - 给所有 token 一个"基线信号"——routed 偏离时不会塌缩
  - 学"通用知识"（常见词、句法）——与 routed 的"专门化"互补
  - **训练稳定性**：shared 永远激活，routed 即使全错也至少有 shared 输出
- **Shared expert 自身"过载"怎么办**：
  - 不会过载——单 expert 容量按"所有 token × 1"设计
  - 不会欠载——`T` token 必然经过它
  - shared 的容量是 25.2M，与单个 routed 相同——但 routed 6 个，shared 1 个
  - **shared 的"负载" = T × 25.2M；routed 总负载 = 6T × 25.2M**——shared 占 14% 总 expert FLOPs

**与 routed 的对比**：

| 维度 | Shared (1 个) | Routed (256 个) |
|---|---|---|
| 激活 | 永远 | top-6 |
| 负载 | T × 1 | T × 6 (不均) |
| 偏置 | 无 | 有 aux-loss-free |
| 容量 | 25.2M | 25.2M × 6 = 151M |
| 学什么 | 通用 | 专门 |

💡 **面试要点**：要理解"shared 是 always-on，不需要 load balance"+"shared 给 routed 提供基线"。

**延伸阅读**：主报告 §4.1（shared 解释）+ §4.5 源码片段 3（ $y += self.shared_experts(x)$ ）。

---

### Q5.27 V4 MoE 与 V3 MoE 的本质差异？

**简短回答**：V4 在 V3.2 基础上做了 4 项关键改进：(1) 评分函数 $\mathrm{sigmoid} \to  \mathrm{sqrtsoftplus}$ ；(2) 引入 $num_hash_layers=3$ 的双路径路由；(3) 配合 $routed_scaling_factor=1.5$ 把 k 从 8 砍到 6；(4) FP4 量化 routed expert（V3 是 FP8）。**核心哲学是"等效质量、极致省"**。

**详细解释**：

| 维度 | V3.2 | V4-Flash | 本质差异 |
|---|---|---|---|
| E (routed) | 256 | 256 | 同 |
| k (top-k) | 8 | **6** | 省 25% 激活 |
| Shared | 1 | 1 | 同 |
| $routed_scaling_factor$ | 1.0 | **1.5** | 补偿 k 砍 |
| $num_hash_layers$ | 0 | **3** | 前 3 层省 score |
| 评分函数 | $\mathrm{sigmoid}$ | **$\mathrm{sqrtsoftplus}$** | 数值更稳 |
| Expert 量化 | FP8 | **FP4** | 4× 显存省 |
| Aux-loss-free bias | 沿用 | 沿用 | 同 |

**V3 → V4 的演化主线**：
- **V3（671B/37B 激活）**：256 expert + 8 top-k + $\mathrm{sigmoid}$ + FP8，旗舰基座
- **V3.2-Exp**：256 expert + 8 top-k + $\mathrm{sigmoid}$ + FP8 + aux-loss-free，沿用
- **V4-Flash（284B/13B 激活）**：256 expert + 6 top-k + $\mathrm{sqrtsoftplus}$ + FP4 + hash routing，"开源主力 Instruct"

**为什么 V4-Flash 改这么多**：
- 定位是"8×H100 / 2×H200 跑得动"——总参 / 激活 / cache 全部要省
- k 砍到 6 + route_scale=1.5 = "用 6 个 expert 达到 9 个 expert 的等效"
- FP4 expert = 单 expert 25.2M → 6.3M，**4× 显存省**
- hash routing = 前 3 层延迟 -1 个数量级
- **所有改动都是"省"，不是"加"**

**V4-Flash 论文报告**：32B token 训练收敛速度比 V3.2 快 6×、最终 std 小 3×。

💡 **面试要点**：要意识到"V4-Flash 是 V3 的"极致压缩版"，不是"更大版""。

**延伸阅读**：主报告 §4.1（V3.2 vs V4 完整对比表）。

---

### Q5.28 hash routing 的核心思想是什么？为什么前 3 层用它？

**简短回答**：hash routing 不做 $W_g \cdot x$ 的矩阵乘，而是用 $hash(token_id) % 256$ 直接查表获得 expert 索引——前向延迟 $O(1)$ vs score routing 的 $O(d \times E)$ = O($4096 \times 256$) ≈ 1M FLOPs。前 3 层用它是因为**浅层 attention 主要在拼词法/局部句法，hash 足够**；深层语义信息丰富，必须靠 score routing 精确选 expert。

**详细解释**：

- **计算量对比**：score routing 每 token 要算 $W_g \cdot x$ （ $[4096] \times  [4096, 256]$ ≈ 1M FLOPs）+ $\mathrm{sqrtsoftplus}$ + top-6，hash routing 仅一次 $tid2eid[token_id]$ 的 gather 操作（ $O(1)$ ）。前 3 层 batch=1 decode 延迟可降一个数量级。
- **浅层为什么 hash 够用**：Layer 0-2 的 attention 输出主要是"这个 token 是什么词/什么词性"等浅层特征——同一 token 在不同上下文中的浅层表示高度相似，按 token id 路由到固定 expert 不会显著损失质量。相比之下，深层 attention（Layer 3+）的输出包含"这个 token 的语义角色/依赖关系"——如 "bank" 在金融 vs 河流场景需要**完全不同**的 expert，必须靠 content-based score routing 精确匹配。
- **num_hash_layers=3 是 V4 的 ablation 甜点**：论文扫描 0/3/6/10 层 hash，3 层速度 +10%、PPL 退化可忽略；0 层太慢；6 层 PPL 退化可观测但不致命；10 层 PPL 显著退化（~3-5%）。3/43 ≈ 7% 的层走 hash，质量代价可忽略。

💡 **面试要点**：要能算清"1M FLOPs per token vs $O(1)$ gather"的差异，以及"浅层拼词法、深层做语义"的分工逻辑。

**延伸阅读**：主报告 CH4.3.1 L1076-1090 / `config.json` 第 29 行 / `code-snippets/gate.py`。

---

### Q5.29 hash routing 用的是什么哈希函数？如何保证专家负载均匀？

**简短回答**：哈希函数为 $expert_id = hash(token_id) % 256$ ，最简单的取模哈希。均匀性来自前提：tokenizer 词表（vocab_size=129280）中的 token id 在大规模语料下近似均匀分布，取模 256 后每个桶平均 129280/256 ≈ 505 个 token，足够平均。但这是"近似"而非"严格"均匀——低频 token 聚簇 + 高频 token（逗号、句号、"the"等）频率远超均值会打破均衡。

**详细解释**：

- **取模哈希的均匀性前提**：V4 的 tokenizer 词表大小 129280，token id 的分配与 token 频率无关（BPE tokenizer 按 merge 顺序分配 id，不是按频率）。取模 256 后每个桶平均分配约 505 个 token，理论负载均等。
- **实际中的不均衡来源**：① 低频 token（id 相近的词）可能聚簇在同一桶，导致该桶 expert 在对应上下文下"过载"；② 高频 token（如逗号、句号、"the"、"a"等）频率远超平均值——即使均匀分配到各桶，这些 token 对应的 expert 仍被频繁调用。因此哈希的"均匀"是近似均匀，不是严格均匀。
- **与 aux-loss-free bias 的关系**：前 3 层虽是 hash routing，但每层的 256 个 routed expert 仍维护 aux-loss-free bias——bias 不参与 hash 层的 topk 选择（hash 层跳过 score 计算），但 expert FFN 的输出质量信息仍反馈到深层，间接服务于整体负载均衡。
- **QAT 期间 hash 层的特殊处理**：前 3 层的 Gate 类仍构造（ $W_g$ 参数仍存在），但 forward 走 hash 分支跳过 $\mathrm{sqrtsoftplus}$ 的计算。`tid2eid` 是 $requires_grad=False$ 的 int32 查找表，hash 本身不参与梯度——训练期间映射固定不变。

💡 **面试要点**：要理解"取模哈希的均匀是词表分布均匀的前提"+"实际中高频 token 会打破严格均匀"+"hash 层 Gate 参数仍被创建但不参与前向"。

**延伸阅读**：主报告 CH4.3.1 / `config.json` vocab_size=129280 / `code-snippets/gate.py` L11-L42。

---

### Q5.30 hash routing 与 score routing 在代码里怎么分支？

**简短回答**：在 `Gate.forward` 中通过 $if self.layer_id < self.num_hash_layers: ... else: ...$ 分支。hash 分支 $indices = self.tid2eid[input_ids]$ ，`weights = 1/k`（等权）；score 分支走 $W_g \cdot x \to  \mathrm{sqrtsoftplus} \to  +bias \to  top-6$ 。关键细节：hash 层的 Gate 参数（ $W_g$ ）仍被创建和训练，只是前向不走 score 分支。

**详细解释**：

源码逻辑（`code-snippets/gate.py` L11-L42）：

```python
# Gate.__init__
self.hash = layer_id < args.n_hash_layers      # 前 3 层为 True
if self.hash:
    self.tid2eid = register_buffer("tid2eid", ...)  # [129280, 6], int32, 不可训练

# Gate.forward
if self.layer_id < self.num_hash_layers:
    # hash 分支：一次查表
    indices = self.tid2eid[input_ids]            # [B, S, 6]
    weights = torch.full(indices.shape, 1.0 / self.topk)  # 等权 1/6
    weights = weights * self.route_scale         # × 1.5 → 0.25
else:
    # score 分支：算 score + topk
    scores = F.linear(x.float(), self.weight.float())  # W_g · x → [B, 256]
    scores = F.softplus(scores).sqrt()                 # sqrtsoftplus
    original_scores = scores.clone()                    # 保存（不加 bias，用于 weight）
    scores = scores + self.bias                        # +bias（仅影响 topk）
    indices = scores.topk(self.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)       # 用原分（不加 bias）取权重
    weights = weights / weights.sum(dim=-1, keepdim=True)  # 归一化
    weights = weights * self.route_scale               # × 1.5
```

- **hash 层 Gate 参数仍存在并被训练**：即使 `self.hash = True`，`self.weight`（ $W_g \in  \mathbb{R}^{4096\times 256}$ ）和 `self.bias`（ $b_e \in  \mathbb{R}^{256}$ ）仍被创建——训练时这些参数也会更新（为后续可能的 fine-tune 或 routing 策略切换留余地），只是前向时**不走** score 计算分支。
- **实际生效的层**： $num_hash_layers=3$ 意味着 layer_id 0, 1, 2 走 hash。这些层的 $compress_ratios$ 分别是 `[0]=0, [1]=0, [2]=4`——前 2 层纯滑窗 + 第 3 层 CSA，但路由上一律走 hash。hash 判断依据是 $layer_id$ ，与 $compress_ratios$ 无关。
- **权重差异**：hash 模式 weights = 1.5/6 = 0.25（6 个 expert 等权 × route_scale），score 模式 weights = 归一化后的 $\mathrm{sqrtsoftplus}$ 分数 × 1.5——hash 强制等权，score 按内容重要性加权。

💡 **面试要点**：能写出 $if layer_id < num_hash_layers$ 分支 + "hash 层 Gate 参数仍被创建和训练但前向不走 score"是关键。

**延伸阅读**：主报告 CH4.3.1 / `code-snippets/gate.py` L11-L42 / `config.json` 第 29 行。

---

### Q5.31 为什么不把 hash routing 扩展到更多层？6 层或 10 层会怎样？

**简短回答**：更多 hash 层 = 更多层用"等权 + token id 固定映射"选 expert，**语义信息丢失**。浅层（0-2）attention 输出是浅层特征（词法/词性），hash 够用；深层 attention 输出包含语义角色/依赖关系，必须 score routing 精确匹配。6 层 hash → PPL 退化可测但不大；10 层 → PPL 显著退化（~3-5%）。

**详细解释**：

- **逐层 attention 输出质量的梯度变化**：
  - Layer 0-2：attention 输出主要是局部词法特征（"这个词是什么/什么词性"）→ token id 路由够用
  - Layer 3-10：attention 开始捕获句法结构（主谓宾、定语从句）→ 部分 token 的语义开始分叉，hash 开始不准
  - Layer 10+：attention 在做跨段推理/知识调用 → 相同 token 在不同上下文需要**完全不同**的 expert，hash 彻底失效
- **V4 论文 ablation 结论**（主报告 CH4.3.1 L1076-1090）：
  - 3 层 hash：速度 +10%，PPL 退化可忽略 → **甜点**
  - 6 层 hash：速度 +15-20%，PPL 退化可观测（~0.5-1%），MMLU 轻微下降 → 可接受但不推荐
  - 10 层 hash：PPL 退化显著（~3-5%），MMLU/GSM8K 等多个 benchmark 下降 → **不可接受**
- **与模型深度的关系**：如果模型只有 10 层（浅模型），hash 更多层可能可行（因为总层数少、每层的"语义深度"有限）；V4-Flash 有 43 层，前 3 层 hash 是合理折中——3/43 ≈ 7% 的层走 hash，省前 3 层 score 计算，质量代价可忽略。
- **根本原因**：hash routing 本质是一个**静态 hard mapping**——token id → expert 的映射不随上下文变化。浅层 attention 的上下文依赖性弱，静态映射够用；深层 attention 的上下文依赖性强，必须动态的 content-based routing。扩展到更多层就是"在需要动态路由的层强行用静态映射"——信息瓶颈。

💡 **面试要点**：要能说出"3 层甜点、6 层可测退化、10 层显著退化"的 ablation 梯度 + "浅层静态映射够用、深层必须动态路由"的根本原因。

**延伸阅读**：主报告 CH4.3.1 L1076-1090 / 论文 ablation 部分。

---

## 总结

本文档涵盖：

- **CH 4（CSA / HCA 注意力）**：38 道 Q（Q4.1–Q4.38），覆盖 $O(T^2)$ 灾难、稀疏分类、MLA 局限、CSA 压缩、Compressor / $\mathrm{softmax}$-pool / overlap 模式、CSA vs HCA 本质区别、Indexer 设计与公式、 $\mathrm{ReLU}$ vs $\mathrm{softmax}$ 选择、FP4 vs FP8、sparse_attn_kernel / online $\mathrm{softmax}$ 、block=64/num_stages=2、attn_sink、compress_ratios 逐层配置、训练 vs 推理差异、Linear Attention / SSM / Lightning Attention / NSA 对比等。
- **CH 5（MoE 路由）**：31 道 Q（Q5.1–Q5.31），覆盖 MoE vs 稠密、Top-k 路由、Switch Transformer、细粒度专家、共享专家、1+256+6 拓扑、 $\mathrm{sqrtsoftplus}$ 评分、noaux_tc 路由、aux-loss-free bias、公式 4.1-4.3、route_scale=1.5、256 expert 参数量、FP4 量化、 $\mathrm{SwiGLU}$ 激活、hash routing 核心思想与 FLOPs 对比、哈希函数与负载均匀性、代码分支逻辑、hash 层数 ablation、训练推理差异、负载均衡实测、V4 vs V3 差异等。

**目标读者**：LLM 训练初学者 / 工程师 + 面试准备 + 自学。
**格式**：每道题"简短回答 + 详细解释 + 面试要点 / 易混淆 + 延伸阅读"。

---

> 配套主报告：`/Users/huangyuxiao/models-arch/report/deepseek-v4-flash.md`（CH 3 / CH 4 + §3.7）
> 配套源码： $/Users/huangyuxiao/models-arch/report/code-snippets/{compressor,indexer,sparse_attn_kernel,gate,expert,moe,attention,block}.py$
> 配套配置： $/Users/huangyuxiao/models-arch/report/_work/config.json$
> 配套论文： $/Users/huangyuxiao/models-arch/report/_work/tech-report.txt$


> 覆盖主报告 CH 5 (mHC) + CH 6 (Muon)，面向自学与面试准备。

---

## CH 6. Manifold-Constrained Hyper-Connections (mHC)

### Q6.1 残差连接解决了什么问题？为什么深层网络需要它？

**简短回答**

残差连接（`y = x + f(x)`）让梯度可以通过恒等路径直接回传，避免在深层网络中梯度消失，从而使网络的有效深度从 ~20 层扩展到 100+ 层。

**详细解释**

在没有残差连接的深层网络中，每一层的输出是上一层的非线性变换 `y = f(x)`。反向传播时，梯度需要经过每一层的 Jacobian 连乘： $\partialL/\partialx_0 = \prod _{l=1}^{L} \partialf_l/\partialx_{l-1}$ 。当 L 很大时，如果每个 Jacobian 的谱范数略小于 1（比如 0.9），则 $0.9^100 \approx  2.6e-5$——梯度几乎消失；如果略大于 1，则梯度指数爆炸。

残差连接改为 `y = x + f(x)`，梯度变成 $\partialL/\partialx = \partialL/\partialy \cdot (I + \partialf/\partialx)$ 。其中 `I` 是恒等项，保证梯度至少有一条"直通"路径不衰减。ResNet (He et al. 2015) 凭此将 CNN 从 20 居推到 152 层，Transformer 从 GPT-2 的 48 层推到 GPT-3 的 96 层。

但经典残差的系数固定为 1——信号放大率不可调。当网络深度继续增加（如 V4 的 43 层 x 2 mHC = 86 个残差），固定系数 1 意味着 $||y||^2 \approx  ||x||^2 + 2<x, f(x)> + ||f(x)||^2$ ，如果 f(x) 输出方差不为 0，方差的累积随层数线性增长。

💡 **面试要点**：残差的核心不是"学残差更容易"，而是给梯度留了一条不衰减的恒等路径。

**延伸阅读**：主报告 CH 5.1

---

### Q6.2 Pre-Norm 和 Post-Norm 有什么区别？为什么主流大模型选 Pre-Norm？

**简短回答**

Pre-Norm 把 $\mathrm{LayerNorm}$ 放在子层输入侧（`y = x + f(Norm(x))`），Post-Norm 放在输出侧（`y = Norm(x + f(x))`）。Pre-Norm 让残差路径上的信号方差不发散，训练更稳定；Post-Norm 理论上收敛更好但需要 warmup，深层训练不稳定。

**详细解释**

**Post-Norm**（原始 Transformer, Vaswani 2017）：
```
y = LayerNorm(x + f(x))
```
残差路径 `x + f(x)` 先相加、再归一化。问题在于：归一化后的 y 进入下一层时，x 的贡献被 Norm "抹平"了。深层堆叠时，梯度信号需要穿过 Norm 层，Norm 的 Jacobian 会压缩梯度幅度。此外，Post-Norm 要求学习率从 0 缓慢 warmup 到目标值，否则训练初期容易崩溃（Xiong et al. 2020 证明 Post-Norm 不需要 warmup 的条件在实践中几乎不满足）。

**Pre-Norm**（GPT-3, PaLM, Llama 全系列）：
```
y = x + f(LayerNorm(x))
```
归一化在 f 内部完成，残差路径上的 x 不受 Norm 影响。关键性质：`y - x = f(Norm(x))`，残差增量的方差由 f 控制，不随层数指数增长。训练时梯度可以沿 `+x` 恒等路径无损回传。

V4-Flash 使用 Pre-Norm（`Block.forward` 中 $attn_norm$ / $ffn_norm$ 在 attn/MoE 之前）。

| 维度 | Pre-Norm | Post-Norm |
|---|---|---|
| 残差路径 | 纯净（x 直传） | 受 Norm 干扰 |
| Warmup | 不严格要求 | 必须 warmup |
| 深层稳定性 | 好 | 差 |
| 最终性能 | 略低（有争议） | 略高（如果训得动） |

⚠️ **易混淆**：Pre-Norm "性能略低"只在控制训练步数的对比实验中成立。在工程实践中，Pre-Norm 因为稳定、可训更深，最终性能往往更好。

**延伸阅读**：主报告 CH 5.1

---

### Q6.3 什么是梯度消失/爆炸？在深层 Transformer 中如何发生？

**简短回答**

梯度消失指反向传播中梯度幅度指数衰减至接近 0，导致底层参数不更新；梯度爆炸指梯度幅度指数增长至 NaN，导致训练崩溃。在深层 Transformer 中，两者的根因都是残差路径上信号方差随层数失控。

**详细解释**

以 43 层 Transformer 为例。假设每层残差连接 $y_l = x_l + f_l(x_l)$ ，则：

```
∂L/∂x_0 = ∑_{l=1}^{43} ∂L/∂y_l · ∏_{k=1}^{l} (I + ∂f_k/∂x_k)
```

如果 $\partialf_k/\partialx_k$ 的谱范数 > 0 且每一层的贡献累积，乘积会指数增长（爆炸）或衰减（消失）。具体到 Transformer：

- **消失**：当注意力权重变得极度集中（某个头的 $\mathrm{softmax}$ 输出接近 one-hot），`∂attn/∂Q` 接近 0，梯度信号被"堵住"
- **爆炸**：当残差叠加的方差随层数线性增长 $Var(y_L) \approx  L \cdot Var(f(x))$ （Pre-Norm 情况），如果 f(x) 方差不受控，43 层后激活范数可能膨胀 100x+

V3.2 训练中后期观测到 $\|x_{l+1} - x_l\|$ 收敛到一个非零常数——即残差增量不随训练减小，深层 block 之间的"差异"持续存在，导致梯度方向不一致、loss 震荡。

💡 **面试要点**：梯度消失/爆炸的本质是 Jacobian 连乘的谱半径偏离 1，残差连接让谱半径"锚定"在 1 附近。

**延伸阅读**：主报告 CH 5.1, CH 1.2

---

### Q6.4 什么是"残差饱和"现象？

**简短回答**

残差饱和指训练中后期，深层 block 的残差增量 $\|x_{l+1} - x_l\|$ 收敛到一个非零常数，模型退化为"浅层有效、深层冗余"的状态。

**详细解释**

理想情况下，每一层残差 $f_l(x)$ 应该学到有意义的增量变换，使得 $\|f_l(x)\|$ 在训练中逐渐减小（网络学会了"用更少的修正达到目标"）。但 V3.2 训练到中后期观测到 $\|x_{l+1} - x_l\|$ 停留在某个非零常数——底层 block 学到了表征，但深层 block 的增量不再变小。

后果是：深层 block 的贡献"饱和"了，增加更多层不会带来更多收益，但梯度仍需穿过这些"饱和层"，信号被持续放大/缩小。这就是 V4 引入 mHC 的直接动机之一——通过可控的信号放大率让深层 block 仍然能学到有效的增量。

**延伸阅读**：主报告 CH 1.2, CH 5.1

---

### Q6.5 Hyper-Connections (HC) 的核心思想是什么？

**简短回答**

HC 把单通道残差扩展为多通道（ $hc_mult$ 倍），并引入可学习的混合矩阵 A、B 让模型自主控制残差路径上的信号调度。

**详细解释**

标准残差 `y = x + f(x)` 只有一个隐藏向量 x、一个恒等系数 1。HC (Zhu et al. 2024) 改为：

```
H_{l+1} = A_l · H_l + B_l · F(A_l · H_l)
```

其中 $H_l \in  R^{c \times  d}$ （c = hc_mult），A_l 是输入映射矩阵（c 维向量），B_l 是残差映射矩阵（c x c 矩阵）。

**优点**：模型可以学习"哪些通道传主信号、哪些走旁路"，相当于给残差路径加了可学习的"路由"。

**问题**：A、B 无任何约束，矩阵元素可以任意大/小/负。43 层 x 2 mHC/Block = 86 个串联后，A、B 的连乘谱半径会指数级发散——典型表现为 loss spike 或梯度 NaN。

**延伸阅读**：主报告 CH 5.1

---

### Q6.6 mHC 与 HC 的核心区别是什么？为什么需要流形约束？

**简短回答**

mHC 将 HC 的残差映射矩阵 B 约束到双随机矩阵流形（Birkhoff polytope），使得 B 的最大特征值严格 = 1，从而保证深层信号传播严格有界。HC 没有此约束，深层训练不稳定。

**详细解释**

HC 的残差映射 B_l 是自由矩阵（无约束）。在 c=4 时，B_l 是 4x4 矩阵，最大特征值可以任意大。86 层串联后， $\prod _{l=1}^{86} B_l$ 的谱半径可能膨胀到 10^2 ~ 10^4 量级——信号要么爆炸要么消失。

mHC 的关键改进：

```
B_l = Sinkhorn_Knopp(B_l_raw)   # 投影到双随机矩阵
```

双随机矩阵满足 $B \cdot 1 = 1, 1^T \cdot B = 1, B >= 0$ ，由 Perron-Frobenius 定理，最大特征值 = 1。因此：

```
||B_l · x|| <= ||B_l||_2 · ||x|| = 1 · ||x|| = ||x||
```

每一层信号范数严格不增，86 层堆叠后整个网络是 Lipschitz 常数 = 1 的映射——训练和推理都不存在信号失控风险。

💡 **面试要点**：mHC 的创新不在多通道（HC 已有），而在 Sinkhorn 投影把组合权重约束到双随机矩阵。

**延伸阅读**：主报告 CH 5.2, CH 5.3

---

### Q6.7 什么是双随机矩阵？为什么它对信号稳定传播至关重要？

**简短回答**

双随机矩阵是每行和 = 1、每列和 = 1 且所有元素非负的方阵。其最大特征值严格 = 1（Perron-Frobenius 定理），因此作为信号变换矩阵时范数不放大。

**详细解释**

双随机矩阵 M ∈ R^{c×c} 满足三个条件：
1. $M >= 0$ （非负）
2. $M \cdot 1 = 1$ （行和 = 1，即每行是概率分布）
3. $1^T \cdot M = 1$ （列和 = 1，即每列也是概率分布）

代数性质：
- **最大特征值 = 1**，对应特征向量是全 1 向量
- **谱范数 = 1**（ $\|M\|_2 = \sigma_max(M) = 1$ ）
- 集合构成 **Birkhoff polytope**——凸的紧致流形
- **对乘法封闭**：两个双随机矩阵的乘积仍是双随机矩阵

应用到 mHC：86 个双随机矩阵串联，每个都不放大信号范数，乘积也不放大——这是 mHC 稳定性的数学根基。

如果 M 是一般矩阵（HC 的情况），最大特征值可以 > 1（信号爆炸）或 < 1（信号衰减），在 86 层串联下不可控。

**延伸阅读**：主报告 CH 5.3

---

### Q6.8 Sinkhorn-Knopp 算法的工作原理是什么？

**简短回答**

Sinkhorn-Knopp 算法通过反复交替做行归一化和列归一化，把任意非负矩阵投影到双随机矩阵。V4 使用 20 次迭代，收敛精度 < 1e-4。

**详细解释**

算法步骤：
```
输入：M ∈ R^{c×c}，M >= 0
预处理：M = softmax(M) + eps    # 先做一次 softmax 行归一
for iter in range(19):           # 再做 19 次交替归一
    M = M / (M.sum(dim=1) + eps) # 行归一
    M = M / (M.sum(dim=0) + eps) # 列归一
输出：M（行和 ≈ 1，列和 ≈ 1）
```

**直觉**：第一步行归一把每行变成概率分布（行和=1），但列和失衡；第二步列归一修复列和，但又会破坏行和——交替迭代逐步逼近"行和=列和=1"。

**收敛性**：Sinkhorn (1964) 证明 n 步后与真实双随机矩阵的差异以 `O(1/n)` 速率收敛，精确上界是 $O(log(1/\delta))$ 步达到 δ 误差。

**V4 的实现细节**（来自 TileLang kernel）：
- 第一次用 $\mathrm{softmax}$ （而非简单除法）做行归一——自带数值稳定（减 max 防 exp 溢出）
- 后续 19 次用简单行/列除法
- eps = 1e-6 防止除零
- 4×4 矩阵，20 次迭代后误差 < 1e-4

💡 **面试要点**：Sinkhorn 不做优化，它是一个投影算子——把任意非负矩阵"最近点"投影到双随机流形。

**延伸阅读**：主报告 CH 5.3, CH 5.5

---

### Q6.9 为什么 Sinkhorn 选 20 次迭代？5 次不行吗？100 次呢？

**简短回答**

20 次是 V4 论文 ablation 的工程最优：5 次时双随机误差仍可观（信号 Lipschitz 上界偏大），100 次浪费算力（20 次后误差已 < 1e-4），20 次在精度和开销间取得平衡。

**详细解释**

| 迭代次数 | 双随机误差量级 | 信号稳定性 | 计算开销 |
|---|---|---|---|
| 5 | ~1e-1 | Lipschitz 上界偏离 1 | 低 |
| 10 | ~1e-2 | 可接受但非最优 | 中 |
| **20** | **< 1e-4** | **严格有界** | **中等** |
| 100 | < 1e-10 | 过度精确 | 浪费 |

每次迭代 = 1 次行归一 + 1 次列归一 = 2c 个除法（c=4 时 = 8 次除法）。20 次迭代 = 160 次除法/token/mHC 模块。V4-Flash 有 43 层 x 2 mHC/Block = 86 个 mHC 模块，总计 86 x 160 = 13760 次除法/token。

这部分开销约占 attn+MoE 总耗时的 +5%。100 次迭代会把开销推到 +25% 而精度收益极小。

**延伸阅读**：主报告 CH 5.3

---

### Q6.10 hc_mult=4 的设计依据是什么？为什么不是 2 或 8？

**简短回答**

V4 论文 ablation 显示 hc_mult=4 在"loss 下降速度 / 最终 ppl / 推理延迟"三维权衡上位于 Pareto 前沿：2 太弱（自由度不够），8 递减收益但开销翻倍。

**详细解释**

| hc_mult | 可学习参数/mHC | 表达能力 | 训练稳定性 | 推理延迟 | 评价 |
|---|---|---|---|---|---|
| 1 | 0 | 基线 | 基线 | 0% | 退化为普通残差 |
| 2 | 2x2+2=6 | 略高 | OK | +7% | 收益小 |
| **4** | **4x4+4+4=24** | **明显高** | **好** | **+15%** | **推荐** |
| 8 | 8x8+8+8=80 | 高 | 好 | +30% | 收益递减 |

**为什么 4 足够**：
- 表达瓶颈：hc_mult=2 时只有 2x2=4 个参数给 comb 矩阵，自由度太低学不到复杂的"信号调度"模式；hc_mult=4 时 4x4=16 个参数，足以学到"主路径/旁路/短接"的差异化模式
- 优化瓶颈：hc_mult=8 时 8x8 矩阵的 Sinkhorn 归一化 cache footprint 翻倍（8x8 vs 4x4），TileLang kernel 开销显著上升

⚠️ **易混淆**：mHC 不增加表征维度——4 通道 d 维隐藏被 reduce 回 1 通道 d 维，表征维度不变。增加的是"训练稳定性"而非"学习容量"。

**延伸阅读**：主报告 CH 5.4

---

### Q6.11 mHC 的三个分支（pre/post/comb）各自的作用是什么？

**简短回答**

pre（4 维向量）在子层前把 4 通道 reduce 为 1 通道输入；post（4 维向量）在子层后把 1 通道输出 expand 为 4 通道；comb（4×4 双随机矩阵）在残差路径上重新混合 4 通道。

**详细解释**

mHC 在每个残差位置维护 4 份隐藏状态。数据流：

```
4 通道输入 ──[α_pre reduce]──> 1 通道 ──[f: attn/MoE]──> 1 通道输出
    |                                                      |
    └──────[α_comb 重新混合]───────────────────────────────┘
                            │
                     [α_post expand]
                            ↓
                       4 通道输出 → 下一层
```

- **α_pre ∈ R^4**：通过 $\mathrm{sigmoid}$ 得到，范围 (0, 1)。对 4 通道加权求和 reduce 为 1 通道——告诉 f "每条残差流对该层多重要"
- **α_post ∈ R^4**：通过 $2 * \mathrm{sigmoid}$ 得到，范围 (0, 2)，中心 = 1。在 f 输出后 expand 为 4 通道——初始化时 ≈ 1（不缩放），让训练早期 mHC 退化为普通残差
- **α_comb ∈ R^{4×4}**：通过 Sinkhorn 投影约束到双随机矩阵。把原始 4 通道输入重新混合后与 f 输出相加

三个权重全部从 `mixes`（24 维向量）通过 $hc_split_sinkhorn$ 拆分得到，端到端可学习。

**代码映射**：
- pre: $hc_split_sinkhorn_kernel_$ L20-L21, $pre = \mathrm{sigmoid}(mixes[0:4] * scale_pre + base_pre) + eps$
- post: L22-L23, $post = 2 * \mathrm{sigmoid}(mixes[4:8] * scale_post + base_post)$
- comb: L24-L25 + L30-L52（Sinkhorn 投影）

**延伸阅读**：主报告 CH 5.2, CH 5.5

---

### Q6.12 公式 (5.1) 输入扩展具体怎么实现？

**简短回答**

把隐藏状态 x 复制 4 份得到 `[T, 4d]`，再用 α_pre（4 维向量）逐通道缩放后加权求和，得到 1 通道输入送入 f。

**详细解释**

```
x ∈ R^{T × d}                         # 原始隐藏状态
x_tile = tile(x, 4) ∈ R^{T × 4d}     # 沿通道维复制 4 次
x_4ch = x_tile.view(T, 4, d)          # reshape 为 4 通道
x_in = Σ_c α_pre[c] * x_4ch[:, c, :] # 逐通道加权求和 → [T, d]
```

在源码中对应 $Block.hc_pre$ 的：
```python
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```
其中 `pre` 是 α_pre（4 维），`x.view(shape)` 把 x reshape 为 `[T, 4, d]`，`pre.unsqueeze(-1)` 扩展为 `[T, 4, 1]`，逐通道相乘后在 dim=2 求和。

**延伸阅读**：主报告 CH 5.2, CH 5.5

---

### Q6.13 公式 (5.4) post + comb 残差输出具体怎么合成？

**简短回答**

$y = \alpha_post \cdot h + \alpha_comb \cdot x_in$ ，其中 h 是 f 的 1 通道输出（expand 为 4 通道），x_in 是残差路径上的 4 通道输入（经 comb 混合），相加得到 4 通道输出。

**详细解释**

```python
# Block.hc_post 的核心实现
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

分解：
- `post.unsqueeze(-1) * x.unsqueeze(-2)`：把 f 的 1 通道输出 x expand 为 4 通道（每通道乘以 post[c]）
- `comb.unsqueeze(-1) * residual.unsqueeze(-2)`：把 4 通道残差 residual（上一层的 4 通道输出）与 4x4 comb 矩阵相乘，得到混合后的 4 通道
- 两者相加 → 4 通道输出 y ∈ R^{T × 4 × d}

4 通道 y 作为下一层 Block 的输入，由下一层的 $hc_pre$ 通过 α_pre reduce 回 1 通道。

**延伸阅读**：主报告 CH 5.3, CH 5.5

---

### Q6.14 mHC 在训练和推理时的差异是什么？

**简短回答**

结构完全相同，但推理时 mixes 的计算是确定性的（无 dropout），且 Sinkhorn 投影只需做一次（同一 token 在同一层的 mixes 相同，batch 内可复用）。训练时 mixes 包含梯度信息，Sinkhorn 需要 autograd 通过。

**详细解释**

| 维度 | 训练 | 推理 |
|---|---|---|
| mixes 计算 | 含梯度，需要 backward | 纯前向，无梯度 |
| Sinkhorn | 需要 autograd 支持 | 纯数值计算 |
| 参数更新 | α_pre/α_post/α_comb_raw 每步更新 | 固定权重 |
| 开销占比 | 较高（+5-15% 训练步时间） | 较低（+15% 推理延迟） |

推理时 mHC 的 +15% 延迟主要来自：每层 2 次 hc_pre/hc_post 调用（43 层 = 86 次），每次含 1 次 Sinkhorn（20 次迭代）+ 1 次 reduce/expand。TileLang fused kernel 把 Sinkhorn 编译为单个 kernel launch，避免了多次 kernel 启动开销。

**延伸阅读**：主报告 CH 5.4, CH 5.5

---

### Q6.15 mHC 对显存的影响有多大？

**简短回答**

mHC 使每层隐藏状态的显存占用增加 4 倍（4 通道 vs 1 通道），但由于只存储残差路径（f 内部仍是 1 通道），且通过 activation checkpointing/recomputation 优化，实际额外显存约 +15-20%。

**详细解释**

V4-Flash 的 $hidden_size=4096$ ，每层 mHC 维护 4 通道 = 4 × 4096 = 16384 维的残差状态。43 层 x 2 mHC/Block = 86 个 mHC 模块。

**朴素计算**：每个 mHC 残差存储 4 份 d 维向量 = 4 × 4096 × 2 bytes (BF16) = 32 KB/token。86 层 = 2.75 MB/token。对 1M 上下文 = 2.75 TB（不可接受）。

**实际优化**：
- Activation checkpointing：不存储中间 mHC 状态，需要时重新计算
- 残差状态只在相邻层间传递，不需要全局存储
- 推理时 residual 是增量更新，不需要 4 份完整副本

V4 论文 §3.4.2 专门描述了 "Cost-Effective and Memory-Efficient Implementation of mHC"，通过 recomputation + fused kernel 把显存开销控制在可接受范围。

**延伸阅读**：主报告 CH 5.4, CH 5.5

---

### Q6.16 mHC 如何与 MoE 协同工作？

**简短回答**

mHC 在 MoE 外面包了一层多通道残差，MoE 内部仍是 1 通道操作。4 通道残差流让不同 MoE expert 的"贡献分布"在多通道上更均匀，缓解了 expert 利用率不均衡的问题。

**详细解释**

在 `Block.forward` 中，MoE 段的数据流：
```
4 通道输入 → hc_pre reduce → 1 通道 → ffn_norm → MoE → 1 通道输出 → hc_post expand → 4 通道
```

MoE 本身只看到 1 通道输入，完全感知不到 mHC 的存在。但 mHC 通过 comb 矩阵的混合，间接影响 MoE 的输入分布：
- 不同通道的 α_pre 权重不同，reduce 后的 1 通道输入是 4 份加权平均
- 训练过程中，4 通道学到的"信号调度"模式会间接影响 MoE 的路由决策

V4 论文报告 mHC + MoE 的组合比单独使用 MoE 训练更稳定，expert 利用率的方差更小。

**延伸阅读**：主报告 CH 5.4

---

### Q6.17 hc_split_sinkhorn kernel 的实现有什么工程亮点？

**简短回答**

TileLang 实现，把 "mixes → 拆分 pre/post/comb → Sinkhorn 20 次迭代" 编译为单个 fused kernel，避免 20 次 kernel launch 的开销，且在 shared memory 中完成 4×4 矩阵操作。

**详细解释**

关键工程细节：

1. **第一次用 $\mathrm{softmax}$ 而非简单除法做行归一**： $\mathrm{softmax}$ 自带减 max 操作防 exp 溢出，比 `M / M.sum(-1)` 更数值稳定
2. **sinkhorn_iters - 1 而非 sinkhorn_iters**：前面 $\mathrm{softmax}$ + 列归一已经等价于 1 次完整 Sinkhorn 步，后续只需 19 次
3. **post 用 $2 * \mathrm{sigmoid}$**： $\mathrm{sigmoid}$ 输出 ∈ (0, 1)，乘 2 后中心点 = 1，让训练早期 mHC 退化为普通残差（降低冷启动难度）
4. **pre 用 $\mathrm{sigmoid}$ 不乘 2**：pre ∈ (0, 1)，初始化时 ≈ 0.5，4 通道加权后总和 ≈ 2.0
5. **comb 用 $\mathrm{softmax}$ + eps**：先 $\mathrm{softmax}$ 行归一再列归一，eps=1e-6 防除零
6. **64 threads/block**：每个 token 的 mixes 独立处理，并行度高

💡 **面试要点**：TileLang fused kernel 是 mHC 推理 +15% 延迟（而非更高）的关键——20 次 Sinkhorn 迭代如果每步都是独立 kernel launch，开销会大得多。

**延伸阅读**：主报告 CH 5.5

---

### Q6.18 mHC 的论文作者是谁？与 Hyper-Connections 是什么关系？

**简短回答**

mHC 由 Xie et al. (2026) 提出，是 DeepSeek V4 论文中引用的独立工作。原始 HC 由 Zhu et al. (2025) 提出。mHC 是 HC 的约束改进版（加 Sinkhorn 流形约束），V4 是首个大规模部署 mHC 的模型。

**详细解释**

论文引用链：
- ResNet: He et al. (2015) — 原始残差
- Transformer: Vaswani et al. (2017) — Post-Norm
- Pre-Norm: Xiong et al. (2020) — 理论分析
- HC: Zhu et al. (2025) — 多通道可学习残差
- mHC: Xie et al. (2026) — 双随机流形约束
- V4: DeepSeek-AI (2026) — 首次在 284B MoE 上大规模部署 mHC

**延伸阅读**：主报告 CH 5.1

---

### Q6.19 为什么选 mHC 不选 Mixture-of-Depths (MoD)？

**简短回答**

MoD 让模型学习跳过某些层（动态深度），但不解决深层信号传播的稳定性问题。mHC 直接约束信号放大率，从根源上保证训练稳定性，且推理路径固定（不引入动态分支），工程部署更简单。

**详细解释**

| 维度 | MoD | mHC |
|---|---|---|
| 核心思想 | 动态跳层 | 多通道可控残差 |
| 深层稳定性 | 未解决 | 根本性解决 |
| 推理延迟 | 动态变化（不可预测） | 固定 +15% |
| 工程复杂度 | 高（需要动态路由 + padding） | 低（固定路径） |
| 与 MoE 兼容性 | 冲突（两层路由叠加） | 正交 |

MoD 的问题是：跳层只是"绕过"了不稳定层，没有修复信号传播本身。在 1M 上下文 + 43 层场景下，即使跳过一半层，剩余 21 层串联仍可能有稳定性问题。

mHC 则从根本上保证每一层的信号放大率 ≤ 1，不依赖跳层。

**延伸阅读**：主报告 CH 5.4

---

### Q6.20 mHC 有什么局限性？

**简短回答**

mHC 的主要局限是：推理延迟增加约 15%、每层额外 24 个可学习参数需要调优、4 通道残差的显存占用增加，以及 Sinkhorn 迭代的计算开销在 batch 较大时显著。

**详细解释**

具体局限：

1. **推理开销**：每层 2 次 hc_pre/hc_post = 86 次/forward pass，包含 86 次 Sinkhorn（20 次迭代）。TileLang fused kernel 把延迟控制在 +15%，但在 batch 极大的推理场景下仍不可忽略
2. **显存压力**：4 通道残差状态占用 4 倍显存，虽然通过 recomputation 优化，但在 1M 上下文下仍是工程挑战
3. **超参敏感**：hc_mult、hc_sinkhorn_iters、hc_eps 三个超参需要 ablation 调优。论文只验证了 hc_mult=4 在 V4-Flash 上的最优性，其他规模可能不同
4. **只解决稳定性不增加容量**：mHC 不增加模型的表达能力（表征维度不变），如果需要更多容量仍需增加参数
5. **与某些训练技巧的兼容性**：gradient checkpointing 与 mHC 的 4 通道残差状态管理增加了实现复杂度

**延伸阅读**：主报告 CH 5.4, CH 5.5

---

### Q6.21 mHC 与 FP8/FP4 量化如何协同？

**简短回答**

mHC 限制每层信号放大率 ≤ 1，让量化误差不至于在深层放大累积。没有 mHC 时，深层激活的数值范围可能远大于浅层，量化误差在深层被放大；有 mHC 后各层激活范围一致，量化精度更均匀。

**详细解释**

V4 使用 FP8（E4M3/E5M2）做训练和推理，FP4 做 routed expert 权重存储。量化的精度与数值范围直接相关——如果某层激活比其他层大 100 倍，相同量化 bit 下的相对误差差 100 倍。

mHC 的双随机约束让每层输出 $y = post \cdot f(x) + comb \cdot residual$ 的范数与输入 `x` 相当（comb 不放大，post 中心在 1），因此各层激活的数值范围一致，FP8/FP4 量化的精度损失均匀分布而非集中在某几层。

**延伸阅读**：主报告 CH 5.4

---

## CH 7. Muon 优化器

### Q7.1 优化器在 LLM 训练中的作用是什么？

**简短回答**

优化器决定"参数往哪个方向更新、更新多少"，是把梯度转化为参数更新的算法。好的优化器能让训练收敛更快、更稳定、泛化更好。

**详细解释**

LLM 训练的目标是最小化损失函数 L(θ)。反向传播计算出梯度 $g = \partialL/\partial\theta$ 后，优化器决定实际的更新方向和步长：

```
θ_{t+1} = θ_t - η · update(g_t, state_t)
```

最朴素的 SGD 直接用 $update = g_t$ 。但在 LLM 训练中：
- 参数量极大（V4-Flash 284B），不同参数的梯度量级差异巨大
- MoE 导致稀疏梯度，大部分参数每步没有新梯度
- 矩阵参数有谱结构（奇异值分布），逐元素更新会破坏这种结构

好的优化器需要：(1) 自适应步长（不同参数不同学习率）；(2) 动量（利用历史梯度平滑更新方向）；(3) 对矩阵参数保持谱结构。

💡 **面试要点**：优化器是 LLM 训练的"油门+方向盘"——决定模型参数怎么走、走多快。

**延伸阅读**：主报告 CH 6.1

---

### Q7.2 AdamW 的核心思想是什么？

**简短回答**

AdamW 结合一阶动量（梯度均值，加速方向）和二阶动量（梯度方差，自适应步长），并使用解耦的 weight decay。它是 LLM 训练的事实标准优化器。

**详细解释**

AdamW 的更新规则：

```
m_t = β1 · m_{t-1} + (1-β1) · g_t           # 一阶动量（梯度 EMA）
v_t = β2 · v_{t-1} + (1-β2) · g_t²          # 二阶动量（梯度方差 EMA）
m_hat = m_t / (1 - β1^t)                     # 偏差修正
v_hat = v_t / (1 - β2^t)                     # 偏差修正
θ_t = θ_{t-1} · (1 - η · λ) - η · m_hat / (√v_hat + ε)   # 更新
```

三个关键组件：
1. **一阶动量 m_t**：梯度的指数移动平均，平滑噪声，让更新方向更一致
2. **二阶动量 v_t**：梯度平方的指数移动平均，给每个参数自适应的学习率——梯度大的参数步长小，梯度小的参数步长大
3. **解耦 weight decay**： $\theta \cdot (1 - \eta\lambda)$ 独立于梯度更新，正则化效果更稳定

典型超参：β1=0.9, β2=0.999, ε=1e-8, λ=0.01-0.1。

**延伸阅读**：主报告 CH 6.1

---

### Q7.3 AdamW 在 MoE 大模型上有什么局限性？

**简短回答**

两个结构性问题：(1) 逐元素二阶动量与 MoE 稀疏梯度冲突——大部分 expert 参数每步没梯度，v_t 变成陈旧的指数滑动平均；(2) 逐元素更新不约束矩阵谱结构——训练后期奇异值分布退化，有效秩下降。

**详细解释**

**问题 1：稀疏梯度导致 v_t 失效**

MoE 中 256 个 routed expert 只激活 6 个。对每个 expert 矩阵 W_e，99%+ 的参数对一个给定 token 没梯度。v_t 在这些参数上只有"上一步的指数滑动平均"，几乎没有新信息流入。导致自适应学习率 $\eta / (\sqrt{v_t} + \epsilon)$ 在有梯度和没梯度的参数上差异巨大——训练信号极度不均衡，expert 利用率不稳定。

**问题 2：谱偏（spectral bias）**

Transformer 的权重矩阵 W ∈ R^{d×d'} 在训练中会形成奇异值分布。AdamW 的更新 $W -= \eta \cdot m / \sqrt{v}$ 是逐元素的——对矩阵整体的奇异值结构没有约束。训练后期容易出现"几个大奇异值 + 大量小奇异值"：

```
Wx ≈ σ_max · u_1 · (v_1^T · x)    # 只有大奇异值方向在工作
```

小奇异值方向几乎"死掉"——等于模型有效秩下降、参数利用率低、容易过拟合。

💡 **面试要点**：AdamW 的核心问题是"把矩阵当一袋标量"——每个元素独立更新，忽略了矩阵的几何结构。

**延伸阅读**：主报告 CH 6.1

---

### Q7.4 矩阵正交化的直觉是什么？

**简短回答**

正交化把矩阵的更新方向变成正交矩阵 O（满足 $O \cdot O^T$ = I），强制所有奇异值 = 1。直觉上，这等价于"重置"矩阵的谱结构，让它始终保持满秩，每个方向同等重要。

**详细解释**

一个矩阵 M ∈ R^{n×m} 的 SVD 分解是 $M = U \cdot \Sigma \cdot V^T$ 。如果 M 的奇异值分布不均匀（有的 σ >> 1，有的 σ ≈ 0），则 M 作为线性变换时：
- 大奇异值方向被放大（主导输出）
- 小奇异值方向被压缩（几乎"看不见"）

正交化把 M 变成 $O \approx  U \cdot V^T$ （去掉 Σ），所有奇异值 = 1。效果：
- 每个方向被平等对待——不偏袒也不忽略任何方向
- 矩阵保持满秩——参数利用率最大化
- 作为优化方向时，不会沿某个奇异值方向过度更新

类比：AdamW 像是"给每个人发不同大小的手电筒"（有些亮有些暗），Muon 像是"给每个人发相同亮度的手电筒"。

**延伸阅读**：主报告 CH 6.2

---

### Q7.5 正交化在 SGD 中有什么作用？与 conditioning 有什么关系？

**简短回答**

正交化改善 loss landscape 的 conditioning（条件数），让优化路径更接近"理想圆"而非"狭长椭圆"，从而加速收敛。

**详细解释**

损失函数的 Hessian 矩阵 H 的条件数 $\kappa = \lambda_max / \lambda_min$ 决定了优化难度：
- κ ≈ 1（well-conditioned）：loss 等高线是圆，SGD 直接指向最优
- κ >> 1（ill-conditioned）：loss 等高线是狭长椭圆，SGD 来回震荡

矩阵参数 W 的 Hessian 等价于它的奇异值结构。如果 W 的奇异值分布不均匀，κ 就大。正交化把奇异值全部拉到 1，等价于把 κ 拉到 1——loss landscape 变得更"圆"，收敛更快。

理论上，对凸优化问题，条件数 κ 决定了梯度下降的收敛速度 $O(\kappa \cdot log(1/\epsilon))$ 。把 κ 降到 1 可以把收敛加速 κ 倍。在大模型训练中，这直接转化为"更少的训练步数达到同样的 loss"。

**延伸阅读**：主报告 CH 6.2

---

### Q7.6 Muon 的核心思想是什么？

**简短回答**

Muon 把权重矩阵 W 的梯度 G 做动量累积后，通过 Newton-Schulz 迭代正交化为 O ≈ $U \cdot V^T$ ，再用 RMS rescale 回与 AdamW 相当的尺度。本质是"把矩阵当矩阵更新"而非"把矩阵当一袋标量更新"。

**详细解释**

Muon 的完整流程（对应 Algorithm 1）：

```python
# Step 1: 动量累积
M_t = μ · M_{t-1} + G_t

# Step 2: Nesterov trick + Newton-Schulz 正交化
O'_t = HybridNewtonSchulz(μ · M_t + G_t)   # O' ≈ U·V^T

# Step 3: RMS rescale
O_t = O'_t · √max(n, m) · γ

# Step 4: Weight decay + 更新
W_t = W_{t-1} · (1 - η · λ) - η · O_t
```

与 AdamW 的对比：

| 维度 | AdamW | Muon |
|---|---|---|
| 更新对象 | 逐元素（标量） | 整矩阵 |
| 二阶信息 | 逐元素方差 v_t | 谱结构（正交化） |
| 状态量 | m_t + v_t（2x 参数量） | M_t（1x 参数量） |
| 对矩阵参数 | 破坏谱结构 | 保持满秩 |

💡 **面试要点**：AdamW 问"这个标量该往哪走"，Muon 问"这个线性变换该往哪走"——后者保留了矩阵的几何结构。

**延伸阅读**：主报告 CH 6.2

---

### Q7.7 Muon 的论文作者是谁？

**简短回答**

原始 Muon 由 Keller Jordan (2024) 提出。V4 使用的是 Jordan et al. (2024) + Liu et al. (2025) 的变体，并自行设计了 Hybrid Newton-Schulz (8+2 分阶段) 的改进版本。

**详细解释**

Muon 的引用链：
- Jordan et al. (2024)：原始 Muon 优化器，使用单一系数的 5 阶多项式做 Newton-Schulz
- Liu et al. (2025)：加入 Nesterov trick + QK-Clip + RMS rescale + weight decay 的工程化版本
- V4 (DeepSeek-AI 2026)：Hybrid Newton-Schulz (8+2 分阶段) + 去掉 QK-Clip（因为 V4 的 $\mathrm{RMSNorm}$ on Q/K 天然防爆炸）

V4 论文 §2.4 引用了 "Muon (Jordan et al., 2024; Liu et al., 2025)"。

**延伸阅读**：主报告 CH 6.1

---

### Q7.8 Newton-Schulz 迭代在干什么？为什么比 SVD 更快？

**简短回答**

Newton-Schulz 迭代通过多项式近似把矩阵 M 逼近正交矩阵 $U \cdot V^T$ （去掉奇异值 Σ）。它只需要矩阵乘法（O(d^3)），不需要做真正的 SVD 分解（O(d^3) 但常数更大、GPU 不友好）。

**详细解释**

给定 M = $U \cdot Σ$·V^T，目标是求 O ≈ $U \cdot V^T$ 。标准方法是先做 SVD 再去掉 Σ，但 SVD 的计算：
- 需要迭代求特征值分解（QR 迭代 / divide-and-conquer）
- 常数因子大（约 10-20x 矩阵乘法）
- 在 GPU 上不容易向量化

Newton-Schulz 的思路：构造一个多项式 p(x)，使得对 σ ∈ (0, 1) 有 $p(\sigma^2) \cdot \sigma \approx  1$ ，然后把这个多项式应用到矩阵上：

```
M_k = a·M_{k-1} + b·(M_{k-1}·M_{k-1}^T)·M_{k-1} + c·(M_{k-1}·M_{k-1}^T)²·M_{k-1}
```

每次迭代只需要 2 次矩阵乘（ $A = M\cdot M^T$ 和 $A\cdot M$ ），GPU 高度优化。10 次迭代 = 20 次矩阵乘，常数远小于 SVD。

收敛保证：只要初始 M 的最大奇异值 ≤ 1（由 Frobenius 归一化保证），Newton-Schulz 多项式保证收敛到 UV^T。

💡 **面试要点**：Newton-Schulz 不做 SVD，它用矩阵乘法"迭代逼近"正交化结果——精度够用、速度快、GPU 友好。

**延伸阅读**：主报告 CH 6.2, CH 6.3

---

### Q7.9 系数 (3.4445, -4.7750, 2.0315) 的来源是什么？

**简短回答**

这三个系数通过数值优化搜索得到，目标是在 8 次迭代内让奇异值 σ ∈ (0, 1) 最快逼近 1。它们定义了一个三次多项式 $p(x) = 3.4445x - 4.7750x^3 + 2.0315x⁵$ ，在 σ² 空间上有良好的收敛性质。

**详细解释**

Newton-Schulz 迭代的每一步：

```
M_k = a·M_{k-1} + b·(M·M^T)·M + c·(M·M^T)²·M
```

对 M 的第 i 个奇异值 σ_i，这一步变成：

```
σ_i^{(k)} = a·σ_i^{(k-1)} + b·(σ_i^{(k-1)})³ + c·(σ_i^{(k-1)})⁵
```

我们要选 (a, b, c) 使得 $\sigma_i$ 快速趋近 1。这是一个多项式优化问题——在 σ ∈ (0, 1) 上找到最好的三次近似。

V4 通过离线数值搜索（在 284B 训练前的小规模 ablation）找到了 (3.4445, -4.7750, 2.0315) 这组系数。这不是理论推导的解析解，而是在"收敛速度 vs 数值稳定性"权衡下的工程最优。

8 次迭代后，σ 从初始的 0.01-0.99 范围收敛到 0.95-1.05 附近（足够好但不够精确），然后交由第 2 阶段的 Chebyshev 系数做精确稳定。

**延伸阅读**：主报告 CH 6.3

---

### Q7.10 为什么 V4 用 8+2 分阶段而不是 10 次统一迭代？

**简短回答**

前 8 次用快速收敛系数把奇异值快速拉到 ≈1 附近，后 2 次用 Chebyshev 系数精确稳定到 1。单一系数无法同时兼顾"快速收敛"和"精确稳定"——快系数在接近 1 时有震荡风险，稳系数前期太慢。

**详细解释**

| 阶段 | 系数 | 目标 | 性质 |
|---|---|---|---|
| 1-8 步 | (3.4445, -4.7750, 2.0315) | 快速收敛 | 数值搜索的最优系数 |
| 9-10 步 | (2, -1.5, 0.5) | 精确稳定 | Chebyshev 迭代经典系数 |

**阶段 2 的数学保证**：(2, -1.5, 0.5) 对应的迭代 $\sigma \to  2\sigma - 1.5\sigma^3 + 0.5\sigma⁵$ 在 σ ∈ [0.5, 1.5] 时有严格收敛保证——误差按几何级数衰减。只要前 8 步把 σ 推到这个范围内，后 2 步就能精确到机器精度。

这是 V4 区别于原始 Muon 和 Liu et al. (2025) 的关键创新：
- 原始 Muon：单一系数，10 次迭代
- Liu et al.：5 系数多项式 + QK-Clip
- V4：3 系数 + 8+2 分阶段，不需要 QK-Clip

💡 **面试要点**：V4 的 Hybrid Newton-Schulz 本质是"快速阶段 + 精确阶段"的两段式设计，比单一系数更稳定。

**延伸阅读**：主报告 CH 6.3

---

### Q7.11 公式 (6.1) Newton-Schulz 目标具体是什么？

**简短回答**

给定 M ∈ R^{n×m} 的 SVD 分解 M = $U \cdot Σ$·V^T，Newton-Schulz 的目标是求 O' ≈ $U \cdot V^T$ ，满足 O' · O'^T = I（半正交矩阵）。

**详细解释**

正交化的"理想结果"是把 M 的奇异值全部变成 1，保留左右奇异向量：
- 输入：M = $U \cdot Σ$·V^T，Σ 的对角元素是 σ_1, σ_2, ..., σ_r
- 目标输出：O' = $U \cdot V^T$ （Σ 被替换为 I）

这个 O' 满足：
- $O' \cdot O'^T = U\cdot V^T\cdot V\cdot U^T = U\cdot I\cdot U^T = I_n$ （如果 n ≤ m）
- 所有奇异值 = 1
- Frobenius 范数 = √min(n, m)

Newton-Schulz 通过迭代逼近这个目标，不需要显式计算 SVD。

**延伸阅读**：主报告 CH 6.2

---

### Q7.12 公式 (6.2) Frobenius 归一化的作用是什么？

**简短回答**

$M_0 = M / \|M\|_F$ 保证 M 的最大奇异值 ≤ 1，这是 Newton-Schulz 多项式收敛的前提条件。如果 σ_max > 1，多项式在 σ > 1 时可能发散。

**详细解释**

Frobenius 范数 $||M||_F = √(\Sigma M_{ij}^2)$ 是所有奇异值平方和的平方根。归一化后：

```
||M_0||_F = 1  →  Σ σ_i² = 1  →  σ_max ≤ 1
```

Newton-Schulz 多项式 $p(\sigma) = a\cdot \sigma + b\cdot \sigma^3 + c\cdot \sigma⁵$ 在 σ ∈ (0, 1) 时单调递增到 1。但如果 σ > 1：
- σ³ 和 σ⁵ 项指数增长
- 多项式可能发散到 +∞ 或 -∞
- 数值不稳定

因此每次迭代前（或至少初始时）必须保证 σ_max ≤ 1。V4 只在第一次迭代前做 Frobenius 归一化，后续迭代由多项式本身保持 σ 在合理范围内。

代码中：`M = M / (norm(M, 'fro') + eps)`，eps=1e-8 防除零。

**延伸阅读**：主报告 CH 6.3

---

### Q7.13 公式 (6.3) Newton-Schulz 的 5 系数 Polynomial 具体形式？

**简短回答**

V4 实际只用 3 系数： $M_k = a\cdot M + b\cdot (MM^T)M + c\cdot (MM^T)^2M$ 。5 系数版本是把 (MM^T)²M 展开为 5 个不同标量项的变体，V4 简化为显式三项。

**详细解释**

完整的 Newton-Schulz 迭代（公式 28）：

```
M_k = a · M_{k-1} + b · (M_{k-1} · M_{k-1}^T) · M_{k-1} + c · (M_{k-1} · M_{k-1}^T)² · M_{k-1}
```

其中：
- 第 1 项 $a \cdot M$ ：保留原始方向
- 第 2 项 $b \cdot (MM^T)M = b \cdot M \cdot (M^T \cdot M)$ ：一阶修正（把大奇异值缩小、小奇异值放大）
- 第 3 项 $c \cdot (MM^T)^2M$ ：高阶修正（加速收敛）

V4 用 3 系数而非 Liu et al. (2025) 的 5 系数，因为 3 系数已经足够表达所需的收敛行为，且减少了超参搜索空间。

**延伸阅读**：主报告 CH 6.3

---

### Q7.14 RMS rescale 因子的作用是什么？

**简短回答**

$O_t = O'_t \cdot \sqrt{max}(n,m) \cdot \gamma$ 把正交化后的更新方向 rescale 回与 AdamW 相当的 RMS 尺度，使 V4 可以直接复用 AdamW 的学习率调度，超参空间不膨胀。

**详细解释**

正交矩阵 O' 的 Frobenius 范数 = √min(n, m)。AdamW 的更新 $m/\sqrt{v}$ 的范数大致是 √(n·m)（与 W 同量级）。如果直接用 O' 作为更新，幅度太小。

RMS rescale 的推导：
```
||O'_t||_F = √min(n, m)           # 正交矩阵范数
||AdamW_update||_F ≈ √(n·m)      # AdamW 更新范数

需要放大因子 = √(n·m) / √min(n,m) = √max(n,m)

O_t = O'_t · √max(n,m) · γ       # γ 是可调的 rescaling factor
```

γ 的作用是微调——让 Muon 的更新幅度与 AdamW 完全匹配。V4 设定 γ 后，可以：
- 用同一份 warmup/decay 学习率调度
- 用同一份 weight decay 超参
- 不需要为 Muon 和 AdamW 分别调参

💡 **面试要点**：RMS rescale 是 Muon 能在工程中落地的关键设计——没有它，Muon 和 AdamW 就需要两套完全独立的超参。

**延伸阅读**：主报告 CH 6.4

---

### Q7.15 Muon 适用于哪些参数？不适用于哪些？

**简短回答**

Muon 适用于二维矩阵参数（attention Q/K/V/O、MoE FFN 权重、mHC comb 矩阵）。不适用于 1D 参数（embedding、prediction head、 $\mathrm{RMSNorm}$ 权重、mHC 的 gating factor），这些用 AdamW。

**详细解释**

V4 的 AdamW vs Muon 分工：

| 模块 | 优化器 | 理由 |
|---|---|---|
| Embedding | AdamW | 离散查找表，每行独立，无谱结构 |
| Prediction head (lm_head) | AdamW | logits 投影，逐行更新 |
| $\mathrm{RMSNorm}$ 权重 | AdamW | 1 维标量，逐元素更新 |
| mHC static bias (b_static) | AdamW | 仅 4 维，Muon 浪费算力 |
| mHC gating factor (α_pre/α_post) | AdamW | 4 维标量 |
| **Attention Q/K/V/O 矩阵** | **Muon** | d×d 矩阵，谱偏最严重 |
| **MoE FFN 权重** | **Muon** | 大矩阵 + 稀疏梯度 |
| **mHC comb 矩阵 (4×4)** | **Muon** | 矩阵结构有意义 |

判断标准：参数是否有"矩阵结构"（即两个维度都有语义）？如果有 → Muon；如果是 1D 或逐元素独立 → AdamW。

⚠️ **易混淆**：mHC 的 **gating factor**（4 维 α_pre/α_post）走 AdamW，但 **comb 矩阵**（4×4）走 Muon。同一个 mHC 模块内部根据参数形态分两个优化器。

**延伸阅读**：主报告 CH 6.4

---

### Q7.16 embedding 和 lm_head 为什么不用 Muon？

**简短回答**

Embedding 是离散查找表（每行独立），lm_head 是 logits 投影——它们都是逐行更新的参数，没有"矩阵整体的谱结构"，正交化对它们没有意义。

**详细解释**

- **Embedding**：vocab_size × d 的矩阵，每个 token 对应一行。训练时只有被采样的 token 对应的行有梯度，其他行不变。这种"逐行独立"的更新方式与矩阵正交化不兼容——正交化是对整个矩阵做的，但大部分行每步没梯度。
- **lm_head**：d × vocab_size 的矩阵，输出是 logits。每个 logits 对应 vocab 中一个 token，更新也是逐行（或逐列）的。没有"整个矩阵作为线性变换"的语义。

对这类参数，AdamW 的逐元素自适应学习率仍然是最优选择。

💡 **面试要点**：Muon 只对"有谱结构的大矩阵"有优势——小参数和 1D 参数用 AdamW 反而更好。

**延伸阅读**：主报告 CH 6.4

---

### Q7.17 Muon vs AdamW 的详细对比

**简短回答**

Muon 用矩阵正交化替代逐元素方差估计，用单个动量缓冲替代双缓冲（m+v），状态量减半但对矩阵参数收敛更快更稳。

**详细解释**

| 维度 | AdamW | Muon |
|---|---|---|
| 更新方式 | 逐元素（标量） | 整矩阵（正交化） |
| 状态量 | m_t + v_t（2x 参数量） | M_t（1x 参数量） |
| 显存 | 每 1B 参数约 16 GB (BF16 m+v) | 每 1B 参数约 8 GB (BF16 M only) |
| 自适应机制 | v_t（逐元素方差） | Newton-Schulz（谱约束） |
| 矩阵参数效果 | 谱偏（有效秩下降） | 满秩保持 |
| 稀疏梯度 | v_t 退化 | 不受影响（正交化不依赖逐元素统计） |
| 适用参数 | 所有 | 仅 2D 矩阵 |
| 计算开销 | 低（ $O(d)$ per parameter） | 高（20 次矩阵乘 per matrix） |
| 收敛速度 | 基线 | 更快（条件数改善） |
| 训练稳定性 | 好（成熟） | 需要工程适配 |

在 V4 的 284B MoE 上，Muon 让训练在 32T tokens 内收敛到更好的 loss，且 loss 曲线更平滑（更少的 spike）。

**延伸阅读**：主报告 CH 6.1, CH 6.2

---

### Q7.18 V4 为什么选择 Muon 而不是继续用 AdamW？

**简短回答**

三个原因：(1) 284B MoE 的稀疏梯度让 AdamW 的 v_t 退化；(2) 大矩阵参数的谱偏导致有效秩下降；(3) 训练预算要求更快收敛（32T tokens，成本极高）。

**详细解释**

V4-Flash 的训练量是 32T tokens（V3 的 2.2 倍）。如果继续用 AdamW：
- MoE 稀疏梯度导致 expert 利用率不稳定，需要更多训练步数补偿
- 矩阵谱偏导致深层模型有效参数量下降，需要更多参数（更大模型）达到同样效果
- 训练周期以月为单位，单次实验调参成本极高

Muon 在 V4 上的优势：
- 正交化天然处理稀疏梯度（不依赖逐元素统计）
- 保持矩阵满秩，参数利用率更高
- 收敛速度更快（V4 论文报告比 AdamW 少 ~30% 步数达到同等 loss）
- 与 mHC 协同：mHC 保证信号传播稳定，Muon 保证参数更新稳定

**延伸阅读**：主报告 CH 6.1, CH 1.2

---

### Q7.19 Muon 在 13B MoE 上的训练稳定性如何？

**简短回答**

V4 论文报告 Muon 在 284B/13B 激活的 MoE 上训练稳定，loss 曲线平滑，不需要特殊的 loss spike 处理。关键是 V4 的 Hybrid Newton-Schulz (8+2) + 去掉 QK-Clip 的组合在工程上比原始 Muon 更鲁棒。

**详细解释**

稳定性保障来自三个设计决策：

1. **8+2 分阶段**：前 8 步快速收敛不会震荡（快系数在 σ 接近 1 时仍有良好行为），后 2 步精确稳定保证 σ 严格 → 1
2. **$\mathrm{RMSNorm}$ on Q/K**：V4 的 attention 架构允许对 Q 和 KV entries 直接应用 $\mathrm{RMSNorm}$ ，attention logits 天然有界（ $Q\cdot K^T / \sqrt{d}$ 不会爆炸），因此不需要 Liu et al. (2025) 的 QK-Clip trick
3. **与 mHC 协同**：mHC 保证每层信号放大率 ≤ 1，Muon 的正交化更新在稳定的激活分布上工作

V4 论文没有报告需要特殊处理 loss spike 的情况，这与 V3 训练中需要仔细调节 lr 和 init_std 的经验形成对比。

**延伸阅读**：主报告 CH 6.3, CH 6.4

---

### Q7.20 Muon 与 mHC 如何协同工作？

**简短回答**

mHC 保证前向传播中信号方差不爆炸/消失，Muon 保证参数更新方向保持矩阵满秩。两者正交互补：mHC 管"信号怎么流"，Muon 管"参数怎么更新"。

**详细解释**

协同效果：

- **mHC 的贡献**：86 个双随机矩阵串联保证前向和反向的信号传播严格有界。这意味着 Muon 收到的梯度 G_t 的数值范围是可控的（不会出现梯度爆炸后 Newton-Schulz 输入 σ_max >> 1 的情况）
- **Muon 的贡献**：正交化更新保持每层权重矩阵的满秩，这意味着 mHC 的"4 通道信号调度"有足够的参数空间来学习有效模式

间接协同：Muon 让 expert 权重保持满秩 → expert 的输出分布更均匀 → mHC 的 4 通道混合更容易找到好的调度模式 → expert 利用率更稳定。

**延伸阅读**：主报告 CH 5.4, CH 6.4

---

### Q7.21 Muon 的工业部署情况如何？

**简短回答**

截至 2026 年，Muon 已被 DeepSeek V4 系列大规模部署。其他公司和实验室也在探索：Moonlight (2025) 使用 Muon 变体训练了 16B MoE 模型，开源社区有 muon-optimizer 等实现。但 AdamW 仍是主流，Muon 的工业验证案例还不够多。

**详细解释**

Muon 的部署现状：
- **DeepSeek V4/V4-Flash/V4-Pro**：284B-1.6T 规模，32-33T tokens 训练，最大的 Muon 部署案例
- **Moonlight (2025)**：16B MoE，使用 Muon 变体，开源
- **学术实验**：多个论文在 100M-1B 规模验证了 Muon 的收敛优势

限制因素：
- Newton-Schulz 需要 10 次矩阵乘，训练步时间增加
- 需要区分 Muon/AdamW 参数并分别维护优化器状态
- 工程复杂度高于 AdamW
- 对 1D 参数无优势，无法完全替代 AdamW

**延伸阅读**：主报告 CH 6.4

---

### Q7.22 Muon 的变体有哪些？与 Shampoo/SOAP 的关系？

**简短回答**

Muon 的变体包括 Moonlight (2025, 加入 adaptive learning rate)、原始 Muon (Jordan 2024)、Liu et al. (2025, Nesterov + QK-Clip)。Shampoo 和 SOAP 是基于矩阵/张量预条件的二阶方法，与 Muon 正交互补。

**详细解释**

| 方法 | 类型 | 核心思想 | 关系 |
|---|---|---|---|
| Muon (Jordan 2024) | 正交化 | Newton-Schulz 把更新方向正交化 | 原始版本 |
| Liu et al. (2025) | 正交化+工程化 | +Nesterov +QK-Clip +RMS rescale | Muon 的工程化改进 |
| V4 Muon | 正交化 | +Hybrid NS (8+2) -QK-Clip | Muon 的 V4 定制版 |
| Moonlight (2025) | 正交化 | +adaptive lr +Moon scheduler | Muon 的另一变体 |
| Shampoo (Gupta 2018) | 二阶预条件 | 左右预处理矩阵 G_l = (GG^T)^{1/4} | 独立的二阶方法 |
| SOAP (Vyas 2025) | 二阶预条件 | Shampoo + Adam 混合 | Shampoo 的改进 |

**Shampoo** 的核心思想：对 W ∈ R^{n×m} 的梯度 G，分别维护左预条件矩阵 L = (Σ $G \cdot G^T$)^{1/2p} 和右预条件矩阵 R = (Σ G^$T \cdot G$)^{1/2p}，更新方向为 L^{-1} · $G \cdot R$^{-1}。这等价于对 W 的奇异值做"软正交化"。

**关键区别**：Muon 是"硬正交化"（奇异值全部变 1），Shampoo 是"软正交化"（奇异值被缩放到更均匀但不完全是 1）。Muon 更激进但计算更简单（只需要矩阵乘），Shampoo 更温和但需要矩阵指数运算。

💡 **面试要点**：Muon 的本质是"一阶优化器 + 正交化"，Shampoo 是"二阶优化器 + 预条件"——两者目标相同（改善谱结构）但路径不同。

**延伸阅读**：主报告 CH 6.1

---

### Q7.23 为什么 AdamW 仍是 1D 参数的最佳选择？

**简短回答**

1D 参数（ $\mathrm{RMSNorm}$ 权重、bias、embedding 行）没有矩阵结构，不存在谱偏问题。AdamW 的逐元素自适应学习率对这些参数是最优的——简单、高效、无需额外计算。

**详细解释**

对 1D 参数，AdamW 的 v_t 提供的是"这个标量的梯度方差有多大"的信息——这恰好是自适应步长需要的。没有"矩阵"的语义，正交化无从谈起。

对 embedding 更具体：每步只有少量行有梯度，v_t 在有梯度的行上提供精确的自适应步长，在没有梯度的行上保持上一步的估计——这对 embedding 更新是合理的。

Muon 的 Newton-Schulz 需要矩阵乘法（O(d^3)），对 4 维的 α_pre/α_post 来说计算量极小但概念上无意义（4 个标量的"正交化"没有实质作用）。

**延伸阅读**：主报告 CH 6.4

---

### Q7.24 Muon 的 Nesterov trick 具体怎么用？

**简短回答**

Nesterov trick 把 Newton-Schulz 的输入从 $M_t$ 改为 $\mu \cdot M_t + G_t$——相当于"先往前看一步"再正交化。这比直接用 M_t 的收敛速度更快。

**详细解释**

标准动量： $M_t = \mu \cdot M_{t-1} + G_t$

不加 Nesterov： $O'_t = NewtonSchulz(M_t)$
加 Nesterov： $O'_t = NewtonSchulz(\mu \cdot M_t + G_t)$

$\mu \cdot M_t + G_t$ 是"如果动量继续走一步"的预估方向——先"展望"未来位置再决定正交化方向。这是 Nesterov accelerated gradient 的经典思想在 Muon 中的应用。

V4 论文 Algorithm 1 Line 5 明确使用 Nesterov trick：
```
O'_t = HybridNewtonSchulz(μ · M_t + G_t)
```

**延伸阅读**：主报告 CH 6.5

---

### Q7.25 Muon 的 weight decay 怎么处理？

**简短回答**

与 AdamW 相同： $W_t = W_{t-1} \cdot (1 - \eta \cdot \lambda) - \eta \cdot O_t$ ，weight decay 独立于梯度更新。这保证正则化效果不受正交化影响。

**详细解释**

Weight decay 的作用是防止参数范数无限增长（L2 正则化）。AdamW 的"解耦"设计把 weight decay 从梯度更新中分离：

```
W_t = W_{t-1} · (1 - η · λ) - η · O_t
```

Muon 沿用了这个设计（Liu et al. 2025 的建议）。关键点：
- $(1 - \eta \cdot \lambda)$ 独立缩放 W，与正交化更新 O_t 无关
- λ 的含义和调参方式与 AdamW 完全相同
- 不需要对正交化后的 O_t 做额外的正则化

V4 论文 Algorithm 1 Line 7： $W_t = W_{t-1} \cdot (1 - \eta\lambda) - \eta \cdot O_t$

**延伸阅读**：主报告 CH 6.5

---

### Q7.26 Muon 的计算开销有多大？

**简短回答**

每次 Newton-Schulz 迭代需要 2 次矩阵乘（`M @ M.T` 和 `A @ M`），10 次迭代共 20 次矩阵乘。相对 forward+backward 一次的 O(d^3) 计算，Newton-Schulz 是 O(10·d^3)——和一次完整前向的 attention 矩阵乘同量级。

**详细解释**

对 V4-Flash 的 d=4096 矩阵：
- 单次 `M @ M.T`：4096 × 4096 矩阵乘 = ~68G FLOPs
- 单次 `A @ M`：同上
- 10 次迭代：~1.36T FLOPs per weight matrix
- V4 有 ~50 个需要 Muon 的矩阵参数 → ~68T FLOPs per training step

对比：
- 一次完整 forward+backward（284B MoE）：~5.7P FLOPs per token
- Muon 开销：~68T / step ≈ forward+backward 的 1-2%

Muon 的额外显存：M_t 缓冲 = 1x 参数量（BF16）。对比 AdamW 的 m_t + v_t = 2x 参数量，Muon 实际上节省了显存。

💡 **面试要点**：Muon 的计算开销与 forward+backward 相比很小（~1-2%），且显存反而比 AdamW 省 1x 参数量。

**延伸阅读**：主报告 CH 6.3

---

### Q7.27 Muon 为什么不需要 QK-Clip trick？

**简短回答**

V4 的 attention 架构允许对 Q 和 KV entries 直接应用 $\mathrm{RMSNorm}$ ，让 Q/K 的数值范围天然有界（单位方差），因此 attention logits $Q\cdot K^T / \sqrt{d}$ 不会爆炸。Liu et al. (2025) 需要 QK-Clip 是因为他们的模型没有这个设计。

**详细解释**

Muon 的正交化更新可能让 Q/K 矩阵的奇异值均匀增大（因为正交化后所有奇异值 = 1，范数可能比原始梯度更大）。如果 Q 和 K 的范数不受控，attention logits $Q\cdot K^T / \sqrt{d}$ 会爆炸 → $\mathrm{softmax}$ 输出退化为 one-hot → 梯度消失。

Liu et al. (2025) 的解决方案是 QK-Clip：对 Q/K 矩阵的更新方向做 clip，限制范数。

V4 不需要这个 trick，因为 V4 的 attention 架构允许在 Q 和 KV entries 上直接应用 $\mathrm{RMSNorm}$ ：
```
Q_normalized = RMSNorm(Q)
K_normalized = RMSNorm(K)
attention_logits = Q_normalized · K_normalized^T / √d
```
$\mathrm{RMSNorm}$ 把 Q/K 的每个向量归一化为单位方差，所以 logits 天然有界。工程上少了一个 clip 算子，减少了 kernel 启动开销。

**延伸阅读**：主报告 CH 6.4

---

### Q7.28 V4 如何实现 Muon 的高效训练？

**简短回答**

V4 论文 §3.4.1 描述了 Muon 的高效实现：使用 hybrid ZeRO 策略分片 Muon 的动量缓冲、融合 Newton-Schulz 为单个 kernel、利用 TileLang 编译优化矩阵乘链。

**详细解释**

关键工程优化：

1. **Hybrid ZeRO**：Muon 的 M_t 缓冲（1x 参数量）使用 ZeRO-3 分片跨 GPU 存储，避免每张卡都保存完整副本
2. **Fused Newton-Schulz kernel**：把 10 次 Newton-Schulz 迭代编译为单个 kernel，减少 kernel launch overhead 和中间结果的 global memory 读写
3. **TileLang 编译**：利用 TileLang DSL 把 Newton-Schulz 的矩阵乘链编译为高效 GPU 代码
4. **与 AdamW 并行管理**：Muon 和 AdamW 参数在同一个训练循环中分别更新，共享学习率调度但维护独立的状态量

V4 论文 §3.4.1 提到 Muon 的训练效率在 284B MoE 上是工程可行的，Newton-Schulz 的开销被其他优化（通信-计算 overlap、ZeRO 分片）所掩盖。

**延伸阅读**：主报告 CH 6.5

---

### Q7.29 Shampoo 作为二阶方法的核心思想是什么？

**简短回答**

Shampoo 对权重矩阵 W ∈ R^{n×m} 分别维护左/右预条件矩阵，用梯度的二阶统计量（ $G \cdot G^T$ 和 G^$T \cdot G$ ）做矩阵级预条件，等价于"软正交化"——让奇异值更均匀但不强制为 1。

**详细解释**

Shampoo (Gupta et al. 2018) 的更新：

```
L_t = (Σ_{k=1}^{t} G_k · G_k^T)^{1/(2p)}     # 左预条件
R_t = (Σ_{k=1}^{t} G_k^T · G_k)^{1/(2p)}     # 右预条件
W_{t+1} = W_t - η · L_t^{-1} · G_t · R_t^{-1}
```

其中 p 是预条件阶数（通常 p=2 或 p=4）。直觉：
- $G\cdot G^T$ 收集了"行方向的梯度方差"→ L 捕捉 W 的行空间结构
- $G^T\cdot G$ 收集了"列方向的梯度方差"→ R 捕捉 W 的列空间结构
- $L^{-1} \cdot G \cdot R^{-1}$ 把梯度投影到预条件空间，等价于在 W 的奇异值空间做"缩放"

与 Muon 的区别：
- Shampoo 保留奇异值的相对大小（"软"调整），Muon 强制所有奇异值 = 1（"硬"重置）
- Shampoo 需要计算矩阵 p-th root（O(n^3)），比 Newton-Schulz 更贵
- Shampoo 的预条件矩阵需要更多显存（n^2 + m^2 per weight matrix）

**延伸阅读**：主报告 CH 6.1

---

### Q7.30 如果让你给一个新 LLM 项目选优化器，你会怎么选？

**简短回答**

如果模型包含 MoE 或大矩阵参数且训练预算紧张，考虑 Muon；否则 AdamW 仍然是安全的选择。具体决策看参数形态、训练规模和工程约束。

**详细解释**

决策树：

```
1. 模型有 MoE？
   → 是：Muon 对 expert 矩阵有明确优势（稀疏梯度问题）
   → 否：看下一步

2. 模型参数量 > 10B？
   → 是：Muon 对大矩阵参数有谱偏改善
   → 否：AdamW 足够（小模型谱偏不严重）

3. 工程团队有优化器定制经验？
   → 是：可以尝试 Muon（需要区分参数类型、维护双优化器）
   → 否：AdamW 更安全（生态成熟、调试简单）

4. 训练预算是否紧张？
   → 是：Muon 的收敛加速可以节省 ~30% 步数
   → 否：AdamW 的简单性更重要
```

⚠️ **易混淆**：Muon 不能完全替代 AdamW——1D 参数和 embedding 仍然需要 AdamW。实际部署是 Muon + AdamW 双优化器。

💡 **面试要点**：选优化器的核心判断是"参数是否有矩阵结构"——有则 Muon，无则 AdamW。

**延伸阅读**：主报告 CH 6.4

---

> **统计**：CH 6 (mHC) 21 Q + CH 7 (Muon) 30 Q = 51 Q。面试要点标注 10 处，易混淆标注 4 处，主报告引用覆盖 CH 5.1-5.5 + CH 6.1-6.5。


## CH 8. 工程实现 -- QA 问答

### Q8.1 什么是混合精度训练？为什么 LLM 需要它？

**简短回答**：混合精度训练是同时使用多种数值精度（如 FP32/FP16/BF16/FP8）完成训练的技术，其核心动机是在不显著损失模型质量的前提下，将显存占用和计算量降低 2-4 倍。

**详细解释**：
传统深度学习训练使用 FP32（32 位浮点）存储所有参数、梯度和优化器状态。以一个 284B 参数模型为例，FP32 下仅参数就占 284B x 4 bytes = 1.14 TB 显存，远超单卡容量。混合精度训练将"对精度敏感"的操作（如梯度累加、权重更新）保留在 FP32，而"对精度不敏感"的操作（如矩阵乘法、激活存储）降到 FP16/BF16/FP8。

具体来说有三层精度：**(1) 前向计算精度**（BF16/FP8 做 matmul，FP32 做累加）；**(2) 权重存储精度**（FP4/FP8 存权重，前向时反量化到 BF16 计算）；**(3) 梯度/优化器状态精度**（梯度用 BF16 通信，优化器状态用 FP32 保持数值精度）。

V4-Flash 走得更远——把 256 个 routed expert 压到 FP4（ $float4_e2m1fn$ ），比 FP8 再省一半显存。这是 V4 能在 2xH200/8xH100 上跑得动的关键。

**面试要点/易混淆**：混合精度不等于"把所有东西降到低精度"——关键在"混合"，不同操作分配不同精度。BF16 和 FP16 都不会训练的面试者在 LLM 岗位很减分。

**延伸阅读**：主报告 CH7.2 FP4+FP8 混合精度策略。


### Q8.2 BF16 / FP16 / FP32 三者有什么区别？

**简短回答**：三者都是 IEEE 754 浮点格式，核心差异在指数位和尾数位的分配——FP32 是 8 指数+23 尾数，FP16 是 5 指数+10 尾数，BF16 是 8 指数+7 尾数。BF16 的动态范围与 FP32 相同（8 位指数），但精度更低（7 位尾数）。

**详细解释**：
具体位分配与数值范围：

| 格式 | 总位宽 | 指数位 | 尾数位 | 动态范围 | 最小正数 | 最大正数 |
|------|--------|--------|--------|---------|---------|---------|
| FP32 | 32 | 8 | 23 | ~10^38 | 1.2e-38 | 3.4e38 |
| FP16 | 16 | 5 | 10 | ~65504 | 6.1e-5 | 65504 |
| BF16 | 16 | 8 | 7 | ~10^38 | 1.2e-38 | 3.4e38 |

BF16 的关键优势：**(1)** 动态范围与 FP32 相同，不会出现 FP16 那种梯度下溢/上溢（FP16 最大只有 65504，大模型梯度经常超过这个范围）；**(2)** BF16 与 FP32 互转只需截断尾数，无需特殊处理，硬件实现简单。代价是精度只有 FP16 的 1/8（7 位 vs 10 位尾数），但在深度网络的梯度噪声面前，这点精度损失基本可忽略。

FP16 在推理时仍有优势——其 10 位尾数对激活值更友好，且 NVIDIA Tensor Core 对 FP16 的支持更成熟。V4-Flash 的 RoPE 维度用 BF16（保位置精度），非 RoPE 维度用 FP8（省显存），这是按"精度需求"分层分配的实际实践。

**面试要点**：问"BF16 和 FP16 哪个好"时，标准答案是"训练用 BF16（防止溢出），推理看场景（FP16 精度更高但范围小）"——不要只说一个。

**延伸阅读**：主报告 CH7.2。


### Q8.3 FP8 的 E4M3 和 E5M2 分别是什么？分别用在什么场景？

**简短回答**：E4M3（4 指数+3 尾数）和 E5M2（5 指数+2 尾数）是 FP8 的两种子格式。E4M3 精度更高（尾数 3 位 → 8 级量化），适合前向传播的权重和激活；E5M2 动态范围更大（指数 5 位 → 范围 ~57344），适合反向传播的梯度。

**详细解释**：

| 格式 | 指数 | 尾数 | 精度 | 动态范围 | 典型用途 |
|------|------|------|------|---------|---------|
| E4M3 | 4 | 3 | 1/8 量化级 | max=448 | 前向权重、激活 |
| E5M2 | 5 | 2 | 1/4 量化级 | max=57344 | 反向梯度 |

E4M3 选 4 位指数的原因是：前向激活值在小范围（-448 到 448）内变化，4 位指数足以覆盖（2^4=16 个档位），3 位尾数比 E5M2 多 1 位精度，对最终 loss 的影响更小。E5M2 选 5 位指数的原因是：梯度值可能非常大（尤其在深层 MoE 的 expert 路由处），需要更大动态范围去吸收 outlier gradient。

V4-Flash 的 $act_quant_kernel$ （`kernel.py` L41）用的是 E4M3（ $float8_e4m3$ ），scale 用 `ue8m0`（无符号 8 指数 0 尾数，即 power-of-2）。block_size=128，即每 128 个元素共享一个 scale。

**面试要点**：要能说清楚"为什么不是一种 FP8 搞定一切"——前向要精度、反向要范围，这是两种不同需求。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` L36-L103。


### Q8.4 FP4 格式 E2M1FN 的 block=32 / E8M0 scale 是什么意思？

**简短回答**：E2M1FN 是 FP4 的一种格式（2 指数+1 尾数，无符号，normal number only），block=32 表示每 32 个权重元素共享一组量化参数（scale），E8M0 是 scale 的存储格式（8 位指数+0 位尾数=power-of-2），使得 scale 可以用快速的位运算代替除法。

**详细解释**：
E2M1FN 的位分配：

- 2 位指数：可表示 4 个指数档位（2^-2 到 2^1），动态范围 ~6
- 1 位尾数：2 级量化精度
- FN（Finite Normal）：不包含 denormal 和 inf/NaN，简化硬件
- 实际可表示的值：{0, 0.5, 1, 1.5, 2, 3, 4, 6}（8 个值，含 0 和负值共 16 个级别）

Block-wise 量化（block=32）意味着每 32 个权重独立计算自己的 scale $s = max(|w_i|) / fp4_max$ ，而不是整层共用一个 scale。block 越小，量化越精确但存储开销越大（每个 block 存一个 scale）。V4 选 32 是因为：(1) FP4 精度低，需要细粒度 scale 来补偿；(2) 32 是 GPU warp size，硬件友好。

E8M0 scale 是最关键的设计创新：scale 强制为 2 的整数次幂（power-of-2），存储为 8 位指数（无尾数）。好处是 `weight / scale` 这个除法变成位运算 shift——$fast_pow2(fast_log2_ceil(amax * fp4_max_inv))$ 。在 GPU 上，位运算比浮点除法快约 10 倍。

**面试要点**：block 大小是精度-开销的权衡——block=1 是 per-element 量化（最优精度但无压缩），block=全张量是 per-tensor 量化（最大压缩但精度差）。问为什么选 32 不选 128 时回答"FP4 精度太低必须用更细的 block 补偿"。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` L129-L177 $fp4_quant_kernel$ 。


### Q8.5 QAT（量化感知训练）和 PTQ（训练后量化）有什么区别？

**简短回答**：QAT 是在训练过程中模拟量化效果，让模型"学会"在低精度下工作；PTQ 是训练完成后对已有权重直接量化，无需重新训练。QAT 精度更高但需要完整训练流程，PTQ 快速但精度损失更大。

**详细解释**：

| 维度 | QAT | PTQ |
|------|-----|-----|
| 时机 | 训练中 | 训练后 |
| 原理 | Straight-Through Estimator (STE) 绕过 quantize 的不可微 | 直接对权重做量化，最多加少量校准数据调 scale |
| 精度损失 | 极小（<0.5% PPL 退化） | 低精度（FP4）时可达 5-10% PPL 退化 |
| 计算开销 | 训练时多 5-15% 算力 | 几乎为零 |
| 适用场景 | 生产部署模型（V4-Flash） | 快速实验、baseline 评估 |

QAT 的核心技巧是 STE（Straight-Through Estimator）：前向时，把 BF16 权重 `w` 量化到 FP4 得到 $w_q$ ，再反量化到 BF16 得到 $w_dq$ ，用 $w_dq$ 算前向输出。反向时，梯度直接穿过量化函数（ $\partialL/\partialw = \partialL/\partialw_dq$ ），假装量化不存在。这让模型逐渐适应"权重被量化"的数值分布。

V4-Flash 的 routed expert 走 QAT——训练全程用 FP4 模拟， $fp4_act_quant$ 在 Indexer 的 Q/K 上也在做 QAT。PTQ 不适合 FP4 的原因是 4-bit 精度太低了，不经过训练适应会直接崩掉。

**面试要点**：区分 QAT 和 PTQ 的关键不在于"是否重训练"，而在于"梯度是否穿过量化函数"——QAT 有 STE 反向传播，PTQ 没有梯度。

**延伸阅读**：主报告 CH7.2、CH7.3。


### Q8.6 V4-Flash 的混合精度策略是怎样的？

**简短回答**：V4-Flash 采用三层混合精度：Attn/shared expert/公共部分用 FP8（E4M3, block=128），256 个 routed expert 用 FP4（E2M1FN, block=32, E8M0 scale），RoPE 维度和累加用 BF16/FP32。存储精度决定显存，计算精度决定质量。

**详细解释**：
精确分配表：

| 组件 | 存储精度 | 计算精度 | 量化粒度 | Scale 格式 |
|------|---------|---------|---------|-----------|
| Embedding | BF16 | BF16 | 无量化 | -- |
| Attention Q/K/V/O | 混合：RoPE dim=BF16, 其余 FP8 | BF16 | block=128 | E4M3/ue8m0 |
| Shared Expert | FP8 | FP32→BF16 | block=128 | E4M3 |
| Routed Experts (x256) | FP4 | FP32（反量化后） | block=32 | E8M0 (power-of-2) |
| $\mathrm{RMSNorm}$ | FP32 | FP32 | 无量化 | -- |
| mHC 权重 | FP32 | FP32 | 无量化 | -- |
| LM Head | BF16 | BF16 | 无量化 | -- |
| KV Cache | 混合：RoPE dim=BF16, 其余 FP8 | -- | -- | -- |

这样分配的逻辑是"精度需求驱动"：**(1)** Routed expert 最多（占模型参数 ~60%），用最激进的 FP4 压缩；**(2)** Attention 需要精确的位置信息，RoPE 维度保留 BF16；**(3)** Shared expert 始终激活，对整体质量影响大，用 FP8 保精度；**(4)** mHC 的 Sinkhorn 迭代对精度敏感（20 次迭代的除法链），用 FP32。

**面试要点**：要能解释"为什么 routed expert 可以 FP4 但 shared expert 不行"——routed expert 稀疏激活（每个 token 只用 6/256），单个 expert 的量化噪声被平均掉了；shared expert 永远激活，噪声直接进入每一层的输出。

**延伸阅读**：主报告 CH7.2；`config.json` 中 $expert_dtype: "fp4"$, $quantization_config$ ；`inference/kernel.py` L36-L177。


### Q8.7 Block-wise 量化和 per-tensor 量化有什么区别？为什么 V4 选 block-wise？

**简短回答**：Per-tensor 量化是整层（或整个张量）共用一个 scale，block-wise 量化是将张量分成小块（block），每块独立计算 scale。Block-wise 量化精度更高但存储开销更大（需要存多个 scale）。V4 选 block-wise 是因为极端精度（FP4）下，per-tensor 的精度损失不可接受。

**详细解释**：
以 V4 expert 矩阵 W ∈ R^{$4096 \times 2048$} 为例：

- Per-tensor：整个 8.4M 元素共享 1 个 scale，量化误差 E[|w - w_q|] ≈ σ_w / 2^b，当 b=4 时约 σ_w/16。如果 W 内元素跨度三个数量级，scale 被 outlier 主导，大部分参数被量化到 0。
- Block-wise（block=32）：每 32 个元素 1 个 scale。 $4096 \times 2048$ / 32 ≈ 262K 个 block，多存 262K×1 byte（E8M0）= 256 KB 的 scale 数据。但量化误差降到 E[|w - w_q|] ≈ σ_block / 16，由于 block 内方差远小于全局方差，精度大幅提升。

存储开销对比：block-wise 额外存储的 scale 占比 = 32/(32×0.5) ≈ 6.25%（scale 1 byte，权重 0.5 byte per element）。用 6% 的额外存储换数倍的精度提升，是 FP4 量化的工程标准做法。

V4 还有一个细节：FP8 用 block=128，FP4 用 block=32。原因：FP8 精度本身较高（4 位指数+3 位尾数 ≈ 256 级），block=128 的精度损失可接受；FP4 精度低（2 位指数+1 位尾数 ≈ 16 级），必须用更细的 block 补偿。

**面试要点**：问"block 选多大"时，标准答案是精度-开销权衡，并且要能说出 V4 的 FP8=128 / FP4=32 的双轨设计。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` $act_quant_kernel$(block_size=128) / $fp4_quant_kernel$(block_size=32)。


### Q8.8 FP4 GEMM 算子是怎么实现的？和 FP8 GEMM 有什么不同？

**简短回答**：FP4 GEMM 的核心流程是"反量化到 BF16 → 矩阵乘 → 累加回 FP32"。与 FP8 GEMM 的区别在于：(1) 权重存储格式是 FP4 而非 FP8；(2) scale 用 E8M0 (power-of-2) 实现快速除法；(3) 当前硬件（H100/H200）上 peak FLOPs 与 FP8 相同（都走 FP16 Tensor Core），但 FP4 权重加载带宽省一半。

**详细解释**：
FP4 GEMM 的具体步骤（以 V4 expert 前向为例）：

```python
# 伪代码：FP4 dequant + GEMM
# w_fp4: [M, K] packed as uint8 (2 FP4 per byte)
# x: [K, N] BF16
def fp4_gemm(w_fp4_packed, scale_e8m0, x_bf16):
    # Step 1: 反量化 FP4 → BF16
    w_bf16 = dequant_fp4(w_fp4_packed, scale_e8m0)  # 位运算 + shift
    # Step 2: BF16 GEMM on Tensor Core
    y = w_bf16 @ x_bf16
    return y
```

$dequant_fp4$ 内部用 $fast_round_scale$ 把 scale 恢复： $scale_fp32 = 2^{e8m0_value - 127}$ 。因为 scale 永远是 2 的幂，`w / scale` 等价于 `w >> log2(scale)`，纯位运算。

与 FP8 GEMM 的关键差异：

| 维度 | FP8 GEMM | FP4 GEMM |
|------|---------|---------|
| 权重位宽 | 8 bit/param | 4 bit/param |
| Scale 格式 | E4M3/FP32 | E8M0 (power-of-2) |
| 反量化操作 | 浮点除法 | 位运算 shift |
| 权重加载带宽 | 基准 | 基准 × 0.5（省一半） |
| 当前硬件 peak | 与 BF16 Tensor Core 相同 | 与 FP8 相同（走同样 TC） |
| 未来硬件 peak | -- | 理论上比 FP8 高 1/3 |

关键 insight：FP4 在现有硬件（H100/H200）上的计算速度并不比 FP8 快（因为 Tensor Core 以 FP16 精度算，FP4 和 FP8 都要先反量化），但**权重加载带宽省一半**，对 memory-bound 的 decode 阶段有实际加速。

**面试要点**：不要错误地说"FP4 在 H100 上比 FP8 快 2x"——当前硬件上都走 FP16 Tensor Core，peak 一样。FP4 的优势在于显存和未来硬件。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` L257-L368。


### Q8.9 FP4 量化对模型质量有什么影响？V4 怎么处理的？

**简短回答**：FP4 量化天然会导致路由专家权重精度下降（16 个量化级别 vs FP8 的 256 个），V4 通过 QAT 训练、block=32 细粒度量化、swiglu_limit=10.0 钳制、route_scale=1.5 补偿四招把影响降到可忽略。

**详细解释**：
FP4 对质量的潜在影响路径有三条：

**路径 1：专家输出失真。** 一个 expert 的 $\mathrm{SwiGLU}$ 三层矩阵都被量化到 FP4，down 投影的输出 $w2(silu(w1\cdot x) * w3\cdot x)$ 中，量化噪声经两层传播。V4 用 QAT 训练让 expert 在训练期就适应 FP4 噪声，相当于"提前学会在低精度下工作"。实测 FP4 vs FP8 的 expert 输出余弦相似度 > 0.98。

**路径 2：数值溢出。** FP4 最大可表示值为 6.0（E2M1FN），而 $\mathrm{SwiGLU}$ 的 `silu(gate) * up` 在训练初期可能超过此值。V4 引入 $swiglu_limit=10.0$ （`Expert.forward` L14-L16），把 `up` 钳制在 [-10, 10]，gate 钳制在 [-inf, 10]。这是在 V3 中不存在的约束。

**路径 3：路由误差。** Gate 网络的评分 $\mathrm{sqrtsoftplus}(W_g \cdot x)$ 不依赖 expert 权重，所以 FP4 不影响"选哪个 expert"。但选中的 6 个 expert 的输出质量略有下降。V4 用 $route_scale=1.5$ 放大 routed 专家输出，让 shared expert（FP8）不至于主导输出，部分补偿 FP4 的质量损失。

V4 技术报告未给出 FP4 vs FP8 在 V4-Flash 上的具体 PPL 对比数字，但从 Benchmark 表现（V4-Flash-Max 在 reasoning 上接近 GPT-5.2）可以推断 FP4 的质量损失在可控范围内。

**面试要点**：不要回避"FP4 有质量损失"这个事实——正确回答是"有损失但通过 QAT + block=32 + swiglu_limit 控制在可接受范围"。

**延伸阅读**：主报告 CH4.1、CH7.2。


### Q8.10 RoPE 的原理是什么？为什么需要位置编码？

**简短回答**：RoPE（Rotary Position Embedding）通过将 attention 的 Q 和 K 向量乘以位置相关的旋转矩阵，使 $Q \cdot K$ 内积仅依赖于相对位置（m-n），而非绝对位置。位置编码的根本原因是 self-attention 本身是置换不变的（permutation invariant）——没有位置信息，"A 给 B"和"B 给 A"是一样的。

**详细解释**：
标准 self-attention 中， $\mathrm{softmax}(QK^T/\sqrt{d})\cdot V$ 对输入序列的任何排列都产生相同的输出（V 的加权和不变）。这导致两个问题：**(1)** 序列中 token 的先后顺序无法被模型感知；**(2)** "距离近的 token 关系密切"这一语言先验无法被利用。

RoPE 的数学：对 d 维向量 q 和 k，按维度对分组（2 维一组 = 1 个复数），第 i 对的旋转角度为 $\theta_i = 1 / base^(2i/d)$ ，位置 m 处的旋转为：

```
q_m[i] = q[i] * cos(m*θ_i) - q[i+1] * sin(m*θ_i)
q_m[i+1] = q[i] * sin(m*θ_i) + q[i+1] * cos(m*θ_i)
```

旋转后 $q_m \cdot k_n$ 仅依赖 `m - n`（差角公式 cos(a-b) = cos a cos b + sin a sin b），实现完美的相对位置编码。同时 RoPE 具有"远程衰减"性质：`|m-n|` 越大， $q_m \cdot k_n$ 的期望值越小，符合"距离远的 token 相关性弱"的先验。

V4-Flash 中 RoPE 仅施加在每头的最后 64 维（ $qk_rope_head_dim=64$ ），前 448 维不带位置信息。这种"partial RoPE"设计是 MLA 体系的继承：让 RoPE 维度和非 RoPE 维度分开管理（RoPE 维度用 BF16，非 RoPE 维度用 FP8）。

**面试要点**：区分"绝对位置编码"（学习型/sinusoidal）和"相对位置编码"（RoPE/ALiBi）——前者把位置信息加在输入上，后者编码在 attention 计算中。RoPE 是目前 LLM 的主流选择。

**延伸阅读**：主报告 CH7.1；`inference/model.py` L199-L229 $precompute_freqs_cis$ 。


### Q8.11 YaRN 是什么？它的 NTK-aware 插值和参数含义是什么？

**简短回答**：YaRN（Yet another RoPE extensioN）是一种将 RoPE 预训练时的上下文长度（如 64K）扩展到更长（如 1M）的方法。核心机制是：低频维度线性插值拉伸（factor=16），高频维度保持原样，中间维度用 ramp 平滑过渡（β_fast=32, β_slow=1）。

**详细解释**：
YaRN 解决的核心问题是"RoPE 外推"——模型在 64K 长度训练后直接用于 1M 长度，高频旋转分量会超出训练时的角度范围（所谓"out-of-distribution rotation"），导致 attention 分数异常。

YaRN 的三个数学组件：

1. **NTK-aware scaling**：对每个频率维度 d，计算"在扩展后的最大长度 L' 下，该维度旋转了 r = L'·θ_d / (2π) 圈"。如果 r < β_fast（高频，旋转圈数多），保持原频率不动（需要高频分辨局部位置）；如果 r > β_slow（低频，旋转圈数少），频率除以 factor（拉伸以覆盖更长序列）。

2. **Ramp 函数**： $\gamma(d) = clamp((d - low) / (high - low), 0, 1)$ ，其中 low 和 high 由 $find_correction_range$ 根据 β_fast/β_slow 计算。 $freq_new = freq / factor * (1-\gamma) + freq * \gamma$——在低频和高频之间做线性插值。

3. **超参含义**（V4-Flash config.json）：
   - `factor=16`：从 64K 扩展到 64K×16=1,024K≈1M
   - $\beta_fast=32$ ：保留 32 圈以上旋转的高频维度（局部位置敏感）
   - $\beta_slow=1$ ：拉伸 1 圈以下的低频维度（覆盖 1M 全长）
   - $original_max_position_embeddings=65536$ ：预训练时的原始长度
   - $rope_theta=10000$ ：RoPE 基础频率

**面试要点**：能解释"为什么不好好训练 1M 而非要外推"——从 64K 到 1M 的训练数据非常少且昂贵，外推是工程上更经济的方案。YaRN 是目前外推方法中理论和实践结合最好的。

**延伸阅读**：主报告 CH7.1；`inference/model.py` L200-L229 $precompute_freqs_cis$ ； $code-snippets/yarn_rope.py$ 。


### Q8.12 sliding_window=128 在 V4-Flash 中扮演什么角色？

**简短回答**：sliding_window=128 在每个 attention 层的作用是保留最近 128 个 token 的**无损**KV 信息，作为 CSA/HCA 压缩 KV 的补充。它解决了压缩 attention 的一个根本问题——同一压缩块内的 token 无法互相看到。

**详细解释**：
在 CSA（m=4 压缩）中，每 4 个 token 才产生 1 个压缩向量。这意味着位置 t 的 query 无法通过压缩向量看到同一压缩块内的其他 token（位置 t-1, t-2, t-3），因为这些 token 的压缩向量还没生成（因果约束）。

滑动窗口 KV 的作用是补全这个缺口：**前 128 个 token 的原始 K 和 V 永远不经压缩，直接参与 attention**。这样：(1) 最近 128 个 token 之间的交互是 dense 的（无信息损失）；(2) 压缩 KV 负责 128 位之后的长程信息；(3) 两者在 attention 中拼接（ $torch.cat([window_kv, compressed_kv])$ ），实现"局部 dense + 远距离 sparse"。

实际源码（`Attention.forward` L507）： $topk_idxs = get_window_topk_idxs(win=128, bsz, seqlen, start_pos)$ ，先把窗口位置的索引算出来，再与压缩 KV 的索引拼接。

**面试要点**：问"既然有 CSA/HCA 压缩了为什么还要滑窗"时，答案分两层：(1) 压缩失去同一块内的 token 间交互；(2) 局部 token 在语言模型中天然更重要（邻近 token 的互信息高于远距离 token）。

**延伸阅读**：主报告 CH3.2、CH3.4；`inference/model.py` L507。


### Q8.13 MQA（KV=1）对 1M 上下文 cache 的影响有多大？

**简短回答**：MQA（Multi-Query Attention）将 KV head 从 64 个减少到 1 个，使 KV cache 的 head 维度缩减 64 倍。在 1M 上下文下，这是 V4-Flash 能把 KV cache 压到 ~4 GB 的关键设计之一（配合 CSA/HCA 压缩）。

**详细解释**：
假设 V4-Flash 用标准 MHA（64 Q heads + 64 KV heads）：
- 每层 KV cache per token = 64 heads × 512 dim × 2 (K+V) × 2 bytes (BF16) = 131,072 bytes = 128 KB
- 1M 上下文 43 层 = 43 × 1M × 128 KB ≈ 5.5 TB —— 完全不可行

换成 MQA（64 Q heads + 1 KV head）：
- 每层 KV cache per token = 1 head × 512 dim × 2 × 2 bytes = 2,048 bytes = 2 KB
- 1M 上下文 43 层 = 43 × 1M × 2 KB ≈ 88 GB —— 仍太大

加上 CSA/HCA 压缩（交替策略，约一半层压缩到 T/4，一半到 T/128）+ 滑窗（128）+ 混合精度（RoPE BF16+非 RoPE FP8）：
- CSA 层（21 层）：window=128 + T/4=262,144 → 262,272 × (64×2 + 448×1) bytes ≈ 151 MB/layer
- HCA 层（20 层）：window=128 + T/128=8,192 → 8,320 × 576 bytes ≈ 4.8 MB/layer
- 合计 ≈ 3.2 GB + 0.1 GB ≈ **3.3 GB**

MQA 是这段压缩链条的第一步——如果 KV heads=64，即使有 CSA/HCA 压缩，KV cache 也是 3.3×64≈211 GB，仍然不可行。所以 V4 的 1M 上下文 = MQA（head 维压缩 64×） × CSA/HCA（时间维压缩 4-128×） × 混合精度（位宽压缩 ~1.78×）。

**面试要点**：能建一个三层压缩的账——"head 维 × 时间维 × 位宽维"，每一步省多少，最终 5.5 TB → 3.3 GB，是三步接力实现的。

**延伸阅读**：主报告 CH3.5；`inference/model.py` L447 $num_key_value_heads=1$ 。


### Q8.14 1M 上下文在 V4-Flash 中是如何实现的？全景图是怎样的？

**简短回答**：V4-Flash 的 1M 上下文是**四层压缩接力**实现的：MQA（head 维 64×）→ CSA/HCA 交替（时间维 4-128×）→ FP8 混合精度存储（位宽维 ~1.78×）→ mHC 残差流复用（通道维 4×）。四层叠加后，1M 上下文下单 token FLOPs 降到 V3.2 的 ~10%，KV cache 降到 ~7%。

**详细解释**：
全景拆解（每层一个"压缩杠杆"）：

| 层 | 机制 | 压缩对象 | 压缩比 | 代价 |
|---|------|---------|--------|------|
| 1 | MQA (KV=1) | head 维度 | 64× | 表达能力略降（shared KV 信息量小）|
| 2 | CSA (m=4, 索引 top-512) | 时间维度（21 层） | 4× | 额外 Indexer 计算 |
| 3 | HCA (m=128, 位置 top-512) | 时间维度（20 层） | 128× | 信息损失大 |
| 4 | 滑窗 (window=128) | 补充局部交互 | -- | 128 个无压缩 KV |
| 5 | FP8 混合存储（RoPE BF16+非 RoPE FP8） | 位宽维度 | ~1.78× | RoPE 维度精度保留 |
| 6 | mHC 残差流复用（hc_mult=4） | 通道维度 | 4× | Sinkhorn 20 次迭代开销 |

这四层压缩配合三层架构支撑：

- **YaRN 位置扩展**：将 RoPE 从 64K 外推到 1M（factor=16, β_fast=32, β_slow=1）
- **因果稀疏**：top-k 选择严格保持因果性（query 只能看到位置 < t 的压缩块）
- **逐层交替**：前 2 层纯滑窗（保 local info），后 41 层 CSA/HCA 交替（均衡精度与速度）

最终效果（V4 技术报告 Figure 1）：
- V4-Flash 单 token FLOPs ≈ 10% of V3.2
- V4-Flash KV cache ≈ 7% of V3.2
- 1M 上下文 Needle-in-haystack：92%
- 长文档检索 MRCR@1M：83.5%

**面试要点**：被问"1M 上下文怎么实现"时，分两层回答：(1) 计算侧——CSA+HCA 稀疏 attention 把 $O(T^2)$ 降到 O(T²/m + T·k)；(2) 存储侧——MQA + FP8 混合存储 + mHC 复用把 cache 从 TB 级压到 GB 级。不要只说一个侧面。

**延伸阅读**：主报告 CH3.5、CH7.1。


### Q8.15 V4-Flash 的训练数据组成是怎样的？

**简短回答**：V4-Flash 的训练数据分两个大阶段——预训练（32T tokens，多样化高质量语料）和后训练（SFT + GRPO + OPD 三阶段）。预训练数据覆盖网页、代码、数学、书籍、多语言，后训练数据按"领域专家 → 蒸馏统一"范式组织。

**详细解释**：

**预训练阶段**（32T tokens）：
DeepSeek 未公开详细的预训练数据配比，但从 V3/V4 论文可以推断大致组成：
- 网页文本（Web pages）：~60-70%，覆盖中英文及多语言
- 代码（Code）：~15-20%，GitHub、Stack Overflow 等
- 数学/科学（Math/Science）：~5-10%，论文、教材、题目
- 书籍（Books）：~5%，长文本
- 其他（QA、论坛等）：~5%

数据处理管线：质量过滤 → 去重（MinHash）→ 文档拼接（packing）→ tokenize（129,280 词表）。V4 继承了 V3 的 Fill-in-Middle（FIM）训练策略。

**后训练阶段**：

| 阶段 | 数据量 | 数据特点 |
|------|--------|---------|
| SFT | 数百万条高质量指令 | 覆盖代码、数学、推理、指令跟随、agent |
| GRPO (RL) | 在线生成 + 奖励信号 | 组内相对比较，奖励来自测试通过率/正确性等 |
| OPD | 主模型自生成的 on-policy 轨迹 | 领域专家提供 token-level logit 监督 |

**面试要点**：区分 pretrain 数据和 post-train 数据的本质差异——pretrain 数据量大但质量参差（需要过滤），post-train 数据量小但质量极高（需要人工标注/模型筛选）。SFT 训练的是"模式匹配"，RL 训练的是"策略优化"。

**延伸阅读**：主报告 CH7.3。


### Q8.16 GRPO（Group Relative Policy Optimization）的原理是什么？

**简短回答**：GRPO 是一种不需要 critic 模型的强化学习算法，核心是对同一个 prompt 采样一组（group）回答，用组内相对比较（relative comparison）替代绝对 reward 来估计优势函数。它避免了传统 PPO 需要训练一个与策略模型同规模的价值网络的高成本。

**详细解释**：
传统 PPO（Proximal Policy Optimization）训练 RLHF 需要 4 个模型：policy、reference、reward、value（critic）。其中 critic 模型与 policy 模型同规模，显存和算力开销巨大。

GRPO 的简化思路：
1. 对一个 prompt，用当前 policy 采样 G 个回答（G=4 或 8）
2. 对每个回答用 reward model 打分，得到 ${r_1, r_2, ..., r_G}$
3. 计算组内标准化优势： $A_i = (r_i - mean(r)) / std(r)$
4. 用 clipped surrogate objective 更新 policy： $L = min(ratio * A_i, clip(ratio, 1-\epsilon, 1+\epsilon) * A_i) - \beta * KL(π\|π_ref)$

关键创新是 **"组内相对比较"**——不需要知道"这个回答绝对值有多好"，只需要知道"在这个组里它相对好/差多少"。这让优势估计（advantage estimation）从"精确值"退化到"相对顺序"，虽然信息量少了，但省掉了 critic 模型。

V4 在每个领域专家训练中都用 GRPO：
- Code expert：reward = test case pass rate
- Math expert：reward = 答案正确性 + 步骤完整性
- Reasoning expert：reward = 多模型投票一致性

**面试要点**：要能说清楚 GRPO 为什么比 PPO 省——"组内比较"替代"价值网络"，把 4 模型压缩到 3 模型（去了 critic）。

**延伸阅读**：主报告 CH7.3；DeepSeek-V3 论文（GRPO 首次在 LLM 缩放中被提出）。


### Q8.17 什么是 On-Policy Distillation (OPD)？为什么 V4 用它替代传统蒸馏？

**简短回答**：On-Policy Distillation (OPD) 是一种特殊的知识蒸馏方法，学生模型（统一主模型）用自己生成的轨迹（而非固定数据集）去学习教师模型（领域专家）的 token-level logit 分布。与传统"off-policy"蒸馏的区别在于：训练样本的分布 = 模型部署时的真实分布，避免了分布漂移。

**详细解释**：
传统蒸馏（off-policy）的流程：准备固定数据集 → 教师模型对数据集打分（存 logit）→ 学生模型用 KL loss 对齐。问题：固定数据集的分布与学生模型部署时遇到的真实分布不一致（分布漂移），导致 KL 对齐的"最优方向"在部署时不是最优。

OPD 的流程：
1. 给定 prompt，学生模型自己生成回答 y ~ π_student(·|prompt)
2. 对生成的每个 token，教师模型给出 token-level logit p_expert(y_t | y_{<t}, prompt)
3. 学生模型用 KL 散度对齐：L = α · KL(π_student || π_expert) + (1-α) · L_SFT

"On-policy"的关键在于——学生模型在"自己会说的话"上学习教师模型的精细分布，不需要额外覆盖"学生不会说的话"（因为那些话在部署时也不会出现）。

V4 的 OPD 流程：
- 4 个领域专家（Code/Math/Reasoning/Agent）分别在各自领域 SFT+GRPO 训练
- 统一主模型通过 OPD 从 4 个专家处学习，一次蒸馏获得全部能力
- 通过控制生成的 CoT 长度实现 3 种推理模式（Non-think / Think High / Think Max）

**面试要点**：区分 on-policy 和 off-policy 的"分布"差异——前者是"学生自己探索 → 老师纠正"，后者是"老师做好的饭 → 学生吃"，前者在长尾分布上泛化更好。

**延伸阅读**：主报告 CH7.3。


### Q8.18 MTP（Multi-Token Prediction）的原理和作用是什么？

**简短回答**：MTP（Multi-Token Prediction）是在标准"预测下一个 token"任务之外，额外添加"预测下下个 token"的辅助训练目标。V4-Flash 有 1 个 MTP 层（ $num_nextn_predict_layers=1$ ），它让每个 token 参与两次训练，在不增加推理成本的前提下提升数据效率 ~1.2-1.3x。

**详细解释**：
标准自回归 LM 训练：给定前缀 x_{1:t}，预测 x_{t+1}，loss = CrossEntropy(p(x_{t+1}), x_{t+1})。

MTP 训练：在标准 loss 之上，额外加一层 MTPBlock 预测 x_{t+2}：
```
h_t = Transformer(x_{1:t})     # 标准 hidden state
e_{t+1} = Embedding(x_{t+1})   # 下一个 token 的 embedding
h'_t = MTPBlock(h_t, e_{t+1})  # 融合预测
p(x_{t+2}) = LMHead(h'_t)      # 预测 x_{t+2}
L_MTP = CrossEntropy(p(x_{t+2}), x_{t+2})
L_total = L_LM + λ · L_MTP
```

MTPBlock 的结构（源码 `model.py` L738-L767）：
- 继承标准 Block（Attention + MoE + mHC）
- 额外加 $e_proj$ （embedding → dim）和 $h_proj$ （hidden → dim）两个投影
- 输入 = $embed(x_{t+1}) + hidden(x_{1:t})$ ，把"已知的下一 token"和"历史 hidden"融合
- 推理时可关掉 MTP 层（标准 generate），也可用于 speculative decoding（用 MTP 草稿 token 加速）

**面试要点**：MTP 不改变模型结构（仅添加额外训练头），推理时可完全关闭，因此是"免费的训练效率提升"。能说清楚 MTP 和 speculative decoding 的关系（MTP 训练出来的头可当作 draft model）是加分项。

**延伸阅读**：主报告 CH7.4；`inference/model.py` L738-L767 `MTPBlock`；`code-snippets/mtpblock.py`。


### Q8.19 32T tokens 的训练规模意味着什么？和 V3 的 14.8T 比如何？

**简短回答**：V4-Flash 在 32T tokens 上预训练，是 V3 公开训练量 14.8T 的 2.2 倍，但模型激活参数却从 V3 的 37B 降到 13B（35%）。这意味着 V4-Flash 用更少的参数、更多的数据达到了更好的性能，遵循"Chinchilla 缩放定律"的"数据比参数更重要"方向。

**详细解释**：
Chinchilla 缩放定律（Hoffmann et al. 2022）的核心结论：对给定计算预算，最优的模型大小和训练 token 数应该等比例缩放。V3 的 37B 激活在 14.8T tokens 上训练，是"参数偏多、数据偏少"；V4-Flash 的 13B 激活在 32T tokens 上训练，是"参数偏少、数据偏多"。

实际对比：

| 维度 | V3 (671B/37B) | V4-Flash (284B/13B) | V4-Pro (1.6T/49B) |
|------|---------------|---------------------|-------------------|
| 预训练 tokens | ~14.8T | 32T | 33T |
| 激活参数 | 37B | 13B | 49B |
| Token/参数比 | ~2.0 | ~11.3 | ~2.0 |
| 推理成本 | 高（8×H800 勉强） | 中（8×H100 舒适） | 高（多机） |

32T tokens 的训练工程挑战：
- 存储：32T tokens × ~4 bytes/token ≈ 128 TB 原始文本
- 算力：13B × 32T × 6 FLOPs ≈ 2.5×10^24 FLOPs（V4 未公开具体 GPU·天数）
- 时间：预估数千 GPU·天（基于 H800 集群）
- 稳定性：Muon + mHC 确保 32T 训练无 loss spike

**面试要点**：不要只说"32T 很大"——要能从 Chinchilla 定律的角度分析"为什么 V4 把参数压到 13B 但数据加到 32T"这一设计选择的动机。

**延伸阅读**：主报告 CH1.1、CH9.2。


### Q8.20 V4-Flash 的后训练三阶段（SFT → RL → OPD）各做了什么？

**简短回答**：后训练三阶段中，SFT（Supervised Fine-Tuning）在高质量指令数据上建立基础能力，GRPO（Group Relative Policy Optimization）用强化学习优化特定领域的策略行为，OPD（On-Policy Distillation）把多个领域专家的能力蒸馏回统一主模型。三者是"奠基 → 强化 → 融合"的递进关系。

**详细解释**：

**阶段 1：SFT**
- 作用：把预训练基座的"补全"能力转换为"遵循指令"能力
- 数据：数百万条高质量（prompt, response）对，覆盖代码、数学、推理、指令跟随、agent 等
- 训练：标准 next-token prediction loss，全参数微调
- 关键：SFT 训练的是"模式匹配"——看到 prompt 类型 → 输出对应格式的回答
- 输出：获得有基础指令跟随能力的模型

**阶段 2：GRPO (RL)**
- 作用：用奖励信号让模型学会"做对的事"而非"像训练数据那样说"
- 数据：模型在线生成（on-policy），无需固定数据集
- 奖励信号：各领域独立——代码=测试通过率，数学=答案正确+步骤完整，agent=任务完成率
- 关键：GRPO 不用 critic 模型，G 个回答组内比相对好坏
- 输出：4 个领域专家模型（Code/Math/Reasoning/Agent）

**阶段 3：OPD**
- 作用：把 4 个领域专家的能力蒸馏回一个统一模型
- 数据：主模型自己生成的 on-policy 轨迹 + 专家提供 token-level logit
- 训练：KL(主模型 || 专家) + SFT loss
- 关键：支持不同 CoT 长度的蒸馏（Non-think: 短/Think High: 中/Think Max: 长）
- 输出：一个统一模型，3 种推理模式

**面试要点**：能说清楚三个阶段分别"输入什么、输出什么、为什么需要"——SFT 给格式，GRPO 给策略，OPD 给统一。三阶段缺一不可。

**延伸阅读**：主报告 CH7.3。


### Q8.21 V4-Flash 的训练稳定性是如何保证的？（含 mHC、Muon）

**简短回答**：V4-Flash 的训练稳定性由三根支柱支撑：(1) mHC 通过 Sinkhorn-Knopp 双随机约束将残差信号传播严格有界化（Lipschitz=1），防止深层信号爆炸/消失；(2) Muon 优化器通过矩阵正交化更新（Newton-Schulz）消除 AdamW 的谱偏问题，保持满秩训练；(3) swiglu_limit=10.0 和 attention Q/K $\mathrm{RMSNorm}$ 防止极端数值出现。

**详细解释**：
V3.2 训练到后期遇到的"残差饱和"问题（深层 block 的 $\|x_{l+1} - x_l\|$ 收敛到非零常数）在 V4 中被 mHC 解决：双随机矩阵的特征值严格 ≤1，86 个 mHC 残差串联后信号范数不会指数级发散。这相当于给网络加了一个 Lipschitz 约束。

Muon 的贡献在另一个维度：AdamW 的逐元素更新不约束矩阵的奇异值分布，训练后期容易出现"几个大奇异值主导"的谱偏。Muon 每一步把更新方向强制正交化（奇异值全拉平到 1），保持参数矩阵的有效秩（effective rank）不退化。

数值稳定的工程措施：
- $swiglu_limit=10.0$ ：FP4 量化下 $\mathrm{SwiGLU}$ 的 up 钳制，防止 NaN/Inf
- Attention Q/K 前加 $\mathrm{RMSNorm}$ ：让 $Q\cdot K^T$ 的数值范围有界，省去 QK-Clip
- Sinkhorn 迭代中用 eps=1e-6 防除零
- RoPE 维度保留 BF16（保位置精度）
- prefill 阶段的 fp32 累加（防止长序列求和溢出）

**面试要点**：稳定性问题要从"信号前向传播"和"梯度反向传播"两个方向分别分析——mHC 解决前向（信号不爆炸），Muon 解决反向（梯度不退化）。

**延伸阅读**：主报告 CH5（mHC）、CH6（Muon）。


### Q8.22 为什么 V4 前 3 层用 hash routing 而不是 score routing？

**简短回答**：前 3 层用 hash routing（按 token id 查表决定 expert）有两个原因：(1) 性能——省去每次前向的 gate matmul ($4096 \times 256$) + $\mathrm{sqrtsoftplus}$ + topk，对 batch=1 decode 有显著加速；(2) 浅层 attention 在拼词法/局部句法，对"哪个专家擅长什么"的要求低，hash 的确定性分配不影响质量。

**详细解释**：
Hash routing 的实现（源码 $Gate.__init__$ L554-L560）：
- 构造查找表 $self.tid2eid \in  R^{vocab\times 6}$ ，`[129280, 6]` int32
- $requires_grad=False$ ，不可训练
- 前向： $indices = self.tid2eid[input_ids]$ ，一次张量 gather

一个 $4096 \times 256$ 的 gate matmul（W_g·x）在 BF16 下约 2M FLOPs，加上 $\mathrm{sqrtsoftplus}$ （256 个 exp 和 log 运算）和 topk（256 个元素排序），对 batch=1 来说是毫秒级的非平凡延迟。省掉 3 层就是 3 倍的这笔开销。

更重要的是，V4 论文 ablation 显示：前 3 层 hash + 后 40 层 score 的组合在 PPL/MMLU 上略优于"全 score"或"前 6 层 hash"。原因是浅层的 attention 分布尚未形成"特定 expert 对特定 token"的精细偏好——hash 分配的"随机但均匀"反而比"score 路由但噪声大"更好。

**面试要点**：不要只说"hash routing 快"——要解释"为什么浅层用 hash 不影响质量"。关键是"浅层还没学会专家偏好，score 路由的效果和随机差不多"。

**延伸阅读**：主报告 CH4.3.1；`inference/model.py` L554-L560。


### Q8.23 swiglu_limit=10.0 为什么是 V4-Flash 独有的？它解决什么问题？

**简短回答**： $swiglu_limit=10.0$ 是对 $\mathrm{SwiGLU}$ 激活函数中 `up` 投影输出的数值钳制（clamp 到 [-10, 10]），防止 FP4 量化下出现 NaN/Inf。V3 没有这个约束是因为 V3 不用 FP4（FP8 的动态范围足够吸收极端值），而 FP4 的最大可表示值仅为 6.0，不做钳制会溢出。

**详细解释**：
$\mathrm{SwiGLU}$ 的计算： $output = w2(silu(w1\cdot x) * clamp(w3\cdot x, -10, 10))$ 。其中 $w3\cdot x$ 是"up projection"的输出。

问题场景：训练初期或遇到 outlier 输入时， $w3\cdot x$ 可能达到 20-50。FP8 的动态范围（E4M3 max=448）足够吸收这个值，但 FP4（E2M1FN max=6.0）会直接饱和。饱和后的梯度为 0（hard clamp 的导数在饱和区为 0），导致该专家再也无法被训练。

V4 的解决：对 up 投影做 soft clamp `torch.clamp(up, min=-10.0, max=10.0)`，gate 投影做 `min=float('-inf'), max=10.0`。选择 10.0 这个阈值是根据训练初期的激活值分布和 FP4 的 max=6.0 反推的——10.0 比 6.0 略高，给 FP4 留一点"smooth saturation"的缓冲。

代码在 `Expert.forward`（`model.py` L587-L608）：
```python
gate = self.w1(x).float()
up = self.w3(x).float()
up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
x = F.silu(gate) * up
```

**面试要点**：要能连接"swiglu_limit"和"FP4 量化"的因果关系——因为 FP4 才需要 clamp，FP8/FP16 不需要。不要孤立的记一个超参。

**延伸阅读**：主报告 CH4.5；`inference/model.py` L587-L608 `Expert`。


### Q8.24 V4-Flash 的 Gate 为什么用 $\mathrm{sqrtsoftplus}$ 而不是 $\mathrm{sigmoid}$ 或 $\mathrm{softmax}$ ？

**简短回答**： $\mathrm{sqrtsoftplus}$ = sqrt(log(1+exp(z))) 是 DeepSeek V4 从 $\mathrm{sigmoid}$ 演进来的新评分函数。比 $\mathrm{sigmoid}$ 好在对负 logits 更敏感（sqrt 放大小值），比 $\mathrm{softmax}$ 好在不过度归一化（各专家评分独立而非互相竞争），比 $\mathrm{softplus}$ 好在数值范围更稳（开方压缩动态范围）。

**详细解释**：
三种评分函数在 256 个 expert 上的行为差异：

| 函数 | z=-10 时 | z=0 时 | z=10 时 | 问题 |
|------|---------|--------|---------|------|
| $\mathrm{sigmoid}$ | ~0.00005 | 0.5 | ~0.9999 | 负值几乎为 0，不区分 |
| $\mathrm{softmax}$ | 归一化后极低 | 中间 | 主导（排他性） | 一个 expert 垄断 |
| $\mathrm{softplus}$(z) | ~0.00005 | 0.69 | 10 | 动态范围太大 |
| **$\mathrm{sqrtsoftplus}$(z)** | ~0.007 | 0.83 | 3.16 | **两者平衡** |

$\mathrm{sqrtsoftplus}$ 的优点：
1. **对负值敏感**：`z=-10` 时 $\mathrm{sqrtsoftplus}$ ≈ 0.007，比 $\mathrm{sigmoid}$ 的 0.00005 大 140 倍——这意味着"暂时不相关的 expert"仍有机会被激活
2. **非排他性**：各专家分数独立（不像 $\mathrm{softmax}$ 你高我就低），适合"多个专家共同处理一个 token"的细粒度 MoE
3. **范围可控**： $z\to \infty$ 时趋近 sqrt(z) 而非 z，最大 256 个 expert 的评分总和在合理范围内

V4 的 evolution 路径：V3 初期用 $\mathrm{softmax}$ → V3.2-Exp 改 $\mathrm{sigmoid}$ → V4 改 $\mathrm{sqrtsoftplus}$ 。这是连续三代的评分函数迭代。

**面试要点**：能说出"$\mathrm{softmax}$ 排他 vs $\mathrm{sigmoid}$ 对负值不敏感 vs $\mathrm{sqrtsoftplus}$ 兼顾"的逻辑链条。

**延伸阅读**：主报告 CH4.2；`inference/model.py` L571 $F.\mathrm{softplus}(scores).sqrt()$ 。


### Q8.25 aux-loss-free 负载均衡的具体更新规则是什么？

**简短回答**：aux-loss-free 通过每个 expert 维护一个可训练偏置 b_e ∈ R（fp32），在每步训练末尾按公式 $b_e ← b_e - \eta \cdot (1/E - f_e)$ 更新（η=0.001, E=256）。f_e 是当前 step 该 expert 被 top-k 选中的频率。关键约束：b 仅影响 topk 选择，不参与 routing weight 计算（不进 $\mathrm{softmax}$ 分母）。

**详细解释**：
完整更新逻辑：

1. 训练 step t 结束时，统计每个 expert e 被选中的次数 `counts[bincount(indices)]`
2. 计算频率： $f_e = counts[e] / total_tokens$
3. 目标频率： $p = 1/256 \approx  0.00390625$
4. 更新偏置： $b_e = b_e - lr_bias * (p - f_e)$
   - 若 $f_e > p + \delta$ （过载）： $b_e ↓$ → 下一 step 该 expert 更难被选中
   - 若 $f_e < p - \delta$ （欠载）： $b_e ↑$ → 下一 step 该 expert 更容易被选中
   - δ 是容忍带（V3 论文给 δ≈0.05）
5. 关键： $b_e$ **仅加在 topk 前的 score 上**（ $score_for_topk = g + b$ ），不参与对 routing weight 的归一化

第 5 点是最核心的设计——源码 `Gate.forward` L27 保存了不带 b 的 $original_scores$ ，L35 从 $original_scores$ 中 gather 权重。这意味着 routing weight（决定 expert 输出的缩放因子）不受 b 影响，即使某个 expert 的 b 很小（"不被选中"），一旦被选中，其输出仍然按原始的 g 值缩放。

**面试要点**：能说清楚"偏置只改选择、不改权重"是 aux-loss-free 的核心——这样负载均衡的副作用（b 的偏置）不会污染模型的前向计算质量。

**延伸阅读**：主报告 CH4.3.2；`inference/model.py` L546-L586 `Gate`。


### Q8.26 routed_scaling_factor=1.5 的作用和设计动机是什么？

**简短回答**： $routed_scaling_factor=1.5$ 在 MoE 输出合成时放大 6 个 routed expert 的加权和，防止始终激活的 shared expert 主导输出。设计动机是"用 6 个 routed expert 达到 8 个的等效表达力"（配合 top-k 从 V3 的 8 砍到 V4 的 6）。

**详细解释**：
V4-Flash 的 MoE 输出公式（伪代码）：
```python
y = shared_expert(x) + 1.5 * sum(w_i * routed_expert_i(x) for i in top-6)
```

如果没有 $route_scale=1.5$ ， $sum(w_i * expert_i(x))$ 的 6 项加权和（w_i 归一化后 ≈ 1/6 量级）在数量级上远小于 shared expert 的输出。这会导致 shared expert（FP8，始终激活）主导最终输出，routed expert 的信息被"淹没"。

具体数值：
- Shared expert 输出：约 $O(1)$ 量级（ $\mathrm{RMSNorm}$ 后）
- 6 个 routed expert 加权和（无缩放）：约 0.1-0.3 量级（每个 w_i 约 1/6 到 1/3，expert 输出 RMS ≈ 1）
- 加 1.5 缩放后：约 0.15-0.45 量级，与 shared 的输出比例更合理

V3.2 用 top-k=8，routed 输出天然有 8/6 ≈ 1.33x 的优势。V4 把 k 从 8 砍到 6（省 25% routed FLOPs），route_scale=1.5 是"对等的补偿"——让 6 个 expert 的加权和 ≈ 9 个 expert 不加权的量级。

**面试要点**：要能链接"k 砍了"和"scale 加了"的因果关系——这是一个经典的"剪枝 + 补偿"组合调参。

**延伸阅读**：主报告 CH4.1、CH4.2。


### Q8.27 Compressor 中的 overlap 机制（ratio=4 时启用）是做什么的？

**简短回答**：overlap 机制在 CSA（ratio=4）中让相邻的压缩组共享 2 个 token 的信息——每 4 个 token 压缩的窗口与前一组有重叠，让压缩边界更平滑，避免边界 token 的信息损失。HCA（ratio=128）不开 overlap，因为 128 的压缩比已经足够稀疏，重叠开销不值得。

**详细解释**：
标准无重叠压缩：token 1-4 → 压缩向量 A，token 5-8 → 压缩向量 B。问题是 token 4 和 5 虽然相邻，但它们在不同压缩组中，query at token 4 和 query at token 5 看到的压缩信息完全不同——边界处出现"信息断层"。

Overlap 机制的实现（源码 `Compressor.forward`）：
- `coff = 1 + overlap = 2`：每个 token 产生 2d 维表示（前 d 维供前一组，后 d 维供后一组）
- prefill 阶段： $overlap_transform$ 把 `[b, s, ratio=4, 2d]` reshape 成 `[b, s, 2*ratio=8, d]`
- 前 4 个位置（t1..t4）取前 d 维 = 组 A 的压缩素材；后 4 个位置（t5..t8）取后 d 维 = 组 B 的压缩素材
- 但 t1..t4 **全部** 参与了组 A 的 $\mathrm{softmax}$-pool（因为 overlap_transform 把位置映射做了调整）

实际效果：每个压缩组有效利用 2m=8 个 token 的信息（前向看 4 个、后向看 4 个），压缩品质提升。

不开 overlap 的 HCA：128 个 token 压缩 1 次，此时 1 个 token 的信息丢失在 128 个 token 的 pool 中微不足道，overlap 的额外开销（kv_state 容量翻倍）性价比太低。

**面试要点**：overlap 本质上是用"通道维翻倍（2d）换时间维平滑"——仅当 ratio 很小时（如 4）才值得。

**延伸阅读**：主报告 CH3.7.1；`inference/model.py` L290 $self.overlap = compress_ratio == 4$ 。


### Q8.28 V4 的 Attention 中 attn_sink 是什么？它解决什么问题？

**简短回答**：attn_sink 是每个 attention head 的一个可学习标量参数，加到 $\mathrm{softmax}$ 分母上，允许 attention 的总权重不等于 1。它解决的是"某些 query 对所有压缩 KV 的分数都很低时， $\mathrm{softmax}$ 强制总分为 1 导致输出无意义"的问题，相当于给每个 head 一个"空槽位"（null position）。

**详细解释**：
标准 $\mathrm{softmax}$ attention 中，∑_j $\mathrm{softmax}$(score_j) = 1。如果 query 对当前 top-512 个压缩 KV 的分数都很低（例如被 indexer 选错了位置）， $\mathrm{softmax}$ 仍然强制把概率全部分配给这些不相关的 KV——输出是一堆噪声的加权和。

attn_sink 的机制（ $sparse_attn_kernel$ L346）：
```python
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
```
最终输出 = (V 的加权和) / (∑exp(score_j) + exp(attn_sink))。语义上：attn_sink 给了一个"虚拟 token"，如果所有真实 KV 的分数都低，attn_sink 占主导，输出趋近于 0（而非噪声）。如果真实 KV 有高分，attn_sink 被淹没在分母中，影响可忽略。

V3 时代引入的设计，V4 完整保留。每个 head 一个独立标量（64 个 head = 64 个 float32 参数）。初始化为 0，让训练早期退化为标准 attention。

**面试要点**：attn_sink 本质上是一个"attention 的截断机制"——用可学习的空槽位分数替代"强制全部注意力到不相关的 KV"。这不是 V4 特有，而是从 V3 继承的成熟设计。

**延伸阅读**：主报告 CH3.6 源码片段 4；`inference/model.py` L736 $self.attn_sink$ 。


### Q8.29 V4-Flash 的 grouped low-rank O 投影（o_groups=8）是什么？

**简短回答**：grouped low-rank O 投影把 64 个 attention head 分成 8 组（每组 8 头），每组共享一个低秩投影矩阵，先把 head 输出从 4096 维压到 1024 维再升回 4096 维。这比直接做 64 头全连接省 ~4x 参数和计算量。

**详细解释**：
标准 O 投影：64 heads × 512 dim = 32768 维直接投影到 d=4096，矩阵大小 32768×4096 ≈ 134M 参数。

Grouped low-rank 投影（o_groups=8, o_lora_rank=1024）：
1. 64 头分成 8 组，每组 8 头：8 × 512 = 4096 维
2. 每组用 $wo_a$ 投影到 1024 维：8 组 × 4096×1024 ≈ 33.6M 参数
3. 所有组拼接后用 $wo_b$ 投影到 d=4096：8×1024=8192 × 4096 ≈ 33.6M 参数
4. 总参数 ≈ 67M（比直接投影的 134M 省一半）

源码（`Attention.forward` L537-L542）：
```python
o = o.view(bsz, seqlen, self.n_local_groups, -1)  # 分组
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)        # 组内降维
x = self.wo_b(o.flatten(2))                        # 全局升维
```

这是从 V3 MLA 体系继承的设计，和低秩 Q 投影（q_lora_rank=1024）对应。

**面试要点**：要理解"grouped"和"low-rank"是两个独立维度——grouped 是指 head 先分组再变换（减少全局连接），low-rank 是指用中间 bottleneck 降低矩阵的参数量。两者叠加实现 4x 节省。

**延伸阅读**：主报告 CH3.4；`inference/model.py` L729-L742。


### Q8.30 FP4 量化的 scale 为什么用 E8M0（power-of-2）而不是 FP32？

**简短回答**：E8M0 把 scale 强制为 2 的整数次幂（如 2^3=8, 2^4=16），使得反量化时的除法 $w = w_fp4 * scale$ 退化为位运算 shift，比 FP32 scale 的浮点除法快约 10 倍。代价是 scale 精度从 "接近连续" 退化到 "8 位指数离散"，但 block=32 的细粒度量化补偿了这一精度损失。

**详细解释**：
反量化公式： $w_bf16 = w_fp4 * scale_e8m0$ 。当 scale 是 2 的整数次幂时：
- $w_fp4 * 2^k = w_fp4 << k$ （整数左移 k 位）
- 在 GPU 上，bit shift 比浮点乘法少 ~3x 延迟、少 ~5x 功耗

E8M0 格式本身：8 位 = 1 位符号 + 7 位指数偏置（实际指数值 = e8m0_value - 127）。例如 scale=2^3 存储为 127+3=130=0x82。可表示的 scale 范围：2^-127 到 2^127。

V4 的 $fast_round_scale$ （`kernel.py` L436-L438）实现：
```python
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))
```
本质是：(1) 用对数化找到 ceil(log2(amax/fp4_max))，(2) 用 2 的幂取整。全程位运算。

FP8 量化用 FP32 scale（不需要 E8M0），因为 FP8 的 block=128 更大、精度更高，浮点除法的开销被 GEMM 主导的计算量隐藏了。

**面试要点**：E8M0 的 trade-off 是"scale 精度 vs 计算速度"——block=32 时 scale 精度由 block 细粒度补偿，所以可以接受 E8M0。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` L129-L177 $fp4_quant_kernel$ 。


### Q8.31 V4-Flash 的训练 pipeline 细节是怎样的？lr schedule / warmup / gradient clipping 怎么配？

**简短回答**：V4-Flash 训练用 cosine decay with warmup（热身 2000 步后 cos 衰减到 0），AdamW 部分 lr≈3e-4、Muon 部分 lr≈0.02（RMS rescale 后绝对值不同但有效 lr 相近），gradient clipping max_norm=1.0（沿用 V3），32T tokens 估算在 ~2048 GPU 规模下约 60 天量级完成。

**详细解释**：
V4-Flash 的训练 pipeline 有 4 个关键超参，各自承担不同的稳定性角色：

**(1) Learning rate schedule: cosine decay with warmup**。标准做法：前 2000 步 lr 从 0 线性增长到 peak（warmup 阶段），之后按余弦函数衰减到 0（cosine decay）。Warmup 在 V4 中尤其关键——训练初期 FP4 量化 + Muon 矩阵更新组合的前几个 batch 数值非常不稳定：FP4 的反量化噪声叠加 Muon 的 Newton-Schulz 正交化（需要矩阵满秩才能收敛），如果 lr 一开始就很大，梯度方向和量级都会剧烈震荡。Warmup 给了 Muon 的 RMS rescale 因子时间适应真实的梯度规模。

**(2) Learning rate 量级**：V4 使用双优化器分工——Embedding/Head/Norm/gating 等小参数用 **AdamW（lr≈3e-4）**，大矩阵（Attention Q/K/V/O、MoE expert 的 w1/w2/w3、mHC comb）用 **Muon（lr≈0.02）**。表面看 Muon 的 lr 是 AdamW 的 ~67 倍，但实际上这是 Muon 的 RMS rescale 机制造成的——Muon 每步对更新方向做正交化后，会乘以 `max(当前参数RMS, 1e-3)` 把更新量 rescale 到与参数规模匹配的水平。经过 rescale 后的"有效 lr"与 AdamW 在同一个数量级。V3.2 全用 AdamW（无 Muon），其 lr 统一约 3e-4，所以 V4 的"表面 lr"虽然变了但"有效更新量"基本不变。

**(3) Gradient clipping: max_norm=1.0**。V4 沿用 V3 的 gradient clipping 设置——对全局梯度的 L2 norm 做 $clip_grad_norm_(max_norm=1.0)$ 。这个值在 FP4 + Muon 场景下尤其重要：Muon 的 Newton-Schulz 迭代（每次 10 步）在遇到极端梯度时可能发散（矩阵接近奇异时正交化不稳定），clipping 提前把梯度的范数限制在 1.0 以内，减少发散概率。实践中 V4 团队可能还加了 per-parameter 的额外 clip（如 `clamp(grad, -10, 10)`），但论文未明确。

**(4) Batch size 和训练规模**：V4-Flash 在 32T tokens 上预训练。基于公开信息估算——V4-Flash 有 284B 总参数 / 13B 激活参数（MoE sparse），每步 forward + backward 的 FLOPs ≈ 6 × 13B × tokens_per_step。假设集群规模 ~2048 GPU（256 个 8×H100 节点），考虑 MoE 稀疏性（top-6/256 ≈ 2.3% expert 利用率但 shared expert 始终激活）和通信开销（all-to-all expert dispatch），有效利用率 ~40-50%，总训练时间约 60 天量级。这个数字是粗略估算，V4 技术报告未公开具体 GPU·天。

**与 V3.2 训练 pipeline 的差异总结**：

| 维度 | V3.2 | V4-Flash |
|------|------|---------|
| 优化器 | 全 AdamW | AdamW（小参数）+ Muon（大矩阵） |
| lr（表面） | ~3e-4 统一 | AdamW~3e-4 / Muon~0.02 |
| lr（有效） | ~3e-4 | 与 V3 同量级（Muon 经 RMS rescale） |
| Gradient clipping | max_norm=1.0 | 同 V3，max_norm=1.0 |
| 训练 tokens | 14.8T | 32T |
| 激活参数 | 37B | 13B |
| 流水线并行 | DualPipe（通信-计算 overlap） | 论文未明确；推理用 TP+EP |

关于 DualPipe：V3 使用 DualPipe 做流水线并行优化（1F1B 交错调度），V4 技术报告未明确说训练中是否沿用 DualPipe，但推理代码中使用了张量并行（TP，按 Q head 切分）+ 专家并行（EP，按 expert 切分 256 experts）。训练时的 DualPipe 细节待 DeepSeek 进一步公开。

💡 **面试要点**：能说出 V4 训练 pipeline 的 4 个关键参数（cosine warmup、双 lr 量级、max_norm=1.0、~2048 GPU×60d）和各自作用即可，不需要精确到每一个超参数字。

**延伸阅读**：主报告 CH6.1-6.4（Muon）、CH9.2（训练规模）；V4 技术报告 §4.2。


### Q8.32 V4-Flash 的 32T tokens 训练数据是怎么构成的？各领域占比多少？

**简短回答**：V4 技术报告未公开详细领域占比，参照 V3 paper 分布：网页文本 ~60-70%、代码 ~15-20%、数学/科学 ~5-10%、书籍 ~3-5%、多语言 ~3-5%。V4 相比 V3 增加了代码和数学比例，因为长上下文 reasoning 需要更强的代码理解和多步推理基础。

**详细解释**：
DeepSeek 系列模型的数据配比一直是"半公开"状态——V3 paper 给出了大致分布但未给精确数字，V4 技术报告则完全未提及。以下基于 V3 paper 和业界实践推断：

**预训练数据组成（32T tokens, 推断）**：

| 领域 | 估算占比 | 说明 |
|------|---------|------|
| 网页文本（Common Crawl + 高质量网页） | 60-70% | 中英文为主，覆盖百科、新闻、论坛、博客等 |
| 代码（GitHub + Stack Overflow + 竞赛平台） | 15-20% | 比 V3 的 ~12-15% 有所提升 |
| 数学 + 科学论文 | 5-10% | arXiv, PubMed, 教材, 竞赛题 |
| 书籍 + 百科全书 | 3-5% | 长文本，训练长上下文理解 |
| 多语言翻译语料 | 3-5% | 覆盖 20+ 语言 |

**V4 vs V3 的数据策略变化**：
V4 增加了代码和数学比例，理由有二：(1) V4-Flash 的目标是长上下文下的强 reasoning——在 1M 上下文中理解复杂代码库、多步数学推导，需要更强的代码/数学预训练基础；(2) "过度训练"（见下）场景下，高质量数据（代码/数学）的边际收益高于普通网页文本。

**数据处理管线**：
1. **去重**：MinHash（文档级模糊去重）+ 精确匹配（段落级去重），两轮去重将重复率控制在 <2%
2. **质量过滤**：训练质量分类器对每篇文档打分——教育水平（是否含实质性内容）、毒性（有害内容过滤）、重复度（模板化内容识别）。低于阈值的文档直接丢弃
3. **文档拼接（packing）**：将短文档拼接以最小化 padding 浪费，用特殊的 document separator token 分隔
4. **Tokenize**：使用 129,280 词表的 BPE tokenizer（与 V3 保持一致），支持 Fill-in-Middle（FIM）格式

**为什么 32T tokens 很重要——Chinchilla 视角**：

Chinchilla 缩放定律（Hoffmann et al. 2022）的结论：对给定计算预算，最优训练 tokens 数 ≈ 20 × 参数量。V4-Flash 有 284B 总参数 / 13B 激活参数——按 Chinchilla 最优，需要的 tokens 仅为 5.7T（用总参）或 260B（用激活参）。32T tokens 远超这两个数字，意味着 V4-Flash 是刻意"过度训练"（overtraining）——用远多于"最优"的数据量训练，以换取更强的推理能力。

"过度训练"对 reasoning 为什么有效？推理能力不像知识那样依赖参数记忆——它更依赖模型在大量不同推理路径上的"练习"。更多 tokens = 更多样化的推理模式被训练到 = 更鲁棒的推理能力。这解释了为什么 V4-Flash（13B 激活，32T tokens）在 AIME 等推理 benchmark 上能接近 GPT-5.2——不是靠参数多，而是靠"练得多"。

💡 **面试要点**：Chinchilla 最优 vs 过度训练——能说出"32T 远大于 5.7T（Chinchilla 最优），过度训练提升 reasoning 能力"是核心得分点。

**延伸阅读**：主报告 CH7.3（后训练数据）、CH9.2（训练规模）；V3 论文 §2.1。


### Q8.33 V4-Flash 后训练的两阶段 Specialist + OPD 具体怎么训练的？

**简短回答**：阶段 1：每个领域（Code/Math/Reasoning/Agent）独立训练一个 Specialist——用该领域的 SFT 数据做监督微调，再用 GRPO 强化学习（各自领域的 reward 函数）。阶段 2：用 OPD（On-Policy Distillation）将多个 Specialist 的 token-level logit 分布蒸馏回一个 Unified Model。两阶段配合实现"分而治之 → 无损融合"。

**详细解释**：
这是对 Q8.16-Q8.18 和 Q8.20 的深层补充——那 4 个 Q 分别讲了 GRPO、OPD、MTP 和三阶段，但没有讲清楚"Specialist → Unified"这一完整管线的具体机制。

**阶段 1：每个领域训练一个 Specialist**

V4 为不同能力域分别训练专家模型（13B 激活参数，从预训练基座初始化），各 Specialist 独立训练互不干扰：

| Specialist | SFT 数据 | GRPO 奖励函数 | 训练目标 |
|-----------|---------|-------------|---------|
| Code | 代码补全、Bug 修复、算法实现 | Test case pass rate（单元测试通过率） | 生成能通过测试的正确代码 |
| Math | 数学证明、计算题、竞赛题 | 答案正确性 + 步骤完整性（中间推导是否合理） | 多步推理正确且步骤可验证 |
| Reasoning | 逻辑推理、常识推断、长链推理 | 多模型投票一致性（多个 judge 模型对推理质量的评分均值） | 推理链连贯且结论正确 |
| Agent/Tool | 工具调用、API 交互、多步任务 | 任务完成率（是否成功调用工具并返回正确结果） | 准确使用工具完成复杂任务 |

**为什么需要 Specialist？** 单个 13B 激活参数的模型 capacity 有限——如果同时训练代码、数学、推理、Agent 四个领域，Gradient 之间会互相干扰（灾难性遗忘）。分开训练让每个 Specialist 在各自领域做到极致，不受其他领域梯度污染。代价是需要后续的 OPD 融合。

**阶段 2：OPD 将多个 Specialist 蒸馏回 Unified Model**

这是 V4 最核心的后训练创新。OPD 与标准知识蒸馏有本质区别：

- **输入**：prompt 从各 Specialist 的 SFT 数据中采样（确保覆盖各领域分布）
- **教师输出**：对于 prompt x，Specialist 给出每个 token 位置上的完整 logit 分布 p_expert(·|x, y_{<t})（不仅是 hard label）
- **学生（Unified Model）** 从预训练基座初始化，同时接收所有领域的 OPD 信号
- **损失函数**： $L = KL(π_teacher || π_student) + 0.1 \times  L_LM$ 。KL 项让学生模仿专家的 token-level 决策分布（不仅是最终答案），0.1 权重的 LM loss 防止学生"只会模仿而不会自己生成"
- **On-policy 关键**：学生模型自己生成回答 y ~ π_student(·|x)，然后教师对这些"学生自己会说的话"打分。这与传统蒸馏"教师对固定数据集打分"的本质区别在于——训练数据的分布 = 学生模型部署时的真实分布，避免了分布漂移（distribution shift）

**Unified Model 的效果**：V4 技术报告显示，Unified Model 在各领域 benchmark 上与对应 Specialist 的性能保持率 > 99%——也就是说"无损蒸馏"基本实现。Unified Model 还额外获得了跨领域泛化能力（例如用代码推理解决数学问题），这是单一 Specialist 不具备的。

**为什么 OPD 优于传统蒸馏**：传统 off-policy 蒸馏用固定数据集（教师已经标好 logit），学生学习的分布是"教师擅长回答的数据分布"，但部署时用户 query 的分布可能完全不同。OPD 让学生"自己探索 → 老师纠正"，训练数据分布始终与部署分布一致（on-policy），在长尾 query 上泛化更好。

💡 **面试要点**：区分 Specialist（单领域强，GRPO 各自 reward）和 Unified Model（多领域统一，OPD 蒸馏融合），以及 OPD 的 "on-policy" 价值——训练分布 = 部署分布，消除分布漂移。

**延伸阅读**：主报告 CH7.3；Q8.16（GRPO）、Q8.17（OPD）、Q8.20（后训练三阶段）。


## CH 9. 源码系统 -- QA 问答

### Q9.1 V4-Flash 仓库的目录结构是怎样的？

**简短回答**：V4-Flash 仓库（DeepSeek-AI/DeepSeek-V4-Flash）是 inference-only 仓库，核心目录：`inference/`（模型定义 + kernel）、`encoding/`（tokenizer/编码工具）、根目录（配置文件）。模型定义在 `inference/model.py`（827 行，13 个类），高性能 kernel 在 `inference/kernel.py`（536 行）。

**详细解释**：
完整目录结构：

```
DeepSeek-V4-Flash/
├── config.json                    # 顶层模型超参（48 keys）
├── inference/
│   ├── model.py                   # 827 行，完整模型定义
│   ├── kernel.py                  # 536 行，TileLang kernel 实现
│   ├── generate.py                # 155 行，采样 + 生成循环
│   ├── convert.py                 # 168 行，权重格式转换
│   ├── config.json                # 推理专用配置
│   └── README.md
├── encoding/
│   ├── encoding_dsv4.py           # 744 行，messages ↔ string
│   └── test_encoding_dsv4.py
├── tokenizer.json                 # 6.4 MB 词表
├── tokenizer_config.json
├── generation_config.json
├── model.safetensors.index.json   # 5.4 MB 权重索引
└── README.md
```

关键文件统计：

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `inference/model.py` | 827 | 13 个类：ModelArgs, ParallelEmbedding, Compressor, Indexer, Attention, Gate, Expert, MoE, Block, ParallelHead, MTPBlock, Transformer + 辅助类 |
| `inference/kernel.py` | 536 | 5 大 kernel：sparse_attn, hc_split_sinkhorn, act_quant(FP8), fp4_quant, GEMM |
| `inference/generate.py` | 155 | Gumbel-max 采样, autoregressive 循环 |

**面试要点**：要知道 "inference-only"——V4 仓库不包含训练代码、optimizer、数据加载。Muon 的实现也不在仓库中（只有论文 Algorithm 1 描述）。

**延伸阅读**：主报告 CH8.1。


### Q9.2 inference/model.py 中 13 个类的组织和职责是什么？

**简短回答**：`model.py` 的 13 个类按"基础设施 → 注意力 → MoE → 整合"四层组织。基础设施层包括 ModelArgs、ParallelEmbedding、 $\mathrm{RMSNorm}$ 和三种 Linear；注意力层包括 Compressor、Indexer、Attention；MoE 层包括 Gate、Expert、MoE；整合层包括 Block、ParallelHead、MTPBlock、Transformer。

**详细解释**：
类分级表（按文件行号顺序）：

| 层 | 类名 | 行号 | 职责 |
|----|------|------|------|
| 基础设施 | `ModelArgs` | L35 | 超参容器（dataclass） |
| 基础设施 | `ParallelEmbedding` | L83 | 词嵌入 + 张量并行 |
| 基础设施 | `Linear` / `ColumnParallelLinear` / `RowParallelLinear` | L123 / L155 / L166 | 三种并行线性层 |
| 基础设施 | $\mathrm{RMSNorm}$ | L183 | Root Mean Square Normalization |
| 注意力 | `Compressor` | L279 | KV 时间维压缩（CSA/HCA 共享） |
| 注意力 | `Indexer` | L380 | CSA 专属的稀疏索引评分 |
| 注意力 | `Attention` | L436 | CSA/HCA 混合注意力整合 |
| MoE | `Gate` | L546 | MoE 路由（hash + score 双路径） |
| MoE | `Expert` | L587 | 单 expert 前向（ $\mathrm{SwiGLU}$ + FP4） |
| MoE | `MoE` | L609 | 256 expert 调度 + 合成 |
| 整合 | `Block` | L647 | 单层 Transformer（attn + MoE + mHC） |
| 整合 | `ParallelHead` | L703 | LM Head（输出 logits） |
| 整合 | `MTPBlock` | L738 | Multi-Token Prediction 块（继承 Block） |
| 整合 | `Transformer` | L769 | 顶层模型（embed → N×Block → head） |

数据流：`Transformer.forward` 是入口，内部调用 $ParallelEmbedding \to  43\times Block.forward(Attention + MoE) \to  \mathrm{RMSNorm} \to  ParallelHead$ 。`Block.forward` 串联 2 个 mHC 残差（attn 后、MoE 后）。

**面试要点**：能说出 `Compressor` 和 `Indexer` 的区别——Compressor 是共享的 KV 压缩器，Indexer 是 CSA 专属的稀疏选择器；两者在 $Attention.__init__$ 中按 $compress_ratio$ 实例化。

**延伸阅读**：主报告 CH8.1; `inference/model.py`。


### Q9.3 inference/kernel.py 的 5 大 kernel 分别是什么？

**简短回答**：`inference/kernel.py`（536 行）包含 5 个用 TileLang DSL 编写的高性能 GPU kernel：(1) $sparse_attn_kernel$——稀疏注意力；(2) $hc_split_sinkhorn_kernel$——mHC 的 Sinkhorn 投影；(3) $act_quant_kernel$——FP8 量化；(4) $fp4_quant_kernel$——FP4 量化；(5) $fp8_gemm$ / $fp4_gemm$——混合精度 GEMM。

**详细解释**：

| Kernel | 行号 | 功能 | 关键设计 |
|--------|------|------|---------|
| $act_quant_kernel$ | L36-L103 | Block-wise FP8 量化（block=128） | E4M3 格式，FP32 scale，支持 inplace 融合 |
| $fp4_quant_kernel$ | L129-L177 | Block-wise FP4 量化（block=32） | E2M1FN 格式，E8M0 scale（power-of-2） |
| $fp8_gemm$ | L257+ | FP8 矩阵乘法 | dequant + GEMM 融合 |
| $sparse_attn_kernel$ | L276-L368 | 稀疏注意力（top-k gather） | Online $\mathrm{softmax}$ + attn_sink + pipeline 双缓冲 |
| $hc_split_sinkhorn_kernel$ | L372-L428 | mHC 的 Sinkhorn-Knopp 投影 | 20 次行列归一化，pre/post/comb 拆分 |

这 5 个 kernel 全部用 `@tilelang.jit` 编译，支持符号维度（运行期确定 batch/seqlen/topk），编译期常量化 head 维度和 head_dim。TileLang 的核心优势是"在一个 DSL 里完成开发和优化"，不需要手写 CUDA C++。

**面试要点**：不要遗漏 $sparse_attn_kernel$ 中的 online $\mathrm{softmax}$ 实现——它和 $\mathrm{FlashAttention}$ 的机制相同（running max + rescale + sum），只是 gather 模式不同（按 topk 索引 gather 而非按连续位置）。

**延伸阅读**：主报告 CH3.6、CH3.7；`inference/kernel.py`。


### Q9.4 一个 token 从输入到输出的完整生命周期是怎样的？

**简短回答**：一个 token 在 V4-Flash 中经历 6 个阶段：encoding（字符串处理）→ tokenize（转为 token ID）→ embedding（转为 4096 维向量）→ 43×Block 前向（CSA/HCA + MoE + mHC）→ LM Head（转为 129280 维 logits）→ sample（Gumbel-max 选择下一 token）。每个阶段在仓库中有对应的代码入口。

**详细解释**：
完整 6 阶段流程（伪代码级）：

```
# 阶段 1: encoding
prompt = encode_messages(messages, thinking_mode="chat")
# encoding/encoding_dsv4.py L506

# 阶段 2: tokenize
input_ids = tokenizer.encode(prompt)        # shape: [T], dtype: int64

# 阶段 3: embedding
x = ParallelEmbedding(input_ids)            # shape: [T, 4096], dtype: bf16
# inference/model.py L83

# 阶段 4: 43×Block 前向
for layer_id in range(43):
    # 4a. mHC hc_pre: 4 通道 → 1 通道
    x, post, comb = block.hc_pre(x, ...)
    
    # 4b. Pre-Norm + Attention (CSA/HCA/滑窗)
    x = attn_norm(x)
    x = block.attn(x, start_pos)            # 按 compress_ratios[layer_id] 选分支
    
    # 4c. mHC hc_post: 1 通道 → 4 通道
    x = block.hc_post(x, residual, post, comb)
    
    # 4d. 残差保存
    residual = x
    
    # 4e. mHC hc_pre → Pre-Norm → MoE → hc_post
    x, post, comb = block.hc_pre(x, ...)
    x = ffn_norm(x)
    x = block.ffn(x, input_ids)             # Gate + Expert + 合成
    x = block.hc_post(x, residual, post, comb)

# 阶段 5: Norm + LM Head
x = RMSNorm(x)
logits = ParallelHead(x)                    # shape: [T, 129280]

# 阶段 6: sample
next_token = sample(logits[-1], temperature=1.0, top_p=1.0)
# inference/generate.py L19: Gumbel-max trick
```

**面试要点**：记住 6 阶段的核心形状变化：str → [T] int64 → [T, 4096] bf16 → [T, 4, 4096] (mHC 后) → [T, 129280] → [1] int64。

**延伸阅读**：主报告 CH2.3。


### Q9.5 Prefill 和 Decode 阶段在 V4-Flash 中有哪些差异？

**简短回答**：Prefill（首次前向）处理完整 prompt，并行计算所有 token；Decode（逐 token 生成）每次只处理 1 个新 token，复用 KV cache 避免重算。两者在 attention、Compressor、Indexer、all-to-all 通信等环节行为不同。

**详细解释**：
关键差异对照：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入形状 | [B, T_prompt, 4096] | [B, 1, 4096] |
| Compressor 行为 | 整段 `unflatten` + $\mathrm{softmax}$-pool（批量压缩） | 逐个 token 流入 $kv_state$ 缓冲区，凑齐 ratio 个才压缩 |
| 触发条件 | $start_pos == 0$ | $start_pos > 0$ |
| Indexer 评分 | 全 prompt token 同时评分 + causal mask | 单 token 评分（无 mask） |
| Top-k 数量 | min(512, end_pos//ratio) | min(512, end_pos//ratio) |
| KV cache 更新 | 逐 position 写入 | 循环缓冲区写入 |
| MoE all-to-all | 需要跨 rank 调度全部 prompt token | batch=1 时点对点拉取 expert 权重 |
| 计算瓶颈 | Compute-bound（大矩阵乘） | Memory-bound（权重加载） |

代码层面的分支（`Attention.forward` L518-L533）：
```python
if start_pos == 0:
    # Prefill: 整段 KV 写入 cache + sparse_attn(q, kv, ...)
    self.kv_cache[:bsz, :seqlen] = kv
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
else:
    # Decode: 单个 KV 更新 cache + sparse_attn(q, kv_cache, ...)
    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

**面试要点**：能说出 Decode 的核心优化——"复用 KV cache 使 Decode 的 attention 从 $O(T^2)$ 降到 $O(T)$"（在压缩后是 O(T/m + k)），这是自回归生成的效率基础。

**延伸阅读**：主报告 CH3.6；`inference/model.py` L518-L533。


### Q9.6 KV cache 在源码中是如何管理的？

**简短回答**：V4-Flash 的 KV cache 采用"异构三层"结构：滑窗 KV（127 个环形 buffer）+ 压缩 KV（T/ratio 个线性存储）+ Indexer KV（T/4 个独立缓存）。每层由 $Attention.__init__$ 分配，`Attention.forward` 更新，Compressor 复用 Attention 主 cache 的尾部。

**详细解释**：
Cache 分配（ $Attention.__init__$ L473）：
```python
kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
self.register_buffer("kv_cache", torch.zeros(max_batch_size, kv_cache_size, head_dim))
```

三层 KV cache 的结构：
1. **滑窗层（window KV）**：128 个原始 K/V 向量，循环写入（ $start_pos % win$ 为索引），永不被压缩
2. **压缩层（compressed KV）**：由 Compressor 每 ratio 个 token 输出 1 个压缩向量，写入 $kv_cache[window_size:]$ 区域
3. **Indexer 层**（仅 CSA 层）：额外的 $kv_cache$ （ $self.kv_cache$ ，大小 = max_seq_len/4 × index_head_dim），由 Indexer Compressor 写入

Decode 时的更新流程：
```python
# Step 1: 写入滑窗 cache
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)

# Step 2: Compressor 累积并可能输出压缩向量
if self.compress_ratio:
    self.compressor(x, start_pos)  # 内部维护 kv_state 缓冲区

# Step 3: Indexer Compressor 同步更新
if self.indexer is not None:
    self.indexer.compressor.kv_cache = self.kv_cache[:, win:]
```

关键设计：Compressor 的 $kv_cache$ 直接指向 Attention 主 cache 的压缩段——"写时共享"，不是 copy。

**面试要点**：要能区分"三层 cache"各自的职责——滑窗=局部无损，压缩=长程压缩，Indexer=稀疏选择。

**延伸阅读**：主报告 CH3.6；`inference/model.py` L466-L475, L508-L533。


### Q9.7 mHC 在源码中的实现是怎样的？（hc_pre / hc_post / hc_split_sinkhorn）

**简短回答**：mHC 源码由三部分组成——$Block.hc_pre$ （4 通道 → 1 通道 reduce）、 $Block.hc_post$ （1 通道 → 4 通道 expand）、以及 $hc_split_sinkhorn_kernel$ （TileLang kernel，把 24 维 mixes 拆分为 pre/post/comb 三个权重并做 Sinkhorn 投影）。

**详细解释**：
三部分源码对照：

**1. hc_pre（reduce）** —— `model.py` L1832-L1840：
```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    x = x.flatten(2).float()                          # [B,S,4,4096] → [B,S,16384]
    rsqrt = torch.rsqrt(x.square().mean(-1) + eps)    # RMSNorm 变体
    mixes = F.linear(x, hc_fn) * rsqrt                # [B,S,16384] → [B,S,24]
    pre, post, comb = hc_split_sinkhorn(mixes, ...)   # 拆分 + Sinkhorn
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)  # 4 通道加权 sum → 1 通道
    return y, post, comb
```

**2. hc_post（expand）** —— `model.py` L1842-L1845：
```python
def hc_post(self, x, residual, post, comb):
    # x: [B,S,d], residual: [B,S,4,d], post: [B,S,4], comb: [B,S,4,4]
    y = post.unsqueeze(-1) * x.unsqueeze(-2)          # f(x) * post（逐通道缩放）
      + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)  # comb @ residual
    return y  # [B,S,4,d]
```

**3. hc_split_sinkhorn_kernel** —— `kernel.py` L372-L428：
```python
# pre = sigmoid(mixes[0:4] * scale_pre + base_pre) + eps
# post = 2 * sigmoid(mixes[4:8] * scale_post + base_post)
# comb = softmax(mixes[8:24] * scale_comb + base_comb)
# for _ in range(20):  # Sinkhorn 迭代
#     comb = comb / comb.sum(-1)  # 行归一
#     comb = comb / comb.sum(-2)  # 列归一
```

关键细节： $post = 2 * \mathrm{sigmoid}(...)$ 而非 $\mathrm{sigmoid}(...)$ ，让 post 的中心在 1.0，初始化时 mHC 退化为普通残差，降低冷启动难度。

**面试要点**：能说清楚三个权重（pre/post/comb）的维度、作用、约束——pre 和 post 是 4 维向量（通过 $\mathrm{sigmoid}$ 约束），comb 是 4×4 双随机矩阵（通过 Sinkhorn 约束）。

**延伸阅读**：主报告 CH5.5；`inference/model.py` L1832-L1858；`inference/kernel.py` L372-L428。


### Q9.8 MoE 的 dispatch 在源码中是如何实现的？

**简短回答**：V4-Flash 的 MoE dispatch（专家调度）在 `MoE.forward` 中用伪代码 $for expert in range(n_experts): mask = (indices == expert); y[mask] += expert(x[mask])$ 实现——这是一个朴素的 Python 循环 + `torch.where`，而非 `torch.scatter` 或自定义 CUDA kernel。

**详细解释**：
`MoE.forward` 核心逻辑（`model.py` L609-L646）：

```python
def forward(self, x, input_ids):
    weights, indices = self.gate(x, input_ids)   # [T, 6], [T, 6]
    y = torch.zeros_like(x, dtype=torch.float32)  # fp32 累加
    
    # 遍历本 rank 持有的局部 experts
    for i in range(self.experts_start_idx, self.experts_start_idx + self.n_local_experts):
        # 找到该 expert 被分配到的 token
        expert_mask = (indices == i)
        token_idx, top_idx = torch.where(expert_mask)
        if token_idx.numel() == 0:
            continue
        # 取出对应 token 的 hidden state 和 weight
        x_expert = x[token_idx]
        w_expert = weights[token_idx, top_idx, None]  # [N, 1]
        # 逐 expert 前向
        expert_out = self.routed_experts[i](x_expert, w_expert)
        # scatter add 回输出
        y[token_idx] += expert_out
    
    if world_size > 1:
        dist.all_reduce(y)              # 跨 rank 汇总
    y += self.shared_experts(x)         # 永远加 shared
    return y
```

关键观察：
- V4-Flash 是 **inference-only** 仓库，dispatch 用 Python 循环（够用）
- 训练时用 DualPipe 的 MegaMoE2 kernel（通信-计算 overlap），不在本仓库
- `bincount(indices)` 统计专家命中次数——训练时用于 aux-loss-free bias 更新，推理时仅 monitoring

**面试要点**：区分"推理时 dispatch"和"训练时 dispatch"——推理时 batch 小、Python 循环 OK；训练时 batch 大、需要 CUDA kernel + all-to-all 通信。

**延伸阅读**：主报告 CH4.4、CH4.5；`inference/model.py` L609-L646。


### Q9.9 V4-Flash 1M 上下文推理的显存账是怎样的？（具体数字）

**简短回答**：V4-Flash 在 1M 上下文下的推理显存约 17 GB（BF16 参考），由 KV cache（~4 GB）、routed expert 权重（FP4, ~3.2 GB）、shared expert + attention + embedding（~3 GB）、mHC + 其他开销（~2 GB）、激活/临时 buffer（~5 GB）组成。

**详细解释**：
逐项显存账（单 batch, BF16 参考）：

| 组件 | 精度 | 计算 | 大小 |
|------|------|------|------|
| KV cache（CSA 21 层） | 混合（BF16+FP8） | 21 × 262,272 × 576 bytes | ~3.17 GB |
| KV cache（HCA 20 层） | 混合 | 20 × 8,320 × 576 bytes | ~0.096 GB |
| KV cache（滑窗 2 层） | BF16 | 2 × 128 × 512 × 2 bytes | ~0.25 MB |
| 压缩 KV cache (Indexer) | FP8 | 21 × 262,272 × 128 bytes | ~0.70 GB |
| Routed experts (256) 权重 | FP4 | 6.44B × 0.5 bytes | ~3.22 GB |
| Shared expert 权重 | FP8 | 25M × 1 byte | ~25 MB |
| Attention (Q/K/V/O/Compressor/Indexer) | FP8/BF16 | 估算 | ~2 GB |
| Embedding (129280 × 4096) | BF16 | 529M × 2 bytes | ~1.06 GB |
| LM Head | BF16 | 同上 | ~1.06 GB |
| mHC 权重（86 个残差） | FP32 | 86 × 24 × 4096 × 4 bytes | ~34 MB |
| $\mathrm{RMSNorm}$ （86+ 个） | FP32 | 86 × 4096 × 4 bytes | ~1.4 MB |
| 激活临时 buffer | BF16/FP32 | 估算 | ~5 GB |
| **合计** | | | **~17.3 GB** |

KV cache 混合精度的关键：RoPE 维度（64 dim）× BF16 = 128 bytes，非 RoPE 维度（448 dim）× FP8 = 448 bytes，总计每向量 576 bytes（vs 纯 BF16 的 1024 bytes）。

**面试要点**：能快速估算 KV cache（约 4 GB）和 expert 权重（约 3.2 GB）两部分最大头的数字，误差在 ±20% 内可接受。

**延伸阅读**：主报告 CH3.5。


### Q9.10 V4-Flash 的推理速度是多少？（tokens/s 量级）

**简短回答**：V4-Flash 在短上下文（<8K）下 decode 速度约 30-50 tokens/s（单 8×H100 节点），在 1M 上下文下约 5-15 tokens/s（FLOPs 增长来自稀疏 attention 的 top-k gather）。具体数字取决于硬件、batch size 和序列长度。

**详细解释**：
影响推理速度的因素（按重要性排序）：

1. **序列长度（最敏感）**：短序列下 attention 只涉及 window=128 个 KV（几乎 dense），快。1M 下需要 gather top-512 个压缩 KV + 128 个滑窗 KV，延迟主要由 $sparse_attn_kernel$ 的随机 gather 决定。

2. **Batch size**：batch=1 decode 是 memory-bound（权重加载主导），加大 batch 变为 compute-bound。V4-Flash 设计目标是单 batch 在 8×H100/2×H200 上运行。

3. **MoE top-6 选择**：每次 decode 激活 7 个 expert（6 routed + 1 shared），总计算量约 7 × 25M × 2 FLOPs（ $\mathrm{SwiGLU}$ matmul）≈ 350M FLOPs/token。在 compute-bound 场景下可忽略，在 memory-bound 下由权重加载带宽主导。

4. **FP4 反量化**：每次 decode 反量化 6 个 routed expert 的权重（约 150 MB），E8M0 位运算几乎无延迟。

V4 技术报告未公开具体 tokens/s 数字，但根据 V3 的 12 tokens/s（H800）和 V4-Flash 的 ~10% FLOPs，短上下文推理可达 30-50 tokens/s。1M 上下文下因 sparse gather 瓶颈降至 5-15 tokens/s。

**面试要点**：区分"理论 FLOPs"和"实测 tokens/s"——memory-bound 场景下（特别是 FP4 权重加载和 sparse gather）实际的 wall-clock 时间由带宽而非 FLOPs 决定。

**延伸阅读**：主报告 CH3.5、CH9.2。


### Q9.11 V4-Flash 的训练需要多少 GPU 天？

**简短回答**：V4 技术报告未公开 V4-Flash 的具体 GPU 天数字。根据公开信息推断：284B 总参 / 13B 激活 / 32T tokens，单次训练估计需要 3000-5000 H800 GPU·天（或等效的 1500-2500 H100 GPU·天）。这个数字是粗略估计，V4 论文未确认。

**详细解释**：
估算依据（基于已知公开信息）：

- 训练数据：32T tokens
- 激活参数：13B
- 训练 FLOPs ≈ 6 × 13B × 32T ≈ 2.5 × 10^24 FLOPs（C = 6ND 规则）
- H800 FP8 dense: ~989 TFLOPS（考虑 MoE 稀疏性和通信开销后有效利用率 ~40-50%）
- 有效 FLOPs/GPU·天 ≈ 989 × 10^12 × 86400 × 0.45 ≈ 3.84 × 10^19
- 预估 GPU·天 ≈ 2.5 × 10^24 / 3.84 × 10^19 ≈ 65,000 GPU·天（单卡）
- 考虑集群规模（千卡级），需要数月训练时间

不确定因素（导致估算误差大）：
- 实际的 MoE 稀疏利用率（top-6 激活意味着大量 expert 参数未被使用）
- 通信开销（DualPipe 的 overlap 效率）
- FP8/FP4 训练的有效吞吐量
- mHC 和 Muon 的额外计算开销

**面试要点**：要诚实承认"V4 论文未公开"这一事实，能给出合理的 C=6ND 估算并说明不确定因素。不要编造一个精确数字。

**延伸阅读**：主报告 CH9.2；V4 技术报告 §4.2。


### Q9.12 V4-Flash 的分布式并行策略（TP/PP/EP）是怎样的？

**简短回答**：V4-Flash 推理环境用 Tensor Parallelism（TP，按 Q head/groups 切分）和 Expert Parallelism（EP，按 expert 切分 256 experts）。训练时还叠加 Pipeline Parallelism（PP）和 Data Parallelism（DP）。V4 的 DualPipe 设计让通信和计算 overlap，把 pipeline bubble 压到最低。

**详细解释**：
三种并行在 V4 中的体现：

**TP（Tensor Parallelism）**——源码直接体现：
- `ColumnParallelLinear`（`model.py` L155）：按输出维度切分，每 rank 持有部分列
- `RowParallelLinear`（`model.py` L166）：按输入维度切分，输出时 all-reduce
- Attention Q head： $n_local_heads = n_heads // world_size$
- O 投影分组： $n_local_groups = o_groups // world_size$
- Indexer head： $n_local_heads = index_n_heads // world_size$

**EP（Expert Parallelism）**——源码直接体现：
- $MoE.__init__$ L609： $assert n_routed_experts % world_size == 0$
- 每 rank 持有 $256 / world_size$ 个 routed expert
- All-to-all 通信调度 expert 输入输出
- 训练时用 MegaMoE2 kernel（通信-计算 fuse）

**PP（Pipeline Parallelism）**——训练时专用（推理不切 pipeline）：
- 43 层切分为若干 pipeline stages
- DualPipe 1F1B：前向和反向交错，隐藏通信延迟

**面试要点**：能区分训练和推理的并行策略——推理通常只用 TP+EP（DP 可选），训练还要加 PP。EP 的 all-to-all 通信是 MoE 模型特有的瓶颈。

**延伸阅读**：主报告 CH4.4、CH4.5；V4 技术报告 §3.1-3.4。


### Q9.13 TileLang kernel 编写有什么特点？和手写 CUDA 比如何？

**简短回答**：TileLang 是一种 Python-embedded DSL，用 $@T.prim_func$ 装饰器定义 kernel，编译期通过 Z3 SMT solver 做形式化整数分析优化，输出 CUDA/PTX 代码。相比手写 CUDA，开发效率高（一行 Python ≈ 十行 CUDA），但性能不输手写（V4 声称在保守默认设置下与手写 CUDA 性能持平）。

**详细解释**：
TileLang kernel 示例（ $sparse_attn_kernel$ 简化版）：
```python
@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    b, m, n, topk = T.symbolic("b"), T.symbolic("m"), T.symbolic("n"), T.symbolic("topk")
    
    @T.prim_func
    def kernel_(q, kv, o, attn_sink, topk_idxs):
        with T.Kernel(m, b, threads=256) as (bx, by):
            # shared memory 分配
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            acc_o = T.alloc_fragment((h, d), FP32)
            # online softmax buffer
            scores_max = T.alloc_fragment(h, FP32)
            T.fill(scores_max, -T.infinity(FP32))
            
            for t in T.Pipelined(num_blocks, num_stages=2):
                # gather KV by index
                idxs[i] = topk_idxs[by, bx, t*block + i]
                kv_shared[i, j] = kv[by, idxs[i], j]
                # QK^T GEMM
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True)
                # online softmax
                ...
            # write output
            T.copy(acc_o, o[by, bx])
    return kernel_
```

TileLang 的三大优势：
1. **开发效率**：`T.gemm` 一行替代 100+ 行 CUDA tiling 代码
2. **形式化验证**：Z3 SMT solver 在编译时验证边界条件、内存访问安全
3. **bitwise repro**：与手写 CUDA 可做到逐位一致的输出（通过 layout annotation 控制累积顺序）

**面试要点**：TileLang 不是替代 CUDA，而是"生成 CUDA"的 DSL——它输出的是 CUDA 代码，只是写代码的方式不同。

**延伸阅读**：主报告 CH3.6、CH3.7；V4 技术报告 §3.2。


### Q9.14 Speculative decoding 在 V4-Flash 中是怎么实现的？

**简短回答**：V4-Flash 仓库未自带 speculative decoding 的专用实现，但提供了两个基础组件：(1) MTP 层（`MTPBlock`，`model.py` L738）可当作"草稿模型"（draft model）预测下 1 个 token；(2) 主模型的标准 generate 循环（`generate.py`）作为"验证模型"。二者组合即可实现 1.2-1.5x 的投机解码加速。

**详细解释**：
投机解码的基本流程（V4-Flash 可用组件）：

1. 主模型 generate 1 个 token（如 `generate.py` 的 `sample` + `Transformer.forward`）
2. 用 MTPBlock 预测 M 个草稿 token（MTPBlock 本身 1 层，预测 1 个额外 token）
3. 主模型一次验证 M 个草稿 token（并行前向检查 logits 是否匹配）
4. 接受匹配的 token，拒绝不匹配的，从第一个不匹配处重新生成

V4-Flash 的 MTPBlock 特点：
- 只预测下 1 个 token（ $num_nextn_predict_layers=1$ ），而不是 V3 的 2 层
- 接受率取决于任务——简单补全高（>80%），创造性任务低（<50%）
- 每层 MTPBlock 有一次 Attention + MoE 计算，比主模型 43 层的开销小得多

投机解码的有效加速比 = 1 / (1 - acceptance_rate × M)，其中 M=1（V4-Flash 的 MTP 层数）。理论上 1.2-1.5x 加速是合理的。

**面试要点**：区分"MTP 训练"和"投机解码"——MTP 在训练时用于提升数据效率，在推理时（可选）用于投机解码加速。两者共享同一个 MTPBlock，但目的不同。

**延伸阅读**：主报告 CH7.4；`inference/model.py` L738-L767 `MTPBlock`；`inference/generate.py`。


### Q9.15 V4-Flash 模型权重文件的结构是怎样的？

**简短回答**：V4-Flash 权重以 safetensors 格式分发，通过 `model.safetensors.index.json`（5.4 MB）索引。推理时 `convert.py`（168 行）将 HF 格式的权重转换为 V4 内部格式（合并低秩矩阵、拆分专家并行列、FP4 pack 等）。

**详细解释**：
权重文件结构：

- **safetensors 分片**：大模型通常分多个 .safetensors 文件（如 model-00001-of-00008.safetensors），每个 ~5 GB
- **index.json**：映射参数名 → 文件偏移量，包含所有参数的 dtype 和 shape
- **权重内容**（按模块分组）：
  - $model.embed_tokens.weight$ ：129280 × 4096，BF16
  - $model.layers.{i}.self_attn.*$ ：Q/K/V/O 投影 + Compressor + Indexer
  - `model.layers.{i}.mlp.gate.*`：Gate 矩阵
  - $model.layers.{i}.mlp.shared_experts.*$ ：Shared expert 的 w1/w2/w3
  - `model.layers.{i}.mlp.experts.{e}.*`：256 个 routed expert（FP4 packed）
  - `model.layers.{i}.hc_*`：mHC 权重
  - `model.layers.{i}.mtp.*`（仅 MTP 层）

`convert.py` 的主要工作：
1. 合并低秩矩阵（wq_a × wq_b → 完整的 W_Q）
2. 按 world_size 切片（TP/EP 并行）
3. FP4 pack：2 个 FP4 值打包为 1 个 uint8
4. 预计算 RoPE freqs_cis（1M 长度）

**面试要点**：能说出 convert.py 的"合并、切片、打包"三步转换——这是大模型部署的标准 pipeline。

**延伸阅读**：主报告 CH8.1；`inference/convert.py`。


### Q9.16 Attention 类中 compress_ratios 是如何决定每层行为的？

**简短回答**： $Attention.__init__$ 通过 $self.compress_ratio = args.compress_ratios[layer_id]$ 读取逐层配置。ratio=0 时只维护滑窗 KV（无压缩），ratio=4 时创建 Compressor+Indexer（CSA 分支），ratio=128 时创建 Compressor 但不创建 Indexer（HCA 分支）。forward 时按分支调用不同的 top-k 选择逻辑。

**详细解释**：
代码中的三分支逻辑（`model.py` L453-L471, L508-L514）：

```python
# __init__ 阶段：按 ratio 创建组件
self.compress_ratio = args.compress_ratios[layer_id]

if self.compress_ratio:                          # ratio > 0
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:                 # CSA
        self.indexer = Indexer(args, self.compress_ratio)
    else:                                        # HCA (ratio=128)
        self.indexer = None
else:                                            # ratio=0 (纯滑窗)
    self.compressor = None
    self.indexer = None

# forward 阶段：按分支选 top-k
if self.compress_ratio:
    if self.indexer is not None:                 # CSA
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                                        # HCA
        compress_topk_idxs = get_compress_topk_idxs(ratio, ...)
    topk_idxs = torch.cat([window_topk_idxs, compress_topk_idxs], dim=-1)
```

V4-Flash 的 $compress_ratios$ 配置（43 项实际使用 + 1 项占位）：
```
[0, 0, 4, 128, 4, 128, ..., 4, 128, 4, 0(占位)]
Layer 0: 滑窗 / Layer 1: 滑窗 / Layer 2-42: CSA/HCA 交替
```

**面试要点**：ratio=0/4/128 不是随便选的——0 给浅层保留 raw token 信息，4 是"轻压缩+索引"的最优压缩比，128 是"计算最小化"的最优压缩比。三个值构成信息-计算的三维权衡。

**延伸阅读**：主报告 CH3.4；`inference/model.py` L453-L471。


### Q9.17 generate.py 中的 Gumbel-max trick 是什么？为什么用它？

**简短回答**：Gumbel-max trick 是一种通过向 logits 加 Gumbel 噪声再取 argmax 来实现精确多项分布采样的方法。V4 用它是为了**避免 GPU-CPU 同步**——`torch.multinomial` 需要在 CPU 上做累积分布计算，而 Gumbel-max 的 argmax 可以在 GPU 上一步完成。

**详细解释**：
标准 `torch.multinomial` 的问题：内部实现需要把 logits 传回 CPU 计算累积概率分布，这在 autoregressive generation 中每步都会触发一次 GPU→CPU 的数据传输（~5-10 us），1000 token 生成就是 5-10 ms 的额外延迟。

Gumbel-max trick（`generate.py` L19）：
```python
def sample(logits, temperature=1.0, top_p=1.0):
    if temperature > 0:
        # Gumbel-max: 加 Gumbel 噪声后取 argmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits = (logits / temperature) + gumbel_noise
    
    if top_p < 1.0:
        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    return torch.argmax(logits, dim=-1)  # GPU 上完成
```

全流程在 GPU 上，零 GPU-CPU 同步。 $temperature=1.0, top_p=1.0$ 是 V4 README 推荐的默认值（V4 不需要 temperature 调节和 top-p 截断）。

**面试要点**：能说出 Gumbel-max 的数学等价性（加 Gumbel(0,1) 噪声后的 argmax 等价于从 $\mathrm{softmax}$ 分布采样）和工程动机（避免 GPU-CPU 同步）。

**延伸阅读**：主报告 CH2.3；`inference/generate.py` L19。


### Q9.18 V4-Flash 的 Pre-Norm 具体放在什么位置？

**简短回答**：V4-Flash 在每个 attention 和 MoE 层之前各放一个 $\mathrm{RMSNorm}$ （Pre-Norm 架构），在 attention Q 之后和 KV 之后的 head 级别再加 $\mathrm{RMSNorm}$ （per-head norm）。整体结构为： $mHC reduce \to  \mathrm{RMSNorm} \to  Attention \to  mHC expand \to  mHC reduce \to  \mathrm{RMSNorm} \to  MoE \to  mHC expand$ 。

**详细解释**：
源码中的 Norm 位置（`Block.forward` L1847-L1859）：

```python
def forward(self, x, start_pos, input_ids):
    # === Attention 残差 ===
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, ...)   # mHC reduce (4→1 通道)
    x = self.attn_norm(x)                                    # Pre-Norm for Attention
    x = self.attn(x, start_pos)                              # Attention (内部含 Q/K 的 head norm)
    x = self.hc_post(x, residual, post, comb)                # mHC expand (1→4 通道)
    
    # === MoE 残差 ===
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, ...)      # mHC reduce
    x = self.ffn_norm(x)                                     # Pre-Norm for MoE
    x = self.ffn(x, input_ids)                               # MoE
    x = self.hc_post(x, residual, post, comb)                # mHC expand
    return x
```

Attention 内部的额外 Norm（`Attention.forward` L470-L543）：
- Q 归一化： $q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)$ （head-level $\mathrm{RMSNorm}$ ）
- KV 归一化： $kv = self.kv_norm(kv)$ （ $\mathrm{RMSNorm}(head_dim)$ ）
- Compressor 输出归一化：`kv = self.norm(kv.to(dtype))`

这 3 个 head-level 的额外 Norm 是 V4 的关键数值稳定设计——让 attention logits $Q\cdot K^T / \sqrt{d}$ 始终有界，省去了 Muon 中 QK-Clip 的需要。

**面试要点**：区分"Block 级 Pre-Norm"和"Attention 内部 per-head Norm"——前者是 Transformer 架构选择，后者是 V4 独创的数值稳定 trick。

**延伸阅读**：主报告 CH5.1、CH6.4；`inference/model.py` L1847-L1859, L470-L543。


### Q9.19 V4-Flash 的 embedding 和 LM head 是否共享权重？

**简短回答**：不共享。`config.json` 中 $tie_word_embeddings: false$ ，即输入 embedding（`ParallelEmbedding`）和输出 head（`ParallelHead`）是两套独立参数，各有自己的权重矩阵。这在 V4 系列中是标准做法。

**详细解释**：
代码层面：
- `ParallelEmbedding`（`model.py` L83）：持有 $self.weight \in  R^{vocab/TP, dim}$ ，一个可学习嵌入矩阵
- `ParallelHead`（`model.py` L703）：持有 `self.weight`，是 `ColumnParallelLinear`（dim → vocab），独立参数

不共享的理由：
1. 输入和输出有不同的语义——输入是"把 token id 映射为稠密向量"，输出是"把 hidden state 映射为词表 logits"
2. V4-Flash 的 vocab_size=129280 较大，共享权重可能限制 LM head 的表达能力
3. 后训练阶段的 SFT 和 RL 主要训练 LM head 参数，独立权重允许 head 做更大的调整

$tie_word_embeddings: false$ 也是 V3 系列的标准配置。

**面试要点**：不要假设"embedding 和 head 一定共享或一定不共享"——这取决于模型设计。GPT-2 共享，GPT-3 及以后的许多模型不共享。V4 不共享。

**延伸阅读**：主报告 CH2.1；`config.json` L58 $tie_word_embeddings: false$ 。


### Q9.20 V4-Flash 的 Tokenizer 有什么特点？（vocab_size=129280）

**简短回答**：V4-Flash 使用 BPE（Byte Pair Encoding）tokenizer，vocab_size=129280（比 V3 的 129280 保持一致）。tokenizer 支持中英文及多语言，通过 $encoding/encoding_dsv4.py$ 处理 OpenAI 兼容的 messages ↔ string 转换。

**详细解释**：
Tokenizer 关键参数（ $tokenizer_config.json$ ）：
- $vocab_size$: 129,280
- $bos_token_id$: 0（`<｜begin▁of▁sentence｜>`）
- $eos_token_id$: 1（`<｜end▁of▁sentence｜>`）
- 聊天模板支持（chat_template）

$encoding_dsv4.py$ 的核心功能：
- $encode_messages(messages, thinking_mode, reasoning_effort)$ → string：把 OpenAI 格式的 messages 转为带 CoT 控制的 prompt 字符串
- $parse_message_from_completion_text(completion_text)$ → response：从模型输出中提取 `<think>/<summary>` 部分
- $render_message$ ：处理 system prompt 和 REASONING_EFFORT 前缀

三种 thinking_mode 的 prompt 差异：
- `"chat"`：直接输出，无 `<think>` 标记
- `"thinking"` + $reasoning_effort="high"$ ：插入 `<think>` 起始 token，引导中等长度推理
- `"thinking"` + $reasoning_effort="max"$ ：加上 REASONING_EFFORT_MAX 前缀指令，引导最长推理

**面试要点**：vocab_size 的选择是一个平衡——太小 token 压缩率低（sequence 变长），太大 embedding 矩阵太大。129280 是 DeepSeek 经过实验选择的"中文+英文+代码"覆盖的最优值。

**延伸阅读**：主报告 CH2.4； $encoding/encoding_dsv4.py$ 。


### Q9.21 V4-Flash 中 FP8 量化的 act_quant_kernel 具体做了什么？

**简短回答**： $act_quant_kernel$ 以 block=128 为单位对比激活张量做 FP8 量化：对每个 128 元素的 block，计算 amax（绝对最大值），用 $amax / fp8_max$ 作为 scale，将每个元素 clamp 到 $[-fp8_max, fp8_max]$ 后除以 scale 并 cast 到 FP8（ $float8_e4m3$ ）。scale 存为 FP32。

**详细解释**：
Kernel 的核心逻辑（`kernel.py` L41-L103 精简）：

```python
@tilelang.jit
def act_quant_kernel(N, block_size=128, out_dtype="float8_e4m3", scale_dtype=FP32, inplace=False):
    fp8_min, fp8_max, fp8_max_inv = -448.0, 448.0, 1/448.0
    
    @T.prim_func
    def kernel_(X, Y, S):  # X: bf16[M,N], Y: fp8[M,N], S: scale[M, N/128]
        with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 128), threads=128):
            # block-wise amax
            T.reduce_absmax(x_local, amax_local, dim=1)
            # scale 计算
            for i in T.Parallel(32):
                s_local[i] = amax_local[i] * fp8_max_inv  # amax / 448
            # 量化
            for i, j in T.Parallel(32, 128):
                Y[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
```

关键设计点：
- block_size=128：每 128 个元素独立 scale（细粒度量化）
- 使用 $T.reduce_absmax$ ：一次 reduction 得到 block 内的最大值
- $fp8_max=448.0$ ：E4M3 格式的最大可表示值
- `inplace=True`：quant 完成后立即 dequant 回 BF16，省一次 global memory 写回
- 所有 scale 计算在 shared memory / register 中完成，零 global memory 开销

**面试要点**：能说出 block size 为什么是 128——FP8 精度较高（256 级量化），block=128 的量化误差可接受；activation 的 outlier 比权重多，需要更大的 block 让 outlier 的影响被平滑。

**延伸阅读**：主报告 CH7.2；`inference/kernel.py` L36-L103。


### Q9.22 V4-Flash 仓库中训练相关的代码在哪里？

**简短回答**：V4-Flash 仓库是 **inference-only**，不包含任何训练代码。没有 `optimizer.py`（Muon 实现）、没有 `train.py`、没有数据加载、没有 loss 计算、没有 aux-loss-free bias 更新逻辑。所有这些在 V4 论文中有文字描述和算法伪代码，但未开源。

**详细解释**：
仓库包含的 vs 不包含的：

**包含（inference 相关）**：
- 模型前向定义（`model.py`，827 行）
- GPU kernel（`kernel.py`，536 行）
- 生成循环（`generate.py`，155 行）
- 权重转换（`convert.py`，168 行）
- Tokenizer 编码（ $encoding/encoding_dsv4.py$ ）

**不包含（training 相关）**：
- Muon 优化器（论文 Algorithm 1）
- Newton-Schulz 迭代 kernel
- 训练循环（forward + backward + step）
- 数据加载管线
- Loss 函数（LM loss + MTP loss + aux loss）
- Aux-loss-free bias 更新
- FP4 QAT 的 STE 反向传播
- 分布式训练的 DualPipe / ZeRO / all-to-all 逻辑
- Activation checkpointing 逻辑

公开可参考的训练信息：
- 论文 §2.4：Muon 算法（Algorithm 1，伪代码）
- 论文 §3.1-3.4：训练框架描述
- 论文 §4.2：训练超参（lr, batch size 等定性描述）
- 论文 §5.2.1：FP4 QAT 训练描述

**面试要点**：这是面试中常见的陷阱问题——面试官问"你怎么实现的 Muon 训练"，如果声称看了仓库代码就知道，会被追问"V4 仓库里根本没有，你怎么知道的"。正确答案是"V4 仓库是推理专用，训练代码未开源，Muon 的细节来自论文 Algorithm 1"。

**延伸阅读**：主报告 CH6、CH8.1；V4 论文 §2.4。


### Q9.23 model.py 中 Block.forward 的完整调用顺序是怎样的？

**简短回答**：`Block.forward` 共 7 步（严格顺序）：hc_pre（4 通道→1 通道）→ attn_norm → Attention（CSA/HCA/滑窗分支）→ hc_post（注入 4 通道残差）→ ffn_norm → MoE（Gate + routed experts + shared expert）→ hc_post（再次注入）。核心特征是 2 次 hc_post 实现 mHC 的 4 通道残差流，而非标准 Transformer 的逐层单残差。

**详细解释**：
`Block.forward` 的 7 步完整调用链（对应源码 `model.py` L1847-L1859）：

```
Step 1: hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
        ── mHC 把 [B,S,4,4096] 的 4 通道 reduce 为 [B,S,4096] 的 1 通道
        ── 同时输出 post 和 comb 权重（供后续 hc_post 使用）

Step 2: attn_norm(x)
        ── RMSNorm (Pre-Norm for Attention)

Step 3: self.attn(x, start_pos, ...)
        ── 根据 compress_ratios[layer_id] 分支：
          · ratio=0 → 纯滑窗 attention（window=128）
          · ratio=4 → CSA (Compressor + Indexer + 滑窗拼接)
          · ratio=128 → HCA (Compressor + 位置 topk + 滑窗拼接)
        ── 内部含 Q/KV 的 per-head RMSNorm

Step 4: hc_post(attn_out, residual, post, comb)
        ── mHC 把 attention 输出注入 4 通道残差流
        ── y = post * attn_out + comb @ residual（逐通道缩放 + 通道间混合）

Step 5: ffn_norm(x)
        ── RMSNorm (Pre-Norm for MoE)

Step 6: self.moe(x, input_ids)
        ── Gate: sqrtsoftplus(W_g·x) + aux-loss-free bias → top-6 experts
        ── Routed experts: 6 个 Expert.forward（SwiGLU, FP4 反量化）
        ── Shared expert: 始终激活（FP8）
        ── 合成: y = shared(x) + route_scale * sum(w_i * routed_i(x))

Step 7: hc_post(moe_out, residual, post, comb)
        ── mHC 把 MoE 输出注入 4 通道残差流（与 Step 4 结构相同）
```

**与标准 Transformer Block 的差异**：

| 维度 | 标准 Transformer | V4-Flash Block |
|------|-----------------|---------------|
| 残差连接数 | 2 个（attn 后 + FFN 后各 1 个） | 2 个 mHC 残差（但每次是 4 通道注入） |
| 残差形式 | `x = x + f(Norm(x))`（标量加） | `y = post * f(x) + comb @ residual`（矩阵混合） |
| 残差路径 | 逐层独立（每层有自己的残差 x_l） | 4 通道共享（4 个信息通道贯穿全部 43 层） |
| Norm 位置 | Pre-Norm 或 Post-Norm | Pre-Norm（attn_norm + ffn_norm） |
| 计算步骤 | 3 步（Norm→Attn→Add, Norm→FFN→Add） | 7 步（含 mHC pre/post 各 2 次） |

mHC 的 4 通道残差流是整个 Block 的核心——不是简单的"x = x + f(x)"，而是每个子层的输出通过 $hc_post$ 注入 4 个信息通道，通道间用组合矩阵 $comb \in  R^{4\times 4}$ （Sinkhorn 双随机约束）进行信息混合。这让不同层之间的信息流动不再是"一层接一层"的链式，而是"所有层共享一个 4 通道记忆体"的网状结构。

💡 **面试要点**：能画出 Block forward 的 7 步 pipeline： $hc_pre \to  attn_norm \to  attn \to  hc_post \to  (repeat for ffn)$——关键在于 mHC 的 pre/post 成对出现，每次子层计算前后各一次。

**延伸阅读**：主报告 CH5.5；`inference/model.py` L1847-L1859 `Block.forward`。


### Q9.24 从 config.json 到 model.py 的超参是怎么映射的？举个具体例子

**简短回答**：映射链为 `config.json` → `ModelArgs`（dataclass, model.py L35-L82）→ 各 class 的 $__init__$ 参数。以 5 个关键字段为例： $hidden_size=4096$ → `ModelArgs.dim` → 所有 Linear 层 in/out features； $n_routed_experts=256$ → $MoE.__init__$ 创建 256 个 `Expert`； $num_hash_layers=3$ → $Gate.__init__$ 判断 $self.layer_id < 3$ 决定 hash vs score 路由； $compress_ratios[0]=0$ → $Attention.__init__$ 不创建 Compressor/Indexer； $hc_mult=4$ → $Block.__init__$ 中 $hc_dim = 4 * dim = 16384$ 。

**详细解释**：
5 个关键 config 字段的完整映射表：

| config.json 字段 | 典型值 | 映射到 ModelArgs 属性 | 被哪些类使用 | 具体作用 |
|-----------------|--------|---------------------|------------|---------|
| $hidden_size$ | 4096 | `args.dim` | `Attention`, `Expert`, `MoE`, `Block`, $\mathrm{RMSNorm}$ 等几乎所有类 | 决定所有 Linear 层的 in/out features，决定了模型的"宽度" |
| $n_routed_experts$ | 256 | $args.n_routed_experts$ | $MoE.__init__$ | 创建 `nn.ModuleList([Expert() for _ in range(256)])`，决定 MoE 的"容量" |
| $num_hash_layers$ | 3 | $args.num_hash_layers$ | $Gate.__init__$ | 在 $Gate.__init__$ 中（L554-L560）： $if self.layer_id < self.num_hash_layers: self.tid2eid = nn.Parameter(randint(0, 256, [vocab, 6]))$ 创建哈希查找表；`Gate.forward`（L568-L573）判断走 hash 还是 score 路由 |
| $compress_ratios$ | `[0,0,4,128,...]` | $args.compress_ratios$ | $Attention.__init__$ | $self.compress_ratio = args.compress_ratios[layer_id]$ ，ratio=0 不创建 Compressor；ratio=4 创建 Compressor+Indexer（CSA）；ratio=128 仅创建 Compressor（HCA） |
| $hc_mult$ | 4 | $args.hc_mult$ | $Block.__init__$ | $hc_dim = args.hc_mult * args.dim = 16384$ ，决定 hc_pre 中 24 维 mixes 的输入维度 |

**具体映射链演练**（以 $num_hash_layers=3$ 为例）：

```python
# Step 1: config.json
{"num_hash_layers": 3}

# Step 2: ModelArgs (model.py L35-L82)
@dataclass
class ModelArgs:
    num_hash_layers: int = 3  # 直接从 config.json 读取

# Step 3: Transformer.__init__ (model.py L769)
for layer_id in range(args.n_layers):
    self.layers.append(Block(args, layer_id))

# Step 4: Block.__init__ → Gate.__init__ (model.py L554)
class Gate:
    def __init__(self, args, layer_id):
        self.layer_id = layer_id
        if layer_id < args.num_hash_layers:  # layer 0,1,2 → True
            self.tid2eid = nn.Parameter(     # 创建 [129280, 6] 哈希查找表
                torch.randint(0, 256, (args.vocab_size, args.top_k)),
                requires_grad=False
            )

# Step 5: Gate.forward (model.py L568-L573)
def forward(self, x, input_ids):
    if self.layer_id < self.num_hash_layers:
        # Hash routing: 跳过 matmul，直接查表
        indices = self.tid2eid[input_ids]
        weights = torch.ones_like(indices, dtype=torch.float32) / self.top_k
    else:
        # Score routing: sqrtsoftplus(W_g @ x) + bias → topk
        scores = F.softplus(F.linear(x, self.weight)).sqrt()
        ...
```

同样的映射逻辑适用于其他所有 config 字段——`config.json` 的每一项都经过 `ModelArgs` → 特定 class 的 $__init__$ → 影响该 class 的 `forward` 行为。

**5 个字段的"一句话"记忆法**：
- $hidden_size=4096$ → 一切 Linear 层的宽度
- $n_routed_experts=256$ → 多少个 Expert 实例
- $num_hash_layers=3$ → 前几层走 hash 路由
- $compress_ratios=[0,0,4,128,...]$ → 每层用什么 attention 模式
- $hc_mult=4$ → mHC 的通道数（4 通道残差流）

💡 **面试要点**：能说出任意 3 个 config 参数在 model.py 中的落地位置和具体作用即可。"能追踪一个参数从 config 到 forward 的完整路径"是高级水平。

**延伸阅读**：主报告 CH2.1（config.json 全解）、CH8.1（model.py 类组织）；`inference/model.py` L35-L82 `ModelArgs`；`config.json`。


## CH 10. 面经高频 -- QA 问答

## 基础题 10 个

### Q10.1 写出 self-attention 的完整公式，并解释每个符号的含义

**简短回答**： $Attention(Q, K, V) = \mathrm{softmax}(QK^T / \sqrt{d_k}) \cdot V$ ，其中 Q（Query）、K（Key）、V（Value）分别是输入 x 的三个线性投影，d_k 是单头的维度，除以 √d_k 是为了防止点积过大导致 $\mathrm{softmax}$ 梯度消失。

**详细解释**：
Self-attention 的 5 步计算：
1. 投影： $Q = xW_Q, K = xW_K, V = xW_V$ （ $W_{Q,K,V} \in  R^{d\times d}$ ）
2. 计算注意力分数： $S = QK^T \in  R^{T\times T}$ ，S_{ij} = q_i · k_j
3. 缩放： $S_scaled = S / \sqrt{d_k}$——除以 √d_k 是因为随机向量点积的方差 ∝ d_k，不缩放的话 $\mathrm{softmax}$ 会饱和（梯度近似 0）
4. 归一化： $A = \mathrm{softmax}(S_scaled)$ ，A 每行和为 1
5. 加权求和： $O = A \cdot V$

多头扩展： $MultiHead(Q, K, V) = Concat(head_1, ..., head_h) \cdot W_O$ ，其中 $head_i = Attention(QW_{Q_i}, KW_{K_i}, VW_{V_i})$ 。

在 causal（自回归）场景下，S 的上三角部分置为 -inf（mask），确保 token t 只能关注位置 < t 的历史。

**面试官视角**：考察对 Transformer 核心算子的基本理解。很多人只会背公式说不清 √d_k 的作用，这是区分"死记硬背"和"真正理解"的分界线。

**答题模板**：(1) 先写公式 $\mathrm{softmax}(QK^T/\sqrt{d}) \cdot V$ ，(2) 解释 $\sqrt{d}$ 为什么必要（方差缩放），(3) 说明 mask 的作用（因果性），(4) 给出 Multi-Head 的扩展形式。

**延伸阅读**：主报告 CH3.1。


### Q10.2 MHA（Multi-Head Attention）相比单头的优势是什么？

**简短回答**：MHA 将 Q、K、V 投影到 h 个不同的子空间（每个 head 关注不同的表示子空间），并行计算，然后拼接输出。优势有三：(1) 不同 head 可以关注不同的位置模式（位置、语法、语义）；(2) 每个 head 的维度 d_h = d/h 更小，计算效率更高；(3) 多头提供了"集成"的效果，单头可能只关注到部分重要关系。

**详细解释**：
Multi-Head 的形式化：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = softmax(Q_i · K_i^T / √d_h) · V_i
```
其中 Q_i = Q · W_{Q_i}（W_{Q_i} ∈ R^{d×d_h}，d_h = d/h）。

优势拆解：
1. **表征多样性**：实战中不同 head 确实学到不同的模式——低级头关注相邻 token（类似 n-gram），中级头关注句法结构，高级头关注语义相关性。有论文可视化验证：不同 head 的 attention pattern 差异很大。
2. **计算效率**：单头 h=1 时 QK^T 矩阵为 T×T，多头下 h 个 T×d_h 矩阵的内积可以并行，且 $h \times  d_h = d$ 总计算量相同但并行度更高。
3. **鲁棒性**：单头如果"看错了"某个位置，整个输出受影响；多头中即使某些 head 的注意力偏差，其他 head 可以"补位"。

V4-Flash 用 64 个 attention head（ $num_attention_heads=64$ ），head_dim=128，但 KV 走 MQA（ $num_key_value_heads=1$ ），即 64 个 Q head 共享 1 组 KV。

**面试官视角**：考察是否理解多头不只是"并行加速"，而是"表征学习"的需要。能举出"低级/中级/高级 head"的具体例子是加分项。

**答题模板**：(1) 核心公式，(2) 三个优势（多样性/效率/鲁棒性），(3) 实际例子（visualization 展示不同 head 关注不同模式），(4) 延伸到 MQA/GQA 的 trade-off。

**延伸阅读**：主报告 CH3.1、CH3.4。


### Q10.3 Pre-Norm 和 Post-Norm 有什么区别？V4-Flash 用哪个？

**简短回答**：Pre-Norm 把 Normalization 放在子层（attention/FFN）之前：`y = x + f(Norm(x))`；Post-Norm 放在子层之后：`y = Norm(x + f(x))`。Pre-Norm 在训练时更稳定（残差路径不需额外 Norm），是现代大模型的主流选择。V4-Flash 用 Pre-Norm。

**详细解释**：
梯度传播的差异：
- Post-Norm： $\partialy/\partialx = \partialNorm/\partial(x+f) \cdot (I + \partialf/\partialx)$ ，Norm 的导数在深层中可能被放大/缩小，导致梯度不稳定。Xiong et al. (2020) 指出 Post-Norm 需要 warmup 来缓解早期训练的不稳定。
- Pre-Norm： $\partialy/\partialx = I + \partialf/\partial(Norm(x)) \cdot \partialNorm/\partialx$ ，残差路径的梯度直接有 I（恒等）项，不会因 Norm 而衰减。这让深层网络的信号传播更稳定。

V4-Flash 的具体位置（源码）：
```python
# Block.forward
x = self.attn_norm(x)    # Pre-Norm → Attention
x = self.attn(x, start_pos)
...
x = self.ffn_norm(x)     # Pre-Norm → MoE
x = self.ffn(x, input_ids)
```

此外，V4 在 attention 内部还对 Q 和 KV 做 head-level 的 $\mathrm{RMSNorm}$ （公式上等价于 $Q ← Q / \|Q\|$ ），确保 attention logits 不会爆炸。

**面试官视角**：考察是否理解 Norm 位置对梯度传播的影响——这是 Transformer 训练的"第一性原理"问题。能画出梯度流图比背结论更重要。

**答题模板**：(1) 写出两种放置的公式，(2) 画出梯度流图分析，(3) 说明 V4 的选择，(4) 补充 V4 的 head-level Norm 创新。

**延伸阅读**：主报告 CH5.1。


### Q10.4 $\mathrm{RMSNorm}$ 和 $\mathrm{LayerNorm}$ 有什么区别？为什么 LLM 偏好 $\mathrm{RMSNorm}$ ？

**简短回答**： $\mathrm{LayerNorm}$ 对输入做"减均值 → 除标准差"的标准化： $(x-\mu)/\sigma$ ，而 $\mathrm{RMSNorm}$ 只做"除 RMS"（不减均值）：`x / RMS(x)`。 $\mathrm{RMSNorm}$ 省去了均值计算（约一半的运算量），同时实验表明两种 Norm 在 Transformer 中效果相当。V4-Flash 全用 $\mathrm{RMSNorm}$ 。

**详细解释**：
公式对比：
- $\mathrm{LayerNorm}$ ： $y = \gamma \cdot (x - \mu) / √(\sigma^2 + \epsilon) + \beta$ ，其中 μ = Σx_i / d，σ² = Σ(x_i-μ)² / d
- $\mathrm{RMSNorm}$ ： $y = \gamma \cdot x / √(\Sigmax_i^2/d + \epsilon)$ （无 β，无 μ）

计算量对比： $\mathrm{LayerNorm}$ 需要 2 次 reduction（mean + variance）， $\mathrm{RMSNorm}$ 只需 1 次（sum of squares）。在大模型的每一层中，Norm 虽然是 $O(d)$ 操作，但 43 层 × 至少 2 个 Norm/layer = 86 个 Norm——省一半的 Norm 计算量约等于省掉一个 attention head。

为什么可以不减均值？Transformer 的输入通常已经是零均值的（前一层的输出经过 normalization），减去均值的效果可以被后续的线性层吸收（线性层的 bias 项可以模拟平移）。Zhang and Sennrich (2019) 的实验证实： $\mathrm{RMSNorm}$ 和 $\mathrm{LayerNorm}$ 在 Transformer 上的最终性能差异 < 0.1 BLEU/ppl。

V4-Flash 中 $\mathrm{RMSNorm}$ 的使用位置：
- $attn_norm$ / $ffn_norm$ ：Block 级 Pre-Norm
- $q_norm$ / $kv_norm$ ：Attention 内 Q/KV 的 head-level Norm
- $hc_pre$ 内部的 $rsqrt(x^2.mean + eps)$ ：mHC 的自定义 Norm
- `model.norm`：最终输出 Norm

**面试官视角**：考察是否关心"实现细节"——LLM 训练中每一点计算节省都会在千卡集群上放大。能说出"两次 reduction vs 一次 reduction"的计算量对比就是加分。

**答题模板**：(1) 写出两个公式，(2) 计算量对比（1 reduction vs 2），(3) 解释为什么不需要减均值（后续线性层可补偿），(4) 说明 V4 的使用位置。

**延伸阅读**：主报告 CH5.1；`inference/model.py` L183 $\mathrm{RMSNorm}$ 类。


### Q10.5 写出 RoPE 的旋转矩阵，说明为什么它编码了相对位置

**简短回答**：RoPE 将 d 维向量按维度对分组（2 维 1 组 = 1 个复数），第 i 对的旋转角度为 θ_i = 1/base^(2i/d)。位置 m 处的旋转为 $q_m[i] = q[i]cos(m\theta_i) - q[i+1]sin(m\theta_i)$ （类似二维旋转）。旋转后的点积 $q_m \cdot k_n$ 等于 $\Sigma q[i]k[i]cos((m-n)\theta_i) + q[i]k[i+1]sin((m-n)\theta_i)$ ，仅依赖相对位置 (m-n)。

**详细解释**：
RoPE 的数学本质是"将复数乘法作为位置编码"：

1. 将 d 维向量按维度对分组为 d/2 个复数：q = [q_0 + iq_1, q_2 + iq_3, ..., q_{d-2} + iq_{d-1}]
2. 对位置 m 做旋转：q_m = $q \cdot e$^{imθ}（复数乘法，e^{imθ} = cos(mθ) + i sin(mθ)）
3. 内积变为： $Re(q_m \cdot conj(k_n))$ = $Re(q \cdot e^{im\theta} \cdot conj(k \cdot e^{in\theta}))$ = $Re(q \cdot conj(k) \cdot e^{i(m-n)\theta})$

最后一步利用了 $e^{im\theta} \cdot e^{-in\theta} = e^{i(m-n)\theta}$——绝对位置 m 和 n 被"消掉"了，只剩相对位置 m-n。这正是 RoPE 最优雅的性质。

V4-Flash 的实现（ $apply_rotary_emb$ ，`model.py` L232）：
```python
def apply_rotary_emb(x, freqs_cis, inverse=False):
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)                     # 转复数
    if inverse:
        freqs_cis = freqs_cis.conj()                   # 反旋转用共轭
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)  # 复数乘法
    return x_out.type_as(x)
```

**面试官视角**：这是一道"数学基础+代码实现"双考合一的问题。能说出复数乘法的视角说明真正理解了 RoPE（而非只记公式），能提到"partial RoPE"（V4 只在最后 64 维用 RoPE）是加分。

**答题模板**：(1) 写出二维旋转矩阵，(2) 用复数视角推导相对位置性质，(3) 给出 V4 的参数（qk_rope_head_dim=64, rope_theta=10000），(4) 解释 partial RoPE 的设计动机。

**延伸阅读**：主报告 CH7.1；`inference/model.py` L232-L250 $apply_rotary_emb$ 。


### Q10.6 KV cache 的内存占用量怎么算？以 V4-Flash 1M 为例

**简短回答**：KV cache 内存 = $L \times  T \times  d_h \times  n_kv_heads \times  2(K+V) \times  bytes_per_element$ 。标准 MHA 下是 $L \times  T \times  d \times  2 \times  2 bytes$ （BF16），V4-Flash 的 MQA + CSA/HCA 压缩 + 混合精度后约 4 GB。

**详细解释**：
分步计算（V4-Flash 1M 上下文）：

**步骤 1：标准 MHA（假设 64 KV heads）**
- Per layer per token = 64 heads × 512 dim × 2 (K+V) × 2 bytes = 128 KB/层/token
- 43 layers × 1M tokens = 43 × 1M × 128 KB ≈ 5.5 TB —— 不可行

**步骤 2：MQA（实际 1 KV head）**
- Per layer per token = 1 head × 512 dim × 2 × 2 = 2 KB/层/token
- 43 layers × 1M tokens = 43 × 1M × 2 KB ≈ 88 GB —— 仍然很大

**步骤 3：CSA/HCA 压缩 + 混合精度（实际方案）**
- CSA 21 layer: (128 + T/4) × 576 bytes = 262,272 × 576 ≈ 151 MB/layer
- HCA 20 layer: (128 + T/128) × 576 bytes = 8,320 × 576 ≈ 4.8 MB/layer
- 合计 ≈ 3.3 GB

**步骤 4：加上 Indexer KV cache（额外 ~0.7 GB）**
- 总 KV cache ≈ **4 GB**

**面试官视角**：考察能否在面试中"现场建账"——不需要精确到小数点，但数量级要对（GB vs TB）。更重要的是，要能说出"三层压缩接力"的逻辑（head 维 → 时间维 → 位宽维）。

**答题模板**：(1) 写出通用公式，(2) 逐步演示 5.5 TB → 88 GB → 4 GB 的压缩过程，(3) 说明每层压缩的机制，(4) 总结"三层压缩接力"。

**延伸阅读**：主报告 CH3.5、Q9.9。


### Q10.7 AdamW 的更新公式是什么？和 SGD 的区别在哪？

**简短回答**：AdamW 将权重衰减（weight decay）与梯度更新解耦： $m_t = \beta₁\cdot m_{t-1} + (1-\beta₁)\cdot g_t$ （动量）， $v_t = \beta₂\cdot v_{t-1} + (1-\beta₂)\cdot g_t^2$ （二阶动量）， $W_t = W_{t-1} - \eta\cdot (m_t/(\sqrt{v_t} + \epsilon) + \lambda\cdot W_{t-1})$ 。与 SGD 的核心区别在于"逐元素的自适应学习率"——每个参数根据其历史梯度的方差自动调节步长。

**详细解释**：
完整 AdamW 更新公式：
```
m_t = β_1 * m_{t-1} + (1-β_1) * g_t        # 一阶动量（梯度方向平滑）
v_t = β_2 * v_{t-1} + (1-β_2) * g_t^2      # 二阶动量（梯度方差估计）
m_hat = m_t / (1-β_1^t)                     # 偏差修正
v_hat = v_t / (1-β_2^t)
W_t = W_{t-1} - η * (m_hat / (√v_hat + ε) + λ * W_{t-1})
```

核心差异（vs SGD + Momentum）：

| 维度 | SGD + Momentum | AdamW |
|------|----------------|-------|
| 学习率 | 全局固定/固定 schedule | 逐元素自适应 |
| 二阶信息 | 无 | 梯度方差估计（RMSProp 继承） |
| 权重衰减 | 与 LR 耦合（ $W -= \eta\lambdaW$ ） | 解耦（ $W -= \eta\lambdaW$ 与梯度更新分离） |
| 对大模型的适用性 | 需要精细调参 | 对超参更鲁棒 |

AdamW 在 LLM 中的局限性（触发 V4 换 Muon 的原因）：
- 逐元素更新对 MoE 稀疏梯度不友好（大量参数没梯度，v_t 过时）
- 不约束矩阵奇异值（谱偏问题，effective rank 下降）

**面试官视角**：考察优化器基础 + 理解 AdamW 的局限。能说出"逐元素 vs 矩阵级"的区别是高级答案。

**答题模板**：(1) 写出完整公式，(2) 解释 bias correction 的作用，(3) 与 SGD 对比，(4) 指出 AdamW 在 MoE 上的局限（引出 Muon）。

**延伸阅读**：主报告 CH6.1。


### Q10.8 MoE 的 top-k 路由是怎么工作的？V4-Flash 的 top-k 参数是多少？

**简短回答**：MoE top-k 路由为每个 token 计算 256 个 expert 的分数，选 top-k 个（V4-Flash 用 k=6），仅激活选中的 expert。分数由 `Gate` 类计算（ $\mathrm{sqrtsoftplus}$(W_g·x) + bias），bias 用于负载均衡。V4 还加了 $route_scale=1.5$ 放大 routed 输出以补偿 k 从 V3 的 8 砍到 6 的容量损失。

**详细解释**：
完整路由流程（每个 token）：
1. Gate 评分： $scores = \mathrm{sqrtsoftplus}(W_g \cdot x) \in  R^256$
2. 加 aux-loss-free bias： $scores_for_topk = scores + b$ （b 仅影响选择）
3. Top-6 选择： $indices = topk(scores_for_topk, k=6)$ → 6 个 expert 索引
4. 提取权重： $weights = original_scores[indices]$ （用不带 b 的原始分）
5. 归一化 + 缩放： $weights /= sum(weights); weights *= 1.5$
6. 前向： $y = sum(w_i * expert_i(x)) + shared_expert(x)$

hash routing 特例（前 3 层）：跳过 Gate 计算，直接用 $tid2eid[input_id]$ 查表。

与 V3 的对比：
- V3：k=8, $\mathrm{sigmoid}$ 评分, 无 route_scale
- V4-Flash：k=6, $\mathrm{sqrtsoftplus}$ 评分, route_scale=1.5

**面试官视角**：考察对 MoE 路由的基本理解。能区分"routing weight（用于前向加权）"和"routing score（用于 top-k 选择）"是理解 aux-loss-free 的前提。

**答题模板**：(1) 写出 6 步路由流程，(2) 说明 V4 的参数（k=6, E=256, route_scale=1.5），(3) 与 V3 对比，(4) 补充 hash routing 的特殊情况。

**延伸阅读**：主报告 CH4.2。


### Q10.9 $\mathrm{FlashAttention}$ 的核心优化是什么？Online $\mathrm{softmax}$ 怎么实现？

**简短回答**： $\mathrm{FlashAttention}$ 的核心优化是"用 SRAM 上的 tiling + online $\mathrm{softmax}$ 替代 HBM 上的完整 attention 矩阵物化"。Online $\mathrm{softmax}$ 通过跟踪 running max（ $m_t$ ）和 running sum（ $\Sigma_t$ ）来实现分块计算 $\mathrm{softmax}$ ，无需存储完整 QK^T 矩阵。

**详细解释**：
标准 attention 的内存瓶颈：QK^T 矩阵大小为 T×T，1M 上下文下是 1T 个元素（~2TB in BF16），远超 GPU HBM 容量。 $\mathrm{FlashAttention}$ 用 tiling 将其分块。

Online $\mathrm{softmax}$ 三件套（核心机制）：
```
初始化：m_0 = -inf, Σ_0 = 0, o_0 = 0
对每个 block t：
  m_t = max(m_{t-1}, block_max_t)                         # 更新全局 max
  s_t = exp(m_{t-1} - m_t)                                 # rescale 因子
  Σ_t = Σ_{t-1} * s_t + block_sum_t                        # 更新全局 sum
  o_t = o_{t-1} * s_t + softmax(block_t) * V_t             # 更新输出
最终：o = o_T / Σ_T
```

关键 insight： $\mathrm{softmax}$ 的"减去 max 再 exp"不是一次性完成的，可以分块迭代。每进一个新 block，旧的累积 o 和 Σ 乘以 $exp(old_max - new_max)$ 这个 rescale 因子，就可以在新 max 基准下继续累加。

V4 的 $sparse_attn_kernel$ （`kernel.py` L276-L368）完整实现了 online $\mathrm{softmax}$ ，额外加了 $attn_sink$ （per-head 偏置）在分母中。

**面试官视角**：这是一道"算法+系统"融合的问题。Online $\mathrm{softmax}$ 本身是数值方法（math），tiling 是系统优化（system）。两者结合才构成 $\mathrm{FlashAttention}$ 。能口述 online $\mathrm{softmax}$ 的三状态更新公式是关键。

**答题模板**：(1) 说明标准 attention 的内存问题，(2) 写出 online $\mathrm{softmax}$ 的三状态更新，(3) 解释 tiling 如何利用 SRAM，(4) 指出 V4 的 sparse_attn_kernel 也是用 online $\mathrm{softmax}$ 。

**延伸阅读**：主报告 CH3.7.3；`inference/kernel.py` L276-L368。


### Q10.10 混合精度训练中的 loss scaling 是什么？BF16 还需要吗？

**简短回答**：Loss scaling 是将 loss 乘以一个大因子（如 2^16），使梯度值变大，防止被 FP16 的动态范围截断为 0。BF16 不需要 loss scaling，因为 BF16 与 FP32 共享相同的 8 位指数（动态范围相同），不会出现梯度下溢。

**详细解释**：
Loss scaling 的必要性（仅 FP16）：
- FP16 的最小正 normal number = 6.1 × 10^-5
- 训练后期某些梯度可能小至此范围以下，直接被截断为 0 → 参数不再更新
- Loss scaling：loss = loss * 2^16，反向传播时梯度同比例放大，确保落在 FP16 可表示范围内
- 更新前：gradient = gradient / 2^16（恢复真实尺度）
- 动态 loss scaling：每 N 步检查是否有 NaN，有则减小 scale，无则增大

BF16 为什么不需要：
- BF16 最小正 normal number = 1.2 × 10^-38（与 FP32 相同）
- 梯度下溢需要 value < 10^-38，实际 LLM 梯度不会这么小
- 代价是 BF16 精度低（7 bit 尾数 vs FP16 的 10 bit）

V4-Flash 的精度策略：
- 主干训练用 BF16（无需 loss scaling）
- 前向计算部分用 FP8（E4M3，范围 448，也不存在下溢问题）
- 梯度通信用 BF16
- 优化器状态用 FP32

**面试官视角**：这是一道检验"是否做过实际大模型训练"的问题——只有实际跑过大模型的人才会遇到 loss scaling 的问题。BF16 不需要 loss scaling 是很多人都忽略的关键事实。

**答题模板**：(1) 解释 loss scaling 的动机（FP16 下溢），(2) 写出 loss scaling 的步骤（乘→反向→除），(3) 说明 BF16 不需要，动态范围与 FP32 相同。

**延伸阅读**：主报告 CH7.2。


## 进阶题 11 个

### Q10.11 MLA（Multi-head Latent Attention）和 MQA/GQA 有什么区别？

**简短回答**：MLA 将 K 和 V 分别投影到低维潜空间（d_c=512），再展开到多头，推理时只需缓存 d_c 维潜在向量。MQA 直接把 KV head 数砍到 1（64 个 Q head 共享 1 组 KV），GQA 是 MQA 和 MHA 之间的折中。MLA 的信息保留能力优于 MQA（潜在空间可以恢复更多头信息），但实现更复杂。

**详细解释**：

| 维度 | MHA | MQA | GQA | MLA |
|------|-----|-----|-----|-----|
| KV heads | = Q heads (64) | 1 | g groups (如 8) | 1（共享）+ 潜在展开 |
| KV cache per token | d × 2 | d/64 × 2 | d/g × 2 | d_c × 2 (≈ d/4) |
| KV 表达能力 | 最丰富 | 最受限 | 中等 | 接近 MHA（潜空间学得更好） |
| 实现复杂度 | 简单 | 最简单 | 简单 | 中等（潜空间投影） |

MLA 的核心公式（V3 体系）：
```
c_K = x · W_{DK}   ∈ R^{d_c}    (d_c = 512)
c_V = x · W_{DV}   ∈ R^{d_c}
K = c_K · W_{UK}   ∈ R^{h×d_h}  (展开到多头)
V = c_V · W_{UV}   ∈ R^{h×d_h}
```

推理时只存 $c_K, c_V$ （各 512 维 = 1024 float16），比 MHA 的 64×128=8192 维省 8x，比 MQA 的 128 维多 8x。但 MLA 的 KV 表达能力远强于 MQA（潜在展开可以学出 64 个不同的"虚拟 head"），在质量上更接近 MHA。

V4-Flash 用的是 MQA（ $num_key_value_heads=1$ ），而非 MLA——因为 V4 走的是"时间维压缩（CSA/HCA）"路线，不需要 MLA 的 head 维压缩。MQA + CSA/HCA 的组合在 V4 中实现了比 V3.2 的 MLA 更好的整体效率。

**面试官视角**：MLA vs MQA 的比较需要在"cache 大小 vs 表达能力"的维度上量化。很多候选人只知道"MLA 比 MQA 好"，但说不清好在哪里。能说出"潜空间可以学出不同的虚拟 head"是高级答案。

**答题模板**：(1) 给出 MQA/GQA/MLA 的定义，(2) 量化对比 cache 大小和表达能力，(3) 解释 V4 为什么选了 MQA 而不是 MLA，(4) 总结"时间维压缩（V4）vs head 维压缩（V3）"两种路线的差异。

**延伸阅读**：主报告 CH1.1、CH3.4。


### Q10.12 aux-loss-free 负载均衡和传统的 aux loss 有什么本质区别？

**简短回答**：传统 aux loss 将负载均衡作为损失函数的一项（ $L_total = L_LM + \lambda\cdot L_aux$ ），副作用是 aux loss 的梯度会"污染"主任务梯度（模型为了均匀而牺牲性能）。aux-loss-free 把负载均衡从损失函数中剥离，改为每步更新 per-expert 偏置 $b_e$ ，b 仅影响 top-k 选择（不影响 routing weight），梯度零污染。

**详细解释**：
两种方法的完整对比：

**传统 aux loss**：
```
L_total = L_LM + λ · Σ_e f_e · p_e
```
- λ 是需要调的超参（太大 → 负载均匀但模型差，太小 → 负载不均）
- aux loss 的梯度直接进入模型参数（W_g 的梯度被 aux term 修改）
- V3 论文报告 aux loss 会降低下游任务性能 ~0.5-1%

**Aux-loss-free bias**（V3.2/V4）：
```
b_e ← b_e - η_bias · (1/E - f_e)   # 每步单独更新
score_for_topk = sqrtsoftplus(W_g·x) + b
weight = sqrtsoftplus(W_g·x)[topk_indices]   # 不带 b！
```
- b_e 不进主损失梯度（独立更新规则）
- b 只影响"谁被选中"（topk），不影响"选中后的权重"
- 不需要调 λ（只有 η_bias 一个超参，通常固定 0.001）
- V3.2 论文报告 MMLU 提升 +0.5% vs aux loss

本质差异：aux loss 试图"在损失空间里拉平 expert 使用率"，aux-loss-free 试图"在路由空间里调节 expert 被选中的概率"。前者是"结果优化"（已经污染主梯度），后者是"过程调控"（只在选择时干预）。

**面试官视角**：考察是否理解"梯度污染"这一根本问题。很多人知道 aux-loss-free 更好但不知道为什么。核心在于"b 不参与路由权重计算"这一关键约束——能准确说出这一点说明真正理解了 aux-loss-free。

**答题模板**：(1) 写出两种方法的更新公式，(2) 说明梯度污染的机制，(3) 强调 b 仅影响 topk 选择，(4) 给出 V4 的具体参数（η_bias=0.001, E=256）。

**延伸阅读**：主报告 CH4.3.2。


### Q10.13 Muon 和 AdamW 在更新机制上的根本区别是什么？为什么 Muon 更适合 MoE？

**简短回答**：AdamW 是"逐元素标量更新"：每个参数独立维护一阶/二阶动量，学习率逐元素自适应。Muon 是"矩阵更新"：对整个参数矩阵 W 的梯度做 Newton-Schulz 正交化（使更新方向 O' ≈ UV^T，O'·O'^T ≈ I），再 rescale 到与 AdamW 相当的 RMS 尺度。Muon 保留矩阵的几何结构（奇异值被拉平），避免 AdamW 的谱偏问题。

**详细解释**：
核心对比：

| 维度 | AdamW | Muon |
|------|-------|------|
| 更新粒度 | 逐元素（per-parameter） | 矩阵级（per-matrix） |
| 信息利用 | 局部梯度二阶矩 | 全局矩阵谱结构 |
| 奇异值约束 | 无（谱偏易退化） | 强制拉平到 1（满秩训练） |
| 自适应 | 逐元素学习率 | 全局 RMS rescale |
| 对 MoE 稀疏梯度的处理 | 无梯度的参数"v_t 过期" | 整体正交化吸收稀疏结构 |
| 计算开销 | O(d²) per step | $O(10 \cdot d^3)$ per step（Newton-Schulz × 10） |

Muon 更适合 MoE 的原因：
1. **稀疏梯度鲁棒**：MoE 中 99% 的 expert 参数每个 step 没有梯度。AdamW 的 v_t 在无梯度参数上只是 β_2 指数衰减，几乎没有新信息。Muon 对整个矩阵做正交化，sparsity 被"结构性吸收"。
2. **谱偏抑制**：MoE 的 expert 矩阵容易出现"几个大方向主导"（因为只有 6/256 的 expert 被激活），AdamW 无法抑制谱偏。Muon 每一步强制拉平奇异值。
3. **与 mHC 协同**：Muon 的满秩更新 + mHC 的 Lipschitz 约束 = 双重稳定性保证。

V4 的模块分工：Embedding/Head/Norm/gating 用 AdamW（小参数无谱结构），大矩阵（Attention/MoE/mHC comb）用 Muon。

**面试官视角**：这是一道"优化器架构决策"的高级题。很多候选人只会说"Muon 比 AdamW 好"，但需要能量化证据（Newton-Schulz 10 次迭代 × 每次 3 次矩阵乘 = 30 次 d³ FLOPs），和说明"为什么只在 MoE 上适用"。

**答题模板**：(1) 对比更新公式（AdamW element-wise vs Muon matrix-wise），(2) 解释 Newton-Schulz 做了什么，(3) 分析 MoE 稀疏梯度场景下的优势，(4) 说明 V4 的 AdamW/Muon 分工。

**延伸阅读**：主报告 CH6.1-6.4。


### Q10.14 CSA 和 HCA 的 trade-off 是什么？V4 为什么交替而不是全用一种？

**简短回答**：CSA（m=4, 带 Indexer 评分）信息保留好但压缩率低（KV 为原来的 1/4），HCA（m=128, 无 Indexer）极端高效但信息损失大（KV 为 1/128）。全用 CSA：FLOPs 仍太大；全用 HCA：质量下降。交替（21 CSA + 20 HCA + 2 滑窗）让 CSA 层"补细节"、HCA 层"省算力"，是信息-效率的逐层权衡。

**详细解释**：
CSA vs HCA 的量化对比（T=1M 单 token）：

| 维度 | CSA (m=4) | HCA (m=128) |
|------|----------|------------|
| 压缩后序列长度 | T/4 ≈ 262,144 | T/128 ≈ 8,192 |
| 压缩比率 | 1/4 | 1/128 |
| KV cache per layer（混合精度） | ~151 MB | ~4.8 MB |
| Attention FLOPs per token | T²/4 + T×512 | T²/128 + T×512 |
| Indexer 开销 | 有（Compressor + Hadamard + FP4 + topk） | 无（位置 topk） |
| 信息损失 | 小（4 个 token 压 1 个） | 大（128 个 token 压 1 个） |

交替策略的分析：
- 如果全部 43 层用 CSA：KV cache ≈ 43 × 151 MB ≈ 6.5 GB + KV 压缩开销 → 仍较大
- 如果全部 43 层用 HCA：KV cache ≈ 43 × 4.8 MB ≈ 0.2 GB → cache 极小但信息损失不可接受
- 交替（21 CSA + 20 HCA）：CSA 层每隔一层"补细节"（恢复 HCA 层丢失的局部信息），HCA 层每隔一层"降开销"（把 FLOPs/cache 压下去）

为什么不动态选（per-token 自适应）？动态选择需要额外的路由网络（类似 MoE gate），增加计算开销和训练复杂度。V4 的设计哲学是"在建模阶段固定结构，推理时零开销"——$compress_ratios$ 列表在训练前确定，推理时直接按 layer_id 查表。

**面试官视角**：这道题考察系统设计的 trade-off 思维。"全用一种"是最常见的 naive 回答，能分析交替的"隔层互补"逻辑是中级水平，能进一步分析"为什么不动态选"是高级水平。

**答题模板**：(1) 量化对比 CSA vs HCA 的数字，(2) 分析全用一种的问题，(3) 解释交替的"互补"机制，(4) 讨论为什么不动态选择。

**延伸阅读**：主报告 CH3.4。


### Q10.15 mHC 的 Sinkhorn-Knopp 投影为什么能保证信号传播稳定？

**简短回答**：Sinkhorn-Knopp 投影将 mHC 的 combination 矩阵 α_comb 约束为双随机矩阵（行和=1, 列和=1）。双随机矩阵的谱范数 σ_max ≤ 1（Perron-Frobenius 定理），因此每个残差层的信号范数满足 ‖α_comb · x‖ ≤ ‖x‖。86 个 mHC 残差串联后，信号传播构成 Lipschitz 常数为 1 的非扩张映射——信号不会爆炸也不会消失。

**详细解释**：
数学链条：
1. **Sinkhorn-Knopp**： $M \to  row_normalize(M) \to  col_normalize(M) \to  ...$ （20 次迭代后 M ≈ 双随机）
2. **双随机性质**： $M\cdot 1 = 1, 1^T\cdot M = 1, M \geq  0$
3. **Perron-Frobenius**：σ_max(M) = 1，特征向量 = 1/√c ·𝟏（全 1 向量）
4. **信号范数界**： $\|M\cdot x\|_2 \leq  \sigma_max(M) \cdot \|x\|_2 = \|x\|_2$ （信号不增）
5. **串联界**： $\|M_86 \cdot M_85 \cdot ... \cdot M_1 \cdot x\|_2 \leq  \|x\|_2$ （86 层不爆炸）

如果没有这个约束（原始 HC）：α_comb 可能是任意矩阵，σ_max 可以大到 10²-10⁴，86 层串联后信号指数级爆炸 → loss spike / NaN。

V4 中具体的参数：
- $hc_mult = 4$ ：4 通道 → α_comb 是 4×4 矩阵
- $hc_sinkhorn_iters = 20$ ：每次 mHC 残差做 20 次行列归一化
- $hc_eps = 1e-6$ ：数值稳定项
- 43 层 × 2 mHC/Block = 86 个 mHC 残差串联

**面试官视角**：这是一道检验数学基础的问题。Perron-Frobenius 定理是矩阵分析的核心工具，能结合到深度学习训练稳定性中说明有较强的理论功底。

**答题模板**：(1) 定义双随机矩阵，(2) 引用 Perron-Frobenius 定理证明 σ_max ≤ 1，(3) 推导 Lipschitz 常数为 1 的信号传播界，(4) 说明如果没有约束会出现什么（HC 的问题）。

**延伸阅读**：主报告 CH5.3。


### Q10.16 V4-Flash 的 KV cache 在 1M 上下文下实际占多少 GB？怎么算？

**简短回答**：V4-Flash 1M 上下文 KV cache 约 4 GB（含主 KV cache + Indexer KV cache）。核心计算：21 层 CSA × 262K 向量 × 576 bytes + 20 层 HCA × 8320 向量 × 576 bytes + 2 层滑窗 × 128 向量 × 1024 bytes + Indexer 开销 ≈ 4 GB。

**详细解释**：
完整计算（混合精度：RoPE dim=64 BF16, 非 RoPE dim=448 FP8）：

**主 KV cache**：
- CSA 层（21 层）：每层存 window=128 + T/4=262144 → 262,272 向量
  - 每向量 = 64×2 (RoPE BF16) + 448×1 (非 RoPE FP8) = 576 bytes
  - 21 × 262,272 × 576 ≈ 3.17 GB
- HCA 层（20 层）：每层存 128 + T/128=8192 → 8,320 向量
  - 20 × 8,320 × 576 ≈ 95.8 MB
- 滑窗层（2 层）：每层存 128 向量（纯 BF16）
  - 2 × 128 × 1024 ≈ 0.26 MB

**Indexer KV cache**（仅 CSA 层）：
- 21 层 × 262,144 向量 × 128 维 (index_head_dim) × 1 byte (FP8) ≈ 704 MB

**总计**：3.17 GB + 0.096 GB + 0.0003 GB + 0.704 GB ≈ **3.97 GB ≈ 4 GB**

加上 expert 权重（FP4, ~3.2 GB）、attention 权重、embedding 等，**总推理显存约 17 GB**。

**面试官视角**：这是一道"现场建账"题，检验能否从架构参数出发计算实际工程数字。不需要答案精确到小数点，但各成分的量级（TB/GB/MB）不能搞错。

**答题模板**：(1) 列出公式 $window_size + T/ratio$ ，(2) 分 CSA/HCA/滑窗三层计算，(3) 加 Indexer 开销，(4) 补充总显存 17 GB。

**延伸阅读**：主报告 CH3.5、Q9.9。


### Q10.17 V4-Flash 的单 token FLOPs 是 V3.2 的百分之多少？为什么？

**简短回答**：V4-Flash 在 1M 上下文下的单 token FLOPs 约为 V3.2 的 **10%**（V4 技术报告 Figure 1）。原因有三：(1) CSA/HCA 交替将 attention 从 $O(T^2)$ 降到 O(T²/4 + T²/128 + T·k)；(2) 激活参数 13B vs V3.2 的 37B（35%）；(3) top-k 从 8 降到 6（25% routed FLOPs 节省）。

**详细解释**：
FLOPs 拆解：

| 组件 | V3.2 (MLA, 37B) | V4-Flash (CSA/HCA, 13B) | 节省比 |
|------|----------------|------------------------|--------|
| Attention | O(T²·d) | O(0.5×T²/4 + 0.5×T²/128) · d + O(T·$k \cdot d$) | ~16× (1M 时) |
| MoE | 8 experts × FFN | 6 experts × FFN | 25% |
| 激活参数 | 37B | 13B | 65% |

关键计算（1M 上下文）：
- V3.2 attention：T²·d/128（MLA 的 head_dim=128）≈ 10¹² × 4096 / 128 ≈ 3.2×10¹³（单层近似，实际 MLA 比这复杂）
- V4-Flash attention：交替后 $528\cdot T^2 + 4.2\times 10^6\cdot T$ 量级（见主报告 CH3.5）
- 加上 MoE 的 25% 节省、激活参数的 65% 节省 → 总计 ~90% 节省

需要说明的是：V4-Flash 的 10% 是 **end-to-end** 的 FLOPs（含 attention + MoE + Norm + mHC 等全部），不只是 attention。10% 还包含了 CSA 的 Indexer 开销、mHC 的 Sinkhorn 开销、以及非 attention 部分的 FLOPs。

**面试官视角**：这道题要求能在面试中从 $O(T^2)$ 推导到 10%。关键是要理解"10% 不是 1/4 或 1/128"，而是多层压缩的叠加效果。能分层拆解 attention FLOPs、MoE FLOPs、其他 FLOPs 的节省是加分项。

**答题模板**：(1) 引用 V4 技术报告 Figure 1 的 10% 数字，(2) 分 attention/MoE/参数三层拆解，(3) 简要估算验证，(4) 说明 10% 是端到端数字含所有开销。

**延伸阅读**：主报告 CH3.5。


### Q10.18 V4-Flash 训练需要多少 GPU·天？（合理估算）

**简短回答**：V4 论文未公开准确数字。基于 C=6ND 规则估算：6 × 13B × 32T ≈ 2.5×10^24 FLOPs。假设 H800 集群利用率 ~45%（考虑 MoE 稀疏和通信），单卡有效 ~3.84×10^19 FLOPs/天，总计 ~65,000 GPU·天（单卡等效）。千卡集群约需 2-3 个月。这是粗略估算，真实数字取决于具体硬件和工程优化水平。

**详细解释**：
估算步骤：
1. 总 FLOPs ≈ 6 × N_active × T_tokens = 6 × 13×10^9 × 32×10^12 ≈ 2.5×10^24
2. H800 FP8 理论峰值：~989 TFLOPS
3. 有效利用率估算：
   - MoE 稀疏性：top-6/256 ≈ 2.3% expert 利用率（但 shared expert 永远激活）
   - 通信开销（all-to-all）：~10-15%
   - DualPipe pipeline bubble：~5-10%
   - 综合利用率：~40-50%
4. 单卡每天有效 FLOPs ≈ 989×10^12 × 86400 × 0.45 ≈ 3.84×10^19
5. GPU·天 ≈ 2.5×10^24 / 3.84×10^19 ≈ 65,000（等效单卡）

千卡集群下 ≈ 65,000 / 1000 ≈ 65 天（约 2 个月）。

不确定因素：实际 MoE 利用率可能低于 40%（expert 不均衡导致某些 expert 过载），FP8 训练的有效吞吐量取决于 kernel 优化程度，V4 的稳定性改进（mHC + Muon）可能减少了因 loss spike 重启的训练时间。

**面试官视角**：这道题的核心不是给出一个精确数字，而是展示"估算的方法论"——C=6ND → FLOPs → 利用率 → GPU·天 → 集群规模 → 时间。能说清楚每一步的不确定因素比给一个精确数字更重要。

**答题模板**：(1) 用 C=6ND 算总 FLOPs，(2) 假设利用率，（3）算 GPU·天，(4) 转换为集群规模和时间，(5) 诚实说明不确定因素。

**延伸阅读**：主报告 CH9.2。


### Q10.19 256 个 expert 的显存占用量是多少？FP4 量化后省多少？

**简短回答**：V4-Flash 的 256 个 routed expert 的 $\mathrm{SwiGLU}$ 参数（w1, w2, w3 各 $4096 \times 2048$ ）共 6.44B 参数。BF16 存储需要 12.9 GB，FP8 需要 6.44 GB，FP4 只需要 **3.22 GB**（省 75% vs BF16）。单 expert 约 25.2M 参数 → BF16: 50.4 MB, FP4: 12.6 MB。

**详细解释**：
逐层计算：

每个 expert 的 $\mathrm{SwiGLU}$ 结构：
- `w1 (gate)`: 4096 × 2048 = 8,388,608 params
- `w2 (down)`: 2048 × 4096 = 8,388,608 params
- `w3 (up)`: 4096 × 2048 = 8,388,608 params
- 合计 per expert ≈ 25.2M params

256 个 expert 合计：
- 25.2M × 256 ≈ 6.44B params
- BF16: 6.44B × 2 bytes = 12.88 GB
- FP8: 6.44B × 1 byte = 6.44 GB
- FP4: 6.44B × 0.5 bytes = 3.22 GB

但 FP4 不是"纯" 4-bit：还需要存 block-wise scale（E8M0 格式，block=32）。scale 开销：
- 256 experts × 3 矩阵 × ($4096 \times 2048$ / 32) × 1 byte = 256 × 3 × 262,144 × 1 ≈ 201 MB

实际 FP4 存储 ≈ 3.22 GB + 0.2 GB ≈ 3.42 GB，vs BF16 的 12.88 GB，节省约 73.5%。

加上 shared expert（25M params, FP8 ≈ 25 MB），整个 MoE 层的权重存储：
- BF16: ~12.9 GB / FP8: ~6.5 GB / FP4: ~3.4 GB

**面试官视角**：考察能否进行精确的参数-显存转换。常见的错误是忘了 shared expert 或算错 $\mathrm{SwiGLU}$ 的 3 个矩阵（很多人只算 2 个）。

**答题模板**：(1) 写出单个 expert 的 3 矩阵参数，(2) 乘 256 得总数，(3) 三种精度对比，(4) 补充 scale 开销。

**延伸阅读**：主报告 CH7.2、Q9.9。


### Q10.20 Sinkhorn 算法的收敛性如何？为什么 V4 选 20 次迭代？

**简短回答**：Sinkhorn 算法将任意非负矩阵投影到双随机矩阵，收敛速度为 $O(log(1/\delta))$ 次迭代达到 δ 误差（Franklin and Lorenz 1989）。V4 选 20 次是因为：(1) 20 次后 4×4 矩阵的行和/列和误差已 < 10^-4；(2) 更少（如 5 次）约束不紧致信号放大；(3) 更多（如 100 次）浪费算力（收敛已饱和）。

**详细解释**：
Sinkhorn 的数学收敛性：
- 每次行/列归一化都使 M 向 Birkhoff polytope 的 KL 投影更近一步
- 误差以 `O(1/n)` 衰减（每次迭代）——线性收敛
- 对 4×4 小矩阵，10 次迭代误差 ≈ 10^-2, 20 次 ≈ 10^-4, 30 次 ≈ 10^-6

V4 的 20 次选择逻辑：
1. $hc_mult=4$ 意味着 α_comb 是 4×4 矩阵——矩阵小，每次迭代只需 4×4=16 次除法/比较
2. 20 次迭代的算力开销：20 × 2(normalizations) × 4×4 = 640 次除法 per mHC
3. 86 个 mHC 残差 × 640 = ~55K 次除法 per token——在 GEMM 开销面前可忽略
4. 误差 10^-4 意味着行和 = 1.0000±0.0001，σ_max = 1.0000±0.0001 → Lipschitz 约束非常紧

为什么不是 5 次？5 次后误差约 10^-2，σ_max 可能达到 1.01-1.02，86 层串联后 $1.02^86 \approx  5.5$——信号放大 5.5 倍，不够稳定。

为什么不是 100 次？100 次后误差 < 10^-6，但 20 次已经足够紧（10^-4 → σ_max 偏差 < 10^-5，86 层效应 < 10^-3）。额外 80 次迭代是纯粹浪费。

**面试官视角**：这道题考察数值方法的理解深度。能引用 Franklin and Lorenz 1989 的收敛性结果是高级答案。实际面试中能说出"O(1/n) 收敛 + 小矩阵 20 次够用"就已经是优秀水平。

**答题模板**：(1) 引用 Sinkhorn 收敛速度 O(log(1/δ))，(2) 分析 4×4 小矩阵每次迭代开销，(3) 解释 20 次的误差和 86 层串联效应，(4) 说明为什么不更少或更多。

**延伸阅读**：主报告 CH5.3。


### Q10.31 hash routing 的原理是什么？为什么不用在所有层？

**简短回答**：hash routing = $token_id % 256$ 直接查表决定 expert 分配， $O(1)$ 延迟 vs score routing 的 $O(d \times E)$ 矩阵乘法。前 3 层用 hash 省 ~3.1M FLOPs/token 且不影响质量（浅层语义简单），全 43 层用 hash 则 PPL 退化显著（深层语义复杂必须 score routing）。3 层是论文 ablation 的甜点。

**详细解释**：
Hash routing 的完整机制：

**(1) 核心原理**：在 $Gate.__init__$ （`model.py` L554-L560）中，为前 3 层（ $layer_id < num_hash_layers$ ）构造一个不可训练的查找表 $self.tid2eid \in  R^{129280\times 6}$ （vocab_size × top-k），每个 token ID 对应 6 个固定的 expert 索引。前向时直接 $indices = self.tid2eid[input_ids]$ ，一次 GPU gather 操作，完全跳过 $W_g\cdot x$ （ $4096 \times 256$ matmul）+ $\mathrm{sqrtsoftplus}$ （256 个 exp/log）+ `topk`（256 个元素排序）。

**(2) 算力节省数字**：
- Score routing 单次开销：matmul $4096\times 256 \approx  1.05M FLOPs$ + $\mathrm{sqrtsoftplus}$ $256\times ~10 ops \approx  2.5K FLOPs$ + topk $256\times log(256) \approx  2K FLOPs$ = ~1.05M FLOPs/layer/token
- Hash routing：gather $6\times int32 \approx  可忽略 $
- 前 3 层省：3 × 1.05M ≈ **3.1M FLOPs/token**。对 batch=1 的 decode 阶段，这是毫秒级的非平凡节省
- 全 43 层省：43 × 1.05M ≈ **45M FLOPs/token**——量级可观，但代价是 PPL 退化

**(3) Trade-off：为什么不全都用 hash routing？**

| 层深度 | 语义特征 | Hash 适用性 | Score 必要性 |
|--------|---------|------------|------------|
| 浅层（0-2） | 词法/局部句法（拼词、词性、相邻搭配） | 高——token 本身的 ID 就能决定需要哪些"基础语言 expert" | 低——Gate 还没学会"哪个 expert 擅长什么"，score 路由 ≈ 随机 |
| 中层（3-20） | 句法结构/局部语义（短语、从句、指代） | 中——token ID 信息不够 | 高——需要根据上下文动态选择 expert |
| 深层（21-42） | 全局语义/推理（跨段落逻辑、长链推理） | 低——同一 token 在不同上下文中含义完全不同 | 极高——expert 选择高度依赖上下文 |

V4 论文的 ablation 实验（推断）表明：3 层 hash + 40 层 score 的组合在 PPL/MMLU 上 **略优于全 score 或全 hash**。原因是浅层的 attention 分布尚未形成"特定 expert 对特定 token"的精细偏好——hash 分配的"随机但均匀"反而比"score 路由但噪声大（因为浅层表示尚未稳定）"更有效。

全 43 层 hash 会怎样？浅层还行，但深层遇到"bank"这个词——在金融上下文中需要金融 expert，在河流上下文中需要地理 expert——hash 按 token ID 查表，两个"bank"分到相同的 6 个 expert，深层语义完全丢失。PPL 退化估计 > 5-10%。

**(4) 实战代码路径**：`config.json` 中 $num_hash_layers=3$ ，`Gate.forward`（L568-L573）：
```python
if self.layer_id < self.num_hash_layers:
    indices = self.tid2eid[input_ids]  # O(1) gather
    weights = torch.ones_like(indices, dtype=torch.float32) / self.top_k
else:
    scores = F.softplus(F.linear(x, self.weight)).sqrt()  # Score routing
    ...
```

🎤 **面试官视角**：考察对 MoE 路由效率 trade-off 的理解深度——不仅仅是"知道 hash routing"而是"理解为什么前 3 层用 hash、深层用 score"。很多候选人能说出 hash 更快但说不清"为什么浅层可以深层不行"。

🎯 **答题模板**：(1) 核心——hash routing = $token_id % 256$ 直接查表 $O(1)$ vs score routing 的 $O(d \times E)$ 矩阵乘；(2) 数字——前 3 层省 ~3.1M FLOPs/token，全 43 层 PPL 退化显著；(3) trade-off——浅层语义简单 hash 够用，深层语义复杂必须 score routing，3 层是 ablation 甜点；(4) 实战——`config.json` $num_hash_layers=3$ ，`Gate.forward` L568-573 判断 $layer_id < 3$ 。

**延伸阅读**：主报告 CH4.3.1；Q8.22；`inference/model.py` L554-L573 `Gate`。


## 情境题 11 个

### Q10.21 如果让你设计一个 100B 参数的 MoE 模型，你会选多少 expert，top-k 取多少？为什么？

**简短回答**：对于 100B 激活参数的 MoE，推荐 E=128 expert, k=8。选择依据：(1) E 不宜太大（256+ 的路由开销和负载均衡难度上升）；(2) k=8 是"质量/成本"的甜点（学术界和工业界大量实验验证）；(3) 1 个 shared expert 始终激活（~10% 激活参数），覆盖通用知识。

**详细解释**：
设计决策树：

**E（expert 数量）的选择**：
- E=64：每个 expert ≈1.5B 参数，太大 → 单 expert 训练慢、显存大
- E=128：每个 expert ≈780M 参数，平衡——单个 expert 在单 GPU 上舒适
- E=256：每个 expert ≈390M 参数，但路由（128 选 8）精度下降，aux-loss-free 收敛慢（V4 需要 32B tokens 才收敛）
- E=512：路由精度更低，通信开销（all-to-all expert dispatch）成为瓶颈

**k（top-k）的选择**：
- k=4：计算省但容量低（4 expert 的输出难以覆盖所有知识）
- k=8：甜点——8 个 expert 既有足够的容量，又不会让路由过于"平均化"
- k=12：容量增加但边际收益递减，且 MoE 计算量增加 50%

**公式估算**：总参数 = $E \times  3 \times  d \times  d_ff + Base_params$ 。假设 d=5120, d_ff=2560：
- Base（attention+embedding+norm）≈ 10B
- 127 expert（128-1 shared）× 3 × 5120 × 2560 ≈ 50B（FP8: ~50 GB）
- + 1 shared expert ≈ 0.4B
- 总计 ≈ 60B 非嵌入参数（100B 总参需调整 d/d_ff）

**面试官视角**：这道题考察系统设计能力和对 MoE scaling law 的理解。优秀的回答需要权衡多个维度（质量、速度、负载均衡、硬件约束），而不是只凭感觉选一个数。能引用 V4 的 E=256/k=6 和 V3 的 E=256/k=8 作为参考点是加分项。

**答题模板**：(1) 列出设计目标（100B 激活），(2) 分步决策 E 和 k（各给 2-3 个候选 + 推演），(3) 估算参数和显存，(4) 说明 shared expert 和负载均衡方案。

**延伸阅读**：主报告 CH4.1。


### Q10.22 KV cache 爆了怎么办？请给 5 种解决方案及其 trade-off

**简短回答**：5 种 KV cache 压缩方案按"质量损失从小到大"排列：(1) MQA/GQA（head 维压缩，质量损失小）；(2) FP8 混合精度（位宽压缩，几乎无损）；(3) 滑动窗口（时间维截断，零实现成本）；(4) Token 丢弃/eviction（选择性丢弃，需注意关键 token 保护）；(5) KV 压缩/量化（如 CSA/HCA，需重新训练或 adapter 调优）。

**详细解释**：
五种方案详细分析：

| 方案 | 压缩倍数 | 质量损失 | 实现难度 | 适用场景 |
|------|---------|---------|---------|---------|
| MQA/GQA | head_h/1 或 head_h/g | 小（~1% PPL） | 低（改 config 重训） | 所有场景 |
| FP8/INT8 KV | 1.8-2× | 极小（<0.1% PPL） | 中（需量化 kernel） | 已训练模型快速部署 |
| 滑动窗口 | 任意（截断到 w） | 中（丢失长程信息） | 低（改 attention mask） | 短程任务为主 |
| Token eviction | 2-4× | 中（可能丢失关键 token） | 中（需重要性评分） | 长文档 QA |
| KV 压缩 (CSA/HCA) | 4-128× | 中-大（需 QAT） | 高（需重新设计架构） | V4 级 1M 上下文 |

推荐组合策略（V4 的做法）：MQA + FP8 混合精度 + CSA/HCA 交替 + 滑窗（四层压缩接力）。单靠一个方案无法把 1M 上下文压到 4 GB——需要多维度的组合拳。

**面试官视角**：这道题考察"系统设计的组合思维"。很多候选人只列 1-2 种方案，能列出 5 种并分析它们的组合效果才是高级答案。关键的 insight 是"没有一个方案能单独解决问题，需要多层压缩接力"。

**答题模板**：(1) 列出 5 种方案，(2) 按"架构层/精度层/算法层"分类，(3) 量化每种方案的压缩比和质量 trade-off，(4) 推荐组合策略（参考 V4 的四层接力）。

**延伸阅读**：主报告 CH3.5。


### Q10.23 训练中某几个 expert 被频繁选中（负载严重不均衡），怎么 debug？

**简短回答**：MoE 负载不均衡的 debug 流程分四步：(1) 检查 aux-loss-free bias 是否正常更新（频率统计是否准确）；(2) 检查前几层的 attention 分布是否异常（某些层可能"吸引"了全部 token）；(3) 检查 Gate 权重是否出现过大的奇异值（导致 $W_g\cdot x$ 在少数 expert 上极大）；(4) 检查 hash routing 层的 hash 函数是否分配均匀。

**详细解释**：
Debug 清单：

**Step 1：bias 更新检查**
- 打印 aux-loss-free b_e 的值——过载 expert 的 b 应该为负（降低被选中概率），欠载 expert 的 b 应该为正
- 检查 b 的更新步长 η_bias = 0.001 是否合理（太小会导致收敛慢）
- 确认 b 没有意外参与路由权重计算（检查代码中 $original_scores$ 的使用）

**Step 2：attention 分布检查**
- 打印各层的 attention entropy——如果某层几乎全是 self-attention（entropy ≈ 0），该层的输出包含了"位置指纹"，会让 Gate 对特定位置产生偏见
- 检查 CSA 的 Indexer 评分是否在少数头上极度集中

**Step 3：Gate 权重检查**
- 打印 W_g 的奇异值分布——如果有几个显著大的奇异值，gate 输出被少数方向主导
- 检查 $\mathrm{sqrtsoftplus}$ 前后得分的分布——是否大部分 expert 分数接近 0
- 对比 $\mathrm{sigmoid}$/$\mathrm{softmax}$/$\mathrm{sqrtsoftplus}$ 的分布差异（可以临时换评分函数诊断）

**Step 4：hash routing 检查**
- 打印 `tid2eid` 中各 expert 的频率分布——理想情况应该接近 1/256
- 检查 tokenizer 是否对某些高频 token 分配了相同的 hash target

**额外工具**：训练框架应提供 per-expert 负载监控（如 TensorBoard 实时展示 256 个 expert 的频率柱状图），并在不均衡超过阈值时告警。

**面试官视角**：这道题考察 debug 的系统方法论。优秀的回答应该是"分层的"——从顶层现象（bias 更新）到中层原因（attention 分布）到底层根因（Gate 权重奇异值），有条理地排查。能提到具体的检查代码/工具（如打印 b_e、画奇异值分布）比泛泛而谈更好。

**答题模板**：(1) 先确认现象（哪些 expert 过载/欠载），(2) 分层排查（bias → attention → Gate → hash），(3) 每层给出具体检查项，(4) 推荐监控工具。

**延伸阅读**：主报告 CH4.3。


### Q10.24 训练中出现 loss spike（损失突然飙升），你会怎么定位原因？

**简短回答**：Loss spike 的排查按"数据 → 模型 → 优化器 → 数值"四层逐级定位。先查是否有坏数据（NaN/Inf 输入），再查模型输出是否异常（activation 的 mean/std 逐层变化），然后查优化器状态（gradient norm、learning rate），最后查浮点异常（FP8/FP4 溢出、除零等）。

**详细解释**：
四层排查法：

**Layer 1：数据（最快检查）**
```python
# 检查输入 token IDs 是否有异常值
assert input_ids.min() >= 0 and input_ids.max() < vocab_size
# 检查 embedding 输出是否有 NaN
assert not torch.isnan(embed_out).any()
```

**Layer 2：模型激活（逐层检查）**
```python
# 记录 loss spike 前后各层的激活统计
for i, layer in enumerate(model.layers):
    # 检查 activation 的 mean/std
    print(f"Layer {i}: mean={x.mean():.4f}, std={x.std():.4f}, max={x.max():.4f}")
    # 异常信号: std 突然增大 10x 或出现 NaN
    x = layer(x)
```

对于 V4 特别要检查：
- swiglu_limit 是否被触发（up 投影输出超过 10.0）
- attention logits 是否异常大（检查 $Q \cdot K$^T 的最大值）
- mHC 的 α_comb 是否成功保持双随机（检查行和/列和是否 ≈ 1）

**Layer 3：优化器状态**
```python
# 检查 gradient norm
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
print(f"Gradient norm: {total_norm:.2f}")
# 检查 learning rate schedule 是否有跳变
# 检查 Muon 的 Newton-Schulz 迭代是否发散（σ 是否偏离 1）
```

**Layer 4：数值精度**
```python
# 检查 FP8/FP4 量化是否有溢出
# 关键在于 scale 计算: amax / fp_max 是否在合理范围
# 检查 Sinkhorn 迭代中是否有除零（eps 是否生效）
```

**V4 特有的防护**：mHC 的 Lipschitz 约束（σ_max ≤ 1）本身就是 loss spike 的第一道防线——如果某个 layer 的激活突然放大，mHC 会限制信号传播不超过 1×。如果仍然 spike，很可能是 mHC 约束本身出了 bug（如 Sinkhorn 未收敛）。

**面试官视角**：这道题考察大规模训练的工程经验。有经验的工程师会先看最简单的（数据），再层层深入。能说出 V4 特有的检查点（swiglu_limit, mHC 双随机检查）说明对架构理解很深。

**答题模板**：(1) 建立四层排查框架，(2) 每层给出具体检查代码/指标，(3) 特别说明 V4 特有检查项，(4) 推荐从 loss spike 恢复的策略（回滚 checkpoint、降低 LR）。

**延伸阅读**：主报告 CH5、CH6、CH7.2。


### Q10.25 1M 上下文推理时 P99 latency 太高，怎么优化？

**简短回答**：1M 上下文 P99 latency 优化的重点在两个方面——(1) 减少 attention 的随机 gather（top-k 的 k 是否可减、gather 模式是否可优化）；(2) 减少 memory-bound 开销（FP4 反量化是否快、KV cache 是否在 HBM 中）。可以考虑的方案：降低 k 从 512 到 256；对长上下文做预先分块（chunked prefill）；用 speculative decoding 抵消 attention 延迟。

**详细解释**：
P99 延迟的瓶颈分析：

**瓶颈 1：稀疏 attention 的随机 gather**
- $sparse_attn_kernel$ 按 $topk_idxs$ 从 KV cache 中 gather 512 个向量
- HBM 随机读取延迟 ≈ 500-800 cycles（vs 顺序读取 ~100 cycles）
- 优化方向：降低 k（512→256，约 2x 减少 gather）、对 gather 索引预排序（提升 cache line 利用率）

**瓶颈 2：KV cache 加载**
- 1M 上下文下，即使压缩后 KV cache 也需要从 HBM 加载
- 混合精度（FP8）虽然减半了容量，但加载时间仍是 O(T/m)
- 优化方向：用 on-disk KV cache（V4 论文 §3.5.2 支持），利用 SSD 存储长尾 KV，热点 KV 保留在 HBM

**瓶颈 3：Expert 权重加载（memory-bound）**
- Decode 时每次加载 6 个 expert 的 FP4 权重（约 75 MB），在 batch=1 下是 memory-bound
- 优化方向：expert 权重预加载到 shared memory；将多个 token 的 expert 加载批量处理（batching）

**推荐的优化策略（按优先级排序）**：
1. 降低 $index_topk$ 从 512 到 256（实验验证 PPL 影响）
2. 启用 on-disk KV cache（V4 已支持）
3. 用 MTP 层做 speculative decoding（1.2-1.5x 加速）
4. 升级硬件（H200 比 H100 带宽大 43%）

**面试官视角**：这道题考察系统性能优化的方法论。优秀的回答需要先 profiling（找到瓶颈），再优化（针对瓶颈设计方案），最后验证（测量 P99 改善）。只列方案不分主次是初级水平。

**答题模板**：(1) 先 profiling 确定瓶颈（gather/加载/计算），(2) 按瓶颈列出优化方案，(3) 给出优先级排序，(4) 估算预期效果（如降 k: ~40% latency 减少）。

**延伸阅读**：主报告 CH3.5、CH9.2。


### Q10.26 FP4 量化后模型 PPL 退化严重，怎么补救？

**简短回答**：FP4 PPL 退化的补救措施有 6 种：(1) 减小 block size（32→16，更细粒度 scale 补偿精度）；(2) 切换到 QAT 训练（而非 PTQ）；(3) 对 outlier 通道单独处理（如保留 top-1% 通道用 FP8）；(4) 增加 swiglu_limit 钳制力度；(5) 调整 Gate 评分的温度（让 expert 选择更平滑）；(6) 对关键层（如前 2 层和最后 2 层）使用 FP8。

**详细解释**：
方案分析和优先级：

**方案 1：更细 block size（16 vs 32）**
- Block=16 时量化粒度翻倍，精度提升约 30%（outlier 影响更局部）
- 代价：scale 存储翻倍（~400 MB → ~800 MB），仍可接受

**方案 2：QAT（量化感知训练）**
- 如果当前是 PTQ，切换到 QAT 是最有效的单点改进
- PTQ 到 QAT 的 PPL 退化可以从 5-10% 降到 <0.5%
- 代价：需要完整的训练流程

**方案 3：outlier 通道保留 FP8**
- V4-Flash 的 RoPE 维度（最后 64 dim）保留 BF16 就是这种思路的体现
- 可以对 expert 权重做 per-channel outlier 检测，top-1% 异常通道保持 FP8
- 额外存储开销 ~1%

**方案 4：调整 swiglu_limit**
- 将 $swiglu_limit$ 从 10.0 降到 6.0（与 FP4 max=6.0 对齐）
- 减少 soft overflow 的影响

**方案 5：Gate 温度调整**
- 在 $\mathrm{sqrtsoftplus}$ 之前除以温度 `T>1`，让 expert 选择更均匀（减少"选错 expert"的影响）
- 实验验证最优 T

**方案 6：关键层 FP8**
- 浅层（layer 0-1）和深层（layer 41-42）用 FP8（总开销 +2 GB，PPL 改善 0.5-1%）

**面试官视角**：这道题考察 FP4 量化的工程经验。优秀的回答要能区分"改进精度"（方案 1-3）和"迂回策略"（方案 4-6），并给出量化的预期效果。

**答题模板**：(1) 先确认是 PTQ 还是 QAT（决定方案优先级），(2) 按效果排序列出 6 种方案，(3) 量化每种方案的效果和代价，(4) 推荐组合策略（如 QAT + block=16 + outlier 保护）。

**延伸阅读**：主报告 CH7.2。


### Q10.27 长文本（>100K）推理时模型性能突然下降，可能是什么原因？

**简短回答**：长文本性能下降的常见原因有三类：(1) 位置编码问题——RoPE 频率在超出训练长度后"旋转过度"，attention 无法正确分辨远近位置；(2) KV cache 压缩过度——HCA 层的 m=128 压缩丢失了关键信息；(3) attention 分布退化——随着序列增长，attention 趋向均匀分布（"lost in the middle"现象），模型无法关注到关键位置。

**详细解释**：
三类原因深入分析：

**原因 1：位置编码（最常见）**
- 症状：性能在某个长度阈值（如 64K 或 128K）突然下降
- 根因：RoPE base=10000 在未经 YaRN 扩展时的有效范围约为 8×base ≈ 80K 维度圈数
- 诊断：检查 RoPE 的频率分布——长文本后半部分的 attention score 是否接近常数？
- 修复：YaRN 扩展（factor 增加、调整 β_fast/β_slow）

**原因 2：KV 压缩信息损失**
- 症状：需要跨长距离推理的任务（如"根据前文第 X 页内容回答"）失败
- 根因：HCA 层 m=128 把 128 个 token 的信息压缩为 1 个向量，长距离检索精度下降
- 诊断：检查 HCA 层的 attention 分布——压缩位置是否被正确关注？
- 修复：增加 CSA 层比例（减少 HCA 层）、减小 HCA 的 m'、或引入 content-aware 压缩

**原因 3：Attention 退化（"lost in the middle"）**
- 症状：模型能回答"开头几段"和"结尾几段"的问题，但中间部分的信息丢失
- 根因：LLM 的 attention 天然偏向序列两端——开头有"task setup"信号，结尾有"recency bias"
- 诊断：在 100K 文本的不同位置插入事实并测试 recall 率
- 修复：position-based attention bias、attention sink 调大、或显式的位置编码增强

**面试官视角**：这道题考察长上下文 LLM 的实战经验。能区分"位置编码问题"和"attention 退化问题"是关键——它们症状相似但根因不同。

**答题模板**：(1) 先确认下降的位置阈值，(2) 区分三大类原因，(3) 给出每类的诊断方法，(4) 推荐针对性修复。

**延伸阅读**：主报告 CH7.1。


### Q10.28 如果要把 V4-Flash 部署到边缘 GPU（如 24GB 显存），你会怎么做？

**简短回答**：24GB 边缘 GPU 部署 V4-Flash 的关键挑战是总显存（推理约需 17 GB + 权重 ~12 GB），勉强可放入但无余量。优化策略：(1) 量化 expert 从 FP4 到 INT4/NF4（再省 ~0.5 GB）；(2) 限制最大上下文长度到 128K（KV cache 从 4 GB 降到 ~0.5 GB）；(3) offload 部分层到 CPU（如后 10 层放到 CPU 内存）；(4) 使用 4-bit 量化 attention 权重；(5) batch=1 推理（无 batch 开销）。

**详细解释**：
边缘部署的显存预算（24GB 目标）：

| 组件 | 原方案 | 优化后 |
|------|--------|--------|
| Expert 权重（FP4） | 3.4 GB | 2.5 GB（NF4 + block=16） |
| Shared + Attention + Embedding | ~5 GB | 3 GB（4-bit 量化 attention） |
| KV cache (1M) | ~4 GB | 0.5 GB（限制到 128K） |
| mHC + Norm + 临时 | ~3 GB | 2 GB |
| 激活 buffer | ~5 GB | 3 GB |
| **总计** | **~20 GB** | **~11 GB** |

限制到 128K 的 KV cache 计算：
- CSA 21 layer: (128 + 131072/4) × 576 × 21 ≈ 397 MB
- HCA 20 layer: (128 + 131072/128) × 576 × 20 ≈ 13 MB
- 合计 ≈ 0.41 GB + Indexer ≈ 0.5 GB

额外优化：
- CPU offload（mmap 模式）：不常用的 expert（尾部 80% expert）权重存在 CPU，用时加载
- Layer offload：后 10 层整个 offload 到 CPU（增加 ~2x 延迟但省 ~4 GB）
- 使用 AWQ/GPTQ 等成熟 INT4 量化工具替代 V4 原生的 FP4 kernel

**面试官视角**：这道题考察模型部署的显存管理经验。优秀的回答需要建"显存账"——逐一列出组件、说明压缩手段、量化预期节省。能区分"权重显存"和"激活显存"的优化策略是关键。

**答题模板**：(1) 算原方案的显存账，(2) 列出优化手段并量化节省，(3) 优先推荐上下文限制（最大杠杆），(4) 给出最终可行性判断。

**延伸阅读**：主报告 CH7.2、Q9.9。


### Q10.29 如果你只能保留 sliding window attention（无压缩），怎样改造 V4-Flash 使它在长上下文上仍可用？

**简短回答**：纯 sliding window attention 下，需要三管齐下：(1) 增大 window size 到 4096-8192（覆盖更多局部上下文）；(2) 在特定层插入 global token（如每 128 个 token 标记一个"摘要 token"，参与所有位置的 attention）；(3) 在模型输入端做文本分块（chunking），每块独立处理后用 cross-attention 融合。这样在 1M 上下文下，KV cache 约 50-100 GB（vs CSA/HCA 的 4 GB），牺牲效率但保留质量。

**详细解释**：
纯滑窗 + 无压缩的改造方案：

**方案 1：增大 window（最简单）**
- window=4096（vs V4 的 128）：KV cache per layer = 4096 × 512 × 2 bytes (BF16) = 4.2 MB
- 43 层：43 × 4.2 MB ≈ 180 MB per sample（1M 上下文，decode 时只存 window 个 KV）
- 问题：prefill 时仍然要算所有 token 的 attention → 无法解决 T² 问题

**方案 2：Global token（滑动窗口 + 稀疏全局注意力）**
- 每 g=128 个 token 选 1 个为"global token"（或 learnable summary token）
- Global token 参加所有位置的 attention（全局 dense），普通 token 只在 window 内交互
- KV cache = window KV + global KV：T/128 × 512 × 2 bytes = 1M/128 × 1024 ≈ 8 MB/layer
- 43 层 × 8 MB ≈ 344 MB（远小于 4 GB 的完整 KV cache）
- 这是 BigBird/Longformer 的经典范式

**方案 3：Chunked cross-attention**
- 将 1M 上下文切成 B=256 个 chunk（每个 ~4096 tokens）
- 每个 chunk 先过 encoder（计算 chunk 内 attention）
- decoder 用 cross-attention 从各 chunk 的 summary 中获取全局信息
- KV cache = B × d × 2 bytes = 256 × 4096 × 2 ≈ 2 MB/layer × 43 ≈ 86 MB

**方案 2 是最推荐的单点改造**——实现复杂度适中，质量损失可控（global token 保留了跨 chunk 信息流），KV cache 约 350 MB（vs 4 GB 的 CSA/HCA 方案）。代价是 global token 的 dense attention 在 1M 下仍需要 O(T²/g) FLOPs。

**面试官视角**：这道题考察在没有"黑科技"（CSA/HCA）时的替代方案设计能力。优秀的回答需要引用已知文献（BigBird/Longformer），并量化每个方案的 KV cache 和 FLOPs。

**答题模板**：(1) 列出 3 种方案，(2) 量化每种方案的 KV cache 和 FLOPs，(3) 推荐方案 2（global token）并说明理由，(4) 与 CSA/HCA 的对比。

**延伸阅读**：主报告 CH3.2、CH3.4。


### Q10.30 V4-Flash vs Llama-4 在 benchmark 上的对比如何？各自的优劣势在哪？

**简短回答**：V4-Flash（284B/13B）和 Llama-4（待定参数）在公开 benchmark 上的直接对比受限于发布时间。基于 V4 论文 Figure 1 的数据，V4-Flash-Max 在 Reasoning 上接近 GPT-5.2 和 Gemini-3.0-Pro，在 Knowledge 上因参数规模限制低于 V4-Pro（49B 激活）。Llama-4 系列通常更注重多模态（视觉+语言），在纯文本推理上 V4 的 1M 上下文 + FP4 高效推理是独特优势。

**详细解释**：
能力雷达图分析：

| 维度 | V4-Flash（284B/13B） | V4-Pro（1.6T/49B） | 特点 |
|------|---------------------|--------------------|----|
| Knowledge (MMLU-Pro, HLE) | 中（参数限制） | 高（接近 Gemini-3.1-Pro） | 显式参数-知识正相关 |
| Reasoning (AIME, GPQA) | 高（Think Max 下接近 GPT-5.2） | 极高（SOTA open model） | 推理能力对参数不太敏感 |
| Coding (Codeforces) | 中-高 | 极高（3206 rating） | 需要 RL 后训练 |
| Agent (SWE-bench, Terminal) | 中-高 | 极高（SOTA） | 需要 OPD 后训练 |
| Long Context (1M) | 极高（83.5% MRCR） | 极高 | V4 独特优势 |
| 推理效率 (FLOPs) | 极高（10% of V3.2） | 高（27% of V3.2） | V4 独特优势 |
| 部署门槛 | 低（8×H100） | 高（多机） | Flash 定位优势 |

Llama-4 系列（Meta 未在 2026-06 前发布完整 benchmark，以下为基于前代和趋势的推测）：
- 优势：多模态原生支持（视觉/语言联合训练）、全球化多语言覆盖、Meta 的硬件资源更充裕
- 劣势：MoE 架构可能不如 V4 激进（不一定有 256 expert），1M 上下文的效率优化可能不如 V4 的 CSA/HCA 组合

V4-Flash 的独特优势：
1. **1M 上下文效率**：CSA+HCA+FP4 组合让 1M 在单节点可跑，Llama-4 尚未公开类似能力
2. **三推理模式**：Non-think/Think High/Think Max 让用户在不同场景下选不同的计算预算
3. **开源 MIT license**：比 Llama-4 的社区许可（通常有限制）更友好

**面试官视角**：这道题考察竞品分析和行业视野。优秀的回答能从"知识/推理/代码/Agent/效率/部署"6 个维度做结构化对比，而不是泛泛说"V4 好"或"Llama 好"。诚实承认"Llama-4 在多模态上更强"比只说 V4 的优势更可信。

**答题模板**：(1) 建立 6 维对比框架，(2) 逐维分析 V4-Flash 和 V4-Pro，(3) 补充 Llama-4 的优劣势推测，(4) 总结 V4-Flash 的独特卖点（1M 效率、三模式、MIT）。

**延伸阅读**：主报告 CH1.3、CH9.3；V4 论文 Figure 1。


### Q10.32 如果要给 V4-Flash 加一个新 attention 机制，你从代码层面怎么最小改动实现？你会修改哪些文件？

**简短回答**：最小改动方案只改 3 个文件：`model.py`（Attention 类加新分支）、`kernel.py`（新 attention 的 TileLang kernel）、`config.json`（加 $use_new_attn: true$ 开关）。核心策略是"if-else 分支 + 单层验证 → 全层扩展 + PPL/tokens/s 对比"，确保改动可控、可回滚。

**详细解释**：

**(1) 改动范围——只改 3 个文件**

| 文件 | 改动内容 | 改动量 |
|------|---------|--------|
| `inference/model.py` | $Attention.__init__$ ：读取 $use_new_attn$ 开关；`Attention.forward`：加 $if self.use_new_attn: ... else: original_code$ 分支 | ~20 行 |
| `inference/kernel.py` | 用 TileLang 写新 attention kernel（与 $sparse_attn_kernel$ 同风格：`@tilelang.jit` 装饰器 + $T.prim_func$ 定义 + online $\mathrm{softmax}$ + attn_sink 兼容） | ~100-150 行 |
| `config.json` | 加 $"use_new_attn": true$ （或 `false` 作为默认关闭） | 1 行 |

不需要改的文件：`generate.py`（生成循环不变）、`convert.py`（权重转换不变）、`encoding/`（tokenizer 不变）、其他 class（Block/Transformer 只调用 `self.attn(x)`，接口不变）。

**(2) 最小改动方案——if-else 分支模式**

```python
# model.py Attention.forward (L518-L533 附近)
class Attention:
    def __init__(self, args, layer_id):
        ...
        self.use_new_attn = getattr(args, 'use_new_attn', False)
        if self.use_new_attn:
            # 新机制需要的额外 buffer（如额外的 cache、投影矩阵等）
            self.new_attn_proj = Linear(args.dim, args.dim, bias=False)
    
    def forward(self, x, start_pos, ...):
        ...
        if self.use_new_attn:
            # === 新 attention 机制 ===
            # Step 1: 用新 kernel 计算 attention（替代 sparse_attn_kernel）
            o = new_attn_kernel(q, kv_cache, self.attn_sink, topk_idxs, ...)
            # Step 2: 可能的后处理（如门控融合、残差缩放等）
            o = o + self.new_attn_proj(x)  # 示例：额外的 skip connection
        else:
            # === 原始代码（不变） ===
            o = sparse_attn_kernel(q, kv_cache, self.attn_sink, topk_idxs, self.softmax_scale)
        ...
```

**(3) 风险管控——四步验证法**

| 步骤 | 操作 | 验证指标 | 回滚条件 |
|------|------|---------|---------|
| 1. 单层替换 | 选 1 个中间层（如 layer 21）替换为 $use_new_attn=True$ ，其余层不变 | 该层的 attention entropy、输出 RMS、gradient norm 是否正常 | 任何 NaN/Inf 或 loss spike |
| 2. 短序列验证 | 在 8K 上下文跑 100 step 训练/推理，对比 PPL | 新机制的 PPL 与原版差异 < 1% | PPL 退化 > 2% |
| 3. 全层扩展 | 43 层全部启用新机制，跑完整 benchmark | MMLU/GPQA/AIME 等 benchmark 与原版对比 | 任一 benchmark 退化 > 3% |
| 4. 长上下文验证 | 在 128K/512K/1M 下测 PPL + tokens/s | 确保长上下文无退化，速度无显著下降 | P99 latency 退化 > 20% |

**(4) 特别注意——KV cache 兼容性**

新 attention 机制最容易出的问题是 KV cache 格式不兼容。V4-Flash 的 KV cache 是"三层异构"结构（滑窗 + 压缩 + Indexer），新机制可能需要：
- **额外的 cache buffer**：如新机制需要存储 per-head 的门控值或 per-token 的重要性分数，需要在 $Attention.__init__$ 的 $register_buffer$ 中新增一个 tensor
- **Cache 读写顺序变更**：如果新机制改变了"哪些 KV 被 gather"，需要同时修改 Compressor 的 $kv_cache$ 指针（因为 Compressor 直接引用 Attention 主 cache 的压缩段）
- **Prefill vs Decode 分支**：确保 $start_pos == 0$ （prefill）和 $start_pos > 0$ （decode）两个分支都正确处理新机制

最小改动原则：**新机制先只改滑窗部分（window KV），压缩 KV 保持原样**。这样 CSA/HCA 的压缩逻辑完全不受影响，风险最小。后续验证通过后再考虑扩展到压缩 KV。

**(5) 为什么不用"继承 Attention 类"的方式？**

更"优雅"的 OOP 设计是 `class NewAttention(Attention): override forward()`。但 V4-Flash 仓库的 $Block.__init__$ 直接实例化 $Attention(args, layer_id)$ ，要支持继承需要改 $Block.__init__$ 的逻辑（加一个 $attention_cls$ 参数）。这虽然代码更干净，但"改动文件数"从 3 个变成 4 个（多了 `model.py` 中 Block 的改动）。if-else 分支虽然"丑"但改动面最小，符合"最小改动"原则。

🎤 **面试官视角**：考察候选人的代码组织能力 + 对 V4 仓库的理解深度。很多人一上来就说"重写 Attention 类"或"用继承"，但优秀的回答会先分析"最小改动面"——哪些文件必须改、哪些可以不动、接口的兼容边界在哪。能提到 KV cache 兼容性和四步验证法是高级答案。

🎯 **答题模板**：(1) 核心——只改 3 个文件（model.py/kernel.py/config.json），用 if-else 分支而非继承；(2) 最小改动方案——在 Attention.forward 加一个 $if self.use_new_attn: ... else: original$ 分支，新 kernel 用 TileLang 写；(3) 风险管控——四步验证（单层→短序列→全层→长上下文），每步有明确验证指标和回滚条件；(4) 特别注意 KV cache 兼容性（先只改滑窗部分、后扩展到压缩 KV）。

**延伸阅读**：主报告 CH3.6（sparse_attn_kernel 源码分析）、CH8.1（model.py 文件结构）；`inference/model.py` L436-L545 `Attention`；`inference/kernel.py` L276-L368。

---

> 章节完成。

**Q 总数**：89 Q（CH8: 33 Q / CH9: 24 Q / CH10: 32 Q）
**字符数**：约 65,000 字符
**CH10 面试标注数**：32 个面经题（基础 10 + 进阶 11 + 情境 11），每题标注「面试官视角」「答题模板」

