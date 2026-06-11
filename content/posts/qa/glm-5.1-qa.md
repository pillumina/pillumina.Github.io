+++
date = '2026-06-11'
draft = false
title = 'GLM-5.1 架构 QA'
categories = ['qa']
tags = ['moe', 'attention', 'model-architecture', 'qa', 'glm', 'dsa', 'mla']
series = ['qa']
summary = '基于 GLM-5.1 主报告的配套 QA。覆盖三代演进、DSA 动态稀疏注意力、MLA 潜 KV 压缩、MoE 路由（256+1, k=8, scaling=2.5）、异步 Agent RL 训练体系等核心主题。'
+++

# GLM-5.1 架构 QA

> 117 问，覆盖 CH1 预备知识 → CH2 前代回顾 → CH3 GLM-5.1 概览 → CH4 DSA 注意力 → CH5 MoE 路由 → CH6 训练体系 → CH7 支撑项 → CH8 源码映射 → CH9 对比总结 → CH10 面经高频
---

## 一、LLM 预备知识（13 Q）

### Q1.1 Transformer Decoder 的核心结构是什么？

**简短回答**：Transformer Decoder 由多层相同的 block 堆叠而成，每层包含 Pre-Norm（RMSNorm）、Masked Multi-Head Attention 和 FFN（Dense 或 MoE），通过残差连接串联。

**详细解释**：每个 Decoder block 的执行顺序为：输入 x → RMSNorm → Attention（Q/K/V 投影 + scaled dot-product + O 投影）→ 残差加 → RMSNorm → FFN → 残差加。Attention 负责 token 间信息交互（建模「哪个 token 与当前 token 相关」），FFN 负责逐 token 的非线性变换（建模「这个 token 的语义是什么」）。GLM-5.1 中 Attention 被替换为 DSA+MLA 的混合设计，FFN 在 75 层中使用 256-expert MoE。

**关键设计**：Pre-Norm（而非 Post-Norm）使深层梯度更稳定；残差连接避免梯度消失；MHA 提供多头的信息收集能力。

**面试要点**：面试官常问「Attention 和 FFN 各自负责什么」，建议用信息交互 vs 语义建模的二分法回答。

**延伸阅读**：主报告 CH 2.2 / Vaswani et al. (2017) Attention Is All You Need

---

### Q1.2 MoE 与 Dense 模型的核心区别是什么？

**简短回答**：Dense 模型每层只有一个 FFN，所有 token 共享同一套参数；MoE 模型每层有多个「专家」（expert FFN），每个 token 通过 Router 选择性激活部分专家。

**详细解释**：MoE（Mixture of Experts）将单层 FFN 扩展为 N 个独立的 expert（GLM-5.1 中 N=256），外加 1 个所有 token 都经过的共享专家。Router 为每个 token 打分，选 top-k（GLM-5.1 中 k=8）个专家激活。这带来两个关键特性：(1) 总参数量极大扩展（N 个 expert 各有一套权重），(2) 每 token 的实际计算量仅略高于 Dense FFN（仅激活 k/N 的 expert）。MoE 的核心价值：庞大的参数库提供多样化能力，极低的单 token 激活率控制推理成本。

| 维度 | Dense FFN | MoE FFN |
|---|---|---|
| 每层参数 | 226M | 9.66B（256 expert 总池） |
| 每 token 激活 | 226M | ~340M（8 expert + 1 shared） |
| 表达能力 | 单一 | 256 种专家模式 |

**面试要点**：记住「total params vs activated params」这对概念——744B 总量 vs 40B 激活。

**延伸阅读**：主报告 CH 2.4 / Shazeer et al. (2017) Outrageously Large Neural Networks

---

### Q1.3 什么是 MLA（多头潜注意力）？

**简短回答**：MLA（Multi-head Latent Attention）通过对 KV 做潜空间压缩来减少 KV cache 大小，将每头的 K V 独立存储改为共享的潜表示存储。

**详细解释**：标准 MHA 每层需缓存 H×d_k 维的 K 和 H×d_v 维的 V（GLM-5.1 中 64×256×2=32768 元素/ token）。MLA 通过 kv_a_proj（6144→512+64）将 KV 压缩到 576 维潜空间，缓存时只存这 576 维，使用时再通过 kv_b_proj（512→64×448）展开。代价是每层增加了 kv_a_proj 和 kv_b_proj 两个轻量投影矩阵的计算，但 KV cache 从 ~936 GB 降至理论 ~19 GB（192K 上下文）。

MLA 最早由 DeepSeek-V2 提出，GLM-5.1 采用了相同原理但维度选择不同（q_lora_rank=2048 vs 1536）。

**面试要点**：MLA 是「用计算换存储」的经典案例——轻微的投影计算换来巨大的 cache 缩减。

**延伸阅读**：主报告 CH 4.1 / DeepSeek-V2 arXiv:2405.04434

---

### Q1.4 GQA 和 MHA 的区别是什么？

**简短回答**：MHA（Multi-Head Attention）中 Q 和 KV 头数相等；GQA（Grouped Query Attention）中 KV 头数少于 Q 头数，多个 Q 头共享一组 KV。

**详细解释**：GQA 的引入是为了减少 KV cache：若 Q 头数=64 而 KV 头数=8（GQA ratio=8），则 KV cache 为 MHA 的 1/8。但 GQA 以有限的 KV 头区分度为代价。

GLM-5.1 的 GQA ratio=1（`num_attention_heads=64` 且 `num_key_value_heads=64`），即全 MHA。这不是设计疏忽：MLA 的 KV 压缩在潜空间完成（kv_lora_rank=512），所有 64 个 attention head 共享压缩 KV 表示，不需要通过减少 KV 头数来节省缓存。MLA 的缓存压缩来自潜空间降维，而非 GQA 的 head 削减。

**面试要点**：MLA 和 GQA 是两类完全不同的 KV cache 压缩方案——MLA 用潜压缩，GQA 用 head 共享。GLM-5.1 用了 MLA 故不需要 GQA。

**延伸阅读**：主报告 CH 2.3（约束 5）/ Ainslie et al. (2023) GQA: Training Generalized Multi-Query

---

### Q1.5 Dense FFN 和 MoE FFN 在结构上有何不同？

**简短回答**：Dense FFN 通常为 SwiGLU 结构（gate+up → SiLU → × → down），所有参数对每个 token 激活；MoE FFN 将 Dense FFN 的权重矩阵替换为多个 expert 副本，由 Router 选择部分 expert 激活。

**详细解释**：SwiGLU 的标准形式为：$\text{FFN}(x) = \text{down}(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x))$，包含 3 个权重矩阵（gate/up/down）。GLM-5.1 中 Dense FFN 的 intermediate_size=12288（每层约 226M 参数）。

MoE FFN 则将 gate/up/down 扩展为 [256, ...] 的三维 tensor：
- `gate_up_proj`: [256, 4096, 6144]（256 个 expert，每个 expert 将 6144→4096×2）
- `down_proj`: [256, 6144, 2048]（256 个 expert，每个 expert 将 2048→6144）

每个 expert 的 intermediate_size=2048（moe_intermediate_size），小于 Dense 的 12288，但 256 个 expert 并列使总参数量巨大。

**面试要点**：MoE 的「expert 更窄但数量多」设计——单 expert 比 Dense FFN 窄（intermediate 2048 vs 12288），但 256 个并联提供了更大的参数容量。

**延伸阅读**：主报告 CH 5.1 / CH 5.5

---

### Q1.6 什么是 Partial RoPE？

**简短回答**：Partial RoPE 只对 Q/K 的部分维度施加旋转位置编码（RoPE），其余维度不编码位置信息，用于在长上下文中保留语义匹配的稳定性。

**详细解释**：GLM-5.1 的 Q/K 每头维度为 256，其中前 192 维为 `qk_nope_head_dim`（无位置编码），后 64 维为 `qk_rope_head_dim`（施加 RoPE）。这意味着 QK 内积 = 内容匹配（前 192 维的点积）+ 位置匹配（后 64 维的旋转点积）。

**为什么不全维度 RoPE？**
全维度 RoPE（如 Llama 的 128 维全旋转）使得 QK 内积完全受位置关系主导——两个 token 的语义再相关，如果位置距离不合适，注意力分数也会被旋转角度扭曲。Partial RoPE 将位置编码限制在 25% 的维度中，让剩余的 75% 维度纯粹用于内容匹配。这在长上下文场景下特别重要——当两个语义相关但距离很远的 token 需要互相 attend 时，内容匹配部分不受位置距离的影响。

**与 MLA 的配合**：GLM-5.1 使用 MLA（Multi-head Latent Attention），KV 被压缩到低秩潜空间。RoPE 施加在压缩 KV 的一个独立分量（`k_pe`，单头 stream）上，不参与潜压缩。这种解耦使 MLA 的低秩压缩不损失位置精度——位置信息走独立的通道。

**面试要点**：区分 Full RoPE（全部 256 维旋转）和 Partial RoPE（仅 64 维旋转）——后者是 MLA 的标准配置。

**延伸阅读**：主报告 CH 4.2 / DeepSeek-V2 论文 §3.1

---

### Q1.7 Sigmoid 路由与 Softmax 路由的核心区别？

**简短回答**：Sigmoid 路由为每个 expert 独立打分（值在 0-1），各 expert 分数不互相影响；Softmax 路由将所有 expert 分数归一化为概率分布（和为 1），expert 间此消彼长。

**详细解释**：

| 维度 | Sigmoid | Softmax |
|---|---|---|
| 独立性 | 完全独立 | 互相竞争 |
| 得分范围 | (0, 1) | [0, 1]，∑=1 |
| 计算复杂度 | O(N)（N 次 sigmoid） | O(N)（N 次 exp + sum） |
| 适合场景 | 复合需求（token 同时需要多种能力） | 独占需求（token 只适合某类 expert） |
| 区分度 | 偏低（需 bias 补偿） | 偏高（自然拉开差距） |

GLM-5.1 选择 sigmoid 的原因：(1) token 可能同时需要代码+数学+常识等多种能力，sigmoid 允许同时给多个 expert 高分；(2) 配合 `e_score_correction_bias` 做负载均衡；(3) 计算略快（无需 sum 再做除法）。

**面试要点**：sigmoid 路由的独立性是 MoE 设计的关键选择——是否支持 token 的「多任务同时激活」决定了评分函数的选择。

**延伸阅读**：主报告 CH 5.3 / GLM-5 paper §2.1

---

### Q1.8 KV Cache 是什么？为什么对推理至关重要？

**简短回答**：KV Cache 是在自回归生成中缓存已计算的 Key 和 Value，避免每个新 token 都重新计算所有历史 token 的 K/V 投影。

**详细解释**：自回归解码时，token t+1 需要与 token 1..t 做注意力。若每步都重新计算所有历史 K/V，计算量为 $O(T^2)$。KV Cache 将已计算的 K/V 存储起来，新 token 只需：(1) 计算自身的 Q/K/V，(2) Q 与缓存的 K 做点积，(3) 与缓存的 V 做加权。这样每步计算降至 $O(T)$。

KV Cache 大小 = 层数 × 上下文长度 × 每层每 token 的 KV 元素数 × 精度字节。对于 GLM-5.1：
- Expanded cache（当前实现）：78 × 192K × 64 × (256+256) × 2B ≈ 936 GB
- MLA 压缩 cache（理论）：78 × 192K × (512+64) × 2B ≈ 19.2 GB

**面试要点**：KV cache 是 LLM 推理的「内存瓶颈」——大 batch / 长上下文时 cache 远大于模型权重。

**延伸阅读**：主报告 CH 2.5 / vLLM PagedAttention 论文

---

### Q1.9 什么是稀疏注意力？与全注意力有何不同？

**简短回答**：稀疏注意力只对部分 token 对计算注意力权重（而非全注意力的所有 $T^2$ 对），以降低长上下文下的计算复杂度。

**详细解释**：全注意力（Full/Causal Attention）计算 $QK^T$ 矩阵的所有 $T^2$ 个元素，复杂度 $O(T^2 d)$。稀疏注意力通过限制每个 query 可关注的 key 范围来降低复杂度。常见策略：

| 稀疏方式 | 机制 | 复杂度 | 代表 |
|---|---|---|---|
| 滑动窗口 | 固定窗口内全注意 | $O(T \cdot w)$ | Mistral |
| 压缩+选择 | 先压缩序列再选 top | $O(T \cdot m)$ | V4-Flash CSA |
| 动态选择 | Indexer 打分选 top-k | $O(T \cdot k)$ | GLM-5.1 DSA |

DSA 属于「动态选择」类：Indexer 从上下文中动态选 top-2048 个最相关 token，仅在这些 token 上做精确注意力。与固定模式（滑动窗口）的关键区别：选择是内容驱动的，能根据具体 query 动态调整。

**面试要点**：稀疏注意力分为「静态稀疏」（滑动窗口）和「动态稀疏」（内容驱动选择），DSA 是后者中的最佳代表。

**延伸阅读**：主报告 CH 3.1 / Tay et al. (2020) Efficient Transformers: A Survey

---

### Q1.10 MTP（多 token 预测）的原理是什么？

**简短回答**：MTP（Multi-Token Prediction）在训练时让模型同时预测当前 token 和未来 1-2 个 token，推理时用辅助预测头做 speculative decoding 加速。

**详细解释**：标准 LM 训练只预测下一个 token（next-token prediction）。MTP 在最后一层后增加 MTP 模块：使用额外的 Transformer block 处理当前 hidden state，预测第 t+2、t+3...个 token。训练 loss 为各步的加权和。

GLM-5.1 的特殊之处：**Parameter Sharing MTP**——训练时 3 个 MTP 层共享参数，推理时用 1 层预测 2 个 future token。优势：
- 训练内存与单层 MTP 一致（避免 3× 的额外参数）
- 接受率高于 DeepSeek-V3.2（accept length 2.76 vs 2.55）
- 在小型 batch 解码场景中加速效果显著

**面试要点**：MTP 的推理加速来自 speculative decoding——辅助模型预测多个 draft token，主模型批量验证。

**延伸阅读**：主报告 CH 7.4 / GLM-5 paper §2.1 / DeepSeek-V3 MTP 设计

---

### Q1.11 SwiGLU 激活函数是什么？为什么 LLM 普遍使用？

**简短回答**：SwiGLU（Swish-Gated Linear Unit）是一种门控激活函数，将输入 x 分为 gate 和 up 两路，gate 路经 SiLU 后与 up 路逐元素相乘，再经 down 投影输出。

**详细解释**：SwiGLU 的数学形式：

$$
\text{SwiGLU}(x) = \text{down}(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x))
$$

其中 $\text{SiLU}(z) = z \cdot \sigma(z)$。与标准 ReLU（FFN(x) = down(ReLU(up(x)))）相比：

| 维度 | ReLU | SwiGLU |
|---|---|---|
| 门控 | 否（单路） | 是（gate+up 双路） |
| 非线性 | ReLU（0 截断） | SiLU（平滑） |
| 参数量 | 2 个矩阵 | 3 个矩阵（gate/up/down） |
| 性能 | 较差 | 更优（LLM 标配） |

SwiGLU 于 2020 年被提出（Shazeer, GLU Variants），在 LLM 中几乎完全取代了 ReLU FFN。

**面试要点**：SwiGLU = SiLU 门控 + Linear 双射——记住「三个矩阵（gate/up/down）」就抓住了 SwiGLU 的参数量特征。

**延伸阅读**：Shazeer (2020) GLU Variants Improve Transformer

---

### Q1.12 RMSNorm 与 LayerNorm 的区别？

**简短回答**：RMSNorm 是 LayerNorm 的简化版——只做均方根缩放（除以 RMS），省略了减均值和重缩放步骤，计算更快且对 Transformer 训练同样有效。

**详细解释**：

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | $(x - \mu)/\sqrt{\sigma^2 + \epsilon} \cdot \gamma + \beta$ | $x / \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon} \cdot \gamma$ |
| 减均值 | 是 | 否 |
| 重缩放参数 | $\gamma, \beta$ | $\gamma$（无 $\beta$） |
| 计算速度 | 较慢 | 快 ~15% |
| 训练稳定性 | 等效 | 等效 |

GLM-5.1 的 `GlmMoeDsaRMSNorm`（L46-L63）使用 $\epsilon=10^{-5}$，等价于 T5LayerNorm。几乎所有现代 LLM（Llama, DeepSeek, GLM）都使用 RMSNorm。

**面试要点**：RMSNorm 的成功说明「减均值」对 Transformer 训练不是必需的——只保留方差归一化即可。

**延伸阅读**：主报告 CH 8.2 / Zhang & Sennrich (2019) Root Mean Square Layer Normalization

---

### Q1.13 BF16、FP8、INT4 三种精度的区别和使用场景？

**简短回答**：BF16 是训练/推理的标准精度（2 bytes/value），FP8 是推理量化的主力（1 byte/value，~2× 显存节省），INT4 是最激进的推理量化（0.5 bytes/value，~4× 显存节省）。

**详细解释**：

| 精度 | 位宽 | 动态范围 | 典型用途 | 精度损失 |
|---|---|---|---|---|
| BF16 | 16 bit | ~10³⁸ | 训练、标准推理 | 无（训练精度） |
| FP8 (E4M3) | 8 bit | ~10³⁸ | 推理加速（H100+ 原生支持） | 极小 |
| INT4 | 4 bit | ~10¹ | 极致内存压缩（消费级部署） | 中（需 QAT 补偿） |

GLM-5.1 的部署策略：
- **BF16**：744B × 2 = 1488 GB 权重（8+ H200 推理）
- **FP8**：744B × 1 = 744 GB（论文 §5 的策略，支持 7 家国产芯片）
- **INT4**：744B × 0.5 = 186 GB（+ QAT 训练，可部署在 2-3 张 H200 上）
- **W4A8** 混合精度：Attention/MLP 用 W8A8，MoE expert 用 W4A8

**面试要点**：量化策略遵循「重要模块保留高精度」原则——Indexer.weights_proj 和 e_score_correction_bias 始终 FP32。

**延伸阅读**：主报告 CH 7.3 / GLM-5 paper §5

---

## 二、前代回顾（9 Q）

### Q2.1 GLM-4 的核心架构特征是什么？

**简短回答**：GLM-4（包括 4.5 变体）是 GLM 系列的首个 MoE 尝试，采用 Dense Transformer 基础 + 初步 MoE 探索，规模为 355B 总参 / 32B 激活参数。

**详细解释**：GLM-4 系列代表 2024 年的 GLM 技术状态。核心特征：(1) Dense Transformer 为主的架构；(2) 初步引入 MoE（以 GLM-4.5 为代表，160 expert，8 路由）；(3) 未引入 DSA 和 MLA——这两项是 GLM-5 的全新贡献。GLM-4 在 SWE-bench 等 Agent 评测上表现一般，其 MoE 的「渐进式引入」策略（前 3 层 Dense + 后 N 层 MoE）在 GLM-5 中延续。

**面试要点**：GLM-4.5 的 355B/32B/160 expert/8 路由 这些数字与 GLM-5.1 的 744B/40B/256 expert/8 路由形成「规模倍增」的对比。

**延伸阅读**：主报告 CH 1.1 / GLM-5 paper Table 10

---

### Q2.2 GLM-5 相比 GLM-4 的三大创新是什么？

**简短回答**：GLM-5 引入了三大核心架构创新：MoE 扩展到 256 专家、MLA 潜注意力压缩、DSA 动态稀疏注意力，配合异步 RL 基础设施实现 Agent 能力飞跃。

**详细解释**：

1. **MoE 扩展（160→256 expert）**：总参数从 355B 增至 744B，激活参数从 32B 增至 40B。更大的专家池提供更丰富的专用能力。
2. **MLA 潜注意力压缩**：全新引入，将 KV cache 降至理论 ~19 GB（192K 下），使长上下文推理在有限 GPU 上可行。
3. **DSA 动态稀疏注意力**：取代全注意力，将注意力复杂度从 $O(T^2)$ 降至 $O(T \cdot k)$，节省 72.5% 注意力计算。

这三大创新在 GLM-5 论文中首次公开，GLM-5.1 完全继承。此外，异步 RL 基础设施（slime 框架）是 post-training 侧的同等重要创新。

**面试要点**：三大创新的定位——DSA 解决「快不快」（计算效率），MLA 解决「省不省」（存储效率），MoE 解决「强不强」（模型容量）。

**延伸阅读**：主报告 CH 1.2 / GLM-5 paper arXiv:2602.15763

---

### Q2.3 GLM-5.1 和 GLM-5 是什么关系？

**简短回答**：GLM-5.1 不是独立的新模型，而是 GLM-5 的 checkpoint 优化 + Agent 工程增强部署版本，核心架构完全一致。

**详细解释**：两者的关系可以从三个维度确认：

1. **架构一致**：78 层、6144 hidden、DSA+MLA、256+1 MoE、k=8，所有核心超参与 GLM-5 论文 Table 10 一致。
2. **config.json 验证**：HF 仓库 `zai-org/GLM-5.1` 的 `config.json` 中的架构字段与论文描述匹配。
3. **差异仅在训练**：GLM-5.1 经过了更多的 Agent 工程优化（异步 Agent RL），在 SWE-bench Pro（58.4）和 Code Arena（1530 Elo）上取得更强的结果。

区别于 GLM-4 → GLM-5 的架构换代，GLM-5 → GLM-5.1 是「架构相同、质量提升」的迭代。

**面试要点**：如果有人问「GLM-5.1 架构上有什么新东西」，正确答案是「没有——架构与 GLM-5 相同，提升来自 checkpoint 优化和 Agent 工程增强」。

**延伸阅读**：主报告 CH 1.3 / HF `zai-org/GLM-5.1` 仓库

---

### Q2.4 GLM-5 论文的技术贡献概括为哪四大支柱？

**简短回答**：DSA 动态稀疏注意力、异步 RL 基础设施、异步 Agent RL 算法、国产芯片全栈适配。

**详细解释**（对应论文四大贡献）：

1. **DSA（§2.1.1）**：引入 Indexer 动态选择 top-2048 token，将注意力复杂度从 $O(T^2)$ 降至 $O(T \cdot k)$，在 RULER@128K 上仅比 Full Attention 低 0.35 分。
2. **异步 RL 基础设施（§3, §4）**：基于 slime 框架，将生成与训练解耦，支持 1k+ 并发 rollout、心跳驱动容错。
3. **异步 Agent RL 算法（§4.1）**：TITO 网关消除重 tokenize 偏差，Direct Double-sided Importance Sampling 简化 old-policy 推断。
4. **国产芯片全栈适配（§5）**：已完成 7 家国产芯片平台（昇腾、摩尔线程、海光、寒武纪、昆仑芯、MetaX、燧原）的深度优化。

**面试要点**：四大支柱中 DSA 是「架构创新」，异步 RL 是「工程创新」，国产芯片适配是「部署创新」——体现了 GLM 团队从模型到系统的全栈能力。

**延伸阅读**：主报告 CH 1.2

---

### Q2.5 论文 Table 10 提及「reduces layer count to 80」但实际只有 78 层，为什么？

**简短回答**：论文标题文字可能源于早期设计稿（80 层 = 7 Dense + 73 MoE 的某种配置），但 Table 10 的数据（3 Dense + 75 MoE = 78 层）与 config.json 的 `num_hidden_layers=78` 一致，最终训练使用 78 层。

**详细解释**：主报告 CH 1.4 讨论了这一差异。通过交叉验证发现问题：

- **论文标题文字**："reduces layer count to 80"（减少层数至 80）
- **论文 Table 10 数据**：3 Dense + 75 MoE = 78 层（与标题矛盾）
- **config.json 真值**：`num_hidden_layers=78`（与 Table 10 一致）
- **代码验证**：`configuration_glm.py` 中 `num_hidden_layers` 默认值为 78

结论：以 config.json 为准，实际训练配置为 78 层。标题中的 "80" 可能是早期设计稿的数字（如 5 Dense + 75 MoE）未被更新。这种不一致不影响模型理解，但提醒我们——论文中的数字必须与 config.json 交叉验证，config.json 才是 golden source。

**面试要点**：遇到论文与代码不一致时，以 `config.json` 和实际模型文件为准——这是验证模型架构的「golden source」。

**延伸阅读**：主报告 CH 1.4

---

### Q2.6 GLM-4.5 的参数规模和 GLM-5.1 相比如何？

**简短回答**：GLM-4.5 为 355B 总参 / 32B 激活 / 160 expert / 8 路由；GLM-5.1 为 744B 总参 / 40B 激活 / 256 expert / 8 路由。总参增长 2.1×，激活参数增长 1.25×。

**详细解释**：从 GLM-4.5 到 GLM-5.1 的规模扩展并非线性翻倍：

| 维度 | GLM-4.5 | GLM-5.1 | 倍数 |
|---|---|---|---|
| 总参数 | 355B | 744B | 2.1× |
| 激活参数 | 32B | 40B | 1.25× |
| 专家数 | 160 | 256 | 1.6× |
| k（每 token 专家） | 8 | 8 | — |
| 层数 | 未公开 | 78 | — |
| hidden_size | 未公开 | 6144 | — |

关键洞察：激活参数增长（1.25×）远小于总参增长（2.1×），说明新增的参数大部分进入了「长尾专家」——这些 expert 很少被激活，但对特定场景至关重要。

**面试要点**：总参增长 2.1× 但激活仅 1.25×——MoE 扩展的精髓是「增加参数池大小但控制每 token 计算量」。

**延伸阅读**：主报告 CH 1.1 / GLM-5 paper Table 10

---

### Q2.7 GLM-5.1 在哪些 benchmark 上有显著提升？

**简短回答**：SWE-bench Pro 58.4（首次开源模型达此水平）、Code Arena 1530 Elo、Artificial Analysis Intelligence Index v4.0 得分 50（开源模型首次）。

**详细解释**：

| Benchmark | GLM-5 | GLM-5.1 | 提升 |
|---|---|---|---|
| SWE-bench Verified | 77.8 | — | — |
| SWE-bench Pro | — | 58.4 | 新评测 |
| Code Arena Elo | — | 1530 | 新评测 |
| Terminal-Bench 2.0 | 56.2 / 60.7† | — | — |
| AAII v4.0 | — | 50 | 开源首次 |

GLM-5.1 的提升集中在 Agent 工程评测（SWE-bench Pro, Code Arena），这与「checkpoint 优化 + Agent 工程增强」的定位一致。在纯推理评测（MMLU, BBH 等）上，GLM-5 论文已有报告。

**面试要点**：GLM-5.1 的强项是 Agent（SWE-bench Pro 58.4），不是纯推理——这是面试中常见的考察点。

**延伸阅读**：主报告 CH 0 / CH 1 / GLM-5 paper Table 7, Table 8

---

### Q2.8 GLM-5 的异步 RL 基础设施核心组件有哪些？

**简短回答**：基于 slime 框架的解耦生成-训练引擎、1k+ 并发 rollout 编排器、心跳容错机制、FP8+MTP+PD 尾延迟优化。

**详细解释**：

1. **解耦架构**：生成引擎和训练引擎部署在不同 GPU 上，推理引擎持续生成轨迹，达到阈值后批量发送训练引擎更新模型。
2. **参数同步**：每 K 步梯度更新后，训练引擎将新权重推回推理引擎。
3. **多任务编排**：Server-based Multi-Task Rollout Orchestrator，各任务注册为独立微服务。
4. **尾延迟优化**：FP8 推理 + MTP speculative decoding + PD（Prefill-Decode）disaggregation。
5. **容错**：心跳驱动故障检测，不健康 server 自动下线（而非重试机制，避免无谓的资源浪费）。

**面试要点**：异步 RL 的难点在于「参数同步的时机」和「尾延迟处理」——GLM-5 的设计是这两方面的重要参考。

**延伸阅读**：主报告 CH 6.2 / GLM-5 paper §3.6, §4.1

---

### Q2.9 GLM-5 适配了哪些国产芯片平台？

**简短回答**：华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、MetaX、燧原，共 7 家国产芯片平台。

**详细解释**：论文 §5 详细描述了国产芯片适配方案。这是 GLM-5 区别于其他开源模型的重要特色——不是"可以跑"，而是"针对每家芯片做了深度优化"。

**关键技术栈**：
- **W4A8 混合精度**：标准 Attention/MLP 用 W8A8（精度敏感），MoE expert 用 W4A8（参数量大，对精度相对鲁棒）。这种分工比全 W8A8 节省 30% 显存
- **QuaRot 异常值抑制**：通过权重旋转（正交变换）分散异常值，避免低比特量化时关键通道的信息丢失。这是 LLM 量化的通用技术，在 GLM-5 的 6144 维 hidden 上效果显著
- **Flex_AWQ_SSZ 缩放校准**：针对不同芯片的指令集特性（如昇腾的 Da Vinci 架构 vs 寒武纪的 MLU 架构）使用不同的 scaling factor 搜索策略

**部署规模**：单台 Atlas 800T A3（8×Ascend 910B）可运行完整 744B 模型。按 W4A8 计算，权重大约 744B × 0.5 bytes ≈ 372GB，加上 KV cache 和激活，8×64GB = 512GB 刚好够用。

**面试要点**：可以提到「7 家国产芯片全栈适配」作为 GLM-5 在部署工程上的独特贡献——说明这不是纯学术项目。

**延伸阅读**：主报告 CH 7.3 / GLM-5 paper §5

---

## 三、GLM-5.1 概览（11 Q）

### Q3.1 GLM-5.1 的总参和激活参数分别是多少？为什么有这么大的差距？

**简短回答**：总参 ~744B，激活参数 ~40B。差距来自 MoE 设计——256 个路由专家中每 token 只激活 top-8（仅 3.1% 的专家参数被激活）。

**详细解释**：

| 参数类别 | 数量 | 占比 |
|---|---|---|
| 总参数 | ~744B | 100% |
| 激活参数（每 token） | ~40B | ~5.4% |
| 路由专家总参数 | 724.6B | 97.3% |
| 激活的专家参数（每 token） | ~22.6B | ~3.1% of 路由专家 |

激活参数的计算：
- Attention + Indexer：78 × 174.4M = 13.6B（始终激活）
- Dense FFN（前 3 层）：3 × 226.5M = 0.68B
- MoE（75 层，8 路由专家 + 1 共享）：75 × 339.8M = 25.5B
- 合计：~39.8B ≈ 40B

**面试要点**：744B vs 40B 的 18.6× 差距是 MoE 架构效率的最直观体现——「用 Dense 模型的推理成本，获得接近万亿参数模型的表达能力」。

**延伸阅读**：主报告 CH 2.4 / config.json

---

### Q3.2 GLM-5.1 的 78 层结构如何分布？

**简短回答**：前 3 层为 Dense FFN（标准 SwiGLU），后 75 层为 MoE FFN（256 路由专家 + 1 共享专家 + top-8 路由）。

**详细解释**：这种「Dense 首 + MoE 尾」的混合设计继承自 GLM-4.5，背后的设计逻辑值得深入理解：

| 层范围 | FFN 类型 | 参数量/层 | 激活量/层 |
|---|---|---|---|
| 1-3 | Dense SwiGLU | ~226.5M | ~226.5M（全部激活） |
| 4-78 | MoE (256+1) | ~9.66B（总池） | ~339.8M（仅 9/257 激活） |

**为什么是 Dense 开头而非 MoE 开头？**
1. **特征分化假说**：浅层（1-3）提取的是通用低级特征（词性、句法结构），这些特征对所有 token 都适用，不需要路由分化。深层（4-78）提取的是任务特定的高级特征（语义、推理模式），不同 token 需要不同专家。
2. **训练稳定性**：训练初期路由器随机，如果第 1 层就用 MoE，路由错误会级联放大。Dense 开头提供稳定的初始表示，路由器在更深的层才介入。
3. **参数量权衡**：3 层 Dense 仅贡献 0.68B（占总参 0.1%），对总参几乎无影响，但提供了不成比例的训练稳定性收益。

**对比**：Qwen3.5-MoE 采用全层 MoE（无 Dense 层），DeepSeek V3 也是全层 MoE。GLM-5.1 的 Dense 开头是一个保守但有理有据的选择。

**面试要点**：「为什么是 3 层而不是 1 层或 5 层」——3 是 GLM-4.5 的实验结论，论文未提供不同 Dense 层数消融，可能来自内部实验。

**延伸阅读**：主报告 CH 2.3（约束 3）/ CH 5.4

---

### Q3.3 GLM-5.1 的核心超参数速览表？

**简短回答**：78 层、hidden=6144、64 Q heads + 64 KV heads（全 MHA）、q_lora_rank=2048、kv_lora_rank=512、256 routed experts + 1 shared、k=8、routed_scaling_factor=2.5、max_position_embeddings=202752。

**详细解释**：最关键的 15 个超参（来自 `config.json`）：

| 参数 | 值 | 含义 |
|---|---|---|
| num_hidden_layers | 78 | 层数 |
| hidden_size | 6144 | 隐层维度 |
| num_attention_heads | 64 | Q 头数 |
| num_key_value_heads | 64 | KV 头数（GQA=1） |
| q_lora_rank | 2048 | Q LoRA 压缩秩 |
| kv_lora_rank | 512 | KV LoRA 压缩秩 |
| qk_head_dim | 256 | QK 每头维度 |
| qk_nope_head_dim | 192 | 无位置部分 |
| qk_rope_head_dim | 64 | RoPE 部分 |
| v_head_dim | 256 | V 每头维度 |
| index_n_heads | 32 | DSA Indexer 头数 |
| index_topk | 2048 | DSA 选择 token 数 |
| n_routed_experts | 256 | 路由专家数 |
| num_experts_per_tok | 8 | 每 token 激活专家数 |
| routed_scaling_factor | 2.5 | 路由权重放大 |

**面试要点**：记住 6 个硬核数字——78/6144/64/256/8/2.5——就可以在面试中快速画出 GLM-5.1 的架构图。

**延伸阅读**：主报告 CH 2.1（完整超参表）

---

### Q3.4 744B 参数的构成是怎样的？

**简短回答**：MoE 路由专家占 97.3%（724.6B），Attention+Indexer 占 1.8%（13.6B），共享专家+Router 占 0.4%（2.95B），Embedding+LM Head 占 0.3%（1.9B），Dense FFN 占 0.1%（0.68B）。

**详细解释**：

| 组件 | 参数量 | 占比 |
|---|---|---|
| MoE 路由专家（75×256） | 724.6B | 97.3% |
| Attention + Indexer（78 层） | 13.6B | 1.8% |
| MoE 共享专家 + Router（75 层） | 2.95B | 0.4% |
| Embedding + LM Head | 1.90B | 0.3% |
| Dense FFN（3 层） | 0.68B | 0.1% |
| 其他（RMSNorm/MTP 等） | ~0.5B | 0.1% |

核心洞察：参数高度集中在 MoE 路由专家中——几乎整个模型的能力都存储在 19,200 个 expert 权重矩阵（75 层 × 256 expert）中。

**面试要点**：如果有人问「744B 参数主要花在哪里」，答案只有一个——MoE 专家（97.3%）。这是所有大规模 MoE 模型的共性。

**延伸阅读**：主报告 CH 2.4 / 附录 C

---

### Q3.5 前 3 层 Dense + 后 75 层 MoE 的设计理由是什么？

**简短回答**：(1) 浅层 token 表征尚未充分分化为可路由的专用特征，过早 MoE 导致路由不稳定；(2) Dense 保证早期语义编码的全覆盖，避免信息在路由筛选阶段丢失；(3) 仅 3 层，对总参影响极小。

**详细解释**：这是一种「渐进式 MoE 化」策略，GLM-4.5 和 GLM-5 两代验证有效：

- **路由稳定性**：浅层（第 1-3 层）的 hidden state 仍包含丰富的通用语义信息（词义、语法、浅层关系），尚未细化为需要不同 expert 处理的专用特征。此时用 MoE 路由可能导致高熵的「随机路由」，不仅浪费计算，还降低了每个 expert 的训练效率。
- **信息完整性**：Dense FFN 的 226.5M 参数对每个 token 都激活，保证了早期处理没有任何信息丢失——所有 token 都经过相同的语义编码步骤。
- **成本可控**：3 层 Dense FFN 仅占总参数的 0.1%，但为后续 75 层 MoE 提供了稳定的特征基础。

**面试要点**：类比理解——Dense 首就像「公共基础课」（所有学生必修），MoE 尾就像「专业课」（按兴趣分流）。没有公共基础，分流会盲目。

**延伸阅读**：主报告 CH 2.3（约束 3）/ CH 5.4

---

### Q3.6 routed_scaling_factor=2.5 的设计意图是什么？

**简短回答**：补偿 sigmoid 评分 + top-8 归一化导致的有效权重过小问题，将每个 expert 的加权输出放大 2.5×，使 MoE 层的输出幅度接近 Dense FFN，保持残差流的数值一致。

**详细解释**：路由权重的完整计算链：

1. sigmoid 打分：每个 expert 得分 ∈ (0, 1)
2. top-8 选择
3. 归一化：8 个选中 expert 的权重除以它们之和
4. × 2.5：放大

假设 8 个 expert 的 sigmoid 分数均为 0.5，归一化后每个权重 = 0.5/4.0 = 0.125。Expert FFN 的输出乘以 0.125 后仅为原始幅度的 12.5%。乘以 2.5 后有效权重 = 0.125 × 2.5 = 0.3125，使 MoE 层输出与 Dense FFN 接近（Dense FFN 不需要路由折扣因子）。

与其他模型对比：V4-Flash 的 scaling_factor=1.5（对应 k=6），M2.7 不使用（等价 1.0）。GLM-5.1 的 2.5 是最激进的。

**面试要点**：2.5 不是「放大信号」——是「恢复信号」。没有这个因子，MoE 层输出只有 Dense 层的 1/8 到 1/4。

**延伸阅读**：主报告 CH 2.3（约束 2）/ CH 5.2

---

### Q3.7 GLM-5.1 为什么 GQA ratio=1（全 MHA）？

**简短回答**：MLA 的 KV 压缩已在潜空间完成（kv_lora_rank=512），不需要通过减少 KV 头数来节省缓存。MLA 的缓存压缩来自潜空间降维，而非 GQA 的 head 削减。

**详细解释**：理解这个设计的两个关键点：

1. **GQA 和 MLA 是两种独立的 KV cache 压缩方案**：
   - GQA：减少 KV 头数（如 64 Q 头 vs 8 KV 头），直接减少存储的 K V 数量
   - MLA：对 KV 做潜空间压缩（576 维替代 32768 维），间接减少存储

2. **MLA 已经解决了 GQA 要解决的问题**：
   - GQA 的目标：减少 KV cache → MLA 通过压缩做到了（~19 GB vs ~936 GB）
   - GQA 的代价：KV 头区分度下降 → MLA 避免了这一代价（64 头独立 k_nope）

MLA + MHA 的组合本质是：用潜空间压缩完成存储优化，用全 MHA 保持注意力精度。这与 DeepSeek-V3 的设计一致（V3 也是 MLA + 全 MHA）。

**面试要点**：MLA + MHA 是「算法级压缩 + 体系级精度」的组合，不要误认为 GQA=1 是设计疏忽。

**延伸阅读**：主报告 CH 2.3（约束 5）

---

### Q3.8 GLM-5.1 KV Cache 的两种估算分别是多少？

**简短回答**：MLA 理论压缩 cache 约 19.2 GB（78 层 × 192K × 576 × 2B），当前 expanded 实现约 936 GB（78 层 × 192K × 64 × 512 × 2B）。

**详细解释**：

| 缓存模式 | 每层缓存 | 78 层总计 | 说明 |
|---|---|---|---|
| MLA 压缩（理论） | 246 MB | **19.2 GB** | 缓存 kv_lora_rank(512) + k_pe(64) |
| 当前 expanded | 12 GB | **936 GB** | 缓存完全展开的 K(64×256) + V(64×256) |

压缩 cache 是 MLA 的设计目标——每 token 仅存储 576 个值（512 压缩 KV + 64 RoPE K_pe），解码时通过 kv_b_proj 展开为 64 头 × 448 维。当前 HF 实现为兼容标准 `DynamicCache` 而展开存储，代码注释标注为「future optimization」。

在 batch 推理中，MLA 压缩缓存的优势更大：所有请求共享同一份 KV cache（~19 GB 不随 batch size 缩放）。

**面试要点**：区分「当前实现」和「理论设计」——面试中问 KV cache 大小时应主动提到两种模式。

**延伸阅读**：主报告 CH 2.5

---

### Q3.9 单 token 的 decode FLOPs 约是多少？

**简短回答**：单 token decode 约 218 GFLOPs（78 层），其中 Indexer 占 58.7%，MoE FFN 占 24.6%，MLA 投影占 11.9%，稀疏注意力仅占 4.8%。

**详细解释**：按阶段划分的单 token 单层（MoE 层）FLOPs：

| 阶段 | FLOPs | 占比 |
|---|---|---|
| Indexer（q_wq_b + wk + 点积评分） | 1.63 G | 58.7% |
| MoE FFN（8 expert + 1 shared） | 0.68 G | 24.6% |
| MLA 投影（q_a/b + kv_a/b + o_proj） | 0.33 G | 11.9% |
| 稀疏注意力（QK^T + AV，仅 top-2048） | 0.13 G | 4.8% |
| **MoE 层合计** | **2.78 G** | — |
| Dense 层（3 层，FFN 用 Dense 替代） | 2.55 G | — |

完整 78 层：3 × 2.55 + 75 × 2.78 + LM Head(1.9G) ≈ **218 GFLOPs/token**

**面试要点**：Indexer 是 decode 阶段最贵的操作（58.7%），因为它需要对所有 192K token 做点积评分——DSA 的「粗筛」代价不小。

**延伸阅读**：主报告 CH 2.6

---

### Q3.10 DSA 相比 Full Attention 节省了多少计算？

**简短回答**：DSA 的 Indexer + 稀疏注意力合计约 1.77 GFLOPs/层，Full Attention 的单层 QK^T 为 6.44 GFLOPs，节省约 72.5%。78 层累计节省约 364 GFLOPs/token。

**详细解释**：

| 方案 | 注意力计算/层 | 说明 |
|---|---|---|
| Full Attention | 6.44 GFLOPs | $2 \times 64 \times T \times T \times 256$，T=192K 不可承受 |
| DSA Indexer | 1.63 GFLOPs | 32 头 Indexer 评分 |
| DSA 稀疏注意力 | 0.13 GFLOPs | 仅 top-2048 token 参与 |
| DSA 合计 | **1.77 GFLOPs** | 节省 72.5% vs Full Attention |

值得强调的是，Full Attention 的 6.44 GFLOPs/层在 192K 上下文下实际上是无法承受的——78 层 × 6.44G = 502 GFLOPs 仅注意力部分，加上 Indexer 和 MoE 后单 token 将超过 700 GFLOPs。DSA 使 78 层 × 192K 上下文在实际部署中可行。

**面试要点**：72.5% 是注意力部分的节省，整体 decode（包括 MoE FFN）的节省幅度约为 62.5%。

**延伸阅读**：主报告 CH 2.6（完整 FLOPs 表）

---

### Q3.11 推理需要多少显存？如何降到消费级？

**简短回答**：BF16+expanded cache 需 ~1488 GB（≥32×H200），FP8+压缩 cache 需 ~765 GB（≥8×H200），INT4+压缩 cache 需 ~207 GB（可部署在 2-3 张 H200 上）。

**详细解释**：

| 配置 | 权重 | KV Cache | 激活值 | 合计 | 所需 GPU |
|---|---|---|---|---|---|
| BF16 + expanded | 1488 GB | 936 GB | ~2 GB | ~2,426 GB | ~60×H200 |
| BF16 + 压缩 | 1488 GB | 19.2 GB | ~2 GB | ~1,509 GB | ~11×H200 |
| FP8 + expanded | 744 GB | 936 GB | ~2 GB | ~1,682 GB | ~14×H200 |
| FP8 + 压缩 | 744 GB | 19.2 GB | ~2 GB | ~765 GB | ~8×H200 |
| INT4 + 压缩 | 186 GB | 19.2 GB | ~2 GB | ~207 GB | 2-3×H200 |

消费级部署建议：(1) INT4 QAT（量化感知训练），(2) MLA 压缩 cache 实现，(3) CPU offloading 辅助。

**面试要点**：面试官可能会问「如何在消费级硬件上跑 744B 模型」——回答的关键是 INT4 + 压缩 cache + CPU offload 的组合。

**延伸阅读**：主报告 CH 2.7 / GLM-5 paper §5

---

## 四、DSA 稀疏注意力（16 Q）

### Q4.1 DSA 的核心思想是什么？

**简短回答**：用一个轻量级 Indexer 替代全注意力的 $O(T^2)$ 暴力扫描，从 T 个位置中动态选出 top-2048 个与当前 query 最相关的 token，仅在这些 token 上计算精确注意力，实现从 $O(T^2)$ 到 $O(T \cdot k)$ 的复杂度降级。

**详细解释**：DSA（Dynamic Sparse Attention）基于一个核心洞察：**90% 的注意力条目在长上下文中是冗余的**。其两阶段设计：

1. **Indexer 粗筛**：对每个 query token，Indexer 用 32 个轻量 attention head 对所有历史 token 打分，选择 top-2048 个最高分的 token。
2. **精确注意力精算**：仅在这 2048 个被选中的 token 上计算标准 scaled dot-product attention。

与固定稀疏模式（如滑动窗口）的本质区别：DSA 的选择是**内容驱动的**——同一个 query 面对不同的上下文会选择不同的 token。这使得 DSA 在「检索特定信息」（如代码中查找函数定义）场景中优于所有固定模式。

DSA 最早由 DeepSeek-V3.2 提出，GLM-5 完整采用并做了自己的训练策略优化。

**面试要点**：DSA 不是「少算注意力」，是「先筛选再精算」——这两步加起来比全注意力快，但引入了 Indexer 可能遗漏关键 token 的风险。

**延伸阅读**：主报告 CH 3.1 / GLM-5 paper §2.1.1

---

### Q4.2 DSA Indexer 的工作流程是什么？

**简短回答**：Indexer 通过 7 步完成 token 筛选：Q 投影（复用 MLA q_resid）→ K 投影（单头 128 维）→ K cache 管理 → 权重投影 → ReLU 逐头打分 → 因果 mask → top-k 选择（2048 tokens）。

**详细解释**：7 步流程（对应 `GlmMoeDsaIndexer.forward` L144-L228）：

1. **Q 投影**：$\text{q} = \text{wq\_b}(\text{q\_resid})$，输入是 MLA q_a_proj 的 2048 维输出，投影到 32×128 维。应用 partial RoPE（前 64 维）。
2. **K 投影**：$\text{k} = \text{k\_norm}(\text{wk}(x))$，6144→128 维。所有 32 个 Indexer head 共享同一个 K。同样是 partial RoPE。
3. **K Cache 管理**：`k_cached = cat([self._cached_keys, k])`，Indexer 维护独立的 KV cache（不放在 DynamicCache 中）。
4. **权重投影**：`weights = weights_proj(x).float() * (32 ** -0.5)`，为 32 个 head 学习重要性权重。
5. **逐头打分**：`scores = einsum("bshd,btd->bsht", q, k_cached) * softmax_scale` → `F.relu(scores)`。使用 ReLU（非 softmax）仅保留正相关。
6. **加权组合**：`index_scores = einsum("bsht,bsh->bst", scores, weights)`，32 头分数加权求和。
7. **Top-k 选择**：`topk_indices = index_scores.topk(2048, dim=-1).indices`。

**面试要点**：Indexer 最巧妙的设计是复用 MLA 的 q_resid（第 1 步）—— Indexer 不从头投影 Q，而是「蹭」MLA 的 Q 压缩结果。

**延伸阅读**：主报告 CH 3.6 / 代码片段 `glm_dsa_indexer.py`

---

### Q4.3 为什么 DSA Indexer 使用 ReLU 而非 softmax？

**简短回答**：ReLU 仅保留正相关的 token pair（将负相关的直接置零），配合因果 mask 后形成「只关注相关 token」的硬筛选，比 softmax 的「概率分布」更适合二阶段的「选或不选」决策。

**详细解释**：在 Indexer 打分语境下，ReLU 和 softmax 的语义差异：

- **ReLU**：分数 > 0 的 token 被考虑（可能有 N 个），分数 ≤ 0 的被完全忽略。通过 `weights_proj` 学习到的重要性权重做加权求和后，正相关 token 的分数被累加。
- **Softmax**：所有 token 的分数被归一化为概率分布（和为 1），即使 token 完全无关也会被分配一个非零概率（噪声）。

DSA 的 Indexer 面临的是「选择问题」而非「概率问题」——我们只需要知道哪些 token 值得关注，不需要一个概率分布。ReLU 的硬阈值行为天然适合这个场景：负相关的 token 直接排除，正相关的 token 按相关性排序。

此外，ReLU 计算更快（一个 `max(0, x)` vs softmax 的 exp + sum）。

**面试要点**：面试官可能会追问「如果两个 token 的 ReLU 分数都是 0，它们在后续 attention 中如何被处理」——答案是它们被 mask 排除（index_mask=-inf），注意力权重为 0。

**延伸阅读**：主报告 CH 3.2（Step 4-5）/ indexer forward 代码

---

### Q4.4 Indexer 的 7 步数据流对应的代码行在哪里？

**简短回答**：在 `modeling_glm_moe_dsa.py` 的 `GlmMoeDsaIndexer.forward`（L144-L228）中，按 Q 投影→K 投影→K Cache→权重投影→ReLU 打分→因果 mask→top-k 的 7 步顺序执行。

**详细解释**：

| 步骤 | 代码行 | 关键操作 |
|---|---|---|
| Step 1: Q 投影 | L177-L181 | `wq_b(q_resid)` → view(32,128) → RoPE |
| Step 2: K 投影 | L184-L187 | `k_norm(wk(x))` → split → RoPE |
| Step 3: K Cache | L191-L201 | `_cached_keys` 追加/重置 |
| Step 4: 权重投影 | L214 | `weights_proj(x).float() * (32**-0.5)` |
| Step 5: ReLU 打分 | L217-L220 | `einsum + relu + einsum` |
| Step 6: 因果 Mask | L222-L223 | `index_scores + attention_mask` |
| Step 7: Top-k | L225-L228 | `index_scores.topk(2048)` |

稀疏 Mask 构建在 `GlmMoeDsaAttention.forward`（L416-L432）中完成：`index_mask.scatter_(-1, topk_indices, 0.0)` → 与 causal_mask 合并。

**面试要点**：记住代码结构的「两阶段」——Indexer 负责选 token（7 步），Attention 负责构建 mask + 执行注意力。

**延伸阅读**：主报告 CH 3.6 / CH 8.4 / 代码片段 `glm_dsa_indexer.py`

---

### Q4.5 DSA 与 V4-Flash 的 CSA 有何关键区别？

**简短回答**：DSA 不做 KV 压缩（在原始序列上打分），选择粒度是单 token 级别，top-k=2048；CSA 先将 KV 压缩（m=4），在压缩块上选 top-512，选择粒度是块级别。

**详细解释**：

| 维度 | V4-Flash CSA | GLM-5.1 DSA |
|---|---|---|
| KV 预处理 | Compressor m=4 压缩 | 无压缩（原始序列） |
| 选择粒度 | 压缩块级别 | **单 token 级别** |
| top-k 大小 | 512 | **2048**（4×） |
| 信息损失 | 压缩引入信息损失 | 无损（粒度细） |
| Indexer 头数 | — | 32 |
| 评分函数 | — | ReLU（DSA 特有） |

DSA 不做 KV 压缩的核心优势：选择粒度是单 token 级别，理论上可以精确找到特定 token。对于 Agent 场景（如代码检索中定位特定函数名、类定义、变量引用），token 级精度至关重要——压缩块可能将关键 token 与噪声混在一起，导致要么整个块被选中（浪费 top-k 预算），要么整个块被忽略（错过关键信息）。

代价是 DSA 的 Indexer 需要对所有原始 KV 位置做评分（K cache 大小为 T×128），而 CSA 只需在压缩后的 m 个块上评分（K cache 大小为 T/4×d_k）。

**面试要点**：DSA vs CSA 的本质区别是「精度 vs 效率」的取舍——DSA 选择精度（token 级选择），CSA 选择效率（压缩块级选择）。

**延伸阅读**：主报告 CH 3.1（对比表）/ DeepSeek V4-Flash 论文

---

### Q4.6 DSA 的训练策略是怎样的？

**简短回答**：从 MLA 密集 checkpoint 出发，通过两阶段 Continued Pre-Training 引入 DSA——Warmup（1000 steps，只训 Indexer，最大 lr=5e-3）+ Sparse Adaptation（20B tokens，联合训练基模型和 Indexer）。

**详细解释**：论文 §2.1.1 详细描述了这一策略的关键设计：

**Warmup 阶段**：
- 目标：让 Indexer 学会「选择哪些 token 值得关注」
- 冻结基模型参数，仅训练 Indexer（wq_b, wk, k_norm, weights_proj）
- 1000 steps，每步 14 条 202,752 token 的序列
- 最大学习率 5e-3（比标准 pre-training 高得多）
- 结果：RULER@128K 仅从 79.21 降至 71.35（-7.86）

**Sparse Adaptation 阶段**：
- 目标：让基模型适应「只看到 2048 个 token」的稀疏注意力
- 联合训练基模型所有参数 + Indexer
- 20B tokens（mid-training 的数据+超参）
- 结果：在 RULER@16K/32K/64K 上**反超** MLA baseline（+0.86/+0.49/+1.72），128K 上仅差 0.35 分

关键发现：仅 20B tokens 就足以让 DSA 模型匹配 MLA 模型的性能——这个训练预算远小于 DeepSeek-V3.2 的 943.7B tokens。

**面试要点**：DSA 不是从头训练的——是从 MLA 密集 checkpoint 「适应」出来的。20B tokens 能做到近乎无损是 DSA 训练策略最大的亮点。

**延伸阅读**：主报告 CH 3.3 / GLM-5 paper §2.1.1, Table 3, Table 6

---

### Q4.7 DSA RL 训练中发现了什么关键问题？

**简短回答**：SGLang 的非确定性 CUDA top-k 实现导致 RL 训练几步后性能急剧退化（伴随熵下降），切换到 `torch.topk`（确定性）后恢复稳定。此外 RL 期间必须**冻结 Indexer 参数**以防止不稳定学习。

**详细解释**：论文 §3.2 报告了这两个 DSA RL 中的关键经验：

**问题 1：非确定性 top-k**
- SGLang 的 CUDA top-k 实现使用并行算法，结果不与 `torch.topk` 完全一致
- 在 DSA 中，Indexer 的 top-k 选择决定了哪些 token 参与注意力计算
- 非确定性导致同一 batch 在不同 forward 中选择不同的 token subset，RL 的训练信号变得混乱
- 表现：熵急剧下降 + 性能退化
- 解决：切换到 `torch.topk`（虽略慢但结果确定）

**问题 2：Indexer 冻结**
- RL 训练中梯度信号比 pre-training 噪声更大（来自稀疏奖励和 exploration）
- Indexer 参数少（约 9.6M/层），对噪声极其敏感
- 冻结 Indexer 不仅加速训练，还防止了 Indexer 的「catastrophic forgetting」
- 成本：RL 期间无法进一步优化 Indexer 的选择策略

**面试要点**：这两点暴露了稀疏注意力在 RL 场景的脆弱性——Indexer 的确定性是 RL 稳定的前提。面试官可能会问「为什么 pre-training 不需要冻结 Indexer 但 RL 需要」。

**延伸阅读**：主报告 CH 3.4 / GLM-5 paper §3.2

---

### Q4.8 DSA 消融实验的结果说明了什么？

**简短回答**：在 GLM-9B 上，DSA 是唯一将 RULER@128K 损失控制在 1 分以内的方法（-0.35 vs Full Attention），所有其他高效注意力方法（滑动窗口、Gated DeltaNet、SimpleGDN）都有实质性精度损失（-5.69 到 -30.35）。

**详细解释**：论文 §2.1.2 的系统消融结果：

| 方法 | RULER@64K | RULER@128K | 与 Full Attn 差距 |
|---|---|---|---|
| Full Attention (baseline) | 85.35 | 75.28 | — |
| SWA Interleave | 65.94 | 44.93 | **-30.35** |
| SWA Pattern (搜索优化) | 83.72 | 69.59 | -5.69 |
| Gated DeltaNet (GDN) | 76.76 | 64.00 | -11.28 |
| SimpleGDN | 81.76 | 67.03 | -8.25 |
| **DSA** | 87.06 | 78.86 | **-0.35** |

三个结论：
1. 所有非 DSA 方法在长上下文检索上都有实质性的精度损失
2. DSA 是唯一将 128K 损失控制在 1 分以内的
3. DSA 在 64K 甚至**超过**了 Full Attention（87.06 vs 85.35），说明稀疏注意力在中等长度下可能有「去噪」效果

**面试要点**：DSA 在 64K 上反超 Full Attention（+1.71）是最有力的论据——稀疏注意力不仅能省计算，还能提升精度（类似 Dropout 的正则化效果）。

**延伸阅读**：主报告 CH 3.5 / GLM-5 paper §2.1.2, Table 4-6

---

### Q4.9 Indexer 为什么维护独立的 KV cache（`self._cached_keys`）？

**简短回答**：HuggingFace 的 `DynamicCache` 按 `num_hidden_layers=78` 精确分配空间（每层一个 slot），没有为 Indexer 预留额外的 KV cache slot。Indexer 因此必须自己管理缓存。

**详细解释**：这是一个实现层面的工程约束：

- `DynamicCache` 在初始化时根据 `config.num_hidden_layers` 创建固定大小的缓存数组
- `self_attn.layer_idx` 用于索引每层的缓存 slot
- Indexer 不是 self_attn 的一部分（它是独立的子模块），没有对应的 layer_idx
- 尝试复用 DynamicCache 会导致索引越界或 shape 不匹配

解决方案：Indexer 在 `self._cached_keys` 中维护自己的 K cache（一个 `nn.Parameter(torch.empty(0))` 或等价的 buffer）。在 prefill 阶段重置（`seq_len > 1 → _cached_keys = None`），decode 阶段逐 token 追加。

**面试要点**：这是典型的「框架兼容性 vs 理想实现」的折中——从设计角度 Indexer K cache 应该和 attention KV cache 放在一起，但 `DynamicCache` 的固定 slot 设计不允许。

**延伸阅读**：主报告 CH 3.6（Step 3）/ CH 8.5

---

### Q4.10 为什么 `weights_proj` 必须保持 FP32（不被 FP8 量化）？

**简短回答**：`weights_proj` 为每个 query 位置的 32 个 Indexer head 生成重要性权重，这些权重直接决定 token 选择的排序。FP8 的量化误差可能导致 top-2048 的排序结果漂移，使关键的 token 被错误排除。

**详细解释**：代码通过 `_keep_in_fp32_modules = ["indexer.weights_proj"]` 明确保护该模块：

- **量化敏感性**：`weights_proj` 的输出维度仅 32（极低维），FP8 的表示精度（E4M3，约 3-4 位有效位数）可能不足以区分 32 个相似的重要性分数
- **误差放大**：weights 在 Step 5 的加权组合中与 scores 矩阵（32 heads × 192K tokens）逐元素相乘。weights 的微小误差会被 192K 的求和操作放大
- **排序敏感**：top-k 选择只关心分数的相对顺序，而非绝对值。FP8 精度可能导致相邻 token 的分数顺序反转，将应被选中的 token 排除在外
- **一致性保证**：`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]` 更进一步保护路由偏置

**面试要点**：不是所有模块都需要 FP32——只有「排序敏感的轻量模块」才需要。大部分 MLA 投影和 MoE expert 可以安全地用 FP8。

**延伸阅读**：主报告 CH 7.3 / CH 8.5 / 代码 `modeling_glm_moe_dsa.py:L662-L663`

---

### Q4.11 Indexer 的 Q 投影为什么复用 MLA 的 `q_resid`？

**简短回答**：避免重复计算——MLA 已经通过 q_a_proj（6144→2048）对 hidden state 做了 Q 方向的语义压缩，Indexer 直接使用这个压缩表示做进一步的查询投影，节省参数量和计算量。

**详细解释**：数据流：

```
hidden_states [B,S,6144]
  → q_a_proj (6144→2048) → q_a_layernorm → q_resid [B,S,2048]
      ├→ q_b_proj (2048→64×256) → MLA Q（注意力用）
      └→ Indexer wq_b (2048→32×128) → Indexer Q（打分用）
```

复用的优势：
1. **参数量节省**：如果 Indexer 从 hidden_states 直接投影，需要 Linear(6144, 4096)=25.2M 参数；而现在 wq_b 只需 Linear(2048, 4096)=8.4M（节省 67%）
2. **语义对齐**：MLA Q 和 Indexer Q 共享相同的语义压缩基础，Indexer 选择的 token 更可能与 MLA 注意力关注的 token 对齐
3. **训练效率**：Indexer 可以与 MLA 的 Q 压缩联合优化

**面试要点**：这是典型的「表征共享」设计——不同模块（MLA Attention 和 Indexer）共享底层投影结果，减少冗余计算。

**延伸阅读**：主报告 CH 3.2（Step 1）/ CH 3.6（Step 1）

---

### Q4.12 Indexer 的 K 为什么只有 128 维且不区分多 head？

**简短回答**：Indexer 的 K 只用于「粗筛打分」而非「精确注意力」，128 维足够表达 token 的基本语义特征。不区分多 head 是为了共享 K cache（所有 32 个 Indexer head 使用同一个 K），节省计算和存储。

**详细解释**：

**128 维的充分性**：
- Indexer 的任务是选出「大致相关」的 token，不需要像 attention 那样精确建模 token 间的语义关系
- 128 维 = 6144 的约 2.1%，是 heavy compression，但筛选任务不需要高维语义表示
- wk 投影：Linear(6144, 128) 仅需 0.79M 参数

**不区分多 head 的原因**：
- Q 有 32 个 head（每个 focus 不同的语义方面），但 K 对所有 head 一致
- 这是类似 MQA（Multi-Query Attention）的设计——「多 query head 共享同一套 key」
- 好处：K cache 只有 T×128 个元素（而非 T×32×128），大幅减少 Indexer 的缓存开销

**Q 多 head + K 单 head 的不对称设计**是精巧的：Q 的 32 个 head 提供了打分角度的多样性，K 的单 head 避免了缓存的 32× 膨胀。

**面试要点**：Q multi-head + K single-head 是 MQA 的变体——「不同的查询角度但相同的键表示」。

**延伸阅读**：主报告 CH 3.2（Step 2）/ CH 3.6（Step 2）

---

### Q4.13 Causal mask 在 Indexer 中如何应用？

**简短回答**：`index_scores = index_scores + attention_mask`，将未来位置（t > s）的分数设为 -inf，确保 top-k 选择只从已见过的 token 中选。

**详细解释**：因果 mask 的应用在 Step 6（L222-L223）：

```
if attention_mask is not None:
    index_scores = index_scores + attention_mask
```

attention_mask 是一个 [B, S, T] 的张量，其中：
- 允许关注的位置：mask = 0.0（不改变分数）
- 禁止关注的位置（未来 token）：mask = -inf（使分数变为 -inf）

加上 -inf 后，ReLU 打分结果仍为 0（ReLU(-inf) = 0），但 top-k 在排序时这些位置自然排到最后，永远不会被选中。这是标准的 Transformer 因果 mask 技巧——不需要 if-else 判断，直接在分数上加 mask 即可。

**面试要点**：mask 放在 Indexer 打分后、top-k 选择前——这是为了保证 token 选择不「偷看」未来。

**延伸阅读**：主报告 CH 3.2（Step 6）/ CH 3.6（Step 6）

---

### Q4.14 稀疏 Mask 是如何从 topk_indices 构建的？

**简短回答**：创建一个全 -inf 的 [B, S, T] tensor，在 topk_indices 指定的位置 scatter 0.0，然后与因果 mask 合并，非 top-2048 位置的注意力权重自然为 0。

**详细解释**（代码 L416-L432）：

```
# Step 1: 创建全 -inf mask
index_mask = torch.full([B, S, T], float("-inf"), ...)

# Step 2: 在 topk_indices 位置填入 0.0
index_mask.scatter_(-1, topk_indices, 0.0)
# 现在 top-2048 位置 = 0.0, 其余 = -inf

# Step 3: 与因果 mask 合并
combined_mask = index_mask.unsqueeze(1) + causal_mask
# causal_mask: [B, 1, S, T]
# 未来位置 = -inf（从 causal_mask 来）
# 过去的非 top-2048 位置 = -inf（从 index_mask 来）
# 过去的 top-2048 位置 = 0.0

# Step 4: 传给 attention 函数
attn_weights = softmax(QK^T / sqrt(d) + combined_mask)
# -inf 位置 → softmax 后 = 0
# 0.0 位置 → softmax 后 = 正常注意力权重
```

这个技巧的精妙之处：不需要修改 attention kernel，只需构造正确的 mask 并传入，标准的 softmax 就会自动忽略非 top-2048 位置的 token。

**面试要点**：`scatter_` + `-inf` 是 Transformer 稀疏注意力的标准 mask 构建技巧——不要自己实现稀疏 attention kernel，而是让标准 softmax attention 通过 mask 自然实现稀疏。

**延伸阅读**：主报告 CH 3.6（稀疏 Mask 构建）

---

### Q4.15 为什么 top-k 选择 2048 而不是其他值？

**简短回答**：2048 是来自 `index_topk` 配置字段的实验选择。太小则稀疏注意力覆盖不足（可能遗漏关键 token），太大则 Indexer 的计算节省有限。在 192K 下，2048 意味着仅 1% 的 token 参与注意力。

**详细解释**：top-k 选择是一个 trade-off：

| k 值 | 覆盖率 | Indexer 成本 | 注意力成本 | 性能 |
|---|---|---|---|---|
| 512 | 0.27% | 低 | 极低 | 显著劣于 Full Attention |
| 1024 | 0.53% | 低 | 低 | — |
| **2048** | **1.07%** | 中 | 低 | 接近 Full Attention |
| 4096 | 2.13% | 中 | 中 | 接近 Full Attention |
| 8192 | 4.27% | 中 | 中高 | 可能无增益 |

从论文消融看，2048 是「性能饱和点」——再大收益递减。与 V4-Flash CSA 的 512 对比，DSA 用 4× 的 top-k 换取了 (1) 不做 KV 压缩的 token 级精度，(2) 更接近 Full Attention 的性能。考虑到 DSA 的 Indexer 本身也对 K cache 打分（有计算成本），top-2048 恰好使 Indexer+稀疏注意力和 Full Attention 之间达到 Pareto 最优。

**面试要点**：2048 不是拍脑袋的数字——它来自 9B scale 的消融实验 + 192K 上下文下的覆盖率分析。面试中如果被问到，应该提到「1% 覆盖率 + 接近 Full Attention 精度」这两个关键数据点。

**延伸阅读**：主报告 CH 2.3（约束 1）/ CH 3.5（消融表）

---

### Q4.16 DSA 的「近乎无损」特性意味着什么？

**简短回答**：DSA 在 RULER@128K 上仅比 Full Attention 低 0.35 分，在所有高效注意力方法中是唯一能做到「精度损失 < 1 分」的方案。这意味着 DSA 不是用精度换速度的折中方案，而是用智能选择换效率的优化方案。

**详细解释**：「近乎无损」有四个层面的含义：

1. **实验验证**：9B 消融实验中 DSA 在多个长度（16K/32K/64K/128K）上的 RULER 分数均接近或超过 Full Attention。744B 规模上的性能印证了这一结论。
2. **理论支撑**：DSA 的「先粗筛后精算」保证了被选中的 2048 个 token 上的是**精确注意力**（非近似），信息损失只可能来自 Indexer 的遗漏——而非 attention 计算的近似。
3. **实用意义**：DSA 使「78 层 × 192K 上下文」的经济部署成为可能。如果没有 DSA，Full Attention 在 192K 下的 QK^T 计算（6.44 GFLOPs/层）将使 78 层 decode 不可承受。
4. **范式意义**：DSA 证明了稀疏注意力不必是「次优方案」——在合适的 selection mechanism 下，稀疏注意力可以与全注意力等价。

**面试要点**：如果有人质疑「稀疏注意力必然有精度损失」，用 RULER@64K 上 DSA 超越 Full Attention（87.06 vs 85.35）的数据反驳。

**延伸阅读**：主报告 CH 3.5 / CH 10.1（三个核心 Insight 第 1 点）
# GLM-5.1 架构 QA（2/2）—— MLA / MoE / 训练 / 源码 / M2.7 对比 / 面经

> 配套 `main-report.md` CH 4-10 | 版本 v0.1 | 撰写日期 2026-06-10

---

## 五、MLA 潜 KV 压缩（11 Q）

### Q5.1 MLA 的核心压缩原理是什么？

**简短回答**：MLA 通过 `kv_a_proj_with_mqa` 将 KV 从 64×256×2=32768 维压缩到 576 维（kv_lora_rank=512 + k_pe=64），缓存时只存这 576 维，使用时通过 `kv_b_proj` 展开为 64 头 × 448 维的 K 和 V。

**详细解释**：MLA 的压缩-展开流程：

```
hidden_states [B,S,6144]
  → kv_a_proj_with_mqa [6144→576]
  → compressed_kv [B,S,576]
    ├→ k_compressed [B,S,512] → kv_a_layernorm
    │   → kv_b_proj [512→64×448] → kv_expanded [B,S,64,448]
    │     ├→ k_nope [B,64,S,192]
    │     └→ value_states [B,64,S,256]
    └→ k_pe [B,S,64] → RoPE → expand(64) → [B,64,S,64]
```

压缩比：压缩前需要 64×(256+256)=32768 维/ token，压缩后只需 512+64=576 维/ token，压缩比 = 32768/576 ≈ 57×。

MLA 最精妙的设计：k_pe（RoPE 位置信息）不参与潜压缩——因为 RoPE 的正交旋转操作不可逆（无法从压缩空间恢复），所以 k_pe 以原始形式存储（64 维），与 k_nope（192 维，从压缩空间展开）分开处理。

**面试要点**：MLA 的 576 是 512（压缩 KV）+ 64（单头 RoPE K），两者分开存储和展开。

**延伸阅读**：主报告 CH 4.1 / DeepSeek-V2 arXiv:2405.04434

---

### Q5.2 q_lora_rank=2048 比 V4-Flash 的 1024 大有什么考量？

**简短回答**：GLM-5.1 的 hidden_size=6144 是 V4（4096）的 1.5×，q_lora_rank 相应提升至 2×（2048 vs 1024），目的是保持 Q 压缩比大致不变（hidden/q_lora_rank ≈ 3 vs V4 的 4），避免 Q 投影的信息瓶颈。

**详细解释**：

| 维度 | V4-Flash | GLM-5.1 | 比值 |
|---|---|---|---|
| hidden_size | 4096 | 6144 | 1.5× |
| q_lora_rank | 1024 | 2048 | 2.0× |
| 压缩比 | 4.0 | 3.0 | — |
| q_b_proj 参数量 | 1024×32×192 | 2048×64×256 | ~2.6× |

更大的 q_lora_rank 意味着：
- Q 投影保留了更多原始 hidden state 信息 → 查询质量更高
- q_b_proj 参数量增加（2048→16384 vs 1024→6144）
- 压缩比更保守（3.0 vs 4.0），更不容易出现信息瓶颈

这反映 GLM-5.1 的「宽而浅」设计哲学：hidden 更大 → Q 需要更多压缩带宽 → q_lora_rank 翻倍。

**面试要点**：q_lora_rank 不是越大越好——更大的 q_lora_rank 增加了 Q 投影的参数量。2048 是 hidden=6144 下的经验最优值。

**延伸阅读**：主报告 CH 2.3（约束 4）/ CH 4.3

---

### Q5.3 为什么 K 的 RoPE 部分（k_pe）是所有 attention head 共享的（单头）？

**简短回答**：RoPE 编码的是 token 间的**相对位置关系**，这个关系对 64 个 attention head 是相同的。将 k_pe 设为单头（64 维而非 64×64=4096 维）避免了冗余存储，同时不损失任何信息——不同 head 通过各自的 k_nope 来区分查询语义。

**详细解释**：K 的构成分为两部分：

- **k_nope**（192 维/头）：来自 `kv_b_proj` 展开的压缩 KV，64 头各不同——携带各个 head 需要的不同语义信息
- **k_pe**（64 维，单头）：RoPE 位置编码，64 头共享——只编码相对位置，与语义无关

k_pe 的处理流程：
```
k_pe [B,S,64]
  → view(B, 1, S, 64)          # 单头（head dim = 1）
  → apply_rotary_pos_emb(cos, sin)  # 施加 RoPE
  → expand(-1, 64, -1, -1)     # 广播到 64 头
  → cat with k_nope             # [B, 64, S, 192+64=256]
```

在 MLA 压缩缓存中，只缓存 64 维的 k_pe（而非 64×64=4096 维），这是 KV cache 从 ~936 GB 降至 ~19 GB 的关键贡献之一。

**面试要点**：k_pe 单头是 MLA 节省 KV cache 的关键设计之一——如果每个 head 都有独立的 RoPE K，压缩缓存的效果会大打折扣。

**延伸阅读**：主报告 CH 4.2（K 侧）

---

### Q5.4 Partial RoPE 的设计意图是什么？

**简短回答**：将 QK 维度分为无位置编码部分（nope，192 维）和有位置编码部分（rope，64 维），使 QK 内积同时编码位置相关和位置无关的信息，在长上下文中保留了语义匹配的稳定性。

**详细解释**：QK 内积的展开：

$$
Q \cdot K^T = Q_{\text{nope}} \cdot K_{\text{nope}}^T + Q_{\text{rope}} \cdot K_{\text{rope}}^T
$$

- **第一项（nope × nope）**：纯内容匹配，不受位置影响。语义相似的 token 即使距离很远也能获得高分——这对长上下文检索至关重要
- **第二项（rope × rope）**：位置相关匹配，得分随相对位置变化。近距离 token 获得更高的位置加分，维持了注意力的局部性偏好（locality bias）

完整的 Full RoPE 会将所有 256 维都施加旋转位置编码，此时语义相似但距离远的 token 可能因为位置旋转向量正交而得分不高。Partial RoPE 通过保留 192 维的「纯内容通道」解决了这个问题。

**面试要点**：Partial RoPE = 位置编码的「软开关」——192 维纯内容 + 64 维纯位置，两者独立发挥作用。

**延伸阅读**：主报告 CH 4.2（Q 侧）/ 附录 B

---

### Q5.5 MLA 中 QK Norm 是如何实现的？

**简短回答**：GLM-5.1 的 MLA 中，Q 使用 `q_a_layernorm` 对压缩后的 Q 残差做 RMSNorm，K 使用 `kv_a_layernorm` 对压缩后的 K（512 维）做 RMSNorm。Indexer 的 K 使用 `k_norm`（LayerNorm）做归一化。

**详细解释**：代码中的三个 Norm 位置：

1. **`q_a_layernorm`**（L365）：`q_resid = q_a_layernorm(q_a_proj(x))`——对压缩后的 2048 维 Q 残差做 RMSNorm，保证送入 q_b_proj 的信号幅度稳定。
2. **`kv_a_layernorm`**（L375）：`k_compressed = kv_a_layernorm(k_compressed)`——对压缩后的 512 维 K 做 RMSNorm，保证从压缩空间展开的 K 和 V 幅度稳定。
3. **`k_norm`**（L184，Indexer 内）：对 Indexer 的 128 维 K 做 LayerNorm，保证 Indexer 打分稳定性。

与 M2.7 的「每层 QK Norm」不同，GLM-5.1 没有在 attention score 计算前对 Q 和 K 做显式的 per-head normalization。GLM-5.1 的 Norm 是在压缩-展开路径中内置的（q_a_layernorm + kv_a_layernorm），其效果类似于在 QK 投影源头上做归一化。

**面试要点**：GLM-5.1 的「隐式 QK Norm」通过 LoRA 路径中的 LayerNorm 实现，与 M2.7 的「显式 per-head QK Norm」是两种不同的设计路线。

**延伸阅读**：主报告 CH 4.2 / CH 4.5

---

### Q5.6 MLA 的 compressed cache 和 expanded cache 有什么区别？

**简短回答**：Compressed cache 存储未展开的 576 维压缩 KV（19.2 GB @192K），expanded cache 存储展开后的 32768 维完整 K V（936 GB @192K）。当前 HF 实现使用 expanded cache 以兼容标准 `DynamicCache`，compressed cache 是论文设计的「future optimization」。

**详细解释**：

| 缓存模式 | 存储内容 | 每层大小 | 78 层总计 | 实现状态 |
|---|---|---|---|---|
| Compressed（理论） | 512 压缩 KV + 64 k_pe | 246 MB | 19.2 GB | 未实现（future optimization） |
| Expanded（当前） | 64×256 K + 64×256 V | 12 GB | 936 GB | HF `DynamicCache` 当前实现 |

Compressed cache 的使用流程（需要实现专门的 MLA Cache 类）：
```
解码时：
1. 从 compressed cache 读取 k_compressed(512) + k_pe(64)
2. kv_b_proj(k_compressed) → 展开为 K(64×192) + V(64×256)
3. k_pe 广播到 64 头
4. 拼接 K = [k_nope, k_pe] = [B, 64, S, 256]
5. 注意力计算
6. K, V 不写回缓存（直接从压缩态读取）
```

代码注释（L279-L282）明确指出 compressed cache decode 路径需要「dedicated MLA cache class」——这是一个重要的工程遗留问题。

**面试要点**：生产部署应该用 compressed cache（~19 GB），当前 HF 实现是为了框架兼容性的权宜之计。面试中主动提到这一点说明你关注了实现细节。

**延伸阅读**：主报告 CH 2.5 / 代码 `modeling_glm_moe_dsa.py:L279-L282`

---

### Q5.7 Muon Split 是什么？为什么在 MLA 中重要？

**简短回答**：Muon Split 是将 MLA 的 Q/K/V 展开投影矩阵按 head 拆分为多个小矩阵，分别做 Muon 正交化的训练技巧。解决了标准 MLA 在使用 Muon 优化器时不同 head 权重被错误耦合的问题，将 MLA 性能从显著劣于 GQA-8 恢复到接近持平。

**详细解释**：问题根源和解决方案：

**问题**：Muon 优化器对投影矩阵做整体正交化时，将 64 个 attention head 的权重耦合在一起。不同 head 应该有不同的更新方向和尺度，整体正交化破坏了这一多样性。

**Muon Split 解决方案**：
- 将 $W_{U}^Q, W_{U}^K, W_{U}^V$（展开投影矩阵）按 head 维度拆分为 64 个独立的小矩阵
- 分别对每个小矩阵做矩阵正交化
- 保持不同 head 的更新独立性

**效果**（论文 Table 1 节选）：

| 模型变体 | MMLU | BBH | HumanEval |
|---|---|---|---|
| GQA-8 (baseline) | 61.2 | 53.3 | 38.5 |
| MLA（标准，无 Split） | 61.5 | 48.9 | 33.5 |
| MLA + Muon Split | 62.5 | 51.8 | 36.7 |

Muon Split 使 MLA 的 BBH 从 -4.4 恢复到 -1.5，HumanEval 从 -5.0 恢复到 -1.8。论文还提到使用 Muon Split 后 attention logit 规模保持稳定，无需裁剪策略。

**面试要点**：Muon Split 是优化器与架构的「合力」而非「对抗」——它暴露了 MLA 在特定优化器下的脆弱性，但解决方案非常简单（按 head 拆分投影矩阵）。

**延伸阅读**：主报告 CH 4.4 / GLM-5 paper §2.1, Table 1

---

### Q5.8 GLM-5.1 MLA 与 V4-Flash MLA 的维度差异意味着什么？

**简短回答**：GLM-5.1 的 q_lora_rank（2048 vs 1024）、v_head_dim（256 vs 128）、num_heads（64 vs 32）均为 V4 的 2×，反映其「大 hidden + 宽 attention」的设计哲学，而 V4 更强调「轻量 + 紧凑」。

**详细解释**：

| 维度 | V4-Flash MLA | GLM-5.1 MLA | 含义 |
|---|---|---|---|
| q_lora_rank | 1024 | 2048 | Q 压缩带宽 2× |
| kv_lora_rank | 512 | 512 | KV 压缩相同 |
| qk_nope_head_dim | 128 | 192 | 内容维度 +50% |
| qk_rope_head_dim | 64 | 64 | 位置维度相同 |
| v_head_dim | 128 | 256 | V 信息容量 2× |
| num_heads | 32 | 64 | 头数 2× |
| GQA | MQA（1 KV head） | 全 MHA（64 KV heads） | 设计路线不同 |

关键洞察：kv_lora_rank 两者一致（512）——说明 512 维潜空间对 KV 压缩是「甜点值」。差异集中在 Q 侧（GLM-5.1 更宽）和 V 侧（GLM-5.1 更大），这与 hidden_size 差异（6144 vs 4096）一致。

**面试要点**：两者的 kv_lora_rank 均为 512 说明「512 是 KV 潜压缩的通用最佳维度」——太少信息不足，太多压缩比优势丧失。

**延伸阅读**：主报告 CH 4.3

---

### Q5.9 `kv_a_proj_with_mqa` 中的 "mqa" 是什么意思？

**简短回答**："mqa" 指 Multi-Query Attention，表示 KV 的 RoPE 部分（k_pe）是单头的（所有 64 个 attention head 共享同一个 k_pe），而非 rope 部分（k_nope）是多头的（来自 kv_b_proj 展开，64 头独立）。

**详细解释**：`kv_a_proj_with_mqa` 的输出维度为 512 + 64 = 576：

```
kv_a_proj_with_mqa: Linear(6144, 576, bias=False)
输出:
  ├→ k_compressed [B,S,512]  # 压缩 KV（潜空间），多头的——展开后 64 头各自不同
  └→ k_pe [B,S,64]           # RoPE 位置信息，单头的（MQA）——64 头共享
```

命名中的 "mqa" 特指 k_pe 是单头的设计——这是 MLA 标准设计的一部分：位置信息（RoPE）对所有 head 是共享的，语义信息（k_nope）是 per-head 的。DeepSeek-V2 的 MLA 也有相同的 "mqa" 概念。

**面试要点**：`kv_a_proj_with_mqa` 的 576 = 512（多头的压缩 KV）+ 64（单头的 RoPE K）。记住 512+64=576 这个算式。

**延伸阅读**：主报告 CH 4.2 / 代码 L370

---

### Q5.10 MLA 中 Q 的压缩-展开路径是怎样的？

**简短回答**：`hidden_states` → `q_a_proj`（6144→2048）→ `q_a_layernorm` → `q_resid` → `q_b_proj`（2048→64×256）→ reshape → split（nope 192 + rope 64）→ RoPE on rope 部分 → cat 组装。

**详细解释**：Q 路径的完整代码流程（L358-L367）：

```python
# Step 1: 压缩
q_resid = q_a_proj(x)              # [B,S,6144] → [B,S,2048]
q_resid = q_a_layernorm(q_resid)   # RMSNorm over 2048

# Step 2: 展开
query_states = q_b_proj(q_resid)   # [B,S,2048] → [B,S,64×256]

# Step 3: 重塑
query_states = query_states.view(B, S, 64, 256)
query_states = query_states.transpose(1, 2)  # [B,64,S,256]

# Step 4: Partial RoPE
q_nope, q_pe = split(query_states, [192, 64], dim=-1)
q_pe = apply_rotary_pos_emb(q_pe, cos, sin)

# Step 5: 组装
query_states = cat([q_nope, q_pe], dim=-1)  # [B,64,S,256]
```

Q 路径的参数量：q_a_proj（12.58M）+ q_b_proj（33.55M）= 46.13M/层。q_resid（2048 维）还被 Indexer 的 wq_b 复用（见 Q4.11）。

**面试要点**：Q 的压缩是可选的（MLA 标准允许无 Q 压缩），但 GLM-5.1 选择压缩以配合 Indexer 复用 q_resid。

**延伸阅读**：主报告 CH 4.1 / CH 4.5

---

### Q5.11 GLM-5 将 head_dim 从 192 增至 256、头数减少 1/3 的原因？

**简短回答**：为了适配不同硬件（非 H800 的 roofline），增大 head_dim 降低 head 数，保持训练计算量不变，同时降低 decode 计算量。

**详细解释**：论文 §2.1 描述了这一调整：

| 调整 | 前 | 后 | 影响 |
|---|---|---|---|
| head_dim | 192 | 256 | +33% |
| num_heads | ~96 | 64 | -33% |
| 每头 QK 维度 | 192 | 256 | 总 QK 维度不变（96×192 = 64×256 = 6144 × 某种投影） |
| Decode 计算 | — | 降低 | 头数更少 → QK^T 的 head 维度减少 |

设计思路：总 QK 维度（num_heads × head_dim）保持不变（6144），但减少 head 数、增大 head_dim 后：
- matmul 操作变成更少的头 × 更大的维度 → 更适合 memory-bound 硬件的 roofline
- decode 阶段每个 head 的 attention 计算量减少（头数少）

这是一个典型「roofline-aware」的架构调整——不是改变模型表达能力，而是让计算 pattern 更适合目标硬件的计算-内存带宽比。

**面试要点**：这是针对非 H800 硬件的 roofline 优化——H800 的高带宽可能让更多小 head 更优，但国产芯片的 roofline 不同。

**延伸阅读**：主报告 CH 4.4 / GLM-5 paper §2.1

---

## 六、MoE 路由（11 Q）

### Q6.1 Sigmoid 路由的完整流程是怎样的？

**简短回答**：Router 投影（Linear 6144→256）→ Sigmoid 独立打分 → +e_score_correction_bias（负载均衡）→ 分组筛选 → Top-8 选择 → 权重归一化 → ×2.5 放大 → Expert dispatch → +共享专家。

**详细解释**：8 步完整流程（`route_tokens_to_experts` L558-L581）：

```
Step 1: router_logits = gate(x)                              # [B*S, 256] linear 投影
Step 2: router_logits = router_logits.sigmoid()               # 每个 expert (0,1) 独立打分
Step 3: scores = router_logits + e_score_correction_bias      # 加偏置做负载均衡
Step 4: topk_indices = torch.topk(scores, k=8)[1]             # 选 top-8
Step 5: topk_weights = router_logits.gather(topk_indices)
        topk_weights /= topk_weights.sum(dim=-1)              # 归一化（和=1）
Step 6: topk_weights = topk_weights * 2.5                     # routed_scaling_factor 放大
Step 7: expert dispatch → index_add_ 加权汇总                  # 8 个 expert 输出加权求和
Step 8: output = expert_output + shared_expert(residual)       # 加共享专家
```

Sigmoid（而非 softmax）是 GLM-5.1 的 MoE 评分函数的标志性选择。

**面试要点**：流程中有三个关键数字——256（expert 总数）、8（top-k）、2.5（缩放因子）。

**延伸阅读**：主报告 CH 5.1 / 代码 `glm_dsa_moe.py`

---

### Q6.2 routed_scaling_factor=2.5 的数学推导是怎样的？

**简短回答**：sigmoid 值在 (0,1)，top-8 归一化后每个 expert 权重约 0.125（假设 8 个分数接近），Expert FFN 输出幅值约为 Dense FFN 的 12.5%。×2.5 后有效权重约 0.3125，使 MoE 层输出接近 Dense 层的幅值水平。

**详细解释**：数学推导：

1. 假设 8 个被选中 expert 的 sigmoid 分数均为 $s=0.5$：
   - 归一化权重：$w_i = \frac{0.5}{0.5 \times 8} = 0.125$
   
2. Expert FFN 输出 $E_i(x)$ 的期望幅值与 Dense FFN 输出 $D(x)$ 相近（均为 SwiGLU 结构，参数量相近）：
   - MoE 加权输出：$y = \sum_{i=1}^{8} w_i \cdot E_i(x) = 8 \times 0.125 \times E(x) = E(x)$
   - 看起来加权后幅值 = 单个 expert 的输出，接近 Dense FFN
   
3. 但实际上 sigmoid 分数分布不均——高分的 expert 分数可能高达 0.9，低分的可能 0.3：
   - 归一化后：$w_{\text{max}} = \frac{0.9}{0.9+7\times 0.3} = \frac{0.9}{3.0} = 0.3$
   - 主 expert 有效权重 ≈ 0.3，其余 7 个权重极低
   - 输出仍偏小
   
4. ×2.5 后：$w_{\text{eff}} \in [0, 2.5]$，典型值约 0.3-0.8，使 MoE 输出幅值与 Dense FFN 对齐

这个因子反映了 GLM 团队对 MoE 残差流数值特性的深入理解——不是简单的调参，而是基于残差流标准化需求的精算。

**面试要点**：2.5 不是"越大越好"——太大可能破坏残差流稳定性，太小则信号衰减。这个值很可能来自内部实验的 sweep 结果。

**延伸阅读**：主报告 CH 5.2 / CH 2.3（约束 2）

---

### Q6.3 Sigmoid vs Softmax 的工程权衡是什么？

**简短回答**：Sigmoid 的优点是独立性（多 expert 可同时高分）和计算简单（无需 exp+sum），缺点是指数区分度低（需 bias 补偿）。Softmax 优点是天然区分度高（分数和为 1 的竞争分布），缺点是互斥性强（不适合 token 同时需要多种能力的场景）。

**详细解释**：

| 维度 | Sigmoid | Softmax | 评价 |
|---|---|---|---|
| Expert 独立性 | 完全独立 | 互相竞争 | Sigmoid 适合复合需求 |
| 分数区分度 | 低（容易相近） | 高（指数拉开） | Softmax 选择更干脆 |
| Top-k 选择 | 需手动 top-k | 天然 top-k | 效率等同 |
| 负载均衡 | 需要 bias 辅助 | 自然实现 | Softmax 天然均衡 |
| 计算复杂度 | 每 expert 1 次 sigmoid | 每 expert 1 次 exp | Sigmoid 略快 |
| 与 MoE 兼容性 | 需要 routing bias | 需要 auxiliary loss | 两种方案都成熟 |

GLM-5.1 的选择说明：在「支持 token 同时激活多个不相关的 expert」和「让 expert 之间有清晰的竞争关系」之间，GLM-5.1 选择了前者。这对于 Agent 场景（token 可能同时需要代码+数学+常识）是合适的。

**面试要点**：两者没有绝对优劣——取决于设计哲学。如果面试中需要做架构选择，先问清楚模型的主要场景。

**延伸阅读**：主报告 CH 5.3

---

### Q6.4 e_score_correction_bias 的作用是什么？

**简短回答**：`e_score_correction_bias` 是一个 256 维的可学习/可调节 bias 向量，加载在 sigmoid 评分之上用于负载均衡——降低过载 expert 的分数，提升欠载 expert 的分数，防止路由坍塌。

**详细解释**：没有路由偏置的风险：

- 某些 expert 的权重矩阵可能「通用性」更强（对所有 token 都有高 sigmoid 分数）
- 这些 expert 会独占 top-8 位，导致 (1) 其他 expert 缺乏训练信号，(2) 推理时负载不均
- 极端情况：8 个 expert 处理 90% token，其余 248 个 expert 闲置——MoE 退化为「k=8 的 mini Dense」

`e_score_correction_bias` 的实现：
- 类型：`register_buffer`（非 `nn.Parameter`）
- 含义：不参与梯度更新（optimizer.step 不更新它），由外部负载均衡机制维护
- 更新方式：论文未公开详细机制，但从 `DeepSeek-V3` 的经验推断，可能基于 expert 的平均激活频率 / 负载统计做增量更新

代码保护：`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`，该 bias 严格保持 FP32（不被 FP8 量化），因为它的值直接影响路由选择。

**面试要点**：区分 `e_score_correction_bias`（buffer，不参与梯度更新）和 Router weight（Parameter，参与梯度更新）。

**延伸阅读**：主报告 CH 5.1（Step 3）/ CH 8.5

---

### Q6.5 n_group=1 意味着什么？与 M2.7 的 n_group=8 有何差异？

**简短回答**：n_group=1 意味着 GLM-5.1 不使用分组路由——所有 256 个 expert 在同一池中竞争 top-8。M2.7 的 n_group=8、topk_group=4 则将 256 expert 分为 8 组，先在 32 expert 组内选 top-4 组，再在组内选 top expert。

**详细解释**：

| 维度 | GLM-5.1 | M2.7 |
|---|---|---|
| n_group | 1（无分组） | 8 |
| topk_group | 1 | 4 |
| 选择方式 | 全局 top-8 | 8 组中选 4 组，每组内再选 |
| 负载均衡 | 依赖 e_score_correction_bias | 分组自然均衡 + auxiliary loss |
| 路由自由度 | 高（256 选 8） | 约束更严格 |

无分组路由的优点：
- 路由更灵活——任何 token 可以从 256 个 expert 中自由选择
- 实现更简单——不需要分组筛选逻辑
- 适合 expert 数量适中的场景（256 expert，全局 top-8 计算开销不大）

分组路由的优点：
- 更强的负载均衡保证——每组限制了 token 流入，避免「赢家通吃」
- 适合更多 expert 的场景（如 >512 expert，全局路由复杂度高）
- 与 auxiliary loss 配合更容易稳定

**面试要点**：GLM-5.1 的无分组路由说明团队对 sigmoid + e_score_correction_bias 的负载均衡能力有信心，不需要分组来辅助。

**延伸阅读**：主报告 CH 5.1（Step 3）/ CH 9.3

---

### Q6.6 共享专家（shared expert）的设计理由是什么？

**简短回答**：共享专家为所有 token 提供通用的语义处理能力（如基础语法、常识、安全对齐），不经过路由选择，所有 token 都必须经过。这保证了 MoE 路由的最低质量基线——即使路由选择不理想，token 仍然获得了来自共享专家的核心处理。

**详细解释**：共享专家在架构中的位置（`GlmMoeDsaMoE.forward`）：

```python
shared_expert_output = self.shared_experts(identity)           # 所有 token 都经过
routed_output = self.experts(hidden_states, topk_indices, ...) # 8 个路由 expert
final_output = routed_output + shared_expert_output            # 相加
```

设计理由：
1. **质量兜底**：即使所有路由 expert 的得分都很低（sigmoid 分数接近 0），共享专家仍提供基本的 FFN 处理
2. **路由稳定性**：共享专家的存在降低了 Router 的决策压力——Router 只需要选「增值的 expert」，基础能力由共享专家保证
3. **训练效率**：共享专家始终收到梯度信号（所有 token 都经过它），训练比路由专家更稳定
4. **残差对齐**：共享专家 37.75M 参数 ≈ 1 个 Dense FFN 的 1/6，为 MoE 层提供类似 Dense 层的「通用基底」

M2.7 没有共享专家，完全依赖路由选择——这是两者 MoE 设计哲学的又一差异。

**面试要点**：共享专家 vs 无共享专家 = 「有安全网 vs 纯竞争」的哲学差异。GLM-5.1 选择留安全网。

**延伸阅读**：主报告 CH 5.1（Step 8）/ CH 9.3

---

### Q6.7 Expert dispatch 的代码实现是怎样的？

**简短回答**：通过 one_hot(mask) + index_add_ 实现。首先将 top-8 的 expert 选择转为 one-hot mask [256, 8, B*S]，然后按 expert 维度遍历，对每个 expert 处理它被分配到的所有 token，加权累加到输出。

**详细解释**：`GlmMoeDsaNaiveMoe.forward` 的核心逻辑（L517-L534）：

```python
# Step 1: One-hot mask
expert_mask = F.one_hot(topk_index, num_classes=256)   # [B*S, 8, 256]
expert_mask = expert_mask.permute(2, 1, 0)              # [256, 8, B*S]

# Step 2: Expert 迭代
final_hidden_states = torch.zeros(B*S, hidden_size)
for expert_idx in range(256):
    if expert_mask[expert_idx].sum() == 0:
        continue                                          # 无 token 分配给此 expert
    
    # 取出分配给此 expert 的所有 token
    token_idx, top_k_pos = expert_mask[expert_idx].nonzero(as_tuple=True)
    current_state = hidden_states[token_idx]
    
    # SwiGLU FFN forward
    gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
    current_hidden = silu(gate) * up
    current_hidden = F.linear(current_hidden, down_proj[expert_idx])
    
    # 加权（乘以 routing weight）
    current_hidden = current_hidden * topk_weights[token_idx, top_k_pos, None]
    
    # 累加到输出
    final_hidden_states.index_add_(0, token_idx, current_hidden)
```

`index_add_` 是个关键操作：当多个 expert 处理同一 token 时（k=8），它们的输出通过 `index_add_` 自然求和。这避免了额外的合并步骤。

**面试要点**：当前实现是 naive 循环——对 256 expert 逐个迭代。生产部署中会用 group_gemm / MoE kernel（如 DeepEP）并行处理。

**延伸阅读**：主报告 CH 5.6 / 代码 `glm_dsa_moe.py`

---

### Q6.8 GLM-5.1 的负载均衡是如何实现的？

**简短回答**：通过 `e_score_correction_bias`（256 维 buffer）动态调节各 expert 的分数，配合 `topk_method: "noaux_tc"` 的无辅助损失策略实现负载均衡。不使用 auxiliary load balancing loss。

**详细解释**：`topk_method: "noaux_tc"` 的含义：

- **noaux**：不使用 auxiliary loss（辅助负载均衡损失）。传统 MoE（如 Switch Transformer, GShard）通过添加 $\alpha \cdot \mathcal{L}_{\text{aux}}$ 来鼓励平衡的路由分布。GLM-5.1 选择了不加辅助损失。
- **tc**：可能代表 token-choice（由 token 选择 expert）或某种特定策略。论文未详细公开。

代替 auxiliary loss 的机制：
1. `e_score_correction_bias`：动态调整每个 expert 的基准分，过载 expert → 降低 bias，欠载 expert → 提升 bias
2. `norm_topk_prob: true`：top-8 的权重归一化确保权重和为 1
3. 256 expert × k=8 的高容量意味着每个 expert 平均每 token 被选中的概率只有 8/256=3.1%，负载自然分布较均匀

**面试要点**：noaux_tc 是相对少见的策略——大多数 MoE 论文使用 auxiliary loss。面试官可能会质疑「没有辅助损失能平衡吗」，回答是「sigmoid + e_score_correction_bias + 大 expert 池自然均衡」。

**延伸阅读**：主报告 CH 5.1 / config.json

---

### Q6.9 每 token 在 MoE 层激活多少参数？如何计算？

**简短回答**：每层 MoE 激活约 339.8M 参数（8 个路由 expert × 37.75M + 1 个共享 expert × 37.75M）。75 层 MoE 合计约 25.5B 激活参数。

**详细解释**：单个 expert 的参数分解（SwiGLU 结构）：

```
gate_up_proj: [4096, 6144]  = 25.17M  # gate(2048→6144) + up(2048→6144) 合并
down_proj:    [6144, 2048]  = 12.58M  # 降维投影
```
单个 expert = 25.17M + 12.58M = **37.75M**

MoE 层激活量：
- 8 个路由 expert：8 × 37.75M = 302M
- 1 个共享 expert：1 × 37.75M = 37.75M
- Router 权重：256 × 6144 = 1.57M
- **每层 MoE 激活合计**：~**339.8M**

75 层 MoE + 3 层 Dense + 78 层 Attention = ~40B 总激活参数。

**面试要点**：区分「expert 权重」（37.75M/expert）和「激活 expert 数」（8+1=9/层）。75 层 × 9 × 37.75M ≈ 25.5B 来自 MoE。

**延伸阅读**：主报告 CH 2.4 / CH 5.5

---

### Q6.10 75 层 MoE 中每层的 expert 权重是共享的还是独立的？

**简短回答**：完全独立的——每层有自己专属的 256 个路由 expert 权重矩阵。75 层 × 256 expert = 19,200 个独立的 expert 权重。不同层的同一编号 expert 学到的是不同语义深度的知识。

**详细解释**：这与 MoE 的架构设计理念一致：
- 浅层 expert（第 4-10 层）：学习浅层语义特征（语法、词性、基础常识）
- 中层 expert（第 20-50 层）：学习复杂语义关系（推理、知识关联、代码逻辑）
- 深层 expert（第 60-78 层）：学习任务特定特征（生成策略、Agent 决策、输出格式化）

如果所有层共享 expert 权重：
- 参数总量大幅减少（744B → 约 70B）
- 但不同深度的语义处理需求完全不同，共享权重会严重限制表达能力
- 这正是 GLM-5.1 选择 744B 而非更小模型的原因

Expert 权重不共享是标准的 MoE 设计选择（Mixtral、DeepSeek-V3 同理），因为 expert 的核心价值就是提供深度特定的专用能力。

**面试要点**：75 层独立 × 256 expert = 19,200 个独立的 FFN 权重矩阵——这是 744B 参数的来源。

**延伸阅读**：主报告 CH 5.5

---

### Q6.11 Router 的 weight 和 e_score_correction_bias 有什么不同的训练方式？

**简短回答**：Router weight（`gate.weight`）是 `nn.Parameter`，参与梯度更新——通过反向传播学习。`e_score_correction_bias` 是 `register_buffer`，不参与梯度更新——由外部负载均衡机制按频率/负载统计动态维护。

**详细解释**：

| 属性 | gate.weight | e_score_correction_bias |
|---|---|---|
| 类型 | `nn.Parameter` | `register_buffer` |
| 梯度 | 参与 | 不参与 |
| 更新方式 | optimizer.step() | 外部机制（频率统计） |
| 维度 | [256, 6144] | [256] |
| FP8 保护 | 是（标准量化） | 是（`_keep_in_fp32_modules_strict`） |
| 作用 | 学习 token→expert 的语义映射 | 调节 expert 负载均衡 |

分离设计的智慧：
- Router 学习「这个 token 应该找哪个 expert」（语义映射，需要梯度）
- Routing bias 维护「这个 expert 是不是太忙了」（负载均衡，不需要梯度，甚至不应该用梯度——否则 bias 会退化回 Router 的辅助变量）

**面试要点**：「哪些 MoE 参数有梯度？哪些没有？」——Router weight 有，routing bias 没有。这是 MoE 负载均衡的关键设计选择。

**延伸阅读**：主报告 CH 5.6 / 代码 `GlmMoeDsaTopkRouter.__init__`

---

## 七、训练体系（9 Q）

### Q7.1 GLM-5.1 的预训练数据构成是怎样的？

**简短回答**：总计 28.5T tokens，包含 Web（改进分类器）、Code（低资源语言增强 +28% 去重后唯一 token）、Math & Science（LLM 评分筛选）三大类。

**详细解释**：

| 数据类别 | 特点 | 说明 |
|---|---|---|
| Web | DCLM + World Knowledge 分类器 | 在 GLM-4.5 数据管线上改进 |
| Code | 模糊去重，低资源语言专用分类器 | 去重后唯一 token 增加 28%；覆盖 Scala/Swift/Lua 等 |
| Math & Science | LLM 评分筛选 | 网页 + 书籍 + 论文中的高质量数理数据 |

训练数据不是等比例混合的——不同阶段的 mid-training 使用不同的数据分布：
- 32K 阶段（1T tokens）：通用分布
- 128K 阶段（500B tokens）：增加长文档比例
- 200K 阶段（50B tokens）：新增 MRCR（多轮共指解析）数据增强超长多轮召回

**面试要点**：28.5T vs DeepSeek-V3 的 14.8T——GLM-5 的训练数据量更接近 Llama-3 的 15T+ 级别，属于较大规模的预训练数据。

**延伸阅读**：主报告 CH 6.1 / GLM-5 paper §2

---

### Q7.2 Mid-training 三阶段上下文扩展的设计逻辑是什么？

**简短回答**：32K（1T tokens）→ 128K（500B tokens）→ 200K（50B tokens），从短到长渐进式扩展，越长的阶段 token 数越少（因为超长序列的 I/O 和计算成本更高）。

**详细解释**：

| 阶段 | 上下文长度 | Token 量 | 数据特点 |
|---|---|---|---|
| Stage 1 | 32K | 1T | 通用分布，建立基础长上下文能力 |
| Stage 2 | 128K | 500B | 增加长文档比例，适应中长上下文 |
| Stage 3 | 200K | 50B | MRCR 数据，超长多轮对话召回 |

设计逻辑：
1. **渐进式**：直接从短上下文跳到 200K 可能导致注意力模式失真（位置编码需要在长距离上适应）
2. **token 递减**：200K 阶段仅 50B tokens 是因为每条序列 200K token，同样 GPU 小时处理的序列数少得多
3. **数据专项化**：200K 阶段加入 MRCR 数据——在超长上下文中准确检索特定信息需要专项训练

这种「先宽后窄再尖」的策略在长上下文训练中很常见（Llama-3 也用类似方法）。

**面试要点**：三个阶段的 token 递减（1T → 500B → 50B）不是训练预算缩减，而是由于超长序列的 batch 效率下降导致的自然设计。

**延伸阅读**：主报告 CH 6.1 / CH 7.1 / GLM-5 paper §2.3

---

### Q7.3 异步 RL 基础设施的架构是怎样的？

**简短回答**：生成引擎（推理）和训练引擎（优化）部署在不同 GPU 设备上，生成引擎持续生成 rollout 轨迹，达到阈值后批量发送给训练引擎更新模型。每 K 步梯度更新后，训练引擎将新权重推回生成引擎。

**详细解释**：slime 框架（the slime framework）的关键组件：

```
┌─────────────────────┐          ┌─────────────────────┐
│   Generation Engines │ ──batch──→ │   Training Engine    │
│  (多个推理 GPU)      │ ←─weights─ │  (训练 GPU)          │
│                     │          │                     │
│  • 1k+ 并发 rollout │          │  • 梯度累积 + 更新  │
│  • MTP spec. decoding│          │  • 参数同步（每 K 步） │
│  • FP8 推理         │          │  • 模型权重推送     │
│  • PD disaggregation│          │                     │
└─────────────────────┘          └─────────────────────┘
            │
    ┌──────────────────┐
    │ Multi-Task       │
    │ Orchestrator     │
    │  • 心跳检测      │
    │  • 容错(下线,非重试)│
    │  • 任务注册      │
    └──────────────────┘
```

关键设计：(1) 解耦避免 GPU 轮流闲置（训练时推理 GPU 空闲，反之亦然），(2) 心跳容错而非重试——不健康 server 直接下线（节省资源），(3) FP8+MTP+PD 三重优化尾延迟。

**面试要点**：异步 RL 的核心挑战是「参数同步时机」和「尾延迟」——GLM-5 的方案是每 K 步同步 + 三重加速。

**延伸阅读**：主报告 CH 6.2 / GLM-5 paper §3.6, §4.1

---

### Q7.4 TITO（Token-In-Token-Out）网关解决了什么问题？

**简短回答**：消除 detokenize → re-tokenize 循环中的 token 边界、空格/规范化、截断等不匹配问题，让训练管线直接消费推理引擎产出的 token ID 流（而非文本）。

**详细解释**：传统的异步 RL 数据流：

```
推理引擎生成文本 → detokenize → 文本传输 → tokenize → 训练引擎消费 token
```

问题：
- **Token 边界偏移**：原 tokenizer 的分词可能与恢复后重新分词的边界不同
- **空格/规范化**：文本传输中可能引入空格变化（不同的平台/编码处理）
- **截断**：文本截断后再 tokenize 可能产生原不存在的 token 序列

TITO 的方案：

```
推理引擎生成 token IDs → 直接传输 token ID 流 → 训练引擎消费 token IDs
```

token IDs 是整数序列，不存在任何文本编码/解码的问题。训练和推理使用完全一致的 token 序列，消除了 RL 中一个重要的分布偏移来源。

**面试要点**：TITO 体现了「在 RL 中保持 token 一致性」的重要性——微小的 tokenization 偏差在 RL 中被累积放大。

**延伸阅读**：主报告 CH 6.3 / GLM-5 paper §4.1.2

---

### Q7.5 Direct Double-sided Importance Sampling 是什么？

**简短回答**：用 rollout 时的 log-probability 直接替代 old-policy 的 log-probability，避免维护多个历史 checkpoint。结合双边裁剪 $[1-\epsilon_\ell, 1+\epsilon_h]$ 丢弃极端偏离的 token。

**详细解释**：标准的 PPO 需要维护 $\pi_{\theta_{\text{old}}}$ 来计算 importance ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$。在异步 RL 中，多个 rollout 可能对应不同的 old-policy 版本，维护多个历史 checkpoint 成本高且复杂。

Direct Double-sided IS 的公式：

$$
r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t))
$$

$$
f(x; \epsilon_\ell, \epsilon_h) = \begin{cases} x, & \text{if } 1-\epsilon_\ell < x < 1+\epsilon_h \\ 0, & \text{otherwise} \end{cases}
$$

关键简化：用 rollout 时的 log-prob（$\log \pi_{\text{rollout}}$）替代 $\log \pi_{\theta_{\text{old}}}$。因为 rollout 使用推理引擎的当前模型参数，而训练引擎使用的是 K 步前的参数——但二者差异在 K 较小时可控。

双边裁剪的好处：既防止过大的 importance ratio（上界 $1+\epsilon_h$），也防止过小的 ratio（下界 $1-\epsilon_\ell$，直接丢弃），两者都表示当前策略与 rollout 策略偏离过大。

**面试要点**：这是 PPO 在异步场景的轻量替代——不用维护 old-policy checkpoint，简化了系统复杂度。

**延伸阅读**：主报告 CH 6.3 / GLM-5 paper §4.1.2

---

### Q7.6 后训练的完整 pipeline 包含哪些阶段？

**简短回答**：SFT（多类 + Interleaved Thinking）→ Reasoning RL（GRPO + IcePop）→ Agentic RL（异步 RL, 三大 Agent 域）→ General RL（三维优化 + 混合奖励）→ On-Policy Cross-Stage Distillation（恢复前序技能）。

**详细解释**（论文 §3 描述）：

| 阶段 | 重点 | 关键技术 |
|---|---|---|
| **1. SFT** | 基础指令跟随 | Interleaved / Preserved / Turn-level Thinking，202K 上下文 |
| **2. Reasoning RL** | 数学/科学/代码/TIR | GRPO + IcePop，确定性 top-k（torch.topk），冻结 Indexer |
| **3. Agentic RL** | SWE / Terminal / Search | 异步 RL，slime 框架，DP-aware routing |
| **4. General RL** | 正确性 + 情感智力 + 质量 | 混合奖励（规则+结果+生成 RM），人工样本锚点 |
| **5. Distillation** | 技能恢复 | On-policy 从前序阶段蒸馏，避免灾难性遗忘 |

Interleaved Thinking 是 SFT 的创新：每次响应和工具调用前都插入思考块，训练模型的 step-by-step reasoning 能力。Preserved Thinking 则跨轮保留思考块（不被历史的 attention mask 截断）。

**面试要点**：5 阶段设计的核心——先打基础（SFT），再练专项（Reasoning + Agentic RL），再统合（General RL），最后恢复（Distillation）。

**延伸阅读**：主报告 CH 6.4 / GLM-5 paper §3.1-§3.5

---

### Q7.7 Muon 优化器的特点是什么？

**简短回答**：Muon 是通过矩阵正交化来优化神经网络参数的优化器，对投影矩阵做牛顿-舒尔茨迭代使其接近正交。GLM-5 使用 Muon + cosine decay + batch size warmup。

**详细解释**：Muon 的核心特点：

- **矩阵级更新**：标准优化器（Adam, SGD）对参数做逐元素更新。Muon 将参数矩阵视为整体，通过牛顿-舒尔茨迭代将其正交化，使更新方向的「矩阵结构」更优
- **适配 Transformer**：Transformer 的参数主要是线性投影矩阵（Q/K/V/O 投影、FFN 矩阵），非常适合 Muon 的矩阵级处理
- **Muon Split**：GLM-5 发现 Muon 对 MLA 的全矩阵正交化会导致不同 head 权重耦合，通过按 head 拆分投影矩阵解决

GLM-5 的训练配置：lr 从 0 warmup 到 2e-4，衰减到 4e-5。Mid-training 从 4e-5 线性到 1e-5。论文沿用 GLM-4.5 的优化器设置，说明 Muon 在 GLM 训练体系中有持续验证。

**面试要点**：Muon 不同于 Adam 的原因是它不做逐元素的自适应学习率——而是做矩阵级的正交化更新。两者的融合（Muon + Adam）是当前 LLM 训练的趋势。

**延伸阅读**：主报告 CH 6.1 / GLM-5 paper Appendix A

---

### Q7.8 训练工程优化有哪些关键技巧？

**简短回答**：(1) 灵活 MTP 放置（与主输出层共 pipeline stage），(2) Pipeline ZeRO2 梯度分片（double buffering），(3) Muon 零冗余通信，(4) Pipeline 激活卸载（CPU offload），(5) 序列分块输出投影，(6) INT4 QAT（位级一致）。

**详细解释**：

| 技巧 | 解决的问题 | 方法 |
|---|---|---|
| 灵活 MTP 放置 | MTP 层增加内存和通信开销 | 与主输出层放在最后一个 pipeline stage |
| Pipeline ZeRO2 | 梯度存储 + 通信开销 | 每 stage 仅保留 1/dp 梯度，2 stage 轮转 |
| Muon 零冗余通信 | Muon 优化器的通信开销 | 限制 all-gather 到各 rank 拥有的参数分片 |
| Pipeline 激活卸载 | warmup 期间显存峰值 | forward 后卸载到 CPU，backward 前加载 |
| 序列分块输出投影 | 长序列投影/ loss 峰值显存 | 切分成小块独立计算 |
| INT4 QAT | 训练-推理量化不一致 | SFT 阶段应用，kernel 位级一致 |

这些优化共同使 744B 模型的训练在工程上成为可能。大多数是成熟的分布式训练技巧，但组合使用在 744B 规模上的经验具有参考价值。

**面试要点**：不要只列技巧名——要说明每个技巧解决了什么问题。面试官更关心你理解「为什么需要」而非「是什么」。

**延伸阅读**：主报告 CH 6.5 / GLM-5 paper §2.4

---

### Q7.9 GRPO + IcePop 是什么？

**简短回答**：GRPO（Group Relative Policy Optimization）是一种通过组内相对比较替代绝对奖励的 RL 算法。IcePop 是 GLM-5 论文引入的 GRPO 变体/改进，用于 Reasoning RL 阶段的数学/科学/代码/TIR 多域混合训练。

**详细解释**：GRPO 的核心思想：

- 为每个 prompt 生成一组候选回答（group），组内做相对比较
- 优势函数 = 组内相对排名（而非绝对奖励值）
- 避免了 PPO 中价值模型（critic）的训练开销
- 适合 reward model 评分噪声大的场景

IcePop 的改进（论文 Eq.(1)）：论文 §3.2 提到了 GRPO+IcePop 的 loss 公式，但详细算法被标注为来自 GLM-5 论文。IcePop 可能引入了：(1) 温度调节的重要性采样，(2) 对极端奖励的裁剪，(3) 与异步 RL 基础设施的兼容层。

从使用方式推断，GRPO+IcePop 在 Reasoning RL 阶段被用在 TIR（Tool-Integrated Reasoning）等需要逐步骤奖励的场景中，而 Agentic RL 阶段则使用 PPO 变体。

**面试要点**：GRPO 的「组内相对比较」消除对价值模型的依赖——这是从 PPO 到 GRPO 的核心简化。

**延伸阅读**：主报告 CH 6.4 / GLM-5 paper §3.2, Eq.(1)

---

## 八、源码与系统（9 Q）

### Q8.1 GLM-5.1 的 HuggingFace 代码仓库结构是怎样的？

**简短回答**：位于 `transformers/models/glm_moe_dsa/` 目录，核心文件包括 `configuration_glm_moe_dsa.py`（超参）、`modeling_glm_moe_dsa.py`（893 行模型实现）、`modular_glm_moe_dsa.py`（自动生成前的母版）、`tokenization_glm_moe_dsa.py`（Tokenizer）。

**详细解释**：

```
transformers/src/transformers/models/glm_moe_dsa/
├── __init__.py
├── configuration_glm_moe_dsa.py        # GlmMoeDsaConfig
├── modeling_glm_moe_dsa.py             # 893 行模型实现（自动生成）
├── modular_glm_moe_dsa.py              # 模块化母版源码
└── tokenization_glm_moe_dsa.py         # Tokenizer (vocab_size=154880)
```

重要说明：`modeling_glm_moe_dsa.py` 的头部注释声明此文由 `modular_glm_moe_dsa.py` 自动生成，不可手动编辑。这说明 GLM 团队使用 HuggingFace 的 `modular_xxx` 机制来分离「开发者手写代码」和「自动生成的 HF 兼容代码」，类似于 DeepSeek 的开发模式。

Transformers 版本：5.4.0（初始）→ 5.10.2（后续升级）。

**面试要点**：`modular_xxx.py` → `modeling_xxx.py` 的自动生成管线是 HF transformers 5.x 的新特性，GLM-5.1 是早期 adopters。

**延伸阅读**：主报告 CH 8.1 / HF 仓库 `zai-org/GLM-5.1`

---

### Q8.2 11 个关键类及其功能是什么？

**简短回答**：从底层到顶层依次为：RMSNorm → RoPE → Indexer → Attention → MLP → Router → NaiveMoe → MoE → DecoderLayer → Model → ForCausalLM。

**详细解释**：

| 类 | 代码行 | 功能 |
|---|---|---|
| `GlmMoeDsaRMSNorm` | L46-L63 | RMSNorm，epsilon=1e-5 |
| `GlmMoeDsaRotaryEmbedding` | L677-L740 | RoPE，theta=1M，dim=64，interleave=true |
| `GlmMoeDsaIndexer` | L104-L228 | DSA 索引器（Q/K 投影 + ReLU 打分 + top-2048） |
| `GlmMoeDsaAttention` | L268-L459 | MLA+ DSA 集成（Q 压缩→展开，KV 压缩→展开，稀疏 mask 构建） |
| `GlmMoeDsaMLP` | L462-L475 | Dense SwiGLU FFN（gate/up/down, intermediate=12288） |
| `GlmMoeDsaTopkRouter` | L478-L495 | MoE Router（Linear + bias） |
| `GlmMoeDsaNaiveMoe` | L498-L535 | Expert 权重 3D tensor 存储 + dispatch |
| `GlmMoeDsaMoE` | L538-L591 | MoE 层（Router + Experts + Shared Expert） |
| `GlmMoeDsaDecoderLayer` | L594-L639 | Decoder Layer（Attn + FFN，Dense/MoE 分支） |
| `GlmMoeDsaModel` | L744-L816 | 78 层主模型循环 |
| `GlmMoeDsaForCausalLM` | L820-L893 | CausalLM wrapper + generate |

辅助函数：`rotate_half`, `apply_rotary_pos_emb`, `repeat_kv`, `eager_attention_forward`。

**面试要点**：记住类之间的层级关系——Indexer 是 Attention 的子模块，Router+NaiveMoe+SharedExpert 是 MoE 的子模块。

**延伸阅读**：主报告 CH 8.2 / SOURCES.md

---

### Q8.3 代码片段速查——核心功能的代码行号

**简短回答**：Indexer 打分 + top-k（L177-L228）、MLA Q 路径（L358-L367）、MLA KV 路径（L370-L388）、稀疏 Mask 构建（L416-L432）、MoE 路由（L558-L581）、Expert Dispatch（L517-L534）、78 层主循环（L799-L810）。

**详细解释**：

| 功能 | 行号 | 核心操作 |
|---|---|---|
| Indexer Q 投影 | L177-L181 | `wq_b(q_resid)` → view(32,128) → RoPE |
| Indexer K 投影 | L184-L187 | `k_norm(wk(x))` → split → RoPE |
| Indexer 打分 | L217-L220 | `einsum + relu + einsum` |
| MLA Q 路径 | L358-L367 | `q_a_proj → q_a_layernorm → q_b_proj → split → RoPE` |
| MLA KV 路径 | L370-L388 | `kv_a_proj_with_mqa → layernorm → kv_b_proj → split → RoPE` |
| 稀疏 Mask | L416-L432 | `torch.full(-inf) → scatter(topk, 0.0) → merge with causal` |
| MoE sigmoid 路由 | L558-L581 | `sigmoid + bias + top8 + normalize + ×2.5` |
| Expert Dispatch | L517-L534 | `one_hot + index_add_` |
| 78 层循环 | L799-L810 | `for layer in layers: x, topk = layer(x, ...) + skip_topk` |

这些代码片段也是面试中经常被要求「手写伪代码」的点（特别是 Indexer 打分和 MLA KV 路径）。

**面试要点**：面试官可能会让你画出 Indexer forward 的伪代码——记住 7 步，尤其是 Q 投影复用 q_resid 这一点。

**延伸阅读**：主报告 CH 8.4 / 代码片段目录

---

### Q8.4 skip_topk 优化机制是什么？

**简短回答**：部分层的 Indexer 标记为 "shared" 类型（`skip_topk=True`），复用上一层的 topk_indices，避免每层都重新运行 Indexer 打分，减少 decode 时的 Indexer 开销（~1.63 GFLOPs/层）。

**详细解释**：代码中的实现（L340-L343, L396-L400）：

```python
# Indexer 类型标记
self.skip_topk = self.indexer is None or self.indexer.type == "shared"

# Attention forward 中的复用逻辑
if not self.skip_topk or prev_topk_indices is None:
    topk_indices = self.indexer(hidden_states, q_resid, ...)
else:
    topk_indices = prev_topk_indices  # 复用上层结果
```

设计意图：相邻层的 Indexer 选择结果可能高度相似（token 的重要性在不同深层之间变化缓慢），复用上一层的 top-2048 选择可以减少 50% 的 Indexer 打分次数。

使用场景推测：
- 浅层（第 1-3 层）：不 skip（语言特征变化快，需要每层独立打分）
- 深层（第 60-78 层）：可以 skip（深层语义变化慢，token 重要性相对稳定）

**面试要点**：这是一个「层间稀疏性共享」优化——利用深层 token 重要性变化慢的特性减少 Indexer 开销。

**延伸阅读**：主报告 CH 8.5 / 代码 L340-L343, L396-L400

---

### Q8.5 Flash-MLA kernel 的适配状态如何？

**简短回答**：当前状态为 `_supports_flash_attn = False`，已注册 `_compatible_flash_implementations = ["kernels-community/flash-mla"]`，但 kernel 仍在适配中。当前回退到标准 attention（eager / SDPA）。

**详细解释**：代码中的相关配置（L649, L664）：

```python
# 类属性
_supports_flash_attn = False
_compatible_flash_implementations = ["kernels-community/flash-mla"]
# 未注册: _flash_attn_implementation 和 _supports_sdpa
```

影响：
- 当前 MLA 的 forward-pass 中的 `attention_interface` 回退到标准 SDPA（PyTorch 的 `scaled_dot_product_attention`）
- 标准 SDPA 需要完整的 QKV 矩阵 [B, 64, S, 256]，在 192K 长上下文下速度可能不理想
- `kernels-community/flash-mla` 是一个社区维护的 MLA-specific fused kernel，专门优化 MLA 的 QK^T 计算模式

Flash-MLA 适配完成后，64 × 2048 × 256 的稀疏注意力计算应该有显著的加速（预估 2-4×），这将直接提升 decode TPS。

**面试要点**：知道 flash-mla kernel 还在开发中说明你关注了实现细节。面试中可以提到「如果 flash-mla 集成完成，预计 decode TPS 提升 2-4×」。

**延伸阅读**：主报告 CH 8.5 / CH 10.2（已知局限）

---

### Q8.6 FP8 量化保护机制是如何实现的？

**简短回答**：通过 `_keep_in_fp32_modules` 和 `_keep_in_fp32_modules_strict` 两个类属性，将量化敏感的模块（Indexer 的 weights_proj 和 routing bias）排除在 FP8 转换之外。

**详细解释**：代码中的保护声明（L662-L663）：

```python
_keep_in_fp32_modules = ["indexer.weights_proj"]
_keep_in_fp32_modules_strict = ["e_score_correction_bias"]
```

两个列表的区别：
- `_keep_in_fp32_modules`：包含此名称的模块保持 FP32（如 `decoder.layers.0.self_attn.indexer.weights_proj`）
- `_keep_in_fp32_modules_strict`：名称精确匹配的模块严格保持 FP32（如 `e_score_correction_bias`）

这两个列表被 HuggingFace 的 `PreTrainedModel._autocast_smart_context_manager` 在 FP8 量化转换时读取，自动跳过列表中的模块。

为什么需要保护：
- `weights_proj`：32 维低维输出，FP8 精度可能导致排序漂移（见 Q4.10）
- `e_score_correction_bias`：直接影响路由选择，任何量化误差都可能导致 expert 分配错误

**面试要点**：FP8 量化不是「全或无」——关键小模块可以保留 FP32。这种「选择性量化」是生产部署的标准做法。

**延伸阅读**：主报告 CH 7.3 / CH 8.5

---

### Q8.7 MTP 参数共享是如何工作的？

**简短回答**：训练时 3 个 MTP 层共享同一套参数（而非 3 套独立参数），推理时用 1 层预测 2 个 future token。参数共享避免了 3× 的 MTP 参数量，同时通过训练-推理的一致性设计保持了较高的接受率。

**详细解释**：与传统 MTP 的对比：

| 方案 | 训练 MTP 层数 | 推理 MTP 层数 | 预测 token 数 | 额外参数 |
|---|---|---|---|---|
| 传统 MTP (DeepSeek-V3) | 1 | 1 | 1 | ~1× |
| MTP×3 (独立参数) | 3 | 3 | 3 | ~3× |
| **GLM-5.1 (参数共享)** | **3 (共享)** | **1** | **2** | **~1×** |

参数共享的关键设计：
- 3 个训练 MTP 层使用相同的 Embedding + Output Head + Transformer Block 参数
- 各层的输入不同（来自不同的 hidden state 位置），但处理方式相同
- 推理时只用 1 层，但通过深度迭代预测 2 个 token

效果（论文 Table 2）：accept length = 2.76（vs DeepSeek-V3.2 的 2.55），说明参数共享并没有削弱 MTP 的预测能力。第 2 个 token 的接受率不下降，归功于训练时的「3 层共享参数」让模型学会了在单一层内同时优化多个 depth 的预测。

**面试要点**：MTP 参数共享是「用训练时的多 depth 经验补偿推理时的单层限制」——模型学到的参数适应了多 depth 的输入分布。

**延伸阅读**：主报告 CH 7.4 / GLM-5 paper §2.1

---

### Q8.8 192K 上下文是如何实现的？

**简短回答**：通过 `max_position_embeddings=202752` + `rope_theta=1,000,000` + DSA 稀疏注意力（top-2048）+ Mid-training 三阶段上下文扩展（32K→128K→200K）共同实现。

**详细解释**：192K 上下文的三大技术支撑：

1. **位置编码**：`rope_theta=1M` 确保 RoPE 在 192K 距离上的旋转角度不重复，维持位置区分度。`rope_interleave=true` 采用交错模式（而非相邻模式）减少高频分量的衰减。
2. **DSA 稀疏注意力**：每层只对 top-2048 个 token 做精确注意力，避免 $O(T^2)$ 的计算爆炸。Indexer 的 top-2048 独立于序列总长度（除 Indexer 本身需要扫描所有 T 个 token）。
3. **渐进式 Mid-Training**：32K（1T tokens）→ 128K（500B）→ 200K（50B），确保模型在每个长度区间都充分训练。

三个技术缺一不可：位置编码保证了 token 位置可区分，DSA 保证了计算可承受，渐进训练保证了模型质量。

**面试要点**：192K 上下文的「真实可用性」取决于 RULER@128K 等长上下文评测——DSA 在 128K 上仅比 Full Attention 低 0.35 分，说明 192K 是可以实际使用的。

**延伸阅读**：主报告 CH 7.1 / CH 3.5

---

### Q8.9 层次化上下文管理策略是什么？

**简短回答**：Keep-recent-k（保留最近 k 轮观察，折叠更早的为占位符）+ Discard-all（超过阈值时丢弃所有工具调用历史，重新开始）。组合使用在 BrowseComp 上从 55.3% 提升至 75.9%。

**详细解释**：论文 §4.2.4 提出的搜索 Agent 上下文管理策略：

**Keep-recent-k**：
- 当交互历史超过 k 轮后，将第 k 轮之前的观察折叠为占位符（placeholder）
- 占位符保留「曾经发生过的交互」这一信息（不丢弃），但不保留完整内容
- 类似滑动窗口但保留了对早期交互的「存在性」记忆

**Discard-all**：
- 当总上下文超过阈值 T 时，丢弃所有工具调用历史
- 从新上下文重新开始（reset to new session）
- 防止历史过长导致 Indexer 选择效率下降

**组合策略**：
- 日常使用：Keep-recent-k（保留近期完整 + 远期占位符）
- 超长上下文：Discard-all（彻底重置）
- 效果：BrowseComp 从 55.3% → 75.9%（+20.6%）

**面试要点**：这是搜索 Agent 的「工程技巧」而非架构创新——但对最终效果影响巨大（+20.6%）。体现了架构设计之外，推理策略同样重要。

**延伸阅读**：主报告 CH 7.5 / GLM-5 paper §4.2.4

---

## 九、与 MiniMax-M2.7 的架构对比（11 Q）

### Q9.1 GLM-5.1 和 M2.7 的规模差异有多大？

**简短回答**：GLM-5.1 的总参（744B）是 M2.7（229.9B）的 3.2×，激活参数（40B）是 M2.7（9.8B）的 4.1×。GLM-5.1 选择「更宽更深」（78 层，hidden 6144），M2.7 选择「深而窄」（62 层，hidden 3072）。

**详细解释**：

| 维度 | GLM-5.1 | M2.7 | 比值 |
|---|---|---|---|
| 总参数 | 744B | 229.9B | 3.2× |
| 激活参数 | 40B | 9.8B | 4.1× |
| 层数 | 78 | 62 | 1.26× |
| hidden_size | 6144 | 3072 | 2.0× |
| 参数量 | 3.2× | 1.0× | — |
| 设计哲学 | 更宽广（wider） | 更深窄（deeper-narrower） | — |

M2.7 的「mini activations」设计（hidden 3072）强调单 token 推理效率，9.8B 激活参数使其在消费级 GPU 上有更好的推理性能。GLM-5.1 用更大的 hidden_size 换取更强的单 token 表达能力，代价是更高的推理成本。

两者代表了 2026 年 Agent LLM 的两个主流规模方向：「宽模型」（GLM-5.1, DeepSeek-V3）和「窄模型」（M2.7, Qwen-3）。

**面试要点**：规模差异是理解 GLM-5.1 vs M2.7 一切差异的基础——几乎所有的架构选择都可以追溯到规模差异。

**延伸阅读**：主报告 CH 9.1

---

### Q9.2 两者在 Attention 策略上的核心差异是什么？

**简短回答**：GLM-5.1 使用 DSA（动态稀疏注意力）+ MLA（KV 压缩），M2.7 使用 Full Attention（全局注意力）+ GQA（KV 头共享）+ QK Norm。GLM-5.1 用「智能选择」降低计算，M2.7 用「全局覆盖」保证质量。

**详细解释**：

| 维度 | GLM-5.1 | M2.7 |
|---|---|---|
| Attention 类型 | **DSA**（top-2048 稀疏） | **Full Attention**（全局） |
| KV 压缩 | **MLA**（潜空间压缩） | GQA（head 数削减） |
| KV 头数 | 64（全 MHA） | 8（GQA ratio=6） |
| QK Norm | 隐式（LoRA 路径） | 显式（per-layer QK Norm） |
| Head dim | 256 | 128 |
| rope_theta | 1M | 5M |

**核心哲学对立**：
- GLM-5.1：「让聪明的 Indexer 决定哪些 token 值得关注」——用算法智能换系统效率
- M2.7：「算力足够，直接看全部」——用计算暴力换算法简洁

这两种策略都已被证明有效——DSA 在 192K 下几乎无损（RULER@128K 仅 -0.35），Full Attention 天然不遗漏任何信息。

**面试要点**：这个对比是面试中最常问的架构 trade-off 题——没有标准答案，关键是要能分析 DSA 的「概率性遗漏风险」和 Full Attention 的「计算成本上限」。

**延伸阅读**：主报告 CH 9.2 / CH 9.5

---

### Q9.3 两者在 MoE 设计上有什么差异？

**简短回答**：GLM-5.1 有 1 个共享专家（所有 token 必须经过）、无分组路由、routed_scaling_factor=2.5。M2.7 无共享专家、使用 8 组 top-4 分组路由、不设缩放因子。GLM-5.1 更「包容」（共享兜底），M2.7 更「竞争」（全路由决定）。

**详细解释**：

| 维度 | GLM-5.1 | M2.7 |
|---|---|---|
| 路由专家 | 256 | 256 |
| 共享专家 | **1** | **0** |
| k | 8 | 8 |
| routed_scaling_factor | **2.5** | 1.0（无） |
| expert 中间维度 | 2048（hidden 的 33%） | 1536（hidden 的 50%） |
| Dense/MoE 混合 | **前 3 Dense + 后 75 MoE** | 全 62 MoE |
| 分组路由 | n_group=1（无） | n_group=8, topk_group=4 |
| 评分函数 | sigmoid | sigmoid |

关键差异解读：
1. **共享专家**：GLM-5.1 的「安全网」——保证即使路由不理想，token 也能获得基本处理
2. **scaling_factor**：GLM-5.1 的 2.5 是最激进的，M2.7 不缩放——反映了对 MoE 输出幅度的不同判断
3. **expert 相对宽度**：M2.7 的 expert 相对更「胖」（1536/3072=50% vs 2048/6144=33%），反映了「窄模型+胖 expert」vs「宽模型+窄 expert」的策略差异
4. **全 MoE vs Dense 首**：M2.7 全 62 层 MoE 说明其对路由稳定性有充分信心

**面试要点**：共享专家的有无是最关键的 MoE 哲学差异——GLM-5.1 的设计更保守（留兜底），M2.7 更激进（纯路由）。

**延伸阅读**：主报告 CH 9.3

---

### Q9.4 两者的 KV cache 效率对比如何？

**简短回答**：GLM-5.1 的 MLA 压缩 KV cache（~19 GB）仅为 M2.7 的 GQA KV cache（~48.8 GB）的 39%。MLA 用潜压缩实现缓存缩减，GQA 用 head 削减实现缓存缩减。

**详细解释**：在 192K 上下文下的 KV cache 对比：

| 维度 | GLM-5.1 (MLA 压缩) | M2.7 (GQA) | 比值 |
|---|---|---|---|
| 每层 K 维度 | 512+64=576 | 8×128=1024 | 0.56× |
| 每层 V 维度 | 0（含在压缩中） | 8×128=1024 | — |
| 每 token 每层 | 576 | 2048 | 0.28× |
| 每层总 cache | 246 MB | 628 MB | 0.39× |
| 78/62 层总计 | **19.2 GB** | **48.8 GB** | **0.39×** |

MLA 的压缩优势来自：(1) 潜空间压缩（512 维替代 4096 维 K）; (2) V 的维度隐含在压缩 KV 中，无需单独缓存; (3) k_pe 仅 64 维（单头共享）。

GQA 的缓存在 batch 推理中会线性放大（每个请求独立），而 MLA 压缩缓存在 batch 中不变（所有请求共享同一份压缩 KV）。这是 MLA 在生产部署中的重要优势。

**面试要点**：「如果 batch=32，两者的 KV cache 差距是多少？」——MLA 压缩 cache 仍是 ~19 GB，GQA cache 则是 32 × 48.8 = ~1.56 TB。

**延伸阅读**：主报告 CH 9.2 / CH 2.5

---

### Q9.5 DSA vs Full Attention 在 Agent 场景中的适用性如何？

**简短回答**：DSA 在长上下文下节省 72.5% 注意力计算，使 78 层 × 192K 经济可行，但有概率遗漏稀疏关键 token 的风险。Full Attention 全局覆盖无遗漏，但 192K 下计算成本不可承受（>6.4 GFLOPs/层）。

**详细解释**：Agent 场景的特殊性：

- **代码检索**：在 192K 上下文中定位特定函数名——可能只出现 2-3 次。如果 Indexer 未能将其选入 top-2048，attention 将错过关键信息。这是 DSA 最脆弱的场景。
- **长文档总结**：关键信息在上下文中密度较高（每 1K token 有多次提及），Indexer 几乎不可能遗漏。这是 DSA 的优势场景。
- **多轮对话**：最近的几轮最重要（recency bias），Indexer 可以通过 k_pe 的位置信息自然给予近轮高分。DSA 表现良好。

风险缓解：
- 当前：Indexer 的 32 个 head 提供了多角度打分（降低单 head 遗漏风险）
- 未来可能：Indexer+attention 的联合训练让 attention 反向传导信号给 Indexer（「你漏了重要 token，下次注意」）
- 工程侧：对超长上下文做分段处理（每段独立编码），在段间做 cross-attention

**面试要点**：面试官最可能追问的就是「DSA 漏 token 怎么办」——准备上述三条缓解方案。

**延伸阅读**：主报告 CH 9.5

---

### Q9.6 MLA vs GQA 的 KV cache 设计空间有什么不同？

**简短回答**：MLA 通过投影矩阵实现「算法级」缓存压缩——添加投影层，增加计算量，减少缓存。GQA 通过减少 KV 头数实现「结构级」缓存压缩——改变模型结构，不增加计算量，减少缓存效果有限。

**详细解释**：两者在 Pareto 前沿上的位置：

| 维度 | MLA | GQA |
|---|---|---|
| 压缩方式 | 投影压缩 | 头数削减 |
| 压缩比 | 高（57×） | 中（6×, ratio=6） |
| 额外计算 | kv_a_proj + kv_b_proj | 几乎无（repeat_kv 免费） |
| 精度损失 | 低（潜空间可恢复） | 中低（KV 头区分度下降） |
| 实现复杂度 | 高（需要 MLA kernel） | 低（标准 attention kernel） |
| 扩展性 | 可继续增加压缩比 | 受 head 数限制 |

MLA 的理想压缩比可以进一步增加（如 kv_lora_rank=256），但需要权衡信息损失。GQA 的压缩比受 head 数限制（M2.7 中 GQA=6 意味着 48→6 KV heads，压缩比 6×，进一步压缩到 GQA=12 会导致 KV 头区分度严重不足）。

在 batch 推理场景：MLA 的压缩缓存在所有 batch 中共享（一致的大小），而 GQA 的缓存随 batch 线性增长。这是 MLA 在服务端部署中不可替代的优势。

**面试要点**：MLA 的「用计算换存储」vs GQA 的「用精度换存储」——前者在高 batch 场景中更有优势。

**延伸阅读**：主报告 CH 9.5 / CH 2.5

---

### Q9.7 两者的代码能力对比如何？

**简短回答**：GLM-5.1 在 SWE-bench Pro 上 58.4 vs M2.7 的 56.2（略胜），差异在统计误差范围内。两个模型在代码能力上都接近闭源顶级模型。

**详细解释**：

| Benchmark | GLM-5.1 | M2.7 |
|---|---|---|
| SWE-bench Pro | **58.4** | 56.2 |
| SWE-bench Verified | 77.8（GLM-5 论文） | — |
| Code Arena Elo | 1530 | — |
| Terminal-Bench 2.0 | 56.2/60.7† | — |

注意：部分评测数据来源不同（GLM-5 论文 vs 开源社区 vs 官方报告），跨模型比较需谨慎。SWE-bench Pro 的 58.4 vs 56.2 差异仅 2.2 分，属同等水平。

两个模型在代码 Agent 能力上的趋同反映了 2026 年 Agent LLM 的设计共识：MoE + 长上下文 + 强化学习后训练。

**面试要点**：「谁更好」的答案是「同一水平，差异在误差范围内」。关注点应放在两者为达到这一水平采取的不同技术路线。

**延伸阅读**：主报告 CH 9.4

---

### Q9.8 两者的推理速度对比如何？

**简短回答**：社区实测 GLM-5.1 约 44 TPS，M2.7 约 45.6 TPS，差距不大。两者在 192K 长上下文下的推理速度都受限于 KV cache 管理和注意力计算。

**详细解释**：推理速度的制约因素：

| 因素 | GLM-5.1 | M2.7 |
|---|---|---|
| 单 token decode | ~218 GFLOPs | ~150 GFLOPs（估算） |
| Indexer 开销 | 1.63 GFLOPs/层（58.7%） | 0（无 Indexer） |
| KV cache 大小 | 19.2 GB（压缩）/ 936 GB（expanded） | 48.8 GB |
| 瓶颈 | Indexer（长上下文）+ MoE dispatch | Full Attention（长上下文） |
| 实测 TPS | ~44 | ~45.6 |

速度对比需要注意：
1. GLM-5.1 的 expanded cache 实现（~936 GB）远大于论文设计的压缩 cache（~19 GB）——如果用压缩 cache，速度可能提升
2. Flash-MLA kernel 还在适配中——对稀疏注意力的 QK^T 矩阵计算有 2-4× 潜力
3. 测试环境不同（GPU 型号、batch size、序列长度等影响 TPS）

**面试要点**：44 vs 45.6 TPS 的差异无实质意义——两者在同一水平。差异主要来自测试环境而非模型本身。压缩 cache 和 flash-mla kernel 到位后，GLM-5.1 可能更快。

**延伸阅读**：主报告 CH 10.2（已知局限）/ CH 9.4

---

### Q9.9 两者的设计哲学如何总结？

**简短回答**：GLM-5.1——「用算法智能换系统效率」（DSA 索引器 + MLA 压缩 + 大 hidden + 共享专家）。M2.7——「用计算暴力换算法简洁」（Full Attention + GQA + 小 hidden + 无共享专家）。

**详细解释**：两种哲学在四个维度上的体现：

| 维度 | GLM-5.1（算法智能） | M2.7（计算暴力） |
|---|---|---|
| Attention | DSA——Indexer 智能选择 | Full Attention——全量计算 |
| KV Cache | MLA——投影压缩 | GQA——头数削减 |
| MoE | 共享专家兜底 + scaling | 全路由竞争 |
| 规模 | 大 hidden（6144）更宽 | 小 hidden（3072）更窄 |

两种路线在 2026 年都是有效的：
- GLM-5.1 路线适合资源受限但需要长上下文能力的场景（如边缘推理、国产芯片部署）
- M2.7 路线适合计算资源充足、追求极致质量的场景（如云端 API、研究机构）

**趋同点**：两者在 MoE 规模（256 expert × k=8）上趋同——说明「大规模 MoE」已经成为 Agent LLM 的共识基础。差异集中在注意力策略上。

**面试要点**：两个哲学没有绝对优劣——取决于约束条件（计算预算 vs 精度需求 vs 部署环境）。

**延伸阅读**：主报告 CH 9.5 / CH 10.1（三个核心 Insight）

---

### Q9.10 共享专家的有无对 MoE 有什么影响？

**简短回答**：有共享专家（GLM-5.1）提供了质量底线——即使路由选择不理想，token 仍获得基本 FFN 处理。无共享专家（M2.7）完全依赖路由——路由质量直接决定 token 处理质量，但有更少的激活参数。

**详细解释**：

| 维度 | 有共享专家 (GLM-5.1) | 无共享专家 (M2.7) |
|---|---|---|
| 质量底线 | 有（所有 token 经过共享 expert） | 无（路由失败 = token 未充分处理） |
| 激活参数 | 9 × 37.75M = 339.8M/层 | 8 × 37.75M = 302M/层 |
| 训练稳定性 | 共享 expert 始终有梯度信号 | 所有 expert 依赖路由分配梯度 |
| 路由压力 | 低（共享专家分摊基础处理） | 高（路由承担全部处理） |
| 冗余 | 共享专家参数「冗余」（所有 token 都用） | 无冗余 |

共享专家的「冗余」在训练中是优势——它始终接收所有 token 的梯度，提供稳定的训练信号。在 GLM-4.5 和 GLM-5 两代的验证中，共享专家的存在对于 MoE 训练的稳定性有显著正面影响。

**面试要点**：不要认为共享专家是「浪费参数」——它的 37.75M/层（仅占总参的 0.4%）是 MoE 训练稳定性的重要保障。类似于 RL 中的 baseline。

**延伸阅读**：主报告 CH 9.3

---

### Q9.11 rope_theta=1M (GLM-5.1) vs 5M (M2.7) 的设计差异说明了什么？

**简短回答**：M2.7 的 5M 是 GLM-5.1 的 5×，因为 M2.7 使用 Full Attention（所有位置的 token 都需要通过 RoPE 的位置点积来区分远近），需要更大的 theta 来保持 192K 远程的区分度。GLM-5.1 的 DSA 通过 Indexer 显式选择 Top-2048 token，不依赖 RoPE 来「找到」相关 token，因此 1M 足够。

**详细解释**：

- **rope_theta 的作用**：控制 RoPE 旋转频率。theta 越大 → 高频分量衰减越慢 → 远距离 token 的位置区分度越高。
- **Full Attention 的需求**：在 192K 上下文中，所有 192K^2 对 token pair 都需要通过 QK 内积（含 RoPE 位置项）来判断相关性。如果 theta 太小，相距 100K+ 的 token 之间的位置编码几乎不可区分（旋转向量接近平行/正交），QK 内积的位置部分退化为常数。
- **DSA 的需求**：Indexer 通过内容匹配（k_norm + weights_proj + 32 头打分）显式选出相关 token，不依赖 RoPE 来建立远程关联。RoPE 只在 Indexer 的 Q/K 投影中作用于前 64 维，提供基本的「近 vs 远」信号，不需要极精细的远程位置区分。

设计洞察：rope_theta 不是越大越好——更大的 theta 在短上下文下可能过度区分相邻 token 的位置，导致注意力过于分散。1M vs 5M 的选择与注意力策略直接耦合。

**面试要点**：rope_theta 的大小是「由注意力策略驱动的」——全注意力需要大 theta，稀疏注意力可以用小 theta。

**延伸阅读**：主报告 CH 7.1 / CH 9.2

---

## 十、面经（17 Q）

### Q10.1 请用 3 分钟介绍 GLM-5.1 的架构。

**简短回答**：GLM-5.1 是 78 层 Transformer Decoder，采用 MoE（256+1 expert, top-8）+ DSA（动态稀疏注意力, top-2048）+ MLA（潜 KV 压缩, kv_lora_rank=512）三位一体架构，总参 744B，激活 40B。

**详细解释**：

**面试官视角**：面试官想看你在 3 分钟内抓住最核心的特征、建立全局认知。不要陷入细节——先给 big picture，再点出最独特的创新。

**答题模板**：

"GLM-5.1 的架构可以归纳为'一个基础，三大创新，一个数字'。

**一个基础**：78 层 Pre-Norm Transformer Decoder，前 3 层 Dense FFN + 后 75 层 MoE FFN。hidden_size=6144，64 个 attention heads。

**三大创新**：
1. **DSA（动态稀疏注意力）**——这是最核心的创新。用一个叫 Indexer 的轻量模块从 192K 上下文中动态选 top-2048 个最相关的 token，只在这些 token 上做精确注意力。复杂度从 $O(T^2)$ 降到 $O(T \cdot k)$，节省 72.5% 注意力计算。在 RULER@128K 上仅比全注意力低 0.35 分——近乎无损。
2. **MLA（潜 KV 压缩）**——对 KV 做 512 维潜空间压缩，KV cache 从 ~936 GB 降到理论 ~19 GB。配合 DSA，实现了长上下文推理的经济性。
3. **MoE（混合专家）**——256 个路由专家 + 1 个共享专家，每 token 激活 top-8。评分函数用 sigmoid（不是 softmax），路由权重缩放因子 2.5（这个值很激进）。

**一个数字**：744B 总参，但每 token 只激活约 40B（5.4%）——这是 MoE 的核心价值。"

**面试要点**：3 分钟说清楚「744B 怎么来的，40B 怎么算的，DSA 为什么是核心创新」就够了。不要读超参表。

**延伸阅读**：主报告 CH 0 / CH 2

---

### Q10.2 DSA 为什么比滑动窗口（SWA）好？

**简短回答**：滑动窗口只关注固定范围内的 token（如最近 4096 个），无法根据内容动态选择——长距离的关键信息被强制忽略。DSA 的 Indexer 根据 query 内容动态打分，可以从整个上下文中选 top-2048——即使那个 token 在 100K 之外。

**详细解释**：

**面试官视角**：这是一个考察「你理解稀疏注意力 vs 全注意力 trade-off 本质」的问题。关键不是背数据，是理解「动态选择」为什么重要。

**答题模板**：

"想象你在看一篇 192K token 的代码库。滑动窗口只看最近的 4K token，如果代码开头定义了关键函数 `authenticate_user`，滑动窗口根本看不到它。

DSA 的 Indexer 会做什么？它对每个历史 token 打分——`authenticate_user` 和你的 query（可能包含 'auth' 语义）会产生高相关性分数（通过 32 个 Indexer head 的多角度匹配），然后被选入 top-2048，参与精确注意力计算。

消融实验证明了这一点：SWA Interleave 在 RULER@128K 上仅 44.93，比 Full Attention 低 30.35 分。DSA 是 78.86，仅低 0.35 分。

本质区别：SWA 是'盲目的空间约束'，DSA 是'智能的内容筛选'。"

**面试要点**：用具体例子（如代码检索场景）来说明，比背数字更有说服力。

**延伸阅读**：主报告 CH 3.5（消融表）

---

### Q10.3 为什么 routed_scaling_factor 要设 2.5 这么激进？

**简短回答**：因为 sigmoid 评分在 (0,1)，top-8 归一化后每个 expert 的权重约 0.125，Expert FFN 的输出被严重衰减。乘以 2.5 将有效权重恢复到约 0.31，使 MoE 层输出幅度与 Dense FFN 对齐，保持 78 层残差流的数值一致。

**详细解释**：

**面试官视角**：考察你能否从数值分析角度解释一个看似奇怪的超参。关注点：(1) 理解衰减的来源，(2) 知道为什么需要恢复，(3) 对比其他模型的选择。

**答题模板**：

"这个数字要放在完整的计算链里理解：

1. sigmoid → 分数在 (0,1)，不是一个概率分布
2. top-8 → 8 个最高分的 expert 被选中
3. 归一化 → 8 个分数除以它们的和。假设均为 0.5，归一化后每个权重 = 0.125
4. Expert FFN 输出乘以 0.125 → 有效幅值仅 12.5% of Dense FFN

在 78 层残差网络中，如果每层 MoE 输出只有 Dense FFN 的 12.5%，信号会逐层衰减。2.5 的缩放将有效权重恢复到约 0.31——不是完全对齐 Dense，但足够了。

为什么比 V4-Flash 的 1.5 激进？因为 V4 的 k=6（6 个 expert 分摊权重，每个约 0.17），GLM 的 k=8（每个约 0.125），起步就低了 26%。2.5 是对 k=8 的补偿。

M2.7 不缩放（1.0），说明 MiniMax 可能通过不同的路由权重计算方式避免了这个问题。"

**面试要点**：2.5 不是孤立的参数——它与 k=8、sigmoid 评分函数、残差网络深度三者耦合。展示这种系统性思考能力。

**延伸阅读**：主报告 CH 5.2

---

### Q10.4 MLA 的 KV cache 为什么能比标准 MHA 小那么多？

**简短回答**：标准 MHA 需要缓存 64 头 × 256 维 × 2（K+V）= 32768 个值/token。MLA 将 KV 压缩到 576 维（512 压缩 KV + 64 RoPE K），缓存量减少 57×。使用时通过轻量投影展开，不影响注意力质量。

**详细解释**：

**面试官视角**：考察你对 MLA 工作流的理解——「压缩了什么，怎么展开的，代价是什么」。

**答题模板**：

"MLA 的关键洞察是：K 和 V 的 64 个头之间高度冗余——它们都是同一个 hidden state 的不同投影。既然如此，应该可以在一个低维潜空间中存储共享信息，用的时候再展开。

具体做法：
1. `kv_a_proj_with_mqa`：6144 → 576 维（512 压缩 KV + 64 RoPE K）
2. 缓存这 576 维（~19 GB @192K）
3. 使用时 `kv_b_proj`：512 → 64 × 448 维（展开为 64 头 × (192 K + 256 V)）
4. k_pe（64 维）广播到 64 头

本质是'用两个轻量投影的重量代价，换 57× 的缓存节省'：
- kv_a_proj：6K→576（3.54M 参数）
- kv_b_proj：512→64×448（14.68M 参数）
- 额外计算：每 token 约 36.5M FLOPs（kv_a + kv_b）

在 batch 推理中优势更大——MLA 压缩缓存所有请求共享（~19 GB 不变），标准 MHA 随 batch 线性增长。"

**面试要点**：说清楚「576 维分成两块（512 压缩 KV + 64 RoPE K）」和「为什么 RoPE 部分不压缩」。

**延伸阅读**：主报告 CH 4.1 / CH 2.5

---

### Q10.5 前 3 层用 Dense FFN 而不是 MoE 的理由是什么？

**简短回答**：浅层的 token 表征尚未分化为可被路由的专用特征——过早 MoE 会导致高熵的随机路由，浪费计算且降低 expert 训练效率。3 层 Dense 保证早期语义编码的完整性，为后续 75 层 MoE 提供稳定的特征基础。

**详细解释**：

**面试官视角**：考察你是否理解 MoE 的前提条件——MoE 要有效，Router 必须能做有意义的 expert 选择。如果输入特征太「原始」，Router 无法区分，expert 也无法专业化。

**答题模板**：

"这是一个'渐进式 MoE 化'的设计。核心逻辑：

第 1-3 层的 hidden state 还在做基础的语义编码——词义消歧、句法解析、浅层关系建模。这些特征是所有 token 共享的，不需要不同的 expert 来处理。如果此时用 MoE：
- Router 看到的是未分化的语义混合，打分几乎是随机的
- 每个 expert 收到的是随机分配的 token，无法专业化
- 训练信号被稀释，expert 退化为「k 个较小的 Dense FFN」

到第 4 层之后，经过 3 层 Dense FFN 的处理，hidden state 已经分化为可路由的专用特征——代码 token 有了代码相关的表示，数学 token 有了数学相关表示。此时 Router 才能有效地将 token 分配到对应的 expert。

这个设计在 GLM-4.5 和 GLM-5 两代中验证有效——3 层是经验最优值。成本极低（占总参数 0.1%），但收益显著（路由稳定性 + 训练效率）。

类比：公共基础课后分流——先让所有学生打好基础，再根据兴趣选专业课。"

**面试要点**：这个答案的关键是「特征分化」这个概念——MoE 路由依赖语义可分性，不是所有层都适合 MoE。

**延伸阅读**：主报告 CH 2.3（约束 3）/ CH 5.4

---

### Q10.6 如何评估 DSA 是否「无损」？给出你的方法论。

**简短回答**：用 (1) RULER 类长上下文检索评测在不同长度上对比 Full Attention，(2) 逐层分析 Indexer 的 recall@2048，(3) 消融 top-k 大小对下游任务的影响，(4) 多样性测试（SWE-bench, RepoQA, MRCR 等不同检索场景）。DSA 当前最有力的证据是 RULER@128K 上仅比 Full Attention 低 0.35，且在 64K 上反超。

**详细解释**：

**面试官视角**：这是一个开放性的方法论问题——考察你设计评测实验的能力，而不仅仅是复述论文结论。

**答题模板**：

"我会从四个维度评估：

**1. 检索评测**（RULER 系列）
RULER 是专门测试长上下文检索能力的 benchmark——在 128K token 中检索特定信息。DSA 在 RULER@128K 上 78.86 vs Full Attention 75.28——DSA 甚至更高，说明稀疏注意力可能过滤了噪声。

**2. Indexer 的 recall 分析**
这是论文没有做的。我会在 Full Attention 模型的 attention weights 上取 top-2048 个最关注的 token，计算 Indexer 选出的 top-2048 与「真实 top-2048」的 recall。如果 recall > 95%，说明 Indexer 选择质量很高。如果 < 80%，说明有系统性的遗漏风险。

**3. Top-k 消融**
在 9B 模型上分别设 k=512, 1024, 2048, 4096，观察下游任务的分数曲线。如果从 1024 到 2048 有显著提升，但 2048 到 4096 持平，说明 2048 是饱和点。这也验证了选择不是盲目的。

**4. 多样性场景测试**
不是所有检索任务都一样——代码检索（函数名只出现 2-3 次）vs 文档总结（关键信息密度高）。应该在 SWE-bench（代码检索）、RepoQA（仓库级别问答）、MRCR（多轮检索）等不同场景下分别评测。

总结：DSA 在全注意力模型的「真实注意力分布」上的 recall 是核心指标——如果 Indexer 能覆盖 95%+ 的高注意力 token，基本可以认为是「无损」的。"

**面试要点**：提出论文没有做的实验（Indexer recall 分析）说明你有独立思考能力，而不是只会复述 paper。

**延伸阅读**：主报告 CH 3.5 / CH 4.16

---

### Q10.7 如果让你改进 DSA，你会怎么做？

**简短回答**：(1) 用 attention 的反向信号在线训练 Indexer（目前 Indexer 在 RL 中被冻结），(2) 引入可学习的 top-k（而不是硬编码 2048），(3) 不同层使用不同的 top-k（浅层更大，深层更小），(4) 对 Indexer 遗漏的 token 做「补偿性恢复」。

**详细解释**：

**面试官视角**：考察你的创新能力——不是让你复述现有架构，是让你设想「如果是我来做，下一步往哪个方向走」。

**答题模板**：

"我有四个改进方向，按优先级排序：

**1. Attention-guided Indexer 训练**
当前 Indexer 在 RL 中被冻结——防止不稳定学习。但更好的做法是让 attention 的反向信号「教」Indexer 哪些 token 应该被选中。具体做法：attention 计算后，对未选中的 token 如果注意力本应很高，产生一个「遗漏损失」，反传给 Indexer 的 weights_proj 和 wq_b。类似知识蒸馏中学生向老师学习。

**2. 动态 top-k**
2048 是固定值——但在短上下文（如 4K）下，2048 意味着 50% 的 token 参与注意力，稀疏优势消失。应根据序列长度动态调整：top-k = min(2048, T × 0.2)——保证「至少选前 20% 或最多 2048 个」。或者更激进：根据 Indexer 分数的分布动态确定——如果分数集中在前 500 个 token，选 500 就够了。

**3. 层异构 top-k**
浅层（第 4-10 层）的注意力模式较分散（关注更多 token 做语义聚合），深层（第 50-78 层）更集中（关注特定 token）。浅层用 top-4096，深层用 top-1024——在总计算量不变的情况下提高浅层的覆盖率。

**4. Fallback token 机制**
对 Indexer 可能遗漏的「稀疏但关键」的 token（如代码中的函数定义），引入一个轻量的 fallback 机制——例如基于 BM25 的快速关键词检索（CPU 侧并行，不阻塞 GPU），把匹配到的 token 强制加入 attention（即使 Indexer 没选）。类似搜索引擎的'付费推广位'——无论相关性分数如何，这些 token 一定被关注。"

**面试要点**：改进方案要有「为什么这样改」的 reasoning，不只是列 idea。第 4 点（fallback mechanism）是 hybrid 思路——经典信息检索 + 神经网络，很有说服力。

**延伸阅读**：主报告 CH 3 / CH 10.3

---

### Q10.8 GLM-5.1 vs M2.7，你会选哪个架构？给出理由。

**简短回答**：取决于部署场景。推理成本敏感（消费级/国产芯片）选 GLM-5.1（DSA+MLA 压缩使长上下文经济可行）。质量敏感（云端 API）且不差钱选 M2.7（Full Attention 零风险）。如果服务端部署且需要高 batch 推理，GLM-5.1 的 MLA 压缩缓存优势不可替代。

**详细解释**：

**面试官视角**：没有一个放之四海而皆准的答案——面试官想看你能不能根据约束条件做决策，而不是给你偏好的模型站台。

**答题模板**：

"我的选择取决于三个约束：

**场景 A：边缘/国产芯片部署** → 选 GLM-5.1
- DSA 节省 72.5% 注意力计算 + MLA 压缩缓存 ~19 GB
- 在单台 Atlas 800T A3 上可运行（FP8/W4A8）
- M2.7 的 Full Attention 在 192K 下计算不可承受

**场景 B：云端高 batch 推理** → 选 GLM-5.1
- MLA 压缩缓存在 batch 中不变（~19 GB）
- M2.7 的 GQA 缓存随 batch 线性增长（32 × 48.8 GB = 1.56 TB）
- 服务端部署中 batch size 是吞吐量的关键

**场景 C：追求极致质量的研究环境** → 选 M2.7
- Full Attention 零遗漏——不需要担心 Indexer 的 recall
- 更简单的 architecture 意味着更可预测的行为
- 研究环境中单个 token 成本不是首要约束

如果必须选一个'默认推荐'——在当前（2026 年）的工程条件下，我选 GLM-5.1。因为 (1) 它的无损特性已经被充分验证（RULER 仅差 0.35），(2) 长上下文推理的经济性是实际部署的主要瓶颈，(3) 国产芯片适配提供了额外的部署灵活性。"

**面试要点**：不要只选一个站队——分场景讨论，展示你的 engineering judgment。结尾给出「如果必须选一个」的答案增加说服力。

**延伸阅读**：主报告 CH 9.5 / CH 10.1

---

### Q10.9 DSA Indexer 的 top-2048 能否在推理时动态调整？

**简短回答**：技术上可以——Indexer 已经有完整的分数矩阵，top-k 只是 score.topk(k) 的参数。但需要评估动态调整对注意力性能和下游任务的影响。减少 k 可以加速推理但可能遗漏关键 token；增大 k 提高覆盖率但 Indexer+注意力计算成本上升。

**详细解释**：

**面试官视角**：这是一个技术可行性和工程影响力的平衡问题。面试官想看你能不能权衡实现成本 vs 收益。

**答题模板**：

"从代码角度看，`self.index_topk` 是一个 Python 整数属性，`index_scores.topk(topk, dim=-1)` 可以在 forward 时传入任意 k 值——技术上完全可行。

但要考虑几个因素：

**何时减少 k**：
- 短上下文（< 16K）：k=2048 覆盖率太高（可能 > 12% token），浪费注意力计算
- 简单任务（如聊天）：不需要关注 2048 个 token，512 可能就够了
- 方案：根据任务难度和序列长度动态调整，k = min(2048, max(512, T × 0.05))

**何时增加 k**：
- 代码检索任务：关键 token 稀疏分布，2048 可能不够
- 超长上下文（> 128K）：绝对覆盖率下降（2048/128K=1.6%），可能需要 4096
- 风险：增加 k 会线性增加稀疏注意力的计算量和 KV cache 访问量

**工程实现**：
最简单的方案是在 generate() 时通过 kwargs 传入 index_topk_override 参数，在 Indexer forward 中覆盖 self.index_topk。不需要修改模型权重。

权衡：动态 k 的收益（更精准的覆盖率控制）是否大于额外复杂度（任务检测、k 值决策逻辑）？我倾向于保守——先用固定 2048，积累足够的任务-specific 的召回率数据后再考虑动态化。"

**面试要点**：技术上可行但不推荐贸然改动——这说明你有工程判断力，不会为了「看起来聪明」而过度工程化。

**延伸阅读**：主报告 CH 3.6（Step 7）

---

### Q10.10 为什么 shared expert 只需要 1 个而不是多个？

**简短回答**：1 个共享专家足够提供通用语义处理的基础能力。多个共享专家会：(1) 增加激活参数（每个 37.75M），(2) 与路由专家产生冗余竞争——共享专家的本质是「兜底」而非「替代」路由专家，(3) 多个共享专家之间的负载分配又变成了一个子路由问题。

**详细解释**：

**面试官视角**：考察你是否理解「共享专家」在 MoE 架构中的角色定位——它是安全网，不是另一组路由专家。

**答题模板**：

"共享专家在 MoE 中的角色是'通用底线'而不是'额外能力'。类比理解：

- 路由专家 = 专科医生（神经科、心脏科、骨科...）
- 共享专家 = 全科医生（处理所有人都会遇到的基础问题）

1 个共享专家的设计逻辑：

1. **功能定位**：共享专家的职责是通用语义处理（基础语法、常识、安全对齐）——这些对所有 token 是统一的，1 个 expert 就够了。

2. **避免冗余**：如果设 2 个共享专家，它们会学到相似的功能（因为输入是相同的）——造成参数浪费。路由专家因为有 Router 的差异化分配，不会出现这个问题。

3. **参数效率**：共享专家 37.75M/层 × 75 = 2.83B，只占总参的 0.4%。增加到 2 个就是 5.66B（0.8%）——对 744B 模型来说不算多，但边际收益可疑。

4. **退化风险**：多个共享专家需要分配负载——本质上是'共享专家层内部的子路由问题'，失去了共享专家的简洁性。

如果未来发现 1 个共享专家在特定任务上有瓶颈（如安全对齐和代码生成需要不同的通用基础），可以将共享专家按能力域分为 2-3 个——但这需要实验验证而非先验假设。"

**面试要点**：回答的关键是区分「共享专家 ≠ 路由专家」——前者是统一的基线，后者是差异化的选择。混淆两者是常见的概念错误。

**延伸阅读**：主报告 CH 6.6 / CH 9.3

---

### Q10.11 异步 RL 中为什么要冻结 Indexer？

**简短回答**：(1) Indexer 只有约 9.6M 参数/层（占每层 Attention 174.4M 的 5.5%），参数少对噪声极其敏感；(2) RL 的延迟奖励信号噪声远大于 pre-training 的 token-level loss；(3) 冻结 Indexer 防止了「catastrophic forgetting」——Indexer 在 pre-training 中学会的选择策略在 RL 中被噪声破坏。

**详细解释**：

**面试官视角**：这是一个具体的工程决策问题——考察你理解 RL 训练的难点（高噪声、稀疏信号、小模块脆弱性）。

**答题模板**：

"论文 §3.2 报告了这个经验教训。三个层面的原因：

**1. 参数脆弱性**
Indexer 的核心组件：wq_b（8.4M）+ wk（0.79M）+ weights_proj（0.2M）+ k_norm（极小）≈ 9.6M 参数。这些参数决定了一个至关重要的决策——哪些 2048 个 token 参与注意力。即使微小的梯度噪声也可能改变 top-2048 的选择，进而影响整个 attention 输出。

**2. RL 信号特征**
与 pre-training 的「每 token 都有明确 ground truth」不同，RL 的奖励信号是稀疏的（一个完整 rollout 只有末尾一个 reward）且有高噪声（环境随机 + exploration noise）。这种信号用于更新 9.6M 参数的 Indexer，很容易导致过拟合到当前的 reward pattern。

**3. 灾难性遗忘**
Indexer 在 pre-training（Warmup + Sparse Adaptation）中通过 20B+ tokens 学会了在 192K 上下文中有效选择 top-2048 token。这是 Indexer 最宝贵的知识。RL 中的奖励信号（如「完成 SWE 任务」）可能是局部的、任务特定的，用它来更新 Indexer 可能导致：(a) Indexer 过度关注与当前奖励相关的 pattern，(b) 丢失通用检索能力。

冻结 Indexer 的代价是 RL 期间无法进一步优化选择策略——但权衡下来，稳定性的收益远大于微调 Indexer 的潜在获益。"

**面试要点**：指出 Indexer 参数量（9.6M）和 Attention 参数量（174.4M）的对比——小模块承担大决策，冻结是明智的保守策略。

**延伸阅读**：主报告 CH 3.4 / GLM-5 paper §3.2

---

### Q10.12 如果 GLM-5.1 的 sigmoid 路由换成 softmax，会有什么影响？

**简短回答**：softmax 会迫使 256 个 expert 形成零和竞争——高分 expert 压制低分 expert 的分数。优点是路由选择更「干脆」（分布熵更低，top-8 的区分度更高），缺点是 expert 间失去了独立性——一个 token 如果同时需要「代码+数学」能力，softmax 可能只选代码 expert 或只选数学 expert，而不是两者都选。

**详细解释**：

**面试官视角**：这是一个假设性的架构设计问题——考察你对评分函数在 MoE 路由中作用的理解深度。

**答题模板**：

"换成 softmax 后的预期影响：

**正面**：
- 路由选择更可靠——softmax 的指数运算天然拉开分数差距，top-8 的选择更「确定」
- 负载均衡更自然——softmax 的概率分布特性使 expert 被选中的概率更均匀
- `e_score_correction_bias` 的调节压力减小——softmax 本身就有较强的负载均衡倾向

**负面**：
- 失去 expert 独立性——softmax 的零和性意味着「一个 expert 分数高 = 其他 expert 分数低」。对于需要复合能力的 token（如「请写一个用数学公式的代码」），softmax 可能压制其中一个能力域的 expert
- `routed_scaling_factor` 可能需要调整——softmax 归一化后的权重分布不同于 sigmoid 归一化，2.5 可能不再是最优值
- 与现有负载均衡机制（e_score_correction_bias + noaux_tc）的兼容性未知——这套机制是为 sigmoid 设计的

**我的判断**：对于 GLM-5.1 的 Agent 场景（token 经常需要复合能力），sigmoid 的独立性更适合。如果未来模型更侧重单一能力评测（如纯数学、纯代码），softmax 可能更优。

这个替换还需要：(1) 重新 sweep routing_scaling_factor，(2) 调整负载均衡机制（可能改回 auxiliary loss），(3) 检查 MoE 层是否有路由坍塌（少数 expert 被所有 token 选中）。"

**面试要点**：不要说 sigmoid 或 softmax 绝对更好——要分析适用场景，展示你理解每个选择的 trade-off。

**延伸阅读**：主报告 CH 5.3 / CH 6.3

---

### Q10.13 MTP 参数共享的优劣是什么？

**简短回答**：优点是训练内存节省 3×（MTP 参数量不翻倍）、推理接受率不下降（2.76 vs 2.55）。劣势是训练-推理的不一致可能引入 gap（训练 3 共享层 vs 推理 1 层），需要额外的设计来缓解（可能通过训练阶段的 depth annealing）。

**详细解释**：

**面试官视角**：考察你是否理解 MTP 的「参数共享」和「训练-推理不一致」这对矛盾，以及论文如何解决。

**答题模板**：

"MTP 参数共享是一个精巧但有一定风险的工程优化。

**优势**：
1. 参数量不变——3 层共享 1 套参数，训练时额外参数量仅为 1× MTP 层（而非 3×）
2. 推理接受率高——2.76 vs DeepSeek-V3.2 的 2.55，说明共享参数质量不差
3. 深度泛化——训练时用 3 个 depth 的输入，模型学会了在单层内处理不同 depth 的预测，类似 test-time scaling

**劣势**：
1. 训练-推理不一致——训练用 3 层（共享参数）但推理用 1 层。虽然参数相同，但 3 层 forward 的输出分布和 1 层 forward 不同（中间有 3 次 RMSNorm 和残差连接）
2. 第 2 个 token 的预测质量——训练时由深度 2 的 MTP 层预测，推理时由深度 1 的同一层做两次 forward 预测。虽然是同一套参数，但状态空间不同
3. 消融数据不完整——论文未公开 Parameter Sharing vs Independent Parameters vs No Sharing 的详细对比

**如何评估**：如果我做审稿人，我会希望看到 (1) MTP sharing 的消融实验（共享 vs 不共享 vs 部分共享），(2) 训练和推理在中间层的 hidden state 分布对比，(3) 不同长度序列上接受率的 breakdown。论文没提供这些，是 MTP 部分的不足。"

**面试要点**：不要只说「参数共享好」——指出 paper 的 incomplete ablation 说明你有批判性思维。

**延伸阅读**：主报告 CH 7.4 / GLM-5 paper §2.1

---

### Q10.14 192K 上下文在实际部署中如何高效运行？

**简短回答**：(1) 使用 MLA 压缩 KV cache（~19 GB 而非 936 GB），(2) 使用 FP8/W4A8 量化模型权重，(3) 使用 PD disaggregation 分离 prefill 和 decode，(4) 使用 MTP speculative decoding 加速，(5) 上下文管理策略（Keep-recent-k + Discard-all）在应用层减少实际上下文。

**详细解释**：

**面试官视角**：考察你的「工程部署思维」——不是「能不能跑」，而是「怎么高效地跑」。

**答题模板**：

"从系统层面，五个层次逐级优化：

**Layer 1: 模型精度**
- FP8 量化权重：744 GB → 744 GB（权重）或 W4A8（MoE expert INT4 + 其他 INT8）
- 保护关键模块：Indexer.weights_proj 和 e_score_correction_bias 保持 FP32

**Layer 2: KV Cache 压缩**
- 当前最大的瓶颈——936 GB expanded cache
- 实现 MLA 压缩 cache（~19 GB）是第一优先级
- 需要专门的 MLA Cache 类替代 DynamicCache

**Layer 3: 计算分离**
- PD Disaggregation：prefill 和 decode 用不同的 GPU（prefill 计算密集，decode 内存密集）
- 独立扩展 prefill 和 decode 的资源

**Layer 4: 推理加速**
- MTP speculative decoding：1 层 MTP 预测 2 个 token，接受率 2.76
- Flash-MLA kernel：社区 kernel 完成适配后注意力计算加速 2-4×

**Layer 5: 应用层优化**
- 层次化上下文管理：Keep-recent-k + Discard-all
- 实际用户使用的平均上下文长度远小于 192K
- 在 128K 实际上下文下，DSA 的 Indexer 开销可接受

最终配置建议：
FP8 + 压缩 KV cache + PD + MTP ≈ 8×H200 可以支撑流畅的 192K 上下文推理。"

**面试要点**：5 层递进展示了系统性的工程思维——从模型精度到应用策略，每层解决不同的问题。

**延伸阅读**：主报告 CH 2.7 / CH 7.5

---

### Q10.15 GLM-5.1 最核心的创新是什么？为什么？

**简短回答**：DSA（动态稀疏注意力）是 GLM-5.1 最核心的架构创新。理由：(1) 论文 35 次提及 DSA，是全文最密集讨论的技术点；(2) 它是将注意力从 $O(T^2)$ 降到 $O(T \cdot k)$ 且保持近乎无损的第一个有效方案；(3) DSA 使 78 层 × 192K 上下文 + 744B 参数的部署在工程上可行——没有 DSA 就没有 GLM-5.1 的实用价值。

**详细解释**：

**面试官视角**：考察你能否识别 paper 中最重要、最具原创性的贡献。不是遍历所有 feature，而是识别「哪个是如果没有就不行的」。

**答题模板**：

"我认为是 DSA，三个层面的论证：

**1. 原创性**：DSA 是 GLM-5 论文中唯一有完整消融实验 + 多种 baseline 对比 + 多个 scale 验证的技术创新。MLA 和 MoE 都建立在已有工作之上（MLA 来自 DeepSeek-V2，MoE 来自 GLM-4.5 的延续），DSA 的「从密集 checkpoint 出发的两阶段适应」训练策略是全新的。

**2. 必要性**：没有 DSA，GLM-5.1 的 78 层 × 192K 上下文推理无法实现——Full Attention 的单层 QK^T 为 6.44 GFLOPs，78 层仅注意力就需要 502 GFLOPs/token。DSA 将注意力降低 72.5% 到 1.77 GFLOPs/层，使部署可行。没有 MLA 只是 KV cache 更大（可以买更多 GPU），但没有 DSA 根本无法运行。

**3. 范式意义**：DSA 证明了稀疏注意力不必是'次优方案'——在合适的 selection mechanism 下，稀疏注意力可以与全注意力等价。这改变了人们对「注意力必须全局」的固有认知，可能成为未来长上下文 LLM 的默认注意力模块。

相比之下，MLA 节省了 KV cache（重要但不原创），MoE 是 GLM-4.5 的延续（重要但非新贡献），异步 RL 是工程创新（重要但在架构文档中居于辅助位置）。DSA 是架构层面唯一的原创性突破。"

**面试要点**：论文 35 次提及 DSA——这个数字很有说服力。说明作者自己也认为 DSA 是最重要的贡献。

**延伸阅读**：主报告 CH 10.1（三个核心 Insight 第 1 点）

---

### Q10.16 面对 DSA 可能遗漏关键 token 的风险，你有什么缓解方案？

**简短回答**：(1) 多粒度选择——同时保留 sliding window 作为 fallback（最近 4K token 始终参与注意力），(2) 基于语法结构的关键 token 预标记——对代码中的函数定义/类声明预标记为「must-attend」，(3) Indexer 置信度估计——当 2048 个被选中 token 的最高分都很低时，扩大 top-k，(4) 双层 attention——稀疏 attention 后做一次轻量级的全局「摘要 attention」。

**详细解释**：

**面试官视角**：这是一个面向「已知局限」的开放性改进题——考察你能否从风险识别到方案设计到实施评估给出完整的思路链。

**答题模板**：

"DSA 的遗漏风险是一个真实存在的问题——特别是在代码检索场景。我有四个互补的缓解方案：

**方案 1：Hybrid Sliding Window（最低成本）**
Indexer 选 top-2048，但同时保证最近 512 个 token 始终在注意力中（强制加入，覆盖 local context 和 recency）。这相当于 'sparse global + dense local' 的 hybrid 方案。额外计算量极小（512 个 token 的 QK^T 几乎免费），但覆盖了最常见的遗漏 type（关键信息就在附近但 Indexer 没选）。

**方案 2：Structure-aware Pre-marking（中等成本）**
在 prefill 阶段，用轻量规则（如 AST parser for code, keyword detector for text）标记「结构关键」的 token——函数定义、类声明、章节标题等。这些 token 直接进入 top-2048（不经过 Indexer 打分竞争）。类似搜索引擎的'权威页面'白名单。prefill 阶段的 AST 解析开销可接受（非推理瓶颈）。

**方案 3：Confidence-gated Dynamic k（低实现成本）**
Indexer 打分后，检查被选中的 2048 个 token 的最高分。如果 max(index_score) < 某个阈值（说明 Indexer 对当前上下文的置信度很低），自动扩展 k 到 4096。类似于模型在「不自信」时多看一些 token。

**方案 4：Lightweight Summary Attention（高成本但全面）**
在稀疏 attention 之后，对全局的 hidden state 做一次轻量级 attention（如 4× 压缩后的线性 attention），生成一个「全局摘要向量」，与稀疏 attention 的输出拼接。计算成本增加约 10-15%，但提供了全局视野作为 fallback。

优先级：方案 1（最低成本）+ 方案 3（低实现成本）作为第一梯队，方案 2 作为代码场景的专项优化，方案 4 作为长期方向的探索。"

**面试要点**：四个方案从低到高按成本排序——说明你的方案有优先级考量，不只是 brainstorm。

**延伸阅读**：主报告 CH 10.2（已知局限）

---

### Q10.17 如果要为 GLM-5.1 设计一个改进版本，你的三个改进方向是什么？

**简短回答**：(1) Attention-guided Indexer 在线微调——让 attention 的反向信号在线训练 Indexer，(2) 混合精度专家量化 + dynamic expert pruning——推理时跳过低贡献的 expert，(3) 层次化 DSA——不同层用不同的 top-k 和 Selection 策略。

**详细解释**：

**面试官视角**：最终的综合性开放题——考察你的架构设计视野、工程判断力和对论文局限的洞察。

**答题模板**：

"我的三个改进方向，分别针对三个不同的优化目标：

**改进 1：Attention-guided Indexer（精度提升）**
当前 Indexer 的训练是离线的（pre-training 中的 Warmup + Sparse Adaptation），之后在 RL 中被冻结。我的方案：在推理服务中收集「Indexer 选择的 top-2048」vs「attention 如果全量计算的真实 top-2048」，用这些有监督信号定期微调 Indexer 的 weights_proj 和 wq_b。类似于 continual learning with distillation——让 Indexer 在实际使用中持续改进。

预期收益：Indexer 的 recall 从 ~95% 提升到 ~98%+。

**改进 2：Dynamic Expert Pruning（效率提升）**
当前 8 个被选中的 expert 全部参与计算，但归一化权重 < 0.05 的 expert 贡献几乎为零。我的方案：在 Router 输出后设一个权重阈值（如 0.05），低于阈值的 expert 不执行 forward。动态 k 从固定的 8 变为自适应的 4-8。对于简单 token（如标点、常用词），可能只需要 3-4 个 expert。

预期收益：MoE 层计算减少 20-30%，激活参数从 40B 降至 ~30-35B。

**改进 3：层次化 DSA 策略（架构优化）**
当前的 DSA 是「一层一策」（所有 78 层用相同的 top-2048）。我的方案：
- 浅层（1-20 层）：top-4096，关注更多 token 做广泛的语义聚合
- 中层（21-55 层）：top-2048，当前最佳设置
- 深层（56-78 层）：top-1024，注意力模式更集中
- 总体计算量：约为当前方案的 85%（浅层增加但深层减少）

此外，在 Indexer 之前加一个 fast hash-based filter（类似 LSH attention），先排除明显不相关的 token（如 hash code 相差大的），减少 Indexer 扫描的候选集大小。

这三个改进分别对应精度、效率、架构三个维度——都是基于 GLM-5.1 当前架构的自然延伸而非推翻重来。"

**面试要点**：三个改进分别对应精度、效率、架构——展示了系统性思考。每个改进都有明确的预期收益和实现复杂度。

**延伸阅读**：主报告 CH 10.3
