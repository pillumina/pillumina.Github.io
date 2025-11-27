
---
title: "[Deterministic RL] deterministic问题的来源和相关工作总结"
date: 2025-11-20T11:30:12+08:00
tags: ["deterministic","RL"]
series: ["aiinfra"]
---


## 理解LLM推理中deterministic问题来源

Wiki上对deterministic算法的定义是:
>“a deterministic algorithm is an algorithm that, given a particular input, will always produce the same output.”

而我们在文中要讨论的，即对于LLM这个context下的deterministic问题，我会先从inference角度（即重复给定一个确定的input，模型的推理为什么无法给定确定的输出）进行问题的理解，再进一步讨论RL工程中的training & inference之间差异，可能会导致RL训练的崩溃问题，并继续讨论业界现在已有的解决方案、与还在`working-in-progress`的工作。

### 浮点数的非结合性
[thinking machines lab针对batch invariant讨论的文章](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，详细地解释了在LLM推理中不确定性的来原，即因为精度有限，GPU浮点数运算中的结合性通常不成立：$$(a+b)+c \neq a+(b+c) $$
[这篇arxiv文章](https://arxiv.org/abs/2506.09501)，则更深入得说明了这个问题：
>Floating-point arithmetic in GPUs exhibits non-associativity, meaning (a+b)+c≠a+(b+c)(a+b)+c=a+(b+c) due to finite precision and rounding errors. This property directly impacts the computation of attention scores and logits in the transformer architecture, where parallel operations across multiple threads can yield different results based on execution order.

浮点数通常可用科学计数的表示来表征大/小数，例如格式$mantissa *10^{exponent}$，如果指数项是不同的，也就是文中说的`add at different scales`，那不同累加序导致的精度损失会更加明显，而这种不同scale的累加是最常见的场景。

但是尽管这是不一致输出的根本原因，但是并没有回答不确定性源自何处。无法帮助我们去理解：浮点数值为何会以不同的顺序相加、这种情况何时会发生，已经如何避免这种情况。


### 为何计算内核不同序add numbers？
一个常见的假说是“**并发执行随机性 + 浮点运算误差**”。这个假说的核心观点，就是如果并发线程的结束顺序是非确定的，并且数值累加顺序如果依赖于并发线程的结束顺序（例如使用atomic add操作），那么最终数值累加的顺序也是非确定的。

### 什么时候真正需要atomic add？

但是问题是，LLM前向的GPU内核实际上很少用atomic add操作。
>简单解释下Atomic Add的含义：PU 会把同一段程序同时扔到很多“小核”（SM）上去跑。这些小核之间天生没有步调一致的机制，谁快谁慢完全看当时心情。于是，如果它们需要把结果写到同一个地方，就会出问题。那atomic add就是，硬件保证所有人的结果最终都会加进去，但谁先谁后、按什么顺序加，完全不保证，因此每次跑出来的累加顺序都可能不一样。
>再举个例子，通过torch.sum()对100个数求和，GPU 可以让 100 个小核各读一个数，这一步完全并行。可最后总得把 100 个数合并成 1 个总和。若用原子加，就是让每个小核随便谁先到，就先把它的数塞到同一个累加器里。硬件只负责“不会丢数”，却不负责“按固定顺序加”。于是同样跑两遍，先加谁后加谁可能不同，结果也就可能出现那一点点浮点误差。


我们回想通常定义的不确定性的含义：同一段 kernel、同一批输入，跑两遍却得到两个略有差异的结果。这叫“run-to-run 非确定性”——哪怕 Python 脚本、依赖库、硬件都没变，第二次跑就是能给你不一样的数。虽然atomic add会导致这个问题，但是更通常的情况是，LLM一次典型的forward，通常一个atomic add也没有。

这主要有两个原因：
1. batch维度上实际上已经“人多势众”，根本不需要在reduction维度再去并行。
2. 大多数neural networking library也用了很多技巧来实现“既保障确定性又快”，例如“分段树形归约”（split/tree reduction），可以先把100个数拆成5组，每组20个数，5个小核并行计算一个局部和。剩下 5 个局部和要么交给一个核顺序“扫尾”（元素少，开销可忽略），要么用信号量（semaphore）按固定顺序让不同线程块依次累加，从而保证先后顺序一致，结果也就 deterministic 了。

不过，在`pytorch`中的`scatter_add`操作（`a[b] += c`），如果不用atomic add性能会特别慢，而LLM中唯一踩这个坑的，是`FlashAttention`的反向传播。

> 不过当前Triton版本FA反向实现，其实和Tri Dao原论文里的算法不完全一样，Triton版为了躲开atomic add，额外多算了一遍中间结果--FLOPS直接多了40%，但是换来了determinstic，也算明码标价。

但是正向传播里，LLM中根本就没有非得用atomic add的算子，所以结论就是：LLM 的前向推理，跑两次、跑一百次，结果**比特级完全一致**；真正可能“每次不一样”的，只出在反向训练阶段，而且基本就 FlashAttention 一家。（也就是前向是“run-to-run deterministic”的）。


### 系统级别批次不变性的缺失（batch invariant）

前向kernel函数的确定性，实际上不等于整个推理服务对外表现确定，也就是还存在额外的**系统级非确定性**。因为真正喂给前向的**张量内容**还可能被其他“外部输入”左右。

举个经典里batch norm的坑：早期 BatchNorm 把“整批统计量”（均值/方差）当常量参与计算。  
同一句话，单条跑 μ₁ σ₁，跟 32 条一起跑 μ₃₂ σ₃₂，算出来的隐藏值就不一样，于是最终 token 概率也不同。  
站在“单条请求”视角：它完全无法预知今晚会不会有 31 个请求来搭伙，所以即使自己 prompt 固定，输出依旧“看运气”——这就是上面所说的**系统级非确定性**。

LLM 推理里虽然早就把 BatchNorm 踢了出去，却**仍缺“批次不变性”（batch invariance）**：同一请求、同一模型权重，只要推理时动态批大小不同，可能会导致`tilesize`不同，导致reduce的计算结果不同。例如，vLLM在不同规模的batch下，把prompt送往不同的batch，而GPU的并行调度器SIMT会把矩阵送往不同的sm、warp，这样计算路径就每次都不一样。

所以针对推理引擎，比如要在kernel层面实现batch invariant才能解决serve层面不确定性的问题。


### 和并行策略相关的Reduction不确定性

TODO: 通信库的通信算子带来的规约不确定性。


## `Batch Invariant`的相关工作

### `batch invariant ops`

Thinking Machines Lab发布了`batch invariant`的[部分kernel算子实现](https://github.com/thinking-machines-lab/batch_invariant_ops/tree/main)。
而从原blog里，提出了三种难度递增的实现。

#### batch invariant的`RMSNorm`
直接让每个Batch元素的reduction顺序固定，不受batch大小影响。

- batch大时，把单个batch元素分配给单个核心，reduction运算在单核心内完成，batch增大时让核心依次处理多个元素，保持reduction策略不变。
- batch小时，若采用"split reduction"（多核心分担reduction以提升并行度）会破坏batch invariant，可以选择忽略小batch优化（小batch本身执行就快，性能损失可以接受），或者采用固定reduction策略（牺牲部分性能来保证batch invariant）。

![batch-inv-rmsnorm](https://pic1.zhimg.com/80/v2-ce80537a575835d21972fe5b063f5bb9_1440w.webp?source=1def8aca)


#### batch invariant的矩阵乘法

将输出张量拆分为2D tiles，每个tile分配给单个核心，reduction在单个核心内部完成。编译固定配置的内核以适配所有形状，虽然会损失20%性能（和cuBLAS相比），但在LLM推理中通常可以接受，因为模型维度（N）比较大，对split-k的需求较低。

![batch-inv-gemm](https://pic1.zhimg.com/80/v2-7a754f390567bf5c6d92ccf2a4267c0a_1440w.webp?source=1def8aca)

#### batch invariant的注意力计算

采用data-parallel策略（沿着Q张量并行，reduction在单核心内完成），更新KV缓存和页表以保证KV布局一致，不受处理token数量的影响。

decode阶段Q长度小，需要拆分KV维度（Split-KV），采用固定拆分大小策略（而非固定拆分数量），确保reduction顺序不变，比如把1000长度的KV拆成3个256长度和1个232长度的片段，而不是4个250长度的片段。

![batch-inv-attn](https://picx.zhimg.com/80/v2-9b088207a9c3ed23e018f1416897134e_1440w.webp?source=1def8aca)

### Sglang / vLLM 实现deterministic inference

SGLang团队的[博客](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)里记录了实现的细节，主要是针对`batch invariant`kernel上，针对chunked prefill、cuda graph等特性做了兼容，具体可以参考[RoadMap](https://github.com/sgl-project/sglang/issues/10278)。

vLLM参考[Enabling Batch Invariant文档](https://docs.vllm.ai/en/latest/features/batch_invariance/)，也可以参考RFC [#28326](https://github.com/vllm-project/vllm/issues/28326)，[#27433](https://github.com/vllm-project/vllm/issues/27433)。



## On-policy RL训练中的训推不一致问题

当讨论on-policy RL训练的时候，[有研究指出](https://fengyao.notion.site/off-policy-rl) train / inference engine之间的不一致也会隐形导致on-policy假设的RL实际变成off-policy。所以当我们追求"真正的" on-policy RL训练时，需要知道：如果不能从两个完全一致的推理请求中获取bitwise相等的结果，那么当然也无法保障训推之间的bitwise一致性。所以基于之前我们对确定性推理实现讨论，直觉上可以知道如果保证了确定性推理，那么通过修改训练这部分stack，也能够实现在bitwise上训推的一致性，从而实现真正的on-policy RL训练。

而业界对这个问题的解决思路上主要分为两种：
- 在训练引擎侧，基于推理引擎(vllm/sglang)确定性推理内核前向实现，进行反向传递的实现，通过对齐kernel的实现，做到训练和采样部分的bitwise一致性（i.e. 0 KL divergence）。

- 拥抱训推分布的不一致（考虑到训练bitwise实现在工程上的工作量，和不同模型适配的工作量），在算法上为off-policy做off-policy correction，进行训推KL散度的偏差抑制，在大多数场景也能实现RL训练的平滑和目标效果。

后续会分别着重分析这两种解决思路。

### 不一致问题分析

[这篇文章](https://fengyao.notion.site/off-policy-rl) 从实验的角度来对rollout-training不一致问题进行了分析，主要得出的结论是，**不同的并行策略**以及**更长的响应长度**会增大二者之间的mismatch，而选择不同的推理后端的影响比较小。

![mismatch-parallel](https://fengyao.notion.site/image/attachment%3A82d124b2-e301-497d-8e8d-5c8b08c12a72%3Avllm_megatron_parallelism.png?table=block&id=279721e3-f6c4-806e-9215-f3811bd6544e&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)


![mismatch-reponse-length](https://fengyao.notion.site/image/attachment%3Ac030a2b2-299e-4449-8438-d01a6145dc8b%3Amax_mean_20_4.png?table=block&id=279721e3-f6c4-80ce-9b09-ee4a9355a7b8&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)


![mismatch-sampler-backend-dapo-32b](https://fengyao.notion.site/image/attachment%3A1f820845-a43e-4ed6-8061-98af976d6b8f%3Asglang_vllm_dapo.png?table=block&id=279721e3-f6c4-80f0-ab62-c1cf298e8c71&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)


![mismatch-sampler-backend-polaris-7b](https://fengyao.notion.site/image/attachment%3A65d1350d-89b3-44f7-8305-bd6e0c2d8672%3Asglang_vllm_pol.png?table=block&id=279721e3-f6c4-8030-95d6-f5ad53d497b9&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)

>这些消融实验可以带来一些经验的归纳，也就是说明现象。但是笔者认为并不能让我们完全理解mismatch产生的原因。笔者认为，不一致性的主要来源，还是因为训练（FSDP、Megatron等）和推理（vLLM、SGLang等）是针对不同计算pattern进行了各自侧重的优化，不论是前向的kernel算子差异带来的数值精度误差累积，还是切分策略带来的通信算子规约顺序带来的精度误差累计，都是mismatch的原因一部分。

而像MoE模型的稀疏以及动态路由特性，会带来比Dense模型更大的mismatch，因为路由机制本身就是数值精度敏感的，一些微小的数值差异，会带来差异巨大的专家激活。除此之外，MoE模型本身的稀疏特性，和Dense模型相比一般规模会更大，而现代的推理引擎，通常针对MoE模型有独特的优化手段（计算、通信），也会放大训推引擎之间的不一致性。


而字节团队的[这篇文章](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)，对训推不一致问题进行了更加深入的理论、实验分析，针对不一致的现象，也提出了更genearal的叙述：
>To achieve the massive throughput required, modern inference engines (e.g., vLLM, SGLang, TensorRT-LLM) employ aggressive optimization strategies like speculative decoding, low-precision computation (INT8/FP8), and specialized, batch-variant CUDA kernels. While maintaining sampling fidelity, the primary objective of modern inference engines is to maximize throughput, often measured in tokens per second. Conversely, training frameworks (e.g., FSDP, DeepSpeed, Megatron-LM) must strike a different balance, prioritizing numerical stability and precision for gradient computation, often using higher-precision formats like FP32 for master weights and optimizer states. This divergence in optimization priorities and constraints creates an inevitable training-inference mismatch.

因此，我们可以回到一开始提到的业界解决on-policy RL训推不一致问题的两个思路，实际上是在性能和一致性上trade-off的取舍，如果希望对齐训推计算（例如之前讨论的batch invariant），势必会带来性能上的劣化。

从这篇文档，能得到很多有用的takeaways，比如实验中衡量不一致性的用的是下面的`vllm-kl`metric：$$\small{\mathbb{E}_{s\sim d_{\textcolor{red}{\pi^\text{vllm}_\theta}}}\left[\text{KL}\left(\textcolor{red}{\pi^\text{vllm}_\theta}\left(\cdot|s\right),\textcolor{blue}{\pi^\text{fsdp}_\theta}\left(\cdot|s\right)\right)\right] = \mathbb{E}_{s\sim d_{\textcolor{red}{\pi^\text{vllm}_\theta}},a\sim {\textcolor{red}{\pi^\text{vllm}_\theta}\left(\cdot|s\right)}} \left[\log\left(\frac{\textcolor{red}{\pi^\text{vllm}_\theta}(a|s)}{\textcolor{blue}{\pi^\text{fsdp}_\theta}(a|s)}\right)\right],}$$
这个metric的异常spike，通常能导致训练侧entropy和rewards的异常波动。![vllm-kl-trigger](https://yingru.notion.site/image/attachment%3A3e26ea60-291f-470c-b2ec-79e7a8815f7f%3Aimage.png?table=block&id=271211a5-58b7-8055-8f35-c4a618ee5fed&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)

而`vllm-kl`的spike同时会导致`fsdp-ppl`和gradient norm的爆炸性波动，这表示`FSDP` engine给推理引擎采样的得到的tokens设置特别小的概率，导致梯度爆炸，从而让RL训练崩溃。

以及mismatch不是均匀分布的，如果推理引擎得到的token概率越接近0，那在训练侧这个token的概率会更严重地被压小，让mismatch更大。

![uniform-mismatch](https://yingru.notion.site/image/attachment%3Aa94ce922-399e-4e5e-b5a0-2ea829327d92%3Aimage.png?table=block&id=271211a5-58b7-8084-87e0-d30121785272&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)

除此之外，还发现OOD tools的返回比如multi-turn TIR会带来更大的mismatch等等问题。

所以综上所述，在当前的RL框架中，训推引擎之间的不一致，是一个不可避免的问题，如果不一致问题非常严重，容易导致训练崩溃这样的严重后果（特别在长稳训练下）。

接下来笔者详细介绍一下，业界针对不一致问题的解决思路和方案。

### 硬对齐训推前反向不同kernel

#### TorchTitan + vLLM

[TorchTitan项目](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/deterministic_vllm_rl)探索了基于vllm的确定性RL的实现，基于vllm的确定性前向实现，补充了vllm operations的反向传播。其具体的实现为：
- 利用vLLM的`batch invariant`前向实现。
```python
# These operations are deterministic when batch_invariance is enabled
y = torch.matmul(a, b)  # Uses vLLM's deterministic matmul
output = flash_attn_varlen_func(q, k, v, num_splits=1)  # Deterministic FA
```
- 实现了自定义的反向函数进行梯度计算。
```python
class FlashAttnWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ...):
        # Use vLLM's forward implementation
        return flash_attn_varlen_func(q, k, v, num_splits=1, ...)

    @staticmethod
    def backward(ctx, grad_output):
        # Compute gradients deterministically
        # (re-compute attention weights and apply chain rule)
        return grad_q, grad_k, grad_v, ...
```
- 提供了torchtitan和vllm侧不同格式的权重转换能力。

#### Slime + SGLang

SGLang团队在Thinking Machines Lab发布的批次不变算子基础之上，通过定制一系列注意力算子和采样逻辑，也**实现了完全确定性推理**。该实现同时保持与**分块预填充 (chunked prefill)、CUDA Graph、Radix Cache 和非贪婪采样 (non-greedy sampling)** 等关键功能的兼容性。SGLang侧的主要增强工作为:
- 集成Thinking Machines Lab的批次不变(batch invariant)算子。
- 实现固定KV分割大小的批次不变注意力算子。支持多种后端，包括 FlashInfer、FlashAttention 3和[Triton](https://zhida.zhihu.com/search?content_id=263564186&content_type=Article&match_order=1&q=Triton&zhida_source=entity)。
- 与关键推理性能相关功能完全兼容，例如分块预填充、CUDA图、基数缓存等，当启用确定性推理时，所有这些功能都仍受支持。
- 支持按请求设置采样种子(per-request sampling seed)，即使在temperature>0的非贪婪采样模式下也能实现确定性推理。

而在slime侧，主要是进行了torch设置：
```
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
```

以及环境变量：
- setting environment variable `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- 针对megatron后端设置`--deterministic-mode`

详细可以看此[PR](https://github.com/THUDM/slime/pull/370)。


### RL算法侧缓解差异（off-policy correction）

#### Mismatch Importance Sampling 

##### TIS（截断重要性采样）

比较早的博客是[(Yao et al.2025)](https://fengyao.notion.site/off-policy-rl)，分析了用重要性采样从算法上缓解训推不一致性的问题。对REINFORCE的梯度表示：$$\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)],$$
转换为：$$\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)} \cdot R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr].$$
而后基于比较经典的[TIS方法](https://ionides.github.io/pubs/ionides08-jcgs.pdf)，可以实现更稳定的重要性采样: $$\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\underbrace{\min\Bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\Bigr)}_{\text{truncated importance ratio}} \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr],$$

扩展到PPO算法，策略梯度为经典的公式: $$\small{ \mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}} \Bigl[ \nabla_\theta \min\Bigl( \frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)}\,\hat A, \;\mathrm{clip}\bigl(\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A \Bigr) \Bigr]}.$$
为了提升吞吐，Hybrid RL系统比如veRL使用vLLM这类推理引擎做rollout采样，而后回到训练侧用训练引擎再做一次 $\pi_{\theta old}$的recompute：$$  

\small{ \mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\mathrm{old}})} \Bigl[ \nabla_\theta \min\Bigl( \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})}\,\hat A, \;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A \Bigr) \Bigr] }.$$
同样的，这种训练和推理的mismatch会出现，那么可以使用TIS进行校准：$$\small{\mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\min\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}, C\Bigr)}_{\text{truncated importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A}, \mathrm{clip}\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}, 1-\epsilon,\;1+\epsilon \Bigr)\,\hat{A}\Bigr)\Bigr]}​$$
文中也做了一些对比实验，表示此类校准确实能减少训推之间的计算分布差异:
![tis-analysis](https://fengyao.notion.site/image/attachment%3A766b9627-d7c4-4f0d-ba10-6eda045390a1%3Agsm8k_int8.png?table=block&id=246721e3-f6c4-803f-b9f1-c1e707b64b02&spaceId=5cbd2ef3-859d-42c5-86d3-a8382485dc0e&width=1420&userId=&cache=v2)


除此之外，不同的IS变种的效果也有所不同。例如Colossal框架使用的PPO-IS格式: $$\small{ \mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\nabla_{\theta}\,\min\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A}, \mathrm{clip}\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})}, 1-\epsilon,\;1+\epsilon \Bigr)\,\hat{A}\Bigr)\Bigr]}$$
以及Nemo-RL框架使用的格式：$$\small{\mathbb{E}_{\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})} }_{\text{importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A}, \mathrm{clip}\Bigl( \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}, 1-\epsilon,\;1+\epsilon \Bigr)\,\hat{A}\Bigr)\Bigr]}​$$
对比下来还是TIS更加稳定，特别是在训推不同量化这种场景下（e.g. FP8/INT8），更加明显。


##### 更多的IS变种

更进一步的，前面介绍的字节的[这篇工作](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)还是更细致分析了不同IS：
- Token-level / Sequence-level TIS
	- 给定upper和lower bound，针对weights超过这个部分的做clip。
- Token-level / Sequence-level MIS
	- 给定upper和lower bound，将超出这个范围的weights置为0，相当与mask out掉，这个策略更加激进，适合处理极端的mismatch
简单来说，有以下的几个结论。

- Token-level的IS是理论**有偏**（biased）的估计，而Sequence-level的IS是**无偏**的估计，通常能有更稳定的训练。![token-seq-compare](https://yingru.notion.site/image/attachment%3A6e07f8c0-50bb-4418-8fa2-878ea8b5283b%3Aimage.png?table=block&id=271211a5-58b7-8008-aab6-ca175a776bdd&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)

- 在复杂的场景（例如TIR），token-level的TIS还是会failed，但是在简单的reasoning RL，当mismatch较小的时候（比如on-policy GRPO）, token-level的TIS够用，可以防止梯度爆炸，但是训练稳定性、训练摸高效果可能由于梯度的bias会有限制。![simper-tis-case](https://yingru.notion.site/image/attachment%3A15dcdfb6-18f5-4b92-91de-fe32f3e4c7d0%3Aimg_v3_02qd_87d5ed2e-e1f2-4b43-9eb5-d52ad649685g.jpg?table=block&id=27b211a5-58b7-80b6-8007-d80f67c22b85&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)


- MIS（masked IS）效果通常比TIS要更好，同样sequence-level的测试，不论在training reward还是评估分数，都能超过不用IS的原始训练。![tis-mis](https://yingru.notion.site/image/attachment%3A07340732-81a9-4d0e-9dc8-0e2450a386b8%3Aimage.png?table=block&id=27b211a5-58b7-808b-92f0-f53e0439f9d6&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)


- sequence-level的MIS在更复杂、更长上下文的自回归任务上，表现的还是比token-level要好，这符合理论预期。 ![seq-mis-token-mis](https://yingru.notion.site/image/attachment%3A6269fa85-8a0c-40ab-8485-9238805b69ce%3Aimg_v3_02qb_d5ff66e7-2585-4275-a6f7-db4a65fbe60g.jpg?table=block&id=27b211a5-58b7-80a9-9837-dc6b5f980e1b&spaceId=effaf72e-4449-4e46-8824-1cc2f447196b&width=1420&userId=&cache=v2)



##### `VeRL`的rollout correction实现

>建议直接参考[verl rollout correction文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)

```yaml
algorithm:
  rollout_correction:
    rollout_is: token                      # IS weights: "token", "sequence", or null
    rollout_is_threshold: 2.0              # Upper threshold for IS weights
    rollout_is_batch_normalize: false      # Batch normalize IS weights to mean=1.0
    rollout_rs: null                       # Rejection sampling: "token", "sequence", "geometric", or null
    rollout_rs_threshold: null             # RS upper threshold (required if rollout_rs is enabled)
    rollout_rs_threshold_lower: null       # RS lower threshold (auto-reciprocal if null)
    rollout_token_veto_threshold: null     # Per-token veto threshold (null = disabled)
    bypass_mode: false  # Skip old_log_prob computation
    use_policy_gradient: false     # Use policy gradient loss (vs PPO loss)

# REQUIRED: Enable log prob calculation
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

具体分为`BypassMode`以及`IS/RS`的实现，实现放在`verl.trainer.ppo.rollout_corr_helper.py`中。

`apply_rollout_correction`在`verl.trainer.ppo.ray_trainer.py`的`RayPPOTrainer`的`fit`中调用，调用场景是`bypass_mode`被使能，也就是不重计算`old_log_prob`，apply以后，实际上设置了`loss_mode`为`rollout_correction`，后续计算的时候, `loss_fn`会选择`compute_policy_loss_with_rollout_correction`（见`verl.trainer.ppo.core_algos.py`），在这个函数中，会`on-the-fly`调用`compute_rollout_correction_and_rejection_mask`计算IS之后的weight和modified的response mask。

而对`bypass_mode`关闭的模式，也就是`decoupled-ppo`模式，会重计算`old_log_prob`，相当于每个`mini batch`都要调用`compute_rollout_correction_and_rejection_mask`计算IS的weight和response make，然后添加到batch中（union），然后正常走正常的流程，比如调用`compute_policy_loss_vanilla`里会处理。


##### `Slime`的IS实现

>可以参考[合入PR](https://github.com/THUDM/slime/pull/429)，和verl类似，都实现了token/sequence/geometric mean级别的TIS、MIS等校准策略。

- `--use-train-infer-is`: Enable training-inference importance sampling
- `--train-infer-is-level`: Aggregation level (token/sequence/geometric)
- `--train-infer-is-mode`: Processing mode (truncate/mask/clip)
- `--train-infer-is-lower-bound`/`--train-infer-is-upper-bound`: Weight bounds
- `--train-infer-is-veto-threshold`: Catastrophic token threshold


#### Routing Replay

>https://arxiv.org/html/2510.11370v1
>https://arxiv.org/html/2510.23027v1       RSPO  routing fluctuations

Rollout Routing Replay 主要解决在专家混合（MoE）大模型中，因其路由机制在训练和推理阶段的行为不一致，导致训练和推理的 logprob产生比较大的差异进而引起强化学习（RL）训练不稳定甚至崩溃的问题。
![router-discrepancy](https://pic1.zhimg.com/80/v2-8cb9c9ea1fd6dc7b8bdd8e5cc68d9031_1440w.webp?source=2c26e567)


Rollout Routing Replay 会在模型进行推理时（Rollout 阶段），记录下每个 token 的 router 分布，然后在后续的训练过程中使用这些 router 分布进行计算。通过这种方式，强制训练过程模仿并对齐推理时的 router 行为，从而弥合两者之间的差异。

![routing-replay](https://picx.zhimg.com/80/v2-ee7c64bc4737825e244d334a3d1d04eb_1440w.webp?source=2c26e567)


>需要注意的是, GSPO论文中提到的Routing Replay, 是训练侧old和target策略之间，如果进行token-level的重要性采样，可能导致专家激活模式在新旧策略之间有差异，这种路由波动可能破坏训练稳定性，GSPO因为引入了seq-level的重要性采样，对单个token的专家波动不敏感，可以不需要routing replay（而GRPO不引入routing replay容易训崩）。而上面讨论的routing replay，主要还是解决训推不一致导致的路由波动带来的问题。
