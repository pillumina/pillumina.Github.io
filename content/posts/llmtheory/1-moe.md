---
title: "MoE环游记：1、从几何意义出发"
date: 2025-08-08T15:05:12+08:00
tags: ["MoE"]
categories: ["Theory"]
series: ["MoE环游记"]
---

MoE（Mixture of Experts）架构的流行自不必多说，近来火出圈的[DeepSeek-V3](https://papers.cool/arxiv/2412.19437)便是MoE架构，传言GPT-5也是MoE架构，国内最近出的一些模型（Qwen3系列相关）也有不少用上了MoE。然而，虽然MoE的研究由来已久，但其应用长时间内都不愠不火，大致上是从去年初的[《Mixtral of Experts》](https://papers.cool/arxiv/2401.04088)开始，MoE才逐渐吸引大家的注意力，其显著优点是参数量大，但训练和推理成本都显著低。

但同时MoE也有一些难题，如训练不稳定、负载不均衡、效果不够好等，这也是它早年没有流行起来的主要原因。不过随着这两年关注度的提升，这些问题在很大程度上已经得到解决，我们在接下来的介绍中会逐一谈到这些内容。

### 问题定义

我们知道，Transformer模型由Attention层和MLP层组成，MoE替换的是模型中MLP层。MLP层又分FFN（FeedForward Network）和GLU（Gated Linear Unit）两种，主流的是GLU，但简单起见我们还是以FFN为例：$$y=f(xW^{(A)})W^{(B)}$$其中$x\in\mathbb{R}^d$ 是输入向量（行向量），$W^{(A)}\in\mathbb{R}^{d\times{D}}$, $W^{(B)}\in\mathbb{R}^{D\times{d}}$ 是两个参数矩阵，$f$是`Element-wise`的激活函数，设$n$是一个能整除$D$的整数，那么上面的FFN可以用分块矩阵等价：
{{< rawhtml >}}
$$ \begin{equation}\boldsymbol{y} = f\big(\boldsymbol{x}\begin{bmatrix}\boldsymbol{W}^{(A)}_1 & \boldsymbol{W}^{(A)}_2 & \cdots & \boldsymbol{W}^{(A)}_n\end{bmatrix}\big)\begin{bmatrix}\boldsymbol{W}^{(B)}_1 \\ \boldsymbol{W}^{(B)}_2 \\ \vdots \\ \boldsymbol{W}^{(B)}_n\end{bmatrix} = \sum_{i=1}^n \underbrace{f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i}_{\boldsymbol{v}_i}\end{equation} $$
{{< /rawhtml >}}

其中
{{< rawhtml >}}$W^{(A)}_i = W^{(A)}_{[:,(i-1)c:ic]}$, $W^{(B)}_i = W^{(B)}_{[(i-1)c:ic,:]}$, $c= D/n${{< /rawhtml >}}，这里的切片按照Python规则来。由此可见，FFN可以等价表示成n个向量
{{< rawhtml >}}$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n${{< /rawhtml >}}
之和，每个向量代表了一个小模型$f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i$的输出，每个小模型计算量相同，这些小模型就是MoE中的“Expert”。

MoE提出的问题是：
>能否只挑k个向量的和来逼近n个向量的和呢？这样就可以将计算量降低到k/n了。


### 模长排序

要解决上述的问题，实质上是要解决**低秩近似**的问题，数学公式就是: 
{{< rawhtml >}}
$$\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \lambda_i \boldsymbol{v}_i - \sum_{i=1}^n\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \lambda_i = k\end{equation}$$ 
{{< /rawhtml >}}
记$\gamma_i = 1 - \lambda_i$，那么它又可以写成：
{{< rawhtml >}}
$$\begin{equation}\mathop{\text{argmin}}_{\gamma_1,\gamma_2,\cdots,\gamma_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \gamma_i = n - k\end{equation}$$
{{< /rawhtml >}}
这个问题的精确求解是比较困难的（NP Hard），但有一个简单的近似解：当$v_i$**两两正交**时，我们有
{{< rawhtml >}}
$$\begin{equation}\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2 = \sum_{i=1}^n \gamma_i^2 \Vert\boldsymbol{v}_i\Vert^2 = \sum_{i=1}^n \gamma_i \Vert\boldsymbol{v}_i\Vert^2\end{equation}$$
{{< /rawhtml >}}
上式最优解显然就是让模长$\Vert\boldsymbol{v}_i\Vert$最小的$n-k$个$\gamma_i$等于1，这又等价于说挑出模长最大的$k$个向量来逼近$n$个向量之和。当$v_i$不满足两两正交的条件时，我们依然用它来作为一个**近似解**。它的几何意义也很直观，**模长越大的向量，在求和过程中越不容易被抵消，从而作用越突出**。


### MoE初现

现在策略已经有了——“挑模长最大的$k$个向量”——可是细想之下我们会发现它并不实用：要挑模长最大的$k$个向量，就得把所有向量的模长都算出来，这又意味着要把所有的$\boldsymbol{v}_i$先算出来，可我们的原本目的却是减少$v_i$的计算量！

为了解决这个矛盾，我们需要重新设计每个Expert模型，使得它的模长可以低成本地计算出来。什么意思呢？首先我们将$v_i$归一化得到$\boldsymbol{e}_i = \boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$，这样每个$e_i$的模长都相同了。接着我们定义  
{{< rawhtml >}}
$$\begin{equation}\underbrace{[\rho_1,\rho_2,\cdots,\rho_n]}_{\boldsymbol{\rho}} = h(\boldsymbol{x}\boldsymbol{W}^{(R)})\quad\in\mathbb{R}_{\geq 0}^n\end{equation}$$
{{< /rawhtml >}}
其中{{< rawhtml >}}$\boldsymbol{W}^{(R)}\in\mathbb{R}^{d\times n}${{< /rawhtml >}}是参数矩阵，{{< rawhtml >}}$h(\cdot)$是一个$\mathbb{R}\to\mathbb{R}_{\geq 0}${{< /rawhtml >}}的激活函数，说白了这就是一个$d$维到$n$维的线性变换加激活函数，所以计算量是比较小的，这部分模型在MoE中被称为“Router”。

$\boldsymbol{\rho}$的作用是什么呢？预测每个Expert的模长！换言之，我们将$\rho_i$作为第$i$个Expert的模长，$\rho_i \boldsymbol{e}_i$才是完整的Expert，它被分解为两部分：计算量比较小的模长$\rho_i$以及计算量比较大的方向$\boldsymbol{e}_i$。为了减少计算量，我们先计算出$\boldsymbol{\rho}$，挑出最大的$k$个后才去计算相应的$e_i$，最后乘上{{< rawhtml >}}$\rho_i${{< /rawhtml >}}并求和：  
{{< rawhtml >}}
$$\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\end{equation}$$
{{< /rawhtml >}}
这便是MoE模型的基本公式。由于计算中只保留了Top-$k$部分，所以它本质上属于一种Sparse模型，而原本的FFN或者$k=n$时的模型，通常称为对应的Dense模型。


### 思路概括

我们再来整理一下整个思路：
>1、一个常规的Dense模型FFN，可以等价改写为$n$个Expert向量$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n$之和;
>
>2、为了节省计算量，我们试图挑出$k$个向量求和来逼近原本的$n$个向量之和 ;
>
>3、转化为数学问题求解后，我们发现挑选规则是模长最大的$k$个向量；
>
>4、直接去算$n$个Expert的模长然后选$k$个实际上是不省计算量的，所以要重新设计Expert；
>
>5、将$\boldsymbol{v}_i$归一化得到$\boldsymbol{e}_i$，然后用另外的小模型（Router）预测模长$\rho_i$，最终的Expert为$\rho_i \boldsymbol{e}_i$；
>
>6、此时，我们就可以先算全体$\rho_i$，挑出$k$个后才去计算$\boldsymbol{e}_i$，达到节省计算量的目的


### 为何如此

可能有些读者疑问，为什么要做这个看似复杂的过程？原本的MoE不是挺好理解的吗？一般的MoE形式为  
$$\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{v}_i\end{equation}$$  
也就是求和前少了对$\boldsymbol{v}_i$的归一化，此时$\rho_i$也没有模长的意义，它纯粹是一个用来对Expert排序的打分模型（即Router）。可为什么将$\rho_i$乘到Expert上去就能让Router学会正确排序Expert呢？笔者发现只有[《Sparse Backpropagation for MoE Training》](https://papers.cool/arxiv/2310.00811)对此给出了一个解释，但还是稍欠直观。

而在本文的几何视角下，我们会发现很多问题就“豁然开朗”了。我们将Expert重新参数化为$\rho_i \boldsymbol{e}_i$后，Dense模型对应于全体$\rho_i \boldsymbol{e}_i$求和，而MoE对应于$\rho_i$选Top-$k$后求和，这是Dense模型的一个有理论保证的逼近。我们没有去考虑Router如何选择Expert，只是每一步都尽可能逼近Dense模型，这可以说是<strong>既要</strong>大参数、<strong>又要</strong>小计算量的最佳选择。

现在$\rho_i$的几何意义是模长而不是概率，所以激活函数$h(\cdot)$就没有归一化的要求了，除了Softmax外，像Sigmoid、ReLU都可以考虑使用，也可以考虑我们在[《Softmax后传：寻找Top-K的光滑近似》](app://obsidian.md/archives/10373)介绍的Top-$k$光滑近似。Router使用非归一化的激活函数，有助于避免$k > 1$时Expert之间的恶性竞争，有时候能取得更好的效果。

最后补充一点，我们前面定义$\boldsymbol{e}_i = \boldsymbol{v}_i/ \Vert\boldsymbol{v}_i\Vert$，目的是让所有$\boldsymbol{e}_i$模长相同，实际操作中不是一定要L2 Normalize，也可以是其他等价操作，比如gamma参数恒等于1的RMS Norm，它更符合我们的输出习惯。

### 文章小结

本文从Dense模型的最佳逼近出发来推导和理解MoE，得到了一种特定的MoE形式，它比现有MoE多了一个Normalize步骤，但能让MoE的几何意义更加明显。当然，不管Normalize与否，MoE之路都只是刚刚开始，更多的困难还在路上。

