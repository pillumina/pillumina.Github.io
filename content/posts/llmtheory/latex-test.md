---
title: "LaTeX Test Page"
date: 2024-01-01
math: true
draft: false
---

# LaTeX 渲染测试

## 基础测试

行内公式：$E = mc^2$

块级公式：
$$E = mc^2$$

## 复杂公式测试

### 原始问题公式1（更稳妥写法，拆成两条显示公式）

{{< rawhtml >}}
$$
\mathbf{y} = f\left(\mathbf{x}\, \big[\,\mathbf{W}^{(A)}_1\; \mathbf{W}^{(A)}_2\; \cdots\; \mathbf{W}^{(A)}_n\,\big]\right)
\begin{bmatrix}
\mathbf{W}^{(B)}_1 \\
\mathbf{W}^{(B)}_2 \\
\vdots \\
\mathbf{W}^{(B)}_n
\end{bmatrix}
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\mathbf{y} = \sum_{i=1}^n f\big(\mathbf{x}\mathbf{W}^{(A)}_i\big)\,\mathbf{W}^{(B)}_i
$$
{{< /rawhtml >}}

## 其他常见LaTeX测试

### 分数和根号：
$$\frac{\sqrt{x^2 + y^2}}{2}$$

### 积分：
$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

### 矩阵（确保每行使用 \\ 换行）：
{{< rawhtml >}}
$$
\begin{pmatrix}
 a & b \\
 c & d
\end{pmatrix}
\begin{bmatrix}
 x \\
 y
\end{bmatrix}
=
\begin{bmatrix}
 ax + by \\
 cx + dy
\end{bmatrix}
$$
{{< /rawhtml >}}
### 求和和乘积：
$$\sum_{i=1}^n i = \frac{n(n+1)}{2}$$

$$\prod_{i=1}^n i = n!$$

### 上下标组合：
$x^{2^3}$, $x_{i_j}$, $x^{(a)}_{(b)}$

### 特殊符号：
$\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta$

$\nabla, \partial, \infty, \sum, \prod, \int, \oint$
