+++
date = '2026-06-15'
draft = false
title = 'LLM 架构计算方法论：从 config.json 到推理显存'
categories = ['aiinfra']
layout = 'single'
math = false
summary = '从 config.json 到参数量、FLOPs、KV Cache、推理显存的完整计算推导。全系列 4 篇，基于 8 个开源模型的实战拆解经验。'
+++

# LLM 架构计算方法论

> 从 config.json 到参数量、FLOPs、KV Cache、推理显存的完整计算推导。基于 8 个开源模型（M2.7 / GLM-5.1 / V4-Flash / Qwen3.5 / Mimo / Kimi / Nemotron / M3）的实战拆解经验。

## 系列文章

| # | 篇名 | 内容 |
|---|------|------|
| 1 | [预备知识与参数分解](part-1/) | CH 1-2：矩阵乘法基础、config.json 字段速查、参数分解四步法 |
| 2 | [FLOPs 估算](part-2/) | CH 3：六种注意力架构的 FLOPs 公式推导与跨架构对比 |
| 3 | [KV Cache 与推理显存](part-3/) | CH 4-5：KV Cache 原理与四种架构缓存策略、推理显存完整拆解 |
| 4 | [M3 实战推演与 Roofline 模型](part-4/) | CH 6-7：MiniMax M3 全链路推演、Roofline 延迟分析 + 公式速查 + 附录 |

## 速查附录

- **公式速查**：包含在[第四篇](part-4/)末尾
- **常见计算错误 Top 12**：包含在[第四篇](part-4/)末尾
- **config.json 字段速查表**：[第四篇附录 A](part-4/#附录-a-常见-configjson-字段速查表)
- **符号与缩写表**：[第四篇附录 B](part-4/#附录-b-符号与缩写表)
- **8 模型计算结果速览**：[第四篇附录 C](part-4/#附录-c-8-个已拆解模型的计算结果速览)
