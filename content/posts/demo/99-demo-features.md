---
title: "博客富文本新功能：Callout、折叠、脚注、ECharts"
date: 2026-05-28T12:00:00+08:00
categories: ["demo"]
tags: ["demo", "features"]
---

本文展示博客新支持的富文本功能：Callout 提示块、折叠内容、脚注增强和 ECharts 交互图表。

## Callout 提示块

Callout 是一种突出重要信息的视觉方式，支持四种类型：

### 示例

- `tip` - 提示信息
- `info` - 背景信息
- `warning` - 注意事项
- `danger` - 危险警告

{{< callout "tip" "提示" >}}
当你学习新概念时，尝试用自己的话复述一遍，这能加深理解。
{{< /callout >}}

{{< callout "info" "背景信息" >}}
Transformer 架构最早由 Google 在 2017 年的论文《Attention Is All You Need》中提出。
{{< /callout >}}

{{< callout "warning" "注意事项" >}}
这个配置选项在生产环境中不建议修改，可能导致服务不稳定。
{{< /callout >}}

{{< callout "danger" "危险警告" >}}
执行此操作将删除所有数据，且无法恢复。请务必确认已备份重要文件。
{{< /callout >}}

### 用法

```markdown
{{</* callout "tip" "标题" */>}}
内容
{{</* /callout */>}}
```

类型可选：`tip`、`info`、`warning`、`danger`

---

## 折叠内容

适用于隐藏较长但需要时可见的内容：

<details>
<summary>点击展开：分布式训练的核心挑战</summary>

分布式训练面临三个核心挑战：

1. **通信开销** - 多节点间需要同步梯度，带宽成为瓶颈
2. **负载均衡** - 不同计算任务耗时不同，需要动态调整
3. **容错处理** - 长训练任务中节点故障如何恢复

```python
# 示例：简单的梯度同步
for param in model.parameters():
    dist.all_reduce(param.grad)
    param.grad /= world_size
```

</details>

<details>
<summary>点击展开：矩阵乘法优化技巧</summary>

矩阵乘法是深度学习中最耗时的操作之一。以下是几个常用优化技巧：

- **Tiling** - 将大矩阵分块以提高缓存命中率
- **向量化** - 使用 SIMD 指令一次处理多个数据
- **混合精度** - 使用 FP16/BF16 减少计算量

</details>

### 用法

```markdown
<details>
<summary>点击展开：标题</summary>

这里是隐藏的内容...

</details>
```

---

## 脚注增强

脚注用于添加补充说明而不打断正文流[^1]：

分布式系统的 CAP 理论指出，一个分布式系统无法同时满足一致性（Consistency）、可用性（Availability）和分区容错性（Partition tolerance）[^cap]。

当你设计系统时，需要根据业务场景权衡这三个特性。比如对于金融交易系统，一致性是首要考虑[^finance]。

[^1]: 脚注会自动编号并显示在文章末尾。
[^cap]: Eric Brewer, "Towards Robust Distributed Systems", PODC, 2000.
[^finance]: 在证券交易系统中，任何不一致都可能导致交易错误，因此通常选择 CP 模型。

---

## ECharts 交互图表

ECharts 支持多种交互式图表：

### 1. 折线图 - 训练曲线

{{< echarts "loss-curve" "350px" >}}
{
  "title": {
    "text": "训练 Loss 曲线",
    "left": "center",
    "textStyle": {"fontSize": 14}
  },
  "tooltip": {"trigger": "axis"},
  "legend": {"bottom": 10, "data": ["Train Loss", "Val Loss"]},
  "grid": {"left": "3%", "right": "4%", "bottom": "15%", "top": "15%", "containLabel": true},
  "xAxis": {
    "type": "category",
    "boundaryGap": false,
    "data": ["Step 100", "500", "1K", "5K", "10K", "50K"]
  },
  "yAxis": {"type": "value", "name": "Loss"},
  "series": [
    {
      "name": "Train Loss",
      "type": "line",
      "smooth": true,
      "data": [2.8, 2.1, 1.6, 0.9, 0.5, 0.2],
      "areaStyle": {"opacity": 0.2},
      "lineStyle": {"width": 2},
      "itemStyle": {"color": "#3B82F6"}
    },
    {
      "name": "Val Loss",
      "type": "line",
      "smooth": true,
      "data": [3.0, 2.4, 1.9, 1.3, 0.8, 0.5],
      "areaStyle": {"opacity": 0.2},
      "lineStyle": {"width": 2},
      "itemStyle": {"color": "#EF4444"}
    }
  ]
}
{{< /echarts >}}

### 2. 柱状图 - 性能对比

{{< echarts "benchmark" "350px" >}}
{
  "title": {
    "text": "推理框架性能对比",
    "left": "center",
    "textStyle": {"fontSize": 14}
  },
  "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
  "legend": {"bottom": 10, "data": ["吞吐量", "延迟"]},
  "grid": {"left": "3%", "right": "4%", "bottom": "15%", "top": "15%", "containLabel": true},
  "xAxis": {
    "type": "category",
    "data": ["vLLM", "TensorRT-LLM", "SGLang", "LightLLM"]
  },
  "yAxis": [
    {
      "type": "value",
      "name": "吞吐量 (tokens/s)",
      "position": "left"
    },
    {
      "type": "value",
      "name": "延迟 (ms)",
      "position": "right"
    }
  ],
  "series": [
    {
      "name": "吞吐量",
      "type": "bar",
      "data": [2450, 3200, 2800, 2100],
      "itemStyle": {"color": "#3B82F6", "borderRadius": [4, 4, 0, 0]}
    },
    {
      "name": "延迟",
      "type": "bar",
      "yAxisIndex": 1,
      "data": [85, 62, 72, 95],
      "itemStyle": {"color": "#10B981", "borderRadius": [4, 4, 0, 0]}
    }
  ]
}
{{< /echarts >}}

### 3. 饼图 - 资源分配

{{< echarts "pie-chart" "350px" >}}
{
  "title": {
    "text": "GPU 内存使用分布",
    "left": "center",
    "textStyle": {"fontSize": 14}
  },
  "tooltip": {"trigger": "item", "formatter": "{b}: {c}GB ({d}%)"},
  "legend": {"bottom": 10, "orient": "horizontal"},
  "series": [
    {
      "type": "pie",
      "radius": ["40%", "70%"],
      "center": ["50%", "45%"],
      "avoidLabelOverlap": true,
      "itemStyle": {"borderRadius": 6, "borderColor": "#fff", "borderWidth": 2},
      "label": {"show": true, "formatter": "{b}\n{c}GB"},
      "emphasis": {
        "label": {"show": true, "fontSize": 14, "fontWeight": "bold"}
      },
      "data": [
        {"value": 45, "name": "模型权重", "itemStyle": {"color": "#3B82F6"}},
        {"value": 20, "name": "KV Cache", "itemStyle": {"color": "#8B5CF6"}},
        {"value": 12, "name": "激活值", "itemStyle": {"color": "#06B6D4"}},
        {"value": 8, "name": "优化器状态", "itemStyle": {"color": "#F59E0B"}},
        {"value": 15, "name": "其他", "itemStyle": {"color": "#6B7280"}}
      ]
    }
  ]
}
{{< /echarts >}}

### 用法

```markdown
{{</* echarts "chart-id" "400px" */>}}
{"xAxis": {"type": "category", "data": ["A", "B"]}, "series": [{"type": "line", "data": [1, 2]}]}
{{</* /echarts */>}}
```

---

## 总结

以上功能可以显著提升技术博客的表达能力：

| 功能 | 适用场景 | 类型 |
|------|----------|------|
| Callout | 重点提示、警告、信息 | 视觉强调 |
| 折叠内容 | 技术细节、代码、扩展阅读 | 内容组织 |
| 脚注 | 参考文献、术语解释 | 学术规范 |
| ECharts | 数据分析、性能对比 | 交互图表 |

有问题或建议？欢迎反馈！
