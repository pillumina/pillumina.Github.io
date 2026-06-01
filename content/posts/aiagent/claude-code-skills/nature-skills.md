---
title: "Nature Skills - 学术论文工具集"
date: 2026-05-28
draft: false
description: "上海交大博士团队开发的学术论文工具集：写作、润色、图表、审稿回复，覆盖论文全流程"
categories: ["Skills"]
tags: ["skills", "claude-code", "nature-skills", "academic", "paper", "writing", "citation", "journal-club"]
---

> GitHub: [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)
> 作者：袁一哲（上海交大博士），专注医疗 AI

## 核心定位

Nature Skills 是一套为学术论文全流程设计的 AI 工具集，涵盖：

- 论文阅读与理解
- 各部分写作与润色
- 图表生成（Nature 风格）
- 审稿意见回复
- 引文检索与管理

**设计原则**：
1. **仅用一手来源**：基于已发表 Nature 论文，非二手总结
2. **显式优于隐式**：每条规则都有明确理由
3. **上下文感知**：论文不同部分应用不同逻辑
4. **输出优先**：返回可直接使用的内容

## 安装

### Claude Code（推荐）

```bash
/plugin marketplace add https://github.com/Yuan1z0825/nature-skills
/plugin install nature-skills
/reload-plugins
```

### Codex

1. 打开 Codex Desktop
2. 添加自定义插件市场
3. 仓库源：`https://github.com/Yuan1z0825/nature-skills.git`
4. 分支：`main`
5. 安装 `nature-skills` 插件

### 手动安装

```bash
# 克隆仓库
git clone https://github.com/Yuan1z0825/nature-skills.git
cd nature-skills

# 安装单个 skill
mkdir -p ~/.claude/skills
cp -R skills/nature-reader ~/.claude/skills/

# 或安装所有
for d in skills/nature-*; do
  cp -R "$d" ~/.claude/skills/
done
```

## Skill 概览

| Skill | 状态 | 用途 | 触发示例 |
|-------|------|------|----------|
| `nature-reader` | Beta | 论文全文阅读，原文对照 | "read this paper" |
| `nature-paper2ppt` | Beta | 论文转中文 PPT（journal club） | "paper PPT", "journal club" |
| `nature-writing` | Draft | 论文各部分写作 | "write abstract" |
| `nature-polishing` | **Stable** | 论文润色（Nature 风格） | "Nature style", "polish" |
| `nature-citation` | Beta | CNS 级别引文检索 | "CNS citation" |
| `nature-response` | Beta | 审稿意见回复 | "rebuttal" |
| `nature-figure` | **Stable** | Nature 风格图表 | "Nature figure" |
| `nature-data` | Draft | 数据可用性声明 | "Data Availability" |
| `nature-academic-search` | Beta | 学术搜索（PubMed） | "search papers" |

## Best Practices

### 按场景选择 Skill

| 场景 | 推荐 Skill | 说明 |
|------|-----------|------|
| 论文精读 | `nature-reader` | 原文对照，关键信息提取 |
| 组会/实验室报告 | `nature-paper2ppt` | 自动生成故事线，中文幻灯片 |
| 论文写作（初稿） | `nature-writing` | 各部分写作模板 |
| 语言润色（定稿） | `nature-polishing` | Nature 风格精修 |
| 图表设计 | `nature-figure` | Nature 风格图表 |
| 引文管理 | `nature-citation` | CNS 级别检索 |
| 审稿回复 | `nature-response` | 礼貌有力 |
| 文献调研 | `nature-academic-search` | PubMed 搜索 |

### nature-paper2ppt 使用流程

1. **提供论文**：可以是 PDF、arXiv ID、或论文 URL
2. **AI 分析**：识别论文类型和论证逻辑
3. **选择图表**：从全文中选择关键图表
4. **生成内容**：中文幻灯片内容 + 演讲备注
5. **创建文件**：实际 PPTX 文件
6. **自查纠错**：检查图表质量、文字溢出、非模板视觉设计

**关键原则**：
- 整片是一个连续的运动叙事
- 禁止 PowerPoint 切换风格
- 每张幻灯片之间要有视觉连贯性

### nature-polishing 使用技巧

**最佳时机**：论文初稿完成后，最终提交前

**使用方式**：
```
润色这段摘要，用 Nature 风格
Polish this paragraph in Nature journal style
```

**输出特点**：
- 学术表达优化
- 逻辑连贯性检查
- 语言地道性

### nature-figure 使用指南

**支持的图表类型**：
- 柱状图/折线图
- 热力图
- 流程图
- 散点图
- ROC 曲线
- 混淆矩阵

**使用方式**：
```
帮我画一个模型架构图
Create a ROC curve with this data
```

### nature-citation 检索策略

**检索范围**：CNS（Cell、Nature、Science）及子刊

**使用方式**：
```
检索这篇论文被哪些 Nature 文章引用过
Find CNS citations for this paper
```

**输出格式**：支持 EndNote、RIS、BibTeX

### nature-response 审稿回复

**回复结构**：
1. 感谢审稿人意见
2. 逐条回应
3. 修改说明（如果有）
4. 再次感谢

**使用方式**：
```
帮我写审稿意见回复
Write rebuttal letters
```

## 详细说明

### nature-reader（论文阅读）

功能：
- 论文全文阅读
- 原文对照模式
- 关键信息提取

### nature-paper2ppt（论文转 PPT）

**Stable 版本**，推荐用于 journal club 和组会。

适用场景：
- Journal club
- 组会报告
- 论文分享

### nature-polishing（论文润色）

**Stable 版本**，推荐使用。

功能：
- Nature 风格语言润色
- 学术表达优化
- 逻辑连贯性检查

### nature-figure（图表生成）

**Stable 版本**，推荐使用。

功能：
- Nature 风格图表设计
- 多类型支持
- 配色方案优化

### nature-citation（引文检索）

功能：
- CNS 级别期刊引文检索
- 引用格式导出
- 多格式支持

### nature-response（审稿回复）

功能：
- 审稿意见分析
- 回复模板生成
- 礼貌且有力的表达

### nature-academic-search（学术搜索）

需要额外配置：

```bash
# 安装 NCBI API（可选，获得更高 PubMed 速率限制）
bash skills/nature-academic-search/install.sh your-email@example.com
```

## 使用方式

安装后直接用自然语言触发：

| 示例 |
|------|
| "帮我把这篇论文做成 journal club PPT" |
| "润色这段摘要，用 Nature 风格" |
| "帮我画一个模型架构图" |
| "写一封审稿意见回复" |
| "搜索这篇论文被哪些 CNS 文章引用过" |

## MCP 配置（nature-academic-search 专用）

```bash
bash skills/nature-academic-search/install.sh your-email@example.com
```

设置 `NCBI_API_KEY` 可获得更高的 PubMed 速率限制。

## 设计哲学

1. **一手来源**：所有建议基于已发表的 Nature 论文，非二手总结
2. **显式规则**：每条规则都有明确理由，可追溯
3. **上下文感知**：摘要/方法/结果/讨论各有不同处理逻辑
4. **输出导向**：返回内容可直接使用，无需二次编辑

---

> 💡 **Tip**: 学术写作相关需求优先使用 nature-skills，尤其是 `nature-polishing` 和 `nature-figure` 是 Stable 版本
