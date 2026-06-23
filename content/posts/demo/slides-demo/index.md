+++
date = '2026-06-23T11:00:00+08:00'
draft = false
title = '幻灯片嵌入功能演示'
categories = ['demo']
tags = ['demo', 'slides', 'features', 'showcase']
summary = '演示在文章详情页嵌入 HTML 幻灯片的能力——支持键盘/滚轮/触屏翻页、全屏播放、懒加载。由 guizang-ppt / frontend-slides 等工具生成的 HTML 放入 static/slides/ 即可。'
slides = '/slides/slides-demo/'
+++

> 本文顶部嵌入的幻灯片是一个真实可交互的 demo。可以用键盘 ← → 翻页、滚轮翻页、点全屏按钮进入沉浸模式。幻灯片下方的正文是常规文章内容。

## 功能简介

博客文章现在支持嵌入独立的 HTML 幻灯片（slides）。适用于：

- **长文的概览版**：读者先快速翻完 slides 把握全貌，再决定是否细读正文
- **演讲/分享的配套材料**：slides 来自 Slidev / reveal.js / guizang-ppt / frontend-slides 等工具的产物
- **教学/教程**：用 slides 做步骤演示，正文做详细解释

## 如何使用

### 1. 准备 slides HTML

用任意工具生成自包含的 HTML 幻灯片（内嵌 CSS/JS，无外部依赖最好），放到：

```
static/slides/{slug}/index.html
```

例如本文的 slides 在 `static/slides/slides-demo/index.html`。

### 2. 在文章 frontmatter 声明

```toml
+++
title = "我的文章"
slides = "/slides/slides-demo/"   # 指向 slides 目录
+++
```

模板会自动在文章顶部（TOC 之后、正文之前）渲染幻灯片嵌入区域。

## 交互方式

嵌入的幻灯片支持：

| 操作 | 效果 |
|------|------|
| **键盘 ← →** | 上一页/下一页 |
| **空格 / PageDown** | 下一页 |
| **PageUp** | 上一页 |
| **Home / End** | 跳到首页/末页 |
| **滚轮** | 翻页（400ms 防抖） |
| **触屏左右滑** | 移动端翻页 |
| **全屏按钮** | 进入沉浸式播放 |
| **ESC** | 退出全屏 |
| **新窗口按钮** | 在新标签页打开 slides |

## 性能考虑

幻灯片 HTML 通常较重（含动画、图片、JS 库），博客侧做了三项优化：

1. **IntersectionObserver 懒加载**——slides iframe 仅在用户滚动到距离 300px 时才创建，首屏不阻塞
2. **content-visibility: auto**——嵌入容器离屏时跳过渲染
3. **iframe 隔离**——slides 的 CSS/JS 与博客主页面完全隔离，互不影响

## 全屏播放

点 slides 右上角的全屏图标，进入覆盖整个视口的沉浸式播放层：

- 毛玻璃半透明背景
- 右上角关闭按钮（或 ESC）
- 底部短暂显示操作提示
- 锁定背景滚动

全屏模式下，slides 内部的键盘事件优先处理（翻页），ESC 退出由博客负责。

## 适配建议

| slides 内容类型 | 建议 |
|-----------------|------|
| 宽表格 | 在 slide 内做横向滚动 |
| 长代码块 | 在 slide 内做纵向滚动 |
| 高清图片 | slides 自己做 lazy load |
| 视频/动画 | 用 `allow="autoplay"` |
| 暗色主题 | slides 自带即可，博客不强制 |

## 技术实现

- **模板**：`layouts/partials/slides_embed.html`——读 `.Params.slides`，渲染 iframe 容器
- **注入点**：`layouts/_default/single.html`——TOC 之后、post-content 之前
- **CSS**：`assets/css/custom.css` 末尾——16:9 容器、全屏 overlay、加载占位符
- **JS**：`layouts/partials/extend_head.html` 末尾——IntersectionObserver + 全屏 Overlay 逻辑

slides 本身用什么工具生成都可以，只要产出是一个自包含的 `index.html`。推荐：

- **guizang-ppt** / **frontend-slides**：中文场景，模板精美
- **Slidev**：开发者友好，Vue 生态
- **reveal.js**：老牌，功能最全

---

下方的幻灯片就是这套功能的真实演示——往上滚看看。
