---
title: "Skill: HyperFrames - 用 HTML 制作视频"
date: 2026-05-28
draft: false
description: "用 HTML + GSAP 制作高质量视频的框架，为 AI Agent 优化"
categories: ["AI Agent", "Claude Code Skills"]
tags: ["hyperframes", "html", "video", "gsap", "animation"]
---

> GitHub: [heygen-com/hyperframes](https://github.com/heygen-com/hyperframes)
> 文档: [hyperframes.heygen.com](https://hyperframes.heygen.com/introduction)

## 核心定位

HyperFrames 是一个开源的 HTML 视频渲染框架，用「写 HTML 来渲染视频」的方式工作。最大的特点是 **AI-First**——AI Agent 天然会写 HTML，不需要额外学习。

**与 Remotion 的核心区别**：

| 特性 | HyperFrames | Remotion |
|------|-------------|----------|
| 编写方式 | HTML + CSS + GSAP | React 组件 (TSX) |
| 构建步骤 | 无，`.html` 直接可用 | 需要 bundler |
| 动画精度 | Seekable，帧级精确 | 依赖 wall-clock |
| 开源许可 | Apache 2.0（完全开源） | 自定义许可证（需付费） |

> HyperFrames 借鉴了 Remotion 的设计思路，代码中保留了对其首创模式的致谢注释。两者的核心分歧在于：**Agent 主要写什么**。Remotion 选择 React 组件，HyperFrames 选择 HTML。

## 安装

### Claude Code 插件市场（推荐）

```bash
/plugin marketplace add heygen-com/hyperframes
/reload-plugins
```

### npx skills（通用）

```bash
npx skills add heygen-com/hyperframes
```

### CLI 工具

```bash
# 全局安装 CLI
npm install -g hyperframes

# 初始化新项目
hyperframes init my-video
cd my-video

# 开发预览
hyperframes preview      # 浏览器预览，live reload

# 渲染输出
hyperframes render       # 输出 MP4
```

**前置要求**：Node.js >= 22, FFmpeg

## 核心概念

### Composition（视频构成）

视频由多个 **clips**（片段）组成，通过 HTML data 属性定义时序：

```html
<div id="stage"
     data-composition-id="my-video"
     data-start="0"
     data-width="1920"
     data-height="1080">

  <!-- 视频片段 -->
  <video
    id="clip-1"
    data-start="0"
    data-duration="5"
    data-track-index="0"
    src="intro.mp4"
    muted
    playsinline>
  </video>

  <!-- 图片叠加 -->
  <img
    id="overlay"
    class="clip"
    data-start="2"
    data-duration="3"
    data-track-index="1"
    src="logo.png">

  <!-- 音频 -->
  <audio
    id="bg-music"
    data-start="0"
    data-duration="9"
    data-track-index="2"
    data-volume="0.5"
    src="music.wav">
  </audio>
</div>
```

### Timeline Contract（动画注册）

HyperFrames 通过 `window.__timelines` 控制 GSAP 动画：

```javascript
window.__timelines = window.__timelines || {};
const tl = gsap.timeline({ paused: true });

// 构建动画
tl.from(".title", { y: 48, opacity: 0, duration: 0.6, ease: "power3.out" }, 0);
tl.to(".accent", { scaleX: 1, duration: 0.5, ease: "power2.out" }, 0.25);

// 注册 — key 必须与 data-composition-id 匹配
window.__timelines["my-video"] = tl;
```

**关键规则**：
- ✅ Timeline 必须 `{ paused: true }`
- ✅ 必须注册到 `window.__timelines["composition-id"]`
- ❌ 不要调用 `tl.play()` — HyperFrames 通过 seek 控制
- ❌ 不要在 async/setTimeout 中创建 timeline
- ❌ 不要使用 `repeat: -1` — 使用有限重复次数

## CLI 命令

| 命令 | 说明 |
|------|------|
| `hyperframes init` | 初始化新项目 |
| `hyperframes lint` | 语法检查 |
| `hyperframes validate` | 验证 composition |
| `hyperframes preview` | 浏览器预览 |
| `hyperframes render` | 渲染 MP4 |
| `hyperframes inspect` | 视觉检查 |
| `hyperframes doctor` | 环境诊断 |

## 确定性原则

HyperFrames 追求 **确定性渲染**：相同输入 = 相同输出。

**禁止项**：
1. `Math.random()` — 每次运行必须产生相同结果（使用种子 PRNG 如 mulberry32）
2. 无限循环 `repeat: -1` — 视频时长有限
3. 异步创建 timeline — 必须同步注册
4. 提前离场 — 场景转换即离场时刻

**必须项**：
1. 入场动画 — 每个可见元素必须有入场动画
2. 视频/音频分离 — 视频 muted，音频用独立 `<audio>` 元素

## Best Practices

### 三条黄金规则（确保正确渲染）

1. **根元素必须有**：`data-composition-id`、`data-width`、`data-height`
2. **时间元素必须有**：`class="clip"`、`data-start`、`data-duration`、`data-track-index`
3. **GSAP timeline 必须**：使用 `{ paused: true }` 并注册到 `window.__timelines`

### 正确的提示词模式

**冷启动** — 从零描述视频：
```
Using `/hyperframes`, create a 10-second product intro with a fade-in title, a background video, and background music.
```

**暖启动** — 转换现有内容：
```
Take a look at this GitHub repo and explain its uses using `/hyperframes`.
Summarize the attached PDF into a 45-second pitch video using `/hyperframes`.
Turn this CSV into an animated bar chart race using `/hyperframes`.
```

**迭代调整** — 像和视频编辑对话：
```
Make the title 2x bigger, swap to dark mode, and add a fade-out at the end.
Add a lower third at 0:03 with my name and title.
```

### 缓动函数选择（对应视觉感受）

| 视觉感受 | GSAP ease |
|----------|-----------|
| 平滑 | `power2.out` |
| 干脆 | `power4.out` |
| 弹跳 | `back.out` |
| 弹性 | `elastic.out` |
| 戏剧性 | `expo.out` |

### 字幕风格选择

| 风格 | 字号 | 字体 | 动画 |
|------|------|------|------|
| Hype（动感） | 72-96px | 粗体 | scale-pop |
| Corporate（商务） | 56-72px | 无衬线 | fade + slide |
| Tutorial（教程） | 48-64px | 等宽 | typewriter |

### 典型工作流

```bash
# 1. 初始化项目
hyperframes init my-video

# 2. 用 Claude Code 打开项目
cd my-video

# 3. 用 /hyperframes 描述视频
# 4. 预览迭代
hyperframes preview

# 5. 小步调整（像和视频编辑对话）

# 6. 渲染输出
hyperframes render --output final.mp4
```

### 性能优化

- 预览时卡顿是正常的（实时渲染）
- 如果最终渲染效果差，先优化：
  - 大面积 `backdrop-filter` 模糊
  - 过大的图片
  - 过重的阴影叠加

### 模板选择

| 模板 | 风格 | 适用场景 |
|------|------|----------|
| `swiss-grid` | 简洁结构 | 企业/技术演示 |
| `yt-graph` | 编辑风格 | 数据故事/动态图表 |
| `play-mode` | 弹性活力 | 社交媒体/产品发布 |
| `vignelli` | 大胆竖版 | 移动端竖屏视频 |

初始化模板：`hyperframes init my-video --example swiss-grid`

## Skill 体系

HyperFrames 提供了多个 skill，覆盖不同场景：

| Skill | 用途 |
|-------|------|
| `hyperframes` | 基础用法：composition、captions、TTS |
| `hyperframes-cli` | CLI 命令：init/lint/preview/render |
| `hyperframes-media` | 资源预处理：TTS、转录、去背景 |
| `hyperframes-registry` | Block/component 安装 |
| `website-to-hyperframes` | 网页转视频 |
| `remotion-to-hyperframes` | Remotion 项目迁移 |
| `gsap` | GSAP 动画（见 [gsap.md](./gsap.md)） |
| `animejs` | Anime.js 动画 |
| `css-animations` | CSS 关键帧动画 |
| `lottie` | Lottie 动画 |
| `three` | Three.js 3D 场景 |
| `waapi` | Web Animations API |

## Block 组件库

50+ 开箱即用的组件：

```bash
hyperframes add flash-through-white   # 闪光转场
hyperframes add instagram-follow      # 社交关注组件
hyperframes add data-chart            # 数据图表
```

浏览完整目录：[hyperframes.heygen.com/catalog](https://hyperframes.heygen.com/catalog/)

## 常见错误

| 错误做法 | 正确做法 |
|----------|----------|
| 请求 React/Vue 组件 | compositions 是纯 HTML |
| 跳过 `/hyperframes` 命令 | agent 需要 skill 上下文 |
| 默认 4K/60fps | 默认 1920x1080 30fps 已经很好且渲染快 |

---

> 💡 **Tip**: 安装 skill 后，直接用自然语言描述视频需求即可触发使用
