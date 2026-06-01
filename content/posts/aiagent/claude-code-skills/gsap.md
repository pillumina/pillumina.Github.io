---
title: "Skill: GSAP - HyperFrames 动画引擎"
date: 2026-05-28
draft: false
description: "GSAP 官方为 HyperFrames 提供的动画技能参考，GreenSock 动画平台"
categories: ["Skills"]
tags: ["skills", "animation", "gsap", "hyperframes", "greensock"]
---

> GitHub: [greensock/gsap-skills](https://github.com/greensock/gsap-skills)
> 官网: [gsap.com](https://gsap.com)

## 核心定位

GSAP (GreenSock Animation Platform) 是专业的 JavaScript 动画库，HyperFrames 使用它作为主要动画引擎。

**重要更新**：GSAP 完全免费，包括所有插件。自 Webflow 收购 GSAP 后，所有插件（包括 formerly Club-only 的 SplitText、MorphSVG 等）对所有人都免费使用，包括商业用途。

## 安装

### Claude Code 插件市场（推荐）

```bash
/plugin marketplace add greensock/gsap-skills
/reload-plugins
```

### npx skills（通用）

```bash
npx skills add https://github.com/greensock/gsap-skills
```

## HyperFrames Contract

HyperFrames 通过 `window.__timelines` 控制 GSAP，所有 timeline 必须遵循此约定：

```javascript
window.__timelines = window.__timelines || {};
const tl = gsap.timeline({ paused: true }));

// 构建动画
tl.from(".title", { y: 48, opacity: 0, duration: 0.6, ease: "power3.out" }, 0);
tl.to(".accent", { scaleX: 1, duration: 0.5, ease: "power2.out" }, 0.25);

// 注册 — key 必须与 data-composition-id 匹配
window.__timelines["main"] = tl;
```

**关键规则**：
- ✅ Timeline 必须 `{ paused: true }`
- ✅ 必须注册到 `window.__timelines["composition-id"]`
- ✅ 注册 key 必须与 composition root 的 `data-composition-id` 匹配
- ❌ 不要调用 `tl.play()` — HyperFrames 通过 seek 控制
- ❌ 不要在 async/setTimeout 中创建 timeline
- ❌ 不要使用 `repeat: -1` — 使用有限重复次数

## 核心方法

| 方法 | 说明 |
|------|------|
| `gsap.to(targets, vars)` | 从当前状态动画到目标状态（最常用） |
| `gsap.from(targets, vars)` | 从目标状态动画到当前状态（入场动画） |
| `gsap.fromTo(targets, fromVars, toVars)` | 显式指定起始和结束状态 |
| `gsap.set(targets, vars)` | 立即应用（duration: 0） |

## Transform 别名（优先使用）

GSAP 的 CSSPlugin 使用一致的变换顺序（translate → scale → rotation → skew），比原生 CSS 更可靠。

| GSAP 属性 | 等价 CSS |
|-----------|----------|
| `x`, `y`, `z` | translateX/Y/Z (px) |
| `xPercent`, `yPercent` | translateX/Y in % |
| `scale`, `scaleX`, `scaleY` | scale |
| `rotation` | rotate (deg) |
| `rotationX`, `rotationY` | 3D rotate |
| `skewX`, `skewY` | skew |
| `transformOrigin` | transform-origin |
| `autoAlpha` | opacity + visibility hidden |

**autoAlpha 优于 opacity**：值为 0 时同时设置 `visibility: hidden`，避免元素仍可点击。

## 常用配置项

| 属性 | 类型 | 说明 |
|------|------|------|
| `duration` | Number | 秒数（默认 0.5） |
| `ease` | String | 缓动函数 |
| `stagger` | Number/Object | 错开时间：`0.1` 或 `{ amount: 0.3, from: "center" }` |
| `repeat` | Number | 有限次数（计算可见时长） |
| `yoyo` | Boolean | 与 repeat 配合，方向交替 |
| `immediateRender` | Boolean | from/fromTo 默认 true |

## Timeline 位置参数

控制 tween 在时间线上的位置（第三个参数）：

| 值 | 含义 |
|---|------|
| `0` | 绝对 0 秒 |
| `1` | 绝对 1 秒 |
| `"+=0.5"` | 前一个结束后 0.5 秒 |
| `"-=0.2"` | 前一个结束前 0.2 秒 |
| `"<"` | 与前一个同时开始 |
| `">"` | 与前一个同时结束 |
| `"<0.2"` | 前一个开始后 0.2 秒 |
| `"labelName"` | 在指定标签处 |

```javascript
tl.to(".a", { x: 100 }, 0);           // 在 0s
tl.to(".b", { y: 50 }, "+=0.5");      // 前一个结束后 0.5s
tl.to(".c", { opacity: 0 }, "<");      // 与 .b 同时开始
tl.to(".d", { scale: 2 }, "<0.2");      // .c 开始后 0.2s
```

## Timeline 标签

命名关键时间点，便于阅读和维护：

```javascript
tl.addLabel("intro", 0);
tl.to(".a", { x: 100 }, "intro");
tl.addLabel("outro", "+=0.5");
tl.to(".b", { opacity: 0 }, "outro");

// 用于预览
tl.seek("outro");
```

## 缓动函数

决定动画的时间曲线。常用场景推荐：

| 场景 | 推荐 | 特点 |
|------|------|------|
| 入场 | `power3.out` | 快出慢停，自然 |
| 离场 | `power2.in` | 慢出快停 |
| 强调 | `elastic.out(1, 0.3)` | 弹性效果 |
| 按钮点击 | `back.out(1.7)` | 轻微回弹 |
| 线性 | `none` | 匀速，用于 scroll-driven |

**内置缓动**：每个都有 `.in`、`.out`、`.inOut` 变体：
- `power1`–`power4`：基础曲线，数字越大越陡
- `back`：带超调
- `bounce`：弹跳
- `elastic`：弹性
- `circ`：圆弧
- `expo`：指数
- `sine`：正弦

## Best Practices

### Timeline 模式

**创建带默认值的 timeline**：
```javascript
const tl = gsap.timeline({
  paused: true,
  defaults: { duration: 0.5, ease: "power2.out" }
});
```

**嵌套实现模块化**：
```javascript
function createIntroAnimation() {
  const tl = gsap.timeline({ paused: true });
  tl.to(".title", { y: 0, opacity: 1, duration: 0.8 })
    .to(".subtitle", { y: 0, opacity: 1, duration: 0.6 }, "-=0.4")
    .to(".cta", { scale: 1, opacity: 1, duration: 0.4 }, "-=0.2");
  return tl;
}

window.__timelines["intro"] = createIntroAnimation();
```

**交错入场动画**：
```javascript
gsap.from('.box', {
  opacity: 0,
  y: 50,
  stagger: 0.1,  // 每个元素延迟 0.1s
  duration: 0.8
});
```

### 性能优化

| 最佳实践 | 说明 |
|----------|------|
| 优先使用 transform 和 opacity | `x`, `y`, `scale`, `rotation`, `opacity` 运行在合成器线程 |
| 使用 CSS `will-change: transform` | 对动画元素启用 GPU 加速 |
| 使用 stagger 代替多个 tween | 减少 DOM 操作 |
| 使用 `gsap.quickTo()` 处理高频更新 | 如鼠标跟随：```javascript
let xTo = gsap.quickTo("#id", "x", { duration: 0.4, ease: "power3" });
document.addEventListener("mousemove", e => xTo(e.pageX));
``` |
| 使用 `autoAlpha` 代替 `opacity` | 0 时同时设置 visibility，避免点击穿透 |

### 常见错误

| 错误 | 正确做法 |
|------|----------|
| 调用 `tl.play()` | HyperFrames 通过 seek 控制 |
| 动画 layout 属性 | 使用 scale/x/y |
| `repeat: -1` | 计算精确次数 |
| 链式 `delay` | 使用 timeline + 位置参数 |
| async 中创建 timeline | 同步创建，注册后让 HyperFrames 控制 |
| `gsap.from()` 用于无限循环 | 使用有限时长 |
| SVG 同时使用 svgOrigin 和 transformOrigin | 二选一 |

## 无障碍支持

使用 `gsap.matchMedia()` 响应用户偏好：

```javascript
let mm = gsap.matchMedia();
mm.add(
  {
    isDesktop: "(min-width: 800px)",
    reduceMotion: "(prefers-reduced-motion: reduce)",
  },
  (context) => {
    const { isDesktop, reduceMotion } = context.conditions;
    gsap.to(".box", {
      rotation: isDesktop ? 360 : 180,
      duration: reduceMotion ? 0 : 2  // 减少动画偏好时跳过动画
    });
  }
);
```

## 相关 Skills

GSAP 技能体系包含多个专门 skill：

| Skill | 用途 |
|-------|------|
| **gsap-core** | 核心 API：tween、easing、stagger |
| **gsap-timeline** | Timeline 进阶：嵌套、标签、播放控制 |
| **gsap-scrolltrigger** | 滚动驱动动画（HyperFrames 通常不需要） |
| **gsap-plugins** | 插件：ScrollToPlugin、Flip、Draggable 等 |
| **gsap-utils** | 工具函数：clamp、mapRange、random 等 |
| **gsap-react** | React 集成：useGSAP、cleanup |
| **gsap-performance** | 性能优化 |
| **gsap-frameworks** | Vue、Svelte 等框架集成 |

---

> 💡 **Tip**: 参阅 [gsap.com/docs](https://gsap.com/docs/v3/) 获取完整文档
