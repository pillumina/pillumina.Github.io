+++
date = '2026-07-09T10:00:00+08:00'
draft = false
title = '拆解 Matt Pocock 的 Agent Skills：设计哲学、工程流与反模式'
categories = ['AIAgent']
tags = ['agent', 'skills', 'claude-code', 'matt-pocock', 'engineering', 'tdd', 'domain-modeling']
summary = '深入拆解 mattpocock/skills（16 万 star）的体系：skill 组织架构、main flow 工作流、Wayfinder 多 session 编排、CONTEXT.md 共享语言、grill-with-docs 面试对齐、TDD 红线、及四个反模式解决方案。'
+++

> Matt Pocock（Total TypeScript 作者）的 [mattpocock/skills](https://github.com/mattpocock/skills) 是当前最系统的 Claude Code skills 集合，16 万 star。本文从工程视角拆解其组织架构、核心工作流、设计决策，以及它试图修复的四个 agent 失效模式。

---

## 1. 背景：问题先行

AI coding agent 已经足够好到让人产生依赖，但远没好到能独立交付。Pocock 把常见失效模式归纳为四条：

**#1 我没有得到我想要的** — 你以为 agent 懂了，看到产出才发现它理解偏了。根源是沟通鸿沟：人的模糊意图无法穿透到 agent 的执行层。

**#2 Agent 极其啰嗦** — agent 被丢进一个代码库，被迫自己摸索术语。结果是 20 个词能说清的它用 200 个——不是模型差，是没有共享语言。

**#3 代码跑不了** — 你和对齐都做了，产出还是残次品。根源是 agent 没有反馈闭环：它写代码，但不知道代码能不能跑、跑对了没有。

**#4 我们造了一个泥球** — 架构腐化是渐进式的。每次"先这样，下次再修"都在堆积，直到整个代码库让 agent 也无法有效操作。

这四个问题驱动了整个 skills 体系的设计。每条 skill 本质上是一个应对特定失效模式的故障恢复程序。

Pocock 的直接对立面是 GSD、BMAD、Spec-Kit 这类"替你管理流程"的框架。他的论点很明确：框架抢走了你的控制权，并在流程中隐藏 bug；skills 是小块的、可组合的、不占有过程的——你始终在线。

---

## 2. 架构：六桶一插件

### 2.1 目录结构

```
skills/
├── engineering/     ← 日常代码工作（21 条）
├── productivity/     ← 日常非代码工具（5 条）
├── misc/            ← 保留但很少用，不推广
├── personal/        ← 绑定作者的本地设置，不推广
├── in-progress/     ← 草稿，未完工
└── deprecated/      ← 已废弃
```

两个受推广的桶（`engineering` 和 `productivity`）中的每条 skill 必须在 README 中有引用、在 `.claude-plugin/plugin.json` 中有注册、在 `docs/` 下有人类可读的文档页。其余四个桶的 skill 不出现在上述任何位置。这形成了一个两层的发布通道：26 条公开 skill 和数量不等的内部/实验 skill，避免了"每条 idea 都往 README 里塞"的熵增。

### 2.2 SKILL.md 格式：YAML 头 + Markdown 体

每条 skill 是一个包含 `SKILL.md` 的目录。格式极其简洁：

```yaml
---
name: wayfinder
description: Plan a huge chunk of work as a shared map
disable-model-invocation: true  # 仅用户触发
---
# skill body (Markdown)
```

没有 JSON schema、没有参数声明、没有 runtime 绑定。skill 的内容就是*自然语言指令*——它写给 agent 看，不是写给程序解析。这种设计的隐含假设是：**模型已经足够理解 Markdown 中的英文流程描述，不需要额外的结构化元数据**。

`disable-model-invocation: true` 是关键开关。标记为 `true` 的 skill 只能被用户通过 `/skill-name` 显式调用，模型不能自己决定调用。标记为 `false` 的 skill 模型可以按上下文自由调用。这个区分反映了两条不同的 agent 交互模式：你让模型替你判断何时做什么（model-invoked），和你告诉模型现在做什么（user-invoked）。

### 2.3 CONTEXT.md：共享语言

pocock 体系最巧妙的设计不在任何 skill 内部，而在一个概念：**共享语言**。每个使用这些 skills 的代码库需要一个 `CONTEXT.md`，里面维护项目特定术语及其精确定义。

```
# CONTEXT.md (示例)
## Language
**Materialization cascade**: The process of creating physical file-system entries for
virtual course entities. Triggers when a lesson transitions from "draft" to "real".
_Avoid_: "making lessons real in the file system", "lesson-to-file sync"
```

效果是 agent 从前 20 个词才能描述一个概念，变成用 2-3 个精确术语。这个做法直接来自《Domain-Driven Design》——建立领域专家和开发者之间的共享语言。Pocock 的转折在于：**agent 本身就是那个需要学习领域语言的"新开发者"**。

`/domain-modeling` skill 是对这条原则的执行面：它接收一个模糊术语，挑战其歧义，找到一个精确替代，必要时通过 ADR（Architecture Decision Record）记录下来。

---

## 3. Main Flow：idea → ship

这是 skills 体系最核心的编排：一条主线 + 两个支线汇入点。它不是"用某个框架管理流程"，而是把现有被验证过的工程实践——面试（grilling）、spec、ticket、TDD、code review——翻译成 agent 可执行的 natural language procedure。

### 3.1 Step 1: `/grill-with-docs`

**有代码库时的入口**。核心是 relentless interview：agent 向你提一系列尖锐问题，把你的模糊 idea 压缩成精确理解。

这个过程同时干两件事：(a) 让 agent 和人对齐意图；(b) 把新发现的术语和决策写进 `CONTEXT.md`。每次 grilling session 之后，共享语言库变大一点，agent 的长期操作能力变强一点。

没有代码库的场景走 `/grill-me`——同样的 relentless interview，但不写文件。底层都是 `/grilling` 这个原始机制，`grill-with-docs` 多加了一层"留纸面痕迹"的持久层。

### 3.2 Step 2: 分叉判断

Grill 结束后，问自己一个问题：**所有的设计问题能纯粹靠对话解决吗？**

- **是** → 跳过，直接进入 spec 判断。
- **否** → 分叉到 `/prototype`：一个 purpose-built throwaway 程序，用来回答一个具体的设计问题（"这个 state model 感觉对吗？""这个 UI 应该长什么样？"）。prototype 只产生答案和截图，不产生生产代码。答案通过 `/handoff` 传回主线，prototype 本身被删除。

这种 "prototype 作为设计问题的回答而非产品代码的前身" 的定位，精准区分了"探索开销"和"交付开销"。

### 3.3 Step 3: 分叉判断 #2

**这是一个多 session 构建吗？**

- **是** → `/to-spec`（把对话线程变成 spec），然后 `/to-tickets`（把 spec 拆成 tracer-bullet tickets，每个声明其 blocking edges）。ticket 的依赖边由实际的 issue tracker 实现（GitHub Issues / Linear / local markdown），产生一个 DAG。任何未被阻塞的 ticket 可以被抓起，用 `/implement` 在新 session 中执行。

- **否** → 直接 `/implement`，在当前上下文窗口完成。

### 3.4 Step 4: `/implement` + `/tdd`

`/implement` 是执行引擎。它内部驱动 `/tdd`（red-green-refactor loop），每条 ticket 完成后自动运行 `/code-review`（Standards + Spec 两个维度的 diff 审查）。

`/tdd` skill 本身值得单独展开。它不只告诉你"先写测试"，而是定义了一套严格的约束：

- **Seam 确认**：写任何测试前，先写下你在哪些 seam 上测试，和用户确认。不允许对未经确认的 seam 写测试。
- **反模式列表**：实现耦合测试（mock 内部协作者）、同义反复测试（assertion 重算了被测代码）、水平切片（批量测试然后批量实现）——这三个反模式都指向一个根因：测试丢失了"规范文档"的角色，变成了"实现的重复品"。
- **垂直切片**：一次一个 seam → 一个 test → 一个最小实现 → 重复。每次循环都是 tracer bullet，响应上一循环的发现。

### 3.5 Context 卫生

Pocock 强调的一个容易被忽略的点：**Steps 1-3 在同一个不打断的上下文窗口中完成**——不要在 grilling/spec/tickets 三者之间 compact 或清空。到 `/implement` 时每个 ticket 起一个新的干净 session。

背后的约束是 "smart zone" 概念：当前 SOTA 模型约 120K token 内推理最锐利。超过这个窗口，即使模型仍能看到所有内容，推理质量已在不可见地退化。所以在 `/to-tickets` 之前不 compact，在 `/implement` 时必须清上下文——两个约束来自同一个上限。

---

## 4. On-ramps：从非零状态进入主线

Main flow 假设你从一个 idea 开始。现实通常是另一种状态：你有一堆 bug、一个打不开的局面、一个看不清的项目。

### 4.1 `/triage` — 从积压中杀出

triage 只处理**不是你创建的** issues——来自外部的 bug 报告、功能请求、任何原始输入的。内部产生的 tickets 不需要 triage，因为它们出生时已是 agent-ready。

triage 流程把每个 issue 移过一组 canonical triage roles（`needs-triage` → `ready-for-implementation` → ...），产生 `/implement` 可以直接执行的工作单元。

### 4.2 `/diagnosing-bugs` — 硬 bug

对那些一瞥无法诊断的 bug——间歇性 flake、回归问题、跨版本引入的性能退化——`/diagnosing-bugs` 强制一条原则：**拒绝猜测，直到拥有一个紧反馈闭环**。

反馈闭环就是一条命令：用来验证 bug 存在但现在还没反馈的命令。找到这条命令之后，才进入 fix + regression test。Pocock 给这个流程加了 post-mortem 环节：如果诊断中发现"这个 bug 这么难定位是因为没做好测试接缝"，手把手转交到 `/improve-codebase-architecture`，把 bug 修复从"修症状"变成"修测试基础设施"。

### 4.3 `/wayfinder` — 大雾中的导航

这是整个体系里最复杂的一条 skill。面对一个太大的、看不清路的目标（绿野项目、巨型 feature 重构），wayfinder 不直接做，而是**绘制地图**。

地图是一个 issue，label `wayfinder:map`。它包含：
- **Destination**：这个任务地图的目标是什么——一句话，每个 session 重读一次以保持方向
- **Decisions so far**：已关闭的 ticket 列表，每个一条要点 + 链接到 ticket body
- **Not yet specified**：雾区——你能看到前面有什么但还无法精确定义成 ticket 的东西
- **Out of scope**：被有意排除在目标之外的工作

每个 ticket 是一个 child issue，携带 `wayfinder:<type>` 标签（`research` / `prototype` / `grilling` / `task`），表示解决方式。worker session 按"取下一个未被阻塞的 frontier ticket → claim → 解决 → 往地图上追加一条 Decisions-so-far → 清空对雾区的贡献"循环执行。

关键约束：**每个 session 只解决一个 ticket**。这不是保守，是认清 agent 的单次推理窗口有限——把多个决策塞进一个窗口，后面的决策会带着前面决策的残余偏见。

---

## 5. 设计决策中的几个亮点

### 5.1 `ask-matt` 路由器

26 条公开 skill，人不方便记住。`/ask-matt` 作为路由器，用自然语言模拟"问一个懂这套体系的人该走哪条路"。这不是传统的 help 命令——它是一个带有体系知识的对话 agent，接收你的起始状态描述，输出应该触发哪条 skill 或哪个 flow。

### 5.2 `/handoff` — context window 间的桥梁

当窗口满了或需要分叉到不同方向时，`/handoff` 把当前对话压缩成一个 Markdown 文件。下个 session 打开时引用该文件。它与 Claude Code 内置的 `/compact` 的区别在于：`/handoff` 是**分叉**（你打开新窗口，旧窗口的 verbatim 历史保留），`/compact` 是**继续**（同一个窗口，旧历史被摘要化后丢弃）。

### 5.3 `/codebase-design` 语汇层

一套 model-invoked 的深层模块化语汇——module、interface、depth、seam、adapter、leverage、locality。这些概念不出现在任何 user-facing skill 的 UI 中，但被 `/tdd` 和 `/improve-codebase-architecture` 在背后引用。Pocock 的取舍值得注意：把设计语汇放到 model-invoked 层，agent 可以在需要时拉取，不会在每次对话中塞进 user prompt 造成 token 膨胀。

---

## 6. 与 AI 工程界其他方法的对比

| | Pocock skills | GSD / Spec-Kit | BMAD |
|---|---|---|---|
| 粒度 | 单 skill = 单过程步骤 | 单框架 = 一个完整工作流 | 单工作流 = agentic methodology |
| 用户控制 | 全程在位（grill/prototype/ticket 都需要人确认） | 框架接管大量决策 | 框架接管 |
| 可组合性 | 高（skills 之间通过 handoff 松耦合） | 低（框架预设路径） | 低 |
| 学习曲线 | 中（需要理解 main flow 的编排逻辑） | 低（遵循预设脚本） | 中 |
| bug 定位 | 容易（per-skill isolation） | 困难（框架内的状态隐蔽） | 中等 |

这个对比本身暴露了 Pocock 哲学的核心张力：你拿回控制权的代价是更多手工操作——更少的自动化，更多的人工确认点。如果你愿意接受这个交换，你的 bug 可追踪性大幅提升；如果你不愿意，框架替你决策，代价是框架出错时你看不见。

---

## 7. 结语

Pocock 的 skills 体系本质上是把 **"有经验的工程师怎么思考和协作"** 翻译成了 agent 可执行的 natural language procedure。它不是 AI 时代的"新方法"，而是工程实践（grilling = 需求评审、spec = 技术规格、ticket = 任务分解、TDD = 反馈闭环、code review = 质量门、handoff = 交接文档）在 agent 媒介上的重写。

这个体系的隐含前提是：**模型质量**。这套东西在 GPT-4 级别以下是跑不起来的——grilling 需要模型能够提出有意义的问题，domain-modeling 需要模型能感知命名歧义，tdd 需要模型能从 seam 确认中提取测试边界。Pocock 的 120K token "smart zone" 假设，本质上把 target 锁定在了 Sonnet 4.6 / Opus 4.8 这个级别的模型上。

从 repo 的结构和 `CONTEXT.md` 的存在本身可以看出，这不仅仅是一堆 skills 的集合——它是一个关于 agent 工程方法论的主张，包装成了一个 GitHub 仓库。
