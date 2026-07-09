+++
date = '2026-07-09T10:00:00+08:00'
draft = false
title = '拆解 Matt Pocock 的 Agent Skills：设计哲学、工程流与失效模式'
categories = ['AIAgent']
tags = ['agent', 'skills', 'claude-code', 'matt-pocock', 'engineering', 'tdd', 'domain-modeling', 'cursor', 'codex']
summary = '深入拆解 mattpocock/skills（16 万 star）的体系：skill 组织架构、main flow 工作流、Wayfinder 多 session 编排、CONTEXT.md 共享语言、四个 agent 失效模式，以及三个能直接用起来的工作流模板。'
+++

> Matt Pocock（Total TypeScript 作者）的 [mattpocock/skills](https://github.com/mattpocock/skills) 是当前最完整的 Claude Code skills 集合，16 万 star。本文拆解它的组织架构、核心工作流和设计决策，但不是介绍文档——我想回答的是：这 26 条 skills 背后，Pocock 看到了什么工程师困境，又是怎么解决的。文末给了三个你能直接在自己的 agent session 里用起来的工作流模板。

---

## 1. 四个失效模式：问题先行

AI coding agent 已经好到让人依赖，但还没好到能独立交付。Pocock 把常见的失效场景归纳为四条：

**#1 我没得到我想要的** — 你以为 agent 懂了，看到产出才发现它理解偏了。根源是沟通损耗：人的模糊意图没有真的穿透到 agent 的执行层。

**#2 Agent 极其啰嗦** — agent 被丢进陌生代码库，被迫自己摸索术语。结果是 20 个词能说清的它用 200 个——不是模型差，是没有共享语言。一个真实的对比：

```
# BEFORE（无共享语言）
"There's a problem when a lesson inside a section of a course is made
'real' (i.e. given a spot in the file system)"

# AFTER（CONTEXT.md 定义了术语后）
"The materialization cascade is failing"
```

每次对话都在重复前一次的长描述。50 次 session 下来，累积浪费的 token 和注意力已经相当大。

**#3 代码跑不了** — 你对齐也做了，spec 也写了，产出还是残次品。根源是缺少反馈闭环：agent 写代码但不运行代码，不知道自己的代码跑对了没有。

**#4 我们造了一个泥球** — 架构腐化是渐进式的。每次"先这样，下次再修"都在堆积，直到整个代码库让 agent 也无法有效操作——agent 需要清晰的模块边界来定位修改点，泥球让它的 token 消耗暴涨但理解深度暴跌。

这四条驱动了整个 skills 体系的设计。每条 skill 本质上是一个针对特定失效模式的应对机制。

Pocock 的对立面是 GSD、BMAD、Spec-Kit 这类"替你管理流程"的框架。他的论点很明确：框架抢走你的控制权，并在流程中隐藏 bug；skills 是小块的、可组合的、不占有过程的——你始终在线。

---

## 2. 架构：目录组织与 skill 定义

### 2.1 目录组织

```
skills/
├── engineering/     ← 日常代码工作（17 条）
├── productivity/     ← 日常非代码工具（5 条）
├── misc/            ← 保留但不用，不推广
├── personal/        ← 绑定作者的本地设置，不推广
├── in-progress/     ← 草稿，未完工
└── deprecated/      ← 已废弃
```

两条规则防止退化：(a) 只有 `engineering` 和 `productivity` 两个受推广桶中的 skill 才出现在 README、plugin.json 和人可读的文档页中；(b) 每个受推广桶自带 README，按"User-invoked"和"Model-invoked"分组列出。结果是：新 idea 可以先丢进 `in-progress/` 试水，成熟后再晋升——不会出现每条 idea 都直接往 README 里塞的熵增。

### 2.2 SKILL.md：YAML 头 + Markdown 体

```yaml
---
name: wayfinder
description: Plan a huge chunk of work as a shared map
disable-model-invocation: true  # 仅用户触发
---
# skill body (Markdown)
```

没有 JSON schema、没有参数声明、没有 runtime 绑定。skill 的内容就是自然语言指令——它写给 agent 看，不是写给程序解析。隐含假设是模型已经足够理解 Markdown 中的英文流程描述。

`disable-model-invocation: true` 是关键开关。标记为 true 的 skill 只能通过 `/skill-name` 显式调用——你告诉模型该做什么。标记为 false 的 skill 模型可以按上下文自由调用——你信任模型的判断。

### 2.3 CONTEXT.md：共享语言

体系里最巧妙的设计不在任何 skill 内部。`/grill-with-docs` 执行时，会把发现的领域术语写进一个 `CONTEXT.md`。这个文件的作用是建立领域专家和开发者之间的共享语言——来自 Eric Evans 的《Domain-Driven Design》。Pocock 的洞察是：agent 就是那个需要学习领域语言的"新开发者"。

下面是一个真实片段，来自 pocock 自己的 `course-video-manager` 仓库：

```markdown
# Matt Pocock Skills

## Language

**Materialization cascade**: The process of creating physical file-system entries for
virtual course entities. Triggers when a lesson transitions from "draft" to "real".
_Avoid_: "making lessons real in the file system", "lesson-to-file sync"

**Triage role**: A canonical state-machine label applied to an **Issue** during triage
(e.g. `needs-triage`, `ready-for-afk`). Each role maps to a real label string.

**Issue tracker**: The tool that hosts a repo's issues — GitHub Issues, Linear, a local
`.scratch/` markdown convention, or similar.
_Avoid_: backlog manager, backlog backend

## Flagged ambiguities
- "backlog" was previously used to mean both the tool hosting issues and the body of
  work inside it — resolved: the tool is the **Issue tracker**; "backlog" is no longer
  used as a domain term.
```

这个文件不是 README——它是词汇表。词汇表不回答"这是什么项目"，回答的是"在这个项目里，每个词到底指什么"。它对 agent 的实际价值是：当 agent 看到 `materialization cascade` 时，它知道这是个已定义的术语，不需要花 500 token 重新描述。50 次 session 下来，这个单一段落节省的 token 和注意力累积起来相当可观。

`/domain-modeling` 是对这条原则的执行：接收模糊术语，挑战歧义，找到精确替代，必要时用 ADR（Architecture Decision Record）记录下来。

---

## 3. Main Flow：idea → ship

这是 skills 体系最核心的编排。它没有发明新方法——它做的是把需求评审、spec、任务分解、TDD、code review 这些被验证过的工程实践，翻译成 agent 可执行的流程指令。

### 3.1 Step 1: `/grill-with-docs`（需求评审的 agent 版本）

有代码库时的入口。核心是一轮 relentless interview：agent 向你提一连串尖锐问题，把你的模糊想法压缩成可执行的精确理解。一个真实的 grilling 对话大概长这样：

```
用户: 我想加一个"课程副本"功能，让讲师可以复制已有课程。
Agent: 复制到什么程度？章节和课程结构要复制，视频文件要复制吗？
用户: 结构全部复制，视频不复制——新课程应该是空壳。
Agent: 复制的课程和原课程是什么关系？原课程更新后，复制的要不要跟着更新？
用户: 不跟——完全独立的两套数据。
Agent: 那讲师信息呢？复制的课程算谁创建的？
用户: 还是原作者。但新开一个"copied by"字段记录这次复制操作的人。
Agent: 好的——我创建了一条 ADR 记录这个设计决策，并在 CONTEXT.md 里加入了
"lesson duplicate"、"copied by" 两个新术语。
```

这个过程同时干两件事：(a) 对齐意图；(b) 把新术语和决策写进 CONTEXT.md。每次 grilling 之后，共享语言库大一点，agent 的长期操作能力强一点。没有代码库时走 `/grill-me`——同样的 relentless interview，但不写文件。

### 3.2 Step 2: 分叉判断（需要 prototype 吗？）

Grill 结束后有一个关键判断：所有的设计问题能靠对话解决吗？

不能 → 分叉到 `/prototype`：一个一次性的原型程序，只用来回答一个具体的设计问题（"这个 state model 感觉对吗？""这个 UI 应该长什么样？"）。prototype 只产生答案，不产生生产代码——答案通过 `/handoff` 传回主线，prototype 本身被删除。

### 3.3 Step 3: 分叉判断 #2（单 session 还是多 session？）

多 session → `/to-spec`（把对话线程变成 spec）→ `/to-tickets`（拆成 tracer-bullet tickets，每个标出阻塞依赖）。依赖边由实际的 issue tracker 实现（GitHub Issues / Linear / local markdown），形成有向无环图。任何未被阻塞的 ticket 可以在新 session 中用 `/implement` 执行。

单 session → 直接 `/implement`，在当前窗口完成。

### 3.4 Step 4: `/implement` + `/tdd`

`/implement` 是执行引擎，内部驱动 `/tdd`。`/tdd` 不只是"先写测试"，而是三层约束：

**seam 确认**：写任何测试前，先写下测试将在哪个 public boundary 上进行，和用户确认。不允许对未经确认的 seam 写测试——避免 agent 自己决定"在哪测"。

**反模式表**：三类典型错误被直接写进了 skill body：
- 实现耦合测试：mock 了内部协作者，测试对重构敏感
- 同义反复测试：断言重算了被测代码自身的逻辑，永远不会 fail
- 水平切片：批量写测试然后批量实现，丢失了每个循环的反馈信号

**垂直切片**：一个 seam → 一个 test → 一个最小实现 → 重复。每次循环是 tracer bullet，响应上一循环的发现。

### 3.5 上下文卫生

被忽略但最关键的一条：Steps 1-3 在同一个不打断的上下文中完成——不要在 grilling、spec、tickets 之间 compact 或清空上下文。到 `/implement` 时每个 ticket 起新的干净 session。

背后的约束是 Pocock 提出的 "smart zone"：当前最好的模型在约 120K token 内推理最锐利。超过这个窗口，即使模型仍能看到所有内容，推理质量已在不可见地退化。

---

## 4. 汇入流：从非零状态进入主线

Main flow 假设你从一个明确的 idea 开始。现实通常不是。

### 4.1 `/triage` — 从积压中杀出来

只处理不是你创建的 issues——外部的 bug 报告、功能请求。内部产生的 tickets 不需要 triage，它们出生时已经是 agent-ready。Triage 把每个 issue 移过一组标准化的 triage role（`needs-triage` → `ready-for-implementation`），产出 `/implement` 可以直接消费的结果。

### 4.2 `/diagnosing-bugs` — 硬 bug

对一瞥无法诊断的 bug，强制一条原则：拒绝猜测，直到拥有一个紧反馈闭环。反馈闭环是一条命令——一条能重现 bug 但现在还没反馈的命令。找到它之后才进入 fix + regression test。复盘环节是关键：如果诊断中发现"这个 bug 这么难定位是因为没有好的测试接缝"，转交到 `/improve-codebase-architecture`——把单次修复升级为测试基础设施的改进。

### 4.3 `/wayfinder` — 大雾中的导航

这是体系里最复杂的一条。面对太大的、看不清路的目标，wayfinder 不直接做，而是先绘制地图。

地图是一个 issue，label `wayfinder:map`。它包含四个部分：

```
## Destination
为 course-video-manager 新增多语言字幕功能——允许讲师生成和使用
多语言字幕，字幕文件与课程视频独立存储。

## Decisions so far
- [字幕存储方案](link) — 独立 bucket，不嵌入视频容器。理由：实时流中
  切换字幕需要低延迟随机访问，嵌入方案不符合
- [多语言索引](link) — 采用 language_code + variant 二元标识

## Not yet specified
- 字幕格式兼容性——SRT vs WebVTT，取决于视频播放器选型结果
- 在线字幕编辑器——是否购买第三方服务 or 自建

## Out of scope
- 实时 AI 字幕生成（属于 tts-pipeline 仓库的范畴，不在此地图内）
```

每个 ticket 是一个 child issue，标签 `wayfinder:<type>`（`research` / `prototype` / `grilling` / `task`），指明解决方式。worker session 严格按照 **每个 session 只解决一个 ticket** 的约束推进。这不是保守——agent 的单次推理窗口有限，把多个决策塞进一个窗口，后面的决策会带着前面决策的残余偏见。

---

## 5. 三条边界：这个体系不做什么

拆解必须包含判断——没有判断的拆解是说明书。

### 5.1 ship 的缺位

Main flow 叫 "idea → ship"，但它在 `/implement` + `/code-review` 就停了。部署、灰度发布、监控告警、回滚——真实的生产交付链路完全没有覆盖。这可能是刻意的（每个团队的 CI/CD 环境差异太大），但缺少指向它们的路径意味着 agent 在"写完代码"和"代码在生产环境运行"之间没有任何指引。

### 5.2 CONTEXT.md 的腐化风险

CONTEXT.md 依赖人的持续维护。人忘记更新、写错了、或一个 PR 改了架构但没更新词汇表——这些都不在 skills 体系的检测范围内。Pocock 的设计假设是 `/grill-with-docs` 的使用频率足够高，自然会保持 CONTEXT.md 新鲜。但如果团队有多个开发者、多个 agent session 在并发运行，没有冲突检测、没有版本对比、没有过期标记，腐化只是时间问题。

### 5.3 跨模型兼容性

这些 skills 的指令密度和推理要求，隐含锁定了模型门槛。Pocock 的 120K token "smart zone" 是 Sonnet 4.6 / Opus 4.8 级别的假设。`/wayfinder` 需要在一个推理周期里理解地图、选择下一个 ticket、判断雾区边界——这些任务在 Haiku 级别模型上的表现完全不可预测。

### 5.4 数量天花板

26 条公开 skill 已经多到需要一个 `/ask-matt` 路由器来导航。目前的路由设计是把所有 skills 的关系硬写在 `ask-matt` 的 SKILL.md 里，每加一条要改路由器——没有自动发现机制，没有依赖图。`in-progress/` 桶里还有 6 条待晋升，天花板只是时间问题。

---

## 6. 与 AI 工程界其他方法的对比

| | Pocock skills | Superpowers | GSD / Spec-Kit |
|---|---|---|---|
| 粒度 | 单 skill = 单过程步骤 | 单 skill = 单编排关卡 | 单框架 = 完整工作流 |
| 用户控制 | 全程在位 | 分阶段在位（写计划→审执行） | 框架接管大量决策 |
| 核心机制 | 人驱动：你告诉 agent 该走哪条 flow | 关卡驱动：agent 在固定关卡（plan→implement→verify）间流转 | 脚本驱动：框架预设的线性路径 |
| 可组合性 | 高 | 中（关卡间可跳跃但不能重排） | 低 |
| 学习曲线 | 中（需要理解 main flow 的编排） | 中（需要理解关卡语义） | 低（遵循预设脚本） |
| bug 定位 | 容易（per-skill isolation） | 中等（关卡边界清晰但关卡内状态隐蔽） | 困难（框架内状态隐蔽） |
| 前置要求 | 需要人能做出高质量的设计判断 | 需要在写 plan 时投入较多 | 跟随脚本即可 |

三种方法代表了三种 agent 编排水位的选择。Pocock 给的是最小粒度的基元，你自己组合流程。Superpowers 给的是固定关卡——plan → implement → verify——你可以在关卡间跳跃但不能重新定义关卡本身。GSD 和 Spec-Kit 给的是预设的端到端流水线，你跟着走就行，走岔了不好修。

---

## 7. 三个你能直接用起来的工作流

不是"理论上应该这样"，而是 Pocock 体系里最成熟、你能在自己的 agent session 里立刻用起来的三条流程。

### 7.1 从模糊想法到拆好的 tickets（30-60 分钟）

适合任何代码库中的新功能开发。

```
Session 1: 对齐
/grill-with-docs     ← agent 面试你，把 idea 精确化
/domain-modeling     ← 命名冲突时介入，统一术语
/to-spec             ← 产出可审查的 spec 文档
/to-tickets          ← 按 tracer-bullet 拆成 tickets，标注依赖关系
                     ← 不要在同一个 session 里开始 implement
```

产出：(a) 一个 spec 文件；(b) N 个 tickets，每个声明了阻塞依赖；(c) 更新过的 CONTEXT.md。

### 7.2 接一个 ticket 的完整闭环（20-40 分钟/ticket）

适合处理自己拆出来的 tickets 或 triage 产出的 issues。

```
Session 2-N: 实现（每个 ticket 新开 session）
/implement           ← agent 驱动：
                     1. 先确认测试 seam
                     2. /tdd: 红 → 绿 → 重构
                     3. /code-review: Standards + Spec 两条轴审查 diff
                     4. 提交，关闭 ticket
                     ← 不要在同一个 session 里 implement 两个 tickets
```

每个 session 只做一件事的纪律，来自一个反直觉的事实：即使上下文窗口大到能装下三个 ticket，每多做一个，前一个 ticket 的上下文残余就会渗透进后面的推理中。

### 7.3 清积压：入库 → 分级 → 分发（单 session）

适合 backlog 堆积的周末下午。

```
/triage          ← agent 对每个 issue：
                  1. 分配 triage role 标签（needs-triage → ... → ready-for-implementation）
                  2. 只有 ready-for-implementation 的 issue 才能被 /implement 消费
                  3. 超出能力范围的 issue 标 needs-spec 或 wont-fix
```

Pocock 特别强调的一条规则：不要 triage 你自己用 `/to-tickets` 产出的 tickets——它们出生时已经是 agent-ready，再跑一遍 triage 等于把成品重新分类。

---

## 8. Takeaways

Pocock 的这套体系，本质是把有经验的工程师怎么思考和协作，翻译成了 agent 可执行的流程指令。它不是 AI 时代的"新方法"——grilling 就是需求评审、spec 就是技术规格、ticket 就是任务分解、TDD 就是反馈闭环、code review 就是质量门、handoff 就是交接文档。只是从人与人之间的协作，换成了人与 agent 之间的协作。

如果你只能从这里带走三件事：

1. **每次 idea 都先用 `/grill-with-docs` 对齐**。没有对齐的执行只是浪费一个 session。5 分钟的 relentless interview 能节省 50 分钟的错误方向。
2. **维护一个 CONTEXT.md**。不需要从零开始——下一次 grilling 中，把 agent 发现的歧义写进去就行。一个只有 10 个术语的词汇表，在 50 次 session 中的 token 节省远超你想象。
3. **每个 implement 只做一个 ticket**。不要贪多——多个决策在一个窗口里相互污染，出来的代码比一次做一个更差，bug 也更难追。

这套体系有一个硬前提：模型质量。在 GPT-4 级别以下是跑不起来的——grilling 需要模型能提有意义的问题，domain-modeling 需要模型能感知命名歧义，TDD 需要模型能从 seam 确认中提取测试边界。120K token 的 smart zone 假设本身就锁定了最低门槛。

但即使你现在用的不是最顶级的模型，这套方法论仍然是可移植的。一个稍弱的模型执行 grilling 时可能问题提得不够锐利、TDD 时可能 seam 判断更粗放——但你仍然得到了一个"不做泥球"的流程骨架。方法论不绑定模型，只是在更好的模型上表现更精准。
