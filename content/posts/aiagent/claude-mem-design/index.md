+++
date = '2026-06-23T10:00:00+08:00'
draft = false
title = '从 claude-mem 看 Agent Memory 的设计哲学'
categories = ['AIAgent']
tags = ['agent', 'memory', 'claude-mem', 'claude-code', 'llm', 'architecture']
summary = '一篇关于 agent 长期记忆的长文。前半部分拆 claude-mem 13.8.0 的实现，后半部分讨论 agent memory 设计里那些绕不开的本质问题——write-time 压缩、provider 抽象、LoMoDe 击败、六个未解难题。'
+++


> 一篇关于 agent 长期记忆的长文。前半部分拆 claude-mem 13.8.0 的实现，后半部分讨论 agent memory 设计里那些绕不开的本质问题。
>
> 写给：天天跟 LLM agent 打交道、对"它为什么这么健忘"有切肤之痛、又不愿意盲信任何 memory plugin 的工程师。

---

## 1. 引子：另一个 memory 插件，凭什么是它

用过 Claude Code、Cursor、Cline 这类 coding agent 的人，多半都遇过同一个问题——**跨 session 失忆**。

下午花了几个小时让 agent 理解一个棘手的 checkpoint 保存 bug，一起翻了几十层调用栈，最终定位到 mcore/HF weight 转换脚本里某个 sharded state_dict 的错误。第二天打开新对话窗口继续，agent 一脸茫然地看着 checkpoint 路径里的报错，要重新建立上下文。中间那个关键的 epiphany（"啊原来这条路径绕过了 validation"）——当时帮你跳过三条死胡同的判断——只剩结论，过程已经丢失。

这就是 agent 的"记忆"问题。它不致命（单次任务还能完成），但持续消耗用户的注意力——重新解释、重新探索、重新犯已经犯过的错。

社区对此有不少尝试：LangChain 出过 Memory 抽象，MemGPT 把 OS 虚拟内存隐喻搬过来，Letta 把它产品化，OpenAI 给 ChatGPT 加了原生 memory，Cursor 有 project memory，Claude Code 生态里 memory plugin 少说十几个。

但 memory 这件事远比"加个数据库存历史"复杂。设计不好的 memory 系统会反噬：

- 注入无关上下文，挤占 attention budget
- 把低质量"记忆"反复塞给模型，污染推理
- 在错误的时刻召回错误的内容，让 agent 在错误方向上越走越远
- 静默把敏感信息写进共享存储

claude-mem（[thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)）是这堆方案里值得花时间拆解的一个。它**生产级**（截至本文 13.8.0，跨 70+ session 持续运行）、**开源**、**踩过真实的工程坑**（hook 兼容性、provider auth、reasoning 模型适配），而且它的设计选择大多可以被讨论——每一个"为什么这样做"背后都有清晰答案，有些是聪明的取舍，有些是工程妥协。

本文前半部分拆 claude-mem 13.8.0 的实现，后半部分从实现抽身出来讨论 agent memory 设计的本质问题。这些问题不是 claude-mem 独有的，而是任何想做长期记忆的 agent 系统都要回答的。

中间会停在一个让社区震动的反直觉发现上：**最朴素的"把对话直接存进文件"方案，在标准 benchmark 上打败了精心设计的 memory 系统**。这个发现逼着重新审视 claude-mem 的整个设计哲学——包括 write-time 压缩这条核心路线到底是不是对的。

---

## 2. 一眼看穿：claude-mem 是什么

一句话：**claude-mem 是一个为 Claude Code 提供跨 session 长期记忆的插件，通过生命周期 hook 捕获每次对话的"观察"，压缩成结构化的 observation 存到本地 SQLite + Chroma，下次 session 启动时把相关的 observation 注入回上下文。**

几个事实给你建立量感（来自我本机 13.8.0 的真实数据）：

- 跨 70+ 个 Claude Code session，存了 1300+ 条 observation、420+ 条 session summary、850+ 条 user prompt
- 后台 worker 进程长驻（系统级，不随 Claude Code 关闭而死）
- 每次工具调用触发一次异步 AI 压缩（用我自己接入的 DeepSeek-V4-Flash）
- 完整代码 ~4MB minified JS（worker + MCP server + viewer UI）
- 单文件 SQLite 数据库 + 本地 Chroma 向量库，**所有数据本地**

它在生态里的位置很明确：**Claude Code plugin**。这意味着它依赖 Claude Code 的 hook 系统、Claude Code 的 plugin enable 机制、Claude Code 的 MCP 协议。这套依赖是它强大的来源（不需要自己造生命周期管理），也是它脆弱的来源（Claude Code 任何 API 变化都可能让它失效）。

OK，量感有了。开始拆。

---

## 3. 架构总览：四角色分工

claude-mem 不是单一进程，是**四个角色协同**：

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Code (你对话的主进程)                                 │
│                                                              │
│  ┌─────────────────────┐      ┌─────────────────────────┐   │
│  │ Lifecycle Hooks     │      │ MCP Client (per session)│   │
│  │ 6 个事件回调         │      │  调用 mcp-search 工具   │   │
│  └──────────┬──────────┘      └────────────┬────────────┘   │
└─────────────┼──────────────────────────────┼────────────────┘
              │ bash one-liner               │ MCP stdio
              ▼                              ▼
   ┌──────────────────────┐       ┌────────────────────────┐
   │  Worker daemon       │       │  MCP server            │
   │  bun × worker-       │       │  node × mcp-server.cjs │
   │  service.cjs         │       │  (per Claude Code      │
   │  PID xxxxx :37777    │       │   session 各起一个)    │
   │  跨 session 长驻     │       └──────────┬─────────────┘
   │                      │                  │
   │  HTTP API /api/...   │                  │
   │  Hook dispatcher     │                  │
   │  Async job queue     │                  │
   │  AI generation       │                  │
   └──────────┬───────────┘                  │
              │                              │
              ├──── SQLite (~/.claude-mem/claude-mem.db)
              │      observations / sdk_sessions /
              │      session_summaries / user_prompts
              │
              ├──── ChromaDB (~/.claude-mem/chroma/)
              │      collection: cm__claude-mem
              │
              ├──── AI Provider (Claude/OpenRouter/Gemini
              │      /任意 OpenAI 兼容端点)
              │
              └──── Viewer UI (http://127.0.0.1:37777/)
```

四个角色各有分工。理解了它们为什么分开，就理解了 claude-mem 的核心架构思想。

### 为什么不是简单的 MCP server？

最朴素的设计是：写一个 MCP server，提供 `read_memory` / `write_memory` 工具，agent 想用就调——简单、清晰、解耦。

claude-mem 没这么干。它的 MCP server 只负责"读"（search、timeline、observation_context 等查询接口），"写"和"压缩"全在 worker 里。

为什么？因为 **memory 写入是高频且不能阻塞对话的**。每次工具调用都触发一次写入，如果写入要等 AI 压缩完成（5-30 秒），那 Claude Code 的工具调用就要被卡住 5-30 秒——这对交互式体验是灾难。

于是 claude-mem 把写入做成了**异步管道**：

1. PostToolUse hook 同步触发 → worker 收到 webhook
2. worker 原子地往 `pending_messages` 表写一行 + 入队一个 generation job（毫秒级）
3. worker 后台 job runner 慢慢消费队列，调 AI 压缩，写入 observations
4. Claude Code 立即继续，不等 AI 完成

这要求 worker 是**独立长驻进程**，不能依附于某个 Claude Code session——session 一关，后台任务就死了。所以 worker 必须是 system-level daemon，跟 Claude Code 解耦。这个决策带来好处（异步、跨 session 共享、可控 retry），也带来麻烦（启动 / 升级 / 状态管理的复杂度）。

### 为什么 hooks 是 bash one-liner？

claude-mem 的 hooks.json 里每个 hook 命令长这样（截短版）：

```bash
node "$_P/scripts/bun-runner.js" "$_P/scripts/worker-service.cjs" hook claude-code observation
```

前面还有一大段路径解析逻辑——`_P` 变量按 cache → marketplace 顺序找最新版本的 plugin 目录。一个 hook 配置就 600+ 字符。

为什么要这样？因为 **Claude Code 的 plugin 安装路径不稳定**：

- `claude plugin update` 后版本号会变（`cache/thedotmack/claude-mem/13.4.0/` → `13.8.0/`）
- 不同用户可能在 cache 或 marketplace 路径下
- `CLAUDE_PLUGIN_ROOT` 环境变量有时设有时不设

hook 命令必须在 runtime 自己解析路径——这是个工程妥协，可读性换鲁棒性。代价是每次 hook 触发要跑一遍 bash + node + bun-runner。但因为 worker 长驻，bun-runner 实际只发一个 HTTP 请求给 worker，几毫秒就返回。

---

## 4. Hook 生命周期：把记忆织进对话

claude-mem 注册了 6 个 hook 事件，覆盖 Claude Code session 的整个生命周期。这张表是 claude-mem 工作机制的核心：

| Hook 事件 | matcher | worker 子命令 | 作用 |
|---|---|---|---|
| `Setup` | `*` | `version-check.js` | 安装时校验版本 |
| `SessionStart` | `startup\|clear\|compact` | `worker-service.cjs start` | 拉起 worker daemon（如未跑） |
| `SessionStart` | 同上 | `hook claude-code context` | **注入历史 observation 到新 session** |
| `UserPromptSubmit` | `*` | `hook claude-code session-init` | 登记当前 session 到 sdk_sessions |
| `PreToolUse` | `Read` | `hook claude-code file-context` | 用户读文件时拉相关 observation |
| `PostToolUse` | `*` | `hook claude-code observation` | **任意工具调用后异步生成 observation** |
| `Stop` | `*` | `hook claude-code summarize` | 一轮对话结束生成 session summary |

把它画成时间轴：

```
Session 启动
   │
   ├─→ SessionStart hook #1: 启动 worker（如未跑）
   ├─→ SessionStart hook #2: 查询历史 obs → 注入 system prompt
   │
   ▼
用户发第一条消息
   │
   ├─→ UserPromptSubmit hook: 写 sdk_sessions 行
   │
   ▼
Agent 推理 + 调工具
   │
   ├─→ PreToolUse(Read) hook: 拉文件相关 obs
   ├─→ [工具执行]
   ├─→ PostToolUse hook: 入队 observation 生成任务
   │       (异步: worker 调 AI → 写 observations 表)
   │
   ▼
Agent 回复完成
   │
   ├─→ Stop hook: 生成 session summary
   │
   ▼
下一轮 / Session 结束
```

### SessionStart 注入实际长什么样

光看表知道"会注入"，但看不到注入的形状。下面是我打开新 session 时 claude-mem 实际塞进 system prompt 的内容（截短版，完整版约 18k tokens）：

```
<claude-mem-context>
<session project="research" sessionId="1de2dbe3-...">
<lastSummary>
Session investigated claude-mem 13.8.0 architecture, hooked DeepSeek V4 Flash
as AI provider via OPENROUTER_BASE_URL, patched thinking:disabled for XML parser
compatibility. Worker daemon running on PID 92456 with 1300+ observations.
</lastSummary>

<observations>
<obs type="bugfix">
  DeepSeekV4 checkpoint save failure root cause identified: sharded_state_dict
  keys mismatched between mcore and HF format. Fix: add key remapping step
  before validation. Files: train.py, validation.py, mcore/checkpoint.py
</obs>

<obs type="change">
  Switched claude-mem provider from `claude` to `openrouter` with DeepSeek base
  URL after OAuth auth failed under proxy setup
</obs>

<obs type="discovery">
  claude-mem's OPENROUTER_BASE_URL supports any OpenAI-compatible endpoint —
  not just openrouter.ai. Pointed at api.deepseek.com/v1 successfully.
</obs>

... (47 more observations, ranked by recency × relevance)
</observations>

<sessions>
  - 2026-06-23 14:06 research: investigating claude-mem hooks
  - 2026-05-30 blog: Hugo theme design refinements
  ... (8 more recent sessions)
</sessions>
</claude-mem-context>
```

几个值得注意的细节：

- **结构化标签**：用 XML 风格 `<claude-mem-context>` / `<observations>` / `<obs>` 嵌套，让模型容易区分"这是注入的记忆"和"这是用户当前 prompt"
- **`lastSummary` 优先**：上一轮 session 的总结单独放最前，因为跟当前任务相关性最高
- **obs 按相关性 + 时间排序**：默认 top-50，新 session 大约消耗 15-25k tokens（取决于 obs 长度）
- **sessions 列表**：最近 10 个 session 的简短描述，让 agent 知道"最近在做什么"

这段文本是 Claude Code 看不见的——它进入 system prompt 区域，模型当成"上下文"读，但用户在对话窗口里看不到。这就是"被动注入"的物理形态。

### PostToolUse：异步队列的精妙

PostToolUse hook 是 claude-mem 最关键的设计。它把"工具调用 → AI 压缩 → 写入 DB"做成了一个**原子入队 + 异步消费**的管道：

1. **同步阶段**（毫秒级）：hook 触发后，worker 原子地往 `pending_messages` 表写一行 + 入队一个 generation job，立即返回 `{continue: true}` 让 Claude Code 继续
2. **异步阶段**（5-30 秒）：worker 后台 job runner 拿出 job，调 AI provider 把 raw tool input/output 压缩成结构化 observation，解析 XML 提取 title/subtitle/narrative/facts/concepts/files/type，写入 `observations` 表（触发器同步到 FTS）+ Chroma（向量索引）

整个过程对 Claude Code 透明——agent 该回回，用户该等等。AI 调用失败有 retry，连续失败会跳过，**不让坏数据进库**。

代价是 observation 不是实时可见的——刚做完工具调用立刻 `mcp__mcp-search__search` 是搜不到的，得等 worker 处理完。这是 async 的必然代价。

---

## 5. Observation：write-time 压缩的艺术

理解 observation 的设计，就理解了 claude-mem 最核心的设计哲学。

### 为什么不是 raw transcript？

最朴素的 memory 设计是：**把所有对话存下来**。完整 transcript，需要时召回。

这个方案有几个致命问题：

1. **token 经济性灾难**：一次工具调用的 input/output 加起来可能 10k+ tokens，直接塞进下次 session 的 system prompt，几次工具调用就把上下文挤满
2. **信噪比低**：raw transcript 里 90% 是冗余（"让我读一下这个文件"、报错堆栈、调试输出），只有 10% 是值得记的（"我们发现了 X、决定了 Y、踩了坑 Z"）
3. **召回难**：raw transcript 是按时间序的，跟当前任务的相关性需要语义理解才能判断

claude-mem 的解法是 **write-time 压缩**：每次工具调用不直接存原始数据，而是让 AI 把它压缩成一条 100-300 字的结构化 observation。压缩在写入时发生，读取时就只需要读压缩后的版本。

这是个聪明的取舍：

- **write-time 多花 AI 算力**（每条 observation 一次 AI 调用）
- **换 read-time 巨大的 token 节省**（每次注入只占 200-400 tokens 而不是 10k+）
- **换召回质量**（压缩后的 observation 已经是"语义单元"，向量和 FTS 都更准）

**前提是写入少、读取多**——你写一次 memory，可能在 N 个未来 session 里被读，压缩摊销才划算。这个假设是不是真的成立，第 8.3 节会回来拷问。

### Type 系统：六类观察

claude-mem 限制每条 observation 必须属于六种 type 之一：

```json
"CLAUDE_MEM_CONTEXT_OBSERVATION_TYPES":
  "bugfix,feature,refactor,discovery,decision,change"
```

我库里 1300+ 条的真实分布：

| Type | 数量 | 占比 |
|---|---|---|
| `discovery` | 619 | 47% |
| `change` | 224 | 17% |
| `feature` | 149 | 11% |
| `bugfix` | 98 | 7% |
| `refactor` | 13 | 1% |
| `decision` | 10 | 1% |

这个分布很有意思——`discovery` 占近一半，`decision` 只占 1%。这背后有两个混在一起的因素：一是 agent 工作流确实是探索 >> 决断（跟人类工程师节奏类似）；二是 claude-mem 给 AI 的 prompt 倾向于把模糊的 observation 归为 `discovery`（"读了一个新文件"、"理解了一个新概念"都算）。两种因素这个数据本身区分不开。

`decision` 类型虽然只占 1%，但它可能是**信息密度最高**的一类——"我们否决了方案 X 因为 Y"，这种"否决理由"对未来工作的价值远大于"我们试了方案 X"。

Type 系统的本质是**给记忆加结构**，让你（和模型）可以按 type 过滤、聚合、检索。没有 type，所有记忆都是同质的糊；有了 type，可以做"只召回 decision 类的记忆"这种精准查询。

### XML 结构化输出

claude-mem 要求 AI 把压缩结果输出成 XML：

```xml
<observation>
  <type>bugfix</type>
  <title>DeepSeekV4 checkpoint save failure root cause identified</title>
  <subtitle>sharded_state_dict keys mismatched between mcore and HF format</subtitle>
  <narrative>Traced the checkpoint save failure through save_checkpoint_and_time() 
            → validation.py → _validate_sharding_for_*. The root cause is that 
            mcore's sharded state_dict uses _extra_state suffixed keys while the 
            validator expects plain key names. Fix: add a key remapping step 
            before validation.</narrative>
  <facts>
    - save_checkpoint_and_time() in train.py is the entry point
    - validation happens in validation.py::_validate_sharding_for_*
    - mcore uses _extra_state suffix on certain tensor keys
  </facts>
  <concepts>checkpoint, sharded_state_dict, mcore, validation</concepts>
  <files_read>
    train.py, validation.py, mcore/checkpoint.py
  </files_read>
  <files_modified></files_modified>
</observation>
```

不同 type 的 observation 形态差异挺大——下面是另外两类典型例子，能感受到 type 系统的语义区分：

```xml
<!-- decision 类：信息密度最高，"否决理由"比"试过什么"更值钱 -->
<observation>
  <type>decision</type>
  <title>Decided NOT to use vLLM's PagedAttention for Ascend NPU port</title>
  <subtitle>NPU memory model doesn't expose page-level controls needed by PagedAttention</subtitle>
  <narrative>Investigated porting PagedAttention to CANN. The block-table abstraction 
            requires virtual→physical page mapping control that Ascend's driver 
            doesn't expose. Decided to use static memory planning + custom KV cache 
            manager instead. Revisit if CANN adds page-level APIs.</narrative>
  <concepts>PagedAttention, KV-cache, CANN, Ascend NPU</concepts>
</observation>

<!-- discovery 类：占 47%，多是"理解了一个新概念/读了一个新文件" -->
<observation>
  <type>discovery</type>
  <title>claude-mem's OPENROUTER_BASE_URL supports any OpenAI-compatible endpoint</title>
  <narrative>The env var defaults to openrouter.ai but accepts any OpenAI-compatible 
            base URL — including api.deepseek.com/v1, Ollama, LM Studio. This lets 
            claude-mem act as a generic OpenAI-consumer, not just OpenRouter-specific.</narrative>
  <concepts>claude-mem, OPENROUTER_BASE_URL, provider abstraction</concepts>
</observation>
```

`decision` 类虽然只占库里 1%，但单条信息密度最高——它记的是**否决理由**（"为什么不用 vLLM PagedAttention"），未来重做相关决策时这种"否决记忆"比"我们试了 X"值钱得多。`discovery` 占近一半，但单条价值参差——有的是关键架构发现，有的只是"读了个新文件"。type 系统目前对这种价值差异不敏感。

为什么用 XML 而不是 JSON？两个原因：

1. **LLM 对 XML 的解析错误率低于 JSON**（特别是带嵌套和长字符串时，JSON 的引号转义容易出错）
2. **XML 的标签结构允许 AI"自然地"组织内容**——`<narrative>` 里可以写长段落，`<facts>` 里可以列要点，不需要严格遵守 JSON 的结构约束

但 XML 也不是免费的午餐——**reasoning 模型会破坏 XML 解析**。这是我们后面要专门讲的坑。

### 双存：SQLite FTS + Chroma 向量

claude-mem 同时维护两套索引：

**SQLite + FTS5**：
- 主存储，所有 observation 字段都在 `observations` 表
- `observations_fts` 虚拟表通过触发器自动同步
- 支持 MATCH 全文检索（关键词级）
- 适合"精确召回"——找包含特定关键词的记忆

**Chroma 向量库**：
- 每个 observation 的 title + subtitle + narrative 计算 embedding 存入 Chroma
- 支持语义相似度检索
- 适合"模糊召回"——找语义相关的记忆，即使没有完全匹配的关键词

为什么两套都要？因为它们是互补的：

- FTS 强在**精确性**：你搜 `_validate_sharding_for_`，FTS 能精确找到这条 obs
- 向量强在**泛化性**：你搜"checkpoint 验证失败"，向量能找到即使没用过这个确切词的相关 obs
- 实际查询时，claude-mem 会做 **hybrid retrieval**：FTS + 向量并行召回，按某种权重合并

代价是**双倍的存储 + 双倍的索引维护**。每次写 observation 都要写 SQLite + 算 embedding + 写 Chroma。但因为是异步的，对用户体验无感。

### Schema：观察的字段设计

`observations` 表的关键字段分四组：

- **内容**：`title` / `subtitle` / `narrative` / `facts` / `concepts`（XML 提取出来的结构化内容）
- **关联**：`memory_session_id`（哪个 session 写的）/ `project`（cwd basename）/ `files_read` / `files_modified` / `prompt_number`
- **元数据**：`type`（6 种之一）/ `created_at_epoch` / `generated_by_model` / `content_hash`（去重）
- **预留**：`relevance_count`（被召回次数，本意是做 relevance decay）/ `discovery_tokens` / `metadata`

写入时通过触发器自动同步到 `observations_fts` 虚拟表（FTS5 全文索引），让 `MATCH` 关键词检索和 SQL 查询共用一套数据。

值得注意：`relevance_count` 字段已经预留了"按召回频率衰减"的能力，但 claude-mem 目前只记账不用——这是个**未完成的设计**，后面讨论 forgetting 时会再提到。

---

## 6. Provider 抽象：从小聪明到大坑

claude-mem 不绑定 Claude，它的 AI generation 走一个可配置的 provider 抽象。`CLAUDE_MEM_PROVIDER` 三个选项：

| Provider | 认证方式 | 适用场景 |
|---|---|---|
| `claude`（默认）| Claude Code keychain OAuth | 你用 Claude Code 官方订阅 |
| `openrouter` | API key + 可选 base URL | 你想用其他模型 / 自部署 |
| `gemini` | Gemini API key | 你想用 Gemini |

最有趣的设计是 `CLAUDE_MEM_OPENROUTER_BASE_URL` 这个环境变量。它默认空字符串（走 openrouter.ai），但你可以填**任何 OpenAI 兼容端点**：

```json
{
  "CLAUDE_MEM_PROVIDER": "openrouter",
  "CLAUDE_MEM_OPENROUTER_BASE_URL": "https://api.deepseek.com/v1",
  "CLAUDE_MEM_OPENROUTER_API_KEY": "sk-...",
  "CLAUDE_MEM_OPENROUTER_MODEL": "deepseek-v4-flash"
}
```

这等于把"OpenRouter provider"重命名为"任意 OpenAI 兼容 provider"。这个小巧的设计让 claude-mem 能接 DeepSeek、Moonshot、本地 vLLM、Ollama、LM Studio……任何符合 OpenAI API 协议的服务。

### 实战：DeepSeek 接入的三个坑

我自己接 DeepSeek-V4-Flash 踩了三个坑，**这些坑文档不写**，是真正考验工程经验的地方。

**坑 1：proxy 环境下的 auth 假设**

我的 Claude Code 走代理（`ANTHROPIC_BASE_URL=http://127.0.0.1:15721` + `PROXY_MANAGED` token），keychain 里没有真实 Anthropic OAuth token。claude-mem worker spawn 时从 keychain 读 OAuth，拿到的是空/无效 token，调 Claude SDK 直接返回 "Not logged in · Please run /login"。

provider 默认是 `claude`，假设所有用户都用 Claude Code 官方订阅。这个假设在 proxy 用户、自托管 Claude Code、SDK 直连用户那里**全部失效**。

**坑 2：免费模型的生命周期短**

切到 OpenRouter 后第一个模型 `xiaomi/mimo-v2-flash:free` 直接 404——下架了。换 `meta-llama/llama-3.3-70b-instruct:free`，限速 429 严重。免费 tier 在 OpenRouter 上不是稳定产品，是限时活动。生产用必须用付费模型。

**坑 3：reasoning 模型 vs XML 解析**

DeepSeek-V4-Flash 是推理模型，响应里带 `reasoning_content` 字段。它对 claude-mem 要求的 XML 输出反应不稳定——有时把所有 token 都用在 reasoning 上，最终 `content` 字段为空，claude-mem 的解析器拿到 "Empty response" 直接丢弃。实测 4 次 SDK 调用里有 2 次空响应，observation 生成成功率只有 50%。

要让 DeepSeek 在结构化输出场景下不输出 reasoning_content，需要往请求 body 里加 `{"thinking": {"type": "disabled"}}`（这是 DeepSeek API 的扩展参数，通过错误响应反向探测得到——`reasoning_effort` 接受 `low/medium/high/max/xhigh` 但无法完全关闭，`thinking: {type: "disabled"}` 是唯一能彻底关掉 reasoning 的方式）。这种 thinking-control API 正在成为 reasoning 模型的标配——结构化输出场景需要关掉 thinking，已经从模型能力演化为 API 层的一等公民。

#### claude-mem 的兼容方式

claude-mem worker 发 OpenAI 兼容请求的代码（minified 后形态）：

```js
body: JSON.stringify({
  model: n,
  messages: c,
  temperature: .3,
  max_tokens: 4096,
  ...s.includes("openrouter.ai") ? { usage: { include: !0 } } : {}
})
```

`s` 是 base URL。这里已经有一个值得借鉴的 pattern——**条件 spread**，只在真打 openrouter.ai 时加 `usage.include` 字段，对其他端点零影响。给 DeepSeek 加 thinking-disabled 沿用同一模式：

```js
body: JSON.stringify({
  model: n,
  messages: c,
  temperature: .3,
  max_tokens: 4096,
  ...s.includes("openrouter.ai") ? { usage: { include: !0 } } : {},
  ...s.includes("deepseek.com") ? { thinking: { type: "disabled" } } : {}
})
```

这是个**自限制的 patch**——只在 baseURL 含 `deepseek.com` 时生效，对其他 provider 完全透明。pattern 的本质是：**per-provider extra body**，根据 base URL 判断要不要往请求里加特定字段。如果 claude-mem 把这个抽象成 `CLAUDE_MEM_OPENROUTER_EXTRA_BODY` 这样的环境变量，用户传任意 JSON 合并到 body，就不需要改源码了——但目前没有这个抽象，只能 patch 源码。

patch 后 DeepSeek 调用成功率从 50% 升到 100%，token 用量降一半（reasoning tokens 全省下来了）。

#### 升级覆盖与对抗性维护

`claude plugin update` 会覆盖 worker-service.cjs，patch 失效。每次升级都要重打。工程化的解法是写一个 **idempotent patch 脚本**：先 grep patch marker 检测是否已应用（幂等），再检测 anchor 字符串是否还在（防止 upstream 改了源码 shape 后误改文件），然后备份 + 应用 + 验证。整个脚本几十行 bash 就能搞定，挂到 `claude plugin update` 之后的钩子里基本无感。

但本质上这是**对抗性维护**——claude-mem 升级随时可能换 anchor 字符串、改 provider 抽象方式，到那时 patch 就要重写或废弃。更彻底的解法还是上游提供扩展点（环境变量 / 钩子 / 插件机制），让 per-provider 适配不需要改源码。这反映了开源工具的一个普遍张力：**默认值 vs 可配置性**——默认行为要简单（不需要用户配置），但边界 case 必须留扩展点（否则用户被迫 hack 源码）。



---

## 7. Agent Memory 设计的本质题

实现拆完了。从 claude-mem 抽身出来看 agent memory 设计本身——这些本质题不是 claude-mem 独有的，而是任何想做长期记忆的 agent 系统都要回答的。

### 7.1 写什么：observation / event / summary 的取舍

任何 memory 系统第一个要回答的问题：**该把什么存下来**。粗看答案似乎是"全部"，但工程上"全部"是最糟的——存储爆炸、召回低效、注入污染。

claude-mem 存的是"经过压缩的情景记忆 + 半语义化事实"——每条 observation 都是某次工具调用的产物，带有 `memory_session_id` 和 `created_at_epoch`，但内容已经是 AI 压缩后的"我们发现了 X / 决定了 Y"这种半结构化结论。它**刻意不做** procedural memory（"做 X 的标准步骤"），那需要更结构化的"技能"抽象，claude-mem 不碰。

值得讨论的几个设计点：

**Raw event vs compressed observation**：raw event 忠实但膨胀，compressed observation 经济但有损。claude-mem 选了 compressed + 把 raw 留在 Claude Code 自己的 jsonl transcript 里，**等于用 Claude Code 的 transcript 兜底"信息无损"**。这是个聪明的分工：memory 系统只管"压缩的、可召回的"，原始数据让宿主负责。

**单条 vs batch**：claude-mem 每次工具调用生成一条 observation。但有些"记忆"是跨多次工具调用形成的——比如"我们花了一下午排这个 bug，试了 A、B、C 三个方案都不行，最后是 D 解决的"。这种 narrative 用单条 observation 表达不出来，需要**跨 observation 聚合**。claude-mem 的 `session_summaries` 部分回答了这个（Stop hook 生成），但聚合粒度还是 session 级，不是 task 级。

**作者视角**：observation 是 agent 写的，但它描述的是 user + agent 协作的过程。这导致 observation 经常出现"用户要求 X / 我们发现了 Y / 我决定 Z"这种主语混乱。未来如果做多 agent，每个 agent 自己的 memory 怎么区分？这是个未解问题。

**结构化程度**：claude-mem 用 XML + 字段（title/subtitle/narrative/facts/concepts/...）。MemGPT 用更自由的"memory block"。Letta 加了"core memory / archival memory"分层。结构化越强，召回越准但表达力越弱；结构化越弱，表达力强但召回难——经典的 schema vs schema-less 取舍。

### 7.2 何时读：被动注入 vs 主动召回 vs 显式查询

memory 系统的读取时机，决定了 agent "意识到"自己有记忆的方式。三种模式：

**1. 被动注入（SessionStart injection）**

新 session 启动时，自动把 top-N 相关记忆塞进 system prompt。claude-mem 默认这么干（50 条 observation + 10 条 summary）。

- **优点**：agent 无需任何动作就有上下文，体验最好
- **缺点 1**：挤占 attention budget——每次启动固定注入 15-30k tokens，不管有没有用
- **缺点 2**：召回决策在 session 启动时做的，但 session 真正的任务要在用户第一个 prompt 后才明朗。所以召回基于"历史相关性"而不是"当前任务相关性"

**2. 主动召回（Tool-call triggered retrieval）**

agent 在执行某个动作前，根据当前上下文召回相关记忆。claude-mem 的 PreToolUse(Read) hook 就是这个模式——读文件 X 时自动拉关于 X 的历史 observation。

- **优点**：召回时机精准，相关性高
- **缺点**：只在 hook 配置的特定时机触发，覆盖不全

**3. 显式查询（On-demand query）**

agent 自己判断"我需要查记忆"，主动调用 `memory_search` 这种工具。

- **优点**：agent 有完全的 agency，最灵活
- **缺点**：依赖 agent 的"我需要查记忆"自我意识——但 LLM 经常意识不到自己需要查

claude-mem 三种都用：SessionStart 注入 + PreToolUse 召回 + 暴露 MCP 工具供显式查。这种"全栈"设计覆盖率最高，但也最重。

最关键的问题是**相关性计算**。三种模式都依赖"哪些 obs 跟当前 context 相关"的判断。claude-mem 用 hybrid（FTS + 向量），但 hybrid 也不是银弹：FTS 偏 literal（搜 "checkpoint" 漏掉 "model saving"），向量偏 fuzzy（语义相关但不精准），hybrid 权重难调。

更深的问题是：**"搜得到"和"该不该现在出现"是两件事**。一条 obs 可能跟当前 context 高度相关（向量相似度高），但当前 session 是个全新任务，旧 obs 反而是干扰。memory 系统需要判断**"相关性 ≠ 有用性"**——前者是检索问题，后者是决策问题。当前主流 memory 系统（包括 claude-mem）都只解决前者。


---

## 8. 开源 Memory 横评：当下社区在尝试什么

把 claude-mem 放回 2026 年开源 agent memory 的版图里看，才能看清它的设计选择处在什么位置。这一节先抽出"设计空间"的几个维度，再过五个有代表性的开源项目，最后讨论一个让社区震动的反直觉发现。

### 8.1 设计空间：五个正交维度

任何一个 agent memory 系统都要在五个维度上做选择，这些维度大致正交：

| 维度 | 选项光谱 |
|---|---|
| **写入触发** | 全自动（每次 tool call） ↔ LLM 自决（"我觉得这值得记"） ↔ 用户显式 |
| **写入内容** | raw event ↔ 压缩 observation ↔ 结构化 fact / triple |
| **读取时机** | 被动注入（system prompt） ↔ 主动召回（hook 触发） ↔ 显式查询（agent call） |
| **存储模型** | flat 文件 ↔ 关系表 ↔ 向量库 ↔ 知识图谱 ↔ 时序图谱 |
| **遗忘机制** | 无 ↔ 时间衰减 ↔ 显式归档 ↔ 自动合并 / 重写 |

claude-mem 的选择：**全自动 / 压缩 observation / 三种读法全用 / SQLite+Chroma / 几乎不遗忘**。它在"写入触发"上偏激进（全自动），在"存储模型"和"遗忘"上偏朴素。

### 8.2 五个代表项目

#### mem0 — hybrid vector + graph + LLM curator

[mem0](https://github.com/mem0ai/mem0) 是目前 star 数最高的开源 memory layer。架构核心是用一个 LLM（默认 GPT-4o-mini）作为 **memory curator**——每条潜在 memory 先让小模型判断值得记吗？是 ADD、UPDATE（更新已有）、还是 DELETE（覆盖过期）。存储是 **vector + graph 双后端**：向量做相似度召回，图谱（Neo4j）做实体关系查询，再加一个 history log 可回溯。

关键设计点：**写入经过 LLM 筛选 + 增量更新**。同一个事实多次出现会被 UPDATE 而不是 INSERT N 条。这跟 claude-mem 的"每次 tool call 一条 observation"是两种哲学——mem0 把 memory 当**结构化知识库**维护，claude-mem 把 memory 当**情景日志**累积。

#### Letta（前身 MemGPT）— 分层记忆 + LLM 作为 memory manager

[Letta](https://github.com/letta-ai/letta) 走 OS 内存层级隐喻：**Core memory**（永远在 prompt 里的 block，agent 自己通过 tool call 编辑）→ **Recall memory**（对话历史归档）→ **Archival memory**（向量检索的长期存储）。

最大设计差异：**LLM 自己管理 memory**——决定什么进 core、什么下沉到 archival、什么时候 search_recall。这跟 claude-mem 的"工程师预设管道自动跑"完全相反。这是个有想象力的方向（agent 自治），但代价是 prompt 消耗大、行为不可预测、调试困难。

#### Zep / Graphiti — 时序知识图谱

[Zep](https://github.com/getzep/graphiti)（论文 [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)）的核心是**时序知识图谱**。每条记忆是图里的节点 + 边，**带时间维度**——同一个事实可以有多条时间戳不同的记录。比如"项目用 React" 在 2024-01 是 truth，2024-06 改成 "项目用 Vue"，Zep 会同时存这两条带时间区间的 triple，召回时按当前时间筛选。

这种设计直接回应了"过期记忆"问题——不是删旧记忆，而是给记忆加 valid interval。代价是图构建复杂、查询需要时序推理、对底层 LLM 能力要求高。

#### Cognee 与 A-MEM — SDK 路线与涌现 schema

另外两条值得注意的路线：[Cognee](https://github.com/topoteretes/cognee) 走 Python SDK 路线（vector + graph + relational 的轻量 library）；[A-MEM](https://arxiv.org/abs/2502.12110)（NeurIPS 2025 poster）受 Zettelkasten 启发，让 schema 从内容涌现而非预设——这跟 claude-mem 的固定 schema 形成对比，是经典的 schema vs schema-less 取舍。


### 8.3 Letta "Filesystem All You Need" 的冲击

2025 年下半年，Letta 团队发了一篇让社区震动的 benchmark：[Benchmarking AI Agent Memory: Is a Filesystem All You Need?](https://www.letta.com/blog/benchmarking-ai-agent-memory/)。他们在 LoCoMo（Long Conversation Memory）benchmark 上比较了几种 memory 方案：

| 方案 | LoCoMo 分数 |
|---|---|
| mem0 | 68.5% |
| **Letta Filesystem（只是把对话存进文件）** | **74.0%** |
| Backboard（后来的 SOTA） | 90.1% |

最朴素的设计——**直接把对话历史写进一个文件**——打败了精心设计的 mem0（vector + graph + LLM curator）。这个结果引发了激烈讨论，[Charles Packer 的 LinkedIn 帖子](https://www.linkedin.com/posts/charles-packer_dont-trust-everything-you-read-online-activity-7361113034754347008-Ic7L)直接质疑了 mem0 基于 68.5% 这个分数自称"state-of-the-art"的宣称。

benchmark 的意义不在数字本身（LoCoMo 有自己的问题），而在揭示的几个反直觉事实：

**1. LLM 的 in-context 能力被严重低估**。当模型能从大段原始 context 里提取需要的信息时，复杂的写入时压缩 + 召回链路反而会**丢失信号**。mem0 的"小模型先过滤再存"流程，相当于让一个不太聪明的模型提前替主模型做判断——这种中间决策一旦错了，主模型连纠正的机会都没有。Letta Filesystem 把决策权完全留给读时的主模型，反而保留了完整信号。

**2. memory 系统的"中间层"是有代价的**。每多一层抽象（压缩、结构化、向量化、图谱化）都可能引入信息损失。压缩是"用信息分辨率换 token 经济性"——只有当压缩准确性极高时才划算，但 LLM 压缩的准确性本身是个问题。

**3. write-time vs read-time compute 是核心张力**。两条路线基于完全不同的假设：

- **write-time compression**（claude-mem、mem0）：未来读很多次，所以现在花算力压缩值得。读取越频繁，压缩摊销越划算。
- **read-time retrieval**（Letta Filesystem、raw RAG）：未来读的次数少，且 LLM 在 raw context 里就能检索到需要的信息。压缩省的 token 不值得牺牲信号保真度。

#### 对 claude-mem 的拷问

claude-mem 走的是激进的 write-time compression——**每次工具调用都触发一次 AI 压缩**。这跟 Claude Code 的实际使用模式有个微妙的张力：写入是确定的（每次 tool call 触发），但读取是间接的——多数 obs 只有在排进 top-50 时才真正被"读"到，剩下的大量 obs 写入后从未召回。

如果一条 obs 写入花 1500 tokens 但整个生命周期只被召回 0.5 次，它的"压缩摊销"就是失败的——花了压缩成本，读取时根本没用上。

这引出一个值得认真考虑的方向：**Claude Code 场景下，Letta Filesystem 风格的方案可能整体更优**。直接把每次工具调用的 raw input/output 写进按日期切分的文件，不压缩、不结构化、不向量化；session 启动时把当前 project 最近 N 天的 raw 文件全量塞进 context；让 LLM 自己在 raw context 里找相关信息。这个方案写入几乎零成本、信号完整无损、召回质量随模型变强只会更好——代价是 context 膨胀快，需要 aggressive 的 file rotation。

这不一定真比 claude-mem 强，但 Letta Filesystem benchmark 让这个方向值得认真考虑。**memory 设计的"默认答案"可能不是压缩，而是 raw + 强模型 in-context**——前提是模型够强、context 够大、读取频率够低。

归根到底，关键变量是 **写入频率 / 读取频率的比值**：读取频繁且可预测（如客服 bot 反复回答同类问题）走 write-time compression 划算；写入海量但每次召回率低（典型如个人开发场景，大部分 obs 永远排不进 top-50）走 read-time retrieval + raw 保真可能更划算。claude-mem 选了 write-time compression 且没有为后者做 fallback——这反映了 memory 设计领域的一个隐性偏见：**默认把 memory 当成"高频检索的知识库"设计，而不是"低频回顾的事件流"设计**。


### 8.4 claude-mem 在版图里的位置

把上面五个项目跟 claude-mem 对照，几个关键差异：

- **跟 mem0**：mem0 把 memory 当**结构化知识库**维护（UPDATE 而非 INSERT，LLM curator 筛选写入），适合长期偏好和事实沉淀；claude-mem 把 memory 当**情景日志**累积（每次 tool call 一条），适合工作过程的连续性。
- **跟 Letta**：claude-mem 走"工程师预设管道"路线，Letta 走"LLM 自治管理 memory"路线。前者可预测、好调试，后者更有想象力但消耗大——"工具基础设施 vs agent 自身能力"的哲学差异。
- **跟 Zep**：claude-mem 没有时序维度，过期记忆处理留白。Zep 的时序图谱是真正的"记忆演化"，但实现复杂度高得多——需要图数据库、时序推理、LLM 维护图结构，跟 claude-mem 的"SQLite 就够"是两个量级的工程投入。
- **跟 A-MEM**：固定 schema（claude-mem）vs 涌现 schema（A-MEM）——召回准 vs 表达力强的经典工程取舍。
- **跟 Letta Filesystem**：write-time compression（claude-mem，每条 obs 一次 AI 调用）vs read-time retrieval（Letta Filesystem，几乎不压缩，靠 LLM 在 raw context 里检索）。这是 8.3 节展开过的核心张力。

claude-mem 整体偏向**朴素工程化路线**——SQL + FTS + 向量、固定 schema、自动管道。在研究维度上不算前卫：没有 graph、没有 temporal、没有 agentic memory management。但这未必是缺陷——Letta filesystem benchmark 提示了一个反直觉的事实：**朴素方案在工程上经常不输复杂方案**，当 LLM 自身的 in-context 能力足够强时，复杂的中间层反而可能丢失信号。

claude-mem 真正差异化的地方不在存储模型，而在**集成深度**——把 Claude Code 的 6 个 hook 全部接住、把 worker / MCP / DB / UI 的角色分工理清楚、把 provider 抽象做得足够灵活。这些都是工程贡献而非研究贡献，但对最终用户体验的影响往往比架构新颖性更大。换句话说，**claude-mem 的核心竞争力是"在 Claude Code 生态里跑得最稳"，不是"memory 设计最先进"**——这两个目标是正交的。




---

## 9. 本质的张力：六个未解难题

agent memory 不是个 solved problem。下面六个张力，每一个都没有完美答案。

### 9.1 Memory Pollution：低质量记忆的累积

LLM 写的 observation 不一定准。它会：

- **过度泛化**："所有 checkpoint 失败都是 mcore 问题"（其实只见过一个 case）
- **过度具体**："这个 bug 是 line 42 的拼写错误"（line 42 改了之后这条 obs 就过期了）
- **编造**：明明没读过那个文件，说"已读 train.py:100"
- **冗余**：同一个事实写十条 obs，因为十次工具调用都触发了

这些低质量 obs 进入库后，会被反复召回，**强化模型的错误认知**。这跟"幻觉"的不同在于：幻觉是当场生成的，下一轮可能就修正了；pollution 是**持久化的幻觉**，会反复影响后续 session。

防御手段：

- 写入前 dedup（content_hash 字段是这个意图，但只防完全相同的 obs）
- 多次出现才升级 confidence（重复观测才相信）
- 用户 review（重，但有效）
- 时间衰减（缓解但不去除）

claude-mem 的 dedup 比较弱（只 hash 全字段），没做多 observation 投票。这是个可以改进的点。

### 9.2 Cascading Errors：错误记忆的连锁影响

如果一条错的 obs 被召回了，它会影响 agent 的当前推理；agent 基于错误推理做出的新动作，又被 PostToolUse 压成新的 observation——**错的 obs 繁殖错的 obs**。

这是 memory 系统特有的失败模式，比单次幻觉严重得多。单次幻觉在下次 session 失忆后就消失了，但 memory 让错误**持久化**。

防御手段：

- 给 obs 加 confidence / source 字段，召回时按 confidence 加权
- 用户能 flag 错误 obs，flag 后该 obs 不再召回（claude-mem 当前不支持）
- 引入"反例"——专门存"这条 obs 是错的，因为 X"的负面记忆

一个值得探索的方向是**meta-memory**——关于"哪些记忆可信、哪些可疑"的二级记忆。这跟人类的元认知对应：人能意识到"我对这件事的记忆可能不准"，agent memory 系统目前普遍没有这一层。

### 9.3 自我连续性：跨 session 的"我"

如果 agent 的 memory 是 session 级 fragment 的累积，那"它是同一个 agent 吗"？

这听起来玄学，但落到工程上很具体——**memory schema 的向后兼容性**。我自己踩过一个实例：claude-mem 某次升级后，`enabledPlugins` 里 `claude-mem@thedotmack` 键消失了，hook 完全不触发但 MCP server 还在跑。表面看一切正常，实际近 5 周没生成任何 observation。新 session 启动时拿不到近期记忆，agent 实质上"失忆"了——它跟我之前配合过的 agent 在记忆上是断开的。

这反映了 memory 系统的稳定性是个**用户心理契约**问题，不只是技术问题。用户跟 agent 建立的不是"工具使用关系"，是"协作关系"——而协作的连续性完全建立在 memory 系统的健康上。几个具体的工程含义：

- **schema 的向后兼容**：升级路径必须考虑"老 memory 怎么办"，不能简单 drop
- **可观测性**：memory 系统要主动暴露健康状态（hook 是否触发、observation 是否增长），让"看起来正常实际失忆"的故障能被发现
- **export/import**：用户应该能导出自己的 memory 并迁移到别的工具——这是数据可携性的基本原则
- **跨 agent 的 memory 共享**：用 Claude 时的 memory 能不能给 GPT 用？技术上和协议上都是开放问题

claude-mem 当前没有 export/import 能力，schema 也是私有的。这意味着用户被绑死在 claude-mem 的格式上，未来换工具的迁移成本很高。

### 9.4 作者权与隐私：这是谁的 memory？

最后一个问题，可能是最重要的。

agent memory 里存的不是只有 agent 的工作——里面有你（user）的想法、你写过的代码、你做过的决策、甚至你的失败。这条 memory 是**你的**，还是**agent 的**，还是**服务商的**？

具体场景：

- claude-mem 把 memory 存本地，**所有权清晰**（你的）
- ChatGPT memory 存云端，OpenAI 用它训练模型了吗？条款说不，但你验证不了
- 如果你给 agent 看了客户的源代码（哪怕是临时调试），相关 observation 进了 memory，下次别的客户的项目里被召回——**数据泄露**
- 跨 session 的"用户画像"被 build 出来：你的工作习惯、技术栈、犯错模式、决断风格……这些是高度敏感的隐私

防御手段：

- project 维度隔离（claude-mem 有 project 字段，但默认不强制隔离）
- 敏感数据 redaction（识别 secret、PII，不写入 memory）
- 用户能审计 / 删除（GDPR 的 right to be forgotten）
- 端到端加密（如果存云端）

claude-mem 因为是本地存储，大部分隐私问题自动规避了。但**跨 project 污染**仍然存在——同一个 cwd 不同项目的 obs 会混在同一个库里，靠 `project` 字段做软隔离。如果用户在多个客户项目间切换，理论上一个项目的 obs 可能影响另一个项目。

这是个**架构层面的隐私**问题，需要设计时就考虑。

### 9.5 Meta-memory：记忆的元认知

前三个张力（pollution、cascading errors、自我连续性）合起来指向同一个更深的缺失：**当前 agent memory 系统普遍没有"元记忆"层**——关于记忆本身的可信度、来源、时效的二级记忆。

人类有这种元认知。说"我记得三年前那份合同里**好像**有这一条"时，"好像"承载的就是元记忆——对自身记忆可信度的判断。这让人能区分"我确定的"和"我大致印象的"，在重要决策时主动核实。agent memory 系统目前**普遍没有这一层**——所有 obs 写入时都被当作同等可信，召回时都按相同权重进入 prompt，模型不知道哪些是"高置信度事实"、哪些是"早期误解、可能已过时"。

claude-mem 的 schema 有 `relevance_count`、`content_hash`、`generated_by_model`、`created_at_epoch` 等字段，但要么记账不用，要么只做技术用途。**没有 confidence、source、evidence、stale_at** 这种元认知字段。后果是几种典型失败：

- **临时方案被记成永久事实**：obs 写"项目用 React"，但其实只是某次实验性尝试，三个月前就改回 Vue 了
- **AI 编造被当作观测**：压缩时 hallucinate 说"已读 train.py:100"，但其实没读过——这条 obs 反复召回强化错误认知
- **早期误解持续污染**：刚开始接触 codebase 时写的 obs 经常是错的；后续 session 没有机制知道"这条来自早期、可能不准"
- **同源错误累积**：同一根因导致的多次工具调用各生成一条 obs，看起来是 N 条独立证据，其实是同一个错的 N 次重复

#### meta-memory 的设计空间

如果要做 meta-memory，几个正交维度：

1. **confidence**——每条 obs 写入时带置信度。可以是生成模型的 logprob、用户事后 review 的标记、或基于"这条 obs 跟其他 obs 是否一致"的派生分数。召回时按 confidence 加权注入
2. **source / evidence**——记录来源（用户告诉的 / agent 观测的 / agent 推断的 / 文件读取的）和支撑证据（"基于读 train.py:42 的内容"）。source 决定可信度，evidence 让召回时可以溯源
3. **多观测投票**——同一事实被多次独立观测后才升级为"高置信度事实"。一次观测进"待定"队列，三次以上才进主库——对应科学界的"可重复性"
4. **时效 / 失效检测**——obs 引用的 file:line 改了或决策被新 obs 推翻时，自动标记 stale 或降权。需要 memory 系统主动维护
5. **反例 / 矛盾记忆**——允许写"obs X 是错的，因为 Y"这种二级 obs。不删除 X，召回时让两者同时出现，让模型自己判断

#### 为什么没人做（包括 claude-mem）

设计空间清晰，但工程实现复杂度让人望而却步：confidence 没有 ground truth 难定；多观测投票需要"事实级"dedup，而语义级 dedup 本身是难题；失效检测需要持续监控 codebase；反例机制让召回路径变复杂。

更深层的原因是**当前 agent memory 领域还处在"先解决有无"阶段**——大家都在抢"如何存如何检索"的基础设施，没到"如何信任如何怀疑"的元层。这是技术成熟度曲线的自然进程：先做能用的，再做能信的。

但这是个**早晚要补的层**。没有 meta-memory，agent memory 的可靠性会随库增长单调下降——错误 obs 累积速度比正确 obs 快（错误的更"有戏剧性"，更容易被模型选中），最终整个 memory 变成不可信的污染源。届时要么全量 reset，要么用户花大量时间手动 prune——两者都不该是生产级方案。

### 9.6 Forgetting：缺席的核心机制

跟 meta-memory 紧密相关的另一个缺失是**主动遗忘**。claude-mem 几乎不做 forgetting（只有用户手动 prune），这反映了当前 agent memory 领域的另一个普遍短板：**重写入轻淘汰**。

claude-mem 的当前库（用了约 6 周）有 1300 条 obs，按线性增长推演：半年约 11000 条，一年约 18000 条。但**召回 top-50 的预算是固定的**——SessionStart 永远只注入 50 条。直接后果：

- **召回"信号密度"单调下降**：库大时 top-50 只是冰山一角，大量相关 obs 排不进
- **召回质量被老 obs 占位**：时间排序的 top-50 里老 obs 越来越多
- **过期 obs 累积**：代码改了、决策推翻了、bug 早修了——这些 obs 没机制失效，永远在库里
- **存储 + 查询性能下降**：向量库 50000 条 vs 1300 条，查询慢一个数量级

#### Forgetting 的设计选项

最小可行的 forgetting 机制，几个候选：

1. **时间衰减权重**：老的 obs 默认权重低，召回时乘以衰减因子。简单但粗暴——很多重要决策是早期做的，不该被时间冲掉。衰减曲线本身也不好定（指数？线性？基于绝对时间还是 last-accessed？）
2. **任务完成后归档**：跟 task 绑定的 obs，task 完成后移到冷存储。需要 task abstraction——claude-mem 目前只有 session 级和 project 级，没有 task 级
3. **自动合并相似 obs**：语义相近的 obs 自动合并成一条带 confidence 的"聚合 obs"。需要语义聚类，错了会丢信息（A-MEM 的 Zettelkasten 链接机制跟这个方向相关）
4. **基于代码状态的失效检测**：obs 引用的 file:line 改了，自动标记 stale 或归档。**这可能是 ROI 最高的方向**——codebase 是 ground truth，obs 跟它对齐就有了客观判据。需要持续监控，开销大但准确
5. **显式归档**：用户主动标记 obs 为"已过期 / 已废弃"。轻量但依赖用户参与
6. **相关性驱动的淘汰**：长期没被召回的 obs 自动降权或归档（"三个月没被读过的 obs 默认进冷库"）。模拟人脑的"不用就忘"。claude-mem 的 `relevance_count` 字段其实已经预留了这个能力——只是没用

#### 为什么 forgetting 是核心机制

forgetting 看起来像 hygiene 功能（保持库干净），但它实际上是**记忆系统长期可用的前提**：没有 forgetting，库增长必然导致召回质量下降、错误 obs 无限累积、用户的 right to be forgotten 无法实现、隐私边界无法管理。

人类记忆的 forgetting 不是 bug 是 feature——它是大脑在有限神经元容量下保持长期可用的核心机制。agent memory 系统的容量限制虽然比大脑宽松（磁盘比神经元便宜），但**召回质量 vs 库大小**的权衡曲线类似：库里东西越多，召回 top-N 的"代表性"越差。

claude-mem 和它的大多数同类目前都没认真对待 forgetting——这反映了领域成熟度（还在"解决存的问题"，没到"解决淘汰的问题"）。但随着这些系统使用时间变长（从月到年到多年），forgetting 一定会从 nice-to-have 变成 must-have。


---

## 10. 结语：能从 claude-mem 学到什么

把 claude-mem 拆到底，能提炼出的核心观察大致是这些：

- **架构层面**：把 memory 系统分成 worker / MCP / hook / DB 四个角色是合理的——读写路径解耦、生命周期由宿主 hook 决定、存储用本地 SQLite + Chroma。这套结构适合任何"agent + 跨 session 记忆"的场景，不止 Claude Code。
- **数据层面**：write-time 压缩成结构化 observation 是 token 经济性上的正确取舍，但代价是压缩错了没法纠回。type 系统让记忆有了结构，但 6 种类型是否穷举，本身就需要持续审视。
- **集成层面**：claude-mem 真正的生产价值不在架构新颖，而在**集成深度**——它把宿主的 6 个 hook 全部接住、把 provider 抽象做得足够灵活（`OPENROUTER_BASE_URL` 那一笔让它能接任意 OpenAI 兼容端点）、把异步队列和 retry 做到了不阻塞交互。这些是工程贡献，不是研究贡献。
- **未解决问题**：forgetting、cascading errors、跨 session 自我连续性、隐私边界——这些问题在 claude-mem 里基本没被解决，是整个 agent memory 领域的共同短板。

### 四个落到 claude-mem 的设计问题

接着核心观察，几个落到 claude-mem 具体设计上的开放问题——没有标准答案，但想清楚它们是 agent memory 下一步演进的前提。（meta-memory 和 forgetting 这两个更深的张力已经在第 9 节展开讨论过，这里不重复。）

**Q1：claude-mem 的 6 种 type 是穷举的吗？**

`bugfix / feature / refactor / discovery / decision / change`——这个分类是预设的、固定的。但实际数据里 `discovery` 占了 47%，说明这个类目**承担了过多**——它把"读了一个新文件"、"理解了一个新概念"、"发现了一个潜在 bug 但没修"等非常不同的认知活动都装在一起了。同时，明显缺失的类目包括：

- **用户偏好**（"用户喜欢简洁回答"、"用户用 zsh"）——procedural / preference 类记忆
- **项目约束**（"这个项目不能用 pandas"、"提交前必须跑 ruff"）——normative 类记忆
- **悬而未决的问题**（"我们还没确定用 A 还是 B"）——open-question 类记忆
- **失败尝试**（"试过 X 方案但不行"）——negative-knowledge 类记忆

A-MEM 的路线是不预设 schema、让记忆类型从内容里涌现。这两条路怎么取舍？预设 schema 召回准但表达力弱，涌现 schema 表达力强但召回难。**有没有可能做成"启动时预设、但允许 LLM 提议新 type"的混合模式？**

**Q2：SessionStart 注入 50 条 obs——这个数字是怎么调出来的？**

`CLAUDE_MEM_CONTEXT_OBSERVATIONS=50` 是默认值。但 50 这个数字是 heuristic 还是有数据支撑？换个数字会怎样？

- 注入 5 条：召回准确度会下降多少？（top-5 太严，相关但排名 6-10 的 obs 全漏）
- 注入 500 条：attention budget 还剩多少？（500 条 obs 大约 100-200k tokens，已经吃掉大半 context window）
- 数字应该跟 observation 平均长度、当前任务类型、上下文窗口大小**动态适配**吗？

这是个值得做 ablation 的方向。Letta filesystem benchmark 的反直觉结果提示我们：**注入策略的"最优值"可能跟模型 in-context 能力强相关**——模型越强，越能从原始 context 里自己筛信息，越不需要 memory 系统预先压缩 + 排序。

**Q3：PreToolUse 只对 Read 触发——为什么不对 Bash、Edit？**

claude-mem 的 hook 配置里 `PreToolUse` 的 matcher 是 `Read`，只在读文件时召回相关 obs。但逻辑上，**Edit 之前的相关 obs 更重要**——避免改错地方；**Bash 之前的相关 obs 可能有用**——之前跑过类似命令的结果。

为什么这么设计？可能的考虑：

- **hook 调用成本**：每次 hook 都要跑 SQL + 向量查询，对 Bash 这种高频工具会显著拖慢
- **召回准确性**：Read 的"相关 obs"容易判断（文件名匹配），Bash 的"相关 obs"难判断（命令意图理解）
- **noise 控制**：每次 Edit 都注入历史 obs，可能让 agent 困惑（"我之前是这么改的，现在还要这么改吗？"）

但代价是覆盖不全——agent 在 Edit 时没有历史 context，可能重复犯之前犯过的错。**这是个值得做 ablation 的设计权衡**：PreToolUse 全工具覆盖 vs 只 Read，对任务完成率和延迟的影响是什么？

**Q4：本地 SQLite 是单机最优——多 agent / 多机器怎么办？**

claude-mem 的所有数据在 `~/.claude-mem/`，单机本地。这是个隐私友好、可部署性好的选择，但限制了几个场景：

- **多 agent 共享 memory**：Claude Code 写的 obs 能给 Cline / 自研 agent 用吗？目前不能，因为格式和存储都是 claude-mem 私有的
- **跨机器同步**：你在笔记本上工作，想换台式机继续，memory 怎么迁移？目前只能手动 copy 数据库
- **团队协作**：团队多个人对同一个项目的 memory 能否共享？涉及隐私和合规
- **商业代码边界**：obs 里可能包含客户代码细节，同步到另一台机器是否合规？

这些问题在个人使用场景下不重要，但**决定了 claude-mem 能否从个人工具升级到团队 / 企业工具**。mem0 / Zep 选了云端 service 路线，部分原因就是这些场景需要中央存储 + 多端访问。claude-mem 选本地是务实，但也是个天花板。

### 落到实操的几个判断

回答完抽象问题，最后给几条具体的使用建议：

**1. claude-mem 适合长期在固定项目工作的开发者**。每天跟同一个 codebase 打交道、跨多日断续推进任务，跨 session 记忆的价值显著。如果是"一天换三个项目、每个 session 独立"的用法，memory 的价值密度很低，注入反而成噪声。

**2. memory generation 应当选结构化能力强的模型，而不是 reasoning 模型**。接 reasoning 模型要么用 thinking-disabled 参数，要么换非 reasoning 变体。这跟"哪种模型更聪明"无关——结构化输出和 chain-of-thought 是两个方向。

**3. 主动做 memory 卫生**。定期打开 viewer UI 看 observation 质量、prune 明显错的、用 `CLAUDE_MEM_EXCLUDED_PROJECTS` 隔离临时项目。memory 系统不会自我修复，越用越脏是必然。同时主动做 health check——hook 静默失败、provider auth 过期、queue 积压都可能让系统"看起来正常实际失忆"几周不自知。

---

memory 不是 agent 的附加品，是 agent 设计的核心一部分。它决定了 agent 的"自我"如何连续、如何学习、如何避免重复犯错——也决定了 agent 的"自我"如何被自己累积的错误记忆所污染。这两面是同一个机制的两面，无法只取其一。

---

> 写于 2026 年 6 月，基于 claude-mem 13.8.0 的真实使用与拆解。
>
> 文中所有数据来自我本机的实际安装（1300+ observations、70+ sessions），所有坑来自调试过程。文中观点仅代表此刻，欢迎讨论。

