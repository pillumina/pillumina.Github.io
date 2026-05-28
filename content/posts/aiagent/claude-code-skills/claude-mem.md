---
title: "Skill: Claude-Mem - 记忆系统工具集"
date: 2026-05-28
draft: false
description: "Claude Code 的记忆系统：跨会话持久化上下文，让 Agent 记住项目历史"
categories: ["AI Agent", "Claude Code Skills"]
tags: ["claude-mem", "memory", "knowledge-base", "codebase"]
---

> GitHub: [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)
> 文档: [docs.claude-mem.ai](https://docs.claude-mem.ai/)
> 版本: v6.5.0

## 核心定位

Claude-Mem 是一个**上下文持久化压缩系统**，专门为 Claude Code 设计。它能够：

- 自动捕获工具使用情况并生成语义摘要
- 将历史上下文在新会话中恢复
- 让 Claude 在跨会话场景下保持对项目的持续理解

**关键洞察**：Agent 的「记忆」应该是持久化的，而不仅仅是当前会话内的。

## 安装

### Claude Code（推荐）

```bash
/plugin marketplace add thedotmack/claude-mem
/plugin install claude-mem
/reload-plugins
```

### npx 一键安装

```bash
npx claude-mem install
```

### 其他 IDE

```bash
# Gemini CLI
npx claude-mem install --ide gemini-cli

# OpenCode
npx claude-mem install --ide opencode
```

> **注意**：npm 安装 `claude-mem` 仅安装 SDK，不注册插件钩子。务必使用 `npx claude-mem install` 或 `/plugin` 命令。

### 前置要求

| 要求 | 最低版本 | 说明 |
|------|----------|------|
| Node.js | 18+ | Claude Code 插件支持 |
| Bun | - | 自动安装，进程管理器 |
| uv | - | 自动安装，向量搜索用 |
| SQLite 3 | - | 自动安装，持久化存储 |

## 核心功能

| 功能 | 说明 |
|------|------|
| **Persistent Memory** | 上下文跨会话持久化 |
| **Progressive Disclosure** | 分层记忆检索，按需加载 |
| **mem-search Skill** | 自然语言查询项目历史 |
| **Web Viewer UI** | http://localhost:37777 |
| **Privacy Control** | `<private>` 标签排除敏感内容 |
| **Beta Channel** | Endless Mode 等实验功能 |

## 工作原理

### 核心组件

1. **5 个生命周期钩子**：SessionStart、UserPromptSubmit、PostToolUse、Stop、SessionEnd
2. **Worker Service**：HTTP API 运行在 37777 端口，带 Web Viewer UI
3. **SQLite 数据库**：存储会话、观察记录、摘要
4. **mem-search Skill**：自然语言查询 + 渐进式披露
5. **Chroma 向量数据库**：混合语义 + 关键词搜索

### 记忆流程

```
会话开始 → SessionStart 钩子 → 加载相关记忆 → 会话进行 → 工具调用
    ↓
PostToolUse 钩子 → 生成观察记录 → 摘要生成 → 存入数据库
    ↓
会话结束 → SessionEnd 钩子 → 持久化上下文
```

## Best Practices

### 三层工作流（必须遵循）

Claude-Mem 提供 4 个 MCP 工具，采用**三层工作流模式**以节省 ~10x token：

| 步骤 | 工具 | 目的 | 成本 |
|------|------|------|------|
| 1 | `search` | 获取紧凑索引 | ~50-100 tokens/结果 |
| 2 | `timeline` | 获取时间上下文 | ~variable |
| 3 | `get_observations` | 仅对过滤后的 ID 获取完整详情 | ~500-1000 tokens/每个 |

**使用示例**：
```typescript
// 步骤 1：搜索索引
search(query="authentication bug", type="observations", obs_type="bugfix", limit=20)

// 步骤 2：获取相关 ID 的时间上下文
timeline(anchor=12345, depth_before=5, depth_after=5)

// 步骤 3：仅对相关 ID 获取完整详情
get_observations(ids=[12345, 67890])
```

**关键规则**：
- ✅ 在 `get_observations` 中批量 ID（一次请求 vs N 次请求）
- ✅ 先搜索 → 再过滤 → 最后获取详情
- ❌ 不要直接获取所有搜索结果的详情

### 隐私控制

使用 `<private>` 标签排除敏感内容：
```
这个 API key 是 <private>sk-xxx</private> 请不要记录
```

标签在钩子层被剥离，敏感信息永不进入数据库。

### 多账户隔离

设置 `CLAUDE_MEM_DATA_DIR` 隔离不同项目/账户：
```bash
# 工作账户
CLAUDE_MEM_DATA_DIR=~/.claude-mem-work npx claude-mem install

# 个人账户（默认）
npx claude-mem install
```

### 增量更新

首次运行后，后续仅处理变更文件：
- SHA256 缓存在 `graphify-out/cache/`
- 未变更文件自动跳过
- 大型重构后用 `graphify . --force` 清除幽灵重复

### 上下文工程原则

> "找到最小的高信号 token 集合，最大化期望结果的概率。"

**渐进式披露**：从紧凑索引开始，仅在需要时展开详情。

## MCP 搜索工具

### 工具列表

| 工具 | 用途 |
|------|------|
| `search` | 语义搜索记忆索引，支持 type/date/project 过滤 |
| `timeline` | 获取特定观察的时间上下文 |
| `get_observations` | 按 ID 批量获取完整观察详情 |
| `observation_context` | 获取上下文摘要 |

### 使用示例

```typescript
// 查找认证相关的 bug 修复
search(query="authentication bug", type="observations", obs_type="bugfix", limit=10)

// 查找特定时间段的观察
search(query="API refactor", dateStart="2026-01-01", dateEnd="2026-03-01")

// 组合过滤
search(query="performance", type="observations", project="my-project", limit=20)
```

## mem-search Skill

```bash
mem-search "how did we handle authentication?"
mem-search "why was this architecture chosen?"
mem-search "what was the issue with the database?"
```

基于自然语言的语义搜索记忆历史，返回关联的观察记录。

## Web Viewer

访问 http://localhost:37777 查看：
- 实时记忆流
- 观察记录列表
- Beta 功能切换（Endless Mode）
- 配置管理
- Bug 报告生成器

## 配置选项

配置文件位于 `~/.claude-mem/settings.json`（首次运行自动创建）。

### 模式与语言

```json
{
  "CLAUDE_MEM_MODE": "code--zh"
}
```

| 模式 | 说明 |
|------|------|
| `code` | 默认英文模式 |
| `code--zh` | 简体中文模式 |

> 注意：`code--zh` 模式内置支持，无需额外安装。

### 其他配置

- AI 模型选择
- Worker 端口
- 数据目录
- 日志级别
- 上下文注入控制

详见 [Configuration Guide](https://docs.claude-mem.ai/configuration)。

## Beta 功能

通过 Web Viewer UI（http://localhost:37777 → Settings）切换 stable/beta 频道。

**Endless Mode**：生物启发的记忆架构，适合超长会话。

详见 [Beta Features](https://docs.claude-mem.ai/beta-features)。

## 为什么值得用

1. **超越关键词搜索**：揭示隐含关系，而非仅匹配关键词
2. **诚实的不确定性**：AMBIGUOUS 标记，区分「找到」和「推测」
3. **持久化**：跨 session 可用，不依赖单一会话
4. **增量友好**：变化时只更新需要的部分
5. **Token 效率**：~10x 节省，通过渐进式披露
6. **隐私保护**：敏感信息永不进入数据库

## 故障排除

遇到问题时：
```bash
cd ~/.claude/plugins/marketplaces/thedotmack
npm run bug-report
```

自动诊断并生成报告。

---

> 💡 **Tip**: 安装后重启 Claude Code，之前会话的上下文会自动出现在新会话中
