---
title: "Skill: graphify - 代码库转知识图谱"
date: 2026-05-28
draft: false
description: "将代码库转化为可导航知识图谱的 Claude Code 技能，支持多 Agent 平台"
categories: ["Skills"]
tags: ["skills", "claude-code", "graphify", "knowledge-graph", "codebase-analysis", "RAG", "tree-sitter"]
---

> GitHub: [safishamsi/graphify](https://github.com/safishamsi/graphify)
> 官网: [graphifylabs.ai](https://graphifylabs.ai)
> YC S26 加速器项目

## 核心定位

在 AI 编程助手（Claude Code、Codex、Cursor 等）中输入 `/graphify`，它会将整个项目——代码、文档、PDF、图片、视频——映射成可查询的知识图谱。

不再是 grep 文件，而是**查询关系**。

## 安装

### pip 安装

> **注意**：官方包名是 `graphifyy`（双 y），PyPI 上其他 `graphify*` 包与其无关。

```bash
# 推荐方式（uv 自动配置 PATH）
uv tool install graphifyy

# 或使用 pipx
pipx install graphifyy

# 或使用 pip
pip install graphifyy
```

### 注册 Skill

```bash
graphify install
```

### 前置要求

| 要求 | 最低版本 | 检查命令 |
|------|----------|----------|
| Python | 3.10+ | `python --version` |
| uv（推荐） | 任意 | `uv --version` |

**macOS**：
```bash
brew install python@3.12 uv
```

**Windows**：
```powershell
winget install astral-sh.uv
```

## 快速开始

```bash
/graphify .
```

运行后会生成三个文件：

```
graphify-out/
├── graph.html       # 交互式可视化，浏览器打开，可点击节点、过滤、搜索
├── GRAPH_REPORT.md  # 审计报告：关键概念、意外连接、建议问题
└── graph.json       # 完整图谱数据，随时可查询
```

### 支持的平台

| 平台 | 安装命令 |
|------|----------|
| Claude Code | `graphify install` |
| Codex | `graphify install --platform codex` |
| Cursor | `graphify cursor install` |
| Gemini CLI | `graphify install --platform gemini` |
| OpenCode | `graphify install --platform opencode` |
| VS Code Copilot | `graphify vscode install` |
| Aider | `graphify install --platform aider` |
| OpenClaw | `graphify install --platform claw` |

> Codex 用户：还需在 `~/.codex/config.toml` 的 `[features]` 下添加 `multi_agent = true`。
> Codex 使用 `$graphify` 而非 `/graphify`。

## 输出内容

### 报告内容

| 内容 | 说明 |
|------|------|
| **God nodes** | 项目中连接最多的概念，所有东西都流经这些节点 |
| **Surprising connections** | 跨文件/模块的意外连接，按意外程度排序 |
| **The "why"** | 内联注释（`# NOTE:`、`# WHY:`、`# HACK:`）、docstring，设计文档 |
| **Suggested questions** | 4-5 个图谱最适合回答的问题 |
| **Confidence tags** | 每个推断关系标记为 EXTRACTED/INFERRED/AMBIGUOUS |

### 支持的文件类型

| 类型 | 扩展名 |
|------|--------|
| 代码（33种语言） | `.py .ts .js .jsx .tsx .go .rs .java .c .cpp .rb .cs .kt .swift ...` |
| MCP 配置 | `.mcp.json mcp.json mcp_servers.json claude_desktop_config.json` |
| 文档 | `.md .mdx .html .txt .rst .yaml .yml` |
| Office | `.docx .xlsx`（需安装 `[office]`） |
| PDF | `.pdf` |
| 图片 | `.png .jpg .webp .gif` |
| 视频/音频 | `.mp4 .mov .mp3 .wav`（需安装 `[video]`） |

**代码本地提取**：使用 tree-sitter AST，无需 API 调用。其他类型通过 AI 助手模型 API 处理。

## Best Practices

### 三阶段提取管线

1. **AST 提取**（免费，本地）：tree-sitter 解析代码，提取函数、类、import、调用图、内联注释
2. **视频/音频转录**（本地）：faster-whisper，有缓存
3. **语义提取**（API 调用）：文档、PDF、图片通过 LLM 子代理

### 置信度标记

| 标记 | 含义 | 置信度 |
|------|------|--------|
| `EXTRACTED` | 源码中直接找到 | 1.0 |
| `INFERRED` | LLM 推断 | 0.55-0.95 |
| `AMBIGUOUS` | 需人工复核 | - |

### 团队协作

**推荐流程**：
```bash
# 1. 一个人运行并提交
/graphify .
git add graphify-out/ && git commit -m "Add knowledge graph"

# 2. 其他人 pull 后，助手立即可用
git pull

# 3. git commit 时自动重建
graphify hook install  # 设置 git merge driver
```

**推荐的 `.gitignore` 添加**：
```
graphify-out/manifest.json    # mtime-based，clone 后失效
graphify-out/cost.json        # 本地 cost 跟踪
# graphify-out/cache/         # 可选：提交以加速
```

### 性能优化

```bash
# 大图谱（>5000 节点）：跳过 HTML 生成
graphify extract . --no-viz

# 仅处理变更文件
graphify . --update

# 强制重建（大型重构后）
graphify . --force

# 限制 VRAM（Ollama 本地推理）
GRAPHIFY_OLLAMA_NUM_CTX=8192 graphify extract --token-budget 4000
```

### 增量更新

- SHA256 缓存在 `graphify-out/cache/`
- 未变更文件自动跳过
- 大型重构后用 `--force` 清除幽灵重复

### Token 节省

| 查询方式 | Token 消耗 |
|----------|-----------|
| 读取原始文件 | 100% |
| 使用图谱查询 | ~1.4% |

benchmark: 52 文件语料库，**71.5x token 节省**。

## 常用命令

### 构建图谱

```bash
/graphify .                        # 为当前目录构建图谱
/graphify ./docs --update          # 仅重新提取变更文件
/graphify . --cluster-only         # 重新聚类，不重新提取
/graphify . --no-viz              # 跳过 HTML，仅生成报告 + JSON
/graphify . --wiki                # 从图谱构建 Markdown wiki
```

### 查询图谱

```bash
/graphify query "what connects auth to the database?"
/graphify path "UserService" "DatabasePool"
/graphify explain "RateLimiter"
```

### 高级功能

```bash
# 添加外部资源
/graphify add https://arxiv.org/abs/1706.03762   # 论文
/graphify add <youtube-url>                       # 视频转录

# 自动重建
graphify hook install              # git commit 时自动重建

# 合并图谱
graphify merge-graphs a.json b.json

# PR 仪表盘
graphify prs                       # CI 状态、review 状态、工作树映射
graphify prs --triage             # AI 排序 review 队列
graphify prs --conflicts          # 共享图谱社区的 PR（合并顺序风险）
```

### 聚类控制

```bash
# 更细粒度的社区
graphify . --cluster-only --resolution 1.5

# 抑制工具类超级节点
graphify . --cluster-only --exclude-hubs 99
```

## 可选扩展

| 扩展 | 用途 | 安装 |
|------|------|------|
| `pdf` | PDF 提取 | `pip install "graphifyy[pdf]"` |
| `office` | Word/Excel 支持 | `pip install "graphifyy[office]"` |
| `video` | 视频/音频转录 | `pip install "graphifyy[video]"` |
| `mcp` | MCP stdio 服务器 | `pip install "graphifyy[mcp]"` |
| `neo4j` | Neo4j 推送 | `pip install "graphifyy[neo4j]"` |
| `gemini` | Gemini API | `pip install "graphifyy[gemini]"` |
| `openai` | OpenAI API | `pip install "graphifyy[openai]"` |
| `ollama` | 本地推理 | `pip install "graphifyy[ollama]"` |
| `all` | 全部安装 | `pip install "graphifyy[all]"` |

## 忽略文件

创建 `.graphifyignore`（语法同 `.gitignore`）：

```
# .graphifyignore
node_modules/
dist/
*.generated.py

# 仅索引 src/，忽略其他
*
!src/
!src/**
```

## 为什么值得用

1. **超越关键词搜索**：揭示隐含关系，而非仅匹配关键词
2. **诚实的不确定性**：AMBIGUOUS 标记，区分「找到」和「推测」
3. **持久化**：跨 session 可用，输出可提交到 git
4. **增量友好**：变化时只更新需要的部分
5. **多 Agent 支持**：Claude Code、Codex、Cursor、Gemini CLI 等
6. **超低 token**：71.5x 节省

---

> 💡 **Tip**: 首次运行 `/graphify .`，之后代码库问题会直接通过图谱回答
