+++
date = '2026-07-09T10:00:00+08:00'
draft = false
title = '从 Pocock Skills 到昇腾诊断：Skill 与 Knowledge 体系设计草案'
categories = ['AIAgent']
tags = ['agent', 'skills', 'claude-code', 'ascend', 'knowledge-base', 'diagnosis', 'matt-pocock']
summary = '以昇腾训练和推理支持场景为背景，综合 Pocock skills 的设计思想，提出一套三层知识架构的 Skill 加 Knowledge 体系设计草案，涉及诊断流程、知识分层、团队协作和持续演化。'
+++

> 这是[上一篇拆解 Matt Pocock skills 的文章](../matt-pocock-skills/)的续篇。上一篇拆解了 Pocock 的设计思想——这篇把这些思想应用到一个真实的场景：昇腾训练和推理支持团队的日常问题定位。这不是一个已实现的系统，而是一个从具体约束出发的体系设计草案。

---

## 1. 场景与约束

角色是昇腾支持工程师，接口 MindSpeed-LLM、MindSpeed-MM、veRL、vllm-ascend、SGLang 等框架在 A2 / A3 / A5 不同架构上的客户问题。问题覆盖三类：

- **功能中断**：框架报错、进程崩溃、通信 hang
- **精度异常**：loss 异常、FP8 衰减、allreduce 精度退化
- **性能退化**：吞吐下降、step time 波动、EP 通信占比过高

有四个约束决定了这个体系不能照搬 Pocock：

1. **团队不能统一使用同一个 AI coding agent**——公司网络限制和工具偏好导致诊断对话分散在多个 AI Chatbot（Claude Code、Kimi、DeepSeek 网页版）甚至纯手工排查中。
2. **已有大量历史案例**，但格式混乱——截图、IM 聊天记录、个人笔记都有。
3. **新 case 每周新增 5-10 个**，且 A2 / A3 / A5 的差异持续扩大。
4. **个人维护能力有限**，任何需要一个人持续手工维护的方案都会在三周内腐化。

---

## 2. 体系概览

三层架构，每层有不同的读写频率和维护人：

```
+-----------------------------------------------------+
| Skill 层：诊断流程（低频变更）                         |
| /diagnose-training-issue                             |
| /diagnose-inference-issue                            |
| /to-postmortem                                       |
| /knowledge-groom                                     |
+-----------------------------------------------------+
| Knowledge 层：诊断规则（中频增长）                      |
| Tier 1: triage-tree.yaml    (30 条分支, 极低频)        |
| Tier 2: pattern_library/    (200 条 case, 周级增长)    |
| Tier 3: postmortems/        (无上限, 日级增长)          |
+-----------------------------------------------------+
| Script 层：工具链（已有，保持不变）                      |
| ascend-profile-analyze / mem-analyze / bench-run /   |
| collect-profiling / machine-ops                      |
+-----------------------------------------------------+
```

一条 case 的完整生命周期：

```
团队成员定位问题（使用任意 agent 或纯手工）
    |
    +-- 方式 A：在 Claude Code / Codex 中协作诊断
    |       +-- session 结束时 agent 自动跑 summary prompt
    |
    +-- 方式 B：用 Kimi / DeepSeek / 网页版对话诊断
    |       +-- 把对话记录粘贴给 /to-postmortem
    |
    +-- 产出：postmortem.md -> postmortems/YYYY-QN/

/knowledge-groom（每周运行）
    |
    +-- 扫 postmortems/ 中未处理的 .md
    +-- 人审批通过的 -> 结构化 YAML -> 追加到 pattern_library/
    +-- 无法结构化的 -> 标记 needs-human-review
    +-- 合并检测：bucket 中的相似 case 自动提示合并
```

---

## 3. Skill 层设计

### 3.1 设计原则

五条原则直接提取自 Pocock 体系，适配到昇腾场景后加了具体语义。完整的对照分析见[上一篇](../matt-pocock-skills/)。

**原则 1：一个 skill 只做一件事，做完就停。**
`/diagnose-training-issue` 的终点是定位 root cause 或标记 need-escalation，不会继续做"顺便帮客户把配置改了"。

**原则 2：人判断不能被自动化取代，但可以被结构化。**
skill 不替人做诊断决策——它给出结构化验证清单，人执行后把结果贴回来，agent 分析结果后给出下一步。诊断过程中 agent 是分析引擎，人是执行终端。

**原则 3：上下文窗口是有限资源，要显式管理。**
长诊断 session 超过 smart zone 后用 `/handoff` 分段，不要在推理退化后继续诊断。

**原则 4：知识不绑定在 skill body 里。**
skill body 只写诊断方法论（如何分类、如何收集症状、如何按平台差异匹配）。具体的 case rules 存在 knowledge/ 下，skill 按需加载。改 case 不需要改 skill。

**原则 5：知识沉淀是 skill 的 side effect，不是人的额外负担。**
每条诊断 skill 执行完毕自动生成 postmortem 草稿。人不写文档——agent 写，人审。

### 3.2 `/diagnose-training-issue`

```
流程：
1. 收集症状
   - 错误信息、环境变量（HCCL_*, ASCEND_*, NPU_*）、框架版本、硬件平台

2. 分类 -> 加载 triage-tree.yaml Tier 1
   - hang / crash -> pattern_library/training/hang/
   - precision -> pattern_library/training/precision/
   - performance -> pattern_library/training/perf/
   - 无法分类 -> Tier 3 向量检索

3. 诊断（加载对应 Tier 2 bucket）
   - 对 bucket 中每个 case 执行 quickly_check（轻量预检命令）
   - 通过预检的 case 执行完整 diagnosis checks 验证
   - 命中 -> 输出 root cause + fix
   - 未命中 -> 进入深度排查

4. 深度排查（Tier 2 未命中）
   - 收集完整 profiler 数据（如果有）
   - Tier 3 向量检索提供启发式提示
   - 人工/agent 联合分析

5. 产出
   - resolution: resolved / escalated / unknown
   - postmortem 草稿（自动生成）
```

### 3.3 `/to-postmortem`

解决的核心问题：团队成员不一定用同一个 agent 做诊断，因此知识注入入口必须与诊断工具解耦。

```
用法：
  /to-postmortem "[粘贴 Kimi/DeepSeek 的完整对话]
                  [或粘贴纯手工排查笔记]"

流程：
1. 从输入中提取症状、执行的命令和输出、排除的假设、最终 root cause 和 fix
2. 输出结构化 YAML 草稿 + postmortem.md
   - 标记 confidence: high | medium | low
   - 标记 novelty: new_pattern | variant | covered
3. 人扫一眼确认 -> done（30 秒内可完成）
```

如果原始对话中包含 `!key` 标记，对应信息在结构化时提升权重。`!key` 是一个极其轻量的约定——人在发现关键线索时打一个前缀，不需要额外的工具或流程。

### 3.4 `/knowledge-groom`

```
触发：手动运行，建议每周一次

流程：
1. 扫 postmortems/ 中新增且通过审批的 .md
2. 对每个 postmortem：
   +-- 尝试结构化 -> YAML
   +-- 成功 -> 追加到 pattern_library/<bucket>.yaml
   +-- 失败 -> 标记 needs-human-review
   +-- novelty = covered -> 跳过，不重复添加
3. 合并检测：相似 case 对自动提示合并
4. 产出 PR：pattern_library/ 变更列表 + 合并建议 + 需人工补充的项
```

---

## 4. Knowledge 层设计

### 4.1 为什么需要三层

500 条平铺的 case entry 对 agent 来说是灾难——每轮诊断都要全量加载，token 膨胀，推理退化。三层架构把搜索空间逐级缩小：Tier 1 把 500 条缩小到约 15 条，Tier 2 做精确匹配，Tier 3 做模糊兜底。

| 层 | 内容 | 条目上限 | 加载时机 | token 消耗 |
|---|---|---|---|---|
| Tier 1 | 症状分类索引 | 30 分支 | 始终加载 | ~2K |
| Tier 2 | 结构化 case rules | 200 条 | 症状匹配后加载一个桶 | ~8K |
| Tier 3 | 原始 postmortem | 无上限 | T1+T2 未命中时向量检索 | ~5K |

总 token 消耗控制在 15K 以内——即使最坏情况（T1 匹配 + T2 bucket 全量 + T3 top-3），也不会超过一次诊断 session 推理窗口的 15%。

### 4.2 Tier 1: triage-tree.yaml

不存储具体 root cause，只做分类路由。每条分支是一组症状正则模式，匹配后加载对应的 Tier 2 bucket。

```yaml
branches:
  - id: training_hang
    symptoms:
      - "timeout" | "hang" | "stuck at step"
      - "NCCL.*timeout" | "HCCL.*timeout"
      - "all_to_all.*timeout"
    goto: pattern_library/training/hang/

  - id: training_precision
    symptoms:
      - "nan" | "loss.*nan" | "fp8.*precision"
      - "bf16.*mismatch" | "allreduce.*round"
    goto: pattern_library/training/precision/

  - id: training_perf_regression
    symptoms:
      - "step time.*spike" | "throughput.*drop"
      - "EP.*bottleneck" | "communication.*slow"
    goto: pattern_library/training/perf/

  - id: uncategorized
    symptoms: []
    goto: null  # 未分类的走 Tier 3
```

设计要求：
- 分支数不超过 30。超过 30 说明分类太细——合并相似分支
- 症状模式是正则兼容的模糊匹配，不做精确匹配
- 一个症状可能匹配多个分支 -> agent 加载所有匹配的 bucket，按 priority 排序

### 4.3 Tier 2: pattern_library/

每个文件一个诊断桶，包含 10 到 30 条结构化 case entry：

```yaml
cases:
  - id: ASCEND-EP-HANG-001
    title: "HCCL buffer undersize for large-scale EP dispatch"
    priority: high
    platforms: ["A5-910C"]
    frameworks: ["mindspeed-llm>=2.4.0"]

    symptoms:
      - "all_to_all_single hangs at step usually after 1000+"
      - "world_size >= 64"

    quickly_check:
      command: "grep -c 'all_to_all' /path/to/error.log"
      expected: "regex:^[1-9]"

    diagnosis:
      - step: 1
        command: "env | grep HCCL_BUFFSIZE"
        expected: ">= 4194304"
        fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"

      - step: 2
        command: "python3 check_ep_topology.py --world_size ${WORLD_SIZE}"
        expected: "all ranks reachable"
        fix_on_mismatch: "escalate to network team"

    root_cause: "HCCL internal buffer insufficient for EP all-to-all"
    fix: "export HCCL_BUFFSIZE=4194304 before training launch"
```

关键设计决策：

- `quickly_check` 的作用不是诊断——是快速过滤。它必须在 5 秒内执行完毕，让 agent 跳过明显不相关的 bucket
- `diagnosis` 是顺序执行，不跳步。任一步 mismatch 且没有 fix_on_mismatch -> agent 标记该 case 不匹配，进入下一条
- `priority: high` 的 case 优先验证——常见且易修复的问题应该在搜索路径上排前面

### 4.4 Tier 3: postmortems/

原始诊断记录，不做结构化。仅用于 T1 和 T2 都未命中时的向量检索兜底。文件由 session-end summary 或 `/to-postmortem` 生成，按季度目录归档。

```
postmortems/
+-- 2026-Q3/
|   +-- 2026-07-05-a5-fp8-precision.md
|   +-- 2026-07-06-vllm-ascend-pd-separate-hang.md
+-- 2026-Q2/
```

---

## 5. 知识注入与协作

核心设计目标：知识注入不绑定到特定 AI 工具。

| | 路径 A：agent 协作诊断 | 路径 B：外部定位 |
|---|---|---|
| 使用场景 | 工程师使用 Claude Code / Codex | 工程师使用 Kimi / DeepSeek 网页版或手工排查 |
| 触发方式 | session 结束时 agent 自动跑 summary | 人运行 `/to-postmortem "[粘贴对话]"` |
| 输出 | postmortem.md（自动生成） | postmortem.md（agent 提取） |
| 人需要做什么 | 扫一眼确认 root cause 和 fix（30s） | 同左 |
| 后续 | `/knowledge-groom` 定期升格到 Tier 2 | 同左 |

不要期望团队成员额外写一份文档。"把今天解决的那个 case 写下来"是一个额外的负担，且质量完全取决于人的自觉和记忆力。正确的方案是 agent 在 session 结束时自动生成草稿（或人粘贴外部对话后 agent 提取），人只做审批——成本从 20 分钟降到 30 秒。

### 团队分工

| 角色 | 职责 |
|------|------|
| 一线工程师 | 诊断定位 + 跑 `/to-postmortem`（或 agent 自动生成）+ 扫一眼确认 |
| 领域 owner | 审批 `/knowledge-groom` 的升格 PR + 手补无法自动结构化的 case |
| 体系维护人 | 维护 triage-tree 分支、审议 bucket 合并、裁决 confidence 争议 |

---

## 6. 与 Pocock 体系的设计映射

| Pocock 概念 | Ascend Skills 对应 | 异同 |
|---|---|---|
| `/grill-with-docs` | `/diagnose-training-issue` | 都是人加 agent 协作，对齐后产出结构化记录 |
| `CONTEXT.md` | `knowledge/` | Pocock 的是静态术语，我们是动态诊断 rules |
| `/domain-modeling` | `/knowledge-groom` | 定期维护知识结构 |
| postmortem（Pocock 没有） | `postmortems/` | 昇腾场景的核心机制——知识持续增长 |
| `disable-model-invocation` | 诊断 skill 设为 user-only | 诊断决策需要人判断 |
| `ask-matt` 路由器 | `/diagnose-training-issue` 入口 | 统一入口，避免记忆负担 |

---

## 7. 演化机制

体系需要自我约束——不加控制的增长会摧毁检索效率。

| 信号 | 触发动作 |
|------|----------|
| Tier 2 bucket 未命中率 > 60% 持续两周 | 审查是否需要拆分或扩大症状匹配 |
| Tier 2 单个 bucket 超过 30 条 case | 输出相似 case 对，建议合并 |
| 某个 case 被命中 >= 5 次且每次都直接解决 | 提升 priority = high |
| 某个 case 标记 needs-human-review 超过 30 天 | 提醒领域 owner 处理 |
| 积累 5 条 needs-skill-update 标记的 case | 审查 skill 本身的诊断流程 |

不增长的设计：Tier 2 上限 200 条（超限强制合并）、Tier 1 上限 30 分支（超限说明分类过细）、Tier 3 有最小质量阈值（缺少 root_cause 或 symptoms 的记录不进入向量索引）。

---

## 8. 接下来的步骤

1. **搭建 `/to-postmortem` 原型**——选 3 到 5 个真实 case，手工转换为结构化 YAML 作为 reference，写 skill body 和 summary prompt，团队试跑验证"人均 30 秒确认"的假设是否成立。
2. **搭建 triage-tree.yaml 框架**——从团队现有 issue 中提取最高频的 5 到 8 个症状组，挂载到 Tier 2 bucket。
3. **跑第一轮 `/knowledge-groom`**——把现有 20 到 30 个 case 文档过一遍，度量自动结构化的成功率。
4. **Tier 3 向量检索**——等 Tier 2 积累到 50 条以上 case 后，验证未命中率，决定是否引入向量检索兜底。
