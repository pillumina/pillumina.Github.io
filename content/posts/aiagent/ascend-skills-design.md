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

## 8. 诊断精度：误诊的代价与回滚

当前设计假设匹配到的 case 就是对的。但现实中可能出现两个问题：

- 两个 case 有几乎相同的症状但完全不同的 root cause——A5 上的 EP hang 和 A3 上的 EP hang 可能都是 `all_to_all timeout`，但一个是 HCCL buffer 问题，一个是 firmware 版本 bug
- agent 可能因为症状模糊匹配到错误的 case，执行了错误的 fix

**需要三个防御层**：

**层一：fix 标注回滚方式**。每个 `fix_on_mismatch` 应该携带回滚指令，修复前 agent 先输出回滚方式。

```yaml
fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"
rollback: "unset HCCL_BUFFSIZE  # 恢复默认值"
```

**层二：串联保护**。如果 agent 连续两次匹配到不同 case（第一次 fix 没解决问题），强制转为人工介入，不再继续尝试第三个 case。这个规则写入 `/diagnose-*` 的 SKILL.md。

**层三：误诊率追踪**。每个 case 被命中但 fix 未解决问题时，标记 `misdiagnosis: true`。这个信号比"未命中率"更关键——高未命中率说明知识库不够大，高误诊率说明知识库有错误信息。追踪方式见 [§13 量化指标](#13-量化指标与回顾)。

---

## 9. 平台差异矩阵：不是三张表，是一个字段

A2、A3、A5 的差异不是三套独立的 case 库——是大量共享规则 + 少量平台特定差异。比如 HCCL 行为在 A3 和 A5 上几乎相同，但 A2 完全不同；FP8 精度问题只在 A5 上出现（A2 和 A3 不支持 FP8）。

当前 `platforms: ["A5-910C"]` 只能表达"这个 case 适用于哪些平台"，不能表达"同一个 root cause 在 A3 上的检查命令完全不同"。

**方案：字段级平台差异**。一个 case 可以有多组 `diagnosis`，分别对应不同平台：

```yaml
cases:
  - id: ASCEND-EP-HANG-001
    title: "HCCL buffer undersize for large-scale EP dispatch"
    priority: high

    symptoms:
      - "all_to_all_single hangs at step after 1000+"

    diagnosis:
      - platforms: ["A5-910C", "A3-910B"]
        steps:
          - command: "env | grep HCCL_BUFFSIZE"
            expected: ">= 4194304"
            fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"
            rollback: "unset HCCL_BUFFSIZE"

      - platforms: ["A2-910A"]
        steps:
          - command: "cat /proc/driver/npu/version"
            expected: ">= 23.0"
            note: "A2 上不存在 HCCL_BUFFSIZE 参数。检查 NPU 驱动版本是否 >= 23.0"
```

此外，需要一份 `platforms/` 目录存放平台级背景知识——类似 Pocock 的 CONTEXT.md，但是平台级而非项目级。agent 在加载 Tier 2 bucket 的同时加载平台差异文件，自动选择匹配的 diagnosis 分支。

```
knowledge/
+-- platforms/
    +-- a2-910a.md    <-- A2 已知特性清单（不支持 FP8、HCCL 行为差异、最多 8 卡）
    +-- a3-910b.md
    +-- a5-910c.md
```

每份文件约 500 字，由领域 owner 维护，极低频率更新。不是要写完整的硬件手册——只写**诊断相关的**平台差异。

---

## 10. 多跳诊断：当第一个 fix 没解决问题

当前的 diagnosis 设计是单跳的：匹配一个 case -> 执行 checks -> 命中 root cause -> fix。但实际诊断经常是多跳的——中间步骤可能是"修复了但仍然不行"，触发下一个 case 的匹配。

```
EP hang (症状)
  -> check 1: HCCL_BUFFSIZE 不够 (发现不够)
  -> fix: 改成 4194304
  -> 重新训练，还是 hang
  -> check 2: UB switch 拓扑问题 (已排除——拓扑正常)
  -> check 3: NPU firmware 版本不匹配
  -> 这才是真正的 root cause
```

**方案：显式诊断链**。在 case schema 中预留一个 `next_on_fail` 字段，指向下一个应该尝试的 case id：

```yaml
cases:
  - id: ASCEND-EP-HANG-001
    next_on_fail: "ASCEND-EP-HANG-003"
    # 如果这个 case 的 fix 被应用但问题仍然存在，尝试 EP-HANG-003
```

这不是要求每条 case 都建立多跳——单跳能解决 70% 的问题。`next_on_fail` 是可选字段，只在已知"修复 X 之后经常需要再检查 Y"的 case 对中使用。它可以逐步从 `/knowledge-groom` 的统计分析中自动生成（如果 groom 发现 case A 被命中后 case B 在同一个 session 中也被命中的概率超过 40%，自动建议建立关联）。

---

## 11. Session 中断与恢复：诊断可能被会议打断

诊断不是连续的时间段。实际场景：

- 你正在跑 agent 给出的 check 命令，被同事打断去开紧急会议。回来时记不住之前在查什么。
- agent 在 session 中间的上下文已经被 compact 了一次，之前的诊断链路丢失了。

Pocock 的 `/handoff` 是 session 级别的交接——你需要的是 session 内部的抗中断机制。

**方案：诊断状态文件**。`/diagnose-training-issue` 的每个 step 执行后，更新一个本地 `diagnosis_state.yaml`：

```yaml
session_id: "2026-07-09-ep-hang-a5"
status: in_progress
current_step: 3
excluded_cases:
  - ASCEND-EP-HANG-001  # fix 应用了但问题未解决
  - ASCEND-EP-HANG-002  # symptoms 不匹配（已验证）
active_case: ASCEND-EP-HANG-003
last_action: "等待用户执行 check_ep_topology.py 并贴回输出"
```

当 session 恢复时，agent 首先读这个文件，而不是从头开始收集症状。状态文件同时解决了另一个问题：如果多个工程师在同一个 issue 上协作，他们可以共享状态文件，避免重复排查。

---

## 12. 人可读的速查表：不通过 agent 也能用

不是每个工程师每次都愿意用 agent。有时候只是想搜一条命令：`grep HCCL_BUFFSIZE` 是多少。当前的 YAML 对 agent 友好，但对人完全不行——没人愿意在 30 个 YAML 文件里翻。

**方案：自动生成 CHEATSHEET.md**。`/knowledge-groom` 每次运行时，除了产出 YAML，也产出 Markdown 速查表：

```markdown
## EP Hang

| 检查命令 | 期望值 | 修复方式 | 平台 |
|----------|--------|---------|------|
| `env | grep HCCL_BUFFSIZE` | >= 4194304 | `export HCCL_BUFFSIZE=4194304` | A5, A3 |
| `python3 check_ep_topology.py` | all reachable | 联系网络组 | A5 |
| `cat /proc/driver/npu/version` | >= 23.0 | 升级 NPU 驱动 | A2 |

## FP8 Precision

| 检查命令 | 期望值 | 修复方式 | 平台 |
|----------|--------|---------|------|
| `grep 'round error' /var/log/npu/device-0.log` | 无匹配 | 降级到 BF16 allreduce | A5 |
```

这份文件不用人手维护——它完全由 groom 自动生成。它有三个作用：
- **离线可用**：agent 挂了或网络不可用时，人仍然能完成基础排查
- **dogfooding 质量**：如果从 YAML 生成出来的速查表让人看不下去，说明 YAML 的结构化有问题——这成为知识质量的自动检测信号
- **新成员 onboarding**：新人不需要理解 skill 体系就能使用知识库

---

## 13. 量化指标与回顾

当前设计没有任何数字来衡量这套体系的效果。六个月后你怎么知道它值得继续投入？不需要构建 dashboard——一个 Markdown 表格加上定期手工更新就够了。

**核心指标**：

| 指标 | 含义 | 数据来源 |
|------|------|---------|
| 命中率 | Tier 2 直接匹配并解决的比例 | `/diagnose-*` session 结束时的 resolution 字段 |
| 误诊率 | 命中了但 fix 没解决问题 | 同 resolution，需要手动标记 `misdiagnosis: true` |
| 平均诊断时间 | 从接手到定位 root cause 的时间 | 人自己在 session 开始时记一下，结束时算差值 |
| 知识增长速度 | 每周新增 postmortem + 成功升格到 Tier 2 的数量 | `/knowledge-groom` 的输出 PR |
| 知识覆盖率 | 每月新问题中，有对应 case 的比例 | 手动回翻一个月内的 issue，标记 "有/无对应 case" |

**记录方式**：一个 `docs/metrics.md`，每两周由体系维护人手工追加一条。格式极其简单：

```markdown
## 2026-W28
- 处理 issue 总数: 12
- Tier 2 命中: 7 (58%)
- Tier 2 命中但误诊: 1
- Tier 2 未命中, Tier 3 辅助定位: 2
- Tier 2 未命中, 纯人工: 2
- 新增 postmortem: 4
- 成功升格到 Tier 2: 2
- 失败升格 (需手补): 1
- 平均诊断时间 (估计): ~45min
```

不需要自动化。有数字比没有数字重要得多——手工记录的数字够用了。关键是**定期回顾**：体系维护人每两周看一遍 metrics，回答一个问题："这三层架构真的在变好用，还是我们在自欺欺人？"

---

## 14. 接下来的步骤

1. **搭建 `/to-postmortem` 原型**——选 3 到 5 个真实 case，手工转换为结构化 YAML 作为 reference，写 skill body 和 summary prompt，团队试跑验证"人均 30 秒确认"的假设。
2. **搭建 triage-tree.yaml 框架**——从团队现有 issue 中提取最高频的 5 到 8 个症状组，挂载到 Tier 2 bucket。
3. **搭建 `platforms/` 平台差异文件**——三份文件，各 500 字，领域 owner 一小时内可以写完初版。
4. **跑第一轮 `/knowledge-groom`**——把现有 20 到 30 个 case 文档过一遍，度量自动结构化的成功率，同时生成第一版 CHEATSHEET.md。
5. **Tier 3 向量检索**——等 Tier 2 积累到 50 条以上 case 后，验证未命中率，决定是否引入向量检索兜底。
6. **建 `docs/metrics.md`**——体系维护人在第二轮 groom 后开始手工记录，形成两周一次的回看节奏。
