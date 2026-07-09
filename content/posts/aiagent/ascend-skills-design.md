+++
date = '2026-07-09T10:00:00+08:00'
draft = false
title = '从 Pocock Skills 到昇腾诊断：Skill 与 Knowledge 体系设计草案'
categories = ['AIAgent']
tags = ['agent', 'skills', 'claude-code', 'ascend', 'knowledge-base', 'diagnosis', 'matt-pocock']
summary = '以昇腾训练和推理支持场景为背景，综合 Pocock skills 的设计思想，提出一套三层知识架构的 Skill 加 Knowledge 体系设计草案，涉及诊断流程、知识分层、冷启动、紧急模式、团队协作和持续演化。'
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

## 2. 体系概览：三层架构

```
+-----------------------------------------------------+
| Skill 层：诊断流程（低频变更）                         |
| /diagnose-training-issue                             |
| /diagnose-inference-issue                            |
| /to-postmortem                                       |
| /knowledge-groom                                     |
+-----------------------------------------------------+
| Knowledge 层：诊断规则（中频增长）                      |
| Tier 1: triage-tree.yaml    (30 个分支, 极低频)        |
| Tier 2: knowledge/          (200 条 case, 周级增长)    |
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

五条原则提取自 Pocock 体系，适配到昇腾场景。完整对照分析见[上一篇](../matt-pocock-skills/)。

**原则 1：一个 skill 只做一件事，做完就停。**
`/diagnose-training-issue` 的终点是定位 root cause 或标记 need-escalation。

**原则 2：人判断不能被自动化取代，但可以被结构化。**
skill 不替人做诊断决策——它给出结构化验证清单，人执行后把结果贴回来，agent 分析结果后给出下一步。

**原则 3：上下文窗口是有限资源，要显式管理。**
长诊断 session 超过 smart zone 后用 `/handoff` 分段。

**原则 4：知识不绑定在 skill body 里。**
skill body 只写诊断方法论。具体的 case rules 存在 knowledge/ 下，skill 按需加载。改 case 不需要改 skill。

**原则 5：知识沉淀是 skill 的 side effect，不是人的额外负担。**
每条诊断 skill 执行完毕自动生成 postmortem 草稿。人不写文档——agent 写，人审。

### 3.2 `/diagnose-training-issue`

```
流程：
1. 收集症状
   - 错误信息、环境变量（HCCL_*, ASCEND_*, NPU_*）、框架版本、硬件平台
   - 自动检测所属框架：
       pip list | grep -i 'mindspeed|vllm|sglang|verl'
       env | grep -i 'MINDSPEED|VLLM|SGLANG'
     检测到的框架用于决定后续加载哪个 namespace（§4.4）

2. 分类 -> 加载 triage-tree.yaml Tier 1
   - hang / crash -> 搜索 training/<framework>/ + common/hccl + common/npu-driver
   - precision -> 搜索 training/<framework>/ + common/cann
   - performance -> 搜索 training/<framework>/ + common/hardware
   - 框架未检测到 -> 只搜 common/
   - 无法分类 -> Tier 3 向量检索

3. 诊断（加载匹配的 Tier 2 namespace）
   - 最多加载 3 个命名空间（框架桶 + 2 个 common 桶）
   - 对每个 case 执行 quickly_check（primary + fallback）
   - 通过预检的 case 执行完整 diagnosis checks 验证
   - 命中 -> 输出 root cause + fix
   - 未命中 -> 进入深度排查

4. 深度排查（Tier 2 未命中）
   - 收集完整 profiler 数据（如果有）
   - Tier 3 向量检索提供启发式提示
   - 人工/agent 联合分析

5. 产出
   - resolution: resolved / escalated / unknown
   - postmortem 草稿（自动生成——框架已自动填充到对应 namespace）
```

### 3.3 `/to-postmortem`

解决的核心问题：团队成员不一定用同一个 agent 做诊断，因此知识注入入口必须与诊断工具解耦。

```
用法：
  /to-postmortem "[粘贴 Kimi/DeepSeek 的完整对话]
                  [或粘贴纯手工排查笔记]"

流程：
1. 从输入中提取症状、执行的命令和输出、排除的假设、最终 root cause 和 fix
2. agent 自动检测或推断所属框架，给出命名空间建议：
   [1] training/mindspeed-llm/   （检测到 mindspeed-llm）
   [2] training/verl/            （检测到 verl）
   [3] common/                   （跨框架，或不确定）
   人输入一个数字确认 -> 约 5 秒
3. 输出结构化 YAML 草稿 + postmortem.md
   - 标记 confidence: high | medium | low
   - 标记 novelty: new_pattern | variant | covered
   - namespace 已由人确认
4. 人扫一眼确认 -> done（30 秒内可完成）
```

命名空间确认不是额外的管理负担——它是一种轻量的质量检查。当人在 "training/mindspeed-llm/" 和 "common/" 之间选择时，本质上在自问"这个问题是这个框架特有的，还是通用的"。这个自问过程本身就能暴露误判。如果对话原文中包含了 `CANN_OP_DEBUG=1` 和 `HCCL_BUFFSIZE`，但检测到的框架是 `vllm-ascend`，人在确认时会犹豫——"等一下，这看起来像 HCCL 的问题，为什么放在 vllm-ascend 下？"——这就是一个可能被纠正的错误分类。

### 3.4 `/knowledge-groom`

```
触发：手动运行，建议每周一次

流程：
1. 扫 postmortems/ 中新增且通过审批的 .md
2. 对每个 postmortem：
   +-- 尝试结构化 -> YAML
   +-- 成功 -> 追加到对应 namespace
   +-- 失败 -> 标记 needs-human-review
3. 跨 namespace 去重检测：
   +-- 如果 training/mindspeed-llm/ 和 inference/vllm-ascend/
   |   各有一条 case 指向相同 root cause（如 HCCL buffer undersize）
   +-- 在 common/hccl/ 建一条权威记录
   +-- 两条框架层 case 加 references 字段指向它
   +-- 框架层保留框架特有的诊断步骤，
   |   common 层只存 root cause 和 base fix
4. namespace 拆分检测：
   +-- 当某个 namespace 积累到超过 30 条 case 时
   +-- groom 报告中标注该 namespace 的内容分布
   +-- 给出拆分为子 namespace 的建议
   +-- 只有这时才建子目录——不是事前猜测
5. 合并检测：同一 namespace 内相似 case 对自动提示合并
6. 产出 PR：变更列表 + 合并建议 + 去重建议 + 需人工补充的项
```

---

## 4. Knowledge 层设计

### 4.1 命名空间设计原则

命名空间的分割维度只有一个约束：**在 case 创建时就能确定的东西可以做分割，需要诊断完成后才能确定的东西留给 groom 去整理**。

- **框架名**——session 开始 30 秒内就能确定（`pip list | grep mindspeed` 的输出是客观事实）。创建 case 时即可分到对应 namespace。
- **root cause 层**（这个问题出在 CANN 还是 HCCL 还是 NPU driver 层？）——这是诊断的目标，不是诊断的输入。诊断完成后才能确定。所以不做 `cann/`、`hccl/` 这类预分割。等到同一个 root cause 在多个框架 namespace 中重复出现时，groom 把它抽取到 `common/` 下。

### 4.2 目录结构

初始 namespace 平铺——框架层和共享层。不做任何预分割子目录。等单个 namespace 积累到超过 30 条 case 时，groom 在报告中给出拆分建议，这时才建子目录。

```
knowledge/
+-- training/
|   +-- mindspeed-llm/      # /to-postmortem 检测或交互确认 -> 进这里
|   +-- mindspeed-mm/
|   +-- verl/
+-- inference/
|   +-- vllm-ascend/
|   +-- sglang/
+-- common/                  # 框架检测失败 -> 兜底
|                           # groom 发现多框架共用 -> 从框架层提升
+-- platforms/
    +-- a2-910a.md           # 平台差异背景知识
    +-- a3-910b.md
    +-- a5-910c.md
```

### 4.3 `common/` 与框架层的引用关系

框架层的 case 可以包含框架特有的诊断步骤（日志路径、环境变量前缀、配置文件名），但在 root cause 层面引用 `common/` 中的权威记录：

```yaml
# training/mindspeed-llm/ep_hang.yaml
cases:
  - id: MSLLM-EP-HANG-001
    title: "EP dispatch timeout due to HCCL buffer undersize"
    symptoms:
      - "all_to_all_single hangs at step after 3000"

    references: common/hccl/buffer_config.yaml#ASCEND-HCCL-BUFFER-001

    diagnosis:
      - step: 1
        command: "grep 'all_to_all' /path/to/mindspeed_llm/logs/rank_0.log"
        expected: "regex:timeout"

    root_cause: "same as referenced case"
    fix: "HCCL_BUFFSIZE=4194304. See referenced case for details and rollback."
```

`references` 是可选的——只有当 root cause 已被确认是多框架共用的底层问题时才填。框架层保留自身的诊断步骤，`common/` 只存 root cause 描述和 base fix。修改一次 `common/`，所有引用它的框架层 case 自动同步。

### 4.4 Tier 1: triage-tree.yaml（分类路由）

triage-tree 的每条分支指向多个 namespace，按顺序搜索。`<detected_framework>` 是 §3.2 步骤 1 中自动检测的结果——不需要人手动声明。

```yaml
branches:
  - id: training_hang
    symptoms:
      - "timeout" | "hang" | "stuck at step"
      - "NCCL.*timeout" | "HCCL.*timeout"
      - "all_to_all.*timeout"
    search_namespaces:        # 按顺序搜索——最多加载 3 个
      - training/<detected_framework>/    # 先搜框架特定
      - common/                           # 不命中时搜共享层
    fallback: Tier 3

  - id: training_precision
    symptoms:
      - "nan" | "loss.*nan" | "fp8.*precision"
      - "bf16.*mismatch" | "allreduce.*round"
    search_namespaces:
      - training/<detected_framework>/
      - common/
    fallback: Tier 3

  - id: uncategorized
    symptoms: []
    search_namespaces: [common/]
    fallback: Tier 3
```

设计要求：分支数不超过 30。症状模式是正则兼容的模糊匹配。框架检测失败时只用 `common/`。三个 namespace 还不够说明症状太模糊——直接走 Tier 3。

### 4.5 Tier 2: 结构化 case entry

```yaml
cases:
  - id: MSLLM-EP-HANG-001
    title: "HCCL buffer undersize for large-scale EP dispatch"
    priority: high
    platforms: ["A5-910C"]
    frameworks: ["mindspeed-llm>=2.4.0"]

    symptoms:
      - "all_to_all_single hangs at step usually after 1000+"
      - "world_size >= 64"

    references: common/hccl/buffer_config.yaml#ASCEND-HCCL-BUFFER-001

    quickly_check:
      primary:
        command: "grep -c 'all_to_all' /path/to/error.log"
        expected: "regex:^[1-9]"
      fallback:
        command: "grep -ci 'timeout|hang|stuck' /path/to/error.log"
        expected: "regex:^[1-9]"

    diagnosis:
      - step: 1
        command: "env | grep HCCL_BUFFSIZE"
        expected: ">= 4194304"
        fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"
        rollback: "unset HCCL_BUFFSIZE"

    next_on_fail: "MSLLM-EP-HANG-003"

    root_cause: "HCCL internal buffer insufficient for EP all-to-all"
    fix: "export HCCL_BUFFSIZE=4194304 before training launch"
```

关键设计决策：
- `quickly_check` 分为 primary 和 fallback——primary 精确但可能因日志格式变化失效，fallback 更模糊但更鲁棒。primary 不匹配但 fallback 匹配时，仍然进入 diagnosis 但标记 `low_confidence`
- `diagnosis` 是顺序执行，不跳步。任一步 mismatch 且没有 fix_on_mismatch -> 标记该 case 不匹配
- `priority: high` 的 case 优先验证
- `next_on_fail` 可选，指向下一个应该尝试的 case
- `references` 可选，指向 `common/` 中的权威记录——框架层保留诊断步骤，common 层留 root cause 描述

### 4.6 Tier 3: postmortems/

原始诊断记录，不做结构化。仅用于 T1 和 T2 都未命中时的向量检索兜底。文件由 session-end summary 或 `/to-postmortem` 生成，按季度目录归档。

### 4.7 三层检索

| 层 | 内容 | 条目上限 | 加载时机 | token 消耗 |
|---|---|---|---|---|
| Tier 1 | 症状分类与 namespace 路由 | 30 分支 | 始终加载 | ~2K |
| Tier 2 | 结构化 case rules | 200 条 | 症状匹配后加载最多 3 个 namespace | ~10K |
| Tier 3 | 原始 postmortem | 无上限 | T1+T2 未命中时向量检索 | ~5K |

总 token 消耗控制在 17K 以内——即使最坏情况（T1 匹配 + 3 个 namespace + T3 top-3），也不会超过一次诊断 session 推理窗口的 15%。

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

不要期望团队成员额外写一份文档。agent 在 session 结束时自动生成草稿（或人粘贴外部对话后 agent 提取），人只做审批——成本从 20 分钟降到 30 秒。

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
| `CONTEXT.md` | `knowledge/`（按命名空间组织） | Pocock 的是静态术语单文件，我们的是 namespace 分割 + references 引用的诊断知识网络 |
| `/domain-modeling` | `/knowledge-groom` | 定期维护知识结构 |
| postmortem（Pocock 没有） | `postmortems/` | 昇腾场景的核心机制——知识持续增长 |
| `disable-model-invocation` | 诊断 skill 设为 user-only | 诊断决策需要人判断 |
| `ask-matt` 路由器 | `/diagnose-training-issue` 入口 | 统一入口，避免记忆负担 |

---

## 7. 演化机制

体系需要自我约束——不加控制的增长会摧毁检索效率。

| 信号 | 触发动作 |
|------|----------|
| 单个 namespace 积累到超过 30 条 case | groom 报告给出内容分布和子 namespace 拆分建议——这时才建子目录 |
| 两个框架 namespace 各有一条 case 指向相同 root cause | groom 在 `common/` 下创建权威记录，两条框架层 case 添加 `references` 指向它 |
| Tier 2 整体未命中率 > 60% 持续两周 | 审查 triage-tree 的 `search_namespaces` 是否需要扩大 |
| 某个 case 被命中 >= 5 次且每次都直接解决 | 提升 priority = high |
| 某个 case 标记 needs-human-review 超过 30 天 | 提醒领域 owner 处理 |
| 积累 5 条 needs-skill-update 标记的 case | 审查 skill 本身的诊断流程 |

不增长的设计：Tier 2 单个 namespace 上限 30 条（超限触发 groom 拆分建议——不是自动合并，是人工确认后建子目录）、Tier 2 总量上限 200 条（超限强制合并）、Tier 1 上限 30 分支（超限说明分类过细——合并相似分支）、Tier 3 有最小质量阈值（缺少 root_cause 或 symptoms 的记录不进入向量索引）。

---

## 8. 冷启动：第一周怎么活

设计假设 Tier 1 和 Tier 2 都有内容。但团队刚采纳这套体系时，triage-tree 只有两个分支，pattern_library 是空的。工程师跑 `/diagnose-training-issue`，agent 发现 Tier 2 为空，直接跳到 Tier 3 向量检索——但 Tier 3 也是空的。体验是：花了几周搭了整套体系，结果 agent 什么都帮不上。这不是架构问题，是采纳曲线问题。

**方案一：手工播种第一批 case**。在体系上线前，领域 owner 手工挑出过去半年最高频的 10 条 root cause，直接写成 Tier 2 格式的 YAML。不需要等 postmortem 积累到 30 条再升格——跳过 groom 流程，手工播种。这 10 条应覆盖约 50% 的日常 issue（Pareto 原理——20% 的 root cause 对应 80% 的 case）。第一周命中率不要求高，20% 就足够让人感到"这东西有用"。

**方案二：空库提示，不要静默退化**。agent 在第一次运行 `/diagnose-training-issue` 时，如果发现 Tier 2 为空，主动输出提示——

```
当前知识库中还没有经过验证的诊断规则。你可以选择：
1. 继续进行深度排查（跳过自动诊断）
2. 先浏览 CHEATSHEET.md 看是否有手动记录的相关命令
3. 直接转人工诊断（ESC 退出）
```

不要让空知识库的体验是静默退化——明确告诉用户"我还没数据，但我有其他能帮你的方式"。

---

## 9. 紧急模式：生产挂了的时候没人想走流程

当前流程需要 15 到 30 分钟走完收集症状到产出 resolution。生产环境挂了的时候，工程师想的是"先让它跑起来，再分析为什么坏"。这不能被自动化，但可以被承认和设计。

**方案：`/diagnose-*` 内建紧急分支**。在 skill body 中显式写一条：

```
如果用户明确表示"这是紧急情况 / 生产中断 / 需要先恢复服务"：
1. 跳过症状分类和 Tier 2 匹配（不改任何配置）
2. 直接加载 CHEATSHEET.md 的"紧急恢复"部分（如果存在）
3. 输出人类可读的排查清单，每项标注 risk（safe / caution）
4. 不记录 postmortem——等事后手动跑 /to-postmortem
```

紧急排查后的 postmortem 可以由人在事后补齐——知识不会丢，但不会在紧急时刻阻塞人。这和 `/to-postmortem` 的设计哲学一致：知识注入是异步的。

CHEATSHEET.md 的"紧急恢复"部分可以手工维护，不是自动生成的。因为紧急场景通常不是"某个 YAML case 能匹配"——是"先检查最近变更了什么，再检查基础链路通不通，再检查日志最后一段报错"。这条路径是人凭经验走的，但至少可以给它一个可被 agent 读的速查格式。

---

## 10. `quickly_check` 的假阴性风险

`quickly_check` 是整个 Tier 2 匹配的性能关键——在 5 秒内过滤掉不相关的 case。但有一个系统性盲区：如果日志格式因为框架升级而改变，`grep` 匹配不到，即使 root cause 完全正确，这条 case 也会被跳过。这不是概率事件——随着框架版本迭代，这是必然事件。

**方案：分层回退，不是单一 check**。`quickly_check` 不应该是单个命令，而应包含 primary 和 fallback（已在 §4.3 的 schema 中体现）：

```yaml
quickly_check:
  primary:                    # 首选——精确但可能因日志格式变化失效
    command: "grep -c 'all_to_all' /path/to/error.log"
    expected: "regex:^[1-9]"
  fallback:                   # 回退——更模糊但更鲁棒
    command: "grep -ci 'timeout|hang|stuck' /path/to/error.log"
    expected: "regex:^[1-9]"
```

agent 的三段逻辑：先跑 primary。不匹配时跑 fallback。fallback 也不匹配时跳过该 case。如果 primary 不匹配但 fallback 匹配，**仍然进入 diagnosis 但标记 `low_confidence`**——不直接跳过，让人决定要不要试这个 fix。成本是一条额外的命令执行，约 2-5 秒，在诊断场景里完全可接受。

---

## 11. 剩余待解决的问题

| 问题 | 为什么暂时不做 |
|------|--------------|
| 跨团队 knowledge 分叉（MindSpeed-LLM vs vllm-ascend 各自维护 Tier 2） | 等 knowledge/ 积累到超过 100 条再考虑 namespace 设计。当前单仓库够用 |
| case authorship 追踪（谁写的、谁最后验证的） | git blame 已解决——每条 YAML 有 commit history。不需要另建元数据 |
| `/knowledge-groom` 的运行频率（当前建议每周，但可能太频繁） | 试试每周跑一次，如果连续三周发现"没有新 postmortem 需要 groom"，改为双周 |
| 诊断图谱的可视化（`next_on_fail` 长了应该能画出 DAG） | Tier 2 少于 50 条时手动追踪就够了。不建可视化——建了没人用会更尴尬 |

---

## 12. 诊断精度：误诊的代价与回滚

当前设计假设匹配到的 case 就是对的。但现实中可能出现两个问题：

- 两个 case 有几乎相同的症状但完全不同的 root cause——A5 上的 EP hang 和 A3 上的 EP hang 可能都是 `all_to_all timeout`，但一个是 HCCL buffer 问题，一个是 firmware 版本 bug
- agent 可能因为症状模糊匹配到错误的 case，执行了错误的 fix

**层一：fix 标注回滚方式**。每个 `fix_on_mismatch` 应该携带回滚指令——

```yaml
fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"
rollback: "unset HCCL_BUFFSIZE  # 恢复默认值"
```

**层二：串联保护**。如果 agent 连续两次匹配到不同 case（第一次 fix 没解决问题），强制转为人工介入，不再继续尝试第三个 case。这个规则写入 `/diagnose-*` 的 SKILL.md。

**层三：误诊率追踪**。每个 case 被命中但 fix 未解决问题时，标记 `misdiagnosis: true`。这个信号比"未命中率"更关键——高未命中率说明知识库不够大，高误诊率说明知识库有错误信息。追踪方式见 §16 的量化指标。

---

## 13. 平台差异矩阵：字段级而非 case 级

A2、A3、A5 的差异不是三套独立的 case 库——是大量共享规则加少量平台特定差异。比如 HCCL 行为在 A3 和 A5 上几乎相同但 A2 完全不同，FP8 精度问题只在 A5 上出现。

**方案：字段级平台差异**。一个 case 可以有多组 `diagnosis`，分别对应不同平台——

```yaml
diagnosis:
  - platforms: ["A5-910C", "A3-910B"]
    steps:
      - command: "env | grep HCCL_BUFFSIZE"
        expected: ">= 4194304"
        fix_on_mismatch: "export HCCL_BUFFSIZE=4194304"

  - platforms: ["A2-910A"]
    steps:
      - command: "cat /proc/driver/npu/version"
        expected: ">= 23.0"
        note: "A2 上不存在 HCCL_BUFFSIZE 参数。检查 NPU 驱动版本"
```

此外，需要一份 `platforms/` 目录存放平台级背景知识——类似 Pocock 的 CONTEXT.md，但是平台级而非项目级。agent 在加载 Tier 2 bucket 的同时加载平台差异文件，自动选择匹配的 diagnosis 分支。每份文件约 500 字，由领域 owner 维护。

---

## 14. 多跳诊断：显式诊断链

当前的 diagnosis 设计是单跳的——匹配一个 case、执行 checks、命中 root cause、fix。但实际诊断经常是多跳的：中间步骤可能是"修复了但仍然不行"，触发下一个 case 的匹配。

**方案：可选字段 `next_on_fail`**，指向下一个应该尝试的 case id——

```yaml
cases:
  - id: ASCEND-EP-HANG-001
    next_on_fail: "ASCEND-EP-HANG-003"
```

`next_on_fail` 是可选字段，只在已知"修复 X 之后经常需要再检查 Y"的 case 对中使用。它可以逐步从 `/knowledge-groom` 的统计分析中自动生成——如果 groom 发现 case A 被命中后 case B 在同一个 session 中也被命中的概率超过 40%，自动建议建立关联。

---

## 15. Session 中断与恢复

诊断不是连续的时间段。你正在跑 agent 给出的 check 命令，被同事打断去开紧急会议——回来时记不住之前在查什么。agent 在 session 中间的上下文可能已被 compact 了一次，诊断链路丢失。

**方案：诊断状态文件**。`/diagnose-training-issue` 的每个 step 执行后，更新本地 `diagnosis_state.yaml`：

```yaml
session_id: "2026-07-09-ep-hang-a5"
status: in_progress
current_step: 3
excluded_cases: [ASCEND-EP-HANG-001, ASCEND-EP-HANG-002]
active_case: ASCEND-EP-HANG-003
last_action: "等待用户执行 check_ep_topology.py 并贴回输出"
```

当 session 恢复时，agent 首先读这个文件，而不是从头开始收集症状。如果多个工程师在同一个 issue 上协作，他们可以共享状态文件，避免重复排查。

---

## 16. 人可读的速查表：自动生成

不是每个工程师每次都愿意用 agent。有时候只是想搜一条命令。YAML 对 agent 友好但对人完全不行。

**方案：自动生成 CHEATSHEET.md**。`/knowledge-groom` 每次运行时，除了产出 YAML，也产出 Markdown 速查表：

```markdown
## EP Hang

| 检查命令 | 期望值 | 修复方式 | 平台 |
|----------|--------|---------|------|
| `env | grep HCCL_BUFFSIZE` | >= 4194304 | `export HCCL_BUFFSIZE=4194304` | A5, A3 |
| `cat /proc/driver/npu/version` | >= 23.0 | 升级 NPU 驱动 | A2 |
```

三作用：离线可用（agent 挂了或用不了时）、dogfooding 知识质量（生成出来的表看不下去说明 YAML 有问题）、新成员 onboarding。

---

## 17. 数据脱敏

客户发送的日志可能包含 token、API key、集群内部 IP。如果 agent 将这些信息写入 postmortem 并提交到仓库，是安全事故。

**方案**：postmortem 生成时增加 `redact()` 步骤——扫描输出中的 `Bearer ...`、`sk-...`、`password=` 模式并自动替换为 `[REDACTED]`。同时支持 `scope: internal_only` 标记，阻止包含敏感信息的 case 进入公开 knowledge 库。

---

## 18. 量化指标与回顾

六个月后你怎么知道这套体系值得继续投入？不需要 dashboard——一个 Markdown 表格加定期手工更新就够了。

核心指标：命中率（Tier 2 直接匹配解决的比例）、误诊率（命中了但 fix 没解决）、平均诊断时间、知识增长速度、知识覆盖率。

记录方式：`docs/metrics.md`，每两周由体系维护人手工追加一条：

```markdown
## 2026-W28
- 处理 issue 总数: 12
- Tier 2 命中: 7 (58%)
- Tier 2 命中但误诊: 1
- 新增 postmortem: 4 / 成功升格: 2
- 平均诊断时间: ~45min
```

有数字比没有数字重要得多——手工记录足够。关键是定期回顾：体系维护人每两周看一遍 metrics，回答一个问题："这三层架构真的在变好用，还是我们在自欺欺人？"

---

## 19. 接下来的步骤

1. **建 namespace 骨架**——创建 `knowledge/` 目录结构（training/mindspeed-llm、training/verl、inference/vllm-ascend、inference/sglang、common/、platforms/）和 `platforms/*.md` 三份文件。不用写 case——只建空目录和文件。
2. **手工播种第一批 case**——领域 owner 挑出过去半年最高频的 10 条 root cause，直接写成 Tier 2 YAML，放入对应 namespace。跳过 groom 流程。目标：第一周命中率达到 20%。
3. **搭建 triage-tree.yaml 框架**——从团队现有 issue 中提取最高频的 5 到 8 个症状组，配好 `search_namespaces` 的搜索顺序。
4. **搭建 `/to-postmortem` 原型**——选 3 到 5 个真实 case，手工转换为结构化 YAML 作为 reference，写 skill body 和 summary prompt，验证命名空间确认交互的"人均 5 秒选择"假设。
5. **跑第一轮 `/knowledge-groom`**——把现有 20 到 30 个 case 文档过一遍，度量自动结构化的成功率，同时生成第一版 CHEATSHEET.md。
6. **建 `docs/metrics.md`**——体系维护人在第二轮 groom 后开始手工记录，形成两周一次的回看节奏。
7. **Tier 3 向量检索**——等 Tier 2 积累到 50 条以上 case 后，验证未命中率，决定是否引入向量检索兜底。
