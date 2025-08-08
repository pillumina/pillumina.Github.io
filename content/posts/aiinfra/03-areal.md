---
title: "[LLM4RL] 异步RL框架: Areal"
date: 2025-08-07T14:40:12+08:00
tags: ["framework", "LLM", "RL"]
series: ["llm4rl"]
---

> https://github.com/inclusionAI/AReaL
> 纯异步RL方案

## 异步PPO训练调用流程

```mermaid
graph TD
    A[用户执行: examples/run_async_ppo.sh] --> B[training/main_async_ppo.py]
    B --> C[AsyncPPOMATHConfig配置解析]
    C --> D[training/utils.py: run_experiment]
    
    D --> E[Ray初始化]
    E --> F[exp_cfg.initial_setup]
    F --> G[AsyncRLExperimentConfig.initial_setup]
    G --> H[创建ExperimentConfig]
    
    H --> I[启动Workers]
    I --> J[MasterWorker]
    I --> K[ModelWorker]
    I --> L[GenerationServer]
    I --> M[GserverManager]
    I --> N[RolloutWorker]
    
    %% MasterWorker训练流程
    J --> J1[MasterWorker._poll_async]
    J1 --> J2[FunctionExecutor.execute_step]
    J2 --> J3[执行数据流图遍历]
    J3 --> J4[发送训练请求到ModelWorker]
    
    %% ModelWorker处理流程
    K --> K1[ModelWorker._poll]
    K1 --> K2[接收MasterWorker请求]
    K2 --> K3[处理训练/推理请求]
    K3 --> K4[执行模型前向/反向传播]
    
    %% Rollout流程
    N --> N1[RolloutWorker._poll_async]
    N1 --> N2[load_next_data]
    N2 --> N3[allocate_new_rollout]
    N3 --> N4[agent.collect_trajectory]
    N4 --> N5[env.step计算奖励]
    N5 --> N6[推送数据到训练端]
    
    %% 生成服务器流程
    L --> L1[GenerationServer._poll]
    L1 --> L2[启动SGLang子进程]
    L2 --> L3[处理生成请求]
    
    %% 生成服务器管理器
    M --> M1[GserverManager._poll]
    M1 --> M2[HTTP服务线程]
    M2 --> M3[请求调度和权重更新]
    
    %% 数据流
    N6 --> O[stream_dataset.py]
    O --> J4
    
    %% 异步通信
    J4 -.->|异步请求| K2
    N3 -.->|HTTP请求| M2
    M2 -.->|调度请求| L3
    
    %% 权重更新
    K4 --> P[参数更新]
    P --> Q[权重同步]
    Q --> M3
    M3 --> R[更新生成服务器权重]
    
    style A fill:#e1f5fe
    style J fill:#f3e5f5
    style K fill:#e8f5e8
    style L fill:#fff3e0
    style M fill:#fce4ec
    style N fill:#f1f8e9
```


### 用户入口到配置解析

- `examples/run_async_ppo.sh` → `training/main_async_ppo.py`

- 通过Hydra解析CLI参数为`AsyncPPOMATHConfig`

- 调用`initial_setup()`生成`ExperimentConfig`

### Worker启动和初始化

- `training/utils.py:run_experiment()`启动Ray集群

- 根据`scheduling_setup()`创建各类Worker

- 每个Worker执行`_configure()`和`_poll()/_poll_async()`

### 训练端数据流

- `MasterWorker._poll_async()` → `FunctionExecutor.execute_step()`

- 通过`request_reply_stream`发送请求到ModelWorker

- ModelWorker处理训练/推理请求，执行模型计算

###  Rollout端数据流

- `RolloutWorker._poll_async()` → `agent.collect_trajectory()`

- 通过`GserverManager`调度生成请求到`GenerationServer`

- 通过`stream_dataset.py`推送轨迹数据到训练端

###  异步通信机制

- 训练端和Rollout端通过TCP Socket通信

- `GserverManager`提供HTTP API进行请求调度

- 权重更新通过文件系统同步

## 全局架构

### 部署形态
- 进程部署架构
> 以单机8卡为例

 `MasterWorker`：1个CPU进程，协调训练流程
 `ModelWorker`：6个GPU进程（GPU0-5），执行模型训练
 `GenerationServer`：2个GPU进程（GPU6-7），运行SGLang推理服务
 `GserverManager`：1个CPU进程，管理生成服务器
 `RolloutWorker`：多个CPU进程，执行智能体逻辑

### 训推资源分配
> 框架支持**分离部署**和**共享部署**两种模式

#### 分离部署
```bash
┌─────────────────────────────────────────────────────────────┐
│                    Ray Cluster (1 Node, 8 GPUs)             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │MasterWorker │  │ModelWorker  │  │ModelWorker  │         │
│  │   (CPU)     │  │   (GPU0)    │  │   (GPU1)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ModelWorker  │  │ModelWorker  │  │ModelWorker  │         │
│  │   (GPU2)    │  │   (GPU3)    │  │   (GPU4)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ModelWorker  │  │GServerMgr   │  │RolloutWorker│         │
│  │   (GPU5)    │  │   (CPU)     │  │   (CPU)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │GenServer    │  │GenServer    │  │RolloutWorker│         │
│  │ (SGLang)    │  │ (SGLang)    │  │   (CPU)     │         │
│  │   (GPU6)    │  │   (GPU7)    │  └─────────────┘         │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘

```


- 训练端：使用4个GPU（d2p2m1 = 2×2×1）

- 推理端：使用4个GPU（d4p1m1 = 4×1×1）

- 优势：完全解耦，互不干扰，性能最优


### 分层关系

```mermaid
graph TB
    subgraph "用户层"
        A[examples/run_async_ppo.sh]
        B[training/main_async_ppo.py]
    end
    
    subgraph "配置层"
        C[AsyncPPOMATHConfig]
        D[ExperimentConfig]
        E[WorkerConfigs]
    end
    
    subgraph "系统层"
        F[Ray集群管理]
        G[Name Resolution]
        H[日志系统]
    end
    
    subgraph "训练端 Workers"
        I[MasterWorker]
        J[ModelWorker]
        K[FunctionExecutor]
    end
    
    subgraph "Rollout端 Workers"
        L[RolloutWorker]
        M[GenerationServer]
        N[GserverManager]
        O[PartialRolloutManager]
    end
    
    subgraph "核心组件"
        P[Agent接口]
        Q[Environment接口]
        R[Model接口]
        S[Dataset接口]
    end
    
    subgraph "通信层"
        T[Request-Reply Stream]
        U[Push-Pull Stream]
        V[HTTP API]
        W[TCP Socket]
    end
    
    subgraph "模型层"
        X[SGLang Backend]
        Y[PyTorch Backend]
        Z[模型并行]
    end
    
    %% 连接关系
    A --> B
    B --> C
    C --> D
    D --> E
    
    E --> F
    F --> G
    F --> H
    
    E --> I
    E --> J
    E --> L
    E --> M
    E --> N
    
    I --> K
    K --> T
    J --> T
    
    L --> O
    O --> V
    M --> V
    N --> V
    
    L --> P
    L --> Q
    J --> R
    I --> S
    
    T --> W
    U --> W
    V --> W
    
    J --> Y
    M --> X
    Y --> Z
    X --> Z
    
    style A fill:#e3f2fd
    style I fill:#f3e5f5
    style L fill:#e8f5e8
    style T fill:#fff3e0
    style X fill:#fce4ec

```


### 全局类图

```mermaid
classDiagram
    %% 基类层
    class AsyncWorker {
        <<abstract>>
        +_configure(config)
        +_poll_async() PollResult
        +run_async()
    }
    
    class Worker {
        <<abstract>>
        +_configure(config)
        +_poll() PollResult
        +run()
    }
    
    %% Worker实现层 - 训练端
    class MasterWorker {
        -config: MasterWorkerConfig
        -func_executor: FunctionExecutor
        -__poll_async()
        -__lazy_init()
    }
    
    class ModelWorker {
        -config: ModelWorkerConfig
        -__request_queue: Queue
        -_poll()
        -handle_request()
    }
    
    %% Worker实现层 - Rollout端
    class RolloutWorker {
        -config: RolloutWorkerConfig
        -agent: Agent
        -env: Environment
        -_poll_async()
        -rollout_task()
    }
    
    class GenerationServer {
        -config: GenerationServerConfig
        -server_process: Process
        -_poll()
        -launch_server_subprocess()
    }
    
    class GserverManager {
        -config: GserverManagerConfig
        -server_urls: List[str]
        -_poll()
        -_schedule_request()
    }
    
    %% 接口层
    class Agent {
        <<interface>>
        +collect_trajectory(prompt, env, obs_queue, act_queue)
    }
    
    class Environment {
        <<interface>>
        +reset()
        +step(action)
    }
    
    class ModelInterface {
        <<interface>>
        +inference(model, data, mb_spec)
        +generate(model, data, mb_spec)
        +train_step(model, data, mb_spec)
    }
    
    %% 配置层
    class AsyncPPOMATHConfig {
        +agent: AgentAbstraction
        +env: EnvServiceAbstraction
        +initial_setup() ExperimentConfig
        +scheduling_setup() ExperimentScheduling
    }
    
    class ExperimentConfig {
        +model_rpcs: List[ModelRPC]
        +model_worker: ModelWorkerConfig
        +generation_server: GenerationServerConfig
        +rollout_worker: RolloutWorkerConfig
    }
    
    %% 继承关系 - 垂直排列减少交叉
    AsyncWorker <|-- MasterWorker
    AsyncWorker <|-- RolloutWorker
    Worker <|-- ModelWorker
    Worker <|-- GenerationServer
    Worker <|-- GserverManager
    
    %% 组合关系 - 水平连接
    MasterWorker --> ModelInterface : uses
    RolloutWorker --> Agent : uses
    RolloutWorker --> Environment : uses
    ModelWorker --> ModelInterface : implements
    
    %% 配置关系 - 底部连接
    AsyncPPOMATHConfig --> ExperimentConfig : creates
    ExperimentConfig --> MasterWorker : configures
    ExperimentConfig --> ModelWorker : configures
    ExperimentConfig --> RolloutWorker : configures
    ExperimentConfig --> GenerationServer : configures
    ExperimentConfig --> GserverManager : configures
```


### 核心模块类图

```mermaid
classDiagram
    %% 基类
    class AsyncWorker {
        <<abstract>>
        +_poll_async() PollResult
    }
    
    class Worker {
        <<abstract>>
        +_poll() PollResult
    }
    
    %% 训练端Workers
    class MasterWorker {
        -func_executor: FunctionExecutor
        -__poll_async()
    }
    
    class ModelWorker {
        -__request_queue: Queue
        -_poll()
    }
    
    %% Rollout端Workers
    class RolloutWorker {
        -agent: Agent
        -env: Environment
        -_poll_async()
    }
    
    class GenerationServer {
        -server_process: Process
        -_poll()
    }
    
    class GserverManager {
        -server_urls: List[str]
        -_poll()
    }
    
    %% 核心接口
    class Agent {
        <<interface>>
        +collect_trajectory()
    }
    
    class Environment {
        <<interface>>
        +step(action)
    }
    
    class ModelInterface {
        <<interface>>
        +train_step()
        +generate()
    }
    
    %% 配置
    class AsyncPPOMATHConfig {
        +initial_setup()
        +scheduling_setup()
    }
    
    %% 继承关系
    AsyncWorker <|-- MasterWorker
    AsyncWorker <|-- RolloutWorker
    Worker <|-- ModelWorker
    Worker <|-- GenerationServer
    Worker <|-- GserverManager
    
    %% 关键关系
    MasterWorker --> ModelInterface
    RolloutWorker --> Agent
    RolloutWorker --> Environment
    ModelWorker --> ModelInterface
    AsyncPPOMATHConfig --> MasterWorker
    AsyncPPOMATHConfig --> ModelWorker
    AsyncPPOMATHConfig --> RolloutWorker
```




## 异步流程机制细节

### 异步完整流程图
```mermaid
sequenceDiagram
    participant User as 用户
    participant MW as MasterWorker
    participant RW as RolloutWorker
    participant GS as GenerationServer
    participant GSM as GserverManager
    participant ZMQ as ZMQ Stream
    participant SD as StreamDataset
    participant MW2 as ModelWorker
    participant NR as NameResolving
    participant FS as 文件系统

    Note over User: 启动异步PPO训练
    User->>User: examples/run_async_ppo.sh<br/>输入：GPU数量、并行策略、模型路径
    User->>MW: training/main_async_ppo.py<br/>输入：AsyncPPOMATHConfig

    Note over MW: 初始化阶段
    MW->>MW: run_experiment(config)<br/>输入：实验配置
    MW->>MW: initial_setup()<br/>输入：worker配置
    MW->>NR: 注册各Worker地址<br/>变量：worker_info, msid2mwid
    Note over MW: 设置版本差异控制参数<br/>变量：max_head_offpolicyness

    Note over RW,GS: Rollout端启动
    RW->>RW: _configure(config)<br/>输入：RolloutWorkerConfig
    RW->>ZMQ: 初始化NameResolvingZmqPusher<br/>变量：experiment_name, trial_name, worker_index
    GS->>GS: _configure(config)<br/>输入：GenerationServerConfig
    GS->>GS: 初始化SGLang后端<br/>变量：model_path, tokenizer_path
    GSM->>GSM: _configure(config)<br/>输入：GserverManagerConfig
    GSM->>GSM: 初始化权重版本跟踪<br/>变量：_last_param_realloc_step

    Note over MW2: 训练端启动
    MW2->>MW2: _configure(config)<br/>输入：ModelWorkerConfig
    MW2->>SD: 初始化PullerStreamDataset<br/>变量：dataset_size, pull_timeout_ms
    MW2->>MW2: 初始化模型和优化器<br/>变量：model_config, optimizer_config

    Note over MW: 训练循环开始
    MW->>MW: __poll_async()<br/>输入：训练控制参数
    MW->>MW: func_executor.execute_step()<br/>输入：数据流图

    Note over RW,GS: 并行生成轨迹
    loop 持续生成轨迹
        RW->>GS: 发送生成请求<br/>输入：prompt, max_tokens
        Note over GS: 使用当前加载的权重版本<br/>变量：current_model_version
        GS->>GS: SGLang生成<br/>输入：模型权重、生成参数
        GS-->>RW: 返回生成结果<br/>输出：generated_text
        RW->>RW: agent.collect_trajectory()<br/>输入：生成结果
        RW->>RW: 计算奖励、构建轨迹<br/>变量：trajectory, reward
        Note over RW: 为轨迹添加版本信息<br/>变量：trajectory.model_version = current_model_version
        RW->>ZMQ: push_stream.push(traj)<br/>输入：轨迹数据(JSON格式)
    end

    Note over ZMQ,SD: 数据传递
    ZMQ->>SD: 接收轨迹数据<br/>输入：JSON序列化数据
    SD->>SD: _pull_data_worker()<br/>后台线程持续拉取
    SD->>SD: 转换为SequenceSample<br/>变量：data_queue, processed_data

    Note over MW2: 训练执行 - 版本差异控制
    MW2->>SD: 获取训练数据<br/>输入：batch_size
    SD-->>MW2: 返回SequenceSample<br/>输出：训练样本
    Note over MW2: 检查数据版本差异<br/>变量：data_version, current_version, max_head_offpolicyness
    MW2->>MW2: validate_data_version(data_version, current_version)<br/>输入：数据版本、当前版本、最大允许差异
    alt 版本差异在允许范围内
        Note over MW2: 接受数据，继续训练<br/>变量：version_diff <= max_head_offpolicyness
        MW2->>MW2: train_step(data)<br/>输入：训练数据、优化器状态
        MW2->>MW2: 计算PPO损失<br/>变量：policy_loss, value_loss, entropy_loss
        MW2->>MW2: 更新模型参数<br/>变量：optimizer.step(), global_step
    else 版本差异过大
        Note over MW2: 丢弃过期数据<br/>变量：version_diff > max_head_offpolicyness
        MW2->>MW2: discard_stale_data(data)<br/>输入：过期数据
        Note over MW2: 记录数据丢弃统计<br/>变量：stale_data_count++
        MW2->>SD: 请求新的训练数据<br/>输入：batch_size
    end

    Note over MW2,FS: 权重同步 - 版本控制
    MW2->>FS: __save_model(save_meta)<br/>输入：model, save_dir, global_step
    Note over FS: 保存权重分片到磁盘<br/>变量：param_realloc_path/model_name/step/
    MW2->>NR: name_resolve.add(model_version, global_step)<br/>输入：experiment, trial, model_name, step
    Note over NR: 原子性更新版本号<br/>变量：model_version = global_step

    Note over GSM,GS: 推理端权重更新 - 数据陈旧性控制
    loop 定期检查新权重
        GSM->>NR: check_new_params()<br/>输入：experiment, trial, model_name
        NR-->>GSM: 返回最新global_step
        alt 有新权重版本
            GSM->>GSM: 发现版本更新<br/>变量：realloc_version > _last_param_realloc_step
            GSM->>GS: flush_requests_and_update_weights(load_dir)<br/>输入：新权重路径
            Note over GS: 中断当前所有推理请求<br/>变量：_interrupt_requests()
            GS->>FS: 读取新权重文件<br/>输入：load_dir
            GS->>GS: update_weights_from_disk(load_dir)<br/>变量：分片rank, 权重文件
            Note over GS: 按TP/PP分片加载权重<br/>变量：新model_version生效
            GS-->>GSM: 权重更新完成
            GSM->>GSM: 更新版本跟踪<br/>变量：_last_param_realloc_step = realloc_version
            Note over GS: 恢复推理服务<br/>变量：使用新权重版本
        else 无新权重
            GSM->>GSM: 继续使用当前权重<br/>变量：_last_param_realloc_step
        end
    end

    Note over MW: 训练控制 - 版本差异监控
    MW->>MW: 检查训练终止条件<br/>输入：epoch, global_step, loss
    MW->>MW: 监控版本差异统计<br/>变量：stale_data_count, version_diff_stats
    Note over MW: 记录版本差异对训练的影响<br/>变量：training_efficiency, data_freshness
    alt 继续训练
        MW->>MW: 更新训练状态<br/>变量：step_info, epoch_step
        Note over MW: 返回训练循环开始
    else 训练完成
        MW->>User: 训练结束<br/>输出：最终模型、训练日志、版本差异统计
    end
```

### 异步带来的算法修正

#### 同步PPO完整流程
先回顾一下ppo的计算流程：
> 我们有一个策略π(a|s)，它决定在状态s下选择动作a的概率。PPO的目标是优化这个策略，使其能够获得更高的累积奖励。

- 数据收集（rollout）
```python
# 使用当前策略π_θ生成轨迹
for episode in range(num_episodes):
    state = env.reset()
    trajectory = []
    
    while not done:
        # 使用当前策略选择动作
        action_probs = π_θ(state)  # 当前策略的概率分布
        action = sample(action_probs)  # 采样动作
        
        # 记录动作概率（用于后续计算重要性比率）
        old_logp = log(action_probs[action])  # 这就是old_logp
        
        # 执行动作
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward, old_logp))
        state = next_state
```

- 计算优势函数
```python
# 使用GAE计算优势函数
advantages = compute_gae(trajectory, γ=0.99, λ=0.95)
returns = compute_returns(trajectory, γ=0.99)
```

- 策略更新
```python
# 对收集的数据进行多次更新
for epoch in range(num_epochs):
    for batch in data_loader:
        # 重新计算当前策略的概率
        current_action_probs = π_θ(batch.states)  # 当前策略
        cur_logp = log(current_action_probs[batch.actions])  # 这就是cur_logp
        
        # 计算重要性比率
        ratio = exp(cur_logp - old_logp)
        
        # PPO损失函数
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        loss = -min(surr1, surr2)
        
        # 更新策略参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```


> 为什么需要重要性采样   ratio = π_θ(a|s) / π_θ_old(a|s) = exp(cur_logp - old_logp)
> 1.  我们想用当前策略π_θ来评估旧策略π_θ_old生成的数据
> 2. 重要性采样修正了这种分布偏移


#### 框架的异步PPO修正机制

- 异步带来的问题，数据生成和训练并行
```python
# 时间线
t=0: 策略π_θ_0生成数据
t=1: 策略π_θ_1生成数据，同时训练π_θ_0的数据
t=2: 策略π_θ_2生成数据，同时训练π_θ_1的数据
...
```
这导致：

- 训练数据来自较旧的策略版本

- 重要性比率可能变得很大或很小

- 策略更新可能不稳定

> 框架引入的修正机制如下：

**机制1： 版本控制**
```python
# 记录数据生成时的策略版本
data = {
    "version_start": model_version_when_generation_started,
    "version_end": model_version_when_generation_ended,
    "old_logp": logprobs_from_generation,
    "actions": actions,
    "rewards": rewards
}
```


**机制2：数据过滤**
```python
# 检查版本差异
version_diff = current_version - data.version_start
if version_diff > max_head_offpolicyness:
    # 数据太旧，丢弃
    continue
```

**机制3：解耦损失（Decoupled Loss）**
```python
# 标准PPO损失
def standard_ppo_loss(cur_logp, old_logp, advantages):
    ratio = exp(cur_logp - old_logp)
    return -min(ratio * advantages, clip(ratio, 1-ε, 1+ε) * advantages)

# AReaL解耦损失
def decoupled_loss(cur_logp, old_logp, prox_logp, advantages):
    # 使用prox_logp作为中间策略
    ratio = exp(cur_logp - prox_logp)
    behav_weight = exp(prox_logp - old_logp)
    return -min(ratio * advantages, clip(ratio, 1-ε, 1+ε) * advantages) * behav_weight
```


#### 修正的合理性分析

**数学基础**
解耦损失可以分解为:
```python
# 标准PPO
ratio = π_θ(a|s) / π_θ_old(a|s)

# AReaL解耦
ratio = π_θ(a|s) / π_prox(a|s)
behav_weight = π_prox(a|s) / π_θ_old(a|s)

# 等价性
ratio * behav_weight = π_θ(a|s) / π_θ_old(a|s)  # 与标准PPO相同
```

**稳定性提升**
```python
# 异步场景下的问题
# 如果π_θ与π_θ_old差异很大
ratio = π_θ(a|s) / π_θ_old(a|s)  # 可能很大或很小

# AReaL的解决方案
# 引入中间策略π_prox，使得：
# π_θ ≈ π_prox ≈ π_θ_old
ratio = π_θ(a|s) / π_prox(a|s)  # 更稳定
behav_weight = π_prox(a|s) / π_θ_old(a|s)  # 更稳定
```

**渐进式更新**
```python
# 标准异步PPO：直接从π_θ_old跳到π_θ
# AReaL：π_θ_old → π_prox → π_θ，分两步更新
```


#### 具体实现
**核心修正机制实现**
```python
# AReaL的解耦损失实现
if proximal_logprobs is not None:
    # 计算行为策略权重
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    
    # 应用权重上限
    if behav_imp_weight_cap is not None:
        behav_mask = (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
    else:
        behav_mask = loss_mask
    
    # 应用行为策略权重
    pg_loss = pg_loss * behav_imp_weight
```

**数学等价性证明**

```python
# 标准PPO损失
L_standard = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
其中 ratio = π_θ(a|s) / π_θ_old(a|s)

# AReaL解耦损失
L_decoupled = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) * behav_weight
其中 ratio = π_θ(a|s) / π_prox(a|s)
     behav_weight = π_prox(a|s) / π_θ_old(a|s)

# 等价性证明
L_decoupled = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) * behav_weight
           = -min((π_θ/π_prox) * A, clip(π_θ/π_prox, 1-ε, 1+ε) * A) * (π_prox/π_θ_old)
           = -min((π_θ/π_θ_old) * A, clip(π_θ/π_prox, 1-ε, 1+ε) * A * (π_prox/π_θ_old))
```



#### 流程图视角
```mermaid
graph TD
    %% 生成阶段 - SGLang推理服务
    A[用户Prompt<br/>packed_prompts] --> B[SGLang推理服务<br/>actor_gen]
    B --> B1[PPOActorInterface.generate<br/>使用策略π_θ_old]
    B1 --> B2[模型前向传播<br/>genstep函数]
    B2 --> B3[采样token<br/>计算logprob]
    B3 --> B4[concat_prompt_to_generation_output<br/>拼接prompt和生成结果]
    B4 --> B5[输出: packed_input_ids<br/>packed_logprobs<old_logp><br/>prompt_mask<br/>seq_no_eos_mask]
    
    %% 推理阶段 - 四个组件并行执行
    B5 --> C[推理阶段开始]
    
    %% Actor推理 - 计算proximal_logp
    C --> D[actor_inf<br/>PPOActorInterface.inference<br/>使用策略π_θ_prox]
    D --> D1[输入: packed_input_ids]
    D1 --> D2[calc_logprobs post_hook<br/>gather_packed_shifted_log_probs]
    D2 --> D3[输出: proximal_logprobs<br/>π_θ_prox<a,s>]
    
    %% Reference推理 - 计算ref_logp
    C --> E[ref_inf<br/>PPOActorInterface.inference<br/>使用策略π_ref]
    E --> E1[输入: packed_input_ids]
    E1 --> E2[calc_logprobs post_hook<br/>gather_packed_shifted_log_probs]
    E2 --> E3[输出: packed_ref_logprobs<br/>π_ref<a,s>]
    
    %% Critic推理 - 计算values
    C --> F[critic_inf<br/>PPOCriticInterface.inference<br/>使用价值网络V_θ]
    F --> F1[输入: packed_input_ids<br/>seq_no_eos_mask]
    F1 --> F2[module.forward<br/>直接输出value]
    F2 --> F3[输出: values<br/>V_θ<s>]
    
    %% Reward推理 - 计算rewards
    C --> G[rew_inf<br/>MultiTaskRewardInterface.inference<br/>使用奖励函数R]
    G --> G1[输入: packed_input_ids<br/>packed_prompts<br/>task_ids]
    G1 --> G2[calculate_task_reward<br/>异步任务处理]
    G2 --> G3[输出: rewards<br/>R<s,a>]
    
    %% 数据汇聚
    D3 --> H[推理结果汇聚]
    E3 --> H
    F3 --> H
    G3 --> H
    
    %% 训练阶段准备
    H --> I[训练数据准备<br/>packed_input_ids<br/>packed_logprobs<old_logp><br/>packed_ref_logprobs<br/>proximal_logprobs<br/>rewards<br/>values<br/>prompt_mask<br/>seq_no_eos_mask]
    
    %% 训练阶段 - 计算current_logp和loss
    I --> J[actor_train<br/>PPOActorInterface.train_step<br/>使用策略π_θ]
    J --> J1[模型前向传播<br/>module.forward]
    J1 --> J2[gather_packed_shifted_log_probs<br/>计算current_logp<br/>π_θ<a,s>]
    J2 --> J3[计算advantages<br/>GAE算法]
    J3 --> J4[计算rewards<br/>KL正则化]
    J4 --> J5[PPO Loss计算<br/>_ppo_actor_loss_from_model_outputs]
    
    %% PPO Loss详细计算
    J5 --> K[PPO Loss计算详情]
    K --> K1[输入: current_logp, old_logp, proximal_logp<br/>advantages, rewards]
    K1 --> K2{use_decoupled_loss?}
    K2 -->|是| K3[解耦损失计算<br/>ratio = exp<current_logp - proximal_logp><br/>behav_weight = exp<proximal_logp - old_logp>]
    K2 -->|否| K4[标准损失计算<br/>ratio = exp<current_logp - old_logp>]
    K3 --> K5[最终损失<br/>loss = -min ratio * advantages, clip ratio * advantages * behav_weight]
    K4 --> K6[最终损失<br/>loss = -min ratio * advantages, clip ratio * advantages]
    K5 --> L[输出: Actor Loss]
    K6 --> L
    
    %% Critic训练
    J3 --> M[critic_train<br/>PPOCriticInterface.train_step<br/>使用价值网络V_θ]
    M --> M1[模型前向传播<br/>计算new_values]
    M1 --> M2[Critic Loss计算<br/>_ppo_critic_loss_from_model_outputs]
    M2 --> M3[输出: Critic Loss]
    
    %% 样式定义
    classDef generateStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef inferenceStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef trainStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef lossStyle fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    
    class B,B1,B2,B3,B4,B5 generateStyle
    class D,D1,D2,D3,E,E1,E2,E3,F,F1,F2,F3,G,G1,G2,G3 inferenceStyle
    class J,J1,J2,J3,J4,J5,M,M1,M2,M3 trainStyle
    class K,K1,K2,K3,K4,K5,K6,L lossStyle
```


**old_logp (π_θ_old)**
```python
# 生成阶段 - SGLang推理服务
# 模型：Actor模型 (策略π_θ_old)
# 时机：生成token时实时计算
# 函数：genstep() -> distrb.log_prob(next_tokens)
# 保存：concat_prompt_to_generation_output() -> packed_logprobs
```



**proximal_logp (π_θ_prox)**
```python
# 推理阶段 - actor_inf组件
# 模型：Actor模型 (策略π_θ_prox，比π_θ_old新，比π_θ旧)
# 时机：生成完成后，训练前
# 函数：PPOActorInterface.inference() -> calc_logprobs()
# 条件：仅当use_decoupled_loss=True时计算
```



**current_logp (π_θ)**
```python
# 训练阶段 - actor_train组件
# 模型：Actor模型 (当前策略π_θ，最新)
# 时机：训练时重新计算
# 函数：PPOActorInterface.train_step() -> gather_packed_shifted_log_probs()
# 作用：用于计算重要性采样比率
```


### 权重同步机制

```mermaid
sequenceDiagram
    participant MW as ModelWorker
    participant FS as 文件系统
    participant NR as NameResolving
    participant GSM as GserverManager
    participant GS as GenerationServer

    Note over MW: 训练完成一次step后
    MW->>MW: __save_model(save_meta)<br/>输入：model, save_dir, global_step
    MW->>FS: 保存权重文件<br/>路径: param_realloc_path/model_name/step/
    Note over MW,FS: 权重以分片形式落盘（TP/PP分片）

    MW->>NR: name_resolve.add(model_version, global_step)<br/>输入：experiment, trial, model_name, step
    NR-->>GSM: model_version更新

    loop 推理端定期检查
        GSM->>NR: check_new_params()<br/>输入：experiment, trial, model_name
        NR-->>GSM: 返回最新global_step
        alt 有新权重
            GSM->>FS: 获取新权重路径
            GSM->>GS: flush_requests_and_update_weights(load_dir)
            GS->>FS: 读取权重分片文件
            GS->>GS: update_weights_from_disk(load_dir)<br/>变量: load_dir, 分片rank
            Note over GS: 按TP/PP分片加载到各自分片
        else 无新权重
            GSM->>GSM: 不做更新
        end
    end

    Note over GS: 新权重生效，推理端继续服务
```


核心机制：

- 训练端：`ModelWorker`在每次`train_step`后保存权重到`param_realloc_path`，并调用`name_resolve.add(model_version, global_step)`，在`NameResolving`服务中记录最新的权重版本号（global_step）。

- 推理端：`GserverManager`定期检查`model_version`，发现新版本（`model_version`和已经加载的对比）时通过HTTP API更新所有`GenerationServer`的权重。
s
- 同步动作：权重更新时会中断正在进行的生成请求，确保推理使用最新权重。


关键函数与变量说明：

- `__save_model(save_meta)`

- 输入：model_name, save_dir, global_step

- 输出：权重文件（分片）落盘

- `name_resolve.add(model_version, global_step)`

- 输入：实验名、trial名、模型名、step

- 输出：NameResolving服务中记录最新step

- `check_new_params()`

- 输入：实验名、trial名、模型名

- 输出：最新step（如果有更新）

- `flush_requests_and_update_weights(load_dir)`

- 输入：权重目录

- 输出：推理端各分片加载新权重

- `update_weights_from_disk(load_dir)`

- 输入：分片rank、load_dir

- 输出：各分片权重加载到内存

变量传递链路：

- `global_step/model_version`：用于标识权重版本

- `param_realloc_path/load_dir`：权重磁盘路径

- 分片rank：决定每个worker加载哪一份权重



### 数据陈旧性控制
> 异步训推协调的核心机制，需要限制陈旧性保证训练稳定性

```python
# GserverManager中的陈旧性检查
def is_staled(self):
    # 检查当前运行的rollout是否过时
    return self.rollout_stat.running > self.config.max_head_offpolicyness
```

协调机制：

- 版本控制：每个生成请求都携带version_start和version_end，记录使用的权重版本

- 陈旧性限制：通过`max_head_offpolicyness`参数控制允许的最大数据陈旧性

- 请求调度：`GserverManager`在分配新rollout时检查容量和陈旧性，拒绝过时的请求

确实存在使用老权重的情况：

- 异步训练允许一定程度的权重陈旧性

- 通过`max_head_offpolicyness`参数控制陈旧性上限

- 这种设计在提高训练效率的同时，通过限制陈旧性保证训练稳定性



### 数据传递机制

> 各个worker之间的通信核心是ZMQ：
> - 高性能：支持零拷贝和批量传输
> - 多种模式：PUSH/PULL、PUB/SUB、REQ/REP等
> - 异步通信：非阻塞I/O，适合高并发场景
> - 跨语言：支持多种编程语言
> - 网络透明：自动处理连接、重连、负载均衡

```python
# zmq的配置举例
# 高性能配置
self.context = zmq.Context.instance(io_threads=8)
self.context.set(zmq.MAX_SOCKETS, 65536)

# 缓冲区优化
self.socket.setsockopt(zmq.SNDHWM, 1000)  # 发送缓冲区
self.socket.setsockopt(zmq.RCVHWM, 1000)  # 接收缓冲区

# 超时设置
self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
```


```mermaid
sequenceDiagram
    participant RW as RolloutWorker
    participant GS as GenerationServer
    participant GSM as GserverManager
    participant ZMQ as ZMQ Stream
    participant SD as StreamDataset
    participant MW as ModelWorker
    participant DM as DataManager
    
    Note over RW,DM: 1. 生成轨迹数据
    RW->>GS: 发送生成请求
    GS->>GS: SGLang生成结果
    GS->>RW: 返回生成结果
    RW->>RW: 计算奖励，构建轨迹
    
    Note over RW,DM: 2. 推送数据到训练端
    RW->>ZMQ: 推送轨迹数据(JSON格式)
    ZMQ->>SD: 接收数据
    SD->>SD: 转换为SequenceSample
    
    Note over RW,DM: 3. 训练端处理数据
    SD->>MW: 提供数据给ModelWorker
    MW->>DM: 存储到DataManager(内存)
    MW->>MW: 执行训练步骤
```



#### 数据传递层次

1. Rollout端到训练端：

- 使用ZMQ Push-Pull Stream传输轨迹数据

- RolloutWorker → NameResolvingZmqPusher → NameResolvingZmqPuller → StreamDataset

```mermaid
graph TB
    subgraph "Rollout端"
        RW[RolloutWorker] --> NP[NameResolvingZmqPusher]
        NP --> ZMQ1[ZMQ PUSH Socket]
    end
    
    subgraph "训练端"
        ZMQ2[ZMQ PULL Socket] --> NP2[NameResolvingZmqPuller]
        NP2 --> SD[StreamDataset]
        SD --> MW[ModelWorker]
    end
    
    subgraph "Name Resolution"
        NR[name_resolve系统]
    end
    
    ZMQ1 -.->|TCP连接| ZMQ2
    NP --> NR
    NP2 --> NR
```

2. 训练端内部：

- 使用Request-Reply Stream传输训练请求

- MasterWorker → ModelWorker通过ZMQ通信

```mermaid
graph TB
    subgraph "MasterWorker"
        MW[MasterWorker] --> NRC[NameResolvingRequestClient]
        NRC --> ZMQ1[ZMQ PUSH Sockets]
        ZMQ2[ZMQ PULL Socket] --> NRC
    end
    
    subgraph "ModelWorker"
        ZMQ3[ZMQ PULL Socket] --> NRS[NameResolvingReplyServer]
        NRS --> MW2[ModelWorker]
        MW2 --> ZMQ4[ZMQ PUSH Socket]
    end
    
    subgraph "通信协议"
        REQ[Request] --> ACK[ACK]
        ACK --> SYN[SYN]
        SYN --> RESP[Response]
    end
    
    ZMQ1 -.->|TCP| ZMQ3
    ZMQ4 -.->|TCP| ZMQ2
```

```python
# 请求发送
def request(self, handlers, handle_type, datas, no_syn=True):
    requests = [
        Payload(
            handler=handler,
            handle_name=handle_type,
            data=data,
            no_syn=no_syn,
        )
        for handler, data in zip(handlers, datas)
    ]
    
    # 发送请求
    for payload in requests:
        idx = self._handler_routing[payload.handler]
        self.send_sockets[idx].send(pickle.dumps(payload))
```


3. 存储分离：

- 训练数据：存储在DataManager中，支持分布式存储和重分布
   `DataManager`为内存存储：
```python
class DataManager:
    def __init__(self, model_topos, msid2mwid, data_transfer_pairs):
        # 核心存储：内存字典
        self.storage: Dict[Hashable, SequenceSample] = {}
        
    def store(self, x: SequenceSample):
        # 存储到内存字典
        self.storage[x.ids[0]] = x
        
    def get(self, data_id: Hashable):
        # 从内存获取
        return self.storage[data_id]
```

支持数据重分布：
```python
def redistribute(self, data_info: SequenceSample, plan: List[RedistribStep]):
    """执行数据重分布"""
    for step in plan:
        if step.comm_type == "bcast":
            self._run_bcast(step, data_infos)
        elif step.comm_type == "gather":
            self._run_gather(step, data_infos)
        elif step.comm_type == "scatter":
            self._run_scatter(step, data_infos)
```


- 推理数据：存储在SGLang服务器的内存中

- 元数据：通过name_resolve系统共享


#### 实现细节
##### `RolloutWorker` 数据发送

```python
# realhf/system/rollout_worker.py
class RolloutWorker(AsyncWorker):
    def _configure(self, config):
        # 初始化ZMQ推送器 - 发送轨迹数据到训练端
        self.push_stream = NameResolvingZmqPusher(
            self.experiment_name,
            self.trial_name, 
            pusher_index=self.worker_index,
            pusher_cnt=self.worker_count,
        )
    
    async def _poll_async(self):
        # 收集轨迹数据
        traj = await self.agent.collect_trajectory()
        
        # 推送数据到训练端
        self.push_stream.push([traj.as_json_serializable()])
```


##### `GenerationServer` 推理服务

```python
# realhf/system/generation_server.py  
class GenerationServer(Worker):
    def launch_server_subprocess(self):
        # 启动SGLang推理服务器
        self.server_process, self.server_port = launch_server_cmd(cmd, port=server_port)
        self.server_addr = f"http://{host}:{self.server_port}"
        
        # 注册服务地址到NameResolving
        name = names.gen_servers(self.experiment_name, self.trial_name)
        name_resolve.add_subentry(name, self.server_addr)
```


##### `GserverManager`负载均衡

```python
# realhf/system/gserver_manager.py
class GserverManager(Worker):
    def _discover_servers(self, n_servers: int):
        # 通过NameResolving发现所有推理服务器
        name = names.gen_servers(self.experiment_name, self.trial_name)
        urls = name_resolve.get_subtree(name)
        return urls
    
    def _run_routing_service(self):
        # HTTP服务，接收推理请求并路由到合适的服务器
        async def schedule_request(req_meta):
            server_idx = self._least_requests_schedule(req_meta)
            return self.server_urls[server_idx]
```

##### `MasterWorker` 训练协调

```python
# realhf/system/master_worker.py
class MasterWorker(AsyncWorker):
    def _configure(self, config):
        # 初始化Request-Reply客户端
        self.func_executor = FunctionExecutor(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            n_subscribers=self.config.n_model_workers,
            handler_routing=self.config.handler_routing,
        )
    
    async def _poll_async(self):
        # 执行训练步骤，通过Request-Reply与ModelWorker通信
        result = await self.func_executor.execute_step(
            step_name="train_step",
            step_kwargs={"batch": batch}
        )
```

##### `ModelWorker` 模型训练
```python
# realhf/system/model_worker.py
class ModelWorker(Worker):
    def _configure(self, config):
        # 初始化Request-Reply服务器
        self.reply_server = NameResolvingReplyServer(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            idx=self.worker_index,
        )
        
        # 注册训练处理函数
        self.reply_server.register_handler("train_step", self._train_step)
    
    def _train_step(self, batch):
        # 执行训练步骤
        loss = self.model.train_step(batch)
        
        # 保存权重并更新版本号
        self.model.save_weights(self.param_realloc_path)
        name = names.model_version(self.experiment_name, self.trial_name, self.model_name.role)
        name_resolve.add(name, self.global_step)
        
        return {"loss": loss, "global_step": self.global_step}
```


##### `StreamDataset` 数据接收

```python
# realhf/system/stream_dataset.py
class StreamDataset:
    def __init__(self, args, puller_index):
        # 初始化ZMQ拉取器 - 接收RolloutWorker推送的数据
        self.puller = NameResolvingZmqPuller(args, puller_index)
    
    def __iter__(self):
        while True:
            # 从ZMQ接收数据
            data = self.puller.pull()
            
            # 转换为训练格式
            sample = SequenceSample.from_json_serializable(data)
            yield sample
```


##### `ZMQ`通信层

```python
# realhf/system/push_pull_stream.py
class NameResolvingZmqPusher(ZMQJsonPusher):
    def __init__(self, experiment_name, trial_name, pusher_index, pusher_cnt):
        # 通过NameResolving获取目标地址
        pullers = name_resolve.get_subtree(names.stream_pullers(experiment_name, trial_name))
        
        # 计算路由关系
        groups = grouping(pusher_cnt, len(pullers))
        puller_index = self._find_target_puller(groups, pusher_index)
        
        # 获取目标地址并连接
        name = names.push_pull_stream(experiment_name, trial_name, f"puller{puller_index}")
        addr = name_resolve.wait(name)
        host, port = addr.split(":")
        super().__init__(host, int(port))

class NameResolvingZmqPuller(ZMQJsonPuller):
    def __init__(self, args, puller_index):
        # 绑定随机端口
        host, port = network.gethostip(), network.find_free_port()
        addr = f"{host}:{port}"
        
        # 注册地址到NameResolving
        name = names.push_pull_stream(args.experiment_name, args.trial_name, f"puller{puller_index}")
        name_resolve.add(name, addr)
        super().__init__(host, port)
```

##### `Request-Reply` 通信层

```python
# realhf/system/request_reply_stream.py
class NameResolvingRequestClient:
    def __init__(self, experiment_name, trial_name, n_subscribers, handler_routing):
        # 创建多个发送socket
        for i in range(n_subscribers):
            s = self.context.socket(zmq.PUSH)
            send_port = s.bind_to_random_port(f"tcp://{host_ip}")
            
            # 注册发送地址
            master_send_name = names.request_reply_stream(experiment_name, trial_name, f"master_send_{i}")
            name_resolve.add(name=master_send_name, value=f"{host_ip}:{send_port}")
            self.send_sockets.append(s)
        
        # 创建接收socket
        self.recv_socket = self.context.socket(zmq.PULL)
        recv_port = self.recv_socket.bind_to_random_port(f"tcp://{host_ip}")
        master_recv_name = names.request_reply_stream(experiment_name, trial_name, "master_recv")
        name_resolve.add(name=master_recv_name, value=f"{host_ip}:{recv_port}")

class NameResolvingReplyServer:
    def __init__(self, experiment_name, trial_name, idx):
        # 等待MasterWorker注册地址
        send_name = names.request_reply_stream(experiment_name, trial_name, "master_recv")
        master_recv_addr = name_resolve.wait(send_name, timeout=300)
        
        recv_name = names.request_reply_stream(experiment_name, trial_name, f"master_send_{idx}")
        master_send_addr = name_resolve.wait(recv_name, timeout=300)
        
        # 连接到MasterWorker
        self.accept(master_send_addr, master_recv_addr)
```

##### 轨迹数据序列化

```python
# 轨迹数据序列化
class Trajectory:
    def as_json_serializable(self):
        return {
            "observations": self.observations,
            "actions": self.actions, 
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "log_probs": self.log_probs,
        }

# ZMQ传输
self.push_stream.push([traj.as_json_serializable()])

# 接收端反序列化
data = self.puller.pull()
sample = SequenceSample.from_json_serializable(data)
```

#### QA
##### 为什么数据流不通过`MasterWorker`而是直接到`ModelWorker`？
> `ModelWorker`直接创建`PullerStreamDataset`，通过`zmq`接收`RolloutWorker`推送的数据。

```python
# realhf/system/model_worker.py
class ModelWorker(Worker):
    def _lazy_setup(self):
        # 在ModelWorker中创建数据集
        datasets = [
            data_api.make_dataset(
                d,
                self.config.base_seed,
                self.__dataset_dp_rank,
                self.__dataset_dp_size,
                self.config.tokenizer_name_or_path,
            )
            for d in self.config.datasets
        ]
        
        # 特殊处理StreamDataset
        if not isinstance(self.__datasets[dataset_id], PullerStreamDataset):
            dataloader_kwargs["collate_fn"] = data_api.SequenceSample.gather
            dataloader_kwargs["batch_size"] = 10240
        else:
            dataloader_kwargs["batch_size"] = None  # StreamDataset不需要batch_size
```

```python
class PullerStreamDataset(Dataset):
    def __init__(self, util, args, dataset_cfgs, pull_timeout_ms=100):
        # 创建后台线程来拉取数据
        self.worker_thread = threading.Thread(target=self._pull_data_worker)
        self.worker_thread.start()
    
    def _pull_data_worker(self):
        # 在后台线程中创建ZMQ拉取器
        stream = NameResolvingZmqPuller(
            self.args,
            puller_index=self.util.dp_rank,
        )
        
        while not self._stop_event.is_set():
            # 从ZMQ接收RolloutWorker推送的数据
            data = stream.pull(timeout_ms=self.pull_timeout_ms)
            processed_data = [SequenceSample.from_json_compatible(x) for x in data]
            # 放入队列供训练使用
            self.data_queue.put_nowait(processed_data)
    
    def __getitem__(self, idx):
        # 从队列中获取数据用于训练
        samples = []
        while True:
            try:
                samples += self.data_queue.get_nowait()
            except queue.Empty:
                break
        return samples
```

**目的是为了控制流和数据流的分离，且减少数据中转** 。`MasterWorker`只是做协调训练步骤，而`ModelWorker`直接接收数据:
```python
# MasterWorker: 控制流
await self.func_executor.execute_step()  # 协调训练步骤

# ModelWorker: 数据流  
stream = NameResolvingZmqPuller(args, puller_index)  # 直接接收数据
```

这里需要理解一点：`StreamDataset`是持续接收`RolloutWorker`的数据的，不是按需获取的。stream过程会把数据缓存在内存的queue中，`MasterWorker`协调训练发生后，`ModelWorker`从内存队列里直接取数据训练。

此外，`RolloutWorker`是按照DP rank分组的，每个`ModelWorker`负责特定分组的`RolloutWorker`,通过`NameResolving`动态发现和链接。



##### `ModelWorker`如何和`RolloutWorker`分组建链？
> 问题的本质rollout worker是按照`dp`分组，那么rollout worker怎么找到对应的model worker的，这其中的服务发现是怎么实现的。

首先理解如何分组的，比如发送者和接受者的个数不同:
```python
def grouping(num_senders, num_receivers):
    groups = {}
    assert num_senders >= num_receivers
    # 每个接收者分配多个发送者
    senders_per_receiver = num_senders // num_receivers
    for receiver_id in range(num_receivers):
        start = receiver_id * senders_per_receiver
        end = (receiver_id + 1) * senders_per_receiver
        groups[receiver_id] = list(range(start, end))
    # 分配剩余的发送者
    remaining = num_senders % num_receivers
    for i in range(remaining):
        groups[i].append(num_receivers * senders_per_receiver + i)
    return groups
```

```python
# 假设有6个RolloutWorker，3个ModelWorker
grouping(6, 3)  # 6个发送者，3个接收者
# 结果：
# {
#   0: [0, 1],  # ModelWorker 0 负责 RolloutWorker 0,1
#   1: [2, 3],  # ModelWorker 1 负责 RolloutWorker 2,3  
#   2: [4, 5]   # ModelWorker 2 负责 RolloutWorker 4,5
# }
```

其次要理解`ModelWorker`如何确定自己的DP Rank:
- 只有数据并行头节点（`tp_rank == 0 and pp_rank == pp_size - 1`）才负责接收数据。
- 每个DP rank对应一个ModelWorker。
- **DP rank通过拓扑结构确定**。

```python
# realhf/system/model_worker.py
class ModelWorker(Worker):
    def _configure(self, cfg):
        # 遍历所有模型分片，找到数据并行头节点
        for s in self.config.shards:
            _pp_size = s.id.topo.get_dim("pipe")
            # 只有pipeline的最后一个stage且tensor rank为0的才是数据并行头
            if not (s.id.tp_rank == 0 and s.id.pp_rank == _pp_size - 1):
                continue
            if src_rpc.model_name == s.id.model_name:
                self.__has_dataset = True
                self.__dataset_dp_size = s.id.topo.get_dim("data")  # 总DP数量
                self.__dataset_dp_rank = s.id.dp_rank               # 当前DP rank
                break
        
        # 注册到NameResolving系统
        if self.__has_dataset:
            name = names.stream_pullers(self.__experiment_name, self.__trial_name)
            name_resolve.add_subentry(name, str(self.__dataset_dp_rank))
```

还要理解`RolloutWorker`是如何找到对应的`ModelWorker`的：
```python
# realhf/system/push_pull_stream.py
class NameResolvingZmqPusher(ZMQJsonPusher):
    def __init__(self, experiment_name, trial_name, pusher_index, pusher_cnt, **kwargs):
        # 1. 获取所有可用的puller（ModelWorker）
        pullers = name_resolve.get_subtree(
            names.stream_pullers(experiment_name, trial_name)
        )
        pullers = list(map(int, pullers))  # 转换为整数列表
        puller_cnt = len(pullers)
        
        # 2. 执行分组算法
        groups = grouping(pusher_cnt, puller_cnt)
        
        # 3. 找到当前pusher属于哪个puller组
        puller_index = None
        for puller_index, pusher_indices in groups.items():
            if pusher_index in pusher_indices:  # 这里有个bug，应该是pusher_index
                break
        
        # 4. 通过NameResolving获取目标地址
        name = names.push_pull_stream(
            experiment_name, trial_name, stream_name=f"puller{puller_index}"
        )
        addr = name_resolve.wait(name)
        host, port = addr.split(":")
        super().__init__(host, int(port), **kwargs)
```

最后理解完整的匹配流程：
1. `ModelWorker`注册
```python
# ModelWorker启动时
if self.__has_dataset:
    name = names.stream_pullers(self.__experiment_name, self.__trial_name)
    name_resolve.add_subentry(name, str(self.__dataset_dp_rank))
    # 例如：注册 "puller0", "puller1", "puller2"
```

2. `RolloutWorker`发现分组
```python
# RolloutWorker启动时
pullers = name_resolve.get_subtree(names.stream_pullers(exp_name, trial_name))
# 获取到 ["0", "1", "2"] 表示有3个ModelWorker

groups = grouping(6, 3)  # 6个RolloutWorker，3个ModelWorker
# 结果：{0: [0,1], 1: [2,3], 2: [4,5]}
```

3. 建立链接
```python
# RolloutWorker 0,1 连接到 ModelWorker 0
# RolloutWorker 2,3 连接到 ModelWorker 1  
# RolloutWorker 4,5 连接到 ModelWorker 2

name = names.push_pull_stream(exp_name, trial_name, f"puller{puller_index}")
addr = name_resolve.wait(name)  # 等待ModelWorker注册地址
```

##### `MasterWorker`如何和`ModelWorker`建链？

与`RolloutWorker-ModelWorker`的`Push-Pull`模式（单向）不同，`MasterWorker-ModelWorker`使用`Request-Reply`模式（双向）。

1. `MasterWorker`创建`Request Client`
```python
# realhf/system/master_worker.py
def __lazy_init(self):
    # 构建handler路由表
    handler_routing = copy.deepcopy(self.config.msid2mwid)
    
    # 为数据并行添加特殊路由
    src_rpc = self.__rpc_srcs[0]
    src_rpc_topo = self.config.model_topos[src_rpc.model_name]
    src_rpc_dp_size = src_rpc_topo.get_dim("data")
    src_rpc_pp_size = src_rpc_topo.get_dim("pipe")
    
    for i in range(src_rpc_dp_size):
        # 找到每个DP rank对应的ModelWorker
        rank = src_rpc_topo.get_rank(data=i, pipe=src_rpc_pp_size - 1, tensor=0)
        handler_routing[f"__data{i}__"] = self.config.msid2mwid[
            config_pkg.ModelShardID.from_parallelism_rank(
                model_name=src_rpc.model_name,
                topo=src_rpc_topo,
                parallelism_rank=rank,
            )
        ]
    
    # 添加简单的worker_index映射
    handler_routing.update({i: i for i in range(self.config.n_model_workers)})
    
    # 创建Request-Reply Stream
    self.__stream = request_reply_stream.make_master_stream(
        self.config.worker_info,
        n_subscribers=self.config.n_model_workers,
        handler_routing=handler_routing,
    )
```

```python
# realhf/system/request_reply_stream.py
class NameResolvingRequestClient:
    def __init__(self, experiment_name, trial_name, n_subscribers, handler_routing):
        self.context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)
        host_ip = socket.gethostbyname(socket.gethostname())
        
        # 1. 为每个ModelWorker创建发送socket
        self.send_sockets: List[zmq.Socket] = []
        for i in range(n_subscribers):
            s = self.context.socket(zmq.PUSH)
            send_port = s.bind_to_random_port(f"tcp://{host_ip}")
            s.setsockopt(zmq.LINGER, 0)
            
            # 注册发送地址到NameResolving
            master_send_name = names.request_reply_stream(
                experiment_name, trial_name, f"master_send_{i}"
            )
            name_resolve.add(name=master_send_name, value=f"{host_ip}:{send_port}")
            self.send_sockets.append(s)
        
        # 2. 创建接收socket
        self.recv_socket = self.context.socket(zmq.PULL)
        recv_port = self.recv_socket.bind_to_random_port(f"tcp://{host_ip}")
        self.recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_address = f"{host_ip}:{recv_port}"
        
        # 注册接收地址
        master_recv_name = names.request_reply_stream(
            experiment_name, trial_name, "master_recv"
        )
        name_resolve.add(name=master_recv_name, value=self.recv_address)
        
        # 3. 等待所有ModelWorker连接
        while (
            len(
                name_resolve.get_subtree(
                    names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME)
                )
            )
            < n_subscribers
        ):
            time.sleep(0.1)
```

2. `ModelWorker`创建`Reply Server`
```python
# realhf/system/model_worker.py
def __lazy_setup(self):
    # 创建与MasterWorker的连接
    self.__stream = request_reply_stream.make_worker_stream(
        self.config.worker_info,
        idx=self.__worker_index,
    )
```

```python
# realhf/system/request_reply_stream.py
class NameResolvingReplyServer:
    def __init__(self, experiment_name, trial_name, idx):
        self.context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)
        
        # 1. 等待MasterWorker注册接收地址
        send_name = names.request_reply_stream(
            experiment_name, trial_name, "master_recv"
        )
        try:
            master_recv_addr = name_resolve.wait(send_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master receive stream.")
            raise e
        
        # 2. 等待MasterWorker注册发送地址
        recv_name = names.request_reply_stream(
            experiment_name, trial_name, f"master_send_{idx}"
        )
        try:
            master_send_addr = name_resolve.wait(recv_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master send stream")
            raise e
        
        # 3. 建立连接
        self.accept(master_send_addr, master_recv_addr)
        
        # 4. 注册到barrier，通知MasterWorker已连接
        name_resolve.add_subentry(
            name=names.request_reply_stream(
                experiment_name, trial_name, PUBSUB_BARRIER_NAME
            ),
            value=socket.gethostbyname(socket.gethostname()),
            keepalive_ttl=1200,
        )
    
    def accept(self, server_send_addr: str, server_recv_addr: str):
        # 连接到MasterWorker的发送socket
        recv_socket = self.context.socket(zmq.PULL)
        recv_socket.connect(f"tcp://{server_send_addr}")
        recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_socket = recv_socket
        
        # 连接到MasterWorker的接收socket
        send_socket = self.context.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{server_recv_addr}")
        send_socket.setsockopt(zmq.LINGER, 0)
        self.send_socket = send_socket
```


##### 为什么`Request-Reply`模式要设计路由表？
> 问题本质是`Push-Pull`模式直接用DP rank分组策略。而`MasterWorker`和`ModelWorker`之间的路由策略要设计特定的路由表。

因为`RolloutWorker-ModelWorker`的数据流场景有以下特点：
- 持续推送：RolloutWorker持续生成数据
- 负载均衡：只需要确保数据均匀分布
- 简单映射：一个RolloutWorker组对应一个ModelWorker
- 无状态：不需要跟踪具体的任务状态

而控制流场景的特点是：
- 精确控制：需要精确指定哪个ModelWorker执行哪个任务
- 复杂拓扑：模型可能有DP、TP、PP等多种并行维度
- 状态管理：需要跟踪请求-响应的状态
- 动态分配：任务可能需要根据负载动态分配

核心还是**复杂模型的并行拓扑问题**，比如还有细粒度的模型分片(tp, pp)等，不是push-pull场景的1：N映射，而是复杂的N:M映射，还需要考虑拓扑、负载、依赖关系等。所以路由表可以确保：
- 每个ModelShardID精确映射到对应的ModelWorker
- 支持一个ModelWorker承载多个模型分片
- 支持复杂的跨模型通信（如Actor-Critic架构）

```python
# 路由表示例
handler_routing = {
    # 模型分片ID -> ModelWorker索引
    "ModelShardID(model_name='actor', dp_rank=0, tp_rank=0, pp_rank=0)": 0,
    "ModelShardID(model_name='actor', dp_rank=1, tp_rank=0, pp_rank=0)": 1,
    
    # 数据并行特殊路由
    "__data0__": 0,  # DP rank 0 -> ModelWorker 0
    "__data1__": 1,  # DP rank 1 -> ModelWorker 1
    
    # 简单索引映射
    0: 0,  # ModelWorker 0
    1: 1,  # ModelWorker 1
}
```


##### 不同并行场景下的路由表长什么样？
**场景1：纯DP（dp=2）**
配置：
- 2个ModelWorker

- 1种模型结构，DP=2

- 每个ModelWorker承载1个DP rank
```python
handler_routing = {
    # 模型分片映射
    ModelShardID(model="actor", dp=0, tp=0, pp=0): 0,  # DP rank 0 -> MW 0
    ModelShardID(model="actor", dp=1, tp=0, pp=0): 1,  # DP rank 1 -> MW 1
    
    # 数据路由映射
    "__data0__": 0,  # 数据0 -> MW 0
    "__data1__": 1,  # 数据1 -> MW 1
    
    # 用于Worker间的直接通信
    0: 0,  # MW 0 -> MW 0
    1: 1,  # MW 1 -> MW 1
}
```

特点：

- 简单的1:1映射

- 每个ModelWorker独立处理一个DP rank

- 数据路由与模型分片路由一致


**场景2: DP + TP （DP=2，TP=2）**
配置：

- 4个ModelWorker

- 1种模型结构，DP=2, TP=2

- 每个ModelWorker承载1个模型分片
```python
handler_routing = {
    # 模型分片映射 (DP=2, TP=2)
    ModelShardID(model="actor", dp=0, tp=0, pp=0): 0,  # (0,0) -> MW 0 副本0的前半
    ModelShardID(model="actor", dp=0, tp=1, pp=0): 1,  # (0,1) -> MW 1 副本0的后半
    ModelShardID(model="actor", dp=1, tp=0, pp=0): 2,  # (1,0) -> MW 2 副本1的前半
    ModelShardID(model="actor", dp=1, tp=1, pp=0): 3,  # (1,1) -> MW 3 副本1的后半
    
    # 数据路由映射 (每个DP rank对应多个TP rank)
    "__data0__": 0,  # DP rank 0 的head -> MW 0 (tp=0)
    "__data1__": 2,  # DP rank 1 的head -> MW 2 (tp=0)
    
    # 直接索引映射
    0: 0, 1: 1, 2: 2, 3: 3,
}
```

- 前向/反向时，MasterWorker会根据dp/tp/pp的rank，查找ModelShardID，路由到对应的worker（卡号）。

- 数据分发时，比如dp=0的数据，直接通过"__data0__"路由到卡0（tp=0的head）；dp=1的数据路由到卡2。

特点：

- 每个DP rank有多个TP分片

- 数据路由指向每个DP rank的head (tp=0)

- 需要TP内部的通信协调

**场景3：DP + TP + PP （DP=2, TP=2, PP=2）**
配置：

- 8个ModelWorker

- 1种模型结构，DP=2, TP=2, PP=2

- 每个ModelWorker承载1个模型分片

```python
handler_routing = {
    # 模型分片映射 (DP=2, TP=2, PP=2)
    # PP=0
    ModelShardID(model="actor", dp=0, tp=0, pp=0): 0,  # (0,0,0) -> MW 0
    ModelShardID(model="actor", dp=0, tp=1, pp=0): 1,  # (0,1,0) -> MW 1
    ModelShardID(model="actor", dp=1, tp=0, pp=0): 2,  # (1,0,0) -> MW 2
    ModelShardID(model="actor", dp=1, tp=1, pp=0): 3,  # (1,1,0) -> MW 3
    # PP=1 (最后一层)
    ModelShardID(model="actor", dp=0, tp=0, pp=1): 4,  # (0,0,1) -> MW 4
    ModelShardID(model="actor", dp=0, tp=1, pp=1): 5,  # (0,1,1) -> MW 5
    ModelShardID(model="actor", dp=1, tp=0, pp=1): 6,  # (1,0,1) -> MW 6
    ModelShardID(model="actor", dp=1, tp=1, pp=1): 7,  # (1,1,1) -> MW 7
    
    # 数据路由映射 (每个dp组的head，通常pp=最后一层, tp=0)
    "__data0__": 4,  # DP rank 0 的最后一层 -> MW 4 (pp=1, tp=0)
    "__data1__": 6,  # DP rank 1 的最后一层 -> MW 6 (pp=1, tp=0)
    
    # 直接索引映射
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
}
```

- 模型函数调用：

MasterWorker根据dp/tp/pp的rank，构造ModelShardID，查找handler_routing，路由到对应worker（卡号）。

- 例如：要调度dp=1, tp=0, pp=1的分片，查找ModelShardID(dp=1, tp=0, pp=1)，得到worker id=6（卡6）。

- 数据分发：

数据分发通常路由到每个dp组的“head”，即pp=最后一层、tp=0的分片。

- 例如：dp=0的数据，查找"__data0__"，得到worker id=4（卡4，dp=0, tp=0, pp=1）。

- dp=1的数据，查找"__data1__"，得到worker id=6（卡6，dp=1, tp=0, pp=1）。


特点：

- 最复杂的3D并行拓扑

- 数据路由指向每个DP rank的最后一层 (pp=1)

- 需要PP内部的流水线协调


**场景4：Actor-Critic架构 (DP=2)**
配置：
- 2个ModelWorker

- Actor和Critic两个模型结构，DP=2

- 每个ModelWorker承载Actor和Critic的同一个DP rank
```python
handler_routing = {
    # Actor模型分片
    ModelShardID(model="actor", dp=0, tp=0, pp=0): 0,  # Actor DP=0 -> MW 0
    ModelShardID(model="actor", dp=1, tp=0, pp=0): 1,  # Actor DP=1 -> MW 1
    
    # Critic模型分片
    ModelShardID(model="critic", dp=0, tp=0, pp=0): 0,  # Critic DP=0 -> MW 0
    ModelShardID(model="critic", dp=1, tp=0, pp=0): 1,  # Critic DP=1 -> MW 1
    
    # 数据路由映射 (Actor和Critic共享)
    "__data0__": 0,  # 数据0 -> MW 0 (Actor和Critic的DP=0)
    "__data1__": 1,  # 数据1 -> MW 1 (Actor和Critic的DP=1)
    
    # 直接索引映射
    0: 0, 1: 1,
}
```

特点：

- 一个ModelWorker承载多个模型

- Actor和Critic共享相同的DP rank

- 支持模型间的参数同步


##### 框架针对不同的拓扑是按照什么顺序切分的？

> 从路由表可以看到，3D并行下不同的切分顺序会影响卡和rank的映射，这个问题是一个分布式并行训练的基础问题，和框架的实现一起来理解。

从代码中可以看到，AReaL框架使用固定的切分顺序：
```python
# realhf/base/topology.py
class ProcessTopology:
    def __init__(self, axes, dims):
        # axes定义了切分顺序，dims定义了每个维度的切分大小
        self.axes = axes  # 切分顺序
        self.dims = dims  # 切分大小
```

```python
# 训练时的拓扑
PipeDataTensorParallelTopology(axes=['pipe', 'data', 'tensor'])

# 推理时的拓扑  
DataPipeTensorParallelTopology(axes=['data', 'pipe', 'tensor'])
```

也就是训练和推理的切分拓扑不同。

**标准顺序：PP -> DP -> TP (训练时)**：
```python
# 8张卡，DP=2, TP=2, PP=2
# 切分顺序：PP -> DP -> TP
rank = pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank

# 映射结果：
# 卡0: pp=0, dp=0, tp=0  (rank=0)
# 卡1: pp=0, dp=0, tp=1  (rank=1)
# 卡2: pp=0, dp=1, tp=0  (rank=2)
# 卡3: pp=0, dp=1, tp=1  (rank=3)
# 卡4: pp=1, dp=0, tp=0  (rank=4)
# 卡5: pp=1, dp=0, tp=1  (rank=5)
# 卡6: pp=1, dp=1, tp=0  (rank=6)
# 卡7: pp=1, dp=1, tp=1  (rank=7)
```

原因：
-  流水线友好：PP维度相邻的rank在物理上相邻，减少流水线通信开销

-  数据并行效率：同一PP stage内的DP rank可以高效进行AllReduce

-  内存局部性：同一PP stage的数据在内存上更接近

**推理时：DP -> PP -> TP**:
原因：

-  数据分发友好：DP rank相邻，便于数据分发

-  推理并行：同一DP组内的PP rank可以并行处理不同batch

-  负载均衡：DP维度优先，便于负载均衡