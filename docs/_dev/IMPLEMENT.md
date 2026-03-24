# IMPLEMENT.md — DictDyna 技术实现蓝图

**版本**: v2.0
**日期**: 2026-03-19
**配套文件**: PLAN.md v2.0

---

## 目录

1. [技术栈](#1-技术栈)
2. [仓库结构](#2-仓库结构)
3. [分阶段实现计划](#3-分阶段实现计划)
4. [核心模块接口设计](#4-核心模块接口设计)
5. [算法伪代码](#5-算法伪代码)
6. [实验脚本与配置模板](#6-实验脚本与配置模板)
7. [测试策略](#7-测试策略)

---

## 与 v1.0 的关键差异

| 项目 | v1.0 | v2.0 | 变更理由 |
|------|------|------|---------|
| 主平台 | CityLearn v2 | **Sinergym v3.11** | CityLearn 中 agent 不影响热动态（预模拟）；Sinergym 的 EnergyPlus 后端提供真实 HVAC 控制 |
| 补充平台 | — | **CityLearn v2.5** | Scalability 验证 + 不同类型 dynamics 泛化性 |
| SAC 实现 | SB3 | **CleanRL 风格纯 PyTorch** | SB3 无原生 Dyna 集成；CleanRL 更透明，Dyna 循环嵌入更自然 |
| 预训练数据 | BDG2 | **Sinergym 离线 rollout** | BDG2 单维能耗时序与 Sinergym 多维状态空间维度不匹配 |
| Reward model | 未定义 | **从预测状态直接计算** | Sinergym reward = f(温度, 功率)，均为状态变量 |
| 新增引用 | — | MrCoM, Dynamic Sparsity | 校核发现的 AAAI 2026 相关工作 |

---

## 1. 技术栈

| 类别 | 工具 | 版本 | 选型理由 |
|------|------|------|---------|
| **语言** | Python | 3.12+ | 现代语法：`X \| Y` 类型联合、`list[T]` 泛型 |
| **深度学习框架** | PyTorch | 2.x | 生态成熟，Gymnasium 集成良好 |
| **RL 算法** | CleanRL 风格纯 PyTorch SAC | 自实现 | 完全控制训练循环，Dyna 集成自然；参考 CleanRL `sac_continuous_action.py` |
| **RL 环境（主）** | Sinergym | v3.11 | EnergyPlus 后端，Gymnasium API，agent 真正控制 HVAC |
| **RL 环境（补充）** | CityLearn | v2.5 | 多建筑 scalability 验证 |
| **Dictionary Learning** | scikit-learn / 自实现 | — | `MiniBatchDictionaryLearning` 或自实现 K-SVD |
| **Sparse Coding** | 自实现（PyTorch） | — | 可微分 sparse encoder + $\ell_1$ |
| **配置管理** | OmegaConf + YAML | latest | 层级配置，CLI override |
| **类型验证** | Pydantic | v2 | Schema validation |
| **日志** | loguru + wandb | latest | 本地日志 + 实验追踪 |
| **绘图** | ultraplot | latest | 出版质量图表 |
| **代码质量** | ruff + ty | latest | Lint + 类型检查 |
| **包管理** | uv | latest | 快速依赖解析 |
| **数据处理** | pandas + numpy | — | 离线数据预处理 |
| **CLI** | Typer | latest | 命令行入口 |
| **后台任务** | tmux | — | 长时间训练会话 |

### 环境搭建

```bash
# 项目初始化
uv init dictdyna
cd dictdyna

# 核心依赖
uv add torch numpy gymnasium
uv add sinergym       # 需要先安装 EnergyPlus (见下方)
uv add citylearn      # 补充实验
uv add pandas scikit-learn omegaconf pydantic loguru typer wandb
uv add ultraplot

# 开发依赖
uv add --dev ruff ty pytest

# EnergyPlus 安装（Sinergym 依赖）
# 方式 1：通过 Sinergym Docker（推荐首次使用）
#   docker pull sailugr/sinergym:latest
# 方式 2：手动安装 EnergyPlus 24.x
#   下载：https://energyplus.net/downloads
#   安装后确认 energyplus 命令可用
# 方式 3：如果环境不支持 EnergyPlus（如无 GUI 服务器）
#   pip install sinergym 会自动尝试下载 EnergyPlus
```

---

## 2. 仓库结构

```
dictdyna/
├── pyproject.toml
├── README.md
├── PLAN.md                          # 论文写作蓝图 (v2.0)
├── IMPLEMENT.md                     # 本文件 (v2.0)
│
├── configs/                         # OmegaConf YAML 配置
│   ├── base.yaml                    # 共享默认值
│   ├── pretrain.yaml                # Phase I: dictionary pretraining
│   ├── train.yaml                   # Phase II: Dyna RL 训练
│   ├── transfer.yaml                # Phase III: few-shot transfer
│   ├── ablation/                    # 消融实验配置
│   │   ├── random_dict.yaml
│   │   ├── fixed_dict.yaml
│   │   ├── independent_dict.yaml
│   │   ├── dict_size_k64.yaml
│   │   ├── dict_size_k128.yaml
│   │   ├── dict_size_k256.yaml
│   │   └── sparsity_sweep.yaml
│   └── env/
│       ├── sinergym_office.yaml     # Sinergym 5zone office 配置
│       ├── sinergym_datacenter.yaml # Sinergym datacenter 配置
│       ├── sinergym_multi.yaml      # 多建筑-多气候区配置
│       ├── sinergym_transfer.yaml   # 迁移实验配置
│       ├── citylearn_scale.yaml     # CityLearn scalability 配置
│       └── citylearn_transfer.yaml  # CityLearn 跨平台迁移
│
├── src/
│   ├── __init__.py
│   │
│   ├── dictionary/                  # Dictionary learning 模块
│   │   ├── __init__.py
│   │   ├── ksvd.py                  # K-SVD 实现
│   │   ├── online_dl.py             # Online dictionary learning
│   │   └── pretrain.py              # 预训练 pipeline（Sinergym rollout → D₀）
│   │
│   ├── world_model/                 # DictDyna world model
│   │   ├── __init__.py
│   │   ├── sparse_encoder.py        # g_θ(s,a;ϕᵢ) 网络
│   │   ├── dict_dynamics.py         # ŝ' = s + D·α transition model
│   │   ├── reward_estimator.py      # 从预测状态计算 Sinergym reward
│   │   └── model_trainer.py         # World model 训练循环
│   │
│   ├── agent/                       # RL agent + Dyna 集成
│   │   ├── __init__.py
│   │   ├── sac.py                   # CleanRL 风格纯 PyTorch SAC
│   │   ├── dyna_sac.py              # Dyna-SAC: SAC + model rollouts
│   │   ├── replay_buffer.py         # Mixed real + simulated buffer
│   │   └── rollout.py               # Model-based rollout 生成
│   │
│   ├── env/                         # 环境封装
│   │   ├── __init__.py
│   │   ├── sinergym_wrapper.py      # Sinergym ↔ DictDyna adapter
│   │   ├── multi_building_sinergym.py # 多 Sinergym 实例并行管理
│   │   ├── citylearn_wrapper.py     # CityLearn ↔ DictDyna adapter
│   │   └── multi_building.py        # 统一多建筑接口
│   │
│   ├── transfer/                    # Few-shot transfer 模块
│   │   ├── __init__.py
│   │   └── adapter.py               # Adapter fine-tuning
│   │
│   ├── data/                        # 数据处理
│   │   ├── __init__.py
│   │   ├── offline_collector.py     # Sinergym 离线数据收集
│   │   └── state_diff.py            # 计算 Δs = s_{t+1} - s_t
│   │
│   ├── schemas.py                   # Pydantic 配置 schemas
│   └── utils.py                     # 共享工具（seeding、device 等）
│
├── scripts/                         # 入口脚本
│   ├── collect_offline.py           # Phase 0: 收集 Sinergym 离线数据
│   ├── pretrain_dictionary.py       # Phase I: 运行 dictionary pretraining
│   ├── train_dyna.py                # Phase II: 运行 Dyna-SAC 训练
│   ├── transfer.py                  # Phase III: 运行 few-shot transfer
│   ├── run_baseline.py              # 运行 baseline 方法（SAC、RBC 等）
│   ├── run_ablation.py              # 运行消融实验
│   ├── run_citylearn.py             # 运行 CityLearn 补充实验
│   ├── evaluate.py                  # 评估训练好的 agent
│   ├── visualize_atoms.py           # 可视化 dictionary atoms
│   └── visualize_results.py         # 生成论文图表
│
├── data/
│   ├── offline_rollouts/            # Sinergym 离线 rollout（gitignored）
│   └── processed/                   # 预处理后数据
│       └── state_diffs/
│
├── output/                          # 实验输出（gitignored）
│   ├── pretrained/                  # 保存的 dictionaries
│   ├── checkpoints/                 # 训练 checkpoints
│   ├── results/                     # 评估指标
│   └── figures/                     # 论文图表
│
└── tests/
    ├── test_dictionary.py
    ├── test_world_model.py
    ├── test_sac.py
    ├── test_dyna_sac.py
    ├── test_sinergym_env.py
    └── test_transfer.py
```

---

## 3. 分阶段实现计划

### Phase 0：基础准备（2–3 个月）

**目标**：建立理解基础，搭建实验环境，运行 baselines。

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–2 | 精读 Losse-FTL 论文 + JAX 代码 | 技术笔记 | 理解 sparse encoding world model 原理 |
| 3–4 | 精读 MB2C、CLUE 论文 + 代码 | 技术笔记 | 理解建筑 MBRL 具体实现 |
| 5–6 | 搭建 Sinergym v3.11 + EnergyPlus | `sinergym_wrapper.py` 通过冒烟测试 | 能 `gym.make` 创建环境并交互 |
| 7–8 | 实现 CleanRL SAC 并在单建筑上训练 | SAC 学习曲线 | Reward > RBC baseline |
| 9–10 | 用 RBC/随机策略收集 Sinergym 离线数据 | `offline_collector.py`，state diffs | 数据维度正确，分布合理 |
| 11–12 | 学习 K-SVD/OMP，实现字典学习 | `ksvd.py` 在 toy data 上工作 | 合成信号重构误差 <5% |

**关键风险**：EnergyPlus 安装环境问题。**应对**：备选 Docker 部署方案。

### Phase 1：单建筑 World Model（2–3 个月）

**目标**：证明 dictionary world model 能预测建筑热动态。

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–3 | 在 Sinergym 离线 rollout 数据上做 dictionary pretraining | 预训练好的 $\mathbf{D}_0$ | Atoms 视觉上多样且非退化 |
| 4–6 | 实现 sparse encoder $g_\theta$ | `sparse_encoder.py` | 输出确实稀疏（$>70\%$ 零值） |
| 7–8 | 实现 dict dynamics model + reward estimator | `dict_dynamics.py`、`reward_estimator.py` | 单步预测 MSE < naive baseline |
| 9–10 | 多步预测评估 | 预测误差 vs. horizon 曲线 | $H \le 5$ 时误差增长 < 指数级 |
| 11–12 | 与 ensemble NN（MB2C 风格）对比 | 对比表格 | Dictionary model 性能可比或更优 |

### Phase 2：单建筑 MBRL（2 个月）

**目标**：Dyna-style planning 提升 sample efficiency。

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–3 | 实现 Dyna-SAC 集成（CleanRL SAC + world model rollouts） | `dyna_sac.py` 可运行 | 训练循环不崩溃 |
| 4–5 | 实现 model rollout + mixed buffer | `rollout.py`、`replay_buffer.py` | Simulated transitions 质量合理 |
| 6–7 | 在单建筑上训练 Dyna-SAC | 学习曲线 vs. SAC | 收敛速度 ≥ 比 model-free SAC 快 30% |
| 8 | 超参数调优（rollout horizon $H$、ratio） | 最优 $H$ 和 ratio | 记录在 wandb |

### Phase 3：多建筑共享字典（2–3 个月）

**目标**：核心创新——shared dictionary 优于独立模型。

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–2 | 实现 MultiBuildingSinergym 封装 | `multi_building_sinergym.py` | 3–5 种建筑类型并行训练 |
| 3–5 | 实现 shared $\mathbf{D}$ + per-building $\phi_i$ | 完整方案 γ 实现 | Shared model 训练稳定 |
| 6–7 | 对比：shared dict vs. independent dict vs. MB2C | 性能对比表 | Shared ≥ independent（多数建筑） |
| 8–9 | 可解释性：可视化 atoms | Atom pattern 图 | Atoms 对应物理模式 |
| 10 | 初步 CityLearn 验证 | CityLearn 上基本运行 | 确认跨平台兼容 |

**核心检查点**：shared dictionary ≥ independent dictionary 性能。若此处失败 → 转向 IDEA 2（SparseAdapt）。

### Phase 4：迁移 + 可扩展性 + CityLearn 补充（2–3 个月）

**目标**：Few-shot transfer、scaling、跨平台验证。

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–3 | Sinergym few-shot transfer（跨气候区、跨建筑类型） | Transfer 曲线（1/3/7 天） | 7 天 adaptation ≥ 全训练性能的 80% |
| 4–5 | CityLearn scalability 实验（5→20→50 栋） | Scaling 图 | 确认 sublinear atom 增长 |
| 6–7 | CityLearn 上完整 DictDyna 训练 | CityLearn 性能表 | 验证跨平台泛化 |
| 8–9 | 鲁棒性测试（不同 seeds、不同初始化） | 统计汇总 | 结果跨 seeds 稳定 |

### Phase 5：消融实验 + 论文写作（2–3 个月）

| 周 | 任务 | 产出物 | 检查点 |
|----|------|--------|--------|
| 1–3 | 运行全部消融实验 | 消融表格/图表 | 全部消融完成 |
| 4–5 | 生成所有论文图表 | 7 张图存于 `output/figures/` | 出版质量（300 DPI） |
| 6–10 | 按 PLAN.md v2.0 写论文 | 初稿 v1 | 所有章节完成 |
| 11–14 | 修订、润色、投稿 Applied Energy | 终稿 | 准备投稿 |

**总计：12–17 个月**（与 v1.0 一致，Applied Energy 滚动投稿无 deadline 压力）

---

## 4. 核心模块接口设计

### 4.1 配置 Schema（`src/schemas.py`）

```python
from pydantic import BaseModel, ConfigDict, Field


class DictionarySchema(BaseModel):
    """Dictionary learning configuration."""
    model_config = ConfigDict(frozen=True)

    n_atoms: int = Field(128, ge=16, description="Number of dictionary atoms K")
    state_dim: int = Field(15, ge=1, description="State dimension d (Sinergym ~10-15)")
    sparsity_lambda: float = Field(0.1, gt=0, description="L1 sparsity weight lambda")
    pretrain_epochs: int = Field(100, ge=1)
    pretrain_lr: float = Field(1e-3, gt=0)
    slow_update_lr: float = Field(1e-5, ge=0, description="D fine-tune rate during RL, 0=fixed")


class SparseEncoderSchema(BaseModel):
    """Sparse encoder g_theta(s,a;phi_i) configuration."""
    model_config = ConfigDict(frozen=True)

    shared_hidden_dims: list[int] = Field(default=[256, 256])
    adapter_dim: int = Field(64, ge=1, description="Per-building adapter hidden dim")
    activation: str = Field("relu", pattern=r"^(relu|gelu|tanh)$")
    sparsity_method: str = Field("l1_penalty", pattern=r"^(l1_penalty|topk|proximal)$")
    topk_k: int = Field(16, ge=1, description="If sparsity_method=topk, keep top-k activations")


class DynaSchema(BaseModel):
    """Dyna-style planning configuration."""
    model_config = ConfigDict(frozen=True)

    rollout_horizon: int = Field(3, ge=1, le=10, description="H: model rollout steps")
    rollouts_per_step: int = Field(10, ge=1, description="M: number of rollouts per real step")
    model_to_real_ratio: float = Field(0.5, ge=0, le=1, description="Fraction of simulated data in batch")
    model_update_freq: int = Field(1, ge=1, description="Update world model every N real steps")


class SACSchema(BaseModel):
    """CleanRL-style SAC configuration."""
    model_config = ConfigDict(frozen=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dims: list[int] = Field(default=[256, 256])
    tau: float = 0.005
    autotune_alpha: bool = True
    initial_alpha: float = 0.2


class TrainSchema(BaseModel):
    """Overall training configuration."""
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    total_timesteps: int = Field(8760 * 3, description="3 years of hourly data")
    eval_freq: int = 8760  # Evaluate every simulated year
    n_buildings: int = Field(3, ge=1)
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    dictionary: DictionarySchema = DictionarySchema()
    encoder: SparseEncoderSchema = SparseEncoderSchema()
    dyna: DynaSchema = DynaSchema()
    sac: SACSchema = SACSchema()
    device: str = "auto"


class TransferSchema(BaseModel):
    """Few-shot transfer configuration."""
    model_config = ConfigDict(frozen=True)

    adaptation_days: int = Field(7, ge=1, description="Days of data for new building")
    freeze_dictionary: bool = True
    freeze_shared_encoder: bool = True
    adapter_lr: float = 1e-3
    adaptation_epochs: int = 50
```

### 4.2 Sinergym 多建筑封装（`src/env/multi_building_sinergym.py`）

```python
import gymnasium
import numpy as np


class MultiBuildingSinergym:
    """Manage multiple Sinergym environments as a unified multi-building interface.

    Each building = one Sinergym instance with distinct env_name (IDF + weather).
    All buildings expose the same observation/action space structure
    (ensured by choosing compatible Sinergym environments).

    Args:
        building_configs: List of dicts with keys:
            env_name: Sinergym registered environment name
            building_id: Unique string identifier
    """

    def __init__(self, building_configs: list[dict]) -> None:
        self.envs: dict[str, gymnasium.Env] = {}
        self.building_ids: list[str] = []
        for cfg in building_configs:
            env = gymnasium.make(cfg["env_name"])
            bid = cfg["building_id"]
            self.envs[bid] = env
            self.building_ids.append(bid)

    @property
    def n_buildings(self) -> int:
        return len(self.building_ids)

    def reset_all(self, seed: int | None = None) -> dict[str, tuple]:
        """Reset all building environments.

        Returns:
            Dict mapping building_id to (obs, info).
        """
        results = {}
        for i, bid in enumerate(self.building_ids):
            s = seed + i if seed is not None else None
            obs, info = self.envs[bid].reset(seed=s)
            results[bid] = (obs, info)
        return results

    def step(self, building_id: str, action: np.ndarray) -> tuple:
        """Step a specific building environment.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        return self.envs[building_id].step(action)

    def close_all(self) -> None:
        for env in self.envs.values():
            env.close()

    def collect_offline_data(
        self,
        policy: str = "random",
        n_episodes: int = 2,
    ) -> dict[str, list[dict]]:
        """Collect offline rollout data for dictionary pretraining.

        Args:
            policy: "random" for random actions, "rbc" for rule-based.
            n_episodes: Number of episodes per building.

        Returns:
            Dict mapping building_id to list of transition dicts
            {"s": ..., "a": ..., "s_next": ..., "r": ...}.
        """
        ...
```

### 4.3 Reward Estimator（`src/world_model/reward_estimator.py`）

```python
import torch


class SinergymRewardEstimator:
    """Estimate reward from predicted state in Sinergym environments.

    Sinergym rewards are typically of the form:
        r = -w_E * E_t - (1 - w_E) * comfort_penalty(T_indoor, T_target)
    Both E_t (HVAC power) and T_indoor are state variables, so reward
    can be computed directly from world model predictions.

    This eliminates the need for a separate learned reward model,
    which would be required in environments like CityLearn where
    reward depends on exogenous signals (electricity price, carbon).

    Args:
        comfort_weight: Weight for comfort penalty (1 - energy_weight).
        temp_target: Target indoor temperature (°C).
        temp_band: Acceptable temperature range around target.
        state_indices: Mapping of variable name to state vector index.
    """

    def __init__(
        self,
        comfort_weight: float = 0.5,
        temp_target: float = 23.0,
        temp_band: float = 2.0,
        state_indices: dict[str, int] | None = None,
    ) -> None:
        self.comfort_weight = comfort_weight
        self.temp_target = temp_target
        self.temp_band = temp_band
        self.state_indices = state_indices or {}

    def estimate(self, predicted_state: torch.Tensor) -> torch.Tensor:
        """Compute reward from predicted next state.

        Args:
            predicted_state: World model output, shape (batch, d).

        Returns:
            Estimated reward, shape (batch,).
        """
        idx_temp = self.state_indices.get("indoor_temp", 0)
        idx_power = self.state_indices.get("hvac_power", 1)

        temp = predicted_state[:, idx_temp]
        power = predicted_state[:, idx_power]

        # Comfort penalty: 0 inside band, linear outside
        comfort_violation = (
            torch.relu(temp - (self.temp_target + self.temp_band))
            + torch.relu((self.temp_target - self.temp_band) - temp)
        )

        energy_weight = 1.0 - self.comfort_weight
        reward = -(energy_weight * power + self.comfort_weight * comfort_violation)
        return reward
```

### 4.4 CleanRL 风格 SAC（`src/agent/sac.py`）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SoftQNetwork(nn.Module):
    """Twin Q-networks for SAC.

    Args:
        state_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        # Q1
        q1_layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            q1_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        q1_layers.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 (identical architecture, independent weights)
        q2_layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            q2_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        q2_layers.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy for continuous SAC.

    Args:
        state_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes.
        action_scale: Scale for tanh squashing.
        action_bias: Bias for tanh squashing.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Returns:
            (action, log_prob) where action is squashed.
        """
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean) for evaluation."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        return torch.tanh(mean) * self.action_scale + self.action_bias
```

### 4.5 Dyna-SAC Agent（`src/agent/dyna_sac.py`）

```python
import torch
import numpy as np
from pathlib import Path


class DynaSAC:
    """Dyna-style SAC with dictionary world model.

    Integrates real environment interaction with model-based
    rollouts. Uses CleanRL-style pure PyTorch SAC (no SB3 dependency).

    At each real step:
    1. Execute action in Sinergym env, collect (s, a, r, s')
    2. Update world model on real data
    3. Generate M simulated rollouts of horizon H
    4. Update SAC policy on mixed real + simulated data

    Args:
        env: MultiBuildingSinergym or single Gymnasium env.
        world_model: DictDynamicsModel instance.
        sparse_encoder: SparseEncoder instance.
        reward_estimator: SinergymRewardEstimator instance.
        config: TrainSchema configuration.
    """

    def __init__(
        self,
        env: object,
        world_model: object,
        sparse_encoder: object,
        reward_estimator: object,
        config: object,
    ) -> None:
        self.env = env
        self.world_model = world_model
        self.sparse_encoder = sparse_encoder
        self.reward_estimator = reward_estimator
        self.config = config

        # Initialize CleanRL SAC components
        self.actor: object = None      # GaussianActor
        self.critic: object = None     # SoftQNetwork
        self.critic_target: object = None
        self.log_alpha: torch.Tensor | None = None

        self.real_buffer: object = None
        self.model_buffer: object = None

    def train(self, total_timesteps: int) -> dict:
        """Run the full Dyna-SAC training loop.

        Args:
            total_timesteps: Total real environment steps.

        Returns:
            Training metrics dict.
        """
        ...

    def _model_rollout(
        self,
        start_states: np.ndarray,
        building_id: str,
        horizon: int,
    ) -> list[tuple]:
        """Generate simulated rollouts from the world model.

        Reward is estimated directly from predicted states using
        SinergymRewardEstimator (no separate reward model needed).

        Args:
            start_states: Starting states, shape (M, d).
            building_id: Building to simulate.
            horizon: Number of rollout steps H.

        Returns:
            List of (s, a, r, s', done) tuples.
        """
        ...

    def _update_world_model(self, batch: dict) -> float:
        """Update world model (encoder + optional dictionary).

        Args:
            batch: Batch of real transitions {s, a, s', building_id}.

        Returns:
            World model loss value.
        """
        ...

    def _update_sac(self, batch: dict) -> dict:
        """Update SAC actor/critic on mixed real + simulated batch.

        Pure PyTorch implementation following CleanRL conventions.

        Args:
            batch: Mixed batch from real + model buffers.

        Returns:
            SAC training metrics.
        """
        ...

    def evaluate(self, n_episodes: int = 3) -> dict:
        """Evaluate current policy without model rollouts."""
        ...

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> "DynaSAC": ...
```

### 4.6 离线数据收集（`src/data/offline_collector.py`）

```python
import gymnasium
import numpy as np
from pathlib import Path
from loguru import logger


class OfflineCollector:
    """Collect offline rollout data from Sinergym for dictionary pretraining.

    Runs episodes with RBC or random policy to gather (s, a, s') transitions
    from diverse building types and climate zones. The collected state
    differences Δs = s' - s become training data for K-SVD.

    Args:
        building_configs: List of dicts with env_name and building_id.
        policy: "random" or "rbc".
        n_episodes: Number of episodes per building.
        output_dir: Directory to save collected data.
    """

    def __init__(
        self,
        building_configs: list[dict],
        policy: str = "rbc",
        n_episodes: int = 2,
        output_dir: str = "data/offline_rollouts",
    ) -> None:
        self.building_configs = building_configs
        self.policy = policy
        self.n_episodes = n_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> dict[str, np.ndarray]:
        """Collect transitions from all buildings.

        Returns:
            Dict mapping building_id to state_diffs array, shape (N, d).
        """
        all_diffs: dict[str, np.ndarray] = {}
        for cfg in self.building_configs:
            bid = cfg["building_id"]
            logger.info(f"Collecting data for {bid} ({cfg['env_name']})")
            diffs = self._collect_building(cfg["env_name"])
            all_diffs[bid] = diffs
            np.save(self.output_dir / f"{bid}_state_diffs.npy", diffs)
            logger.info(f"  Collected {len(diffs)} transitions")
        return all_diffs

    def _collect_building(self, env_name: str) -> np.ndarray:
        """Collect state differences from a single building."""
        all_diffs = []
        for ep in range(self.n_episodes):
            env = gymnasium.make(env_name)
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                if self.policy == "random":
                    action = env.action_space.sample()
                else:
                    action = self._rbc_action(obs, env)
                next_obs, _, terminated, truncated, _ = env.step(action)
                all_diffs.append(next_obs - obs)
                obs = next_obs
                done = terminated or truncated
            env.close()
        return np.array(all_diffs)

    def _rbc_action(self, obs: np.ndarray, env: object) -> np.ndarray:
        """Simple rule-based control action."""
        # Default: return midpoint of action space
        low = env.action_space.low
        high = env.action_space.high
        return (low + high) / 2.0
```

---

## 5. 算法伪代码

（与 v1.0 基本一致，更新环境名称和 reward 估计方式）

### Algorithm 1: DictDyna Training（论文用）

```
Algorithm 1: DictDyna Training Procedure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: N buildings {env_1, ..., env_N} (Sinergym instances),
       pretrained dictionary D_0 (from offline rollouts),
       rollout horizon H, rollouts per step M,
       sparsity weight λ_s, dict learning rate η_D

Output: Trained policy π, world model (D, g_θ, {ϕ_i})

 1: Initialize D ← D_0 (pretrained on Sinergym offline data)
 2: Initialize shared encoder params θ, adapters {ϕ_1,...,ϕ_N}
 3: Initialize SAC policy π (CleanRL-style: actor, twin-Q, α)
 4: Initialize replay buffers B_real, B_model ← empty

 5: for episode = 1, 2, ... do
 6:    for each building i = 1, ..., N do
 7:       s ← env_i.reset()
 8:       for t = 1, ..., T do
 9:          -- Real Interaction (Sinergym) --
10:          a ← π(s)
11:          s', r ← env_i.step(a)
12:          B_real ← B_real ∪ {(s, a, r, s', i)}
13:
14:          -- World Model Update --
15:          Sample batch {(s_j, a_j, r_j, s'_j, i_j)} from B_real
16:          α ← g_θ(s_j, a_j; ϕ_{i_j})
17:          ŝ ← s_j + D · α
18:          L_WM ← ||s'_j - ŝ||² + λ_s · ||α||₁
19:          Update θ, {ϕ_i} by ∇L_WM
20:          Optionally update D with learning rate η_D
21:          Normalize D columns to unit norm
22:
23:          -- Model Rollouts --
24:          for m = 1, ..., M do
25:             s_0 ← sample from B_real
26:             for h = 0, ..., H-1 do
27:                a_h ← π(s_h)
28:                α_h ← g_θ(s_h, a_h; ϕ_i)
29:                s_{h+1} ← s_h + D · α_h
30:                r_h ← RewardEstimator(s_{h+1})  // 直接从状态计算
31:                B_model ← B_model ∪ {(s_h, a_h, r_h, s_{h+1})}
32:             end for
33:          end for
34:
35:          -- Policy Update (CleanRL SAC) --
36:          Sample mixed batch from B_real ∪ B_model
37:          Update π (actor, twin-Q, α) on mixed batch
38:
39:          s ← s'
40:       end for
41:    end for
42: end for
```

### Algorithm 2: Few-shot Transfer（与 v1.0 一致，仅更新环境名）

---

## 6. 实验脚本与配置模板

### 6.1 基础配置（`configs/base.yaml`）

```yaml
seed: 42
device: auto

dictionary:
  n_atoms: 128
  sparsity_lambda: 0.1
  pretrain_epochs: 100
  pretrain_lr: 1e-3
  slow_update_lr: 1e-5

encoder:
  shared_hidden_dims: [256, 256]
  adapter_dim: 64
  activation: relu
  sparsity_method: l1_penalty
  topk_k: 16

dyna:
  rollout_horizon: 3
  rollouts_per_step: 10
  model_to_real_ratio: 0.5
  model_update_freq: 1

sac:
  actor_lr: 3e-4
  critic_lr: 3e-4
  alpha_lr: 3e-4
  hidden_dims: [256, 256]
  tau: 0.005
  autotune_alpha: true
  initial_alpha: 0.2

training:
  total_timesteps: 26280  # 3 years hourly
  eval_freq: 8760
  n_eval_episodes: 3
  log_interval: 100
  save_freq: 8760
  batch_size: 256
  buffer_size: 100000
  gamma: 0.99

reward:
  comfort_weight: 0.5
  temp_target: 23.0
  temp_band: 2.0

wandb:
  project: dictdyna
  entity: null
  tags: []
```

### 6.2 Sinergym 多建筑配置（`configs/env/sinergym_multi.yaml`）

```yaml
platform: sinergym

buildings:
  - env_name: Eplus-5zone-hot-continuous-v1
    building_id: office_hot
  - env_name: Eplus-5zone-mixed-continuous-v1
    building_id: office_mixed
  - env_name: Eplus-5zone-cool-continuous-v1
    building_id: office_cool
  - env_name: Eplus-datacenter-mixed-continuous-stochastic-v1
    building_id: datacenter_mixed

# State/action space info (will be auto-detected from env)
# Listed here for reference
state_variables:
  - outdoor_temperature
  - outdoor_humidity
  - direct_solar_radiation
  - diffuse_solar_radiation
  - indoor_temperature
  - hvac_power
  - month
  - day_of_month
  - hour
```

### 6.3 Pretraining 配置（`configs/pretrain.yaml`）

```yaml
defaults:
  - base

dictionary:
  n_atoms: 128
  pretrain_epochs: 200
  pretrain_lr: 1e-3

data:
  source: sinergym_rollout
  collection_policy: rbc
  n_episodes_per_building: 2
  buildings:
    - env_name: Eplus-5zone-hot-continuous-v1
      building_id: office_hot
    - env_name: Eplus-5zone-mixed-continuous-v1
      building_id: office_mixed
    - env_name: Eplus-5zone-cool-continuous-v1
      building_id: office_cool
    - env_name: Eplus-datacenter-mixed-continuous-stochastic-v1
      building_id: datacenter_mixed
  normalize: true
  output_path: data/processed/state_diffs/

output:
  dict_path: output/pretrained/dict_k128.pt
```

### 6.4 CityLearn Scalability 配置（`configs/env/citylearn_scale.yaml`）

```yaml
platform: citylearn

# Scalability experiment: 5 → 20 → 50 buildings
scale_levels:
  - n_buildings: 5
    name: citylearn_5b
  - n_buildings: 20
    name: citylearn_20b
  - n_buildings: 50
    name: citylearn_50b

# Note: CityLearn dynamics = storage system (battery SOC),
# NOT thermal dynamics. This tests DictDyna generalization
# to different dynamics types.
```

### 6.5 tmux 会话模板

```bash
# Phase 0: 收集 Sinergym 离线数据
tmux new-session -d -s collect \
    "cd dictdyna && uv run python scripts/collect_offline.py -c configs/pretrain.yaml"

# Phase I: Dictionary pretraining
tmux new-session -d -s pretrain \
    "cd dictdyna && uv run python scripts/pretrain_dictionary.py -c configs/pretrain.yaml"

# Phase II: Dyna-SAC 训练（5 个 seeds）
for seed in 1 2 3 4 5; do
    tmux new-session -d -s "train_s${seed}" \
        "cd dictdyna && uv run python scripts/train_dyna.py -c configs/train.yaml -o seed=${seed}"
done

# Phase III: Transfer 实验
for days in 1 3 7; do
    tmux new-session -d -s "transfer_d${days}" \
        "cd dictdyna && uv run python scripts/transfer.py -c configs/transfer.yaml -d ${days}"
done

# CityLearn 补充实验
tmux new-session -d -s citylearn \
    "cd dictdyna && uv run python scripts/run_citylearn.py -c configs/env/citylearn_scale.yaml"

# Baselines
for method in rbc sac mb2c; do
    tmux new-session -d -s "baseline_${method}" \
        "cd dictdyna && uv run python scripts/run_baseline.py -m ${method}"
done

# 全部消融实验
tmux new-session -d -s ablations \
    "cd dictdyna && uv run python scripts/run_ablation.py --all --seeds 5"

# 监控
tmux list-sessions
```

---

## 7. 测试策略

### 关键测试用例

| 模块 | 测试 | 验证内容 |
|------|------|---------|
| `KSVDDictionary` | `test_fit_reconstruct` | Dictionary 能以低 MSE 重构训练数据 |
| `KSVDDictionary` | `test_atom_unit_norm` | 所有 atoms 拟合后具有单位 L2 范数 |
| `SparseEncoder` | `test_output_shape` | 输出形状为 (batch, K) |
| `SparseEncoder` | `test_sparsity_topk` | Top-k 产生恰好 k 个非零值 |
| `SparseEncoder` | `test_adapter_isolation` | 不同 building_id 产生不同输出 |
| `DictDynamicsModel` | `test_forward_residual` | 输出 = 输入 + D·α（残差结构） |
| `DictDynamicsModel` | `test_normalize_atoms` | 归一化后所有 atoms 具有单位范数 |
| `SinergymRewardEstimator` | `test_reward_from_state` | 给定已知状态，reward 计算正确 |
| `GaussianActor` | `test_action_bounds` | 输出在 action space 范围内 |
| `SoftQNetwork` | `test_twin_independence` | Q1 和 Q2 输出不同值 |
| `DynaSAC` | `test_rollout_shape` | Rollout 产生正确数量的 transitions |
| `DynaSAC` | `test_mixed_buffer` | Mixed buffer 同时包含 real 和 simulated 数据 |
| `MultiBuildingSinergym` | `test_multi_env_create` | 能创建多个独立的 Sinergym 实例 |
| `OfflineCollector` | `test_collect_diffs` | 收集的 state diffs 维度正确 |
| Transfer | `test_freeze_params` | Transfer 时仅 adapter 参数有梯度 |

### 代码质量检查

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ty check .
```

---

## 附录 A：方法创新设计（INNOVATION_PLAN）

> 来源：4 个并行 Agent 调研（2026-03-21），竞争格局分析 + 创新设计

### 竞争格局

| 论文 | 年份/期刊 | 关键差异 |
|------|----------|---------|
| **SINDy-RL** | 2025 Nature Comms | **最强竞品**：固定基函数库（多项式/三角函数），DictDyna 用数据驱动学习的字典原子 |
| **LOSSE** | 2024 ICLR | 随机稀疏特征 + 线性 FTL，DictDyna 用学习的 overcomplete dictionary |
| **MoW** | 2026 ICLR | MoE 路由选择动力学模块，DictDyna 用 topk 稀疏编码选择原子 |
| **DreamerV3** | 2025 Nature | 通用 MBRL baseline，dense latent，DictDyna 加入可解释稀疏结构 |

**DictDyna 差异化定位**：首个将数据驱动学习的字典用于 MBRL world model 的工作。

### 创新 1：稀疏编码探索奖励（已实现）

```
r_intrinsic(s,a) = η / sqrt(N(support(α)))
```
- topk 稀疏编码的 support 是状态的天然离散哈希
- "世界模型即探索模块" — 零额外参数

### 创新 2：自适应稀疏度 k(s)（待实现）

```
k(s) = MLP(s) → [k_min, k_max]
```
- 瞬态需要更多原子（高 k），稳态只需少量原子（低 k）

### 创新 3：奖励加权字典学习（已实现）

```
L_WM = Σ w_i · ||s'_i - ŝ'_i||² + λ · ||α_i||₁
w_i = 1 + β · |δ_i|
```

### 创新 4：多建筑共享字典（已实现）

```
3 Sinergym envs → TaggedReplayBuffer → Shared D + Per-building adapter ϕ_i → Shared SAC
```

### 原子可解释性

| 类别 | 数量 | 占比 | 代表维度 |
|------|------|------|---------|
| Weather | 55 | 43% | wind_dir, outdoor_hum |
| Indoor | 32 | 25% | indoor_temp, indoor_hum |
| Temporal | 22 | 17% | month, day, hour |
| HVAC | 15 | 12% | heat_setpt, cool_setpt, power |
| Solar | 3 | 2% | diffuse/direct solar |

### 需引用的关键论文

| 论文 | 引用理由 |
|------|---------|
| SINDy-RL (Nature Comms 2025) | 最强竞品，稀疏字典 + Dyna |
| MoW (ICLR 2026) | 模块化多任务 world model |
| PWM (ICLR 2025) | 多任务 world model |
| Network Sparsity in DRL (ICML 2025 Oral) | 稀疏性在 RL 中的价值 |
| Value-Aligned WM (OpenReview 2025) | 奖励感知 world model |

---

## 附录 B：多建筑共享字典优化方案（SHARED_DICT_RESEARCH）

> 来源：Phase 3 诊断后的调研（2026-03-22）

### 诊断：Phase 3 Shared D 仅 +0.6% 的根因

1. **灾难性遗忘**：顺序训练 hot→mixed→cool，shared encoder 忘记之前建筑。cool（最后训练）赢 +541，hot（最先训练）输 -358
2. **World Model 比 Identity 差**：4 个维度（12/13/15/16）的预测误差是恒等映射的 182x。rollout 注入噪声而非信号
3. **Adapter 错位**：用错误建筑的 adapter 反而预测更好（2/3 case），因为 trunk 在训练 cool 时已偏移

### 6 种优化方案评估

| 方案 | 预期收益 | 实现复杂度 | 状态 |
|------|---------|----------|------|
| 1. Interleaved Replay (Joint WM Training) | +8-15% | 极低 (30min) | **已实现** |
| 2. EWC | +3-8% | 低-中 | 不推荐（Joint 已解决遗忘） |
| 3. PCGrad 梯度投影 | +5-12% | 中 | 待评估 |
| 4. **Shared-Private Dict Split (64+64)** | +8-20% | 中 | **已实现** |
| 5. Multi-Task Loss Weighting | +1-5% | 极低 | 不推荐（收益低） |
| 6. Full Joint Training | +8-15% | 极低 | 已实现（= 方案 1） |

### 已实施方案

**方案 1：Joint WM Training**：每步 WM 更新时从所有建筑采样训练。将 +0.6% 提升到 +3%。

**方案 4：Shared-Private Dict Split**：128 原子 = 64 shared（frozen）+ 64 private（trainable per-building）。cool Ep3 达到全局最佳 -4871。

---

## 附录 C：Transfer 改进方案（TRANSFER_IMPROVEMENT_RESEARCH）

> 来源：Phase 4 优化调研（2026-03-23）

### 5 种 Transfer 改进方案

| 方案 | 预期收益 | 新颖性 | 状态 |
|------|---------|--------|------|
| 1. MAML-Style Bi-Level Optimization | +5-15% | 中 | 待评估 |
| 2. **Context-Conditioned WM (CaDM/DALI)** | +10-25% | **高** | 推荐#1 |
| 3. Test-Time Training (TARL) | +3-8% | 低-中 | 可作补充 |
| 4. Meta-Learned LR + 优化改进 | +2-5% | 无 | 工程优化 |
| 5. Ensemble Source Selection | +2-8% | 低 | 可作补充 |

### 推荐方案：Context-Conditioned World Model ✅ 已实现并验证

**核心思想**：用 ContextEncoder 从短历史推断建筑 context vector z，替代离散 adapter 路由。Transfer 时只需 ~10 transitions 推断 z，零梯度步骤。

**已实现文件**：
- `src/world_model/context_encoder.py` — ContextEncoder + ContextConditionedEncoder
- `src/world_model/context_dynamics.py` — ContextDynamicsModel
- `src/world_model/model_trainer.py` — ContextWorldModelTrainer
- CLI: `--context` flag for multi-train / transfer

**实验结果（3 seeds, source-only dict）**：
- 1d: +52.5±3.4%，3d: **+60.2±5.1%**，7d: +39.5±0.8%
- vs adapter: 3d 优势翻倍（+30% → +60%），7d std 从 11.6% 降至 0.8%

**论文叙事**：
- Shared dictionary D = 通用热物理基
- Context vector z = 建筑热指纹
- Sparse code alpha(s,a,z) = 该建筑在此状态如何使用物理基

### 关键引用

| 论文 | 期刊 | 相关性 |
|------|------|--------|
| CaDM (Lee et al.) | ICML 2020 | Context-conditioned dynamics, 3.5x 改善 |
| DALI | NeurIPS 2025 | Context encoder zero-shot, +96% |
| GrBAL (Nagabandi et al.) | ICLR 2019 | MAML for dynamics models |
| TARL | ICML 2025 | Test-time adaptation in RL |

---

## 附录 D：pyproject.toml

```toml
[project]
name = "dictdyna"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0",
    "numpy>=1.26",
    "gymnasium>=1.0",
    "sinergym>=3.11",
    "citylearn>=2.5",
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "omegaconf>=2.3",
    "pydantic>=2.9",
    "loguru>=0.7",
    "typer>=0.12",
    "wandb>=0.18",
    "ultraplot>=0.3",
]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "SIM", "RUF", "N"]
ignore = ["E501", "N805", "N806"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
```
