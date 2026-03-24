# PLAN.md — DictDyna 论文写作蓝图

**版本**: v2.0
**日期**: 2026-03-19
**状态**: 校核修订版 — 基于可行性审查更新

---

## 论文元信息

| 字段 | 内容 |
|------|------|
| **工作标题** | DictDyna: Dictionary Learning based Shared World Model for Transferable Model Based Reinforcement Learning in Building Energy Management |
| **目标期刊** | Applied Energy（首选）/ AAAI 2027（备选） |
| **格式要求** | Applied Energy: 单栏，无严格页数限制；AAAI 备选: 双栏 7 页 |
| **字数预算** | ~8000–10000 词（Applied Energy 典型长度） |
| **写作语言** | English |
| **审稿方式** | Applied Energy: Single-blind；AAAI: Double-blind |
| **附加要求** | CRediT author statement；Data availability statement；代码开源 |

---

## 贡献摘要

本文有四项贡献：

1. **方法创新**：首个将 dictionary learning 引入 MBRL world model 的工作，提出 shared dictionary dynamics model，将多建筑 transition dynamics 分解为共享 basis atoms 与建筑特定的 sparse codes。
2. **迁移学习**：提出 few-shot adaptation 机制——新建筑仅需学习轻量 adapter 参数即可复用共享字典，用 1–7 天数据即可达到有效控制。
3. **可解释性**：dictionary atoms 提供对建筑热动态模式的物理可解释聚类（如太阳得热响应、夜间散热衰减等）。
4. **多平台实验验证**：在 Sinergym（EnergyPlus 高保真热动态控制）和 CityLearn v2（社区级储能管理）两个互补平台上全面验证 sample efficiency、transfer 性能和 scalability，对比 model-free 和现有 MBRL baselines。

---

## 决策日志

| # | 决策 | 理由 | 日期 |
|---|------|------|------|
| D1 | ~~AAAI 为首选~~ → Applied Energy 为首选投稿目标 | 平台切换后论文偏向"方法+领域贡献"；Applied Energy 无页数限制，可充分展示双平台验证；滚动投稿无 deadline 压力 | 2026-03-19 |
| D2 | 采用方案 γ（混合）：离线预训练 dictionary，RL 训练中缓慢微调 | 兼顾稳定性与适应性，避免端到端联合优化不稳定 | 2026-03-19 |
| D3 | ~~CityLearn v2 作为核心实验平台~~ → Sinergym 为主平台，CityLearn 为补充 | CityLearn 的预模拟架构中 agent 仅控制储能设备，transition dynamics 不包含真实建筑热动态；Sinergym 的 EnergyPlus 后端提供真正的 HVAC 控制 → 热动态 world model | 2026-03-19 |
| D4 | ~~SAC (SB3)~~ → SAC (CleanRL 风格纯 PyTorch 实现) | SB3 是 model-free 框架，没有原生 Dyna 集成；CleanRL 风格单文件实现更透明，Dyna 训练循环集成更自然 | 2026-03-19 |
| D5 | Dyna-style 短视野 rollout（1–5 步） | 控制模型误差累积；MBPO 已验证有效性 | 2026-03-19 |
| D6 | ~~BDG2 数据集用于 dictionary pretraining~~ → Sinergym 离线 rollout 数据 | BDG2 提供单维能耗时序，与 Sinergym 多维状态空间（温度、湿度、HVAC 功率等）维度不匹配；使用 Sinergym 离线数据保证预训练字典与 RL 训练的状态空间完全一致 | 2026-03-19 |
| D7 | 多建筑通过多个 Sinergym 实例并行实现 | Sinergym 原生为单建筑设计；不同 IDF 文件 + 不同天气文件 = 不同建筑类型 | 2026-03-19 |
| D8 | Sinergym 中 reward 直接从预测状态计算，不需要单独 reward model | Sinergym reward = f(室内温度, HVAC 功率)，两者均为状态变量，world model 可直接预测 | 2026-03-19 |
| D9 | BDG2 保留为 Discussion 中 future work；BOPTEST 作为 future high-fidelity validation | BDG2 可用于大规模验证 atom 物理含义；BOPTEST Modelica 模型是更高保真度的验证路径 | 2026-03-19 |

---

## 章节结构与字数预算

| # | 章节 | 字数 | 占比 | 核心内容 |
|---|------|------|------|---------|
| 1 | Abstract | 300 | 3% | IMRAD 结构，3–5 个关键数字 |
| 2 | Introduction | 1200 | 13% | 背景 → 研究空白 → 贡献 |
| 3 | Related Work | 1200 | 13% | 三个主题：Building MBRL、Sparse WM、Multi-task WM |
| 4 | Methodology | 2800 | 30% | DictDyna 架构、数学推导、训练流程、迁移机制 |
| 5 | Experiments | 2500 | 27% | 双平台设置、主实验、消融、迁移、可解释性 |
| 6 | Discussion & Conclusion | 1000 | 11% | 稀疏性讨论、局限性、未来工作 |
| — | **合计** | **~9000** | 100% | — |

> 注：Applied Energy 无严格页数限制，字数预算从 6500 扩展到 ~9000，
> 允许更充分的实验描述和讨论。

---

## 第 1 章：Abstract（~300 词）

**最后写**，所有章节确认后再撰写。

### 结构（IMRAD）

1. **背景 + 空白**（2–3 句，~60 词）：
   - 建筑占全球能耗约 30%；RL 用于 HVAC 控制前景广阔但 sample efficiency 低。
   - 现有 MBRL 方法为每栋建筑学习独立模型，忽略了不同建筑共享的热动态模式。

2. **方法**（~100 词）：
   - 提出 DictDyna：基于 shared dictionary learning 的 MBRL world model。
   - Overcomplete dictionary $\mathbf{D}$ 捕获通用建筑热动态模式；per-building sparse encoder $g_\theta(\cdot;\phi_i)$ 编码个体差异。
   - 基于 Dyna-style planning 进行策略优化，在多种建筑上共享字典结构。

3. **结果**（~100 词）：
   - 在 Sinergym（EnergyPlus 高保真 HVAC 控制）上 N 种建筑实验：sample efficiency 较 model-free SAC 提升 X%，较 MB2C 提升 Y%。
   - CityLearn v2 上验证社区级 scalability。
   - Few-shot transfer 到新建筑仅需 Z 天数据即达全训练性能的 W%。
   - Dictionary atoms 呈现可解释的热动态模式。

4. **结论**（~40 词）：
   - DictDyna 实现了可扩展、可迁移、可解释的建筑能源管理 MBRL。

### 待填数字
- [ ] 相比 SAC 的 sample efficiency 提升（%）
- [ ] 相比 MB2C 的 sample efficiency 提升（%）
- [ ] Few-shot transfer 所需天数及达到全训练性能的比例
- [ ] 测试建筑数量与类型

---

## 第 2 章：Introduction（~1200 词）

### §2.1 研究背景（~300 词）

**核心论点：**
- 建筑消耗全球约 30% 的终端能源，贡献约 26% 的能源相关 CO₂ 排放（IEA, 2023）。HVAC 系统是商业建筑的主要能耗来源。
- RL 作为数据驱动的 HVAC 最优控制方法已被广泛研究，DQN、PPO、SAC 在仿真中实现了 10–30% 的节能（Khabbazi et al., 2025）。
- 然而，RL 在真实建筑中的部署面临核心瓶颈：**sample inefficiency**——每个 timestep 对应 1 小时真实运行，需要数月交互才能学到合理策略。

**必引文献：** IEA (2023), Khabbazi et al. (2025), Campoy-Nieves et al. (2025)

### §2.2 研究空白（~500 词）

**核心论点（3 个 gap）：**

**Gap 1：建筑能耗领域没有 latent world model。**
- Model-based RL 通过学习 transition model 并在 imagination 中规划来提升 sample efficiency。Dreamer (Hafner et al., 2023) 和 MBPO (Janner et al., 2019) 已在 robotics/games 中验证。
- 现有建筑 MBRL（MB2C、CLUE、HDMBPO）使用简单的 ensemble NNs 或 GP 作为 world model——没有任何工作采用 learned latent dynamics model。

**Gap 2：MBRL 中没有跨建筑知识共享。**
- 不同类型/气候区的建筑共享大量热动态模式（天气响应、日夜周期、围护结构热惯性），但所有现有方法为每栋建筑学习独立模型。
- Multi-task world model（Newt/Hansen et al., 2025; MrCoM/Xiong et al., AAAI 2026）在 robotics 领域已展示 shared world model 的跨场景泛化能力，但尚未应用于建筑。

**Gap 3：Sparse representation 在建筑 world model 中未被探索。**
- Dictionary learning 天然编码"共性 + 差异"：shared atoms $\mathbf{D}$ 捕获通用模式，sparse codes $\boldsymbol{\alpha}_i$ 编码建筑特性。
- Losse-FTL (Liu et al., ICLR 2024) 证明了 sparse features 在 RL world model 中的有效性，但仅在 Atari/MuJoCo 领域。
- Dynamic Sparsity (Pandaram et al., AAAI 2026) 分析了稀疏性假设的适用性——建筑热动态比机器人接触动力学具有更强的结构化稀疏性。

**必引文献：** Hafner et al. (2023), Janner et al. (2019), Ding et al. (2024) ×2, Gao et al. (2026), Hansen et al. (2025), Liu et al. (2024), Xiong et al. (2026), Pandaram et al. (2026)

### §2.3 贡献（~300 词）

**核心论点：**
- 简洁陈述 research question。
- 列出 4 项编号贡献（与上文"贡献摘要"一致），每项贡献对应一个 gap。
- 强调双平台验证的全面性。

### §2.4 论文组织（~50 词）

一句话路线图。

### 图表
- **[Figure 1]**：DictDyna 总体架构图（dictionary pretraining → Dyna RL training → few-shot transfer）。"主图"，应突出放置。

---

## 第 3 章：Related Work（~1200 词）

**组织方式**：按主题（Thematic），分 3 个方向

### §3.1 Model Based RL for Building Energy Management（~450 词）

**核心论点：**
- 简要回顾 model-free RL 在建筑控制中的应用（SAC、PPO、DQN——已成熟但数据需求大）。
- 详细回顾现有 MBRL：MB2C（ensemble NN + MPPI）、CLUE（GP + meta-kernel，7 天训练）、HDMBPO（物理-数据混合 MBPO）。
- 指出局限：全部是 per-building 独立模型，无 transfer 能力，无 learned latent dynamics。

**必引文献：** Ding et al. (2024) MB2C, Ding et al. (2024) CLUE, Gao et al. (2026), Khabbazi et al. (2025), Gao & Wang (2023), Campoy-Nieves et al. (2025)

### §3.2 World Model 中的 Sparse / Efficient Representation（~400 词）

**核心论点：**
- World model 受益于紧凑、结构化的表示以提高泛化能力和效率。
- Losse-FTL (ICLR 2024)：sparse encoding + linear regression 实现在线 world model，解决了 catastrophic forgetting。
- CompACT (2026)、Sparse Imagination (2025)、DDP-WM (2026)：视觉领域的 sparse/efficient world model 方法。
- Dictionary learning in RL：Tang et al. (2022) DLRL-ASR 系列——唯一将 dictionary learning 与 RL 结合的工作，但未用于 world model。
- **Dynamic Sparsity (Pandaram et al., AAAI 2026)**：分析了 MuJoCo 中 world model 稀疏性假设的有效性，发现全局稀疏性罕见但局部 state-dependent 稀疏性存在。DictDyna 的差异：建筑热动态受天气驱动，具有更强的周期性结构化稀疏性，与机器人接触动力学不同。
- **空白**：没有工作将 dictionary learning 与 world model 结合。

**必引文献：** Liu et al. (2024), Tang et al. (2022), Chun et al. (2025), Kim et al. (2026), Yin et al. (2026), **Pandaram et al. (AAAI 2026)**

### §3.3 Multi-task World Model 与 RL 中的 Transfer（~350 词）

**核心论点：**
- Newt (Hansen et al., 2025)：大规模 multi-task world model，shared representation 用于连续控制。
- **MrCoM (Xiong et al., AAAI 2026)**：Meta-regularized contextual world model，通过 latent state decomposition + meta-regularization 实现跨场景泛化。在 DMControl 验证。DictDyna 与 MrCoM 的区别：DictDyna 用 dictionary learning 提供更强的结构化先验（overcomplete basis + sparsity），且 dictionary atoms 具有物理可解释性，这是 MrCoM 的 latent decomposition 不具备的。
- CityLearn v2 (Nweye et al., 2024)：标准多建筑 RL benchmark，但所有方法均为 model-free。
- Feature decorrelation (Lee et al., 2023) 提升 RL 表示质量——与 dictionary learning 的 coherence reduction 思想一致。
- **空白**：没有 shared world model 用于多建筑能源管理；没有基于 dictionary 的 transfer 方法。

**必引文献：** Hansen et al. (2025), **Xiong et al. (AAAI 2026)**, Nweye et al. (2024), Lee et al. (2023)

### 总结段落
- 综合：dictionary learning × world model × building energy management 的交叉领域完全空白。DictDyna 填补这一空白。

---

## 第 4 章：Methodology（~2800 词）

核心贡献章节，分配最多篇幅。

### §4.1 问题定义（~350 词）

**核心论点：**
- 将多建筑 HVAC 控制定义为一组 MDP $\{\mathcal{M}_i\}_{i=1}^N$，共享相同动作空间结构但具有不同 transition dynamics。
- 每个 MDP：$\mathcal{M}_i = (\mathcal{S}, \mathcal{A}, T_i, R_i, \gamma)$
  - $\mathcal{S} \subseteq \mathbb{R}^d$：状态空间（室内/室外温度、湿度、太阳辐射、HVAC 运行状态、时间特征等）
  - $\mathcal{A} \subseteq \mathbb{R}^m$：动作空间（供暖/制冷温度设定点）
  - $T_i: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：建筑特定的 transition dynamics（由建筑围护结构、HVAC 系统特性决定）
  - $R_i$：奖励函数（能耗 + 热舒适惩罚）
  - $\gamma$：折扣因子
- 关键洞察：不同建筑的 $T_i$ 共享结构相似性——天气驱动的热响应、HVAC 系统的基本物理特性是通用的，建筑围护结构参数和系统容量不同。
- 目标：学习一个 shared world model，捕获共有热动态结构的同时高效适配个体建筑。
- **Sinergym 环境特点**：agent 的动作（设定点调整）直接影响室内温度、HVAC 能耗等状态变量，transition dynamics 完整包含建筑热物理过程。

**数学约定：**
- 所有符号首次出现时定义
- 下标约定：$i$ 表建筑索引，$t$ 表时间步，$k$ 表 dictionary atom 索引

### §4.2 DictDyna 架构概览（~250 词）

**核心论点：**
- 三阶段架构：(I) Dictionary Pretraining，(II) Dyna-style RL Training，(III) Few-shot Transfer。
- 引用 [Figure 1] 展示总体架构。
- 引入核心 world model 方程：

$$\hat{s}_{t+1}^{(i)} = s_t^{(i)} + \mathbf{D} \cdot g_\theta\left(s_t^{(i)}, a_t^{(i)}; \phi_i\right) \tag{1}$$

- 简要说明各组件含义，详细推导留给后续小节。

**图表：**
- **[Figure 1]**：完整架构图（若未在 Introduction 放置）

### §4.3 Phase I: Dictionary Pretraining（~500 词）

**核心论点：**
- 使用 Sinergym 环境中以 RBC 或随机策略收集的**离线交互数据**预训练 shared dictionary。
- 对每种建筑类型/气候区组合，运行 1-2 年模拟收集 $(s_t, a_t, s_{t+1})$ 轨迹。
- 从轨迹中计算状态差分 $\Delta s_j = s_{j+1} - s_j$ 作为训练信号——这保证了预训练字典的状态空间与 RL 训练**完全一致**。
- Dictionary learning 目标函数：

$$\min_{\mathbf{D}, \{\boldsymbol{\alpha}_j\}} \sum_{j=1}^{N_{\text{data}}} \left\| \Delta s_j - \mathbf{D} \boldsymbol{\alpha}_j \right\|_2^2 + \lambda \|\boldsymbol{\alpha}_j\|_1 \quad \text{s.t.} \quad \|\mathbf{d}_k\|_2 = 1, \forall k \tag{2}$$

- 逐项解释：
  - $\mathbf{D} \in \mathbb{R}^{d \times K}$：overcomplete dictionary（$K > d$），每列 $\mathbf{d}_k$ 是一个"热动态模式原子"
  - $\boldsymbol{\alpha}_j \in \mathbb{R}^K$：样本 $j$ 的 sparse coefficient vector
  - $\lambda$：sparsity regularization 权重，控制重构精度与稀疏度之间的平衡
  - 原子单位范数约束防止平凡缩放解
- 算法：K-SVD 或 Online Dictionary Learning (Mairal et al., 2009)
- **预训练数据多样性**：不同建筑类型（办公、数据中心）和不同气候区（hot-dry、mixed、cool-humid）的离线数据混合训练，确保字典原子覆盖多种热动态模式。
- 讨论预期学到的 atoms：太阳得热模式、制冷/供暖瞬态、天气锋面响应等。

**数学：** Eq. (2) + K-SVD 更新规则（简要，引用原论文）

### §4.4 Phase II: Dyna-style RL Training（~800 词）

**核心论点：**

**§4.4.1 Sparse Encoder Network（~300 词）：**
- Sparse encoder $g_\theta(\cdot; \phi_i)$ 是一个轻量 MLP，将 $(s_t, a_t)$ 映射到 sparse activation $\boldsymbol{\alpha} \in \mathbb{R}^K$。
- 架构：shared layers（参数 $\theta$）+ building-specific adapter layer（参数 $\phi_i$）。
- 稀疏性强制方式：$\ell_1$ penalty、proximal gradient step 或 top-$k$ hard thresholding。
- Adapter $\phi_i$ 很小（例如单线性层或 low-rank adaptation），每栋建筑的额外参数量极少。

**§4.4.2 World Model Training（~250 词）：**
- World model 损失函数：

$$\mathcal{L}_{\text{WM}} = \mathbb{E}\left[\left\|s_{t+1}^{(i)} - \hat{s}_{t+1}^{(i)}\right\|_2^2\right] + \lambda_s \left\|g_\theta(s_t^{(i)}, a_t^{(i)}; \phi_i)\right\|_1 \tag{3}$$

- Dictionary 更新策略：固定或以降低的学习率缓慢微调（$\eta_D \ll \eta_\theta$）。
- 慢更新的理由：dictionary 应保持为稳定的 shared basis，类似 complementary learning systems 中的"slow weights"。

**§4.4.3 Dyna Planning 集成（~250 词）：**
- 每个真实环境步骤：
  1. 在 Sinergym 真实环境中执行动作 $a_t$，观察 $(s_{t+1}, r_t)$。
  2. 更新 world model 参数（$g_\theta$，可选更新 $\mathbf{D}$）。
  3. 从当前状态出发，用 world model 生成 $M$ 条长度为 $H$ 的 simulated rollouts。
  4. **Reward 估计**：Sinergym 的 reward 函数仅依赖状态变量（室内温度、HVAC 功率），因此可直接从 world model 预测的 $\hat{s}_{t+1}$ 计算 reward，无需单独的 reward model。
  5. 将 simulated transitions 加入 replay buffer。
  6. 用 mixed real + simulated data 更新 policy（SAC）。
- Rollout horizon $H \in \{1, 2, 3, 5\}$：短视野控制 compounding error（遵循 MBPO 原则）。
- Model-to-real data ratio：超参数，控制 simulated vs. real experience 的比例。

**图表：**
- **[Figure 2]**：Phase II 详细计算流程图
- **[Algorithm 1]**：DictDyna Training Procedure（伪代码）

### §4.5 Phase III: Few-shot Transfer（~350 词）

**核心论点：**
- 对于训练时未见过的新建筑 $j$：
  1. 固定 shared dictionary $\mathbf{D}$ 和 shared encoder 参数 $\theta$。
  2. 初始化新 adapter $\phi_j$（随机或从最相似的已有建筑复制）。
  3. 收集少量交互数据（1–7 天）。
  4. 仅微调 $\phi_j$。
  5. 用 adapted world model 进行 Dyna planning。
- 理论解释：shared dictionary 张成了通用热动态空间；adaptation 只需学习新建筑在此空间中的"坐标"。
- 参数效率：新增一栋建筑仅需 $|\phi_j|$ 个新参数，远少于从头训练的 $|\theta| + |\phi|$。
- **迁移场景**：
  - 同建筑类型跨气候区（同一 IDF，不同天气文件）
  - 跨建筑类型（不同 IDF 文件）
  - 跨平台（Sinergym → CityLearn，验证泛化性）

**数学：**
$$\phi_j^* = \arg\min_{\phi_j} \sum_{t} \left\|s_{t+1}^{(j)} - s_t^{(j)} - \mathbf{D} \cdot g_\theta(s_t^{(j)}, a_t^{(j)}; \phi_j)\right\|_2^2 + \lambda_s \|g_\theta(\cdot; \phi_j)\|_1 \tag{4}$$

### §4.6 计算复杂度分析（~200 词）

**核心论点：**
- 参数量对比：DictDyna（$|\mathbf{D}| + |\theta| + N|\phi|$）vs. 独立模型（$N \times |\theta_{\text{full}}|$）。
- Sublinear scaling：随 $N$ 增大，shared components（$\mathbf{D}$, $\theta$）被摊销。
- 单步推理成本：一次 MLP forward pass + 一次矩阵乘法——与 ensemble NN 相当或更低。

### 第 4 章图表汇总
- **[Figure 1]**：DictDyna 三阶段架构总览
- **[Figure 2]**：Phase II 详细计算流程
- **[Algorithm 1]**：DictDyna training procedure
- **[Table 1]**：符号表（notation definitions）

---

## 第 5 章：Experiments（~2500 词）

### §5.1 实验设置（~600 词）

**§5.1.1 主平台：Sinergym（~250 词）**
- Sinergym v3.11 (Campoy-Nieves et al., Energy and Buildings, 2025)：EnergyPlus 后端，Gymnasium API。
- Agent 直接控制 HVAC 温度设定点 → 真正影响建筑热动态。
- 建筑配置：多种建筑类型 × 多种气候区
  - 建筑类型：5zone office, datacenter, residential（不同 IDF 文件）
  - 气候区：hot-dry (USA_AZ_Tucson), mixed (USA_NY_NewYork), cool-humid (USA_WA_Seattle)
  - 总计 N 种建筑-气候组合
- 状态空间：~10–15 维（室内/室外温度、湿度、直射/散射辐射、HVAC 功率、时间特征等）
- 动作空间：1–2 维连续（供暖/制冷温度设定点）
- 奖励：能耗 + 热舒适违规的加权组合（Sinergym 标准 reward）

**§5.1.2 补充平台：CityLearn v2（~100 词）**
- CityLearn v2.5 (Nweye et al., 2024)：社区级多建筑 RL benchmark。
- 用于：(1) scalability 实验（5→20→50 栋建筑），(2) 验证 DictDyna 在储能管理 dynamics（非热动态）上的泛化性。
- 注意：CityLearn 中 agent 控制储能设备（电池充放电），transition dynamics 是储能系统动态而非建筑热动态。这一差异恰好证明 DictDyna 的 dictionary 结构对不同类型 dynamics 的泛化能力。

**§5.1.3 Baselines（~100 词）：**
- RBC（Rule-Based Control）：Sinergym 默认 + CityLearn 内置
- SAC（model-free）：CleanRL 风格 PyTorch 实现
- MB2C（MBRL, ensemble NN）：复现
- CLUE（MBRL, GP）：复现
- Independent DictDyna（per-building dictionary，无共享）：消融 baseline
- MARLISA（MARL）：CityLearn 上的额外 baseline

**§5.1.4 Dictionary Pretraining（~100 词）：**
- 数据来源：Sinergym 离线 rollout（RBC 策略 + 随机策略，每种建筑-气候组合 2 年数据）
- 预处理：归一化、计算状态差分 $\Delta s$、过滤异常值
- K-SVD，$K$ 个 atoms（$K = 64, 128, 256$）

**§5.1.5 超参数：**
- 以表格形式报告关键超参数。

**图表：**
- **[Table 2]**：超参数汇总
- **[Table 3]**：Sinergym + CityLearn 环境配置

### §5.2 主实验：Sample Efficiency（~500 词）

**核心论点：**
- **Sinergym 主实验**：对比学习曲线——DictDyna vs. 所有 baselines，横轴为 episodes/years。
- 报告 cumulative reward、energy savings（%）、comfort violations（收敛后）。
- 展示 DictDyna 比 model-free 方法收敛更快，最终性能优于现有 MBRL。
- 统计显著性：报告 M 个 random seeds 的 mean ± std。
- **CityLearn 补充**：在 CityLearn 上运行相同方法，展示跨平台一致性。

**图表：**
- **[Figure 3]**：学习曲线（cumulative reward vs. episodes），Sinergym 主实验
- **[Table 4]**：最终性能对比（energy savings、comfort），双平台

### §5.3 迁移实验（~500 词）

**核心论点：**
- **同类型跨气候区迁移**：在 3 种气候区的 office 建筑上训练 → 迁移到第 4 种气候区。
- **跨建筑类型迁移**：在 office 建筑上训练 → 迁移到 datacenter/residential。
- 对比：DictDyna transfer（固定 $\mathbf{D}$，微调 $\phi$）vs. 从头训练 vs. 直接 zero-shot。
- 报告不同 adaptation 数据量下的性能（1、3、7 天）。
- **跨平台迁移（如可行）**：Sinergym 训练 → CityLearn 适配，展示更广泛的泛化性。

**图表：**
- **[Figure 4]**：Transfer 性能 vs. adaptation 数据量
- **[Table 5]**：Transfer 结果（1/3/7 天的 energy savings）

### §5.4 消融实验（~400 词）

**核心论点：**

| 消融 | 问题 | 预期发现 |
|------|------|---------|
| Random vs. pretrained $\mathbf{D}$ | Pretraining 是否重要？ | 预训练 dictionary 收敛更快 |
| Fixed vs. fine-tuned $\mathbf{D}$ | Slow update 是否必要？ | Fine-tuning 有轻微益处 |
| Shared vs. independent dictionary | Sharing 是否有帮助？ | Shared 在 transfer 中优势显著 |
| Dictionary size $K$ | 需要多少 atoms？ | 性能在 $K^*$ 附近趋于饱和 |
| Sparsity $\lambda$ | 多稀疏合适？ | 中等稀疏度最优 |

**图表：**
- **[Figure 5]**：消融实验结果（bar chart 或 line plot）

### §5.5 可解释性分析（~300 词）

**核心论点：**
- 可视化 top dictionary atoms 的时间模式。
- **在 Sinergym 上，atoms 的物理含义更加直接**：
  - 原子直接对应 "HVAC 设定点变化 → 室内温度响应" 的模式
  - 可以叠加天气数据做条件分析（日照强时 vs. 阴天）
  - 不同气候区激活不同的 atom 子集
- 展示 atoms 对应物理有意义的热动态（太阳得热、夜间散热、HVAC 响应延迟）。
- 分析不同建筑类型中哪些 atoms 被最频繁激活——展示 shared vs. building-specific patterns。

**图表：**
- **[Figure 6]**：学到的 dictionary atoms 可视化

### §5.6 Scalability（~200 词）

**核心论点：**
- 在 CityLearn 上做 scaling 实验：5 → 20 → 50 栋建筑。
- 报告 dictionary atom 数量与建筑数量的关系——验证亚线性增长。
- 报告参数量对比：DictDyna vs. 独立建模。

**图表：**
- **[Figure 7]**：Scaling 曲线（atoms / 参数量 vs. 建筑数）

---

## 第 6 章：Discussion & Conclusion（~1000 词）

### §6.1 Discussion（~600 词）

**启示：**
- DictDyna 表明，通过 dictionary learning 进行结构化表示共享可显著降低 RL 在真实建筑中部署的数据需求。
- 可解释的 dictionary atoms 弥合了黑箱 RL 与领域理解之间的鸿沟。

**与文献对比：**
- 与 MB2C/CLUE 对比：DictDyna 的优势来自 shared representation，而不仅仅是 model-based planning。
- 与 MrCoM (Xiong et al., AAAI 2026) 对比：DictDyna 的 dictionary structure 为建筑热动态提供了更强的 inductive bias；dictionary atoms 具有 MrCoM latent decomposition 不具备的物理可解释性。

**关于稀疏性假设（新增，重要）：**
- Dynamic Sparsity (Pandaram et al., AAAI 2026) 表明在机器人任务中全局稀疏性假设不一定成立，真正存在的是局部的、state-dependent 稀疏性。
- 然而，建筑热动态与机器人动力学在本质上不同：
  1. 建筑热响应是缓变的（以小时为尺度），没有机器人接触事件那样的突变不连续性。
  2. 驱动力（天气、占用）具有强周期模式（日夜、季节），天然适合被有限数量的字典原子捕获。
  3. 不同建筑的热动态共享同一组物理机制（传导、对流、辐射），只是参数不同——这是 dictionary learning"共性+差异"框架的理想应用场景。
- Figure 6 的 atoms 可视化提供了对此假设的直接实证验证。

**局限性（诚实、建设性）：**
- Sinergym 虽然使用 EnergyPlus，但仍是仿真；真实建筑动态更复杂（传感器噪声、未建模的扰动）。建议后续用 BOPTEST（Modelica 高保真模型）或现场测试验证。
- Dictionary size $K$ 目前需要 grid search；自适应方法值得探索。
- 当前假设所有建筑共享相同的 state/action spaces；异构空间需要扩展。
- 预训练数据来自仿真环境；未来可探索使用 BDG2 等大规模真实建筑数据（需解决维度匹配问题）。

### §6.2 Conclusion（~400 词）

**核心论点：**
1. 重述：提出 DictDyna，首个基于 dictionary learning 的 shared world model 用于建筑 MBRL。
2. 关键发现（4 条编号）：
   - (1) 相比 model-free baselines 提升 X% sample efficiency。
   - (2) 仅 Z 天 adaptation 数据即可有效迁移至新建筑。
   - (3) Dictionary atoms 揭示可解释的热动态模式。
   - (4) 参数量随建筑数 sublinear 增长。
3. 未来工作：
   - 通过 BOPTEST 高保真 Modelica 模型或现场测试进行真实部署验证。
   - 利用 BDG2 等大规模真实建筑数据进行字典预训练，扩大适用范围。
   - 自适应 dictionary expansion 以支持异构建筑类型。
   - 将 physics-informed priors 融入 dictionary atoms。
4. 收尾：DictDyna 开辟了 dictionary learning × world model × 可持续建筑控制交叉方向的新路线。

---

## 图表清单

| ID | 类型 | 描述 | 章节 | 状态 |
|----|------|------|------|------|
| Fig.1 | 示意图 | DictDyna 三阶段架构总览 | §2/§4 | ⬜ |
| Fig.2 | 示意图 | Phase II 详细计算流程 | §4 | ⬜ |
| Fig.3 | 折线图 | 学习曲线（reward vs. episodes），Sinergym 主实验 | §5.2 | ⬜ |
| Fig.4 | 折线图 | Transfer 性能 vs. adaptation 数据量 | §5.3 | ⬜ |
| Fig.5 | 柱状图 | 消融实验结果 | §5.4 | ⬜ |
| Fig.6 | 热力图 | Dictionary atoms 可视化 | §5.5 | ⬜ |
| Fig.7 | 折线图 | Scaling 曲线（atoms/参数量 vs. 建筑数） | §5.6 | ⬜ |
| Alg.1 | 伪代码 | DictDyna training procedure | §4 | ⬜ |
| Tab.1 | 表格 | 符号定义（notation） | §4 | ⬜ |
| Tab.2 | 表格 | 超参数汇总 | §5.1 | ⬜ |
| Tab.3 | 表格 | Sinergym + CityLearn 环境配置 | §5.1 | ⬜ |
| Tab.4 | 表格 | 主实验性能对比（双平台） | §5.2 | ⬜ |
| Tab.5 | 表格 | Transfer 实验结果 | §5.3 | ⬜ |

---

## 参考文献规划

### 必引（核心方法）
1. Liu et al. (ICLR 2024) — Losse-FTL, sparse encoding world model
2. Tang et al. (2022) — DLRL-ASR, dictionary learning in RL
3. Hafner et al. (2023) — DreamerV3, latent world model SOTA
4. Janner et al. (NeurIPS 2019) — MBPO, Dyna-style MBRL
5. Aharon et al. (2006) — K-SVD algorithm

### 必引（建筑 MBRL）
6. Ding et al. (2024) — MB2C
7. Ding et al. (2024) — CLUE
8. Gao et al. (2026) — HDMBPO

### 必引（平台）
9. **Campoy-Nieves et al. (Energy and Buildings, 2025) — Sinergym**
10. Nweye et al. (2024) — CityLearn v2

### 必引（新增——校核后发现的关键文献）
11. **Xiong et al. (AAAI 2026) — MrCoM, meta-regularized multi-scenario world model**
12. **Pandaram et al. (AAAI 2026) — Dynamic Sparsity, sparsity assumptions in world models**

### 应引（上下文）
13. Khabbazi et al. (2025) — 现场部署综述
14. Hansen et al. (2025) — Newt, multi-task world model
15. Lee et al. (2023) — Feature decorrelation in RL
16. Chun et al. (2025) — Sparse Imagination
17. Kim et al. (2026) — CompACT
18. Blum et al. (2021) — BOPTEST framework

### 可引（空间允许时）
19. Ha & Schmidhuber (2018) — World Models
20. Gao & Wang (2023) — Model-based vs model-free 对比
21. IEA 建筑能耗统计
22. Miller et al. (2020) — BDG2（在 Discussion future work 中提及）

---

## 术语映射表

| 术语 | 使用场景 | 避免混用 |
|------|---------|---------|
| dictionary atom | $\mathbf{D}$ 的列向量的正式称呼 | basis vector, component |
| sparse code / sparse coefficient | $\boldsymbol{\alpha}$ | activation, encoding（有歧义） |
| world model | 学到的 transition dynamics | environment model, simulator |
| adapter / adapter layer | 建筑特定参数 $\phi_i$ | fine-tuning layer |
| few-shot transfer | 用少量数据适配 | zero-shot, one-shot（除非精确） |
| Dyna-style planning | 将 model-based planning 集成到 model-free RL | imagination, dreaming |
| thermal dynamics | 建筑传热过程 | thermal behavior（过于模糊） |

---

## 推荐写作顺序

1. **第 4 章：Methodology** — 核心贡献，决定其他所有章节
2. **第 5 章：Experiments** — 方法章节的自然延伸
3. **第 2 章：Introduction** — 写完方法后更容易把握定位
4. **第 3 章：Related Work** — Introduction 后细化定位
5. **第 6 章：Discussion & Conclusion** — 有了完整上下文才能解读
6. **第 1 章：Abstract** — 最后写

---

## 写作前 TODO

- [ ] 复现 Losse-FTL toy example（Phase 0 检查点）
- [ ] 搭建 Sinergym v3.11 + EnergyPlus 环境，跑通官方 tutorial
- [ ] 在 Sinergym 单建筑上运行 model-free SAC baseline，建立 reward 基准
- [ ] 用 Sinergym 离线 rollout 数据做字典预训练的初步实验（K-SVD）
- [ ] 实现 DictDyna world model + CleanRL SAC + Dyna training loop
- [ ] 搭建多 Sinergym 实例的多建筑环境封装
- [ ] 运行 Sinergym 主实验 + CityLearn 补充实验
- [ ] 运行消融实验和迁移实验
- [ ] 生成所有论文图表
- [ ] 按本计划完成论文写作
