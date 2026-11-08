# 训练算法设计

本文说明两个任务各自的奖励函数、策略/价值网络结构，以及 PPO 一整个迭代里「采集 → 计算优势 → 更新网络」的完整流程：

- **静态站立任务**（`Isaac-M20-Wheeled-v0`）：把速度指令固定为零，即"站稳"。§1–§5 以它为主线。
- **速度指令任务**（`Isaac-M20-Wheeled-Velocity-v0`）：跟踪 `(vx, vy, ωz)`，5 阶段课程 + 阶段间续训，见 [§6](#6-速度指令任务-m20-第一版正式配置)。

配置入口：

- 奖励权重与观测/动作：`tasks/m20_env_cfg.py`（站立）、`tasks/m20_velocity_env_cfg.py`（速度指令）
- 奖励公式：`tasks/rewards.py`；观测项与指令采样：`tasks/observations.py`、`tasks/commands.py`
- 网络与 PPO 超参：`tasks/agents/rsl_rl_ppo_cfg.py`
- 课程与续训：`scripts/train.py`；验收测试：`scripts/eval.py`

## 1. 奖励函数（站立任务）

采用 legged_gym（Rudin et al., *Learning to Walk in Minutes*）与 IsaacLab `AnymalC` 任务的标准奖励集，把速度指令固定为零，即"站稳"。

**符号约定**：惩罚项的函数返回**正量纲值**，符号由权重携带——与 `isaaclab.envs.mdp` 一致（例如 `lin_vel_z_l2` 返回 `v_z²`，配置里给权重 −2.0）。所以下表「公式」列写的都是函数返回值，`rewards.py` 里不出现前导负号。函数返回负值 × 负权重 = 正奖励，会把惩罚项变成激励。

| 分项 | 权重 | 函数返回 | 含义 |
| --- | --- | --- | --- |
| `track_lin_vel_xy` | **+1.5** | `exp(-‖v_xy‖² / 0.25)` | 线速度跟踪核，指令为 0 → 越静止越接近 1 |
| `track_ang_vel_z` | **+0.75** | `exp(-ω_z² / 0.25)` | 偏航角速度跟踪核，指令为 0 |
| `track_wheel_vel` | **+0.5** | `exp(-‖ω_wheel‖² / 2.0²)` | 四个轮子静止（轮腿机器人专属） |
| `lin_vel_z` | −2.0 | `v_z²` | 禁止上下颠簸 |
| `ang_vel_xy` | −0.05 | `Σ ω_xy²` | 禁止翻滚/俯仰角速度 |
| `flat_orientation` | −2.5 | `gx² + gy²` | 机身水平（`g` 为重力在机体系的投影） |
| `base_height` | **−20.0** | `(h − 0.50)²` | 基座高度二次惩罚（论文用的是二次型，不是 exp 奖励） |
| `joint_deviation` | −0.5 | `Σ_legs abs(q − q_default)` | **仅 12 个腿关节**偏离默认站立姿态（legged_gym 的 `stand_still`） |
| `joint_vel` | −1e-4 | `Σ q̇²` | 关节速度正则 |
| `joint_acc` | −2.5e-7 | `Σ q̈²` | 关节加速度正则，抑制高频抖动 |
| `joint_torques` | −1e-5 | `Σ τ²` | 省力矩 |
| `action_rate` | −0.01 | `Σ(aₜ − aₜ₋₁)²` | 动作平滑 |
| `termination_penalty` | **−200** | `1[非超时终止]` | 摔倒的重大惩罚 |
| `alive` | +0.15 | `1` | 存活奖励 |

**`termination_penalty` 是最关键的一项**。缺少它时，摔倒的代价接近零，而 `action_rate` / `joint_torques` 等正则项是每步累积的——于是"尽快摔倒结束回合"反而收益更高，策略会主动学怎么更快结束（表现为 episode length 持续下降）。论文里这一项通常是 −200 到 −400。

**`base_height` 必须是二次惩罚**：权重 −20 作用在 `(h−h*)²` 上。换成 `exp` 奖励形式时，远离目标高度的区域梯度几乎为零，等于没有推力。

**轮组的两个专门处理**（轮腿机器人区别于普通四足的地方）：

1. `track_wheel_vel` 用 exp 核把四个轮速压到零。不能用 L2：`joint_vel` 是 −1e-4 的量级，而轮速量纲是 rad/s，ω ≈ 1 rad/s 时贡献只有 −4e-4，**零速附近的梯度几乎是死的**；exp 核在原点梯度非零且有界。σ_w = 2.0 rad/s 吸收了四个轮子的求和。
2. `joint_deviation` **只算 12 个腿关节**。轮子的绝对角度是无界的（`continuous` revolute，不 wrap），转两圈就是 4π → 每步罚 6.3，能把整套奖励淹没。轮子的"别动"由 `track_wheel_vel` 负责，不用关节角。

**终止条件**（触发即本回合结束、重置）：

- `base_height`：基座高度 < 0.25 m（趴下）
- `bad_orientation`：倾角 > 0.7 rad（翻倒）
- `time_out`：满 10 s（50 Hz × 200 步），**超时不触发 `termination_penalty`**

## 2. 网络设计

rsl_rl 的 Actor-Critic，两个 MLP 共享同一份观测输入（`obs_groups` 里 actor 和 critic 都吃 `policy` 组）。

**站立任务**：

```text
观测 o ∈ R^57
  │
  ├─ Actor  (策略)    57 → 512 → 256 → 128 → μ∈R^16
  │                     ELU    ELU    ELU    (对角高斯, init σ=1.0)
  │
  └─ Critic (价值)    57 → 512 → 256 → 128 → V∈R^1
                        ELU    ELU    ELU
```

**速度指令任务**（`M20VelocityRslRlPpoRunnerCfg`）：

```text
观测 o ∈ R^56
  │
  ├─ Actor  (策略)    56 → 256 → 256 → μ∈R^16    ELU, 对角高斯 init σ=0.9
  └─ Critic (价值)    56 → 256 → 256 → V∈R^1     ELU
```

- **站立观测 57 维**：`base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(16) + joint_vel(16) + last_action(16)`。
- **速度指令观测 56 维**：见 §6.1，多 3 维指令、少 4 维（腿/轮关节拆开算）。
- **动作 16 维** ∈ [-1,1]，都是**目标值**，不是力矩：
  - 腿 12 维：`q* = q_default + a × 0.25 rad`（`JointPositionActionCfg`）
  - 轮 4 维：`ω* = a × scale`（`JointVelocityActionCfg`）；站立 `scale=20`，速度指令任务 `scale=8.0`
- **PD 伺服环**（`ImplicitActuatorCfg`）在 200 Hz 物理步上把目标跟踪成力矩：腿 kp=80、kd=2，轮 kp=0、kd=2（速度伺服），力矩上限 76.4 / 21.6 N·m。策略负责决策，PD 只负责跟踪。
- **激活** ELU；**无观测归一化**（`actor_obs_normalization=False`）。
- **策略头**：输出均值，方差固定为 `init_noise_std` 的对角高斯（不学方差），采样得到动作。

站立任务关键超参（`RslRlPpoAlgorithmCfg`）：γ=0.99、λ(GAE)=0.95、clip=0.2、熵系数 0.01、价值损失系数 1.0、`num_learning_epochs=5`、`num_mini_batches=4`、lr=1e-3（`schedule="adaptive"`，KL 超 0.01 就降 lr）、`max_grad_norm=1.0`。速度指令任务的超参见 §6.5。

## 3. 运行与更新流程

**时间尺度**：物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`，每 4 个物理步出一次动作）。站立回合 10 s = 200 个策略步，速度指令回合 20 s = 1000 个策略步。

**一次 PPO 迭代**：

1. **采集**：N 个并行环境各推 `num_steps_per_env=24` 步，得到 N × 24 条 (oₜ, aₜ, rₜ, oₜ₊₁, done) 样本（站立 N=4096 → 98,304；速度指令 N=512 → 12,288，N=1024 → 24,576）。
2. **算优势**：Critic 给出每步 `V(o)`，用 GAE（γ=0.99、λ=0.95）沿时间反向算优势 `Âₜ`，回报 `Rₜ = Âₜ + V(oₜ)` 作为价值回归目标。
3. **多轮更新**：把样本打乱分成 minibatch，循环 `num_learning_epochs` 遍，每遍对 Actor 和 Critic 各做一次梯度下降：
   - **Critic**：`loss_V = 0.5·(V − R)²`（价值损失系数 1.0）；
   - **Actor**：PPO-clip 目标 `min(rₜ·Â, clip(rₜ, 1±0.2)·Â) + entropy_coef·熵`，`rₜ = π(a|o)/π_old(a|o)`；
   - 梯度范数裁剪到 1.0。
4. **自适应学习率**：每遍后若新旧策略 KL 超过 `desired_kl`，lr 自动下调。
5. 清空 buffer，用更新后的策略进入下一轮采集。

**训练总量**：站立 `max_iterations=3000` 轮，每 `save_interval=100` 轮存一个 `model_*.pt`；速度指令任务分 5 阶段共 3950 轮，详见 §6.6。

**为什么直驱力矩开局会大面积趴下**：没有 PD 兜底，随机初始化的策略输出的是乱力矩，机器人生成就倒。这属正常——`base_height`/`flat_orientation` 的信号会把策略逐步拉向"撑住"的力矩组合，通常几百轮内趴下率开始下降。

## 4. 域随机化

站立任务在 `tasks/m20_env_cfg.py` 的 `EventCfg` 里全开；**速度指令任务第一版默认关闭**（见 §6.7），只保留重置时的小扰动。

`startup` 类在场景建立时抽一次，`reset` 类在**每个环境每次重置时重新抽**——策略面对的不是一台机器人，而是一族机器人。

| 项 | 模式 | 范围 | 目的 |
| --- | --- | --- | --- |
| `physics_material` | startup | 静/动摩擦 0.5–1.25，restitution 0 | 地面/轮子材质差异 |
| `reset_base` | reset | xy ±0.10 m，z 0–0.04 m，roll/pitch ±0.10 rad，yaw ±0.30 rad；线速度 ±0.1 m/s | 初始姿态偏差 |
| `reset_joints` | reset | **腿关节** ±0.15 rad，速度 ±0.5 rad/s | 编码器/标定的初始误差 |
| `reset_wheels` | reset | 轮关节角度归零（无偏移） | 连续关节角度会跨回合累加，必须清零 |
| `base_mass` | startup | 机身质量 **+(-2.0, +3.0) kg**（叠加），重算惯量 | 载荷变化（相机、电池、雷达） |
| `base_com` | startup | 质心 x/y ±0.03 m，z ±0.02 m | 载荷不在几何中心 |
| `push_robot` | interval | 每 4–8 s 给 x/y 一个 ±0.5 m/s 的瞬时速度 | 外力扰动、抗推 |

### 课程调度

站立任务上表写的是**满幅范围**，实际生效的是 `满幅 × level`，`level ∈ [0, 1]`（`tasks/dr.py`）：

```text
level = clamp(iteration / (max_iterations × --dr_ramp), 0, 1)
```

默认 `--dr_ramp 0.4`，3000 轮就是**前 1200 轮从 0 线性爬满**，之后固定 1.0。`level = 0` 时等价于完全没有随机化的基线。

为什么要有这条：满幅 DR 从第 0 轮开满，策略还没学会站稳就被随机质量和推力反复打断，表现就是**存活率先升后降**——它学到的那点站立能力被越来越难的扰动抹掉，而它又还没强到能适应。课程把任务始终放在"略高于当前能力"的区间里。

注意 `reset_wheels` **不参与课程**（它用 `mdp.` 而不是 `dr.`），因为它不是随机化，是每回合必须执行的清零。质量和质心是 `startup` 项，抽一次就固定，也不参与课程。

几点设计取舍：

- **质量和质心必须是 `startup`**。IsaacLab 的 `randomize_rigid_body_mass` / `randomize_rigid_body_com` 是「读出当前值再加偏移」，不是「在默认值上重算」。放在 `mode="reset"` 里的话，一个回合结束再加一次，一个环境训练中重置几百次，质量就累加几百份——晚期整批机器人重到站不起来，**表现正是存活率先升后降**。`startup` 只在建场景时抽一次，但抽样是按环境独立的，4096 个环境依然构成一个质量分布。
- **推力用速度脉冲**（`push_by_setting_velocity`）而不是持续外力：真机上人推一把、撞一下，都是在几十毫秒内改变动量，速度突变更接近这个物理过程，实现也更便宜。
- **轮子不参与 `reset_joints`**。轮子是连续关节，初始角度没有物理意义。
- **策略观测里没有质量/质心**，所以它只能学一个"对这一族参数都稳"的策略，这正是 sim2real 需要的。

如果加了 DR 之后训练曲线开始抖或跌，**先降 `push_robot` 的幅度**（±0.5 → ±0.3 m/s），它对回报的冲击最直接；其次是 `base_mass` 的上界。

## 5. 看板指标

| 指标 | 期望趋势 |
| --- | --- |
| `Episode_Reward/base_height` | 往 1 爬（高度到位） |
| `Episode_Reward/flat_orientation` | 往 0 收（越平越好，本就是负分） |
| `Episode_Reward/base_motion` | 往 0 收（不再漂移） |
| `Episode_Termination/base_height` / `bad_orientation` | 从 ~100% 往 0 降 |
| `Episode_Reward/track_wheel_vel` | 往 0.5 爬（轮子不再乱转） |
| `Loss/learning_rate` | 自适应，学不动时自动降 |

加了域随机化之后，`Episode_Reward/*` 的绝对值会整体下降，这是**正常的**——同样的策略在更难的初始条件下拿不到那么多分。看趋势而不是绝对值。

## 6. 速度指令任务（M20 第一版正式配置）

`Isaac-M20-Wheeled-Velocity-v0`，配置在 `tasks/m20_velocity_env_cfg.py`。它是**独立注册的第二个任务**：站立任务一字未改，两者各自训练、各自出 checkpoint。

**为什么必须是两个任务**：观测维度不同（57 vs 56），Actor 第一层的输入矩阵形状不同，两边的 `model_*.pt` 不能互换；日志也分开放（`logs/rsl_rl/m20_wheeled/` vs `logs/rsl_rl/m20_velocity/`）。

**旧站立模型也不能直接续训速度任务**——观测维度不同，连前 6 维的顺序都不同（站立把线速度放前面，速度任务把角速度放前面），所以连"截断 + 补齐"都救不了。它的能力改由课程的阶段 1（零指令站稳）重新建立：见 §6.6。

### 6.1 观测（56 维）

| 观测量 | 维度 | 说明 |
| --- | --- | --- |
| 机身角速度 | 3 | IMU，机体系 |
| 投影重力 | 3 | 姿态信息 |
| 机身线速度 | 3 | X、Y、Z，机体系 |
| 腿部关节位置偏差 | 12 | 相对默认位置（`mdp.joint_pos_rel`） |
| 腿部关节速度 | 12 | 关节运动状态 |
| 车轮角速度 | 4 | 四个轮子（`mdp.joint_vel_rel`） |
| 速度指令 | 3 | 前后、左右、偏航 |
| 上一时刻动作 | 16 | 动作连续性 |
| **总计** | **56** | Actor 输入 |

两条容易错的细节：

- **线速度和指令都在机体系**。训练、播放、部署三处必须同一个坐标系，混用世界系会让策略在某些朝向下突然失灵。
- **指令只取前 3 列**。`vel_command_b` 是 4 列（第 4 列是 heading 目标），`heading_command=False` 时它无意义；观测函数 `observations.velocity_command` 切前 3 列，否则拼接后是 57 维。

### 6.2 动作（16 维）

| 动作 | 维度 | 控制方式 |
| --- | --- | --- |
| FL/FR/RL/RR 腿部关节 | 4 × 3 | 位置目标 `q* = q_default + 0.25·a` |
| 四个车轮 | 4 | 速度目标 `ω* = 8.0·a` |
| **总计** | **16** | 连续动作 |

轮速上限 = `8.0 × 0.09 m = 0.72 m/s`，覆盖满幅前进指令 0.5 m/s 的 69%，留出跟踪误差和打滑余量。**指令范围和轮动作 scale 必须一起改**：只把 `lin_vel_x` 调到 1.0 而 scale 还是 8.0，策略会一直顶着 0.72 m/s 的饱和点白丢奖励。

腿部仍用 PD：`τ = K_p(q_des − q) − K_d q̇`。**车轮控制保持现有实现（速度目标接口）**，不与腿部一起改——同时动动作空间和控制接口，出问题无法归因。

### 6.3 奖励（8 项 + 终止惩罚）

σ 是误差尺度，kernel 统一是 `exp(-e²/σ²)`（注意分母是 σ²，不是 σ）。惩罚项函数返回正值，符号由权重携带。

| # | 奖励项 | 函数 | 权重 |
| --- | --- | --- | --- |
| 1 | 前后、左右速度跟踪 | `exp(-[(vx−vx*)²+(vy−vy*)²]/0.25²)` | **+2.0** |
| 2 | 偏航角速度跟踪 | `exp(-(ωz−ωz*)²/0.3²)` | **+1.0** |
| 3 | Roll、Pitch 稳定 | `exp(-(φ²+θ²)/0.2²)` | **+0.5** |
| 4 | 机身高度 | `exp(-(h−0.50)²/0.05²)` | **+0.5** |
| 5 | 轮地纵向滑移 | `mean_4((v_roll − v_x)/0.5)²` | **−0.1** |
| 6 | 动作变化率 | `Σ(aₜ − aₜ₋₁)²` | **−0.01** |
| 7 | 关节力矩 | `Σ τ² / 40` | **−0.005** |
| 8 | 横滚/俯仰角速度 | `Σ ω_xy²` | **−0.05** |
| — | 摔倒终止 | `1[非超时终止]` | **−200** |

```text
r = 2·r_v + 1·r_ω + 0.5·r_ori + 0.5·r_h
    − 0.1·p_slip − 0.01·p_action − 0.005·p_torque − 0.05·p_ang
```

三条**刻意不做**的事：

- **不惩罚车轮绝对转速**。前进和后退本来就要求轮子转，惩罚绝对转速是在给"按要求运动"收税（站立任务里 `track_wheel_vel` 可以这么做，因为那里指令恒为 0）。
- **不强制四个轮子始终接触地面**。四轮着地的约束会破坏 Trot 步态，接触机制沿用现有有效实现。
- **不惩罚横向滑移**。车轮不是全向轮，侧移时横向轮胎滑动是物理必然；惩罚它等于禁止侧移。`p_slip` 只算**纵向**：`v_roll = −0.09·θ̇` 与机身 `v_x` 的差。

`p_torque` 的 `/40` 归一：腿力矩量级 76 N·m，`(τ/40)²` 让单项落在 0–1 附近，否则 −0.005 的权重对 `τ²` 完全无感。`p_slip` 的 `/0.5` 同理：0.5 m/s 的滑移记 1.0。

**`wheel_sign` 符号必须按 URDF 推**。`hipx` 轴 `(-1,0,0)`，`hipy`/`knee`/`wheel` 轴 `(0,-1,0)`，关节 origin `rpy` 全 0。绕 `−y` 以 `θ̇>0` 转时轮底接触点速度为 `+θ̇R x̂`，无滑动要求机身速度 `−θ̇R x̂`，所以 `v_roll = −0.09·θ̇`（`rewards.py` 里 `wheel_sign=-1.0`）。若 `Episode_Reward/wheel_slip` 一直贴着 0 而机器人明明在直行，说明 USD 导入翻了关节速度符号，把 `wheel_sign` 改成 `+1.0`。

### 6.4 指令（混合采样）

`MixedVelocityCommand`（`tasks/commands.py`），机体系 `(vx, vy, ωz)`，`heading_command=False`（直接给角速度，不让环境反算 heading；观测/奖励/部署三处口径一致）。

**每 2–4 s 重采样一次**。重采样不是三个轴独立均匀抽，而是按类型混合：

| 指令类型 | 采样比例 |
| --- | --- |
| 单方向运动 | 50% |
| 两个方向组合 | 30% |
| 三个方向组合 | 10% |
| 零速度指令 | 10% |

实现要点：**可用轴由 range 宽度决定**——某个轴的 `(min, max)` 退化成 `(0, 0)` 时它不参与采样，权重按 `[0.5, 0.3, 0.1][:可用轴数]` 重新归一化。所以阶段 2（只有 `vx`）自动退化成"50/30/10 全落在 `vx` 上 + 10% 零指令"，阶段 1（三个轴全零）则 100% 是零指令，不需要为每个阶段写一套采样逻辑。

零指令占 10% 是刻意的：没有它，策略从没学过"站着别动"，而真机上松手就是要停。10% 也不会让它退化成"躺平最省事"——零指令只值 10% 的样本，而跟踪奖励在非零指令上仍是主项。

### 6.5 PPO 超参（速度指令任务）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| Learning Rate | 3e-4 | `schedule="adaptive"` |
| Gamma | 0.99 | 折扣因子 |
| GAE Lambda | 0.95 | GAE 参数 |
| Clip Parameter | 0.2 | PPO 裁剪范围 |
| Entropy Coef | 0.005 | 熵系数 |
| Value Loss Coef | 1.0 | 价值损失系数 |
| Max Grad Norm | 1.0 | 梯度裁剪 |
| Desired KL | 0.01 | KL 目标 |
| Learning Epochs | 5 | 每轮更新遍数 |
| Mini Batches | 8 | 1024 环境时 |
| Clipped Value Loss | True | 价值裁剪 |
| Init Action Std | 0.9 | 初始动作标准差 |
| Actor / Critic Hidden | [256, 256] | 两层 MLP |
| Activation | ELU | 统一 |

| 采样参数 | 值 |
| --- | --- |
| `num_envs` | 512，稳定后 1024 |
| `num_steps_per_env` | 24 |
| `episode_length_s` | 20 |
| `policy_dt` / `physics_dt` / `decimation` | 0.02 s / 0.005 s / 4 |
| `save_interval` | 50 |
| `max_iterations` | 3950（5 阶段合计） |

样本量：`1024 × 24 = 24,576` 条/轮；3950 轮 ≈ 9700 万条环境转移。512 环境时约为一半（4850 万）。注意 RSL-RL 每轮还对这批 rollout 做 5 遍 × 8 个 minibatch 的更新，所以不能把 9700 万理解成"网络只训练了 9700 万次"。

环境数量按显存选（Isaac Sim 本身要吃掉一部分）：

| 显存 | 起步 | 可试 |
| --- | --- | --- |
| 16 GB | 128 | 256 |
| 24 GB | 512 | 1024 |
| 48 GB | 1024 | 2048–4096 |
| 80 GB | 2048 | 4096+ |

训练期间**只跑一个 Isaac Lab 训练进程**：PhysX GPU 缓冲、显存余量和 PPO 更新峰值是互相挤的，两个进程同时抢卡的表现是帧率崩到个位数，而不是 OOM——很难定位。

### 6.6 课程（5 阶段，阶段间续训）

`scripts/train.py` 的 `STAGES`。**每个阶段都从上一阶段的最佳模型继续训练，不重新初始化**。顺序是"先站稳 → 前后 → 左右 → 偏航 → 全部混合"，每个轴先单独学会，再与其它轴叠加：

| 阶段 | 名称 | 轮数 | 起始 → 结束指令范围 | warmup |
| --- | --- | --- | --- | --- |
| 1 | `1_stand` | 300 | `vx=0`，`vy=0`，`ωz=0`（零指令） | 0 |
| 2 | `2_forward_backward` | 700 | `vx: 0 → ±0.4`，`vy=0`，`ωz=0` | 150 |
| 3 | `3_lateral` | 1000 | `vx ±0.4`，`vy: 0 → ±0.2`，`ωz=0` | 250 |
| 4 | `4_yaw` | 700 | `vx ±0.4`，`vy ±0.2`，`ωz: 0 → ±0.6` | 250 |
| 5 | `5_mixed` | 1250 | `vx: ±0.4 → ±0.5`，`vy ±0.2`，`ωz: ±0.6 → ±0.8` | 250 |

合计 3950 轮（`max_iterations` 口径，不是 PPO 的 `num_learning_epochs`）。

**阶段 1 是零指令阶段**，它的作用是把后面所有阶段的基线定下来：站住高度、机身水平、轮子不空转、车身不漂移。零指令下跟踪项的含义就是"别动"，所以它不需要额外的奖励项。单独成一阶段（而不是塞进阶段 2 的 warmup）的理由是它能出一个自己的 checkpoint——如果阶段 2 之后的跟踪分还不如阶段 1 的"站着"，问题在指令和动作尺度上，不在基线。

**旧站立模型的 57 维观测搬不过来**（见下文），所以这条课程必须自己重新建立站立基线，这一步就是阶段 1。

**warmup 是阶段内的线性 lerp**：每个阶段的指令范围都从"上一阶段的范围"出发，用前 `warmup` 轮线性长到本阶段的目标范围，之后固定。理由有两条：

1. 阶段 2 是第一次给非零指令，`vx` 直接给 ±0.4 会让策略一开局就满幅乱冲。从 `0` 起步，前 150 轮只是在阶段 1 的基线之上慢慢长出指令。
2. 后三个阶段刚换 checkpoint 时策略对新维度（`vy` 或 `ωz`）完全没概念，瞬间换满幅等于把上一阶段的能力打断。lerp 让新维度从 0 长出来，旧维度不受冲击。

**旧站立模型怎么用**：它的观测是 57 维，速度指令任务是 56 维，**不能 `runner.load` 续训**。做法是把它的能力当作阶段 1（零指令站稳）要重新达到的目标——那个阶段本身就是"站稳 + 轮子不转"，策略会在几百轮内重新学到；不需要、也没法把权重搬过来。若要搬运，只能在 `assets`/`observations` 层面把两边对齐成同一维度，那是另一件事，第一版不做。

**阶段间续训流程**（`main()` 里）：

```text
for stage in stages:
    runner = OnPolicyRunner(env, agent_cfg, log_dir=<run>/stage{N}_{name})
    ckpt = 上一阶段目录里最新的 model_*.pt   # 单阶段运行时是 STAGES 里前一阶段目录
    if ckpt: runner.load(ckpt)
    schedule.start_stage(stage)              # 设指令范围到 ranges_from
    runner.learn(stage.iterations, init_at_random_ep_len=True)
```

**只建一个 env，跨阶段复用**：`gym.make` 一次，各阶段重新 `OnPolicyRunner` 但共享场景。重建 env 会重新抽 startup 随机化、重置 PhysX 缓冲，既慢又让阶段间对比失去可比性。

`--stage N` 单独跑某一阶段时会自动去前一阶段目录找 checkpoint，找不到就打 `[WARN]` 从头开始——不会静默地用一个错维度的权重。不给 `--run_name` 时它会自动沿用 `<experiment>/` 下最新的 run 目录（否则每次 `--stage N` 都会新建一个空目录、每次都找不到上一阶段）。

**每阶段的轮数是参数**：`--iters_stage1 … --iters_stage5` 分别覆盖对应阶段的预算，`--iterations` 是"给所有选中阶段同一个值"的简写，`--stage_scale` 才是按比例整体缩放。优先级：`--iters_stageN` > `--iterations` > 默认值 × `--stage_scale`。warmup 始终被裁到不超过该阶段预算的一半，所以缩到几次迭代也不会把整段吃光。

**每阶段跑完，脚本会打印下一阶段的建议命令**（`play --checkpoint …` / `eval --checkpoint …` / `train --stage N+1`），这就是"跑完先进模拟器看，再决定是否训练下一步"的工作流。

**每阶段该跑多少轮，要按 §6.8 的验收指标决定，不是机械跑满。** 表里给的是上限。

### 6.7 域随机化：第一版关闭

`m20_velocity_env_cfg.set_domain_randomization(env_cfg, enabled)`，默认 `enabled=False`（`--dr off`）：

| 项 | 关闭（默认） | `--dr full` |
| --- | --- | --- |
| 摩擦 | 1.0 / 1.0 | 0.5–1.25 |
| 机身质量 | +0 kg | +(-2, +3) kg |
| 质心 | 0 | x/y ±0.03，z ±0.02 |
| 推力 | 无（interval 1e9） | 每 4–8 s ±0.5 m/s |

**必须在 `gym.make` 之前调用**：`startup` 事件在建场景时被深拷贝进 manager，之后再改 cfg 不生效。

顺序是刻意的：**先固定地形和动力学参数，把速度跟踪训好，再加入摩擦、质量、地形随机化**。第一版就开满 DR，速度跟踪和鲁棒性会互相干扰，出问题时无法归因是"策略不会走"还是"扰动太大"。重置时的小扰动（姿态、关节、轮角）仍然保留——它是每个回合必须做的，不算随机化。

### 6.8 验收（`scripts/eval.py`）

**不要只看平均 Episode Reward**。用固定指令序列测，且**不在训练环境里评价**（eval 脚本强制关掉 DR、关掉指令重采样）：

```bash
python -m isaaclab_m20.scripts.eval \
    --checkpoint logs/rsl_rl/m20_velocity/<run>/stage5_mixed/model_1250.pt
```

| 测试 | 指令 `(vx, vy, ωz)` | 运动 |
| --- | --- | --- |
| 1 | `(0.30, 0, 0)` | 前进 |
| 2 | `(-0.30, 0, 0)` | 后退 |
| 3 | `(0, 0.20, 0)` | 左移 |
| 4 | `(0, -0.20, 0)` | 右移 |
| 5 | `(0, 0, 0.50)` | 左转 |
| 6 | `(0, 0, -0.50)` | 右转 |
| 7 | `(0.30, 0, 0.30)` | 前进并左转 |
| 8 | `(-0.30, 0.20, -0.30)` | 后退、左移并右转 |
| 9 | `(0, 0, 0)` | 保持静止 |

| 指标 | 验收标准 |
| --- | --- |
| 前进 / 后退速度 RMSE | < 0.10 m/s |
| 左右速度 RMSE | < 0.12 m/s |
| 偏航速度 RMSE | < 0.15 rad/s |
| Roll RMS | < 5° |
| Pitch RMS | < 5° |
| 平地摔倒率 | < 2%（= 20 次/1000 策略步） |
| 零指令时机身速度 | < 0.05 m/s，`|ωz|` < 0.10 rad/s |

这些是**工程验收目标，不是 PPO 必然能达到的结果**。单项不过关时按下面顺序查：

1. 摔倒率超标 → 先看是不是 Stage 4 的 `ωz ±0.8` 配 `vx ±0.5` 太激进，回到 Stage 3 的验收再前进。
2. 速度 RMSE 大但方向对 → 轮动作 scale 是否够（`8.0 × 0.09 = 0.72 m/s`），以及 `p_slip` 权重是否把轮子压得太死。
3. 侧移差 → 检查 `vy` 的 exp 核 σ=0.25 是否对 0.2 m/s 的指令太大（σ 不随指令收缩，小指令下梯度偏平，必要时单独给 vy 一个 σ）。
4. 零指令漂移 → 10% 零指令占比是否真的在生效（看 `Episode_Reward` 里零指令段的 track 项）。

### 6.9 训练与播放

```bash
# 全课程一次跑完（不推荐首版）：512 环境起步，稳定后改 1024
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --run_name m20_v1

# 推荐：一次只跑一个阶段，跑完 play/eval 看完再决定下一步
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --stage 1 --run_name m20_v1
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --stage 2 --run_name m20_v1

# 每个阶段的轮数都是参数（--iters_stage1 … --iters_stage5）
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --stage 2 \
    --run_name m20_v1 --iters_stage2 1200

# 一次指定全部阶段的长度
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --stage all \
    --iters_stage1 200 --iters_stage2 600 --iters_stage3 900 \
    --iters_stage4 600 --iters_stage5 1000

# 冒烟测试：第 1 阶段 20 轮
python -m isaaclab_m20.scripts.train --headless --num_envs 128 --stage 1 --iters_stage1 20

# 播放：W/S 改 vx，A/D 改 vy，Q/E 改 ωz，空格清零，P 推一把，R 重置
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 --checkpoint <path>

# 固定指令 + 无窗口出视频
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --cmd_vx 0.3 --cmd_wz 0.3 --headless --video

# 验收
python -m isaaclab_m20.scripts.eval --checkpoint <path>

# 曲线
tensorboard --logdir logs/rsl_rl/m20_velocity
```

`--stage N` 时如果没给 `--run_name`，脚本会自动沿用 `<experiment>/` 下最新的那个 run 目录，所以"先 `--stage 1`、看完再 `--stage 2`"不需要每次手抄 run 名。

日志布局：

```text
logs/rsl_rl/m20_velocity/<run_name>/
├── stage1_stand/{env.yaml, agent.yaml, model_*.pt}
├── stage2_forward_backward/...
├── stage3_lateral/...
├── stage4_yaw/...
└── stage5_mixed/...
```

### 6.10 看这些指标

| 指标 | 期望趋势 |
| --- | --- |
| `Episode_Reward/track_lin_vel_xy` | 往 2.0 爬。**长期 < 0.5 就是速度跟踪没学会** |
| `Episode_Reward/track_ang_vel_z` | 往 1.0 爬（阶段 4 之后才有意义，前 3 阶段 ωz 指令恒为 0） |
| `Episode_Reward/flat_orientation` | 往 0.5 爬（越平越接近 1） |
| `Episode_Reward/base_height` | 往 0.5 爬 |
| `Episode_Reward/wheel_slip` | 小幅负值。**恒为 0 说明 `wheel_sign` 反了** |
| `Episode_Reward/joint_torques` | 小幅负值 |
| `Episode_Termination/base_height` / `bad_orientation` | 从高往低降，阶段 2 末应 < 10% |
| `Loss/learning_rate` | 自适应，学不动时自动降 |

终端里每 50 轮打印 `[cmd] iteration=... stage=... ranges=...`（当前阶段与当前指令范围），退火期间能看到范围在长。**换阶段那一刻跟踪分会掉一下**（新维度从 0 长出来），看它能不能在 warmup 内爬回来；如果爬不回来，说明上一阶段的 checkpoint 没被正确加载（检查 `[INFO] Resumed from ...` 那行）。

**训练总量参考**：`512 × 24 × 3950 ≈ 4850 万` 条转移（1024 环境翻倍）。在 4090 上，1024 环境约 20–30 min / 1000 轮量级，整条课程通常在 2–4 小时；512 环境更快但样本效率略低。
