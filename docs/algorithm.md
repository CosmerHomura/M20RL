# 训练算法设计

本文以**静态站立任务**（`Isaac-M20-Wheeled-v0`）为主线说明奖励函数、策略/价值网络结构，以及 PPO 一整个迭代里「采集 → 计算优势 → 更新网络」的完整流程；**速度指令任务**（`Isaac-M20-Wheeled-Velocity-v0`）的差异集中在 [§6](#6-速度指令任务第二个任务)。配置入口见 `isaaclab_m20/tasks/`：

本文说明当前静态站立任务的奖励函数、策略/价值网络结构，以及 PPO 一整个迭代里「采集 → 计算优势 → 更新网络」的完整流程。配置入口见 `isaaclab_m20/tasks/`：

- 奖励权重与观测/动作：`tasks/m20_env_cfg.py`（站立）、`tasks/m20_velocity_env_cfg.py`（速度指令）
- 奖励公式：`tasks/rewards.py`
- 网络与 PPO 超参：`tasks/agents/rsl_rl_ppo_cfg.py`

## 1. 奖励函数

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

```text
观测 o ∈ R^57
  │
  ├─ Actor  (策略)    57 → 512 → 256 → 128 → μ∈R^16
  │                     ELU    ELU    ELU    (对角高斯, init σ=1.0)
  │
  └─ Critic (价值)    57 → 512 → 256 → 128 → V∈R^1
                        ELU    ELU    ELU
```

- **观测 57 维**：`base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(16) + joint_vel(16) + last_action(16)`。
- **动作 16 维** ∈ [-1,1]，都是**目标值**，不是力矩：
  - 腿 12 维：`q* = q_default + a × 0.25 rad`（`JointPositionActionCfg`）
  - 轮 4 维：`ω* = a × 20 rad/s`（`JointVelocityActionCfg`）
- **PD 伺服环**（`ImplicitActuatorCfg`）在 200 Hz 物理步上把目标跟踪成力矩：腿 kp=80、kd=2，轮 kp=0、kd=2（速度伺服），力矩上限 76.4 / 21.6 N·m。策略负责决策，PD 只负责跟踪。
- **激活** ELU，三层 MLP [512, 256, 128]；**无观测归一化**（`actor_obs_normalization=False`）。
- **策略头**：输出均值，方差固定为 `init_noise_std=1.0` 的对角高斯（不学方差），采样得到动作。

关键超参（`RslRlPpoAlgorithmCfg`）：γ=0.99、λ(GAE)=0.95、clip=0.2、熵系数 0.01、价值损失系数 1.0、`num_learning_epochs=5`、`num_mini_batches=4`、lr=1e-3（`schedule="adaptive"`，KL 超 0.01 就降 lr）、`max_grad_norm=1.0`。

## 3. 运行与更新流程

**时间尺度**：物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`，每 4 个物理步出一次动作）。回合 10 s = 200 个策略步。

**一次 PPO 迭代**：

1. **采集**：4096 个并行环境各推 `num_steps_per_env=24` 步，得到 4096 × 24 = **98,304** 条 (oₜ, aₜ, rₜ, oₜ₊₁, done) 样本。
2. **算优势**：Critic 给出每步 `V(o)`，用 GAE（γ=0.99、λ=0.95）沿时间反向算优势 `Âₜ`，回报 `Rₜ = Âₜ + V(oₜ)` 作为价值回归目标。
3. **多轮更新**：把 98,304 条样本打乱分成 4 个 minibatch，循环 `num_learning_epochs=5` 遍，每遍对 Actor 和 Critic 各做一次梯度下降：
   - **Critic**：`loss_V = 0.5·(V − R)²`，权重 1.0；
   - **Actor**：PPO-clip 目标 `min(rₜ·Â, clip(rₜ, 1±0.2)·Â) + 0.01·熵`，`rₜ = π(a|o)/π_old(a|o)`；
   - 梯度范数裁剪到 1.0。
4. **自适应学习率**：每遍后若新旧策略 KL 超过 `desired_kl=0.01`，lr 自动减半式下调。
5. 清空 buffer，用更新后的策略进入下一轮采集。

**训练总量**：`max_iterations=3000` 轮 × 98,304 步 ≈ 3 亿步。每 `save_interval=100` 轮存一个 `checkpoints/model_*.pt`。

**为什么直驱力矩开局会大面积趴下**：没有 PD 兜底，随机初始化的策略输出的是乱力矩，机器人生成就倒。这属正常——`base_height`/`flat_orientation` 的信号会把策略逐步拉向"撑住"的力矩组合，通常几百轮内趴下率开始下降。

## 4. 域随机化

在 `tasks/m20_env_cfg.py` 的 `EventCfg` 里。`startup` 类在场景建立时抽一次，`reset` 类在**每个环境每次重置时重新抽**——策略面对的不是一台机器人，而是一族机器人。

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

上表写的是**满幅范围**。实际生效的是 `满幅 × level`，`level ∈ [0, 1]`（`tasks/dr.py`）：

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

## 6. 速度指令任务（第二个任务）

`Isaac-M20-Wheeled-Velocity-v0`，配置在 `tasks/m20_velocity_env_cfg.py`。它是**独立注册的第二个任务**，站立任务（`Isaac-M20-Wheeled-v0`）一字未改，两者各自训练、各自出 checkpoint。

### 为什么必须是两个任务

观测维度不同：速度指令要把 3 维指令拼进观测，57 → 60 维。Actor 第一层是 `57×512` 的矩阵，加载 60 维输入的 checkpoint 直接 shape mismatch；反之亦然。**两边的 `model_*.pt` 不能互换**，日志也分开放（`logs/rsl_rl/m20_wheeled/` vs `logs/rsl_rl/m20_velocity/`）。

### 指令

`mdp.UniformVelocityCommandCfg`，机体系 `(vx, vy, ωz)`：

| 维度 | 满幅范围 | 说明 |
| --- | --- | --- |
| `lin_vel_x` | ±3.0 m/s | 前后。上限由动作尺度决定，见下 |
| `lin_vel_y` | ±0.6 m/s | 横移。髋只有外展 + 俯仰两个自由度，没有髋偏航，所以横向**不能纯靠轮子滚**，得靠腿侧向蹬/抬；范围给窄一点 |
| `ang_vel_z` | ±1.0 rad/s | 偏航角速度（约 57 °/s） |
| 重采样 | 每 4 s | `heading_command=False`，直接给角速度而不是让环境算 heading 误差 |

上表是**满幅**，只有最后一档才会用到；实际生效的是下面指令课程里的分档值。

**速度上限由动作尺度决定，不是由电机决定。** 轮动作是 `ω* = a × scale`，`a` 被 `clip_actions=1.0` 夹在 ±1，所以策略能要求的最大轮速就是 `scale`。轮半径 0.09 m（URDF 里 `<cylinder radius="0.09"/>`）：

```text
地面速度上限 = scale × 0.09 m
站立任务 scale=20 → 1.8 m/s     速度指令任务 scale=45 → 4.05 m/s
```

指令上限取到上限的 ~74%，留出跟踪误差、低摩擦地面打滑和推力的余量。**关键是二者必须一起改**：只把指令调到 3 m/s 而 wheel scale 还是 20，策略会一直顶在 1.8 m/s 的饱和点上，每一步都拿不到跟踪分，而它本来也不可能做到——这是纯粹的白丢奖励。

电机本身（`velocity_limit_sim=200 rad/s`，约 18 m/s）远不是瓶颈，瓶颈是动作缩放。

`heading_command=False` 是刻意的：heading 模式下指令是"朝向某个绝对角度"，环境内部再转成 ωz 误差，观测里那 3 维的含义就不再是纯粹的角速度了。直接给 ωz，观测、奖励、部署三处口径一致。

### 奖励差异

其余项（`lin_vel_z` / `ang_vel_xy` / `flat_orientation` / `base_height` / `joint_acc` / `joint_torques` / `action_rate` / `termination_penalty` / `alive`）与站立任务完全相同，差异如下：

| 分项 | 站立任务 | 速度指令任务 |
| --- | --- | --- |
| `track_lin_vel_x` / `_y` | `exp(-‖v_xy‖²/0.25)` | **拆成两个通道**，σ 随指令大小收缩 |
| `track_ang_vel_z` | `exp(-ω_z²/0.25)` | `exp(-(ω_z − cmd_ωz)²/0.25)` |
| `track_wheel_vel` | +0.5，轮子不许转 | **删除**；换成 `wheel_roll` +0.5，轮速必须与机身前进速度一致 |
| `joint_deviation` | 一直生效 | 一直生效（**没有门控**） |
| `joint_vel` | 全部 16 个关节 | **只算 12 个腿关节** |
| 轮动作 scale | 20 rad/s（1.8 m/s） | 45 rad/s（4.05 m/s） |

### 速度跟踪为什么拆成两个通道

`exp(-e²/σ)` 的 σ 是**误差尺度**：误差到几个 σ 后奖励就归零。所以 σ 必须和"这个通道上有意义的误差"成比例，而 x / y 两个轴的范围相差 5 倍（±3.0 vs ±0.6 m/s），一个 σ 服务不了两边：

```text
命令 (vx=0, vy=0.6)，机器人以 0.5 m/s 向前滑移：
  合并 2-范数 + σ=0.5   → exp(-0.25/0.5)  = 0.61   只损失 39%
  完全不横移（误差 0.6） → exp(-0.36/0.5)  = 0.49   还能拿一半分
```

第二条更要命：**横移通道几乎没有梯度**。σ=0.5 是为 ±3.0 的 x 通道配的，套到 ±0.6 上就是废的。

**而且 σ 还要随指令大小收缩。** 固定 σ 无法同时满足两端：0.0625 让 3 m/s 跟踪完全没有梯度，0.5 让"保持 vx=0"这件事免费。所以 σ 跟着 |cmd| 线性从 `sigma_min` 长到 `sigma_max`：

| 通道 | 权重 | σ_min（指令≈0） | σ_max（满幅） | fade |
| --- | --- | --- | --- | --- |
| x | 1.5 | 0.0625 | 0.5 | 1.5 m/s |
| y | 1.0 | 0.02 | 0.1 | 0.3 m/s |

效果：命令 `vx=0` 时 0.25 m/s 的滑移只值 0.37；命令 `vx=3` 时 1 m/s 的误差还值 0.135，梯度仍在。

**这里有一个物理原因，不只是奖励问题。** 轮子在 x 方向是自由滚动的，对前向几乎没有阻力。所以"向左操控还往前滑"不是摩擦不够，是**没有任何一项在管**。要压住它只能靠主动控制轮速，而这就要求奖励在那个方向上足够陡。

**`joint_vel` 必须把轮子排除**。轮速 40 rad/s 时四个轮子贡献 `4 × 1600 × 1e-4 = 0.64`，是跟踪奖励上限的 40%，等于对"按要求前进"这件事额外收 40% 的税。轮的平滑由 `action_rate` 和 `joint_acc` 管就够了。

### 姿态惩罚：不门控

legged_gym 的 `stand_still` 是"整个指令向量接近零"才罚姿态，前提是**足式机器人一动就必须抬腿**。M20 不满足这个前提：`vx` 靠轮子滚，腿不需要做任何事。

但**"横移需要腿"不等于"横移时腿可以随便摆"**。曾经试过把门控放在 `(vy, ωz)` 上（`vx` 不参与、横移/转向时放开），结果是横移那一段完全没有姿态约束——**后腿左右一前一后**就是这么长出来的，而横移恰恰是腿动得最多、最需要"每步回中"的时候。

所以现在是无条件生效。-0.5 这个权重数值上并不小：横移时 12 个关节的 L1 偏差 3~4 rad 就是 −1.5~−2.0，和整条跟踪通道同量级。之前显得"太小"，是因为门控把它在横移时整个关掉了。

**默认站姿左右对称，所以这一项同时也是对称项。** 后左 `hipy` 比后右超前 δ，L1 罚的是两个偏差之和；两个都回到默认值时该和为 0。注意它在"对称但整体偏移"的方向上是平的（都在 +0.3 和 一个 +0.3 一个 −0.3 代价相同），所以如果重训后不对称还在，下一步是加 6 组 `|q_left − q_right|` 配对项——但那一项会禁止小跑，只能在加地形时一起做。

`wheel_roll` 同理：删掉 `track_wheel_vel` 之后没有任何一项知道"轮子在空转、机身没动"。它把机身前进速度 `v_x` 和无滑动滚动速度 `v_roll = -0.09 · mean(ω_wheel)` 对齐（符号来源见下），空转和拖拽都拿不到分。

**符号必须按 URDF 推，不能拍脑袋。** `hipx` 轴是 `(-1,0,0)`，`hipy` / `knee` / `wheel` 轴都是 `(0,-1,0)`，且各关节 origin 的 `rpy` 全为 0。绕 y 的旋转不改变 y 方向，所以默认姿态下轮轴在机身系里就是 `(0,-1,0)`。绕 `-y` 以 `θ̇ > 0` 转时，轮底接触点速度为 `+θ̇R x̂`；无滑动要求接触点静止，于是机身速度是 `-θ̇R x̂`。所以是 `v_roll = -0.09·θ̇`；写成 `+` 就是反过来惩罚正确行为，策略会去学倒着滚。若 `Episode_Reward/wheel_roll` 一直贴着 0 而机器人明明在直行，说明 USD 导入把关节速度符号翻了一次，把 `wheel_sign` 改成 `+1.0` 即可。

`wheel_roll` 在零指令下也是 1.0（轮子不转、机身不动就是完美滚动），所以它在"站着"和"正常滚"之间是中性的，不构成"别动"或"必须动"的偏置。

### 指令课程（分档）

与域随机化课程并列的第二条调度。**不是线性爬坡，是分档跳变**（`COMMAND_STAGES`，`scripts/train.py`）：

| 阶段 | 结束于训练进度 | vx | vy | ωz |
| --- | --- | --- | --- | --- |
| 0 站立 | 5% | 0 | 0 | 0 |
| 1 低速 | 25% | ±0.6 | ±0.15 | ±0.3 |
| 2 中速 | 50% | ±1.5 | ±0.35 | ±0.6 |
| 3 高速 | 75% | ±2.2 | ±0.50 | ±0.8 |
| 4 满幅 | 100% | ±3.0 | ±0.60 | ±1.0 |

3000 轮下就是 0–150 / 150–750 / 750–1500 / 1500–2250 / 2250–3000。`--no-cmd_stages` 关掉课程，从第一步就用满幅。

为什么不用线性爬坡：线性爬坡把绝大部分轮次花在低速段，而低速段几乎没什么可学的；真正有信息量的 2–3 m/s 区间被推到训练末尾才出现。分档能在有限轮次里更快进入有用的速度区间。

阶段 0 刻意很短（150 轮）。它的作用不是从零训练一个站立策略（站立任务已经做过了），而是先把站姿、机身高度和姿态惩罚的基线定下来，再让指令去要求运动。

`UniformVelocityCommand` 每次重采样都读 `cfg.ranges`，所以直接改 term 的 config 就生效（与 events 不同，那里是深拷贝，见 §4）。另外训练启动时会把指令缓冲清零一次：建场景时抽的指令用的是满幅，而第一次重采样要等 4 s，不擦掉的话策略开局 4 s 就被要求跑 3 m/s。

分档的另一半理由是探索。从满幅冷启动，策略连"该往哪个方向走"都分不清，跟踪分长期贴着 0；分档是同一个任务族里更容易的那一支，梯度也更有方向。

**跳档时的奖励会短暂塌一下**（比如中速 1.5 跳到高速 2.2，误差 1.5 m/s 配 σ=0.5 时 `exp(-4.5)≈0.01`），几百轮内会爬回来。如果某一档迟迟不恢复，说明跨度太大，在表里插一档。

### 其余设置

- 回合 **20 s**（站立是 10 s）：一个回合里能放 5 段指令，而不是只有 1 段，同一个回合内的切换也是策略要面对的。
- 域随机化、推力、课程调度、PD 参数、动作缩放全部继承站立任务，没有第二套。
- `max_iterations=3000`、网络结构与超参也完全一致，只是 `experiment_name` 变成 `m20_velocity`。

### 训练与播放

```bash
# 训练（建议 4096 环境，4090 上占一半显存）。--no-cmd_stages 关掉分档课程
python -m isaaclab_m20.scripts.train --headless --num_envs 4096 \
    --task Isaac-M20-Wheeled-Velocity-v0 --run_name vel_v1

# 播放：键盘 W/S 改 vx，A/D 改 vy，Q/E 改 ωz，空格清零，P 推一把，R 重置
# --task 决定加载哪个任务的 checkpoint（自动找 m20_velocity/ 下最新）
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0

# 或无窗口跑固定指令出视频
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --cmd_vx 0.6 --cmd_wz 0.3 --headless --video
```

播放时脚本会把指令重采样时间改成无穷大，并在每个策略步覆写指令缓冲——否则训练里的 4 s 重采样会不停覆盖人工输入。

### 看这些指标

| 指标 | 期望趋势 |
| --- | --- |
| `Episode_Reward/track_lin_vel_x` | 往 1.5 爬 |
| `Episode_Reward/track_lin_vel_y` | 往 1.0 爬。**这一条长期贴着 0.2 以下，就是横移没学会** |
| `Episode_Reward/track_ang_vel_z` | 往 0.75 爬 |
| `Episode_Reward/track_wheel_vel` | **不存在**，看到这一项说明跑的是站立任务 |
| `Episode_Reward/wheel_roll` | 往 0.5 爬。**贴着 0 说明 `wheel_sign` 反了**（见上），不是学不会 |
| `Episode_Termination/base_height` / `bad_orientation` | 从高往低降 |
| `Episode_Reward/alive` | 上限 0.15 × 1000 步 = 150（20 s 回合） |
| `Episode_Reward/joint_deviation` | 一直有小幅负值。**恒为 0 说明跑的是旧配置**（门控版） |
| `Loss/learning_rate` | 自适应 |

终端里每 50 轮打印 `[dr] iteration=... level=...`（域随机化档位），每次换档打印 `[cmd] iteration=... stage=...`。**换档那一刻跟踪分会掉一下**，这是跳档的正常现象，看它能不能在几百轮内爬回来。

跟踪分数上不去最常见的原因是**指令范围太大**：3 m/s 加 1.0 rad/s 同时抽，一部分组合本来就接近侧翻边界。先把 `ang_vel_z` 收到 ±0.6、或在 `COMMAND_STAGES` 里把 3.0 m/s 那一档往后挪。
