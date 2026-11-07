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
| `lin_vel_x` | ±1.0 m/s | 前后 |
| `lin_vel_y` | ±0.4 m/s | 横移。髋只有外展 + 俯仰两个自由度，没有髋偏航，所以横向**不能纯靠轮子滚**，得靠腿侧向蹬/抬；范围给窄一点 |
| `ang_vel_z` | ±1.0 rad/s | 偏航角速度 |
| 重采样 | 每 4 s | `heading_command=False`，直接给角速度而不是让环境算 heading 误差 |

上表是**满幅**。实际生效的是 `满幅 × cmd_level`，见下面的指令课程。

`heading_command=False` 是刻意的：heading 模式下指令是"朝向某个绝对角度"，环境内部再转成 ωz 误差，观测里那 3 维的含义就不再是纯粹的角速度了。直接给 ωz，观测、奖励、部署三处口径一致。

### 奖励差异

其余项（`lin_vel_z` / `ang_vel_xy` / `flat_orientation` / `base_height` / `joint_vel` / `joint_acc` / `joint_torques` / `action_rate` / `termination_penalty` / `alive`）与站立任务完全相同，只有三处不同：

| 分项 | 站立任务 | 速度指令任务 |
| --- | --- | --- |
| `track_lin_vel_xy` | `exp(-‖v_xy‖²/0.25)` | `exp(-‖v_xy − cmd_xy‖²/0.25)` |
| `track_ang_vel_z` | `exp(-ω_z²/0.25)` | `exp(-(ω_z − cmd_ωz)²/0.25)` |
| `track_wheel_vel` | +0.5，轮子不许转 | **删除**。要机器人动，就不许罚它动 |
| `joint_deviation` | 一直生效 | 只在 `‖cmd‖ < 0.1` 时生效（`stand_still`） |

`stand_still` 那一条是 legged_gym 的做法：一直罚"偏离默认站姿"，等于和横移/转向需要的抬腿动作对着干，策略会学到"站着不动最划算"。门控之后，静止段保持姿态、运动段释放自由度。

**但门控只在指令课程下才成立。** 满幅均匀采样时，`‖cmd‖ < 0.1` 的命中率是

```text
半径 0.1 的球体积 4.19e-3 / 采样盒体积 (2 × 0.8 × 2 = 3.2) = 0.13%
```

也就是约 770 个环境步才命中一次——姿态惩罚在满幅采样下等于**从来没生效过**，配置里写的行为并不存在。所以指令范围必须从零开始爬（见下），早期所有环境都命中这条门控，站姿先练出来，之后才逐步放开。

### 指令课程

与域随机化课程并列的第二条调度（`--cmd_ramp`，默认 0.5）：

```text
cmd_level = clamp(iteration / (max_iterations × --cmd_ramp), 0, 1)
采样范围 = 满幅 × cmd_level
```

`cmd_level = 0` 时指令恒为 0，等价于站立任务；1500 轮后爬到满幅。`UniformVelocityCommand` 每次重采样都读 `cfg.ranges`，所以直接改 term 的 config 就生效（与 events 不同，那里是深拷贝，见 §4）。

另外训练启动时会把指令缓冲清零一次：建场景时抽的指令用的是满幅，而第一次重采样要等 4 s，不擦掉的话策略开局 4 s 就被要求跑 1 m/s。

这一步不只服务 `stand_still`。从满幅 `vy ±0.4` 冷启动，策略连"该往哪个方向走"都分不清，跟踪分长期贴着 0；从小到大是同一个任务族里更容易的那一支，梯度也更有方向。

### 其余设置

- 回合 **20 s**（站立是 10 s）：一个回合里能放 5 段指令，而不是只有 1 段，同一个回合内的切换也是策略要面对的。
- 域随机化、推力、课程调度、PD 参数、动作缩放全部继承站立任务，没有第二套。
- `max_iterations=3000`、网络结构与超参也完全一致，只是 `experiment_name` 变成 `m20_velocity`。

### 训练与播放

```bash
# 训练（建议 4096 环境，4090 上占一半显存）。--cmd_ramp 0 关掉指令课程
python -m isaaclab_m20.scripts.train --headless --num_envs 4096 \
    --task Isaac-M20-Wheeled-Velocity-v0 --run_name vel_v1

# 播放：键盘 W/S 改 vx，A/D 改 vy，Q/E 改 ωz，空格清零，P 推一把，R 重置
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --checkpoint logs/rsl_rl/m20_velocity/vel_v1/checkpoints/model_3000.pt

# 或无窗口跑固定指令出视频
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --checkpoint <path> --cmd_vx 0.6 --cmd_wz 0.3 --headless --video
```

播放时脚本会把指令重采样时间改成无穷大，并在每个策略步覆写指令缓冲——否则训练里的 4 s 重采样会不停覆盖人工输入。

### 看这些指标

| 指标 | 期望趋势 |
| --- | --- |
| `Episode_Reward/track_lin_vel_xy` | 往 1.5 爬；满幅约 0.8–1.1 就算收敛 |
| `Episode_Reward/track_ang_vel_z` | 往 0.75 爬 |
| `Episode_Reward/track_wheel_vel` | **不存在**，看到这一项说明跑的是站立任务 |
| `Episode_Termination/base_height` / `bad_orientation` | 从高往低降 |
| `Episode_Reward/alive` | 上限 0.15 × 1000 步 = 150（20 s 回合） |
| `Episode_Reward/joint_deviation` | 早期（`cmd_level` 小）是活跃的，后期趋近 0，符合预期 |
| `Loss/learning_rate` | 自适应 |

终端里 `[dr] iteration=... level=... cmd_level=...` 每 50 轮打印一次，`level` 是域随机化、`cmd_level` 是指令范围。**前 1500 轮 `cmd_level` 很小的时候跟踪分低是正常的**，此时指令本身就只有零点几 m/s。

跟踪分数上不去最常见的原因是**指令范围太大**：±1.0 m/s 加 ±1.0 rad/s 同时抽，一部分组合本来就接近侧翻边界。先把 `ang_vel_z` 收到 ±0.6、或把 `--cmd_ramp` 调到 0.7 再试。
