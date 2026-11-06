# 训练算法设计（静态站立）

本文说明当前静态站立任务的奖励函数、策略/价值网络结构，以及 PPO 一整个迭代里「采集 → 计算优势 → 更新网络」的完整流程。配置入口见 `isaaclab_m20/tasks/`：

- 奖励权重与观测/动作：`tasks/m20_env_cfg.py`
- 奖励公式：`tasks/rewards.py`
- 网络与 PPO 超参：`tasks/agents/rsl_rl_ppo_cfg.py`

## 1. 奖励函数

采用 legged_gym（Rudin et al., *Learning to Walk in Minutes*）与 IsaacLab `AnymalC` 任务的标准奖励集，把速度指令固定为零，即"站稳"。

**符号约定**：惩罚项的函数返回**正量纲值**，符号由权重携带——与 `isaaclab.envs.mdp` 一致（例如 `lin_vel_z_l2` 返回 `v_z²`，配置里给权重 −2.0）。所以下表「公式」列写的都是函数返回值，`rewards.py` 里不出现前导负号。函数返回负值 × 负权重 = 正奖励，会把惩罚项变成激励。

| 分项 | 权重 | 函数返回 | 含义 |
| --- | --- | --- | --- |
| `track_lin_vel_xy` | **+1.5** | `exp(-‖v_xy‖² / 0.25)` | 线速度跟踪核，指令为 0 → 越静止越接近 1 |
| `track_ang_vel_z` | **+0.75** | `exp(-ω_z² / 0.25)` | 偏航角速度跟踪核，指令为 0 |
| `lin_vel_z` | −2.0 | `v_z²` | 禁止上下颠簸 |
| `ang_vel_xy` | −0.05 | `Σ ω_xy²` | 禁止翻滚/俯仰角速度 |
| `flat_orientation` | −2.5 | `gx² + gy²` | 机身水平（`g` 为重力在机体系的投影） |
| `base_height` | **−20.0** | `(h − 0.50)²` | 基座高度二次惩罚（论文用的是二次型，不是 exp 奖励） |
| `joint_deviation` | −0.5 | `Σ abs(q − q_default)` | 关节偏离默认站立姿态（legged_gym 的 `stand_still`） |
| `joint_vel` | −1e-4 | `Σ q̇²` | 关节速度正则 |
| `joint_acc` | −2.5e-7 | `Σ q̈²` | 关节加速度正则，抑制高频抖动 |
| `joint_torques` | −1e-5 | `Σ τ²` | 省力矩 |
| `action_rate` | −0.01 | `Σ(aₜ − aₜ₋₁)²` | 动作平滑 |
| `termination_penalty` | **−200** | `1[非超时终止]` | 摔倒的重大惩罚 |
| `alive` | +0.15 | `1` | 存活奖励 |

**`termination_penalty` 是最关键的一项**。缺少它时，摔倒的代价接近零，而 `action_rate` / `joint_torques` 等正则项是每步累积的——于是"尽快摔倒结束回合"反而收益更高，策略会主动学怎么更快结束（表现为 episode length 持续下降）。论文里这一项通常是 −200 到 −400。

**`base_height` 必须是二次惩罚**：权重 −20 作用在 `(h−h*)²` 上。换成 `exp` 奖励形式时，远离目标高度的区域梯度几乎为零，等于没有推力。

**终止条件**（触发即本回合结束、重置）：

- `base_height`：基座高度 < 0.25 m（趴下）
- `bad_orientation`：倾角 > 0.7 rad（翻倒）
- `time_out`：满 10 s（50 Hz × 200 步），**超时不触发 `termination_penalty`**

## 2. 网络设计

rsl_rl 的 Actor-Critic，两个 MLP 共享同一份观测输入（`obs_groups` 里 actor 和 critic 都吃 `policy` 组）。

```
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

**训练总量**：`max_iterations=2000` 轮 × 98,304 步 ≈ 2 亿步。每 `save_interval=100` 轮存一个 `checkpoints/model_*.pt`。

**为什么直驱力矩开局会大面积趴下**：没有 PD 兜底，随机初始化的策略输出的是乱力矩，机器人生成就倒。这属正常——`base_height`/`flat_orientation` 的信号会把策略逐步拉向"撑住"的力矩组合，通常几百轮内趴下率开始下降。

## 4. 看板指标

| 指标 | 期望趋势 |
| --- | --- |
| `Episode_Reward/base_height` | 往 1 爬（高度到位） |
| `Episode_Reward/flat_orientation` | 往 0 收（越平越好，本就是负分） |
| `Episode_Reward/base_motion` | 往 0 收（不再漂移） |
| `Episode_Termination/base_height` / `bad_orientation` | 从 ~100% 往 0 降 |
| `Loss/learning_rate` | 自适应，学不动时自动降 |
