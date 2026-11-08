# M20 轮足机器人（Isaac Lab + rsl_rl）

用 Isaac Lab 的 ManagerBasedRLEnv 训练 M20 轮足机器人：PPO 输出 12 个腿部关节的位置目标 + 4 个车轮的速度目标，PD 伺服环在 200 Hz 跟踪成力矩，后端是 rsl_rl，GPU 上并行数百到数千个环境。

两个**互相独立**的任务，各自训练、各自的 checkpoint：

| gym id | 任务 | 观测 | 回合 | 日志 |
| --- | --- | --- | --- | --- |
| `Isaac-M20-Wheeled-v0` | 静态站立 | 57 维（无指令） | 10 s | `logs/rsl_rl/m20_wheeled/` |
| `Isaac-M20-Wheeled-Velocity-v0` | `(vx, vy, ωz)` 速度指令跟踪 | 56 维 | 20 s | `logs/rsl_rl/m20_velocity/<run>/stage{N}_*/` |

两者的 checkpoint **不能互换**（观测维度不同）。速度指令任务是第一版正式训练配置：5 阶段课程、8 项奖励、域随机化默认关闭、阶段间自动续训。

详细安装与排错见 [docs/isaaclab_setup.md](docs/isaaclab_setup.md)，奖励函数/网络结构/PPO 流程/课程与验收见 [docs/algorithm.md](docs/algorithm.md)。

## 环境要求

- Isaac Sim 5.1 + IsaacLab 2.3.2（Python 3.11，driver ≥ 580）
- rl 后端：`pip install "isaaclab[rsl-rl]" --extra-index-url https://pypi.nvidia.com`
- 首次启动前 `export OMNI_KIT_ACCEPT_EULA=YES`

## 目录

```text
model/
├── urdf/              # 官方 URDF（训练资产由它导入生成）
├── M20_urdf.usd       # 生成的训练 USD（不进 git）
├── usd/               # 官方 USD（组合层，需要完整的 configuration/ 子层）
└── M20.xml + meshes/  # 官方 MJCF（备用）
isaaclab_m20/
├── assets/m20.py      # ArticulationCfg：USD、初始姿态、PD 增益
├── tasks/
│   ├── m20_env_cfg.py            # 站立任务：场景/观测/动作/奖励/域随机化
│   ├── m20_velocity_env_cfg.py   # 速度指令任务：56 维观测、8 项奖励、5 阶段
│   ├── observations.py           # 56 维观测的拆分项（腿/轮/指令）
│   ├── commands.py               # MixedVelocityCommand：50/30/10/10 混合采样
│   ├── rewards.py                # 奖励函数（站立 14 项 + 速度 8 项）
│   ├── dr.py                     # 域随机化档位（站立任务用）
│   └── agents/rsl_rl_ppo_cfg.py  # PPO 超参（两个任务各一套）
└── scripts/
    ├── check_m20.py   # 加载 USD，打印关节/质量，验证资产
    ├── train.py       # 训练（速度任务含 5 阶段课程与续训）
    ├── play.py        # 播放 / 键盘给指令 / 录视频
    └── eval.py        # 固定指令验收测试（RMSE、倾角、摔倒率）
```

## 快速开始

仓库根目录执行（脚本是模块入口，不能 `cd` 进 `scripts/`）：

```bash
# 1. 从官方 URDF 导入一次训练 USD，然后验证（16 个关节、总质量 ≈38.5 kg）
python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf

# 2. 训练静态站立（4096 环境，模型很小）
python -m isaaclab_m20.scripts.train --headless --num_envs 4096

# 3. 播放站立策略（加载最新 checkpoint，16 个环境）。视口里 P 推一下，R 重置
python -m isaaclab_m20.scripts.play --num_envs 16

# 4. 训练速度指令任务：5 阶段课程，512 环境起步，阶段间自动续训
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --run_name m20_v1

# 5. 只跑某个阶段（自动从上一阶段最新 checkpoint 续）
python -m isaaclab_m20.scripts.train --headless --num_envs 1024 --stage 3 --run_name m20_v1

# 6. 冒烟测试：总轮数缩到 10%
python -m isaaclab_m20.scripts.train --headless --num_envs 128 --stage_scale 0.1 --run_name smoke

# 7. 播放速度指令策略。W/S = vx，A/D = vy，Q/E = 偏航角速度，空格清零
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0

# 8. 固定指令 + 无界面出视频
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --cmd_vx 0.3 --cmd_wz 0.3 --headless --video

# 9. 验收：9 组固定指令，输出 RMSE / 倾角 / 摔倒率 / PASS-FAIL
python -m isaaclab_m20.scripts.eval

# 10. 曲线
tensorboard --logdir logs/rsl_rl --bind_all --port 6006
```

## 控制与观测

**时间尺度**：物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`，4 个物理步出一次动作）。

**静态站立任务**：没有速度指令，目标是原地站稳、保持高度（0.50 m）、机身水平、不漂移，轮速压零。回合 10 s。

**速度指令任务**（`docs/algorithm.md` §6 是完整说明）：

- **观测 56 维**：`base_ang_vel(3) + projected_gravity(3) + base_lin_vel(3) + 腿关节位置偏差(12) + 腿关节速度(12) + 车轮角速度(4) + 指令(3) + last_action(16)`。线速度和指令都在**机体系**，训练/播放/部署必须同一个坐标系。
- **动作 16 维** ∈ [-1, 1]，都是**目标值**：腿 12 维 `q* = q_default + 0.25·a`（rad），轮 4 维 `ω* = 8.0·a`（rad/s，轮半径 0.09 m → 0.72 m/s 上限）。
- **指令**：机体系 `(vx, vy, ωz)`，每 2–4 s 重采样，混合采样 50% 单轴 / 30% 双轴 / 10% 三轴 / 10% 零指令。
- **奖励 8 项**：线速度跟踪 +2.0、偏航跟踪 +1.0、姿态 +0.5、高度 +0.5、纵向滑移 −0.1、动作变化率 −0.01、力矩 −0.005、横滚俯仰角速度 −0.05，摔倒终止 −200。**不惩罚车轮绝对转速、不强制四轮着地、不惩罚横向滑移。**
- **终止**：高度 < 0.28 m、倾角 > 0.8 rad、超时 20 s。
- **课程 5 阶段**（合计 3950 iterations，每阶段内有线性 warmup）：零指令站稳（300）→ 前进/后退 `vx ±0.4`（700）→ 加横移 `vy ±0.2`（1000）→ 加偏航 `ωz ±0.6`（700）→ 三维混合 `vx ±0.5 / vy ±0.2 / ωz ±0.8`（1250）。每个轴先单独学会，再与其它轴叠加。
- **域随机化默认关闭**（摩擦 1.0、质量/质心固定、无推力），`--dr full` 打开。

**PD 伺服环**（`ImplicitActuatorCfg`，200 Hz）：腿 kp=80 / kd=2，轮 kp=0 / kd=2，力矩上限 76.4 / 21.6 N·m。

## 显存与环境数

| 显存 | 速度指令任务起步 | 可试 |
| --- | --- | --- |
| 16 GB | 128 | 256 |
| 24 GB | 512 | 1024 |
| 48 GB | 1024 | 2048–4096 |
| 80 GB | 2048 | 4096+ |

**训练期间只跑一个 Isaac Lab 进程。** 两个进程抢同一张卡的表现是帧率崩到个位数，而不是干脆的 OOM。

## 调参入口

| 想改的东西 | 文件 |
| --- | --- |
| 奖励权重、终止条件、域随机化（站立） | `isaaclab_m20/tasks/m20_env_cfg.py` |
| 观测布局、奖励权重、动作尺度、终止、DR 开关（速度指令） | `isaaclab_m20/tasks/m20_velocity_env_cfg.py` |
| 奖励函数本身 | `isaaclab_m20/tasks/rewards.py` |
| 观测拆分项（腿/轮/指令） | `isaaclab_m20/tasks/observations.py` |
| 指令混合采样比例 | `isaaclab_m20/tasks/commands.py` |
| 课程阶段、轮数、warmup、续训逻辑 | `isaaclab_m20/scripts/train.py`（`STAGES`） |
| 验收阈值与测试指令 | `isaaclab_m20/scripts/eval.py` |
| PPO 超参 | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
