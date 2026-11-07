# M20 轮足机器人（Isaac Lab + rsl_rl）

用 Isaac Lab 的 ManagerBasedRLEnv 训练 M20 轮足机器人：PPO 输出关节位置目标，PD 伺服环在 200 Hz 跟踪成力矩，后端是 rsl_rl，GPU 上可并行 4096 个环境。

两个**互相独立**的任务，各自训练、各自的 checkpoint：

| gym id | 任务 | 观测 | 回合 |
| --- | --- | --- | --- |
| `Isaac-M20-Wheeled-v0` | 静态站立 | 57 维（无指令） | 10 s |
| `Isaac-M20-Wheeled-Velocity-v0` | `(vx, vy, ωz)` 速度指令跟踪 | 60 维（3 维指令 + 57） | 20 s |

两者的 checkpoint **不能互换**（观测维度不同），日志分别在 `logs/rsl_rl/m20_wheeled/` 和 `logs/rsl_rl/m20_velocity/`。

详细安装与排错见 [docs/isaaclab_setup.md](docs/isaaclab_setup.md)，奖励函数/网络结构/PPO 更新流程见 [docs/algorithm.md](docs/algorithm.md)。

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
├── tasks/             # 场景/观测/动作/指令/奖励 + gym 注册
└── scripts/
    ├── check_m20.py   # 加载 USD，打印关节/质量，验证资产
    ├── train.py
    └── play.py
```

## 快速开始

仓库根目录执行（脚本是模块入口，不能 `cd` 进 `scripts/`）：

```bash
# 1. 从官方 URDF 导入一次训练 USD，然后验证（16 个关节、总质量 ≈38.5 kg）
python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf

# 2. 训练（静态站立）
python -m isaaclab_m20.scripts.train --headless --num_envs 4096

# 3. 播放（加载最新 checkpoint，16 个环境）。视口里按 P 推一下机器人，R 重置
python -m isaaclab_m20.scripts.play --num_envs 16

# 4. 无界面录视频，每 3 秒自动推一次
python -m isaaclab_m20.scripts.play --num_envs 16 --push_every 3 --headless --video

# 5. 速度指令任务：训练
python -m isaaclab_m20.scripts.train --headless --num_envs 4096 \
    --task Isaac-M20-Wheeled-Velocity-v0 --run_name vel_v1

# 6. 速度指令任务：播放。键盘 W/S = vx，A/D = vy，Q/E = 偏航角速度，空格清零
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --checkpoint logs/rsl_rl/m20_velocity/vel_v1/checkpoints/model_3000.pt

# 7. 或固定指令、无界面出视频
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --checkpoint <path> --cmd_vx 0.6 --cmd_wz 0.3 --headless --video

# 8. 曲线
tensorboard --logdir logs/rsl_rl --bind_all --port 6006
```

## 控制与观测

- 物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`）。
- **静态站立任务**：没有速度指令，目标就是原地站稳、保持高度、机身水平、不漂移，轮速压零。回合 10 s。
- **速度指令任务**：指令 `(vx, vy, ωz)`，满幅 ±1.0 m/s、±0.4 m/s、±1.0 rad/s，每 4 s 重采样一次；回合 20 s（含 5 段指令）。奖励里没有轮速归零项，腿姿态惩罚只在指令接近零时生效。指令范围本身也走课程（`--cmd_ramp`，默认 0.5）：从零开始爬，否则那条门控几乎永远不触发。
- 观测：机身线速度 3 + 角速度 3 + 重力投影 3 + 关节角 16 + 关节速度 16 + 上一拍动作 16 = 57 维；速度指令任务在最前面再加 3 维指令 = 60 维。
- 动作 16 维 ∈ [-1, 1]，均为**目标值**：腿部 12 维为位置目标 `q* = q_default + a × 0.25 rad`，轮部 4 维为速度目标 `ω* = a × 20 rad/s`。
- PD 伺服环（`ImplicitActuatorCfg`，200 Hz）：腿 kp=80 / kd=2，轮 kp=0 / kd=2，力矩上限 76.4 / 21.6 N·m。
- 奖励：跟踪项、机身水平、高度（0.50 m）、腿姿态、省力矩、动作平滑、存活、摔倒重罚。
- 终止：高度 < 0.25 m、倾角 > 0.7 rad、超时。

## 调参入口

| 想改的东西 | 文件 |
| --- | --- |
| 奖励权重、指令范围、终止条件（站立） | `isaaclab_m20/tasks/m20_env_cfg.py` |
| 奖励权重、指令范围（速度指令） | `isaaclab_m20/tasks/m20_velocity_env_cfg.py` |
| 奖励函数 | `isaaclab_m20/tasks/rewards.py` |
| PPO 超参 | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
