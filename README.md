# M20 轮足机器人（Isaac Lab + rsl_rl）

用 Isaac Lab 的 ManagerBasedRLEnv 训练 M20 轮足机器人：PPO 输出 12 个腿部关节的位置目标 + 4 个车轮的速度目标，PD 伺服环在 200 Hz 跟踪成力矩，后端是 rsl_rl，GPU 上并行数百到数千个环境。

两个**互相独立**的任务，各自训练、各自的 checkpoint：

| gym id | 任务 | 观测 | 回合 | 日志 |
| --- | --- | --- | --- | --- |
| `Isaac-M20-Wheeled-v0` | 静态站立 | 57 维（无指令） | 10 s | `logs/rsl_rl/m20_wheeled/` |
| `Isaac-M20-Wheeled-Velocity-v0` | `(vx, vy, ωz)` 速度指令跟踪 | 56 维 | 20 s | `logs/rsl_rl/m20_velocity/<run>/stage{N}_*/` |

两者的 checkpoint **不能互换**（观测维度不同）。速度指令任务是第一版正式训练配置：4 阶段课程（站稳 → 前后 → 全向 → 更快、含起步/急停切换）、10 项每步奖励 + 摔倒 −10（追踪用有界柯西核、零指令下只有轮速一项带门控、腿关节加速度抗抖、无存活加成、无位置保持、每步奖励有 0 下限）、域随机化默认关闭、阶段间自动续训。

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
├── envs.py            # M20VelocityEnv：速度任务的每步奖励下限（防自杀策略）
├── tasks/
│   ├── m20_env_cfg.py            # 站立任务：场景/观测/动作/奖励/域随机化
│   ├── m20_velocity_env_cfg.py   # 速度指令任务：56 维观测、10 项奖励、4 阶段
│   ├── observations.py           # 56 维观测的拆分项（腿/轮/指令）
│   ├── commands.py               # MixedVelocityCommand：50/30/10/10 混合采样
│   ├── rewards.py                # 奖励函数（站立 14 项 + 速度 10 项）
│   ├── dr.py                     # 域随机化档位（站立任务用）
│   └── agents/rsl_rl_ppo_cfg.py  # PPO 超参（两个任务各一套）
└── scripts/
    ├── check_m20.py   # 加载 USD，打印关节/质量，验证资产
    ├── train.py       # 训练（速度任务含 4 阶段课程与续训）
    ├── play.py        # 播放 / 键盘给指令 / 录视频
    └── eval.py        # 固定指令验收测试（RMSE、倾角、摔倒率）
```

## 快速开始

仓库根目录执行（脚本是模块入口，不能 `cd` 进 `scripts/`）：

```bash
# 1. 从官方 URDF 导入一次训练 USD，然后验证（16 个关节、总质量 ≈38.5 kg）
python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf

# 2. 训练静态站立（默认 4096 环境，模型很小）
python -m isaaclab_m20.scripts.train --headless

# 3. 播放站立策略（加载最新 checkpoint，16 个环境）。视口里 P 推一下，R 重置
python -m isaaclab_m20.scripts.play --num_envs 16

# 4. 训练速度指令任务：4 阶段课程，阶段间自动续训（默认 4096 环境，不用传 --num_envs）
python -m isaaclab_m20.scripts.train --headless --run_name m20_v1

# 5. 推荐流程：一次只跑一个阶段，跑完进模拟器看，再决定下一步。
#    --stage N 自动接上一阶段：从同一次 run 里第 N-1 阶段的最新 checkpoint 续训。
#    重新训练同一阶段＝覆盖该阶段目录（旧的 model_*.pt 先清掉，避免更高迭代号被当成
#    本阶段结果）。想保留上一次的对比就加 --new_run 另开一个时间戳目录；
#    --keep_logs 则原地保留旧文件。
#    checkpoint 文件名 = 训练结束时已完成的累计迭代数（rsl_rl 原本会少 1，这里改掉了）。
python -m isaaclab_m20.scripts.train --headless --stage 1 --run_name m20_v1

# 6. 只看某个阶段的模型：play 加 --stage N，取最新 run 里 stageN_* 的最新 checkpoint。
#    阶段编号：1 站稳 / 2 前后 / 3 全向（横移+偏航） / 4 更快、含起步急停切换
#    （--stage 只在速度任务里有，所以给了它就自动选速度任务，不用再写 --task）
python -m isaaclab_m20.scripts.play --stage 1
# 看完再训下一段（这段从上面 stage 1 的最新模型继续）
python -m isaaclab_m20.scripts.train --headless --stage 2 --run_name m20_v1
python -m isaaclab_m20.scripts.play --stage 2

# 7. 每个阶段的轮数都是参数：单阶段直接给 --iterations
python -m isaaclab_m20.scripts.train --headless --stage 2 \
    --run_name m20_v1 --iterations 1200
#    （--iters_stage1 … --iters_stage4 只在 --stage all 跑整套课程时才用）

# 8. 冒烟测试：第 1 阶段只跑 20 轮（约 1 分钟）
python -m isaaclab_m20.scripts.train --headless --num_envs 128 --stage 1 --iterations 20

# 9. 播放速度指令策略。W/S = vx，A/D = vy，Q/E = 偏航角速度，空格清零
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0

# 10. 固定指令 + 无界面出视频（--stage 也能和它们一起用，给了 --stage 就不用写 --task）
python -m isaaclab_m20.scripts.play --stage 4 --cmd_vx 0.3 --cmd_wz 0.3 \
    --headless --video

# 11. 验收：9 组固定指令 + 1 组起步/停下切换，输出 RMSE / 倾角 / 摔倒率 / PASS-FAIL
python -m isaaclab_m20.scripts.eval --stage 4

# 12. 曲线
tensorboard --logdir logs/rsl_rl --bind_all --port 6006
```

## 控制与观测

**时间尺度**：物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`，4 个物理步出一次动作）。

**静态站立任务**：没有速度指令，目标是原地站稳、保持高度（0.50 m）、机身水平、不漂移，轮速压零。回合 10 s。

**速度指令任务**（`docs/algorithm.md` §6 是完整说明）：

- **观测 56 维**：`base_ang_vel(3) + projected_gravity(3) + base_lin_vel(3) + 腿关节位置偏差(12) + 腿关节速度(12) + 车轮角速度(4) + 指令(3) + last_action(16)`。线速度和指令都在**机体系**，训练/播放/部署必须同一个坐标系。
- **动作 16 维** ∈ [-1, 1]，都是**目标值**：腿 12 维 `q* = q_default + 0.25·a`（rad），轮 4 维 `ω* = 8.0·a`（rad/s，轮半径 0.09 m → 0.72 m/s 上限）。
- **指令**：机体系 `(vx, vy, ωz)`，每 2–4 s 重采样，混合采样 50% 单轴 / 30% 双轴 / 10% 三轴 / 10% 零指令。
- **奖励 10 项每步 + 摔倒 −10**：任务项 `track_lin_vel_xy +2.0`（σ 0.25，有界柯西核 `1/(1+e²/σ²)`）、`track_ang_vel_z +1.0`（σ 0.3，同核）；机身状态项 `base_height −0.5`（`((h−0.50)/0.05)²`，平方误差不是奖金）、`flat_orientation −0.5`（姿态**平方**惩罚，0.2）、`lin_vel_z −0.2`（上下弹跳，0.2 m/s）、`ang_vel_xy −0.2`（姿态**角速度**，补"幅度小但抖得快"）；轮/姿态保持 `wheel_speed −0.2`（**唯一带门控**的一项，只在命令严格为零时生效，压住"机身不动但轮子空转"）；正则项 `joint_deviation −0.05`（只腿）、`leg_joint_acc −0.02`（腿关节加速度 /200）、`action_rate −0.01`（按 0.05 归一，与加速度项一起抗抖）；`termination_penalty −10`（**只在真摔倒时**扣，20 s 超时不扣）。**没有存活加成、没有位置保持项、没有任何死区**——存活加成与摔倒惩罚不是同一件事的两面（一个每步发钱、一个只罚一次），而 `envs.py` 的每步 0 下限已经让"活着"永远不比"摔死"差；"零指令不许动"由第 1 项自己承担（指令为零就是跟踪零速度），被推之后回到原处是**第四种能力**，先在 `eval` 第 9 行的 `drift` 列里单独量，不塞进同一个奖励。**不做与指令无关的绝对速度惩罚、不强制四轮着地、不惩罚横向滑移。**
- **终止**：高度 < 0.33 m、倾角 > 0.5 rad、超时 20 s。两个门限与摔倒惩罚共用同一对常量（结束回合的判据就是扣钱的判据）。
- **奖励下限**：速度任务用 `isaaclab_m20/envs.py` 的 `M20VelocityEnv`，它把**每步**奖励下限压到 0（摔倒的 −10 照付）。没有这一道，跟不住指令时每步净亏、摔倒只付一次钱，策略会算出来"秒速摔倒更划算"。代价是 tensorboard 的 `Episode_Reward/*` 仍是未 clip 的原始值，看学习信号请用 `Train/mean_reward`。
- **课程 4 阶段**（合计 3250 iterations，每阶段内有线性 warmup）：零指令站稳（300，**100% 零指令**）→ 前进/后退 `vx ±0.4`（700，**零指令 50%**）→ 全向 `vx ±0.5 / vy ±0.2 / ωz ±0.6`（1000，**零指令 30%**）→ 更快、含起步急停 `ωz ±0.8`（1250，**零指令 20%，每条指令只保持 1.5–3 s**）。每个阶段都保留一部分零指令，是为了让"站住"在学新动作时被反复复习——不保留的话，后面的阶段会把停止能力训掉（这正是之前"放出来一听指令就倒、一 hold 住就溜"的原因）；零指令占比逐阶段下降，重心从站立移到跟踪。
- **域随机化默认关闭**（摩擦 1.0、质量/质心固定、无推力），`--dr full` 打开。

**PD 伺服环**（`ImplicitActuatorCfg`，200 Hz）：腿 kp=80 / kd=2，轮 kp=0 / kd=2，力矩上限 76.4 / 21.6 N·m。

## 显存与环境数

两个任务默认都是 **4096 环境**（`--num_envs` 不传就用这个值），4090 24 GB 跑满没问题：

| 显存 | 建议环境数 |
| --- | --- |
| 16 GB | 2048 |
| 24 GB | **4096（默认）** |
| 48 GB | 4096–8192 |
| 80 GB | 8192+ |

只有在显存更小或只是冒烟测试时才需要传 `--num_envs`。

**训练期间只跑一个 Isaac Lab 进程。** 两个进程抢同一张卡的表现是帧率崩到个位数，而不是干脆的 OOM。

## 调参入口

| 想改的东西 | 文件 |
| --- | --- |
| 奖励权重、终止条件、域随机化（站立） | `isaaclab_m20/tasks/m20_env_cfg.py` |
| 观测布局、奖励权重、动作尺度、终止、DR 开关（速度指令） | `isaaclab_m20/tasks/m20_velocity_env_cfg.py` |
| 奖励函数本身 | `isaaclab_m20/tasks/rewards.py` |
| 观测拆分项（腿/轮/指令） | `isaaclab_m20/tasks/observations.py` |
| 指令混合采样比例 | `isaaclab_m20/tasks/commands.py` |
| 课程阶段、轮数、warmup、续训逻辑（含每阶段的零指令占比与最后一阶段的保持时间） | `isaaclab_m20/scripts/train.py`（`STAGES`） |
| 重训同一阶段时覆盖还是另开目录 | `--new_run` 另开时间戳目录、`--keep_logs` 原地保留旧文件；**默认覆盖该阶段目录** |
| 验收阈值与测试指令 | `isaaclab_m20/scripts/eval.py` |
| PPO 超参 | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
