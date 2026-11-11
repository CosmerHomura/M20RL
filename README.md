# M20 轮足机器人（Isaac Lab + rsl_rl）

用 Isaac Lab 的 ManagerBasedRLEnv 训练 M20 轮足机器人：PPO 输出 12 个腿部关节的位置目标 + 4 个车轮的速度目标，PD 伺服环在 200 Hz 跟踪成力矩，后端是 rsl_rl，GPU 上并行数百到数千个环境。

两个**互相独立**的任务，各自训练、各自的 checkpoint：

| gym id | 任务 | 观测 | 回合 | 日志 |
| --- | --- | --- | --- | --- |
| `Isaac-M20-Wheeled-v0` | 静态站立 | 57 维（无指令） | 10 s | `logs/rsl_rl/m20_wheeled/` |
| `Isaac-M20-Wheeled-Velocity-v0` | `(vx, vy, ωz)` 速度指令跟踪 | 56 维 | 20 s | `logs/rsl_rl/m20_velocity/<run>/stage{N}_*/` |

两者的 checkpoint **不能互换**（观测维度不同）。速度指令任务是第一版正式训练配置：4 阶段课程（站稳 → 前后 → 全向 → 更快、含起步/急停切换）、11 项每步奖励 + 摔倒 −10（追踪用有界柯西核；两道**模式门**：STAND 下压轮子空转、ROTATE 下不许平移；观测第 37–39 维是解算出来的目标，不是原始指令；无存活加成、无位置保持、每步奖励**没有** 0 下限）、域随机化默认关闭、阶段间自动续训。

详细安装与排错见 [docs/isaaclab_setup.md](docs/isaaclab_setup.md)，奖励函数/网络结构/PPO 流程/课程与验收见 [docs/algorithm.md](docs/algorithm.md)。

## 环境要求

- Isaac Sim 5.1 + IsaacLab 2.3.2（Python 3.11，driver ≥ 580）
- rl 后端：`pip install "isaaclab[rsl-rl]" --extra-index-url https://pypi.nvidia.com`
- 首次启动前 `export OMNI_KIT_ACCEPT_EULA=YES`

## 目录

```text
model/
├── usd/               # 官方 USD（默认：M20.usd + configuration/ 子层；缺 17 MB 的
│                      #   M20_base.usd 时会自动回退到 M20_urdf.usd 并打印 WARN）
├── urdf/              # 官方 URDF（回退路线 M20_USD=urdf 的导入源）
├── M20_urdf.usd       # URDF 导入版（不进 git，只有 M20_USD=urdf 时才用）
└── M20.xml + meshes/  # 官方 MJCF（备用；关节轴与限位可在这里核对）
isaaclab_m20/
├── assets/m20.py      # ArticulationCfg：USD、初始姿态、PD 增益
├── envs.py            # M20VelocityEnv：摔倒惩罚按终止标志加（只罚真摔倒，不做奖励裁剪）
├── tasks/
│   ├── m20_env_cfg.py            # 站立任务：场景/观测/动作/奖励/域随机化
│   ├── m20_velocity_env_cfg.py   # 速度指令任务：56 维观测、11 项奖励、4 阶段
│   ├── observations.py           # 56 维观测的拆分项（腿/轮/指令）
│   ├── commands.py               # MixedVelocityCommand：50/30/10/10 混合采样
│   ├── rewards.py                # 奖励函数（站立 14 项 + 速度 11 项）
│   ├── dr.py                     # 域随机化档位（站立任务用）
│   └── agents/rsl_rl_ppo_cfg.py  # PPO 超参（两个任务各一套）
├── cli/train_args.py  # train 的全部命令行参数（只有 argparse，不含仿真逻辑）
├── training/          # 课程 stages.py / 调度 curriculum.py / 目录规则 checkpoints.py / 诊断 diagnostics.py
└── scripts/           # 薄入口：启动 app，然后调 cli + training
    ├── check_m20.py   # 加载 USD，打印关节/质量，验证资产
    ├── check_joints.py# 关节配置审计：正则解析、默认姿态一致性、零动作落位
    ├── diagnose_command.py # 指令链路体检：谁在改命令缓冲
    ├── train.py       # 训练入口（课程表在 training/stages.py，调度在 training/curriculum.py）
    ├── play.py        # 播放 / 键盘给指令 / 录视频
    └── eval.py        # 固定指令验收测试（RMSE、倾角、摔倒率、净位移）
docs/
├── control.md         # 控制细节：时间尺度、执行器参数、两环带宽、驻车环、硬件对应、调试清单
├── algorithm.md       # 奖励、课程、PPO 超参、验收
└── isaaclab_setup.md  # 环境搭建与速查
```

## 快速开始

仓库根目录执行（脚本是模块入口，不能 `cd` 进 `scripts/`）：

```bash
# 1. 验证资产（默认加载官方 USD model/usd/M20.usd；层链不完整会自动回退并打 WARN）
python -m isaaclab_m20.scripts.check_m20 --headless

#    回退路线：从官方 URDF 导入一份再验证（会折叠 4 个无质量传感器 link）
M20_USD=urdf python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf

# 1b. 关节配置审计：正则解析到的 joint_ids、init_state 与 USD 默认姿态是否一致、
#     零动作 200 步后腿落在哪（怀疑"腿在折叠"或"默认姿态对不上"时先跑这个）
python -m isaaclab_m20.scripts.check_joints --headless

# 2. 训练静态站立（默认 4096 环境，模型很小）
python -m isaaclab_m20.scripts.train --headless

# 3. 播放站立策略（加载最新 checkpoint，16 个环境）。视口里 P 推一下，R 重置
python -m isaaclab_m20.scripts.play --num_envs 16

# 4. 训练速度指令任务：4 阶段课程，阶段间自动续训（默认 4096 环境，不用传 --num_envs）
python -m isaaclab_m20.scripts.train --headless --run_name m20_v1

# 5. 推荐流程：一次只跑一个阶段，跑完进模拟器看，再决定下一步。
#    --stage N 自动接上一阶段：从同一次 run 里第 N-1 阶段的最新 checkpoint 续训。
#    重新训练**同一阶段＝接着训**：目录复用、加载该阶段最新模型，然后跑多少轮由你说了算
#    ——默认该阶段预算，`--iterations N` 就是这次跑 N 轮（脚本不替你算差额、也不判断够不够）。
#    续训时**环境回到停下那一刻**：命令 ramp 从该阶段已达的迭代位置接上（warmup 中断就从中断处继续，
#    不会跳到满幅），采样分布用该阶段的；同时把当前 cfg 与首次训练的 env.yaml 逐项对比，
#    有差异（指令范围/奖励权重/域随机化/环境数）会 WARN，并把本次配置另存为 env.resume-<时间戳>.yaml。
#    要真的从头重来才加 --overwrite（旧目录改名为 stageN_xxx.bak-<时间戳>，不删）。--new_run 另开一个 run。
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

## 命令行参数总表

所有脚本都以 `python -m isaaclab_m20.scripts.<名字>` 运行。下面几张表是**全部**参数（默认值取自代码），`--help` 的输出与之一致。

### train.py —— 训练

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--task` | `Isaac-M20-Wheeled-Velocity-v0` | 任务 id；站立任务是 `Isaac-M20-Wheeled-v0` |
| `--num_envs` | 不传＝4096 | 并行环境数。24 GB 卡直接 4096；只有冒烟测试才降到 128 之类 |
| `--seed` | `0` | 随机种子 |
| `--stage` | `all` | `1`…`4`（站稳 / 前后 / 全向 / 更快）或 `all`。给了某个阶段：已训练过就**接着训**，没训练过就从上一阶段最新模型起 |
| `--stage_scale` | `1.0` | 所有阶段预算的倍率（`0.05` 用来冒烟）。**只缩轮数，不缩指令** |
| `--cmd_scale` | `1.0` | 所有阶段**指令范围**的倍率（三轴同比缩，warmup 两端一起缩；预算、零指令占比、保持时间不变）。`0.7` = 阶段 2 只要 `±1.75` 而不是 `±2.5`。这是"策略还吃不下这个速度"的旋钮。注意：续训时改它会让 `env.yaml` 对比报 WARN，这是设计如此 |
| `--iterations` | 不传＝该阶段预算 | **这次跑多少轮**。新阶段、续训都一样是字面值，脚本不做差额、也不判断够不够 |
| `--warmup_frac` | 不传＝阶段自带 warmup（且不超过预算一半） | 把指令 ramp 写成**预算的比例**：`0.5` = 前一半线性涨到本阶段满幅、后一半在范围内随机。只对自带 warmup 的阶段（2、3）生效。预算比原计划大很多时用它，例如阶段 2 跑 4000 轮时 `--warmup_frac 0.5` → 前 2000 轮爬坡 |
| `--iters_stage1` … `--iters_stage4` | 该阶段预算 × `--stage_scale` | 只在 `--stage all` 有意义：分别指定每个阶段的轮数（优先级最高）。单跑某阶段请用 `--iterations` |
| `--run_name` | 不传＝沿用最新 run | `logs/rsl_rl/m20_velocity/<run_name>/` 下的 run 目录名，阶段是其子目录 |
| `--new_run` | 关 | 另开一个时间戳 run（保留旧 run 作对比），并从"任意 run 里上一阶段最新模型"起步 |
| `--overwrite` | 关 | 真的从头重来：把该阶段旧目录**改名**为 `<stage>.bak-<时间戳>`（永不删除） |
| `--log_root` | `logs/rsl_rl` | 日志根目录（checkpoint、`env.yaml`/`agent.yaml`、tensorboard event 都在里面） |
| `--video` | 关 | 录第一轮的视频 |
| `--video_length` | `200` | 视频长度（策略步） |
| `--video_folder` | `videos` | 视频输出目录 |
| `--zero_actions` | 关 | 保持默认姿态、不训练。隔离测试：这样还崩就说明问题在模型/仿真，不在策略 |
| `--health_every` | `24` | 每 N 个 env step 打印一行 `[health]`（`0` 关闭） |
| `--per_term_every` | `200` | 每 N 个 env step 打印 `[terms]`：每个奖励项的 `value*weight*dt` 与回合累计值。这是唯一可比的尺度（`0` 关闭） |
| `--dr` | `off` | `off`＝名义机器人（平地、摩擦 1.0、无质量/质心扰动、无推力）；`full`＝开摩擦/质量/质心随机化与推力 |
| `--dr_ramp` | `0.4` | **仅站立任务**：域随机化从 0 到满幅所占的训练比例 |
| `--learning_rate` / `--lr` | 不传（cfg `3e-4`） | PPO 的**起始**学习率。adaptive 下它只是起点（KL 控制器每轮改它），`--fixed_lr` 下才是整段不变的速率。只改速率，不动预算 / 指令范围 / 奖励 |
| `--fixed_lr` | 关 | 把学习率钉死在 `learning_rate`（`schedule="fixed"`），不用 cfg 的 adaptive-in-KL。adaptive 是默认（原计划设置），但它会让 Adam 步长在 ~0 与 ~2e-3 之间摆 |
| `--desired_kl` | 不传（cfg `0.01`） | 覆盖 adaptive 的 KL 目标。**注意 rsl_rl 比的是 16 个动作维的 KL 之和**，所以 0.01 相当于单维 6e-4——太紧，40 步一轮在 `lr=1e-5` 都能超，结果 lr 永久贴地板。这个网络可用的区间是 **0.05–0.08** |

### play.py —— 在仿真里看模型

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--stage` | 不传 | `1`…`4`：取最新 run 的 `stageN_*` 里迭代号最大的 checkpoint。**给了它就不用再写 `--task`**（阶段只存在于速度任务） |
| `--checkpoint` | 不传＝最新模型 | 可以是 `model_*.pt` 文件、阶段目录、或 run 目录（目录取其中迭代号最大者） |
| `--task` | 由 `--stage` 决定 | 任务 id；不给 `--stage` 时默认站立任务 |
| `--num_envs` | `16` | 播放时的环境数 |
| `--log_root` | `logs/rsl_rl` | 找 checkpoint 的根目录 |
| `--video` / `--video_length` / `--video_folder` | 关 / `500` / `videos` | 出视频而不是开窗口 |
| `--real_time` / `--no-real-time` | 有窗口时开 | 按仿真步长节流 |
| `--cmd_vx` `--cmd_vy` `--cmd_wz` | 全 `0` | 固定速度指令（m/s、m/s、rad/s）；不传则用键盘 |
| `--cmd_step` | `0.25` | 键盘每次按键给指令加多少 |
| `--status_every` | `50` | 每 N 个策略步打印一行 `[status]`（50 步 = 1 s）：指令缓冲、`|v_xy|`、高度、倾角、腿/轮最大关节速度、`standing`。用来区分"指令没进环境"和"轮子空转"；`0` 关闭 |
| `--push_speed` | `1.0` | 按键/自动推力的大小（m/s 的水平速度冲量） |
| `--push_every` | `0.0` | 每 N 秒仿真时间自动推一次（`0` 关闭），做视频用 |

键盘：`W/S` 前后、`A/D` 左右、`Q/E` 偏航、`空格` 指令清零、`P` 推一下、`R` 重置。

**指令归谁**：`play`（以及 `eval`）里指令归人，不归采样器。命令项本身每 2–4 s 和**每次回合重置**都会重采样，写入发生在我们之后，于是会出现"命令行给了 2.0 m/s，机器人却站着"——阶段 1 的采样范围是全零，重采样就等于清零。两边都调 `freeze_command_resampling()` 把它冻住，并改走 `M20VelocityEnv.set_command()`：指令被存成 `command_override`，每步开头与每次重置之后都写回一次（重置发生在 `step` 内部、观测构建之前，光写缓冲盖不住这个顺序）。自证输出：启动的 `[INFO] Policy obs: ... command slice obs[37:40]`、每步 `[status]` 里的 `cmd=` 与 `obs_cmd=`、以及首次写入后的 `[INFO] write check`。三者都对而机器人不动，才是策略或抓地的问题。详见 `docs/control.md` §7.1。

### eval.py —— 验收（固定指令序列）

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--stage` | 不传 | `1`…`4`：评测最新 run 的该阶段 |
| `--checkpoint` | 不传＝最新模型 | `model_*.pt` 文件、阶段目录或 run 目录 |
| `--task` | `Isaac-M20-Wheeled-Velocity-v0` | 任务 id |
| `--num_envs` | `16` | 评测环境数（同一组指令并行跑，最后取平均） |
| `--seconds` | `8.0` | 每组测试的时长（扣掉 settle 之后） |
| `--settle` | `0.5` | 切换指令后丢弃的过渡时间（s） |
| `--seed` | `12345` | 评测种子（与训练种子分开，避免背题） |
| `--log_root` | `logs/rsl_rl` | 找 checkpoint 的根目录 |
| `--json_out` | 不传 | 把结果写成 JSON 到这个路径 |
| `--strict` | 关 | 有任何一项 FAIL 就以非零码退出（CI 用） |

输出 12 组：`(0.3,0,0)`、`(−0.3,0,0)`、`(0,0.2,0)`、`(0,−0.2,0)`、`(0,0,0.5)`、`(0,0,−0.5)`、`(0.3,0,0.3)`、`(−0.3,0.2,−0.3)`、`(0,0,0)`（含 `drift` 净位移）、第 10 组的"起步→停住"切换测试，以及新增的 **`(1.0,0,0)` 与 `(2.0,0,0)`** 两条速度包线测试（前九组都在 0.5 m/s 以下，没有这两条，只会慢慢爬的策略也能全过）。

### check_joints.py —— 关节/奖励配置审计

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--task` | `Isaac-M20-Wheeled-Velocity-v0` | 要建的任务 |
| `--num_envs` | `1` | 环境数（1 个就够） |
| `--usd` | 不传＝按 `assets/m20.py` | 覆盖机器人 USD（对比 `model/usd/M20.usd` 与 `model/M20_urdf.usd` 时用） |
| `--steps` | `200` | 零动作跑多少步再读"落位"姿态 |
| `--settle` | 关 | 每 25 步也打印一次姿态，用来看还在不在动 |

### check_m20.py / inspect_usd.py

| 脚本 | 参数 | 默认 | 作用 |
| --- | --- | --- | --- |
| `check_m20.py` | `--usd` | `model/M20_urdf.usd` | 要检查的 USD |
| `check_m20.py` | `--convert urdf` | 不传 | 先把 `model/urdf/M20.urdf` 重新导入成 USD |
| `inspect_usd.py` | `--usd` | `model/usd/M20.usd` | 打印 USD 的 prim/关节结构 |

### 所有脚本共享的 Isaac 启动参数（由 AppLauncher 提供）

| 参数 | 作用 |
| --- | --- |
| `--headless` | 不开窗口（服务器必用） |
| `--device cuda:0` / `cpu` | 计算设备 |
| `--enable_cameras` | 需要相机/RTX 传感器时打开 |
| `--livestream 1` / `2` | 用 WebRTC 把视口推流出去（无显示器又要看画面时） |
| `--kit_args "..."` | 透传给 Isaac Sim 的额外参数 |

环境变量：`M20_USD=urdf` 强制用 `model/M20_urdf.usd`（默认优先官方 `model/usd/M20.usd`，层链不完整时自动回退并打印 WARN）。

## 控制与观测

**时间尺度**：物理 200 Hz（`sim.dt=0.005`），策略 50 Hz（`decimation=4`，4 个物理步出一次动作）。

**静态站立任务**：没有速度指令，目标是原地站稳、保持高度（0.50 m）、机身水平、不漂移，轮速压零。回合 10 s。

**速度指令任务**（`docs/algorithm.md` §6 是完整说明）：

- **观测 56 维**：`base_ang_vel(3) + projected_gravity(3) + base_lin_vel(3) + 腿关节位置偏差(12) + 腿关节速度(12) + 车轮角速度(4) + 指令目标(3) + last_action(16)`。第 37–39 维不是原始指令，而是 `tasks/control.py` 解算出来的**目标**：指令系速度 `(vx*, vy*)` 用 `R(−ψ_rel)` 旋到机体系，第三维是外环航向 PD 给出的偏航速率目标 `ω*`。线速度和目标都在**机体系**，训练/播放/部署必须同一个坐标系。
- **动作 16 维** ∈ [-1, 1]，都是**目标值**：腿 12 维 `q* = q_default + 0.25·a`（rad），轮 4 维 `ω* = 27.78·a`（rad/s，轮半径 0.09 m → **2.5 m/s 上限**，比人走路快）。
- **指令**：机体系 `(vx, vy, ωz)`，每 2–4 s 重采样，混合采样 50% 单轴 / 30% 双轴 / 10% 三轴 / 10% 零指令。
- **奖励 11 项每步 + 摔倒 −10**：任务项 `track_lin_vel_xy +2.0`（σ 0.25，有界柯西核 `1/(1+e²/σ²)`，比的是**机体系解算目标** `v_body`）、`track_ang_vel_z +1.0`（σ 0.3，同核，比的是外环给的 `ω*`）；机身状态项 `base_height −0.5`（`((h−0.50)/0.05)²`，平方误差不是奖金）、`flat_orientation −0.5`（姿态**平方**惩罚，0.2）、`lin_vel_z −0.2`（上下弹跳，0.2 m/s）、`ang_vel_xy −0.2`（姿态**角速度**，补"幅度小但抖得快"）；轮/姿态保持 `wheel_speed −0.2`（**STAND 门**：只在模式标签是 STAND 时生效，压住"机身不动但轮子空转"）；正则项 `joint_deviation −1.0`（只腿、12 关节**取均值**、1 rad 归一——第一版是 `Σ/0.3`、−0.05，实测把"把腿收回来"这一项算成 367、比正项上限大 9 倍，见 `docs/algorithm.md` §6.3）、`leg_joint_acc −0.02`（腿关节加速度 /2000）、`action_rate −0.01`（按 1.0 归一，与加速度项一起抗抖——这两个尺度不能按"可读"收紧，否则会把探索噪声放大成 −40/步）；模式项 `zero_xy_speed −1.0`（**ROTATE 门**：纯旋转时不许平移，`‖v_xy/0.15‖²`；与 STAND 门互斥，因为一个环境每步只有一个模式标签）；摔倒 `−10` 由 `M20VelocityEnv` 在 `terminated & ~truncated` 的步上加（**只罚真摔倒，20 s 超时不罚**）。**没有存活加成、没有位置保持项、没有任何死区**——"零指令不许动"由第 1 项自己承担（指令为零就是跟踪零速度），被推之后回到原处是**第四种能力**，先在 `eval` 第 9 行的 `drift` 列里单独量，不塞进同一个奖励。**不做与指令无关的绝对速度惩罚、不强制四轮着地、不惩罚横向滑移。**
- **终止**：高度 < 0.33 m、倾角 > 0.5 rad、超时 20 s。两个门限与摔倒惩罚共用同一对常量（结束回合的判据就是扣钱的判据）。
- **没有奖励裁剪**：`isaaclab_m20/envs.py` 的 `M20VelocityEnv` **不**把每步奖励压到 0（legged_gym 的 `only_positive_rewards` 试过又删了：策略早期每步为负，被 clip 成常数 0、整轮梯度消失、平均奖励恒为 0）。PPO 不需要非负奖励，`A = R − V` 归一化只缩放不改序。所以 `Train/mean_reward` 与 `Episode_Reward/*` 是同一套数字，可以直接读；每步为负不是故障。
- **课程 4 阶段**（合计 4200 iterations，每阶段内有线性 warmup）：零指令站稳（900，**100% 零指令**；默认 `adaptive` 下 200 轮左右就走平，900 是覆盖 `--fixed_lr` 的上界）→ 前进/后退 `vx ±2.5`（700，**零指令 50%**，warmup 350 轮从 0 长到满包线）→ 全向 `vx ±2.5 / vy ±0.4 / ωz ±0.8`（1200，**零指令 30%**）→ 顶速、含起步急停 `vx ±2.5 / vy ±0.5 / ωz ±1.0`（1400，**零指令 20%，每条指令只保持 1.5–3 s**）。速度包线 2.5 m/s 由轮动作 `scale=27.78` 决定（`× 0.09 m`），不是电机上限；旧值 `scale=8.0` 只有 0.72 m/s、比走路还慢。每个阶段都保留一部分零指令，是为了让"站住"在学新动作时被反复复习——不保留的话，后面的阶段会把停止能力训掉（这正是之前"放出来一听指令就倒、一 hold 住就溜"的原因）；零指令占比逐阶段下降，重心从站立移到跟踪。
- **域随机化默认关闭**（摩擦 1.0、质量/质心固定、无推力），`--dr full` 打开。

**PD 伺服环**（`ImplicitActuatorCfg`，200 Hz）：腿 kp=80 / kd=2，轮 kp=0 / kd=2，力矩上限 76.4 / 21.6 N·m。

**零指令时的轮子驻车环**（`isaaclab_m20/envs.py` 的 `M20VelocityEnv`）：轮子是 `stiffness=0` 的速度伺服，而 PhysX 不建模滚动阻力——静摩擦只阻止滑动、不阻止滚动，所以零指令下唯一的阻力就是 `damping=2`，机身只要倾斜 0.9° 就会以 `F_bias/(4·damping) ≈ 5 mm/s` 持续蠕动，8 s 就是 4 cm，正好卡在验收线附近。因此当命令缓冲严格为零时，env 把四个轮子动作列替换成位置外环的输出：`ω* = clip(15·(q_hold − q), ±3 rad/s)`，再由原有的速度环跟踪（`τ = damping·(ω* − q̇)`）。外环**没有独立的 kd**（阻尼就是共享的 `damping=2`，比外环快 47 倍），所以它是一阶过阻尼、不会振；等效驻车刚度 `15×2 = 30 N·m/rad ≈ 0.5 N·m/deg`，比腿的 80 软。**不切增益**（执行器始终是速度伺服，所以没有力矩阶跃），**替换值写回 action**（`last_action` 与 `action_rate` 看到的是真正下发的命令），**轮角不 wrap**（转 2π 车身走 0.57 m，缠绕会把方向弄反），`q_hold` 只在指令刚归零的上升沿和回合重置时重锚，误差超过 0.3 rad 就重锚（被拖动/打滑）。参数在 `M20WheeledVelocityEnvCfg`：`wheel_park_enabled / wheel_park_kp / wheel_park_omega_max / wheel_park_error_clamp`。控制细节（时间尺度、执行器参数、两环带宽、调试清单）见 [docs/control.md](docs/control.md)，设计动机见 `docs/algorithm.md` §6.2.1。

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
| 摔倒惩罚量级 | `isaaclab_m20/tasks/m20_velocity_env_cfg.py` 的 `fall_penalty`（默认 −10；由 `isaaclab_m20/envs.py` 在 `terminated & ~truncated` 的步上加，不是奖励项） |
| 终端诊断输出 | `--health_every 24`（`[health]` 行：`|v_xy|`/`standing`/`rew`/`neg`/`fall_rew`/`timeout_rew`）、`--per_term_every 200`（`[terms]` 行：逐项 `value×weight×dt` 的每步实际贡献 + `_episode_sums` 累计值；`sum=` 应等于同一时刻的 `rew=`） |
| 观测拆分项（腿/轮/指令） | `isaaclab_m20/tasks/observations.py` |
| 指令混合采样比例 | `isaaclab_m20/tasks/commands.py` |
| 课程阶段、轮数、warmup、续训逻辑（含每阶段的零指令占比与最后一阶段的保持时间） | `isaaclab_m20/training/stages.py`（`STAGES`）与 `training/curriculum.py`（`CommandSchedule`） |
| 重训同一阶段时覆盖还是接着训 | **默认接着训**：复用该阶段目录、加载其中最新的 `model_*.pt`；跑多少轮由命令行决定（默认该阶段预算，`--iterations N` 就是 N 轮，脚本不做差额计算）。要从头重来加 `--overwrite`（旧目录改名为 `.bak-<时间戳>`，不删）；要另开目录加 `--new_run` |
| 验收阈值与测试指令 | `isaaclab_m20/scripts/eval.py` |
| PPO 超参 | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
