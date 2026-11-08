# Isaac Lab 训练（云端 4090）

训练在 Isaac Lab + rsl_rl 上跑，重点利用**并行环境数**：站立任务在 4090 上可以开 4096 个环境；速度指令任务因为观测/网络更大、还要留显存给 PPO 更新，从 512 起步、稳定后 1024。

资产来自 Deep Robotics 官方仓库 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model/tree/main/M20)：

```text
model/
├── urdf/                # 官方 URDF（训练资产由它导入生成）
├── M20_urdf.usd         # 生成的训练 USD（不进 git，首次用 --convert urdf 生成）
├── usd/                 # 官方 USD 及其配置子层（备用，见第 4 节）
└── M20.xml + meshes/    # 官方 MJCF（备用转换路径 + 本机快速验证）
```

代码在 `isaaclab_m20/`：

```text
isaaclab_m20/
├── assets/m20.py              # ArticulationCfg：USD 路径、初始姿态、腿/轮执行器增益
├── tasks/rewards.py           # 奖励项函数（两个任务共用）
├── tasks/m20_env_cfg.py       # 静态站立：场景、观测、动作、奖励、终止、事件
├── tasks/m20_velocity_env_cfg.py  # 速度指令：继承上者，换观测/奖励/指令
├── tasks/dr.py                # 域随机化的课程缩放
├── tasks/__init__.py          # gym.register(...) 两个任务 id
├── tasks/agents/rsl_rl_ppo_cfg.py
└── scripts/
    ├── check_m20.py           # 加载 USD，打印关节/质量，验证资产
    ├── train.py
    └── play.py
```

## 1. 环境要求

| 项 | 要求 |
| --- | --- |
| 系统 | Ubuntu 22.04（推荐）/ 24.04 |
| 驱动 | 580.65 以上（Isaac Sim 5.x）；`nvidia-smi` 确认 |
| Python | 3.11（IsaacLab 2.3.2 对应 Isaac Sim 5.x） |
| 显存 | ≥ 16 GB（4090 24 GB：站立任务 4096 环境没问题，速度指令任务建议 512–1024） |
| 内存 | ≥ 32 GB |
| 磁盘 | ≥ 20 GB（Isaac Sim wheels 本身就有十几 GB） |

如果你的镜像里已经是 Python 3.12，就装 IsaacLab 3.0 + Isaac Sim 6.2 那一支；本文命令按 2.3.2 写，3.0 的包名一致，只是版本号不同。

## 2. 安装

纯净的 PyTorch/CUDA 镜像里没有 Isaac Sim，需要装完整套（会从 NVIDIA 源拉几 GB）：

```bash
python -V                              # 必须是 3.11
pip install --upgrade pip
pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com
```

系统依赖（headless 录制视频才需要，纯训练可以跳过）：

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
```

验证：

```bash
export OMNI_KIT_ACCEPT_EULA=YES          # 首次启动跳过交互式 EULA 询问
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 没有这个环境变量时，第一次 `AppLauncher` 会问 `Do you accept the EULA? (Yes/No):`。
> 用 `python - <<'PY'` 跑脚本时 stdin 被 heredoc 占用，会直接报
> `Unable to bootstrap inner kit kernel: EOF when reading a line`。
>
> 装完不需要 `isaaclab.sh` 或额外 source；pip 包已经把 Isaac Sim 的 Python 模块装进同一个环境。

## 3. 上传代码

仓库根目录必须同时有 `isaaclab_m20/` 和 `model/`（至少 `model/urdf/M20.urdf`，训练 USD 由它生成）。

在仓库根目录执行（不要 `cd isaaclab_m20/scripts`，模块入口需要根目录在 `sys.path` 上）：

```bash
python -m isaaclab_m20.scripts.train --help
```

## 4. 先验证 USD（务必先做）

训练加载的是 `model/M20_urdf.usd`，由官方 URDF 一次性导入生成（这个文件不进 git，换机器要重新生成）：

```bash
# 首次：从官方 URDF 导入（约 1 分钟，首次会自动下载 URDF 导入扩展）
python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf

# 之后每次：只做检查
python -m isaaclab_m20.scripts.check_m20 --headless
```

它会加载 USD，打印 body / joint 名字、默认关节角、根状态、各刚体质量。对照预期：

- 16 个 joint，名字为 `fl_hipx_joint` … `hr_wheel_joint`（每条腿 hipx / hipy / knee / wheel）；
- **17 个 body**（base + 16 个腿部链接）；
- 默认关节角：hipx = 0.0，hipy = 0.61，knee = -1.22，wheel = 0.0；
- 总质量 ≈ 38.5 kg；
- 根高度 0.50 m。

官方 URDF 里 `front_camera_link` / `rear_camera_link` / `front_lidar_link` / `rear_lidar_link` 都是**空标签**（`<link name="x"/>`，没有 inertial / visual / collision）。保留它们会生成 4 个**零质量零惯量**的刚体，在 PhysX GPU 求解器里构成奇异质量矩阵，并在组合后的 stage 里留下 `/visuals/<link>` 的悬空引用——表现为运行时偶发的 `CUDA error: an illegal memory access`。所以导入时用 `merge_fixed_joints=True` 把它们并入 `base_link`。如果 body 数是 21 而不是 17，说明这个开关没生效，重新 `--convert urdf`。

**为什么不直接用 `model/usd/M20.usd`**：那是个组合层，`M20.usd`（1.4 KB）的默认变体早已选中 `PhysX / Sensors / Robot`，但它的子层链末端是 `configuration/M20_physics.usd` → `configuration/M20_base.usd`（17 MB，几何+质量）。这个 17 MB 文件一缺，整条链就塌成一个空 prim，报错是 `Failed to find an articulation when resolving '/World/envs/env_0/Robot'`。如果你确实想用官方 USD，把本地 `model/usd/configuration/M20_base.usd`（17,196,921 字节）传全，再 `--usd model/usd/M20.usd --headless` 检查即可，关节和质量应当一致。

**如果对不上**，按下面顺序排查：

1. **关节名不对** → 改 `isaaclab_m20/assets/m20.py` 和 `tasks/m20_env_cfg.py` 里的正则（`.*_hipx_joint` 等）；
2. **`FileNotFoundError: model/M20_urdf.usd`** → 还没跑 `--convert urdf`；
3. **质量/惯量为 0 或整个模型塌掉** → 检查 `init_state.pos` 的 z 是不是 0.50；
4. **力矩上限不对** → 执行器配置里已经写死 `effort_limit_sim`（76.4 / 21.6），不依赖 USD 自带值。

导入时刷屏的 `Unresolved reference prim path .../visuals/front_lidar_link` 是相机/雷达的视觉网格引用缺失，headless 训练用不到，忽略。

## 5. 训练

```bash
tmux new -s m20
# 站立任务（4096 环境，模型很小，24 GB 卡放得下）
python -m isaaclab_m20.scripts.train --headless --num_envs 4096

# 速度指令任务：5 阶段课程 + 阶段间续训（512 起步，稳定后改 1024）
python -m isaaclab_m20.scripts.train --headless --num_envs 512 --run_name m20_v1
```

- 站立任务每个 iteration 采集 4096 × 24 = 98,304 步；默认跑 3000 iterations。
- 速度指令任务每个 iteration 采集 512 × 24 = 12,288 步（1024 环境翻倍），5 阶段合计 3950 iterations。
- 日志：`logs/rsl_rl/m20_wheeled/<时间戳>/`（站立，每 100 iterations 存一个 checkpoint）；速度指令任务按阶段分目录，`logs/rsl_rl/m20_velocity/<run_name>/stage{N}_<name>/`，每阶段每 50 iterations 存一个 `model_*.pt`。
- 想给这次实验起名：`--run_name v1`，日志就落在 `logs/rsl_rl/m20_velocity/v1/`。
- 显存不够就降 `--num_envs`。**训练期间只跑一个 Isaac Lab 进程**——两个进程抢同一张卡的表现是帧率崩到个位数，而不是干脆的 OOM。
- 断点续训：站立任务 `--resume --run_name v1`（加载该目录下最新 checkpoint）；速度指令任务默认每阶段自动从上一阶段最新 checkpoint 续训，也可 `--stage 3 --resume_from <path>` 手指定。
- 只有一份配置文件、想快速看流程通不通：`--stage_scale 0.1 --num_envs 128 --run_name smoke`。

粗略预期（4090，单场景，无相机）：站立任务 3000 iterations 几小时量级；速度指令任务 1024 环境下约 20–30 min / 1000 iterations，整条 5 阶段课程通常 2–4 小时。

### 看曲线

```bash
tensorboard --logdir logs/rsl_rl --bind_all --port 6006
```

重点看：

- `Episode_Reward/flat_orientation` / `base_height`：是否在往 1 靠（机身水平、高度到位）；
- `Episode_Reward/base_motion`：越接近 0 越稳（不再漂移）；
- `Episode_Termination/base_height` / `bad_orientation`：翻车/趴下比例是否在降；
- `Loss/learning_rate`：`schedule="adaptive"`，学不动时会自动降 lr。

## 6. 播放与评估

```bash
# 最新 checkpoint，16 个环境（默认站立任务，自动找 m20_wheeled/ 下最新）
python -m isaaclab_m20.scripts.play --num_envs 16

# 指定 checkpoint
python -m isaaclab_m20.scripts.play \
    --checkpoint logs/rsl_rl/m20_wheeled/v1/checkpoints/model_2000.pt \
    --num_envs 16

# 无界面录制视频
python -m isaaclab_m20.scripts.play --headless --video
```

`play.py` 加载 checkpoint 后就地站立推演，视频写到 `videos/`，`--video_length` 是策略步数（50 Hz，500 步 = 10 s）。

`--task` 决定加载哪个任务的模型：不传 `--checkpoint` 时，脚本在 `logs/rsl_rl/<experiment>/` 下自动找最新 checkpoint——站立是 `m20_wheeled/`、速度指令是 `m20_velocity/`，两个目录的 checkpoint 维度不同（57 vs 56），按目录隔离就不会串。速度指令任务的 checkpoint 按阶段多一层目录（`.../stage5_mixed/model_1250.pt`），脚本递归查找、按修改时间取最新，所以默认拿到的就是最后一阶段的最新模型。

### 推一把试试

打开视口时可以用键盘：

| 键 | 作用 |
| --- | --- |
| `P` | 给机身一个随机水平方向的瞬时速度冲量 |
| `R` | 重置所有环境 |
| `W` / `S` | 前进 / 后退速度指令 ±`--cmd_step`（仅速度指令任务） |
| `A` / `D` | 左 / 右横移指令（仅速度指令任务） |
| `Q` / `E` | 偏航角速度指令（仅速度指令任务） |
| `Space` | 指令清零（仅速度指令任务） |

```bash
# 手动推：P 键推一次，--push_speed 是冲量大小（m/s）
python -m isaaclab_m20.scripts.play --num_envs 16 --push_speed 1.0

# 自动推：每 3 秒推一次，适合录视频
python -m isaaclab_m20.scripts.play --num_envs 16 --push_every 3 --headless --video

# 速度指令任务：键盘实时给指令（自动加载 m20_velocity/ 下最新 checkpoint）
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0

# 速度指令任务：固定指令（headless 时也可用）
python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
    --cmd_vx 0.6 --cmd_vy 0.0 --cmd_wz 0.3 --headless --video
```

速度指令任务里，`play.py` 会把指令重采样时间改成无穷大，并在每个策略步覆写指令缓冲——否则训练配置里 2–4 s 一次的重采样会不停覆盖人工输入。速度指令范围以当前 checkpoint 所在阶段的 `cfg.ranges` 为准（打印在启动日志里），键盘 ±`--cmd_step` 就在这个范围之外手动推。

### 验收测试

`logs/` 里最好的 checkpoint 不等于"能跑起来就等于学会了"。固定指令序列的自动验收：

```bash
# 9 组固定指令（前进/后退/左右横移/左右转/两个组合/零指令），输出 RMSE、roll-pitch RMS、摔倒率
python -m isaaclab_m20.scripts.eval

# 指定 checkpoint，结果同时写 JSON，任一项不过就返回非零退出码
python -m isaaclab_m20.scripts.eval \
    --checkpoint logs/rsl_rl/m20_velocity/m20_v1/stage5_mixed/model_1250.pt \
    --json_out eval.json --strict
```

eval 会强制**关掉域随机化、关掉指令重采样**，机器人是标称参数——测的是部署会看到的东西，不是训练环境的鲁棒性。阈值：前进/后退 RMSE < 0.10 m/s、横移 < 0.12 m/s、偏航 < 0.15 rad/s、roll/pitch RMS < 5°、摔倒率 < 2%（20 次/1000 策略步）、零指令机身速度 < 0.05 m/s。

推力用的和训练时的 `push_robot` 是同一个机制（`write_root_velocity_to_sim`，瞬时改变机身速度），所以可以直接对比：训练时是 ±0.5 m/s，`--push_speed 1.0` 就是**两倍于训练强度**的测试。

视口模式下默认按实时速度播放；headless 不限速。想强制关掉限速加 `--no-real_time`。

## 7. 控制与观测速查

两个任务共用同一套场景、动作、PD 与终止条件，差别只在观测/奖励/指令：

| gym id | 任务 | 观测 | 回合 | 日志目录 |
| --- | --- | --- | --- | --- |
| `Isaac-M20-Wheeled-v0` | 静态站立，无指令 | 57 维 | 10 s | `logs/rsl_rl/m20_wheeled/` |
| `Isaac-M20-Wheeled-Velocity-v0` | `(vx, vy, ωz)` 跟踪 | 56 维 | 20 s | `logs/rsl_rl/m20_velocity/<run>/stage{N}_*/` |

观测维度：

- 站立 57 = base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(16) + joint_vel(16) + last_action(16)；
- 速度指令 56 = base_ang_vel(3) + projected_gravity(3) + base_lin_vel(3) + 腿关节位置偏差(12) + 腿关节速度(12) + 车轮角速度(4) + 指令(3) + last_action(16)。

**两边的 checkpoint 不能互换**（观测维度不同）。

| 项 | 配置 |
| --- | --- |
| 控制频率 | 50 Hz（`sim.dt=0.005` × `decimation=4`） |
| 腿动作 | `JointPositionActionCfg(scale=0.25, use_default_offset=True)` |
| 轮动作 | `JointVelocityActionCfg`，`use_default_offset=False`；站立 scale=20（1.8 m/s），速度指令任务 scale=8.0（0.72 m/s） |
| PD 伺服 | 腿 kp=80 / kd=2，轮 kp=0 / kd=2，200 Hz |
| 力矩上限 | 腿 76.4 N·m，轮 21.6 N·m |
| 指令（速度指令任务） | 机体系 `(vx, vy, ωz)`，每 2–4 s 重采样；50% 单轴 / 30% 双轴 / 10% 三轴 / 10% 零指令；5 阶段课程，满幅 vx ±0.5、vy ±0.2、ωz ±0.8 |
| 奖励 | 站立 14 项 / 速度指令 **8 项 + 摔倒 −200**（不惩罚轮速、不强制四轮着地、不惩罚横向滑移），完整权重表见 [docs/algorithm.md](algorithm.md) §1 与 §6.3 |
| 终止 | 站立：高度 < 0.25、倾角 > 0.7、超时；速度指令：高度 < 0.28、倾角 > 0.8、超时 |
| 域随机化 | 站立全开（带 `--dr_ramp` 课程）；速度指令第一版**默认关闭**，`--dr full` 打开 |

## 8. 常见问题

- **`ModuleNotFoundError: No module named 'isaaclab'`**：用了别的 python。确认 `which python` 是装 IsaacLab 的那个环境。
- **`ModuleNotFoundError: isaaclab_m20`**：没在仓库根目录跑，或者用 `python isaaclab_m20/scripts/train.py` 直接跑脚本。用 `python -m ...`。
- **启动时报 Omniverse/Carb 错误**：必须是 `AppLauncher` 先启动、再 import 任何 `isaaclab` 模块；三个脚本都已经按这个顺序写好了，不要再把 import 往上挪。
- **机器人一出生就塌**：`init_state.pos.z=0.50` 和 USD 里的默认关节角不匹配；用 `--convert` 时再确认 `fix_base=False`。
- **轮子一直空转 / 车子原地打滑**：站立任务检查轮的摩擦材质（`EventCfg.physics_material` 会把摩擦随机到 0.5–1.25）和 `damping`；速度指令任务默认是标称摩擦 1.0，先把 `--dr off`（默认）下的跟踪跑通，再开 `--dr full`。
- **`RuntimeError: CUDA out of memory`**：降 `--num_envs`。
- **headless 录视频报 Vulkan/EGL**：装 `libvulkan1`，并确认驱动正常；纯训练不需要相机，不加 `--video` 即可。
- **训练不收敛**：先看 `Episode_Termination/*` 翻车比例有没有降，再看 `Episode_Reward/base_height`。如果一出生就趴下，把 `reset_joints` 的 `position_range` 调小、`base_motion` 权重调低，先让机器人稳住不动。

## 9. 奖励函数 / 网络 / PPO 流程

奖励公式、Actor-Critic 网络结构、一次 PPO 迭代的完整更新流程见 [docs/algorithm.md](algorithm.md)。

## 10. 调参入口

| 想改的东西 | 文件 |
| --- | --- |
| 奖励权重、终止条件（站立） | `isaaclab_m20/tasks/m20_env_cfg.py` |
| 奖励权重、指令范围（速度指令） | `isaaclab_m20/tasks/m20_velocity_env_cfg.py` |
| 奖励函数本身 | `isaaclab_m20/tasks/rewards.py` |
| PPO 超参（lr、epochs、网络大小） | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
