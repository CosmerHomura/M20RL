# Isaac Lab 训练（云端 4090）

训练在 Isaac Lab + rsl_rl 上跑，重点利用**并行环境数**：4090 上可以开 4096 个环境。

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
├── tasks/rewards.py           # 奖励项函数
├── tasks/m20_env_cfg.py       # 场景、观测、动作、指令、奖励、终止、事件
├── tasks/__init__.py          # gym.register("Isaac-M20-Wheeled-v0")
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
| 显存 | ≥ 16 GB（4090 24 GB，开 4096 环境没问题） |
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
python -m isaaclab_m20.scripts.train --headless --num_envs 4096
```

- 每个 iteration 采集 4096 × 24 = 98,304 步；默认跑 2000 iterations（约 2 亿步）。
- 日志：`logs/rsl_rl/m20_wheeled/<时间戳>/`，每 100 iterations 存一个 `checkpoints/model_*.pt`。
- 想给这次实验起名：`--run_name v1`，日志就落在 `logs/rsl_rl/m20_wheeled/v1/`。
- 显存不够就降到 `--num_envs 2048`；这个场景很轻，一般不会不够。
- 断点续训：`--resume --run_name v1`（加载该目录下最新的 checkpoint）。

粗略预期（4090，单场景，无相机）：每秒几万步，2000 iterations 大概几小时量级。

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
# 最新 checkpoint，16 个环境
python -m isaaclab_m20.scripts.play --num_envs 16

# 指定 checkpoint
python -m isaaclab_m20.scripts.play \
    --checkpoint logs/rsl_rl/m20_wheeled/v1/checkpoints/model_2000.pt \
    --num_envs 16

# 无界面录制视频
python -m isaaclab_m20.scripts.play --checkpoint <path> --headless --video
```

`play.py` 加载 checkpoint 后就地站立推演，视频写到 `videos/`，`--video_length` 是策略步数（50 Hz，500 步 = 10 s）。

## 7. 控制与观测速查

**当前任务是静态站立**：无速度指令，目标原地站稳、保持高度 0.50 m、机身水平、不漂移。PPO 输出关节**位置目标**，PD 伺服环在 200 Hz 跟踪成力矩。

观测 57 维：base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(16) + joint_vel(16) + last_action(16)。

| 项 | 配置 |
| --- | --- |
| 控制频率 | 50 Hz（`sim.dt=0.005` × `decimation=4`） |
| 腿动作 | `JointPositionActionCfg(scale=0.25, use_default_offset=True)` |
| 轮动作 | `JointVelocityActionCfg(scale=20)`，`use_default_offset=False` |
| PD 伺服 | 腿 kp=80 / kd=2，轮 kp=0 / kd=2，200 Hz |
| 力矩上限 | 腿 76.4 N·m，轮 21.6 N·m |
| 奖励 | 共 14 项，完整权重表见 [docs/algorithm.md](algorithm.md) 第 1 节 |
| 终止 | 高度 < 0.25、倾角 > 0.7、超时 |

## 8. 常见问题

- **`ModuleNotFoundError: No module named 'isaaclab'`**：用了别的 python。确认 `which python` 是装 IsaacLab 的那个环境。
- **`ModuleNotFoundError: isaaclab_m20`**：没在仓库根目录跑，或者用 `python isaaclab_m20/scripts/train.py` 直接跑脚本。用 `python -m ...`。
- **启动时报 Omniverse/Carb 错误**：必须是 `AppLauncher` 先启动、再 import 任何 `isaaclab` 模块；三个脚本都已经按这个顺序写好了，不要再把 import 往上挪。
- **机器人一出生就塌**：`init_state.pos.z=0.50` 和 USD 里的默认关节角不匹配；用 `--convert` 时再确认 `fix_base=False`。
- **轮子一直空转 / 车子原地打滑**：检查轮的摩擦材质（`EventCfg.physics_material` 会把摩擦随机到 0.7–1.2）和 `damping`。
- **`RuntimeError: CUDA out of memory`**：降 `--num_envs`。
- **headless 录视频报 Vulkan/EGL**：装 `libvulkan1`，并确认驱动正常；纯训练不需要相机，不加 `--video` 即可。
- **训练不收敛**：先看 `Episode_Termination/*` 翻车比例有没有降，再看 `Episode_Reward/base_height`。如果一出生就趴下，把 `reset_joints` 的 `position_range` 调小、`base_motion` 权重调低，先让机器人稳住不动。

## 9. 奖励函数 / 网络 / PPO 流程

奖励公式、Actor-Critic 网络结构、一次 PPO 迭代的完整更新流程见 [docs/algorithm.md](algorithm.md)。

## 10. 调参入口

| 想改的东西 | 文件 |
| --- | --- |
| 奖励权重、指令范围、终止条件 | `isaaclab_m20/tasks/m20_env_cfg.py` |
| 奖励函数本身 | `isaaclab_m20/tasks/rewards.py` |
| PPO 超参（lr、epochs、网络大小） | `isaaclab_m20/tasks/agents/rsl_rl_ppo_cfg.py` |
| PD 增益、力矩上限、默认姿态 | `isaaclab_m20/assets/m20.py` |
