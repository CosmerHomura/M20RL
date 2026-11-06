# M20 PPO Gait Training

在 MuJoCo 中用 stable-baselines3 的 PPO 训练 M20 轮足机器人追踪速度指令 `(vx, vy, yaw_rate)`。

- 设计文档：`docs/superpowers/specs/2026-11-06-m20-ppo-gait-training-design.md`
- 实现计划：`docs/superpowers/plans/2026-11-06-m20-ppo-gait-training.md`
- 性能基准：`docs/benchmark.md`

## 环境与动作

- 控制频率 50 Hz（`sim_dt=0.002`，10 个子步），每回合上限 500 步（10 s）。
- 观测 60 维：机身系线速度 3 + 角速度 3 + 重力投影 3 + 指令 3 + 关节角 16 + 关节速度 16 + 上一拍动作 16。
- 动作 16 维 ∈ [-1, 1]，顺序 `fl, fr, hl, hr × (hipx, hipy, knee, wheel)`：
  - 腿部：目标角偏移，`q_target = q_default + a × 0.4 rad`，内部 PD（kp=100、kd=5，500 Hz）跟踪后饱和到 ±76.4 N·m。
  - 轮部：直接力矩 `a × 21.6 N·m`。
- 奖励：前向/侧向/偏航追踪、机身水平、高度、省力、动作平滑、零指令静止，外加存活奖励。
- 终止：高度 < 0.25 m、俯仰/横滚 > 0.7 rad、机身碰撞体触地。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

脚本必须以模块方式运行（`python -m scripts.xxx`），否则 `import m20rl` 会失败。

### 检查环境（M0）

```bash
.venv/Scripts/python.exe -m scripts.check_env
```

打开 MuJoCo viewer，用随机动作观察模型、默认站姿与关节正负号。

### 训练

```bash
# M1：只追前向速度
.venv/Scripts/python.exe -m scripts.train --reward-mode forward_only --total-timesteps 5000000 --run-name m1_forward

# M2：完整速度指令追踪
.venv/Scripts/python.exe -m scripts.train --reward-mode full --total-timesteps 20000000 --run-name m2_full
```

常用参数：`--n-envs`、`--device`、`--seed`、`--no-eval`、`--log-root`。
输出目录 `logs/<run-name>/` 含 `tb/`（TensorBoard）、`checkpoints/`、`best_model/`、`final_model.zip`、`final_vecnormalize.pkl`。

```bash
.venv/Scripts/python.exe -m tensorboard.main --logdir logs/m1_forward/tb
```

### 回放与录像

```bash
# 实时可视化
.venv/Scripts/python.exe -m scripts.play --model logs/m1_forward/final_model \
    --vecnormalize logs/m1_forward/final_vecnormalize.pkl --command 0.4 0 0

# 录视频（需要 imageio-ffmpeg）
.venv/Scripts/python.exe -m scripts.play --model logs/m1_forward/final_model \
    --vecnormalize logs/m1_forward/final_vecnormalize.pkl --command 0.4 0 0 \
    --video logs/m1_forward/rollout.mp4
```

`--command VX VY YAW` 在该次回放中固定不变。**必须复用训练时的 `final_vecnormalize.pkl`**，否则观测分布不一致，策略会失效。

### 训练时实时观看

训练环境跑在子进程里，viewer 挂不上去。另开一个终端，用 `watch.py` 盯着 run 目录，它会自动加载最新的 checkpoint（每 100k 步一次）在 viewer 里跑：

```bash
.venv/Scripts/python.exe -m scripts.watch --run logs/m2_full
# 每回合随机指令：
.venv/Scripts/python.exe -m scripts.watch --run logs/m2_full --random-commands
```

### 性能基准

```bash
.venv/Scripts/python.exe -m scripts.benchmark --steps 1000 --out docs/benchmark.md
```

本机（12 逻辑核，CPU-only torch）实测：10 环境约 4100–4500 env-steps/s，训练约 800–2000 train-steps/s。GPU 不会有帮助——MuJoCo 在 CPU 上仿真，策略网络很小，传输开销大于计算收益。
