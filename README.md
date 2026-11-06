# M20 PPO Gait Training

在 MuJoCo 中用 stable-baselines3 的 PPO 训练 M20 轮足机器人追踪速度指令。

- 设计文档：`docs/superpowers/specs/2026-11-06-m20-ppo-gait-training-design.md`
- 实现计划：`docs/superpowers/plans/2026-11-06-m20-ppo-gait-training.md`

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.train --reward-mode forward_only --total-timesteps 5000000
.venv/Scripts/python.exe -m scripts.play --model logs/<run>/final_model --vecnormalize logs/<run>/final_vecnormalize.pkl
```
