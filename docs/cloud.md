# 云端（Linux + GPU）跑训练

本机开发环境是 Windows；云端一般是 Linux。代码本身已用 `pathlib` 和模块化入口，换平台只需改命令写法。

## 先说结论：4090 基本帮不上忙

MuJoCo 物理仿真跑在 **CPU** 上，PPO 的策略网络只有约 8 万参数。把 PPO 放到 GPU 上，每步的 CPU↔GPU 传输开销通常大于计算收益，实测往往更慢。租 4090 的真正价值是**附带的那台机器通常有更多 CPU 核**——并行仿真环境数可以直接拉高。

**先在云上跑基准，用数据决定：**

```bash
python -m scripts.benchmark --steps 1000 --out docs/benchmark.md
```

输出里如果出现 `training: device=cuda` 行，就和 `device=cpu` 比较，哪个快用哪个；`n_envs` 取 `CPU 核数 - 2`。

## 1. 上传代码

代码都在 git 里，最省事的是打包当前提交（不含 `.venv/`、`logs/`）：

```bash
# 本地
git archive --format=zip -o m20rl.zip HEAD
scp m20rl.zip user@host:~/

# 云端
unzip m20rl.zip -d M20RL && cd M20RL
```

如果云端机器能访问 GitHub，也可以建远端仓库后 `git clone`。

## 2. 安装

```bash
cd M20RL
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # 应为 True
```

Linux 上 PyPI 的 `torch` 默认带 CUDA 12 wheel，`requirements.txt` 不需要改。若显存/驱动不匹配，按 PyTorch 官网的命令重装 torch。

## 3. 训练

```bash
# 先确认基准，再定 n_envs / device
python -m scripts.benchmark --steps 1000

# M1（1~2 小时，视核数）
python -m scripts.train --reward-mode forward_only --total-timesteps 5000000 \
    --n-envs 14 --device cpu --run-name m1_forward

# M2 完整速度指令
python -m scripts.train --reward-mode full --total-timesteps 20000000 \
    --n-envs 14 --device cpu --run-name m2_full
```

注意是 `--total-timesteps`（连字符）。`n_envs` 从 10 提到 14 左右通常能线性提速；超过物理核数收益会下降。

**建议用 `tmux` / `nohup` 跑**，避免 SSH 断连杀掉训练：

```bash
tmux new -s train
# Ctrl-b d 脱离，tmux attach -t train 回来
```

TensorBoard 在云端看：

```bash
python -m tensorboard.main --logdir logs/m2_full/tb --bind_all --port 6006
# 本地 ssh -L 6006:localhost:6006 user@host，然后浏览器开 localhost:6006
```

## 4. 无显示器环境的注意事项

训练本身不需要渲染。只有回放/录像需要 OpenGL：

- `scripts.play --render-mode none` 可以直接跑，不需要显示。
- 录像（`--video`）在无头服务器上要指定 EGL：

  ```bash
  export MUJOCO_GL=egl
  sudo apt install -y libegl1 libgl1     # 若缺库
  python -m scripts.play ... --video out.mp4
  ```

- `scripts.watch.py` 需要真实窗口，云端无头环境用不了，留到本地看。

## 5. 取回结果

训练产物都在 `logs/<run-name>/`：

```bash
# 只取最关键的
scp user@host:~/M20RL/logs/m2_full/final_model.zip ./
scp user@host:~/M20RL/logs/m2_full/final_vecnormalize.pkl ./

# 或整个 run（含 TensorBoard 日志和视频）
scp -r user@host:~/M20RL/logs/m2_full ./
```

本地回放时**必须同时用 `final_model.zip` 和 `final_vecnormalize.pkl`**，否则观测分布不一致。

## 6. 常见坑

| 现象 | 处理 |
|---|---|
| `No module named m20rl` | 必须用 `python -m scripts.xxx`，并在仓库根目录执行 |
| 训练很慢 | 先看 `time/fps`；提高 `--n-envs`，确认没用 `--device cuda` 拖后腿 |
| 子进程启动失败 | Linux 默认 `fork`，一般没问题；仍保留 `if __name__ == "__main__"` 保护 |
| 显存被别的任务占用 | 本项目训练本身几乎不用显存，`--device cpu` 即可 |
