"""Measure simulation throughput and PPO training speed on this machine."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from m20rl.config import EnvConfig, TrainConfig
from m20rl.env import M20Env


def make_env(rank: int):
    def _thunk():
        return M20Env(env_config=EnvConfig())
    return _thunk


def _vec_env(n_envs: int):
    fns = [make_env(i) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def bench_stepping(n_envs: int, steps: int = 2000) -> float:
    venv = _vec_env(n_envs)
    venv.reset()
    n_actions = venv.action_space.shape[0]
    actions = np.random.uniform(-1, 1, (n_envs, n_actions)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(steps):
        venv.step(actions)
    elapsed = time.perf_counter() - start
    venv.close()
    return (steps * n_envs) / elapsed


def bench_training(device: str, total_timesteps: int = 10_000, n_envs: int = 10) -> float:
    venv = _vec_env(n_envs)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    cfg = TrainConfig()
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        policy_kwargs=cfg.policy_kwargs,
        device=device,
        verbose=0,
        seed=cfg.seed,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - start
    venv.close()
    return total_timesteps / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-timesteps", type=int, default=10_000)
    parser.add_argument("--train-envs", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = ["# M20 benchmark", ""]
    for n_envs in (1, 4, 10):
        fps = bench_stepping(n_envs, steps=args.steps)
        line = f"stepping: n_envs={n_envs:2d} -> {fps:8.1f} env-steps/s"
        print(line)
        lines.append(f"- {line}")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("CUDA not available - skipping GPU benchmark")
        lines.append("- CUDA not available, GPU benchmark skipped")

    for device in devices:
        fps = bench_training(device, args.train_timesteps, args.train_envs)
        line = f"training: device={device} -> {fps:8.1f} train-steps/s"
        print(line)
        lines.append(f"- {line}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written to {out}")


if __name__ == "__main__":
    main()
