"""Watch a training run: replay the newest checkpoint in the MuJoCo viewer.

Training runs in subprocesses, so a viewer cannot attach to those envs.
Instead this script keeps a single viewer open and swaps in each new
checkpoint as the trainer writes it.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from m20rl.config import EnvConfig
from m20rl.env import M20Env


def newest_checkpoint(run_dir):
    checkpoints = list((run_dir / "checkpoints").glob("ppo_*_steps.zip"))
    if checkpoints:
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    final = run_dir / "final_model.zip"
    if final.exists():
        return final
    return None


def vecnormalize_path(checkpoint):
    if checkpoint.name == "final_model.zip":
        return checkpoint.parent / "final_vecnormalize.pkl"
    return checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")


def load_stats(venv, path):
    loaded = VecNormalize.load(str(path), venv.venv)
    venv.obs_rms = loaded.obs_rms
    venv.ret_rms = loaded.ret_rms
    venv.clip_obs = loaded.clip_obs


def sample_command(env):
    return np.array([
        env.np_random.uniform(*env.cfg.vx_range),
        env.np_random.uniform(*env.cfg.vy_range),
        env.np_random.uniform(*env.cfg.yaw_range),
    ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="run directory, e.g. logs/m2_full")
    parser.add_argument("--command", type=float, nargs=3, default=[0.4, 0.0, 0.0])
    parser.add_argument("--random-commands", action="store_true",
                        help="sample a random command every episode")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="seconds between checkpoint checks")
    args = parser.parse_args()

    run_dir = Path(args.run)
    env = M20Env(env_config=EnvConfig(reward_mode="full"), render_mode="human")
    env.cfg.command_resample_steps = 10 ** 9
    venv = VecNormalize(
        DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,
        training=False)

    model = None
    loaded = None
    try:
        while True:
            checkpoint = newest_checkpoint(run_dir)
            if checkpoint is None:
                print(f"waiting for a checkpoint under {run_dir}")
                time.sleep(args.poll)
                continue

            if checkpoint != loaded:
                try:
                    model = PPO.load(str(checkpoint), device="cpu")
                    load_stats(venv, vecnormalize_path(checkpoint))
                except Exception as exc:  # file may still be being written
                    print(f"checkpoint not ready ({exc}), retrying")
                    time.sleep(1.0)
                    continue
                loaded = checkpoint
                print(f"loaded {checkpoint.name}")

            obs = venv.reset()
            command = sample_command(env) if args.random_commands else np.array(
                args.command, dtype=np.float64)
            env.command = command
            obs = np.asarray(
                venv.normalize_obs(env._get_obs()), dtype=np.float32)[None, :]
            print(f"command vx={command[0]:+.2f} vy={command[1]:+.2f} "
                  f"yaw={command[2]:+.2f}")

            for _ in range(env.cfg.episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = venv.step(action)
                env.render()
                time.sleep(1 / env.metadata["render_fps"])
                if dones[0]:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(env, "viewer", None) is not None:
            env.viewer.close()
        venv.close()


if __name__ == "__main__":
    main()
