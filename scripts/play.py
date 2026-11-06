"""Roll out a trained PPO policy on the M20 model."""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from m20rl.config import EnvConfig
from m20rl.env import M20Env


def apply_command(venv, env, command):
    """Pin the command and return the observation matching it.

    ``reset`` samples a random command, so the observation it returns is stale
    once we override the command. Rebuild it from the env state instead.
    """
    env.command = np.asarray(command, dtype=np.float64)
    obs = venv.normalize_obs(env._get_obs())
    return np.asarray(obs, dtype=np.float32)[None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"],
                        default="human")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    render_mode = args.render_mode
    if args.video and render_mode != "rgb_array":
        render_mode = "rgb_array"

    env = M20Env(env_config=EnvConfig(reward_mode="full"), render_mode=render_mode)
    env.cfg.command_resample_steps = 10 ** 9
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(args.vecnormalize, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model, device="cpu")

    venv.reset()
    obs = apply_command(venv, env, args.command)

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=50)

    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)
            frame = env.render()
            if writer is not None and frame is not None:
                writer.append_data(frame)
            if render_mode == "human":
                time.sleep(1 / 50)
            if dones[0]:
                venv.reset()
                obs = apply_command(venv, env, args.command)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
        venv.close()


if __name__ == "__main__":
    main()
