"""Play a trained M20 static-standing policy.

Examples::

    python -m isaaclab_m20.scripts.play --checkpoint logs/rsl_rl/m20_wheeled/<run>/checkpoints/model_2000.pt
    python -m isaaclab_m20.scripts.play --checkpoint <path> --headless --video
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Play a trained M20 policy.")
parser.add_argument("--task", type=str, default="Isaac-M20-Wheeled-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model_*.pt. Defaults to the newest checkpoint under --log_root.")
parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
parser.add_argument("--video", action="store_true", help="Record a video instead of opening the viewer.")
parser.add_argument("--video_length", type=int, default=500, help="Video length in policy steps.")
parser.add_argument("--video_folder", type=str, default=str(REPO_ROOT / "videos"))
parser.add_argument("--real_time", action="store_true", help="Throttle the loop to the simulated step time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Everything below needs the Omniverse app running.
# --------------------------------------------------------------------------- #
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_m20.tasks  # noqa: F401 - registers Isaac-M20-Wheeled-v0
from isaaclab_m20.tasks.agents.rsl_rl_ppo_cfg import M20RslRlPpoRunnerCfg
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


def resolve_checkpoint() -> Path:
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted(Path(args_cli.log_root).rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt under {args_cli.log_root}; pass --checkpoint.")
    return candidates[-1]


def main() -> None:
    entry_point = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry_point.split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    fit_viewer_to_grid(env_cfg)

    checkpoint = resolve_checkpoint()
    print(f"[INFO] Loading {checkpoint}")

    agent_cfg = M20RslRlPpoRunnerCfg()
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="m20_stand",
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Wrote video to {args_cli.video_folder}")
            break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
