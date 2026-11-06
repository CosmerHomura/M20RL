"""Train the M20 wheeled task with rsl_rl PPO in Isaac Lab.

Example::

    python -m isaaclab_m20.scripts.train --headless --num_envs 4096
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Train M20 wheeled locomotion with rsl_rl.")
parser.add_argument("--task", type=str, default="Isaac-M20-Wheeled-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=None, help="Override the runner's iteration count.")
parser.add_argument("--run_name", type=str, default=None, help="Sub-directory inside logs/rsl_rl/<experiment>.")
parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint of --run_name.")
parser.add_argument("--video", action="store_true", help="Record a video of the first training iteration.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in policy steps.")
parser.add_argument("--video_folder", type=str, default=str(REPO_ROOT / "videos"))
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
import yaml
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_m20.tasks  # noqa: F401 - registers Isaac-M20-Wheeled-v0
from isaaclab_m20.tasks.agents.rsl_rl_ppo_cfg import M20RslRlPpoRunnerCfg
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


class _LogDumper(yaml.SafeDumper):
    """Dumper for the params log: stringify values YAML has no type for."""


def _represent_unknown(dumper, data):
    return dumper.represent_str(str(data))


# SceneEntityCfg defaults to joint_ids=slice(None), which plain SafeDumper rejects.
_LogDumper.add_multi_representer(slice, _represent_unknown)
_LogDumper.add_multi_representer(object, _represent_unknown)


def main() -> None:
    entry_point = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry_point.split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed
    fit_viewer_to_grid(env_cfg)

    agent_cfg = M20RslRlPpoRunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_root_path = os.path.join(args_cli.log_root, agent_cfg.experiment_name)
    if args_cli.run_name:
        log_dir = os.path.join(log_root_path, args_cli.run_name)
    else:
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to {log_dir}")
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    with open(os.path.join(log_dir, "params", "env.yaml"), "w") as f:
        yaml.dump(env_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "w") as f:
        yaml.dump(agent_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)
    print(f"[INFO] Environment: {args_cli.task}  num_envs={env_cfg.scene.num_envs}")
    print_dict(agent_cfg.to_dict(), nesting=0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video of the first iteration.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Action space: {env.action_space}  Observation space: {env.observation_space}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume and args_cli.run_name:
        checkpoints = sorted(Path(log_dir, "checkpoints").glob("model_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {Path(log_dir, 'checkpoints')}")
        resume_path = str(checkpoints[-1])
        print(f"[INFO] Resuming from {resume_path}")
        runner.load(resume_path)

    start = time.time()
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    print(f"[INFO] Training finished in {(time.time() - start) / 60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
