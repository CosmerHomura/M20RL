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
parser.add_argument(
    "--zero_actions", action="store_true",
    help="Hold the default pose instead of training. Isolation test: if this still "
         "crashes, the fault is in the model/sim, not in the policy.")
parser.add_argument(
    "--health_every", type=int, default=24,
    help="Print sim-health stats every N env steps (0 disables).")
parser.add_argument(
    "--dr_ramp", type=float, default=0.4,
    help="Fraction of training over which the domain randomization ramps "
         "from 0 (nominal robot) to 1 (full range). 0 disables the curriculum "
         "and applies the full range from the first step.")
parser.add_argument(
    "--cmd_stages", action=argparse.BooleanOptionalAction, default=True,
    help="Velocity tasks only: walk the command range through the staged "
         "schedule (stand -> low -> mid -> high -> full). --no-cmd_stages "
         "samples the full range from the first step.")
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
import yaml
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_m20.tasks  # noqa: F401 - registers both M20 task ids
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


class _LogDumper(yaml.SafeDumper):
    """Dumper for the params log: stringify values YAML has no type for."""


def _represent_unknown(dumper, data):
    return dumper.represent_str(str(data))


# SceneEntityCfg defaults to joint_ids=slice(None), which plain SafeDumper rejects.
_LogDumper.add_multi_representer(slice, _represent_unknown)
_LogDumper.add_multi_representer(object, _represent_unknown)


def attach_diagnostics(env, every: int, zero_actions: bool) -> None:
    """Log simulation health, optionally replacing the policy output with zeros.

    PhysX reports a divergence as a bare ``CUDA error 700`` (illegal memory
    access) with no hint about which quantity blew up, and the CUDA context is
    dead by the time it surfaces. Printing the worst-case joint speed and base
    height every iteration makes the growth visible a few seconds before the
    crash instead of after it.
    """
    unwrapped = env.unwrapped
    original_step = env.step
    state = {"n": 0}

    def step(action):
        if zero_actions:
            action = torch.zeros_like(action)
        out = original_step(action)
        state["n"] += 1
        if every > 0 and state["n"] % every == 0:
            robot = unwrapped.scene["robot"]
            joint_vel = robot.data.joint_vel
            root_z = robot.data.root_pos_w[:, 2]
            root_vel = robot.data.root_lin_vel_w.abs().max()
            print(
                f"[health] step={state['n']:8d}"
                f"  max|qdot|={joint_vel.abs().max().item():10.2f}"
                f"  max|v|={root_vel.item():8.2f}"
                f"  z=[{root_z.min().item():.3f},{root_z.max().item():.3f}]"
                f"  finite={torch.isfinite(joint_vel).all().item()}",
                flush=True,
            )
        return out

    env.step = step


# Command-range stages: (end of the stage as a fraction of training, vx, vy, wz).
# The robot learns to stand, then to hold a low, medium and high speed, and
# finally sees the full range. Values are half-widths in m/s and rad/s.
#
# Staged rather than a linear ramp because a ramp spends most of its iterations
# in the slow band, where there is nothing left to learn, and reaches the
# interesting 2-3 m/s regime only near the end. Stage 0 is deliberately short:
# it exists to settle the stance and the base-height/posture terms, not to
# train a standing policy from scratch (the standing task already did that).
COMMAND_STAGES: tuple[tuple[float, float, float, float], ...] = (
    (0.05, 0.0, 0.00, 0.0),   # 站立
    (0.25, 0.6, 0.15, 0.3),   # 低速
    (0.50, 1.5, 0.35, 0.6),   # 中速
    (0.75, 2.2, 0.50, 0.8),   # 高速
    (1.00, 3.0, 0.60, 1.0),   # 满幅
)


def command_stage(progress: float) -> int:
    """Index of the stage covering ``progress`` in [0, 1)."""
    for index, (end, *_) in enumerate(COMMAND_STAGES):
        if progress < end:
            return index
    return len(COMMAND_STAGES) - 1


def attach_curriculum(env, max_iterations: int, num_steps_per_env: int,
                      ramp_fraction: float, cmd_stages: bool) -> None:
    """Schedule the domain randomization and the command range.

    Randomization: a linear ramp from 0 to full. Full randomization from the
    first step means the policy is asked to survive pushes and payload changes
    before it has learned to stand at all -- survival typically rises for a
    while and then collapses. Starting at 0 and ramping keeps the task in the
    learnable band: the nominal robot first, the distribution later.

    Commands: the staged table above, so that the tracking task widens at the
    pace the policy can absorb. Sampling +-3 m/s from step 0 asks for a reward
    density the untrained policy cannot reach anywhere, while a linear ramp
    spends most of its iterations in the slow band where there is nothing left
    to learn. The short standing stage exists to establish the stance and the
    base-height/posture terms before any command asks for motion.
    """
    from isaaclab_m20.tasks import dr

    diag_every = 50
    ramp_iterations = max(int(max_iterations * ramp_fraction), 1)
    plain_dr = ramp_fraction <= 0.0

    command_term = None
    if cmd_stages:
        manager = getattr(env.unwrapped, "command_manager", None)
        if manager is not None and "base_velocity" in manager.active_terms:
            command_term = manager.get_term("base_velocity")
            dr.register_command_term(command_term)
            _, vx, vy, wz = COMMAND_STAGES[0]
            dr.set_command_ranges(vx, vy, wz)
            # The commands drawn at scene construction used the full range and
            # the first resample is 4 s away; start the robot from standing.
            dr.zero_command(command_term)
            bounds = ", ".join(
                f"{i}:{end:.0%} vx+-{vx:g}/vy+-{vy:g}/wz+-{wz:g}"
                for i, (end, vx, vy, wz) in enumerate(COMMAND_STAGES))
            print(f"[INFO] Command stages (fraction of training): {bounds}")

    if plain_dr:
        dr.set_level(1.0)
        print("[INFO] --dr_ramp 0: full domain randomization from step 0.")
    else:
        print(f"[INFO] Domain randomization ramps 0 -> 1 over {ramp_iterations} iterations.")

    original_step = env.step
    state = {"n": 0, "stage": -1}

    def step(action):
        out = original_step(action)
        state["n"] += 1
        if state["n"] % num_steps_per_env == 0:
            iteration = state["n"] // num_steps_per_env
            level = 1.0 if plain_dr else dr.set_level(iteration / ramp_iterations)
            if command_term is not None:
                stage = command_stage(iteration / max(max_iterations, 1))
                if stage != state["stage"]:
                    _, vx, vy, wz = COMMAND_STAGES[stage]
                    dr.set_command_ranges(vx, vy, wz)
                    state["stage"] = stage
                    print(f"[cmd] iteration={iteration:5d} stage={stage} "
                          f"vx=+-{vx:g} vy=+-{vy:g} wz=+-{wz:g}", flush=True)
            if iteration % diag_every == 0:
                print(f"[dr] iteration={iteration:5d} level={level:.2f}", flush=True)
        return out

    env.step = step


def main() -> None:
    entry_point = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry_point.split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed
    fit_viewer_to_grid(env_cfg)

    agent_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    module_name, class_name = agent_entry.split(":")
    agent_cfg = getattr(importlib.import_module(module_name), class_name)()
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
    if args_cli.zero_actions:
        print("[INFO] --zero_actions: holding the default pose, actions are forced to 0.")
    attach_curriculum(env, agent_cfg.max_iterations, agent_cfg.num_steps_per_env,
                      args_cli.dr_ramp, args_cli.cmd_stages)
    attach_diagnostics(env, args_cli.health_every, args_cli.zero_actions)

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
