"""Play a trained M20 policy.

``--task`` picks the task and, unless ``--checkpoint`` is given, the newest
checkpoint under ``logs/rsl_rl/<experiment>/`` is loaded automatically -- the
two tasks log to different directories and their checkpoints are not
interchangeable, so this never mixes them up.

Examples::

    # static standing: default task, newest m20_wheeled checkpoint
    python -m isaaclab_m20.scripts.play

    # velocity command: steer with the keyboard (W/S/A/D/Q/E, Space)
    python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0

    # velocity command, pinned checkpoint and fixed command (headless video)
    python -m isaaclab_m20.scripts.play --task Isaac-M20-Wheeled-Velocity-v0 \
        --checkpoint logs/rsl_rl/m20_velocity/<run>/checkpoints/model_*.pt \
        --cmd_vx 0.6 --cmd_wz 0.3 --headless --video
"""
from __future__ import annotations

import argparse
import importlib
import math
import random
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
                    help="Path to model_*.pt. Defaults to the newest checkpoint under "
                         "<log_root>/<experiment>, where <experiment> is the task's log dir.")
parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
parser.add_argument("--video", action="store_true", help="Record a video instead of opening the viewer.")
parser.add_argument("--video_length", type=int, default=500, help="Video length in policy steps.")
parser.add_argument("--video_folder", type=str, default=str(REPO_ROOT / "videos"))
parser.add_argument(
    "--real_time", action=argparse.BooleanOptionalAction, default=None,
    help="Throttle the loop to the simulated step time. Defaults to on with a "
         "viewer, off when headless.")
parser.add_argument(
    "--push_speed", type=float, default=1.0,
    help="Horizontal velocity impulse in m/s applied by the push key / auto-push.")
parser.add_argument(
    "--push_every", type=float, default=0.0,
    help="Apply a push every N seconds of simulated time (0 disables). Useful "
         "for videos; with a viewer you can also press P by hand.")
parser.add_argument("--cmd_vx", type=float, default=0.0, help="Constant forward command, m/s.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Constant lateral command, m/s.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Constant yaw-rate command, rad/s.")
parser.add_argument(
    "--cmd_step", type=float, default=0.25,
    help="Amount W/S, A/D, Q/E add to the command per key press.")
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

import isaaclab_m20.tasks  # noqa: F401 - registers both M20 task ids
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


def resolve_checkpoint(experiment_name: str) -> Path:
    """Return --checkpoint, else the newest model under <log_root>/<experiment_name>.

    The search is scoped to the task's experiment directory: standing and
    velocity log to ``m20_wheeled/`` and ``m20_velocity/`` respectively, and
    their checkpoints are not interchangeable (57 vs 56 observation dims). A
    flat "newest model anywhere under log_root" would hand a standing checkpoint
    to the velocity env and crash on a shape mismatch.

    The velocity run nests its checkpoints one level deeper, one directory per
    curriculum stage (``<run>/stage5_mixed/model_1250.pt``), so the search is
    recursive and ordered by modification time: the newest stage wins.
    """
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    run_dir = Path(args_cli.log_root) / experiment_name
    candidates = sorted(run_dir.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt under {run_dir}; pass --checkpoint.")
    return candidates[-1]


def shove(env, speed: float) -> None:
    """Instantaneous horizontal velocity change -- the same disturbance training used."""
    robot = env.unwrapped.scene["robot"]
    angle = random.uniform(0.0, 2.0 * math.pi)
    root_vel = robot.data.root_vel_w.clone()
    root_vel[:, 0] += speed * math.cos(angle)
    root_vel[:, 1] += speed * math.sin(angle)
    robot.write_root_velocity_to_sim(root_vel)


class KeyboardControls:
    """Viewer controls. Needs a focused viewer window.

    P push the robot        R reset every environment
    W/S vx +/-              A/D vy +/-          Q/E wz +/-
    Space zero the command  (command keys only when the task has commands)
    """

    def __init__(self, speed: float, command_step: float, steer: bool):
        import carb.input
        import omni.appwindow

        self._ki = carb.input
        self.speed = speed
        self.command_step = command_step
        self.steer = steer
        self.pending_push = False
        self.reset_requested = False
        self.command = [0.0, 0.0, 0.0]
        self.command_changed = False

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        if event.type != self._ki.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input
        K = self._ki.KeyboardInput
        if key == K.P:
            self.pending_push = True
            return True
        if key == K.R:
            self.reset_requested = True
            return True
        if not self.steer:
            return True
        steps = {K.W: (0, 1), K.S: (0, -1), K.A: (1, 1), K.D: (1, -1),
                 K.Q: (2, 1), K.E: (2, -1)}
        if key in steps:
            axis, sign = steps[key]
            self.command[axis] += sign * self.command_step
            self.command_changed = True
        elif key == K.SPACE:
            self.command = [0.0, 0.0, 0.0]
            self.command_changed = True
        return True

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)


def write_command(command_term, command) -> None:
    """Overwrite the (vx, vy, wz) columns of the command buffer.

    The velocity command buffer is 4 columns -- index 3 is the heading target,
    unused with ``heading_command=False`` -- so only the first three are written.
    """
    buffer = getattr(command_term, "vel_command_b", None)
    if buffer is None:
        buffer = command_term.command
    buffer[:, : len(command)] = torch.tensor(
        command, device=buffer.device, dtype=buffer.dtype)


def main() -> None:
    entry_point = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry_point.split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    fit_viewer_to_grid(env_cfg)
    has_command = getattr(env_cfg, "commands", None) is not None

    agent_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    module_name, class_name = agent_entry.split(":")
    agent_cfg = getattr(importlib.import_module(module_name), class_name)()
    agent_cfg.device = args_cli.device

    checkpoint = resolve_checkpoint(agent_cfg.experiment_name)
    print(f"[INFO] Task={args_cli.task}  experiment={agent_cfg.experiment_name}")
    print(f"[INFO] Loading {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="m20_velocity" if has_command else "m20_stand",
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    real_time = args_cli.real_time if args_cli.real_time is not None else not args_cli.headless
    push_period = int(round(args_cli.push_every / dt)) if args_cli.push_every > 0 else 0

    command_term = None
    if has_command:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        ranges = command_term.cfg.ranges
        print(f"[INFO] Command ranges: vx {ranges.lin_vel_x}  vy {ranges.lin_vel_y}  "
              f"wz {ranges.ang_vel_z}")
        # Training resamples the command every 4 s. Stop it here so the operator
        # owns the command; the per-step write below is the backstop.
        try:
            command_term.cfg.resampling_time_range = (1.0e6, 1.0e6)
            command_term.time_left.fill_(1.0e6)
        except Exception:
            pass

    keyboard = None
    if not args_cli.headless:
        try:
            keyboard = KeyboardControls(args_cli.push_speed, args_cli.cmd_step, steer=has_command)
            print("[INFO] Viewer controls:  P = push the robot   R = reset")
            if has_command:
                print("[INFO] Steer:  W/S = vx   A/D = vy   Q/E = yaw   Space = zero")
                keyboard.command = [args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz]
        except Exception as exc:  # no window or no input interface available
            print(f"[WARN] Keyboard control unavailable: {exc}")
    if push_period:
        print(f"[INFO] Auto-push every {args_cli.push_every:g} s at {args_cli.push_speed:g} m/s")

    pending_command = None
    if has_command and keyboard is None:
        pending_command = [args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz]
        print(f"[INFO] Fixed command: vx={pending_command[0]:g}  vy={pending_command[1]:g}  "
              f"wz={pending_command[2]:g}")

    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        if command_term is not None:
            if keyboard is not None:
                write_command(command_term, keyboard.command)
                if keyboard.command_changed:
                    print(f"[INFO] command -> vx={keyboard.command[0]:+.2f}  "
                          f"vy={keyboard.command[1]:+.2f}  wz={keyboard.command[2]:+.2f}")
                    keyboard.command_changed = False
            else:
                write_command(command_term, pending_command)
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        timestep += 1

        if push_period and timestep % push_period == 0:
            shove(env, args_cli.push_speed)
        if keyboard is not None:
            if keyboard.pending_push:
                shove(env, args_cli.push_speed)
                keyboard.pending_push = False
            if keyboard.reset_requested:
                env.unwrapped.reset()
                obs = env.get_observations()
                keyboard.reset_requested = False

        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Wrote video to {args_cli.video_folder}")
            break
        if real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if keyboard is not None:
        keyboard.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
