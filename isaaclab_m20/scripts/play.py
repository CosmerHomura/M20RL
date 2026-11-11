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

# Stage directory suffixes, in course order: <run>/stage3_multi_direction/ ...
# Kept here as plain data so play.py never imports train.py (which builds an
# argparse parser and starts the Isaac app at import time).
STAGE_SLUGS = ("stand", "forward_backward", "multi_direction", "agile")

parser = argparse.ArgumentParser(description="Play a trained M20 policy.")
parser.add_argument("--task", type=str, default=None,
                    help="Gym task id. Defaults to the velocity task when --stage is given "
                         "(stages only exist there), otherwise the standing task.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model_*.pt, a stage directory, or a run directory. "
                         "Defaults to the newest checkpoint under "
                         "<log_root>/<experiment>, where <experiment> is the task's log dir.")
parser.add_argument("--stage", type=int, default=None,
                    choices=range(1, len(STAGE_SLUGS) + 1), metavar="{1..4}",
                    help="Play one curriculum stage instead of the newest model: takes the "
                         "highest model_*.pt in <log_root>/<experiment>/<newest run>/stageN_*. "
                         "e.g. --stage 2 to look at the forward/backward stage.")
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
parser.add_argument(
    "--status_every", type=int, default=50,
    help="Print one status line every N policy steps (0 disables). 50 steps = 1 s "
         "at 50 Hz. The line shows the command buffer, |v_xy|, height, tilt and the "
         "worst leg/wheel joint speed -- enough to tell 'command not reaching the "
         "robot' from 'wheels spinning without grip'.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None:
    # --stage only exists in the velocity task's curriculum, so it selects the
    # task too instead of failing on a missing m20_velocity/ stage directory.
    args_cli.task = ("Isaac-M20-Wheeled-Velocity-v0" if args_cli.stage is not None
                     else "Isaac-M20-Wheeled-v0")
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
from isaaclab_m20.tasks.commands import freeze_command_resampling
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


def iterations(path: Path) -> int:
    """Iteration number encoded in a model_<n>.pt name (-1 if it has none)."""
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def newest_run(experiment_dir: Path) -> Path:
    """Most recently written run directory under <log_root>/<experiment>."""
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()] \
        if experiment_dir.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"No run directories under {experiment_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def resolve_stage_checkpoint(experiment_name: str, stage: int) -> Path:
    """Checkpoint of one curriculum stage of the newest run.

    ``--stage 3`` means "<newest run>/stage3_multi_direction/model_<highest>.pt",
    which is the model that stage ended on -- the loop is "train a stage, look at
    it, decide", and looking at it should not require reading the directory
    listing to find out which iteration number the stage stopped at.
    """
    experiment_dir = Path(args_cli.log_root) / experiment_name
    run_dir = newest_run(experiment_dir)
    if stage < 1 or stage > len(STAGE_SLUGS):
        raise ValueError(f"--stage must be 1..{len(STAGE_SLUGS)}, got {stage}")
    stage_dir = run_dir / f"stage{stage}_{STAGE_SLUGS[stage - 1]}"
    if not stage_dir.is_dir():
        present = sorted(path.name for path in run_dir.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"No stage {stage} ({stage_dir.name}) in {run_dir}. Present: "
            f"{', '.join(present) or 'none'}. --stage only applies to the velocity "
            f"task's curriculum; train that stage first or pass --checkpoint.")
    candidates = list(stage_dir.rglob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt in {stage_dir}; train that stage first.")
    return max(candidates, key=iterations)


def resolve_checkpoint(experiment_name: str) -> Path:
    """Return --checkpoint, else --stage, else the newest model of the task.

    The search is scoped to the task's experiment directory: standing and
    velocity log to ``m20_wheeled/`` and ``m20_velocity/`` respectively, and
    their checkpoints are not interchangeable (57 vs 56 observation dims). A
    flat "newest model anywhere under log_root" would hand a standing checkpoint
    to the velocity env and crash on a shape mismatch.

    The velocity run nests its checkpoints one level deeper, one directory per
    curriculum stage (``<run>/stage4_agile/model_4200.pt``), so the search is
    recursive and ordered by modification time: the newest stage wins. The
    iteration number in those names is cumulative over the whole run, not the
    stage's own budget -- stage 4 ends around 4200, not 1400.

    ``--checkpoint`` may also be a directory (a stage directory or the run
    directory), in which case its highest iteration number is used.
    """
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint).expanduser()
        if path.is_dir():
            candidates = list(path.rglob("model_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No model_*.pt under {path}")
            return max(candidates, key=iterations)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if args_cli.stage is not None:
        return resolve_stage_checkpoint(experiment_name, args_cli.stage)
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


def status_line(env, command_term, wheel_ids, timestep: int, obs=None) -> str:
    """One line that answers "is the command reaching the robot, and is it moving".

    Everything here is read-only and env-averaged, so it can run while the robot
    moves. The four numbers that matter when a command seems ignored:
    ``standing`` (1.00 = the command buffer is still zero, so the parking brake is
    also still holding the wheels), ``|v_xy|`` (actual speed), ``qdot_wheel``
    (wheels spinning with no speed = they are not loaded) and ``qdot_leg``.

    ``obs_cmd`` is the command as the *policy* sees it, taken out of the policy
    observation tensor (columns 37:40). If ``cmd=`` and ``obs_cmd=`` ever disagree,
    something between the buffer and the observation is rewriting or mapping it.
    """
    unwrapped = env.unwrapped
    data = unwrapped.scene["robot"].data
    speed = float(data.root_lin_vel_b[:, :2].norm(dim=1).mean())
    wz = float(data.root_ang_vel_b[:, 2].mean())
    height = float(data.root_pos_w[:, 2].mean())
    tilt = float(
        torch.rad2deg(torch.acos(torch.clamp(-data.projected_gravity_b[:, 2], -1.0, 1.0))).mean()
    )

    qdot = data.joint_vel.abs()
    if wheel_ids is not None and len(wheel_ids) > 0:
        is_wheel = torch.zeros(qdot.shape[1], dtype=torch.bool, device=qdot.device)
        is_wheel[wheel_ids] = True
        qdot_wheel = float(qdot[:, is_wheel].max()) if bool(is_wheel.any()) else 0.0
        qdot_leg = float(qdot[:, ~is_wheel].max()) if bool((~is_wheel).any()) else 0.0
    else:
        qdot_wheel = 0.0
        qdot_leg = float(qdot.max())

    command = ""
    standing = "n/a"
    if command_term is not None:
        buffer = getattr(command_term, "vel_command_b", None)
        if buffer is not None:
            vx, vy, wz_cmd = (float(value) for value in buffer[0, :3])
            command = f"cmd=({vx:+.2f},{vy:+.2f},{wz_cmd:+.2f}) "
            zero = (buffer[:, :3].abs() <= 1e-3).all(dim=1).float().mean()
            standing = f"{float(zero):.2f}"

    observed = ""
    policy_obs = obs
    if isinstance(policy_obs, dict):
        policy_obs = policy_obs.get("policy")
    if policy_obs is not None and policy_obs.ndim == 2 and policy_obs.shape[1] >= 40:
        ox, oy, ow = (float(value) for value in policy_obs[0, 37:40])
        observed = f"obs_cmd=({ox:+.2f},{oy:+.2f},{ow:+.2f}) "

    seconds = timestep * unwrapped.step_dt
    return (f"[status] t={seconds:5.1f}s {command}{observed}|v_xy|={speed:.3f} wz={wz:+.2f} "
            f"h={height:.3f} tilt={tilt:4.1f}deg qdot_leg={qdot_leg:5.1f} "
            f"qdot_wheel={qdot_wheel:6.2f} standing={standing}")


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
    if args_cli.stage is not None:
        print(f"[INFO] Stage {args_cli.stage} ({STAGE_SLUGS[args_cli.stage - 1]}) of the newest run")
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
    try:
        sliced = obs["policy"] if isinstance(obs, dict) else obs
        obs_dim = int(sliced.shape[-1])
        if sliced.ndim == 2 and sliced.shape[1] >= 40:
            # A startup dump of the command the policy actually receives: if this
            # does not track the value printed by [INFO] command, the command is
            # not reaching the observation and nothing else matters.
            print(f"[INFO] Policy obs: {obs_dim} dims, command slice obs[37:40] = "
                  f"{[round(float(v), 3) for v in sliced[0, 37:40]]}")
        else:
            print(f"[INFO] Policy obs: {obs_dim} dims (no command slice)")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[WARN] Could not read the policy obs shape: {exc}")
    dt = env.unwrapped.step_dt
    real_time = args_cli.real_time if args_cli.real_time is not None else not args_cli.headless
    push_period = int(round(args_cli.push_every / dt)) if args_cli.push_every > 0 else 0

    command_term = None
    if has_command:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        ranges = command_term.cfg.ranges
        print(f"[INFO] Command ranges: vx {ranges.lin_vel_x}  vy {ranges.lin_vel_y}  "
              f"wz {ranges.ang_vel_z}")
        # Training resamples the command every few seconds and on every reset.
        # Stop that here, otherwise the term overwrites the operator's command
        # right after we write it (a zero-range stage then looks like "the
        # command never arrived"). The per-step write below is the backstop.
        freeze_command_resampling(command_term)

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
    overwrite_reports = 0
    wheel_ids = None
    if args_cli.status_every:
        try:
            ids, _ = env.unwrapped.scene["robot"].find_joints(
                [".*_wheel_joint"], preserve_order=True)
            wheel_ids = ids or None
        except Exception as exc:  # noqa: BLE001 - status line is diagnostics only
            print(f"[WARN] Could not resolve wheel joint ids: {exc}")
    while simulation_app.is_running():
        start = time.time()
        intended = None
        if command_term is not None:
            if keyboard is not None:
                intended = list(keyboard.command)
                write_command(command_term, keyboard.command)
                if keyboard.command_changed:
                    print(f"[INFO] command -> vx={keyboard.command[0]:+.2f}  "
                          f"vy={keyboard.command[1]:+.2f}  wz={keyboard.command[2]:+.2f}")
                    keyboard.command_changed = False
            else:
                intended = list(pending_command)
                write_command(command_term, pending_command)
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if intended is not None and overwrite_reports < 3:
            buffer = getattr(command_term, "vel_command_b", None)
            if buffer is not None:
                written = torch.tensor(intended, device=buffer.device, dtype=buffer.dtype)
                if not torch.allclose(buffer[0, :3], written, atol=1.0e-3):
                    print("[WARN] The command was overwritten inside env.step: wrote "
                          f"{written.tolist()} but read {buffer[0, :3].tolist()}. "
                          "Something is still resampling the command term.")
                    overwrite_reports += 1
        timestep += 1
        if args_cli.status_every and timestep % args_cli.status_every == 0:
            print(status_line(env, command_term, wheel_ids, timestep, obs), flush=True)

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
