"""Fixed-command acceptance test for the M20 velocity policy.

Runs the plan's test battery and compares the result against the plan's
acceptance table. Nothing here trains or explores: the commands are fixed, the
robot is the nominal one, and the numbers are what deployment would see.

=================  ===============  ===============
#                  command (vx, vy, wz)  motion
=================  ===============  ===============
1                  (0.30, 0, 0)     forward
2                  (-0.30, 0, 0)    backward
3                  (0, 0.20, 0)     strafe left
4                  (0, -0.20, 0)    strafe right
5                  (0, 0, 0.50)     turn left
6                  (0, 0, -0.50)    turn right
7                  (0.30, 0, 0.30)   forward + turn left
8                  (-0.30, 0.20, -0.30) backward + strafe left + turn right
9                  (0, 0, 0)        hold still
10                 (0.30,0,0) <-> (0,0,0)   switch go/stop, 3 cycles
=================  ===============  ===============

Acceptance (from the plan):

* forward / backward speed RMSE < 0.10 m/s
* lateral speed RMSE < 0.12 m/s
* yaw-rate RMSE < 0.15 rad/s
* roll RMS < 5 deg, pitch RMS < 5 deg
* flat-ground fall rate < 2%
* zero command: body speed near 0
* switch test: go-phase speed error < 0.12 m/s and, once a stop phase has
  settled, residual speed < 0.08 m/s (course stage 4)

The fall rate is reported per 1000 policy steps. An episode is 20 s at 50 Hz,
i.e. 1000 steps, so "2% of episodes fall" and "20 falls per 1000 steps" are the
same threshold.

Usage::

    python -m isaaclab_m20.scripts.eval --stage 4        # that stage of the newest run
    python -m isaaclab_m20.scripts.eval --stage 2        # look at one stage on its own
    python -m isaaclab_m20.scripts.eval --checkpoint logs/rsl_rl/m20_velocity/<run>/stage4_agile
    python -m isaaclab_m20.scripts.eval            # newest checkpoint of the newest run
    python -m isaaclab_m20.scripts.eval --seconds 10 --num_envs 32 --strict

``--checkpoint`` takes a ``model_*.pt`` file or a directory (stage or run); the
file names carry the cumulative iteration count, so stage 4 ends around 3250.
``--stage N`` picks the highest checkpoint of ``<newest run>/stageN_*`` instead,
which is the "check this stage before spending the next one" entry point.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VELOCITY_TASK = "Isaac-M20-Wheeled-Velocity-v0"
POLICY_DT = 0.02          # 50 Hz

# Course order of the stage directory suffixes: <run>/stage3_multi_direction/ ...
STAGE_SLUGS = ("stand", "forward_backward", "multi_direction", "agile")

parser = argparse.ArgumentParser(description="Fixed-command acceptance test for M20.")
parser.add_argument("--task", type=str, default=VELOCITY_TASK)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a model_*.pt, a stage directory, or a run directory. Default: "
                         "newest checkpoint of the newest run under logs/rsl_rl/m20_velocity.")
parser.add_argument("--stage", type=int, default=None,
                    choices=range(1, len(STAGE_SLUGS) + 1), metavar="{1..4}",
                    help="Evaluate one curriculum stage (highest model_*.pt in "
                         "<newest run>/stageN_*) instead of the newest model overall.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=8.0,
                    help="Duration of each test, after the settle time.")
parser.add_argument("--settle", type=float, default=0.5,
                    help="Seconds discarded after each command switch (steady state only).")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
parser.add_argument("--json_out", type=str, default=None, help="Write the results as JSON here.")
parser.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any test fails (useful in CI).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_m20.tasks  # noqa: F401 - registers the task ids


# --------------------------------------------------------------------------- #
# Test battery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Test:
    label: str
    command: tuple[float, float, float]
    motion: str


TESTS: tuple[Test, ...] = (
    Test("1", (0.30, 0.00, 0.00), "forward"),
    Test("2", (-0.30, 0.00, 0.00), "backward"),
    Test("3", (0.00, 0.20, 0.00), "strafe left"),
    Test("4", (0.00, -0.20, 0.00), "strafe right"),
    Test("5", (0.00, 0.00, 0.50), "turn left"),
    Test("6", (0.00, 0.00, -0.50), "turn right"),
    Test("7", (0.30, 0.00, 0.30), "forward + turn left"),
    Test("8", (-0.30, 0.20, -0.30), "back + strafe left + turn right"),
    Test("9", (0.00, 0.00, 0.00), "hold still"),
)

# Acceptance thresholds from the plan.
MAX_VX_RMSE = 0.10      # m/s
MAX_VY_RMSE = 0.12      # m/s
MAX_WZ_RMSE = 0.15      # rad/s
MAX_TILT_RMS = 5.0      # deg, roll and pitch
MAX_FALLS_PER_1000 = 20.0   # = 2% of 20 s episodes
MAX_STILL_SPEED = 0.05      # m/s, zero-command test
MAX_STILL_YAW = 0.10        # rad/s, zero-command test

# Switching test (course stage 4): alternate "go" and "stop" commands and check
# that the policy both reaches the speed and comes back to rest, within one run.
SWITCH_COMMAND = (0.30, 0.00, 0.00)
SWITCH_PHASE_S = 2.0        # each command holds this long
SWITCH_SETTLE_S = 1.0       # ignored at the start of each phase (transient)
MAX_SWITCH_GO_RMSE = 0.12   # m/s, speed error in the measured part of a "go" phase
MAX_SWITCH_STOP_SPEED = 0.08  # m/s, residual speed once a "stop" phase has settled


def resolve_stage_dir(experiment_dir: Path, stage: int) -> Path:
    """``<newest run>/stageN_*`` of the experiment, for ``--stage``."""
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()] \
        if experiment_dir.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"No run directories under {experiment_dir}")
    run_dir = max(runs, key=lambda path: path.stat().st_mtime)
    stage_dir = run_dir / f"stage{stage}_{STAGE_SLUGS[stage - 1]}"
    if not stage_dir.is_dir():
        present = sorted(path.name for path in run_dir.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"No stage {stage} ({stage_dir.name}) in {run_dir}. Present: "
            f"{', '.join(present) or 'none'}")
    return stage_dir


def resolve_checkpoint(log_root: Path, explicit: str | None) -> Path:
    """Newest checkpoint of the newest run, unless one is given explicitly.

    ``explicit`` may be a ``model_*.pt`` file, a stage directory, or the whole
    run directory: pointing at a directory takes its highest iteration number,
    so ``--checkpoint logs/rsl_rl/m20_velocity/m20_v1/stage4_agile``
    works without spelling out which ``model_<n>.pt`` that stage ended on.
    """

    def iterations(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    if explicit:
        path = Path(explicit)
        if path.is_dir():
            candidates = list(path.rglob("model_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No model_*.pt under {path}")
            return max(candidates, key=iterations)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    runs = [p for p in log_root.glob("*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs under {log_root}")
    candidates = [p for p in log_root.rglob("model_*.pt")]
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt under {log_root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, iterations(p)))


def tilt_deg(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch in degrees from the body-frame gravity projection.

    For roll phi and pitch theta, g = (sin theta, -sin phi cos theta,
    -cos phi cos theta), so pitch = asin(gx) and roll = atan2(-gy, -gz). Using
    asin/atan2 here (rather than assuming small angles as the reward kernel does)
    is deliberate: the acceptance table is written in degrees of real tilt.
    """
    gx, gy, gz = projected_gravity[:, 0], projected_gravity[:, 1], projected_gravity[:, 2]
    pitch = torch.asin(gx.clamp(-1.0, 1.0))
    roll = torch.atan2(-gy, -gz.clamp(max=-1e-6))
    return torch.rad2deg(roll), torch.rad2deg(pitch)


@dataclass
class Result:
    label: str
    motion: str
    command: tuple[float, float, float]
    rmse_vx: float
    rmse_vy: float
    rmse_wz: float
    roll_rms: float
    pitch_rms: float
    falls_per_1000: float
    mean_speed: float
    mean_yaw: float
    checks: list[tuple[str, bool, str]]

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def run_test(env, policy, test: Test, steps: int, settle_steps: int,
             device: torch.device) -> Result:
    """Hold one command for ``steps`` policy steps and collect the metrics."""
    command = torch.tensor(test.command, dtype=torch.float, device=device).expand(
        env.unwrapped.num_envs, 3)
    term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]

    sum_sq = torch.zeros(3, device=device)
    sum_sq_roll = torch.zeros((), device=device)
    sum_sq_pitch = torch.zeros((), device=device)
    sum_abs_speed = torch.zeros((), device=device)
    sum_abs_yaw = torch.zeros((), device=device)
    counted = 0
    falls = 0

    obs = env.get_observations()

    for step in range(steps + settle_steps):
        # The command manager resamples on resets even with resampling disabled,
        # so write the command every step: this test measures the policy against
        # a fixed target, not against whatever the sampler drew.
        term.vel_command_b[:, :3] = command
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        terminated = env.unwrapped.termination_manager.terminated
        falls += int(terminated.sum().item())
        if step < settle_steps:
            continue

        keep = ~terminated
        if not bool(keep.any()):
            continue
        lin_vel = robot.data.root_lin_vel_b[keep, :2]
        ang_vel_z = robot.data.root_ang_vel_b[keep, 2]
        error = torch.stack(
            (lin_vel[:, 0] - command[keep, 0],
             lin_vel[:, 1] - command[keep, 1],
             ang_vel_z - command[keep, 2]), dim=1)
        sum_sq += torch.sum(torch.square(error), dim=0)
        roll, pitch = tilt_deg(robot.data.projected_gravity_b[keep])
        sum_sq_roll += torch.sum(torch.square(roll))
        sum_sq_pitch += torch.sum(torch.square(pitch))
        sum_abs_speed += torch.sum(torch.linalg.vector_norm(lin_vel, dim=1))
        sum_abs_yaw += torch.sum(torch.abs(ang_vel_z))
        counted += int(keep.sum().item())

    counted = max(counted, 1)
    rmse = torch.sqrt(sum_sq / counted).tolist()
    rmse_vx, rmse_vy, rmse_wz = rmse
    roll_rms = math.sqrt(float(sum_sq_roll) / counted)
    pitch_rms = math.sqrt(float(sum_sq_pitch) / counted)
    falls_per_1000 = 1000.0 * falls / max(steps * env.unwrapped.num_envs, 1)
    mean_speed = float(sum_abs_speed) / counted
    mean_yaw = float(sum_abs_yaw) / counted

    # The message shows the measured value next to the limit, not as an
    # inequality.  "0.304 < 0.1" read as a statement of fact (and, being false,
    # looked like a bug); "0.304 (limit 0.1)" cannot be misread.
    checks: list[tuple[str, bool, str]] = []
    if test.command[0] != 0.0:
        checks.append(("vx RMSE", rmse_vx < MAX_VX_RMSE,
                       f"{rmse_vx:.3f} (limit {MAX_VX_RMSE:.2f})"))
    if test.command[1] != 0.0:
        checks.append(("vy RMSE", rmse_vy < MAX_VY_RMSE,
                       f"{rmse_vy:.3f} (limit {MAX_VY_RMSE:.2f})"))
    if test.command[2] != 0.0:
        checks.append(("wz RMSE", rmse_wz < MAX_WZ_RMSE,
                       f"{rmse_wz:.3f} (limit {MAX_WZ_RMSE:.2f})"))
    if test.command == (0.0, 0.0, 0.0):
        checks.append(("still speed", mean_speed < MAX_STILL_SPEED,
                       f"{mean_speed:.3f} (limit {MAX_STILL_SPEED:.2f})"))
        checks.append(("still yaw", mean_yaw < MAX_STILL_YAW,
                       f"{mean_yaw:.3f} (limit {MAX_STILL_YAW:.2f})"))
    checks.append(("roll RMS", roll_rms < MAX_TILT_RMS,
                   f"{roll_rms:.1f} (limit {MAX_TILT_RMS:.0f})"))
    checks.append(("pitch RMS", pitch_rms < MAX_TILT_RMS,
                   f"{pitch_rms:.1f} (limit {MAX_TILT_RMS:.0f})"))
    checks.append(("falls/1000", falls_per_1000 < MAX_FALLS_PER_1000,
                   f"{falls_per_1000:.1f} (limit {MAX_FALLS_PER_1000:.0f})"))

    return Result(test.label, test.motion, test.command, rmse_vx, rmse_vy, rmse_wz,
                  roll_rms, pitch_rms, falls_per_1000, mean_speed, mean_yaw, checks)


def run_switching_test(env, policy, device: torch.device,
                       label: str = "10") -> Result:
    """Alternate go/stop commands and check both halves of the switch.

    Course stage 4 trains exactly this: the same policy has to hold a speed when
    the command is non-zero and come back to rest when it becomes zero. A policy
    that only ever sees long holds can pass tests 1-9 and still creep for a metre
    after every stop, which is what this row is for.

    Phases are 2 s of "go" and 2 s of "stop", repeated three times; the first
    second of each phase is discarded as the transient, the second is measured.
    """
    term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs

    phase_steps = int(round(SWITCH_PHASE_S / POLICY_DT))
    settle_steps = int(round(SWITCH_SETTLE_S / POLICY_DT))
    phases = [(1.0, SWITCH_COMMAND), (0.0, (0.0, 0.0, 0.0))] * 3

    sum_sq = torch.zeros(3, device=device)
    sum_sq_roll = torch.zeros((), device=device)
    sum_sq_pitch = torch.zeros((), device=device)
    sum_abs_speed = torch.zeros((), device=device)
    sum_abs_yaw = torch.zeros((), device=device)
    go_counted = 0
    stop_counted = 0
    falls = 0
    total_steps = 0

    obs = env.get_observations()
    for _, command_tuple in phases:
        command = torch.tensor(command_tuple, dtype=torch.float, device=device).expand(num_envs, 3)
        is_go = command_tuple[0] != 0.0 or command_tuple[1] != 0.0 or command_tuple[2] != 0.0
        for step in range(phase_steps):
            term.vel_command_b[:, :3] = command
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            total_steps += 1

            terminated = env.unwrapped.termination_manager.terminated
            falls += int(terminated.sum().item())
            if step < settle_steps:
                continue
            keep = ~terminated
            if not bool(keep.any()):
                continue

            lin_vel = robot.data.root_lin_vel_b[keep, :2]
            ang_vel_z = robot.data.root_ang_vel_b[keep, 2]
            roll, pitch = tilt_deg(robot.data.projected_gravity_b[keep])
            sum_sq_roll += torch.sum(torch.square(roll))
            sum_sq_pitch += torch.sum(torch.square(pitch))
            if is_go:
                error = torch.stack(
                    (lin_vel[:, 0] - command[keep, 0],
                     lin_vel[:, 1] - command[keep, 1],
                     ang_vel_z - command[keep, 2]), dim=1)
                sum_sq += torch.sum(torch.square(error), dim=0)
                go_counted += int(keep.sum().item())
            else:
                sum_abs_speed += torch.sum(torch.linalg.vector_norm(lin_vel, dim=1))
                sum_abs_yaw += torch.sum(torch.abs(ang_vel_z))
                stop_counted += int(keep.sum().item())

    measured = max(go_counted + stop_counted, 1)
    go_counted = max(go_counted, 1)
    stop_counted = max(stop_counted, 1)
    rmse_vx, rmse_vy, rmse_wz = torch.sqrt(sum_sq / go_counted).tolist()
    roll_rms = math.sqrt(float(sum_sq_roll) / measured)
    pitch_rms = math.sqrt(float(sum_sq_pitch) / measured)
    stop_speed = float(sum_abs_speed) / stop_counted
    stop_yaw = float(sum_abs_yaw) / stop_counted
    falls_per_1000 = 1000.0 * falls / max(total_steps * num_envs, 1)

    checks: list[tuple[str, bool, str]] = [
        ("go speed RMSE", rmse_vx < MAX_SWITCH_GO_RMSE,
         f"{rmse_vx:.3f} (limit {MAX_SWITCH_GO_RMSE:.2f})"),
        ("stop speed", stop_speed < MAX_SWITCH_STOP_SPEED,
         f"{stop_speed:.3f} (limit {MAX_SWITCH_STOP_SPEED:.2f})"),
        ("stop yaw", stop_yaw < MAX_STILL_YAW,
         f"{stop_yaw:.3f} (limit {MAX_STILL_YAW:.2f})"),
        ("roll RMS", roll_rms < MAX_TILT_RMS,
         f"{roll_rms:.1f} (limit {MAX_TILT_RMS:.1f})"),
        ("pitch RMS", pitch_rms < MAX_TILT_RMS,
         f"{pitch_rms:.1f} (limit {MAX_TILT_RMS:.1f})"),
        ("falls/1000", falls_per_1000 < MAX_FALLS_PER_1000,
         f"{falls_per_1000:.1f} (limit {MAX_FALLS_PER_1000:.0f})"),
    ]
    return Result(label, "switch go/stop x3", SWITCH_COMMAND, rmse_vx, rmse_vy, rmse_wz,
                  roll_rms, pitch_rms, falls_per_1000, stop_speed, stop_yaw, checks)


def print_result(result: Result) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    failures = ", ".join(f"{name} {detail}" for name, ok, detail in result.checks if not ok)
    print(f"[{verdict}] {result.label:>2}  {result.motion:<30} "
          f"cmd=({result.command[0]:+.2f},{result.command[1]:+.2f},{result.command[2]:+.2f})  "
          f"rmse vx/vy/wz={result.rmse_vx:.3f}/{result.rmse_vy:.3f}/{result.rmse_wz:.3f}  "
          f"r/p={result.roll_rms:.1f}/{result.pitch_rms:.1f} deg  "
          f"falls/1000={result.falls_per_1000:.1f}"
          + (f"  <- {failures}" if failures else ""), flush=True)


def main() -> None:
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(__import__(module_name, fromlist=["_"]), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Evaluation runs the nominal robot and no pushes: the numbers are meant to
    # describe deployment on flat ground, not the robustness of a randomized one.
    from isaaclab_m20.tasks.m20_velocity_env_cfg import set_domain_randomization
    set_domain_randomization(env_cfg, False)
    # Hold each command for as long as the test needs; the fixed command is also
    # rewritten every step, but disabling the sampler keeps the reset path quiet.
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)

    agent_module, agent_class = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(__import__(agent_module, fromlist=["_"]), agent_class)()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device

    explicit = args_cli.checkpoint
    if explicit is None and args_cli.stage is not None:
        explicit = str(resolve_stage_dir(Path(args_cli.log_root) / agent_cfg.experiment_name,
                                         args_cli.stage))
    checkpoint = resolve_checkpoint(Path(args_cli.log_root) / agent_cfg.experiment_name,
                                    explicit)
    print(f"[INFO] Checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = torch.device(agent_cfg.device)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

    steps = int(round(args_cli.seconds / POLICY_DT))
    settle_steps = int(round(args_cli.settle / POLICY_DT))
    num_envs = env.unwrapped.num_envs
    print(f"[INFO] {steps} steps per test ({args_cli.seconds:.1f} s) + "
          f"{settle_steps} settle steps, {num_envs} envs\n")

    results: list[Result] = []
    for test in TESTS:
        result = run_test(env, policy, test, steps, settle_steps, device)
        results.append(result)
        print_result(result)

    # Course stage 4's own acceptance: does the same policy also stop when told?
    switching = run_switching_test(env, policy, device)
    results.append(switching)
    print_result(switching)

    env.close()

    print("\n=== summary ===")
    header = (f"{'#':>2}  {'motion':<30} {'vx':>7} {'vy':>7} {'wz':>7} "
              f"{'roll':>6} {'pitch':>6} {'falls':>7}  verdict")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:>2}  {r.motion:<30} {r.rmse_vx:7.3f} {r.rmse_vy:7.3f} "
              f"{r.rmse_wz:7.3f} {r.roll_rms:6.1f} {r.pitch_rms:6.1f} "
              f"{r.falls_per_1000:7.1f}  {'PASS' if r.passed else 'FAIL'}")

    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} tests passed")
    print(f"thresholds: vx<{MAX_VX_RMSE} vy<{MAX_VY_RMSE} wz<{MAX_WZ_RMSE} "
          f"roll/pitch<{MAX_TILT_RMS}deg falls<{MAX_FALLS_PER_1000}/1000 steps "
          f"still<{MAX_STILL_SPEED} m/s {MAX_STILL_YAW} rad/s")
    print(f"row 10 (switch): vx column = go-phase speed error "
          f"(<{MAX_SWITCH_GO_RMSE}), mean stop-phase speed "
          f"(<{MAX_SWITCH_STOP_SPEED}), stop-phase yaw (<{MAX_STILL_YAW})")

    if args_cli.json_out:
        payload = {
            "checkpoint": str(checkpoint),
            "num_envs": num_envs,
            "seconds": args_cli.seconds,
            "results": [
                {
                    "label": r.label, "motion": r.motion, "command": list(r.command),
                    "rmse_vx": r.rmse_vx, "rmse_vy": r.rmse_vy, "rmse_wz": r.rmse_wz,
                    "roll_rms_deg": r.roll_rms, "pitch_rms_deg": r.pitch_rms,
                    "falls_per_1000_steps": r.falls_per_1000,
                    "mean_speed": r.mean_speed, "mean_yaw": r.mean_yaw,
                    "passed": r.passed,
                    "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in r.checks],
                }
                for r in results
            ],
        }
        Path(args_cli.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Results written to {args_cli.json_out}")

    if args_cli.strict and passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
