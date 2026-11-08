"""Audit the M20 joint configuration exactly as the training env sees it.

The question this answers: is ``joint_deviation = -0.367`` per step (raw value
``367``, i.e. legs ~1.6 rad away from the default pose) a *reward* problem or a
*joint configuration* problem? It prints, in one run:

1. the articulation's joint order, default positions, limits and actuator group
   for every joint;
2. what the reward's regexes actually resolve to (a regex that silently matches
   five joints instead of twelve changes every number in the reward);
3. whether ``ArticulationCfg.init_state.joint_pos`` and the USD's
   ``default_joint_pos`` agree -- the position action adds its offsets to
   ``default_joint_pos``, so a mismatch there drives the legs to a pose nobody
   designed;
4. where the legs settle after N steps of *zero* action (the same target the
   policy commands at ``a = 0``), with the raw ``joint_deviation`` value the
   reward computes and a per-group breakdown.

Usage::

    python -m isaaclab_m20.scripts.check_joints --headless
    python -m isaaclab_m20.scripts.check_joints --headless --steps 300 --settle
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Audit M20 joint config vs the reward.")
parser.add_argument(
    "--task", type=str, default="Isaac-M20-Wheeled-Velocity-v0",
    help="Task to build. The velocity task is the one whose reward is being questioned.")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to build (1 is enough).")
parser.add_argument(
    "--usd", type=str, default=None,
    help="Override the robot USD (e.g. model/usd/M20.usd vs model/M20_urdf.usd). "
         "Default: whatever M20_USD dictates in isaaclab_m20/assets/m20.py.")
parser.add_argument("--steps", type=int, default=200,
                    help="Zero-action steps before reading the settled pose.")
parser.add_argument(
    "--settle", action="store_true",
    help="Also print the pose every 25 steps, to show whether it is still moving.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Everything below needs the Omniverse app running.
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
import importlib  # noqa: E402
import torch  # noqa: E402

import isaaclab_m20.tasks  # noqa: E402,F401  (registers the gym ids)
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_m20.assets import M20_CFG, M20_USD_SOURCE  # noqa: E402
from isaaclab_m20.tasks import rewards  # noqa: E402
from isaaclab_m20.tasks.observations import LEG_JOINT_NAMES, WHEEL_JOINT_NAMES  # noqa: E402

GROUPS = (
    ("hipx", ".*_hipx_joint"),
    ("hipy", ".*_hipy_joint"),
    ("knee", ".*_knee_joint"),
    ("wheel", ".*_wheel_joint"),
)


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(4, 68 - len(title)))


def fmt_gain(value) -> str:
    """Actuator gains are scalars in some Isaac Lab versions and per-joint tensors in others.

    A per-joint tensor is worth seeing whole, so multi-element values print as
    ``min/mean/max`` instead of being collapsed to "only one element" errors.
    """
    if value is None:
        return "?"
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        return "?"
    if tensor.numel() == 1:
        return f"{tensor.item():.1f}"
    return f"{tensor.min().item():.1f}/{tensor.mean().item():.1f}/{tensor.max().item():.1f}"


def group_indices(robot, pattern: str) -> list[int]:
    return [index for index, name in enumerate(robot.joint_names) if re.fullmatch(pattern, name)]


def print_joint_table(robot) -> None:
    section("joint table (articulation order)")
    default = robot.data.default_joint_pos[0].tolist()
    current = robot.data.joint_pos[0].tolist()
    try:
        limits = robot.data.soft_joint_pos_limits[0].tolist()
    except Exception:  # pragma: no cover - diagnostics only
        limits = None
    group_of: dict[int, str] = {}
    for group_name, actuator in robot.actuators.items():
        for joint_index in actuator.joint_indices.tolist():
            group_of[joint_index] = group_name
    print(f"  {'id':>3} {'joint':<22} {'default':>8} {'current':>8} {'limits':>18}  actuator")
    for index, name in enumerate(robot.joint_names):
        span = ""
        if limits is not None:
            span = f"[{limits[index][0]:+.2f},{limits[index][1]:+.2f}]"
        print(f"  {index:>3} {name:<22} {default[index]:+8.3f} {current[index]:+8.3f}"
              f" {span:>18}  {group_of.get(index, '?')}")

    section("actuator groups")
    for group_name, actuator in robot.actuators.items():
        indices = actuator.joint_indices.tolist()
        print(f"  {group_name}: {len(indices)} joints, stiffness={fmt_gain(getattr(actuator, 'stiffness', None))}, "
              f"damping={fmt_gain(getattr(actuator, 'damping', None))}")
        print("    gains are shown as min/mean/max when the actuator stores one value per joint")
        print(f"    {[robot.joint_names[i] for i in indices]}")


def check_init_state_matches_default(robot) -> None:
    """``cfg.init_state.joint_pos`` is what the USD's default pose should be."""
    expected: dict[str, float] = {}
    for pattern, value in M20_CFG.init_state.joint_pos.items():
        for name in robot.joint_names:
            if re.fullmatch(pattern, name):
                expected[name] = float(value)

    section("cfg.init_state.joint_pos vs articulation default_joint_pos")
    default = robot.data.default_joint_pos[0].tolist()
    worst = 0.0
    for index, name in enumerate(robot.joint_names):
        want = expected.get(name)
        if want is None:
            print(f"  [WARN] {name:<22} has no entry in M20_CFG.init_state.joint_pos")
            continue
        got = default[index]
        delta = abs(want - got)
        worst = max(worst, delta)
        flag = "" if delta < 1e-3 else "   <-- MISMATCH"
        print(f"  {name:<22} cfg={want:+.3f}  usd={got:+.3f}{flag}")
    if worst < 1e-3:
        print("  [OK] every joint matches: the position action's default offset is the designed stance")
    else:
        print(f"  [FAIL] worst mismatch {worst:.3f} rad")
        print("         The position action adds its offsets to default_joint_pos, so the policy")
        print("         would hold a pose that no one designed. Fix init_state or the USD import.")


def check_resolution(scene, robot) -> None:
    section("regex resolution (this is what the reward actually sees)")
    for label, patterns, expected in (("legs", LEG_JOINT_NAMES, 12),
                                      ("wheels", WHEEL_JOINT_NAMES, 4)):
        cfg = SceneEntityCfg("robot", joint_names=patterns)
        cfg.resolve(scene)
        indices = list(cfg.joint_ids)
        names = [robot.joint_names[index] for index in indices]
        flag = "[OK]" if len(names) == expected else f"[FAIL] expected {expected}"
        print(f"  {label}: {flag} {len(names)} joints -> {names}")
        if len(set(indices)) != len(indices):
            print("  [FAIL] duplicate joint ids resolved -- the regexes overlap")


def deviation_report(robot, label: str) -> None:
    """Raw ``joint_deviation`` plus per-group RMS, in radians."""
    legs = [index for _, pattern in GROUPS if pattern != ".*_wheel_joint"
            for index in group_indices(robot, pattern)]
    deviation = robot.data.joint_pos[:, legs] - robot.data.default_joint_pos[:, legs]
    raw_old = torch.sum(torch.square(deviation / 0.3), dim=1).mean().item()
    print(f"  [{label}] joint_deviation raw, old sum/0.3 form = {raw_old:.1f}"
          f"   (new mean/1.0 form = {torch.mean(torch.square(deviation), dim=1).mean().item():.3f})")
    print(f"  [{label}] |dq| mean={deviation.abs().mean().item():.3f} rad, "
          f"max={deviation.abs().max().item():.3f} rad")
    for group_name, pattern in GROUPS:
        indices = group_indices(robot, pattern)
        group_dev = robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]
        print(f"    {group_name:<5} |dq| mean={group_dev.abs().mean().item():.3f} rad  "
              f"q mean={robot.data.joint_pos[:, indices].mean().item():+.3f}  "
              f"|qdot| mean={robot.data.joint_vel[:, indices].abs().mean().item():.2f} rad/s")

    # A knee sitting on its limit is the difference between "the PD is pushing the
    # wrong way" and "the legs are merely sagging": within the action range
    # (default +- 0.25 rad) the policy can never command a joint to its limit.
    try:
        limits = robot.data.soft_joint_pos_limits[0]
    except Exception:  # pragma: no cover - diagnostics only
        return
    positions = robot.data.joint_pos[0]
    pinned: list[str] = []
    for index, name in enumerate(robot.joint_names):
        low, high = limits[index, 0].item(), limits[index, 1].item()
        if not math.isfinite(low) or not math.isfinite(high):
            continue
        if min(positions[index].item() - low, high - positions[index].item()) < 0.05:
            pinned.append(f"{name}={positions[index].item():+.2f}")
    if pinned:
        print(f"  [{label}] pinned at a limit (<0.05 rad): {', '.join(pinned)}")
    else:
        print(f"  [{label}] no joint is pinned at a limit")


def build_env() -> gym.Env:
    """Build the task through its registered config, the same way train.py does."""
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 0
    if args_cli.usd:
        env_cfg.scene.robot.spawn.usd_path = str(Path(args_cli.usd).expanduser().resolve())
    print(f"[INFO] robot USD: {env_cfg.scene.robot.spawn.usd_path} ({M20_USD_SOURCE})")
    return gym.make(args_cli.task, cfg=env_cfg)


def main() -> None:
    env = build_env()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    zero_action = torch.zeros(unwrapped.num_envs,
                              unwrapped.action_manager.total_action_dim,
                              device=unwrapped.device)

    print(f"[INFO] task={args_cli.task}  num_envs={unwrapped.num_envs}  "
          f"joints={len(robot.joint_names)}  bodies={len(robot.body_names)}")
    print(f"[INFO] action_dim={unwrapped.action_manager.total_action_dim}  "
          f"decimation={unwrapped.cfg.decimation}  step_dt={unwrapped.step_dt}")

    print_joint_table(robot)
    check_init_state_matches_default(robot)
    check_resolution(unwrapped.scene, robot)

    section("zero-action hold (what the PD does when the policy outputs a = 0)")
    unwrapped.reset()
    print("  after reset:")
    deviation_report(robot, "reset")
    for step in range(1, args_cli.steps + 1):
        with torch.no_grad():
            unwrapped.step(zero_action)
        if args_cli.settle and step % 25 == 0:
            print(f"  step {step:4d}:")
            deviation_report(robot, f"step {step}")
    print(f"  after {args_cli.steps} zero-action steps:")
    deviation_report(robot, "settled")

    section("what the reward would charge")
    # A fresh, resolved SceneEntityCfg: the default argument inside rewards.py has
    # joint_ids = slice(None) until the manager resolves it, which would silently
    # include the four (unbounded) wheel angles in this number.
    leg_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
    leg_cfg.resolve(unwrapped.scene)
    new_form = rewards.joint_deviation_l2(unwrapped, scale=1.0, asset_cfg=leg_cfg).mean().item()
    # The old code summed over the 12 leg joints, so its value is 12x this mean.
    old_form = 12.0 * rewards.joint_deviation_l2(unwrapped, scale=0.3, asset_cfg=leg_cfg).mean().item()
    dt = unwrapped.step_dt
    print(f"  new form mean_legs((dq/1.0)^2) = {new_form:.3f}  -> per-step "
          f"{-1.0 * new_form * dt:+.4f} at weight -1.0")
    print(f"  old form sum_legs((dq/0.3)^2)  = {old_form:.1f}  -> per-step "
          f"{-0.05 * old_form * dt:+.4f} at the old weight -0.05")
    print(f"  the per-step positive side caps at {(2.0 + 1.0) * dt:.3f} "
          "(tracking + yaw, both <= 1.0)")

    env.close()
    del env


if __name__ == "__main__":
    main()
    simulation_app.close()
