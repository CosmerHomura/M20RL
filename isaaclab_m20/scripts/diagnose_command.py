# Copyright (c) 2024-2026, M20RL contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose why an externally written velocity command disappears.

``play`` and ``eval`` own the command, but something between the write and the
policy's observation keeps resetting it to zero: the status line reads
``cmd=(0,0,0)`` one step after a non-zero write. This script isolates the path in
one run, with no policy and no keyboard:

1. what the command term actually is (class, attributes, timers, resampling);
2. whether a write to ``vel_command_b`` survives being read back immediately;
3. whether the buffer keeps its value across ``env.step`` with zero actions;
4. whether the policy observation (columns 37:40) carries the same value.

Run it and paste the whole output:

.. code-block:: bash

    python -m isaaclab_m20.scripts.diagnose_command --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace the velocity command path.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-M20-Wheeled-Velocity-v0")
parser.add_argument("--command", type=float, nargs=3, default=[0.25, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"), help="command to write (m/s, m/s, rad/s)")
parser.add_argument("--steps", type=int, default=5, help="policy steps to watch")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_m20.tasks  # noqa: F401,E402 - registers the task ids
from isaaclab_m20.tasks.commands import freeze_command_resampling  # noqa: E402

COMMAND_SLICE = slice(37, 40)


def describe(term, label: str) -> None:
    """Print the state of the command term under ``label``."""
    buffer = getattr(term, "vel_command_b", None)
    values = "missing" if buffer is None else [round(float(v), 4) for v in buffer[0, :3]]
    print(f"[{label}] buffer id={id(buffer)} value={values}", flush=True)


def main() -> None:
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(__import__(module_name, fromlist=["_"]), class_name)()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    term = unwrapped.command_manager.get_term("base_velocity")
    print(f"[INFO] env class        = {type(unwrapped).__name__}", flush=True)
    print(f"[INFO] command term     = {type(term).__name__}", flush=True)
    print(f"[INFO] term attributes  = {sorted(vars(term))}", flush=True)
    print(f"[INFO] cfg ranges       = vx {term.cfg.ranges.lin_vel_x} "
          f"vy {term.cfg.ranges.lin_vel_y} wz {term.cfg.ranges.ang_vel_z}", flush=True)
    print(f"[INFO] cfg resampling   = {term.cfg.resampling_time_range}", flush=True)
    print(f"[INFO] cached resampling= {getattr(term, '_resampling_time_range', 'n/a')}", flush=True)
    print(f"[INFO] time_left        = {getattr(term, 'time_left', 'n/a')}", flush=True)
    describe(term, "start")

    # 1. Does a plain buffer write survive an immediate read-back?
    command = torch.tensor(args_cli.command, dtype=torch.float32)
    term.vel_command_b[:, :3] = command.to(term.vel_command_b.device)
    describe(term, "write")

    # 2. Same write through the env's pin hook, which also re-asserts after resets.
    pin = getattr(unwrapped, "set_command", None)
    print(f"[INFO] env has set_command = {callable(pin)}", flush=True)
    if callable(pin):
        pin(command)
        describe(term, "pinned")

    # 3. Freeze resampling and watch a few steps with zero actions.
    freeze_command_resampling(term)
    print(f"[INFO] after freeze: cfg={term.cfg.resampling_time_range} "
          f"cached={getattr(term, '_resampling_time_range', 'n/a')} "
          f"time_left={getattr(term, 'time_left', 'n/a')} "
          f"noop={term._resample_command}", flush=True)

    actions = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    obs = env.get_observations()
    for step in range(args_cli.steps):
        if callable(pin):
            pin(command)
        else:
            term.vel_command_b[:, :3] = command.to(term.vel_command_b.device)
        with torch.inference_mode():
            obs, reward, _, _, extras = env.step(actions)
        seen = obs[COMMAND_SLICE] if obs.ndim == 2 else obs["policy"][:, COMMAND_SLICE]
        print(f"[step {step}] buffer={[round(float(v), 4) for v in term.vel_command_b[0, :3]]} "
              f"obs={[round(float(v), 4) for v in seen[0]]} "
              f"episode_len={int(unwrapped.episode_length_buf[0])} "
              f"reward={float(reward[0]):+.4f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
