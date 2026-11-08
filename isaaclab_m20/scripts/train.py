"""Train M20 with rsl_rl PPO in Isaac Lab.

The velocity task trains as a five-stage course, each stage continuing from the
previous stage's checkpoint:

======  ==================  ========  =========================================
stage   name                iters     command range (half-widths)
======  ==================  ========  =========================================
1       stand               300       vx 0,           vy 0,      wz 0
2       forward_backward    700       vx +-0.4,       vy 0,      wz 0
3       lateral             1000      vx +-0.4,       vy +-0.2,  wz 0
4       yaw                 700       vx +-0.4,       vy +-0.2,  wz +-0.6
5       mixed               1250      vx +-0.5,       vy +-0.2,  wz +-0.8
======  ==================  ========  =========================================

3950 iterations total, in the order "stand -> forward/back -> lateral -> yaw ->
everything", each axis introduced on its own before it is combined with the
others.

Stage 1 is a zero-command stage: the tracking terms are then "be still", so it
trains the baseline the later stages build on -- hold the ride height, keep the
body level, do not spin the wheels, do not drift. That is the answer to the "we
already have a standing policy" problem: that policy was trained on 57
observations and cannot be loaded into a 56-observation actor, so the course has
to re-establish standing under the new observation layout first. Keeping it as
its own stage (rather than as a warmup inside stage 2) gives it its own
checkpoint, which is what makes the later stages comparable: if forward/backward
never improves on stage 1, the problem is the command, not the baseline.

After stage 1 every stage opens with a short warmup during which the new axis is
interpolated from zero up to its target width over 150-250 iterations, instead of
appearing at full width on the first step.

Example runs::

    # the full course, 512 envs
    python -m isaaclab_m20.scripts.train

    # one stage at a time (each continues from the previous stage's checkpoint)
    python -m isaaclab_m20.scripts.train --stage 2
    python -m isaaclab_m20.scripts.train --stage 3

    # 1024 envs once 512 is stable
    python -m isaaclab_m20.scripts.train --num_envs 1024

    # smoke test: 5% of every stage's iteration budget
    python -m isaaclab_m20.scripts.train --stage_scale 0.05

The legacy standing task (``Isaac-M20-Wheeled-v0``) has no command term, so for
it this script runs a single plain run, as before.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VELOCITY_TASK = "Isaac-M20-Wheeled-Velocity-v0"

parser = argparse.ArgumentParser(description="Train M20 wheeled locomotion with rsl_rl.")
parser.add_argument("--task", type=str, default=VELOCITY_TASK)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override the number of environments (velocity task: 512 to start, "
                         "1024 once that is stable).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stage", type=str, default="all",
                    help="Velocity task: which course stage to run ('1'..'5' or 'all'). "
                         "Each stage continues from the previous stage's latest checkpoint "
                         "unless --resume_from says otherwise.")
parser.add_argument("--stage_scale", type=float, default=1.0,
                    help="Multiply every stage's iteration budget (e.g. 0.05 for a smoke test).")
parser.add_argument("--iterations", type=int, default=None,
                    help="Override the per-stage iteration budget for the selected stages.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Checkpoint for the first selected stage. Without it, stage 1 starts "
                         "from scratch and stages 2-5 continue from the previous stage.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Sub-directory inside logs/rsl_rl/<experiment>; stages get their own "
                         "sub-sub-directory inside it.")
parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
parser.add_argument("--video", action="store_true", help="Record a video of the first iteration.")
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
    "--dr", type=str, default="off", choices=("off", "full"),
    help="Velocity task only: 'off' trains on the nominal robot (flat ground, "
         "friction 1.0, no payload spread, no pushes) -- the first version's setting. "
         "'full' enables the friction/mass/CoM ranges and the shoves.")
parser.add_argument(
    "--dr_ramp", type=float, default=0.4,
    help="Standing task only (no command term): fraction of training over which the "
         "domain randomization ramps from 0 to full.")
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

import isaaclab_m20.tasks  # noqa: F401 - registers the M20 task ids
from isaaclab_m20.tasks import dr
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid


class _LogDumper(yaml.SafeDumper):
    """Dumper for the params log: stringify values YAML has no type for."""


def _represent_unknown(dumper, data):
    return dumper.represent_str(str(data))


# SceneEntityCfg defaults to joint_ids=slice(None), which plain SafeDumper rejects.
_LogDumper.add_multi_representer(slice, _represent_unknown)
_LogDumper.add_multi_representer(object, _represent_unknown)


# --------------------------------------------------------------------------- #
# Course definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    """One stage of the course.

    ``ranges_from`` -> ``ranges_to`` are (vx, vy, wz) half-widths in m/s / m/s /
    rad/s. The warmup interpolates linearly between them over ``warmup``
    iterations and then holds ``ranges_to``. ``ranges_from`` is written as the
    previous stage's ``ranges_to`` -- not as zeros -- because the axes learned
    earlier must keep their full width while the new axis grows in.
    """

    name: str
    iterations: int
    ranges_from: tuple[float, float, float]
    ranges_to: tuple[float, float, float]
    warmup: int


STAGES: tuple[Stage, ...] = (
    # Zero command: the standing baseline under the new 56-dim observation
    # layout. Every term still runs; the tracking terms just mean "be still".
    Stage("1_stand", 300, (0.00, 0.0, 0.0), (0.00, 0.0, 0.0), 0),
    Stage("2_forward_backward", 700, (0.00, 0.0, 0.0), (0.40, 0.0, 0.0), 150),
    Stage("3_lateral", 1000, (0.40, 0.0, 0.0), (0.40, 0.2, 0.0), 250),
    Stage("4_yaw", 700, (0.40, 0.2, 0.0), (0.40, 0.2, 0.6), 250),
    Stage("5_mixed", 1250, (0.40, 0.2, 0.6), (0.50, 0.2, 0.8), 250),
)


def select_stages(selector: str, scale: float, iterations: int | None) -> list[tuple[int, Stage]]:
    """Which stages this run executes, with the iteration budget applied.

    Returns ``(index_in_STAGES, stage)`` pairs so the caller can find the
    preceding stage's checkpoint for the first selected stage.
    """
    if selector == "all":
        indices = list(range(len(STAGES)))
    else:
        try:
            index = int(selector) - 1
        except ValueError as exc:
            raise SystemExit(f"--stage must be 1..{len(STAGES)} or 'all', got {selector!r}") from exc
        if not 0 <= index < len(STAGES):
            raise SystemExit(f"--stage must be 1..{len(STAGES)} or 'all', got {selector!r}")
        indices = [index]

    out: list[tuple[int, Stage]] = []
    for index in indices:
        stage = STAGES[index]
        budget = iterations if iterations is not None else int(round(stage.iterations * scale))
        if budget <= 0:
            raise SystemExit(f"stage {stage.name} would run {budget} iterations")
        # Never spend more than half of a trimmed stage on the warmup: with
        # --stage_scale 0.05 the budget is 35 iterations for stage 2 and a
        # 150-iteration warmup would hold the new axis near zero the whole run.
        # ``max(..., 1)`` keeps a 0-iteration warmup (stage 1) at 0 rather than 1.
        warmup = min(stage.warmup, budget // 2) if stage.warmup else 0
        out.append((index, Stage(stage.name, budget, stage.ranges_from, stage.ranges_to, warmup)))
    return out


class CommandSchedule:
    """Drives the command range through the current stage, one iteration at a time.

    Attached to ``env.step``: rsl_rl calls ``learn`` once per stage, but the env
    wrapper lives across all of them, so the schedule has to know which stage it
    is in. ``start_stage`` is called right before each ``runner.learn``.
    """

    def __init__(self, command_term, steps_per_iteration: int) -> None:
        self.term = command_term
        self.steps_per_iteration = steps_per_iteration
        self.stage: Stage | None = None
        self.iteration = 0
        self._steps = 0
        self._applied: tuple[float, float, float] | None = None
        dr.register_command_term(command_term)

    # -- stage lifecycle ---------------------------------------------------
    def start_stage(self, stage: Stage) -> None:
        self.stage = stage
        self.iteration = 0
        self._steps = 0
        self._applied = None
        self._apply(stage.ranges_from, announce=True)
        # The buffer still holds commands drawn under the previous stage's
        # range (up to 4 s of them). For stage 1 the new range starts at zero,
        # so wipe them: the robot must not be asked for 0.4 m/s on step 0.
        if max(stage.ranges_from) == 0.0:
            dr.zero_command(self.term)

    def _apply(self, ranges: tuple[float, float, float], announce: bool = False) -> None:
        if self._applied is not None and all(
                abs(a - b) < 1e-4 for a, b in zip(ranges, self._applied)):
            return
        dr.set_command_ranges(*ranges)
        self._applied = ranges
        if announce:
            stage = self.stage.name if self.stage else "?"
            print(f"[cmd] stage={stage} vx=+-{ranges[0]:.3f} "
                  f"vy=+-{ranges[1]:.3f} wz=+-{ranges[2]:.3f}", flush=True)

    def ranges_at(self, iteration: int) -> tuple[float, float, float]:
        """Range after ``iteration`` iterations of the current stage."""
        assert self.stage is not None
        stage = self.stage
        if stage.warmup <= 0:
            return stage.ranges_to
        t = min(max(iteration / stage.warmup, 0.0), 1.0)
        return tuple(
            low + (high - low) * t
            for low, high in zip(stage.ranges_from, stage.ranges_to)
        )

    # -- per-step hook -----------------------------------------------------
    def on_step(self) -> None:
        if self.stage is None:
            return
        self._steps += 1
        if self._steps % self.steps_per_iteration:
            return
        self.iteration = self._steps // self.steps_per_iteration
        self._apply(self.ranges_at(self.iteration))
        if self.iteration % 50 == 0:
            r = self._applied or self.ranges_at(self.iteration)
            print(f"[cmd] iter={self.iteration:5d} vx=+-{r[0]:.3f} "
                  f"vy=+-{r[1]:.3f} wz=+-{r[2]:.3f}", flush=True)


def attach_stage_schedule(env, schedule: CommandSchedule) -> None:
    original_step = env.step

    def step(action):
        out = original_step(action)
        schedule.on_step()
        return out

    env.step = step


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


def attach_dr_ramp(env, max_iterations: int, num_steps_per_env: int,
                   ramp_fraction: float) -> None:
    """Standing task only: ramp the domain-randomization level over training."""
    ramp_iterations = max(int(max_iterations * ramp_fraction), 1)
    plain = ramp_fraction <= 0.0
    if plain:
        dr.set_level(1.0)
        print("[INFO] --dr_ramp 0: full domain randomization from step 0.")
    else:
        print(f"[INFO] Domain randomization ramps 0 -> 1 over {ramp_iterations} iterations.")

    original_step = env.step
    state = {"n": 0}

    def step(action):
        out = original_step(action)
        state["n"] += 1
        if not plain and state["n"] % num_steps_per_env == 0:
            dr.set_level((state["n"] // num_steps_per_env) / ramp_iterations)
        return out

    env.step = step


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest ``model_<iteration>.pt`` under ``directory``, by iteration number.

    Sorted numerically, not lexicographically: the runner numbers checkpoints by
    the global iteration count, so ``model_1200.pt`` sorts *before*
    ``model_700.pt`` as a string and a plain ``sorted(...)[-1]`` would resume
    from a checkpoint 500 iterations old.

    Searched recursively because rsl_rl puts them in ``<log_dir>/checkpoints``
    in some versions and directly in ``log_dir`` in others.
    """
    candidates = []
    for path in directory.rglob("model_*.pt"):
        try:
            candidates.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            continue
    candidates.sort()
    return candidates[-1][1] if candidates else None


def main() -> None:
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed
    fit_viewer_to_grid(env_cfg)

    agent_module, agent_class = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(importlib.import_module(agent_module), agent_class)()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device

    # The velocity task carries a command term; the standing task does not.
    is_velocity_task = hasattr(env_cfg, "commands")
    stages = select_stages(args_cli.stage, args_cli.stage_scale, args_cli.iterations) \
        if is_velocity_task else []

    if is_velocity_task:
        # Must happen before gym.make: friction/mass/CoM are startup terms.
        from isaaclab_m20.tasks.m20_velocity_env_cfg import set_domain_randomization
        enabled = set_domain_randomization(env_cfg, args_cli.dr == "full")
        print(f"[INFO] Domain randomization: "
              f"{'full' if enabled else 'off (nominal robot, no pushes)'}")
        # The commands drawn on the very first reset use whatever range the
        # config carries, and that happens during gym.make -- before the stage
        # schedule can intervene. Point the config at the first selected stage
        # so the first rollout is already on the right distribution.
        first_vx, first_vy, first_wz = stages[0][1].ranges_from
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-first_vx, first_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-first_vy, first_vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-first_wz, first_wz)
        agent_cfg.max_iterations = sum(stage.iterations for _, stage in stages)
    else:
        agent_cfg.max_iterations = args_cli.iterations or agent_cfg.max_iterations

    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = Path(args_cli.log_root, agent_cfg.experiment_name, run_name)
    print(f"[INFO] Logging to {log_root_path}")
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
    attach_diagnostics(env, args_cli.health_every, args_cli.zero_actions)

    schedule = None
    if is_velocity_task:
        manager = env.unwrapped.command_manager
        schedule = CommandSchedule(manager.get_term("base_velocity"), agent_cfg.num_steps_per_env)
        attach_stage_schedule(env, schedule)
    else:
        attach_dr_ramp(env, agent_cfg.max_iterations, agent_cfg.num_steps_per_env,
                       args_cli.dr_ramp)

    resume_path = Path(args_cli.resume_from) if args_cli.resume_from else None

    planned: list[tuple[str, int, Stage | None]] = []
    if stages:
        for stage_index, stage in stages:
            planned.append((stage.name, stage.iterations, stage))
    else:
        planned.append(("standing", agent_cfg.max_iterations, None))

    for position, (label, iterations, stage) in enumerate(planned):
        stage_dir = log_root_path if stage is None else log_root_path / label
        stage_dir.mkdir(parents=True, exist_ok=True)
        with open(stage_dir / "env.yaml", "w") as f:
            yaml.dump(env_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)
        with open(stage_dir / "agent.yaml", "w") as f:
            yaml.dump(agent_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)

        if resume_path is None and stage is not None:
            # Running one stage per invocation: pick up where the previous stage
            # of the same run left off, so "--stage 2" after "--stage 1" really
            # does continue rather than start from a random policy.
            stage_index = next(i for i, s in enumerate(STAGES) if s.name == stage.name)
            if stage_index > 0:
                previous = log_root_path / STAGES[stage_index - 1].name
                resume_path = latest_checkpoint(previous)
                if resume_path is not None:
                    print(f"[INFO] Stage {stage.name}: continuing from the previous "
                          f"stage's checkpoint {resume_path}")
                else:
                    print(f"[WARN] Stage {stage.name}: no checkpoint from "
                          f"{previous} -- starting from a fresh policy. Run "
                          f"--stage {stage_index} first, or pass --resume_from.")

        if resume_path is not None and not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"\n[INFO] ===== stage {label}: {iterations} iterations =====")
        if resume_path is not None:
            print(f"[INFO] Continuing from {resume_path}")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(stage_dir),
                                device=agent_cfg.device)
        if resume_path is not None:
            runner.load(str(resume_path))

        if schedule is not None and stage is not None:
            schedule.start_stage(stage)

        start = time.time()
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        print(f"[INFO] stage {label} finished in {(time.time() - start) / 60:.1f} min")

        # Stage N+1 continues from the checkpoint this stage ended on.
        found = latest_checkpoint(stage_dir)
        if found is None:
            raise RuntimeError(f"stage {label} produced no checkpoint in {stage_dir}")
        resume_path = found

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
