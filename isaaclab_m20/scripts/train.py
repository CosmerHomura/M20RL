"""Train M20 with rsl_rl PPO in Isaac Lab.

The velocity task trains as a six-stage course, each stage continuing from the
previous stage's checkpoint:

======  ==================  ========  =========================================
stage   name                iters     command range (half-widths)
======  ==================  ========  =========================================
1       stand               300       vx 0,           vy 0,      wz 0
2       forward_backward    700       vx +-0.4,       vy 0,      wz 0
3       lateral             1000      vx +-0.4,       vy +-0.2,  wz 0
4       yaw                 700       vx +-0.4,       vy +-0.2,  wz +-0.6
5       mixed               1250      vx +-0.5,       vy +-0.2,  wz +-0.8
6       static_dynamic      1000      vx +-0.5,       vy +-0.2,  wz +-0.8
======  ==================  ========  =========================================

4950 iterations total, in the order "stand -> forward/back -> lateral -> yaw ->
everything -> stop/go switching", each axis introduced on its own before it is
combined with the others.

Stage 6 keeps stage 5's ranges and changes the *distribution* instead: 35% of the
commands are "stop" (instead of 10%) and each command holds 1.5-3 s (instead of
2-4 s), so a 20 s episode contains several start/stop/side-step switches rather
than two or three long holds. That is what the masked terms in the reward were
built for -- the same policy has to be stiff when the command is zero and
permissive when it is not, within one episode.

Stage 1 is a zero-command stage: the tracking terms are then "be still", so it
trains the baseline the later stages build on -- hold the ride height, keep the
body level, do not spin the wheels, do not drift. That is the answer to the "we
already have a standing policy" problem: that policy was trained on 57
observations and cannot be loaded into a 56-observation actor, so the course has
to re-establish standing under the new observation layout first. Keeping it as
its own stage (rather than as a warmup inside stage 2) gives it its own
checkpoint, which is what makes the later stages comparable: if forward/backward
never improves on stage 1, the problem is the command, not the baseline.

Every stage that introduces a new axis opens with a short warmup during which
that axis is interpolated from zero up to its target width over 150-250
iterations, instead of appearing at full width on the first step. Stage 6 has no
warmup: its ranges are stage 5's, only the sampling distribution changes.

Two log-directory rules, both about the "train a stage, look at it, decide"
loop:

* Retraining a stage overwrites it. A single ``--stage N`` reuses the newest run
  of the experiment and clears that stage's directory first, so a second
  attempt at stage 2 replaces the first instead of leaving two sets of
  checkpoints whose iteration numbers disagree. ``--new_run`` keeps the earlier
  attempt as a separate timestamped run instead, and ``--keep_logs`` appends to
  the existing stage directory.
* A stage starts from the previous stage's newest checkpoint in the same run.
  That is what makes the loop safe to interrupt: stage N+1 picks up whatever
  stage N actually produced, and ``--resume_from`` overrides it for one run.

Example runs::

    # the whole course in one process, 4096 envs (the default)
    python -m isaaclab_m20.scripts.train

    # one stage at a time -- the recommended loop: train, look at the result in
    # the simulator (play / eval), decide, then spend the next stage's budget.
    # --stage N automatically continues from stage N-1's latest checkpoint, and
    # --iterations is how long to train it (--iters_stageN means the same thing
    # here, so don't pass both).
    python -m isaaclab_m20.scripts.train --stage 1
    python -m isaaclab_m20.scripts.train --stage 2
    python -m isaaclab_m20.scripts.train --stage 2 --iterations 1200

    # --stage all runs the six stages back to back; only then do per-stage
    # budgets differ, and that is what --iters_stageN is for
    python -m isaaclab_m20.scripts.train --stage all --iters_stage1 200 \
        --iters_stage2 600 --iters_stage3 900 --iters_stage4 600 \
        --iters_stage5 1000 --iters_stage6 800

    # fewer envs only when the GPU is smaller than 24 GB, or for a smoke test
    python -m isaaclab_m20.scripts.train --num_envs 128

    # smoke test: 20 iterations of the standing stage, about a minute
    python -m isaaclab_m20.scripts.train --stage 1 --iterations 20 --num_envs 128

The legacy standing task (``Isaac-M20-Wheeled-v0``) has no command term, so for
it this script runs a single plain run, as before.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
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

    ``zero_prob`` and ``resampling`` override the command term's sampling for
    this stage (``None`` keeps the config default: 0.10 and 2-4 s). Stage 6 uses
    them to change the *distribution* -- more stop commands, shorter holds --
    while keeping stage 5's ranges, which is what makes it a stop/go switching
    stage rather than a wider-range stage.
    """

    name: str
    iterations: int
    ranges_from: tuple[float, float, float]
    ranges_to: tuple[float, float, float]
    warmup: int
    zero_prob: float | None = None
    resampling: tuple[float, float] | None = None


STAGES: tuple[Stage, ...] = (
    # Zero command: the standing baseline under the new 56-dim observation
    # layout. Every term still runs; the tracking terms just mean "be still".
    Stage("1_stand", 300, (0.00, 0.0, 0.0), (0.00, 0.0, 0.0), 0),
    Stage("2_forward_backward", 700, (0.00, 0.0, 0.0), (0.40, 0.0, 0.0), 150),
    Stage("3_lateral", 1000, (0.40, 0.0, 0.0), (0.40, 0.2, 0.0), 250),
    Stage("4_yaw", 700, (0.40, 0.2, 0.0), (0.40, 0.2, 0.6), 250),
    Stage("5_mixed", 1250, (0.40, 0.2, 0.6), (0.50, 0.2, 0.8), 250),
    # Same full range as stage 5, different distribution: one command in three
    # is "stop" and holds are shorter, so the policy has to start, stop and
    # change direction several times per episode.
    Stage("6_static_dynamic", 1000, (0.50, 0.2, 0.8), (0.50, 0.2, 0.8), 0,
          zero_prob=0.35, resampling=(1.5, 3.0)),
)

_LAST_STAGE = len(STAGES)


def stage_by_number(number: int) -> Stage:
    """Course stage by its 1-based number, with a readable error on a typo."""
    if not 1 <= number <= _LAST_STAGE:
        raise SystemExit(f"--stage must be 1..{_LAST_STAGE} or 'all', got {number}")
    return STAGES[number - 1]


def stage_label(number: int, stage: Stage) -> str:
    """Checkpoint directory for a stage, e.g. ``stage3_lateral``.

    The number is part of the directory name because the course order is what
    makes the checkpoints comparable, and it is spelled out here (one place)
    because the log tree, the resume lookup, and the docs must agree -- they did
    not, and "find the checkpoint under stage1_stand" pointed at a directory
    that was written as ``1_stand``.
    """
    _, _, slug = stage.name.partition("_")
    return f"stage{number}_{slug or stage.name}"


parser = argparse.ArgumentParser(description="Train M20 wheeled locomotion with rsl_rl.")
parser.add_argument("--task", type=str, default=VELOCITY_TASK)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override the number of environments. Both tasks default to 4096, "
                         "which a 24 GB card runs comfortably; drop it (e.g. 128) only for a "
                         "smoke test.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stage", type=str, default="all",
                    help=f"Velocity task: which course stage to run "
                         f"('1'..'{_LAST_STAGE}' or 'all'). "
                         "Each stage continues from the previous stage's latest checkpoint "
                         "unless --resume_from says otherwise.")
parser.add_argument("--stage_scale", type=float, default=1.0,
                    help="Multiply every stage's iteration budget (e.g. 0.05 for a smoke test).")
parser.add_argument("--iterations", type=int, default=None,
                    help="Iteration budget for the stage(s) being run. With a single "
                         "--stage this is the normal way to say how long to train it "
                         "(--stage 2 --iterations 1200). --iters_stageN is for --stage all, "
                         "where several stages run back to back with different budgets.")
for _number, _course_stage in enumerate(STAGES, start=1):
    parser.add_argument(
        f"--iters_stage{_number}", type=int, default=None,
        help=f"Iterations for stage {_number} ({_course_stage.name}) inside a --stage all "
             f"run; default {_course_stage.iterations} x --stage_scale. When you run that "
             f"stage alone, --iterations does the same job more directly.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Checkpoint for the first selected stage. Without it, stage 1 starts "
                         "from scratch and stages 2-6 continue from the previous stage.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Sub-directory inside logs/rsl_rl/<experiment>; stages get their own "
                         "sub-sub-directory inside it. Without it a single --stage reuses the "
                         "newest run of that experiment, so re-running a stage retrains (and "
                         "overwrites) that stage in place.")
parser.add_argument("--new_run", action="store_true",
                    help="Start a new timestamped run directory instead of reusing the newest "
                         "one. Use this to keep an earlier attempt of the same stage for "
                         "comparison; without it, the same --stage overwrites that stage.")
parser.add_argument("--keep_logs", action="store_true",
                    help="Keep an existing stage directory when retraining it. Off by default: "
                         "stale model_<iter>.pt files from the previous attempt would otherwise "
                         "be picked up as this stage's result and as stage N+1's starting point.")
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


def select_stages(
    selector: str,
    scale: float,
    iterations: int | None,
    per_stage: dict[int, int | None] | None = None,
) -> list[tuple[int, Stage]]:
    """Which stages this run executes, with the iteration budget applied.

    The budget of stage ``N`` comes from, in order of precedence: ``per_stage[N]``
    (``--iters_stageN``), then ``iterations`` (``--iterations`` for every selected
    stage), then the stage's default budget times ``--stage_scale``.

    Returns ``(index_in_STAGES, stage)`` pairs so the caller can find the
    preceding stage's checkpoint for the first selected stage.
    """
    per_stage = per_stage or {}
    if selector == "all":
        indices = list(range(len(STAGES)))
    else:
        try:
            number = int(selector)
        except ValueError as exc:
            raise SystemExit(f"--stage must be 1..{len(STAGES)} or 'all', got {selector!r}") from exc
        stage_by_number(number)  # validates the 1..N range
        indices = [number - 1]

    out: list[tuple[int, Stage]] = []
    for index in indices:
        stage = STAGES[index]
        explicit = per_stage.get(index)
        if explicit is not None:
            budget = explicit
        elif iterations is not None:
            budget = iterations
        else:
            budget = int(round(stage.iterations * scale))
        if budget <= 0:
            raise SystemExit(f"stage {stage.name} would run {budget} iterations")
        # Never spend more than half of a trimmed stage on the warmup: with
        # --stage_scale 0.05 the budget is 35 iterations for stage 2 and a
        # 150-iteration warmup would hold the new axis near zero the whole run.
        # A stage with no warmup (stage 1) keeps 0, so it never steals iterations.
        warmup = min(stage.warmup, budget // 2) if stage.warmup else 0
        out.append((index, Stage(stage.name, budget, stage.ranges_from, stage.ranges_to,
                                 warmup, stage.zero_prob, stage.resampling)))
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
        # Sampling defaults, restored for stages that do not override them: a
        # stage 6 override must not leak into a later run of another stage.
        self._base_zero_prob = getattr(command_term.cfg, "zero_prob", None)
        self._base_resampling = command_term.cfg.resampling_time_range
        dr.register_command_term(command_term)

    # -- stage lifecycle ---------------------------------------------------
    def start_stage(self, stage: Stage) -> None:
        self.stage = stage
        self.iteration = 0
        self._steps = 0
        self._applied = None
        self.apply_sampling(stage, announce=True)
        self._apply(stage.ranges_from, announce=True)
        # The buffer still holds commands drawn under the previous stage's
        # range (up to 4 s of them). For stage 1 the new range starts at zero,
        # so wipe them: the robot must not be asked for 0.4 m/s on step 0.
        if max(stage.ranges_from) == 0.0:
            dr.zero_command(self.term)

    def apply_sampling(self, stage: Stage, announce: bool = False) -> None:
        """Apply the stage's command-distribution overrides (zero_prob, hold time).

        ``MixedVelocityCommand`` reads both off ``self.cfg`` on every resample, so
        writing them here takes effect from the next resample on -- no need to
        recreate the term. Stage 6 uses this to hold the same ranges as stage 5
        while flipping far more commands to "stop" and shortening the holds.
        """
        cfg = self.term.cfg
        if stage.zero_prob is not None:
            cfg.zero_prob = float(stage.zero_prob)
        elif self._base_zero_prob is not None:
            cfg.zero_prob = self._base_zero_prob
        if stage.resampling is not None:
            cfg.resampling_time_range = (float(stage.resampling[0]), float(stage.resampling[1]))
        else:
            cfg.resampling_time_range = self._base_resampling
        if hasattr(self.term, "_resampling_time_range"):
            # The command term caches the tuple at construction in some versions;
            # keep the private copy in sync when it exists.
            self.term._resampling_time_range = cfg.resampling_time_range
        if announce:
            name = stage.name
            zero_prob = getattr(cfg, "zero_prob", None)
            print(f"[cmd] stage={name} hold={cfg.resampling_time_range[0]:.2f}-"
                  f"{cfg.resampling_time_range[1]:.2f}s"
                  + (f" zero_prob={zero_prob:.2f}" if zero_prob is not None else ""),
                  flush=True)

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

    For the velocity task the line also carries ``|v_xy|`` (mean body speed, the
    quantity the static-hold terms act on) and ``standing`` (the fraction of
    envs whose command is zero, i.e. the mask those terms multiply). A stage-1
    run that wanders shows up there immediately; ``standing=0.00`` would mean the
    mask is broken rather than the reward being too weak.
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
            # "does the robot actually stand still when told to" is otherwise
            # only visible in tensorboard hours later: |v_xy| is the quantity the
            # static-hold terms act on, and `standing` is the mask they use. If
            # standing is 0.00 during stage 1, the mask is broken, not the reward.
            speed_xy = robot.data.root_lin_vel_b[:, :2].norm(dim=1).mean()
            extra = f"  |v_xy|={speed_xy.item():.3f}"
            command_manager = getattr(unwrapped, "command_manager", None)
            if command_manager is not None:
                try:
                    term = command_manager.get_term("base_velocity")
                except KeyError:
                    term = None
                flag = getattr(term, "is_standing_env", None) if term else None
                if flag is not None:
                    extra += f"  standing={flag.float().mean().item():.2f}"
            print(
                f"[health] step={state['n']:8d}"
                f"  max|qdot|={joint_vel.abs().max().item():10.2f}"
                f"  max|v|={root_vel.item():8.2f}"
                f"  z=[{root_z.min().item():.3f},{root_z.max().item():.3f}]"
                f"{extra}"
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


def latest_run_dir(experiment_dir: Path) -> Path | None:
    """Newest run directory under ``logs/rsl_rl/<experiment>/``."""
    if not experiment_dir.is_dir():
        return None
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def rename_final_checkpoint(stage_dir: Path, last_iteration: int | None) -> Path | None:
    """Rename the stage's final checkpoint to the number of iterations it completed.

    rsl_rl names the checkpoints after the loop *index*, and its final save uses
    that index too, so ``--iterations 300`` ends with ``model_299.pt``: the file
    name is one less than the number of iterations that were actually run. The
    loader does not care (the counter lives inside the checkpoint), but the
    directory listing then lies about the budget, and reading
    ``model_999.pt`` after asking for 1000 iterations looks like a truncated run.

    Only the exact off-by-one is corrected. If the newest file is some other
    number -- a mid-training save, because the harness saved nothing at the end
    -- it is left untouched rather than relabelled with an iteration count it
    does not contain.
    """
    newest = latest_checkpoint(stage_dir)
    if newest is None or last_iteration is None:
        return newest
    completed = last_iteration + 1
    try:
        number = int(newest.stem.split("_")[-1])
    except ValueError:
        return newest
    if number + 1 != completed:
        return newest
    target = newest.with_name(f"model_{completed}.pt")
    if target.exists():
        return newest
    newest.rename(target)
    print(f"[INFO] final checkpoint renamed to {target.name} "
          f"({completed} iterations completed)")
    return target


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
    per_stage = {index: getattr(args_cli, f"iters_stage{index + 1}") for index in range(len(STAGES))}
    stages = select_stages(args_cli.stage, args_cli.stage_scale, args_cli.iterations, per_stage) \
        if is_velocity_task else []

    if is_velocity_task:
        # Must happen before gym.make: friction/mass/CoM are startup terms.
        from isaaclab_m20.tasks.m20_velocity_env_cfg import set_domain_randomization
        enabled = set_domain_randomization(env_cfg, args_cli.dr == "full")
        print(f"[INFO] Domain randomization: "
              f"{'full' if enabled else 'off (nominal robot, no pushes)'}")
        # One stage per invocation: --iterations is the way to set its length;
        # --iters_stageN only matters for --stage all.
        if args_cli.stage != "all" and len(stages) == 1:
            running = stages[0][0] + 1
            ignored = [n for n, value in per_stage.items()
                       if value is not None and n + 1 != running]
            if ignored:
                flags = ", ".join(f"--iters_stage{n + 1}" for n in sorted(ignored))
                print(f"[WARN] --stage {running} ignores {flags} (other stages are not "
                      f"running); use --iterations for the length of this stage.")
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

    if stages:
        total = sum(stage.iterations for _, stage in stages)
        print(f"[INFO] Course plan: {len(stages)} stage(s), {total} iterations "
              f"({env_cfg.scene.num_envs} envs x {agent_cfg.num_steps_per_env} steps/env)")
        for number, stage in stages:
            sampling = []
            if stage.zero_prob is not None:
                sampling.append(f"zero_prob {stage.zero_prob:.2f}")
            if stage.resampling is not None:
                sampling.append(f"hold {stage.resampling[0]:.1f}-{stage.resampling[1]:.1f}s")
            extra = f"  [{', '.join(sampling)}]" if sampling else ""
            print(f"    {stage_label(number + 1, stage):<24} {stage.iterations:5d} it  "
                  f"vx=+-{stage.ranges_to[0]:.2f} vy=+-{stage.ranges_to[1]:.2f} "
                  f"wz=+-{stage.ranges_to[2]:.2f}  warmup {stage.warmup}{extra}")

    experiment_dir = Path(args_cli.log_root, agent_cfg.experiment_name)
    if args_cli.run_name:
        run_name = args_cli.run_name
    elif args_cli.new_run:
        # Explicit opt-in to a second directory: the earlier attempt of the same
        # stage stays on disk for comparison instead of being overwritten.
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("[INFO] --new_run: starting a fresh run directory")
    elif is_velocity_task and args_cli.stage != "all":
        # One stage per invocation: stay inside the run the previous stage
        # wrote to. A fresh timestamped directory would make "--stage 2" search
        # for stage 1's checkpoint in an empty folder and silently start over.
        previous_run = latest_run_dir(experiment_dir)
        run_name = previous_run.name if previous_run else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if previous_run is not None:
            print(f"[INFO] Continuing run {previous_run} "
                  f"(a stage that is retrained overwrites its own directory; "
                  f"--new_run starts a separate one instead)")
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = experiment_dir / run_name
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
            planned.append((stage_label(stage_index + 1, stage), stage.iterations, stage))
    else:
        planned.append(("standing", agent_cfg.max_iterations, None))

    for position, (label, iterations, stage) in enumerate(planned):
        stage_dir = log_root_path if stage is None else log_root_path / label
        if stage is not None and not args_cli.keep_logs and stage_dir.exists() \
                and any(stage_dir.iterdir()):
            # Retraining a stage overwrites it by default. The old checkpoints
            # have to go: latest_checkpoint() would otherwise hand a higher
            # iteration number from the previous attempt to stage N+1 as if it
            # were this attempt's final model.
            old = sorted(stage_dir.rglob("model_*.pt"))
            print(f"[INFO] {label}: overwriting the previous attempt "
                  f"({len(old)} checkpoint(s) removed from {stage_dir})")
            shutil.rmtree(stage_dir)
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
                previous = log_root_path / stage_label(stage_index, STAGES[stage_index - 1])
                resume_path = latest_checkpoint(previous)
                if resume_path is not None:
                    print(f"[INFO] Stage {stage.name}: continuing from the previous "
                          f"stage's checkpoint {resume_path}")
                else:
                    print(f"[WARN] Stage {stage.name}: no checkpoint from "
                          f"{previous} -- starting from a fresh policy. Run "
                          f"--stage {stage_index} first, or pass --resume_from "
                          f"(a --new_run directory starts empty, so it has to be "
                          f"filled from stage 1).")

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

        # Stage N+1 continues from the checkpoint this stage ended on. rsl_rl
        # names its last save after the loop index, one below the iteration
        # count, so fix the name before it becomes anything's starting point.
        found = rename_final_checkpoint(
            stage_dir, getattr(runner, "current_learning_iteration", None))
        if found is None:
            raise RuntimeError(f"stage {label} produced no checkpoint in {stage_dir}")
        resume_path = found

        if stage is not None:
            number = next(i for i, s in enumerate(STAGES) if s.name == stage.name) + 1
            print(f"\n[INFO] Stage {label} done -- checkpoint {found}")
            print("[INFO] Look at it in the simulator before spending the next stage:")
            print(f"    python -m isaaclab_m20.scripts.play --stage {number}"
                  f"    # = --checkpoint '{found}'")
            print(f"    python -m isaaclab_m20.scripts.eval --stage {number}")
            if number < _LAST_STAGE:
                print(f"    then: python -m isaaclab_m20.scripts.train --stage {number + 1}"
                      f"   (continues from this checkpoint)")
            else:
                print("[INFO] That was the last stage of the course.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
