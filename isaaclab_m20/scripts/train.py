"""Train M20 with rsl_rl PPO in Isaac Lab.

The velocity task trains as a four-stage course, each stage continuing from the
previous stage's checkpoint:

======  ==================  ========  ================================  =========
stage   name                iters     command range (half-widths)       zero cmd
======  ==================  ========  ================================  =========
1       stand               900       vx 0,        vy 0,     wz 0        100%
2       forward_backward    700       vx +-0.8,    vy 0,     wz 0        50%
3       multi_direction     1200      vx +-1.4,    vy +-0.4, wz +-0.8    30%
4       agile               1400      vx +-2.0,    vy +-0.5, wz +-1.0    20%
======  ==================  ========  ================================  =========

4200 iterations total, in the order "stand -> forward/back -> everything ->
faster and stop/go", one axis group introduced per stage.

The speed envelope is 2.0 m/s, set by the wheel action scale: ``scale=22.2``
rad/s times the 0.09 m wheel. It is a policy choice, not a hardware limit (the
actuator allows 200 rad/s and 240 N per wheel against a 338 N friction ceiling);
the earlier ``scale=8.0`` capped the robot at 0.72 m/s, slower than walking, and
that cap was the reason the whole course looked slow. Changing it changes the
meaning of the four wheel actions, so the course restarts from stage 1.

Stage 1 is budgeted at 900 to cover the *fixed*-rate mode (``--fixed_lr``), where
the standing curve only flattens out around iteration 700-900; with the default
adaptive schedule it converges much sooner (m20_v5 hit the same plateau at
~200 iterations because the scheduler had pushed lr to 1.5e-3), so treat 900 as
an upper bound and stop the stage when the curve is flat. The budget cannot be
separated from the rate: change ``learning_rate`` or ``--fixed_lr`` and the
iteration count that matters changes with it -- judge by the curve, not the
budget.

The last column is the point of the course. Every stage after the first keeps a
share of zero commands, so the standing skill is rehearsed while the moving
skills are learned -- without it, stage 2 and later would train the "stop"
behaviour out of the actor and it would have to be relearned (that was the bug
behind "stage 4 falls over the moment `eval` holds still"). The share shrinks
stage by stage, so the emphasis moves from standing to tracking, and stage 4
keeps 20% stops with 1.5-3 s holds: one 20 s episode then contains several
start/stop/side-step switches rather than two or three long holds.

Stage 1 is 100% zero command: the tracking terms are then "be still", so it
trains the baseline the later stages build on -- hold the ride height, keep the
body level, do not spin the wheels, do not drift. That is the answer to the "we
already have a standing policy" problem: that policy was trained on 57
observations and cannot be loaded into a 56-observation actor, so the course has
to re-establish standing under the new observation layout first. Keeping it as
its own stage (rather than as a warmup inside stage 2) gives it its own
checkpoint, which is what makes the later stages comparable: if forward/backward
never improves on stage 1, the problem is the command, not the baseline.

Every stage that introduces a new axis or widens an existing one opens with a
warmup during which the range is interpolated linearly over 300-400 iterations,
instead of appearing at full width on the first step. ``--warmup_frac`` turns that
into a share of the stage's budget, which is what a stage given a much larger
budget than planned needs (``--stage 2 --iterations 4000 --warmup_frac 0.5``
spends its first 2000 iterations growing vx to +-0.8).

Two log-directory rules, both about the "train a stage, look at it, decide"
loop:

* Re-running a stage continues it. The stage directory is reused and the newest
  ``model_*.pt`` in it is loaded, then exactly ``--iterations`` iterations run
  (default: the stage's budget). The script does not subtract what the stage has
  already done and never decides that a stage has had enough -- the operator
  decides. Nothing is ever deleted. To retrain a stage from scratch, pass
  ``--overwrite``: the old attempt is renamed to ``<stage>.bak-<timestamp>``, so
  an accidental retrain can be undone.
* A continued stage gets the environment it stopped with: the command ramp is
  put back at the stage iteration it reached (an interrupted warmup resumes
  mid-ramp instead of jumping to the full range), the sampling distribution is
  the stage's, and the cfg is compared against the ``env.yaml`` written by the
  first attempt -- any difference (command range, reward weights, domain
  randomisation, env count) is printed as a warning, and the current config is
  saved as ``env.resume-<stamp>.yaml`` so the original record survives.
* A stage that does not exist yet starts from the previous stage's newest
  checkpoint in the same run. ``--new_run`` starts a separate timestamped run and
  seeds it from the previous stage of whichever run has one.

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

    # --stage all runs the four stages back to back; only then do per-stage
    # budgets differ, and that is what --iters_stageN is for
    python -m isaaclab_m20.scripts.train --stage all --iters_stage1 200 \
        --iters_stage2 600 --iters_stage3 900 --iters_stage4 1000

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
    this stage (``None`` keeps the config default: 0.10 and 2-4 s). Every stage
    after the first uses ``zero_prob`` to keep a share of stop commands, which is
    what stops the later stages from training the standing behaviour out of the
    actor; stage 4 also shortens the hold so one episode contains several
    start/stop switches.
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
    Stage("1_stand", 900, (0.00, 0.0, 0.0), (0.00, 0.0, 0.0), 0, zero_prob=1.0),
    # Half the commands stay zero, so standing is rehearsed while the first
    # moving axis is learned. 0.8 m/s is 40% of the 2.0 m/s envelope: enough to
    # force a real gait, slow enough that falling is not the first lesson.
    Stage("2_forward_backward", 700, (0.00, 0.0, 0.0), (0.80, 0.0, 0.0), 300,
          zero_prob=0.5),
    # Lateral and yaw arrive together: they are the two axes the wheels cannot
    # produce on their own and the ones that break the standing skill first.
    # vx is already at 0.8 and grows to 1.4 while vy/wz grow in from 0.
    Stage("3_multi_direction", 1200, (0.80, 0.0, 0.0), (1.40, 0.40, 0.80), 400,
          zero_prob=0.3),
    # The last 0.6 m/s of vx plus 20% stops and shorter holds: accelerate, stop,
    # turn and side-step several times per episode, at the top of the envelope.
    Stage("4_agile", 1400, (1.40, 0.40, 0.80), (2.00, 0.50, 1.00), 300,
          zero_prob=0.2, resampling=(1.5, 3.0)),
)

_LAST_STAGE = len(STAGES)


def stage_by_number(number: int) -> Stage:
    """Course stage by its 1-based number, with a readable error on a typo."""
    if not 1 <= number <= _LAST_STAGE:
        raise SystemExit(f"--stage must be 1..{_LAST_STAGE} or 'all', got {number}")
    return STAGES[number - 1]


def stage_label(number: int, stage: Stage) -> str:
    """Checkpoint directory for a stage, e.g. ``stage3_multi_direction``.

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
                         "A stage that already has a run is continued from its newest "
                         "checkpoint; a new stage continues from the previous one.")
parser.add_argument("--stage_scale", type=float, default=1.0,
                    help="Multiply every stage's iteration budget (e.g. 0.05 for a smoke test).")
parser.add_argument("--iterations", type=int, default=None,
                    help="How many iterations to run now. Plain and literal: it is the "
                         "same number whether the stage is new or being continued, and "
                         "the script never decides on its own that a stage has had enough. "
                         "--iters_stageN is only for --stage all, "
                         "where several stages run back to back with different budgets.")
parser.add_argument("--warmup_frac", type=float, default=None,
                    help="Length of the command ramp as a fraction of this stage's budget "
                         "(0.5 = the first half grows linearly, the rest samples across the "
                         "full range). Default: the stage's own warmup, capped at half the "
                         "budget. Only stages that declare a warmup are affected.")
for _number, _course_stage in enumerate(STAGES, start=1):
    parser.add_argument(
        f"--iters_stage{_number}", type=int, default=None,
        help=f"Iterations for stage {_number} ({_course_stage.name}) inside a --stage all "
             f"run; default {_course_stage.iterations} x --stage_scale. When you run that "
             f"stage alone, --iterations does the same job more directly.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Sub-directory inside logs/rsl_rl/<experiment>; stages get their own "
                         "sub-sub-directory inside it. Without it a single --stage reuses the "
                         "newest run of that experiment.")
parser.add_argument("--new_run", action="store_true",
                    help="Start a new timestamped run directory instead of reusing the newest "
                         "one. Use this to keep an earlier attempt of the same stage for "
                         "comparison.")
parser.add_argument("--overwrite", action="store_true",
                    help="Retrain a stage from scratch instead of continuing it. The previous "
                         "attempt is RENAMED to <stage>.bak-<timestamp>, never deleted, so a "
                         "mistaken retrain can be undone. Without this flag, re-running a stage "
                         "continues it: newest checkpoint loaded, command ramp put back where "
                         "it stopped, current cfg compared against the stage's env.yaml.")
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
    "--per_term_every", type=int, default=200,
    help="Print the per-step reward decomposition (value*weight*dt for every term, "
         "plus the accumulated episode sums) every N env steps. This is the only "
         "scale in which the reward weights are comparable; 0 disables.")
parser.add_argument(
    "--dr", type=str, default="off", choices=("off", "full"),
    help="Velocity task only: 'off' trains on the nominal robot (flat ground, "
         "friction 1.0, no payload spread, no pushes) -- the first version's setting. "
         "'full' enables the friction/mass/CoM ranges and the shoves.")
parser.add_argument(
    "--dr_ramp", type=float, default=0.4,
    help="Standing task only (no command term): fraction of training over which the "
         "domain randomization ramps from 0 to full.")
parser.add_argument(
    "--fixed_lr", action="store_true",
    help="Pin the learning rate at ``learning_rate`` (schedule='fixed') instead of "
         "the cfg's adaptive-in-KL schedule. Adaptive is the default because it is "
         "the plan's setting, but it also swings Adam's step size (lr moved between "
         "~0 and ~2e-3 in m20_v5); with --fixed_lr the rate stays at 3e-4 and a "
         "stage needs a longer budget before it plateaus.")
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
    warmup_frac: float | None = None,
) -> list[tuple[int, Stage]]:
    """Which stages this run executes, with the iteration budget applied.

    The budget of stage ``N`` comes from, in order of precedence: ``per_stage[N]``
    (``--iters_stageN``), then ``iterations`` (``--iterations`` for every selected
    stage), then the stage's default budget times ``--stage_scale``.

    ``warmup_frac`` (``--warmup_frac``) turns the ramp into a fraction of that
    budget, which is what a stage that is given a much bigger budget than planned
    needs: stage 2 over 4000 iterations should spend the first 2000 growing the
    range instead of reaching +-0.4 after 150 and sampling there for 39 minutes.

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
        # 300-iteration warmup would hold the new axis near zero the whole run.
        # A stage with no warmup (stage 1) keeps 0, so it never steals iterations.
        # --warmup_frac replaces the absolute number with a share of the budget;
        # it still cannot exceed the budget, and stages without a ramp stay flat.
        if warmup_frac is not None and stage.warmup:
            warmup = min(int(round(budget * warmup_frac)), budget)
        else:
            warmup = min(stage.warmup, budget // 2) if stage.warmup else 0
        out.append((index, Stage(stage.name, budget, stage.ranges_from, stage.ranges_to,
                                 warmup, stage.zero_prob, stage.resampling)))
    return out


def interpolate_ranges(stage: Stage, done: int,
                       warmup: int | None = None) -> tuple[float, float, float]:
    """Command range after ``done`` iterations of ``stage`` (linear warmup).

    Shared by the schedule and by the pre-``gym.make`` setup, so the range the
    env is built with equals the range the schedule will apply on step one.
    """
    span = stage.warmup if warmup is None else max(int(warmup), 0)
    if span <= 0:
        return stage.ranges_to
    t = min(max(done / span, 0.0), 1.0)
    return tuple(
        low + (high - low) * t
        for low, high in zip(stage.ranges_from, stage.ranges_to)
    )


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
        self.warmup = 0
        # How far into the stage the previous run already got. Resuming has to
        # continue the ramp from there, not restart it: a stage stopped at 60% of
        # its warmup must resume on the range it was actually training at.
        self.offset = 0
        self._steps = 0
        self._applied: tuple[float, float, float] | None = None
        # Sampling defaults, restored for stages that do not override them: a
        # stage 4 override must not leak into a later run of another stage.
        self._base_zero_prob = getattr(command_term.cfg, "zero_prob", None)
        self._base_resampling = command_term.cfg.resampling_time_range
        dr.register_command_term(command_term)

    # -- stage lifecycle ---------------------------------------------------
    def start_stage(self, stage: Stage, warmup: int | None = None,
                    resume_iteration: int = 0) -> None:
        """Enter ``stage``, optionally continuing it from ``resume_iteration``.

        ``resume_iteration`` is how many iterations this stage had already done,
        so the env comes back exactly where training stopped: same command range
        (mid-warmup ranges stay mid-warmup) and same sampling distribution. Pass
        0 for a fresh stage.

        ``warmup`` overrides the stage's own (possibly ``--stage_scale``-trimmed)
        warmup; when continuing, pass the *untrimmed* one so the ramp the stage was
        climbing stays the same ramp.
        """
        self.stage = stage
        self.warmup = stage.warmup if warmup is None else max(int(warmup), 0)
        self.offset = max(int(resume_iteration), 0)
        self.iteration = 0
        self._steps = 0
        self._applied = None
        self.apply_sampling(stage, announce=True)
        self._apply(self.ranges_at(self.offset), announce=True)
        # The buffer still holds commands drawn under the previous stage's
        # range (up to 4 s of them). For stage 1 the new range starts at zero,
        # so wipe them: the robot must not be asked for 0.4 m/s on step 0.
        if self.offset == 0 and max(stage.ranges_from) == 0.0:
            dr.zero_command(self.term)

    def apply_sampling(self, stage: Stage, announce: bool = False) -> None:
        """Apply the stage's command-distribution overrides (zero_prob, hold time).

        ``MixedVelocityCommand`` reads both off ``self.cfg`` on every resample, so
        writing them here takes effect from the next resample on -- no need to
        recreate the term. The course uses this to keep a share of stop commands
        alive in every stage after the first, and to shorten the holds in stage 4
        so an episode contains several start/stop switches.
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
        """Range after ``iteration`` iterations of the current stage.

        The stage's own offset is included, so resuming mid-warmup keeps the
        range it was training at instead of jumping to ``ranges_to``.
        """
        assert self.stage is not None
        stage = self.stage
        return interpolate_ranges(stage, getattr(self, "offset", 0) + iteration,
                                  getattr(self, "warmup", stage.warmup))

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
            print(f"[cmd] stage_iter={self.offset + self.iteration:5d} "
                  f"vx=+-{r[0]:.3f} vy=+-{r[1]:.3f} wz=+-{r[2]:.3f}", flush=True)


def attach_stage_schedule(env, schedule: CommandSchedule) -> None:
    original_step = env.step

    def step(action):
        out = original_step(action)
        schedule.on_step()
        return out

    env.step = step


def reward_term_contributions(env, manager) -> dict[str, float] | None:
    """Per-step contribution of every reward term: ``value * weight * dt``.

    ``_episode_sums`` answers "what did this term contribute over the episode",
    which is what tensorboard plots -- but it is dt-scaled and keeps growing
    until the episode ends, so a term can look healthy while its per-step value
    is already negative (or the other way round). This evaluates each term once
    more and returns the number that actually enters ``Train/mean_reward`` every
    step, which is the only scale in which the weights are comparable.

    The extra evaluation is read-only: every term in ``rewards.py`` only reads
    ``robot.data`` / the command buffer, so calling it twice in a step is safe.
    Terms that raise are skipped rather than taking the run down.
    """
    term_cfgs = getattr(manager, "_term_cfgs", None)
    if not term_cfgs:
        return None
    dt = getattr(env, "step_dt", None)
    if dt is None:
        dt = getattr(manager, "_dt", None)
    if dt is None:
        return None
    names = getattr(manager, "_term_names", None)
    if not names or len(names) != len(term_cfgs):
        names = [getattr(term_cfg, "name", f"term{index}")
                 for index, term_cfg in enumerate(term_cfgs)]
    contributions: dict[str, float] = {}
    for name, term_cfg in zip(names, term_cfgs):
        # Class-based terms keep state inside an instance the manager owns; calling
        # the class again would build a fresh one and report zeros.
        if isinstance(term_cfg.func, type):
            continue
        try:
            value = term_cfg.func(env, **term_cfg.params)
            value = torch.as_tensor(value, dtype=torch.float32)
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            continue
        contributions[name] = float((value * term_cfg.weight * dt).mean().item())
    return contributions or None


def episode_sums(manager) -> dict[str, float] | None:
    """Mean of ``_episode_sums`` per term: the accumulated episode reward.

    This is the number tensorboard's ``Episode_Reward/*`` is built from, so
    printing it next to the per-step contributions lets one check that the two
    agree in sign and order of magnitude.
    """
    sums = getattr(manager, "_episode_sums", None)
    if not sums:
        return None
    values: dict[str, float] = {}
    for name, tensor in sums.items():
        try:
            values[name] = float(tensor.mean().item())
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            continue
    return values or None


def format_terms(values: dict[str, float], unit: str) -> str:
    """One line, biggest contributors first, so the tail of the list is quiet."""
    ordered = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
    body = "  ".join(f"{name}={value:+.3f}" for name, value in ordered)
    return f"[terms] per-step ({unit}): {body}  sum={sum(values.values()):+.3f}"


def attach_diagnostics(env, every: int, zero_actions: bool, per_term_every: int = 0) -> None:
    """Log simulation health, optionally replacing the policy output with zeros.

    PhysX reports a divergence as a bare ``CUDA error 700`` (illegal memory
    access) with no hint about which quantity blew up, and the CUDA context is
    dead by the time it surfaces. Printing the worst-case joint speed and base
    height every iteration makes the growth visible a few seconds before the
    crash instead of after it.

    For the velocity task the line also carries ``|v_xy|`` (mean body speed -- the
    quantity the tracking kernel prices), ``standing`` (the fraction of envs whose
    command buffer is zero, which is both the stage-1 set point and the gate that
    ``wheel_speed`` multiplies), and ``rew``/``neg`` (the per-step reward and the
    fraction of envs where it is negative). A stage-1 run that wanders shows up in
    ``|v_xy|`` immediately: the command is zero, so anything above ~0.05 m/s is
    drift. ``standing=0.00`` would mean the command buffer is not zero rather than
    the reward being too weak. ``neg=1.00`` is *not* an error by itself -- with no
    reward floor a bad policy legitimately earns a negative step -- but if it sits
    at 1.00 for a whole stage while ``Train/mean_reward`` does not rise, the
    command range is ahead of the policy.

    ``fall_rew`` / ``timeout_rew`` are the mean of the tensor actually handed to
    rsl_rl on the steps where the episode ends by a fall / by the time-out. They
    are the end-to-end proof that the fall penalty arrives (``fall_rew`` should
    read ≈ -10, which ``M20VelocityEnv`` adds from the termination flags) and that
    a time-out pays nothing extra (``timeout_rew`` >= 0).

    ``per_term_every`` adds a ``[terms] per-step`` line every that many steps: the
    contribution of each reward term as ``value * weight * dt``, i.e. the number
    that enters ``Train/mean_reward`` every step. ``Episode_Reward/*`` is the
    accumulated version (``_episode_sums``) and it grows with episode length, so
    a term whose weight is far too large can still look bounded there while its
    per-step cost is the whole positive side of the reward. This line is the one
    to read when ``Train/mean_reward`` sits at some value and the question is
    "which term is doing that". Its ``sum=`` should equal ``rew=`` on the health
    line at the same step (up to the fall penalty, which the env adds later).
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
            # only visible in tensorboard hours later: |v_xy| is the residual the
            # tracking kernel prices, and `standing` is the fraction of envs whose
            # command is exactly zero (the same gate `wheel_speed` applies). It is
            # derived from the command buffer -- the same source `standing_mask`
            # reads -- rather than from `is_standing_env`, so this line reports the
            # gate the reward actually applies. If it is 0.00 during stage 1, the
            # command buffer is not zero, not the reward weights.
            speed_xy = robot.data.root_lin_vel_b[:, :2].norm(dim=1).mean()
            extra = f"  |v_xy|={speed_xy.item():.3f}"
            command_manager = getattr(unwrapped, "command_manager", None)
            if command_manager is not None:
                try:
                    term = command_manager.get_term("base_velocity")
                except KeyError:
                    term = None
                buffer = getattr(term, "vel_command_b", None) if term else None
                if buffer is not None:
                    gate = (buffer[:, :3].abs() <= 1e-3).all(dim=1)
                    extra += f"  standing={gate.float().mean().item():.2f}"
            # Per-step reward as the reward manager summed it (no clamp since the
            # floor was removed). A large negative `rew` here is not a broken
            # reward -- it is the regime where the penalties outweigh the tracking
            # credit. If `neg` sits near 1.00 for a whole stage and
            # `Train/mean_reward` does not climb, the command range is ahead of
            # the policy.
            reward_buf = getattr(unwrapped, "reward_buf", None)
            if reward_buf is not None:
                extra += (
                    f"  rew={reward_buf.mean().item():+.2f}"
                    f"  neg={(reward_buf < 0).float().mean().item():.2f}"
                )
            # What the *learner* actually receives on the terminal steps. `rew` above
            # is the sum from the reward manager; `out[1]` is the tensor handed to
            # rsl_rl, after `M20VelocityEnv` added the fall penalty. A fall must show
            # `fall_rew` ≈ -10, and a time-out must show `timeout_rew` >= 0 (it pays
            # nothing extra). These two numbers are the end-to-end proof that the
            # penalty is applied once, on falls only.
            #
            # Shape differs by where this wrapper sits: the raw env returns
            # (obs, reward, terminated, truncated, extras), while
            # ``RslRlVecEnvWrapper`` returns (obs, reward, dones, infos) with the
            # time-out flag in ``infos["time_outs"]``. Handle both.
            terminated = truncated = None
            if len(out) == 5:
                terminated, truncated = out[2], out[3]
            elif len(out) == 4 and isinstance(out[3], dict):
                terminated = out[2]
                truncated = out[3].get("time_outs")
                if truncated is None:
                    truncated = torch.zeros_like(terminated)
            if terminated is not None:
                fell = terminated & ~truncated
                if bool(fell.any()):
                    extra += f"  fall_rew={out[1][fell].mean().item():+7.2f}"
                if bool(truncated.any()):
                    extra += f"  timeout_rew={out[1][truncated].mean().item():+6.2f}"
            print(
                f"[health] step={state['n']:8d}"
                f"  max|qdot|={joint_vel.abs().max().item():10.2f}"
                f"  max|v|={root_vel.item():8.2f}"
                f"  z=[{root_z.min().item():.3f},{root_z.max().item():.3f}]"
                f"{extra}"
                f"  finite={torch.isfinite(joint_vel).all().item()}",
                flush=True,
            )
            # Which term is actually paying for the number above. The health line
            # says `rew=-3.4`; this says it is `joint_deviation=-1.9` and
            # `action_rate=-1.2`, without waiting for tensorboard -- and in the
            # only scale where weights are comparable (the dt-scaled episode sums
            # grow with episode length and hide the per-step cost).
            if per_term_every > 0 and state["n"] % per_term_every == 0:
                manager = getattr(unwrapped, "reward_manager", None)
                if manager is not None:
                    steps = reward_term_contributions(unwrapped, manager)
                    if steps:
                        print(format_terms(steps, "value*weight*dt"), flush=True)
                    totals = episode_sums(manager)
                    if totals:
                        body = "  ".join(
                            f"{name}={value:+.1f}" for name, value in totals.items())
                        print(f"[terms] episode sums: {body}", flush=True)
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


def checkpoint_iteration(path: Path) -> int:
    """Iteration count encoded in a ``model_<n>.pt`` name, 0 if unreadable."""
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def stage_baseline(log_root_path: Path, stage_index: int) -> Path | None:
    """Newest checkpoint of the stage before ``stage_index``, i.e. its baseline.

    rsl_rl's counter is cumulative, so the number in this stage's filenames only
    makes sense next to the *tail number of the previous stage's last model*:
    stage 2 running from 900 to 3150 starts at 900.
    """
    if stage_index <= 0:
        return None
    return latest_checkpoint(
        log_root_path / stage_label(stage_index, STAGES[stage_index - 1]))


def stage_relative_iteration(checkpoint: Path, log_root_path: Path,
                             stage_index: int) -> int:
    """How many iterations into its own stage ``checkpoint`` was written.

    Only used to put the command ramp back where it was: a stage interrupted at
    iteration 400 of its 1000 must resume on the range it was training at, and a
    stage interrupted at 400 of its 150 warmup must resume mid-ramp. It is *not*
    used to decide how long the stage still has to run -- that is the operator's
    call (see the module docstring).
    """
    baseline = stage_baseline(log_root_path, stage_index)
    base = checkpoint_iteration(baseline) if baseline is not None else 0
    return max(checkpoint_iteration(checkpoint) - base, 0)


def _plain(value):
    """Tuples to lists, recursively, so YAML round-trips compare equal."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _flatten(node, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        flat: dict[str, object] = {}
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): _plain(node)}


def env_differences(previous_path: Path, env_cfg, limit: int = 40) -> list[str] | None:
    """Leaves where the current env cfg differs from the saved ``env.yaml``.

    Dumped through the same ``_LogDumper`` on both sides so sentinels and tuples
    serialise identically; returns ``None`` when the old file cannot be read.

    The command range is skipped on purpose: ``CommandSchedule`` rewrites it every
    few steps and the pre-``gym.make`` setup points it at the resumed position, so
    it is *expected* to differ from the value stored when the stage started. The
    range actually in force is printed separately ("First rollout range").
    """
    try:
        previous = yaml.safe_load(previous_path.read_text())
        current = yaml.safe_load(yaml.dump(env_cfg.to_dict(), Dumper=_LogDumper,
                                           sort_keys=False))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(previous, dict):
        return None
    old_flat = _flatten(previous)
    new_flat = _flatten(current or {})
    ignored = ("commands.base_velocity.ranges.",)
    diffs = []
    for key in sorted(set(old_flat) | set(new_flat)):
        if key.startswith(ignored):
            continue
        before = old_flat.get(key, "<absent>")
        after = new_flat.get(key, "<absent>")
        if before == after:
            continue
        diffs.append(f"{key}: {before!r} -> {after!r}")
        if len(diffs) >= limit:
            break
    return diffs


def latest_checkpoint_across_runs(experiment_dir: Path, stage_index: int) -> Path | None:
    """Newest checkpoint for course stage ``stage_index`` in *any* run.

    Used by ``--new_run``: the new directory is empty by construction, so a
    stage other than the first has to borrow the previous stage from an earlier
    run instead of silently starting from a random policy.
    """
    if not experiment_dir.is_dir() or stage_index <= 0:
        return None
    label = stage_label(stage_index, STAGES[stage_index - 1])
    best: tuple[float, Path] | None = None
    for run_dir in experiment_dir.iterdir():
        stage_dir = run_dir / label
        if not run_dir.is_dir() or not stage_dir.is_dir():
            continue
        found = latest_checkpoint(stage_dir)
        if found is None:
            continue
        mtime = found.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, found)
    return None if best is None else best[1]


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
    if args_cli.fixed_lr:
        # The cfg ships with the plan's adaptive schedule; pin the rate here so
        # the two modes can be compared without editing the config file.
        agent_cfg.algorithm.schedule = "fixed"
        print(f"[INFO] Schedule: fixed at lr={agent_cfg.algorithm.learning_rate:g}"
              f" (cfg default is 'adaptive'; --fixed_lr overrides it)")
    else:
        print(f"[INFO] Schedule: {agent_cfg.algorithm.schedule}"
              f" (start lr={agent_cfg.algorithm.learning_rate:g}; pass --fixed_lr to pin it)")

    # The velocity task carries a command term; the standing task does not.
    is_velocity_task = hasattr(env_cfg, "commands")
    per_stage = {index: getattr(args_cli, f"iters_stage{index + 1}") for index in range(len(STAGES))}
    stages = select_stages(args_cli.stage, args_cli.stage_scale, args_cli.iterations, per_stage,
                           args_cli.warmup_frac) \
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
                  f"(re-running a stage continues it from its newest checkpoint; "
                  f"--overwrite moves that attempt aside and retrains, --new_run "
                  f"starts a separate directory)")
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = experiment_dir / run_name
    print(f"[INFO] Logging to {log_root_path}")

    if stages:
        # The commands of the very first reset are drawn from whatever range the
        # config carries, and that happens inside gym.make -- before the schedule
        # can intervene. Point it at the range the first selected stage trains at
        # *now*: ranges_from for a fresh stage, the interrupted position for a
        # stage being continued. Otherwise a resumed stage starts its first 2-4 s
        # on the wrong distribution, which is exactly "starting over".
        first_number, first_stage = stages[0]
        first_done = 0
        if args_cli.stage != "all":
            candidate = latest_checkpoint(
                log_root_path / stage_label(first_number + 1, first_stage))
            if candidate is not None:
                first_done = stage_relative_iteration(
                    candidate, log_root_path, first_number)
        first_vx, first_vy, first_wz = interpolate_ranges(
            first_stage, first_done, first_stage.warmup)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-first_vx, first_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-first_vy, first_vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-first_wz, first_wz)
        print(f"[INFO] First rollout range: vx=+-{first_vx:.3f} vy=+-{first_vy:.3f} "
              f"wz=+-{first_wz:.3f}" +
              (f" (stage iteration {first_done})" if first_done else ""))

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
    attach_diagnostics(
        env, args_cli.health_every, args_cli.zero_actions, args_cli.per_term_every)

    schedule = None
    if is_velocity_task:
        manager = env.unwrapped.command_manager
        schedule = CommandSchedule(manager.get_term("base_velocity"), agent_cfg.num_steps_per_env)
        attach_stage_schedule(env, schedule)
    else:
        attach_dr_ramp(env, agent_cfg.max_iterations, agent_cfg.num_steps_per_env,
                       args_cli.dr_ramp)

    planned: list[tuple[str, int, Stage | None]] = []
    if stages:
        for stage_index, stage in stages:
            planned.append((stage_label(stage_index + 1, stage), stage.iterations, stage))
    else:
        planned.append(("standing", agent_cfg.max_iterations, None))

    for position, (label, iterations, stage) in enumerate(planned):
        stage_dir = log_root_path if stage is None else log_root_path / label
        stage_index = -1
        if stage is not None:
            stage_index = next(i for i, s in enumerate(STAGES) if s.name == stage.name)
        resume_path: Path | None = None
        resuming_own_stage = False
        resumed_iterations = 0

        if stage_dir.exists() and any(stage_dir.iterdir()):
            if args_cli.overwrite:
                # Moved, not deleted: a mistaken retrain stays recoverable.
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = stage_dir.with_name(f"{stage_dir.name}.bak-{stamp}")
                stage_dir.rename(backup)
                print(f"[INFO] {label}: --overwrite, previous attempt moved to "
                      f"{backup.name} (nothing was deleted)")
            else:
                # Re-running a stage continues it. That is what "keep training"
                # has to mean: resuming from the previous stage's checkpoint
                # instead would throw this stage's work away.
                resume_path = latest_checkpoint(stage_dir)
                resuming_own_stage = resume_path is not None
                if resuming_own_stage:
                    print(f"[INFO] {label}: earlier attempt found -- continuing it from "
                          f"{resume_path.name} (use --overwrite to retrain from scratch)")
                    resumed_iterations = stage_relative_iteration(
                        resume_path, log_root_path, stage_index)
                    # Same environment as when it stopped: if anything in the cfg
                    # changed (command range, rewards, DR, env count), resuming is
                    # not the same run and the operator has to know.
                    previous_env = stage_dir / "env.yaml"
                    if previous_env.exists():
                        diff = env_differences(previous_env, env_cfg)
                        if diff is None:
                            print(f"[WARN] {label}: could not read {previous_env.name}; "
                                  f"cannot confirm the environment is unchanged")
                        elif diff:
                            print(f"[WARN] {label}: env cfg differs from the one this stage "
                                  f"was trained with ({previous_env.name}):")
                            for line in diff[:10]:
                                print(f"        {line}")
                            if len(diff) > 10:
                                print(f"        ... {len(diff) - 10} more difference(s)")
                            print("        Resuming on a different environment is not the same "
                                  "run; pass --overwrite to start the stage cleanly instead.")
                        else:
                            print(f"[INFO] {label}: env cfg matches {previous_env.name}")
                else:
                    print(f"[INFO] {label}: directory holds logs but no checkpoint, so "
                          f"this stage starts over")
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Continuing a stage keeps the first attempt's env.yaml/agent.yaml as the
        # record of how it was trained; this run's config goes beside them.
        if resuming_own_stage:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            env_yaml = stage_dir / f"env.resume-{stamp}.yaml"
            agent_yaml = stage_dir / f"agent.resume-{stamp}.yaml"
        else:
            env_yaml = stage_dir / "env.yaml"
            agent_yaml = stage_dir / "agent.yaml"
        with open(env_yaml, "w") as f:
            yaml.dump(env_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)
        with open(agent_yaml, "w") as f:
            yaml.dump(agent_cfg.to_dict(), f, Dumper=_LogDumper, sort_keys=False)

        if resume_path is None and stage is not None:
            # Fresh stage: pick up where the previous stage of this run left off,
            # so "--stage 2" after "--stage 1" really does continue rather than
            # start from a random policy.
            if stage_index > 0:
                previous = log_root_path / stage_label(stage_index, STAGES[stage_index - 1])
                resume_path = latest_checkpoint(previous)
                if resume_path is None and args_cli.new_run:
                    # A --new_run directory is empty by construction, so seed it
                    # from the previous stage of whichever run has one.
                    resume_path = latest_checkpoint_across_runs(experiment_dir, stage_index)
                    if resume_path is not None:
                        print(f"[INFO] --new_run: seeding {stage.name} from "
                              f"{resume_path} (previous stage, earlier run)")
                if resume_path is not None:
                    print(f"[INFO] Stage {stage.name}: continuing from the previous "
                          f"stage's checkpoint {resume_path}")
                else:
                    print(f"[WARN] Stage {stage.name}: no checkpoint from "
                          f"{previous} and no earlier run to borrow from -- starting "
                          f"from a fresh policy. Run --stage {stage_index} first.")

        if resume_path is not None and not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        # How much to run now. No arithmetic against what the stage already did:
        # how long a stage should be trained is the operator's decision, so the
        # default is simply the stage budget and --iterations replaces it.
        if args_cli.iterations is not None:
            run_iterations = int(args_cli.iterations)
        else:
            run_iterations = iterations

        print(f"\n[INFO] ===== stage {label}: {run_iterations} iterations =====")
        if resume_path is not None:
            baseline = stage_baseline(log_root_path, stage_index)
            source = (f"baseline {baseline.name}" if baseline is not None
                      else "no previous stage, absolute baseline 0")
            print(f"[INFO] Continuing from {resume_path} "
                  f"(global iteration {checkpoint_iteration(resume_path)}; {source}"
                  + (f"; stage iteration {resumed_iterations}" if resuming_own_stage else "")
                  + ")")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(stage_dir),
                                device=agent_cfg.device)
        if resume_path is not None:
            runner.load(str(resume_path))

        if schedule is not None and stage is not None:
            # Continuing a stage resumes its ramp where it stopped (same command
            # range as the last iteration it trained), so the env looks exactly
            # like it did at the end of the previous attempt. The ramp length is
            # this stage's effective one (--warmup_frac / --stage_scale applied),
            # and ``resumed_iterations`` is measured on the same ramp, so a stage
            # interrupted at 900 of a 2000-iteration ramp picks up at 45%.
            schedule.start_stage(
                stage,
                warmup=stage.warmup,
                resume_iteration=resumed_iterations)

        start = time.time()
        runner.learn(num_learning_iterations=run_iterations, init_at_random_ep_len=True)
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
