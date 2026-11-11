"""The training course: what each stage asks for, and how the budget is spent.

This module is the single place where the course is defined. Changing a stage's
range or budget is a one-line edit here; ``cli`` only chooses *which* stages run,
``curriculum`` applies the ranges to a live command term, and the docs quote the
same table.

============================  ========  ================================  =========
stage                         iters     command range (half-widths)       zero cmd
============================  ========  ================================  =========
1  stand                       900       vx 0,     vy 0,    wz 0           100%
2  forward_backward            700       vx +-2.5, vy 0,    wz 0            50%
3  multi_direction            1200       vx +-2.5, vy +-0.4, wz +-0.8       30%
4  agile                      1400       vx +-2.5, vy +-0.5, wz +-1.0       20%
============================  ========  ================================  =========

4200 iterations in total, in the order "stand -> forward/back -> everything ->
faster and stop/go", one axis group introduced per stage. The 2.5 m/s envelope is
set by the wheel action scale (``scale=27.78`` rad/s per unit times the 0.09 m
wheel) and is a policy choice, not a hardware limit: the actuator allows
200 rad/s and 240 N per wheel against a 338 N friction ceiling.

Every stage after the first keeps a share of zero commands, so standing is
rehearsed while the moving skills are learned -- without it the later stages
train the "stop" behaviour out of the actor and it has to be relearned. The share
shrinks stage by stage and stage 4 also shortens the hold (1.5-3 s), so one 20 s
episode contains several start/stop switches instead of two long holds.

Stage 1 is 100% zero command: the tracking terms then mean "be still", which
trains the baseline the later stages build on -- hold the ride height, keep the
body level, do not spin the wheels, do not drift. It is its own stage (rather
than a warmup inside stage 2) so that it has its own checkpoint: if
forward/backward never improves on it, the problem is the command, not the
baseline.

Its 900 iterations cover the *fixed*-rate mode (``--fixed_lr``), where standing
flattens out around iteration 700-900; with the default adaptive schedule it
converges much sooner. The budget cannot be separated from the rate: change
``learning_rate`` or ``--fixed_lr`` and the number that matters changes with it,
so judge by the curve, not by the budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One stage of the course.

    ``ranges_from`` -> ``ranges_to`` are (vx, vy, wz) half-widths in m/s, m/s and
    rad/s. The warmup interpolates linearly between them over ``warmup``
    iterations and then holds ``ranges_to``. ``ranges_from`` is written as the
    previous stage's ``ranges_to`` -- not as zeros -- because the axes learned
    earlier must keep their full width while the new axis grows in.

    ``zero_prob`` and ``resampling`` override the command term's sampling for this
    stage (``None`` keeps the config default: 0.10 and 2-4 s).
    """

    name: str
    iterations: int
    ranges_from: tuple[float, float, float]
    ranges_to: tuple[float, float, float]
    warmup: int
    zero_prob: float | None = None
    resampling: tuple[float, float] | None = None


STAGES: tuple[Stage, ...] = (
    # Zero command: the standing baseline under the 56-dim observation layout.
    Stage("1_stand", 900, (0.00, 0.0, 0.0), (0.00, 0.0, 0.0), 0, zero_prob=1.0),
    # Half the commands stay zero, so standing is rehearsed while vx grows all
    # the way to the envelope.
    Stage("2_forward_backward", 700, (0.00, 0.0, 0.0), (2.50, 0.0, 0.0), 350,
          zero_prob=0.5),
    # Lateral and yaw arrive together: the two axes the wheels cannot produce on
    # their own, and the ones that break the standing skill first.
    Stage("3_multi_direction", 1200, (2.50, 0.0, 0.0), (2.50, 0.40, 0.80), 400,
          zero_prob=0.3),
    # 20% stops and shorter holds: accelerate, stop, turn and side-step several
    # times per episode, at the top of the envelope.
    Stage("4_agile", 1400, (2.50, 0.40, 0.80), (2.50, 0.50, 1.00), 300,
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

    The number is part of the name because the course order is what makes the
    checkpoints comparable, and it is spelled out in one place because the log
    tree, the resume lookup and the docs must agree -- they did not once, and
    "find it under stage1_stand" pointed at a directory written as ``1_stand``.
    """
    _, _, slug = stage.name.partition("_")
    return f"stage{number}_{slug or stage.name}"


def select_stages(
    selector: str,
    scale: float,
    iterations: int | None,
    per_stage: dict[int, int | None] | None = None,
    warmup_frac: float | None = None,
    cmd_scale: float = 1.0,
) -> list[tuple[int, Stage]]:
    """Which stages this run executes, with the iteration budget applied.

    The budget of stage ``N`` comes from, in order of precedence: ``per_stage[N]``
    (``--iters_stageN``), then ``iterations`` (``--iterations`` for every selected
    stage), then the stage's default budget times ``--stage_scale``.

    ``warmup_frac`` turns the ramp into a fraction of that budget, which is what a
    stage given a much bigger budget than planned needs: stage 2 over 4000
    iterations should spend the first 2000 growing the range instead of reaching
    the full width after 150 and sampling there for the rest.

    ``cmd_scale`` multiplies the *command ranges* -- all three axes, both ends of
    the ramp -- and leaves the budget, the zero-command share and the hold time
    alone. That is the knob for "the policy cannot hold this speed yet", as
    opposed to ``--stage_scale``, which only shrinks the iteration count.

    Returns ``(index_in_STAGES, stage)`` pairs so the caller can find the
    preceding stage's checkpoint for the first selected stage.
    """
    per_stage = per_stage or {}
    if cmd_scale <= 0.0:
        raise SystemExit(f"--cmd_scale must be positive, got {cmd_scale}")
    if selector == "all":
        indices = list(range(len(STAGES)))
    else:
        try:
            number = int(selector)
        except ValueError as exc:
            raise SystemExit(f"--stage must be 1..{len(STAGES)} or 'all', got {selector!r}") from exc
        stage_by_number(number)  # validates the 1..N range
        indices = [number - 1]

    def _scaled(ranges: tuple[float, float, float]) -> tuple[float, float, float]:
        if cmd_scale == 1.0:
            return ranges
        return tuple(value * cmd_scale for value in ranges)  # type: ignore[return-value]

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
        if warmup_frac is not None and stage.warmup:
            warmup = min(int(round(budget * warmup_frac)), budget)
        else:
            warmup = min(stage.warmup, budget // 2) if stage.warmup else 0
        out.append((index, Stage(stage.name, budget, _scaled(stage.ranges_from),
                                 _scaled(stage.ranges_to), warmup, stage.zero_prob,
                                 stage.resampling)))
    return out


def interpolate_ranges(stage: Stage, done: int,
                       warmup: int | None = None) -> tuple[float, float, float]:
    """Command range after ``done`` iterations of ``stage`` (linear warmup).

    Shared by the live schedule and by the pre-``gym.make`` setup, so the range
    the env is built with equals the range the schedule applies on step one.
    """
    span = stage.warmup if warmup is None else max(int(warmup), 0)
    if span <= 0:
        return stage.ranges_to
    t = min(max(done / span, 0.0), 1.0)
    return tuple(
        low + (high - low) * t
        for low, high in zip(stage.ranges_from, stage.ranges_to)
    )
