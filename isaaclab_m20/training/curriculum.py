"""Live curriculum: drives the command range and the DR level inside ``env.step``.

rsl_rl calls ``runner.learn`` once per stage, but the env wrapper lives across all
of them, so something has to know which stage is running. That something is
``CommandSchedule``: it is attached to ``env.step`` and applies the current
stage's range one iteration at a time.

The schedule also owns the *resume* semantics, which is why it takes
``resume_iteration``: a stage that was interrupted at 40% of its warmup must come
back on the range it was training at, not on the range the stage ends at. See
``docs/algorithm.md`` section 6.6 for the three rules that follow from this.
"""
from __future__ import annotations

from isaaclab_m20.tasks import dr
from isaaclab_m20.training.stages import Stage, interpolate_ranges


class CommandSchedule:
    """Drives the command range through the current stage, iteration by iteration.

    Attached to ``env.step``; ``start_stage`` is called right before each
    ``runner.learn``.
    """

    def __init__(self, command_term, steps_per_iteration: int) -> None:
        self.term = command_term
        self.steps_per_iteration = steps_per_iteration
        self.stage: Stage | None = None
        self.iteration = 0
        self.warmup = 0
        # How far into the stage the previous run already got. Resuming has to
        # continue the ramp from there, not restart it.
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

        ``warmup`` overrides the stage's own (possibly ``--stage_scale``-trimmed)
        warmup; when continuing, pass the *untrimmed* one so the ramp the stage
        was climbing stays the same ramp. Pass ``resume_iteration=0`` for a
        fresh stage.
        """
        self.stage = stage
        self.warmup = stage.warmup if warmup is None else max(int(warmup), 0)
        self.offset = max(int(resume_iteration), 0)
        self.iteration = 0
        self._steps = 0
        self._applied = None
        self.apply_sampling(stage, announce=True)
        self._apply(self.ranges_at(self.offset), announce=True)
        # The buffer still holds commands drawn under the previous stage's range
        # (up to 4 s of them). For stage 1 the new range starts at zero, so wipe
        # them: the robot must not be asked for 0.4 m/s on step 0.
        if self.offset == 0 and max(stage.ranges_from) == 0.0:
            dr.zero_command(self.term)

    def apply_sampling(self, stage: Stage, announce: bool = False) -> None:
        """Apply the stage's command-distribution overrides (zero_prob, hold time).

        ``MixedVelocityCommand`` reads both off ``self.cfg`` on every resample, so
        writing them here takes effect from the next resample on -- no need to
        recreate the term.
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
            zero_prob = getattr(cfg, "zero_prob", None)
            print(f"[cmd] stage={stage.name} hold={cfg.resampling_time_range[0]:.2f}-"
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
