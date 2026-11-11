"""Velocity-command sampling with an axis mix.

``UniformVelocityCommand`` draws the three axes independently and uniformly.
Over a range that is wide enough to be useful, that means a *pure* command has
probability zero -- the probability of hitting ``vy == 0`` exactly is zero -- so
a policy trained on it sees almost only coupled commands, and the decoupled
ones (straight ahead, straight back, spin on the spot) that the acceptance test
in ``scripts/eval.py`` measures never get any training signal.

This term draws from the mix the training plan asks for instead:

===============  ======  ===================================================
mode             share   axes that are non-zero
===============  ======  ===================================================
single axis      50%     one of vx / vy / wz, the other two exactly 0
two axes         30%     any two of the three
three axes       10%     all three
zero command     10%     none -- stop and hold
===============  ======  ===================================================

An axis whose sampling range is ``(0, 0)`` is not available at all, so the mix
degrades correctly during the early training stages: with only ``vx`` enabled
(single-axis course) every non-zero draw is a pure forward/backward command,
and the zero share is still 10%. That is exactly the "learn forward and
backward first" behaviour the staged plan wants, without a second code path.

Each component is drawn uniformly inside its range, and each command is held
for 2-4 s (``resampling_time_range``) before being redrawn: the policy has to
hold a command, not chase noise.

Both numbers are read off the config on every resample, so a course stage can
change them at runtime without recreating the term. The last stage does exactly
that: 35% zero commands and 1.5-3 s holds, to train stop/go switching rather
than long holds (see ``STAGES`` in ``scripts/train.py``).
"""
from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from isaaclab_m20.tasks import control

_ZERO_EPS = 1e-6


class MixedVelocityCommand(UniformVelocityCommand):
    """Uniform-in-range commands, sampled with 1-3 active axes.

    Alongside the twist this term publishes a **mode label** (``STAND`` / ``MOVE``
    / ``ROTATE`` / ``COMBINED``) in ``mode_buf``.  The reward gates read the label
    instead of re-testing the floats: the sampler writes exact zeros today, so a
    float test would work, but any future command filter or noise would make "the
    label says MOVE while the numbers say 0.001" a silent inconsistency.  The
    label is written by the sampler, i.e. by the same code that chose the numbers.
    """

    cfg: MixedVelocityCommandCfg

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self.mode_buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _write_modes(self, env_ids, command: torch.Tensor) -> None:
        """Derive the mode label from the twist that was just written."""
        moving = command[:, :2].norm(dim=1) > _ZERO_EPS
        spinning = command[:, 2].abs() > _ZERO_EPS
        mode = torch.full((command.shape[0],), control.MODE_STAND,
                          dtype=torch.long, device=command.device)
        mode = torch.where(moving & spinning,
                           torch.full_like(mode, control.MODE_COMBINED), mode)
        mode = torch.where(moving & ~spinning,
                           torch.full_like(mode, control.MODE_MOVE), mode)
        mode = torch.where(~moving & spinning,
                           torch.full_like(mode, control.MODE_ROTATE), mode)
        self.mode_buf[env_ids] = mode

    def _resample_command(self, env_ids):
        # Heading mode replaces the yaw-rate command with a feedback law on the
        # heading error, so a pre-sampled third column would be overwritten one
        # step later anyway. This task runs with heading_command=False; keep the
        # stock sampler for the other case rather than silently fighting it.
        if self.cfg.heading_command:
            super()._resample_command(env_ids)
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_envs = env_ids.numel()

        ranges = self.cfg.ranges
        bounds = torch.tensor(
            [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z],
            dtype=torch.float,
            device=self.device,
        )  # (3, 2)
        available = (bounds[:, 1] - bounds[:, 0]) > 0.0
        axis_ids = available.nonzero(as_tuple=False).flatten()
        num_axes = axis_ids.numel()

        command = torch.zeros(num_envs, 3, device=self.device)
        if num_axes == 0:
            # Nothing is being asked for: every environment gets the zero
            # command. (Happens only if a stage is misconfigured to all zeros.)
            self.vel_command_b[env_ids, :3] = command
            self.is_standing_env[env_ids] = True
            self.is_heading_env[env_ids] = False
            self._write_modes(env_ids, command)
            return

        # How many axes this draw activates: 1 / 2 / 3 with the 50/30/10 shares,
        # renormalised over the axes this stage actually enables.
        weights = torch.tensor(
            [self.cfg.single_axis_prob, self.cfg.two_axis_prob, self.cfg.three_axis_prob],
            dtype=torch.float,
            device=self.device,
        )[:num_axes]
        weights = weights / weights.sum()
        cumulative = torch.cumsum(weights, dim=0)

        draw = torch.rand(num_envs, device=self.device)
        standing = draw < self.cfg.zero_prob
        # Re-draw the non-standing share conditionally, so the zero share does
        # not eat into the 50/30/10 proportion between the other modes.
        share = max(1.0 - self.cfg.zero_prob, 1e-6)
        conditional = ((draw - self.cfg.zero_prob) / share).clamp_(0.0, 1.0 - 1e-6)
        active_count = torch.searchsorted(cumulative, conditional) + 1
        active_count = active_count.clamp_(max=num_axes)

        # Sample every enabled axis uniformly inside its range ...
        unit = torch.rand(num_envs, num_axes, device=self.device)
        low = bounds[axis_ids, 0]
        high = bounds[axis_ids, 1]
        sampled = low + (high - low) * unit

        # ... then keep only the first `active_count` axes of a random ordering,
        # which selects a uniformly random subset of that size.
        order = torch.argsort(torch.rand(num_envs, num_axes, device=self.device), dim=1)
        keep = torch.arange(num_axes, device=self.device).unsqueeze(0) < active_count.unsqueeze(1)
        mask = torch.zeros(num_envs, num_axes, dtype=torch.bool, device=self.device)
        mask.scatter_(1, order, keep)
        command[:, axis_ids] = sampled * mask.to(sampled.dtype)

        # Standing environments are exactly zero on all three axes.
        command[standing] = 0.0

        self.vel_command_b[env_ids, :3] = command
        self.is_standing_env[env_ids] = standing
        self.is_heading_env[env_ids] = False
        self._write_modes(env_ids, command)


def freeze_command_resampling(term) -> None:
    """Stop a command term from ever resampling over a command written from outside.

    ``play`` and ``eval`` own the command: they overwrite it every step. The term
    would still resample on its own timer -- ``UniformVelocityCommand`` resamples
    every ``resampling_time_range`` seconds *and* on every reset -- and that write
    lands after ours, so the robot silently follows a freshly sampled command
    instead of the one on the command line. With a zero-range stage (stage 1) the
    sampled command is zero, which reads as "the command never arrived".

    Setting ``cfg.resampling_time_range`` alone is not enough: the term copies it
    into ``_resampling_time_range`` in ``__init__`` and keeps a separate
    ``time_left`` countdown. So rewrite the config, the cached tuple and the
    countdown, and finally make ``_resample_command`` a no-op -- that is what
    covers the reset path, which is what zeroes the buffer at every episode end.

    Being a no-op, ``is_standing_env`` / ``is_heading_env`` keep their initial
    values. Nothing in this project reads them any more (the reward and the parking
    brake both read the command buffer), so that is fine.
    """
    if term is None:
        return

    infinite = (1.0e9, 1.0e9)
    try:
        term.cfg.resampling_time_range = infinite
    except Exception as exc:  # noqa: BLE001 - the no-op below is the real fix
        print(f"[WARN] Could not rewrite resampling_time_range: {exc}")
    if hasattr(term, "_resampling_time_range"):
        try:
            term._resampling_time_range = infinite
        except Exception:  # noqa: BLE001
            pass
    for attribute in ("time_left", "_time_left"):
        countdown = getattr(term, attribute, None)
        if torch.is_tensor(countdown):
            countdown.fill_(1.0e9)
    try:
        term._resample_command = lambda env_ids=None: None
    except Exception as exc:  # noqa: BLE001 - fall back to the timer above
        print(f"[WARN] Could not disable command resampling: {exc}")


@configclass
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` plus the axis-mix shares."""

    class_type: type = MixedVelocityCommand

    # Shares of the four draw modes. They have to sum to 1; the counts are
    # renormalised against the axes a stage enables, so the mix still works
    # when only one axis is sampled.
    single_axis_prob: float = 0.50
    two_axis_prob: float = 0.30
    three_axis_prob: float = 0.10
    zero_prob: float = 0.10
