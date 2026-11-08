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
"""
from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class MixedVelocityCommand(UniformVelocityCommand):
    """Uniform-in-range commands, sampled with 1-3 active axes."""

    cfg: MixedVelocityCommandCfg

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
