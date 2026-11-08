"""Environment classes that are more than a bare ``ManagerBasedRLEnv``.

Right now there is exactly one: :class:`M20VelocityEnv`, which gives the
velocity task a **reward floor** (``only_positive_rewards`` in legged_gym). The
task is still registered as a gym id in ``tasks/__init__.py``; only the
``entry_point`` differs from the standing task.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class M20VelocityEnv(ManagerBasedRLEnv):
    """Velocity task with a per-step reward floor.

    Why the floor exists
    --------------------
    The reward is a mix of positive tracking terms and per-step penalties. The
    positives shrink when the policy tracks badly, but the penalties do not -- a
    flailing policy pays ``joint_deviation`` ~1.0/step while earning < 1.0/step of
    tracking. Over a 20 s episode that is ~-20 reward per second of life, against a
    one-off -10 for falling: **dying early becomes the cheaper policy**, and the
    agent reliably discovers it ("it dies instantly, yet the mean reward goes up").

    Flooring the *per-step* reward at 0 removes that gradient without touching the
    fall penalty:

    * staying alive and behaving badly now pays 0 per step (total ~0),
    * falling pays -10 and no more,

    so surviving is always worth at least as much as dying, and the positives still
    teach tracking (a good policy earns ~4/step, i.e. ~4000 per episode).

    The floor is applied to everything *except* the fall penalty, which is why the
    fall term is added back before clipping and subtracted afterwards -- the same
    order legged_gym uses (clip the step reward, then add the terminal one).

    Time-out is not failure
    -----------------------
    Surviving the full 20 s is the thing being trained, so it pays exactly nothing
    extra: no penalty, no bonus. Only a real fall is penalised, and a fall is
    ``terminated & ~truncated`` -- the time-out is reported through ``truncated``
    (its termination term carries ``time_out=True``), so it can never be scored as
    a fall. ``_check_timeout_is_not_a_fall`` warns if that flag is ever removed.
    This is also why the reward task has no ``termination_penalty`` on
    ``mdp.is_terminated``, which fires on the time-out as well.

    Consequences worth knowing
    --------------------------
    * ``Episode_Reward/*`` in tensorboard still logs the **raw**, unclipped term
      sums (the reward manager accumulates them before this class sees the
      buffer). Read ``Train/mean_reward`` for what the learner optimises; the
      per-term curves are still the right place to see *which* term is large.
    * A catastrophic-but-alive policy is indistinguishable from a merely bad one
      (both pay 0/step), so the floor trades some gradient for stability. That is
      the accepted trade: without it the gradient points at the floor.

    Because the floor guarantees "living badly" >= 0 while falling is only -10, the
    velocity task does not also carry a per-step survival bonus: the floor is what
    makes survival valuable, and a bonus on top would just rescale the return.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._fall_penalty = self._read_fall_penalty()
        self._check_timeout_is_not_a_fall()

    def _check_timeout_is_not_a_fall(self) -> None:
        """Warn if the time-out is configured as a failure instead of a truncation.

        Reaching 20 s is the demonstration of the competence being trained, not a
        failure: it must pay exactly nothing. That is enforced by
        ``fell = terminated & ~truncated``, which is correct only while the time-out
        term is declared with ``time_out=True`` (Isaac Lab then reports it through
        ``truncated``). Flip that flag and every completed episode would silently
        pay the fall penalty, so the mistake is reported instead of absorbed.
        """
        term = getattr(self.cfg.terminations, "time_out", None)
        if term is None or not bool(getattr(term, "time_out", False)):
            print(
                "[WARN] M20VelocityEnv: the 'time_out' termination is not marked "
                "time_out=True, so surviving the whole episode would be scored as "
                "a fall and pay the fall penalty. Normal time-outs must pay 0.",
                flush=True,
            )

    def _read_fall_penalty(self) -> float:
        """|weight| of the ``termination_penalty`` term, or 0.0 if there is none.

        Returns 0.0 when the term is missing, and ``step`` then skips the floor
        entirely -- rewards would flow through unclipped and without a fall
        penalty. That is a silent regression, so it is reported loudly here: the
        term is part of the contract between this class and the reward config, not
        an optional extra.
        """
        try:
            term_cfg = getattr(self.cfg.rewards, "termination_penalty", None)
        except AttributeError:  # pragma: no cover - a cfg without that term
            term_cfg = None
        weight = 0.0
        if term_cfg is not None:
            weight = abs(float(getattr(term_cfg, "weight", 0.0) or 0.0))

        if weight <= 0.0:
            print(
                "[WARN] M20VelocityEnv: cfg.rewards has no usable "
                "'termination_penalty' term, so the per-step reward floor and the "
                "fall penalty are BOTH disabled. Restore the term (weight < 0) if "
                "that is not intentional.",
                flush=True,
            )
        return weight

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        if self._fall_penalty <= 0.0:
            return obs, rew, terminated, truncated, extras

        # A real fall is a termination that is not the episode time-out. Both
        # flags come straight out of the base class, so this stays correct even if
        # the termination terms change.
        #
        # ``rew`` already contains the negative ``termination_penalty`` weight, so
        # adding the magnitude back recovers the ordinary step reward, clamping
        # floors that part at zero, and subtracting the magnitude again re-applies
        # the penalty unclipped. The net order is therefore "clip the step reward,
        # *then* pay the fall penalty", which is what legged_gym's
        # ``only_positive_rewards`` does. The subtraction relies on
        # ``rewards.fall_terminated`` evaluating the same height / tilt condition
        # from robot data in the same step (it does not read
        # ``termination_manager``, whose flags are computed after rewards).
        fell = (terminated & ~truncated).to(device=rew.device, dtype=rew.dtype)
        penalised = rew + self._fall_penalty * fell  # back to the step part only
        floored = torch.clamp(penalised, min=0.0)
        return (
            obs,
            floored - self._fall_penalty * fell,  # re-apply, unclipped
            terminated,
            truncated,
            extras,
        )
