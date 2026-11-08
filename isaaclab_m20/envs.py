"""Environment classes that are more than a bare ``ManagerBasedRLEnv``.

Right now there is exactly one: :class:`M20VelocityEnv`. The task is still
registered as a gym id in ``tasks/__init__.py``; only the ``entry_point`` differs
from the standing task.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class M20VelocityEnv(ManagerBasedRLEnv):
    """Velocity task with a fall penalty taken from the termination flags.

    What this class does
    --------------------
    It adds the fall penalty (``cfg.fall_penalty``, default -10) to the reward of
    the steps that actually ended in a fall, using ``terminated & ~truncated``
    straight out of the base class. There are two reasons for doing it here
    instead of with a reward term:

    * a reward term would have to re-derive "did it fall" from the robot state,
      and that copy of the condition can drift away from ``TerminationsCfg``
      (height and tilt thresholds live in two places, one of which is never
      exercised until the day it is wrong);
    * the termination manager computes its flags *after* the reward manager, so a
      reward term can only ever read the previous step's flags -- i.e. it would
      pay on the wrong step, and miss the last one entirely because the episode
      resets.

    Here the flags are the ones that ended the episode, so the penalty can
    neither fire twice nor miss a fall.

    There is deliberately **no reward floor**
    -----------------------------------------
    An earlier version clamped the per-step reward at 0
    (legged_gym's ``only_positive_rewards``) to remove the "dying early is
    cheaper" policy, and added an ``alive`` bonus to lift the ordinary step above
    that clamp. Both are gone, for one measured reason: the clamp is only safe
    while the ordinary step reward is *positive*, and while the policy is bad it
    is not. With the old normalisation scales the noise alone put a step at
    ~-45, so every step was clamped to a constant 0, ``Train/mean_reward`` was
    identically 0 for the whole run, and nothing could be learned.

    The conclusion is that the clamp treated a symptom. PPO does not need
    non-negative rewards: the advantage is ``A = R - V`` and normalising it
    rescales without changing the ordering, so a step that is *less* bad than the
    critic expects still produces a positive advantage. What actually made dying
    attractive was the reward *shape* (penalties that do not shrink when the
    policy stops trying), and that is fixed in the weights, not in a clamp.

    Time-out is not failure
    -----------------------
    Surviving the full 20 s is the thing being trained, so it pays exactly
    nothing extra: no penalty, no bonus. Only a real fall is penalised --
    ``terminated & ~truncated`` -- because the time-out is reported through
    ``truncated`` (its termination term carries ``time_out=True``).
    ``_check_timeout_is_not_a_fall`` warns if that flag is ever removed.
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
        """The fall penalty, negative, from ``cfg.fall_penalty``.

        Falls back to the old ``rewards.termination_penalty`` weight when the
        dedicated field is missing, so a config written before the penalty moved
        here still behaves the same. A penalty of exactly 0 is reported because it
        would silently make falling free -- that is a policy decision, not a
        default.
        """
        penalty = float(getattr(self.cfg, "fall_penalty", 0.0) or 0.0)
        if penalty == 0.0:
            term_cfg = getattr(self.cfg.rewards, "termination_penalty", None)
            weight = 0.0 if term_cfg is None else float(getattr(term_cfg, "weight", 0.0) or 0.0)
            if weight < 0.0:
                penalty = weight
            else:
                print(
                    "[WARN] M20VelocityEnv: cfg.fall_penalty is 0 and there is no "
                    "'termination_penalty' reward term, so falling is free and the "
                    "policy will learn to fall as soon as tracking gets hard.",
                    flush=True,
                )
        return penalty

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        if self._fall_penalty >= 0.0:
            return obs, rew, terminated, truncated, extras

        # A real fall is a termination that is not the episode time-out. Both flags
        # come straight out of the base class, so this stays correct even if the
        # termination terms change -- no second copy of the height/tilt test.
        fell = (terminated & ~truncated).to(device=rew.device, dtype=rew.dtype)
        return obs, rew + self._fall_penalty * fell, terminated, truncated, extras
