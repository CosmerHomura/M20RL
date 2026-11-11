"""Environment classes that are more than a bare ``ManagerBasedRLEnv``.

Right now there is exactly one: :class:`M20VelocityEnv`. The task is still
registered as a gym id in ``tasks/__init__.py``; only the ``entry_point`` differs
from the standing task.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class M20VelocityEnv(ManagerBasedRLEnv):
    """Velocity task with a fall penalty and a wheel parking brake.

    What this class does
    --------------------
    Two things, both of them consequences of a decision that would otherwise be
    spread over the reward and the action term:

    1. it adds the fall penalty (``cfg.fall_penalty``, default -10) to the reward
       of the steps that actually ended in a fall, using
       ``terminated & ~truncated`` straight out of the base class;
    2. it parks the wheels (a position outer loop, see below) while the velocity
       command is exactly zero.

    The fall penalty is here instead of in a reward term for two reasons:

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

    The wheel parking brake
    ----------------------
    The wheels are a velocity servo with ``stiffness = 0``, and PhysX does not
    model rolling resistance: nothing resists rolling except the servo's
    ``damping``. So a body that sits a fraction of a degree off level rolls
    slowly forever, at the equilibrium ``qdot = F_bias / (4 * damping)``. With
    ``damping = 2`` that is a couple of cm/s -- enough to wander out of the
    acceptance box (``eval`` row 9 measures both speed *and* drift).

    A position loop can hold the wheel still, which a damping term cannot, so
    while the command is zero the four wheel columns of the action are replaced
    by the output of a cascade: the position error is turned into a wheel speed,
    and the existing velocity servo tracks it.

    ``omega* = clip(kp * (q_hold - q), +-omega_max)``, ``tau = damping * (omega* - qdot)``

    This is how a real wheel module behaves when the robot is parked (position
    outer loop, velocity inner loop) and it needs no gain switching: the
    actuator stays a pure velocity servo, so there is no torque step at the
    transition. The policy's wheel action simply has no authority while the
    command is zero, and the replacement is written back into the action so that
    ``last_action``/``action_rate`` see what was actually commanded.

    ``q`` is a continuous joint and is never wrapped: the wheel angle is a
    legitimate odometry counter, and rolling it by 2*pi moves the body by
    0.57 m. So the error is used unwrapped and clamped; if it ever exceeds the
    clamp the wheel was dragged (or skidded) and the hold is re-seeded at the
    current angle instead of commanding a large spin back.

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
        self._init_wheel_park()

    # -- wheel parking brake --------------------------------------------------
    def _iter_action_terms(self) -> list[tuple[str, object]]:
        """Return ``(name, term)`` pairs in action-vector order.

        IsaacLab keeps the terms in a dict keyed by name (``_terms``) next to an
        ordered ``_term_names`` list. Iterating that dict yields its *keys*, which
        is how ``'str' object has no attribute 'action_dim'`` happens, so walk the
        names and look each term up instead. Both the dict form and a plain
        sequence are accepted, because the container changed shape between
        IsaacLab releases; the public ``active_terms`` API is the last resort.
        """
        manager = self.action_manager
        names = list(getattr(manager, "_term_names", []) or [])
        terms = getattr(manager, "_terms", None)

        pairs: list[tuple[str, object]] = []
        if isinstance(terms, dict):
            pairs = [(name, terms.get(name)) for name in names] if names else list(terms.items())
        elif terms is not None:
            terms = list(terms)
            if names and len(names) == len(terms):
                pairs = list(zip(names, terms))
            else:
                pairs = [(f"term_{index}", item) for index, item in enumerate(terms)]
        if pairs:
            return pairs

        for name in list(getattr(manager, "active_terms", []) or []):
            try:
                pairs.append((name, manager.get_term(name)))
            except Exception:  # noqa: BLE001 - a disabled term must not break startup
                continue
        return pairs

    def _init_wheel_park(self) -> None:
        """Resolve the wheel columns of the action vector and the hold buffers.

        Everything is discovered from the action manager rather than hard-coded:
        the wheel columns are found by walking the action terms up to
        ``wheel_velocities``, the joint ids and the scale come from that term. If
        any of that cannot be resolved the brake disables itself with a warning
        instead of raising -- a training run without the brake is still a
        training run.
        """
        self._park_enabled = bool(getattr(self.cfg, "wheel_park_enabled", True))
        self._park_kp = float(getattr(self.cfg, "wheel_park_kp", 15.0))
        self._park_omega_max = float(getattr(self.cfg, "wheel_park_omega_max", 3.0))
        self._park_error_clamp = float(getattr(self.cfg, "wheel_park_error_clamp", 0.3))
        self._park_slice: slice | None = None
        self._park_joint_ids: torch.Tensor | None = None
        self._park_scale = 0.0
        if not self._park_enabled:
            print("[INFO] M20VelocityEnv: wheel parking brake disabled by config.", flush=True)
            return

        term_name = "wheel_velocities"
        offset = 0
        term = None
        # Walk the action terms in configuration order, because the action vector is
        # the concatenation of their outputs; the wheel columns start wherever the
        # leg term ends. Inactive terms are ``None`` and contribute nothing.
        for name, candidate in self._iter_action_terms():
            if candidate is None:
                continue
            if name == term_name:
                term = candidate
                break
            offset += candidate.action_dim
        if term is None:
            print(
                f"[WARN] M20VelocityEnv: no '{term_name}' action term, so the wheel parking "
                "brake is disabled and the wheels will creep while standing.",
                flush=True,
            )
            self._park_enabled = False
            return

        try:
            self._park_scale = abs(float(term.cfg.scale))
        except (TypeError, ValueError):
            self._park_scale = 0.0
        if self._park_scale <= 0.0:
            print(
                f"[WARN] M20VelocityEnv: '{term_name}.scale' is not a positive number, so the "
                "wheel parking brake is disabled.",
                flush=True,
            )
            self._park_enabled = False
            return
        self._park_slice = slice(offset, offset + term.action_dim)
        joint_ids = term.joint_ids
        if not isinstance(joint_ids, torch.Tensor):
            # ``joint_ids`` is a slice when the term resolves to "every joint of
            # the articulation"; the brake needs explicit ids to index its
            # per-wheel hold buffer.
            print(
                f"[WARN] M20VelocityEnv: '{term_name}' resolves to a slice of joints rather "
                "than explicit ids, so the wheel parking brake is disabled.",
                flush=True,
            )
            self._park_enabled = False
            return
        self._park_joint_ids = joint_ids.to(self.device)
        self._park_robot = self.scene["robot"]
        num_wheels = int(term.action_dim)
        self._park_q_hold = torch.zeros(self.num_envs, num_wheels, device=self.device)
        self._park_was_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _standing_command_mask(self) -> torch.Tensor | None:
        """Envs whose velocity command is exactly zero, from the command buffer.

        The buffer is the same source the observation and the gated wheel-speed
        reward use, so "the reward thinks it is standing" and "the brake thinks it
        is standing" cannot disagree. Returns ``None`` when the task has no
        velocity command at all (the standing task), in which case the brake stays
        out of the way.
        """
        manager = getattr(self, "command_manager", None)
        if manager is None:
            return None
        try:
            term = manager.get_term("base_velocity")
        except (KeyError, ValueError):
            return None
        buffer = getattr(term, "vel_command_b", None)
        if buffer is None:
            return None
        return (buffer[:, :3].abs() <= 1e-3).all(dim=1)

    def _park_wheels(self, action: torch.Tensor) -> torch.Tensor:
        """Replace the wheel columns with the position-loop command when parked.

        Returns a new tensor only when something actually changes, so the common
        case (driving) stays allocation-free.
        """
        if not self._park_enabled or self._park_slice is None or self._park_scale <= 0.0:
            return action
        standing = self._standing_command_mask()
        if standing is None:
            return action
        if not bool(standing.any()):
            # Driving: nothing to hold, and no state to update either. The hold is
            # seeded on the rising edge of "the command just became zero" below.
            self._park_was_standing = standing
            return action

        q = self._park_robot.data.joint_pos[:, self._park_joint_ids]

        # Re-seed the hold at an episode reset or on the rising edge of "the
        # command just became zero": the robot should stop where it is, not walk
        # back to where it was the last time it stood still.
        capture = (self.episode_length_buf <= 1) | (standing & ~self._park_was_standing)
        if bool(capture.any()):
            self._park_q_hold[capture] = q[capture]

        error = self._park_q_hold - q
        stale = error.abs() > self._park_error_clamp
        if bool(stale.any()):
            # Dragged or skidded further than the clamp: accept the new position
            # rather than commanding a large spin to undo it.
            self._park_q_hold[stale] = q[stale]
            error = torch.where(stale, torch.zeros_like(error), error)
        error = error.clamp(-self._park_error_clamp, self._park_error_clamp)

        omega = (self._park_kp * error).clamp(-self._park_omega_max, self._park_omega_max)
        if bool(standing.any()):
            action = action.clone()
            action[standing, self._park_slice] = (omega / self._park_scale)[standing]
        self._park_was_standing = standing
        return action

    # -- configuration checks -------------------------------------------------

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
        # The parking brake replaces the wheel columns while the command is zero.
        # This happens before the action term runs, so ``last_action`` (an
        # observation) and ``action_rate`` (a reward term) both see the command
        # that was actually sent, not the policy's ignored output.
        action = self._park_wheels(action)
        obs, rew, terminated, truncated, extras = super().step(action)
        if self._fall_penalty >= 0.0:
            return obs, rew, terminated, truncated, extras

        # A real fall is a termination that is not the episode time-out. Both flags
        # come straight out of the base class, so this stays correct even if the
        # termination terms change -- no second copy of the height/tilt test.
        fell = (terminated & ~truncated).to(device=rew.device, dtype=rew.dtype)
        return obs, rew + self._fall_penalty * fell, terminated, truncated, extras
