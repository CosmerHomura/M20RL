"""The single place where a velocity command becomes a control target.

Everything downstream -- the policy observation, the reward, the parking brake,
``play`` and ``eval`` -- reads the numbers produced here.  Nothing else is
allowed to re-derive them: the failure mode this module exists to prevent is two
copies of ``psi_star`` (or of the command-frame rotation) drifting apart, which
turns "the reward improved" into an unattributable statement.

The semantics
-------------
The operator gives a command in a **command frame** that starts at the robot's
heading when the command was issued and rotates with the commanded yaw rate::

    psi_cmd(t) = psi0 + integral( wz_cmd dt )

``vx_cmd, vy_cmd`` are velocities expressed in that frame, so a pure lateral
command asks the robot to *travel sideways*, not to face sideways.  Two ways to
satisfy it: turn until the body points along the travel direction and drive
forward, or keep the heading and drive at an angle.  This module picks the
cheaper one per environment (two candidate headings ``alpha_f`` / ``alpha_r``,
with hysteresis) and hands the resulting **heading set point** to a yaw PD.  The
PD output, ``omega_target``, is what the policy tracks -- the policy never sees
``psi*`` itself, only "how fast should I be rotating now".

Two frames of reference, one transform
--------------------------------------
``psi_rel = psi - psi_cmd`` is how far the robot has rotated away from the
command frame.  A command-frame velocity maps to the body frame as::

    v_body = R(-psi_rel) . v_cmd
    R(t)   = [[cos t, -sin t], [sin t, cos t]]

i.e. ``v_body = (cos*psi_rel * vx + sin*psi_rel * vy,
                -sin*psi_rel * vx + cos*psi_rel * vy)``.
Getting the sign of ``psi_rel`` wrong rotates the target the wrong way, and the
robot drives in a mirrored direction while every diagnostic still looks sane.

Why the yaw loop takes a rate command, not a heading
----------------------------------------------------
``omega_target = psi_dot_cmd + Kp * e_psi - Kd * (psi_dot_meas - psi_dot_cmd)``.
``Kp`` has units of 1/s, ``Kd`` is dimensionless, and this is *not* a
second-order position loop: the inner "plant" is the policy tracking a yaw-rate
command, a cascade whose closed-loop behaviour cannot be read off ``2*sqrt(Kp)``.
The two gains are therefore initial values to be judged by the yaw RMSE in
``eval`` and the settling time in ``play``, not a damping ratio.

In ROTATE mode (heading fixed, yaw rate commanded) the heading error would grow
without bound as the robot spins, so the PD is bypassed and ``omega_target`` is
the commanded rate itself.

Heading measurement
-------------------
``psi`` comes from the root quaternion (yaw, ZYX); ``psi_dot_meas`` is the
*Euler* heading rate, not the body z gyro reading::

    psi_dot_meas = (w_y * sin(phi) + w_z * cos(phi)) / cos(theta)

with ``cos(theta)`` clamped at 0.5 (|pitch| <= 60 deg).  Near vertical the
division is meaningless and the expression degrades to ``w_z``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Command modes.  The sampler writes one of these into the command buffer next to
# the twist, so the reward reads a *label* rather than re-testing floats.  A
# float test would still work today (the sampler writes exact zeros), but the
# moment a command filter or noise is added, "the label says MOVE while the
# numbers say 0.001" becomes a silent inconsistency.
MODE_STAND = 0
MODE_MOVE = 1
MODE_ROTATE = 2
MODE_COMBINED = 3

MODE_NAMES = ("STAND", "MOVE", "ROTATE", "COMBINED")

COMMAND_TERM = "base_velocity"

# --------------------------------------------------------------------------- #
# Yaw outer loop defaults (see the module docstring for why these are only
# starting values).
# --------------------------------------------------------------------------- #
DEFAULT_KP = 1.2          # 1/s
DEFAULT_KD = 0.25         # dimensionless
DEFAULT_OMEGA_MAX = 1.5   # rad/s, clamp on omega_target
DEFAULT_OMEGA_RATE = 60.0  # rad/s^2, slew limit on omega_target
DEFAULT_ALPHA_TAU = 0.05  # s, first-order filter on the branch angle
DEFAULT_ALPHA_RATE = 3.0  # rad/s, slew limit on the branch angle
DEFAULT_HYSTERESIS = 0.15  # rad, margin before switching travel branch

_ZERO_EPS = 1e-6


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to (-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw of a (w, x, y, z) quaternion, ZYX convention."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class CommandTargets:
    """What the current command asks for, in the frames the consumers need.

    Attributes:
        v_body: (n, 2) target linear velocity in the **body** frame (m/s).
        omega:  (n,)   target yaw rate (rad/s).
        mode:   (n,)   one of ``MODE_*``, written by the sampler.
        psi_rel: (n,)  body heading minus command-frame heading (rad).
        psi_star: (n,) heading set point the PD is tracking (rad), for logging.
    """

    v_body: torch.Tensor
    omega: torch.Tensor
    mode: torch.Tensor
    psi_rel: torch.Tensor
    psi_star: torch.Tensor


class CommandTargetsComputer:
    """Per-environment state behind :func:`command_targets`.

    Stateful on purpose: ``psi_cmd`` is an integral and ``alpha_filt`` is
    filtered, so they cannot be recomputed from the current step alone.  Both are
    reset per environment when the episode ends.
    """

    def __init__(self, env):
        self.env = env
        self.device = env.device
        num_envs = env.num_envs
        self.psi_cmd = torch.zeros(num_envs, device=self.device)
        self.alpha_filt = torch.zeros(num_envs, device=self.device)
        self.omega = torch.zeros(num_envs, device=self.device)
        self.reverse = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stamp = -1
        self._zeros = torch.zeros(num_envs, device=self.device)
        self.targets = CommandTargets(
            v_body=torch.zeros(num_envs, 2, device=self.device),
            omega=torch.zeros(num_envs, device=self.device),
            mode=torch.zeros(num_envs, dtype=torch.long, device=self.device),
            psi_rel=torch.zeros(num_envs, device=self.device),
            psi_star=torch.zeros(num_envs, device=self.device),
        )

    # -- inputs ------------------------------------------------------------ #
    def _knob(self, name: str, default: float) -> float:
        cfg = getattr(self.env, "cfg", None)
        value = getattr(cfg, name, None) if cfg is not None else None
        try:
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _robot(self):
        return self.env.scene["robot"]

    def _command(self) -> torch.Tensor:
        """(n, 3) twist in the command frame; zeros when there is no term."""
        manager = getattr(self.env, "command_manager", None)
        term = manager.get_term(COMMAND_TERM) if manager is not None else None
        if term is None:
            return torch.zeros(self.env.num_envs, 3, device=self.device)
        buffer = getattr(term, "vel_command_b", None)
        if buffer is None:
            buffer = getattr(term, "command", None)
        if buffer is None:
            return torch.zeros(self.env.num_envs, 3, device=self.device)
        return buffer[:, :3]

    def _mode(self, command: torch.Tensor) -> torch.Tensor:
        """Mode label from the term if it publishes one, else derived from the twist."""
        manager = getattr(self.env, "command_manager", None)
        term = manager.get_term(COMMAND_TERM) if manager is not None else None
        buffer = getattr(term, "mode_buf", None) if term is not None else None
        if buffer is not None:
            return buffer.to(dtype=torch.long)

        moving = command[:, :2].norm(dim=1) > _ZERO_EPS
        spinning = command[:, 2].abs() > _ZERO_EPS
        mode = torch.full_like(command[:, 2], MODE_STAND, dtype=torch.long)
        mode = torch.where(moving & spinning, torch.full_like(mode, MODE_COMBINED), mode)
        mode = torch.where(moving & ~spinning, torch.full_like(mode, MODE_MOVE), mode)
        mode = torch.where(~moving & spinning, torch.full_like(mode, MODE_ROTATE), mode)
        return mode

    # -- reset ------------------------------------------------------------- #
    def reset(self, env_ids=None) -> None:
        """Re-seed the integral and the branch angle for the given environments."""
        ids = slice(None) if env_ids is None else env_ids
        robot = self._robot()
        psi = heading_from_quat(robot.data.root_quat_w)
        command = self._command()

        vx, vy = command[:, 0], command[:, 1]
        moving = command[:, :2].norm(dim=1) > _ZERO_EPS
        alpha_forward = torch.where(moving, torch.atan2(vy, vx), self._zeros)
        # A reverse command is satisfied either by turning round (forward branch)
        # or by keeping the heading and driving backwards (reverse branch).  At a
        # reset the reverse branch is always the closer one, so start there.
        self.reverse[ids] = False
        self.psi_cmd[ids] = psi[ids]
        self.alpha_filt[ids] = alpha_forward[ids]
        self.omega[ids] = 0.0
        self._stamp = -1

    # -- one update per environment step ----------------------------------- #
    def update(self) -> CommandTargets:
        env = self.env
        dt = float(getattr(env, "step_dt", 0.02))
        kp = self._knob("yaw_kp", DEFAULT_KP)
        kd = self._knob("yaw_kd", DEFAULT_KD)
        omega_max = self._knob("yaw_omega_max", DEFAULT_OMEGA_MAX)
        omega_rate = self._knob("yaw_omega_rate", DEFAULT_OMEGA_RATE)
        alpha_tau = max(self._knob("yaw_alpha_tau", DEFAULT_ALPHA_TAU), 1e-6)
        alpha_rate = self._knob("yaw_alpha_rate", DEFAULT_ALPHA_RATE)
        hysteresis = self._knob("yaw_branch_hysteresis", DEFAULT_HYSTERESIS)

        command = self._command()
        mode = self._mode(command)
        vx, vy, wz = command[:, 0], command[:, 1], command[:, 2]

        # The command frame rotates with the commanded yaw rate.
        self.psi_cmd = self.psi_cmd + wz * dt

        robot = self._robot()
        psi = heading_from_quat(robot.data.root_quat_w)
        gravity = robot.data.projected_gravity_b
        body_rate = robot.data.root_ang_vel_b

        moving = command[:, :2].norm(dim=1) > _ZERO_EPS
        alpha_forward = torch.where(moving, torch.atan2(vy, vx), self._zeros)
        alpha_reverse = torch.where(moving, wrap_to_pi(alpha_forward + torch.pi), self._zeros)

        # Pick the travel branch: the candidate heading closer to where the body
        # already points wins, and the other one must be better by `hysteresis`
        # before we switch, so a command near the deciding angle does not chatter.
        error_forward = wrap_to_pi(self.psi_cmd + alpha_forward - psi).abs()
        error_reverse = wrap_to_pi(self.psi_cmd + alpha_reverse - psi).abs()
        switch_to_reverse = (~self.reverse) & (error_reverse + hysteresis < error_forward)
        switch_to_forward = self.reverse & (error_forward + hysteresis < error_reverse)
        self.reverse = self.reverse ^ (switch_to_reverse | switch_to_forward)

        alpha_target = torch.where(self.reverse, alpha_reverse, alpha_forward)
        delta = wrap_to_pi(alpha_target - self.alpha_filt)
        delta = delta.clamp(-alpha_rate * dt, alpha_rate * dt)
        self.alpha_filt = wrap_to_pi(self.alpha_filt + (dt / (alpha_tau + dt)) * delta)

        psi_star = self.psi_cmd + self.alpha_filt
        psi_rel = wrap_to_pi(psi - self.psi_cmd)

        cos_rel = torch.cos(psi_rel)
        sin_rel = torch.sin(psi_rel)
        v_body_x = cos_rel * vx + sin_rel * vy
        v_body_y = -sin_rel * vx + cos_rel * vy

        # Euler heading rate from the body rates (see the module docstring).
        pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        cos_pitch = torch.cos(pitch).clamp(min=0.5)
        psi_dot_meas = (body_rate[:, 1] * torch.sin(roll)
                        + body_rate[:, 2] * torch.cos(roll)) / cos_pitch

        error_psi = wrap_to_pi(psi_star - psi)
        omega = wz + kp * error_psi - kd * (psi_dot_meas - wz)
        # ROTATE: the heading is free, so the PD would fight the spin.
        omega = torch.where(mode == MODE_ROTATE, wz, omega)
        omega = omega.clamp(-omega_max, omega_max)
        slew = (omega - self.omega).clamp(-omega_rate * dt, omega_rate * dt)
        self.omega = (self.omega + slew).clamp(-omega_max, omega_max)

        self.targets = CommandTargets(
            v_body=torch.stack((v_body_x, v_body_y), dim=1),
            omega=self.omega,
            mode=mode,
            psi_rel=psi_rel,
            psi_star=psi_star,
        )
        return self.targets


def computer(env) -> CommandTargetsComputer:
    """The computer attached to ``env`` (created on first use)."""
    instance = getattr(env, "_m20_command_targets", None)
    if instance is None:
        instance = CommandTargetsComputer(env)
        env._m20_command_targets = instance
    return instance


def command_targets(env) -> CommandTargets:
    """Current targets, refreshed at most once per environment step.

    Called from the reward manager and the observation manager, which both run
    after ``command_manager.compute()`` and share one outcome per step; the stamp
    keeps the integral from advancing twice inside a single step.
    """
    instance = computer(env)
    stamp = int(getattr(env, "common_step_counter", 0))
    if stamp != instance._stamp:
        instance._stamp = stamp
        instance.update()
    return instance.targets


def mode_mask(env, *modes: int) -> torch.Tensor:
    """1.0 where the current command mode is one of ``modes``, else 0.0."""
    targets = command_targets(env)
    mask = torch.zeros_like(targets.omega)
    for mode in modes:
        mask = torch.maximum(mask, (targets.mode == mode).to(dtype=mask.dtype))
    return mask


def mode_name(mode: int) -> str:
    return MODE_NAMES[int(mode)] if 0 <= int(mode) < len(MODE_NAMES) else "?"
