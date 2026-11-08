"""Reward terms for the two M20 tasks.

``Isaac-M20-Wheeled-v0`` (static standing) keeps the legged_gym / AnymalC term
set it has always used. ``Isaac-M20-Wheeled-Velocity-v0`` uses the eight-term
set of the first formal training plan:

    r = 2.0 · r_vel
      + 1.0 · r_yaw
      + 0.5 · r_ori                     roll/pitch attitude
      + 0.5 · r_height
      - 0.1 · p_slip                    wheel/ground longitudinal slip
      - 0.01 · p_action                 action rate
      - 0.005 · p_torque                normalised joint torque
      - 0.05 · p_ang                    roll/pitch rate
      - 200 · 1[terminated]             fall penalty

The four exp kernels are ``exp(-e²/σ²)``: σ is the *error scale* of the
channel, and the kernel is ~0 once the error is a few σ. Note the division by
``σ²``, not by ``σ``: this is the plan's formulation, and it is the reason the
σ values are 0.25 m/s, 0.3 rad/s, 0.2 rad and 0.05 m -- read them as "the error
that already costs most of the channel", not as a width.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``ang_vel_xy_l2`` returns ``w_xy²`` and the config gives it weight -0.05).
Do not add a leading minus here -- it would flip the term into a bonus.

Two things are deliberately NOT in the set, because both would fight the task:

* no penalty on the wheels' absolute speed -- driving forward *requires* the
  wheels to turn, so ``|w_wheel| -> 0`` is a reward for refusing the command;
* no term that keeps all four wheels on the ground -- the trot gait lifts two
  legs at a time on purpose. ``p_slip`` is a mean over the four wheels and is
  weighted 0.1 for the same reason: a wheel that is in the air for part of the
  cycle must not be able to dominate the reward.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab_m20.tasks import observations
from isaaclab_m20.tasks.observations import LEG_JOINT_NAMES, WHEEL_JOINT_NAMES

# Every ``*_wheel`` link in model/urdf/M20.urdf is
# <cylinder length="0.054" radius="0.09"/>. The wheel action scale times this
# radius is the speed ceiling; it is also the rolling-speed gain below.
WHEEL_RADIUS = 0.09


def leg_asset_cfg() -> SceneEntityCfg:
    """A fresh SceneEntityCfg for the 12 leg joints.

    Fresh per call, not a module-level constant: the manager resolves the cfg in
    place (it writes ``joint_ids`` back onto it), and two reward terms sharing
    one instance would then share those indices. Here that happens to be
    harmless, but it is not a habit worth keeping.
    """
    return SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)


def wheel_asset_cfg() -> SceneEntityCfg:
    """A fresh SceneEntityCfg for the 4 wheel joints."""
    return SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


# --------------------------------------------------------------------------- #
# Standing task (Isaac-M20-Wheeled-v0): the command is hard-wired to zero
# --------------------------------------------------------------------------- #
def track_lin_vel_xy_zero(env, sigma: float = 0.25,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-|v_xy|^2/sigma), in [0, 1]: the legged_gym tracking kernel at a zero command."""
    lin_vel = _robot(env, asset_cfg).data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel), dim=1) / sigma)


def track_ang_vel_z_zero(env, sigma: float = 0.25,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-w_z^2/sigma), in [0, 1]: yaw-rate tracking at a zero command."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(ang_vel_z) / sigma)


def wheel_vel_exp(env, sigma: float = 2.0,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg(
                      "robot", joint_names=[".*_wheel_joint"])) -> torch.Tensor:
    """exp(-|w_wheel|^2 / sigma^2), in [0, 1]: the four wheels hold still.

    Standing only. A velocity-command task must NOT use this -- it rewards the
    robot for not doing what it was told.
    """
    wheel_vel = _robot(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]
    return torch.exp(-torch.sum(torch.square(wheel_vel), dim=1) / (sigma * sigma))


# --------------------------------------------------------------------------- #
# Velocity-command task (Isaac-M20-Wheeled-Velocity-v0): 8 terms + fall penalty
# --------------------------------------------------------------------------- #
def track_lin_vel_xy_cmd(env, sigma: float = 0.25, command_name: str = "base_velocity",
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-((vx-cmd_x)^2 + (vy-cmd_y)^2) / sigma^2), in [0, 1].

    One kernel over both linear axes, as in the plan. ``sigma = 0.25 m/s``
    against a command range of +-0.5 / +-0.2: a 0.25 m/s error keeps
    ``exp(-1) = 0.37``, a 0.5 m/s error (the whole forward range) keeps 0.018.
    That is a usable gradient across the whole range, which is what makes the
    exponential preferable to an L2 penalty here -- the L2 version has no
    gradient left once the channel has saturated.

    Both velocities are body-frame. So is the command, so the comparison is
    frame-consistent by construction.
    """
    robot = _robot(env, asset_cfg)
    velocity = robot.data.root_lin_vel_b[:, :2]
    command = observations.velocity_command(env, command_name)[:, :2]
    error = torch.sum(torch.square(velocity - command), dim=1)
    return torch.exp(-error / (sigma * sigma))


def track_ang_vel_z_cmd(env, sigma: float = 0.3, command_name: str = "base_velocity",
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(w_z - cmd_wz)^2 / sigma^2), in [0, 1]: yaw-rate tracking."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    cmd = observations.velocity_command(env, command_name)[:, 2]
    return torch.exp(-torch.square(ang_vel_z - cmd) / (sigma * sigma))


def flat_orientation_exp(env, sigma: float = 0.2,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(gx^2 + gy^2) / sigma^2), in [0, 1]: roll and pitch small.

    ``g`` is gravity projected on the base frame. For roll φ and pitch θ,
    ``gx = sin θ`` and ``gy = -sin φ cos θ``, so ``gx^2 + gy^2 = φ^2 + θ^2`` to
    first order -- the plan's ``(φ² + θ²)`` without an ``asin``/``atan2`` per
    step. At ``sigma = 0.2 rad`` (11.5 deg) a 10 deg tilt keeps 0.72, a 20 deg
    tilt keeps 0.13.

    The exp form is what makes this usable at all here: as a *penalty*
    (``-2.5·(gx²+gy²)``, the standing task) it is a spring towards level with a
    constant stiffness; as an exp kernel it saturates, so a recovered robot can
    score almost 1 again instead of being marked down for the rest of the
    episode.
    """
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    return torch.exp(-(torch.square(gravity[:, 0]) + torch.square(gravity[:, 1]))
                     / (sigma * sigma))


def base_height_exp(env, target_height: float = 0.50, sigma: float = 0.05,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(h - h*)^2 / sigma^2), in [0, 1]: the base holds its ride height.

    ``sigma = 0.05 m`` is deliberately tight: base height is the cheapest proxy
    for "the legs are still carrying the robot" that this model has -- there are
    no foot contact sensors -- and a 10 cm squat should cost the channel.
    """
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.exp(-torch.square(height - target_height) / (sigma * sigma))


def wheel_slip_l2(env, slip_scale: float = 0.5,
                  wheel_radius: float = WHEEL_RADIUS,
                  wheel_sign: float = -1.0,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg(
                      "robot", joint_names=[".*_wheel_joint"])) -> torch.Tensor:
    """mean_wheel ((v_roll - v_x) / slip_scale)^2, positive (weight is negative).

    ``v_roll = wheel_sign * radius * qdot`` is the speed the wheel implies when
    it rolls without slipping; ``v_x`` is the body's forward speed in the base
    frame. ``v_roll - v_x`` is therefore the longitudinal slip at the contact
    patch: positive while the wheels spin faster than the robot moves (wheelspin
    on the spot, or on a low-friction surface), negative while the robot is
    dragged along on locked wheels. Both waste the only forward actuator M20
    has, and neither is visible to the tracking term when the command itself is
    small.

    Only the LONGITUDINAL direction is penalised, on purpose. The wheels are not
    omnidirectional: during a lateral command the tyres scrub sideways, and that
    scrub is the mechanism that produces the lateral motion. Penalising it would
    penalise the behaviour the task is asking for.

    ``slip_scale = 0.5 m/s`` is a reference, not a limit: a 0.5 m/s slip costs
    1.0 per wheel, so the term stays small next to the 2.0 tracking channel at
    ``weight = -0.1``.

    Sign note: every wheel joint turns about ``(0, -1, 0)`` in the base frame at
    the nominal stance. A spin about -y moves the contact point at the bottom of
    the wheel along +x, so rolling without slip puts the *body* at
    ``v_x = -radius * qdot``. Hence ``wheel_sign = -1``. If the trained policy
    drives backwards when commanded forwards, this sign is the first suspect:
    flip it to +1.0.
    """
    robot = _robot(env, asset_cfg)
    wheel_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    v_roll = wheel_sign * wheel_radius * wheel_vel            # (num_envs, 4)
    v_forward = robot.data.root_lin_vel_b[:, 0:1]             # (num_envs, 1)
    slip = (v_roll - v_forward) / slip_scale
    return torch.mean(torch.square(slip), dim=1)


# --------------------------------------------------------------------------- #
# Shared by both tasks
# --------------------------------------------------------------------------- #
def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical velocity squared (positive; weight is negative)."""
    return torch.square(_robot(env, asset_cfg).data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, scale: float = 1.0,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll/pitch rate squared, normalised by ``scale`` (rad/s).

    ``scale = 1.0`` keeps the standing task's raw ``sum(w_xy^2)``. The velocity
    task states it explicitly as "normalised squared penalty": 1 rad/s of roll
    rate is one unit of cost, which at ``weight = -0.05`` is a 0.05-per-rad/s
    tax on wobble -- small enough to permit the body sway a trot needs, large
    enough to stop the robot from rocking itself over.
    """
    ang_vel = _robot(env, asset_cfg).data.root_ang_vel_b[:, :2]
    return torch.sum(torch.square(ang_vel / scale), dim=1)


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the base xy plane (positive; weight is negative)."""
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    return torch.square(gravity[:, 0]) + torch.square(gravity[:, 1])


def base_height_l2(env, target_height: float = 0.50,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic height error (positive; weight is negative)."""
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.square(height - target_height)


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"])
        ) -> torch.Tensor:
    """L1 distance from the default stand pose over the LEG joints (positive; weight is negative).

    Standing task only in this version -- the eight-term velocity set has no
    posture channel. It is kept here (and one config line away in the velocity
    task) because it is the term that keeps the left and right legs mirrored:
    the default stance is symmetric, so a symmetric deviation costs the sum of
    both offsets while an asymmetric one costs the same and looks wrong.

    Wheels are deliberately excluded. A wheel joint's absolute angle is
    unbounded -- it grows without limit as the wheel rolls -- so penalising it
    against a default of 0 adds a term that never saturates and swamps the rest
    of the reward after a couple of turns.
    """
    robot = _robot(env, asset_cfg)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env, asset_cfg).data.joint_vel), dim=1)


def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env, asset_cfg).data.joint_acc), dim=1)


def joint_torques_l2(env, torque_scale: float = 1.0,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum((tau / torque_scale)^2), positive (weight is negative).

    ``torque_scale`` is the expression of "normalised": with ``scale = 40 N·m``
    one unit of cost is 40 N·m on one joint, so the leg actuators (76.4 N·m
    limit) reach 3.6 at saturation and the wheels (21.6 N·m) 0.29. The plan's
    ``weight = -0.005`` then costs a fully loaded leg 0.018 per step -- a
    regulariser, not a task term. Raw ``tau^2`` would instead scale with
    (76.4)^2 and need a weight near 1e-5, which is where the standing task is.
    """
    torque = _robot(env, asset_cfg).data.applied_torque
    return torch.sum(torch.square(torque / torque_scale), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    action_manager = env.action_manager
    return torch.sum(
        torch.square(action_manager.action - action_manager.prev_action), dim=1)
