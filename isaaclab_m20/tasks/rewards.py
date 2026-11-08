"""Reward terms for the two M20 tasks.

``Isaac-M20-Wheeled-v0`` (static standing) keeps the legged_gym / AnymalC term
set it has always used. ``Isaac-M20-Wheeled-Velocity-v0`` uses the plan's term
set plus the static-hold and posture channels:

    r = + 0.5   · 1[alive]             per-step survival (not paid on the time-out step)
      + 2.0   · r_vel                 speed tracking (exp, sigma 0.25 m/s)
      + 1.0   · r_yaw                 yaw-rate tracking (exp, sigma 0.3 rad/s)
      + 0.5   · r_height              ride height (exp, sigma 0.05 m)
      - 0.5   · p_ori                 (gx² + gy²) / 0.2²        roll/pitch
      - 1.0   · p_v                   sum((v_xy - cmd_xy)²/0.15²)            quad
      - 0.5   · p_yaw                 ((w_z - cmd_wz)²/0.2²)                quad
      - 1.0   · p_pos                 1[cmd==0]·sum((p_xy - p_hold)²/0.25²) hold
      - 0.5   · p_wheel               1[cmd==0]·mean((r·qdot_w)²/0.3²)      gated
      - 0.2   · p_pose                sum((q - q_default)²/0.3²) legs only
      - 0.02  · p_qdot                sum((qdot / 20)²)          legs only
      - 0.1   · p_slip                wheel/ground longitudinal slip
      - 0.01  · p_action              action rate
      - 0.005 · p_torque              normalised joint torque
      - 0.05  · p_ang                 roll/pitch rate
      - 200   · 1[fallen]             fall penalty, NOT the time-out

The survival term and the fall-only penalty are a pair. ``mdp.is_alive`` is 1
until the terminating step and ``fall_terminated`` is 1 only when a *fall* ended
the episode, so standing 20 s and paying the time-out is worth ~+500, while
collapsing at step 40 is worth ~+20 - 200. A terminal penalty on
``mdp.is_terminated`` scores both the same and, perversely, makes ending the
episode early free.

The exp kernels are ``exp(-e²/σ²)``: σ is the *error scale* of the channel, and
the kernel is ~0 once the error is a few σ. Note the division by ``σ²``, not by
``σ``: this is the plan's formulation, and it is the reason the σ values are
0.25 m/s, 0.3 rad/s, 0.2 rad and 0.05 m -- read them as "the error that already
costs most of the channel", not as a width.

Three deliberate departures from the plan's eight channels:

* attitude is a *quadratic* (``orientation_sq``) instead of an exp kernel: on
  flat ground the body should be level at all times, so the cost has to be
  stiff near zero, and an exp kernel flattens out exactly where the hold-still
  command needs gradient. ``flat_orientation_exp`` is kept for the terrain
  version, where a large tilt can be legitimate;
* the "no wheel-speed penalty" rule became "no *unmasked absolute* wheel-speed
  penalty". What is penalised is the gap between the rolling speed the command
  asks for and the rolling speed the wheels have -- so a zero command brakes the
  wheels and a 0.3 m/s command costs nothing when the wheels roll at 0.3 m/s.
  Penalising ``qdot_w²`` directly would tax driving, which is the plan's
  original objection;
* the hold-still channels are *command-relative quadratics* (``p_v``, ``p_yaw``)
  plus a gated *position hold* (``p_pos``) and wheel gate (``p_wheel``), rather
  than zero-command-masked absolute penalties. A mask around the base channels
  would be a silent failure mode: if the command term does not flag standing (or
  the task has no command at all) the channel switches itself off and a "stand
  still" robot creeps. First-version numbers were 0.5/0.25² and 0.1/4², so a
  0.05 m/s creep cost 0.02 per step: with the exp tracking term flat at zero
  error, nothing held the base in place, and stage 1 wandered -- typically
  drifting forward and slowly rotating, because yaw had *only* the exp kernel and
  no quadratic at all.

  There is **no deadband** on any of them. A zero command means zero motion: the
  wheels have ``stiffness = 0`` and ``damping = 2``, i.e. the chassis is a
  free-rolling cart whose only brake is wheel damping, so the last few cm/s are
  set by contact impulses and are genuinely expensive to remove -- but the answer
  is to charge for the *displacement* those cm/s produce (``p_pos``, which pays
  for the integral and therefore stops the drift at its source, and also pays for
  coming back after a push), not to declare the residual acceptable. Only
  ``p_wheel`` and ``p_pos`` are gated (strictly-zero command, read from the
  command buffer at ``eps = 1e-3``), because an ungated wheel term taxes every
  skid-steer turn and an ungated position term forbids driving away.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``ang_vel_xy_l2`` returns ``w_xy²`` and the config gives it weight -0.05).
Do not add a leading minus here -- it would flip the term into a bonus.

Also deliberately absent: any term that keeps all four wheels on the ground --
the trot gait lifts two legs at a time on purpose. ``p_slip`` is a mean over the
four wheels and is weighted 0.1 for the same reason: a wheel that is in the air
for part of the cycle must not be able to dominate the reward.
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


# --------------------------------------------------------------------------- #
# Command-relative quadratics: (what the base/wheels do) - (what was asked).
# No deadband: a zero command means zero motion, full stop. The two base terms
# are ungated (subtracting the command keeps them free while driving); the wheel
# term and the position hold are gated on a *strictly* zero command, and the gate
# is read from the command buffer (see ``standing_mask``). A masked term that
# never fires (an env whose command term forgets to flag standing, a task with no
# command at all) fails silently; a subtraction cannot.
# --------------------------------------------------------------------------- #
def standing_mask(env, command_name: str = "base_velocity",
                  eps: float = 1e-3) -> torch.Tensor:
    """1.0 where the commanded twist is (strictly) zero, else 0.0 -- shape (n,).

    Read from the command buffer itself, **not** from ``is_standing_env``: that
    flag has to be written by the sampler, and a sampler that forgets to write it
    (or a task with no command term) silently zeroes a whole reward channel --
    which is one of the ways a "stand still" command starts sliding. The buffer is
    the ground truth; ``eps = 1e-3`` makes the gate strict, so a 0.05 m/s command
    counts as *driving* and does not receive the stand-still treatment.

    Used by ``wheel_roll_l2_error`` and ``stand_position_l2``. The base-velocity
    terms are command-relative and need no gate: subtracting the command already
    makes them free while driving.
    """
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd = observations.velocity_command(env, command_name)
    return (cmd.abs() <= eps).all(dim=1).to(dtype=torch.float32)


def lin_vel_xy_l2_error(env, scale: float = 0.2, deadband: float = 0.0,
                        command_name: str = "base_velocity",
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum(((max(0, |v_xy - cmd_xy| - deadband)) / scale)^2) (positive).

    The tracking term is an exp kernel, so at a zero command a 0.05 m/s drift
    still scores ``2.0 * exp(-0.04) = 1.92`` -- it loses 0.08 -- and the gradient
    there is exactly zero, so nothing pushes the base back. This quadratic is
    the missing stiffness: with ``scale = 0.15`` and no deadband, 0.05 m/s costs
    0.11, 0.1 m/s 0.44, 0.2 m/s 1.78, and the gradient grows with the error
    instead of vanishing.

    ``deadband`` (m/s, per axis) exists only as a tuning knob and defaults to
    0.0: a zero command means the base must not move, so there is no free zone.
    ``stiffness = 0`` on the wheels means the chassis is a free-rolling cart with
    no parking brake, which makes the last cm/s expensive to remove -- but that is
    a reason to add the position hold below (which pays for the *displacement*
    those cm/s produce, and so stops the drift at its source), not a reason to
    score the residual as acceptable.

    It is measured against the command, not against zero, so it does not tax
    driving: a robot tracking 0.3 m/s perfectly pays 0 here, and the exp kernel
    still provides the long-range behaviour where the quadratic would explode.
    """
    cmd = observations.velocity_command(env, command_name)[:, :2]
    lin_vel = _robot(env, asset_cfg).data.root_lin_vel_b[:, :2]
    excess = torch.clamp((lin_vel - cmd).abs() - deadband, min=0.0)
    return torch.sum(torch.square(excess / scale), dim=1)


def yaw_rate_l2_error(env, scale: float = 0.2, deadband: float = 0.0,
                      command_name: str = "base_velocity",
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(max(0, |w_z - cmd_wz| - deadband) / scale)^2 (positive).

    Yaw is the one channel where "stands still but slowly rotates" was the
    cheapest escape: the exp kernel peaks at zero error, so a 0.5 rad/s spin
    only lost 0.94 of its 1.0 channel. ``scale = 0.2 rad/s``: 0.1 rad/s costs
    0.25, 0.3 rad/s costs 2.25, and a full-range 0.8 rad/s command costs 0 when
    it is actually followed. ``deadband`` defaults to 0.0 -- a zero yaw command
    means no rotation.

    Roll/pitch rate stays in ``ang_vel_xy_l2`` (there is no roll-rate command to
    measure against, and the target is always zero).
    """
    cmd = observations.velocity_command(env, command_name)[:, 2]
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    excess = torch.clamp((ang_vel_z - cmd).abs() - deadband, min=0.0)
    return torch.square(excess / scale)


# Where each env's "hold here" reference is anchored. Per-env state, so it lives
# in a module-level cache keyed by the env instance rather than in the cfg.
_STAND_REF: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _stand_reference(env, mask: torch.Tensor, pos_xy: torch.Tensor) -> torch.Tensor:
    """Position each env must hold while its command is zero -- shape (n, 2).

    The reference is re-anchored on the **rising edge** of the stand mask (and on
    every episode reset), i.e. "hold where you stopped". Anchoring it to the spawn
    point instead would fight the driving stages: after 3 s at 0.5 m/s the robot is
    1.5 m away, and a sampled zero command would then demand walking back to the
    spawn -- which teaches "never drive far" rather than "stop cleanly". A push
    still produces drift away from the reference, so recovery is still rewarded.
    """
    key = id(env)
    state = _STAND_REF.get(key)
    reset = env.episode_length_buf <= 1
    if state is None or state[0].shape != pos_xy.shape or state[0].device != pos_xy.device:
        return pos_xy.clone()
    reference, last_mask = state
    reference = reference.clone()
    reanchor = ((mask > 0.5) & (last_mask < 0.5)) | reset
    if bool(reanchor.any()):
        reference[reanchor] = pos_xy[reanchor]
    return reference


def stand_position_l2(env, scale: float = 0.25, command_name: str = "base_velocity",
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1[cmd == 0] * sum(((p_xy - p_xy_hold) / scale)^2) (positive).

    The displacement counterpart of the velocity quadratics, and the term that
    makes "the command is zero, so the robot must not move" actually hold. A
    velocity penalty alone is a damper: it charges for the *rate*, so a constant
    0.05 m/s creep costs 0.11 per step -- noticeable, but a position term charges
    for the *integral*, and 0.05 m/s becomes 1 m of drift (cost 16 per step) within
    one 20 s episode. That is also what recovers from a push: the pusher moves the
    base, drift grows, and the term pays for coming back.

    Gated on a strictly-zero command (see ``standing_mask``) and measured from the
    position at which the command went to zero, so it is free while driving and
    never asks the robot to return to where the episode started.

    ``scale = 0.25 m``: 0.05 m costs 0.04, 0.25 m costs 1.0, 1 m costs 16. Read it
    together with ``lin_vel_xy_l2_error`` (``scale = 0.15 m/s``): the velocity term
    sets how hard the base brakes, this one sets how hard it comes back.
    """
    mask = standing_mask(env, command_name)
    pos_xy = _robot(env, asset_cfg).data.root_pos_w[:, :2]
    reference = _stand_reference(env, mask, pos_xy)
    _STAND_REF[id(env)] = (reference, mask.clone())
    drift = pos_xy - reference
    return mask * torch.sum(torch.square(drift / scale), dim=1)


def wheel_roll_l2_error(env, scale: float = 0.3, deadband: float = 0.0,
                        command_name: str = "base_velocity",
                        wheel_radius: float = WHEEL_RADIUS, wheel_sign: float = -1.0,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg(
                            "robot", joint_names=WHEEL_JOINT_NAMES)) -> torch.Tensor:
    """standing * mean_wheel (((r*qdot_w) - cmd_vx) / scale)^2 (positive).

    With a zero command the wheels should not be rolling at all, so this is the
    channel that stops a "stand still" robot from rolling away: ``wheel_sign * r
    * qdot`` is the ground speed the wheels are trying to produce, and at a zero
    command it must be zero.

    **Gated on a strictly-zero command** (``standing_mask``, read from the
    command buffer, ``eps = 1e-3``), for two reasons:

    * while driving, the base-velocity terms above already charge tracking error,
      and an ungated wheel term would charge the same mistake twice;
    * skid-steer yaw needs the left/right wheels to roll in opposite directions,
      so an ungated term taxes every turn in place (0.06/step at 0.5 rad/s). That
      is a real cost for a legal command, so the term is off while driving.

    ``scale = 0.3 m/s`` = 3.3 rad/s of rolling speed. ``deadband`` defaults to
    0.0 for the same reason as the base terms: a zero command means no rolling.
    The mean over four wheels (not the sum) keeps one lifted, free-spinning wheel
    from dominating.
    """
    mask = standing_mask(env, command_name)
    cmd_x = observations.velocity_command(env, command_name)[:, 0]
    robot = _robot(env, asset_cfg)
    wheel_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    rolling = wheel_sign * wheel_radius * wheel_vel
    excess = torch.clamp((rolling - cmd_x.unsqueeze(1)).abs() - deadband, min=0.0)
    return mask * torch.mean(torch.square(excess / scale), dim=1)


def joint_deviation_l2(env, scale: float = 0.3,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg(
                           "robot", joint_names=LEG_JOINT_NAMES)) -> torch.Tensor:
    """sum_legs ((q - q_default) / scale)^2 (positive; weight is negative).

    The posture channel: it keeps the legs near the stance they were designed
    for instead of whatever folded shape happens to balance. Legs only -- a wheel
    angle is unbounded (see ``joint_deviation_l1``).

    ``scale = 0.3 rad`` makes a 0.3 rad offset on one joint cost 1.0; 12 joints
    at 0.1 rad cost ``12 * 0.11 = 1.3``, which the default weight -0.2 turns into
    -0.27 per step, about an eighth of the tracking channel.

    Caveat: it pulls towards ``default_joint_pos``. If that nominal stance is
    itself pitched (unequal hip offsets front to back), this term *enforces* the
    pitch instead of correcting it -- the attitude term is what levels the body.
    Keep the weight modest and fix the default pose rather than out-weighting it.
    """
    robot = _robot(env, asset_cfg)
    deviation = (robot.data.joint_pos[:, asset_cfg.joint_ids]
                 - robot.data.default_joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(deviation / scale), dim=1)


def joint_vel_legs_l2(env, scale: float = 20.0,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg(
                          "robot", joint_names=LEG_JOINT_NAMES)) -> torch.Tensor:
    """sum_legs (qdot / scale)^2 over the 12 leg joints (positive; weight negative).

    Legs only, same reason as ``joint_deviation_l2``: wheel speed is commanded
    and is covered by ``wheel_roll_l2_error`` when it matters. ``scale = 20
    rad/s`` is half the leg actuator's 40 rad/s limit.
    """
    joint_vel = _robot(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel / scale), dim=1)


def orientation_sq(env, scale: float = 0.2,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(gx^2 + gy^2) / scale^2 (positive; weight is negative).

    The hard version of ``flat_orientation_exp``: same error measure, but the
    cost grows as the square of the tilt instead of saturating. On flat ground
    the tilt should be small at all times, so a quadratic is the right shape --
    it is the *stiffness near zero* that a hold-still command needs. The exp
    kernel is kept for the day the robot trains on terrain, where a large tilt
    can be legitimate and must not be scored to death.
    """
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    tilt = torch.square(gravity[:, 0]) + torch.square(gravity[:, 1])
    return tilt / (scale * scale)


def fall_terminated(env, minimum_height: float = 0.33, max_tilt: float = 0.5,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1.0 for envs that ended in a *failure*, 0.0 otherwise (positive; weight is negative).

    Deliberately not ``mdp.is_terminated``: that fires on the 20 s time-out too,
    so a policy that stands perfectly for the whole episode would pay the same
    -200 as one that collapses at step 40. With the terminal penalty costing the
    same either way, nothing rewards *staying* upright -- it only rewards ending
    fast. This is the fall-only version.

    The two conditions mirror ``TerminationsCfg`` exactly (height below
    ``minimum_height``, tilt from ``acos(-g_z)`` above ``max_tilt``) and must be
    kept in sync with it: the term that ends the episode is the term that pays
    the penalty. It is evaluated through the same projected-gravity reading
    ``mdp.bad_orientation`` uses, so it does not depend on
    ``TerminationManager.time_outs`` existing in this Isaac Lab version.
    """
    robot = _robot(env, asset_cfg)
    height = robot.data.root_pos_w[:, 2]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    fallen = (height < minimum_height) | (tilt > max_tilt)
    return fallen.to(dtype=torch.float32)
