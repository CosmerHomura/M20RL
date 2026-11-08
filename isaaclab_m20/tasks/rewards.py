"""Reward terms for the two M20 tasks.

``Isaac-M20-Wheeled-v0`` (static standing) keeps the legged_gym / AnymalC term
set it has always used. ``Isaac-M20-Wheeled-Velocity-v0`` uses ten terms:

    r = + 0.5   · 1[alive]             per-step survival (not paid on the time-out step)
      + 2.0   · r_vel                 speed tracking, 1/(1 + e²/0.25²)
      + 1.0   · r_yaw                 yaw-rate tracking, 1/(1 + e²/0.3²)
      + 0.5   · r_height              ride height (exp, sigma 0.05 m)
      - 0.5   · p_ori                 (gx² + gy²) / 0.2²                    roll/pitch
      - 1.0   · p_pos                 1[cmd==0]·sum((p_xy - p_hold)²/0.25²) hold
      - 0.2   · p_pose                sum((q - q_default)²/0.3²)   legs only
      - 0.02  · p_qacc                sum((qdd / 200)²)            legs only
      - 0.01  · p_action              sum((Δa / 0.05)²)
      - 200   · 1[fallen]             fall penalty, NOT the time-out

**Ten, not sixteen.** The earlier set had a pair of channels for each axis group
-- an exp kernel *and* a quadratic for linear velocity, again for yaw -- plus a
gated wheel-speed term and regularisers that measured almost nothing (leg joint
velocity was 1% of the alive bonus, joint torque less). Two springs of different
stiffness on the same error cannot be attributed or tuned, which is exactly why
the table kept growing. The consolidation replaced pairs instead of adding to
them:

* one bounded kernel per axis group (``track_lin_vel_xy``, ``track_ang_vel_z``).
  The Cauchy form 1/(1+e²/σ²) keeps a usable gradient across the whole command
  range, where exp is already flat at 0.018 by the far end;
* ``stand_position_l2`` alone carries "a zero command means zero motion". It
  prices the *displacement* a drift produces -- 0.05 m/s costs 0.04/step after a
  second and 16/step after twenty -- which is what the acceptance test measures;
  a velocity channel would only double-charge the same drift;
* ``leg_joint_acc_l2`` (new) replaces the leg joint-velocity channel, and
  ``action_rate_l2`` is normalised by 0.05. Together they are the only thing that
  makes joint chatter cost anything: at the old weights a 3 rad/s swing cost
  0.005 per step.

The survival term and the fall-only penalty are a pair. ``mdp.is_alive`` is 1
until the terminating step and ``fall_terminated`` is 1 only when a *fall* ended
the episode, so standing 20 s and paying the time-out is worth ~+500, while
collapsing at step 40 is worth ~+20 - 200. A terminal penalty on
``mdp.is_terminated`` scores both the same and, perversely, makes ending the
episode early free.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``ang_vel_xy_l2`` returns ``w_xy²`` and the config gives it weight -0.05).
Do not add a leading minus here -- it would flip the term into a bonus.

Gate convention: only ``stand_position_l2`` is masked, and its mask is read from
the **command buffer** (strictly zero, ``eps = 1e-3``) rather than from
``is_standing_env``. A flag that nobody sets switches a whole channel off
silently; a buffer cannot. There is no deadband anywhere -- a zero command means
a stopped robot.

Also deliberately absent: any term that keeps all four wheels on the ground (the
trot lifts two legs on purpose), and any absolute wheel-speed or base-speed
penalty, which would tax driving itself. Zero-command braking is handled by
``p_pos`` (the wheels cannot roll while the base holds position).
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
# Velocity-command task (Isaac-M20-Wheeled-Velocity-v0)
# --------------------------------------------------------------------------- #
# One tracking channel per axis group, not two. The first version had an exp
# kernel AND a quadratic for linear velocity, and again for yaw; two springs of
# different stiffness on the same error, which is impossible to tune and was the
# reason the reward table kept growing. The kernel below is the Cauchy form
# 1 / (1 + e^2/sigma^2): bounded (so a badly-tracking early policy still gets a
# usable gradient instead of a 0/1 cliff), gently saturating, and -- unlike the
# masked stand-still channels -- it keeps its shape near zero error.
#
# Zero command needs no second channel for "do not move": `stand_position_l2`
# prices the *displacement*, which is what a drift actually produces.
# --------------------------------------------------------------------------- #
def track_lin_vel_xy(env, sigma: float = 0.25, command_name: str = "base_velocity",
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1 / (1 + ((vx-cmd_x)^2 + (vy-cmd_y)^2) / sigma^2), in (0, 1].

    ``sigma = 0.25 m/s`` against commands of +-0.5 / +-0.2: a 0.25 m/s error
    scores 0.5, a 0.5 m/s error 0.2 -- a gradient across the whole range, where an
    exp kernel is already flat at 0.02 by then.

    Both velocities are body-frame, and so is the command, so the comparison is
    frame-consistent by construction.
    """
    robot = _robot(env, asset_cfg)
    velocity = robot.data.root_lin_vel_b[:, :2]
    command = observations.velocity_command(env, command_name)[:, :2]
    error = torch.sum(torch.square(velocity - command), dim=1)
    return 1.0 / (1.0 + error / (sigma * sigma))


def track_ang_vel_z(env, sigma: float = 0.3, command_name: str = "base_velocity",
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1 / (1 + (w_z - cmd_wz)^2 / sigma^2), in (0, 1]: yaw-rate tracking."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    cmd = observations.velocity_command(env, command_name)[:, 2]
    error = torch.square(ang_vel_z - cmd)
    return 1.0 / (1.0 + error / (sigma * sigma))


def base_height_exp(env, target_height: float = 0.50, sigma: float = 0.05,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(h - h*)^2 / sigma^2), in [0, 1]: the base holds its ride height.

    ``sigma = 0.05 m`` is deliberately tight: base height is the cheapest proxy
    for "the legs are still carrying the robot" that this model has -- there are
    no foot contact sensors -- and a 10 cm squat should cost the channel.
    """
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.exp(-torch.square(height - target_height) / (sigma * sigma))


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


def action_rate_l2(env, action_scale: float = 1.0) -> torch.Tensor:
    """sum(((a - a_prev) / action_scale)^2) (positive).

    Raw ``sum((a - a_prev)^2)`` at weight -0.01 is *not* a smoothness penalty: a
    leg chattering by 0.1 action units per step costs 0.16 * 0.01 = 0.0016, i.e.
    0.3% of the 0.5 alive bonus -- free chatter. ``action_scale`` makes the number
    interpretable: with 0.05, a 0.05-unit step costs 0.05, a 0.10-unit step 0.20.
    """
    action_manager = env.action_manager
    delta = (action_manager.action - action_manager.prev_action) / action_scale
    return torch.sum(torch.square(delta), dim=1)


def leg_joint_acc_l2(env, acc_scale: float = 200.0,
                     asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """sum(((qdd / acc_scale)^2) over the 12 leg joints) (positive).

    The anti-chatter channel. Leg velocity alone cannot price it: 12 joints
    swinging at 3 rad/s cost 0.005 under any sane velocity weight, which is 1% of
    the 0.5 alive bonus. Joint *acceleration* separates the two regimes -- a gait
    accelerating at 100 rad/s^2 costs 12 * 0.25 * 0.02 = 0.06, a 400 rad/s^2 buzz
    costs 0.96.

    Legs only: the wheels reach 200 rad/s and are velocity-servoed, so their
    smoothness belongs in ``action_rate_l2``, not here. (This is deliberately a
    different function from ``joint_acc_l2`` above, which is the all-joint
    regulariser the standing task uses.)
    """
    robot = env.scene["robot"]
    if asset_cfg is None:
        asset_cfg = leg_asset_cfg()
    return torch.sum(
        torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids] / acc_scale), dim=1)


# --------------------------------------------------------------------------- #
# Standing hold: (where the robot is) - (where it stopped).
# No deadband: a zero command means zero motion, full stop. The tracking kernels
# are command-relative and stay ungated (subtracting the command keeps them free
# while driving); the position hold is gated on a *strictly* zero command, and the
# gate is read from the command buffer (see ``standing_mask``). A masked term that
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

    Used by ``stand_position_l2`` -- the only gated term in the set. The tracking
    kernels are command-relative and need no gate: subtracting the command already
    makes them free while driving.
    """
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd = observations.velocity_command(env, command_name)
    return (cmd.abs() <= eps).all(dim=1).to(dtype=torch.float32)


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

    ``scale = 0.25 m``: 0.05 m costs 0.04, 0.25 m costs 1.0, 1 m costs 16. It is
    the whole "stay put" mechanism now -- the tracking kernels only measure how
    well the commanded velocity is followed, this one measures where the robot
    ended up.
    """
    mask = standing_mask(env, command_name)
    pos_xy = _robot(env, asset_cfg).data.root_pos_w[:, :2]
    reference = _stand_reference(env, mask, pos_xy)
    _STAND_REF[id(env)] = (reference, mask.clone())
    drift = pos_xy - reference
    return mask * torch.sum(torch.square(drift / scale), dim=1)


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


def orientation_sq(env, scale: float = 0.2,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(gx^2 + gy^2) / scale^2 (positive; weight is negative).

    The cost grows as the square of the tilt instead of saturating. On flat
    ground the tilt should be small at all times, so a quadratic is the right
    shape -- it is the *stiffness near zero* that a hold-still command needs. On
    terrain, where a large tilt can be legitimate, swap this for an exp kernel so
    a climbing robot is not scored to death for the rest of the episode.
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
