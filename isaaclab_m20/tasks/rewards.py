"""Reward terms for the two M20 tasks.

``Isaac-M20-Wheeled-v0`` (static standing) keeps the legged_gym / AnymalC term
set it has always used. ``Isaac-M20-Wheeled-Velocity-v0`` uses ten per-step
terms, plus a fall penalty that is applied by the environment rather than by a
reward term:

    r = + 2.0   · r_vel                 speed tracking, 1/(1 + e²/0.25²)
      + 1.0   · r_yaw                 yaw-rate tracking, 1/(1 + e²/0.3²)
      - 0.5   · p_h                   ((h - 0.50)/0.05)²                  ride height
      - 0.5   · p_ori                 (gx² + gy²) / 0.2²                    roll/pitch
      - 0.2   · p_vz                  (v_z / 0.2)²                     vertical bounce
      - 0.2   · p_ang                 sum(w_xy² / 1²)                       wobble
      - 0.2   · p_wheel               1[cmd==0]·mean((qdot_w / 2)²)   spinning wheels
      - 0.05  · p_pose                sum((q - q_default)²/0.3²)   legs only
      - 0.02  · p_qacc                sum((qdd / 2000)²)           legs only
      - 0.01  · p_action              sum((Δa / 1.0)²)

    terminal: -10 on the fall step only, applied by ``M20VelocityEnv`` (see
    ``fall_penalty`` on the env cfg). It is not one of the terms below because the
    termination flags are only known after the reward manager has run.

**Ten, not sixteen.** The earlier set had a pair of channels for each axis group
-- an exp kernel *and* a quadratic for linear velocity, again for yaw -- plus a
gated position hold and regularisers that measured almost nothing (leg joint
velocity was 1% of the height term, joint torque less). Two springs of different
stiffness on the same error cannot be attributed or tuned, which is exactly why
the table kept growing. The consolidation replaced pairs instead of adding to
them:

* one bounded kernel per axis group (``track_lin_vel_xy``, ``track_ang_vel_z``).
  The Cauchy form 1/(1+e²/σ²) keeps a usable gradient across the whole command
  range, where exp is already flat at 0.018 by the far end;
* ride height is a plain quadratic error, not an exp bonus: on flat ground the
  target is a set point, so what the channel needs is stiffness around it, and a
  quadratic has no ceiling to saturate against;
* ``leg_joint_acc_l2`` replaces the leg joint-velocity channel, and
  ``action_rate_l2`` (both at their un-scaled magnitudes) make joint chatter cost
  something.

**No survival bonus, and no reward floor either.** Both were tried and both were
compensations for other mistakes, so neither is in this file. The floor
(``only_positive_rewards``, a clamp at 0 on the per-step reward) is only safe
while the ordinary step reward is positive; while the policy is bad it is not, so
the clamp turned every step into a constant 0 and the gradient disappeared -- the
symptom was a mean reward of exactly 0 for a whole run. An ``alive`` bonus then
had to be added to lift the step above the clamp. The root cause was the reward
shape (penalties that do not shrink when the policy stops trying) plus
normalisation scales that priced exploration noise at -30/step; both are fixed,
which leaves no reason for either the clamp or the bonus. PPO does not need
non-negative rewards: ``A = R - V`` normalised rescales without reordering, so a
step that is merely less bad than the critic expected still gets a positive
advantage.

**Normalisation scales are anti-noise, not readability.** ``action_scale = 0.05``
and ``acc_scale = 200`` read nicely ("a 0.05 action step counts as 1") but they
multiply the *exploration* noise by the same factor: actions are resampled every
control step from N(0, sigma), so at ``init_noise_std = 0.5`` the step-to-step
difference has std 0.7 and ``sum((da/0.05)²)`` is ~3200 -- -32 per step at weight
-0.01, dwarfing the +3 the tracking terms can pay. At ``action_scale = 1.0`` /
``acc_scale = 2000`` the same noise costs 0.08 / 0.17. The honest cost: a quadratic
cannot tell noise from chatter, so these two terms are mild on real vibration
until the noise floor drops -- vibration is mostly priced by the return it loses.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``ang_vel_xy_l2`` returns ``w_xy²`` and the config gives it weight -0.2).
Do not add a leading minus here -- it would flip the term into a bonus.

Gate convention: ``wheel_speed_l2`` is the only masked term, and the mask is read
from the **command buffer** (strictly zero, ``eps = 1e-3``) rather than from
``is_standing_env``. A flag that nobody sets switches a whole channel off
silently; a buffer cannot. There is no deadband anywhere. Standing is held by the
tracking kernels themselves (a zero command asks for zero velocity) -- a separate
gated ``||v_xy||²`` term would be a second spring on the same error.

**No position hold in this revision.** Getting back to the spot you were pushed
from is a third capability, on top of "move", "stop" and "stand". It is verified
on its own (net displacement in the acceptance test) before it earns a reward
term; stacking it on the same reward as the other three makes every failure
ambiguous.

Also deliberately absent: any term that keeps all four wheels on the ground (the
trot lifts two legs on purpose), and any un-gated wheel-speed or base-speed
penalty, which would tax driving itself.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab_m20.tasks import observations
from isaaclab_m20.tasks.observations import LEG_JOINT_NAMES, WHEEL_JOINT_NAMES


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
# Zero command needs no second channel for "do not move": the kernels are already
# relative to the command, so a zero command asks for zero velocity, and the only
# thing they cannot see is a wheel spinning under a parked body -- that is what
# the gated `wheel_speed_l2` is for.
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


# --------------------------------------------------------------------------- #
# Shared by both tasks
# --------------------------------------------------------------------------- #
def lin_vel_z_l2(env, scale: float = 1.0,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical velocity squared, normalised by ``scale`` (m/s).

    ``scale = 1.0`` keeps the standing task's raw ``v_z²``. The velocity task
    passes ``0.2`` so that a 0.2 m/s of vertical bounce is one unit of cost:
    with no foot contacts, a bouncing body is the cheapest way to look like it is
    walking, and the roll/pitch channels do not see a bounce at all.
    """
    lin_vel_z = _robot(env, asset_cfg).data.root_lin_vel_b[:, 2]
    return torch.square(lin_vel_z / scale)


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


def base_height_l2(env, target_height: float = 0.50, scale: float = 1.0,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic height error, normalised by ``scale`` (positive; weight is negative).

    Both tasks use the same function with different normalisation: the standing
    task keeps the raw metres (``scale = 1.0``, weight -20), the velocity task
    prices 5 cm as one unit (``scale = 0.05``, weight -0.5) so that height is
    stiffness around a set point rather than a bonus with a ceiling. On flat
    ground 0.50 m is where the legs carry the robot; losing 5 cm means the knees
    have folded and the next event is the fall termination.
    """
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.square((height - target_height) / scale)


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

    ``action_scale`` is left at 1.0 on purpose. It is tempting to normalise by
    the smallest step that "should" be free (0.05 action units), but the same
    normalisation multiplies the *exploration noise* by the same factor: actions
    are resampled every control step from N(0, sigma), so with ``init_noise_std =
    0.5`` the step-to-step difference has std 0.7 and ``sum((da / 0.05)^2)`` is
    about 3200 -- i.e. -32 reward per step at weight -0.01, dwarfing the +3 the
    tracking terms can pay, so the noise (not the policy) would set the cost.

    At ``action_scale = 1.0`` the same noise costs ``8 * 0.01 = 0.08``, and a
    policy that has learned to move smoothly pays almost nothing -- which is the
    honest way round: the term prices *unnecessary* motion, not exploration.
    """
    action_manager = env.action_manager
    delta = (action_manager.action - action_manager.prev_action) / action_scale
    return torch.sum(torch.square(delta), dim=1)


def leg_joint_acc_l2(env, acc_scale: float = 2000.0,
                     asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """sum(((qdd / acc_scale)^2) over the 12 leg joints) (positive).

    The anti-chatter channel. Leg velocity alone cannot price it: 12 joints
    swinging at 3 rad/s cost 0.005 under any sane velocity weight, which is 1% of
    the +0.5 height term. Joint *acceleration* separates the two regimes.

    The catch, and why ``acc_scale`` is 2000 rather than 200: the exploration
    noise lands in this term too. Actions are resampled every control step from
    N(0, sigma), so with ``init_noise_std = 0.5`` the joints see accelerations
    around 1700 rad/s^2 before the policy has learned anything. At ``acc_scale =
    200`` that is ``12 * 72 = 870`` per step, i.e. two orders of magnitude more
    than the whole positive side of the reward, so the noise alone would decide
    whether the policy learns anything. At 2000 the same noise costs
    ``12 * 0.72 * 0.02 = 0.17``.

    Consequence worth accepting: a quadratic term cannot tell noise from chatter
    (both are just qdd), so a weight that tolerates the noise is also mild on the
    chatter -- 400 rad/s^2 costs 0.01/step. Vibration is therefore priced mostly
    by the return it loses, not by this term. Raise the weight once the policy
    moves smoothly enough that the noise floor is lower.

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
# The one gated channel: wheels spinning under a parked body. That is the only
# state the base-frame kernels cannot observe -- the body is still, so every
# tracking channel is satisfied. The gate is a *strictly* zero command read from
# the command buffer (see ``standing_mask``) rather than from ``is_standing_env``,
# because a flag that nobody sets switches a channel off silently. No deadband: a
# zero command means zero motion.
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

    Used by the one gated term, ``wheel_speed_l2``. The tracking kernels are
    command-relative and need no gate: subtracting the command already makes them
    free while driving.
    """
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd = observations.velocity_command(env, command_name)
    return (cmd.abs() <= eps).all(dim=1).to(dtype=torch.float32)


def wheel_speed_l2(env, scale: float = 2.0, command_name: str = "base_velocity",
                   asset_cfg: SceneEntityCfg = SceneEntityCfg(
                       "robot", joint_names=WHEEL_JOINT_NAMES)) -> torch.Tensor:
    """1[cmd == 0] * mean_4 ((qdot_wheel / scale)^2) (positive).

    The one thing the body-frame terms cannot see: with the legs holding the base
    up, a wheel can be commanded to any speed without moving the robot, so a
    policy can idle with four spinning wheels and still track a zero command. It
    also fixes the reverse case -- wheels that are *not* spinning while the base
    creeps, which is the robot skating.

    Gated, unlike the body terms, because when a command is present the wheels are
    supposed to spin: ungated, an in-place turn would pay for the differential
    rolling that produces it. ``scale = 2 rad/s`` is 0.18 m/s of rolling, a fifth
    of the 0.72 m/s the wheel action can reach.

    Mean over the four wheels, not sum: a single noisy wheel should not outweigh
    the whole posture channel.
    """
    mask = standing_mask(env, command_name)
    wheel_vel = _robot(env, asset_cfg).data.joint_vel
    return mask * torch.mean(torch.square(wheel_vel / scale), dim=1)


def joint_deviation_l2(env, scale: float = 1.0,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg(
                           "robot", joint_names=LEG_JOINT_NAMES)) -> torch.Tensor:
    """mean_legs ((q - q_default) / scale)^2 (positive; weight is negative).

    The posture channel: it keeps the legs near the stance they were designed
    for instead of whatever folded shape happens to balance. Legs only -- a wheel
    angle is unbounded (see ``joint_deviation_l1``).

    **Mean, not sum, and ``scale = 1.0``, not 0.3.** The first version summed
    twelve quadratic terms over ``0.3 rad`` and cost ``-0.367`` per step while
    the whole positive side of the reward is capped at ``0.06``
    (``(2.0 + 1.0) * dt``): one folded knee was worth more than perfect tracking,
    so the policy's gradient was almost entirely "unfold the legs" and the
    tracking channels never got to speak. With a mean over the twelve joints and
    a 1 rad scale the same numbers read: nominal ``kp = 80`` sag (0.15 rad, from
    84.6 N per leg and a 0.143 m knee-to-wheel arm) costs ``-0.0005``, a seriously
    folded leg (1.65 rad) costs ``-0.054`` -- the same
    order as the reward the policy can earn, which is what a shaping term has to
    be for PPO to trade it off against tracking.

    Caveat: it pulls towards ``default_joint_pos``. If that nominal stance is
    itself pitched (unequal hip offsets front to back), this term *enforces* the
    pitch instead of correcting it -- the attitude term is what levels the body.
    Keep the weight modest and fix the default pose rather than out-weighting it.
    """
    robot = _robot(env, asset_cfg)
    deviation = (robot.data.joint_pos[:, asset_cfg.joint_ids]
                 - robot.data.default_joint_pos[:, asset_cfg.joint_ids])
    return torch.mean(torch.square(deviation / scale), dim=1)


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

    Used by the **standing** task, which runs on the stock ``ManagerBasedRLEnv`` and
    has no other way to attach a terminal penalty. The velocity task does *not* use
    this: ``isaaclab_m20.envs.M20VelocityEnv`` applies ``cfg.fall_penalty`` on the
    steps where ``terminated & ~truncated``, straight from the termination flags,
    so that task cannot drift out of sync with ``TerminationsCfg``.

    Deliberately not ``mdp.is_terminated``: that fires on the 20 s time-out too,
    so a policy that stands perfectly for the whole episode would pay the same
    penalty as one that collapses at step 40. With the terminal penalty costing
    the same either way, nothing rewards *staying* upright -- it only rewards
    ending fast. This is the fall-only version.

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
