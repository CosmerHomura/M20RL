"""Reward terms for the two M20 tasks.

``Isaac-M20-Wheeled-v0`` (static standing) keeps the legged_gym / AnymalC term
set it has always used. ``Isaac-M20-Wheeled-Velocity-v0`` uses eighteen per-step
terms, plus a fall penalty that is applied by the environment rather than by a
reward term:

    r = + 2.0    · r_vel                speed tracking, exp(-e²/0.5²)      [body target]
      + 1.0    · r_yaw                yaw-rate tracking, exp(-e²/0.5²)   [ω*]
      - 1.0    · p_rot_xy             1[mode=ROTATE]·sum((v_xy/0.15)²)
      - 10.0   · p_h                  (h - 0.50)²                        ride height
      - 0.5    · p_ori                (gx² + gy²) / 0.2²                  roll/pitch
      - 1.0    · p_vz                 (v_z / 1.0)²                   vertical bounce
      - 0.05   · p_ang                sum(w_xy² / 1²)                     wobble
      - 0.6    · p_axle               wheel axle alignment, 100% STAND / 25% MOVE
      - 1.0    · p_contact            number of leg links touching (>5 N)
      - 0.08   · p_hip                sum_4 |q - q_d| (hipx), x2 when standing
      - 0.01   · p_pose               sum_8 |q - q_d| (hipy, knee), x2 standing
      - 0.4    · p_qacc               sum((qdd / 2000)²)                 legs only
      - 1e-4   · p_torque             sum(tau²)
      - 2e-5   · p_power              sum(|tau·qdot|)
      - 0.01   · p_action             sum((Δa / 1.0)²)
      - 0.01   · p_smooth             sum((a - 2a_-1 + a_-2)²)
      - 2.0    · p_limit              soft joint-limit violation, 12 legs
      - 0.02   · p_stand_qd           gated leg/wheel speed, |v*|<0.1, |wz*|<0.1

    terminal: -10 on the fall step only, applied by ``M20VelocityEnv`` (see
    ``fall_penalty`` on the env cfg). It is not one of the terms below because the
    termination flags are only known after the reward manager has run.

**Eighteen, and sixteen of them are legbot_lab's.** The alignment is deliberate:
that project's channels are the ones a team with a working wheeled-legged robot
converged on, and inventing a different set only makes failures unattributable.
Weights are theirs, re-expressed in our normalisation (their ``1e-7 * sum(qdd^2)``
is our ``-0.4 * sum((qdd/2000)²)``; their ``1e-4 * sum(tau²)`` is our
``-1e-4 * sum((tau/1)²)``). Two channels are ours and have no counterpart:
``flat_orientation`` (their tilt handling is implicit in the height/contact terms)
and ``zero_xy_speed`` (they command yaw but not "spin in place without moving").

Earlier revisions of this file went through ten and eleven channels; the lessons
that survived are worth keeping in one place:

* one kernel per axis group (``track_lin_vel_xy``, ``track_ang_vel_z``), both
  against the *resolved* target from ``tasks/control.py`` -- the body-frame
  velocity and the outer-loop yaw-rate target. The kernel is the standard
  ``exp(-e²/σ²)`` (σ = 0.5, i.e. legged_gym's ``tracking_sigma = 0.25`` in that
  notation). Its band is the point: a policy that is 2.5 m/s away gets nothing,
  so the curriculum has to ask for speeds the policy can reach;
* ride height is a plain quadratic error, not an exp bonus: on flat ground the
  target is a set point, so what the channel needs is stiffness around it, and a
  quadratic has no ceiling to saturate against;
* posture is L1 and split by joint group (``hip_pos`` for the four hipx,
  ``joint_pos`` for hipy + knee) and doubled when the command is "stand". The L2
  sum over twelve joints at 0.3 rad that came before charged -0.367/step for a
  pose that is merely sagging under its own weight -- more than the entire
  positive side of the reward;
* the mode gates: ``wheel_axis_alignment`` (100 % STAND / 25 % MOVE) and
  ``zero_xy_speed`` (ROTATE). Both read the label the sampler wrote, never a
  float re-test here. ``standing_joint_vel`` uses its own looser gate
  (|v*|<0.1, |wz*|<0.1) because it prices creep, not the label of the moment.

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
control step from N(0, sigma), so at ``init_noise_std = 0.3`` the step-to-step
difference has std 0.42 and ``sum((da/0.05)²)`` is ~1150 -- -11 per step at weight
-0.01, dwarfing the +3 the tracking terms can pay. At ``action_scale = 1.0`` /
``acc_scale = 2000`` the same noise costs 0.03 / 0.06. The honest cost: a quadratic
cannot tell noise from chatter, so these two terms are mild on real vibration
until the noise floor drops -- vibration is mostly priced by the return it loses.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``ang_vel_xy_l2`` returns ``w_xy²`` and the config gives it weight -0.2).
Do not add a leading minus here -- it would flip the term into a bonus.

Gate convention: three terms carry a gate and each one states which, in its own
docstring -- ``zero_xy_speed_l2`` (ROTATE), ``wheel_axis_alignment_l2``
(STAND/MOVE weighting), ``standing_joint_vel_l2`` (|v*|<0.1 and |wz*|<0.1). They
read the **mode label the sampler wrote** or the command buffer, never
``is_standing_env``: a flag that nobody sets switches a whole channel off
silently, a buffer and a label cannot. There is no deadband anywhere. Standing is
held by the tracking kernels themselves (a zero command asks for zero velocity) --
a separate gated ``||v_xy||²`` term would be a second spring on the same error.

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

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_m20.tasks import control, observations
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
# reason the reward table kept growing. What is left is the standard exponential
# kernel exp(-e^2 / sigma^2), the one used by legged_gym's tracking_lin_vel and
# by legbot_lab (where it is written exp(-e^2 / 0.25), i.e. sigma = 0.5 here).
# It was a Cauchy kernel for a while; two arguments turned it back. It has no
# literature behind it, and its heavy tail means a policy that is nowhere near
# the command still collects something -- which hides exactly the failure this
# project kept hitting ("the reward was 1.0 and the robot never moved"). The exp
# form says plainly: outside the sigma band you get nothing, so the commands you
# train on have to be commands the policy can reach. That is the curriculum's
# job, not the kernel's.
#
# Zero command needs no second channel for "do not move": the kernels are already
# relative to the command, so a zero command asks for zero velocity. What they
# cannot see is a wheel spinning under a parked body (and a wheel rolled onto its
# edge) -- that is what `standing_joint_vel_l2` and `wheel_axis_alignment_l2` are
# for, and why both are gated on the standing label / zero command.
# --------------------------------------------------------------------------- #
def track_lin_vel_xy(env, sigma: float = 0.5, command_name: str = "base_velocity",
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-|v_body - v_body*|^2 / sigma^2), in (0, 1].

    ``sigma = 0.5 m/s`` against a 2.5 m/s envelope is legged_gym's
    ``tracking_sigma = 0.25`` written as ``exp(-e^2 / sigma^2)`` (0.25 is
    ``sigma^2`` there), and the same number legbot_lab uses. Reading it as a band:
    within 0.25 m/s the kernel is above 0.78, at 0.5 m/s it is 0.37, at 1.0 m/s
    0.018 -- the command has to be reachable, which is why the course ramps the
    range instead of sampling the envelope from the start.

    The target is ``control.command_targets(env).v_body``: the command-frame
    velocity rotated into the body frame with ``R(-psi_rel)``. Comparing it to
    ``root_lin_vel_b`` is therefore the same error the operator asked for, in the
    frame both quantities are measured in. The raw command is *not* used here --
    using it directly would demand a body velocity the operator never asked for
    whenever the robot is not aligned with its command frame.
    """
    robot = _robot(env, asset_cfg)
    velocity = robot.data.root_lin_vel_b[:, :2]
    target = control.command_targets(env).v_body
    error = torch.sum(torch.square(velocity - target), dim=1)
    return torch.exp(-error / (sigma * sigma))


def track_ang_vel_z(env, sigma: float = 0.5, command_name: str = "base_velocity",
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(w_z - omega*)^2 / sigma^2), in (0, 1]: yaw-rate tracking.

    ``sigma = 0.5 rad/s`` -- the same band legbot_lab uses (``sigma^2 = 0.25``)
    against a 1.0 rad/s command, so 0.2 rad/s error scores 0.85 and 0.5 rad/s 0.37.

    ``omega*`` is the outer loop's target (feed-forward plus heading PD), not
    ``wz_cmd``: in MOVE/COMBINED the robot is expected to steer while it drives,
    and in ROTATE the two are equal by construction.
    """
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    target = control.command_targets(env).omega
    error = torch.square(ang_vel_z - target)
    return torch.exp(-error / (sigma * sigma))


def zero_xy_speed_l2(env, scale: float = 0.15,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1[ROTATE] * |v_xy|^2 / scale^2 (positive; weight is negative).

    The third gate. While the command is a pure spin the operator asked for *no
    translation*, and the tracking kernel cannot say that: with a zero linear
    target it would already ask for zero velocity, but a spin is the one case
    where "stay in place" is the whole task, so it gets its own stiffness
    (``scale = 0.15 m/s`` -- 50 mm/s costs ~0.1/step at weight -1.0) rather than
    relying on the gentler tracking kernel.

    Gated on the mode label, so a moving command never pays it.
    """
    mask = control.mode_mask(env, control.MODE_ROTATE)
    velocity = _robot(env, asset_cfg).data.root_lin_vel_b[:, :2]
    return mask * torch.sum(torch.square(velocity / scale), dim=1)


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
    0.3`` the step-to-step difference has std 0.42 and ``sum((da / 0.05)^2)`` is
    about 1150 -- i.e. -11 reward per step at weight -0.01, dwarfing the +3 the
    tracking terms can pay, so the noise (not the policy) would set the cost.

    At ``action_scale = 1.0`` the same noise costs ``2.9 * 0.01 = 0.03``, and a
    policy that has learned to move smoothly pays almost nothing -- which is the
    honest way round: the term prices *unnecessary* motion, not exploration.
    """
    action_manager = env.action_manager
    delta = (action_manager.action - action_manager.prev_action) / action_scale
    return torch.sum(torch.square(delta), dim=1)


def joint_power(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum_j |tau_j * qdot_j| over the asset's joints (positive; weight is negative).

    Mechanical power, legbot_lab's ``joint_power``. Unlike ``joint_torques_l2`` it
    only charges torque that is *doing work*: a leg holding a static load has high
    torque and zero velocity, and should not be taxed for carrying the robot. That
    is the term that separates "pushing hard" from "spinning fast".

    Raw watts, so the weight has to be small (1e-5 scale). A wheel at 20 N·m and
    20 rad/s is 400 W on its own; a leg in stance is tens of watts.
    """
    robot = _robot(env, asset_cfg)
    ids = asset_cfg.joint_ids
    torque = robot.data.applied_torque[:, ids]
    velocity = robot.data.joint_vel[:, ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


# Second-order action difference needs a_{t-2}, which the action manager does not
# keep. Cached here per env, one tensor per environment instance.
_PREV_PREV_ACTION: dict[int, torch.Tensor] = {}


def action_smoothness_l2(env, action_scale: float = 1.0) -> torch.Tensor:
    """1[t > 2] * sum(((a_t - 2 a_{t-1} + a_{t-2}) / action_scale)^2) (positive).

    legbot_lab's ``action_smoothness_l2``: the *second* difference, which prices
    acceleration of the action signal rather than its size. ``action_rate_l2``
    already prices speed of change; together they are a cheap damping filter, and
    the second difference is what catches the 2-step oscillation an RL policy
    falls into when a single-step term is all it sees.

    Gated on ``episode_length_buf > 2`` like theirs: on the first two steps there
    is no ``a_{t-2}``, and the reset zeroes the history, so the difference is an
    artefact rather than a policy choice. Cache is keyed by env id and refreshed
    every call, so it costs one clone per step.
    """
    action_manager = env.action_manager
    action = action_manager.action
    prev = action_manager.prev_action
    key = id(env)
    prev_prev = _PREV_PREV_ACTION.get(key)
    if prev_prev is None or prev_prev.shape != action.shape:
        prev_prev = prev.clone()
    second = (action - 2.0 * prev + prev_prev) / action_scale
    _PREV_PREV_ACTION[key] = prev.clone()
    fresh = (getattr(env, "episode_length_buf", torch.zeros(action.shape[0], device=action.device)) > 2)
    return torch.sum(torch.square(second), dim=1) * fresh.to(dtype=action.dtype)


_WHEEL_BODY_IDS: dict[int, torch.Tensor] = {}


def wheel_body_ids(robot) -> torch.Tensor:
    """Body indices whose name contains "wheel", cached per articulation.

    Resolved by substring rather than by a ``body_names`` regex on the asset cfg:
    the link names come from the URDF and differ between the imported and the
    official asset, while "the bodies called ...wheel" is stable.
    """
    key = id(robot)
    ids = _WHEEL_BODY_IDS.get(key)
    if ids is None:
        ids = torch.tensor([i for i, name in enumerate(robot.body_names) if "wheel" in name.lower()],
                           dtype=torch.long, device=robot.device)
        _WHEEL_BODY_IDS[key] = ids
    return ids


def wheel_axis_alignment_l2(env, tolerance_deg: float = 2.0, scale: float = 0.2,
                            stand_weight: float = 1.0, move_weight: float = 0.25,) -> torch.Tensor:
    """mean_wheels ((relu(|axle.fwd| - sin tol)^2 + relu(|axle.up| - sin tol)^2) / scale^2),
    weighted by mode (standing 100 %, driving 25 %, rotating/strafing 0 %).

    The wheeled-legged channel that a legged robot has no use for, and the one
    that actually decides whether the robot *looks* right standing still: each
    wheel's axle has to be horizontal and perpendicular to the direction of
    travel. A leg that has drifted puts its wheel at an angle, the wheel then
    rolls on an edge, and nothing else in the reward notices -- the body can be
    level, the height correct and the wheels still.

    The axle is the wheel body's local +y (the URDF's rotation axis). Both
    components are compared in the *robot's* frame, so the test is "is the axle
    square to the body", not "is it square to the world".

    Weighting follows legbot_lab: posture of the wheels matters when parked
    (100 %), is still worth something when driving straight (25 %), and must be
    off when the command asks for a turn or a strafe, because those *require* a
    misaligned axle (differential rolling, or the wheels skating sideways).

    A 2 deg deadband is subtracted before squaring, so a small, harmless
    misalignment is free and only a real one is priced; ``scale = 0.2`` makes
    ~11 deg of misalignment one unit.
    """
    robot = env.scene["robot"]
    ids = wheel_body_ids(robot)
    quat = robot.data.body_quat_w[:, ids]                       # (n, k, 4)
    count = quat.shape[1]
    axle_local = torch.zeros(quat.shape[0], count, 3, device=quat.device, dtype=quat.dtype)
    axle_local[..., 1] = 1.0
    axle = quat_apply(quat.reshape(-1, 4), axle_local.reshape(-1, 3)).reshape(-1, count, 3)
    forward = quat_apply(robot.data.root_quat_w, torch.tensor(
        [1.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3))
    up = torch.tensor([0.0, 0.0, 1.0], device=quat.device, dtype=quat.dtype).expand_as(forward)
    tolerance = math.sin(math.radians(tolerance_deg))
    forward_error = torch.clamp(
        (axle * forward.unsqueeze(1)).sum(dim=-1).abs() - tolerance, min=0.0)
    vertical_error = torch.clamp((axle * up.unsqueeze(1)).sum(dim=-1).abs() - tolerance, min=0.0)
    per_wheel = (torch.square(forward_error) + torch.square(vertical_error)) / (scale * scale)
    weight = (stand_weight * control.mode_mask(env, control.MODE_STAND)
              + move_weight * control.mode_mask(env, control.MODE_MOVE))
    return weight * torch.mean(per_wheel, dim=1)


_LEG_CONTACT_IDS: dict[int, torch.Tensor] = {}


def undesired_contacts_legs(env, threshold: float = 5.0,
                            sensor_name: str = "contact_forces") -> torch.Tensor:
    """Number of leg-link bodies whose contact force exceeded ``threshold`` (positive).

    Thighs and shins touching the ground is a fault on flat ground, and the height
    and tilt terminations do not catch it: the robot can rest on a knee with the
    base still 0.4 m up and level. legbot_lab prices it (5 N) and lets the body
    itself terminate the episode; here it is a priced fault, since our height/tilt
    terminations already end the episode when the body comes down.

    One count per *body*, using the maximum force over the sensor's history
    window, so a single strong poke costs one unit and a persistent drag costs one
    unit per step it lasts.

    Body selection is by substring ("hip" or "knee", not "base"), resolved once
    per sensor: the link names come from the URDF, so a hard-coded regex would be
    one asset swap away from silently matching nothing.
    """
    sensor = env.scene[sensor_name]
    key = id(sensor)
    ids = _LEG_CONTACT_IDS.get(key)
    if ids is None:
        kept = [i for i, name in enumerate(sensor.body_names)
                if ("hip" in name.lower() or "knee" in name.lower()) and "base" not in name.lower()]
        ids = torch.tensor(kept, dtype=torch.long, device=env.device)
        _LEG_CONTACT_IDS[key] = ids
    forces = sensor.data.net_forces_w_history[:, :, ids, :]        # (n, history, bodies, 3)
    magnitude = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1)  # (n, bodies)
    return (magnitude > threshold).sum(dim=1).to(dtype=torch.float32)


def leg_joint_acc_l2(env, acc_scale: float = 2000.0,
                     asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """sum(((qdd / acc_scale)^2) over the 12 leg joints) (positive).

    The anti-chatter channel. Leg velocity alone cannot price it: 12 joints
    swinging at 3 rad/s cost 0.005 under any sane velocity weight, which is 1% of
    the +0.5 height term. Joint *acceleration* separates the two regimes.

    The catch, and why ``acc_scale`` is 2000 rather than 200: the exploration
    noise lands in this term too. Actions are resampled every control step from
    N(0, sigma), so with ``init_noise_std = 0.3`` the joints see accelerations
    around 1000 rad/s^2 before the policy has learned anything. At ``acc_scale =
    200`` that is ``12 * 25 = 300`` per step, i.e. orders of magnitude more than
    the whole positive side of the reward, so the noise alone would decide
    whether the policy learns anything. At 2000 the same noise costs
    ``12 * 0.25 * 0.02 = 0.06``.

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
# Standing gates. ``standing_mask`` is the strict-zero test used by the parking
# brake and by nothing else in the reward set; ``standing_joint_vel_l2`` uses the
# looser 0.1 m/s / 0.1 rad/s band that legbot_lab uses, because it prices the
# twitch that makes a stopped robot creep rather than the exact label.
# Both read the command buffer, never ``is_standing_env``: a flag that nobody sets
# switches a channel off silently, a buffer cannot. No deadband on motion itself:
# a zero command means zero motion.
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

    Kept as a tool for the parking brake and for diagnostics; no reward term in
    this revision uses it directly. ``standing_joint_vel_l2`` deliberately uses a
    looser band (0.1 m/s / 0.1 rad/s): measuring creep is not the same as
    recognising an exact zero command. The tracking kernels are command-relative
    and need no gate at all: subtracting the command already makes them free while
    driving.
    """
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd = observations.velocity_command(env, command_name)
    return (cmd.abs() <= eps).all(dim=1).to(dtype=torch.float32)


def standing_joint_vel_l2(env, leg_scale: float = 1.0, wheel_scale: float = 5.0,
                          stand_scale: float = 1.0, speed_threshold: float = 0.1,
                          yaw_threshold: float = 0.1, command_name: str = "base_velocity",
                          ) -> torch.Tensor:
    """1[standing still] * ( mean_12((qdot_leg/1)²) + mean_4((qdot_wheel/5)²) ) (positive).

    legbot_lab's ``standing_joint_vel_l2``: the leg velocities normalised by 1
    rad/s, the wheel velocities by 5 rad/s, averaged within each group and then
    added. The gate is the loose "standing still" test (|v_xy*| < 0.1 m/s and
    |wz*| < 0.1 rad/s), not the parking brake's exact zero, because this term is
    about *holding a posture* rather than about preventing creep.

    Two reasons the wheels get their own scale: with the legs propping the base up
    a wheel can be commanded to any speed without moving the robot (idle spin, and
    the base-frame kernels cannot see it), and 5 rad/s is 0.45 m/s of rolling at
    the 0.09 m radius -- loose enough that it prices idling, not driving, and the
    term is gated off entirely when a command is present.

    Averaged within each group so a single noisy wheel cannot outweigh the whole
    posture channel. The 12 leg joints were previously not priced at all while
    standing; that is what let the legs churn under a parked body.
    """
    robot = _robot(env, SceneEntityCfg("robot"))
    leg_ids = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)[0]
    wheel_ids = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)[0]
    leg = torch.mean(torch.square(robot.data.joint_vel[:, leg_ids] / leg_scale), dim=1)
    wheel = torch.mean(torch.square(robot.data.joint_vel[:, wheel_ids] / wheel_scale), dim=1)
    gate = _standing_multiplier(env, stand_scale, speed_threshold, yaw_threshold,
                                command_name) / max(stand_scale, 1e-6)
    return gate * (leg + wheel)


def joint_pos_deviation_l1(env, joint_names: list[str] | None = None, scale: float = 1.0,
                           stand_scale: float = 2.0, speed_threshold: float = 0.1,
                           yaw_threshold: float = 0.1,
                           command_name: str = "base_velocity") -> torch.Tensor:
    """sum_j |q_j - q_default_j| / scale  over the named joints, x2 while standing still.

    The posture channel, in the shape legbot_lab uses: **L1**, split into a hip
    group and a thigh/shin group with their own weights (-0.08 and -0.01 there,
    i.e. the hip is 8x the price of the other two joints per radian), and doubled
    when the robot has been asked to stand still.

    Why L1 and why split: L1 does not concentrate the gradient on whichever joint
    is worst. L2 does -- the first version of this channel summed twelve squares
    over 0.3 rad and cost -0.367/step against a +0.06 positive side, so a single
    folded knee outbid perfect tracking and the policy's whole gradient became
    "unfold the legs". L1 keeps every joint's pull proportional to its own error,
    and splitting the groups lets the hips (which decide stance width and pitch)
    be priced separately from the knees (which carry the load).

    The standing multiplier is the same idea as legbot_lab's ``stand_still_scale
    = 2``: holding a *pose* matters most exactly when the command is zero, and the
    gate is read from the command buffer (|v_xy*| < 0.1 m/s and |wz*| < 0.1 rad/s)
    rather than from a mode label, so it degrades to "all zeros = standing" even
    if no label was ever written.

    Legs only: a wheel angle is unbounded and means nothing absolute.
    """
    robot = _robot(env, SceneEntityCfg("robot", joint_names=joint_names))
    ids = (torch.arange(robot.num_joints, device=robot.device)
           if joint_names is None else robot.find_joints(joint_names, preserve_order=True)[0])
    deviation = (robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]) / scale
    total = torch.sum(torch.abs(deviation), dim=1)
    return total * _standing_multiplier(env, stand_scale, speed_threshold, yaw_threshold,
                                        command_name)


def _standing_multiplier(env, stand_scale: float, speed_threshold: float,
                         yaw_threshold: float, command_name: str) -> torch.Tensor:
    """1.0 normally, ``stand_scale`` where the command is (nearly) zero.

    "Nearly" is the point: legbot_lab's gate is |v_xy*| < 0.1 m/s and |wz*| < 0.1
    rad/s, because a real operator's "stand still" arrives as a small number, not
    as an exact zero. That is a different gate from the parking brake's, which has
    to be exact -- a 0.05 m/s command is still a command to *drive*, so the brake
    must stay off, but it is also slow enough that holding a clean pose should
    pay. Two gates, two purposes; both read the same buffer.
    """
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.full((env.num_envs,), float(stand_scale), device=env.device)
    cmd = observations.velocity_command(env, command_name)
    still = ((torch.linalg.vector_norm(cmd[:, :2], dim=1) < speed_threshold)
             & (cmd[:, 2].abs() < yaw_threshold))
    return torch.where(still, torch.full_like(cmd[:, 0], float(stand_scale)),
                       torch.ones_like(cmd[:, 0]))


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
