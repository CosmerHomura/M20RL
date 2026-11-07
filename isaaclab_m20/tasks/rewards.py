"""Reward terms for the two M20 tasks.

The term set and default weights follow the standard legged-locomotion reward
formulation used by legged_gym (Rudin et al., "Learning to Walk in Minutes")
and IsaacLab's own AnymalC task.

* ``Isaac-M20-Wheeled-v0`` -- static standing; the velocity command is hard
  wired to zero, so the tracking kernels compare against zero.
* ``Isaac-M20-Wheeled-Velocity-v0`` -- (vx, vy, wz) command tracking, using the
  ``*_cmd`` kernels that compare against ``env.command_manager``.

Standing reward:

    r = 1.5·exp(-|v_xy|^2/0.25) + 0.75·exp(-w_z^2/0.25)
        + 0.5·exp(-|w_wheel|^2/4)                     # wheels stand still
        - 2.0·v_z^2 - 0.05·|w_xy|^2 - 2.5·(gx^2+gy^2)
        - 20·(h - 0.5)^2 - 0.5·sum_legs|q - q_default|
        - 1e-4·sum(qd^2) - 2.5e-7·sum(qdd^2) - 1e-5·sum(tau^2)
        - 0.01·sum(a - a_prev)^2
        - 200·1[terminated]  (+ 0.15 alive)

Velocity-command reward: the same set, with

    track   = 1.5·exp(-(v_x − cmd_x)^2/sigma_x) + 1.0·exp(-(v_y − cmd_y)^2/sigma_y)
              + 0.75·exp(-(w_z − cmd_wz)^2/0.25)
    posture = -0.5·sum_legs|q − q_default|      (always on, no stand_still gate)
    roll    = 0.5·exp(-(v_x + 0.09·mean(qd_wheel))^2/0.25)

and no wheel-standstill term -- a moving robot's wheels are supposed to turn.
sigma_x and sigma_y are per-axis and shrink as the command shrinks; see the
docstrings of the three terms below.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``lin_vel_z_l2`` returns ``v_z^2`` and the config gives it weight -2.0).
Do not add a leading minus here -- it would flip the term into a bonus.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


# Every ``*_wheel`` link in model/urdf/M20.urdf is
# <cylinder length="0.054" radius="0.09"/>. The wheel action scale times this
# radius is the speed ceiling; it is also the rolling-speed gain below.
WHEEL_RADIUS = 0.09


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


# -- standing task (Isaac-M20-Wheeled-v0): command is hard-wired to zero ------
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


# -- velocity-command task (Isaac-M20-Wheeled-Velocity-v0) --------------------
def track_lin_vel_cmd(env, axis: int = 0, sigma_max: float = 0.5,
                      sigma_min: float = 0.0625, fade: float = 1.5,
                      command_name: str = "base_velocity",
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(v_axis - cmd_axis)^2 / sigma), in [0, 1]: one axis of velocity tracking.

    Two things made the old single-kernel version too loose, and both come from
    using one ``sigma`` for two axes whose ranges differ by 5x (+-3.0 m/s vs
    +-0.6 m/s):

    * ``sigma`` is an *error scale*. 0.5 is right for the drive axis, where a
      1 m/s error is a real tracking failure against a 3 m/s command. On the
      lateral axis a 0.6 m/s error is 100% of the range and still scores
      ``exp(-0.72) = 0.49`` -- a policy that never moves sideways keeps half
      the lateral reward, so the channel carries almost no gradient.
    * the kernel was shared, so a forward slip during a lateral command was
      only one component of a 2-norm. Wheels roll freely along x, so there is
      no friction to stop that slip: the only thing that can is this term.

    So the axes are split, and ``sigma`` additionally follows the *magnitude*
    of the command on that axis: ``sigma_min`` when asked to hold zero
    (a 0.25 m/s slip must not be free), widening to ``sigma_max`` at ``fade``
    so that high-speed tracking still has a usable gradient instead of being a
    step function.
    """
    robot = _robot(env, asset_cfg)
    velocity = robot.data.root_lin_vel_b[:, axis]
    command = env.command_manager.get_command(command_name)[:, axis]
    sigma = sigma_min + (sigma_max - sigma_min) * torch.clamp(
        torch.abs(command) / fade, max=1.0)
    return torch.exp(-torch.square(velocity - command) / sigma)


def track_ang_vel_z_cmd(env, sigma: float = 0.25, command_name: str = "base_velocity",
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(w_z - cmd_wz)^2 / sigma), in [0, 1]: yaw-rate tracking."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(ang_vel_z - cmd) / sigma)


def wheel_roll_vel_exp(env, sigma: float = 0.25,
                       wheel_radius: float = WHEEL_RADIUS,
                       wheel_sign: float = -1.0,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg(
                           "robot", joint_names=[".*_wheel_joint"])) -> torch.Tensor:
    """exp(-(v_x - v_roll)^2 / sigma), in [0, 1]: the wheels are actually rolling.

    ``v_roll`` is the speed the wheels imply when they roll without slipping.
    Where they agree with the body's forward speed the wheels are doing the
    work; where they disagree the robot is skating -- wheels spinning on the
    spot (v_roll > v_x) or being dragged along with the wheels locked
    (v_roll < v_x). Same idea as the wheel term in the Booster T1 skating
    work, ``exp(-(v_x - r*sum(w_i c_i)/sum(c_i))^2 / sigma)``, minus the
    contact indicator: M20 has no foot sensor, so it uses the mean over all
    four wheels.

    Sign: every wheel joint in the URDF turns about ``(0, -1, 0)`` in the base
    frame at the nominal stance (hipx = 0; the hipy and knee axes point the
    same way and rotations about y leave y alone). A spin about -y moves the
    contact point at the bottom of the wheel along +x, so rolling without slip
    puts the *body* at ``v_x = -r*dq``. Hence ``wheel_sign = -1``. Flip it to
    +1 if ``Episode_Reward/wheel_roll`` sits near 0 while the robot visibly
    drives straight -- that would mean the joint-velocity sign came through
    the USD importer the other way.

    At zero command this term still pays 1.0 (standing still on locked wheels
    is perfect rolling), so it is neutral between driving and standing, not an
    incentive to do either.
    """
    robot = _robot(env, asset_cfg)
    wheel_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    v_roll = wheel_sign * wheel_radius * torch.mean(wheel_vel, dim=1)
    v_forward = robot.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(v_forward - v_roll) / sigma)


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical velocity squared (positive; weight is negative)."""
    return torch.square(_robot(env, asset_cfg).data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll/pitch rate squared (positive; weight is negative)."""
    ang_vel = _robot(env, asset_cfg).data.root_ang_vel_b[:, :2]
    return torch.sum(torch.square(ang_vel), dim=1)


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

    Applied unconditionally -- there is no ``stand_still`` gate. legged_gym
    gates it because a walking robot must leave the nominal pose to move; M20
    does not. The default stance is where the legs belong while the wheels are
    doing the work, and it is also the target every lateral step should return
    to.

    Because the target pose is symmetric left-to-right, minimizing this term
    also pulls the legs into symmetric configurations: a rear-left hipy ahead
    of the rear-right one costs the sum of both offsets, while both at the
    default costs zero. Note the cost is flat among symmetric offsets (both at
    +0.3 scores the same as one at +0.3 and one at -0.3), so if an asymmetry
    survives, the targeted fix is a pairwise |q_left - q_right| term -- but
    that term also forbids trotting, so it belongs with the terrain work, not
    here.

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


def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env, asset_cfg).data.applied_torque), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    action_manager = env.action_manager
    return torch.sum(
        torch.square(action_manager.action - action_manager.prev_action), dim=1)
