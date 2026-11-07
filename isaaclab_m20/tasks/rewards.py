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

    track  = 1.5·exp(-|v_xy − cmd_xy|^2/0.25) + 0.75·exp(-(w_z − cmd_wz)^2/0.25)
    posture = -0.5·sum_legs|q − q_default| · 1[|cmd| < 0.1]   (stand_still)

and no wheel-standstill term -- a moving robot's wheels are supposed to turn.

Sign convention: penalty terms return a POSITIVE magnitude and carry their sign
in the reward weight, exactly like ``isaaclab.envs.mdp`` (e.g.
``lin_vel_z_l2`` returns ``v_z^2`` and the config gives it weight -2.0).
Do not add a leading minus here -- it would flip the term into a bonus.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


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
def track_lin_vel_xy_cmd(env, sigma: float = 0.25, command_name: str = "base_velocity",
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-|v_xy - cmd_xy|^2 / sigma), in [0, 1]: the legged_gym tracking kernel."""
    lin_vel = _robot(env, asset_cfg).data.root_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel - cmd), dim=1) / sigma)


def track_ang_vel_z_cmd(env, sigma: float = 0.25, command_name: str = "base_velocity",
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-(w_z - cmd_wz)^2 / sigma), in [0, 1]: yaw-rate tracking."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(ang_vel_z - cmd) / sigma)


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

    Wheels are deliberately excluded. A wheel joint's absolute angle is
    unbounded -- it grows without limit as the wheel rolls -- so penalising it
    against a default of 0 adds a term that never saturates and swamps the rest
    of the reward after a couple of turns.
    """
    robot = _robot(env, asset_cfg)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


def joint_deviation_l1_when_still(env, threshold: float = 0.1,
                                  command_name: str = "base_velocity",
                                  asset_cfg: SceneEntityCfg = SceneEntityCfg(
                                      "robot",
                                      joint_names=[".*_hipx_joint", ".*_hipy_joint",
                                                   ".*_knee_joint"])) -> torch.Tensor:
    """Leg deviation from the default stance, but only while the command is ~zero.

    This is legged_gym's ``stand_still``: penalise leaving the nominal posture
    only when the robot was not asked to move. Applying it unconditionally would
    fight the leg motion a lateral or turning command needs.
    """
    deviation = joint_deviation_l1(env, asset_cfg)
    command = env.command_manager.get_command(command_name)[:, :3]
    still = (torch.linalg.norm(command, dim=1) < threshold).float()
    return deviation * still


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
