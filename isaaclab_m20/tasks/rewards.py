"""Reward terms for the M20 static-standing task.

The term set and default weights follow the standard legged-locomotion reward
formulation used by legged_gym (Rudin et al., "Learning to Walk in Minutes")
and IsaacLab's own AnymalC task, adapted to a zero velocity command:

    r = 1.5·exp(-|v_xy|^2/0.25) + 0.75·exp(-w_z^2/0.25)
        - 2.0·v_z^2 - 0.05·|w_xy|^2 - 2.5·(gx^2+gy^2)
        - 20·(h - 0.5)^2 - 0.5·sum|q - q_default|
        - 1e-4·sum(qd^2) - 2.5e-7·sum(qdd^2) - 1e-5·sum(tau^2)
        - 0.01·sum(a - a_prev)^2
        - 200·1[terminated]  (+ 0.15 alive)

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


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L1 distance from the default stand pose (positive; weight is negative)."""
    robot = _robot(env, asset_cfg)
    return torch.sum(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)


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
