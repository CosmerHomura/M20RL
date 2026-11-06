"""Reward terms for the M20 static-standing task.

The term set and default weights follow the standard legged-locomotion reward
formulation used by legged_gym (Rudin et al., "Learning to Walk in Minutes")
and IsaacLab's own AnymalC task, adapted to a zero velocity command:

    r = 1.5·exp(-|v_xy|²/0.25) + 0.75·exp(-ω_z²/0.25)
        - 2.0·v_z² - 0.05·|ω_xy|² - 2.5·(gx²+gy²)
        - 20·(h - 0.5)² - 0.5·Σ|q - q_default|
        - 1e-4·Σq̇² - 2.5e-7·Σq̈² - 1e-5·Στ² - 0.01·Σ(a - a_prev)²
        - 200·1[terminated]  (+ 0.15 alive)

The exp tracking terms with a zero command are what make "stand still" the
optimum; the quadratic height and termination terms are what stop the policy
from taking the cheap way out and just falling over.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def track_lin_vel_xy_zero(env, sigma: float = 0.25,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-|v_xy|²/σ): the legged_gym tracking kernel evaluated at a zero command."""
    lin_vel = _robot(env, asset_cfg).data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel), dim=1) / sigma)


def track_ang_vel_z_zero(env, sigma: float = 0.25,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-ω_z²/σ): yaw-rate tracking at a zero command."""
    ang_vel_z = _robot(env, asset_cfg).data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(ang_vel_z) / sigma)


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical bounce penalty."""
    return -torch.square(_robot(env, asset_cfg).data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll/pitch rate penalty."""
    ang_vel = _robot(env, asset_cfg).data.root_ang_vel_b[:, :2]
    return -torch.sum(torch.square(ang_vel), dim=1)


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    return -(torch.square(gravity[:, 0]) + torch.square(gravity[:, 1]))


def base_height_l2(env, target_height: float = 0.50,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic height error; the standard -20·(h-h*)² shape, not an exp kernel."""
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return -torch.square(height - target_height)


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L1 distance from the default stand pose (legged_gym's `stand_still`)."""
    robot = _robot(env, asset_cfg)
    return -torch.sum(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)


def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return -torch.sum(torch.square(_robot(env, asset_cfg).data.joint_vel), dim=1)


def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return -torch.sum(torch.square(_robot(env, asset_cfg).data.joint_acc), dim=1)


def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return -torch.sum(torch.square(_robot(env, asset_cfg).data.applied_torque), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    action_manager = env.action_manager
    return -torch.sum(
        torch.square(action_manager.action - action_manager.prev_action), dim=1)
