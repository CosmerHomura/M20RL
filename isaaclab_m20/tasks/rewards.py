"""Reward terms for the M20 static-standing task.

Each function returns one reward term per environment; weights live in
``m20_env_cfg.py``. The task is pure stand-still: stay at the nominal height,
keep the base level, and do not drift. There is no velocity command and no PD
layer -- the policy commands joint torques directly and must learn gravity
compensation end-to-end.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def flat_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    return -(torch.square(gravity[:, 0]) + torch.square(gravity[:, 1]))


def base_height(env, target_height: float = 0.50, sigma: float = 20.0,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    height = _robot(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.exp(-sigma * torch.square(height - target_height))


def base_motion(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base linear and angular velocity so the robot stays put."""
    robot = _robot(env, asset_cfg)
    lin = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)
    ang = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)
    return -(lin + ang)


def joint_torques(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    torque = _robot(env, asset_cfg).data.applied_torque
    return -torch.sum(torch.square(torque), dim=1)


def action_rate(env) -> torch.Tensor:
    action_manager = env.action_manager
    return -torch.sum(
        torch.square(action_manager.action - action_manager.prev_action), dim=1)
