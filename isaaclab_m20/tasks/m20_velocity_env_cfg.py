"""Velocity-command version of the M20 task: track (vx, vy, wz).

Registered separately as ``Isaac-M20-Wheeled-Velocity-v0``. The static-standing
task (``Isaac-M20-Wheeled-v0`` in ``m20_env_cfg.py``) is untouched -- the two
differ in the observation, the reward set and the presence of a command term,
so they cannot share a checkpoint and are kept as independent configs.

Difference from the standing task:

* observation 57 -> 60 dims: the 3-dim (vx, vy, wz) command is prepended
* the tracking kernels measure the error against the command instead of zero
* the wheel-standstill reward is gone -- a moving robot's wheels must turn
* the leg-posture penalty only applies while the command is ~zero (legged_gym's
  ``stand_still``), otherwise it fights the leg motion steering needs
* 20 s episodes so one episode contains five 4 s command segments
"""
from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from isaaclab.utils import configclass

from isaaclab_m20.tasks import rewards
from isaaclab_m20.tasks.m20_env_cfg import M20WheeledEnvCfg


@configclass
class VelocityCommandsCfg:
    """Uniform sampling of a body-frame velocity command."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # One command per 4 s. Episode is 20 s, so five segments per episode.
        resampling_time_range=(4.0, 4.0),
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class VelocityObservationsCfg:
    """60 dims: 3 command + 3 lin vel + 3 ang vel + 3 gravity + 16 q + 16 qd + 16 a."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class VelocityRewardsCfg:
    """Same legged_gym set as the standing task, with command tracking."""

    # -- task: exp tracking kernels against the commanded velocity ------------
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy_cmd, weight=1.5, params={"sigma": 0.25})
    track_ang_vel_z = RewardTermCfg(
        func=rewards.track_ang_vel_z_cmd, weight=0.75, params={"sigma": 0.25})

    # -- base stability -------------------------------------------------------
    lin_vel_z = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewardTermCfg(func=rewards.flat_orientation_l2, weight=-2.5)
    base_height = RewardTermCfg(
        func=rewards.base_height_l2, weight=-20.0, params={"target_height": 0.50})

    # -- posture and smoothness ----------------------------------------------
    joint_deviation = RewardTermCfg(
        func=rewards.joint_deviation_l1_when_still, weight=-0.5,
        params={"threshold": 0.1})
    joint_vel = RewardTermCfg(func=rewards.joint_vel_l2, weight=-1e-4)
    joint_acc = RewardTermCfg(func=rewards.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewardTermCfg(func=rewards.joint_torques_l2, weight=-1e-5)
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01)

    # -- survival -------------------------------------------------------------
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.15)


@configclass
class M20WheeledVelocityEnvCfg(M20WheeledEnvCfg):
    """Inherits scene, actions, terminations, randomisation and sim settings."""

    observations: VelocityObservationsCfg = VelocityObservationsCfg()
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()
    commands: VelocityCommandsCfg = VelocityCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Five command segments per episode instead of one.
        self.episode_length_s = 20.0
