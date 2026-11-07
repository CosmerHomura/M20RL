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
* a wider wheel action scale, so the command range is actually reachable (below)
* the wheel joints are out of the joint-velocity penalty; a wheel spinning at
  the commanded speed is the goal, not a cost

Speed ceiling: the wheel action scale times the 0.09 m wheel radius is the
fastest the policy can ask for. ``scale=45`` -> 45 * 0.09 = 4.05 m/s, and the
3.0 m/s top command uses about 74% of that, leaving headroom for tracking error,
slip on a low-friction surface and the pushes. The standing task keeps
``scale=20`` (1.8 m/s cap) because its wheels are supposed to stay still anyway.
"""
from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
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
            # The full range, reached in the last stage of the staged command
            # schedule (see COMMAND_STAGES in scripts/train.py). 3.0 m/s is
            # ~74% of the 4.05 m/s the action scale allows: keep the command
            # under that ceiling, because a command the actuators cannot reach
            # is a permanent reward loss the policy can do nothing about.
            lin_vel_x=(-3.0, 3.0),
            # Lateral motion is limited by the legs (hipx 外展 + hipy 俯仰, no
            # hip yaw), so the wheels cannot roll sideways.
            lin_vel_y=(-0.6, 0.6),
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
    # sigma is the error scale of the kernel, not a speed. exp(-e^2/sigma) is
    # ~0 once the error is a few sigma, so sigma has to be a similar fraction of
    # the command range: 0.25 was tuned for legged_gym's +-1 m/s, and at +-2.5
    # m/s it would mean "within 10% or nothing" -- a near-step reward with no
    # gradient to climb. 0.5 keeps the same relative sharpness.
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy_cmd, weight=1.5, params={"sigma": 0.5})
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
    # Legs only. Four wheels at 40 rad/s contribute 4 * 1600 * 1e-4 = 0.64 per
    # step -- 40% of the maximum tracking reward -- which is a direct tax on the
    # behaviour being asked for. Wheel smoothness is already covered by
    # action_rate and joint_acc.
    joint_vel = RewardTermCfg(
        func=rewards.joint_vel_l2, weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"])})
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
        # Wheel target range: w* = a * scale, and ground speed = w * 0.09 m
        # radius, so 45 rad/s is a 4.05 m/s ceiling. The top command (3.0 m/s)
        # uses ~74% of that. Set here rather than as an actions subclass so the
        # standing task's 20 rad/s (1.8 m/s) is provably untouched.
        self.actions.wheel_velocities.scale = 45.0
