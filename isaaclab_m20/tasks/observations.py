"""Observation terms for the M20 velocity task.

The policy sees exactly 56 numbers. The order is fixed by the declaration order
of the terms in ``ObservationsCfg.PolicyCfg`` and is the layout the deployed
controller has to reproduce:

===========  ====  ============================================================
count        dims  term
===========  ====  ============================================================
3            0-2   base angular velocity, body frame (IMU gyro)
3            3-5   projected gravity, body frame (roll/pitch attitude)
3            6-8   base linear velocity, body frame
12           9-20  leg joint positions, relative to the default stance
12           21-32 leg joint velocities
4            33-36 wheel joint velocities
3            37-39 velocity command (vx, vy, wz), body frame
16           40-55 previous action
===========  ====  ============================================================

Body frame everywhere: the linear velocity, the gravity projection and the
command all live in the base frame, and training and deployment must agree on
that -- a command interpreted in the world frame is a different task, and the
policy would drive in the wrong direction whenever it yaws.

Only one term needs code here. The leg and wheel quantities are
``isaaclab.envs.mdp.joint_pos_rel`` / ``joint_vel_rel`` restricted to the leg or
wheel joint names, which is also how the action manager resolves them, so the
observation and the action index the joints in the same order.
"""
from __future__ import annotations

import torch

# The joint groups. Kept in one place because the observation, the action and
# several reward terms must all resolve to the same 12 / 4 joints in the same
# order; ``SceneEntityCfg`` resolves ``joint_names`` expression by expression,
# so an identical list gives identical indices everywhere.
LEG_JOINT_NAMES = [".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"]
WHEEL_JOINT_NAMES = [".*_wheel_joint"]


def velocity_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    """The ``(vx, vy, wz)`` command, 3 dims.

    ``UniformVelocityCommand`` keeps four columns in ``vel_command_b``: the
    fourth is the heading target, which this task does not use
    (``heading_command=False``). Slicing here instead of returning the whole
    buffer is what keeps the observation exactly 56 wide -- with the heading
    column it would be 57 and every downstream shape (actor input, checkpoint,
    deployment) would be off by one.
    """
    command = env.command_manager.get_command(command_name)
    return command[:, :3]
