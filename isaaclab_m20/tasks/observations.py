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
3            37-39 command target: (vx*, vy*) in the body frame, yaw rate target
16           40-55 previous action
===========  ====  ============================================================

Body frame everywhere: the linear velocity, the gravity projection and the
command target all live in the base frame, and training and deployment must
agree on that -- a command interpreted in the world frame is a different task,
and the policy would drive in the wrong direction whenever it yaws.

Slots 37-39 are not the raw command. They are the *resolved* target from
``isaaclab_m20.tasks.control``: the command-frame velocity rotated into the body
frame (``v_body = R(-psi_rel) . v_cmd``) plus the yaw-rate target the outer loop
produced. The policy therefore sees "which way and how fast, in my own frame, and
how fast should I be rotating" -- one meaning shared by all four modes, instead
of a mode-dependent reading of the same three numbers.

Only one term needs code here. The leg and wheel quantities are
``isaaclab.envs.mdp.joint_pos_rel`` / ``joint_vel_rel`` restricted to the leg or
wheel joint names, which is also how the action manager resolves them, so the
observation and the action index the joints in the same order.
"""
from __future__ import annotations

import torch

from isaaclab_m20.tasks import control

# The joint groups. Kept in one place because the observation, the action and
# several reward terms must all resolve to the same 12 / 4 joints in the same
# order; ``SceneEntityCfg`` resolves ``joint_names`` expression by expression,
# so an identical list gives identical indices everywhere.
LEG_JOINT_NAMES = [".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"]
WHEEL_JOINT_NAMES = [".*_wheel_joint"]


def command_target(env) -> torch.Tensor:
    """The ``(vx_body*, vy_body*, omega*)`` target, 3 dims -- observation slots 37-39.

    Read from ``control.command_targets`` so the observation, the reward and the
    outer loop cannot disagree: a second copy of the rotation is exactly the kind
    of drift this project keeps paying for.
    """
    targets = control.command_targets(env)
    return torch.cat((targets.v_body, targets.omega.unsqueeze(1)), dim=1)


def velocity_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    """The raw ``(vx, vy, wz)`` command, 3 dims -- command frame, for bookkeeping.

    ``UniformVelocityCommand`` keeps four columns in ``vel_command_b``: the
    fourth is the heading target, which this task does not use
    (``heading_command=False``). Slicing here instead of returning the whole
    buffer is what keeps the observation exactly 56 wide -- with the heading
    column it would be 57 and every downstream shape (actor input, checkpoint,
    deployment) would be off by one.

    Not the policy input any more: the policy reads :func:`command_target`. This
    is the operator's request, still in the command frame, and is what ``play``
    and ``eval`` print next to the resolved target when something looks wrong.
    """
    term = env.command_manager.get_term(command_name)
    # ``vel_command_b`` is the 2.x name; older releases called the same buffer
    # ``command``, so fall back rather than break on a minor-version bump.
    buffer = getattr(term, "vel_command_b", None)
    if buffer is None:
        buffer = term.command
    return buffer[:, :3]
