"""Velocity-command environment for M20: the first formal training task.

This is the task the training plan describes: one policy that takes a
``(vx, vy, wz)`` command and coordinates 12 leg joints and 4 wheels to follow it
on flat ground. The static-standing task in ``m20_env_cfg.py`` is untouched and
stays available as a separate gym id; only the shared scene is imported from it.

What is fixed by the plan and implemented here
----------------------------------------------
==================  ==========================================================
item                value
==================  ==========================================================
observation         56 dims, body frame (see ``observations.py``)
action              16 dims: 12 leg position targets + 4 wheel speed targets
control rate        50 Hz (decimation 4 on 200 Hz physics)
episode             20 s
rewards             13 weighted terms + fall penalty (see ``rewards.py``)
command mix         50% single axis / 30% two axes / 10% three / 10% zero,
                    held 2-4 s (last stage: 35% zero, held 1.5-3 s)
domain random.      OFF: nominal friction, mass and CoM, no pushes
terrain             flat plane
==================  ==========================================================

Why the command range defaults to zero
--------------------------------------
The config carries the *first* course stage, which is a zero command ("stand
still"), not the final one, so a plain
``train.py --task Isaac-M20-Wheeled-Velocity-v0`` run cannot ask an untrained
policy for 0.5 m/s on the first step. ``scripts/train.py`` rewrites these ranges
per stage through ``dr.set_command_ranges`` and the command distribution through
``CommandSchedule.apply_sampling``; see ``STAGES`` there for the six stages of the
course.

Wheel action scale
------------------
8.0 rad/s per unit action against a 0.09 m wheel radius is 0.72 m/s at full
action, i.e. 69% of the +-0.5 m/s final forward command and ~2.9x the +-0.25 m/s
forward acceptance target. The old 20.0 (~1.8 m/s) is faster than anything this
task commands, and every rad/s of unusable range is a rad/s the policy has to
search through. Speed is worth revisiting only after the tracking numbers in
``scripts/eval.py`` are met.
"""
from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_m20.tasks import commands, dr, observations, rewards
from isaaclab_m20.tasks.m20_env_cfg import M20SceneCfg, fit_viewer_to_grid

# Base ride height. The USD spawns the base at z = 0.50 and the legs' default
# stance (hipy 0.61, knee -1.22) keeps it there, so the height reward and the
# fall termination are both measured against this number.
BASE_HEIGHT = 0.50

# Fall definition, shared by TerminationsCfg and the fall-only penalty so that
# the term that ends the episode is the term that pays it.
# Height: 0.33 m is 17 cm below the ride height -- under anything the legs can
# hold, but still above a folded robot's belly, which is the "sprawled but not
# below 0.28 m" case that used to survive to the time-out. Tilt: 0.5 rad
# (29 deg) against an acceptance target of 5 deg, so a real tip-over ends the
# episode but leaning into an acceleration does not.
MIN_BASE_HEIGHT = 0.33
MAX_TILT_RAD = 0.5


def leg_obs_cfg() -> SceneEntityCfg:
    """A fresh SceneEntityCfg over the 12 leg joints (see ``rewards.leg_asset_cfg``)."""
    return SceneEntityCfg("robot", joint_names=observations.LEG_JOINT_NAMES)


def wheel_obs_cfg() -> SceneEntityCfg:
    """A fresh SceneEntityCfg over the 4 wheel joints."""
    return SceneEntityCfg("robot", joint_names=observations.WHEEL_JOINT_NAMES)


@configclass
class VelocityCommandsCfg:
    """Command sampling term.

    ``ranges`` has to be given explicitly: ``UniformVelocityCommandCfg``
    declares it as ``MISSING`` (every task that uses the term passes its own),
    and leaving it out fails ``cfg.validate()`` at ``gym.make`` with
    "Missing values detected ... commands.base_velocity.ranges".
    """

    base_velocity = commands.MixedVelocityCommandCfg(
        asset_name="robot",
        # Held 2-4 s per the plan: long enough that the policy has to *hold* a
        # command, short enough that one episode of 20 s covers ~6 commands.
        resampling_time_range=(2.0, 4.0),
        # Yaw is commanded as a rate. Heading mode would replace the third
        # column with a feedback law and the command mix below would be
        # meaningless on that axis.
        heading_command=False,
        rel_standing_envs=0.0,     # standing is the 10% zero share instead
        rel_heading_envs=0.0,
        debug_vis=False,
        # Stage 1 of the course: a zero command, i.e. "stand still". The config
        # carries the *first* stage rather than the widest one so that a bare
        # ``gym.make`` cannot ask an untrained policy for 0.5 m/s on the first
        # step. ``scripts/train.py`` widens these per stage at runtime.
        ranges=commands.MixedVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """12 leg position targets + 4 wheel velocity targets.

    The leg scale is the plan's 0.25 rad around the default stance, which is
    what makes ``q_des = q_default + 0.25 * a`` and lets the un-learned policy
    start from the stance the USD was authored in.

    The wheel term is a *speed* target: ``JointVelocityActionCfg`` drives the
    implicit actuator's velocity setpoint, and the actuator's ``kd = 2`` turns
    the error into torque. ``use_default_offset=False`` because a wheel's
    default angle is arbitrary -- adding an offset there would make the target
    depend on how far the wheel happens to have rolled.
    """

    leg_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=observations.LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=observations.WHEEL_JOINT_NAMES,
        scale=8.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """The 56-dim actor input. Declaration order IS the layout."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # -- 3: IMU gyro, body frame -----------------------------------------
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        # -- 3: attitude, body frame (roll/pitch as a gravity projection) -----
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        # -- 3: body-frame linear velocity ------------------------------------
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        # -- 12: leg joints relative to the default stance ---------------------
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": leg_obs_cfg()})
        # -- 12: leg joint velocities -----------------------------------------
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": leg_obs_cfg()})
        # -- 4: wheel angular velocities --------------------------------------
        wheel_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": wheel_obs_cfg()})
        # -- 3: the command itself (must be part of the input) ----------------
        velocity_command = ObservationTermCfg(func=observations.velocity_command)
        # -- 16: previous action, for action smoothness ------------------------
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            # No sensor noise in the first version: the plan asks for a fixed
            # robot and clean state, and noise on top of a not-yet-converged
            # policy makes the tracking numbers harder to attribute.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Task terms, attitude, the hold-still quadratics, then regularisers.

    Weights are the plan's; every penalty term returns a positive magnitude and
    carries its sign here (see the module docstring of ``rewards.py``).

    Three channels are *command-relative quadratics* (``lin_vel_xy_error``,
    ``yaw_rate_error``, ``wheel_roll_error``): they measure "what the base and
    wheels are doing minus what was asked", so with a zero command they are a
    hard brake and with a non-zero command they are a tracking penalty that is
    zero when the command is followed. ``stand_position`` adds the displacement
    half of the same idea. The two base terms are **not** masked by
    ``is_standing_env``: a mask that silently evaluates to zero disables the
    channel and lets a "stand still" robot creep. Only the wheel term and the
    position hold are gated, and their gate is read from the command buffer
    (strictly zero, ``eps = 1e-3``), not from a flag -- because an ungated wheel
    term taxes every skid-steer turn and an ungated position term forbids driving
    away from the spawn point.

    Why quadratics at all when the tracking terms are already there: the exp
    kernels are flat at zero error (zero gradient), so on their own they never
    ask for the last 0.1 m/s to go away. ``scale`` is the error that costs 1.0
    (before the weight): 0.15 m/s, 0.2 rad/s, 0.3 m/s of wheel rolling.

    Why the quadratics, and why **no deadband**: a wheeled base with
    ``stiffness = 0`` on the wheels has no parking brake -- it is a free-rolling
    cart whose only brake is ``damping = 2``. That makes the last few cm/s
    expensive to remove, but a zero command still has to mean zero motion, so the
    residual is charged rather than excused. What actually stops the drift is
    ``stand_position``: a velocity penalty charges the rate, a position penalty
    charges the integral, and 0.05 m/s becomes 1 m of drift (cost 16 per step)
    inside one 20 s episode. The same term pays for coming back after a push.

    NOT present, on purpose: an absolute (command-independent) wheel-speed or
    base-speed penalty -- that is a reward for refusing the command; a
    four-wheel-contact term (the trot lifts two legs); and a lateral tyre-slip
    penalty (the scrub is how lateral motion happens).

    What *is* present and was missing in the first version: ``alive`` (+0.5 per
    step) and a ``termination_penalty`` that only fires on a real fall. Without
    the pair, a 20 s time-out and a head-first collapse cost exactly the same
    -200, so surviving the episode was worth nothing and ending it early cost
    nothing either -- which is why a policy that fell over still reported
    full-length episodes.
    """

    # -- task ---------------------------------------------------------------
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy_cmd, weight=2.0, params={"sigma": 0.25})
    track_ang_vel_z = RewardTermCfg(
        func=rewards.track_ang_vel_z_cmd, weight=1.0, params={"sigma": 0.3})

    # -- attitude and height ------------------------------------------------
    # Quadratic, not exp: on flat ground the tilt should be near zero at all
    # times and a hold-still command needs stiffness there (see rewards.py).
    flat_orientation = RewardTermCfg(
        func=rewards.orientation_sq, weight=-0.5, params={"scale": 0.2})
    base_height = RewardTermCfg(
        func=rewards.base_height_exp, weight=0.5,
        params={"target_height": BASE_HEIGHT, "sigma": 0.05})

    # -- hold still: command-relative quadratics + position hold --------------
    # These are what make "command == 0" mean "stay here", not merely "do not
    # accelerate". The base terms are ungated (subtracting the command makes them
    # free while driving); the wheel term and the position hold are gated on a
    # strictly-zero command read from the command buffer. No deadband anywhere:
    # within it the base would be free to creep, and the acceptance criteria say
    # a zero command means a stopped robot.
    lin_vel_xy_error = RewardTermCfg(
        func=rewards.lin_vel_xy_l2_error, weight=-1.0,
        params={"scale": 0.15})
    yaw_rate_error = RewardTermCfg(
        func=rewards.yaw_rate_l2_error, weight=-0.5,
        params={"scale": 0.2})
    stand_position = RewardTermCfg(
        func=rewards.stand_position_l2, weight=-1.0,
        params={"scale": 0.25})
    wheel_roll_error = RewardTermCfg(
        func=rewards.wheel_roll_l2_error, weight=-0.5,
        params={"scale": 0.3})

    # -- posture and smoothness ----------------------------------------------
    joint_deviation = RewardTermCfg(
        func=rewards.joint_deviation_l2, weight=-0.2, params={"scale": 0.3})
    joint_vel_legs = RewardTermCfg(
        func=rewards.joint_vel_legs_l2, weight=-0.02, params={"scale": 20.0})
    wheel_slip = RewardTermCfg(
        func=rewards.wheel_slip_l2, weight=-0.1, params={"slip_scale": 0.5})
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01)
    joint_torques = RewardTermCfg(
        func=rewards.joint_torques_l2, weight=-0.005, params={"torque_scale": 40.0})
    ang_vel_xy = RewardTermCfg(
        func=rewards.ang_vel_xy_l2, weight=-0.05, params={"scale": 1.0})

    # -- fall and survival ---------------------------------------------------
    # Without the penalty the cheapest way out of a bad command is to fall over:
    # the episode ends and the remaining per-step penalties are never paid.
    # Without the survival bonus, standing to the time-out and collapsing at
    # step 40 are worth the same, so nothing asks for staying upright.
    # Per-step +0.5 against a per-step ceiling of ~3.5.
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewardTermCfg(
        func=rewards.fall_terminated, weight=-200.0,
        params={"minimum_height": MIN_BASE_HEIGHT, "max_tilt": MAX_TILT_RAD})


@configclass
class TerminationsCfg:
    """Fall and timeout.

    Height, not attitude alone: a robot that has pitched onto its knees can
    still project gravity near vertical, but its base has dropped. ``0.33 m`` is
    17 cm below the 0.50 m ride height -- below anything the legs can hold, above
    the lowest body point of a folded robot. ``bad_orientation`` at ``0.5 rad``
    (29 deg) catches the other failure mode, a slow tip-over that keeps some
    height.

    Both limits live in the module constants ``MIN_BASE_HEIGHT`` / ``MAX_TILT_RAD``
    because ``RewardsCfg.termination_penalty`` re-tests the same two conditions:
    the fall that ends the episode is the fall that is penalised. Keep them in
    sync -- the reward function takes them as parameters precisely so that one
    edit moves both.

    The plan also lists "unexpected contact". M20's articulation config carries
    no contact sensor, so there is nothing to test against; a collision is
    instead caught by the two terms above, which is what the height and
    orientation limits do in practice.
    """

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": MIN_BASE_HEIGHT})
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": MAX_TILT_RAD})


@configclass
class EventCfg:
    """Per-episode reset spread, with domain randomization OFF by default.

    The plan's third implementation rule is "fix terrain and dynamics, get
    velocity tracking working, then add friction / mass / terrain
    randomization". So the *nominal* values live here and ``--dr full`` in
    ``scripts/train.py`` widens them (see ``set_domain_randomization`` below).

    Reset spread is kept even with DR off. It is not domain randomization of the
    robot -- it is the initial condition: without it the policy memorises one
    exact starting pose and stumbles the first time a real robot starts 3 cm off.
    """

    # -- reset: initial state (always active, nominal values) ----------------
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,     # not the dr wrapper: no LEVEL scaling
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.03),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.30, 0.30),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": leg_obs_cfg(),
            "position_range": (-0.10, 0.10),
            "velocity_range": (-0.5, 0.5),
        },
    )
    # Continuous wheel joints: zero the accumulated angle each episode, without
    # an offset. Bookkeeping, not randomization -- a non-zero range here would
    # only make the first step noisier for no benefit.
    reset_wheels = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": wheel_obs_cfg(),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- startup: nominal dynamics (widened by --dr full) -------------------
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    # -- interval: shoves ---------------------------------------------------
    # Kept registered but gated by ``dr.LEVEL``: at LEVEL 0 the wrapper returns
    # without touching the robot, which is "no disturbance" rather than a
    # (0, 0) velocity command that would *freeze* the base every few seconds.
    push_robot = EventTermCfg(
        func=dr.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# Full ranges for ``--dr full``. Written out here rather than applied through a
# global LEVEL multiplier: friction, mass and CoM are startup terms sampled once
# per environment, so "ramping" them at runtime would do nothing -- they have to
# change before the scene is built, which is exactly when this is called.
FULL_FRICTION = (0.5, 1.25)
FULL_MASS = (-2.0, 3.0)
FULL_COM = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)}


def set_domain_randomization(env_cfg: ManagerBasedRLEnvCfg, enabled: bool) -> bool:
    """Turn the dynamics randomization and the shoves on or off, before gym.make.

    Returns the state actually applied, so the caller can log it. Must be called
    after the config is built and before the environment exists: the event
    manager deep-copies the term configs at construction.
    """
    events = env_cfg.events
    if enabled:
        events.physics_material.params["static_friction_range"] = FULL_FRICTION
        events.physics_material.params["dynamic_friction_range"] = FULL_FRICTION
        events.base_mass.params["mass_distribution_params"] = FULL_MASS
        events.base_com.params["com_range"] = dict(FULL_COM)
        dr.set_level(1.0)
    else:
        # Back to nominal. Nothing to undo for the startup terms in a fresh
        # process, but this keeps the function honest if it is called twice.
        events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        events.base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        events.base_com.params["com_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        dr.set_level(0.0)
    return enabled


@configclass
class M20WheeledVelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: M20SceneCfg = M20SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # 200 Hz physics, 50 Hz policy.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        # See m20_env_cfg.py: the PhysX GPU contact buffers have to be sized for
        # the environment count or the tensor API fails with an unrelated-looking
        # "Failed to get DOF velocities" error on the next read.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 23
        self.sim.physx.gpu_collision_stack_size = 64 * 2 ** 20

        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, -2.5, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.viewer.resolution = (1280, 720)


__all__ = [
    "BASE_HEIGHT",
    "EventCfg",
    "M20WheeledVelocityEnvCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "VelocityCommandsCfg",
    "fit_viewer_to_grid",
    "set_domain_randomization",
]
