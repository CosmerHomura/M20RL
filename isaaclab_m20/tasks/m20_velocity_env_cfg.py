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
rewards             10 weighted terms + fall penalty (see ``rewards.py``)
command mix         50% single axis / 30% two axes / 10% three / 10% zero,
                    held 2-4 s; stages 2-4 keep 50/30/20% zero commands
                    (stage 4 also shortens the hold to 1.5-3 s)
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
``CommandSchedule.apply_sampling``; see ``STAGES`` there for the four stages of
the course.

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
    """Eleven entries: two task channels, three state channels, three
    regularisers, two gated hold channels and the fall.

    Weights are the plan's; every penalty term returns a positive magnitude and
    carries its sign here (see the module docstring of ``rewards.py``).

    Ten, not sixteen. The previous revision had a matching pair of channels for
    each axis group -- an exp kernel *and* a quadratic for linear velocity, again
    for yaw, plus a gated wheel-speed term and three regularisers that measured
    almost nothing -- which is two springs of different stiffness on the same
    error: impossible to attribute, impossible to tune, and the reason the table
    kept growing. The fix was to replace pairs, not to add a fourth one:

    * one bounded kernel per axis group (``track_lin_vel_xy``, ``track_ang_vel_z``,
      Cauchy form ``1/(1+e^2/sigma^2)``) instead of exp + quadratic;
    * ``stand_position`` alone carries "a zero command means zero motion": it
      prices the *displacement* a drift produces, which is what the acceptance
      test measures. At 0.05 m/s the drift costs 0.04/step after one second and
      16/step after twenty -- the velocity channels never needed to be doubled;
    * ``leg_joint_acc_l2`` replaces the velocity channel (too weak to price
      chatter by an order of magnitude) and ``action_rate_l2`` is normalised, so
      jitter is no longer free.

    Two terms are gated, both on a strictly-zero command read from the command
    buffer (``eps = 1e-3``): ``stand_position`` (where the robot ended up) and
    ``wheel_speed`` (wheels spinning under a still body). Everything else is
    ungated because it is either relative to the command or meaningful while
    driving -- an ungated position term would forbid driving away from the anchor
    point, and an ungated wheel-speed term would tax in-place turns.

    NOT present, on purpose: an absolute (command-independent) wheel-speed or
    base-speed penalty -- that is a reward for refusing the command; a
    four-wheel-contact term (the trot lifts two legs); and a lateral tyre-slip
    penalty (the scrub is how lateral motion happens).

    Also absent: a per-step survival bonus. It reads like the fall penalty's
    mirror image, but a bonus is collected every step and a penalty once, so it
    mostly rescales the return, and with the reward floor in
    ``isaaclab_m20.envs`` every step is already worth at least zero -- dying can
    no longer be cheaper than living badly. The fall penalty is then a strictly
    negative event on its own.
    """

    # -- task ---------------------------------------------------------------
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy, weight=2.0, params={"sigma": 0.25})
    track_ang_vel_z = RewardTermCfg(
        func=rewards.track_ang_vel_z, weight=1.0, params={"sigma": 0.3})

    # -- attitude and height ------------------------------------------------
    # Quadratic, not exp: on flat ground the tilt should be near zero at all
    # times and a hold-still command needs stiffness there (see rewards.py).
    flat_orientation = RewardTermCfg(
        func=rewards.orientation_sq, weight=-0.5, params={"scale": 0.2})
    base_height = RewardTermCfg(
        func=rewards.base_height_exp, weight=1.0,
        params={"target_height": BASE_HEIGHT, "sigma": 0.05})
    # Attitude *rate*, which the quadratic tilt does not cover: a small-amplitude
    # wobble keeps the mean tilt near zero and is otherwise almost free.
    ang_vel_xy = RewardTermCfg(
        func=rewards.ang_vel_xy_l2, weight=-0.2, params={"scale": 1.0})

    # -- hold still ----------------------------------------------------------
    # One of the two gated terms, and it is gated on a strictly-zero command read
    # from the command buffer. No deadband: within one the base would be free to
    # creep, and 0.25 m of creep already costs 1.0 per step.
    stand_position = RewardTermCfg(
        func=rewards.stand_position_l2, weight=-1.0,
        params={"scale": 0.25})
    # Spinning wheels under a still body. The body terms cannot see this: with the
    # legs holding the base up, the wheels can be commanded to any speed without
    # moving the robot, and ``stand_position`` stays at zero. Gated like the term
    # above for the same reason -- when driving, the wheels are *supposed* to
    # spin, and a rollout that did not gate this would tax every command.
    wheel_speed = RewardTermCfg(
        func=rewards.wheel_speed_l2, weight=-0.5, params={"scale": 2.0})

    # -- posture and smoothness ----------------------------------------------
    joint_deviation = RewardTermCfg(
        func=rewards.joint_deviation_l2, weight=-0.1, params={"scale": 0.3})
    leg_joint_acc = RewardTermCfg(
        func=rewards.leg_joint_acc_l2, weight=-0.02, params={"acc_scale": 200.0})
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2, weight=-0.01, params={"action_scale": 0.05})

    # -- fall ----------------------------------------------------------------
    # No survival bonus. It looks like the mirror image of the fall penalty, but
    # it is not: a bonus is paid every step and a penalty once, so the bonus
    # mostly rescales the return. What actually stops the policy from falling on
    # purpose is the reward floor in ``isaaclab_m20.envs``, which makes every
    # step worth at least zero -- so dying cannot be cheaper than living badly.
    # With the floor, the fall penalty alone is a strictly negative event.
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
