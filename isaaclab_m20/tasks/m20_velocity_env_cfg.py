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
rewards             18 weighted per-step terms, aligned term-for-term with
                    legbot_lab's 16 (see ``rewards.py``) and a terminal -10
                    applied by ``M20VelocityEnv`` on falls only; the perception
                    is ``tasks/control.py``'s resolved target, not the raw
                    command
command mix         50% single axis / 30% two axes / 10% three / 10% zero,
                    held 2-4 s; every stage keeps a zero-command share
                    (100/50/50/50/30/20%) and the last stage shortens the
                    hold to 1.5-3 s
domain random.      OFF: nominal friction, mass and CoM, no pushes
terrain             flat plane
==================  ==========================================================

Why the command range defaults to zero
--------------------------------------
The config carries the *first* course stage, which is a zero command ("stand
still"), not the final one, so a plain
``train.py --task Isaac-M20-Wheeled-Velocity-v0`` run cannot ask an untrained
policy for anything above zero on the first step. ``isaaclab_m20.training.
curriculum.CommandSchedule`` rewrites these ranges per stage through
``dr.set_command_ranges``, and applies the command distribution through
``apply_sampling``; the six stages themselves live in
``isaaclab_m20.training.stages.STAGES``.

Wheel action scale
------------------
27.78 rad/s per unit action against a 0.09 m wheel radius is **2.5 m/s at full
action**, which is the top speed the last two stages command and the whole course
ramps up to in three forward bands.
That is the speed envelope the whole course is built around. It is a policy
choice, not a hardware one: the wheel actuator allows 200 rad/s (18 m/s at the
contact) and its 21.6 N*m is 240 N per wheel against a 338 N friction ceiling for
the 34.5 kg robot. The earlier 8.0 (0.72 m/s) was picked to leave exploration
headroom and instead capped the robot below walking speed; 22.2 (2.0 m/s) was the
first correction and 27.78 is the current one. Changing the scale changes the
meaning of the four wheel actions, so any change restarts the course from stage 1.
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_m20.tasks import commands, control, dr, observations, rewards
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
        # |a| = 1 asks for 27.78 rad/s. With the 0.09 m wheel that is 2.5 m/s, the
        # speed the top stage commands. The motor itself allows far more (the
        # actuator velocity limit is 200 rad/s = 18 m/s, and 21.6 N*m / 0.09 m =
        # 240 N per wheel against a 338 N friction ceiling for the whole robot),
        # so the scale -- not the hardware -- sets the envelope. 8.0 was the old
        # value and capped the robot at 0.72 m/s, slower than walking; 22.2 gave
        # 2.0 m/s and is what the shared checkpoints were trained with.
        scale=27.78,
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
        # -- 3: the resolved command target (body-frame v*, yaw-rate target) ---
        command_target = ObservationTermCfg(func=observations.command_target)
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
    """Eighteen per-step channels, aligned with legbot_lab's sixteen plus two of
    our own. The fall penalty is *not* here -- it is applied by
    ``isaaclab_m20.envs.M20VelocityEnv`` from the termination flags (see
    ``fall_penalty`` on the env cfg).

    Weights are legbot_lab's (README_cn.md, "奖励函数 16 项"), re-expressed in our
    normalisation so that a weight means the same thing on both sides: their
    ``1e-7 * sum(qdd^2)`` is our ``-0.4 * sum((qdd/2000)^2)``, their
    ``1e-4 * sum(tau^2)`` is our ``-1e-4 * sum(tau^2 / 1^2)``. Where we keep a
    different number it is noted next to the term.

    Every penalty term returns a positive magnitude and carries its sign here (see
    the module docstring of ``rewards.py``).

    The revision that came before this one had a pair of channels for each axis
    group -- an exp kernel *and* a quadratic for linear velocity, again for yaw --
    plus a gated position hold and three regularisers that measured almost
    nothing. Two springs of different stiffness on the same error cannot be
    attributed or tuned, which is why the table kept growing. This set replaces
    pairs instead of adding to them:

    * one exp kernel per axis group (``track_lin_vel_xy``, ``track_ang_vel_z``,
      ``exp(-e^2/sigma^2)`` with ``sigma = 0.5`` -- legged_gym's
      ``tracking_sigma = 0.25`` written in this form): the standard shape, and
      honest about its band -- beyond ~1 m/s of error it gives nothing, so the
      course ramps the command range instead of sampling the envelope from the
      start;
    * height is a quadratic error, not an exp bonus -- on flat ground 0.50 m is a
      set point, so the channel wants stiffness around it and no ceiling;
    * ``leg_joint_acc`` replaces the leg joint-velocity channel and
      ``action_rate`` is normalised, so joint chatter is no longer free.

    The wheeled-legged channels are the ones a legged stack does not have:
    ``wheel_axis_alignment`` (a wheel rolled onto its edge is invisible to every
    body-frame term) and ``undesired_contacts`` (resting on a knee keeps the base
    high and level, so neither termination fires).

    **Two mode-gated terms**, both reading the label the sampler wrote (never a
    float re-test here):

    * ``wheel_axis_alignment`` -- weighted 100 % STAND / 25 % MOVE / 0 % otherwise
      by the function itself: a turn or a strafe *needs* a misaligned axle.
    * ``zero_xy_speed`` -- ROTATE: a pure yaw command must not translate. The
      tracking kernel only asks for a zero *linear* target, which is a soft way to
      say "stay put"; this is the hard one that the spin task is actually about.

    ``standing_joint_vel`` uses a looser gate on purpose (``|v*| < 0.1`` and
    ``|wz*| < 0.1`` rather than exact zero): it is pricing the twitch that makes a
    stopped robot creep, not the label of the moment.

    Everything else is ungated because it is either relative to the command or
    meaningful while driving.

    Deliberately absent, and each one was a deliberate deletion:

    * **No gated ``||v_xy||^2``.** With a zero command the tracking kernel already
      asks for zero velocity -- a second quadratic on the same error would be the
      two-springs problem again, on the axis that matters most.
    * **No position hold (``p_hold``).** "Get back to where you were pushed from"
      is a third capability, on top of move / stop / stand. It is measured on its
      own (net displacement of the hold test) before it earns a reward term;
      stacking it on the same reward makes every failure ambiguous.
    * **No survival bonus (``alive``).** It was tried twice: once as "the fall
      penalty's opposite", then again as "the counterweight to the reward floor".
      Both were compensations for other mistakes. The floor is gone (it clamped
      every negative step to a constant 0 and erased the gradient -- see
      ``envs.py``), and with it the need for a bonus that lifts the ordinary step
      above it. PPO does not require non-negative rewards: the advantage is
      ``R - V`` and normalising rescales without reordering, so a step that is
      merely less bad than the critic expected still gets a positive advantage.
      What a survival bonus does add is a constant offset per step, which inflates
      the return scale without telling the policy anything about *how* to survive.
    * **No reward floor either.** The clamp used to live in ``envs.py``; it is
      deleted, so ``Train/mean_reward`` is now the sum of the real terms and can
      legitimately go negative while the policy is bad.
    * **No absolute (command-independent) wheel- or base-speed penalty**, which
      would tax obeying the command; **no four-wheel-contact term** (the trot
      lifts two legs); **no lateral tyre-slip penalty** (the scrub is how lateral
      motion happens).
    """

    # -- task ---------------------------------------------------------------
    # Both kernels compare against ``control.command_targets``: the command-frame
    # velocity rotated into the body frame, and the outer loop's yaw-rate target.
    # That is the only definition of "on target" in this project.
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy, weight=2.0, params={"sigma": 0.5})
    track_ang_vel_z = RewardTermCfg(
        func=rewards.track_ang_vel_z, weight=1.0, params={"sigma": 0.5})
    # The third gate: a pure spin must not translate. Zero linear target alone does
    # not express that strongly enough, and "stay in place while spinning" is the
    # whole task of the ROTATE mode rather than a side condition.
    zero_xy_speed = RewardTermCfg(
        func=rewards.zero_xy_speed_l2, weight=-1.0, params={"scale": 0.15})

    # -- base state ----------------------------------------------------------
    # Quadratic, not exp, and in raw SI units like legbot_lab's: the terms below
    # carry their own scale factor (metres, m/s, rad/s squared), so the weight is
    # the only number to compare against theirs.  ``base_height`` is theirs (-10,
    # 0.49 m) raised to -25: at -10 the first stage-2 run settled at h = 0.396 m
    # (10 cm low, 0.108/step, which is 1.8x everything the positive side can
    # earn) and still moved, so the crouch was not a deliberate trade -- the term
    # simply was not paying for it.  -25 makes 10 cm cost 0.27/step and 5 cm
    # 0.068/step: stiff enough to hold ride height, still an order of magnitude
    # softer than the -200 equivalent this file started with.
    base_height = RewardTermCfg(
        func=rewards.base_height_l2, weight=-25.0,
        params={"target_height": BASE_HEIGHT, "scale": 1.0})
    flat_orientation = RewardTermCfg(
        func=rewards.orientation_sq, weight=-0.5, params={"scale": 0.2})
    # Vertical bounce: with no foot contacts, a bouncing body is the cheapest way
    # to look like it is walking, and neither the tilt nor the height term sees it.
    lin_vel_z = RewardTermCfg(
        func=rewards.lin_vel_z_l2, weight=-1.0, params={"scale": 1.0})
    # Attitude *rate*, which the quadratic tilt does not cover: a small-amplitude
    # wobble keeps the mean tilt near zero and is otherwise almost free.
    ang_vel_xy = RewardTermCfg(
        func=rewards.ang_vel_xy_l2, weight=-0.05, params={"scale": 1.0})

    # -- wheels and contacts (the wheeled-legged channels) --------------------
    # Wheel axle alignment, gated by mode inside the function: square to the body
    # when parked (100 %), worth something while driving straight (25 %), off when
    # the command requires a misaligned axle (turn / strafe).
    wheel_axis_alignment = RewardTermCfg(
        func=rewards.wheel_axis_alignment_l2, weight=-0.6)
    # Leg links touching the ground. The height and tilt terminations do not catch
    # a robot resting on a knee with the base still level and high enough.
    undesired_contacts = RewardTermCfg(
        func=rewards.undesired_contacts_legs, weight=-1.0,
        params={"threshold": 5.0})

    # -- posture, smoothness, effort -----------------------------------------
    # L1 and split by joint group, the way legbot_lab does it: -0.08 for the four
    # hips (they set how wide the stance is), -0.01 for thighs/shins (they are
    # expected to move), and both doubled when the command is "stand". The earlier
    # L2 sum over 12 joints at 0.3 rad charged -0.367/step for a pose that is
    # merely sagging under its own weight, i.e. more than the whole positive side
    # of the reward -- "unfold the legs" became the only gradient.
    hip_pos = RewardTermCfg(
        func=rewards.joint_pos_deviation_l1, weight=-0.08,
        params={"joint_names": [".*_hipx_joint"], "scale": 1.0, "stand_scale": 2.0})
    joint_pos = RewardTermCfg(
        func=rewards.joint_pos_deviation_l1, weight=-0.01,
        params={"joint_names": [".*_hipy_joint", ".*_knee_joint"],
                "scale": 1.0, "stand_scale": 2.0})
    # acc_scale 2000 and action_scale 1.0 are *not* arbitrary: the exploration
    # noise is what these terms mostly price early on (see rewards.py), and at
    # 200 / 0.05 the noise alone cost ~-30/step. The weights below are
    # legbot_lab's, expressed in our normalisation, so they are 20x (-0.02 ->
    # -0.4) and 10x (-0.005 -> -0.05 via 1e-4) heavier than before -- the point
    # of that is that a jittering leg is no longer cheaper than a still one.
    leg_joint_acc = RewardTermCfg(
        func=rewards.leg_joint_acc_l2, weight=-0.4, params={"acc_scale": 2000.0})
    joint_torques = RewardTermCfg(
        func=rewards.joint_torques_l2, weight=-1e-4, params={"torque_scale": 1.0})
    joint_power = RewardTermCfg(
        func=rewards.joint_power, weight=-2e-5)
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2, weight=-0.01, params={"action_scale": 1.0})
    action_smoothness = RewardTermCfg(
        func=rewards.action_smoothness_l2, weight=-0.01, params={"action_scale": 1.0})
    # Soft-limit violations on the twelve leg joints only: a wheel's angle is
    # unbounded, so testing its limit would be meaningless.
    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": leg_obs_cfg()})
    # Leg and wheel velocity while the command asks for nothing. Gated on a looser
    # threshold than the parking brake (0.1 m/s / 0.1 rad/s vs exact zero): a
    # leftover twitch under a stop command is what makes the robot creep.
    standing_joint_vel = RewardTermCfg(
        func=rewards.standing_joint_vel_l2, weight=-0.02)

    # The fall penalty is not a reward term: ``isaaclab_m20.envs.M20VelocityEnv``
    # adds it on the steps where ``terminated & ~truncated``, i.e. exactly the
    # steps the termination manager ended with a fall. A reward term would have to
    # re-derive that condition from the robot state (a second copy of the
    # height/tilt test) and could only read the *previous* step's flags, because
    # the termination manager runs after the reward manager. The magnitude lives
    # on the env cfg as ``fall_penalty = -10``.


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
    so that a single edit moves the whole rule. The penalty itself is applied by
    ``M20VelocityEnv`` from these flags, so there is no second copy of the test to
    keep in sync: the fall that ends the episode is exactly the fall that is
    penalised.

    The plan also lists "unexpected contact", and the scene now has a contact
    sensor for it: a net force above ``10 N`` on the base means the body took a
    hit rather than the legs. legbot_lab terminates on exactly this (their only
    other termination is the timeout); here it is added on top of the height and
    tilt tests, which catch the slow failures a single contact event does not.
    """

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": MIN_BASE_HEIGHT})
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": MAX_TILT_RAD})
    illegal_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*base.*"]),
            "threshold": 10.0,
        },
    )


@configclass
class EventCfg:
    """Per-episode reset spread, with domain randomization OFF by default.

    The plan's third implementation rule is "fix terrain and dynamics, get
    velocity tracking working, then add friction / mass / terrain
    randomization". So the *nominal* values live here and ``--dr full`` in
    ``scripts/train.py`` widens them (see ``set_domain_randomization`` below).

    Reset spread is kept even with DR off. It is not domain randomization of the
    robot -- it is the initial condition: without it the policy memorises one
    exact starting pose and stumbles the first time a real robot starts a couple
    of centimetres off.

    What it deliberately does *not* do any more is drop the robot from a height.
    The spread used to include ``z`` up to 3 cm plus ``+-0.10 rad`` of leg offset;
    at the nominal stance the wheels only just touch the ground, so both of those
    lifted them by a centimetre or two. The policy sees a speed command from step
    zero, spins the wheels up while airborne, and lands with a skid -- an
    unlearnable transient at the start of every episode, in exactly the episodes
    where the fast bands need clean data. The spread now stays within 6 mm of
    contact. If drop robustness is wanted later, add it back as its own curriculum
    stage (0 cm -> 1 cm -> 2 cm) once the flat-ground tracking works.
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
                # Spawn *on* the ground, not 0-3 cm above it. With a gap here the
                # robot free-falls for ~78 ms (4 control steps) while the policy
                # already sees a speed command and spins the wheels up; the wheels
                # then touch down with a surface speed of up to 2.5 m/s, the
                # friction cone saturates for a step or two and the episode starts
                # with a skid. That transient is unlearnable and it poisons the
                # first advantage estimates of every episode -- which is exactly
                # what made the high-speed bands look slow. 5 mm is enough to keep
                # the solver from starting in penetration.
                "z": (0.0, 0.005),
                # Tilt also lifts wheels off the ground (half-track 0.30 m, so
                # 0.05 rad lifts one side by 1.5 cm); 0.02 rad keeps that under
                # 6 mm while still randomising the start.
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
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
            # +-0.05 rad, not +-0.10: at the nominal stance the wheels just touch
            # the ground, so a 0.10 rad knee offset lifts or buries a wheel by
            # ~2 cm -- the same drop-then-skid artefact as the spawn gap above.
            "position_range": (-0.05, 0.05),
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
class VelocitySceneCfg(M20SceneCfg):
    """``M20SceneCfg`` plus a contact sensor over the robot's bodies.

    Two reward channels and one termination need contact forces: leg links that
    should not touch the ground, and the base taking a net hit. The standing task
    has no use for them, and a sensor costs a PhysX query patch per step, so it
    lives here rather than in the shared scene: subclassing keeps the standing
    task's ``M20SceneCfg`` byte-identical.

    ``history_length=3`` is what ``undesired_contacts_legs`` reads (it takes the
    max over the window); ``update_period=0.0`` makes the sensor run every physics
    step, which is what makes that window meaningful at 200 Hz.
    """

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        update_period=0.0,
        debug_vis=False,
    )


@configclass
class M20WheeledVelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Fall penalty, applied by ``isaaclab_m20.envs.M20VelocityEnv`` on the steps
    # that ended in a fall (``terminated & ~truncated``). -10 rather than -200
    # because the real cost of falling is losing the rest of the episode's positive
    # rewards; a large penalty on top of that only inflates the value scale the
    # critic has to fit. Zero would make falling free, so it is kept negative.
    fall_penalty: float = -10.0

    # -- wheel parking brake (see ``isaaclab_m20.envs.M20VelocityEnv``) --------
    # While the velocity command is exactly zero the env replaces the four wheel
    # action columns with a position loop: ``omega* = kp * (q_hold - q)``, clipped
    # to +-``omega_max``, and the existing velocity servo (``damping = 2``) tracks
    # it. The wheels are a pure velocity servo with ``stiffness = 0`` and PhysX
    # does not model rolling resistance, so without this a body a fraction of a
    # degree off level creeps at ``F_bias / (4 * damping)`` -- a few cm/s, enough
    # to leave the acceptance box over an 8 s hold.
    #
    # Gains: the loop has no damping of its own -- ``kd`` stays the shared velocity
    # servo ``damping = 2`` (``kd / I = 700 1/s``), which is already far faster than
    # the position loop, so the cascade is overdamped and does not ring. Only the
    # stiffness is ours: kp = 15 1/s is a 67 ms time constant and an equivalent
    # parking stiffness of ``kp * damping = 30 N*m/rad`` (0.5 N*m/deg, well under the
    # legs' 80 N*m/rad). It is deliberately soft: 25 1/s braked the wheel hard
    # enough to be felt as a lurch at the go/stop edge. ``omega_max = 3 rad/s`` is
    # 0.27 m/s: a fast walk, but capped so that a wheel that got dragged cannot be
    # commanded into a large spin back. ``error_clamp = 0.3 rad`` is 2.7 cm of
    # rolling; beyond it the wheel was dragged or skidded and the hold is simply
    # re-seeded at the current angle.
    wheel_park_enabled: bool = True
    wheel_park_kp: float = 15.0
    wheel_park_omega_max: float = 3.0
    wheel_park_error_clamp: float = 0.3

    # -- yaw outer loop (see ``isaaclab_m20.tasks.control``) ------------------
    # ``omega_target = psi_dot_cmd + kp * e_psi - kd * (psi_dot_meas - psi_dot_cmd)``
    # feeds the third command slot of the observation. Kp is 1/s, Kd is
    # dimensionless, and the numbers are starting values: the plant inside the loop
    # is the policy tracking a yaw-rate command, so "critical damping" is not a
    # property of this pair -- judge them by the yaw RMSE in ``eval`` and the
    # settling time in ``play``.
    #
    # ``omega_max`` / ``omega_rate`` are limits on the *target*, not on the robot:
    # without them a 180 degree branch switch asks for a rate the policy cannot
    # deliver, the tracking term saturates, and the critic sees a discontinuity.
    # ``alpha_*`` shape how fast the travel-direction set point may move (branch
    # switches and command changes), which is what keeps the PD output smooth.
    yaw_kp: float = 1.2
    yaw_kd: float = 0.25
    yaw_omega_max: float = 1.5
    yaw_omega_rate: float = 60.0
    yaw_alpha_tau: float = 0.05
    yaw_alpha_rate: float = 3.0
    yaw_branch_hysteresis: float = 0.15

    # Where the operator's "+vx" lives, in degrees, relative to the asset's +x.
    # 0 keeps the asset convention: the vendor URDF puts ``fl/fr`` and the
    # ``front_camera`` site at +x, so "+vx" drives toward that end.  If the real
    # robot's nose is the other end, set 180 and everything downstream (the
    # target, the reward, the embedded observation slice 37:40) flips together --
    # but the policy is then solving a different task, so retrain from stage 1.
    # It is a task definition, not a display setting.
    command_frame_yaw: float = 0.0

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
