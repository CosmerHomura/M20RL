"""Manager-based environment configuration for the M20 wheeled quadruped."""
from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_m20.assets import M20_CFG
from isaaclab_m20.tasks import dr, rewards


@configclass
class M20SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """12 leg position targets + 4 wheel velocity targets, tracked by the PD servo.

    The policy emits targets, not torques: q* = q_default + a*0.25 rad for the
    legs and ω* = a*20 rad/s for the wheels. The implicit actuator's PD runs at
    the 200 Hz physics rate and turns them into motor torque.
    """

    leg_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
        scale=0.25,
        use_default_offset=True,
    )
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=20.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
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
class RewardsCfg:
    """Standard legged-locomotion reward set with a zero velocity command.

    Weights follow legged_gym (Rudin et al.) and IsaacLab's AnymalC task; see
    docs/algorithm.md for the table and the reasoning behind each term.
    """

    # -- task: exp tracking kernels evaluated at a zero command ---------------
    track_lin_vel_xy = RewardTermCfg(
        func=rewards.track_lin_vel_xy_zero, weight=1.5, params={"sigma": 0.25})
    track_ang_vel_z = RewardTermCfg(
        func=rewards.track_ang_vel_z_zero, weight=0.75, params={"sigma": 0.25})
    # Wheels: a standing robot's wheels must be still. The generic joint_vel term
    # is ~1e-4 on a quantity of order 1, i.e. no useful gradient near zero speed.
    track_wheel_vel = RewardTermCfg(
        func=rewards.wheel_vel_exp, weight=0.5, params={"sigma": 2.0})

    # -- base stability -------------------------------------------------------
    lin_vel_z = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewardTermCfg(func=rewards.flat_orientation_l2, weight=-2.5)
    base_height = RewardTermCfg(
        func=rewards.base_height_l2, weight=-20.0, params={"target_height": 0.50})

    # -- posture and smoothness ----------------------------------------------
    joint_deviation = RewardTermCfg(func=rewards.joint_deviation_l1, weight=-0.5)
    joint_vel = RewardTermCfg(func=rewards.joint_vel_l2, weight=-1e-4)
    joint_acc = RewardTermCfg(func=rewards.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewardTermCfg(func=rewards.joint_torques_l2, weight=-1e-5)
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01)

    # -- survival -------------------------------------------------------------
    # The large negative termination term is what stops the policy from learning
    # to fall over: without it, ending the episode early avoids the per-step
    # smoothness/torque penalties and is the cheaper option.
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.15)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class EventCfg:
    """Domain randomization plus the per-episode reset randomization.

    Startup terms are sampled once when the scene is built; reset terms are
    re-sampled for every environment every time its episode restarts, so a
    single policy has to cope with a distribution of robots rather than one.
    """

    # -- startup: contact properties ------------------------------------------
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # -- reset: initial state -------------------------------------------------
    reset_base = EventTermCfg(
        func=dr.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.04),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.30, 0.30),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.1, 0.1),
            },
        },
    )
    # Leg joints only: a wheel's absolute angle is meaningless, so offsetting it
    # buys nothing and just adds noise to the first step.
    reset_joints = EventTermCfg(
        func=dr.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
            ),
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.5, 0.5),
        },
    )
    # Wheels are continuous joints: their absolute angle accumulates across
    # episodes and PhysX never wraps it. Zero it every reset so the joint_pos
    # observation does not drift upward all run -- but do not add an offset,
    # the angle itself carries no information.
    # Deliberately NOT curriculum-scaled: this is a bookkeeping reset, not a
    # randomization, so it has to run at every level including 0.
    reset_wheels = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- startup: dynamics ----------------------------------------------------
    # Payload on the body: cameras, a lid, a different battery. Additive in kg,
    # and the inertia is recomputed so the tensor stays consistent with the mass.
    #
    # MUST be mode="startup", not "reset". IsaacLab's mass/com randomizers read
    # the *current* value from PhysX and add to it, so re-running them on every
    # reset compounds: an environment that resets 300 times ends up adding a
    # sample 300 times, and its mass climbs without bound. That shows up exactly
    # as survival rising and then collapsing late in training. With startup the
    # sample is drawn once, but per environment -- 4096 environments still give
    # the policy a distribution of robots to survive.
    base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    # Centre-of-mass shift: mounted equipment is never perfectly centred.
    # Startup for the same reason as base_mass.
    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )

    # -- interval: impulsive pushes -------------------------------------------
    # Instantaneous velocity change, i.e. a shove. Episode is 10 s, so this
    # lands roughly once or twice per episode.
    push_robot = EventTermCfg(
        func=dr.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class M20WheeledEnvCfg(ManagerBasedRLEnvCfg):
    scene: M20SceneCfg = M20SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # 200 Hz physics, 50 Hz policy: the actuator PD runs at the physics rate.
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        # GPU contact buffers. gpu_total_aggregate_pairs_capacity defaults to
        # 1024*1024 in IsaacLab; 16*1024 (64x smaller) overflows once several
        # thousand envs share the GPU, and a PhysX buffer overflow surfaces as a
        # generic "Failed to get DOF velocities from backend" on the next tensor
        # API call -- not as a PhysX-looking error.
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


def fit_viewer_to_grid(env_cfg: ManagerBasedRLEnvCfg, num_envs: int | None = None) -> None:
    """Pull the viewport back to frame the whole env grid.

    The default viewer follows env_0 up close, which shows one robot. Call this
    after overriding ``scene.num_envs`` to see every robot at once.
    """
    n = int(num_envs or env_cfg.scene.num_envs)
    if n < 4:
        # One or two robots: keep the close-up that follows env_0.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (2.5, -2.5, 1.6)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
        return

    cols = max(int(math.ceil(math.sqrt(n))), 1)
    span = (cols - 1) * env_cfg.scene.env_spacing
    center = span / 2.0
    dist = max(span * 0.9, 6.0)

    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (center, center - dist, dist * 0.8)
    env_cfg.viewer.lookat = (center, center, 0.0)
