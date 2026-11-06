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
from isaaclab_m20.tasks import rewards


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
    """16 direct joint-torque commands (12 legs + 4 wheels), no PD layer."""

    leg_efforts = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
        scale=76.4,
    )
    wheel_efforts = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=21.6,
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
    flat_orientation = RewardTermCfg(func=rewards.flat_orientation, weight=2.0)
    base_height = RewardTermCfg(
        func=rewards.base_height, weight=2.0,
        params={"target_height": 0.50, "sigma": 20.0})
    base_motion = RewardTermCfg(func=rewards.base_motion, weight=1.0)
    joint_torques = RewardTermCfg(func=rewards.joint_torques, weight=1e-5)
    action_rate = RewardTermCfg(func=rewards.action_rate, weight=0.01)
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.2)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class EventCfg:
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
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
        # 200 Hz physics, 50 Hz policy, direct torque control (no PD).
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 23

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
