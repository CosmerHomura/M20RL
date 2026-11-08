"""Articulation configuration for the M20 wheeled quadruped."""
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
_OFFICIAL_USD = _MODEL_DIR / "usd" / "M20.usd"
_CONVERTED_USD = _MODEL_DIR / "M20_urdf.usd"

# The vendor asset is the authoritative source for the joint order, the joint
# limits and the default pose -- no conversion in between. It is a layer stack:
# model/usd/M20.usd (1.4 KB) sublayers configuration/M20_physics.usd, which in
# turn sublayers configuration/M20_base.usd (17 MB of meshes and masses). When the
# configuration/ directory is present the stack composes normally; if a sublayer
# is ever lost the stage collapses to an empty prim and the spawner fails with
# "Failed to find an articulation", which is what pushed this repo towards the
# URDF import originally.
#
# M20_USD=urdf forces the imported file instead. That import remains useful: it
# can fold the four massless sensor links (front/rear camera, front/rear lidar)
# into base_link, see scripts/check_m20.py --convert urdf.
_PREFERENCE = os.environ.get("M20_USD", "official").strip().lower()
if _PREFERENCE != "urdf" and _OFFICIAL_USD.is_file():
    M20_USD_PATH = str(_OFFICIAL_USD)
else:
    # Deliberately not an error when the file is missing: check_m20.py imports
    # this module to run --convert urdf, which is what creates it.
    M20_USD_PATH = str(_CONVERTED_USD)

M20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=M20_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hipx_joint": 0.0,
            ".*_hipy_joint": 0.61,
            ".*_knee_joint": -1.22,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # PD servo loop, the standard legged-robot arrangement: the policy sends
        # position targets and the drive tracks them at the physics rate.
        # tau = kp * (q* - q) + kd * (qd* - qd), clamped by effort_limit_sim.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
            effort_limit_sim=76.4,
            velocity_limit_sim=40.0,
            stiffness=80.0,
            damping=2.0,
        ),
        # Wheels are velocity controlled: zero stiffness turns the drive into a
        # velocity servo whose gain is `damping`.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=21.6,
            velocity_limit_sim=200.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
