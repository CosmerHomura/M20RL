"""Articulation configuration for the M20 wheeled quadruped."""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# USD imported from the official DeepRoboticsLab/deep_robotics_model M20 URDF.
#
# The bundled USD (model/usd/M20.usd) is a layer stack: M20_physics.usd sublayers
# the 17 MB M20_base.usd, and when that sublayer is missing the whole stage
# composes to a single empty prim -- no ArticulationRootAPI, and the spawner
# fails with "Failed to find an articulation". Importing the official URDF is
# the reliable path and produces the same joint names and default pose.
#
# Generate it once with:
#   python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf
M20_USD_PATH = str(Path(__file__).resolve().parents[2] / "model" / "M20_urdf.usd")

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
        # Direct torque control, no PD layer: stiffness and damping are both
        # zero, so the implicit drive contributes no torque of its own and the
        # effort target written by JointEffortActionCfg is applied as-is.
        # Gravity compensation is learned end-to-end by the policy.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
            effort_limit_sim=76.4,
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=21.6,
            velocity_limit_sim=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
