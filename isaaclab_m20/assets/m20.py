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
# turn sublayers configuration/M20_base.usd (17 MB of meshes and masses).
#
# That last layer is the fragile part: model/*.usd and model/usd/ are gitignored,
# so a checkout can end up with the two small layers and without the 17 MB one.
# USD then logs "Could not load sublayer @M20_base.usd@" and skips it, the stage
# composes to an empty prim, and the spawner fails with
#   RuntimeError: Failed to find an articulation ... 'USD ArticulationRootAPI'
# which says nothing about the real cause. So the stack is only used when every
# layer is present and M20_base.usd looks like the real 17 MB file.
#
# M20_USD=urdf forces the URDF import instead (scripts/check_m20.py --convert
# urdf). That path is also the one that folds the four massless sensor links
# (front/rear camera, front/rear lidar) into base_link.
_OFFICIAL_LAYERS = (
    _OFFICIAL_USD,
    _MODEL_DIR / "usd" / "configuration" / "M20_physics.usd",
    _MODEL_DIR / "usd" / "configuration" / "M20_base.usd",
)


def _official_stack_usable() -> bool:
    """True only when every layer of the vendor stack exists and is not a stub."""
    for layer in _OFFICIAL_LAYERS:
        if not layer.is_file() or layer.stat().st_size < 1024:
            return False
    return _OFFICIAL_LAYERS[-1].stat().st_size > 1_000_000


_PREFERENCE = os.environ.get("M20_USD", "official").strip().lower()
if _PREFERENCE == "urdf":
    M20_USD_PATH = str(_CONVERTED_USD)
    M20_USD_SOURCE = "urdf (M20_USD=urdf)"
elif _official_stack_usable():
    M20_USD_PATH = str(_OFFICIAL_USD)
    M20_USD_SOURCE = "official layer stack"
else:
    # Deliberately not an error: check_m20.py imports this module to run
    # --convert urdf, which is what creates the fallback file.
    M20_USD_PATH = str(_CONVERTED_USD)
    M20_USD_SOURCE = "urdf (official stack incomplete)"
    _missing = [layer.name for layer in _OFFICIAL_LAYERS if not layer.is_file()]
    print(
        "[WARN] isaaclab_m20.assets: the vendor USD layer stack is incomplete"
        f" (missing: {', '.join(_missing) or 'M20_base.usd is a stub'}),"
        f" falling back to {M20_USD_PATH}.\n"
        "       Copy the whole model/usd/ tree to use the vendor asset: the model/"
        "*.usd and model/usd/ paths are gitignored, so a fresh checkout only has"
        " the small layers."
    )

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
