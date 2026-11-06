from pathlib import Path

import mujoco

XML = Path(__file__).resolve().parents[1] / "model" / "M20.xml"


def test_wheel_joints_are_unlimited():
    model = mujoco.MjModel.from_xml_path(str(XML))
    for leg in ("fl", "fr", "hl", "hr"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_wheel_joint")
        assert jid >= 0, f"{leg}_wheel_joint not found"
        assert model.jnt_limited[jid] == 0, f"{leg}_wheel_joint is limited"


def test_leg_joints_keep_their_ranges():
    model = mujoco.MjModel.from_xml_path(str(XML))
    for leg in ("fl", "fr", "hl", "hr"):
        for joint_type in ("hipx", "hipy", "knee"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint_type}_joint")
            assert model.jnt_limited[jid] == 1
            assert model.jnt_range[jid, 1] > model.jnt_range[jid, 0]
