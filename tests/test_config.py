from m20rl.config import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    JOINT_NAMES,
    MAX_TORQUE,
    PD_KD,
    PD_KP,
    EnvConfig,
    RewardConfig,
    TrainConfig,
)


def test_joint_order_is_fixed():
    assert JOINT_NAMES[:4] == [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    ]
    assert JOINT_NAMES[4:8] == [
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    ]
    assert JOINT_NAMES[8:12] == [
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    ]
    assert JOINT_NAMES[12:16] == [
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
    ]


def test_dimensions():
    cfg = EnvConfig()
    assert cfg.observation_dim == 60
    assert cfg.action_dim == 16
    assert cfg.n_substeps == 10
    assert len(DEFAULT_JOINT_POS) == 16
    assert len(ACTION_SCALE) == 16
    assert len(MAX_TORQUE) == 16
    assert len(PD_KP) == 16
    assert len(PD_KD) == 16


def test_default_stance_and_action_scale():
    for i in range(4):
        assert DEFAULT_JOINT_POS[i * 4 + 0] == 0.0
        assert DEFAULT_JOINT_POS[i * 4 + 1] == 0.61
        assert DEFAULT_JOINT_POS[i * 4 + 2] == -1.22
        assert DEFAULT_JOINT_POS[i * 4 + 3] == 0.0
        assert ACTION_SCALE[i * 4 + 0] == 0.4
        assert ACTION_SCALE[i * 4 + 1] == 0.4
        assert ACTION_SCALE[i * 4 + 2] == 0.4
        assert ACTION_SCALE[i * 4 + 3] == 21.6
        assert MAX_TORQUE[i * 4 + 3] == 21.6
        assert MAX_TORQUE[i * 4 + 0] == 76.4
        assert PD_KP[i * 4 + 3] == 0.0
        assert PD_KD[i * 4 + 3] == 0.0


def test_full_reward_weights():
    w = RewardConfig(mode="full").weights()
    assert w["track_vx"] == 1.5
    assert w["track_vy"] == 1.0
    assert w["track_yaw"] == 1.0
    assert w["orientation"] == 1.0
    assert w["height"] == 1.0
    assert w["energy"] == 1e-5
    assert w["smooth"] == 0.01
    assert w["still"] == 1.0
    assert w["alive"] == 0.2


def test_forward_only_reward_weights():
    w = RewardConfig(mode="forward_only").weights()
    assert w["track_vx"] == 1.5
    assert w["track_vy"] == 0.0
    assert w["track_yaw"] == 0.0
    assert w["still"] == 0.0


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.n_envs == 10
    assert cfg.device == "cpu"
    assert cfg.policy_kwargs == {"net_arch": [256, 256]}
    assert cfg.n_steps == 2048
    assert cfg.batch_size == 512
    assert cfg.n_epochs == 5
    assert cfg.ent_coef == 0.01
