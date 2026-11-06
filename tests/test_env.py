import numpy as np
import pytest
import mujoco

from m20rl.config import EnvConfig
from m20rl.env import M20Env


@pytest.fixture
def env():
    e = M20Env()
    e.reset(seed=0)
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (60,)
    assert env.action_space.shape == (16,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (60,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info == {}


def test_default_stance_puts_wheels_on_ground():
    cfg = EnvConfig(joint_noise=0.0, height_noise=(0.0, 0.0), tilt_noise=0.0)
    e = M20Env(env_config=cfg)
    e.reset(seed=0)
    assert abs(e.data.qpos[2] - 0.50) < 1e-9
    for leg in ("fl", "fr", "hl", "hr"):
        body_id = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel")
        wheel_z = e.data.xpos[body_id, 2]
        assert abs(wheel_z - 0.09) < 0.005, f"{leg} wheel z={wheel_z}"
    e.close()


def test_reset_height_within_randomisation_range():
    e = M20Env()
    for seed in range(5):
        e.reset(seed=seed)
        assert 0.495 - 1e-9 <= e.data.qpos[2] <= 0.520 + 1e-9
    e.close()


def test_step_returns_gymnasium_tuple(env):
    obs, reward, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
    assert obs.shape == (60,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_components" in info


def test_reward_components_present(env):
    _, _, _, _, info = env.step(np.zeros(16))
    expected = {
        "track_vx", "track_vy", "track_yaw", "orientation",
        "height", "energy", "smooth", "still", "alive",
    }
    assert expected.issubset(info["reward_components"])


def test_action_is_clipped_to_torque_limits(env):
    env.step(np.full(16, 10.0))
    assert np.all(np.abs(env.tau) <= env.action_scale + 1e-9)


def test_fall_terminates_episode():
    e = M20Env()
    e.reset(seed=0)
    e.data.qpos[2] = 0.10
    mujoco.mj_forward(e.model, e.data)
    _, _, terminated, _, info = e.step(np.zeros(16))
    assert terminated is True
    assert info["reward_components"]["fall_penalty"] < 0.0
    e.close()


def test_episode_truncates_at_limit():
    e = M20Env(env_config=EnvConfig(episode_steps=5))
    e.reset(seed=0)
    e._is_terminated = lambda: False
    for _ in range(4):
        _, _, terminated, truncated, _ = e.step(np.zeros(16))
        assert terminated is False
        assert truncated is False
    _, _, terminated, truncated, _ = e.step(np.zeros(16))
    assert truncated is True
    e.close()
