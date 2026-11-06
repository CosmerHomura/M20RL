"""MuJoCo environment for the M20 wheeled quadruped."""
from __future__ import annotations

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from .config import ACTION_SCALE, DEFAULT_JOINT_POS, JOINT_NAMES, EnvConfig, RewardConfig


class M20Env(gym.Env):
    """Torque-controlled wheeled quadruped tracking body-frame velocity commands."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_config=None, reward_config=None, render_mode=None,
                 render_size=(480, 640)):
        super().__init__()
        self.cfg = env_config or EnvConfig()
        self.rcfg = reward_config or RewardConfig(mode=self.cfg.reward_mode)
        self.render_mode = render_mode
        self.render_size = render_size

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ], dtype=np.int32)
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_adr = self.model.jnt_dofadr[self.joint_ids]
        self.ctrl_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ], dtype=np.int32)

        self.action_scale = np.array(ACTION_SCALE, dtype=np.float64)
        self.default_joint_pos = np.array(DEFAULT_JOINT_POS, dtype=np.float64)
        # Collision geoms belonging to the torso (visual geoms have contype == 0).
        self.base_geom_ids = frozenset(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.base_body_id
            and self.model.geom_contype[g] != 0
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.observation_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.cfg.action_dim,), dtype=np.float32)

        self.tau = np.zeros(self.cfg.action_dim)
        self.last_action = np.zeros(self.cfg.action_dim)
        self.command = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.viewer = None
        self._renderer = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        qpos[0:3] = 0.0
        qpos[2] = self.cfg.standing_height + self.np_random.uniform(*self.cfg.height_noise)

        euler = np.array([
            self.np_random.uniform(-self.cfg.tilt_noise, self.cfg.tilt_noise),
            self.np_random.uniform(-self.cfg.tilt_noise, self.cfg.tilt_noise),
            0.0,
        ])
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, euler, "xyz")
        qpos[3:7] = quat
        qpos[self.qpos_adr] = self.default_joint_pos + self.np_random.uniform(
            -self.cfg.joint_noise, self.cfg.joint_noise, size=self.cfg.action_dim)
        self.data.qvel[:] = 0.0

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_friction[floor_id, 0] = self.np_random.uniform(
            *self.cfg.friction_range)

        mujoco.mj_forward(self.model, self.data)

        self.command = self._sample_command()
        self.last_action = np.zeros(self.cfg.action_dim)
        self.tau = np.zeros(self.cfg.action_dim)
        self.step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------ observations
    def _base_velocity(self):
        """Body-frame linear and angular velocity (local frame)."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, res, 1)
        return res[3:6].copy(), res[0:3].copy()

    def _projected_gravity(self):
        rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def _get_obs(self):
        lin_vel, ang_vel = self._base_velocity()
        joint_pos = self.data.qpos[self.qpos_adr] - self.default_joint_pos
        joint_vel = self.data.qvel[self.qvel_adr]
        obs = np.concatenate([
            lin_vel,
            ang_vel,
            self._projected_gravity(),
            self.command,
            joint_pos,
            joint_vel,
            self.last_action,
        ])
        return obs.astype(np.float32)

    def _sample_command(self):
        return np.zeros(3, dtype=np.float64)

    # --------------------------------------------------------------- gym API
    def step(self, action):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self.render_size[0],
                    width=self.render_size[1],
                )
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
