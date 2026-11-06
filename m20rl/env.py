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
        if self.np_random.random() < self.cfg.zero_command_prob:
            return np.zeros(3, dtype=np.float64)
        if self.cfg.reward_mode == "forward_only":
            return np.array(
                [self.np_random.uniform(*self.cfg.vx_range), 0.0, 0.0],
                dtype=np.float64,
            )
        return np.array([
            self.np_random.uniform(*self.cfg.vx_range),
            self.np_random.uniform(*self.cfg.vy_range),
            self.np_random.uniform(*self.cfg.yaw_range),
        ], dtype=np.float64)

    # --------------------------------------------------------------- gym API
    def _roll_pitch(self):
        rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
        return roll, pitch

    def _is_terminated(self):
        if self.data.qpos[2] < self.cfg.fall_height:
            return True
        roll, pitch = self._roll_pitch()
        if abs(roll) > self.cfg.fall_tilt or abs(pitch) > self.cfg.fall_tilt:
            return True
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.base_geom_ids or contact.geom2 in self.base_geom_ids:
                return True
        return False

    def _compute_reward(self, action, prev_action):
        lin_vel, ang_vel = self._base_velocity()
        vx, vy = lin_vel[0], lin_vel[1]
        yaw_rate = ang_vel[2]
        height = self.data.qpos[2]
        gravity = self._projected_gravity()
        cmd = self.command
        w = self.rcfg.weights()

        track_vx = np.exp(-self.rcfg.sigma * (vx - cmd[0]) ** 2)
        track_vy = np.exp(-self.rcfg.sigma * (vy - cmd[1]) ** 2)
        track_yaw = np.exp(-self.rcfg.sigma * (yaw_rate - cmd[2]) ** 2)
        orientation = -(gravity[0] ** 2 + gravity[1] ** 2)
        height_term = np.exp(
            -self.rcfg.height_sigma * (height - self.cfg.standing_height) ** 2)
        energy = -np.sum(self.tau ** 2)
        smooth = -np.sum((action - prev_action) ** 2)
        if np.linalg.norm(cmd) < self.rcfg.still_command_threshold:
            still = -(vx ** 2 + vy ** 2)
        else:
            still = 0.0

        components = {
            "track_vx": w["track_vx"] * float(track_vx),
            "track_vy": w["track_vy"] * float(track_vy),
            "track_yaw": w["track_yaw"] * float(track_yaw),
            "orientation": w["orientation"] * float(orientation),
            "height": w["height"] * float(height_term),
            "energy": w["energy"] * float(energy),
            "smooth": w["smooth"] * float(smooth),
            "still": w["still"] * float(still),
            "alive": w["alive"],
        }
        return float(sum(components.values())), components

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        prev_action = self.last_action
        self.tau = action * self.action_scale
        self.data.ctrl[self.ctrl_ids] = self.tau
        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward, components = self._compute_reward(action, prev_action)
        terminated = self._is_terminated()
        if terminated:
            reward -= self.rcfg.fall_penalty
            components["fall_penalty"] = -self.rcfg.fall_penalty

        truncated = self.step_count >= self.cfg.episode_steps
        self.last_action = action
        obs = self._get_obs()

        if self.step_count % self.cfg.command_resample_steps == 0:
            self.command = self._sample_command()

        return obs, float(reward), bool(terminated), bool(truncated), {
            "reward_components": components,
        }

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
