# M20 轮足机器人 PPO 步态训练 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MuJoCo 中用 stable-baselines3 的 PPO 训练 M20 轮足机器人，使其能追踪 `vx / vy / yaw_rate` 速度指令并保持稳定站立行走。

**Architecture:** 自建 `M20Env(gymnasium.Env)` 封装 MuJoCo 仿真（观测 60 维、动作 16 维力矩），用 `SubprocVecEnv` 并行 10 个环境，外包 `VecNormalize`，交给 SB3 PPO 训练。配置集中在 `config.py`，训练与回放脚本分离。

**Tech Stack:** Python 3.12、MuJoCo 3.x、gymnasium 1.x、stable-baselines3 2.x、PyTorch、TensorBoard、pytest。

**设计文档：** `docs/superpowers/specs/2026-11-06-m20-ppo-gait-training-design.md`

## Global Constraints

- 平台：Windows 11 + Git Bash；路径一律用 `pathlib.Path`，不写死反斜杠。
- Python 3.12；项目内使用 `.venv`，命令统一为 `.venv/Scripts/python.exe`。
- 控制频率 50 Hz：`sim_dt=0.002`，`control_dt=0.02`，`n_substeps=10`。
- episode 上限 500 步（10 s）。
- 观测 60 维 = 机身系线速度 3 + 机身系角速度 3 + 重力投影 3 + 指令 3 + 关节位置 16 + 关节速度 16 + 上一拍动作 16。
- 动作 16 维 ∈ [-1, 1]，映射为力矩；腿部关节上限 76.4 N·m，轮关节上限 21.6 N·m。
- 关节顺序固定：`fl, fr, hl, hr` × `hipx, hipy, knee, wheel`。
- 默认站姿：每个 `hipx=0.0, hipy=0.61, knee=-1.22, wheel=0.0`；站立高度 0.50 m。
- 速度指令范围：`vx ∈ [-0.6, 0.6]`、`vy ∈ [-0.3, 0.3]`、`yaw ∈ [-0.6, 0.6] rad/s`；10% 概率零指令；每 4 s（200 控制步）重采样。
- 基座初始高度 0.495 ~ 0.520 m；关节初始角 ±0.1 rad；地面摩擦系数 0.7 ~ 1.2。
- 训练默认 `device="cpu"`、`n_envs=10`（本机 12 逻辑核）。
- 不使用 IsaacGym / rsl_rl / legged_gym（Windows + 4GB 显存不可行）。
- 所有奖励权重与超参只允许定义在 `m20rl/config.py`，`env.py` 与脚本中不得硬编码数值。

---

## 文件结构总览

| 文件 | 职责 |
|---|---|
| `requirements.txt` | 依赖清单 |
| `conftest.py` | 把项目根加入 `sys.path`，供 pytest 导入 `m20rl` |
| `m20rl/config.py` | 常量 + `EnvConfig` / `RewardConfig` / `TrainConfig` |
| `m20rl/env.py` | `M20Env`：MuJoCo 步进、观测、奖励、终止、渲染 |
| `m20rl/callbacks.py` | `RewardLogCallback`、`SaveAllCallback` |
| `scripts/train.py` | 训练入口 |
| `scripts/play.py` | 回放与录像 |
| `scripts/benchmark.py` | CPU/GPU 与吞吐基准 |
| `scripts/check_env.py` | M0 人工可视化检查 |
| `tests/test_model.py` | XML 模型断言 |
| `tests/test_config.py` | 配置维度/权重断言 |
| `tests/test_env.py` | 环境行为测试 |
| `tests/test_train_smoke.py` | 训练/回放冒烟测试 |

---

### Task 1: 项目骨架与依赖安装

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `conftest.py`
- Create: `m20rl/__init__.py`
- Create: `scripts/__init__.py`
- Create: `README.md`

**Interfaces:**
- Produces: 可导入的 `m20rl` 包、可运行的 `.venv`、pytest 可用的 `sys.path` 配置。

- [ ] **Step 1: 初始化 git 仓库并创建 `.gitignore`**

```bash
cd /f/study/AI/M20RL
git init
```

`.gitignore` 内容：

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
logs/
*.zip
*.pkl
```

- [ ] **Step 2: 创建 `requirements.txt`**

```text
mujoco>=3.2,<4
gymnasium>=1.0,<2
stable-baselines3>=2.4,<3
torch>=2.4
numpy>=1.26,<3
tensorboard>=2.16
imageio>=2.34
imageio-ffmpeg>=0.5
pytest>=8.0
```

- [ ] **Step 3: 创建虚拟环境并安装依赖**

```bash
cd /f/study/AI/M20RL
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Expected: 安装成功。`torch` 若下载 CUDA 版体积较大（约 2 GB），属正常。

- [ ] **Step 4: 验证依赖可导入**

Run:

```bash
.venv/Scripts/python.exe -c "import mujoco, gymnasium, stable_baselines3, torch, imageio; print('mujoco', mujoco.__version__); print('gymnasium', gymnasium.__version__); print('sb3', stable_baselines3.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected: 四行版本号输出，最后一行形如 `torch 2.x.x cuda True` 或 `cuda False`（都算通过）。

- [ ] **Step 5: 创建包标记与 pytest 路径配置**

`m20rl/__init__.py`：

```python
"""M20 wheeled quadruped RL package."""
```

`scripts/__init__.py`：

```python
"""Training, playback and benchmarking entry points."""
```

`conftest.py`：

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

`README.md`：

```markdown
# M20 PPO Gait Training

在 MuJoCo 中用 stable-baselines3 的 PPO 训练 M20 轮足机器人追踪速度指令。

- 设计文档：`docs/superpowers/specs/2026-11-06-m20-ppo-gait-training-design.md`
- 实现计划：`docs/superpowers/plans/2026-11-06-m20-ppo-gait-training.md`

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.train --reward-mode forward_only --total-timesteps 5000000
.venv/Scripts/python.exe -m scripts.play --model logs/<run>/final_model --vecnormalize logs/<run>/final_vecnormalize.pkl
```
```

- [ ] **Step 6: 提交**

```bash
git add .gitignore requirements.txt conftest.py m20rl/__init__.py scripts/__init__.py README.md
git commit -m "chore: project skeleton and dependencies"
```

---

### Task 2: 轮子关节限位显式声明

**Files:**
- Create: `tests/test_model.py`
- Modify: `model/M20.xml:107,144,181,220`（4 个 wheel joint）

**Interfaces:**
- Produces: 4 个轮关节 `jnt_limited == 0`，供 `M20Env` 假设轮子可无限旋转。

- [ ] **Step 1: 写失败测试**

`tests/test_model.py`：

```python
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
```

- [ ] **Step 2: 运行测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_model.py -v
```

Expected: 两种情况之一。
- 若 `autolimits=true` 已按预期把无 `range` 的轮关节设为无限位 → 两个测试直接 PASS（说明不是 bug，但要继续下一步把行为显式钉死）。
- 若有轮关节 FAIL → 说明版本行为不同，必须进入 Step 3 修改 XML。

- [ ] **Step 3: 给 4 个 wheel joint 显式加 `limited="false"`**

在 `model/M20.xml` 中把这 4 行：

```xml
<joint name="fl_wheel_joint" pos="0 0 0" axis="0 -1 0" actuatorfrcrange="-21.6 21.6"/>
<joint name="fr_wheel_joint" pos="0 0 0" axis="0 -1 0" actuatorfrcrange="-21.6 21.6"/>
<joint name="hl_wheel_joint" pos="0 0 0" axis="0 -1 0" actuatorfrcrange="-21.6 21.6"/>
<joint name="hr_wheel_joint" pos="0 0 0" axis="0 -1 0" actuatorfrcrange="-21.6 21.6"/>
```

改为：

```xml
<joint name="fl_wheel_joint" pos="0 0 0" axis="0 -1 0" limited="false" actuatorfrcrange="-21.6 21.6"/>
<joint name="fr_wheel_joint" pos="0 0 0" axis="0 -1 0" limited="false" actuatorfrcrange="-21.6 21.6"/>
<joint name="hl_wheel_joint" pos="0 0 0" axis="0 -1 0" limited="false" actuatorfrcrange="-21.6 21.6"/>
<joint name="hr_wheel_joint" pos="0 0 0" axis="0 -1 0" limited="false" actuatorfrcrange="-21.6 21.6"/>
```

- [ ] **Step 4: 重新运行测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_model.py -v
```

Expected: `2 passed`。

- [ ] **Step 5: 提交**

```bash
git add model/M20.xml tests/test_model.py
git commit -m "fix: pin wheel joints to unlimited rotation"
```

---

### Task 3: 配置模块

**Files:**
- Create: `m20rl/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `JOINT_NAMES: list[str]`（16 个，顺序固定）
  - `DEFAULT_JOINT_POS: list[float]`（16 个）
  - `ACTION_SCALE: list[float]`（16 个）
  - `EnvConfig(...)`，含属性 `n_substeps: int = 10`、`observation_dim: int = 60`、`action_dim: int = 16`
  - `RewardConfig(mode=...)`，方法 `weights() -> dict[str, float]`，键为
    `track_vx, track_vy, track_yaw, orientation, height, energy, smooth, still, alive`
  - `TrainConfig(...)`

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`：

```python
from m20rl.config import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    JOINT_NAMES,
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


def test_default_stance_and_scale():
    for i in range(4):
        assert DEFAULT_JOINT_POS[i * 4 + 0] == 0.0
        assert DEFAULT_JOINT_POS[i * 4 + 1] == 0.61
        assert DEFAULT_JOINT_POS[i * 4 + 2] == -1.22
        assert DEFAULT_JOINT_POS[i * 4 + 3] == 0.0
        assert ACTION_SCALE[i * 4 + 0] == 76.4
        assert ACTION_SCALE[i * 4 + 1] == 76.4
        assert ACTION_SCALE[i * 4 + 2] == 76.4
        assert ACTION_SCALE[i * 4 + 3] == 21.6


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
    assert cfg.batch_size == 256
    assert cfg.ent_coef == 0.01
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'm20rl.config'`。

- [ ] **Step 3: 实现 `m20rl/config.py`**

```python
"""All tunable configuration for the M20 PPO training project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = PROJECT_ROOT / "model" / "M20.xml"

LEGS = ("fl", "fr", "hl", "hr")
JOINT_TYPES = ("hipx", "hipy", "knee", "wheel")

# Order is fixed and must match the <actuator> order in model/M20.xml.
JOINT_NAMES = [f"{leg}_{jt}_joint" for leg in LEGS for jt in JOINT_TYPES]

# hipy forward + knee folded back: two 0.25 m segments, 0.41 m vertical drop.
# With base at 0.50 m the wheel centre sits at 0.09 m == wheel radius.
DEFAULT_JOINT_POS = [0.0, 0.61, -1.22, 0.0] * 4

# Leg joints: 76.4 N·m, wheel joints: 21.6 N·m.
ACTION_SCALE = [76.4, 76.4, 76.4, 21.6] * 4


@dataclass
class EnvConfig:
    xml_path: str = str(XML_PATH)
    sim_dt: float = 0.002
    control_dt: float = 0.02
    episode_steps: int = 500
    standing_height: float = 0.50
    fall_height: float = 0.25
    fall_tilt: float = 0.7
    command_resample_steps: int = 200
    zero_command_prob: float = 0.1
    vx_range: tuple[float, float] = (-0.6, 0.6)
    vy_range: tuple[float, float] = (-0.3, 0.3)
    yaw_range: tuple[float, float] = (-0.6, 0.6)
    joint_noise: float = 0.1
    height_noise: tuple[float, float] = (-0.005, 0.02)
    tilt_noise: float = 0.05
    friction_range: tuple[float, float] = (0.7, 1.2)
    reward_mode: str = "full"

    @property
    def n_substeps(self) -> int:
        return int(round(self.control_dt / self.sim_dt))

    @property
    def observation_dim(self) -> int:
        return 3 + 3 + 3 + 3 + 16 + 16 + 16

    @property
    def action_dim(self) -> int:
        return 16


@dataclass
class RewardConfig:
    mode: str = "full"
    track_vx: float = 1.5
    track_vy: float = 1.0
    track_yaw: float = 1.0
    orientation: float = 1.0
    height: float = 1.0
    energy: float = 1e-5
    smooth: float = 0.01
    still: float = 1.0
    alive: float = 0.2
    fall_penalty: float = 10.0
    sigma: float = 4.0
    height_sigma: float = 20.0
    still_command_threshold: float = 0.1

    def weights(self) -> dict[str, float]:
        weights = {
            "track_vx": self.track_vx,
            "track_vy": self.track_vy,
            "track_yaw": self.track_yaw,
            "orientation": self.orientation,
            "height": self.height,
            "energy": self.energy,
            "smooth": self.smooth,
            "still": self.still,
            "alive": self.alive,
        }
        if self.mode == "forward_only":
            weights["track_vy"] = 0.0
            weights["track_yaw"] = 0.0
            weights["still"] = 0.0
        return weights


@dataclass
class TrainConfig:
    n_envs: int = 10
    total_timesteps: int = 5_000_000
    seed: int = 0
    device: str = "cpu"
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 50_000
    n_eval_episodes: int = 5
    save_freq: int = 100_000
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: `6 passed`。

- [ ] **Step 5: 提交**

```bash
git add m20rl/config.py tests/test_config.py
git commit -m "feat: central configuration for env, rewards and PPO"
```

---

### Task 4: M20Env 核心（初始化 / reset / 观测）

**Files:**
- Create: `m20rl/env.py`
- Create: `tests/test_env.py`

**Interfaces:**
- Consumes: `m20rl.config` 中的 `EnvConfig`、`RewardConfig`、`JOINT_NAMES`、`DEFAULT_JOINT_POS`、`ACTION_SCALE`。
- Produces: `M20Env(env_config=None, reward_config=None, render_mode=None, render_size=(480, 640))`，属性
  `cfg: EnvConfig`、`rcfg: RewardConfig`、`model: mujoco.MjModel`、`data: mujoco.MjData`、
  `observation_space`（Box 60）、`action_space`（Box 16）、`command: np.ndarray(3)`、`last_action`、`tau`、`step_count`；
  方法 `reset(seed=None, options=None) -> (obs, info)`、`_get_obs() -> np.ndarray(float32, 60)`、`render()`、`close()`。

- [ ] **Step 1: 写失败测试**

`tests/test_env.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'm20rl.env'`。

- [ ] **Step 3: 实现 `m20rl/env.py`（本任务只实现到观测/渲染）**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: `4 passed`。

- [ ] **Step 5: 提交**

```bash
git add m20rl/env.py tests/test_env.py
git commit -m "feat: M20 env reset, observations and rendering"
```

---

### Task 5: step / 奖励 / 终止条件

**Files:**
- Modify: `m20rl/env.py`（替换 `step` 占位，新增 `_compute_reward`、`_roll_pitch`、`_is_terminated`）
- Modify: `tests/test_env.py`（追加测试）

**Interfaces:**
- Consumes: Task 4 的 `M20Env`。
- Produces: `step(action) -> (obs, reward: float, terminated: bool, truncated: bool, info: dict)`，
  `info["reward_components"]` 至少包含 `track_vx, track_vy, track_yaw, orientation, height, energy, smooth, still, alive`；
  终止时额外含 `fall_penalty`（负值）。

- [ ] **Step 1: 追加失败测试**

在 `tests/test_env.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: 新增的 5 个测试 FAIL（`NotImplementedError`）。

- [ ] **Step 3: 实现 step、奖励与终止**

在 `m20rl/env.py` 中把 `step` 占位替换为：

```python
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
```

并在 `step` 之前新增：

```python
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
```

- [ ] **Step 4: 运行全部环境测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: `9 passed`（4 个旧测试 + 5 个新测试）。

- [ ] **Step 5: 提交**

```bash
git add m20rl/env.py tests/test_env.py
git commit -m "feat: torque stepping, velocity-tracking rewards and termination"
```

---

### Task 6: 速度指令采样与随机化

**Files:**
- Modify: `m20rl/env.py`（替换 `_sample_command`）
- Modify: `tests/test_env.py`（追加测试）

**Interfaces:**
- Consumes: `EnvConfig.vx_range / vy_range / yaw_range / zero_command_prob / command_resample_steps / reward_mode`。
- Produces: `_sample_command() -> np.ndarray(3, float64)`；`forward_only` 模式下 `vy` 与 `yaw` 恒为 0。

- [ ] **Step 1: 追加失败测试**

在 `tests/test_env.py` 末尾追加：

```python
def test_command_sampling_respects_ranges(env):
    cmds = np.array([env._sample_command() for _ in range(1000)])
    assert cmds[:, 0].min() >= -0.6 and cmds[:, 0].max() <= 0.6
    assert cmds[:, 1].min() >= -0.3 and cmds[:, 1].max() <= 0.3
    assert cmds[:, 2].min() >= -0.6 and cmds[:, 2].max() <= 0.6
    zero_fraction = np.mean(np.all(np.isclose(cmds, 0.0), axis=1))
    assert 0.05 < zero_fraction < 0.15


def test_forward_only_mode_zeroes_vy_and_yaw():
    e = M20Env(env_config=EnvConfig(reward_mode="forward_only"))
    e.reset(seed=0)
    for _ in range(200):
        cmd = e._sample_command()
        assert cmd[1] == 0.0
        assert cmd[2] == 0.0
    e.close()


def test_command_is_resampled_periodically():
    e = M20Env(env_config=EnvConfig(command_resample_steps=2))
    e.reset(seed=0)
    seen = set()
    for _ in range(6):
        e.step(np.zeros(16))
        seen.add(tuple(np.round(e.command, 6)))
    assert len(seen) > 1
    e.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: 新增 3 个测试中 `test_command_sampling_respects_ranges` 与 `test_forward_only_mode_zeroes_vy_and_yaw` FAIL（当前恒返回零）；`test_command_is_resampled_periodically` FAIL（命令永远不变）。

- [ ] **Step 3: 实现指令采样**

在 `m20rl/env.py` 中把 `_sample_command` 替换为：

```python
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
```

- [ ] **Step 4: 运行全部测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest -v
```

Expected: 全部通过（`test_model.py` 2 + `test_config.py` 6 + `test_env.py` 12）。

- [ ] **Step 5: 提交**

```bash
git add m20rl/env.py tests/test_env.py
git commit -m "feat: random velocity commands with resampling"
```

---

### Task 7: 可视化脚本与 M0 验收

**Files:**
- Create: `scripts/check_env.py`
- Modify: `tests/test_env.py`（追加渲染测试）

**Interfaces:**
- Consumes: `M20Env(render_mode="human" | "rgb_array")`、`env.viewer`。
- Produces: 可人工运行的 M0 检查脚本；离屏渲染返回 `(H, W, 3) uint8`。

- [ ] **Step 1: 追加渲染测试**

在 `tests/test_env.py` 末尾追加：

```python
def test_rgb_array_render():
    e = M20Env(render_mode="rgb_array", render_size=(120, 160))
    e.reset(seed=0)
    frame = e.render()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    e.close()


def test_render_without_mode_returns_none(env):
    assert env.render() is None
```

- [ ] **Step 2: 运行测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_env.py -v
```

Expected: `14 passed`。

- [ ] **Step 3: 创建 `scripts/check_env.py`**

```python
"""Manual M0 check: open the MuJoCo viewer and drive the robot with random actions."""
import time

import numpy as np

from m20rl.config import EnvConfig
from m20rl.env import M20Env


def main():
    env = M20Env(env_config=EnvConfig(reward_mode="forward_only"), render_mode="human")
    env.reset(seed=0)
    env.render()
    print("viewer open — close the window or press Ctrl+C to stop")
    try:
        while env.viewer.is_running():
            action = np.random.uniform(-0.3, 0.3, size=16).astype(np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(1 / 50)
            if terminated or truncated:
                print(
                    f"episode ended: height={env.data.qpos[2]:.3f} "
                    f"reward={reward:.2f} cmd={np.round(env.command, 2)}"
                )
                env.reset()
                env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 人工运行 M0 检查（需要显示器）**

Run:

```bash
.venv/Scripts/python.exe -m scripts.check_env
```

人工验收清单：
1. 窗口打开，机器人以默认站姿站在地面上，轮子贴地、没有穿模或弹飞。
2. 随机力矩下机器人会动、会摔，摔倒后 episode 打印并重置，不会崩溃。
3. 观察 `fl_hipy / fl_knee` 的弯曲方向是否符合「大腿前摆、小腿后折」；若完全反了（腿向后折或穿进地板），把 `config.py` 中 `DEFAULT_JOINT_POS` 的 `0.61` 改为 `-0.61`、`-1.22` 改为 `1.22`，重跑本步骤确认。

- [ ] **Step 5: 提交**

```bash
git add scripts/check_env.py tests/test_env.py
git commit -m "feat: visual M0 check script and render tests"
```

---

### Task 8: 性能基准（CPU vs GPU / 并行吞吐）

**Files:**
- Create: `scripts/benchmark.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `M20Env`、`EnvConfig`、SB3 PPO。
- Produces: `bench_stepping(n_envs: int, steps: int = 2000) -> float`（env-steps/s）；
  `bench_training(device: str, total_timesteps: int = 10_000, n_envs: int = 10) -> float`（train-steps/s）。

- [ ] **Step 1: 写失败测试**

`tests/test_benchmark.py`：

```python
from scripts.benchmark import bench_stepping


def test_bench_stepping_single_env_positive():
    fps = bench_stepping(1, steps=20)
    assert fps > 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_benchmark.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'scripts.benchmark'`。

- [ ] **Step 3: 实现 `scripts/benchmark.py`**

```python
"""Measure simulation throughput and PPO training speed on this machine."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from m20rl.config import EnvConfig
from m20rl.env import M20Env


def make_env(rank: int, reward_mode: str = "forward_only"):
    def _thunk():
        return M20Env(env_config=EnvConfig(reward_mode=reward_mode))
    return _thunk


def _vec_env(n_envs: int):
    fns = [make_env(i) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def bench_stepping(n_envs: int, steps: int = 2000) -> float:
    venv = _vec_env(n_envs)
    venv.reset()
    actions = np.random.uniform(-1, 1, (n_envs, 16)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(steps):
        venv.step(actions)
    elapsed = time.perf_counter() - start
    venv.close()
    return (steps * n_envs) / elapsed


def bench_training(device: str, total_timesteps: int = 10_000, n_envs: int = 10) -> float:
    venv = _vec_env(n_envs)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        policy_kwargs={"net_arch": [256, 256]},
        device=device,
        verbose=0,
        seed=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - start
    venv.close()
    return total_timesteps / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-timesteps", type=int, default=10_000)
    parser.add_argument("--train-envs", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = ["# M20 benchmark", ""]
    for n_envs in (1, 4, 10):
        fps = bench_stepping(n_envs, steps=args.steps)
        line = f"stepping: n_envs={n_envs:2d} -> {fps:8.1f} env-steps/s"
        print(line)
        lines.append(f"- {line}")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("CUDA not available — skipping GPU benchmark")
        lines.append("- CUDA not available, GPU benchmark skipped")

    for device in devices:
        fps = bench_training(device, args.train_timesteps, args.train_envs)
        line = f"training: device={device} -> {fps:8.1f} train-steps/s"
        print(line)
        lines.append(f"- {line}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_benchmark.py -v
```

Expected: `1 passed`。

- [ ] **Step 5: 跑一次完整基准并记录**

Run:

```bash
.venv/Scripts/python.exe -m scripts.benchmark --steps 1000 --out docs/benchmark.md
```

Expected: 输出 `stepping` 三行与 `training` 一或两行；`docs/benchmark.md` 生成。

判定规则：
- 若 `training: device=cuda` 的 steps/s 高于 `cpu` **10% 以上**，把 `TrainConfig.device` 默认值改为 `"cuda"`，并在 `README.md` 记录实测数字。
- 否则保持 `"cpu"`，在 `README.md` 记录「实测 GPU 无显著收益」。

- [ ] **Step 6: 提交**

```bash
git add scripts/benchmark.py tests/test_benchmark.py docs/benchmark.md
git commit -m "feat: CPU/GPU throughput benchmark"
```

---

### Task 9: 训练脚本与回调（含冒烟测试）

**Files:**
- Create: `m20rl/callbacks.py`
- Create: `scripts/train.py`
- Create: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `M20Env`、`EnvConfig`、`RewardConfig`、`TrainConfig`。
- Produces:
  - `RewardLogCallback()`：把各奖励分量写入 TensorBoard 的 `reward/<key>`
  - `SaveAllCallback(save_freq: int, save_path: str)`：同时保存模型与 `VecNormalize` 统计
  - `train.py` 命令行参数：`--reward-mode {full,forward_only}`、`--total-timesteps`、`--n-envs`、`--device`、`--seed`、`--run-name`、`--no-eval`、`--log-root`
  - 输出：`<log-root>/<run-name>/final_model.zip`、`final_vecnormalize.pkl`

- [ ] **Step 1: 写失败测试**

`tests/test_train_smoke.py`：

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_smoke_writes_model_and_vecnormalize(tmp_path):
    run_dir = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.train",
            "--reward-mode", "forward_only",
            "--total-timesteps", "1024",
            "--n-envs", "2",
            "--no-eval",
            "--run-name", "smoke",
            "--log-root", str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "final_vecnormalize.pkl").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_train_smoke.py -v
```

Expected: FAIL，脚本不存在（`can't open file ... scripts/train.py`）。

- [ ] **Step 3: 实现 `m20rl/callbacks.py`**

```python
"""Training callbacks for logging reward components and saving normalisation stats."""
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

COMPONENT_KEYS = (
    "track_vx", "track_vy", "track_yaw", "orientation",
    "height", "energy", "smooth", "still", "fall_penalty",
)


class RewardLogCallback(BaseCallback):
    """Log the mean of every reward component to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for key in COMPONENT_KEYS:
            values = [
                info["reward_components"][key]
                for info in infos
                if "reward_components" in info and key in info["reward_components"]
            ]
            if values:
                self.logger.record_mean(f"reward/{key}", float(np.mean(values)))
        return True


class SaveAllCallback(BaseCallback):
    """Periodically save both the policy and the VecNormalize statistics."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo",
                 verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            base = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(base)
            venv = self.model.get_vec_normalize_env()
            if venv is not None:
                venv.save(str(f"{base}_vecnormalize.pkl"))
            if self.verbose:
                print(f"saved checkpoint {base}")
        return True
```

- [ ] **Step 4: 实现 `scripts/train.py`**

```python
"""Train a PPO policy for the M20 wheeled quadruped."""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from m20rl.callbacks import RewardLogCallback, SaveAllCallback
from m20rl.config import EnvConfig, TrainConfig
from m20rl.env import M20Env


def make_env(rank: int, seed: int, reward_mode: str):
    def _thunk():
        env = M20Env(env_config=EnvConfig(reward_mode=reward_mode))
        env.reset(seed=seed)
        return env
    return _thunk


def build_vec_env(n_envs: int, seed: int, reward_mode: str):
    fns = [make_env(i, seed + i, reward_mode) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=["full", "forward_only"],
                        default="forward_only")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--log-root", default="logs")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed)
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.device is not None:
        cfg.device = args.device

    run_name = args.run_name or f"{args.reward_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.log_root) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    venv = build_vec_env(cfg.n_envs, cfg.seed, args.reward_mode)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=cfg.policy_kwargs,
        tensorboard_log=str(log_dir / "tb"),
        device=cfg.device,
        seed=cfg.seed,
        verbose=1,
    )

    callbacks = [
        RewardLogCallback(),
        SaveAllCallback(save_freq=cfg.save_freq, save_path=str(log_dir / "checkpoints")),
    ]
    eval_env = None
    if not args.no_eval:
        eval_env = build_vec_env(1, cfg.seed + 1000, args.reward_mode)
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ))

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )
    model.save(log_dir / "final_model")
    venv.save(str(log_dir / "final_vecnormalize.pkl"))
    print(f"saved model and normalisation to {log_dir}")

    venv.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行冒烟测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_train_smoke.py -v
```

Expected: `1 passed`（约 10–40 s）。

- [ ] **Step 6: 提交**

```bash
git add m20rl/callbacks.py scripts/train.py tests/test_train_smoke.py
git commit -m "feat: PPO training script, reward logging and checkpointing"
```

---

### Task 10: M1 里程碑——只追前向速度的训练

**Files:**
- Modify: `README.md`（记录 M1 结果）
- Output: `logs/m1_forward/`（被 gitignore）

**Interfaces:**
- Consumes: `scripts/train.py --reward-mode forward_only`。
- Produces: 一个能稳定向前行走的策略，用于验证整条训练链路。

- [ ] **Step 1: 启动 M1 训练（5M 步）**

Run:

```bash
.venv/Scripts/python.exe -m scripts.train --reward-mode forward_only --total-timesteps 5000000 --n-envs 10 --run-name m1_forward
```

Expected: 启动后打印 PPO 网络结构与 rollout 日志；本机 12 核 CPU 预计 30–60 分钟。训练中不要关闭终端。

- [ ] **Step 2: 观察训练曲线**

另开终端 Run:

```bash
.venv/Scripts/python.exe -m tensorboard.main --logdir logs/m1_forward/tb
```

浏览器打开 `http://localhost:6006`，检查：

- `rollout/ep_rew_mean`：应整体上升并在后期趋于平台。
- `reward/track_vx`：应从接近 0 上升到 0.8 以上。
- `reward/fall_penalty`：后期应明显减少。
- `rollout/ep_len_mean`：应从几十步上升到接近 500。

- [ ] **Step 3: 判定 M1 是否通过（基于 TensorBoard 指标）**

指标判定：

- ✅ 通过：`reward/track_vx` 终值 > 0.8，`rollout/ep_len_mean` 终值 > 400，`reward/fall_penalty` 后期接近 0。
- ❌ 不通过（原地不动）：`reward/track_vx` 长期低于 0.3 → 把 `RewardConfig.ent_coef` 提高到 `0.02` 重训。
- ❌ 不通过（一直摔）：`reward/fall_penalty` 长期明显负值、`ep_len_mean` 偏低 → 把 `RewardConfig.orientation` 与 `height` 各提高到 `1.5`，重训。

可视化确认在 Task 11 完成后执行（命令见 Task 12 Step 3，把模型路径换成 `logs/m1_forward/...`）。

- [ ] **Step 4: 在 `README.md` 记录 M1 结果**

在 `README.md` 末尾追加：

```markdown
## 训练记录

### M1 — 只追前向速度（forward_only, 5M 步）

- 命令：`scripts/train.py --reward-mode forward_only --total-timesteps 5000000 --n-envs 10`
- 结果：<填写 ep_rew_mean 终值、ep_len_mean 终值、人工观察是否稳定前进>
- 调参：<若无改动写「默认超参」；否则记录改了哪些权重与原因>
```

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs: record M1 forward-walking results"
```

---

### Task 11: 回放与录像脚本

**Files:**
- Create: `scripts/play.py`
- Create: `tests/test_play_smoke.py`

**Interfaces:**
- Consumes: `PPO.load(path)`、`VecNormalize.load(path, venv)`、`M20Env(render_mode=...)`。
- Produces: 命令行可运行的回放脚本，参数
  `--model`、`--vecnormalize`、`--steps`、`--command VX VY YAW`、`--render-mode {human,rgb_array,none}`、`--video`。

- [ ] **Step 1: 写失败测试**

`tests/test_play_smoke.py`：

```python
import subprocess
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from m20rl.env import M20Env

ROOT = Path(__file__).resolve().parents[1]


def test_play_runs_headless(tmp_path):
    venv = DummyVecEnv([lambda: M20Env()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    venv.reset()
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=1)
    model.save(tmp_path / "model")
    venv.save(str(tmp_path / "vecnorm.pkl"))
    venv.close()

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.play",
            "--model", str(tmp_path / "model"),
            "--vecnormalize", str(tmp_path / "vecnorm.pkl"),
            "--steps", "20",
            "--render-mode", "none",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_play_smoke.py -v
```

Expected: FAIL，`can't open file ... scripts/play.py`。

- [ ] **Step 3: 实现 `scripts/play.py`**

```python
"""Roll out a trained PPO policy on the M20 model."""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from m20rl.config import EnvConfig
from m20rl.env import M20Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"],
                        default="human")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    render_mode = args.render_mode
    if args.video and render_mode != "rgb_array":
        render_mode = "rgb_array"

    env = M20Env(env_config=EnvConfig(reward_mode="full"), render_mode=render_mode)
    env.cfg.command_resample_steps = 10 ** 9
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(args.vecnormalize, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model, device="cpu")

    obs = venv.reset()
    env.command = np.array(args.command, dtype=np.float64)

    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=50)

    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)
            frame = env.render()
            if writer is not None and frame is not None:
                writer.append_data(frame)
            if render_mode == "human":
                time.sleep(1 / 50)
            if dones[0]:
                obs = venv.reset()
                env.command = np.array(args.command, dtype=np.float64)
                env.cfg.command_resample_steps = 10 ** 9
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
        venv.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_play_smoke.py -v
```

Expected: `1 passed`。

- [ ] **Step 5: 运行全部测试**

Run:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: 全部通过（约 1–2 分钟，含两个子进程冒烟测试）。

- [ ] **Step 6: 提交**

```bash
git add scripts/play.py tests/test_play_smoke.py
git commit -m "feat: policy playback with optional video recording"
```

---

### Task 12: M2 里程碑——完整速度追踪训练与调参

**Files:**
- Modify: `README.md`（记录 M2 结果与调参）
- Output: `logs/m2_full/`、`logs/m2_full/video.mp4`（被 gitignore）

**Interfaces:**
- Consumes: `scripts/train.py --reward-mode full`、`scripts/play.py`。
- Produces: 能同时追踪 `vx / vy / yaw` 的最终策略与演示视频。

- [ ] **Step 1: 启动 M2 训练（20M 步起步）**

Run:

```bash
.venv/Scripts/python.exe -m scripts.train --reward-mode full --total-timesteps 20000000 --n-envs 10 --run-name m2_full
```

Expected: 本机预计 2–4 小时。若中途中断，可从最近 checkpoint 继续（见 Step 4）。

- [ ] **Step 2: 观察完整速度追踪指标**

Run:

```bash
.venv/Scripts/python.exe -m tensorboard.main --logdir logs/m2_full/tb
```

判定目标：
- `reward/track_vx`、`reward/track_vy`、`reward/track_yaw` 三项在后期都 > 0.6（满分 1.0）。
- `rollout/ep_len_mean` 接近 500。
- `reward/fall_penalty` 曲线贴地（接近 0）。

- [ ] **Step 3: 依次验证三种指令**

Run:

```bash
.venv/Scripts/python.exe -m scripts.play --model logs/m2_full/final_model --vecnormalize logs/m2_full/final_vecnormalize.pkl --steps 500 --command 0.3 0 0
.venv/Scripts/python.exe -m scripts.play --model logs/m2_full/final_model --vecnormalize logs/m2_full/final_vecnormalize.pkl --steps 500 --command 0 0.2 0
.venv/Scripts/python.exe -m scripts.play --model logs/m2_full/final_model --vecnormalize logs/m2_full/final_vecnormalize.pkl --steps 500 --command 0 0 0.3
```

人工验收：三次分别表现为「向前走」「向左侧移」「原地左转」，且都不摔倒。

- [ ] **Step 4: 调参规则（仅在 Step 2/3 不达标时执行）**

| 现象 | 调整 |
|---|---|
| 机器人原地不动或只轻微抖动 | `RewardConfig.ent_coef` 0.01 → 0.02；确认 `reward/track_*` 不上升 |
| 一直摔倒 | `orientation` 与 `height` 权重 1.0 → 1.5 |
| 能走但速度明显偏慢 | 检查 `energy` 是否过大：1e-5 → 5e-6 |
| 动作抖动/抽搐 | `smooth` 0.01 → 0.02 |
| 零指令时不静止 | `still` 1.0 → 1.5 |
| 转向迟钝 | `track_yaw` 1.0 → 1.5 |

每次只改一项，改动后重训并记录。修改位置统一在 `m20rl/config.py` 的 `RewardConfig`。

从 checkpoint 继续训练：

```bash
.venv/Scripts/python.exe -c "from stable_baselines3 import PPO; from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize; from m20rl.env import M20Env; from m20rl.config import EnvConfig; venv=VecNormalize.load('logs/m2_full/checkpoints/ppo_<NUM>_steps_vecnormalize.pkl', DummyVecEnv([lambda: M20Env(env_config=EnvConfig(reward_mode='full'))])); m=PPO.load('logs/m2_full/checkpoints/ppo_<NUM>_steps', env=venv); m.learn(total_timesteps=10000000, reset_num_timesteps=False); m.save('logs/m2_full/final_model'); venv.save('logs/m2_full/final_vecnormalize.pkl')"
```

（把 `<NUM>` 替换为实际 checkpoint 数字。）

- [ ] **Step 5: 录制演示视频**

Run:

```bash
.venv/Scripts/python.exe -m scripts.play --model logs/m2_full/final_model --vecnormalize logs/m2_full/final_vecnormalize.pkl --steps 500 --command 0.3 0 0 --video logs/m2_full/video.mp4
```

Expected: 生成 `logs/m2_full/video.mp4`，播放可见机器人前进。

- [ ] **Step 6: 在 `README.md` 记录 M2 结果**

追加：

```markdown
### M2 — 完整速度追踪（full, 20M 步）

- 命令：`scripts/train.py --reward-mode full --total-timesteps 20000000 --n-envs 10`
- 最终指标：track_vx=<值>, track_vy=<值>, track_yaw=<值>, ep_len_mean=<值>
- 调参记录：<逐条列出改了哪项、为什么、效果如何>
- 演示视频：logs/m2_full/video.mp4
```

- [ ] **Step 7: 提交**

```bash
git add README.md
git commit -m "docs: record M2 full velocity-tracking results"
```

---

## 自检结果

**1. 规格覆盖检查**

| 设计文档章节 | 实现任务 |
|---|---|
| §3 项目结构 | Task 1 |
| §4 XML 修正（轮子限位） | Task 2 |
| §4.1 默认站姿 | Task 3（常量）+ Task 4（几何测试）+ Task 7（人工校准） |
| §5 观测/动作 | Task 3、Task 4 |
| §6 奖励/终止/时间步 | Task 5 |
| §7 随机化与指令 | Task 6 |
| §8 训练超参 / VecNormalize / 日志 | Task 3、Task 9 |
| §2 CPU/GPU 决策与实测 | Task 8 |
| §9 M0 | Task 4–8（环境 + 基准 + 人工检查） |
| §9 M1 | Task 10 |
| §9 M3 录制验证 | Task 11、Task 12 |
| §10 风险（Windows spawn） | Task 9（`if __name__ == "__main__"` + SubprocVecEnv） |
| §11 不做的事 | 无对应任务（符合预期） |

**2. 占位符扫描**：无 TBD/TODO；Task 12 中的 `<NUM>` 是运行时可确定的 checkpoint 编号，已在同一步给出替换说明。

**3. 类型/命名一致性**：`M20Env` 构造签名、`EnvConfig`/`RewardConfig`/`TrainConfig` 字段、`weights()` 返回值键、`reward_components` 键、`make_env(rank, seed, reward_mode)`、`build_vec_env(n_envs, seed, reward_mode)` 在各任务间保持一致。
