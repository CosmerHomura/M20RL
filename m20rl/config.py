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
