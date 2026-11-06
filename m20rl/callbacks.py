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
