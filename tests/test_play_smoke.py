import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from m20rl.env import M20Env

ROOT = Path(__file__).resolve().parents[1]


def test_play_runs_headless():
    work_dir = Path(tempfile.mkdtemp(prefix="m20rl_play_"))
    try:
        venv = DummyVecEnv([lambda: M20Env()])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
        venv.reset()
        model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=1)
        model.save(work_dir / "model")
        venv.save(str(work_dir / "vecnorm.pkl"))
        venv.close()

        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.play",
                "--model", str(work_dir / "model"),
                "--vecnormalize", str(work_dir / "vecnorm.pkl"),
                "--steps", "20",
                "--render-mode", "none",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
