import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_smoke_writes_model_and_vecnormalize():
    work_dir = Path(tempfile.mkdtemp(prefix="m20rl_smoke_"))
    try:
        run_dir = work_dir / "smoke"
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.train",
                "--reward-mode", "forward_only",
                "--total-timesteps", "1024",
                "--n-envs", "2",
                "--no-eval",
                "--run-name", "smoke",
                "--log-root", str(work_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert (run_dir / "final_model.zip").exists()
        assert (run_dir / "final_vecnormalize.pkl").exists()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
