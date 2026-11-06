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
