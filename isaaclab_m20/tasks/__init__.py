"""Gym registration for the M20 wheeled tasks.

The standing task is a plain ``ManagerBasedRLEnv``. The velocity task uses
``isaaclab_m20.envs:M20VelocityEnv``, which applies the fall penalty from the
termination flags (falls only, never the time-out), adds no reward floor (see that
module for why the floor was removed), and parks the wheels with a position outer
loop while the velocity command is exactly zero.
"""
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-M20-Wheeled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m20_env_cfg:M20WheeledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20RslRlPpoRunnerCfg",
    },
)

gym.register(
    id="Isaac-M20-Wheeled-Velocity-v0",
    entry_point="isaaclab_m20.envs:M20VelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m20_velocity_env_cfg:M20WheeledVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20VelocityRslRlPpoRunnerCfg",
    },
)
