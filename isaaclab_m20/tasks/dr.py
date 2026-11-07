"""Curriculum-scheduled domain randomization.

Every randomization range in the config has a *full* value. These wrappers
multiply it by a module-level ``LEVEL`` in [0, 1] before handing it to the
IsaacLab event function:

    effective_range = (lo * LEVEL, hi * LEVEL)

LEVEL = 0 turns the randomization off completely and reproduces the
un-randomized baseline; LEVEL = 1 is the full range written in the config. The
training loop ramps LEVEL up over the first part of training, so the policy
learns to stand on the nominal robot before it has to survive a distribution
of robots.

Why a module global instead of editing term params at runtime: the manager
deep-copies the config into its terms, so mutating ``env_cfg.events`` after the
scene is built has no effect. The event functions, on the other hand, are
looked up at call time, so a module global is visible to them.
"""
from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

_LEVEL: float = 0.0


def get_level() -> float:
    return _LEVEL


def set_level(level: float) -> float:
    """Clamp and store the curriculum level. Returns the value actually set."""
    global _LEVEL
    _LEVEL = min(max(float(level), 0.0), 1.0)
    return _LEVEL


def _scale(ranges: dict[str, tuple[float, float]], level: float) -> dict[str, tuple[float, float]]:
    return {key: (lo * level, hi * level) for key, (lo, hi) in ranges.items()}


# --------------------------------------------------------------------------- #
# Recorded intents (reset)
# --------------------------------------------------------------------------- #
def reset_root_state_uniform(env, env_ids, pose_range, velocity_range,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    mdp.reset_root_state_uniform(
        env, env_ids,
        pose_range=_scale(pose_range, _LEVEL),
        velocity_range=_scale(velocity_range, _LEVEL),
        asset_cfg=asset_cfg,
    )


def reset_joints_by_offset(env, env_ids, position_range, velocity_range,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    mdp.reset_joints_by_offset(
        env, env_ids,
        position_range=tuple(v * _LEVEL for v in position_range),
        velocity_range=tuple(v * _LEVEL for v in velocity_range),
        asset_cfg=asset_cfg,
    )


# --------------------------------------------------------------------------- #
# Disturbance (interval)
# --------------------------------------------------------------------------- #
# Mass and centre-of-mass randomization deliberately have no wrapper here: they
# read the current value out of PhysX and add to it, so they compound if run
# again on every reset. They are mode="startup" in the config instead, which
# also means there is nothing to schedule.
# --------------------------------------------------------------------------- #
def push_by_setting_velocity(env, env_ids, velocity_range,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # At level 0 the scaled range is (0, 0), which would *set* the base velocity
    # to zero every interval -- that is a disturbance, not the absence of one.
    # Skip the call entirely instead.
    if _LEVEL <= 0.0:
        return
    mdp.push_by_setting_velocity(
        env, env_ids,
        velocity_range=_scale(velocity_range, _LEVEL),
        asset_cfg=asset_cfg,
    )
