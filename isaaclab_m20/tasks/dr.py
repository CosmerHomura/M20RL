"""Curriculum-scheduled difficulty.

Two independent levels are scheduled here:

* ``LEVEL`` -- scales every domain-randomization range.
* ``COMMAND_LEVEL`` -- scales the velocity-command sampling range.

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


# --------------------------------------------------------------------------- #
# Velocity command range (velocity task only)
# --------------------------------------------------------------------------- #
# ``UniformVelocityCommand`` reads ``cfg.ranges`` inside ``_resample_command``,
# i.e. every time a command is drawn, so mutating the term's config takes effect
# at the next resample. Nothing is deep-copied away from us here, unlike events.
_COMMAND_TERM = None
_FULL_COMMAND_RANGES: tuple | None = None


def register_command_term(command_term) -> None:
    """Capture the full command ranges, before the first scaling."""
    global _COMMAND_TERM, _FULL_COMMAND_RANGES
    _COMMAND_TERM = command_term
    ranges = command_term.cfg.ranges
    _FULL_COMMAND_RANGES = (tuple(ranges.lin_vel_x), tuple(ranges.lin_vel_y),
                            tuple(ranges.ang_vel_z))


def set_command_level(level: float) -> float:
    """Scale the command sampling range by ``level``. Returns the clamped level."""
    if _COMMAND_TERM is None or _FULL_COMMAND_RANGES is None:
        return 0.0
    level = min(max(float(level), 0.0), 1.0)
    ranges = _COMMAND_TERM.cfg.ranges
    ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (
        tuple(v * level for v in full) for full in _FULL_COMMAND_RANGES)
    return level


def zero_command(command_term) -> None:
    """Overwrite the (vx, vy, wz) columns with zeros.

    The commands drawn at scene construction used the full range, and the first
    resample is up to 4 s away -- long enough for the untrained policy to be
    asked for 1 m/s straight out of the gate. Call this after
    ``set_command_level(0)`` so training really does start from standing.
    """
    buffer = getattr(command_term, "vel_command_b", None)
    if buffer is not None:
        buffer[:, :3] = 0.0
