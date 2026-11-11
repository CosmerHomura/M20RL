"""The ``[health]`` / ``[terms]`` / ``[track]`` lines, attached to ``env.step``.

Everything here is read-only diagnostics: each function grabs the manager's own
state or re-evaluates a reward term, and every block is wrapped so a diagnostic
can never take a run down. Nothing in this module is needed for training to work
-- delete the ``attach_diagnostics`` call and the run is unchanged.
"""
from __future__ import annotations

import torch

from isaaclab_m20.tasks import control


def reward_term_contributions(env, manager) -> dict[str, float] | None:
    """Per-step contribution of every reward term: ``value * weight * dt``.

    ``_episode_sums`` answers "what did this term contribute over the episode",
    which is what tensorboard plots -- but it is dt-scaled and keeps growing
    until the episode ends, so a term can look healthy while its per-step value
    is already negative (or the other way round). This evaluates each term once
    more and returns the number that actually enters ``Train/mean_reward`` every
    step, which is the only scale in which the weights are comparable.

    The extra evaluation is read-only: every term in ``rewards.py`` only reads
    ``robot.data`` / the command buffer, so calling it twice in a step is safe.
    Terms that raise are skipped rather than taking the run down.
    """
    term_cfgs = getattr(manager, "_term_cfgs", None)
    if not term_cfgs:
        return None
    dt = getattr(env, "step_dt", None)
    if dt is None:
        dt = getattr(manager, "_dt", None)
    if dt is None:
        return None
    names = getattr(manager, "_term_names", None) or []
    # ``_term_cfgs`` is a list in some releases and a dict keyed by name in others;
    # iterating the dict would hand us the keys, so normalise to pairs first.
    if isinstance(term_cfgs, dict):
        pairs = [(name, term_cfgs[name]) for name in names if name in term_cfgs]
        if not pairs:
            pairs = list(term_cfgs.items())
    else:
        term_cfgs = list(term_cfgs)
        pairs = list(zip(names, term_cfgs)) if len(names) == len(term_cfgs) else [
            (getattr(term_cfg, "name", f"term{index}"), term_cfg)
            for index, term_cfg in enumerate(term_cfgs)
        ]
    contributions: dict[str, float] = {}
    for name, term_cfg in pairs:
        # Class-based terms keep state inside an instance the manager owns; calling
        # the class again would build a fresh one and report zeros.
        if isinstance(getattr(term_cfg, "func", None), type):
            continue
        try:
            value = term_cfg.func(env, **term_cfg.params)
            value = torch.as_tensor(value, dtype=torch.float32)
            contributions[name] = float((value * term_cfg.weight * dt).mean().item())
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            continue
    return contributions or None


def episode_sums(manager) -> dict[str, float] | None:
    """Mean of ``_episode_sums`` per term: the accumulated episode reward.

    This is the number tensorboard's ``Episode_Reward/*`` is built from, so
    printing it next to the per-step contributions lets one check that the two
    agree in sign and order of magnitude.
    """
    sums = getattr(manager, "_episode_sums", None)
    if not sums:
        return None
    items = getattr(sums, "items", None)
    if items is None:
        # A plain list: fall back to the manager's names, or skip the readout.
        names = getattr(manager, "_term_names", None) or []
        if len(names) != len(sums):
            return None
        items = lambda: zip(names, sums)  # noqa: E731 - local shim, one use
    values: dict[str, float] = {}
    for name, tensor in items():
        try:
            values[name] = float(tensor.mean().item())
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            continue
    return values or None


def format_terms(values: dict[str, float], unit: str) -> str:
    """One line, biggest contributors first, so the tail of the list is quiet."""
    ordered = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
    body = "  ".join(f"{name}={value:+.3f}" for name, value in ordered)
    return f"[terms] per-step ({unit}): {body}  sum={sum(values.values()):+.3f}"


def tracking_by_mode(env) -> str | None:
    """Tracking credit split by mode label.

    ``Episode_Reward/track_lin_vel_xy`` is a mean over *all* envs, and a standing
    env sits near the top of the kernel (command 0, robot nearly still -> ~2.0 of
    a 2.0 maximum). With half the envs standing, that single number can read ~1.2
    while the moving half tracks almost nothing. This line separates them, using
    the same kernel and the same sigma as the reward term, so
    ``stand=2.0 move=0.2`` is unmistakable and cannot be confused with
    "1.1 on average, so it is learning".
    """
    try:
        term = env.command_manager.get_term("base_velocity")
        modes = getattr(term, "mode_buf", None)
        if modes is None:
            return None
        sigma = float(env.cfg.rewards.track_lin_vel_xy.params.get("sigma", 0.25))
        targets = control.command_targets(env).v_body
        vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
        value = 1.0 / (1.0 + ((vel - targets) / sigma).pow(2).sum(dim=1))
        groups = (
            ("stand", modes == control.MODE_STAND),
            ("move", (modes == control.MODE_MOVE) | (modes == control.MODE_COMBINED)),
            ("rot", modes == control.MODE_ROTATE),
        )
        parts = []
        for name, mask in groups:
            share = mask.float().mean().item()
            mean = value[mask].mean().item() if bool(mask.any()) else float("nan")
            parts.append(f"{name}={mean:.2f}({share * 100:.0f}%)")
        # Per-step credit = value * weight * dt, the scale the terms line uses.
        weight = float(env.cfg.rewards.track_lin_vel_xy.weight)
        dt = float(getattr(env, "step_dt", 0.02))
        return (f"[track] {parts[0]} {parts[1]} {parts[2]}  "
                f"(1.00 raw = {weight * dt:.3f}/step; episode sums are ~1000x this)")
    except Exception:  # noqa: BLE001 - diagnostics must never break training
        return None


def attach_diagnostics(env, every: int, zero_actions: bool, per_term_every: int = 0,
                       runner_box: dict | None = None) -> None:
    """Log simulation health, optionally replacing the policy output with zeros.

    PhysX reports a divergence as a bare ``CUDA error 700`` (illegal memory
    access) with no hint about which quantity blew up, and the CUDA context is
    dead by the time it surfaces. Printing the worst-case joint speed and base
    height every iteration makes the growth visible a few seconds before the
    crash instead of after it.

    For the velocity task the line also carries ``|v_xy|`` (mean body speed -- the
    quantity the tracking kernel prices), ``stand``/``move``/``rot`` (the shares of
    each mode label; the sampler writes the label and the reward gates
    ``wheel_speed``/``zero_xy_speed`` read it), ``w*`` (the yaw-rate target the
    outer loop is asking for), and ``rew``/``neg`` (the per-step reward and the
    fraction of envs where it is negative). A stage-1 run that wanders shows up in
    ``|v_xy|`` immediately: the command is zero, so anything above ~0.05 m/s is
    drift. ``stand=0.00`` during stage 1 would mean the labels are wrong rather
    than the reward being too weak. ``neg=1.00`` is *not* an error by itself -- with no
    reward floor a bad policy legitimately earns a negative step -- but if it sits
    at 1.00 for a whole stage while ``Train/mean_reward`` does not rise, the
    command range is ahead of the policy.

    ``fall_rew`` / ``timeout_rew`` are the mean of the tensor actually handed to
    rsl_rl on the steps where the episode ends by a fall / by the time-out. They
    are the end-to-end proof that the fall penalty arrives (``fall_rew`` should
    read ≈ -10, which ``M20VelocityEnv`` adds from the termination flags) and that
    a time-out pays nothing extra (``timeout_rew`` >= 0).

    ``lr`` / ``sig`` close the loop on the adaptive schedule: ``lr`` is the rate the
    optimizer is using *right now* (rsl_rl rewrites it in place every round) and
    ``sig`` is the mean action standard deviation. A rate stuck at 1e-5 with a
    ``sig`` that has collapsed means the KL controller is chasing a target that the
    update can never satisfy; a rate climbing to 1e-2 with ``sig`` still rising means
    the controller is over-correcting. Read them next to ``Train/mean_reward``: the
    schedule is only "working" if the reward is moving.

    ``per_term_every`` adds a ``[terms] per-step`` line every that many steps: the
    contribution of each reward term as ``value * weight * dt``, i.e. the number
    that enters ``Train/mean_reward`` every step. ``Episode_Reward/*`` is the
    accumulated version (``_episode_sums``) and it grows with episode length, so
    a term whose weight is far too large can still look bounded there while its
    per-step cost is the whole positive side of the reward. This line is the one
    to read when ``Train/mean_reward`` sits at some value and the question is
    "which term is doing that". Its ``sum=`` should equal ``rew=`` on the health
    line at the same step (up to the fall penalty, which the env adds later).
    """
    unwrapped = env.unwrapped
    original_step = env.step
    state = {"n": 0}
    def step(action):
        if zero_actions:
            action = torch.zeros_like(action)
        out = original_step(action)
        state["n"] += 1
        if every > 0 and state["n"] % every == 0:
            robot = unwrapped.scene["robot"]
            joint_vel = robot.data.joint_vel
            root_z = robot.data.root_pos_w[:, 2]
            root_vel = robot.data.root_lin_vel_w.abs().max()
            # "does the robot actually stand still when told to" is otherwise
            # only visible in tensorboard hours later: |v_xy| is the residual the
            # tracking kernel prices; `stand=`/`move=`/`rot=` are the shares of
            # each *mode label* -- the sampler writes the label, and the reward
            # gates (`wheel_speed` on STAND, `zero_xy_speed` on ROTATE) read the
            # label, not re-tested floats. `w*` is the yaw-rate target the outer
            # loop is asking for, which is what `track_ang_vel_z` compares to.
            speed_xy = robot.data.root_lin_vel_b[:, :2].norm(dim=1).mean()
            extra = f"  |v_xy|={speed_xy.item():.3f}"
            command_manager = getattr(unwrapped, "command_manager", None)
            if command_manager is not None:
                try:
                    term = command_manager.get_term("base_velocity")
                except KeyError:
                    term = None
                modes = getattr(term, "mode_buf", None) if term else None
                if modes is not None:
                    extra += (
                        f"  stand={(modes == control.MODE_STAND).float().mean().item():.2f}"
                        f"  move={(modes == control.MODE_MOVE).float().mean().item():.2f}"
                        f"  rot={(modes == control.MODE_ROTATE).float().mean().item():.2f}"
                    )
                try:
                    omega_star = control.command_targets(unwrapped).omega
                    extra += f"  w*={omega_star.abs().mean().item():.2f}"
                except Exception:  # noqa: BLE001 - diagnostics must never stop a run
                    pass
            # Per-step reward as the reward manager summed it (no clamp since the
            # floor was removed). A large negative `rew` here is not a broken
            # reward -- it is the regime where the penalties outweigh the tracking
            # credit. If `neg` sits near 1.00 for a whole stage and
            # `Train/mean_reward` does not climb, the command range is ahead of
            # the policy.
            reward_buf = getattr(unwrapped, "reward_buf", None)
            if reward_buf is not None:
                extra += (
                    f"  rew={reward_buf.mean().item():+.2f}"
                    f"  neg={(reward_buf < 0).float().mean().item():.2f}"
                )
            # What the *learner* actually receives on the terminal steps. `rew` above
            # is the sum from the reward manager; `out[1]` is the tensor handed to
            # rsl_rl, after `M20VelocityEnv` added the fall penalty. A fall must show
            # `fall_rew` ≈ -10, and a time-out must show `timeout_rew` >= 0 (it pays
            # nothing extra). These two numbers are the end-to-end proof that the
            # penalty is applied once, on falls only.
            #
            # Shape differs by where this wrapper sits: the raw env returns
            # (obs, reward, terminated, truncated, extras), while
            # ``RslRlVecEnvWrapper`` returns (obs, reward, dones, infos) with the
            # time-out flag in ``infos["time_outs"]``. Handle both.
            terminated = truncated = None
            if len(out) == 5:
                terminated, truncated = out[2], out[3]
            elif len(out) == 4 and isinstance(out[3], dict):
                terminated = out[2]
                truncated = out[3].get("time_outs")
                if truncated is None:
                    truncated = torch.zeros_like(terminated)
            if terminated is not None:
                fell = terminated & ~truncated
                if bool(fell.any()):
                    extra += f"  fall_rew={out[1][fell].mean().item():+7.2f}"
                if bool(truncated.any()):
                    extra += f"  timeout_rew={out[1][truncated].mean().item():+6.2f}"
            # Optimizer state of the stage that is currently running. `lr` is
            # rewritten in place by the adaptive schedule, so this is the rate the
            # next round will actually use; `sig` is the mean action std (entropy is
            # 16 * (1.419 + ln sig)). Both are read defensively -- a runner that has
            # not been created yet (first steps) simply prints nothing.
            runner = (runner_box or {}).get("runner")
            if runner is not None:
                try:
                    lr = getattr(runner.alg, "learning_rate", None)
                    if lr is not None:
                        extra += f"  lr={float(lr):.1e}"
                    std = getattr(runner.alg.policy, "std", None)
                    if std is not None:
                        extra += f"  sig={std.detach().mean().item():.2f}"
                except Exception:  # noqa: BLE001 - diagnostics must never stop a run
                    pass
            print(
                f"[health] step={state['n']:8d}"
                f"  max|qdot|={joint_vel.abs().max().item():10.2f}"
                f"  max|v|={root_vel.item():8.2f}"
                f"  z=[{root_z.min().item():.3f},{root_z.max().item():.3f}]"
                f"{extra}"
                f"  finite={torch.isfinite(joint_vel).all().item()}",
                flush=True,
            )
            # Which term is actually paying for the number above. The health line
            # says `rew=-3.4`; this says it is `joint_deviation=-1.9` and
            # `action_rate=-1.2`, without waiting for tensorboard -- and in the
            # only scale where weights are comparable (the dt-scaled episode sums
            # grow with episode length and hide the per-step cost).
            if per_term_every > 0 and state["n"] % per_term_every == 0:
                manager = getattr(unwrapped, "reward_manager", None)
                if manager is not None:
                    steps = reward_term_contributions(unwrapped, manager)
                    if steps:
                        print(format_terms(steps, "value*weight*dt"), flush=True)
                    totals = episode_sums(manager)
                    if totals:
                        body = "  ".join(
                            f"{name}={value:+.1f}" for name, value in totals.items())
                        print(f"[terms] episode sums: {body}", flush=True)
                    split = tracking_by_mode(unwrapped)
                    if split:
                        print(split, flush=True)
        return out

    env.step = step
