"""Train the M20 course: parse the flags, start Isaac Sim, hand over to ``training``.

This file is deliberately thin. It exists to enforce one ordering constraint --
Isaac Sim's app has to be running before anything imports ``isaaclab_m20.tasks``
-- and to wire the pieces together:

    cli/train_args.py     the flags
    training/stages.py    what each stage asks for
    training/curriculum.py  how the range moves inside env.step
    training/checkpoints.py where runs live and how a stage continues
    training/diagnostics.py the [health]/[terms]/[track] lines

Course at a glance (see ``training/stages.py`` for the ranges and the reasoning):

    1 stand 900 -> 2 forward/back 700 -> 3 all directions 1200 -> 4 agile 1400

Typical use::

    # the whole course, ~4200 iterations
    python -m isaaclab_m20.scripts.train --run_name m20_v8

    # one stage, look at it in the simulator before spending the next one
    python -m isaaclab_m20.scripts.train --stage 1 --iterations 900 --run_name m20_v8
    python -m isaaclab_m20.scripts.play  --stage 1
    python -m isaaclab_m20.scripts.eval  --stage 1
    python -m isaaclab_m20.scripts.train --stage 2 --iterations 1200
"""
from __future__ import annotations

import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_m20.cli.train_args import parse

args_cli = parse()

from isaaclab.app import AppLauncher  # noqa: E402 - must come after parse(): --headless lives here

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Everything below needs the Omniverse app running.
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_m20.tasks  # noqa: E402,F401 - registers the M20 task ids
from isaaclab_m20.tasks.m20_env_cfg import fit_viewer_to_grid  # noqa: E402
from isaaclab_m20.training.checkpoints import (  # noqa: E402
    LogDumper,
    checkpoint_iteration,
    env_differences,
    latest_checkpoint,
    latest_checkpoint_across_runs,
    latest_run_dir,
    rename_final_checkpoint,
    stage_baseline,
    stage_relative_iteration,
)
from isaaclab_m20.training.curriculum import (  # noqa: E402
    CommandSchedule,
    attach_dr_ramp,
    attach_stage_schedule,
)
from isaaclab_m20.training.diagnostics import attach_diagnostics  # noqa: E402
from isaaclab_m20.training.stages import (  # noqa: E402
    STAGES,
    Stage,
    _LAST_STAGE,
    interpolate_ranges,
    select_stages,
    stage_label,
)


def main() -> None:
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed
    fit_viewer_to_grid(env_cfg)

    agent_module, agent_class = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(importlib.import_module(agent_module), agent_class)()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    if args_cli.learning_rate is not None:
        # The rate the schedule *starts* from. With adaptive it is only the
        # starting point (the KL controller moves it every round); with
        # --fixed_lr it is the rate that stays for the whole stage.
        agent_cfg.algorithm.learning_rate = float(args_cli.learning_rate)
        print(f"[INFO] learning_rate: {agent_cfg.algorithm.learning_rate:g} "
              f"(cfg default is {3.0e-4:g}; adaptive moves it, --fixed_lr pins it)")
    if args_cli.desired_kl is not None:
        # Only meaningful for the adaptive schedule, and only in the sense the
        # controller uses it: a *sum over the 16 action dims*.
        agent_cfg.algorithm.desired_kl = float(args_cli.desired_kl)
        print(f"[INFO] desired_kl: {agent_cfg.algorithm.desired_kl:g} "
              f"(summed over {16} action dims; cfg default is 0.01)")
    if args_cli.fixed_lr:
        # The cfg ships with the plan's adaptive schedule; pin the rate here so
        # the two modes can be compared without editing the config file.
        agent_cfg.algorithm.schedule = "fixed"
        print(f"[INFO] Schedule: fixed at lr={agent_cfg.algorithm.learning_rate:g}"
              f" (cfg default is 'adaptive'; --fixed_lr overrides it)")
    else:
        print(f"[INFO] Schedule: {agent_cfg.algorithm.schedule}"
              f" (start lr={agent_cfg.algorithm.learning_rate:g}; pass --fixed_lr to pin it)")

    # The velocity task carries a command term; the standing task does not.
    is_velocity_task = hasattr(env_cfg, "commands")
    per_stage = {index: getattr(args_cli, f"iters_stage{index + 1}") for index in range(len(STAGES))}
    stages = select_stages(args_cli.stage, args_cli.stage_scale, args_cli.iterations, per_stage,
                           args_cli.warmup_frac, args_cli.cmd_scale) \
        if is_velocity_task else []

    if is_velocity_task:
        # Must happen before gym.make: friction/mass/CoM are startup terms.
        from isaaclab_m20.tasks.m20_velocity_env_cfg import set_domain_randomization
        enabled = set_domain_randomization(env_cfg, args_cli.dr == "full")
        print(f"[INFO] Domain randomization: "
              f"{'full' if enabled else 'off (nominal robot, no pushes)'}")
        # One stage per invocation: --iterations is the way to set its length;
        # --iters_stageN only matters for --stage all.
        if args_cli.stage != "all" and len(stages) == 1:
            running = stages[0][0] + 1
            ignored = [n for n, value in per_stage.items()
                       if value is not None and n + 1 != running]
            if ignored:
                flags = ", ".join(f"--iters_stage{n + 1}" for n in sorted(ignored))
                print(f"[WARN] --stage {running} ignores {flags} (other stages are not "
                      f"running); use --iterations for the length of this stage.")
        agent_cfg.max_iterations = sum(stage.iterations for _, stage in stages)
    else:
        agent_cfg.max_iterations = args_cli.iterations or agent_cfg.max_iterations

    if stages:
        total = sum(stage.iterations for _, stage in stages)
        print(f"[INFO] Course plan: {len(stages)} stage(s), {total} iterations "
              f"({env_cfg.scene.num_envs} envs x {agent_cfg.num_steps_per_env} steps/env)")
        if args_cli.cmd_scale != 1.0:
            print(f"[INFO] --cmd_scale {args_cli.cmd_scale:g}: every command range below "
                  f"is scaled by it (the scheduled ranges come from the stage table)")
        for number, stage in stages:
            sampling = []
            if stage.zero_prob is not None:
                sampling.append(f"zero_prob {stage.zero_prob:.2f}")
            if stage.resampling is not None:
                sampling.append(f"hold {stage.resampling[0]:.1f}-{stage.resampling[1]:.1f}s")
            extra = f"  [{', '.join(sampling)}]" if sampling else ""
            print(f"    {stage_label(number + 1, stage):<24} {stage.iterations:5d} it  "
                  f"vx=+-{stage.ranges_to[0]:.2f} vy=+-{stage.ranges_to[1]:.2f} "
                  f"wz=+-{stage.ranges_to[2]:.2f}  warmup {stage.warmup}{extra}")

    experiment_dir = Path(args_cli.log_root, agent_cfg.experiment_name)
    if args_cli.run_name:
        run_name = args_cli.run_name
    elif args_cli.new_run:
        # Explicit opt-in to a second directory: the earlier attempt of the same
        # stage stays on disk for comparison instead of being overwritten.
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("[INFO] --new_run: starting a fresh run directory")
    elif is_velocity_task and args_cli.stage != "all":
        # One stage per invocation: stay inside the run the previous stage
        # wrote to. A fresh timestamped directory would make "--stage 2" search
        # for stage 1's checkpoint in an empty folder and silently start over.
        previous_run = latest_run_dir(experiment_dir)
        run_name = previous_run.name if previous_run else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if previous_run is not None:
            print(f"[INFO] Continuing run {previous_run} "
                  f"(re-running a stage continues it from its newest checkpoint; "
                  f"--overwrite moves that attempt aside and retrains, --new_run "
                  f"starts a separate directory)")
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = experiment_dir / run_name
    print(f"[INFO] Logging to {log_root_path}")

    if stages:
        # The commands of the very first reset are drawn from whatever range the
        # config carries, and that happens inside gym.make -- before the schedule
        # can intervene. Point it at the range the first selected stage trains at
        # *now*: ranges_from for a fresh stage, the interrupted position for a
        # stage being continued. Otherwise a resumed stage starts its first 2-4 s
        # on the wrong distribution, which is exactly "starting over".
        first_number, first_stage = stages[0]
        first_done = 0
        if args_cli.stage != "all":
            candidate = latest_checkpoint(
                log_root_path / stage_label(first_number + 1, first_stage))
            if candidate is not None:
                first_done = stage_relative_iteration(
                    candidate, log_root_path, first_number)
        first_vx, first_vy, first_wz = interpolate_ranges(
            first_stage, first_done, first_stage.warmup)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-first_vx, first_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-first_vy, first_vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-first_wz, first_wz)
        print(f"[INFO] First rollout range: vx=+-{first_vx:.3f} vy=+-{first_vy:.3f} "
              f"wz=+-{first_wz:.3f}" +
              (f" (stage iteration {first_done})" if first_done else ""))

    print(f"[INFO] Environment: {args_cli.task}  num_envs={env_cfg.scene.num_envs}")
    print_dict(agent_cfg.to_dict(), nesting=0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video of the first iteration.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Action space: {env.action_space}  Observation space: {env.observation_space}")
    if args_cli.zero_actions:
        print("[INFO] --zero_actions: holding the default pose, actions are forced to 0.")
    # The runner is created per stage, after this wrapper; a one-slot box lets the
    # health line read the *current* optimizer state of whichever stage is running.
    runner_box: dict[str, object] = {}
    attach_diagnostics(
        env, args_cli.health_every, args_cli.zero_actions, args_cli.per_term_every,
        runner_box)

    schedule = None
    if is_velocity_task:
        manager = env.unwrapped.command_manager
        schedule = CommandSchedule(manager.get_term("base_velocity"), agent_cfg.num_steps_per_env)
        attach_stage_schedule(env, schedule)
    else:
        attach_dr_ramp(env, agent_cfg.max_iterations, agent_cfg.num_steps_per_env,
                       args_cli.dr_ramp)

    planned: list[tuple[str, int, Stage | None]] = []
    if stages:
        for stage_index, stage in stages:
            planned.append((stage_label(stage_index + 1, stage), stage.iterations, stage))
    else:
        planned.append(("standing", agent_cfg.max_iterations, None))

    for position, (label, iterations, stage) in enumerate(planned):
        stage_dir = log_root_path if stage is None else log_root_path / label
        stage_index = -1
        if stage is not None:
            stage_index = next(i for i, s in enumerate(STAGES) if s.name == stage.name)
        resume_path: Path | None = None
        resuming_own_stage = False
        resumed_iterations = 0

        if stage_dir.exists() and any(stage_dir.iterdir()):
            if args_cli.overwrite:
                # Moved, not deleted: a mistaken retrain stays recoverable.
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = stage_dir.with_name(f"{stage_dir.name}.bak-{stamp}")
                stage_dir.rename(backup)
                print(f"[INFO] {label}: --overwrite, previous attempt moved to "
                      f"{backup.name} (nothing was deleted)")
            else:
                # Re-running a stage continues it. That is what "keep training"
                # has to mean: resuming from the previous stage's checkpoint
                # instead would throw this stage's work away.
                resume_path = latest_checkpoint(stage_dir)
                resuming_own_stage = resume_path is not None
                if resuming_own_stage:
                    print(f"[INFO] {label}: earlier attempt found -- continuing it from "
                          f"{resume_path.name} (use --overwrite to retrain from scratch)")
                    resumed_iterations = stage_relative_iteration(
                        resume_path, log_root_path, stage_index)
                    # Same environment as when it stopped: if anything in the cfg
                    # changed (command range, rewards, DR, env count), resuming is
                    # not the same run and the operator has to know.
                    previous_env = stage_dir / "env.yaml"
                    if previous_env.exists():
                        diff = env_differences(previous_env, env_cfg)
                        if diff is None:
                            print(f"[WARN] {label}: could not read {previous_env.name}; "
                                  f"cannot confirm the environment is unchanged")
                        elif diff:
                            print(f"[WARN] {label}: env cfg differs from the one this stage "
                                  f"was trained with ({previous_env.name}):")
                            for line in diff[:10]:
                                print(f"        {line}")
                            if len(diff) > 10:
                                print(f"        ... {len(diff) - 10} more difference(s)")
                            print("        Resuming on a different environment is not the same "
                                  "run; pass --overwrite to start the stage cleanly instead.")
                        else:
                            print(f"[INFO] {label}: env cfg matches {previous_env.name}")
                else:
                    print(f"[INFO] {label}: directory holds logs but no checkpoint, so "
                          f"this stage starts over")
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Continuing a stage keeps the first attempt's env.yaml/agent.yaml as the
        # record of how it was trained; this run's config goes beside them.
        if resuming_own_stage:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            env_yaml = stage_dir / f"env.resume-{stamp}.yaml"
            agent_yaml = stage_dir / f"agent.resume-{stamp}.yaml"
        else:
            env_yaml = stage_dir / "env.yaml"
            agent_yaml = stage_dir / "agent.yaml"
        with open(env_yaml, "w") as f:
            yaml.dump(env_cfg.to_dict(), f, Dumper=LogDumper, sort_keys=False)
        with open(agent_yaml, "w") as f:
            yaml.dump(agent_cfg.to_dict(), f, Dumper=LogDumper, sort_keys=False)

        if resume_path is None and stage is not None:
            # Fresh stage: pick up where the previous stage of this run left off,
            # so "--stage 2" after "--stage 1" really does continue rather than
            # start from a random policy.
            if stage_index > 0:
                previous = log_root_path / stage_label(stage_index, STAGES[stage_index - 1])
                resume_path = latest_checkpoint(previous)
                if resume_path is None and args_cli.new_run:
                    # A --new_run directory is empty by construction, so seed it
                    # from the previous stage of whichever run has one.
                    resume_path = latest_checkpoint_across_runs(experiment_dir, stage_index)
                    if resume_path is not None:
                        print(f"[INFO] --new_run: seeding {stage.name} from "
                              f"{resume_path} (previous stage, earlier run)")
                if resume_path is not None:
                    print(f"[INFO] Stage {stage.name}: continuing from the previous "
                          f"stage's checkpoint {resume_path}")
                else:
                    print(f"[WARN] Stage {stage.name}: no checkpoint from "
                          f"{previous} and no earlier run to borrow from -- starting "
                          f"from a fresh policy. Run --stage {stage_index} first.")

        if resume_path is not None and not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        # How much to run now. No arithmetic against what the stage already did:
        # how long a stage should be trained is the operator's decision, so the
        # default is simply the stage budget and --iterations replaces it.
        if args_cli.iterations is not None:
            run_iterations = int(args_cli.iterations)
        else:
            run_iterations = iterations

        print(f"\n[INFO] ===== stage {label}: {run_iterations} iterations =====")
        if resume_path is not None:
            baseline = stage_baseline(log_root_path, stage_index)
            source = (f"baseline {baseline.name}" if baseline is not None
                      else "no previous stage, absolute baseline 0")
            print(f"[INFO] Continuing from {resume_path} "
                  f"(global iteration {checkpoint_iteration(resume_path)}; {source}"
                  + (f"; stage iteration {resumed_iterations}" if resuming_own_stage else "")
                  + ")")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(stage_dir),
                                device=agent_cfg.device)
        runner_box["runner"] = runner
        if resume_path is not None:
            runner.load(str(resume_path))

        if schedule is not None and stage is not None:
            # Continuing a stage resumes its ramp where it stopped (same command
            # range as the last iteration it trained), so the env looks exactly
            # like it did at the end of the previous attempt. The ramp length is
            # this stage's effective one (--warmup_frac / --stage_scale applied),
            # and ``resumed_iterations`` is measured on the same ramp, so a stage
            # interrupted at 900 of a 2000-iteration ramp picks up at 45%.
            schedule.start_stage(
                stage,
                warmup=stage.warmup,
                resume_iteration=resumed_iterations)

        start = time.time()
        runner.learn(num_learning_iterations=run_iterations, init_at_random_ep_len=True)
        print(f"[INFO] stage {label} finished in {(time.time() - start) / 60:.1f} min")

        # Stage N+1 continues from the checkpoint this stage ended on. rsl_rl
        # names its last save after the loop index, one below the iteration
        # count, so fix the name before it becomes anything's starting point.
        found = rename_final_checkpoint(
            stage_dir, getattr(runner, "current_learning_iteration", None))
        if found is None:
            raise RuntimeError(f"stage {label} produced no checkpoint in {stage_dir}")
        resume_path = found

        if stage is not None:
            number = next(i for i, s in enumerate(STAGES) if s.name == stage.name) + 1
            print(f"\n[INFO] Stage {label} done -- checkpoint {found}")
            print("[INFO] Look at it in the simulator before spending the next stage:")
            print(f"    python -m isaaclab_m20.scripts.play --stage {number}"
                  f"    # = --checkpoint '{found}'")
            print(f"    python -m isaaclab_m20.scripts.eval --stage {number}")
            if number < _LAST_STAGE:
                print(f"    then: python -m isaaclab_m20.scripts.train --stage {number + 1}"
                      f"   (continues from this checkpoint)")
            else:
                print("[INFO] That was the last stage of the course.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
