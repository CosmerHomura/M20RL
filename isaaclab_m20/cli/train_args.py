"""Every ``train.py`` flag in one function.

Kept separate from ``scripts/train.py`` because that script has to start the
Isaac Sim app *between* parsing and importing the task modules; a 1300-line file
where the parser sits in the middle of a hundred other things hides that ordering.
The parser here only touches ``argparse`` and the course table, so it can be built
and inspected without an app.

The per-stage iteration flags (``--iters_stage1..N``) are generated from
``STAGES``, so adding a stage to the course adds its flag automatically instead
of silently leaving it out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_m20.training.stages import STAGES, _LAST_STAGE

REPO_ROOT = Path(__file__).resolve().parents[2]

VELOCITY_TASK = "Isaac-M20-Wheeled-Velocity-v0"
STANDING_TASK = "Isaac-M20-Wheeled-v0"


def build_parser() -> argparse.ArgumentParser:
    """The full ``train.py`` argument set, plus the AppLauncher's own flags."""
    parser = argparse.ArgumentParser(description="Train M20 wheeled locomotion with rsl_rl.")
    parser.add_argument("--task", type=str, default=VELOCITY_TASK)
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override the number of environments. Both tasks default to 4096, "
                             "which a 24 GB card runs comfortably; drop it (e.g. 128) only for a "
                             "smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=str, default="all",
                        help=f"Velocity task: which course stage to run "
                             f"('1'..'{_LAST_STAGE}' or 'all'). "
                             "A stage that already has a run is continued from its newest "
                             "checkpoint; a new stage continues from the previous one.")
    parser.add_argument("--stage_scale", type=float, default=1.0,
                        help="Multiply every stage's iteration budget (e.g. 0.05 for a smoke test).")
    parser.add_argument("--cmd_scale", type=float, default=1.0,
                        help="Multiply every stage's command range on all three axes (e.g. 0.7 "
                             "makes stage 2 ask for +-1.75 m/s instead of +-2.5). Budget, "
                             "zero-command share and hold time are untouched. This is the "
                             "'the policy cannot hold that speed yet' knob -- --stage_scale only "
                             "shrinks the number of iterations.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="How many iterations to run now. Plain and literal: it is the "
                             "same number whether the stage is new or being continued, and "
                             "the script never decides on its own that a stage has had enough. "
                             "--iters_stageN is only for --stage all, "
                             "where several stages run back to back with different budgets.")
    parser.add_argument("--warmup_frac", type=float, default=None,
                        help="Length of the command ramp as a fraction of this stage's budget "
                             "(0.5 = the first half grows linearly, the rest samples across the "
                             "full range). Default: the stage's own warmup, capped at half the "
                             "budget. Only stages that declare a warmup are affected.")
    for _number, _course_stage in enumerate(STAGES, start=1):
        parser.add_argument(
            f"--iters_stage{_number}", type=int, default=None,
            help=f"Iterations for stage {_number} ({_course_stage.name}) inside a --stage all "
                 f"run; default {_course_stage.iterations} x --stage_scale. When you run that "
                 f"stage alone, --iterations does the same job more directly.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Sub-directory inside logs/rsl_rl/<experiment>; stages get their own "
                             "sub-sub-directory inside it. Without it a single --stage reuses the "
                             "newest run of that experiment.")
    parser.add_argument("--new_run", action="store_true",
                        help="Start a new timestamped run directory instead of reusing the newest "
                             "one. Use this to keep an earlier attempt of the same stage for "
                             "comparison.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain a stage from scratch instead of continuing it. The previous "
                             "attempt is RENAMED to <stage>.bak-<timestamp>, never deleted, so a "
                             "mistaken retrain can be undone. Without this flag, re-running a stage "
                             "continues it: newest checkpoint loaded, command ramp put back where "
                             "it stopped, current cfg compared against the stage's env.yaml.")
    parser.add_argument("--log_root", type=str, default=str(REPO_ROOT / "logs" / "rsl_rl"))
    parser.add_argument("--video", action="store_true", help="Record a video of the first iteration.")
    parser.add_argument("--video_length", type=int, default=200, help="Video length in policy steps.")
    parser.add_argument("--video_folder", type=str, default=str(REPO_ROOT / "videos"))
    parser.add_argument(
        "--zero_actions", action="store_true",
        help="Hold the default pose instead of training. Isolation test: if this still "
             "crashes, the fault is in the model/sim, not in the policy.")
    parser.add_argument(
        "--health_every", type=int, default=24,
        help="Print sim-health stats every N env steps (0 disables).")
    parser.add_argument(
        "--per_term_every", type=int, default=200,
        help="Print the per-step reward decomposition (value*weight*dt for every term, "
             "plus the accumulated episode sums) every N env steps. This is the only "
             "scale in which the reward weights are comparable; 0 disables.")
    parser.add_argument(
        "--dr", type=str, default="off", choices=("off", "full"),
        help="Velocity task only: 'off' trains on the nominal robot (flat ground, "
             "friction 1.0, no payload spread, no pushes) -- the first version's setting. "
             "'full' enables the friction/mass/CoM ranges and the shoves.")
    parser.add_argument(
        "--dr_ramp", type=float, default=0.4,
        help="Standing task only (no command term): fraction of training over which the "
             "domain randomization ramps from 0 to full.")
    parser.add_argument(
        "--learning_rate", "--lr", type=float, default=None, dest="learning_rate",
        help="Starting learning rate for PPO (cfg default 3e-4). With the default adaptive "
             "schedule this is where the KL controller starts, not where it stays; with "
             "--fixed_lr it is the rate that holds for the whole stage. Only the rate "
             "changes -- budgets, command ranges and rewards are untouched.")
    parser.add_argument(
        "--fixed_lr", action="store_true",
        help="Pin the learning rate at ``learning_rate`` (schedule='fixed') instead of "
             "the cfg's adaptive-in-KL schedule. Adaptive is the default because it is "
             "the plan's setting, but it also swings Adam's step size (lr moved between "
             "~0 and ~2e-3 in m20_v5); with --fixed_lr the rate stays at 3e-4 and a "
             "stage needs a longer budget before it plateaus.")
    parser.add_argument(
        "--desired_kl", type=float, default=None,
        help="Override the adaptive schedule's KL target (cfg default 0.01). rsl_rl "
             "compares this against the KL *summed over the 16 action dims*, so 0.01 is "
             "0.0006 per dim -- tight enough that a 40-step round exceeds it even at "
             "lr=1e-5, which pins the rate to the 1e-5 floor. 0.05-0.08 is the usable "
             "range for this network. Ignored with --fixed_lr.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the flags and apply the two post-processing rules argparse cannot."""
    args = build_parser().parse_args(argv)
    if args.video:
        # Recording needs the camera pipeline, which the launcher only builds when
        # asked before the app starts.
        args.enable_cameras = True
    return args
