"""The two log-directory rules: continue a stage, or start a new one.

Both rules exist because the workflow is "train a stage, look at it, decide":

* Re-running a stage continues it. The newest ``model_*.pt`` in its directory is
  loaded and exactly ``--iterations`` iterations run. Nothing is subtracted,
  nothing is deleted, and the script never decides a stage has had enough.
  ``--overwrite`` renames the old attempt to ``<stage>.bak-<stamp>`` instead.
* A continued stage gets the environment it stopped with: the command ramp is put
  back where it was (see ``curriculum``), and the current cfg is compared against
  the ``env.yaml`` written by the first attempt, with any difference printed.

Everything here is filesystem and YAML work; no Isaac Sim objects.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from isaaclab_m20.training.stages import STAGES, stage_label


class LogDumper(yaml.SafeDumper):
    """Dumper for the params log: stringify values YAML has no type for."""


def _represent_unknown(dumper, data):
    return dumper.represent_str(str(data))


# SceneEntityCfg defaults to joint_ids=slice(None), which plain SafeDumper rejects.
LogDumper.add_multi_representer(slice, _represent_unknown)
LogDumper.add_multi_representer(object, _represent_unknown)


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest ``model_<iteration>.pt`` under ``directory``, by iteration number.

    Sorted numerically, not lexicographically: the runner numbers checkpoints by
    the global iteration count, so ``model_1200.pt`` sorts *before*
    ``model_700.pt`` as a string and a plain ``sorted(...)[-1]`` would resume from
    a checkpoint 500 iterations old.

    Searched recursively because rsl_rl puts them in ``<log_dir>/checkpoints`` in
    some versions and directly in ``log_dir`` in others.
    """
    candidates = []
    for path in directory.rglob("model_*.pt"):
        try:
            candidates.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            continue
    candidates.sort()
    return candidates[-1][1] if candidates else None


def latest_run_dir(experiment_dir: Path) -> Path | None:
    """Newest run directory under ``logs/rsl_rl/<experiment>/``."""
    if not experiment_dir.is_dir():
        return None
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def checkpoint_iteration(path: Path) -> int:
    """Iteration count encoded in a ``model_<n>.pt`` name, 0 if unreadable."""
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def stage_baseline(log_root_path: Path, stage_index: int) -> Path | None:
    """Newest checkpoint of the stage before ``stage_index``, i.e. its baseline.

    rsl_rl's counter is cumulative, so the number in this stage's filenames only
    makes sense next to the tail number of the previous stage's last model: stage
    2 running from 900 to 3150 starts at 900.
    """
    if stage_index <= 0:
        return None
    return latest_checkpoint(
        log_root_path / stage_label(stage_index, STAGES[stage_index - 1]))


def stage_relative_iteration(checkpoint: Path, log_root_path: Path,
                             stage_index: int) -> int:
    """How many iterations into its own stage ``checkpoint`` was written.

    Used only to put the command ramp back where it was: a stage interrupted at
    iteration 400 of its 1000 must resume on the range it was training at, and one
    interrupted at 400 of its 150 warmup must resume mid-ramp. It is *not* used to
    decide how much longer the stage has to run -- that is the operator's call.
    """
    baseline = stage_baseline(log_root_path, stage_index)
    base = checkpoint_iteration(baseline) if baseline is not None else 0
    return max(checkpoint_iteration(checkpoint) - base, 0)


def _plain(value):
    """Tuples to lists, recursively, so YAML round-trips compare equal."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _flatten(node, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        flat: dict[str, object] = {}
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): _plain(node)}


def env_differences(previous_path: Path, env_cfg, limit: int = 40) -> list[str] | None:
    """Leaves where the current env cfg differs from the saved ``env.yaml``.

    Dumped through the same ``LogDumper`` on both sides so sentinels and tuples
    serialise identically; returns ``None`` when the old file cannot be read.

    The command range is skipped on purpose: ``CommandSchedule`` rewrites it every
    few steps and the pre-``gym.make`` setup points it at the resumed position, so
    it is *expected* to differ from the value stored when the stage started. The
    range actually in force is printed separately ("First rollout range").
    """
    try:
        previous = yaml.safe_load(previous_path.read_text())
        current = yaml.safe_load(yaml.dump(env_cfg.to_dict(), Dumper=LogDumper,
                                           sort_keys=False))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(previous, dict):
        return None
    old_flat = _flatten(previous)
    new_flat = _flatten(current or {})
    ignored = ("commands.base_velocity.ranges.",)
    diffs = []
    for key in sorted(set(old_flat) | set(new_flat)):
        if key.startswith(ignored):
            continue
        before = old_flat.get(key, "<absent>")
        after = new_flat.get(key, "<absent>")
        if before == after:
            continue
        diffs.append(f"{key}: {before!r} -> {after!r}")
        if len(diffs) >= limit:
            break
    return diffs


def latest_checkpoint_across_runs(experiment_dir: Path, stage_index: int) -> Path | None:
    """Newest checkpoint for course stage ``stage_index`` in *any* run.

    Used by ``--new_run``: the new directory is empty by construction, so a stage
    other than the first has to borrow the previous stage from an earlier run
    instead of silently starting from a random policy.
    """
    if not experiment_dir.is_dir() or stage_index <= 0:
        return None
    label = stage_label(stage_index, STAGES[stage_index - 1])
    best: tuple[float, Path] | None = None
    for run_dir in experiment_dir.iterdir():
        stage_dir = run_dir / label
        if not run_dir.is_dir() or not stage_dir.is_dir():
            continue
        found = latest_checkpoint(stage_dir)
        if found is None:
            continue
        mtime = found.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, found)
    return None if best is None else best[1]


def rename_final_checkpoint(stage_dir: Path, last_iteration: int | None) -> Path | None:
    """Rename the stage's final checkpoint to the number of iterations it completed.

    rsl_rl names the checkpoints after the loop *index* and its final save uses
    that index too, so ``--iterations 300`` ends with ``model_299.pt``: the file
    name is one less than the number of iterations that actually ran. The loader
    does not care (the counter lives inside the checkpoint), but the directory
    listing then lies about the budget.

    Only the exact off-by-one is corrected. If the newest file is some other
    number -- a mid-training save, because the harness saved nothing at the end --
    it is left untouched rather than relabelled with an iteration count it does
    not contain.
    """
    newest = latest_checkpoint(stage_dir)
    if newest is None or last_iteration is None:
        return newest
    completed = last_iteration + 1
    try:
        number = int(newest.stem.split("_")[-1])
    except ValueError:
        return newest
    if number + 1 != completed:
        return newest
    target = newest.with_name(f"model_{completed}.pt")
    if target.exists():
        return newest
    newest.rename(target)
    print(f"[INFO] final checkpoint renamed to {target.name} "
          f"({completed} iterations completed)")
    return target
