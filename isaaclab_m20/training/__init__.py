"""Training-side logic: course definition, curriculum scheduling, checkpoints, diagnostics.

Nothing here creates an app, parses arguments or touches Isaac Sim's launcher --
that is ``isaaclab_m20.cli``'s job, and the thin entry points in
``isaaclab_m20.scripts`` glue the two together. The split exists because the
training logic is what changes week to week (stages, budgets, log rules) while
the entry points should stay boring.

Import order matters in this project: ``AppLauncher`` must start the Omniverse
app *before* anything imports ``isaaclab.envs``/``isaaclab_m20.tasks``. So
``stages`` imports nothing but ``dataclasses`` (it is the one module ``cli``
needs), ``checkpoints`` adds only ``yaml``, and ``curriculum``/``diagnostics``
import the task package at module level -- they are imported from the entry
points *after* the app is up, and must not be imported earlier.
"""
