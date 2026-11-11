"""Command-line definitions, kept apart from the code they configure.

``scripts/`` starts Isaac Sim and calls into ``training/`` or ``tasks/``;
``cli/`` only declares arguments and parses them. That split exists because the
entry-point scripts have a hard ordering constraint -- the Omniverse app must be
running before ``isaaclab_m20.tasks`` can be imported -- and every argument added
to a script was making that constraint harder to see.

Modules:

``train_args``
    Every ``train.py`` flag, including the per-stage iteration overrides.

Nothing here imports Isaac Lab: parsing is pure ``argparse``, so the flags can be
inspected (and unit-tested) without starting a simulator.
"""
