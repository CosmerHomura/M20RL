"""Print the prim hierarchy and physics schemas of a USD file.

Useful when an asset fails to load with "Please ensure that the prim has
'USD ArticulationRootAPI' applied": the output shows the default prim, every
prim path, and which physics APIs each one carries.

    python -m isaaclab_m20.scripts.inspect_usd --headless
    python -m isaaclab_m20.scripts.inspect_usd --headless --usd model/usd/configuration/M20_robot.usd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Inspect a USD file.")
parser.add_argument("--usd", type=str, default=str(REPO_ROOT / "model" / "usd" / "M20.usd"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Everything below needs the Omniverse app running.
# --------------------------------------------------------------------------- #
from pxr import Usd, UsdPhysics  # noqa: E402


def describe(prim) -> str:
    tags = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        tags.append("ArticulationRoot")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        tags.append("RigidBody")
    type_name = prim.GetTypeName()
    if type_name == "PhysicsFixedJoint":
        tags.append("FixedJoint")
    elif type_name in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
        tags.append(type_name.replace("Physics", ""))
    return ",".join(tags)


def main() -> None:
    usd_path = Path(args_cli.usd).expanduser()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    default_prim = stage.GetDefaultPrim()
    print(f"[USD] {usd_path}")
    print(f"[USD] default prim: {default_prim.GetPath() if default_prim else None}")
    print(f"[USD] root layer:   {stage.GetRootLayer().identifier}")

    counts = {}
    for prim in stage.Traverse():
        tags = describe(prim)
        if tags:
            counts[tags] = counts.get(tags, 0) + 1
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]  {tags}")

    print("\n[USD] summary:")
    for tags, count in sorted(counts.items()):
        print(f"  {tags or '(no physics API)'}: {count}")
    print(
        "\n[USD] The spawned prim must carry ArticulationRoot. If it is on a child "
        "(e.g. /M20/base_link), set M20_CFG.prim_path to that child path."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
