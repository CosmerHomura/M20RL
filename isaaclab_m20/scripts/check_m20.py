"""Inspect the M20 USD that the training task loads, and regenerate it if needed.

The task loads ``model/M20_urdf.usd``, imported once from the official URDF::

    python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf
    python -m isaaclab_m20.scripts.check_m20 --headless

Why not the bundled USD: ``model/usd/M20.usd`` is a layer stack whose
``configuration/M20_physics.usd`` sublayers a 17 MB ``M20_base.usd``. If that
sublayer is missing or truncated (it is not tracked by git), the stage composes
to a single empty prim, the variant sets select nothing, and the spawner dies
with "Failed to find an articulation". The URDF import yields the same 16 joint
names and the same default pose, so it is the reliable path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Check or regenerate the M20 USD.")
parser.add_argument("--usd", type=str, default=None, help="USD to inspect. Defaults to model/M20_urdf.usd.")
parser.add_argument(
    "--convert", choices=["urdf"], default=None,
    help="Re-import model/urdf/M20.urdf into model/M20_urdf.usd first.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# Everything below needs the Omniverse app running.
# --------------------------------------------------------------------------- #
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils import configclass

from isaaclab_m20.assets import M20_CFG, M20_USD_PATH, M20_USD_SOURCE


def convert_urdf() -> str:
    """Import model/urdf/M20.urdf and return the generated USD path."""
    cfg = UrdfConverterCfg(
        asset_path=str(REPO_ROOT / "model" / "urdf" / "M20.urdf"),
        usd_dir=str(REPO_ROOT / "model"),
        usd_file_name="M20_urdf.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        # The URDF declares front/rear camera and lidar links as bare
        # <link name="..."/> with no inertial, visual or collision. Left alone
        # they become four rigid bodies with zero mass and zero inertia in the
        # articulation -- a singular mass matrix in the PhysX GPU solver, and an
        # unresolved /visuals/<link> reference in the composed stage. Merging
        # fixed joints folds them into base_link and removes both problems.
        merge_fixed_joints=True,
        self_collision=False,
        collision_from_visuals=False,
        # IsaacLab actuators write the real gains at runtime; keep the USD
        # drives present but at zero so nothing fights the implicit actuators.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[INFO] Wrote {converter.usd_path}")
    return converter.usd_path


def check(usd_path: str) -> None:
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            f"USD not found: {usd_path}\n"
            "Generate it from the official URDF with:\n"
            "  python -m isaaclab_m20.scripts.check_m20 --headless --convert urdf"
        )

    @configclass
    class CheckSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = M20_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=M20_CFG.spawn.replace(usd_path=usd_path),
        )

    print(f"[CHECK] USD: {usd_path}")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(CheckSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()

    robot = scene["robot"]
    print(f"[CHECK] bodies ({len(robot.body_names)}): {robot.body_names}")
    print(f"[CHECK] joints ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"[CHECK] default joint pos: {[round(v, 4) for v in robot.data.default_joint_pos[0].tolist()]}")
    print(f"[CHECK] default root state: {robot.data.default_root_state[0].tolist()}")
    try:
        masses = robot.root_physx_view.get_masses()[0]
        print(f"[CHECK] masses: {[round(v, 3) for v in masses.tolist()]} (total {masses.sum():.3f} kg)")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[WARN] could not read masses: {exc}")

    print(
        "\n[CHECK] Expect 16 joints: 4 x hipx/hipy/knee/wheel per leg; 17 bodies "
        "(base + 16 leg links; the empty camera/lidar links are merged away); "
        "total mass ~38.5 kg; root height 0.50 above the ground plane."
    )
    print("[CHECK] If the joint order or names differ, fix the regexes in "
          "isaaclab_m20/assets/m20.py and isaaclab_m20/tasks/m20_env_cfg.py.")
    sim.clear_instance()


def main() -> None:
    # Spawners resolve relative paths against the current directory, so pin it.
    usd_path = str(Path(args_cli.usd).expanduser().resolve()) if args_cli.usd else M20_USD_PATH
    if args_cli.convert == "urdf":
        usd_path = str(Path(convert_urdf()).expanduser().resolve())
    print(f"[CHECK] asset source: {M20_USD_SOURCE}")
    check(usd_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
