"""Opt-in headless/visual acceptance with a local or published Hub simulator.

G1_BRAINCO_HUB_CHECKOUT=/path/to/unitree-g1-mujoco MUJOCO_GL=egl \
    python -m pytest tests/robots/test_unitree_g1_brainco_sim.py -v

For a published snapshot, set G1_BRAINCO_HUB_PATH=OWNER/REPO@COMMIT instead.
Run this file directly with --viewer for bounded visual inspection.
Only local-checkout mode redirects Hub file resolution. Published mode uses
the normal downloader. Robot construction, environment factory/import, MuJoCo,
DDS body commands and hand commands/feedback are real in both modes.
Each run is isolated in a subprocess because the Unitree DDS factory is global.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

DEFAULT_HUB_PATH = "davidwei79/unitree-g1-mujoco-brainco@99806d8c7e9b8dc4c1cd500450c3f563a52a8953"


@pytest.mark.skipif(
    not (os.environ.get("G1_BRAINCO_HUB_CHECKOUT") or os.environ.get("G1_BRAINCO_HUB_PATH")),
    reason="Requires matching Hub checkout or pinned Hub revision",
)
@pytest.mark.parametrize("repeat", range(2))
def test_live_brainco_simulation(repeat):
    checkout = os.environ.get("G1_BRAINCO_HUB_CHECKOUT")
    revision = os.environ.get("G1_BRAINCO_HUB_PATH")
    assert not (checkout and revision), "Select only one Hub source"
    source_args = [checkout] if checkout else ["--hub-path", revision]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *source_args, "--headless", "--motion", "combined"],
        capture_output=True,
        text=True,
        timeout=180 if revision else 60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: real LeRobot BrainCo simulation" in result.stdout
    for failure in ("Error closing sim_env", "Warning during close", "did not stop cleanly", "Traceback"):
        assert failure not in result.stdout + result.stderr


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bounded BrainCo Robot API acceptance; simulation only.")
    parser.add_argument("hub_checkout", nargs="?", type=Path, help="Local Hub checkout (test resolver)")
    parser.add_argument("--hub-path", help="Published OWNER/REPO@40-character-commit; real Hub download")
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--viewer", action="store_true", help="Open the native MuJoCo viewer (requires OpenGL)"
    )
    display.add_argument("--headless", action="store_true", help="No window (default)")
    parser.add_argument(
        "--cycles", type=int, choices=range(1, 4), default=1, help="Repeat the selected sequence, then exit"
    )
    parser.add_argument(
        "--motion",
        choices=("fingers", "combined"),
        default="fingers",
        help="Individual hand controls (default), or combined arm/hand regression",
    )
    args = parser.parse_args(argv)
    if args.hub_checkout and args.hub_path:
        parser.error("Select only one local checkout or --hub-path")
    if not args.hub_checkout and not args.hub_path:
        args.hub_path = DEFAULT_HUB_PATH
    if args.hub_checkout and not (args.hub_checkout / "env.py").is_file():
        parser.error("Local checkout must contain env.py")
    if args.hub_path and not re.fullmatch(r"[^/@\s]+/[^/@\s]+@[0-9a-f]{40}", args.hub_path):
        parser.error("--hub-path must pin a full 40-character commit, not main or a branch")
    return args


@pytest.mark.parametrize("source", ["owner/repo", "owner/repo@main", "owner/repo@abc123"])
def test_remote_source_requires_commit(source):
    with pytest.raises(SystemExit):
        parse_args(["--hub-path", source])


def test_viewer_options_are_bounded():
    source = "owner/repo@" + "a" * 40
    args = parse_args(["--hub-path", source, "--viewer", "--cycles", "2"])
    assert args.viewer and args.cycles == 2
    with pytest.raises(SystemExit):
        parse_args(["--hub-path", source, "--cycles", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--hub-path", source, "--viewer", "--headless"])


def test_default_downloads_pinned_model_not_dataset():
    args = parse_args(["--viewer"])
    assert args.hub_checkout is None
    assert args.hub_path == DEFAULT_HUB_PATH
    assert args.motion == "fingers"


def finger_phases(features):
    names = ("thumb flexion", "thumb opposition", "index", "middle", "ring", "pinky")
    phases = [("Open both hands", dict.fromkeys(features, 0.0))]
    for side in ("left", "right"):
        for motor, name in enumerate(names):
            target = dict.fromkeys(features, 0.0)
            target[f"hands.{side}.motor_{motor}.pos"] = 0.5
            phases.append((f"Move {side} {name}", target))
            phases.append((f"Reopen {side} {name}", dict.fromkeys(features, 0.0)))
    return phases


def test_finger_sequence_is_individual_and_returns_to_open():
    features = [f"hands.{side}.motor_{motor}.pos" for side in ("left", "right") for motor in range(6)]
    phases = finger_phases(features)
    assert len(phases) == 25
    moved = []
    for index, (_, target) in enumerate(phases):
        active = [name for name, value in target.items() if value]
        assert len(active) == index % 2
        moved.extend(active)
    assert moved == features
    assert not any(phases[-1][1].values())


def run_live(hub_path: Path | None, *, hub_revision=None, viewer=False, cycles=1, motion="fingers"):
    def reject_serial_device(event, arguments):
        if event == "open" and isinstance(arguments[0], str):
            assert not arguments[0].startswith(("/dev/tty", "/dev/serial")), arguments[0]

    sys.addaudithook(reject_serial_device)
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
    from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

    source = Path(sys.modules[UnitreeG1.__module__].__file__).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2] / "src"), source

    robot = UnitreeG1(
        UnitreeG1Config(
            end_effector="brainco",
            is_simulation=True,
            sim_publish_images=False,
            sim_onscreen=viewer,
            sim_hub_path=hub_revision or "lerobot/unitree-g1-mujoco",
            cameras={},
            gravity_compensation=False,
        )
    )
    print(f"LeRobot: {sys.modules[UnitreeG1.__module__].__file__}", flush=True)
    print(f"Hub source: {hub_revision or hub_path}", flush=True)
    print(
        f"Mode: {'native viewer' if viewer else 'headless'}; {cycles} cycle(s), motion={motion}; no dataset",
        flush=True,
    )
    resolver = nullcontext()
    if hub_path is not None:
        resolver = patch(
            "lerobot.envs.factory._download_hub_file",
            return_value=("local/brainco-test", "env.py", str(hub_path / "env.py"), "local-working-tree"),
        )
    try:
        with resolver:
            robot.connect()
        assert robot.sim_env.brainco_interface_version == 1
        assert not robot.sim_env.simulator.unitree_bridge.dds_hands
        assert robot.sim_env.sim_env.mj_model.opt.gravity[2] == -9.81
        features = tuple(robot._hands_ft)
        obs = robot.get_observation()
        assert set(features) <= obs.keys()
        phases = [
            ("Open both hands", dict.fromkeys(features, 0.0)),
            ("Partially close both hands", dict.fromkeys(features, 0.35)),
            ("Close left hand; open right hand", {n: 0.55 if ".left." in n else 0.0 for n in features}),
            (
                "Close right hand; open left hand; bend right elbow",
                {n: 0.55 if ".right." in n else 0.0 for n in features},
            ),
            ("Reopen both hands", dict.fromkeys(features, 0.0)),
        ]
        if motion == "fingers":
            phases = finger_phases(features)
            # Hold the initial body pose without commanding an arm movement.
            robot.send_action({name: value for name, value in obs.items() if name.endswith(".q")})
        phases *= cycles
        for index, (label, hands) in enumerate(phases):
            print(f"Observe {index + 1}/{len(phases)}: {label}", flush=True)
            # An arm target and both hands travel through the same Robot API.
            action = dict(hands)
            if motion == "combined":
                action["kRightElbow.q"] = 0.35 if index % 5 == 3 else 0.15
            deadline = time.monotonic() + (1.5 if motion == "fingers" else 3.0)
            while time.monotonic() < deadline:
                if viewer and not robot.sim_env.sim_env.viewer.is_running():
                    raise KeyboardInterrupt
                robot.send_action(action)
                obs = robot.get_observation()
                assert all(np.isfinite(obs[name]) for name in features)
                time.sleep(0.02)
            measured = np.array([obs[name] for name in features])
            target = np.array([hands[name] for name in features])
            error = float(np.max(np.abs(measured - target)))
            print(f"phase={index} max_hand_error={error:.4f} elbow={obs['kRightElbow.q']:.4f}", flush=True)
            assert error < 0.03, (hands, measured)
            if motion == "combined":
                assert abs(obs["kRightElbow.q"] - action["kRightElbow.q"]) < 0.15
        assert not np.any(robot.sim_env.sim_env.mj_data.warning.number)
        assert "bc_stark_sdk" not in sys.modules
    finally:
        robot.disconnect()
        assert robot.subscribe_thread is None or not robot.subscribe_thread.is_alive()
    print(
        f"PASS: real LeRobot BrainCo simulation; gravity on, {motion}, clean thread shutdown",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer else "egl"
    try:
        run_live(
            args.hub_checkout.resolve() if args.hub_checkout else None,
            hub_revision=args.hub_path,
            viewer=args.viewer,
            cycles=args.cycles,
            motion=args.motion,
        )
    except KeyboardInterrupt:
        print("Stopped by user; acceptance sequence incomplete.", flush=True)
        raise SystemExit(130) from None
