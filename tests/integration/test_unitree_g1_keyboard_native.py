"""Keyboard joint selection through real native physics, for both embodiments."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators.unitree_g1 import keyboard_g1
from tests.integration.test_unitree_g1_keyboard_cli import run_keyboard_cli

pytestmark = pytest.mark.skipif(not os.getenv("G1_KINEMATICS_ASSETS"), reason="Requires pinned G1 models")


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_every_keyboard_joint_matches_native_model(monkeypatch, tmp_path, embodiment):
    root = Path(os.environ["G1_KINEMATICS_ASSETS"])
    config = UnitreeG1Config(
        embodiment=embodiment, simulation_urdf=str(root / f"{embodiment}.urdf"), gravity_compensation=True
    )
    keyboard_config = keyboard_g1.UnitreeG1KeyboardConfig(calibration_dir=tmp_path)
    TeleoperateConfig(robot=config, teleop=keyboard_config)
    teleop = keyboard_g1.UnitreeG1Keyboard(keyboard_config)
    clock = [0.0]
    monkeypatch.setattr(keyboard_g1.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        keyboard_g1, "create_key_listener", lambda *a, **k: SimpleNamespace(stop=lambda: None)
    )
    robot = UnitreeG1(config)
    try:
        robot.connect()
        sim = robot._native
        assert list(teleop.action_features) == sim.keys
        for i, key in enumerate(sim.keys):
            lo, hi = teleop.limits[key]
            assert lo >= sim.lower[i] - 1e-7 and hi <= sim.upper[i] + 1e-7
            robot.reset()
            teleop.connect()
            teleop.send_feedback(robot.get_observation())
            initial = teleop.get_action()
            teleop._on_key("enter")
            teleop._on_key("l" if i < sim.size // 2 else "r")
            teleop._on_key(str(i % (sim.size // 2) + 1))
            for _ in range(5):
                clock[0] += 0.11
                teleop.send_feedback(robot.get_observation())
                teleop._on_key("+")
                action = teleop.get_action()
                assert [k for k in action if action[k] != initial[k]] == [key]
                robot.send_action(action)
                for _ in range(40):
                    robot.step_simulation()
            for _ in range(200):
                robot.step_simulation()
            actual = np.array([robot.get_observation()[k] for k in sim.keys])
            target = np.array([action[k] for k in sim.keys])
            np.testing.assert_allclose(actual, target, atol=0.025)
            assert actual[i] > 0.07
            teleop.disconnect()
    finally:
        teleop.disconnect()
        robot.disconnect()


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_native_keyboard_cli(tmp_path, embodiment):
    root = Path(os.environ["G1_KINEMATICS_ASSETS"])
    run_keyboard_cli(
        tmp_path,
        [
            f"--robot.embodiment={embodiment}",
            f"--robot.simulation_urdf={root / (embodiment + '.urdf')}",
            "--robot.gravity_compensation=true",
        ],
    )
