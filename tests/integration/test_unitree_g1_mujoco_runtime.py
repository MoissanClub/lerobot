# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Real supported-arm dynamics, never DDS or hardware discovery."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config

pytestmark = pytest.mark.skipif(not os.environ.get("G1_KINEMATICS_ASSETS"), reason="Set G1_KINEMATICS_ASSETS")


@pytest.fixture(params=["g1_29", "g1_23"])
def robot(request):
    root = Path(os.environ["G1_KINEMATICS_ASSETS"])
    robot = UnitreeG1(
        UnitreeG1Config(
            embodiment=request.param,
            simulation_urdf=str(root / f"{request.param}.urdf"),
            gravity_compensation=True,
        )
    )
    robot.connect()
    yield robot
    robot.disconnect()


def test_joint_motion_feedback_and_reset(robot):
    assert robot.is_connected
    sim = robot._native
    target = np.zeros(sim.size)
    target[[0, sim.size // 2]] = -0.3
    target[[3, sim.size // 2 + 3]] = 0.5
    robot.send_action(sim.arm_action(target))
    for _ in range(600):
        robot.step_simulation()
    actual = np.array([robot.get_observation()[k] for k in sim.keys])
    np.testing.assert_allclose(actual, target, atol=0.025)
    np.testing.assert_allclose(sim.model.opt.gravity, [0, 0, -9.81])
    assert np.all(np.abs(sim.last_torque) <= sim.torque_limits)
    robot.reset()
    np.testing.assert_allclose(sim.data.qpos, 0)
    robot.disconnect()
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected


def test_invalid_action_is_atomic(robot):
    sim = robot._native
    before = sim.target.copy()
    for action in ({"wrong.q": 1}, {sim.keys[0]: np.nan}, {"kLeftHipPitch.q": 1}, {sim.keys[0]: 100}):
        with pytest.raises(ValueError):
            robot.send_action(action)
        np.testing.assert_array_equal(sim.target, before)


def test_gravity_matches_mujoco_and_reduces_sag(robot):
    sim = robot._native
    q = np.zeros(sim.size)
    q[[0, sim.size // 2]] = -0.7
    q[[3, sim.size // 2 + 3]] = 0.7
    body = sim.scatter_arm(q)
    sim.reset(body)
    np.testing.assert_allclose(sim.gravity(), sim.data.qfrc_bias[sim.vadr], atol=1e-7)
    errors = []
    for enabled in (False, True):
        sim.reset(body)
        sim.config.gravity_compensation = enabled
        for _ in range(400):
            sim.step()
        errors.append(np.linalg.norm(sim.data.qpos[sim.qadr] - q))
    assert errors[0] > 0.005
    assert errors[1] < errors[0] * 0.1


@pytest.mark.skipif(not os.environ.get("G1_RENDER_TESTS"), reason="Set G1_RENDER_TESTS=1 for EGL rendering")
def test_robot_camera_changes_with_motion(robot):
    robot.disconnect()
    robot.config.simulation_mesh_dir = str(Path(os.environ["G1_KINEMATICS_ASSETS"]) / "meshes")
    robot.connect()
    first = robot.render_simulation(320, 240)
    sim = robot._native
    q = np.zeros(sim.size)
    q[[0, sim.size // 2]] = -0.8
    q[[3, sim.size // 2 + 3]] = 1
    robot.send_action(sim.arm_action(q))
    for _ in range(500):
        robot.step_simulation()
    second = robot.render_simulation(320, 240)
    assert first.shape == second.shape == (240, 320, 3)
    assert second.std() > 3
    assert np.mean(abs(first.astype(float) - second)) > 0.1


def test_native_runs_without_cartesian_dependencies(robot, monkeypatch):
    robot.disconnect()
    for module in ("pinocchio", "casadi", "lerobot.robots.unitree_g1.g1_cartesian_control"):
        monkeypatch.setitem(sys.modules, module, None)
    robot.connect()
    robot.send_action({"kLeftShoulderPitch.q": -0.2})
    assert robot.get_observation()["kLeftShoulderPitch.q"] < 0


def test_zero_remote_input_holds_and_locomotion_is_rejected(robot):
    robot.send_action({"remote.lx": 0.0})
    before = robot._native.target.copy()
    with pytest.raises(ValueError, match="locomotion"):
        robot.send_action({"remote.lx": 1.0})
    np.testing.assert_array_equal(robot._native.target, before)


def test_native_viewer_lifecycle(robot, monkeypatch):
    from unittest.mock import MagicMock

    import mujoco.viewer

    robot.disconnect()
    viewer = MagicMock()
    launch = MagicMock(return_value=viewer)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    robot.config.sim_onscreen = True
    robot.connect()
    launch.assert_called_once_with(robot._native.model, robot._native.data)
    robot.step_simulation()
    viewer.sync.assert_called_once()
    robot.disconnect()
    viewer.close.assert_called_once()


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_standard_teleoperate_cli_without_ik(embodiment):
    assets = Path(os.environ["G1_KINEMATICS_ASSETS"])
    args = [
        "lerobot-teleoperate",
        "--robot.type=unitree_g1",
        "--robot.is_simulation=true",
        f"--robot.embodiment={embodiment}",
        f"--robot.simulation_urdf={assets / (embodiment + '.urdf')}",
        "--robot.sim_publish_images=false",
        "--robot.sim_onscreen=false",
        "--robot.gravity_compensation=true",
        "--robot.cameras={}",
        "--teleop.type=unitree_g1",
        "--teleop.id=simulation_review",
        "--display_data=false",
        "--teleop_time_s=0.1",
    ]
    code = (
        "import sys,runpy; sys.modules['pinocchio']=None; sys.modules['casadi']=None; "
        "sys.modules['lerobot.robots.unitree_g1.g1_cartesian_control']=None; "
        f"sys.argv={args!r}; runpy.run_module('lerobot.scripts.lerobot_teleoperate', run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
