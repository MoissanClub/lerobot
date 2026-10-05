# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import json
import logging
import time
from contextlib import ExitStack
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.robots.unitree_g1 import g1_vr_processor as module
from lerobot.robots.unitree_g1.g1_arm_sdk import G1ArmSDKConfig
from lerobot.robots.unitree_g1.g1_motion import UnitreeG1Motion, UnitreeG1MotionConfig
from lerobot.robots.unitree_g1.g1_vr_control import ARM_KEYS, synthetic_input
from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig


@pytest.mark.parametrize(
    "axis,sign,direction",
    [
        (0, 1, "forward"),
        (0, -1, "backward"),
        (1, 1, "left"),
        (1, -1, "right"),
        (2, 1, "up"),
        (2, -1, "down"),
    ],
)
def test_pickup_guidance_direction_and_units(axis, sign, direction):
    delta = np.zeros(3)
    delta[axis] = sign * 0.12
    assert module.pickup_guidance("Left", delta) == f"Left hand: move {direction} 12.0 cm"


def test_pickup_guidance_reports_ready_and_prioritizes_largest_error():
    assert "READY" in module.pickup_guidance("Right", np.array([0.02, 0.01, 0.01]))
    assert module.pickup_guidance("Right", np.array([0.08, -0.12, 0.001])).endswith(
        "move right 12.0 cm, forward 8.0 cm"
    )


def test_processor_pause_requires_explicit_restart(monkeypatch):
    ik = Mock()
    ik.solve.return_value = (np.ones(14) * 0.01, np.zeros(14))
    monkeypatch.setattr(module, "G1VRKinematics", lambda _: ik)
    processor = module.G1VRActionProcessor("unused")
    processor._current_transition = {TransitionKey.OBSERVATION: dict.fromkeys(ARM_KEYS, 0.0)}
    sample = synthetic_input((np.eye(4), np.eye(4)))
    sample["control.start"] = True
    assert processor.action(sample)["control.enabled"] == 1
    sample["control.start"] = False
    sample["head.tracked"] = False
    assert processor.action(sample)["control.enabled"] == 0
    sample["head.tracked"] = True
    assert processor.action(sample)["control.enabled"] == 0
    processor.reset()
    assert processor.enabled is False


def test_bounded_mapper_anchors_saturates_returns_and_preserves_session_origin(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    class Kinematics:
        def __init__(self, _):
            pass

        def fk(self, q):
            left, right = np.eye(4), np.eye(4)
            left[0, 3], right[0, 3] = q[0], q[7]
            return left, right

        def solve(self, wrists, measured, max_step):
            q = measured.copy()
            q[0], q[7] = wrists[0][0, 3], wrists[1][0, 3]
            return q, np.zeros(14)

    monkeypatch.setattr(module, "G1VRKinematics", Kinematics)
    processor = module.G1VRActionProcessor("unused", arm_test=True)
    obs = dict.fromkeys(ARM_KEYS, 0.0)
    processor._current_transition = {TransitionKey.OBSERVATION: obs}
    sample = synthetic_input((np.eye(4), np.eye(4)))
    original = sample["left.grip_pos"].copy()
    sample["control.start"] = True
    first = processor.action(sample)
    assert [first[k] for k in ARM_KEYS] == [0.0] * 14
    sample["control.start"] = False
    sample["left.grip_pos"][2] -= 1.0
    moved = processor.action(sample)
    assert moved[ARM_KEYS[0]] == pytest.approx(0.02)
    assert moved[ARM_KEYS[7]] == 0.0
    assert moved["base.vx"] == moved["base.vy"] == moved["base.yaw"] == 0.0
    assert moved["control.enabled"] == 1.0
    assert "VR workspace boundary" in caplog.text
    assert "move backward" in caplog.text
    # Repeated farther requests hold the boundary without releasing authority.
    for distance in (2.0, 3.0, 1.0):
        sample["left.grip_pos"][2] = original[2] - distance
        saturated = processor.action(sample)
        assert saturated[ARM_KEYS[0]] == pytest.approx(0.02)
        assert saturated["control.enabled"] == 1.0
    sample["left.grip_pos"] = original.copy()
    sample["left.grip_pos"][2] -= 0.05
    inside = processor.action(sample)
    assert inside[ARM_KEYS[0]] == pytest.approx(0.01)
    assert inside["control.enabled"] == 1.0
    assert "VR workspace regained" in caplog.text
    # Moving out again must produce guidance again, without pressing start.
    caplog.clear()
    sample["left.grip_pos"][2] -= 1.0
    assert processor.action(sample)[ARM_KEYS[0]] == pytest.approx(0.02)
    assert "VR workspace boundary" in caplog.text
    sample["left.grip_pos"] = original.copy()
    returned = processor.action(sample)
    assert returned[ARM_KEYS[0]] == pytest.approx(0.0)
    sample["control.pause"] = True
    obs[ARM_KEYS[0]] = 0.01
    processor.action(sample)
    sample["control.pause"], sample["control.start"] = False, True
    processor.action(sample)
    assert processor.origin[0] == 0.0
    sample["control.start"] = False
    sample["left.grip_pos"][2] -= 1.0
    assert processor.action(sample)[ARM_KEYS[0]] == pytest.approx(0.02)
    sample["left.tracked"] = False
    assert processor.action(sample)["control.enabled"] == 0.0
    sample["left.tracked"] = True
    assert processor.action(sample)["control.enabled"] == 0.0


def test_pickup_requires_both_wrists_before_arm_activation(monkeypatch):
    ik = Mock()
    ik.fk.return_value = (np.eye(4), np.eye(4))
    ik.solve.return_value = (np.ones(14), np.zeros(14))
    monkeypatch.setattr(module, "G1VRKinematics", lambda _: ik)
    processor = module.G1VRActionProcessor("unused", arm_test=True)
    obs = dict.fromkeys(ARM_KEYS, 0.01)
    processor._current_transition = {TransitionKey.OBSERVATION: obs}
    sdk = G1ArmSDKConfig(network_interface="robot-test")
    robot = UnitreeG1Motion(
        UnitreeG1MotionConfig(
            mode="arms",
            enable_motion=True,
            arm_test=True,
            arm_sdk=sdk,
        )
    )
    robot._connected = True
    robot.arm_config = sdk
    robot.arm = Mock(active=False)
    robot.arm.observation.return_value = obs
    robot.arm.activate.side_effect = lambda: setattr(robot.arm, "active", True)
    robot.voice = Mock()
    for i, distances in enumerate(((0.2, 0.2), (0.0, 0.2), (0.04, 0.04))):
        wrists = [np.eye(4), np.eye(4)]
        for wrist, distance in zip(wrists, distances, strict=True):
            wrist[0, 3] = distance
        sample = synthetic_input(wrists)
        sample["control.start"] = i == 0
        action = processor.action(sample)
        action["control.live"] = 1.0
        robot.send_action(action)
        if i < 2:
            assert not action["control.enabled"]
            robot.arm.activate.assert_not_called()
            robot.arm.send.assert_not_called()
            ik.solve.assert_not_called()
            robot.voice.update.assert_called_with(False)
        else:
            assert action["control.enabled"]
            robot.arm.activate.assert_called_once()
            np.testing.assert_allclose([action[k] for k in ARM_KEYS], 0.01)
            robot.voice.update.assert_called_with(True)


@pytest.mark.parametrize("cancel", ["pause", "stale", "quit"])
def test_pickup_wait_cannot_resume_after_cancel_without_start(monkeypatch, cancel):
    ik = Mock()
    ik.fk.return_value = (np.eye(4), np.eye(4))
    monkeypatch.setattr(module, "G1VRKinematics", lambda _: ik)
    processor = module.G1VRActionProcessor("unused", arm_test=True)
    processor._current_transition = {TransitionKey.OBSERVATION: dict.fromkeys(ARM_KEYS, 0.0)}
    far = np.eye(4)
    far[0, 3] = 0.2
    sample = synthetic_input((far, far))
    sample["control.start"] = True
    assert not processor.action(sample)["control.enabled"]
    sample["control.start"] = False
    if cancel == "stale":
        sample["captured_at"] -= 1.0
    else:
        sample[f"control.{cancel}"] = True
    assert not processor.action(sample)["control.enabled"]
    assert not processor.waiting_for_pickup
    near = synthetic_input((np.eye(4), np.eye(4)))
    assert not processor.action(near)["control.enabled"]
    ik.solve.assert_not_called()


def test_initial_pickup_waits_for_tracking_then_guides_without_another_start(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    ik = Mock()
    ik.fk.return_value = (np.eye(4), np.eye(4))
    monkeypatch.setattr(module, "G1VRKinematics", lambda _: ik)
    processor = module.G1VRActionProcessor("unused", arm_test=True)
    processor._current_transition = {TransitionKey.OBSERVATION: dict.fromkeys(ARM_KEYS, 0.0)}
    far = np.eye(4)
    far[0, 3] = 0.2
    sample = synthetic_input((far, far))
    sample.update({"control.start": True, "left.tracked": False, "right.tracked": False})
    assert not processor.action(sample)["control.enabled"]
    assert processor.waiting_for_pickup
    assert "Start to get picked up" in caplog.text
    assert "Waiting for tracking: left, right" in caplog.text
    assert "XR tracking paused" not in caplog.text
    sample["control.start"] = False
    sample["left.tracked"] = sample["right.tracked"] = True
    assert not processor.action(sample)["control.enabled"]
    assert "Left hand: move backward 20.0 cm" in caplog.text
    near = synthetic_input((np.eye(4), np.eye(4)))
    assert processor.action(near)["control.enabled"]
    near["left.tracked"] = False
    assert not processor.action(near)["control.enabled"]
    assert not processor.waiting_for_pickup
    near["left.tracked"] = True
    assert not processor.action(near)["control.enabled"]


@pytest.mark.parametrize("mode", ["arms", "walk", "combined"])
def test_physical_config_requires_explicit_enable(mode):
    with pytest.raises(ValueError, match="enable_motion"):
        UnitreeG1MotionConfig(assets="unused", mode=mode)


def test_replay_cannot_activate_physical_robot():
    robot = UnitreeG1Motion(
        UnitreeG1MotionConfig(
            assets="unused", mode="arms", enable_motion=True, contract="unused", report_path="unused"
        )
    )
    robot._connected = True
    robot.arm = Mock()
    action = dict.fromkeys(robot.action_features, 0.0)
    action.update({"control.enabled": 1.0, "control.created_at": time.monotonic()})
    with pytest.raises(ValueError, match="Replay"):
        robot.send_action(action)
    robot.arm.activate.assert_not_called()
    robot.arm.send.assert_not_called()


def test_failed_arm_activation_does_not_announce_following():
    robot = UnitreeG1Motion(
        UnitreeG1MotionConfig(
            mode="arms",
            enable_motion=True,
            arm_test=True,
            arm_sdk=G1ArmSDKConfig(network_interface="robot-test"),
        )
    )
    robot._connected = True
    robot.arm_config = robot.config.arm_sdk
    robot.arm = Mock(active=False)
    robot.arm.observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    robot.arm.activate.side_effect = RuntimeError("starting pose rejected")
    robot.voice = Mock()
    action = dict.fromkeys(robot.action_features, 0.0)
    action.update({"control.created_at": time.monotonic(), "control.enabled": 1.0, "control.live": 1.0})
    with pytest.raises(RuntimeError, match="starting pose"):
        robot.send_action(action)
    robot.voice.update.assert_not_called()


def test_replay_provenance_cannot_be_spoofed(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"input.live": True, "control.start": True}) + "\n")
    teleop = XRControllers(XRControllersConfig(replay_path=str(path)))
    teleop.connect()
    try:
        action = teleop.get_action()
        assert action["input.live"] is False
        assert action["control.start"] is True
        assert teleop.get_action()["control.quit"] is True
    finally:
        teleop.disconnect()
    assert not teleop.is_connected


def test_xr_failed_start_closes_backend():
    backend = Mock()
    backend.connect.side_effect = RuntimeError("no XR runtime")
    teleop = XRControllers(XRControllersConfig(), session_factory=lambda _: backend)
    with pytest.raises(RuntimeError, match="no XR runtime"):
        teleop.connect()
    backend.close.assert_called_once()
    assert not teleop.is_connected


def test_xr_recording_round_trips_arrays_and_refuses_overwrite(tmp_path):
    path = tmp_path / "xr.jsonl"
    backend = Mock()
    backend.read.return_value = {"captured_at": time.monotonic(), "head.pos": np.ones(3)}
    teleop = XRControllers(XRControllersConfig(record_path=str(path)), session_factory=lambda _: backend)
    teleop.connect()
    teleop.get_action()
    teleop.disconnect()
    backend.close.assert_called_once()
    row = json.loads(path.read_text())
    assert row["head.pos"] == [1.0, 1.0, 1.0]
    assert row["input.live"]
    replay = XRControllers(XRControllersConfig(replay_path=str(path)))
    replay.connect()
    try:
        action = replay.get_action()
        assert action["head.pos"] == row["head.pos"]
        assert not action["input.live"]
    finally:
        replay.disconnect()
    with pytest.raises(FileExistsError):
        teleop.connect()
    assert json.loads(path.read_text()) == row


def test_xr_recording_file_closes_when_backend_start_fails(tmp_path):
    backend = Mock()
    backend.connect.side_effect = RuntimeError("no headset")
    teleop = XRControllers(
        XRControllersConfig(record_path=str(tmp_path / "xr.jsonl")), session_factory=lambda _: backend
    )
    with pytest.raises(RuntimeError, match="no headset"):
        teleop.connect()
    assert teleop._record_stream is None
    backend.close.assert_called_once()


def test_xr_runtime_start_requires_eula():
    with pytest.raises(ValueError, match="EULA"):
        XRControllersConfig(cloudxr_config="config.env")


def test_xr_replay_cannot_open_live_runtime():
    with pytest.raises(ValueError, match="Replay"):
        XRControllersConfig(replay_path="input.jsonl", cloudxr_config="config.env", accept_cloudxr_eula=True)


def test_robot_closes_transport_even_if_report_write_fails():
    robot = UnitreeG1Motion(UnitreeG1MotionConfig(assets="unused"))
    closed = Mock()
    robot._stack = ExitStack()
    robot._stack.callback(closed)
    robot.report = Mock()
    robot.report.write.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        robot.disconnect()
    closed.assert_called_once()


def test_fk_outside_workspace_projects_to_boundary_without_error(monkeypatch):
    class Kinematics:
        def fk(self, q):
            left, right = np.eye(4), np.eye(4)
            left[0, 3] = 2 * q[0]
            return left, right

        def solve(self, wrists, measured, max_step):
            # Simulate an IK candidate whose FK overshoots the requested wrist.
            q = measured.copy()
            q[0] = wrists[0][0, 3]
            return q, np.zeros(14)

    monkeypatch.setattr(module, "G1VRKinematics", lambda _: Kinematics())
    processor = module.G1VRActionProcessor("unused", arm_test=True)
    processor._current_transition = {TransitionKey.OBSERVATION: dict.fromkeys(ARM_KEYS, 0.0)}
    sample = synthetic_input((np.eye(4), np.eye(4)))
    sample["control.start"] = True
    processor.action(sample)
    sample["control.start"] = False
    sample["left.grip_pos"][2] -= 1.0
    for _ in range(3):
        result = processor.action(sample)
        assert result["control.enabled"] == 1
        assert result[ARM_KEYS[0]] == pytest.approx(0.01, abs=1e-7)
        assert result[ARM_KEYS[7]] == 0
