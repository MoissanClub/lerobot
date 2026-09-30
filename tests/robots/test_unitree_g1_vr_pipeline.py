# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import json
import time
from contextlib import ExitStack
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.robots.unitree_g1 import g1_vr_processor as module
from lerobot.robots.unitree_g1.g1_motion import UnitreeG1Motion, UnitreeG1MotionConfig
from lerobot.robots.unitree_g1.g1_vr_control import ARM_KEYS, synthetic_input
from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig


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
