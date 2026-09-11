# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_with_hands import G1WithHands, G1WithHandsConfig
from lerobot.robots.unitree_g1.hand_system import HandConfig


class Device:
    def __init__(self, config=None):
        self.is_connected = False
        self.is_calibrated = True
        self.action_features = {"q": float}
        self.observation_features = {"q": float}
        self.cameras = {}
        self.value = 0.0
        self.writes = 0
        self.fail = None

    def connect(self, calibrate=True):
        self.is_connected = True
        if self.fail == "connect":
            raise RuntimeError("connect failure")

    def disconnect(self):
        self.is_connected = False
        if self.fail == "disconnect":
            raise RuntimeError("disconnect failure")

    def configure(self):
        pass

    def calibrate(self):
        pass

    def get_observation(self):
        if self.fail == "read":
            raise RuntimeError("read failure")
        return {"q": self.value}

    def validate_action(self, action):
        if set(action) != {"q"} or not np.isfinite(action["q"]) or not 0 <= action["q"] <= 1:
            raise ValueError("invalid hand action")
        return dict(action)

    def send_action(self, action):
        if self.fail == "write":
            raise RuntimeError("write failure")
        self.writes += 1
        if action:
            self.value = action["q"]
        return dict(action)


@pytest.fixture
def robot(tmp_path):
    config = G1WithHandsConfig(
        calibration_dir=tmp_path, hands={s: HandConfig(side=s) for s in ("left", "right")}
    )
    result = G1WithHands(config, body_factory=Device, hand_factory=Device)
    yield result
    for child in [result.body, *result.hands.values()]:
        child.fail = None
    result.disconnect()


def test_namespaces_and_independent_dispatch(robot):
    robot.connect()
    assert set(robot.action_features) == {"q", "hands.left.q", "hands.right.q"}
    action = {"q": 0.7, "hands.left.q": 0.2, "hands.right.q": 0.8}
    assert robot.send_action(action) == action
    assert robot.get_observation() == action
    robot.send_action({"hands.left.q": 0.5})
    assert robot.hands["right"].writes == 1 and robot.body.writes == 1


@pytest.mark.parametrize("bad", [{"hands.center.q": 0.2}, {"hands.left.q": np.nan}, {"hands.right.q": 2.0}])
def test_validate_before_writes(robot, bad):
    robot.connect()
    with pytest.raises(ValueError):
        robot.send_action({"q": 0.4, **bad})
    assert robot.body.writes == 0
    assert all(hand.writes == 0 for hand in robot.hands.values())


@pytest.mark.parametrize("side", ["left", "right"])
def test_connect_rollback(robot, side):
    robot.hands[side].fail = "connect"
    with pytest.raises(RuntimeError):
        robot.connect()
    assert not robot.body.is_connected
    assert not any(hand.is_connected for hand in robot.hands.values())
    robot.hands[side].fail = None
    robot.connect()
    assert robot.is_connected


@pytest.mark.parametrize("failure", ["read", "write"])
def test_failure_disconnects_all(robot, failure):
    robot.connect()
    robot.hands["right"].fail = failure
    with pytest.raises(RuntimeError):
        if failure == "read":
            robot.get_observation()
        else:
            robot.send_action({"hands.right.q": 0.5})
    assert not robot.body.is_connected and not robot.is_connected


def test_disconnect_continues_after_error(robot):
    robot.connect()
    robot.hands["right"].fail = "disconnect"
    with pytest.raises(ExceptionGroup):
        robot.disconnect()
    assert not robot.body.is_connected and not robot.hands["left"].is_connected


def test_no_hands_preserves_body(tmp_path):
    robot = G1WithHands(G1WithHandsConfig(calibration_dir=tmp_path), body_factory=Device)
    assert robot.action_features == robot.body.action_features
    with pytest.raises(RuntimeError):
        robot.get_observation()
    robot.connect()
    assert robot.send_action({"q": 0.3}) == {"q": 0.3}
    assert robot.get_observation() == robot.body.get_observation()
    robot.disconnect()


def test_side_mismatch():
    with pytest.raises(ValueError):
        G1WithHandsConfig(hands={"left": HandConfig(side="right")})
