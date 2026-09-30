# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline physical-backend tests: never initialize DDS or contact a robot."""

import copy
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_arm_sdk import (
    ARM_KEYS,
    ARM_SLOTS,
    G1ArmSDK,
    G1ArmSDKConfig,
    arm_command,
)


def configuration(**kwargs):
    settings = {
        "network_interface": "robot-test",
        "read_only": False,
        "expected_mode_machine": 5,
        "kp": [30.0] * 14,
        "kd": [1.5] * 14,
        "lower": [-2.0] * 14,
        "upper": [2.0] * 14,
        "torque_limits": [5.0] * 14,
    }
    settings.update(kwargs)
    return G1ArmSDKConfig(**settings)


def state(tick=1):
    return SimpleNamespace(
        tick=tick,
        mode_pr=0,
        mode_machine=5,
        motor_state=[SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0) for _ in range(35)],
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
    )


class FakeTransport:
    def __init__(self):
        self.commands = []
        self.writers = 0
        self.closed = False

    def connect(self, cfg, callback):
        self.callback = callback
        callback(state())

    def enable_writer(self):
        self.writers += 1

    def wait_ready(self, disabled_command):
        assert disabled_command.motor_cmd[29].q == 0

    def write(self, msg):
        self.commands.append(copy.deepcopy(msg))

    def close(self):
        self.closed = True


@pytest.fixture
def backend():
    transport = FakeTransport()
    now = [10.0]
    robot = G1ArmSDK(configuration(), transport=transport, clock=lambda: now[0])
    robot.connect()
    with patch("threading.Thread.start"):
        robot.activate()
    robot.thread = None
    yield robot, transport, now
    robot.close = lambda: None  # No real thread was started; avoid masking asserted faults.
    if robot.lease is not None:
        robot.lease.close()


def test_serialized_message_owns_only_arms_and_weight():
    cmd = arm_command(np.arange(14) / 100, np.zeros(14), 0.25, 5, configuration())
    assert cmd.mode_machine == 5 and cmd.mode_pr == 0
    assert cmd.motor_cmd[29].q == 0.25
    for index, motor in enumerate(cmd.motor_cmd):
        if index not in ARM_SLOTS and index != 29:
            assert (motor.q, motor.dq, motor.tau, motor.kp, motor.kd, motor.mode) == (0, 0, 0, 0, 0, 0)
    restored = type(cmd).deserialize(cmd.serialize())
    assert restored.motor_cmd[22].q == pytest.approx(0.07)
    assert restored.motor_cmd[29].q == 0.25


def test_readonly_lifecycle_has_no_writer():
    io = FakeTransport()
    robot = G1ArmSDK(configuration(read_only=True), transport=io)
    robot.connect()
    assert len(robot.observation()) == 87
    with pytest.raises(RuntimeError):
        robot.activate()
    with pytest.raises(RuntimeError):
        robot.send(dict.fromkeys(ARM_KEYS, 0.0))
    robot.close()
    robot.close()
    with pytest.raises(RuntimeError):
        robot.observation()
    with pytest.raises(RuntimeError):
        robot.connect()
    assert io.writers == 0 and not io.commands


@pytest.mark.parametrize(
    "change",
    [
        {"is_simulation": True},
        {"embodiment": "g1_23"},
        {"controller": "SonicWholeBodyController"},
        {"gravity_compensation": True},
    ],
)
def test_configuration_prevents_wrong_authority(change):
    values = {"is_simulation": False, "arm_sdk": configuration()}
    values.update(change)
    with pytest.raises(ValueError):
        UnitreeG1Config(**values)


def test_explicit_motion_gains_required():
    with pytest.raises(ValueError):
        G1ArmSDKConfig(network_interface="eth0", read_only=False)


@pytest.mark.parametrize("failure", ["duplicate", "regression", "mode", "nan"])
def test_invalid_feedback_never_refreshes_authority(backend, failure):
    robot, _, now = backend
    msg = state(2)
    if failure == "duplicate":
        msg.tick = 1
    elif failure == "regression":
        msg.tick = 0
    elif failure == "mode":
        msg.mode_machine = 6
    else:
        msg.motor_state[15].q = float("nan")
    now[0] += 0.2
    robot._receive(msg)
    with pytest.raises(RuntimeError):
        robot.observation()


@pytest.mark.parametrize("bad", ["partial", "leg", "nan", "delta"])
def test_invalid_command_latches_fault(backend, bad):
    robot, io, _ = backend
    action = dict.fromkeys(ARM_KEYS, 0.0)
    if bad == "partial":
        action.pop(ARM_KEYS[0])
    elif bad == "leg":
        action["kLeftHipPitch.q"] = 0.0
    elif bad == "nan":
        action[ARM_KEYS[0]] = float("nan")
    else:
        action[ARM_KEYS[0]] = 0.2
    with pytest.raises(ValueError):
        robot.send(action)
    assert robot.fault and not io.commands
    with pytest.raises(RuntimeError):
        robot.send(dict.fromkeys(ARM_KEYS, 0.0))


def test_slew_limits_both_derivatives_and_initial_blend(backend):
    robot, io, now = backend
    positions = [np.zeros(14)]
    for step in range(1, 150):
        now[0] += 0.02
        robot._receive(state(step + 1))
        action = dict.fromkeys(ARM_KEYS, 0.0)
        action[ARM_KEYS[0]] = 0.02 if step < 80 else -0.02
        robot.send(action)
        robot._step()
        positions.append(robot.previous.copy())
    velocity = np.diff(positions, axis=0) / 0.02
    assert np.max(np.abs(velocity)) <= 0.1 + 1e-8
    assert np.max(np.abs(np.diff(velocity, axis=0) / 0.02)) <= 0.5 + 1e-8
    assert io.commands[0].motor_cmd[29].q == pytest.approx(0.01)
    assert io.commands[-1].motor_cmd[29].q == 1.0


def test_watchdog_releases_even_when_caller_disappears():
    io = FakeTransport()
    robot = G1ArmSDK(configuration(), transport=io)
    robot.connect()
    robot.activate()
    deadline = time.monotonic() + 2
    while robot.active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not robot.active and robot.fault
    with pytest.raises(RuntimeError):
        robot.close()
    assert robot.release_sent and io.commands[-1].motor_cmd[29].q == 0


@pytest.mark.parametrize("field,value", [("q", 0.2), ("dq", 1.0), ("tau_est", 100.0)])
def test_measured_envelope_faults(backend, field, value):
    robot, _, now = backend
    msg = state(2)
    setattr(msg.motor_state[15], field, value)
    now[0] += 0.02
    robot._receive(msg)
    with pytest.raises(RuntimeError):
        robot._step()


def test_robot_facade_does_not_touch_legacy_backend():
    from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

    backend = G1ArmSDK(configuration(read_only=True), transport=FakeTransport())
    robot = UnitreeG1(UnitreeG1Config(is_simulation=False, arm_sdk=backend.config))
    with (
        patch("lerobot.robots.unitree_g1.g1_arm_sdk.G1ArmSDK", return_value=backend),
        patch.object(robot, "_ChannelFactoryInitialize", side_effect=AssertionError("legacy DDS")),
        patch.object(robot, "_send_zero_torque", side_effect=AssertionError("whole body")),
    ):
        robot.connect()
        assert robot.is_connected and set(robot.action_features) == set(ARM_KEYS)
        assert "kRightElbow.q" in robot.get_observation()
        with pytest.raises(RuntimeError):
            robot.reset()
        with pytest.raises(RuntimeError):
            robot.publish_lowcmd({})
        robot.disconnect()
        assert not robot.is_connected


def test_sdk_transport_topics_and_bounded_write_without_dds():
    from unittest.mock import MagicMock

    from lerobot.robots.unitree_g1.g1_arm_sdk import ArmSDKTransport

    publisher = MagicMock()
    publisher.Write.return_value = True
    with (
        patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize") as initialize,
        patch("unitree_sdk2py.core.channel.ChannelSubscriber") as sub,
        patch("unitree_sdk2py.core.channel.ChannelPublisher", return_value=publisher) as pub,
        patch("socket.if_nameindex", return_value=[(1, "robot-test")]),
    ):
        io = ArmSDKTransport()
        cfg = configuration()

        def callback(msg):
            pass

        io.connect(cfg, callback)
        initialize.assert_called_once_with(0, "robot-test")
        assert sub.call_args.args[0] == "rt/lowstate"
        pub.assert_not_called()
        io.enable_writer()
        assert pub.call_args.args[0] == "rt/arm_sdk"
        cmd = arm_command(np.zeros(14), np.zeros(14), 0.1, 5, cfg)
        io.write(cmd)
        publisher.Write.assert_called_once_with(cmd, cfg.period_s)
        io.close()
        publisher.Close.assert_called_once()
        sub.return_value.Close.assert_called_once()


def test_release_failure_is_reported():
    io = FakeTransport()
    robot = G1ArmSDK(configuration(), transport=io)
    robot.connect()
    robot.activate()
    io.write = lambda msg: (_ for _ in ()).throw(RuntimeError("link lost"))
    with pytest.raises(RuntimeError, match="release write failed"):
        robot.close()
    assert not robot.release_sent and io.closed


def test_initialize_failure_closes_without_publish():
    io = FakeTransport()
    io.connect = lambda cfg, callback: (_ for _ in ()).throw(RuntimeError("init failed"))
    robot = G1ArmSDK(configuration(), transport=io)
    with pytest.raises(RuntimeError, match="init failed"):
        robot.connect()
    assert io.closed and not io.commands


def test_live_feedback_but_absent_commands_faults(backend):
    robot, _, now = backend
    now[0] += 0.2
    robot._receive(state(2))
    robot.last_step = now[0]
    with pytest.raises(RuntimeError, match="Caller command timeout"):
        robot._step()


def test_tick_wrap_is_valid(backend):
    robot, _, now = backend
    robot.tick = 2**32 - 1
    now[0] += 0.01
    robot._receive(state(0))
    assert robot.tick == 0 and not robot.fault


def test_double_publisher_is_rejected(backend):
    robot = G1ArmSDK(configuration(), transport=FakeTransport())
    robot.connect()
    with pytest.raises(BlockingIOError):
        robot.activate()
    assert robot.transport.writers == 0
    robot.close()


def test_worker_start_failure_can_be_cleaned_up():
    io = FakeTransport()
    robot = G1ArmSDK(configuration(), transport=io)
    robot.connect()
    with (
        patch("threading.Thread.start", side_effect=RuntimeError("start failed")),
        pytest.raises(RuntimeError, match="start failed"),
    ):
        robot.activate()
    robot.close()
    assert io.closed and robot.lease is None and not robot.active
