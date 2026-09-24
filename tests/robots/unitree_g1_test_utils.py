#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared mocked G1 fixtures; no simulator downloads or hardware access."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import NUM_MOTORS, G1_29_JointIndex

TOKEN_DIM = 64

# ---------------------------------------------------------------------------
# Controller stubs
# ---------------------------------------------------------------------------


class TokenController:
    """Stand-in for SonicWholeBodyController: declares its own action and proprio spaces."""

    control_dt = 0.02

    def __init__(self, dim: int = TOKEN_DIM):
        self.action_ft = {f"motion_token.{i}.pos": float for i in range(dim)}
        self.observation_ft = {f"motion_token_state.{i}.pos": float for i in range(dim)}
        self.kp = np.full(NUM_MOTORS, 100.0, np.float32)
        self.kd = np.full(NUM_MOTORS, 2.0, np.float32)
        self.default_angles = np.zeros(NUM_MOTORS, np.float32)
        self.reset_calls = 0

    def run_step(self, action: dict, lowstate) -> dict:
        return {}

    def reset(self) -> None:
        self.reset_calls += 1

    def observation_state(self) -> dict[str, float]:
        return dict.fromkeys(self.observation_ft, 0.0)


class LocomotionOnlyController:
    """Stand-in for GR00T/Holosoma: declares no optional features, so the robot's defaults apply."""

    control_dt = 0.02

    def run_step(self, action: dict, lowstate) -> dict:
        return {}

    def reset(self) -> None:
        pass


def make_camera_config(width: int = 640, height: int = 480, use_depth: bool = False):
    """Duck-typed camera config.

    ``_cameras_ft`` reads the resolution/rgb/depth attributes, and ``RobotConfig`` validates
    that width, height and fps are all set.
    """
    return SimpleNamespace(width=width, height=height, fps=30, use_rgb=True, use_depth=use_depth)


# ---------------------------------------------------------------------------
# SDK mocks
# ---------------------------------------------------------------------------


def _make_lowstate_msg_mock():
    """Create a mock that mimics the SDK LowState_ message."""
    msg = MagicMock()
    msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
        idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
    )
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = 0
    return msg


def _make_sdk_mocks():
    """Create mocks for the Unitree SDK modules used by UnitreeG1."""
    lowcmd_default = MagicMock()
    lowcmd_default.mode_pr = 0
    lowcmd_default.motor_cmd = [MagicMock() for _ in range(35)]

    crc_mock = MagicMock()
    crc_mock.Crc.return_value = 0

    lowstate_msg = _make_lowstate_msg_mock()

    subscriber_mock = MagicMock()
    subscriber_mock.Read.return_value = lowstate_msg

    return {
        "lowcmd_default": lowcmd_default,
        "crc_mock": crc_mock,
        "subscriber_mock": subscriber_mock,
        "publisher_mock": MagicMock(),
        "lowstate_msg": lowstate_msg,
    }


@pytest.fixture(name="make_robot")
def make_robot():
    """Factory for a UnitreeG1 with every hardware dependency mocked out.

    Controllers are injected directly rather than resolved by name, so no controller
    checkpoint is ever downloaded.
    """
    mocks = _make_sdk_mocks()
    module = "lerobot.robots.unitree_g1.unitree_g1"

    with ExitStack() as stack:
        # require_package would refuse to build the robot without the Unitree SDK installed.
        stack.enter_context(patch(f"{module}.require_package", MagicMock()))
        stack.enter_context(
            patch(
                f"{module}.make_cameras_from_configs",
                lambda cfgs: {name: MagicMock(is_connected=True) for name in cfgs},
            )
        )
        stack.enter_context(patch(f"{module}.G1_29_ArmIK", MagicMock()))
        stack.enter_context(patch(f"{module}._SDKChannelFactoryInitialize", MagicMock()))
        stack.enter_context(
            patch(f"{module}._SDKChannelPublisher", MagicMock(return_value=mocks["publisher_mock"]))
        )
        stack.enter_context(
            patch(f"{module}._SDKChannelSubscriber", MagicMock(return_value=mocks["subscriber_mock"]))
        )
        stack.enter_context(
            patch(f"{module}.unitree_hg_msg_dds__LowCmd_", MagicMock(return_value=mocks["lowcmd_default"]))
        )
        stack.enter_context(patch(f"{module}.hg_LowCmd", MagicMock))
        stack.enter_context(patch(f"{module}.hg_LowState", MagicMock))
        stack.enter_context(patch(f"{module}.CRC", MagicMock(return_value=mocks["crc_mock"])))

        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        built = []

        def _factory(controller=None, cameras=None, **config_kwargs):
            cfg = UnitreeG1Config(
                is_simulation=True,
                gravity_compensation=False,
                cameras=cameras or {},
                **config_kwargs,
            )
            robot = UnitreeG1(cfg)
            if controller is not None:
                robot.controller = controller
                # observation_features/action_features are cached_property; drop any cached value
                # so the injected controller is taken into account.
                robot.__dict__.pop("observation_features", None)
                robot.__dict__.pop("action_features", None)
            built.append(robot)
            return robot

        yield _factory, mocks

        for robot in built:
            if robot.is_connected:
                robot.disconnect()


def arm_for_publish(robot, mocks, kp_value: float = 50.0, kd_value: float = 1.0):
    """Attach the state that ``connect()`` would normally set up, for publish-only tests."""
    robot.msg = mocks["lowcmd_default"]
    robot.crc = mocks["crc_mock"]
    robot.lowcmd_publisher = mocks["publisher_mock"]
    robot.kp = np.full(NUM_MOTORS, kp_value, np.float32)
    robot.kd = np.full(NUM_MOTORS, kd_value, np.float32)
    for cmd in robot.msg.motor_cmd:
        cmd.q = -1.0  # sentinel: untouched joints keep this value
    return robot


def published_targets(robot):
    return {motor.name: robot.msg.motor_cmd[motor.value].q for motor in G1_29_JointIndex}


def hardware_mode(robot):
    """Switch to the real-robot branch after construction.

    Building with ``is_simulation=False`` would make ``__init__`` import the ZMQ bridge, and
    pyzmq is not installed in the test environment, so the flag is flipped once the robot exists.
    """
    robot.config.is_simulation = False
    return robot
